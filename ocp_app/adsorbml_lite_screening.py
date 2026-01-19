# ocp_app/core/adsorbml_lite_screening.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple
from functools import lru_cache
import inspect
import hashlib

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from ase.io import read as ase_read
from ase.geometry import find_mic

from .ads_sites import AdsSite, add_adsorbate_on_site, _oxide_o_based_ads_position, ANION_SYMBOLS

ReactionMode = Literal["HER", "CO2RR"]
OxideAnchorMode = Literal["cation", "anion_o"]
SurfactantClass = Literal["none", "cationic", "anionic", "nonionic"]


@dataclass
class ScreeningSettings:
    # relaxation
    relax_ads_only: bool = True
    fmax: float = 0.05
    max_steps: int = 150

    # sanity filters
    min_ads_surf_dist: float = 0.75
    max_lateral_disp: float = 2.5

    # Energy blow-up filtering
    # NOTE:
    # - CHGNet energy is extensive (total), so absolute total-energy thresholds are dangerous for large slabs.
    # - Use per-atom and/or large positive per-atom cutoff instead.
    max_energy_abs_per_atom: float = 50.0       # very conservative absolute cap (|E|/N)
    max_energy_pos_per_atom: float = 20.0       # catches catastrophic positive energies (E/N)
    max_energy_abs_total: float = 1.0e9         # hard safety cap (kept huge by default)

    # CO2RR placement
    co2rr_clearance: float = 1.2

    # oxide anchor mode for CO2RR
    oxide_anchor_mode: OxideAnchorMode = "cation"
    oxide_anchor_height: float = 1.8  # Å above O (anion_o mode)

    # slab/bulk heuristic warning
    min_vacuum_z_for_slab: float = 5.0  # Å; if estimated vacuum < this, treat as bulk-like (warn)

    # surfactant scenario (CO2RR): coarse-grained class flag for heuristics
    surfactant_class: SurfactantClass = "none"


@dataclass
class ScreenResult:
    label: str
    kind: str
    adsorbate: str
    energy: float
    lateral_disp: float
    valid: bool
    reason: str
    site: AdsSite
    atoms_relaxed: Optional[Atoms] = None
    anchor_pos: Optional[Tuple[float, float, float]] = None

    # diagnostics (safe extras)
    dmin: float = float("nan")
    e_per_atom: float = float("nan")
    converged: bool = True
    n_atoms: int = 0
    anchor_mode: str = ""


# ------------------------ CHGNet loader (Cached) ------------------------

@lru_cache(maxsize=1)
def _get_chgnet_calculator_cached(device: str = "auto"):
    try:
        import torch
    except Exception as e:
        raise RuntimeError("torch is required for CHGNet-based screening") from e

    try:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
    except Exception as e:
        raise RuntimeError("CHGNet is not available. Install: pip install chgnet") from e

    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device

    model = None
    load_err = None
    for load_kwargs in ({}, {"model_name": "MP2023"}):
        try:
            model = CHGNet.load(**load_kwargs)
            break
        except Exception as e:
            load_err = e
            model = None

    if model is None:
        try:
            model = CHGNet.load()
        except Exception:
            raise RuntimeError(f"Failed to load CHGNet pretrained model: {load_err}")

    try:
        return CHGNetCalculator(model=model, use_device=dev)
    except Exception:
        return CHGNetCalculator(model=model, device=dev)


def _get_chgnet_calculator(device: str = "auto"):
    return _get_chgnet_calculator_cached(device)
def relax_slab_chgnet(
    slab: Atoms,
    surfactant_class: SurfactantClass = "none",
    top_z_tol: float = 2.0,
    jiggle_amp: float = 0.05,
    seed: Optional[int] = None,
    fmax: float = 0.05,
    max_steps: int = 200,
    device: str = "auto",
) -> Tuple[Atoms, Dict[str, object]]:
    """Pre-relax a slab with CHGNet, optionally relaxing only a class-specific subset of top-layer atoms.

    Important: This is a *scenario* tool. It does not explicitly model surfactant molecules,
    solvation, or constant-potential conditions. It only changes which atoms are allowed to relax.
    """
    s = slab.copy()
    pos = s.get_positions()
    if len(pos) == 0:
        return s, {"converged": True, "n_relax": 0, "surfactant_class": str(surfactant_class)}

    # Identify "top" atoms by a simple z-window.
    zmax = float(pos[:, 2].max())
    top_mask = pos[:, 2] >= (zmax - float(top_z_tol))
    top_idx = [i for i, ok in enumerate(top_mask) if bool(ok)]

    syms = s.get_chemical_symbols()

    # Class-specific subset selection (fallback to all top atoms if empty).
    relax_idx: List[int]
    if str(surfactant_class) == "cationic":
        relax_idx = [i for i in top_idx if syms[i] in ANION_SYMBOLS]
    elif str(surfactant_class) == "anionic":
        relax_idx = [i for i in top_idx if syms[i] not in ANION_SYMBOLS]
    elif str(surfactant_class) == "nonionic":
        relax_idx = list(top_idx)
    else:
        relax_idx = list(top_idx)

    if not relax_idx:
        relax_idx = list(top_idx)

    # Optional jiggle to explore nearby surface states (scenario proxy).
    try:
        amp = float(jiggle_amp)
    except Exception:
        amp = 0.0

    seed_used: Optional[int] = None
    if amp > 0.0 and len(relax_idx) > 0:
        if seed is None:
            # Stable seed derived from slab + class + key hyperparameters
            s0 = f"{s.get_chemical_formula()}|{tuple(np.round(s.get_cell().lengths(),4).tolist())}|{tuple(np.round(s.get_cell().angles(),3).tolist())}|{str(surfactant_class)}|{float(top_z_tol):.3f}|{amp:.3f}"
            seed_used = int(hashlib.sha256(s0.encode("utf-8")).hexdigest()[:8], 16)
        else:
            seed_used = int(seed)

        rng = np.random.default_rng(seed_used)
        disp = rng.normal(loc=0.0, scale=amp, size=(len(relax_idx), 3))
        pos2 = s.get_positions()
        pos2[np.array(relax_idx, dtype=int)] = pos2[np.array(relax_idx, dtype=int)] + disp
        s.set_positions(pos2)

    fix_idx = [i for i in range(len(s)) if i not in set(relax_idx)]
    if fix_idx and (len(fix_idx) < len(s)):
        s.set_constraint(FixAtoms(indices=fix_idx))

    calc = _get_chgnet_calculator(device=device)
    s.calc = calc

    dyn = FIRE(s, logfile=None)
    converged = bool(dyn.run(fmax=float(fmax), steps=int(max_steps)))

    try:
        e_total = float(s.get_potential_energy())
    except Exception:
        e_total = float("nan")

    meta = {
        "converged": bool(converged),
        "E_total": float(e_total) if np.isfinite(e_total) else None,
        "n_relax": int(len(relax_idx)),
        "n_fix": int(len(fix_idx)),
        "top_z_tol": float(top_z_tol),
        "jiggle_amp": float(jiggle_amp),
        "seed_used": int(seed_used) if seed_used is not None else None,
        "surfactant_class": str(surfactant_class),
    }
    return s, meta



# ------------------------ CO2RR template placement ------------------------

_ADS_TEMPLATE_FILES = {
    "CO": "CO_box.cif",
    "COOH": "COOH_box.cif",
    "HCOO": "HCOO_box.cif",
    "OCHO": "OCHO_box.cif",
}


def _load_ads_template(ref_dir: str, ads: str) -> Atoms:
    ads_clean = ads.replace("*", "").upper()
    if ads_clean not in _ADS_TEMPLATE_FILES:
        raise ValueError(f"Unsupported adsorbate template: {ads}")
    cif_path = f"{ref_dir}/{_ADS_TEMPLATE_FILES[ads_clean]}"

    a = ase_read(cif_path).copy()

    syms = a.get_chemical_symbols()
    anchor_idx = 0
    for i, s in enumerate(syms):
        if s == "C":
            anchor_idx = i
            break

    pos = a.get_positions()
    pos -= pos[anchor_idx].copy()
    if pos[:, 2].mean() < 0:
        pos[:, 2] *= -1
    a.set_positions(pos)
    return a


def _build_slab_ads_co2rr(
    slab: Atoms,
    site: AdsSite,
    ads: str,
    settings: ScreeningSettings,
    ref_dir: str,
    oxide_anchor_mode_override: Optional[OxideAnchorMode] = None,
) -> Tuple[Atoms, int, np.ndarray, str]:
    slab0 = slab.copy()
    pos_slab = slab0.get_positions()
    z_top_global = float(pos_slab[:, 2].max())

    x, y, z = site.position

    anchor_mode = oxide_anchor_mode_override or settings.oxide_anchor_mode

    if anchor_mode == "anion_o":
        x, y, z = _oxide_o_based_ads_position(
            slab0,
            AdsSite(kind=site.kind, position=(x, y, z), surface_indices=site.surface_indices),
            h_oh=float(settings.oxide_anchor_height),
            extra_z=0.0,
        )

    ads_atoms = _load_ads_template(ref_dir=ref_dir, ads=ads)
    n_slab = len(slab0)

    syms = ads_atoms.get_chemical_symbols()
    anchor_local = 0
    for i, s in enumerate(syms):
        if s == "C":
            anchor_local = i
            break

    pos_ads = ads_atoms.get_positions()
    z_min_ads = float(pos_ads[:, 2].min())

    # base Z: global top + clearance (default)
    target_z_base = z_top_global + float(settings.co2rr_clearance)

    # if site z is already near top surface, respect it
    if z > z_top_global - 0.5:
        target_z_base = max(target_z_base, float(z))

    base_z = target_z_base - z_min_ads
    base = np.array([float(x), float(y), float(base_z)], dtype=float)

    ads_atoms.set_cell(slab0.get_cell())
    ads_atoms.set_pbc(slab0.get_pbc())
    ads_atoms.translate(base)

    slab_ads = slab0 + ads_atoms
    anchor_global = n_slab + anchor_local
    init_anchor_xy = slab_ads.get_positions()[anchor_global][:2].copy()

    return slab_ads, anchor_global, init_anchor_xy, str(anchor_mode)


# ------------------------ robust helpers ------------------------

def _estimate_vacuum_z(atoms: Atoms) -> float:
    """Heuristic vacuum estimate along cartesian z (assumes typical slab alignment)."""
    try:
        cell = atoms.get_cell()
        Lz = float(np.linalg.norm(np.asarray(cell)[2]))
        pos = atoms.get_positions()
        zspan = float(pos[:, 2].max() - pos[:, 2].min())
        vac = Lz - zspan
        return float(max(vac, 0.0))
    except Exception:
        return float("nan")


def _min_dist_ads_to_surf(atoms: Atoms, n_slab: int) -> float:
    """Robust minimum distance between ads atoms and slab atoms with MIC."""
    if n_slab <= 0 or len(atoms) <= n_slab:
        return 999.0

    dmat = atoms.get_all_distances(mic=True)
    ads_idx = list(range(n_slab, len(atoms)))
    slab_idx = list(range(0, n_slab))
    sub = dmat[np.ix_(ads_idx, slab_idx)]
    return float(np.min(sub))


def _lateral_disp(anchor_xy0: np.ndarray, anchor_xy1: np.ndarray, cell, pbc) -> float:
    """PBC-correct lateral displacement using MIC on general cells (including non-orthogonal)."""
    dxy = anchor_xy1 - anchor_xy0
    d3 = np.array([float(dxy[0]), float(dxy[1]), 0.0], dtype=float)
    pbc2 = [bool(pbc[0]), bool(pbc[1]), False]
    vec, dist = find_mic(d3, cell=cell, pbc=pbc2)
    # vec is MIC-corrected displacement vector
    return float(np.linalg.norm(np.asarray(vec)[:2]))


def _reason_key(reason: str) -> str:
    if not reason:
        return "unknown"
    if reason.startswith("exception:"):
        return "exception"
    # strip details like collision(dmin=...)
    return reason.split("(", 1)[0].strip()


def summarize_outcomes(outs: List[ScreenResult]) -> Dict[str, object]:
    """Counts by reason, plus totals/valids."""
    total = len(outs)
    n_valid = sum(1 for o in outs if o.valid)
    keys = [_reason_key(o.reason) for o in outs]
    uniq, cnt = np.unique(np.array(keys, dtype=object), return_counts=True)
    by_reason = {str(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
    return {
        "n_total": int(total),
        "n_valid": int(n_valid),
        "by_reason": by_reason,
    }


# ------------------------ core pipeline ------------------------

def screen_sites_adsorbml_lite(
    slab: Atoms,
    sites: List[AdsSite],
    reaction: ReactionMode,
    mtype: Literal["metal", "oxide"],
    adsorbates: List[str],
    top_k: int = 6,
    ref_dir: str = "ref_gas",
    settings: Optional[ScreeningSettings] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    device: str = "auto",
    return_raw: bool = False,
) -> Dict[str, List[ScreenResult]] | Tuple[Dict[str, List[ScreenResult]], Dict[str, List[ScreenResult]], Dict[str, Dict[str, object]]]:
    """
    Returns:
      - default: results_by_ads (valid top-k only)
      - return_raw=True: (results_by_ads, raw_by_ads, stats_by_ads)

    Notes:
      - Ranking uses CHGNet *total* energy, which is consistent within a fixed slab size.
      - For oxide-CO2RR, an optional coarse-grained surfactant_class flag can switch anchor modes:
          * cationic  -> cation anchor
          * anionic   -> anion_o anchor
          * nonionic  -> evaluates both anchor modes (cation and anion_o)
      - This is scenario logic and does not explicitly model surfactant molecules/solvation.
    """
    if settings is None:
        settings = ScreeningSettings()

    # slab/bulk heuristic warning
    vacz = _estimate_vacuum_z(slab)
    bulk_like = (float(vacz) < float(settings.min_vacuum_z_for_slab)) and bool(slab.get_pbc()[2])

    calc = _get_chgnet_calculator(device=device)

    # anchor-mode expansion for oxide-CO2RR
    anchor_modes: List[Optional[OxideAnchorMode]] = [None]
    if (reaction == "CO2RR") and (mtype == "oxide"):
        surf = str(getattr(settings, "surfactant_class", "none") or "none")
        if surf == "anionic":
            anchor_modes = ["anion_o"]
        elif surf == "cationic":
            anchor_modes = ["cation"]
        elif surf == "nonionic":
            anchor_modes = ["cation", "anion_o"]
        else:
            anchor_modes = [settings.oxide_anchor_mode]
    else:
        anchor_modes = [None]

    mult = int(len(anchor_modes)) if ((reaction == "CO2RR") and (mtype == "oxide")) else 1
    n_total = int(max(len(adsorbates), 1) * max(len(sites), 1) * max(mult, 1))
    job_idx = 0

    results_by_ads: Dict[str, List[ScreenResult]] = {}
    raw_by_ads: Dict[str, List[ScreenResult]] = {}
    stats_by_ads: Dict[str, Dict[str, object]] = {}

    for ads in (adsorbates or []):
        outs: List[ScreenResult] = []

        for i, site in enumerate(sites or []):
            # Expand anchor modes only for oxide-CO2RR
            run_modes = anchor_modes if ((reaction == "CO2RR") and (mtype == "oxide")) else [None]

            for am in run_modes:
                job_idx += 1
                if progress_cb is not None:
                    progress_cb(job_idx, n_total, f"{ads} screening")

                suffix = ""
                if (reaction == "CO2RR") and (mtype == "oxide") and (am is not None) and (len(run_modes) > 1):
                    suffix = f"_{am}"

                label = f"{ads.replace('*','')}_{site.kind}_{i}{suffix}"

                try:
                    if reaction == "HER":
                        mode = "default" if mtype == "metal" else "oxide_o"
                        atoms0 = add_adsorbate_on_site(slab, site, symbol="H", dz=0.0, mode=mode)
                        n_slab = len(slab)
                        anchor_global = len(atoms0) - 1
                        anchor_xy0 = atoms0.get_positions()[anchor_global][:2].copy()
                        anchor_mode_used = ""
                    else:
                        atoms0, anchor_global, anchor_xy0, anchor_mode_used = _build_slab_ads_co2rr(
                            slab, site, ads=ads, settings=settings, ref_dir=ref_dir, oxide_anchor_mode_override=am
                        )
                        n_slab = len(slab)

                    atoms = atoms0.copy()
                    atoms.calc = calc

                    if settings.relax_ads_only:
                        atoms.set_constraint(FixAtoms(indices=list(range(n_slab))))

                    dyn = FIRE(atoms, logfile=None)
                    converged = bool(dyn.run(fmax=float(settings.fmax), steps=int(settings.max_steps)))

                    e_total = float(atoms.get_potential_energy())
                    n_atoms = int(len(atoms))
                    e_pa = float(e_total / max(n_atoms, 1))

                    # Adsorbate lateral displacement (xy)
                    pos = atoms.get_positions()
                    anchor_xy1 = pos[anchor_global][:2].copy()
                    disp = float(np.linalg.norm(anchor_xy1 - anchor_xy0))

                    # Minimum distance between adsorbate atoms and surface atoms
                    dmin = float("inf")
                    try:
                        dvec, _ = find_mic(pos[n_slab:] - pos[:n_slab].mean(axis=0), atoms.get_cell(), atoms.get_pbc())
                        # not exact; fallback to brute-force below
                        _ = dvec
                    except Exception:
                        pass
                    try:
                        ads_pos = pos[n_slab:]
                        slab_pos = pos[:n_slab]
                        for ap in ads_pos:
                            dd = np.linalg.norm(slab_pos - ap, axis=1)
                            dmin = min(dmin, float(dd.min()))
                    except Exception:
                        dmin = float("nan")

                    if (np.isfinite(dmin)) and (dmin < float(settings.min_ads_surf_dist)):
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "too_close", site,
                                                atoms_relaxed=atoms if return_raw else None,
                                                anchor_pos=tuple(pos[anchor_global]),
                                                dmin=dmin, e_per_atom=e_pa, converged=converged, n_atoms=n_atoms,
                                                anchor_mode=str(anchor_mode_used)))
                        continue

                    if disp > float(settings.max_lateral_disp):
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "migrated", site,
                                                atoms_relaxed=atoms if return_raw else None,
                                                anchor_pos=tuple(pos[anchor_global]),
                                                dmin=dmin, e_per_atom=e_pa, converged=converged, n_atoms=n_atoms,
                                                anchor_mode=str(anchor_mode_used)))
                        continue

                    # Energy blow-up checks (robust against slab size)
                    if (not np.isfinite(e_total)) or (not np.isfinite(e_pa)):
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "energy_nan", site,
                                                atoms_relaxed=atoms if return_raw else None,
                                                anchor_pos=tuple(pos[anchor_global]),
                                                dmin=dmin, e_per_atom=e_pa, converged=converged, n_atoms=n_atoms,
                                                anchor_mode=str(anchor_mode_used)))
                        continue
                    if abs(e_total) > float(settings.max_energy_abs_total):
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "energy_blowup_total", site,
                                                atoms_relaxed=atoms if return_raw else None,
                                                anchor_pos=tuple(pos[anchor_global]),
                                                dmin=dmin, e_per_atom=e_pa, converged=converged, n_atoms=n_atoms,
                                                anchor_mode=str(anchor_mode_used)))
                        continue
                    if abs(e_pa) > float(settings.max_energy_abs_per_atom):
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "energy_blowup_per_atom", site,
                                                atoms_relaxed=atoms if return_raw else None,
                                                anchor_pos=tuple(pos[anchor_global]),
                                                dmin=dmin, e_per_atom=e_pa, converged=converged, n_atoms=n_atoms,
                                                anchor_mode=str(anchor_mode_used)))
                        continue
                    if e_pa > float(settings.max_energy_pos_per_atom):
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "energy_pos_per_atom", site,
                                                atoms_relaxed=atoms if return_raw else None,
                                                anchor_pos=tuple(pos[anchor_global]),
                                                dmin=dmin, e_per_atom=e_pa, converged=converged, n_atoms=n_atoms,
                                                anchor_mode=str(anchor_mode_used)))
                        continue

                    outs.append(ScreenResult(label, site.kind, ads, e_total, disp, True, "ok", site,
                                            atoms_relaxed=atoms if return_raw else None,
                                            anchor_pos=tuple(pos[anchor_global]),
                                            dmin=dmin, e_per_atom=e_pa, converged=converged, n_atoms=n_atoms,
                                            anchor_mode=str(anchor_mode_used)))

                except Exception as e:
                    outs.append(
                        ScreenResult(
                            label=label,
                            kind=site.kind,
                            adsorbate=ads,
                            energy=float("nan"),
                            lateral_disp=float("nan"),
                            valid=False,
                            reason=f"exception:{str(e)}",
                            site=site,
                            anchor_mode=str(am) if am is not None else "",
                        )
                    )

        # rank valids by total energy (consistent within fixed slab size)
        valids = [o for o in outs if o.valid and np.isfinite(o.energy)]
        valids.sort(key=lambda x: x.energy)

        results_by_ads[ads] = valids[: int(max(top_k, 1))]
        raw_by_ads[ads] = outs
        stats = summarize_outcomes(outs)
        stats["bulk_like_warning"] = bool(bulk_like)
        stats["vacuum_z_est"] = float(vacz) if np.isfinite(vacz) else None
        stats["surfactant_class"] = str(getattr(settings, "surfactant_class", "none") or "none")
        if (reaction == "CO2RR") and (mtype == "oxide"):
            stats["anchor_modes_evaluated"] = list(anchor_modes)
        stats_by_ads[ads] = stats

    if return_raw:
        return results_by_ads, raw_by_ads, stats_by_ads
    return results_by_ads


def union_topk_sites(
    results_by_ads: Dict[str, List[ScreenResult]],
    union_max_sites: int = 10,
    xy_bin: float = 0.25,
) -> Tuple[Dict[str, AdsSite], Dict[str, Atoms], List[ScreenResult]]:
    """
    adsorbate별 top-k를 합쳐서 중복(xy)을 binning으로 제거하고 union set 생성.
    """
    all_items: List[ScreenResult] = []
    for ads, items in results_by_ads.items():
        all_items.extend(items)

    all_items.sort(key=lambda x: x.energy)

    seen = set()
    union_items: List[ScreenResult] = []
    for it in all_items:
        if it.anchor_pos is None:
            continue
        x, y, _ = it.anchor_pos
        key = (round(float(x) / xy_bin), round(float(y) / xy_bin))
        if key in seen:
            continue
        seen.add(key)
        union_items.append(it)
        if len(union_items) >= int(max(union_max_sites, 1)):
            break

    site_map: Dict[str, AdsSite] = {}
    struct_map: Dict[str, Atoms] = {}
    for j, it in enumerate(union_items):
        label = f"ML_{j}_{it.adsorbate.replace('*','')}_{it.kind}"
        site_map[label] = it.site
        if it.atoms_relaxed is not None:
            struct_map[label] = it.atoms_relaxed

    return site_map, struct_map, union_items
