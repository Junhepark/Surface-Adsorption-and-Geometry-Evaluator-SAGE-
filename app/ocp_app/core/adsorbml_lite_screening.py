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

from .ads_sites import (
    AdsSite,
    add_adsorbate_on_site,
    _oxide_o_based_ads_position,
    oxide_surface_seed_position,
    expand_oxide_channels_for_adsorbate,
    detect_metal_111_sites,
    detect_oxide_surface_sites,
    ANION_SYMBOLS,
)

ReactionMode = Literal["HER", "CO2RR", "ORR"]
OxideAnchorMode = Literal["cation", "anion_o", "bridge_mixed"]
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

    # site-tracking metadata
    seed_kind: str = "unknown"
    initial_site_kind: str = "unknown"
    final_site_kind: str = "unknown"
    migration_basis: str = "none"
    migration_type: str = "none"
    migrated: bool = False
    final_site_match_dist: float = float("nan")
    initial_anchor_xy: Optional[Tuple[float, float]] = None
    final_anchor_xy: Optional[Tuple[float, float]] = None
    seed_surface_indices: str = ""
    final_surface_indices: str = ""
    qc_flags: str = ""
    classification_mode: str = "auto"
    site_family: str = "unknown"


def _normalize_site_kind(kind: object) -> str:
    k = str(kind or "").strip().lower()
    aliases = {
        "top": "ontop",
        "atop": "ontop",
        "on-top": "ontop",
        "on_top": "ontop",
        "o_top": "anion_ontop",
        "o-top": "anion_ontop",
        "anion_o": "anion_ontop",
    }
    return aliases.get(k, k if k else "unknown")


def _kind_priority(kind: str) -> int:
    return {
        "ontop": 0,
        "anion_ontop": 0,
        "bridge": 1,
        "hollow": 2,
        "fcc": 2,
        "hcp": 3,
        "unknown": 99,
    }.get(_normalize_site_kind(kind), 50)


def _fmt_surface_indices(indices: object) -> str:
    try:
        vals = tuple(int(i) for i in (indices or ()))
    except Exception:
        vals = tuple()
    return ",".join(str(i) for i in vals)


def _mic_xy_delta(cell, pbc, xy0: np.ndarray, xy1: np.ndarray) -> tuple[np.ndarray, float]:
    xy0 = np.asarray(xy0, dtype=float).reshape(2)
    xy1 = np.asarray(xy1, dtype=float).reshape(2)
    d3 = np.array([float(xy1[0] - xy0[0]), float(xy1[1] - xy0[1]), 0.0], dtype=float)
    vec, _ = find_mic(d3, cell=cell, pbc=[bool(pbc[0]), bool(pbc[1]), False])
    vec = np.asarray(vec, dtype=float)
    return vec[:2], float(np.linalg.norm(vec[:2]))


def _classify_site_from_candidates(slab_only, anchor_xy: np.ndarray, candidates: list[AdsSite], *, unknown_tol: float = 0.9) -> dict[str, object]:
    anchor_xy = np.asarray(anchor_xy, dtype=float).reshape(2)
    best = None
    best_key = None
    for idx, site in enumerate(candidates or []):
        cand_xy = np.asarray(site.position[:2], dtype=float)
        _, dist = _mic_xy_delta(slab_only.get_cell(), slab_only.get_pbc(), cand_xy, anchor_xy)
        kind = _normalize_site_kind(getattr(site, "kind", "unknown"))
        key = (float(dist), _kind_priority(kind), int(idx))
        if best is None or key < best_key:
            best = {
                "kind": kind,
                "match_dist": float(dist),
                "surface_indices": tuple(int(i) for i in getattr(site, "surface_indices", ()) or ()),
            }
            best_key = key
    if best is None:
        return {"kind": "unknown", "match_dist": float("nan"), "surface_indices": tuple()}
    if np.isfinite(best["match_dist"]) and float(best["match_dist"]) > float(unknown_tol):
        best["kind"] = "unknown"
    return best


def _classify_metal_site_xy(slab_only, anchor_xy: np.ndarray) -> dict[str, object]:
    try:
        cands = detect_metal_111_sites(slab_only)
    except Exception:
        cands = []
    out = _classify_site_from_candidates(slab_only, anchor_xy, cands, unknown_tol=0.9)
    if out.get("kind") in ("fcc", "hollow"):
        # legacy best-effort hcp split by 2nd layer proximity (only when candidates expose hcp/fcc).
        try:
            pos = slab_only.get_positions()
            top = pos[np.asarray([i for i in range(len(slab_only)) if slab_only[i].symbol != "H"], int), :2]
        except Exception:
            top = None
    return out


def _classify_oxide_site_xy(slab_only, anchor_xy: np.ndarray) -> dict[str, object]:
    try:
        cands = detect_oxide_surface_sites(slab_only)
    except Exception:
        cands = []
    return _classify_site_from_candidates(slab_only, anchor_xy, cands, unknown_tol=1.1)


def _classify_oxide_anion_xy(slab_only, anchor_xy: np.ndarray) -> dict[str, object]:
    try:
        pos = slab_only.get_positions()
        sym = slab_only.get_chemical_symbols()
        top_idx = [i for i, s in enumerate(sym) if (s in ANION_SYMBOLS)]
        if not top_idx:
            return {"kind": "unknown", "match_dist": float("nan"), "surface_indices": tuple()}
        z = pos[top_idx, 2]
        zmax = float(np.max(z))
        cand = [i for i in top_idx if (zmax - float(pos[i,2])) < 0.8]
        cand = cand or top_idx
        best = None
        best_key = None
        for i in cand:
            xy = np.asarray(pos[i, :2], dtype=float)
            _, dist = _mic_xy_delta(slab_only.get_cell(), slab_only.get_pbc(), xy, anchor_xy)
            key = (float(dist), int(i))
            if best is None or key < best_key:
                best = {"kind": "anion_ontop", "match_dist": float(dist), "surface_indices": (int(i),)}
                best_key = key
        if best is None:
            return {"kind": "unknown", "match_dist": float("nan"), "surface_indices": tuple()}
        if np.isfinite(best["match_dist"]) and float(best["match_dist"]) > 1.0:
            best["kind"] = "unknown"
        return best
    except Exception:
        return {"kind": "unknown", "match_dist": float("nan"), "surface_indices": tuple()}


def _resolve_site_tracking(
    *,
    slab_only,
    mtype: str,
    seed_kind: object,
    initial_xy: np.ndarray,
    final_anchor_xy: np.ndarray,
    classification_mode: str = "auto",
    disp_threshold: float = 2.5,
) -> dict[str, object]:
    seed_kind_n = _normalize_site_kind(seed_kind)
    initial_kind = seed_kind_n
    mode = str(classification_mode or "auto")
    if mode == "oxide_anion":
        final = _classify_oxide_anion_xy(slab_only, final_anchor_xy)
        site_family = "oxide_anion"
    elif str(mtype).lower() == "metal":
        final = _classify_metal_site_xy(slab_only, final_anchor_xy)
        site_family = "metal"
    else:
        final = _classify_oxide_site_xy(slab_only, final_anchor_xy)
        site_family = "oxide_cation"
    final_kind = _normalize_site_kind(final.get("kind", "unknown"))
    _, disp_mic = _mic_xy_delta(slab_only.get_cell(), slab_only.get_pbc(), initial_xy, final_anchor_xy)

    migration_basis = "none"
    migrated = False
    if final_kind == "unknown":
        migration_basis = "unclassified_final_site"
        migrated = True
    elif final_kind != initial_kind:
        migration_basis = "site_kind_changed"
        migrated = True
    elif float(disp_mic) > float(disp_threshold):
        migration_basis = "lateral_disp"
        migrated = True

    if migrated and (initial_kind not in ("unknown", "") and final_kind not in ("unknown", "")):
        migration_type = f"{initial_kind}_to_{final_kind}" if final_kind != initial_kind else "same_kind_displaced"
    elif migrated:
        migration_type = migration_basis
    else:
        migration_type = "none"

    qc_flags = []
    if migrated:
        qc_flags.append("migrated_site")
    if final_kind == "unknown":
        qc_flags.append("unclassified_site")

    return {
        "seed_site_kind": seed_kind_n,
        "initial_site_kind": initial_kind,
        "final_site_kind": final_kind,
        "classification_mode": mode,
        "site_family": site_family,
        "migration_basis": migration_basis,
        "migration_type": migration_type,
        "migrated": bool(migrated),
        "lateral_disp_mic(Å)": float(disp_mic),
        "final_site_match_dist(Å)": float(final.get("match_dist", float("nan"))),
        "final_surface_indices": _fmt_surface_indices(final.get("surface_indices", ())),
        "qc_flags": list(qc_flags),
        "initial_anchor_x(Å)": float(np.asarray(initial_xy, dtype=float).reshape(2)[0]),
        "initial_anchor_y(Å)": float(np.asarray(initial_xy, dtype=float).reshape(2)[1]),
        "final_anchor_x(Å)": float(np.asarray(final_anchor_xy, dtype=float).reshape(2)[0]),
        "final_anchor_y(Å)": float(np.asarray(final_anchor_xy, dtype=float).reshape(2)[1]),
    }


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
    "O": "O_box.cif",
    "OH": "OH_box.cif",
    "OOH": "OOH_box.cif",
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
    n_migrated = sum(1 for o in outs if bool(getattr(o, "migrated", False)) or (_reason_key(getattr(o, "reason", "")) == "migrated"))
    return {
        "n_total": int(total),
        "n_valid": int(n_valid),
        "n_migrated": int(n_migrated),
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

    # anchor-mode expansion
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

    mult = int(len(anchor_modes)) if ((reaction == "CO2RR") and (mtype == "oxide")) else 1
    n_total = int(max(len(adsorbates), 1) * max(len(sites), 1) * max(mult, 1))
    job_idx = 0

    results_by_ads: Dict[str, List[ScreenResult]] = {}
    raw_by_ads: Dict[str, List[ScreenResult]] = {}
    stats_by_ads: Dict[str, Dict[str, object]] = {}

    for ads in (adsorbates or []):
        outs: List[ScreenResult] = []

        for i, site in enumerate(sites or []):
            ads_clean = str(ads).replace("*", "").upper()
            if (reaction == "CO2RR") and (mtype == "oxide"):
                run_modes = anchor_modes
            elif (reaction == "ORR") and (mtype == "oxide"):
                run_modes = list(expand_oxide_channels_for_adsorbate(ads_clean))
            else:
                run_modes = [None]

            for am in run_modes:
                job_idx += 1
                if progress_cb is not None:
                    progress_cb(job_idx, n_total, f"{ads} screening")

                suffix = ""
                if (mtype == "oxide") and (am is not None) and (len(run_modes) > 1):
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

                    # Adsorbate lateral displacement (xy) + site tracking
                    pos = atoms.get_positions()
                    anchor_xy1 = pos[anchor_global][:2].copy()
                    disp = _lateral_disp(anchor_xy0, anchor_xy1, atoms.get_cell(), atoms.get_pbc())
                    tracking = _resolve_site_tracking(
                        slab_only=atoms[:n_slab],
                        mtype=mtype,
                        seed_kind=("anion_ontop" if ((reaction == "HER") and (mtype == "oxide")) else site.kind),
                        initial_xy=anchor_xy0,
                        final_anchor_xy=anchor_xy1,
                        classification_mode=("oxide_anion" if ((reaction == "HER") and (mtype == "oxide")) else "auto"),
                        disp_threshold=float(settings.max_lateral_disp),
                    )
                    migrated_flag = bool(tracking["migrated"])
                    qc_flags = list(tracking["qc_flags"])

                    # Minimum distance between adsorbate atoms and surface atoms
                    try:
                        dmin = _min_dist_ads_to_surf(atoms, n_slab)
                    except Exception:
                        dmin = float("nan")

                    base_kwargs = dict(
                        atoms_relaxed=atoms if return_raw else None,
                        anchor_pos=tuple(pos[anchor_global]),
                        dmin=dmin,
                        e_per_atom=e_pa,
                        converged=converged,
                        n_atoms=n_atoms,
                        anchor_mode=str(anchor_mode_used),
                        seed_kind=tracking["seed_site_kind"],
                        initial_site_kind=tracking["initial_site_kind"],
                        final_site_kind=tracking["final_site_kind"],
                        migration_basis=tracking["migration_basis"],
                        migration_type=tracking["migration_type"],
                        migrated=bool(migrated_flag),
                        final_site_match_dist=float(tracking["final_site_match_dist(Å)"]),
                        initial_anchor_xy=(float(tracking["initial_anchor_x(Å)"]), float(tracking["initial_anchor_y(Å)"])),
                        final_anchor_xy=(float(tracking["final_anchor_x(Å)"]), float(tracking["final_anchor_y(Å)"])),
                        seed_surface_indices=_fmt_surface_indices(getattr(site, "surface_indices", ())),
                        final_surface_indices=tracking["final_surface_indices"],
                        classification_mode=tracking["classification_mode"],
                        site_family=tracking["site_family"],
                    )

                    if (np.isfinite(dmin)) and (dmin < float(settings.min_ads_surf_dist)):
                        qc_flags2 = qc_flags + ["too_close"]
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "too_close", site, qc_flags=";".join(qc_flags2), **base_kwargs))
                        continue

                    if migrated_flag:
                        qc_flags2 = qc_flags + ["migrated_pre_screen"]
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "migrated", site, qc_flags=";".join(qc_flags2), **base_kwargs))
                        continue

                    # Energy blow-up checks (robust against slab size)
                    if (not np.isfinite(e_total)) or (not np.isfinite(e_pa)):
                        qc_flags2 = qc_flags + ["energy_nan"]
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "energy_nan", site, qc_flags=";".join(qc_flags2), **base_kwargs))
                        continue
                    if abs(e_total) > float(settings.max_energy_abs_total):
                        qc_flags2 = qc_flags + ["energy_blowup_total"]
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "energy_blowup_total", site, qc_flags=";".join(qc_flags2), **base_kwargs))
                        continue
                    if abs(e_pa) > float(settings.max_energy_abs_per_atom):
                        qc_flags2 = qc_flags + ["energy_blowup_per_atom"]
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "energy_blowup_per_atom", site, qc_flags=";".join(qc_flags2), **base_kwargs))
                        continue
                    if e_pa > float(settings.max_energy_pos_per_atom):
                        qc_flags2 = qc_flags + ["energy_pos_per_atom"]
                        outs.append(ScreenResult(label, site.kind, ads, e_total, disp, False, "energy_pos_per_atom", site, qc_flags=";".join(qc_flags2), **base_kwargs))
                        continue

                    outs.append(ScreenResult(label, site.kind, ads, e_total, disp, True, "ok", site, qc_flags=";".join(qc_flags), **base_kwargs))

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
                            seed_kind=_normalize_site_kind(site.kind),
                            initial_site_kind=_normalize_site_kind(site.kind),
                            final_site_kind="unknown",
                            migration_basis="exception",
                            migration_type="exception",
                            migrated=False,
                            seed_surface_indices=_fmt_surface_indices(getattr(site, "surface_indices", ())),
                            qc_flags="exception",
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
    Merge per-adsorbate top-k results, remove xy-duplicates via binning,
    and build a unified site/structure map.
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
        kind_label = str(getattr(it, "final_site_kind", getattr(it, "kind", "unknown")) or getattr(it, "kind", "unknown"))
        label = f"ML_{j}_{it.adsorbate.replace('*','')}_{kind_label}"
        site_map[label] = it.site
        if it.atoms_relaxed is not None:
            struct_map[label] = it.atoms_relaxed

    return site_map, struct_map, union_items
