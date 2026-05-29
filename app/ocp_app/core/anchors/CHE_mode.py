from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.geometry import find_mic

from ocp_app.core.gas_refs import get_h2_ref
from ocp_app.core.ads_sites import (
    AdsSite,
    detect_metal_111_sites,
    detect_oxide_surface_sites,
    expand_oxide_channels_for_adsorbate,
    oxide_surface_seed_position,
    detect_oxide_oer_cation_sites,
    oxide_oer_slab_suitability,
    select_representative_sites,
    ANION_SYMBOLS,
)
from ocp_app.core.anchors.common import (
    calc,
    MODEL_NAME,
    DEVICE,
    ZPE_CORR,
    TDS_CORR,
    NET_CORR,
    H0S,
    MIGRATE_THR,
    recommend_h0s_for_relaxation,
    default_min_clearance_for_scope,
    VAC_WARN_MIN,
    UNUSUAL_DDELTA,
    ensure_pbc3,
    layer_indices,
    normalize_relaxation_scope,
    relax_freeH,
    site_energy_two_stage,
)

from ocp_app.core.anchors.local_zpe import compute_local_h_thermo_correction
from ocp_app.core.oxide_descriptor import run_oxide_descriptor_profile
from ocp_app.core.anchors.oxide_her import (
    _generate_oxide_her_oanchor_sites,
    _project_oxide_her_sites_to_otop,
)

# =====================================================================
# Unified CHE workflow (Metal & Oxide)
#  - HER: ΔG_H (H* adsorption)
#  - CO2RR: ΔG_ads for COOH*, CO*, HCOO*, OCHO* (reaction-descriptor based)
#  - OER: stepwise free energies for OOH*, O*, OH* (4e⁻ Norskov CHE)
# =====================================================================

# --- HER CHE correction (calibrated on Ni(111)) ---
STANDARD_CHE_CORR = 0.24  # eV

THREE_STAGE_OXIDE_HER_CAUTION = (
    "Caution: The O–H and reactive-H stages are evaluated using the OCP-based "
    "relaxed-state workflow, whereas the H₂ pairing stage is only an approximate "
    "release proxy rather than an explicit reaction barrier. The final-stage result "
    "should therefore be used as a supportive screening indicator, not as a "
    "quantitatively validated kinetic metric."
)


DESCRIPTOR_D1_DISP_THRESH_A = 1.20
DESCRIPTOR_D2_DISP_THRESH_A = 1.00

# --- CO2RR adsorbate-specific constant shifts (ZPE + TΔS lumped) ---
# Defaults used when thermo_CO2RR.json is missing or cannot be parsed
DEFAULT_ADS_CORR: Dict[str, float] = {
    "COOH": 0.0,
    "CO": 0.0,
    "HCOO": 0.0,
    "OCHO": 0.0,
}

# Adsorbate template file names in ref_gas/
ADS_TEMPLATE_FILES: Dict[str, str] = {
    "CO": "CO_box.cif",
    "COOH": "COOH_box.cif",
    "HCOO": "HCOO_box.cif",
    "OCHO": "OCHO_box.cif",
    # ── ORR intermediates ──
    "OH":  "OH_box.cif",
    "OOH": "OOH_box.cif",
    "O":   "O_box.cif",
}

THERMO_CO2RR_NAME = "thermo_CO2RR.json"

# ── ORR/OER oxygen-intermediate settings ──────────────────────────────────────────────────────
# Uses the Norskov CHE approach without direct O2 energy calculation:
# E_O2_eff = 2·E_H2O - 2·E_H2 + 4×1.23 eV
ORR_EQUIL_POTENTIAL = 1.23  # V vs RHE (standard O2 reduction equilibrium potential)

# ORR/OER ZPE + TΔS correction defaults (can be overridden via thermo_ORR.json)
DEFAULT_ORR_CORR: Dict[str, float] = {
    "OOH": 0.40,
    "O":   0.05,
    "OH":  0.35,
}

THERMO_ORR_NAME = "thermo_ORR.json"

# Number of electrons consumed per ORR intermediate (4e⁻ pathway)
ORR_N_ELECTRONS: Dict[str, int] = {"OOH": 1, "O": 2, "OH": 3}

# OER AEM strict benchmark placement policy.  These are applied only in
# reaction_mode="OER" on oxide surfaces.  HER placement and HER descriptor
# logic are intentionally not connected to this table.
OER_AEM_CATION_HEIGHT_GRID: Dict[str, Tuple[float, ...]] = {
    "O": (1.55, 1.75, 1.95),
    "OH": (1.65, 1.85, 2.05),
    "OOH": (1.80, 2.05, 2.30),
}

# OER AEM stabilization policy:
# Do not use the v7 guided FixBondLength relaxation.  It tended to produce
# artificial fragmentation after constraint release on rutile IrO2.
# Instead, keep the v6 local-open-direction placement and apply only a short,
# unconstrained adsorbate relaxation for oxide-OER cation-bound oxygen
# intermediates.  This is intentionally limited to reaction_mode="OER" and
# does not touch HER.
OER_AEM_SHORT_RELAX_STEPS = 80
OER_AEM_SHORT_RELAX_FMAX = 0.08

# OER diagnostic relaxation modes. These apply only to reaction_mode="OER"
# on oxide cation-bound oxygen intermediates. HER/CO2RR/ORR paths do not use
# these switches.
OER_RELAXATION_MODES = {"placement_only", "single_point", "short_relax", "normal_relax"}
DEFAULT_OER_RELAXATION_MODE = "short_relax"
# ─────────────────────────────────────────────────────────────────────


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)



def _load_ads_corr(ref_dir: str | Path = "ref_gas") -> Dict[str, float]:
    """
    Load per-adsorbate ZPE correction values from the ΔZPE_ads block in
    thermo_CO2RR.json.  Falls back to DEFAULT_ADS_CORR if file is missing
    or cannot be parsed.
    """
    ref_dir = Path(ref_dir)
    thermo_path = ref_dir / THERMO_CO2RR_NAME

    corr = DEFAULT_ADS_CORR.copy()
    if not thermo_path.is_file():
        return corr

    try:
        data = json.loads(thermo_path.read_text())
    except Exception:
        return corr

    zpe_block = (
        data.get("ΔZPE_ads (eV)")
        or data.get("dZPE_ads (eV)")
        or data.get("dZPE_ads")
        or {}
    )

    for key in corr.keys():
        if key in zpe_block:
            try:
                corr[key] = float(zpe_block[key])
            except Exception:
                pass

    return corr


def _load_co2rr_thermo(
    ref_dir: str | Path = "ref_gas",
) -> Tuple[Optional[dict], Dict[str, float], Path]:
    """
    Load CO2RR thermodynamic data and per-adsorbate corrections.

    Returns
    -------
    (thermo_data or None, ads_corr dict, thermo_path)
    """
    ref_dir = Path(ref_dir)
    thermo_path = ref_dir / THERMO_CO2RR_NAME

    if not thermo_path.is_file():
        return None, DEFAULT_ADS_CORR.copy(), thermo_path

    try:
        data = json.loads(thermo_path.read_text())
    except Exception:
        return None, DEFAULT_ADS_CORR.copy(), thermo_path

    ads_corr = _load_ads_corr(ref_dir)
    return data, ads_corr, thermo_path

def _load_orr_thermo(ref_dir: str | Path = "ref_gas") -> Dict[str, float]:
    """
    Load per-adsorbate ZPE+TΔS corrections for ORR intermediates from
    thermo_ORR.json.  Falls back to DEFAULT_ORR_CORR if file is missing
    or cannot be parsed.
    """
    ref_dir = Path(ref_dir)
    corr = DEFAULT_ORR_CORR.copy()
    p = ref_dir / THERMO_ORR_NAME
    if not p.is_file():
        return corr
    try:
        data = json.loads(p.read_text())
        zpe = (
            data.get("ΔZPE_ads (eV)")
            or data.get("dZPE_ads (eV)")
            or data.get("dZPE_ads")
            or {}
        )
        for k in corr:
            if k in zpe:
                try:
                    corr[k] = float(zpe[k])
                except Exception:
                    pass
    except Exception:
        pass
    return corr


def _get_ads_box_energies(
    adspecies: tuple[str, ...],
    calc,
    ref_dir: str | Path = "ref_gas",
    steps: int = 200,
    fmax: float = 0.05,
    GROOT: Optional[Path] = None,
) -> dict[str, float]:
    """
    Compute gas-phase 'box' energies for CO2RR/ORR adsorbate templates.

    Reads template CIF files from ref_gas/ (e.g. CO_box.cif, COOH_box.cif),
    relaxes each in a periodic box using the provided calculator, and returns
    a dict of {adsorbate_name: energy_eV}.
    """
    ref_dir = Path(ref_dir)
    E_ads_box: dict[str, float] = {}

    unique_ads = {a.replace("*", "").upper() for a in adspecies}
    for ads_clean in unique_ads:
        if ads_clean not in ADS_TEMPLATE_FILES:
            continue

        cif_name = ADS_TEMPLATE_FILES[ads_clean]
        cif_path = ref_dir / cif_name
        if not cif_path.is_file():
            # OCHO box energy is not used in the current code path; skip if
            # template is missing.  Add the same fallback as _load_ads_template
            # if needed in the future.
            continue

        mol = read(cif_path).copy()
        mol = ensure_pbc3(mol, vac_z=10.0)
        mol.calc = calc

        if steps > 0:
            dyn = BFGS(mol, logfile=None)
            dyn.run(fmax=fmax, steps=steps)

        E_ads_box[ads_clean] = float(mol.get_potential_energy())

        # (Optional) Save relaxed box structure under GROOT/ads_refs/
        try:
            if GROOT is not None:
                out_dir = Path(GROOT) / "ads_refs"
                out_dir.mkdir(parents=True, exist_ok=True)
                write(out_dir / f"{ads_clean}_box_relaxed.cif", mol)
        except Exception:
            pass

    return E_ads_box


# ---------------------------------------------------------------------
# Gas reference helpers (CO2RR)
# ---------------------------------------------------------------------
GAS_REF_FILES: Dict[str, str] = {
    "CO2": "CO2_box.cif",
    "H2O": "H2O_box.cif",
    "CO": "CO_box.cif",
    "H2": "H2_box.cif",
}


def _box_molecule(at, cell: float = 15.0):
    """Place a molecule in a cubic periodic box and center it."""
    at = at.copy()
    at.set_cell([float(cell), float(cell), float(cell)])
    at.set_pbc(True)
    at.center()
    return at


def _get_gas_box_energies(
    species: Iterable[str],
    calc,
    ref_dir: str | Path = "ref_gas",
    steps: int = 200,
    fmax: float = 0.05,
    cell: float = 15.0,
    GROOT: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Compute gas-phase reference energies (CO2, H2O, etc.) in a large box using the same calculator.

    Priority:
      1) ref_gas/*_box.cif if present (reproducible references)
      2) ASE built-in molecules (fallback)
    """
    from ase.build import molecule  # type: ignore

    ref_dir = Path(ref_dir)
    out: Dict[str, float] = {}

    for sp in species:
        sp_u = str(sp).upper()
        mol = None

        cif_name = GAS_REF_FILES.get(sp_u)
        if cif_name is not None:
            cif_path = ref_dir / cif_name
            if cif_path.is_file():
                try:
                    mol = read(cif_path).copy()
                except Exception:
                    mol = None

        if mol is None:
            # ASE molecule() expects names like 'CO2', 'H2O', 'H2', 'CO'
            try:
                mol = molecule(sp_u)
            except Exception as e:
                hint = (
                    f"Provide a valid CIF reference at {ref_dir}/{cif_name}"
                    if cif_name
                    else "Provide a valid gas reference CIF"
                )
                raise RuntimeError(
                    f"Unable to build gas molecule '{sp_u}'. {hint}."
                ) from e

        mol = _box_molecule(mol, cell=cell)
        mol.calc = calc

        if steps > 0:
            dyn = BFGS(mol, logfile=None)
            dyn.run(fmax=fmax, steps=steps)

        out[sp_u] = float(mol.get_potential_energy())

        # Save relaxed references for debugging/reuse
        try:
            if GROOT is not None:
                out_dir = Path(GROOT) / "gas_refs"
                out_dir.mkdir(parents=True, exist_ok=True)
                write(out_dir / f"{sp_u}_box_relaxed.cif", mol)
        except Exception:
            pass

    return out


def _load_or_compute_gas_refs(
    required: Iterable[str],
    calc,
    GROOT: Path,
    ref_dir: str | Path = "ref_gas",
    steps: int = 200,
    fmax: float = 0.05,
    cell: float = 15.0,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Load cached gas refs from GROOT/gas_refs.json; compute missing with _get_gas_box_energies."""
    required_u = [str(s).upper() for s in required]
    cache_path = Path(GROOT) / "gas_refs.json"
    gas_E: Dict[str, float] = {}
    src: Dict[str, str] = {}

    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            if isinstance(cached, dict):
                for k, v in cached.items():
                    try:
                        gas_E[str(k).upper()] = float(v)
                        src[str(k).upper()] = "cache"
                    except Exception:
                        pass
        except Exception:
            pass

    missing = [s for s in required_u if s not in gas_E]
    if missing:
        computed = _get_gas_box_energies(
            missing,
            calc,
            ref_dir=ref_dir,
            steps=steps,
            fmax=fmax,
            cell=cell,
            GROOT=GROOT,
        )
        for k, v in computed.items():
            gas_E[str(k).upper()] = float(v)
            src[str(k).upper()] = "computed"
        try:
            cache_path.write_text(json.dumps(gas_E, indent=2))
        except Exception:
            pass

    return gas_E, src


# --- Relaxation steps ---
STEPS_FAST = 300
STEPS_NORMAL = 600
STEPS_TIGHT = 900


def _choose_steps(heavy: bool, relax_mode: str) -> Tuple[int, int, int, int]:
    """
    Determine relaxation step counts for slab/ads/H2 based on system size
    and the chosen relaxation mode.

    Returns
    -------
    (slab_steps, z_steps, free_steps, h2_steps)
    """
    mode = relax_mode.lower()
    if mode not in ("fast", "normal", "tight"):
        raise ValueError(f"Unknown relax_mode='{relax_mode}'")

    base_steps = STEPS_NORMAL
    if mode == "fast":
        base_steps = STEPS_FAST
    elif mode == "tight":
        base_steps = STEPS_TIGHT

    if heavy and mode == "tight":
        slab_steps = STEPS_NORMAL
    else:
        slab_steps = base_steps

    z_steps = base_steps
    free_steps = base_steps
    h2_steps = base_steps

    return slab_steps, z_steps, free_steps, h2_steps


def _check_duplicate_convergence(
    current_pos: np.ndarray,
    previous_positions: List[dict[str, object] | np.ndarray],
    tol: float = 0.15,
    *,
    cell=None,
    pbc=None,
    current_kind: str | None = None,
    current_surface_indices: tuple[int, ...] | None = None,
) -> bool:
    """Duplicate-site detection.

    Backward compatible with legacy position-only lists, but can also use
    basin identity (site kind + surface indices) plus MIC-corrected XY distance.
    """
    cur_kind = _canonicalize_metal_kind(current_kind) if current_kind is not None else None
    cur_sidx = tuple(int(i) for i in (current_surface_indices or ())) if current_surface_indices else tuple()

    for prev in previous_positions:
        if isinstance(prev, dict):
            prev_pos = np.asarray(prev.get("pos", [np.nan, np.nan, np.nan]), dtype=float)
            prev_kind = prev.get("kind")
            prev_sidx = tuple(int(i) for i in (prev.get("surface_indices") or ())) if prev.get("surface_indices") else tuple()
            if cur_kind is not None and prev_kind is not None:
                if _canonicalize_metal_kind(prev_kind) == cur_kind and prev_sidx and cur_sidx and prev_sidx == cur_sidx:
                    return True
            if cell is not None and pbc is not None and np.isfinite(prev_pos[:2]).all() and np.isfinite(current_pos[:2]).all():
                _, dxy = _mic_xy_delta(cell, pbc, np.asarray(prev_pos[:2], dtype=float), np.asarray(current_pos[:2], dtype=float))
                dz = float(abs(float(prev_pos[2]) - float(current_pos[2]))) if np.isfinite(prev_pos[2]) and np.isfinite(current_pos[2]) else 0.0
                if dxy < tol and dz < max(0.25, tol):
                    return True
            dist = np.linalg.norm(current_pos - prev_pos)
            if dist < tol:
                return True
        else:
            prev_pos = np.asarray(prev, dtype=float)
            dist = np.linalg.norm(current_pos - prev_pos)
            if dist < tol:
                return True
    return False


# ---------------------------------------------------------------------
# Site coordinate helpers
# ---------------------------------------------------------------------
def site_xy_by_layers_metal(at) -> Dict[str, np.ndarray]:
    """Extract representative ontop/bridge/hollow xy coordinates from a metal (111)-like slab.

    Note: for metal surfaces we currently expose a single 3-fold hollow basin
    rather than distinguishing fcc vs hcp.
    """
    try:
        candidates = detect_metal_111_sites(at, max_sites_per_kind=200)
    except Exception:
        candidates = []

    pos = at.get_positions()
    cell = at.get_cell()
    try:
        center_xy = np.asarray(pos[:, :2].mean(axis=0), dtype=float)
    except Exception:
        center_xy = np.zeros(2, dtype=float)

    reps: Dict[str, np.ndarray] = {}
    for want in ("ontop", "bridge", "hollow"):
        group = [s for s in candidates if _canonicalize_metal_kind(getattr(s, "kind", want)) == want]
        if not group:
            continue
        best = None
        best_key = None
        for idx, site in enumerate(group):
            cand_xy = np.asarray(site.position[:2], dtype=float)
            _, dist = _mic_xy_delta(cell, at.get_pbc(), center_xy, cand_xy)
            key = (float(dist), int(idx))
            if best is None or key < best_key:
                best = cand_xy
                best_key = key
        if best is not None:
            reps[want] = np.asarray(best, dtype=float)

    return reps


def site_xy_by_layers_oxide(at) -> Dict[str, np.ndarray]:
    """
    Extract xy site coordinates from the top-layer cations of a generic
    oxide slab.
    """
    pos = at.get_positions()
    top_idx = layer_indices(at, n=1)[0]

    top_metal = [i for i in top_idx if at[i].symbol != "O"]
    use_idx = top_metal if len(top_metal) >= 2 else list(top_idx)

    if len(use_idx) == 0:
        z_sorted = np.argsort(pos[:, 2])[::-1]
        use_idx = z_sorted[:3].tolist()

    xy_cand = pos[use_idx][:, :2]
    ref = xy_cand[0]
    d = np.linalg.norm(xy_cand - ref, axis=1)
    order = np.argsort(d)
    if len(order) < 3:
        order = np.concatenate([order, np.tile(order[-1], 3 - len(order))])

    i0, i1, i2 = order[0], order[1], order[2]
    ontop_xy = xy_cand[i0]
    bridge_xy = 0.5 * (xy_cand[i0] + xy_cand[i1])
    hollow_xy = (xy_cand[i0] + xy_cand[i1] + xy_cand[i2]) / 3.0

    return {
        "fcc": hollow_xy,
        "hcp": hollow_xy,
        "bridge": bridge_xy,
        "ontop": ontop_xy,
    }


# ---------------------------------------------------------------------
# Common slab preparation
# ---------------------------------------------------------------------
def _prepare_slab(
    user_slab_cif: str,
    out_root: Path,
    vac_z: float,
    relax_mode: str,
):
    out_root.mkdir(parents=True, exist_ok=True)
    GROOT = out_root / "gas"
    UROOT = out_root / "sample"
    for d in (GROOT, UROOT / "slab", UROOT / "sites"):
        d.mkdir(parents=True, exist_ok=True)

    # Slab PBC + vacuum correction
    slab_u_raw = ensure_pbc3(read(user_slab_cif), vac_z=vac_z)
    write(UROOT / "slab/user_slab_raw.cif", slab_u_raw)
    vac_warning = bool(vac_z < VAC_WARN_MIN)

    n_atoms = len(slab_u_raw)
    heavy = n_atoms > 120
    slab_steps, z_steps, free_steps, h2_steps = _choose_steps(heavy, relax_mode)

    # H2 ref
    h2_rel, E_H2 = get_h2_ref(calc, "ref_gas", h2_steps, 0.03, 10.0)
    write(GROOT / "H2_box.cif", h2_rel)

    # slab relaxation
    slab_u_rel, E_slab_u = relax_freeH(
        slab_u_raw,
        steps=slab_steps,
        fmax=0.03,
        relaxation_scope="partial",
        n_fix_layers=2,
    )
    write(UROOT / "slab/user_slab_relaxed.cif", slab_u_rel)

    # Check energy difference before/after relaxation
    _, E_slab_u_raw = relax_freeH(
        slab_u_raw,
        steps=0,
        fmax=0.03,
        relaxation_scope="partial",
        n_fix_layers=2,
    )
    slab_relax_drop = bool(abs(E_slab_u - E_slab_u_raw) > 0.60)

    n_top_u = len(layer_indices(slab_u_rel, n=1)[0])

    meta_flags = {
        "slab_relax_drop": slab_relax_drop,
        "vac_warning": vac_warning,
        "n_top": n_top_u,
        "n_atoms": n_atoms,
    }

    return (
        slab_u_rel,
        E_slab_u,
        E_H2,
        slab_steps,
        z_steps,
        free_steps,
        h2_steps,
        meta_flags,
        GROOT,
        UROOT,
    )


def _build_target_sites(
    mtype: str,
    slab_u_rel,
    sites: Iterable[str],
    user_ads_sites: Optional[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """
    Build the final list of target-site records.

    Each record preserves the human-facing site label together with the seed
    metadata needed for post-relaxation tracking.
    """
    target_sites: List[Dict[str, object]] = []
    mtype_norm = str(mtype).lower()
    site_names = [str(s) for s in (sites or ())]

    if user_ads_sites:
        projected_sites = user_ads_sites
        if mtype_norm == "oxide":
            projected_sites = _project_oxide_her_sites_to_otop(
                slab_u_rel,
                user_ads_sites,
                dz=0.0,
                extra_z=0.0,
            )
        for label, site in projected_sites.items():
            kind = getattr(site, "kind", "unknown")
            pos = np.asarray(getattr(site, "position", []), dtype=float)
            surface_indices = tuple(int(i) for i in getattr(site, "surface_indices", ()) or ())
            if pos.shape[0] < 2:
                continue
            xyz = (
                np.asarray(pos[:3], dtype=float)
                if pos.shape[0] >= 3
                else np.asarray([pos[0], pos[1], np.nan], dtype=float)
            )
            target_sites.append({
                "site_label": str(label),
                "site_kind": str(kind),
                "xy": np.asarray(pos[:2], dtype=float),
                "initial_xyz": xyz,
                "surface_indices": surface_indices,
                "seed_source": "user_ads_sites_projected" if mtype_norm == "oxide" else "user_ads_sites",
            })
        return target_sites

    if mtype_norm == "metal":
        xy_map = site_xy_by_layers_metal(slab_u_rel)
        sites_iter = list(site_names)
        if not sites_iter:
            sites_iter = ["ontop", "bridge", "hollow"]
        for site_name in sites_iter:
            if site_name in xy_map:
                xy = np.asarray(xy_map[site_name], dtype=float)
                target_sites.append({
                    "site_label": str(site_name),
                    "site_kind": str(site_name),
                    "xy": xy,
                    "initial_xyz": np.asarray([xy[0], xy[1], np.nan], dtype=float),
                    "surface_indices": tuple(),
                    "seed_source": "geometry_default",
                })
        return target_sites

    if mtype_norm == "oxide":
        max_sites = max(1, len(site_names)) if site_names else 6
        oxide_sites = _generate_oxide_her_oanchor_sites(
            slab_u_rel,
            max_sites=int(max_sites),
        )
        oxide_sites = _project_oxide_her_sites_to_otop(
            slab_u_rel,
            oxide_sites,
            dz=0.0,
            extra_z=0.0,
        )

        for idx, site in enumerate(oxide_sites or [], start=1):
            pos = np.asarray(getattr(site, "position", []), dtype=float)
            if pos.shape[0] < 2:
                continue
            xyz = (
                np.asarray(pos[:3], dtype=float)
                if pos.shape[0] >= 3
                else np.asarray([pos[0], pos[1], np.nan], dtype=float)
            )
            target_sites.append({
                "site_label": f"o_top_{idx}",
                "site_kind": str(getattr(site, "kind", "o_top")),
                "xy": np.asarray(pos[:2], dtype=float),
                "initial_xyz": xyz,
                "surface_indices": tuple(int(i) for i in getattr(site, "surface_indices", ()) or ()),
                "seed_source": "oxide_o_anchor",
            })

        if target_sites:
            return target_sites

        xy_map = site_xy_by_layers_oxide(slab_u_rel)
        for site_name in site_names:
            if site_name not in xy_map:
                continue
            xy = np.asarray(xy_map[site_name], dtype=float)
            target_sites.append({
                "site_label": str(site_name),
                "site_kind": str(site_name),
                "xy": xy,
                "initial_xyz": np.asarray([xy[0], xy[1], np.nan], dtype=float),
                "surface_indices": tuple(),
                "seed_source": "geometry_default_fallback",
            })
        return target_sites

    raise ValueError(f"Unknown mtype='{mtype}'.")


# ---------------------------------------------------------------------
# HER mode
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# HER mode
# ---------------------------------------------------------------------

def _run_her_che(
    mtype: str,
    user_slab_cif: str,
    out_root: Path,
    sites: Iterable[str],
    vac_z: float,
    layers: int,
    export_absolute: bool,
    use_net_corr: bool,
    gas: str,
    relax_mode: str,
    user_ads_sites: Optional[Mapping[str, object]],
    her_guardrail: bool = False,
    her_site_preference: str = "ontop",
    her_use_net_corr: bool = True,
    thermo_mode: str = "CHE correction (fast screening)",
    zpe_target_mode: str = "Best-ranked by CHE",
    zpe_target_label: str | None = None,
    local_zpe_cutoff: float = 2.5,
    local_zpe_max_neighbors: int = 3,
    oxide_descriptor_mode: str = "Basic HER screening",
    oxide_descriptor_max_reactive_per_kind: int = 2,
    oxide_descriptor_pair_limit: int = 6,
    her_relaxation_scope: str = "partial",
    her_n_fix_layers: int = 2,
):
    if gas != "H2":
        raise NotImplementedError("HER CHE_mode currently supports only 'H2' gas.")

    (
        slab_u_rel,
        E_slab_u,
        E_H2,
        slab_steps,
        z_steps,
        free_steps,
        h2_steps,
        meta_flags,
        GROOT,
        UROOT,
    ) = _prepare_slab(user_slab_cif, out_root, vac_z, relax_mode)

    net_corr = STANDARD_CHE_CORR if use_net_corr else 0.0

    rows: List[Dict[str, object]] = []
    final_h_positions: List[dict[str, object] | np.ndarray] = []

    target_sites = _build_target_sites(mtype, slab_u_rel, sites, user_ads_sites)

    if not target_sites:
        raise RuntimeError(f"No target adsorption sites were built for mtype='{mtype}'.")

    if str(her_relaxation_scope).strip().lower() == "auto":
        resolved_her_relaxation_scope = "rigid" if str(mtype).lower() == "oxide" else "partial"
    else:
        resolved_her_relaxation_scope = normalize_relaxation_scope(her_relaxation_scope)

    # --- Optional HER guardrail (single-site; cheap) ---
    her_guard = None
    if bool(her_guardrail):
        try:
            z_steps_g = int(min(int(z_steps), 120))
            free_steps_g = int(min(int(free_steps), 150))
            out_cif_g = UROOT / "sites/user_guardrail_H.cif"
            her_guard = _compute_her_guardrail_from_prepared(
                slab_u_rel=slab_u_rel,
                E_slab_u=float(E_slab_u),
                E_H2=float(E_H2),
                target_sites=target_sites,
                z_steps=z_steps_g,
                free_steps=free_steps_g,
                site_preference=str(her_site_preference),
                use_net_corr=bool(her_use_net_corr),
                out_cif=out_cif_g,
                relaxation_scope=resolved_her_relaxation_scope,
                n_fix_layers=int(her_n_fix_layers),
            )
            if her_guard is not None:
                her_guard["z_steps_used"] = int(z_steps_g)
                her_guard["free_steps_used"] = int(free_steps_g)
                try:
                    hg_csv = out_root / "results_her_guardrail.csv"
                    pd.DataFrame([her_guard]).to_csv(
                        hg_csv, index=False, float_format="%.6f"
                    )
                    her_guard["csv_path"] = str(Path(hg_csv).resolve())
                except Exception:
                    pass
        except Exception as e:
            her_guard = {"mode": "HER_GUARDRAIL", "error": str(e)}
    descriptor_primary_results = (
        str(mtype).lower() == "oxide"
        and str(oxide_descriptor_mode) != "Basic HER screening"
    )

    if descriptor_primary_results:
        # Oxide descriptor mode is now D2-primary.  The legacy O-anchor HER
        # candidate loop is intentionally skipped, so anion_ontop rows do not
        # appear in the primary results table.  D1 is still evaluated inside
        # run_oxide_descriptor_profile() when Full 2-stage mode is selected.
        try:
            oxide_descriptor_summary = run_oxide_descriptor_profile(
                slab_u_rel=slab_u_rel,
                E_slab_u=float(E_slab_u),
                E_H2=float(E_H2),
                d1_rows_df=pd.DataFrame(),
                d1_targets=target_sites,
                out_root=out_root,
                z_steps=int(z_steps),
                free_steps=int(free_steps),
                use_net_corr=bool(use_net_corr),
                descriptor_mode=str(oxide_descriptor_mode),
                max_reactive_per_kind=int(oxide_descriptor_max_reactive_per_kind),
                pair_limit=int(oxide_descriptor_pair_limit),
                # Fixed oxide-descriptor policy:
                # D1 remains constrained/rigid; D2_Hreact is forced to partial
                # inside oxide_descriptor.py. Passing "rigid" here prevents
                # legacy UI selections from changing oxide descriptor semantics.
                relaxation_scope="rigid",
                n_fix_layers=int(her_n_fix_layers),
                thermo_mode=str(thermo_mode),
                local_zpe_cutoff=float(local_zpe_cutoff),
                local_zpe_max_neighbors=int(local_zpe_max_neighbors),
                standard_che_corr=float(STANDARD_CHE_CORR),
            )
        except Exception as e:
            oxide_descriptor_summary = {
                "error": str(e),
                "descriptor_mode": str(oxide_descriptor_mode),
                "D2_seed_quality": "failed",
                "D2_seed_warning": str(e),
            }

        df = _oxide_d2_summary_to_primary_results_df(
            oxide_descriptor_summary,
            net_corr=float(net_corr),
        )
        if df.empty:
            df = pd.DataFrame([{
                "mode": "HER",
                "result_role": "oxide_D2_primary",
                "descriptor": "D2",
                "site_label": "D2_unresolved",
                "requested_site": "metal_cation_Hstar",
                "initial_geom_site": "metal_cation_Hstar",
                "relaxed_site": "unresolved",
                "final_site_kind": "unresolved",
                "binding_class": "unresolved",
                "D2_descriptor_valid": False,
                "D2_selection_rule": "min_abs_deltaG_among_valid_metal_centered_Hstar",
                "thermo_mode": str(thermo_mode),
                "ΔG_H_CHE (eV)": np.nan,
                "ΔG_H_local (eV)": np.nan,
                "ΔG_H (eV)": np.nan,
                "ΔE_H_user (eV)": np.nan,
                "local_thermo_corr (eV)": np.nan,
                "local_zpe_status": "not_available",
                "H_lateral_disp(Å)": np.nan,
                "is_duplicate": False,
                "qc_flags": str((oxide_descriptor_summary or {}).get("error", "D2_primary_result_unavailable")),
            }])

        out_csv = out_root / "results_sites_her.csv"
        df.to_csv(out_csv, index=False, float_format="%.6f")

        meta = {
            "mode": "HER",
            "user_slab": str(Path(user_slab_cif).resolve()),
            "relax_mode": relax_mode,
            "steps": {"slab": slab_steps, "H": z_steps, "H2": h2_steps},
            "HER_RELAXATION_SCOPE": str(resolved_her_relaxation_scope),
            "HER_N_FIX_LAYERS": int(her_n_fix_layers),
            "thermo": {"NET_CORR": net_corr, "standard": f"{STANDARD_CHE_CORR:.2f} eV"},
            "E_H2": E_H2,
            "HER_GUARDRAIL": her_guard,
            "thermo_mode": str(thermo_mode),
            "zpe_target_mode": str(zpe_target_mode),
            "zpe_target_label": zpe_target_label,
            "local_zpe_cutoff": float(local_zpe_cutoff),
            "local_zpe_max_neighbors": int(local_zpe_max_neighbors),
            "OXIDE_DESCRIPTOR_MODE": str(oxide_descriptor_mode),
            "OXIDE_DESCRIPTOR_PRIMARY_RESULTS": True,
            "OXIDE_DESCRIPTOR_LEGACY_OANCHOR_SKIPPED": True,
            "OXIDE_DESCRIPTOR_SUMMARY": oxide_descriptor_summary,
            "OXIDE_DESCRIPTOR_D2_CANDIDATES_CSV": (oxide_descriptor_summary or {}).get("D2_candidates_csv", ""),
            "OXIDE_DESCRIPTOR_D3_CANDIDATES_CSV": (oxide_descriptor_summary or {}).get("D3_candidates_csv", ""),
            "OXIDE_DESCRIPTOR_CAUTION": (oxide_descriptor_summary or {}).get("caution", ""),
            "Model": MODEL_NAME,
            "Device": DEVICE,
            "warnings": {
                "slab_relax_drop": meta_flags["slab_relax_drop"],
                "vac_warning": meta_flags["vac_warning"],
            },
            "OER_SLAB_QC": oer_slab_qc,
        }
        (out_root / "meta_her.json").write_text(json.dumps(meta, indent=2))
        return str(out_csv), meta

    for site_seed in target_sites:
        label = str(site_seed.get("site_label", "unknown"))
        kind = str(site_seed.get("site_kind", "unknown"))
        xy = np.asarray(site_seed.get("xy", [np.nan, np.nan]), dtype=float)

        anchor_xyz = np.asarray(site_seed.get("initial_xyz", [np.nan, np.nan, np.nan]), dtype=float)
        if str(mtype).lower() == "metal":
            Au, E_uH, _disp_raw, relax_meta = site_energy_two_stage(
                slab_u_rel,
                xy,
                H0S,
                z_steps,
                free_steps,
                return_meta=True,
                mtype="metal",
            )
        else:
            h0s_used = recommend_h0s_for_relaxation(
                resolved_her_relaxation_scope,
                mtype=mtype,
                site_kind=kind,
                default_h0s=H0S,
            )
            Au, E_uH, _disp_raw, relax_meta = site_energy_two_stage(
                slab_u_rel,
                xy,
                h0s_used,
                z_steps,
                free_steps,
                relaxation_scope=resolved_her_relaxation_scope,
                n_fix_layers=int(her_n_fix_layers),
                return_meta=True,
                anchor_xyz=anchor_xyz,
                mtype=mtype,
                site_kind=kind,
                min_clearance=default_min_clearance_for_scope(resolved_her_relaxation_scope, mtype=mtype),
            )

        dE_u = E_uH - E_slab_u - 0.5 * E_H2
        dG_u = dE_u + net_corr

        h_pos_final = Au.get_positions()[-1]
        slab_only = Au[: len(slab_u_rel)]
        tracking = _resolve_site_tracking(
            slab_only=slab_only,
            mtype=mtype,
            seed_kind=kind,
            initial_xy=xy,
            final_anchor_xy=np.asarray(h_pos_final[:2], dtype=float),
            classification_mode="oxide_anion" if str(mtype).lower() == "oxide" else "auto",
            disp_threshold=float(MIGRATE_THR),
        )
        final_meta = tracking.get("final_site_meta", {}) if isinstance(tracking, dict) else {}
        is_duplicate = _check_duplicate_convergence(
            h_pos_final,
            final_h_positions,
            cell=slab_only.get_cell(),
            pbc=slab_only.get_pbc(),
            current_kind=str(final_meta.get("kind", tracking.get("final_site_kind", kind))),
            current_surface_indices=tuple(final_meta.get("surface_indices", ()) or ()),
        )
        if not is_duplicate:
            final_h_positions.append({
                "pos": np.asarray(h_pos_final, dtype=float),
                "kind": str(final_meta.get("kind", tracking.get("final_site_kind", kind))),
                "surface_indices": tuple(final_meta.get("surface_indices", ()) or ()),
            })

        qc_flags = list(tracking["qc_flags"])
        if is_duplicate:
            qc_flags.append("duplicate_final_site")
        tracking["initial_surface_indices"] = _fmt_surface_indices(site_seed.get("surface_indices", ()))

        row: Dict[str, object] = {
            "mode": "HER",
            "site": kind,
            "site_label": label,
            "seed_source": site_seed.get("seed_source"),
            "structure_cif": str((UROOT / f"sites/user_{label}_H.cif").resolve()),
            "n_slab_atoms": int(meta_flags["n_atoms"]),
            "thermo_mode": str(thermo_mode),
            "her_relaxation_scope": str(resolved_her_relaxation_scope),
            "her_n_fix_layers": int(her_n_fix_layers),
            "selected_h0": _safe_float((relax_meta or {}).get("selected_h0")),
            "h0_candidates": str((relax_meta or {}).get("h0_candidates", "")),
            "placement_mode": str((relax_meta or {}).get("placement_mode", "slab_top")),
            "z_relax_n_steps": int((relax_meta or {}).get("z_relax_n_steps", 0)),
            "z_relax_converged": (relax_meta or {}).get("z_relax_converged", None),
            "z_relax_relaxed_atoms": int((relax_meta or {}).get("z_relax_relaxed_atoms", 0)),
            "fine_relax_n_steps": int((relax_meta or {}).get("fine_relax_n_steps", 0)),
            "fine_relax_converged": (relax_meta or {}).get("fine_relax_converged", None),
            "fine_relax_relaxed_atoms": int((relax_meta or {}).get("fine_relax_relaxed_atoms", 0)),
            "total_relax_n_steps": int((relax_meta or {}).get("total_relax_n_steps", 0)),
            "ΔG_H_CHE (eV)": float(dG_u),
            "ΔG_H_local (eV)": np.nan,
            "local_thermo_corr (eV)": np.nan,
            "zpe_scope": "none",
            "zpe_selected_atoms": "",
            "zpe_warning": "",
            "seed_site_kind": tracking["seed_site_kind"],
            "initial_site_kind": tracking["initial_site_kind"],
            "final_site_kind": tracking["final_site_kind"],
            "relaxed_site": tracking["final_site_kind"],
            "site_tracking_mode": tracking["classification_mode"],
            "site_family": tracking["site_family"],
            "migration_basis": tracking["migration_basis"],
            "migration_type": tracking["migration_type"],
            "initial_surface_indices": tracking["initial_surface_indices"],
            "final_surface_indices": tracking["final_surface_indices"],
            "qc_flags": ";".join(qc_flags),
            "MODEL": MODEL_NAME,
            "DEVICE": DEVICE,
            "layers": layers,
            "vac_z(Å)": vac_z,
            "E_slab_user (eV)": E_slab_u,
            "E_slab+H_user (eV)": E_uH,
            "ΔE_H_user (eV)": dE_u,
            "E_slab (eV)": E_slab_u,
            "ΔE_H (eV)": dE_u,
            "initial_anchor_x(Å)": tracking["initial_anchor_x(Å)"],
            "initial_anchor_y(Å)": tracking["initial_anchor_y(Å)"],
            "final_anchor_x(Å)": tracking["final_anchor_x(Å)"],
            "final_anchor_y(Å)": tracking["final_anchor_y(Å)"],
            "H_lateral_disp(Å)": float(tracking["lateral_disp_mic(Å)"]),
            "final_site_match_dist(Å)": float(tracking["final_site_match_dist(Å)"]),
            "migrated": bool(tracking["migrated"]),
            "is_duplicate": is_duplicate,
            "slab_relax_drop": meta_flags["slab_relax_drop"],
            "vac_warning": meta_flags["vac_warning"],
        }

        if export_absolute:
            row["ΔG_H (eV)"] = dG_u

        rows.append(row)
        write(UROOT / f"sites/user_{label}_H.cif", Au)

    # Second-pass HER thermochemistry refinement
    if rows and str(thermo_mode) != "CHE correction (fast screening)":
        df0 = pd.DataFrame(rows)
        selected_idx = []

        if str(thermo_mode) == "Local ZPE correction (selected structure)":
            if str(zpe_target_mode) == "User-selected site label" and zpe_target_label:
                mask = df0["site_label"].astype(str) == str(zpe_target_label)
                selected_idx = df0.index[mask].tolist()[:1]
            if not selected_idx:
                selected_idx = df0["ΔG_H_CHE (eV)"].abs().sort_values().index.tolist()[:1]
        elif str(thermo_mode) == "Local ZPE correction (all structures)":
            selected_idx = df0.index.tolist()

        for ridx in selected_idx:
            try:
                row = rows[int(ridx)]
                cif_path = Path(str(row["structure_cif"]))
                if not cif_path.is_file():
                    row["zpe_warning"] = "structure_cif_missing"
                    continue

                slab_ads = read(str(cif_path))
                zpe_meta = compute_local_h_thermo_correction(
                    slab_ads,
                    n_slab_atoms=int(row["n_slab_atoms"]),
                    cutoff=float(local_zpe_cutoff),
                    max_neighbors=int(local_zpe_max_neighbors),
                )
                local_corr = float(zpe_meta["local_corr_eV"])
                row["ΔG_H_local (eV)"] = float(row["ΔE_H_user (eV)"] + local_corr)
                row["local_thermo_corr (eV)"] = local_corr
                row["zpe_scope"] = "selected" if str(thermo_mode) == "Local ZPE correction (selected structure)" else "all"
                row["zpe_selected_atoms"] = ",".join(str(i) for i in zpe_meta.get("selected_indices", []))
                row["zpe_warning"] = ";".join(zpe_meta.get("warnings", [])) if zpe_meta.get("warnings") else ""
            except Exception as e:
                rows[int(ridx)]["zpe_warning"] = f"local_zpe_failed:{e}"

    df = pd.DataFrame(rows)
    oxide_descriptor_summary = None

    if str(thermo_mode) == "CHE correction (fast screening)":
        df["ΔG_H (eV)"] = df["ΔG_H_CHE (eV)"]
    elif str(thermo_mode) == "Local ZPE correction (selected structure)":
        df["ΔG_H (eV)"] = df["ΔG_H_local (eV)"].where(df["ΔG_H_local (eV)"].notna(), df["ΔG_H_CHE (eV)"])
    else:
        df["ΔG_H (eV)"] = df["ΔG_H_local (eV)"].where(df["ΔG_H_local (eV)"].notna(), df["ΔG_H_CHE (eV)"])

    if export_absolute and "ΔG_H (eV)" in df.columns:
        df = df.assign(abs_val=lambda x: x["ΔG_H (eV)"].abs()).sort_values(["abs_val"])
    else:
        df = df.assign(abs_val=lambda x: x["ΔE_H_user (eV)"].abs()).sort_values(["abs_val"])

    if str(mtype).lower() == 'oxide' and str(oxide_descriptor_mode) != 'Basic HER screening':
        try:
            oxide_descriptor_summary = run_oxide_descriptor_profile(
                slab_u_rel=slab_u_rel,
                E_slab_u=float(E_slab_u),
                E_H2=float(E_H2),
                d1_rows_df=df.drop(columns=["abs_val"], errors="ignore").copy(),
                d1_targets=target_sites,
                out_root=out_root,
                z_steps=int(z_steps),
                free_steps=int(free_steps),
                use_net_corr=bool(use_net_corr),
                descriptor_mode=str(oxide_descriptor_mode),
                max_reactive_per_kind=int(oxide_descriptor_max_reactive_per_kind),
                pair_limit=int(oxide_descriptor_pair_limit),
                relaxation_scope=str(resolved_her_relaxation_scope),
                n_fix_layers=int(her_n_fix_layers),
                thermo_mode=str(thermo_mode),
                local_zpe_cutoff=float(local_zpe_cutoff),
                local_zpe_max_neighbors=int(local_zpe_max_neighbors),
                standard_che_corr=float(STANDARD_CHE_CORR),
            )
        except Exception as e:
            oxide_descriptor_summary = {'error': str(e), 'descriptor_mode': str(oxide_descriptor_mode)}
            if str(oxide_descriptor_mode) in {'D3_pair only (H2 pairing proxy)', 'Full 3-stage profile (experimental)'}:
                oxide_descriptor_summary['caution'] = THREE_STAGE_OXIDE_HER_CAUTION

    out_csv = out_root / "results_sites_her.csv"
    df.drop(columns=["abs_val"], errors="ignore").to_csv(
        out_csv,
        index=False,
        float_format="%.6f",
    )

    meta = {
        "mode": "HER",
        "user_slab": str(Path(user_slab_cif).resolve()),
        "relax_mode": relax_mode,
        "steps": {"slab": slab_steps, "H": z_steps, "H2": h2_steps},
        "HER_RELAXATION_SCOPE": str(resolved_her_relaxation_scope),
        "HER_N_FIX_LAYERS": int(her_n_fix_layers),
        "OXIDE_DESCRIPTOR_RELAXATION_POLICY": "D1=constrained/rigid O-site OH; D2_Hreact=partial metal-cation-centered H*",
        "thermo": {"NET_CORR": net_corr, "standard": f"{STANDARD_CHE_CORR:.2f} eV"},
        "E_H2": E_H2,
        "HER_GUARDRAIL": her_guard,
        "thermo_mode": str(thermo_mode),
        "zpe_target_mode": str(zpe_target_mode),
        "zpe_target_label": zpe_target_label,
        "local_zpe_cutoff": float(local_zpe_cutoff),
        "local_zpe_max_neighbors": int(local_zpe_max_neighbors),
        "OXIDE_DESCRIPTOR_MODE": str(oxide_descriptor_mode),
        "OXIDE_DESCRIPTOR_SUMMARY": oxide_descriptor_summary,
        "OXIDE_DESCRIPTOR_D2_CANDIDATES_CSV": (oxide_descriptor_summary or {}).get("D2_candidates_csv", ""),
        "OXIDE_DESCRIPTOR_D3_CANDIDATES_CSV": (oxide_descriptor_summary or {}).get("D3_candidates_csv", ""),
        "OXIDE_DESCRIPTOR_CAUTION": (oxide_descriptor_summary or {}).get("caution", ""),
        "Model": MODEL_NAME,
        "Device": DEVICE,
        "warnings": {
            "slab_relax_drop": meta_flags["slab_relax_drop"],
            "vac_warning": meta_flags["vac_warning"],
        },
        "OER_SLAB_QC": oer_slab_qc,
    }
    (out_root / "meta_her.json").write_text(json.dumps(meta, indent=2))

    return str(out_csv), meta


# ---------------------------------------------------------------------
# CO2RR / ORR: adsorbate template loading and slab placement
# ---------------------------------------------------------------------
def _load_ads_template(ads: str, ref_dir: str | Path = "ref_gas"):
    """
    Load an adsorbate template CIF from ref_gas/ and normalize its
    coordinates so that the anchor atom sits at the origin.

    Anchor selection:
      - CO / COOH: C atom
      - HCOO: O-O midpoint (bidentate starting geometry)
      - OCHO: O with lowest z (O-anchored intermediate)

    Atoms below the anchor plane (z < 0) are reflected to z > 0 to
    reduce the chance of any atom penetrating below the slab top.

    If the OCHO template is missing, HCOO_box.cif is reused with atoms
    reordered to O-C-H-O as a fallback.
    """
    ads_clean = ads.replace("*", "").upper()
    if ads_clean not in ADS_TEMPLATE_FILES:
        raise ValueError(f"Unsupported adsorbate '{ads}'")

    ref_dir = Path(ref_dir)
    cif_path = ref_dir / ADS_TEMPLATE_FILES[ads_clean]

    # --- OCHO fallback: if OCHO template missing, reuse HCOO_box and reorder to O-C-H-O ---
    if ads_clean == "OCHO" and (not cif_path.is_file()):
        fallback = ref_dir / ADS_TEMPLATE_FILES.get("HCOO", "HCOO_box.cif")
        if not fallback.is_file():
            raise FileNotFoundError(
                f"Adsorbate template not found: {cif_path} (and no HCOO fallback at {fallback})"
            )
        a = read(fallback).copy()
        syms = a.get_chemical_symbols()
        idx_O = [i for i, s in enumerate(syms) if s == "O"]
        idx_C = [i for i, s in enumerate(syms) if s == "C"]
        idx_H = [i for i, s in enumerate(syms) if s == "H"]
        if len(syms) == 4 and len(idx_O) >= 2 and len(idx_C) >= 1 and len(idx_H) >= 1:
            order = [idx_O[0], idx_C[0], idx_H[0], idx_O[1]]
            a = a[order]
        # else: keep as-is; downstream anchor selection remains robust
    else:
        if not cif_path.is_file():
            raise FileNotFoundError(f"Adsorbate template not found: {cif_path}")
        a = read(cif_path).copy()

    pos = a.get_positions()
    symbols = a.get_chemical_symbols()

    def _rodrigues_rotate(P: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        # Rodrigues rotation formula
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=float,
        )
        R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        return (R @ P.T).T

    # ---------------- anchor selection ----------------
    anchor_pos = None
    anchor_mode = "C"

    if ads_clean == "HCOO":
        o_idx = [i for i, s in enumerate(symbols) if s == "O"]
        if len(o_idx) >= 2:
            # O-O midpoint anchor (better for bidentate starting geometry)
            anchor_pos = pos[o_idx[:2]].mean(axis=0)
            anchor_mode = "O2_mid"

            # Rotate so that O-O vector lies in the xy plane (v_z -> 0)
            v = pos[o_idx[1]] - pos[o_idx[0]]
            v_proj = np.array([v[0], v[1], 0.0], dtype=float)
            nv = np.linalg.norm(v)
            nvproj = np.linalg.norm(v_proj)
            if nv > 1e-8 and nvproj > 1e-8:
                v_n = v / nv
                vp_n = v_proj / nvproj
                axis = np.cross(v_n, vp_n)
                na = np.linalg.norm(axis)
                if na > 1e-8:
                    axis = axis / na
                    angle = float(
                        np.arccos(np.clip(np.dot(v_n, vp_n), -1.0, 1.0))
                    )
                    pos = _rodrigues_rotate(pos - anchor_pos, axis, angle) + anchor_pos
        elif len(o_idx) == 1:
            anchor_pos = pos[o_idx[0]].copy()
            anchor_mode = "O"
        else:
            anchor_pos = pos[0].copy()
            anchor_mode = "atom0"

    elif ads_clean == "OCHO":
        # O-anchored convention: pick the O closest to slab (min-z) as anchor
        o_idx = [i for i, s in enumerate(symbols) if s == "O"]
        if len(o_idx) >= 1:
            if len(o_idx) > 1:
                iz_local = int(np.argmin(pos[np.asarray(o_idx, int), 2]))
                iz = int(o_idx[iz_local])
            else:
                iz = int(o_idx[0])
            anchor_pos = pos[iz].copy()
            anchor_mode = "O(min_z)"
        else:
            anchor_pos = pos[0].copy()
            anchor_mode = "atom0"

    elif ads_clean in ("O", "OH", "OOH"):
        # ORR/OER oxygen intermediates must be O-anchored.  For OOH*,
        # the proximal O is taken as the O atom with the lowest z in the
        # template.  This makes the adsorbate state explicit instead of
        # relying on the CO2RR C-anchor fallback.
        o_idx = [i for i, s in enumerate(symbols) if s == "O"]
        if len(o_idx) >= 1:
            if ads_clean == "OOH" and len(o_idx) > 1:
                iz_local = int(np.argmin(pos[np.asarray(o_idx, int), 2]))
                iz = int(o_idx[iz_local])
                anchor_mode = "O_prox(min_z)"
            else:
                iz = int(o_idx[0])
                anchor_mode = "O"
            anchor_pos = pos[iz].copy()
        else:
            anchor_pos = pos[0].copy()
            anchor_mode = "atom0_no_O"

    else:
        # CO / COOH: anchor atom = C preferred
        c_idx = None
        for i, s in enumerate(symbols):
            if s == "C":
                c_idx = i
                break
        if c_idx is None:
            c_idx = 0
            anchor_mode = "atom0"
        anchor_pos = pos[c_idx].copy()
        anchor_mode = "C" if symbols[c_idx] == "C" else anchor_mode

    # ---------------- normalize coordinates ----------------
    pos = pos - anchor_pos
    # reflect to keep z >= 0 relative to anchor plane
    pos[:, 2] = np.abs(pos[:, 2])
    a.set_positions(pos)

    # store anchor mode as Atoms.info (non-essential; safe to ignore elsewhere)
    try:
        a.info = dict(a.info) if getattr(a, "info", None) is not None else {}
        a.info["anchor_mode"] = anchor_mode
    except Exception:
        pass

    return a


def _rotation_matrix_from_z_to_vec(vec: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix that maps +z onto vec.

    Used only for OER/ORR molecular oxygen intermediates.  HER placement is
    not routed through this helper.
    """
    v = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return np.eye(3)
    b = v / n
    a = np.array([0.0, 0.0, 1.0], dtype=float)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-10:
        return np.eye(3)
    if c < -1.0 + 1e-10:
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)
    vx = np.array([
        [0.0, -b[2], b[1]],
        [b[2], 0.0, -b[0]],
        [-b[1], b[0], 0.0],
    ], dtype=float)
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _oxide_oer_open_direction(
    slab,
    target_cation_index: int,
    *,
    neighbor_cutoff: float = 2.75,
    upward_bias: float = 1.20,
) -> np.ndarray:
    """Estimate an open direction from a target oxide cation toward vacuum.

    The vector is biased upward and away from neighboring lattice oxygens.
    It is used only for cation-bound OER AEM placement of *O/*OH/*OOH.
    This avoids putting the adsorbate oxygen directly into an O-rich bridge
    region where it tends to detach or form a non-AEM lattice-O state.
    """
    try:
        pos = np.asarray(slab.get_positions(), dtype=float)
        syms = slab.get_chemical_symbols()
        ci = int(target_cation_index)
        if ci < 0 or ci >= len(pos):
            return np.array([0.0, 0.0, 1.0], dtype=float)
        cpos = pos[ci]
        z_top = float(np.max(pos[:, 2]))
        vec = np.array([0.0, 0.0, float(upward_bias)], dtype=float)
        for j, sj in enumerate(syms):
            if sj not in ANION_SYMBOLS:
                continue
            r = cpos - pos[j]
            d = float(np.linalg.norm(r))
            if not np.isfinite(d) or d < 1e-8 or d > float(neighbor_cutoff):
                continue
            # Weight nearby and top-near oxygen atoms more strongly.
            z_weight = 1.0 + max(0.0, 1.2 - abs(float(pos[j, 2]) - float(cpos[2])))
            vec += (r / d) * (z_weight / max(d, 0.8))
        # Enforce a positive vacuum-facing component.
        if vec[2] < 0.25:
            vec[2] = 0.25 + 0.5 * max(0.0, z_top - float(cpos[2]))
        n = float(np.linalg.norm(vec))
        if not np.isfinite(n) or n < 1e-10:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return vec / n
    except Exception:
        return np.array([0.0, 0.0, 1.0], dtype=float)


def _build_adsorbate_on_site(
    slab,
    xy: np.ndarray,
    ads: str,
    dz: float = 1.8,
    ref_dir: str | Path = "ref_gas",
    target_anchor_xyz: np.ndarray | None = None,
    target_anchor_dir: np.ndarray | None = None,
):
    """
    Place an adsorbate template on the slab at the given xy position,
    offset above the slab top by dz.  Returns the combined slab+ads
    Atoms object.

    target_anchor_dir is used only for OER/ORR molecular oxygen intermediates
    to align the template's normalized +z direction with a local open direction.
    HER placement does not use this argument.
    """
    from ase import Atoms  # type: ignore

    ads_clean = ads.replace("*", "").upper()

    slab = slab.copy()
    pos = slab.get_positions()
    z_top = float(pos[:, 2].max())

    # species-specific dz
    dz_eff = float(dz)
    if ads_clean in ("HCOO", "OCHO"):
        dz_eff = min(dz_eff, 1.4)

    if target_anchor_xyz is not None:
        taa = np.asarray(target_anchor_xyz, dtype=float).reshape(3)
        base = np.array([float(taa[0]), float(taa[1]), float(taa[2])], dtype=float)
    else:
        base = np.array([float(xy[0]), float(xy[1]), z_top + dz_eff], dtype=float)

    ads_atoms = _load_ads_template(ads_clean, ref_dir=ref_dir)
    ads_atoms.set_cell(slab.get_cell())
    ads_atoms.set_pbc(slab.get_pbc())

    if target_anchor_dir is not None and ads_clean in {"OH", "OOH"}:
        # Templates are normalized with anchor at origin and positive z away
        # from the surface.  Rotate that local +z direction toward the local
        # open direction from the exposed cation.
        R = _rotation_matrix_from_z_to_vec(np.asarray(target_anchor_dir, dtype=float))
        p0 = np.asarray(ads_atoms.get_positions(), dtype=float)
        ads_atoms.set_positions(p0 @ R.T)

    ads_atoms.translate(base)

    slab_ads: Atoms = slab + ads_atoms  # type: ignore
    return slab_ads


def _relax_slab_ads(
    slab_ads,
    n_slab_atoms: int,
    steps: int,
    fmax: float = 0.05,
    relax_ads: bool = False,
):
    """
    Relax the adsorbate on a fixed slab using BFGS.

    The slab atoms (indices 0..n_slab_atoms-1) are frozen via FixAtoms;
    only the adsorbate atoms are allowed to move.

    Returns
    -------
    slab_ads_rel : ase.Atoms
        Relaxed (or unrelaxed) slab+ads structure.
    energy : float
        Potential energy of slab_ads_rel (eV).
    opt_meta : dict
        Lightweight optimizer diagnostics:
            - elapsed_s: wall time spent in optimizer
            - n_steps:  optimizer steps actually taken (best-effort)
            - converged: optimizer convergence flag (best-effort)
            - error: exception string (present only if optimizer raised)
    """
    c = FixAtoms(indices=list(range(int(n_slab_atoms))))
    slab_ads.set_constraint(c)
    slab_ads.calc = calc

    opt_meta: Dict[str, object] = {"elapsed_s": 0.0, "n_steps": 0, "converged": None}

    if relax_ads and int(steps) > 0:
        dyn = BFGS(slab_ads, logfile=None)
        t0 = time.perf_counter()
        err = None
        try:
            dyn.run(fmax=float(fmax), steps=int(steps))
        except Exception as e:
            err = str(e)
        t1 = time.perf_counter()
        opt_meta["elapsed_s"] = float(t1 - t0)

        # steps taken (ASE optimizers differ slightly; best-effort)
        n_steps = None
        try:
            if hasattr(dyn, "get_number_of_steps"):
                n_steps = int(dyn.get_number_of_steps())
            elif hasattr(dyn, "nsteps"):
                n_steps = int(getattr(dyn, "nsteps"))
        except Exception:
            n_steps = None
        opt_meta["n_steps"] = int(n_steps) if n_steps is not None else 0

        # converged flag (best-effort)
        conv = None
        try:
            if hasattr(dyn, "converged"):
                conv = bool(dyn.converged())
        except Exception:
            conv = None
        opt_meta["converged"] = conv

        if err is not None:
            opt_meta["error"] = err

    try:
        energy = float(slab_ads.get_potential_energy())
    except Exception:
        energy = float("nan")

    return slab_ads, energy, opt_meta


def _co2rr_anchor_xy(
    ads_coords3: np.ndarray, ads_symbols: list[str], ads_clean: str
) -> tuple[np.ndarray, float, str]:
    """Return (anchor_xy, anchor_z, anchor_mode) for migration/QA metrics."""
    ads_clean = ads_clean.upper()
    if ads_coords3.shape[0] == 0:
        return np.array([np.nan, np.nan], dtype=float), float("nan"), "none"

    if ads_clean == "HCOO":
        o_idx = [i for i, s in enumerate(ads_symbols) if s == "O"]
        if len(o_idx) >= 2:
            xy = ads_coords3[o_idx[:2], :2].mean(axis=0)
            z = float(ads_coords3[o_idx[:2], 2].min())
            return xy, z, "O2_mid"
        if len(o_idx) == 1:
            xy = ads_coords3[o_idx[0], :2]
            z = float(ads_coords3[o_idx[0], 2])
            return xy, z, "O"
        xy = ads_coords3[:, :2].mean(axis=0)
        z = float(ads_coords3[:, 2].min())
        return xy, z, "COM_fallback"

    if ads_clean == "OCHO":
        o_idx = [i for i, s in enumerate(ads_symbols) if s == "O"]
        if len(o_idx) >= 1:
            if len(o_idx) > 1:
                iz_local = int(np.argmin(ads_coords3[np.asarray(o_idx, int), 2]))
                iz = int(o_idx[iz_local])
            else:
                iz = int(o_idx[0])
            xy = ads_coords3[iz, :2]
            z = float(ads_coords3[iz, 2])
            return xy, z, "O(min_z)"
        xy = ads_coords3[:, :2].mean(axis=0)
        z = float(ads_coords3[:, 2].min())
        return xy, z, "COM_fallback"

    if ads_clean in ("O", "OH", "OOH"):
        o_idx = [i for i, s in enumerate(ads_symbols) if s == "O"]
        if len(o_idx) >= 1:
            if ads_clean == "OOH" and len(o_idx) > 1:
                iz = int(o_idx[int(np.argmin(ads_coords3[np.asarray(o_idx, int), 2]))])
                return ads_coords3[iz, :2], float(ads_coords3[iz, 2]), "O_prox(min_z)"
            iz = int(o_idx[0])
            return ads_coords3[iz, :2], float(ads_coords3[iz, 2]), "O"
        xy = ads_coords3[:, :2].mean(axis=0)
        z = float(ads_coords3[:, 2].min())
        return xy, z, "COM_fallback_no_O"

    # CO / COOH: anchor = C if present
    c_idx = None
    for i, s in enumerate(ads_symbols):
        if s == "C":
            c_idx = i
            break
    if c_idx is None:
        xy = ads_coords3[:, :2].mean(axis=0)
        z = float(ads_coords3[:, 2].min())
        return xy, z, "COM_fallback"
    return ads_coords3[c_idx, :2], float(ads_coords3[c_idx, 2]), "C"


def _co2rr_internal_broken(
    ads_coords3: np.ndarray, ads_symbols: list[str], ads_clean: str
) -> bool:
    """Very lightweight bond-sanity checks to catch obvious fragmentation."""
    ads_clean = ads_clean.upper()
    if ads_coords3.shape[0] < 2:
        return True

    def dist(i, j) -> float:
        return float(np.linalg.norm(ads_coords3[i] - ads_coords3[j]))

    idx_C = [i for i, s in enumerate(ads_symbols) if s == "C"]
    idx_O = [i for i, s in enumerate(ads_symbols) if s == "O"]
    idx_H = [i for i, s in enumerate(ads_symbols) if s == "H"]

    if ads_clean == "CO":
        if len(idx_C) != 1 or len(idx_O) != 1:
            return True
        return dist(idx_C[0], idx_O[0]) > 1.6

    if ads_clean == "COOH":
        if len(idx_C) < 1 or len(idx_O) < 2:
            return True
        c = idx_C[0]
        co = sorted(dist(c, o) for o in idx_O)
        if len(co) < 2 or co[0] > 1.9 or co[1] > 1.9:
            return True
        if len(idx_H) >= 1:
            oh = min(dist(h, o) for h in idx_H for o in idx_O)
            if oh > 1.35:
                return True
        return False

    if ads_clean in ("HCOO", "OCHO"):
        if len(idx_C) < 1 or len(idx_O) < 2:
            return True
        c = idx_C[0]
        co = sorted(dist(c, o) for o in idx_O)
        if len(co) < 2 or co[0] > 1.9 or co[1] > 1.9:
            return True
        if len(idx_H) >= 1:
            ch = float(np.linalg.norm(ads_coords3[idx_H[0]] - ads_coords3[c]))
            if ch > 1.35:
                return True
        return False

    # ── ORR bond sanity checks ──────────────────────────────────────
    if ads_clean == "O":
        return len(idx_O) != 1

    if ads_clean == "OH":
        if len(idx_O) != 1 or len(idx_H) != 1:
            return True
        return dist(idx_O[0], idx_H[0]) > 1.20

    if ads_clean == "OOH":
        if len(idx_O) != 2 or len(idx_H) != 1:
            return True
        oo = dist(idx_O[0], idx_O[1])
        oh = min(dist(h, o) for h in idx_H for o in idx_O)
        return oo > 1.65 or oh > 1.20
    # ────────────────────────────────────────────────────────────────

    return False


def _co2rr_min_slab_dist(slab_coords3: np.ndarray, ads_coords3: np.ndarray) -> float:
    if slab_coords3.shape[0] == 0 or ads_coords3.shape[0] == 0:
        return float("nan")
    d = np.linalg.norm(
        ads_coords3[:, None, :] - slab_coords3[None, :, :], axis=2
    )
    return float(np.min(d))


def _oxygen_adsorbate_state_metrics(
    slab_coords3: np.ndarray,
    ads_coords3: np.ndarray,
    ads_symbols: list[str],
    ads_clean: str,
    surface_channel: object = None,
    slab_symbols: list[str] | None = None,
    target_cation_indices: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Classify ORR/OER oxygen-intermediate binding state.

    This is a QA helper for *O, *OH and *OOH only.  It does not affect HER.
    A row can be migrated and still usable only if the proximal O remains
    surface-bound and the internal O-H/O-O bonds remain intact.
    """
    ads_clean = str(ads_clean).replace("*", "").upper()
    out: dict[str, object] = {
        "oxygen_adsorbate": bool(ads_clean in {"O", "OH", "OOH"}),
        "oxygen_anchor_index_local": -1,
        "oxygen_anchor_symbol": "",
        "oxygen_anchor_slab_dist(Å)": float("nan"),
        "oxygen_nearest_slab_index": -1,
        "oxygen_nearest_slab_symbol": "",
        "oxygen_anchor_cation_dist(Å)": float("nan"),
        "oxygen_anchor_target_cation_dist(Å)": float("nan"),
        "oxygen_anchor_target_cation_index": -1,
        "oxygen_anchor_target_cation_symbol": "",
        "oxygen_anchor_anion_dist(Å)": float("nan"),
        "oxygen_bound_to_cation": False,
        "oxygen_ads_bound": False,
        "oxygen_internal_bond_ok": False,
        "oxygen_surface_bond_ok": False,
        "valid_for_oer_summary": False,
        "oer_state_class": "not_oxygen_intermediate",
        "oer_state_note": "",
    }
    if ads_clean not in {"O", "OH", "OOH"}:
        return out
    if ads_coords3 is None or ads_coords3.shape[0] == 0 or slab_coords3 is None or slab_coords3.shape[0] == 0:
        out["oer_state_class"] = "missing_geometry"
        return out

    idx_O = [i for i, s in enumerate(ads_symbols) if s == "O"]
    idx_H = [i for i, s in enumerate(ads_symbols) if s == "H"]
    if not idx_O:
        out["oer_state_class"] = "no_anchor_O"
        return out

    if ads_clean == "OOH" and len(idx_O) > 1:
        anchor_i = int(idx_O[int(np.argmin(ads_coords3[np.asarray(idx_O, int), 2]))])
    else:
        anchor_i = int(idx_O[0])

    anchor = np.asarray(ads_coords3[anchor_i], dtype=float)
    slab_dists = np.linalg.norm(slab_coords3 - anchor[None, :], axis=1)
    nearest_slab_i = int(np.argmin(slab_dists)) if len(slab_dists) else -1
    d_to_slab = float(slab_dists[nearest_slab_i]) if nearest_slab_i >= 0 else float("nan")

    slab_symbols = list(slab_symbols or [])
    nearest_sym = str(slab_symbols[nearest_slab_i]) if (nearest_slab_i >= 0 and nearest_slab_i < len(slab_symbols)) else ""
    cation_idx = [i for i, sym in enumerate(slab_symbols) if sym not in ANION_SYMBOLS]
    anion_idx = [i for i, sym in enumerate(slab_symbols) if sym in ANION_SYMBOLS]
    d_cation = float(np.min(np.linalg.norm(slab_coords3[np.asarray(cation_idx, int)] - anchor[None, :], axis=1))) if cation_idx else float("nan")
    d_anion = float(np.min(np.linalg.norm(slab_coords3[np.asarray(anion_idx, int)] - anchor[None, :], axis=1))) if anion_idx else float("nan")
    target_cation_indices = tuple(int(i) for i in (target_cation_indices or tuple()) if 0 <= int(i) < len(slab_coords3))
    target_cation_index = int(target_cation_indices[0]) if target_cation_indices else -1
    d_target_cation = (
        float(np.linalg.norm(slab_coords3[target_cation_index] - anchor))
        if target_cation_index >= 0 else float("nan")
    )
    target_cation_symbol = (
        str(slab_symbols[target_cation_index])
        if (target_cation_index >= 0 and target_cation_index < len(slab_symbols)) else ""
    )

    out["oxygen_anchor_index_local"] = int(anchor_i)
    out["oxygen_anchor_symbol"] = str(ads_symbols[anchor_i])
    out["oxygen_anchor_slab_dist(Å)"] = d_to_slab
    out["oxygen_nearest_slab_index"] = int(nearest_slab_i)
    out["oxygen_nearest_slab_symbol"] = nearest_sym
    out["oxygen_anchor_cation_dist(Å)"] = d_cation
    out["oxygen_anchor_target_cation_dist(Å)"] = d_target_cation
    out["oxygen_anchor_target_cation_index"] = int(target_cation_index)
    out["oxygen_anchor_target_cation_symbol"] = target_cation_symbol
    out["oxygen_anchor_anion_dist(Å)"] = d_anion

    # Internal-bond sanity.  Use permissive upper bounds because ML relaxation
    # may stretch weakly bound adsorbates, but broken OOH/OH should not enter
    # OER summaries.
    internal_ok = True
    if ads_clean == "O":
        internal_ok = (len(idx_O) == 1)
    elif ads_clean == "OH":
        if len(idx_O) != 1 or len(idx_H) != 1:
            internal_ok = False
        else:
            oh = float(np.linalg.norm(ads_coords3[idx_O[0]] - ads_coords3[idx_H[0]]))
            internal_ok = (0.60 <= oh <= 1.25)
            out["oxygen_OH_bond(Å)"] = oh
    elif ads_clean == "OOH":
        if len(idx_O) != 2 or len(idx_H) != 1:
            internal_ok = False
        else:
            oo = float(np.linalg.norm(ads_coords3[idx_O[0]] - ads_coords3[idx_O[1]]))
            oh = min(float(np.linalg.norm(ads_coords3[h] - ads_coords3[o])) for h in idx_H for o in idx_O)
            internal_ok = (1.05 <= oo <= 1.75) and (0.60 <= oh <= 1.25)
            out["oxygen_OO_bond(Å)"] = oo
            out["oxygen_OH_bond(Å)"] = oh

    # Surface-bound criterion for proximal O.  For AEM cation-bound OER,
    # require a plausible cation-O anchor distance and reject cases where the
    # closest binding partner is lattice oxygen.  This prevents bridge/lattice
    # O placements from being mixed into *OH/*O/*OOH cation-site summaries.
    ch = str(surface_channel or "").strip().lower()
    max_cation_dist = {"O": 2.30, "OH": 2.45, "OOH": 2.55}.get(ads_clean, 2.45)
    cation_bound = bool(np.isfinite(d_cation) and (1.10 <= d_cation <= float(max_cation_dist)))
    target_cation_bound = bool(np.isfinite(d_target_cation) and (1.10 <= d_target_cation <= float(max_cation_dist)))
    # A neighboring lattice O can be geometrically close on rutile-like oxides.
    # Do not reject a cation-bound AEM state merely because a lattice O is
    # slightly closer than the target cation.  Reject only nonphysical O-O
    # collision distances; ambiguous bridge/peroxo cases remain QA-visible via
    # oxygen_anchor_anion_dist(Å).
    anion_too_close_cut = {"O": 1.42, "OH": 1.50, "OOH": 1.50}.get(ads_clean, 1.50)
    anion_too_close = bool(np.isfinite(d_anion) and d_anion < float(anion_too_close_cut))

    if ch == "adsorbate_on_cation":
        # Strict OER AEM: anchor must be bound to the intended exposed cation.
        surf_ok = bool((target_cation_bound if target_cation_index >= 0 else cation_bound) and not anion_too_close)
    else:
        surf_ok = bool(np.isfinite(d_to_slab) and (0.80 <= d_to_slab <= 2.75))

    invalid_channel = ch in {
        "protonated_lattice_oxygen",
        "anion_o",
        "lattice_protonation_disabled_for_aem",
        "fallback_site",
    }

    out["oxygen_bound_to_cation"] = bool((target_cation_bound if target_cation_index >= 0 else cation_bound) and not anion_too_close)
    out["oxygen_internal_bond_ok"] = bool(internal_ok)
    out["oxygen_surface_bond_ok"] = bool(surf_ok)
    out["oxygen_ads_bound"] = bool(internal_ok and surf_ok)
    if invalid_channel:
        out["oer_state_class"] = "invalid_oer_channel"
        out["oer_state_note"] = f"surface_channel={surface_channel} is excluded from AEM OER summaries"
    elif not internal_ok:
        out["oer_state_class"] = "broken_oxygen_intermediate"
    elif not surf_ok:
        out["oer_state_class"] = "misbound_or_detached_oxygen_intermediate" if ch == "adsorbate_on_cation" else "detached_oxygen_intermediate"
        if ch == "adsorbate_on_cation":
            out["oer_state_note"] = f"target_cation_dist={d_target_cation:.3f}; nearest_cation_dist={d_cation:.3f}; anion_dist={d_anion:.3f}; anion_too_close={anion_too_close}; nearest={nearest_sym}"
    else:
        out["oer_state_class"] = "bound_oxygen_intermediate"
        out["valid_for_oer_summary"] = True
    return out



def _normalize_site_kind(kind: object) -> str:
    k = str(kind or "").strip().lower()
    aliases = {
        "cation": "oer_cation",
        "oer-cation": "oer_cation",
        "oer_cation_top": "oer_cation",
        "top": "ontop",
        "atop": "ontop",
        "on-top": "ontop",
        "on_top": "ontop",
        "3fold": "hollow",
        "threefold": "hollow",
        "three-fold": "hollow",
        "o_top": "anion_ontop",
        "o-top": "anion_ontop",
        "anion_o": "anion_ontop",
    }
    return aliases.get(k, k if k else "unknown")


def _canonicalize_metal_kind(kind: object) -> str:
    k = _normalize_site_kind(kind)
    if k in {"fcc", "hcp", "hollow"}:
        return "hollow"
    return k


def _kind_priority(kind: str) -> int:
    return {
        "ontop": 0,
        "anion_ontop": 0,
        "bridge": 1,
        "hollow": 2,
        "fcc": 2,
        "hcp": 2,
        "unknown": 99,
    }.get(_normalize_site_kind(kind), 50)


def _fmt_surface_indices(indices: object) -> str:
    try:
        vals = tuple(int(i) for i in (indices or ()))
    except Exception:
        vals = tuple()
    return ",".join(str(i) for i in vals)


def _oxide_d2_summary_to_primary_results_df(summary: Optional[dict], *, net_corr: float = STANDARD_CHE_CORR) -> pd.DataFrame:
    """Build the user-facing oxide HER result table from the selected D2 descriptor.

    Local-ZPE-aware version.  The selected D2 candidate row is treated as the
    source of truth when available.  If the oxide descriptor module produced
    ΔG_H_local, local_thermo_corr, and local_zpe_status columns, they are kept
    in the primary HER table instead of being silently collapsed back to the
    standard CHE value.
    """
    if not isinstance(summary, dict) or not summary:
        return pd.DataFrame()

    label = str(summary.get("D2_site_label", "NA"))
    d2_csv = str(summary.get("D2_candidates_csv", "") or "").strip()
    selected: dict[str, object] = {}

    if d2_csv:
        try:
            p = Path(d2_csv).expanduser()
            if p.is_file():
                cand = pd.read_csv(str(p))
                if isinstance(cand, pd.DataFrame) and not cand.empty:
                    if "site_label" in cand.columns and label not in {"", "NA", "nan", "None"}:
                        hit = cand[cand["site_label"].astype(str) == label]
                        if not hit.empty:
                            selected = dict(hit.iloc[0])
                    if not selected:
                        # fallback: reproduce the D2 selection rule directly
                        work = cand.copy()
                        valid_col = "D2_descriptor_valid" if "D2_descriptor_valid" in work.columns else "descriptor_valid"
                        if valid_col in work.columns:
                            valid = work[work[valid_col].astype(str).str.lower().isin(["true", "1", "yes"])]
                            if not valid.empty:
                                work = valid
                        e_col = "ΔG_H (eV)" if "ΔG_H (eV)" in work.columns else ("ΔG (eV)" if "ΔG (eV)" in work.columns else None)
                        if e_col is not None:
                            vals = pd.to_numeric(work[e_col], errors="coerce")
                            work = work.loc[vals[np.isfinite(vals)].index].copy()
                            if not work.empty:
                                work["_abs_e"] = pd.to_numeric(work[e_col], errors="coerce").abs()
                                selected = dict(work.loc[work["_abs_e"].idxmin()].drop(labels=["_abs_e"], errors="ignore"))
        except Exception:
            selected = {}

    def _get(key: str, fallback=np.nan):
        if key in selected:
            return selected.get(key)
        return summary.get(key, fallback)

    # Prefer the actual selected descriptor value.  If local ZPE was requested
    # and succeeded, oxide_descriptor.py should already have made ΔG_H use the
    # local-corrected value.
    dG = _safe_float(_get("ΔG_H (eV)", summary.get("D2_Hreact (eV)", np.nan)))
    dG_che = _safe_float(_get("ΔG_H_CHE (eV)", summary.get("D2_Hreact_CHE (eV)", np.nan)))
    dG_local = _safe_float(_get("ΔG_H_local (eV)", summary.get("D2_Hreact_local (eV)", np.nan)))
    dE = _safe_float(_get("ΔE_H_user (eV)", np.nan))

    if not np.isfinite(dG) and np.isfinite(dG_local):
        dG = dG_local
    if not np.isfinite(dG) and np.isfinite(dG_che):
        dG = dG_che
    if not np.isfinite(dG_che) and np.isfinite(dE):
        dG_che = float(dE) + float(net_corr)
    if not np.isfinite(dE) and np.isfinite(dG_che):
        dE = float(dG_che) - float(net_corr)
    if not np.isfinite(dE) and np.isfinite(dG):
        dE = float(dG) - float(net_corr)

    thermo_mode_val = str(_get("thermo_mode", summary.get("D2_thermo_mode", summary.get("thermo_mode", "CHE correction (fast screening)"))))
    local_corr = _safe_float(_get("local_thermo_corr (eV)", summary.get("D2_local_thermo_corr (eV)", np.nan)))
    local_status = str(_get("local_zpe_status", summary.get("D2_local_zpe_status", "not_requested")) or "not_requested")

    site_kind = str(_get("site_kind", summary.get("D2_binding_class", "metal_centered_Hstar")) or "metal_centered_Hstar")
    final_kind = str(_get("final_site_kind", summary.get("D2_final_site_kind", site_kind)) or site_kind)
    binding_class = str(_get("D2_binding_class", _get("binding_class", summary.get("D2_binding_class", final_kind))) or final_kind)
    disp = _safe_float(_get("H_lateral_disp(Å)", summary.get("D2_seed_disp(Å)", np.nan)))
    cif = str(_get("structure_cif", summary.get("D2_structure_cif", "")) or "")
    qc_flags = str(_get("qc_flags", summary.get("D2_qc_flags", "")) or "")

    if label in {"", "NA", "nan", "None"}:
        label = str(_get("site_label", "D2_selected") or "D2_selected")

    row = {
        "mode": "HER",
        "result_role": "oxide_D2_primary",
        "descriptor": "D2",
        "D2_policy": "surface_metal_cation_centered_Hstar",
        "oxide_descriptor_relaxation_policy": "D1=constrained/rigid; D2_Hreact=partial",
        "D1_relaxation_policy": str(summary.get("D1_fixed_relaxation_policy", "anchor_oh_hookean_preOH")),
        "D2_relaxation_scope": str(summary.get("D2_relaxation_scope", summary.get("D2_fixed_relaxation_scope", "partial"))),
        "site": site_kind,
        "site_label": label,
        "requested_site": site_kind,
        "initial_geom_site": site_kind,
        "relaxed_site": final_kind,
        "final_site_kind": final_kind,
        "binding_class": binding_class,
        "D2_binding_class": binding_class,
        "D2_descriptor_valid": True if np.isfinite(dG) else False,
        "D2_selection_rule": str(summary.get("D2_selection_rule", "min_abs_deltaG_among_valid_metal_centered_Hstar")),
        "placement_mismatch": False,
        "migrated_actual": bool(np.isfinite(disp) and disp > MIGRATE_THR),
        "migrated": bool(np.isfinite(disp) and disp > MIGRATE_THR),
        "migration_destination": final_kind,
        "migration_path": f"{site_kind} -> {final_kind}",
        "actual_migration_path": f"{site_kind} -> {final_kind}",
        "site_transition_type": "stable" if (np.isfinite(disp) and disp <= MIGRATE_THR) else "relaxed_D2_state",
        "structure_cif": cif,
        "thermo_mode": thermo_mode_val,
        "ΔG_H_CHE (eV)": dG_che,
        "ΔG_H_local (eV)": dG_local,
        "ΔG_H (eV)": dG,
        "local_thermo_corr (eV)": local_corr,
        "local_delta_zpe_eV": _safe_float(_get("local_delta_zpe_eV", np.nan)),
        "local_delta_ts_eV": _safe_float(_get("local_delta_ts_eV", np.nan)),
        "local_zpe_status": local_status,
        "local_zpe_n_vib_atoms": int(_safe_float(_get("local_zpe_n_vib_atoms", 0), 0.0)),
        "local_zpe_warnings": str(_get("local_zpe_warnings", summary.get("D2_local_zpe_warnings", "")) or ""),
        "local_zpe_error": str(_get("local_zpe_error", summary.get("D2_local_zpe_error", "")) or ""),
        "ΔE_H_user (eV)": dE,
        "ΔE_H (eV)": dE,
        "abs_ΔG_H (eV)": abs(dG) if np.isfinite(dG) else np.nan,
        "H_lateral_disp(Å)": disp,
        "is_duplicate": False,
        "qc_flags": qc_flags,
        "nearest_metal_symbol": str(summary.get("D2_nearest_metal_symbol", _get("nearest_metal_symbol", "NA"))),
        "nearest_metal_distance(Å)": _safe_float(summary.get("D2_nearest_metal_distance(Å)", _get("nearest_metal_distance(Å)", np.nan))),
        "nearest_anion_symbol": str(summary.get("D2_nearest_anion_symbol", _get("nearest_anion_symbol", "NA"))),
        "nearest_anion_distance(Å)": _safe_float(summary.get("D2_nearest_anion_distance(Å)", _get("nearest_anion_distance(Å)", np.nan))),
    }
    return pd.DataFrame([row])

def _mic_xy_delta(cell, pbc, xy0: np.ndarray, xy1: np.ndarray) -> tuple[np.ndarray, float]:
    xy0 = np.asarray(xy0, dtype=float).reshape(2)
    xy1 = np.asarray(xy1, dtype=float).reshape(2)
    d3 = np.array([float(xy1[0] - xy0[0]), float(xy1[1] - xy0[1]), 0.0], dtype=float)
    vec, _ = find_mic(d3, cell=cell, pbc=[bool(pbc[0]), bool(pbc[1]), False])
    vec = np.asarray(vec, dtype=float)
    return vec[:2], float(np.linalg.norm(vec[:2]))


def _classify_site_from_candidates(
    slab_only,
    anchor_xy: np.ndarray,
    candidates: list[AdsSite],
    *,
    unknown_tol: float,
) -> dict[str, object]:
    anchor_xy = np.asarray(anchor_xy, dtype=float).reshape(2)
    best: dict[str, object] | None = None
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
        return {
            "kind": "unknown",
            "match_dist": float("nan"),
            "surface_indices": tuple(),
        }
    if np.isfinite(best["match_dist"]) and float(best["match_dist"]) > float(unknown_tol):
        best["kind"] = "unknown"
    return best


def _classify_metal_site_xy_legacy(slab_only, anchor_xy: np.ndarray) -> str:
    """Legacy xy-based metal-site classifier returning ontop/bridge/hollow."""
    try:
        pos = slab_only.get_positions()
        top = layer_indices(slab_only, n=1)[0]
        top_xy = pos[np.asarray(top, int), :2]
        d = np.linalg.norm(top_xy - anchor_xy[None, :], axis=1)
        order = np.argsort(d)
        if len(order) < 1:
            return "unknown"
        d1 = float(d[order[0]])
        tol_atop = 0.45
        tol = 0.35

        if d1 < tol_atop:
            return "ontop"

        if len(order) >= 2:
            mid = 0.5 * (top_xy[order[0]] + top_xy[order[1]])
            if float(np.linalg.norm(mid - anchor_xy)) < tol:
                return "bridge"

        if len(order) >= 3:
            tri = top_xy[order[:3]]
            centroid = tri.mean(axis=0)
            if float(np.linalg.norm(centroid - anchor_xy)) < tol:
                try:
                    layers2 = layer_indices(slab_only, n=2)
                    if len(layers2) >= 2:
                        second = layers2[1]
                        sec_xy = pos[np.asarray(second, int), :2]
                        d2 = float(np.min(np.linalg.norm(sec_xy - centroid[None, :], axis=1)))
                        if d2 < tol_atop:
                            return "hollow"
                except Exception:
                    pass
                return "hollow"

        return "unknown"
    except Exception:
        return "unknown"


def _classify_metal_site_xy(slab_only, anchor_xy: np.ndarray) -> dict[str, object]:
    """Robust metal-site classification using ontop/bridge/hollow taxonomy."""
    try:
        candidates = detect_metal_111_sites(slab_only, max_sites_per_kind=200)
    except Exception:
        candidates = []
    out = _classify_site_from_candidates(slab_only, anchor_xy, candidates, unknown_tol=0.85)
    out["kind"] = _canonicalize_metal_kind(out.get("kind", "unknown"))
    if out.get("kind") == "unknown":
        legacy = _classify_metal_site_xy_legacy(slab_only, np.asarray(anchor_xy, dtype=float))
        if legacy != "unknown":
            out["kind"] = _canonicalize_metal_kind(legacy)
    out["classification_mode"] = "metal_geom"
    return out


def _classify_oxide_geom_site_xy(slab_only, anchor_xy: np.ndarray) -> dict[str, object]:
    try:
        candidates = detect_oxide_surface_sites(slab_only, max_sites_per_kind=200, z_tol=1.2)
    except Exception:
        candidates = []
    out = _classify_site_from_candidates(slab_only, anchor_xy, candidates, unknown_tol=1.00)
    out["classification_mode"] = "oxide_geom"
    return out


def _classify_oxide_oer_cation_site_xy(slab_only, anchor_xy: np.ndarray, tol: float = 1.65) -> dict[str, object]:
    """Classify an OER oxygen-intermediate anchor by exposed cation sites only.

    This avoids reclassifying oxide OER AEM rows as metal-like ontop/bridge/fcc
    sites.  It is intentionally used only when classification_mode=
    "oxide_oer_cation".
    """
    try:
        candidates = detect_oxide_oer_cation_sites(slab_only, max_sites=200)
    except Exception:
        candidates = []
    out = _classify_site_from_candidates(slab_only, anchor_xy, candidates, unknown_tol=float(tol))
    if out.get("kind") not in ("unknown", "", None):
        out["kind"] = "oer_cation"
    out["classification_mode"] = "oxide_oer_cation"
    return out


def _classify_oxide_anion_site_xy(slab_only, anchor_xy: np.ndarray, tol: float = 0.75) -> dict[str, object]:
    try:
        pos = slab_only.get_positions()
        syms = slab_only.get_chemical_symbols()
        z = pos[:, 2]
        anion_idx = np.asarray([i for i, sym in enumerate(syms) if sym in ANION_SYMBOLS], dtype=int)
        if anion_idx.size == 0:
            return {"kind": "unknown", "match_dist": float("nan"), "surface_indices": tuple(), "classification_mode": "oxide_anion_top"}
        z_max = float(np.max(z[anion_idx]))
        top_idx = anion_idx[(z_max - z[anion_idx]) < 0.8]
        if top_idx.size == 0:
            top_idx = anion_idx
        best = None
        best_key = None
        for idx in top_idx.tolist():
            cand_xy = np.asarray(pos[idx, :2], dtype=float)
            _, dist = _mic_xy_delta(slab_only.get_cell(), slab_only.get_pbc(), cand_xy, anchor_xy)
            key = (float(dist), 0, int(idx))
            if best is None or key < best_key:
                best = {"kind": "anion_ontop", "match_dist": float(dist), "surface_indices": (int(idx),), "classification_mode": "oxide_anion_top"}
                best_key = key
        if best is None:
            return {"kind": "unknown", "match_dist": float("nan"), "surface_indices": tuple(), "classification_mode": "oxide_anion_top"}
        if np.isfinite(best["match_dist"]) and float(best["match_dist"]) > float(tol):
            best["kind"] = "unknown"
        return best
    except Exception:
        return {"kind": "unknown", "match_dist": float("nan"), "surface_indices": tuple(), "classification_mode": "oxide_anion_top"}


def _resolve_site_tracking(
    *,
    slab_only,
    mtype: str,
    seed_kind: object,
    initial_xy: np.ndarray,
    final_anchor_xy: np.ndarray,
    classification_mode: str = "auto",
    disp_threshold: float = MIGRATE_THR,
) -> dict[str, object]:
    initial_xy = np.asarray(initial_xy, dtype=float).reshape(2)
    final_anchor_xy = np.asarray(final_anchor_xy, dtype=float).reshape(2)
    seed_kind_norm = _normalize_site_kind(seed_kind)

    if classification_mode == "oxide_anion":
        initial_kind = "anion_ontop"
        final_meta = _classify_oxide_anion_site_xy(slab_only, final_anchor_xy)
        site_family = "oxide_anion_top"
    elif classification_mode == "oxide_oer_cation":
        initial_kind = "oer_cation"
        final_meta = _classify_oxide_oer_cation_site_xy(slab_only, final_anchor_xy)
        site_family = "oxide_oer_cation"
    elif str(mtype).lower() == "metal":
        initial_kind = _canonicalize_metal_kind(seed_kind_norm)
        final_meta = _classify_metal_site_xy(slab_only, final_anchor_xy)
        site_family = "metal_geom"
    else:
        initial_kind = seed_kind_norm
        final_meta = _classify_oxide_geom_site_xy(slab_only, final_anchor_xy)
        site_family = "oxide_geom"

    _, disp_mic = _mic_xy_delta(slab_only.get_cell(), slab_only.get_pbc(), initial_xy, final_anchor_xy)
    final_kind = _normalize_site_kind(final_meta.get("kind", "unknown"))
    label_changed = bool(initial_kind not in ("", "unknown") and final_kind not in ("", "unknown") and final_kind != initial_kind)
    displaced = bool(np.isfinite(disp_mic) and float(disp_mic) > float(disp_threshold))
    migrated = bool(label_changed or displaced)

    if label_changed and displaced:
        migration_basis = "both"
    elif label_changed:
        migration_basis = "reclassified"
    elif displaced:
        migration_basis = "displacement"
    else:
        migration_basis = "none"

    if not migrated:
        migration_type = "none"
    elif migration_basis == "displacement" and final_kind == initial_kind:
        migration_type = f"{initial_kind}_displaced"
    else:
        migration_type = f"{initial_kind}_to_{final_kind}"

    qc_flags: List[str] = []
    if final_kind == "unknown":
        qc_flags.append("unclassified_site")
    if label_changed:
        qc_flags.append("site_reclassified")
    if displaced:
        qc_flags.append("migrated_site")

    return {
        "seed_site_kind": seed_kind_norm,
        "initial_site_kind": initial_kind,
        "final_site_kind": final_kind,
        "classification_mode": str(final_meta.get("classification_mode", classification_mode)),
        "site_family": site_family,
        "final_site_match_dist(Å)": float(final_meta.get("match_dist", float("nan"))),
        "initial_anchor_x(Å)": float(initial_xy[0]),
        "initial_anchor_y(Å)": float(initial_xy[1]),
        "final_anchor_x(Å)": float(final_anchor_xy[0]),
        "final_anchor_y(Å)": float(final_anchor_xy[1]),
        "lateral_disp_mic(Å)": float(disp_mic),
        "migrated": bool(migrated),
        "migration_basis": migration_basis,
        "migration_type": migration_type,
        "qc_flags": list(qc_flags),
        "initial_surface_indices": "",
        "final_surface_indices": _fmt_surface_indices(final_meta.get("surface_indices", ())),
    }

# ---------------------------------------------------------------------
# Optional HER guardrail for CO2RR (single-site; cheap)
# ---------------------------------------------------------------------
def _pick_guardrail_site(
    target_sites: list[dict[str, object]],
    preference: str = "ontop",
) -> dict[str, object] | None:
    """Pick one target-site record, preferring `preference` if available."""
    if not target_sites:
        return None
    pref = _normalize_site_kind(preference)
    for site in target_sites:
        kind = _normalize_site_kind(site.get("site_kind"))
        label = _normalize_site_kind(site.get("site_label"))
        if kind == pref or label == pref:
            return site
    return target_sites[0]


def _compute_her_guardrail_from_prepared(
    slab_u_rel,
    E_slab_u: float,
    E_H2: float,
    target_sites: list[dict[str, object]],
    z_steps: int,
    free_steps: int,
    site_preference: str = "ontop",
    use_net_corr: bool = True,
    out_cif: Path | None = None,
    relaxation_scope: str = "partial",
    n_fix_layers: int = 2,
) -> dict[str, object] | None:
    """Compute a single H* adsorption as a guardrail using the prepared slab and H2 ref."""
    pick = _pick_guardrail_site(target_sites, preference=site_preference)
    if pick is None:
        return None

    label = str(pick.get("site_label", "unknown"))
    kind = str(pick.get("site_kind", "unknown"))
    xy = np.asarray(pick.get("xy", [np.nan, np.nan]), dtype=float)
    anchor_xyz = np.asarray(pick.get("initial_xyz", [np.nan, np.nan, np.nan]), dtype=float)
    h0s_used = recommend_h0s_for_relaxation(relaxation_scope, mtype="oxide", site_kind=kind, default_h0s=H0S)
    Au, E_uH, _disp_raw = site_energy_two_stage(
        slab_u_rel,
        xy,
        h0s_used,
        int(z_steps),
        int(free_steps),
        relaxation_scope=relaxation_scope,
        n_fix_layers=int(n_fix_layers),
        anchor_xyz=anchor_xyz,
        mtype="oxide",
        site_kind=kind,
        min_clearance=default_min_clearance_for_scope(relaxation_scope, mtype="oxide"),
    )

    slab_only = Au[: len(slab_u_rel)]
    h_xy = np.asarray(Au.get_positions()[-1, :2], dtype=float)
    tracking = _resolve_site_tracking(
        slab_only=slab_only,
        mtype="oxide" if str(kind).lower().startswith("anion") else "metal",
        seed_kind=kind,
        initial_xy=xy,
        final_anchor_xy=h_xy,
        classification_mode="oxide_anion" if str(kind).lower().startswith("anion") else "auto",
        disp_threshold=float(MIGRATE_THR),
    )

    dE_u = float(E_uH) - float(E_slab_u) - 0.5 * float(E_H2)
    net_corr = float(STANDARD_CHE_CORR if use_net_corr else 0.0)
    dG_u = dE_u + net_corr

    if out_cif is not None:
        try:
            out_cif.parent.mkdir(parents=True, exist_ok=True)
            write(out_cif, Au)
        except Exception:
            pass

    row: dict[str, object] = {
        "mode": "HER_GUARDRAIL",
        "site": str(kind),
        "site_label": str(label),
        "seed_site_kind": tracking["seed_site_kind"],
        "initial_site_kind": tracking["initial_site_kind"],
        "final_site_kind": tracking["final_site_kind"],
        "site_tracking_mode": tracking["classification_mode"],
        "migration_basis": tracking["migration_basis"],
        "migration_type": tracking["migration_type"],
        "qc_flags": ";".join(tracking["qc_flags"]),
        "E_slab_user (eV)": float(E_slab_u),
        "E_slab+H_user (eV)": float(E_uH),
        "ΔE_H_user (eV)": float(dE_u),
        "ΔG_H (eV)": float(dG_u),
        "H_lateral_disp(Å)": float(tracking["lateral_disp_mic(Å)"]),
        "final_site_match_dist(Å)": float(tracking["final_site_match_dist(Å)"]),
        "migrated": bool(tracking["migrated"]),
        "NET_CORR (eV)": float(net_corr),
    }
    if out_cif is not None:
        row["structure_cif"] = str(Path(out_cif).resolve())
    return row

# ---------------------------------------------------------------------
# CO2RR / ORR mode (reaction-descriptor based ΔE/ΔG)
# ---------------------------------------------------------------------
def _run_co2rr_che(
    mtype: str,
    user_slab_cif: str,
    out_root: Path,
    sites: Iterable[str],
    vac_z: float,
    layers: int,
    export_absolute: bool,
    relax_mode: str,
    user_ads_sites: Optional[Mapping[str, object]],
    adspecies: Tuple[str, ...],
    her_guardrail: bool = False,
    her_site_preference: str = "ontop",
    her_use_net_corr: bool = True,
    reaction_mode: str = "CO2RR",   # "CO2RR" | "OER"
    orr_u: float = 0.0,
    oer_relaxation_mode: str = DEFAULT_OER_RELAXATION_MODE,
    oer_manual_cation_indices: Optional[Iterable[int]] = None,
):
    """
    Reaction-descriptor based ΔE/ΔG screening for CO2RR or oxygen intermediates.

    CO2RR adsorbates: COOH*, CO*, HCOO*, OCHO*
    OER adsorbates: OOH*, O*, OH*

    When reaction_mode is "OER", the oxygen-intermediate reference scheme
    and thermodynamic corrections are used instead of the CO2RR defaults.
    """
    _rxn_mode = str(reaction_mode).upper()
    if _rxn_mode == "ORR":
        raise ValueError("ORR mode has been removed in SAGE v1.1.0. Use OER, HER, or CO2RR.")
    _is_oxygen = _rxn_mode in {"OER", "OER_COMPETITION", "OXYGEN"}
    _is_orr = False
    _is_oer = _rxn_mode in {"OER", "OER_COMPETITION", "OXYGEN"}
    _oer_relax_mode = str(oer_relaxation_mode or DEFAULT_OER_RELAXATION_MODE).strip().lower()
    if _oer_relax_mode not in OER_RELAXATION_MODES:
        _oer_relax_mode = DEFAULT_OER_RELAXATION_MODE
    _manual_oer_cations = []
    if oer_manual_cation_indices is not None:
        try:
            _manual_oer_cations = [int(i) for i in oer_manual_cation_indices if int(i) >= 0]
        except Exception:
            _manual_oer_cations = []
    (
        slab_u_rel,
        E_slab_u,
        _E_H2_prepare,
        slab_steps,
        z_steps,
        free_steps,
        _h2_steps,
        meta_flags,
        GROOT,
        UROOT,
    ) = _prepare_slab(user_slab_cif, out_root, vac_z, relax_mode)

    target_sites = _build_target_sites(mtype, slab_u_rel, sites, user_ads_sites)

    # OER on oxides requires a different slab/site policy from oxide-HER.
    # HER uses lattice-O/O-anchor sites; AEM OER requires exposed cation sites
    # for cation-bound *OH/*O/*OOH.  This branch is intentionally isolated
    # from HER logic.
    oer_slab_qc = None
    if _is_oer and str(mtype).lower() == "oxide":
        try:
            oer_slab_qc = oxide_oer_slab_suitability(slab_u_rel)
        except Exception as e:
            oer_slab_qc = {"oer_slab_suitability": "qc_failed", "oer_slab_warning": str(e)}
        syms = slab_u_rel.get_chemical_symbols()
        pos_all = np.asarray(slab_u_rel.get_positions(), dtype=float)
        if _manual_oer_cations:
            # Diagnostic/manual Ir_cus mode: use user-specified cation indices
            # rather than the automatic top-cation detector. This is useful for
            # IrO2(110) benchmarking where the active Ir_cus atom is known.
            oer_cation_sites = []
            for ci in _manual_oer_cations:
                if 0 <= int(ci) < len(syms) and syms[int(ci)] not in ANION_SYMBOLS:
                    p_ci = pos_all[int(ci)]
                    oer_cation_sites.append(AdsSite(kind="oer_cation", position=(float(p_ci[0]), float(p_ci[1]), float(p_ci[2])), surface_indices=(int(ci),)))
        elif user_ads_sites:
            # Respect Step-3 Geometry representative sites for oxide-OER.
            # Previous versions redetected OER cation sites here, making the
            # separate OER max-site cap override the UI representative-site
            # slider and sometimes breaking OH/O/OOH triplet consistency.
            oer_cation_sites = []
            for _label, _site in (user_ads_sites or {}).items():
                try:
                    _kind = str(getattr(_site, "kind", "")).strip().lower()
                    _sidx = tuple(int(i) for i in (getattr(_site, "surface_indices", ()) or ()))
                    if _kind not in {"oer_cation", "cation"}:
                        continue
                    if len(_sidx) != 1:
                        continue
                    _ci = int(_sidx[0])
                    if not (0 <= _ci < len(syms)) or syms[_ci] in ANION_SYMBOLS:
                        continue
                    _p = np.asarray(getattr(_site, "position", pos_all[_ci]), dtype=float)
                    if _p.shape[0] < 3:
                        _p = pos_all[_ci]
                    oer_cation_sites.append(AdsSite(kind="oer_cation", position=(float(_p[0]), float(_p[1]), float(_p[2])), surface_indices=(_ci,)))
                except Exception:
                    continue
            if not oer_cation_sites:
                oer_cation_sites = detect_oxide_oer_cation_sites(slab_u_rel, max_sites=max(1, len(list(sites or [])) or 12))
        else:
            oer_cation_sites = detect_oxide_oer_cation_sites(slab_u_rel, max_sites=max(1, len(list(sites or [])) or 12))
        if not oer_cation_sites:
            raise RuntimeError(
                "OER AEM oxide mode requires at least one exposed/top-near cation site. "
                "The current slab appears unsuitable for cation-bound *OH/*O/*OOH placement. "
                "Use a cation-exposed termination such as rutile IrO2(110) with Ir_cus sites, "
                "or provide manual cation indices for diagnostic benchmarking."
            )
        target_sites = []
        for k, site in enumerate(oer_cation_sites):
            sidx = tuple(int(i) for i in getattr(site, "surface_indices", ()) or ())
            ci = int(sidx[0]) if sidx else -1
            label_sym = syms[ci] if (0 <= ci < len(syms)) else "M"
            pos_site = np.asarray(site.position[:3], dtype=float)
            target_sites.append({
                "site_label": f"oer_cation_{k}_{label_sym}{ci}",
                "site_kind": "oer_cation",
                "xy": np.asarray(pos_site[:2], dtype=float),
                "initial_xyz": pos_site,
                "surface_indices": sidx,
                "seed_source": "oxide_oer_manual_cation" if _manual_oer_cations else "oxide_oer_aem_cation_detector",
                "active_cation_index": ci,
                "active_cation_symbol": label_sym,
            })

    # Load thermodynamic corrections (ORR vs CO2RR branch)
    if _is_oxygen:
        ads_corr = _load_orr_thermo()
        thermo_data = None
        thermo_path = Path("ref_gas") / THERMO_ORR_NAME
    else:
        thermo_data, ads_corr, thermo_path = _load_co2rr_thermo()
    gas_E: Dict[str, float] = {}
    gas_src: Dict[str, str] = {}
    if thermo_data is not None:
        E_gas_block = thermo_data.get("E_gas (eV)") or thermo_data.get("E_gas") or {}
        if isinstance(E_gas_block, dict):
            for k, v in E_gas_block.items():
                try:
                    kk = str(k).upper()
                    gas_E[kk] = float(v)
                    gas_src[kk] = "thermo"
                except Exception:
                    pass

    # Always prefer the internally prepared H2 reference (same calc/pipeline).
    gas_E["H2"] = float(_E_H2_prepare)
    gas_src["H2"] = "get_h2_ref"

    # Determine required gas references (ORR does not need CO2; H2O is required)
    required_gas = ["H2O"] if _is_oxygen else ["CO2", "H2O"]
    missing_required = [
        k for k in required_gas if k not in gas_E or not np.isfinite(gas_E[k])
    ]
    if missing_required:
        computed_E, computed_src = _load_or_compute_gas_refs(
            missing_required,
            calc,
            GROOT=GROOT,
            ref_dir="ref_gas",
            steps=min(200, max(0, int(free_steps))),
            fmax=0.05,
            cell=15.0,
        )
        for k in missing_required:
            ku = str(k).upper()
            if ku in computed_E and np.isfinite(computed_E[ku]):
                gas_E[ku] = float(computed_E[ku])
                gas_src[ku] = computed_src.get(ku, "computed")

    # Final validation
    _required_keys = ["H2", "H2O"] if _is_oxygen else ["CO2", "H2", "H2O"]
    for k in _required_keys:
        if k not in gas_E or not np.isfinite(gas_E[k]):
            _mode_label = ("OER" if _is_oer else "CO2RR")
            raise RuntimeError(
                f"{_mode_label} gas reference energies are missing/invalid. "
                f"Missing: {k}. "
                "Provide ref_gas/thermo_CO2RR.json (or thermo_ORR.json) with an 'E_gas (eV)' block, "
                "or ref_gas/{H2O_box.cif,CO2_box.cif}, or allow ASE fallback molecule()."
            )

    E_CO2 = gas_E.get("CO2", 0.0)   # Not used in ORR mode
    E_H2 = gas_E["H2"]
    E_H2O = gas_E["H2O"]

    # --- Optional companion: HER guardrail (single-site; cheap) ---
    her_guard = None
    if her_guardrail:
        try:
            z_steps_g = min(int(z_steps), 120)
            free_steps_g = min(int(free_steps), 150)
            hg_cif = out_root / "her_guardrail_H.cif"
            her_guard = _compute_her_guardrail_from_prepared(
                slab_u_rel,
                float(E_slab_u),
                float(E_H2),
                target_sites,
                z_steps=z_steps_g,
                free_steps=free_steps_g,
                site_preference=her_site_preference,
                use_net_corr=her_use_net_corr,
                out_cif=hg_cif,
            )
            if her_guard is not None:
                her_guard["z_steps_used"] = int(z_steps_g)
                her_guard["free_steps_used"] = int(free_steps_g)
                try:
                    hg_csv = out_root / "results_her_guardrail.csv"
                    pd.DataFrame([her_guard]).to_csv(
                        hg_csv, index=False, float_format="%.6f"
                    )
                    her_guard["csv_path"] = str(Path(hg_csv).resolve())
                except Exception:
                    pass
        except Exception as e:
            her_guard = {"mode": "HER_GUARDRAIL", "error": str(e)}

    rows: List[Dict[str, object]] = []

    for site_seed in target_sites:
        label = str(site_seed.get("site_label", "unknown"))
        kind = str(site_seed.get("site_kind", "unknown"))
        xy = np.asarray(site_seed.get("xy", [np.nan, np.nan]), dtype=float)
        for ads in adspecies:
            ads_clean = ads.replace("*", "").upper()
            if ads_clean not in ADS_TEMPLATE_FILES:
                continue

            seed_variants = [{
                "site_label": str(label),
                "site_kind": str(kind),
                "xy": np.asarray(xy, float),
                "target_anchor_xyz": None,
                "target_anchor_dir": None,
                "oxide_seed_mode": None,
                "surface_channel": None,
                "seed_source": site_seed.get("seed_source"),
                "surface_indices": tuple(int(i) for i in site_seed.get("surface_indices", ()) or ()),
            }]
            if _is_oxygen and mtype == "oxide":
                seed_variants = []
                site_xyz = site_seed.get("initial_xyz", None)
                if site_xyz is None:
                    site_xyz = (float(xy[0]), float(xy[1]), float(np.max(slab_u_rel.get_positions()[:, 2])))
                site_obj = AdsSite(
                    kind=str(kind),
                    position=(float(site_xyz[0]), float(site_xyz[1]), float(site_xyz[2])),
                    surface_indices=tuple(int(i) for i in site_seed.get("surface_indices", ()) or ()),
                )
                # OER benchmark mode uses strict cation-bound AEM placement.
                # ORR keeps the broader exploratory channel policy.  HER is not
                # connected to this branch.
                channel_policy = "oer_aem_cation" if _is_oer else "default"
                channels = expand_oxide_channels_for_adsorbate(ads_clean, policy=channel_policy)
                for ch in channels:
                    x0, y0, z0_default, surface_channel = oxide_surface_seed_position(
                        slab_u_rel, site_obj, ads_clean, channel=ch
                    )
                    target_dir = None
                    if _is_oer and surface_channel == "adsorbate_on_cation":
                        try:
                            _sidx = tuple(int(i) for i in (site_seed.get("surface_indices", ()) or ()))
                            _ci = int(_sidx[0]) if _sidx else int(site_seed.get("active_cation_index", -1))
                            target_dir = _oxide_oer_open_direction(slab_u_rel, _ci)
                        except Exception:
                            target_dir = np.array([0.0, 0.0, 1.0], dtype=float)
                    base_label = f"{label}:{ch}" if len(channels) > 1 else str(label)
                    # Multi-start heights are used only for OER strict cation
                    # benchmark mode.  This is the standard way to avoid single
                    # bad initial z placements dominating *OH/*O/*OOH screening.
                    if _is_oer and surface_channel == "adsorbate_on_cation":
                        default_h = {"O": 1.45, "OH": 1.80, "OOH": 2.05}.get(ads_clean, 1.80)
                        z_cation = float(z0_default) - float(default_h)
                        heights = OER_AEM_CATION_HEIGHT_GRID.get(ads_clean, (float(default_h),))
                    else:
                        z_cation = None
                        heights = (None,)
                    for h_trial in heights:
                        if h_trial is None:
                            z0 = z0_default
                            site_label_seed = base_label
                            h_val = None
                        else:
                            z0 = float(z_cation) + float(h_trial)
                            site_label_seed = f"{base_label}:h{float(h_trial):.2f}"
                            h_val = float(h_trial)
                        seed_variants.append({
                            "site_label": site_label_seed,
                            "oer_base_site_label": base_label,
                            "oer_start_height_A": h_val,
                            "site_kind": "oer_cation" if (_is_oer and surface_channel == "adsorbate_on_cation") else str(kind),
                            "xy": np.asarray([x0, y0], float),
                            "target_anchor_xyz": np.asarray([x0, y0, z0], float),
                            "target_anchor_dir": None if target_dir is None else np.asarray(target_dir, float),
                            "oxide_seed_mode": str(ch),
                            "surface_channel": str(surface_channel),
                            "seed_source": site_seed.get("seed_source"),
                            "surface_indices": tuple(int(i) for i in site_seed.get("surface_indices", ()) or ()),
                            "active_cation_index": site_seed.get("active_cation_index"),
                            "active_cation_symbol": site_seed.get("active_cation_symbol"),
                        })

            for seed in seed_variants:
                slab_ads = _build_adsorbate_on_site(
                    slab_u_rel,
                    np.asarray(seed["xy"], float),
                    ads_clean,
                    dz=1.8,
                    ref_dir="ref_gas",
                    target_anchor_xyz=seed["target_anchor_xyz"],
                    target_anchor_dir=seed.get("target_anchor_dir"),
                )

                site_label_file = str(seed["site_label"]).replace(":", "__")
                initial_cif_path = UROOT / f"sites/user_{site_label_file}_{ads_clean}_initial.cif"
                try:
                    write(initial_cif_path, slab_ads)
                except Exception:
                    pass

                # OER oxide AEM diagnostic modes.  HER/CO2RR/ORR paths keep
                # their original free_steps/fmax behavior.
                oer_diag = bool(
                    _is_oer
                    and str(mtype).lower() == "oxide"
                    and str(seed.get("surface_channel", "")).lower() == "adsorbate_on_cation"
                    and ads_clean in {"O", "OH", "OOH"}
                )
                if oer_diag:
                    if _oer_relax_mode in {"placement_only", "single_point"}:
                        ads_relax_steps = 0
                        ads_relax_fmax = float(OER_AEM_SHORT_RELAX_FMAX)
                        relax_ads_flag = False
                    elif _oer_relax_mode == "short_relax":
                        ads_relax_steps = int(min(int(free_steps), int(OER_AEM_SHORT_RELAX_STEPS)))
                        ads_relax_fmax = float(OER_AEM_SHORT_RELAX_FMAX)
                        relax_ads_flag = True
                    else:  # normal_relax
                        ads_relax_steps = int(free_steps)
                        ads_relax_fmax = 0.05
                        relax_ads_flag = True
                else:
                    ads_relax_steps = int(free_steps)
                    ads_relax_fmax = 0.05
                    relax_ads_flag = True

                slab_ads_rel, E_slab_ads, opt_meta_ads = _relax_slab_ads(
                    slab_ads,
                    n_slab_atoms=meta_flags["n_atoms"],
                    steps=ads_relax_steps,
                    fmax=ads_relax_fmax,
                    relax_ads=bool(relax_ads_flag),
                )
                if isinstance(opt_meta_ads, dict):
                    opt_meta_ads["oer_diagnostic_mode"] = str(_oer_relax_mode if oer_diag else "not_oer_diagnostic")
                    opt_meta_ads["oer_diagnostic_active"] = bool(oer_diag)
                    opt_meta_ads["oer_ads_relax_steps_used"] = int(ads_relax_steps)
                    opt_meta_ads["oer_ads_relax_fmax_used"] = float(ads_relax_fmax)

                relaxed_cif_path = UROOT / f"sites/user_{site_label_file}_{ads_clean}.cif"
                write(relaxed_cif_path, slab_ads_rel)

                # --- lateral displacement & QA metrics (anchor-based) ---
                coords = slab_ads_rel.get_positions()
                symbols_all = slab_ads_rel.get_chemical_symbols()
                n_slab = int(meta_flags["n_atoms"])
    
                slab_coords3 = coords[:n_slab, :]
                ads_coords3 = coords[n_slab:, :]
                ads_symbols = list(symbols_all[n_slab:])
                anchor_xy, anchor_z, anchor_mode = _co2rr_anchor_xy(
                    ads_coords3, ads_symbols, ads_clean
                )
                slab_only = slab_ads_rel[:n_slab]

                migrate_thr = (
                    float(MIGRATE_THR)
                    if ads_clean == "CO"
                    else float(max(MIGRATE_THR, 2.5))
                )
                tracking = _resolve_site_tracking(
                    slab_only=slab_only,
                    mtype=mtype,
                    seed_kind=seed["site_kind"],
                    initial_xy=np.asarray(seed["xy"], float),
                    final_anchor_xy=anchor_xy,
                    classification_mode=("oxide_oer_cation" if (_is_oer and str(mtype).lower() == "oxide" and str(seed.get("surface_channel", "")).lower() == "adsorbate_on_cation") else "auto"),
                    disp_threshold=float(migrate_thr),
                )
                tracking["initial_surface_indices"] = _fmt_surface_indices(seed.get("surface_indices", ()))
                disp = float(tracking["lateral_disp_mic(Å)"])

                z_top = (
                    float(np.max(slab_coords3[:, 2]))
                    if slab_coords3.shape[0]
                    else float("nan")
                )
                anchor_z_above_top = (
                    float(anchor_z - z_top)
                    if np.isfinite(anchor_z) and np.isfinite(z_top)
                    else float("nan")
                )

                min_slab_dist = _co2rr_min_slab_dist(slab_coords3, ads_coords3)
                broken = (
                    bool(_co2rr_internal_broken(ads_coords3, ads_symbols, ads_clean))
                    if ads_coords3.shape[0]
                    else True
                )
                crashed = bool(min_slab_dist < 0.70) if np.isfinite(min_slab_dist) else False
                desorbed = (
                    bool((anchor_z_above_top > 4.0) or (min_slab_dist > 3.0))
                    if np.isfinite(min_slab_dist)
                    else False
                )

                migrated = bool(tracking["migrated"])

                oxygen_metrics = _oxygen_adsorbate_state_metrics(
                    slab_coords3,
                    ads_coords3,
                    ads_symbols,
                    ads_clean,
                    surface_channel=seed.get("surface_channel"),
                    slab_symbols=list(symbols_all[:n_slab]),
                    target_cation_indices=tuple(int(i) for i in seed.get("surface_indices", ()) or ()),
                ) if _is_oxygen else {}

                legacy_internal_broken = bool(broken)
                oxygen_override_broken = False
                # O* is monatomic. Legacy CO2RR internal-bond QA can falsely mark it
                # as broken because there is no internal bond to check. For oxygen
                # intermediates, accept monatomic O* when the oxygen-specific cation
                # binding metrics are valid. Detached/crashed/desorbed states are
                # still rejected by the separate checks below.
                if (
                    _is_oxygen
                    and ads_clean == "O"
                    and legacy_internal_broken
                    and bool(oxygen_metrics.get("oxygen_adsorbate", False))
                    and bool(oxygen_metrics.get("oxygen_internal_bond_ok", False))
                    and bool(oxygen_metrics.get("oxygen_surface_bond_ok", False))
                    and bool(oxygen_metrics.get("oxygen_bound_to_cation", False))
                ):
                    broken = False
                    oxygen_override_broken = True
                if _is_oxygen:
                    oxygen_metrics["legacy_internal_broken"] = bool(legacy_internal_broken)
                    oxygen_metrics["legacy_internal_broken_overridden_for_oer"] = bool(oxygen_override_broken)
                    oxygen_metrics["oxygen_qa_policy"] = "oer_monatomic_O_surface_binding_override" if oxygen_override_broken else "oxygen_default"

                qa = "ok"
                if crashed:
                    qa = "crashed"
                elif desorbed:
                    qa = "desorbed"
                elif broken:
                    qa = "broken"
                elif bool(oxygen_metrics.get("oxygen_adsorbate", False)) and not bool(oxygen_metrics.get("valid_for_oer_summary", False)):
                    # Do not allow detached or wrong-channel oxygen intermediates to be
                    # silently retained as "migrated" OER candidates.
                    state_class = str(oxygen_metrics.get("oer_state_class", "invalid_oxygen_intermediate"))
                    qa = "detached" if "detached" in state_class else state_class
                elif (
                    _is_oer
                    and bool(oxygen_metrics.get("valid_for_oer_summary", False))
                    and str(tracking.get("site_family", "")).lower() == "oxide_oer_cation"
                    and str(tracking.get("initial_site_kind", "")).lower() == "oer_cation"
                    and str(tracking.get("final_site_kind", "")).lower() == "oer_cation"
                ):
                    # OER cation-bound intermediates often relax laterally while remaining
                    # bonded to the same exposed-cation basin.  Treat this as a relaxed
                    # bound state, not as a physical migration, when the OER site taxonomy
                    # remains stable.
                    qa = "bound_relaxed"
                    migrated = False
                    tracking["migrated"] = False
                    tracking["migration_basis"] = "none"
                    tracking["migration_type"] = "none"
                    tracking["qc_flags"] = [f for f in tracking.get("qc_flags", []) if f != "migrated_site"]
                elif migrated:
                    qa = "bound_migrated" if bool(oxygen_metrics.get("valid_for_oer_summary", False)) else "migrated"

                relaxed_site = tracking["final_site_kind"]
                if relaxed_site == "unknown":
                    relaxed_site = None

                qc_flags = list(tracking["qc_flags"])
                if crashed:
                    qc_flags.append("crashed_adsorbate")
                if desorbed:
                    qc_flags.append("desorbed_adsorbate")
                if broken:
                    qc_flags.append("broken_adsorbate")

                # --- reaction descriptor ---
                # CO2(g) + H+ + e- → COOH*
                # CO2(g) + 2H+ + 2e- → CO* + H2O(l)
                # CO2(g) + H+ + e- → HCOO* / OCHO*
                # ORR (Norskov CHE): E_O2_eff = 2·E_H2O - 2·E_H2 + 4×1.23 eV
                if ads_clean == "COOH":
                    E_reagents = E_CO2 + 0.5 * E_H2
                    ref_rxn = "CO2 + 1/2 H2"
                elif ads_clean == "CO":
                    E_reagents = E_CO2 + E_H2 - E_H2O
                    ref_rxn = "CO2 + H2 - H2O"
                elif ads_clean in ("HCOO", "OCHO"):
                    E_reagents = E_CO2 + 0.5 * E_H2
                    ref_rxn = "CO2 + 1/2 H2"
                # ── ORR/OER oxygen intermediates ─────────────────────────────
                elif ads_clean == "OOH":
                    if _is_oer:
                        # OER AEM standard CHE intermediate:
                        # ΔG_*OOH = G_*OOH - G_* - (2H2O - 3/2 H2)
                        E_reagents = 2.0 * E_H2O - 1.5 * E_H2
                        ref_rxn = "2H2O - 3/2 H2 [OER_AEM]"
                    else:
                        # ORR descriptor: O2 + (H+ + e-) -> OOH*
                        _E_O2_eff = 2 * E_H2O - 2 * E_H2 + 4 * ORR_EQUIL_POTENTIAL
                        E_reagents = _E_O2_eff + 0.5 * E_H2
                        ref_rxn = "O2_eff + 1/2 H2 [ORR]"
                elif ads_clean == "O":
                    if _is_oer:
                        # OER AEM standard CHE intermediate:
                        # ΔG_*O = G_*O - G_* - (H2O - H2)
                        E_reagents = E_H2O - E_H2
                        ref_rxn = "H2O - H2 [OER_AEM]"
                    else:
                        # ORR descriptor: O2 + 2(H+ + e-) -> O* + H2O
                        _E_O2_eff = 2 * E_H2O - 2 * E_H2 + 4 * ORR_EQUIL_POTENTIAL
                        E_reagents = _E_O2_eff + E_H2 - E_H2O
                        ref_rxn = "O2_eff + H2 - H2O [ORR]"
                elif ads_clean == "OH":
                    if _is_oer:
                        # OER AEM standard CHE intermediate:
                        # ΔG_*OH = G_*OH - G_* - (H2O - 1/2 H2)
                        E_reagents = E_H2O - 0.5 * E_H2
                        ref_rxn = "H2O - 1/2 H2 [OER_AEM]"
                    else:
                        # ORR descriptor: O2 + 3(H+ + e-) -> OH* + H2O
                        _E_O2_eff = 2 * E_H2O - 2 * E_H2 + 4 * ORR_EQUIL_POTENTIAL
                        E_reagents = _E_O2_eff + 1.5 * E_H2 - E_H2O
                        ref_rxn = "O2_eff + 3/2 H2 - H2O [ORR]"
                # ─────────────────────────────────────────────────────────────
                else:
                    E_reagents = 0.0
                    ref_rxn = "custom"
    
                dE_raw = float(E_slab_ads - E_slab_u)
                dE_ads = float(E_slab_ads - E_slab_u - E_reagents)
                g_corr = float(ads_corr.get(ads_clean, 0.0))
                dG_ads = float(dE_ads + g_corr)
    
                row: Dict[str, object] = {
                    "mode": ("OER" if _is_oer else "CO2RR"),
                    "adsorbate": ads_clean,
                    "site": seed["site_kind"],
                    "site_label": seed["site_label"],
                    "seed_source": seed.get("seed_source"),
                    "seed_site_kind": tracking["seed_site_kind"],
                    "initial_site_kind": tracking["initial_site_kind"],
                    "final_site_kind": tracking["final_site_kind"],
                    "site_tracking_mode": tracking["classification_mode"],
                    "site_family": tracking["site_family"],
                    "migration_basis": tracking["migration_basis"],
                    "migration_type": tracking["migration_type"],
                    "initial_surface_indices": tracking["initial_surface_indices"],
                    "final_surface_indices": tracking["final_surface_indices"],
                    "qc_flags": ";".join(qc_flags),
                    "MODEL": MODEL_NAME,
                    "DEVICE": DEVICE,
                    "layers": layers,
                    "vac_z(Å)": vac_z,
                    "E_slab_user (eV)": float(E_slab_u),
                    "E_slab+ads_user (eV)": float(E_slab_ads),
                    "ads_relax_elapsed_s": float((opt_meta_ads or {}).get("elapsed_s", 0.0)),
                    "ads_relax_n_steps": int((opt_meta_ads or {}).get("n_steps", 0)),
                    "ads_relax_converged": (opt_meta_ads or {}).get("converged", None),
                    "ads_relax_error": (opt_meta_ads or {}).get("error", None),
                    "oer_diagnostic_mode": (opt_meta_ads or {}).get("oer_diagnostic_mode", None),
                    "oer_diagnostic_active": (opt_meta_ads or {}).get("oer_diagnostic_active", None),
                    "oer_ads_relax_steps_used": (opt_meta_ads or {}).get("oer_ads_relax_steps_used", None),
                    "oer_ads_relax_fmax_used": (opt_meta_ads or {}).get("oer_ads_relax_fmax_used", None),
                    "initial_structure_cif": str(Path(initial_cif_path).resolve()),
                    "structure_cif": str(Path(relaxed_cif_path).resolve()),
                    "ΔE_ads_user (eV)": float(dE_ads),
                    "ΔE_raw(slab+ads - slab) (eV)": float(dE_raw),
                    "E_ref_reagents (eV)": float(E_reagents),
                    "G_correction (eV)": float(g_corr),
                    "ref_rxn": ref_rxn,
                    "ΔG_ads (eV)": float(dG_ads) if export_absolute else None,
                    "ΔG_ads_raw_reference_applied (eV)": float(dG_ads) if export_absolute else None,
                    "OER_reference_scheme": ("standard_OER_AEM_H2O_H2" if _is_oer and ads_clean in {"OH", "O", "OOH"} else "other"),
                    "oxygen_intermediate_role": ("OER_AEM" if _is_oer and ads_clean in {"OH", "O", "OOH"} else "other"),
                    "initial_anchor_x(Å)": tracking["initial_anchor_x(Å)"],
                    "initial_anchor_y(Å)": tracking["initial_anchor_y(Å)"],
                    "final_anchor_x(Å)": tracking["final_anchor_x(Å)"],
                    "final_anchor_y(Å)": tracking["final_anchor_y(Å)"],
                    "ads_lateral_disp(Å)": float(disp),
                    "final_site_match_dist(Å)": float(tracking["final_site_match_dist(Å)"]),
                    "oxide_seed_mode": seed.get("oxide_seed_mode"),
                    "surface_channel": seed.get("surface_channel"),
                    "active_cation_index": seed.get("active_cation_index"),
                    "active_cation_symbol": seed.get("active_cation_symbol"),
                    "oer_base_site_label": seed.get("oer_base_site_label"),
                    "oer_start_height_A": seed.get("oer_start_height_A"),
                    "ads_anchor_mode": anchor_mode,
                    "ads_anchor_z_above_top(Å)": float(anchor_z_above_top),
                    "ads_min_slab_dist(Å)": float(min_slab_dist),
                    "migrate_thr(Å)": float(migrate_thr),
                    "migrated": bool(migrated),
                    "desorbed": bool(desorbed),
                    "broken": bool(broken),
                    "crashed": bool(crashed),
                    "qa": qa,
                    "relaxed_site": relaxed_site,
                    **oxygen_metrics,
                    "slab_relax_drop": meta_flags["slab_relax_drop"],
                    "vac_warning": meta_flags["vac_warning"],
                }
                rows.append(row)

    # Write metadata even when no rows are produced
    if not rows:
        _out_name = ("results_sites_oer.csv" if _is_oer else ("results_sites_orr.csv" if _is_orr else "results_sites_co2rr.csv"))
        out_csv = out_root / _out_name
        pd.DataFrame([]).to_csv(out_csv, index=False)
        _mode_str = ("OER" if _is_oer else "CO2RR")
        meta = {
            "mode": _mode_str,
            "user_slab": str(Path(user_slab_cif).resolve()),
            "relax_mode": relax_mode,
            "steps": {"slab": slab_steps, "ads": free_steps},
            "adspecies": list(adspecies),
            "OER_RELAXATION_MODE": str(_oer_relax_mode) if _is_oer else None,
            "OER_MANUAL_CATION_INDICES": list(_manual_oer_cations) if _is_oer else [],
            "ADS_CORR_effective (eV)": ads_corr,
            "THERMO_FILE": str(thermo_path.resolve()),
            "GAS_REF_E_USED (eV)": gas_E,
            "GAS_REF_SOURCE": gas_src,
            "HER_GUARDRAIL": her_guard,
            "DISP_METRIC": "anchor_xy",
            "MIGRATE_THR_CO(Å)": float(MIGRATE_THR),
            "MIGRATE_THR_COOH_HCOO_OCHO(Å)": float(max(MIGRATE_THR, 2.5)),
            "Model": MODEL_NAME,
            "Device": DEVICE,
            "warnings": {
                "slab_relax_drop": meta_flags["slab_relax_drop"],
                "vac_warning": meta_flags["vac_warning"],
            },
        }
        if thermo_data is not None:
            meta["THERMO_RAW"] = thermo_data
        _meta_name_empty = ("meta_oer.json" if _is_oer else ("meta_orr.json" if _is_orr else "meta_co2rr.json"))
        (out_root / _meta_name_empty).write_text(json.dumps(meta, indent=2))
        return str(out_csv), meta

    # Sort results by absolute descriptor value
    df = pd.DataFrame(rows)
    key = (
        "ΔG_ads (eV)"
        if export_absolute and "ΔG_ads (eV)" in df.columns
        else "ΔE_ads_user (eV)"
    )
    df = df.assign(abs_val=lambda x: x[key].abs()).sort_values(["abs_val"])

    _out_name2 = ("results_sites_oer.csv" if _is_oer else ("results_sites_orr.csv" if _is_orr else "results_sites_co2rr.csv"))
    out_csv = out_root / _out_name2
    df.drop(columns=["abs_val"], errors="ignore").to_csv(
        out_csv,
        index=False,
        float_format="%.6f",
    )

    orr_summary_csv = None
    if _is_oxygen:
        try:
            _oer_summary = _summarize_oer_descriptor_table(df.drop(columns=["abs_val"], errors="ignore"), U=float(orr_u))
            if _oer_summary is not None and (not _oer_summary.empty):
                orr_summary_csv = out_root / "results_oer_competition_summary.csv"
                _oer_summary.to_csv(orr_summary_csv, index=False, float_format="%.6f")
        except Exception:
            orr_summary_csv = None

    _mode_str2 = ("OER" if _is_oer else "CO2RR")
    meta = {
        "mode": _mode_str2,
        "user_slab": str(Path(user_slab_cif).resolve()),
        "relax_mode": relax_mode,
        "steps": {"slab": slab_steps, "ads": free_steps},
        "adspecies": list(adspecies),
        "ADS_CORR_effective (eV)": ads_corr,
        "THERMO_FILE": str(thermo_path.resolve()),
        "GAS_REF_E_USED (eV)": gas_E,
        "GAS_REF_SOURCE": gas_src,
        "HER_GUARDRAIL": her_guard,
        "OXYGEN_SUMMARY_CSV": str(orr_summary_csv) if orr_summary_csv is not None else None,
        "DISP_METRIC": "anchor_xy",
        "MIGRATE_THR_CO(Å)": float(MIGRATE_THR),
        "MIGRATE_THR_COOH_HCOO_OCHO(Å)": float(max(MIGRATE_THR, 2.5)),
        "Model": MODEL_NAME,
        "Device": DEVICE,
        "warnings": {
            "slab_relax_drop": meta_flags["slab_relax_drop"],
            "vac_warning": meta_flags["vac_warning"],
        },
    }
    if thermo_data is not None:
        meta["THERMO_RAW"] = thermo_data

    _meta_name = ("meta_oer.json" if _is_oer else ("meta_orr.json" if _is_orr else "meta_co2rr.json"))
    (out_root / _meta_name).write_text(json.dumps(meta, indent=2))

    return str(out_csv), meta


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def run_metal_che(
    user_slab_cif: str,
    out_root: str | Path = "calc/metal_che",
    sites: Iterable[str] = ("ontop", "bridge", "hollow"),
    vac_z: float = 20.0,
    layers: int = 7,
    export_absolute: bool = True,
    use_net_corr: bool = True,
    gas: str = "H2",
    relax_mode: str = "Normal",
    user_ads_sites: Optional[Mapping[str, object]] = None,
    her_guardrail: bool = False,
    her_site_preference: str = "ontop",
    her_use_net_corr: bool = True,
    thermo_mode: str = "CHE correction (fast screening)",
    zpe_target_mode: str = "Best-ranked by CHE",
    zpe_target_label: str | None = None,
    local_zpe_cutoff: float = 2.5,
    local_zpe_max_neighbors: int = 3,
    oxide_descriptor_mode: str = "Basic HER screening",
    oxide_descriptor_max_reactive_per_kind: int = 2,
    oxide_descriptor_pair_limit: int = 6,
    her_relaxation_scope: str = "rigid",
    her_n_fix_layers: int = 2,
):
    return _run_her_che(
        "metal",
        user_slab_cif,
        Path(out_root),
        sites,
        vac_z,
        layers,
        export_absolute,
        use_net_corr,
        gas,
        relax_mode,
        user_ads_sites,
        her_guardrail=her_guardrail,
        her_site_preference=her_site_preference,
        her_use_net_corr=her_use_net_corr,
        thermo_mode=thermo_mode,
        zpe_target_mode=zpe_target_mode,
        zpe_target_label=zpe_target_label,
        local_zpe_cutoff=local_zpe_cutoff,
        local_zpe_max_neighbors=local_zpe_max_neighbors,
        oxide_descriptor_mode=oxide_descriptor_mode,
        oxide_descriptor_max_reactive_per_kind=oxide_descriptor_max_reactive_per_kind,
        oxide_descriptor_pair_limit=oxide_descriptor_pair_limit,
        her_relaxation_scope=her_relaxation_scope,
        her_n_fix_layers=her_n_fix_layers,
    )


def run_oxide_che(
    user_slab_cif: str,
    out_root: str | Path = "calc/oxide_che",
    sites: Iterable[str] = ("hollow", "bridge", "ontop"),
    vac_z: float = 30.0,
    layers: int = 7,
    export_absolute: bool = True,
    use_che_shift: bool = True,
    gas: str = "H2",
    relax_mode: str = "Normal",
    user_ads_sites: Optional[Mapping[str, object]] = None,
    her_guardrail: bool = False,
    her_site_preference: str = "ontop",
    her_use_net_corr: bool = True,
    thermo_mode: str = "CHE correction (fast screening)",
    zpe_target_mode: str = "Best-ranked by CHE",
    zpe_target_label: str | None = None,
    local_zpe_cutoff: float = 2.5,
    local_zpe_max_neighbors: int = 3,
    oxide_descriptor_mode: str = "Basic HER screening",
    oxide_descriptor_max_reactive_per_kind: int = 2,
    oxide_descriptor_pair_limit: int = 6,
    her_relaxation_scope: str = "rigid",
    her_n_fix_layers: int = 2,
):
    return _run_her_che(
        "oxide",
        user_slab_cif,
        Path(out_root),
        sites,
        vac_z,
        layers,
        export_absolute,
        use_che_shift,
        gas,
        relax_mode,
        user_ads_sites,
        her_guardrail=her_guardrail,
        her_site_preference=her_site_preference,
        her_use_net_corr=her_use_net_corr,
        thermo_mode=thermo_mode,
        zpe_target_mode=zpe_target_mode,
        zpe_target_label=zpe_target_label,
        local_zpe_cutoff=local_zpe_cutoff,
        local_zpe_max_neighbors=local_zpe_max_neighbors,
        oxide_descriptor_mode=oxide_descriptor_mode,
        oxide_descriptor_max_reactive_per_kind=oxide_descriptor_max_reactive_per_kind,
        oxide_descriptor_pair_limit=oxide_descriptor_pair_limit,
        her_relaxation_scope=her_relaxation_scope,
        her_n_fix_layers=her_n_fix_layers,
    )


def run_metal_co2rr_che(
    user_slab_cif: str,
    out_root: str | Path = "calc/metal_co2rr",
    sites: Iterable[str] = ("ontop", "bridge", "hollow"),
    vac_z: float = 20.0,
    layers: int = 7,
    export_absolute: bool = True,
    relax_mode: str = "Normal",
    user_ads_sites: Optional[Mapping[str, object]] = None,
    adspecies: Tuple[str, ...] = ("CO*", "COOH*", "HCOO*", "OCHO*"),
    her_guardrail: bool = False,
    her_site_preference: str = "ontop",
    her_use_net_corr: bool = True,
):
    return _run_co2rr_che(
        "metal",
        user_slab_cif,
        Path(out_root),
        sites,
        vac_z,
        layers,
        export_absolute,
        relax_mode,
        user_ads_sites,
        adspecies,
        her_guardrail=her_guardrail,
        her_site_preference=her_site_preference,
        her_use_net_corr=her_use_net_corr,
    )


def run_oxide_co2rr_che(
    user_slab_cif: str,
    out_root: str | Path = "calc/oxide_co2rr",
    sites: Iterable[str] = ("hollow", "bridge", "ontop"),
    vac_z: float = 30.0,
    layers: int = 7,
    export_absolute: bool = True,
    relax_mode: str = "Normal",
    user_ads_sites: Optional[Mapping[str, object]] = None,
    adspecies: Tuple[str, ...] = ("CO*", "COOH*", "HCOO*", "OCHO*"),
    her_guardrail: bool = False,
    her_site_preference: str = "ontop",
    her_use_net_corr: bool = True,
):
    return _run_co2rr_che(
        "oxide",
        user_slab_cif,
        Path(out_root),
        sites,
        vac_z,
        layers,
        export_absolute,
        relax_mode,
        user_ads_sites,
        adspecies,
        her_guardrail=her_guardrail,
        her_site_preference=her_site_preference,
        her_use_net_corr=her_use_net_corr,
    )


# =====================================================================
# OER competition Public API
# Uses the same OOH*/O*/OH* oxygen-intermediate backend as ORR, but writes
# OER-oriented output names and summary metadata.
# =====================================================================

def run_metal_oer_che(
    user_slab_cif: str,
    out_root: str | Path = "calc/metal_oer",
    sites: Iterable[str] = ("ontop", "bridge", "hollow"),
    vac_z: float = 20.0,
    layers: int = 7,
    export_absolute: bool = True,
    relax_mode: str = "Normal",
    user_ads_sites: Optional[Mapping[str, object]] = None,
    adspecies: Tuple[str, ...] = ("OOH*", "O*", "OH*"),
    her_guardrail: bool = False,
    her_site_preference: str = "ontop",
    her_use_net_corr: bool = True,
    orr_u: float = 0.0,
    oer_relaxation_mode: str = DEFAULT_OER_RELAXATION_MODE,
):
    """
    OER-competition screening on metal surfaces.

    This is not a HER branch.  It evaluates OOH*/O*/OH* descriptors and
    produces an OER penalty summary for anodic candidate screening.
    """
    return _run_co2rr_che(
        "metal",
        user_slab_cif,
        Path(out_root),
        sites,
        vac_z,
        layers,
        export_absolute,
        relax_mode,
        user_ads_sites,
        adspecies,
        her_guardrail=her_guardrail,
        her_site_preference=her_site_preference,
        her_use_net_corr=her_use_net_corr,
        reaction_mode="OER",
        orr_u=orr_u,
        oer_relaxation_mode=oer_relaxation_mode,
    )


def run_oxide_oer_che(
    user_slab_cif: str,
    out_root: str | Path = "calc/oxide_oer",
    sites: Iterable[str] = ("hollow", "bridge", "ontop"),
    vac_z: float = 30.0,
    layers: int = 7,
    export_absolute: bool = True,
    relax_mode: str = "Normal",
    user_ads_sites: Optional[Mapping[str, object]] = None,
    adspecies: Tuple[str, ...] = ("OOH*", "O*", "OH*"),
    her_guardrail: bool = False,
    her_site_preference: str = "ontop",
    her_use_net_corr: bool = True,
    orr_u: float = 0.0,
    oer_relaxation_mode: str = DEFAULT_OER_RELAXATION_MODE,
    oer_manual_cation_indices: Optional[Iterable[int]] = None,
):
    """
    OER-competition screening on oxide surfaces.

    AEM OER summaries use cation/mixed oxygen-intermediate channels and
    explicitly exclude lattice-O protonation fallback rows.
    """
    return _run_co2rr_che(
        "oxide",
        user_slab_cif,
        Path(out_root),
        sites,
        vac_z,
        layers,
        export_absolute,
        relax_mode,
        user_ads_sites,
        adspecies,
        her_guardrail=her_guardrail,
        her_site_preference=her_site_preference,
        her_use_net_corr=her_use_net_corr,
        reaction_mode="OER",
        orr_u=orr_u,
        oer_relaxation_mode=oer_relaxation_mode,
        oer_manual_cation_indices=oer_manual_cation_indices,
    )


def _oer_risk_from_eta(eta_oer: float) -> tuple[str, float, str]:
    """Map OER overpotential proxy to competition-risk metadata.

    In anodic organic-oxidation screening, low ηOER is a penalty because it
    indicates that water oxidation can compete strongly with the target organic
    oxidation pathway.
    """
    try:
        eta = float(eta_oer)
    except Exception:
        return "unknown", float("nan"), "eta_OER unavailable"
    if eta < 0.35:
        return "high", 1.0, "low OER overpotential proxy; high OER-competition risk"
    if eta < 0.60:
        return "medium", 0.5, "moderate OER-competition risk"
    return "low", 0.0, "high OER overpotential proxy; lower OER-competition risk"


def _oer_steps_from_intermediates(dg_oh: float, dg_o: float, dg_ooh: float) -> tuple[dict[str, float], float, str]:
    """Return standard 4-step OER AEM free energies and ηOER.

    Inputs are standard CHE intermediate free energies:
      ΔG_*OH  = G_*OH  - G_* - (H2O - 1/2 H2)
      ΔG_*O   = G_*O   - G_* - (H2O - H2)
      ΔG_*OOH = G_*OOH - G_* - (2H2O - 3/2 H2)
    """
    steps = {
        "OER_step1_H2O_to_OH": float(dg_oh),
        "OER_step2_OH_to_O": float(dg_o - dg_oh),
        "OER_step3_O_to_OOH": float(dg_ooh - dg_o),
        "OER_step4_OOH_to_O2": float(4.92 - dg_ooh),
    }
    pds = max(steps, key=steps.get)
    eta = float(max(steps.values()) - 1.23)
    return steps, eta, pds


def _summarize_oer_descriptor_table(df: pd.DataFrame, U: float = 0.0) -> pd.DataFrame:
    """
    Convert an oxygen-intermediate descriptor table into OER step thermodynamics.

    Important: ΔG_ads (eV) must already be referenced to standard OER AEM H2O/H2 intermediates.

    OER safety policy:
      - If `valid_for_oer_summary` exists, only rows explicitly marked True are used.
      - Otherwise, rows with qa in {ok, bound_migrated} are used.
      - Rows from lattice-O protonation / anion_o fallback channels are not mixed
        into adsorbate-evolution-mechanism OER summaries.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    need = {"adsorbate", "ΔG_ads (eV)", "site_label"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    work = df.copy()
    mode_series = work["mode"].astype(str).str.upper() if "mode" in work.columns else pd.Series(["OER"] * len(work), index=work.index)
    is_oer_table = True

    if "valid_for_oer_summary" in work.columns:
        valid_mask = work["valid_for_oer_summary"].astype(str).str.lower().isin(["true", "1", "yes"])
        work = work.loc[valid_mask].copy()
    elif "qa" in work.columns:
        qa = work["qa"].astype(str).str.strip().str.lower()
        work = work.loc[qa.isin(["ok", "bound_relaxed", "bound_migrated"])].copy()

    if "surface_channel" in work.columns:
        bad_channels = {
            "protonated_lattice_oxygen",
            "anion_o",
            "lattice_protonation_disabled_for_aem",
            "fallback_site",
        }
        ch = work["surface_channel"].astype(str).str.strip().str.lower()
        work = work.loc[~ch.isin(bad_channels)].copy()

    if work.empty:
        return pd.DataFrame()

    use_base_label = False
    if "oer_base_site_label" in work.columns:
        _base = work["oer_base_site_label"].where(~work["oer_base_site_label"].isna(), "").astype(str).str.strip()
        use_base_label = bool((_base != "").any())
    if use_base_label:
        group_cols = [c for c in ["oer_base_site_label", "site", "oxide_seed_mode", "surface_channel"] if c in work.columns]
    else:
        group_cols = [c for c in ["site_label", "site", "oxide_seed_mode", "surface_channel"] if c in work.columns]

    rows = []
    for keys, sub in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        vals: dict[str, float] = {}
        qa_map: dict[str, str] = {}
        state_map: dict[str, str] = {}
        height_map: dict[str, object] = {}
        ref_map: dict[str, str] = {}

        # Multi-start policy: for each intermediate, keep the lowest-energy valid geometry.
        for ads_name, sub_ads in sub.groupby(sub["adsorbate"].astype(str).str.upper(), dropna=False):
            tmp = sub_ads.copy()
            tmp["_g"] = pd.to_numeric(tmp.get("ΔG_ads (eV)"), errors="coerce")
            tmp = tmp[np.isfinite(tmp["_g"])].copy()
            if tmp.empty:
                continue
            ridx = tmp["_g"].idxmin()
            r = tmp.loc[ridx]
            a = str(r.get("adsorbate", "")).upper()
            vals[a] = float(r["_g"])
            qa_map[a] = str(r.get("qa", ""))
            state_map[a] = str(r.get("oer_state_class", ""))
            height_map[a] = r.get("oer_start_height_A", None)
            ref_map[a] = str(r.get("ref_rxn", ""))

        # ORR legacy summary requires explicit OOH/O/OH.
        if not {"OOH", "O", "OH"}.issubset(vals.keys()):
            continue
        # Explicit OER AEM summary.  In OER mode these vals use standard H2O/H2
        # references.  In ORR mode this is kept as a formal reverse descriptor only.
        steps_oer_explicit, eta_oer_explicit, pds_oer_explicit = _oer_steps_from_intermediates(
            vals["OH"], vals["O"], vals["OOH"]
        )
        dgooh_minus_dgoh = float(vals["OOH"] - vals["OH"])

        # Scaling-proxy OER summary, useful as a diagnostic fallback and sanity check.
        dg_ooh_scaling = float(vals["OH"] + 3.20)
        steps_oer_scaling, eta_oer_scaling, pds_oer_scaling = _oer_steps_from_intermediates(
            vals["OH"], vals["O"], dg_ooh_scaling
        )
        steps_oer_scaling = {f"{k}_scaling_proxy": v for k, v in steps_oer_scaling.items()}

        explicit_available = {"OOH", "O", "OH"}.issubset(vals.keys())
        # OOH-OH scaling sanity check.  Valid geometry alone is insufficient for
        # trusting explicit OOH thermodynamics on oxides; peroxide-like OOH can be
        # over-/under-stabilized depending on site and model.
        scaling_deviation = float(abs(dgooh_minus_dgoh - 3.20))
        if 2.80 <= dgooh_minus_dgoh <= 3.60:
            explicit_ooh_confidence = "usable"
        elif (2.50 <= dgooh_minus_dgoh < 2.80) or (3.60 < dgooh_minus_dgoh <= 3.90):
            explicit_ooh_confidence = "caution"
        else:
            explicit_ooh_confidence = "unreliable"

        if is_oer_table:
            if explicit_available and explicit_ooh_confidence == "usable" and (0.35 <= float(eta_oer_explicit) <= 1.20):
                source = "explicit"
                recommendation_basis = "explicit_OOH_passed_scaling_and_eta_sanity"
            else:
                source = "scaling_proxy"
                if not explicit_available:
                    recommendation_basis = "explicit_triplet_missing"
                elif explicit_ooh_confidence != "usable":
                    recommendation_basis = "scaling_proxy_due_to_explicit_OOH_scaling_deviation"
                else:
                    recommendation_basis = "scaling_proxy_due_to_eta_outside_IrO2_sanity_window"

        eta_selected = eta_oer_explicit if source == "explicit" else eta_oer_scaling
        pds_selected = pds_oer_explicit if source == "explicit" else pds_oer_scaling
        risk, penalty, note = _oer_risk_from_eta(eta_selected)
        if is_oer_table and source == "scaling_proxy":
            note = str(note) + "; recommended value uses OOH-OH scaling proxy"
        benchmark_label = (
            "explicit_benchmark_like" if (source == "explicit" and explicit_ooh_confidence == "usable")
            else "scaling_proxy_recommended" if source == "scaling_proxy"
            else "diagnostic_only"
        )

        row = {
            **key_map,
            "OER_reference_scheme": "standard_OER_AEM_H2O_H2",
            "OER_summary_source": source,
            "OER_explicit_available": bool(explicit_available),
            "OOH_OH_scaling_deviation_from_3p20 (eV)": float(scaling_deviation),
            "explicit_OOH_confidence": explicit_ooh_confidence,
            "eta_OER_recommended (V)": float(eta_selected),
            "OER_recommendation_basis": recommendation_basis,
            "benchmark_consistency_label": benchmark_label,
            "ΔG_OOH (eV)": float(vals["OOH"]),
            "ΔG_O (eV)": float(vals["O"]),
            "ΔG_OH (eV)": float(vals["OH"]),
            "ΔG_OOH_minus_ΔG_OH (eV)": dgooh_minus_dgoh,
            "ΔG_OOH_scaling_proxy (eV)": dg_ooh_scaling,
            "ΔG_OOH_minus_ΔG_OH_scaling_proxy (eV)": 3.20,
            **steps_oer_explicit,
            **steps_oer_scaling,
            "eta_OER_explicit (V)": float(eta_oer_explicit),
            "eta_OER_scaling_proxy (V)": float(eta_oer_scaling),
            "eta_OER (V)": float(eta_selected),
            "OER_PDS_explicit": pds_oer_explicit,
            "OER_PDS_scaling_proxy": pds_oer_scaling,
            "OER_PDS": pds_selected,
            "chosen_h_OH_A": height_map.get("OH", None),
            "chosen_h_O_A": height_map.get("O", None),
            "chosen_h_OOH_A": height_map.get("OOH", None),
            "OER_competition_penalty": float(penalty),
            "OER_risk_class": risk,
            "OER_penalty_note": note,
            "OER_competition_context": "penalty_for_anodic_organic_oxidation_screening",
            "qa_OOH": qa_map.get("OOH", ""),
            "qa_O": qa_map.get("O", ""),
            "qa_OH": qa_map.get("OH", ""),
            "state_OOH": state_map.get("OOH", ""),
            "state_O": state_map.get("O", ""),
            "state_OH": state_map.get("OH", ""),
            "ref_rxn_OH": ref_map.get("OH", ""),
            "ref_rxn_O": ref_map.get("O", ""),
            "ref_rxn_OOH": ref_map.get("OOH", ""),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty and "eta_OER (V)" in out.columns:
        _sort_cols = ["eta_OER (V)"]
        if "site_label" in out.columns:
            _sort_cols.append("site_label")
        elif "oer_base_site_label" in out.columns:
            _sort_cols.append("oer_base_site_label")
        out = out.sort_values(_sort_cols, kind="mergesort").reset_index(drop=True)
        # v16: make representative-site selection explicit in the summary table.
        # Rank is based on the already-selected/recommended eta value.
        out["OER_site_rank_by_recommended_eta"] = np.arange(1, len(out) + 1, dtype=int)
        out["OER_representative_site"] = False
        if len(out) > 0:
            out.loc[out.index[0], "OER_representative_site"] = True
        label_col = "oer_base_site_label" if "oer_base_site_label" in out.columns else ("site_label" if "site_label" in out.columns else None)
        if label_col is not None and len(out) > 0:
            out["OER_representative_site_label"] = str(out.loc[out.index[0], label_col])
    return out
