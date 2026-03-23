from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from ocp_app.core.gas_refs import get_h2_ref
from ocp_app.core.ads_sites import AdsSite, expand_oxide_channels_for_adsorbate, oxide_surface_seed_position
from ocp_app.core.anchors.common import (
    calc,
    MODEL_NAME,
    DEVICE,
    ZPE_CORR,
    TDS_CORR,
    NET_CORR,
    H0S,
    MIGRATE_THR,
    VAC_WARN_MIN,
    UNUSUAL_DDELTA,
    ensure_pbc3,
    layer_indices,
    relax_freeH,
    site_energy_two_stage,
)

# =====================================================================
# Unified CHE workflow (Metal & Oxide)
#  - HER: ΔG_H (H* adsorption)
#  - CO2RR: ΔG_ads for COOH*, CO*, HCOO*, OCHO* (reaction-descriptor based)
#  - ORR: stepwise free energies for OOH*, O*, OH* (4e⁻ Norskov CHE)
# =====================================================================

# --- HER CHE correction (calibrated on Ni(111)) ---
STANDARD_CHE_CORR = 0.24  # eV

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

# ── ORR settings ──────────────────────────────────────────────────────
# Uses the Norskov CHE approach without direct O2 energy calculation:
# E_O2_eff = 2·E_H2O - 2·E_H2 + 4×1.23 eV
ORR_EQUIL_POTENTIAL = 1.23  # V vs RHE (standard O2 reduction equilibrium potential)

# ORR ZPE + TΔS correction defaults (can be overridden via thermo_ORR.json)
DEFAULT_ORR_CORR: Dict[str, float] = {
    "OOH": 0.40,
    "O":   0.05,
    "OH":  0.35,
}

THERMO_ORR_NAME = "thermo_ORR.json"

# Number of electrons consumed per ORR intermediate (4e⁻ pathway)
ORR_N_ELECTRONS: Dict[str, int] = {"OOH": 1, "O": 2, "OH": 3}
# ─────────────────────────────────────────────────────────────────────


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
    previous_positions: List[np.ndarray],
    tol: float = 0.15,
) -> bool:
    """
    Check whether the final H position is essentially identical to any
    previously converged site (duplicate site detection).
    """
    for prev in previous_positions:
        dist = np.linalg.norm(current_pos - prev)
        if dist < tol:
            return True
    return False


# ---------------------------------------------------------------------
# Site coordinate helpers
# ---------------------------------------------------------------------
def site_xy_by_layers_metal(at) -> Dict[str, np.ndarray]:
    """
    Extract fcc / bridge / ontop xy coordinates from the top layer of a
    metal (111)-like slab.
    """
    layers = layer_indices(at, n=3)
    nL = len(layers)
    pos = at.get_positions()

    if nL < 2:
        z = pos[:, 2]
        top_idx = np.where(z > np.max(z) - 2.0)[0]
    else:
        top_idx = layers[0]

    top_xy = pos[top_idx][:, :2]

    ref = top_xy[0]
    d = np.linalg.norm(top_xy - ref, axis=1)
    order = np.argsort(d)
    if len(order) < 3:
        order = np.concatenate([order, np.tile(order[-1], 3 - len(order))])

    i0, i1, i2 = order[0], order[1], order[2]
    ontop_xy = top_xy[i0]
    bridge_xy = 0.5 * (top_xy[i0] + top_xy[i1])
    hollow_xy = (top_xy[i0] + top_xy[i1] + top_xy[i2]) / 3.0

    return {
        "fcc": hollow_xy,
        "hcp": hollow_xy,
        "bridge": bridge_xy,
        "ontop": ontop_xy,
    }


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
    slab_u_rel, E_slab_u = relax_freeH(slab_u_raw, steps=slab_steps, fmax=0.03)
    write(UROOT / "slab/user_slab_relaxed.cif", slab_u_rel)

    # Check energy difference before/after relaxation
    _, E_slab_u_raw = relax_freeH(slab_u_raw, steps=0, fmax=0.03)
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
) -> List[Tuple[str, str, np.ndarray]]:
    """
    Build the final list of (label, kind, xy) site coordinates.

    If user_ads_sites is provided, those are used directly; otherwise
    geometry-based defaults are generated from the relaxed slab.
    """
    target_sites: List[Tuple[str, str, np.ndarray]] = []

    if user_ads_sites:
        for label, site in user_ads_sites.items():
            kind = getattr(site, "kind", "unknown")
            pos = np.asarray(getattr(site, "position", []), dtype=float)
            if pos.shape[0] >= 2:
                target_sites.append((str(label), str(kind), pos[:2]))
    else:
        if mtype == "metal":
            xy_map = site_xy_by_layers_metal(slab_u_rel)
        elif mtype == "oxide":
            xy_map = site_xy_by_layers_oxide(slab_u_rel)
        else:
            raise ValueError(f"Unknown mtype='{mtype}'.")

        for s in sites:
            if s in xy_map:
                target_sites.append((s, s, np.asarray(xy_map[s], dtype=float)))

    return target_sites


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
    final_h_positions: List[np.ndarray] = []

    target_sites = _build_target_sites(mtype, slab_u_rel, sites, user_ads_sites)

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

    for label, kind, xy in target_sites:
        Au, E_uH, disp = site_energy_two_stage(
            slab_u_rel,
            np.asarray(xy, dtype=float),
            H0S,
            z_steps,
            free_steps,
        )

        dE_u = E_uH - E_slab_u - 0.5 * E_H2
        dG_u = dE_u + net_corr

        h_pos_final = Au.get_positions()[-1]
        is_duplicate = _check_duplicate_convergence(h_pos_final, final_h_positions)
        if not is_duplicate:
            final_h_positions.append(h_pos_final)

        row: Dict[str, object] = {
            "mode": "HER",
            "site": kind,
            "site_label": label,
            "MODEL": MODEL_NAME,
            "DEVICE": DEVICE,
            "layers": layers,
            "vac_z(Å)": vac_z,
            "E_slab_user (eV)": E_slab_u,
            "E_slab+H_user (eV)": E_uH,
            "ΔE_H_user (eV)": dE_u,
            "E_slab (eV)": E_slab_u,
            "ΔE_H (eV)": dE_u,
            "H_lateral_disp(Å)": float(disp),
            "migrated": bool(disp > MIGRATE_THR),
            "is_duplicate": is_duplicate,
            "slab_relax_drop": meta_flags["slab_relax_drop"],
            "vac_warning": meta_flags["vac_warning"],
        }

        if export_absolute:
            row["ΔG_H (eV)"] = dG_u

        rows.append(row)
        write(UROOT / f"sites/user_{label}_H.cif", Au)

    df = pd.DataFrame(rows)
    if export_absolute and "ΔG_H (eV)" in df.columns:
        df = df.assign(abs_val=lambda x: x["ΔG_H (eV)"].abs()).sort_values(
            ["abs_val"]
        )
    else:
        df = df.assign(
            abs_val=lambda x: x["ΔE_H_user (eV)"].abs()
        ).sort_values(["abs_val"])

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
        "thermo": {"NET_CORR": net_corr, "standard": f"{STANDARD_CHE_CORR:.2f} eV"},
        "E_H2": E_H2,
        "HER_GUARDRAIL": her_guard,
        "Model": MODEL_NAME,
        "Device": DEVICE,
        "warnings": {
            "slab_relax_drop": meta_flags["slab_relax_drop"],
            "vac_warning": meta_flags["vac_warning"],
        },
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


def _build_adsorbate_on_site(
    slab,
    xy: np.ndarray,
    ads: str,
    dz: float = 1.8,
    ref_dir: str | Path = "ref_gas",
    target_anchor_xyz: np.ndarray | None = None,
):
    """
    Place an adsorbate template on the slab at the given xy position,
    offset above the slab top by dz.  Returns the combined slab+ads
    Atoms object.
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


def _classify_metal_site_xy(slab_only, anchor_xy: np.ndarray) -> str:
    """Classify relaxed adsorption site on close-packed metal surfaces using xy geometry."""
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
                        d2 = float(
                            np.min(np.linalg.norm(sec_xy - centroid[None, :], axis=1))
                        )
                        if d2 < tol_atop:
                            return "hcp"
                except Exception:
                    pass
                return "fcc"

        return "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------
# Optional HER guardrail for CO2RR (single-site; cheap)
# ---------------------------------------------------------------------
def _pick_guardrail_site(
    target_sites: list[tuple[str, str, np.ndarray]],
    preference: str = "ontop",
) -> tuple[str, str, np.ndarray] | None:
    """Pick one (label, kind, xy) from target_sites, preferring `preference` if available."""
    if not target_sites:
        return None
    pref = (preference or "").strip().lower()

    for (label, kind, xy) in target_sites:
        if str(kind).lower() == pref or str(label).lower() == pref:
            return (label, kind, np.asarray(xy, dtype=float))

    alias = {
        "top": "ontop",
        "on-top": "ontop",
        "on_top": "ontop",
        "atop": "ontop",
    }
    pref2 = alias.get(pref, pref)
    for (label, kind, xy) in target_sites:
        if str(kind).lower() == pref2 or str(label).lower() == pref2:
            return (label, kind, np.asarray(xy, dtype=float))

    (label, kind, xy) = target_sites[0]
    return (label, kind, np.asarray(xy, dtype=float))


def _compute_her_guardrail_from_prepared(
    slab_u_rel,
    E_slab_u: float,
    E_H2: float,
    target_sites: list[tuple[str, str, np.ndarray]],
    z_steps: int,
    free_steps: int,
    site_preference: str = "ontop",
    use_net_corr: bool = True,
    out_cif: Path | None = None,
) -> dict[str, object] | None:
    """
    Compute a single H* adsorption as a guardrail using the *already prepared* slab and H2 ref.
    """
    pick = _pick_guardrail_site(target_sites, preference=site_preference)
    if pick is None:
        return None

    label, kind, xy = pick
    Au, E_uH, disp = site_energy_two_stage(
        slab_u_rel,
        np.asarray(xy, dtype=float),
        H0S,
        int(z_steps),
        int(free_steps),
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
        "E_slab_user (eV)": float(E_slab_u),
        "E_slab+H_user (eV)": float(E_uH),
        "ΔE_H_user (eV)": float(dE_u),
        "ΔG_H (eV)": float(dG_u),
        "H_lateral_disp(Å)": float(disp),
        "migrated": bool(float(disp) > float(MIGRATE_THR)),
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
    reaction_mode: str = "CO2RR",   # "CO2RR" | "ORR"
    orr_u: float = 0.0,
):
    """
    Reaction-descriptor based ΔE/ΔG screening for CO2RR or ORR.

    CO2RR adsorbates: COOH*, CO*, HCOO*, OCHO*
    ORR adsorbates:   OOH*, O*, OH*

    When reaction_mode="ORR", the ORR reference reaction scheme and
    thermodynamic corrections are used instead of the CO2RR defaults.
    """
    _is_orr = (str(reaction_mode).upper() == "ORR")
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

    # Load thermodynamic corrections (ORR vs CO2RR branch)
    if _is_orr:
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
    required_gas = ["H2O"] if _is_orr else ["CO2", "H2O"]
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
    _required_keys = ["H2", "H2O"] if _is_orr else ["CO2", "H2", "H2O"]
    for k in _required_keys:
        if k not in gas_E or not np.isfinite(gas_E[k]):
            _mode_label = "ORR" if _is_orr else "CO2RR"
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

    for label, kind, xy in target_sites:
        for ads in adspecies:
            ads_clean = ads.replace("*", "").upper()
            if ads_clean not in ADS_TEMPLATE_FILES:
                continue

            seed_variants = [{
                "site_label": str(label),
                "site_kind": str(kind),
                "xy": np.asarray(xy, float),
                "target_anchor_xyz": None,
                "oxide_seed_mode": None,
                "surface_channel": None,
            }]
            if _is_orr and mtype == "oxide":
                seed_variants = []
                site_obj = AdsSite(kind=str(kind), position=(float(xy[0]), float(xy[1]), float(np.max(slab_u_rel.get_positions()[:, 2]))), surface_indices=())
                channels = expand_oxide_channels_for_adsorbate(ads_clean)
                for ch in channels:
                    x0, y0, z0, surface_channel = oxide_surface_seed_position(
                        slab_u_rel, site_obj, ads_clean, channel=ch
                    )
                    site_label_seed = f"{label}:{ch}" if len(channels) > 1 else str(label)
                    seed_variants.append({
                        "site_label": site_label_seed,
                        "site_kind": str(kind),
                        "xy": np.asarray([x0, y0], float),
                        "target_anchor_xyz": np.asarray([x0, y0, z0], float),
                        "oxide_seed_mode": str(ch),
                        "surface_channel": str(surface_channel),
                    })

            for seed in seed_variants:
                slab_ads = _build_adsorbate_on_site(
                    slab_u_rel,
                    np.asarray(seed["xy"], float),
                    ads_clean,
                    dz=1.8,
                    ref_dir="ref_gas",
                    target_anchor_xyz=seed["target_anchor_xyz"],
                )

                slab_ads_rel, E_slab_ads, opt_meta_ads = _relax_slab_ads(
                    slab_ads,
                    n_slab_atoms=meta_flags["n_atoms"],
                    steps=free_steps,
                    fmax=0.05,
                    relax_ads=True,
                )

                write(UROOT / f"sites/user_{str(seed['site_label']).replace(':', '__')}_{ads_clean}.cif", slab_ads_rel)

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
                disp = float(np.linalg.norm(anchor_xy - np.asarray(seed["xy"], float)))
    
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
    
                # Migration threshold: COOH/HCOO/OCHO have larger metric
                # variation due to rotation/rearrangement, so threshold is relaxed
                migrate_thr = (
                    float(MIGRATE_THR)
                    if ads_clean == "CO"
                    else float(max(MIGRATE_THR, 2.5))
                )
                migrated = bool(disp > migrate_thr)
    
                qa = "ok"
                if crashed:
                    qa = "crashed"
                elif desorbed:
                    qa = "desorbed"
                elif broken:
                    qa = "broken"
                elif migrated:
                    qa = "migrated"
    
                # relaxed site re-classification (metal only; best-effort)
                relaxed_site = None
                if mtype == "metal":
                    try:
                        slab_only = slab_ads_rel[:n_slab]
                        relaxed_site = _classify_metal_site_xy(slab_only, anchor_xy)
                    except Exception:
                        relaxed_site = None
    
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
                # ── ORR intermediates ────────────────────────────────────────
                elif ads_clean == "OOH":
                    # O2 + (H⁺+e⁻) → OOH*
                    _E_O2_eff = 2 * E_H2O - 2 * E_H2 + 4 * ORR_EQUIL_POTENTIAL
                    E_reagents = _E_O2_eff + 0.5 * E_H2
                    ref_rxn = "O2_eff + 1/2 H2"
                elif ads_clean == "O":
                    # O2 + 2(H⁺+e⁻) → O* + H2O
                    _E_O2_eff = 2 * E_H2O - 2 * E_H2 + 4 * ORR_EQUIL_POTENTIAL
                    E_reagents = _E_O2_eff + E_H2 - E_H2O
                    ref_rxn = "O2_eff + H2 - H2O"
                elif ads_clean == "OH":
                    # O2 + 3(H⁺+e⁻) → OH* + H2O
                    _E_O2_eff = 2 * E_H2O - 2 * E_H2 + 4 * ORR_EQUIL_POTENTIAL
                    E_reagents = _E_O2_eff + 1.5 * E_H2 - E_H2O
                    ref_rxn = "O2_eff + 3/2 H2 - H2O"
                # ─────────────────────────────────────────────────────────────
                else:
                    E_reagents = 0.0
                    ref_rxn = "custom"
    
                dE_raw = float(E_slab_ads - E_slab_u)
                dE_ads = float(E_slab_ads - E_slab_u - E_reagents)
                g_corr = float(ads_corr.get(ads_clean, 0.0))
                dG_ads = float(dE_ads + g_corr)
    
                row: Dict[str, object] = {
                    "mode": "ORR" if _is_orr else "CO2RR",
                    "adsorbate": ads_clean,
                    "site": seed["site_kind"],
                    "site_label": seed["site_label"],
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
                    "ΔE_ads_user (eV)": float(dE_ads),
                    "ΔE_raw(slab+ads - slab) (eV)": float(dE_raw),
                    "E_ref_reagents (eV)": float(E_reagents),
                    "G_correction (eV)": float(g_corr),
                    "ref_rxn": ref_rxn,
                    "ΔG_ads (eV)": float(dG_ads) if export_absolute else None,
                    "ads_lateral_disp(Å)": float(disp),
                    "oxide_seed_mode": seed.get("oxide_seed_mode"),
                    "surface_channel": seed.get("surface_channel"),
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
                    "slab_relax_drop": meta_flags["slab_relax_drop"],
                    "vac_warning": meta_flags["vac_warning"],
                }
                rows.append(row)

    # Write metadata even when no rows are produced
    if not rows:
        _out_name = "results_sites_orr.csv" if _is_orr else "results_sites_co2rr.csv"
        out_csv = out_root / _out_name
        pd.DataFrame([]).to_csv(out_csv, index=False)
        _mode_str = "ORR" if _is_orr else "CO2RR"
        meta = {
            "mode": _mode_str,
            "user_slab": str(Path(user_slab_cif).resolve()),
            "relax_mode": relax_mode,
            "steps": {"slab": slab_steps, "ads": free_steps},
            "adspecies": list(adspecies),
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
        _meta_name_empty = "meta_orr.json" if _is_orr else "meta_co2rr.json"
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

    _out_name2 = "results_sites_orr.csv" if _is_orr else "results_sites_co2rr.csv"
    out_csv = out_root / _out_name2
    df.drop(columns=["abs_val"], errors="ignore").to_csv(
        out_csv,
        index=False,
        float_format="%.6f",
    )

    orr_summary_csv = None
    if _is_orr:
        try:
            _orr_summary = _summarize_orr_descriptor_table(df.drop(columns=["abs_val"], errors="ignore"), U=float(orr_u))
            if _orr_summary is not None and (not _orr_summary.empty):
                orr_summary_csv = out_root / "results_orr_summary.csv"
                _orr_summary.to_csv(orr_summary_csv, index=False, float_format="%.6f")
        except Exception:
            orr_summary_csv = None

    _mode_str2 = "ORR" if _is_orr else "CO2RR"
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

    _meta_name = "meta_orr.json" if _is_orr else "meta_co2rr.json"
    (out_root / _meta_name).write_text(json.dumps(meta, indent=2))

    return str(out_csv), meta


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def run_metal_che(
    user_slab_cif: str,
    out_root: str | Path = "calc/metal_che",
    sites: Iterable[str] = ("ontop", "bridge", "fcc", "hcp"),
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
    )


def run_oxide_che(
    user_slab_cif: str,
    out_root: str | Path = "calc/oxide_che",
    sites: Iterable[str] = ("fcc", "bridge", "ontop"),
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
    )


def run_metal_co2rr_che(
    user_slab_cif: str,
    out_root: str | Path = "calc/metal_co2rr",
    sites: Iterable[str] = ("ontop", "bridge", "fcc"),
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
    sites: Iterable[str] = ("fcc", "bridge", "ontop"),
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
# ORR Public API
# run_metal_orr_che / run_oxide_orr_che
# Internally calls _run_co2rr_che with reaction_mode="ORR"
# =====================================================================

def run_metal_orr_che(
    user_slab_cif: str,
    out_root: str | Path = "calc/metal_orr",
    sites: Iterable[str] = ("ontop", "bridge", "fcc"),
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
):
    """
    ORR screening on metal surfaces.
    Intermediates: OOH*, O*, OH* (Norskov 4e⁻ CHE)
    Reference potential: 1.23 V vs RHE (E_O2 = 2·E_H2O - 2·E_H2 + 4×1.23 eV)
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
        reaction_mode="ORR",
        orr_u=orr_u,
    )


def run_oxide_orr_che(
    user_slab_cif: str,
    out_root: str | Path = "calc/oxide_orr",
    sites: Iterable[str] = ("fcc", "bridge", "ontop"),
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
):
    """
    ORR screening on oxide surfaces.
    Intermediates: OOH*, O*, OH* (Norskov 4e⁻ CHE)
    Reference potential: 1.23 V vs RHE (E_O2 = 2·E_H2O - 2·E_H2 + 4×1.23 eV)
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
        reaction_mode="ORR",
        orr_u=orr_u,
    )

def _summarize_orr_descriptor_table(df: pd.DataFrame, U: float = 0.0) -> pd.DataFrame:
    """
    Convert descriptor table (ΔG_OOH, ΔG_O, ΔG_OH) into ORR/OER step thermodynamics.
    One-electron CHE shift is applied to ORR steps as ΔG(U)=ΔG(0)-U.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    need = {"adsorbate", "ΔG_ads (eV)", "site_label"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    group_cols = [c for c in ["site_label", "site", "oxide_seed_mode", "surface_channel"] if c in df.columns]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        vals = {}
        qa_map = {}
        for _, r in sub.iterrows():
            a = str(r.get("adsorbate", "")).upper()
            g = pd.to_numeric(r.get("ΔG_ads (eV)"), errors="coerce")
            if pd.notna(g):
                vals[a] = float(g)
                qa_map[a] = str(r.get("qa", ""))
        if not {"OOH", "O", "OH"}.issubset(vals.keys()):
            continue

        d1_orr_0 = 4.92 - vals["OOH"]
        d2_orr_0 = vals["OOH"] - vals["O"]
        d3_orr_0 = vals["O"] - vals["OH"]
        d4_orr_0 = vals["OH"]
        d1_orr_u = d1_orr_0 - float(U)
        d2_orr_u = d2_orr_0 - float(U)
        d3_orr_u = d3_orr_0 - float(U)
        d4_orr_u = d4_orr_0 - float(U)
        steps_orr_0 = {
            "ORR_step1_O2_to_OOH": d1_orr_0,
            "ORR_step2_OOH_to_O": d2_orr_0,
            "ORR_step3_O_to_OH": d3_orr_0,
            "ORR_step4_OH_to_H2O": d4_orr_0,
        }
        steps_orr_u = {
            "ORR_step1_O2_to_OOH@U": d1_orr_u,
            "ORR_step2_OOH_to_O@U": d2_orr_u,
            "ORR_step3_O_to_OH@U": d3_orr_u,
            "ORR_step4_OH_to_H2O@U": d4_orr_u,
        }
        U_lim = min(steps_orr_0.values())
        pds = min(steps_orr_0, key=steps_orr_0.get)
        oer1 = vals["OH"]
        oer2 = vals["O"] - vals["OH"]
        oer3 = vals["OOH"] - vals["O"]
        oer4 = 4.92 - vals["OOH"]
        steps_oer_0 = {
            "OER_step1_H2O_to_OH": oer1,
            "OER_step2_OH_to_O": oer2,
            "OER_step3_O_to_OOH": oer3,
            "OER_step4_OOH_to_O2": oer4,
        }
        pds_oer = max(steps_oer_0, key=steps_oer_0.get)
        row = {
            **key_map,
            "ΔG_OOH (eV)": vals["OOH"],
            "ΔG_O (eV)": vals["O"],
            "ΔG_OH (eV)": vals["OH"],
            **steps_orr_0,
            **steps_orr_u,
            **steps_oer_0,
            "ORR_U_input (V)": float(U),
            "U_lim_ORR (V)": float(U_lim),
            "eta_ORR (V)": float(1.23 - U_lim),
            "ORR_PDS": pds,
            "max_ΔG_ORR@U (eV)": float(max(steps_orr_u.values())),
            "eta_OER (V)": float(max(steps_oer_0.values()) - 1.23),
            "OER_PDS": pds_oer,
            "qa_OOH": qa_map.get("OOH", ""),
            "qa_O": qa_map.get("O", ""),
            "qa_OH": qa_map.get("OH", ""),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "eta_ORR (V)" in out.columns:
        out = out.sort_values(["eta_ORR (V)", "site_label"], kind="mergesort").reset_index(drop=True)
    return out

