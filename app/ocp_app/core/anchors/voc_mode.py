from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from ase import Atoms, Atom
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms, FixCartesian
from ase.geometry import find_mic

from ocp_app.core.ads_sites import (
    AdsSite,
    detect_metal_111_sites,
    detect_oxide_surface_sites,
    select_representative_sites,
    ANION_SYMBOLS,
)
from ocp_app.core.anchors.common import (
    calc,
    MODEL_NAME,
    DEVICE,
    ensure_pbc3,
    build_relax_constraints,
    site_energy_two_stage,
    H0S,
)
from ocp_app.core.anchors.CHE_mode import (
    _prepare_slab,
    _choose_steps,
    _get_gas_box_energies,
    _safe_float,
    STANDARD_CHE_CORR,
)
from ocp_app.core.voc_registry import (
    VOC_PRESETS,
    VOC_ADSORBATES,
    VOC_TEMPLATE_FILES,
    clean_adsorbate_label,
    get_voc_preset,
    normalize_voc_state,
    state_components,
    is_voc_proximity_state,
    allowed_oxide_site_classes_for_state,
    get_oxide_voc_site_policy,
)

VOC_PROXY_WARNING = (
    "SAGE-VOC reports UMA/OCP ΔE_proxy and co-adsorption proximity proxies. "
    "These are pre-screening descriptors, not definitive electrochemical ΔG values."
)


VOC_DESCRIPTOR_VALID_QA = {
    "ok_single_point_proxy",
    "ok_short_relax_proxy",
    "ok_normal_relax_proxy",
    "ok_local_flex_proxy",
    "ok_rigid_proxy",
    "ok_frozen_pose_proxy",
    "ok_axis_locked_proxy",
    "surface_distorted_but_bound",
    "ok_metal_che_her_like",
    "ech_diagnostic_valid",
}

def _is_valid_descriptor_qa(qa_value: object) -> bool:
    return str(qa_value or "").strip().lower() in VOC_DESCRIPTOR_VALID_QA


def _invalid_descriptor_note(qa_value: object) -> str:
    q = str(qa_value or "").strip().lower()
    if not q:
        return "qa_missing"
    return f"descriptor_energy_masked_due_to_qa:{q}"


def _masked_energy_payload(
    *,
    comps: Tuple[str, ...],
    target_key: str,
    qa_value: object,
    dE_proxy_raw: float,
    dE_prox_raw: float,
) -> Dict[str, object]:
    """Return user-facing descriptor columns and raw diagnostic columns.

    Rejected VOC rows may still have finite potential energies because UMA/OCP
    evaluated the final coordinates.  Those values are useful for debugging, but
    they are not valid VOC adsorption/proximity descriptors after fragmentation,
    desorption, separation, or diagnostic-only relaxation.  Therefore, rejected
    rows keep raw diagnostic energies while all descriptor columns are masked
    as NaN.
    """
    is_valid = _is_valid_descriptor_qa(qa_value)

    raw_proxy = float(dE_proxy_raw) if np.isfinite(float(dE_proxy_raw)) else float("nan")
    raw_prox = float(dE_prox_raw) if np.isfinite(float(dE_prox_raw)) else float("nan")

    if is_valid:
        proxy = raw_proxy
        prox = raw_prox
        note = ""
    else:
        proxy = float("nan")
        prox = float("nan")
        note = _invalid_descriptor_note(qa_value)

    is_target_single = len(comps) == 1 and comps[0] == target_key
    is_h_voc = len(comps) > 1 and set(comps) == {"H", target_key}
    is_oh_voc = len(comps) > 1 and set(comps) == {"OH", target_key}

    return {
        "descriptor_energy_valid": bool(is_valid),
        "descriptor_energy_mask_note": note,

        # User-facing descriptor columns: masked unless QA-valid.
        "ΔE_proxy (eV)": proxy,
        "ΔE_ads_user (eV)": proxy,
        "ΔE_VOC_ads_proxy (eV)": proxy if is_target_single else float("nan"),
        "ΔE_H_VOC_proximity_proxy (eV)": prox if is_h_voc else float("nan"),
        "ΔE_OH_VOC_proximity_proxy (eV)": prox if is_oh_voc else float("nan"),
        "ΔE_proximity_proxy (eV)": prox if len(comps) > 1 else float("nan"),

        # Raw diagnostic columns: retained only when the row is rejected.
        "ΔE_raw_proxy_diagnostic (eV)": float("nan") if is_valid else raw_proxy,
        "ΔE_raw_ads_user_diagnostic (eV)": float("nan") if is_valid else raw_proxy,
        "ΔE_raw_VOC_ads_proxy_diagnostic (eV)": raw_proxy if (not is_valid and is_target_single) else float("nan"),
        "ΔE_raw_H_VOC_proximity_diagnostic (eV)": raw_prox if (not is_valid and is_h_voc) else float("nan"),
        "ΔE_raw_OH_VOC_proximity_diagnostic (eV)": raw_prox if (not is_valid and is_oh_voc) else float("nan"),
        "ΔE_raw_proximity_diagnostic (eV)": raw_prox if (not is_valid and len(comps) > 1) else float("nan"),
    }


def _row_can_supply_voc_energy(row: object) -> bool:
    """Return whether a prior single-state row can be used in a proximity formula."""
    if not isinstance(row, dict):
        return False
    if not _is_valid_descriptor_qa(row.get("qa")):
        return False
    if not bool(row.get("descriptor_energy_valid", False)):
        return False
    return np.isfinite(_safe_float(row.get("E_state_user (eV)")))


def _candidate_ref_gas_dirs(ref_dir: str | Path = "ref_gas") -> list[Path]:
    """Return candidate ref_gas directories without accepting directory-only hits.

    This intentionally differs from a simple directory resolver.  In the current
    SAGE deployment, cwd/ref_gas may contain legacy CO2RR/OER templates but not
    the newly added VOC templates, while /app/ref_gas may contain the VOC CIFs.
    Therefore the VOC loader must search for the requested template file itself.
    """
    ref_path = Path(ref_dir)
    here = Path(__file__).resolve()
    raw: list[Path] = []

    if ref_path.is_absolute():
        raw.append(ref_path)
    else:
        raw.extend([
            Path.cwd() / ref_path,
            Path('/app') / ref_path,
            Path('/app/ref_gas'),
            here.parent / ref_path,
            here.parents[1] / ref_path,
            here.parents[2] / ref_path,
            here.parents[3] / ref_path,
            here.parents[4] / ref_path,
            here.parents[5] / ref_path,
            ref_path,
        ])

    out: list[Path] = []
    seen: set[str] = set()
    for cand in raw:
        try:
            c = cand.resolve()
        except Exception:
            c = cand
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def resolve_ref_gas_template(template_name: str, ref_dir: str | Path = "ref_gas") -> Path:
    """Resolve a ref_gas template by checking file existence, not just directory existence.

    CHE_mode can use Path('ref_gas') because the legacy templates are usually in
    the current working directory.  VOC templates are newly added, so an old
    cwd/ref_gas may exist without CH3CHO_box.cif.  This resolver searches all
    plausible ref_gas roots and returns the first one that actually contains the
    requested template.
    """
    checked: list[str] = []
    t = str(template_name)

    ref_path = Path(ref_dir)
    if ref_path.is_file() and ref_path.name == t:
        return ref_path.resolve()

    for d in _candidate_ref_gas_dirs(ref_dir):
        p = d / t
        checked.append(str(p))
        try:
            if p.is_file():
                return p.resolve()
        except Exception:
            pass

    raise FileNotFoundError(
        "VOC adsorbate template not found. Checked:\n  - " + "\n  - ".join(checked)
    )


# -----------------------------------------------------------------------------
# Basic geometry helpers
# -----------------------------------------------------------------------------

def _safe_label(s: str) -> str:
    out = str(s or "").replace("*", "star").replace("+", "__plus__")
    out = out.replace("/", "_").replace(" ", "_").replace(":", "__")
    return out


def _top_z(atoms: Atoms) -> float:
    if atoms is None or len(atoms) == 0:
        return 0.0
    return float(np.max(atoms.get_positions()[:, 2]))


def _nearest_surface_distance(slab: Atoms, ads_coords: np.ndarray) -> float:
    if slab is None or len(slab) == 0 or ads_coords is None or ads_coords.size == 0:
        return float("nan")
    sp = np.asarray(slab.get_positions(), dtype=float)
    d = np.linalg.norm(ads_coords[:, None, :] - sp[None, :, :], axis=2)
    return float(np.min(d))


def _mic_xy_distance(cell, pbc, xy0, xy1) -> float:
    a = np.asarray([float(xy1[0]) - float(xy0[0]), float(xy1[1]) - float(xy0[1]), 0.0], dtype=float)
    try:
        vec, dist = find_mic(a, cell, pbc)
        return float(np.linalg.norm(vec[:2]))
    except Exception:
        return float(np.linalg.norm(a[:2]))


def _offset_xy_for_coadsorption(slab: Atoms, site: AdsSite, offset_A: float = 1.25) -> np.ndarray:
    """Return a deterministic lateral offset from the primary site.

    The offset is along the cell a-vector direction in XY when available. This is
    not meant to define a reaction barrier; it only seeds a local co-adsorption
    geometry for proximity screening.
    """
    xy = np.asarray(site.position[:2], dtype=float)
    try:
        a = np.asarray(slab.get_cell()[0, :2], dtype=float)
        n = float(np.linalg.norm(a))
        if np.isfinite(n) and n > 1e-8:
            return xy + (a / n) * float(offset_A)
    except Exception:
        pass
    return xy + np.asarray([float(offset_A), 0.0], dtype=float)


def _outer_h_xy_for_ech(slab: Atoms, site: AdsSite, offset_A: float = 2.10) -> np.ndarray:
    """Return an outer-side H_ads xy target for H*+CH3CHO*.

    CH3CHO* stays centered on the selected site using the ordinary reduction
    precursor pose. The extra H* is moved only to the outer side of the site,
    i.e. farther from the slab xy center.
    """
    site_xy = np.asarray(site.position[:2], dtype=float)
    try:
        pos = np.asarray(slab.get_positions(), dtype=float)
        center_xy = np.mean(pos[:, :2], axis=0)
        vec = site_xy - center_xy
        n = float(np.linalg.norm(vec))
        if not np.isfinite(n) or n < 1e-8:
            raise ValueError
        u = vec / n
    except Exception:
        try:
            avec = np.asarray(slab.get_cell()[0, :2], dtype=float)
            n = float(np.linalg.norm(avec))
            if not np.isfinite(n) or n < 1e-8:
                raise ValueError
            u = avec / n
        except Exception:
            u = np.asarray([1.0, 0.0], dtype=float)
    return site_xy + float(offset_A) * u


def _ech_outward_unit_xy(slab: Atoms, site: AdsSite) -> np.ndarray:
    """Unit vector from slab xy center toward the selected site, with cell-a fallback."""
    site_xy = np.asarray(site.position[:2], dtype=float)
    try:
        pos = np.asarray(slab.get_positions(), dtype=float)
        center_xy = np.mean(pos[:, :2], axis=0)
        vec = site_xy - center_xy
        n = float(np.linalg.norm(vec))
        if np.isfinite(n) and n > 1e-8:
            return vec / n
    except Exception:
        pass
    try:
        avec = np.asarray(slab.get_cell()[0, :2], dtype=float)
        n = float(np.linalg.norm(avec))
        if np.isfinite(n) and n > 1e-8:
            return avec / n
    except Exception:
        pass
    return np.asarray([1.0, 0.0], dtype=float)


def _near_carbonyl_h_xy_for_ech(slab: Atoms, site: AdsSite, offset_A: float = 1.55) -> np.ndarray:
    """Return a near-carbonyl H_ads xy seed for ECH sensitivity testing.

    The CH3CHO* moiety is still placed using the same standalone pose.  Only
    H_ads is moved closer to the carbonyl-anchor region.  The offset is chosen
    to remain a reactant-proximity seed, not an initial C-H product bond.
    """
    site_xy = np.asarray(site.position[:2], dtype=float)
    return site_xy + float(offset_A) * _ech_outward_unit_xy(slab, site)


def _ech_seed_policies_for_state(state: str) -> list[str]:
    """Return seed policies for ECH/co-adsorption states.

    ECH is disabled in the stable VOC branch.  Ordinary direct-reduction and
    oxidation states are evaluated once with the default policy; H*+CH3CHO* and
    H*+H* are skipped if they survive in an old session-state selection.
    """
    norm = normalize_voc_state(state)
    if norm in {"H*+CH3CHO*", "H*+H*", "OH*+CH3CHO*"}:
        return []
    return ["default"]

def _ech_seed_role_for_policy(policy: str) -> str:
    p = str(policy or "default")
    if p == "outer_H":
        return "coadsorption_retention_test"
    if p == "near_carbonyl_H":
        return "h_transfer_sensitivity_test"
    if p == "adjacent_H_pair":
        return "her_h_h_competition_test"
    if p == "metal_HER_like":
        return "metal_CHE_HER_like_H_reference"
    if p == "metal_HER_neighbor":
        return "metal_HER_like_neighbor_H_coadsorption_test"
    if p == "metal_outer_fallback":
        return "metal_outer_fallback_H_coadsorption_test"
    return "default"


def _is_metal_voc_context(material_type: str | None) -> bool:
    return str(material_type or "").strip().lower() == "metal"


def _canonical_metal_site_kind(kind: object) -> str:
    k = str(kind or "").strip().lower()
    if k in {"fcc", "hcp"}:
        return "hollow"
    return k or "unknown"


def _select_neighbor_metal_h_site_for_ech(
    slab: Atoms,
    primary_site: AdsSite,
    *,
    min_xy_A: float = 1.05,
    max_xy_A: float = 3.60,
) -> tuple[AdsSite, str]:
    """Pick a real neighboring metal adsorption site for ECH H*.

    For metallic ECH, the co-adsorbed H* should be an HER-like adsorbed H
    on a detected metal surface site, not an arbitrary xy offset and not an
    oxide surface-OH fallback.  Prefer nearby hollow/bridge sites that do not
    overlap the primary VOC site.
    """
    try:
        candidates = detect_metal_111_sites(slab, max_sites_per_kind=200)
    except TypeError:
        candidates = detect_metal_111_sites(slab)
    except Exception:
        candidates = []

    site_xy = np.asarray(primary_site.position[:2], dtype=float)
    scored: list[tuple[tuple[float, float, int], AdsSite]] = []
    pref = {"hollow": 0, "bridge": 1, "ontop": 2, "fcc": 0, "hcp": 0}
    for i, cand in enumerate(candidates or []):
        try:
            cxy = np.asarray(cand.position[:2], dtype=float)
            dxy = _mic_xy_dist_for_slab(slab, site_xy, cxy)
            if dxy < float(min_xy_A) or dxy > float(max_xy_A):
                continue
            kind = _canonical_metal_site_kind(getattr(cand, "kind", "unknown"))
            # Prefer a real neighboring HER-like basin; then prefer moderate proximity.
            key = (float(pref.get(kind, 5)), abs(float(dxy) - 1.80), int(i))
            scored.append((key, cand))
        except Exception:
            continue

    if scored:
        scored.sort(key=lambda x: x[0])
        return scored[0][1], "metal_HER_neighbor_site"

    # Fallback: keep the previous conservative outer-H policy, but label it clearly.
    oxy = _outer_h_xy_for_ech(slab, primary_site, offset_A=2.10)
    fallback_site = AdsSite(
        kind="metal_outer_fallback",
        position=(float(oxy[0]), float(oxy[1]), float(_local_surface_z(slab, primary_site))),
        surface_indices=tuple(getattr(primary_site, "surface_indices", ()) or ()),
    )
    return fallback_site, "metal_outer_xy_fallback"


def _add_metal_her_like_h_seed(
    current_atoms: Atoms,
    base_slab: Atoms,
    h_site: AdsSite,
    *,
    height: float = 1.00,
    placement_note: str = "metal_HER_like_H_ads_site",
    ech_seed_policy: str = "metal_HER_like",
) -> Tuple[Atoms, dict]:
    """Append H* on a metallic HER-like adsorption site.

    This is only a seed builder.  Standalone metallic H* energies are evaluated
    in the run loop with CHE_mode.site_energy_two_stage so that ΔG_H_CHE matches
    the HER benchmark workflow.
    """
    slab0 = ensure_pbc3(base_slab)
    out = ensure_pbc3(current_atoms.copy())
    surface_z = _local_surface_z(slab0, h_site)
    support_z = _site_support_z(slab0, h_site, fallback_surface_z=surface_z)
    target_xyz = _site_target_xyz(
        slab0,
        h_site,
        "H",
        xy_override=np.asarray(h_site.position[:2], dtype=float),
        height=float(height),
        surface_z=surface_z,
    )
    out.append(Atom("H", tuple(float(x) for x in target_xyz)))
    idx = len(out) - 1
    support_meta = _site_support_metadata(slab0, h_site)
    return ensure_pbc3(out), {
        "ads_key": "H",
        "ads_start": int(idx),
        "ads_stop": int(idx + 1),
        "anchor_indices_local": (0,),
        "anchor_indices_global": (int(idx),),
        "anchor_xy_initial": (float(target_xyz[0]), float(target_xyz[1])),
        "anchor_xyz_initial": tuple(float(x) for x in target_xyz),
        "anchor_target_xyz": tuple(float(x) for x in target_xyz),
        "support_z_initial": float(support_z),
        "anchor_height_A": float(target_xyz[2] - support_z),
        "anchor_mode": "metal_HER_like_H_ads",
        "reduction_h_placement": str(placement_note),
        "H_descriptor_source": "metal_CHE_HER_like_seed",
        "H_placement_policy": str(placement_note),
        "ech_seed_policy": str(ech_seed_policy),
        "ech_seed_role": _ech_seed_role_for_policy(ech_seed_policy),
        **support_meta,
    }


# -----------------------------------------------------------------------------
# Oxide VOC site classification / routing
# -----------------------------------------------------------------------------

def _is_oxide_cation_symbol(sym: str) -> bool:
    return str(sym) not in ANION_SYMBOLS


def _surface_indices_by_z_window(atoms: Atoms, *, z_window: float = 4.0) -> list[int]:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    if pos.size == 0:
        return []
    zmax = float(np.max(pos[:, 2]))
    return [int(i) for i, p in enumerate(pos) if (zmax - float(p[2])) <= float(z_window)]


def _mic_xy_delta_vec_for_slab(atoms: Atoms, xy0, xy1) -> np.ndarray:
    v = np.asarray([float(xy1[0]) - float(xy0[0]), float(xy1[1]) - float(xy0[1]), 0.0], dtype=float)
    try:
        vec, _dist = find_mic(v, atoms.get_cell(), atoms.get_pbc())
        return np.asarray(vec[:2], dtype=float)
    except Exception:
        return v[:2]


def _mic_xy_dist_for_slab(atoms: Atoms, xy0, xy1) -> float:
    return float(np.linalg.norm(_mic_xy_delta_vec_for_slab(atoms, xy0, xy1)))


def _local_surface_z(slab: Atoms, site: AdsSite, *, radius: float = 2.8, top_z_window: float = 4.0) -> float:
    """Local z reference for corrugated oxide facets.

    Using global zmax on spinel/oxide facets can place VOCs above a remote top-O
    atom and make the molecule float over the actual target site.  Prefer local
    atoms around the selected site or site-defining atom indices.
    """
    try:
        pos = np.asarray(slab.get_positions(), dtype=float)
        if pos.size == 0:
            return _top_z(slab)
        site_xy = np.asarray(site.position[:2], dtype=float)
        zmax = float(np.max(pos[:, 2]))
        near = []
        for p in pos:
            if (zmax - float(p[2])) > float(top_z_window):
                continue
            if _mic_xy_dist_for_slab(slab, site_xy, p[:2]) <= float(radius):
                near.append(float(p[2]))
        if near:
            return float(max(near))
        idx = tuple(int(i) for i in (getattr(site, 'surface_indices', ()) or ()))
        if idx:
            zs = [float(pos[i, 2]) for i in idx if 0 <= i < len(pos)]
            if zs:
                return float(max(zs))
    except Exception:
        pass
    return _top_z(slab)


def _make_oxide_voc_site_record(
    slab: Atoms,
    *,
    site_class: str,
    label: str,
    indices: tuple[int, ...],
    position: tuple[float, float, float],
    rank: float = 0.0,
    warning: str = "",
) -> Dict[str, object]:
    site = AdsSite(kind=str(site_class), position=tuple(float(x) for x in position), surface_indices=tuple(int(i) for i in indices))
    syms = slab.get_chemical_symbols()
    elems = [str(syms[int(i)]) for i in indices if 0 <= int(i) < len(syms)]
    return {
        "site_label": str(label),
        "site_kind": str(site_class),
        "xy": np.asarray(position[:2], dtype=float),
        "initial_xyz": np.asarray(position[:3], dtype=float),
        "surface_indices": tuple(int(i) for i in indices),
        "site": site,
        "seed_source": "oxide_voc_routed_site",
        "oxide_site_class": str(site_class),
        "oxide_site_label": str(label),
        "oxide_site_elements": "-".join(elems),
        "oxide_site_rank": float(rank),
        "site_taxonomy_warning": str(warning),
    }


def _pick_representative_by_label(records: list[Dict[str, object]], max_per_label: int) -> list[Dict[str, object]]:
    buckets: dict[str, list[Dict[str, object]]] = {}
    for r in records:
        buckets.setdefault(str(r.get("oxide_site_label", r.get("site_label", "site"))), []).append(r)
    out: list[Dict[str, object]] = []
    for _label, items in sorted(buckets.items()):
        items2 = sorted(items, key=lambda x: float(x.get("oxide_site_rank", 0.0)))
        out.extend(items2[:max(1, int(max_per_label))])
    return out


def _generate_routed_oxide_voc_sites(slab: Atoms, *, policy: str = "standard_routed") -> list[Dict[str, object]]:
    pol = get_oxide_voc_site_policy(policy)
    pos = np.asarray(slab.get_positions(), dtype=float)
    syms = slab.get_chemical_symbols()
    if pos.size == 0:
        return []
    zmax = float(np.max(pos[:, 2]))
    zwin = float(pol.get("surface_z_window_A", 4.0))
    top_idx = _surface_indices_by_z_window(slab, z_window=zwin)
    # If the facet is O-terminated and cations sit slightly deeper, expand once.
    cation_idx = [i for i in top_idx if _is_oxide_cation_symbol(syms[i])]
    if not cation_idx:
        top_idx = _surface_indices_by_z_window(slab, z_window=max(6.0, zwin + 2.0))
        cation_idx = [i for i in top_idx if _is_oxide_cation_symbol(syms[i])]
    anion_idx = [i for i in top_idx if not _is_oxide_cation_symbol(syms[i])]

    raw: list[Dict[str, object]] = []
    for i in cation_idx:
        el = str(syms[i])
        p = tuple(float(x) for x in pos[i])
        # Highest atoms within an element are preferred; deeper cations are still
        # allowed when the surface is O-terminated, but they are ranked later.
        raw.append(_make_oxide_voc_site_record(
            slab,
            site_class="cation_top",
            label=f"{el}_top",
            indices=(i,),
            position=p,
            rank=float(zmax - pos[i, 2]),
        ))

    for i in anion_idx:
        el = str(syms[i])
        p = tuple(float(x) for x in pos[i])
        raw.append(_make_oxide_voc_site_record(
            slab,
            site_class="anion_top",
            label=f"{el}_top",
            indices=(i,),
            position=p,
            rank=float(zmax - pos[i, 2]),
        ))

    # Cation-cation bridges.  Allow moderately corrugated pairs, because spinel
    # facets are not flat metal(111) surfaces.
    for a_i, i in enumerate(cation_idx):
        for j in cation_idx[a_i + 1:]:
            dxy = _mic_xy_dist_for_slab(slab, pos[i, :2], pos[j, :2])
            dz = abs(float(pos[i, 2]) - float(pos[j, 2]))
            if dxy > 4.5 or dz > 3.0:
                continue
            vec = _mic_xy_delta_vec_for_slab(slab, pos[i, :2], pos[j, :2])
            xy = pos[i, :2] + 0.5 * vec
            z = max(float(pos[i, 2]), float(pos[j, 2]))
            pair = sorted([str(syms[i]), str(syms[j])])
            label = f"{pair[0]}-{pair[1]}_bridge"
            raw.append(_make_oxide_voc_site_record(
                slab,
                site_class="cation_cation_bridge",
                label=label,
                indices=(i, j),
                position=(float(xy[0]), float(xy[1]), z),
                rank=float(dxy + 0.25 * dz),
            ))

    # Cation-oxygen bridge candidates for extended_scan and selected COOH cases.
    for i in cation_idx:
        for j in anion_idx:
            dxy = _mic_xy_dist_for_slab(slab, pos[i, :2], pos[j, :2])
            dz = abs(float(pos[i, 2]) - float(pos[j, 2]))
            if dxy > 4.0 or dz > 3.0:
                continue
            vec = _mic_xy_delta_vec_for_slab(slab, pos[i, :2], pos[j, :2])
            xy = pos[i, :2] + 0.5 * vec
            z = max(float(pos[i, 2]), float(pos[j, 2]))
            label = f"{syms[i]}-{syms[j]}_bridge"
            raw.append(_make_oxide_voc_site_record(
                slab,
                site_class="cation_oxygen_bridge",
                label=label,
                indices=(i, j),
                position=(float(xy[0]), float(xy[1]), z),
                rank=float(dxy + 0.25 * dz),
            ))

    out: list[Dict[str, object]] = []
    out.extend(_pick_representative_by_label([r for r in raw if r.get("oxide_site_class") == "cation_top"], int(pol.get("cation_top_per_element", 1))))
    out.extend(_pick_representative_by_label([r for r in raw if r.get("oxide_site_class") == "anion_top"], int(pol.get("anion_top_per_element", 1))))
    out.extend(_pick_representative_by_label([r for r in raw if r.get("oxide_site_class") == "cation_cation_bridge"], int(pol.get("cation_pair_bridge_per_pair", 1))))
    if int(pol.get("cation_oxygen_bridge_per_pair", 0)) > 0:
        out.extend(_pick_representative_by_label([r for r in raw if r.get("oxide_site_class") == "cation_oxygen_bridge"], int(pol.get("cation_oxygen_bridge_per_pair", 1))))

    # Fallback: if bridge routing fails on a difficult facet, create a best
    # available cation-cation bridge from the two highest/nearest cations.
    if cation_idx and not any(r.get("oxide_site_class") == "cation_cation_bridge" for r in out) and len(cation_idx) >= 2:
        pairs = []
        for a_i, i in enumerate(cation_idx):
            for j in cation_idx[a_i + 1:]:
                dxy = _mic_xy_dist_for_slab(slab, pos[i, :2], pos[j, :2])
                dz = abs(float(pos[i, 2]) - float(pos[j, 2]))
                pairs.append((dxy + 0.25 * dz, i, j, dxy, dz))
        if pairs:
            _rank, i, j, dxy, dz = sorted(pairs)[0]
            vec = _mic_xy_delta_vec_for_slab(slab, pos[i, :2], pos[j, :2])
            xy = pos[i, :2] + 0.5 * vec
            z = max(float(pos[i, 2]), float(pos[j, 2]))
            pair = sorted([str(syms[i]), str(syms[j])])
            out.append(_make_oxide_voc_site_record(
                slab,
                site_class="cation_cation_bridge",
                label=f"{pair[0]}-{pair[1]}_bridge",
                indices=(i, j),
                position=(float(xy[0]), float(xy[1]), z),
                rank=float(_rank),
                warning="fallback_bridge_from_nearest_cations",
            ))

    # Last fallback: wrap legacy oxide sites, but mark them as diagnostic so the
    # CSV is never empty and the user can see why routing failed.
    if not out:
        try:
            legacy = select_representative_sites(detect_oxide_surface_sites(slab), per_kind=1)
        except Exception:
            legacy = []
        for k, s in enumerate(legacy):
            p = np.asarray(getattr(s, "position", [0.0, 0.0, _top_z(slab)]), dtype=float)
            out.append(_make_oxide_voc_site_record(
                slab,
                site_class="cation_top",
                label=f"oxide_fallback_{getattr(s, 'kind', 'site')}",
                indices=tuple(int(i) for i in (getattr(s, 'surface_indices', ()) or ())),
                position=(float(p[0]), float(p[1]), float(p[2]) if p.shape[0] > 2 else _top_z(slab)),
                rank=float(k),
                warning="oxide_voc_routing_fallback_used;check_surface_termination",
            ))

    seen: set[tuple] = set()
    dedup: list[Dict[str, object]] = []
    for r in out:
        key = (str(r.get("oxide_site_class")), str(r.get("oxide_site_label")), tuple(r.get("surface_indices", ())))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def _state_allowed_on_oxide_site(state: str, site_rec: Mapping[str, object]) -> bool:
    cls = str(site_rec.get("oxide_site_class", "")).strip().lower()
    # Manual/legacy geometry sites represent generic adsorption basins
    # (ontop/bridge/fcc/hollow), not routed oxide chemistry classes.  They must
    # not be filtered by cation_top/cation_bridge routing tables.
    if not cls or cls in {"manual", "legacy_geometry", "geometry_representative"}:
        return True
    return cls in set(allowed_oxide_site_classes_for_state(state))


# -----------------------------------------------------------------------------
# Adsorbate template loading / placement
# -----------------------------------------------------------------------------

def _load_voc_template(ads_key: str, ref_dir: str | Path = "ref_gas") -> Tuple[Atoms, dict]:
    key = clean_adsorbate_label(ads_key)
    if key == "H":
        return Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[15.0, 15.0, 15.0], pbc=True), {"anchor_mode": "atom0"}

    template = VOC_TEMPLATE_FILES.get(key)
    if template is None and key in VOC_ADSORBATES:
        template = VOC_ADSORBATES[key].template
    if template is None:
        raise ValueError(f"Unsupported VOC adsorbate template: {ads_key!r}")

    path = resolve_ref_gas_template(str(template), ref_dir=ref_dir)

    atoms = read(str(path)).copy()
    spec = VOC_ADSORBATES.get(key)
    meta = {
        "ads_key": key,
        "template": str(template),
        "anchor_mode": getattr(spec, "anchor_mode", "auto") if spec is not None else "auto",
        "role": getattr(spec, "role", "voc_proxy") if spec is not None else "voc_proxy",
        "warning": getattr(spec, "warning", "") if spec is not None else "",
    }
    return atoms, meta


def _anchor_indices_and_position(mol: Atoms, ads_key: str, anchor_mode: str | None = None) -> Tuple[np.ndarray, Tuple[int, ...], str]:
    """Return the chemically meaningful adsorbate anchor coordinate.

    The previous VOC placement effectively used a generic C-first anchor. That
    is unsafe for aldehydes/carboxylates on oxide facets because the actual
    contact atom/group can differ by species.  This helper keeps the legacy
    geometry-site family (ontop/bridge/fcc) but makes the adsorbate anchor
    explicit.
    """
    key = clean_adsorbate_label(ads_key)
    syms = mol.get_chemical_symbols()
    pos = np.asarray(mol.get_positions(), dtype=float)
    mode = str(anchor_mode or "auto").strip().lower()

    def idxs(sym: str) -> List[int]:
        return [i for i, s in enumerate(syms) if s == sym]

    c_idx = idxs("C")
    o_idx = idxs("O")

    def _nearest_c_to_o() -> int | None:
        if not c_idx or not o_idx:
            return int(c_idx[0]) if c_idx else None
        cpos = pos[c_idx]
        opos = pos[o_idx]
        d = np.linalg.norm(cpos[:, None, :] - opos[None, :, :], axis=2)
        ii = int(np.argmin(np.min(d, axis=1)))
        return int(c_idx[ii])

    def _nearest_o_to_c(c: int | None = None) -> int | None:
        if not o_idx:
            return None
        if c is None:
            if not c_idx:
                return int(o_idx[0])
            c = _nearest_c_to_o()
        cpos = pos[int(c)]
        d = np.linalg.norm(pos[o_idx] - cpos[None, :], axis=1)
        return int(o_idx[int(np.argmin(d))])

    # Explicit modes first.
    if mode in {"carbonyl_o", "aldehyde_o", "o_atom"} and o_idx:
        oi = _nearest_o_to_c()
        if oi is not None:
            return pos[oi].copy(), (int(oi),), "carbonyl_o" if key == "CH3CHO" else "o_atom"

    if mode in {"carbonyl_c", "acyl_c", "carboxyl_c", "c_atom"} and c_idx:
        ci = _nearest_c_to_o()
        if ci is not None:
            label = "carbonyl_c" if key in {"CH3CHO", "CH3CO"} else ("carboxyl_c" if key in {"COOH", "CH3COOH"} else "c_atom")
            return pos[ci].copy(), (int(ci),), label

    if mode == "o_o_midpoint" and len(o_idx) >= 2:
        # Use the two closest O atoms as the carboxylate anchor pair.
        if len(o_idx) == 2:
            chosen = tuple(int(i) for i in o_idx[:2])
        else:
            dmin = 1e99
            chosen = (int(o_idx[0]), int(o_idx[1]))
            for a_i in range(len(o_idx)):
                for b_i in range(a_i + 1, len(o_idx)):
                    da = np.linalg.norm(pos[o_idx[a_i]] - pos[o_idx[b_i]])
                    if da < dmin:
                        dmin = float(da)
                        chosen = (int(o_idx[a_i]), int(o_idx[b_i]))
        return pos[list(chosen)].mean(axis=0), chosen, "o_o_midpoint"

    # Species-specific defaults.
    if key == "H":
        return pos[0].copy(), (0,), "H_atom"

    if key == "CH3CHO":
        # Oxide VOC seeds are most stable when the carbonyl oxygen is the
        # placement anchor; this avoids C-first aldehyde seeds floating away from
        # corrugated oxide facets.
        oi = _nearest_o_to_c()
        if oi is not None:
            return pos[oi].copy(), (int(oi),), "carbonyl_o"

    if key in {"CH3CH2O", "CH3CH2OH", "OH", "O"} and o_idx:
        return pos[o_idx[0]].copy(), (int(o_idx[0]),), "o_atom"

    if key == "CH3COO" and len(o_idx) >= 2:
        chosen = tuple(int(i) for i in o_idx[:2])
        return pos[list(chosen)].mean(axis=0), chosen, "o_o_midpoint"

    if key in {"CH3CO", "CO", "COOH", "CH3COOH"} and c_idx:
        ci = _nearest_c_to_o()
        if ci is not None:
            label = "carboxyl_c" if key in {"COOH", "CH3COOH"} else ("carbonyl_c" if key == "CH3CO" else "c_atom")
            return pos[ci].copy(), (int(ci),), label

    # Generic fallback: C first, then O, then atom 0.
    for sym in ("C", "O"):
        ids = idxs(sym)
        if ids:
            return pos[ids[0]].copy(), (int(ids[0]),), f"fallback_{sym}"
    return pos[0].copy(), (0,), "atom0"



def _rotation_matrix_from_vectors(v_from, v_to):
    """Return a 3x3 rotation matrix that rotates v_from onto v_to."""
    a = np.asarray(v_from, dtype=float)
    b = np.asarray(v_to, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3)
    a = a / na
    b = b / nb
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.eye(3)
    if c < -1.0 + 1e-10:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]], dtype=float)
        return np.eye(3) + 2.0 * (K @ K)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    K = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]], dtype=float)
    return np.eye(3) + K + (K @ K) * ((1.0 - c) / max(s * s, 1e-12))


def _direction_position_for_upright(mol: Atoms, ads_key: str, anchor_indices: Tuple[int, ...]) -> Optional[np.ndarray]:
    """Choose a non-anchor point that should point toward +surface normal."""
    key = clean_adsorbate_label(ads_key)
    syms = mol.get_chemical_symbols()
    pos = np.asarray(mol.get_positions(), dtype=float)
    anchors = tuple(int(i) for i in (anchor_indices or ()))
    if not anchors:
        return None
    anchor_center = pos[list(anchors)].mean(axis=0)

    c_idx = [i for i, s in enumerate(syms) if s == "C"]
    o_idx = [i for i, s in enumerate(syms) if s == "O"]
    h_idx = [i for i, s in enumerate(syms) if s == "H"]
    non_anchor = [i for i in range(len(pos)) if i not in set(anchors)]

    def nearest(indices):
        if not indices:
            return None
        d = np.linalg.norm(pos[indices] - anchor_center[None, :], axis=1)
        return int(indices[int(np.argmin(d))])

    if key == "CH3CHO":
        # O-down aldehyde seed: carbonyl C should point away from the surface.
        ci = nearest(c_idx)
        if ci is not None:
            return pos[ci].copy()

    if key in {"CO", "CH3CO", "COOH", "CH3COOH"}:
        # C-down seeds: nearest O / oxygenated moiety should remain above C.
        oi = nearest(o_idx)
        if oi is not None:
            return pos[oi].copy()

    if key == "CH3COO":
        ci = nearest(c_idx)
        if ci is not None:
            return pos[ci].copy()

    if key == "CH3CH2OH":
        # Ethanol is a product-retention/desorption proxy, not an ethoxy-like
        # O-bound intermediate.  If O--C is forced to point straight upward,
        # the tetrahedral C--O--H geometry necessarily pushes the hydroxyl H
        # partly toward the slab, which commonly relaxes into artificial
        # O--H dissociation.  Use the O-centered bisector of O--C and O--H as
        # the outward direction so both the ethyl group and hydroxyl H start on
        # the vacuum side of the O anchor.
        ci = nearest(c_idx)
        hi = nearest(h_idx)
        if ci is not None and hi is not None:
            vc = pos[int(ci)].copy() - anchor_center
            vh = pos[int(hi)].copy() - anchor_center
            nc = float(np.linalg.norm(vc))
            nh = float(np.linalg.norm(vh))
            if nc > 1e-8 and nh > 1e-8:
                v = vc / nc + vh / nh
                nv = float(np.linalg.norm(v))
                if nv > 1e-8:
                    return anchor_center + v / nv
        if ci is not None:
            return pos[ci].copy()

    if key == "CH3CH2O":
        ci = nearest(c_idx)
        if ci is not None:
            return pos[ci].copy()

    if non_anchor:
        return pos[non_anchor].mean(axis=0)
    return None


def _orient_adsorbate_upright(mol: Atoms, ads_key: str, anchor_indices: Tuple[int, ...], normal=(0.0, 0.0, 1.0)) -> Atoms:
    """Rotate VOC template so the molecular body points away from the slab.

    This is a rigid rotation around the selected anchor atom/group.  It prevents
    directional VOCs such as CH3CHO from being translated into the surface cage
    after the anchor is matched to an ontop/bridge/fcc target.
    """
    anchors = tuple(int(i) for i in (anchor_indices or ()))
    if not anchors or len(mol) <= 1:
        return mol

    pos = np.asarray(mol.get_positions(), dtype=float)
    anchor_center = pos[list(anchors)].mean(axis=0)
    direction_pos = _direction_position_for_upright(mol, ads_key, anchors)
    if direction_pos is None:
        return mol

    v = np.asarray(direction_pos, dtype=float) - anchor_center
    n = np.asarray(normal, dtype=float)
    if np.linalg.norm(v) < 1e-8 or np.linalg.norm(n) < 1e-8:
        return mol
    n = n / max(np.linalg.norm(n), 1e-12)

    R = _rotation_matrix_from_vectors(v, n)
    new_pos = (pos - anchor_center) @ R.T + anchor_center

    body = [i for i in range(len(new_pos)) if i not in set(anchors)]
    if body:
        rel = new_pos - anchor_center
        mean_proj = float(np.mean(rel[body] @ n))
        if mean_proj < 0.0:
            R2 = _rotation_matrix_from_vectors(-n, n)
            new_pos = (new_pos - anchor_center) @ R2.T + anchor_center

    mol.set_positions(new_pos)
    return mol


def _normalize_template_for_placement(mol: Atoms, ads_key: str, anchor_mode: str | None = None) -> Tuple[Atoms, Tuple[int, ...], str]:
    """Return an oriented template and anchor metadata.

    The raw molecule is rotated around its reactive anchor so the molecular body
    points toward +z before translation to the surface target.  This keeps the
    legacy ontop/bridge/fcc site family while avoiding buried CH3CHO/organic
    seeds on corrugated oxide surfaces.
    """
    a = mol.copy()
    _anchor_pos, anchor_indices, anchor_label = _anchor_indices_and_position(a, ads_key, anchor_mode=anchor_mode)
    a = _orient_adsorbate_upright(a, ads_key, tuple(int(i) for i in anchor_indices), normal=(0.0, 0.0, 1.0))
    _anchor_pos2, anchor_indices2, anchor_label2 = _anchor_indices_and_position(a, ads_key, anchor_mode=anchor_mode)
    return a, tuple(int(i) for i in anchor_indices2), anchor_label2

def _site_support_z(slab: Atoms, site: AdsSite, *, fallback_surface_z: Optional[float] = None) -> float:
    """Local z support based primarily on the atoms defining the selected site."""
    try:
        pos = np.asarray(slab.get_positions(), dtype=float)
        idx = tuple(int(i) for i in (getattr(site, "surface_indices", ()) or ()))
        if idx:
            zs = [float(pos[i, 2]) for i in idx if 0 <= i < len(pos)]
            if zs:
                return float(max(zs))
    except Exception:
        pass
    if fallback_surface_z is not None and np.isfinite(float(fallback_surface_z)):
        return float(fallback_surface_z)
    return _local_surface_z(slab, site)




def _compact_formula_from_elements(elements: Iterable[str]) -> str:
    """Return a compact formula-like string from a support-element list."""
    try:
        counts: dict[str, int] = {}
        for el in elements:
            s = str(el)
            if not s:
                continue
            counts[s] = counts.get(s, 0) + 1
        return " ".join(f"{el}{counts[el]}" for el in sorted(counts.keys()))
    except Exception:
        return ""


def _site_support_metadata(slab: Atoms, site: AdsSite) -> Dict[str, object]:
    """Element/atom metadata for the atoms defining the selected geometry site.

    VOC QA must know whether bridge_1/fcc_2 is metal-containing or O-only.
    This helper is VOC-only metadata; it does not change HER/OER site logic.
    """
    idx = tuple(int(i) for i in (getattr(site, "surface_indices", ()) or ()))
    elems: list[str] = []
    try:
        syms = slab.get_chemical_symbols()
        elems = [str(syms[i]) for i in idx if 0 <= i < len(syms)]
    except Exception:
        elems = []
    cations = [e for e in elems if _is_oxide_cation_symbol(e)]
    anions = [e for e in elems if not _is_oxide_cation_symbol(e)]
    return {
        "site_support_indices": tuple(int(i) for i in idx),
        "site_support_elements": "-".join(elems),
        "site_support_formula": _compact_formula_from_elements(elems),
        "site_support_cation_count": int(len(cations)),
        "site_support_anion_count": int(len(anions)),
        "site_support_is_o_only": bool(len(elems) > 0 and len(cations) == 0),
        "site_support_has_cation": bool(len(cations) > 0),
    }



def _select_surface_oxygen_for_reduction_h(
    slab: Atoms,
    site: AdsSite,
    *,
    prefer_adjacent: bool = False,
    z_window: float = 3.0,
    min_adjacent_xy: float = 1.05,
    max_adjacent_xy: float = 4.25,
) -> int | None:
    """Pick an *exposed top-surface O* atom for reduction-route H* placement.

    The previous reduction preview/run could choose an O atom that was laterally
    close to the selected site but vertically buried in the spinel/oxide cage.
    That produced H seeds inside the slab.  For SAGE-VOC reduction, H* is a
    surface-OH-like seed, so first restrict to the highest available surface-O
    layer and only then rank by lateral proximity/adjacency.  This helper is
    reduction-only; HER/OER/VOC-oxidation paths do not call it.
    """
    try:
        pos = np.asarray(slab.get_positions(), dtype=float)
        if pos.size == 0:
            return None
        syms = slab.get_chemical_symbols()
        site_xy = np.asarray(site.position[:2], dtype=float)
        zmax = float(np.max(pos[:, 2]))

        raw: list[tuple[float, float, int]] = []  # (dxy, dz_from_top, index)
        for i, (sym, p) in enumerate(zip(syms, pos)):
            if str(sym) not in ANION_SYMBOLS:
                continue
            dz_from_top = zmax - float(p[2])
            if dz_from_top > float(z_window):
                continue
            dxy = _mic_xy_dist_for_slab(slab, site_xy, p[:2])
            raw.append((float(dxy), float(dz_from_top), int(i)))
        if not raw:
            return None

        # Keep only the exposed O layer.  On corrugated spinel facets, zmax can
        # be a cation; use the highest oxygen layer as the reference.
        min_o_dz = min(c[1] for c in raw)
        top_o = [c for c in raw if c[1] <= min_o_dz + 0.75]
        if not top_o:
            top_o = raw

        if prefer_adjacent:
            adj = [c for c in top_o if float(min_adjacent_xy) <= c[0] <= float(max_adjacent_xy)]
            if adj:
                # For H*+CH3CHO*, prefer an adjacent exposed O near the primary
                # VOC site, but never sacrifice top-surface character.
                adj.sort(key=lambda c: (c[1], abs(c[0] - 1.80), c[0]))
                return int(adj[0][2])

        # For standalone H*, choose the nearest exposed top-O; sort by exposure
        # first to avoid buried/subsurface oxygen even when it is laterally close.
        top_o.sort(key=lambda c: (c[1], c[0]))
        return int(top_o[0][2])
    except Exception:
        return None


def _surface_h_target_from_oxygen(slab: Atoms, oxygen_index: int, *, oh_length: float = 0.98) -> np.ndarray:
    """Return an outward O-H target coordinate for H_on_surface_O seeds."""
    pos = np.asarray(slab.get_positions(), dtype=float)
    oi = int(oxygen_index)
    if oi < 0 or oi >= len(pos):
        return np.asarray([np.nan, np.nan, np.nan], dtype=float)
    # Use a conservative +z outward normal.  The slabify step already orients
    # the active surface toward +z for SAGE-VOC oxide slabs.
    return np.asarray([float(pos[oi, 0]), float(pos[oi, 1]), float(pos[oi, 2]) + float(oh_length)], dtype=float)


def _add_reduction_surface_h(
    current_atoms: Atoms,
    base_slab: Atoms,
    site: AdsSite,
    *,
    prefer_adjacent: bool = False,
) -> Tuple[Atoms, dict]:
    """Append reduction-route H* as H_on_surface_O.

    This is deliberately separate from _add_adsorbate_template(..., 'H') so that
    HER/OER and oxidation behavior remains unchanged.  For H*+CH3CHO*, call this
    after placing CH3CHO*; the H position is still selected from the clean slab
    so the H target is a real surface O rather than a VOC atom or a shifted local
    support plane.
    """
    slab0 = ensure_pbc3(base_slab)
    out = ensure_pbc3(current_atoms.copy())
    oi = _select_surface_oxygen_for_reduction_h(slab0, site, prefer_adjacent=prefer_adjacent)
    if oi is None:
        # Fallback keeps the calculation runnable, but metadata exposes that the
        # H seed is not a true surface-OH placement.
        target_xyz = _site_target_xyz(slab0, site, "H", height=0.60, surface_z=_local_surface_z(slab0, site))
        support_site = site
        support_z = _site_support_z(slab0, site, fallback_surface_z=_local_surface_z(slab0, site))
        placement_note = "fallback_no_surface_oxygen_found"
    else:
        target_xyz = _surface_h_target_from_oxygen(slab0, int(oi), oh_length=0.98)
        op = np.asarray(slab0.get_positions()[int(oi)], dtype=float)
        support_site = AdsSite(kind="surface_oxygen_top", position=(float(op[0]), float(op[1]), float(op[2])), surface_indices=(int(oi),))
        support_z = float(op[2])
        placement_note = "reduction_H_on_surface_O"

    out.append(Atom("H", tuple(float(x) for x in target_xyz)))
    idx = len(out) - 1
    support_meta = _site_support_metadata(slab0, support_site)
    return ensure_pbc3(out), {
        "ads_key": "H",
        "ads_start": int(idx),
        "ads_stop": int(idx + 1),
        "anchor_indices_local": (0,),
        "anchor_indices_global": (int(idx),),
        "anchor_xy_initial": (float(target_xyz[0]), float(target_xyz[1])),
        "anchor_xyz_initial": tuple(float(x) for x in target_xyz),
        "anchor_target_xyz": tuple(float(x) for x in target_xyz),
        "support_z_initial": float(support_z),
        "anchor_height_A": float(target_xyz[2] - support_z),
        "anchor_mode": "surface_OH_H_on_surface_O",
        "reduction_h_placement": placement_note,
        "reduction_h_surface_o_index": int(oi) if oi is not None else -1,
        **support_meta,
    }




def _add_ech_h_ads_at_xy(
    current_atoms: Atoms,
    base_slab: Atoms,
    site: AdsSite,
    *,
    xy_override: np.ndarray,
    height: float = 1.00,
    ech_seed_policy: str = "outer_H",
) -> Tuple[Atoms, dict]:
    """Append H* as an adsorbed-H ECH co-adsorbate, not surface-OH."""
    slab0 = ensure_pbc3(base_slab)
    out = ensure_pbc3(current_atoms.copy())
    surface_z = _local_surface_z(slab0, site)
    support_z = _site_support_z(slab0, site, fallback_surface_z=surface_z)
    target_xyz = _site_target_xyz(
        slab0,
        site,
        "H",
        xy_override=np.asarray(xy_override, dtype=float),
        height=float(height),
        surface_z=surface_z,
    )
    out.append(Atom("H", tuple(float(x) for x in target_xyz)))
    idx = len(out) - 1
    policy = str(ech_seed_policy or "outer_H")
    support_meta = _site_support_metadata(slab0, site)
    return ensure_pbc3(out), {
        "ads_key": "H",
        "ads_start": int(idx),
        "ads_stop": int(idx + 1),
        "anchor_indices_local": (0,),
        "anchor_indices_global": (int(idx),),
        "anchor_xy_initial": (float(target_xyz[0]), float(target_xyz[1])),
        "anchor_xyz_initial": tuple(float(x) for x in target_xyz),
        "anchor_target_xyz": tuple(float(x) for x in target_xyz),
        "support_z_initial": float(support_z),
        "anchor_height_A": float(target_xyz[2] - support_z),
        "anchor_mode": f"ECH_H_ads_{policy}",
        "reduction_h_placement": f"ECH_{policy}_H_ads_not_surface_OH",
        "coadsorption_seed_policy": f"standalone_CH3CHO_{policy}" if policy in {"outer_H", "near_carbonyl_H"} else str(policy),
        "ech_seed_policy": policy,
        "ech_seed_role": _ech_seed_role_for_policy(policy),
        **support_meta,
    }

def _rotation_matrix_axis_angle(axis, angle_rad: float) -> np.ndarray:
    """Return a right-handed axis-angle rotation matrix."""
    try:
        a = np.asarray(axis, dtype=float)
        n = float(np.linalg.norm(a))
        if n < 1e-12:
            return np.eye(3)
        a = a / n
        x, y, z = a
        c = float(np.cos(float(angle_rad)))
        si = float(np.sin(float(angle_rad)))
        C = 1.0 - c
        return np.asarray([
            [c + x*x*C,     x*y*C - z*si, x*z*C + y*si],
            [y*x*C + z*si,  c + y*y*C,    y*z*C - x*si],
            [z*x*C - y*si,  z*y*C + x*si, c + z*z*C],
        ], dtype=float)
    except Exception:
        return np.eye(3)




def _rotate_ch3cho_carbonyl_away_from_xy(
    ads_atoms: Atoms,
    *,
    carbonyl_c_index: int,
    carbonyl_o_index: int,
    direction_xy: Optional[np.ndarray],
) -> Atoms:
    """Rotate CH3CHO around carbonyl-C so the C→O xy projection points away from direction_xy.

    For the ECH H*+CH3CHO* seed, H_ads should approach the carbonyl-carbon side.
    Therefore the carbonyl oxygen is rotated to the opposite azimuth from the
    H_ads direction, while preserving the same tilted reduction precursor.
    """
    if direction_xy is None:
        return ads_atoms
    a = ads_atoms.copy()
    pos = np.asarray(a.get_positions(), dtype=float)
    c = int(carbonyl_c_index)
    o = int(carbonyl_o_index)
    desired = -np.asarray(direction_xy, dtype=float)[:2]
    nd = float(np.linalg.norm(desired))
    if not np.isfinite(nd) or nd < 1e-8:
        return a
    cur = pos[o, :2] - pos[c, :2]
    nc = float(np.linalg.norm(cur))
    if not np.isfinite(nc) or nc < 1e-8:
        return a
    cur = cur / nc
    desired = desired / nd
    cross_z = float(cur[0] * desired[1] - cur[1] * desired[0])
    dot = float(np.clip(cur[0] * desired[0] + cur[1] * desired[1], -1.0, 1.0))
    ang = float(np.arctan2(cross_z, dot))
    Rz = _rotation_matrix_axis_angle([0.0, 0.0, 1.0], ang)
    anchor = pos[c].copy()
    newpos = (pos - anchor) @ Rz.T + anchor
    a.set_positions(newpos)
    return a

def _orient_ch3cho_reduction_tilted(mol: Atoms, *, tilt_deg: float = 18.0, lateral_direction_xy: Optional[np.ndarray] = None) -> Tuple[Atoms, Tuple[int, ...], str]:
    """Reduction-only CH3CHO placement: carbonyl-C down, carbonyl O tilted upward.

    Oxidation uses the generic CH3CHO placement/QC already tuned above.  For
    acetaldehyde reduction, the reducible carbonyl group should approach the
    surface as a C/O tilted precursor rather than being forced into the
    oxidation-style O-down pose.  This helper is called only from the
    VOC-reduction branch.
    """
    a = mol.copy()
    syms = a.get_chemical_symbols()
    pos = np.asarray(a.get_positions(), dtype=float)
    c_idx = [i for i, x in enumerate(syms) if str(x).upper() == "C"]
    o_idx = [i for i, x in enumerate(syms) if str(x).upper() == "O"]
    if not c_idx or not o_idx:
        _ap, ai, al = _anchor_indices_and_position(a, "CH3CHO", anchor_mode="carbonyl_c")
        return _orient_adsorbate_upright(a, "CH3CHO", ai, normal=(0.0, 0.0, 1.0)), tuple(ai), str(al)

    # Carbonyl pair = shortest C-O pair.
    best = None
    for c in c_idx:
        for o in o_idx:
            dco = float(np.linalg.norm(pos[int(o)] - pos[int(c)]))
            if best is None or dco < best[0]:
                best = (dco, int(c), int(o))
    c = int(best[1])
    o = int(best[2])
    anchor_center = pos[c].copy()

    # First orient C->O roughly outward (+z), then apply a mild deterministic
    # tilt so the carbonyl group is not perfectly vertical and can relax toward
    # a nearby H/surface-OH without starting as a product-like C-H bond.
    v = pos[o] - pos[c]
    R1 = _rotation_matrix_from_vectors(v, np.asarray([0.0, 0.0, 1.0], dtype=float))
    p1 = (pos - anchor_center) @ R1.T + anchor_center

    # Tilt around the cell-y-like axis in template coordinates.  The exact
    # azimuth is not meant to encode a mechanism; it simply seeds a C/O tilted
    # reducible precursor reproducibly.
    R2 = _rotation_matrix_axis_angle([0.0, 1.0, 0.0], np.deg2rad(float(tilt_deg)))
    p2 = (p1 - anchor_center) @ R2.T + anchor_center

    a.set_positions(p2)
    if lateral_direction_xy is not None:
        a = _rotate_ch3cho_carbonyl_away_from_xy(
            a, carbonyl_c_index=c, carbonyl_o_index=o, direction_xy=lateral_direction_xy
        )
        return a, (int(c),), "reduction_carbonyl_c_tilted_H_attacked"
    return a, (int(c),), "reduction_carbonyl_c_tilted"


def _add_reduction_ch3cho_template(
    slab: Atoms,
    site: AdsSite,
    *,
    ref_dir: str | Path = "ref_gas",
    xy_override: Optional[np.ndarray] = None,
    toward_xy: Optional[np.ndarray] = None,
    height: float = 1.35,
    surface_z: Optional[float] = None,
) -> Tuple[Atoms, dict]:
    """Append CH3CHO* as a VOC-reduction tilted carbonyl precursor.

    This function is intentionally reduction-only.  It does not alter the
    oxidation CH3CHO* placement/QC path.
    """
    key = "CH3CHO"
    slab0 = ensure_pbc3(slab)
    support_meta = _site_support_metadata(slab0, site)
    support_z_initial = _site_support_z(slab0, site, fallback_surface_z=surface_z)
    target_xyz = _site_target_xyz(slab0, site, key, xy_override=xy_override, height=height, surface_z=surface_z)

    mol, meta = _load_voc_template(key, ref_dir=ref_dir)
    mol_place, anchor_local, anchor_mode = _orient_ch3cho_reduction_tilted(mol, tilt_deg=18.0, lateral_direction_xy=(None if toward_xy is None or xy_override is None else np.asarray(toward_xy, dtype=float)[:2] - np.asarray(xy_override, dtype=float)[:2]))
    anchor_pos = np.asarray(mol_place.get_positions()[int(anchor_local[0])], dtype=float)

    mol_place.set_cell(slab0.get_cell())
    mol_place.set_pbc(slab0.get_pbc())
    ads_initial_internal_bonds = _infer_descriptor_internal_bonds(
        np.asarray(mol_place.get_positions(), dtype=float),
        mol_place.get_chemical_symbols(),
    )
    mol_place.translate(np.asarray(target_xyz, dtype=float) - np.asarray(anchor_pos, dtype=float))
    mol_place, seed_auto_lift_A, seed_clearance_min_dist_A = _lift_adsorbate_clear_of_slab(
        slab0, mol_place, min_dist_A=1.00, step_A=0.15, max_lift_A=1.50
    )
    # The actual anchor coordinate may have changed due to the seed-clearance lift.
    target_xyz_actual = np.asarray(mol_place.get_positions()[int(anchor_local[0])], dtype=float)

    start = len(slab0)
    out = slab0 + mol_place
    stop = len(out)
    anchor_global = tuple(int(start + i) for i in anchor_local)
    return ensure_pbc3(out), {
        "ads_key": key,
        "ads_start": int(start),
        "ads_stop": int(stop),
        "anchor_indices_local": tuple(int(i) for i in anchor_local),
        "anchor_indices_global": anchor_global,
        "anchor_xy_initial": (float(target_xyz_actual[0]), float(target_xyz_actual[1])),
        "anchor_xyz_initial": tuple(float(x) for x in target_xyz_actual),
        "anchor_target_xyz": tuple(float(x) for x in target_xyz_actual),
        "support_z_initial": float(support_z_initial),
        "anchor_height_A": float(target_xyz_actual[2] - support_z_initial),
        "seed_auto_lift_A": float(seed_auto_lift_A),
        "seed_clearance_min_dist_A": float(seed_clearance_min_dist_A),
        "seed_collision_fixed": bool(seed_auto_lift_A > 0.0),
        "ads_initial_internal_bonds": ads_initial_internal_bonds,
        **support_meta,
        "anchor_mode": str(anchor_mode),
        "voc_route_context": "reduction",
        "reduction_ch3cho_placement": "carbonyl_C_down_tilted_raised",
        "template": str(meta.get("template", "")),
        "role": str(meta.get("role", "")),
        "template_warning": str(meta.get("warning", "")),
    }

def _anchor_center_xyz(atoms: Atoms, group: dict) -> np.ndarray:
    ids = tuple(int(i) for i in group.get("anchor_indices_global", ()))
    if not ids:
        return np.asarray([np.nan, np.nan, np.nan], dtype=float)
    try:
        pos = np.asarray(atoms.get_positions(), dtype=float)
        valid = [i for i in ids if 0 <= i < len(pos)]
        if not valid:
            return np.asarray([np.nan, np.nan, np.nan], dtype=float)
        return pos[valid].mean(axis=0)
    except Exception:
        return np.asarray([np.nan, np.nan, np.nan], dtype=float)


def _anchor_site_distance(atoms: Atoms, group: dict) -> float:
    """Distance between the current reactive anchor and the intended target xyz."""
    try:
        anchor = _anchor_center_xyz(atoms, group)
        target = np.asarray(group.get("anchor_target_xyz", (np.nan, np.nan, np.nan)), dtype=float)
        if anchor.shape[0] < 3 or target.shape[0] < 3 or (not np.all(np.isfinite(anchor))) or (not np.all(np.isfinite(target))):
            return float("nan")
        return float(np.linalg.norm(anchor - target))
    except Exception:
        return float("nan")


def _anchor_slab_distance(slab: Atoms, atoms_rel: Atoms, group: dict) -> float:
    """Minimum distance from reactive anchor atom/group to slab atoms."""
    try:
        anchor = _anchor_center_xyz(atoms_rel, group)
        if not np.all(np.isfinite(anchor)):
            return float("nan")
        sp = np.asarray(slab.get_positions(), dtype=float)
        if sp.size == 0:
            return float("nan")
        return float(np.min(np.linalg.norm(sp - anchor[None, :], axis=1)))
    except Exception:
        return float("nan")


def _adsorbate_com_height_from_support(atoms_rel: Atoms, group: dict) -> float:
    """Adsorbate COM height relative to the original local support z."""
    try:
        start = int(group.get("ads_start", -1))
        stop = int(group.get("ads_stop", -1))
        if start < 0 or stop <= start or stop > len(atoms_rel):
            return float("nan")
        coords = np.asarray(atoms_rel.get_positions()[start:stop], dtype=float)
        support_z = _safe_float(group.get("support_z_initial", float("nan")))
        if not np.isfinite(support_z):
            return float("nan")
        return float(np.mean(coords[:, 2]) - support_z)
    except Exception:
        return float("nan")

def _site_target_xyz(
    slab: Atoms,
    site: AdsSite,
    ads_key: str,
    *,
    xy_override: Optional[np.ndarray] = None,
    height: float = 1.75,
    surface_z: Optional[float] = None,
) -> np.ndarray:
    """Target coordinate for the adsorbate anchor atom/group.

    x/y follow the selected legacy geometry site. z is derived from the atoms
    that define that site, not from a remote global top atom. This preserves
    ontop/bridge/fcc labels while avoiding floating seeds on corrugated oxides.
    """
    xy = np.asarray(xy_override if xy_override is not None else site.position[:2], dtype=float)
    support_z = _site_support_z(slab, site, fallback_surface_z=surface_z)
    return np.asarray([float(xy[0]), float(xy[1]), float(support_z + float(height))], dtype=float)



def _lift_adsorbate_clear_of_slab(
    slab: Atoms,
    mol: Atoms,
    *,
    min_dist_A: float = 1.00,
    step_A: float = 0.15,
    max_lift_A: float = 1.50,
) -> Tuple[Atoms, float, float]:
    """Lift a newly placed VOC template along +z until it does not clip the slab.

    This is a seed-generation guardrail for large VOC templates on corrugated
    or low bridge/hollow sites. It does not represent a thermodynamic correction.
    """
    out = mol.copy()
    total = 0.0
    dmin = _nearest_surface_distance(slab, np.asarray(out.get_positions(), dtype=float))
    while np.isfinite(dmin) and dmin < float(min_dist_A) and total + float(step_A) <= float(max_lift_A) + 1e-12:
        out.translate([0.0, 0.0, float(step_A)])
        total += float(step_A)
        dmin = _nearest_surface_distance(slab, np.asarray(out.get_positions(), dtype=float))
    return out, float(total), float(dmin) if np.isfinite(dmin) else float("nan")


def _voc_initial_anchor_height(ads_key: str) -> float:
    """Conservative VOC anchor height above local support atoms.

    This restores the pre-contact-hotfix placement scale.  The previous
    OER-style policy patch raised CH3CHO/OH/CO anchors to ~1.6-2.1 Å, which
    made single-point VOC seeds behave like gas-phase separated molecules on
    corrugated oxides.  Keep the OER-style relaxation policy, but seed the
    initial contact with shorter heights.
    """
    key = clean_adsorbate_label(ads_key)
    if key == "H":
        return 1.00
    if key == "OH":
        return 1.15
    if key in {"CH3COO", "CH3COOH", "COOH"}:
        return 1.25
    if key in {"O", "CO", "CH3CO"}:
        return 1.15
    if key == "CH3CH2OH":
        # Ethanol is a neutral product-retention/desorption proxy, not an
        # ethoxy-like chemisorbed intermediate.  Start it farther from the
        # surface so the O--H / alkyl side does not form a second artificial
        # surface contact on bridge/hollow sites.
        return 1.80
    if key in {"CH3CHO", "CH3CH2O"}:
        return 1.10
    return 1.10


def _add_adsorbate_template(
    slab: Atoms,
    site: AdsSite,
    ads_key: str,
    *,
    ref_dir: str | Path = "ref_gas",
    xy_override: Optional[np.ndarray] = None,
    height: float = 1.75,
    surface_z: Optional[float] = None,
) -> Tuple[Atoms, dict]:
    key = clean_adsorbate_label(ads_key)
    slab0 = ensure_pbc3(slab)
    support_meta = _site_support_metadata(slab0, site)
    support_z_initial = _site_support_z(slab0, site, fallback_surface_z=surface_z)
    target_xyz = _site_target_xyz(slab0, site, key, xy_override=xy_override, height=height, surface_z=surface_z)

    if key == "H":
        a = slab0.copy()
        a.append(Atom("H", tuple(float(x) for x in target_xyz)))
        idx = len(a) - 1
        return ensure_pbc3(a), {
            "ads_key": key,
            "ads_start": idx,
            "ads_stop": idx + 1,
            "anchor_indices_local": (0,),
            "anchor_indices_global": (idx,),
            "anchor_xy_initial": (float(target_xyz[0]), float(target_xyz[1])),
            "anchor_xyz_initial": tuple(float(x) for x in target_xyz),
            "anchor_target_xyz": tuple(float(x) for x in target_xyz),
            "support_z_initial": float(support_z_initial),
            "anchor_height_A": float(height),
            **support_meta,
            "anchor_mode": "H_atom",
        }

    mol, meta = _load_voc_template(key, ref_dir=ref_dir)
    mol_place, anchor_local, anchor_mode = _normalize_template_for_placement(mol, key, meta.get("anchor_mode"))
    anchor_pos, _anchor_idx_check, _anchor_label_check = _anchor_indices_and_position(mol_place, key, anchor_mode=anchor_mode)

    mol_place.set_cell(slab0.get_cell())
    mol_place.set_pbc(slab0.get_pbc())
    ads_initial_internal_bonds = _infer_descriptor_internal_bonds(
        np.asarray(mol_place.get_positions(), dtype=float),
        mol_place.get_chemical_symbols(),
    )
    mol_place.translate(np.asarray(target_xyz, dtype=float) - np.asarray(anchor_pos, dtype=float))

    product_clearance_lift_A = 0.0
    product_clearance_min_dist_A = float("nan")
    if key == "CH3CH2OH":
        # Product ethanol should not start as a two-point chemisorbed geometry.
        # Keep a single weak O-directed placement above the selected site and
        # lift the whole molecule if any atom clips the surface/cage.
        mol_place, product_clearance_lift_A, product_clearance_min_dist_A = _lift_adsorbate_clear_of_slab(
            slab0,
            mol_place,
            min_dist_A=1.35,
            step_A=0.10,
            max_lift_A=2.50,
        )

    start = len(slab0)
    out = slab0 + mol_place
    stop = len(out)
    anchor_global = tuple(int(start + i) for i in anchor_local)
    return ensure_pbc3(out), {
        "ads_key": key,
        "ads_start": int(start),
        "ads_stop": int(stop),
        "anchor_indices_local": tuple(int(i) for i in anchor_local),
        "anchor_indices_global": anchor_global,
        "anchor_xy_initial": (float(target_xyz[0]), float(target_xyz[1])),
        "anchor_xyz_initial": tuple(float(x) for x in target_xyz),
        "anchor_target_xyz": tuple(float(x) for x in target_xyz),
        "support_z_initial": float(support_z_initial),
        "ads_initial_internal_bonds": ads_initial_internal_bonds,
        **support_meta,
        "anchor_mode": str(anchor_mode),
        "product_seed_policy": "weak_product_retention_single_O_anchor" if key == "CH3CH2OH" else "",
        "product_clearance_lift_A": float(product_clearance_lift_A),
        "product_clearance_min_dist_A": float(product_clearance_min_dist_A),
        "template": str(meta.get("template", "")),
        "role": str(meta.get("role", "")),
        "template_warning": str(meta.get("warning", "")),
    }


def _add_state_to_slab(
    slab: Atoms,
    site: AdsSite,
    state: str,
    *,
    ref_dir: str | Path = "ref_gas",
    placement_route: str = "",
    ech_seed_policy: str = "default",
    material_type: str = "",
) -> Tuple[Atoms, list[dict]]:
    comps = state_components(state)
    if not comps:
        raise ValueError(f"Empty VOC descriptor state: {state!r}")
    a = slab.copy()
    groups: list[dict] = []

    surface_z0 = _local_surface_z(slab, site)

    if len(comps) == 1:
        c0 = clean_adsorbate_label(comps[0])
        # Reduction-route H* on an oxide is a surface hydroxyl-like seed.
        # This branch is VOC-reduction-only because H* is not part of the
        # acetaldehyde oxidation state list.
        if c0 == "H":
            if _is_metal_voc_context(material_type):
                a, meta = _add_metal_her_like_h_seed(
                    a,
                    slab,
                    site,
                    height=1.00,
                    placement_note="metal_HER_like_H_ads_site",
                    ech_seed_policy="metal_HER_like",
                )
            else:
                a, meta = _add_reduction_surface_h(a, slab, site, prefer_adjacent=False)
                meta["H_descriptor_source"] = "oxide_surface_H_or_OH_proxy"
                meta["H_placement_policy"] = str(meta.get("reduction_h_placement", "surface_OH_like"))
            groups.append(meta)
            return a, groups
        if c0 == "CH3CHO" and str(placement_route).strip().lower() == "reduction":
            a, meta = _add_reduction_ch3cho_template(a, site, ref_dir=ref_dir, height=1.35, surface_z=surface_z0)
            groups.append(meta)
            return a, groups
        # VOC mode is a local interfacial-proximity descriptor.  Initial heights
        # should seed a near-surface contact, not a gas-phase molecule floating
        # above a rigid slab.  The anchor itself is later constrained by the
        # staged relaxation policy.
        h0 = _voc_initial_anchor_height(c0)
        a, meta = _add_adsorbate_template(a, site, comps[0], ref_dir=ref_dir, height=h0, surface_z=surface_z0)
        groups.append(meta)
        return a, groups

    # Co-adsorption: put the VOC/intermediate on the selected site and seed
    # H*/OH* nearby.  Do not let a leading "H*" in "H*+CH3CHO*" occupy the
    # primary VOC site.
    comps_clean = [clean_adsorbate_label(c) for c in comps]

    # ECH-specific H*+CH3CHO* branch. Keep CH3CHO* centered on the selected
    # site with the same generic VOC pose.  The stable policy uses only outer_H.
    # The previous near_carbonyl_H xy-offset seed is disabled because it was not
    # carbonyl-coordinate-aware and could place H in nonphysical positions.
    if set(comps_clean) == {"H", "CH3CHO"} and len(comps_clean) == 2:
        policy = "outer_H"
        ch3cho_xy = np.asarray(site.position[:2], dtype=float)
        h_site = site
        h_placement_note = "outer_H"
        if _is_metal_voc_context(material_type):
            h_site, h_placement_note = _select_neighbor_metal_h_site_for_ech(slab, site)
            policy = "metal_HER_neighbor" if h_placement_note == "metal_HER_neighbor_site" else "metal_outer_fallback"
            h_xy = np.asarray(h_site.position[:2], dtype=float)
        else:
            h_xy = _outer_h_xy_for_ech(slab, site, offset_A=2.10)
        h_ch3cho = _voc_initial_anchor_height("CH3CHO")
        a, meta_ch3cho = _add_adsorbate_template(
            a,
            site,
            "CH3CHO*",
            ref_dir=ref_dir,
            xy_override=ch3cho_xy,
            height=h_ch3cho,
            surface_z=surface_z0,
        )
        meta_ch3cho["coadsorption_seed_policy"] = f"standalone_CH3CHO_{policy}"
        meta_ch3cho["coadsorption_role"] = "standalone_CH3CHO_pose"
        meta_ch3cho["ech_seed_policy"] = policy
        meta_ch3cho["ech_seed_role"] = _ech_seed_role_for_policy(policy)
        meta_ch3cho["ech_carbonyl_azimuth"] = "same_as_generic_standalone_CH3CHO"
        groups.append(meta_ch3cho)
        if _is_metal_voc_context(material_type):
            a, meta_h = _add_metal_her_like_h_seed(
                a,
                slab,
                h_site,
                height=1.00,
                placement_note=h_placement_note,
                ech_seed_policy=policy,
            )
            meta_h["coadsorption_seed_policy"] = f"standalone_CH3CHO_{policy}"
        else:
            a, meta_h = _add_ech_h_ads_at_xy(
                a,
                slab,
                site,
                xy_override=h_xy,
                height=1.10,
                ech_seed_policy=policy,
            )
        meta_h["coadsorption_role"] = policy
        groups.append(meta_h)
        return a, groups

    # H*+H* offset-based placement is disabled in the stable ECH policy.
    # A reliable HER-competition descriptor needs a true site-pair search rather
    # than a blind lateral xy shift.
    if len(comps_clean) == 2 and comps_clean.count("H") == 2:
        raise ValueError(
            "H*+H* ECH competition seed is disabled in the stable policy; "
            "implement site-pair-based H placement before enabling this state."
        )

    primary_i = 0
    for ii, cc in enumerate(comps_clean):
        if cc not in {"H", "OH", "O"}:
            primary_i = ii
            break
    ordered = [comps[primary_i]] + [c for ii, c in enumerate(comps) if ii != primary_i]

    c1 = clean_adsorbate_label(ordered[0])
    h1 = _voc_initial_anchor_height(c1)
    a, meta1 = _add_adsorbate_template(a, site, ordered[0], ref_dir=ref_dir, height=h1, surface_z=surface_z0)
    groups.append(meta1)

    for j, comp in enumerate(ordered[1:], start=1):
        cj = clean_adsorbate_label(comp)
        xyj = _offset_xy_for_coadsorption(slab, site, offset_A=1.25 * j)
        hj = _voc_initial_anchor_height(cj)
        a, metaj = _add_adsorbate_template(a, site, comp, ref_dir=ref_dir, xy_override=xyj, height=hj, surface_z=surface_z0)
        groups.append(metaj)
    return a, groups


# -----------------------------------------------------------------------------
# Relaxation / energy / QA helpers
# -----------------------------------------------------------------------------


def _slab_distortion_metrics(initial_atoms: Atoms, relaxed_atoms: Atoms, n_slab_atoms: int, *, top_z_window: float = 1.50) -> dict:
    """Quantify adsorbate-induced slab reconstruction.

    VOC proxy descriptors should preserve the reference surface basin.  If a
    relaxed state lowers its energy by pulling slab atoms toward the VOC, the
    resulting energy is a reconstruction diagnostic rather than a usable
    adsorption/proximity descriptor.
    """
    out = {
        "slab_rmsd(Å)": float("nan"),
        "slab_max_disp(Å)": float("nan"),
        "top_slab_rmsd(Å)": float("nan"),
        "top_slab_max_disp(Å)": float("nan"),
        "top_slab_max_lift(Å)": float("nan"),
        "surface_distorted": False,
        "surface_distortion_note": "",
    }
    try:
        n = int(max(0, min(n_slab_atoms, len(initial_atoms), len(relaxed_atoms))))
        if n <= 0:
            return out
        p0 = np.asarray(initial_atoms.get_positions()[:n], dtype=float)
        p1 = np.asarray(relaxed_atoms.get_positions()[:n], dtype=float)
        disp_vec = p1 - p0
        disp = np.linalg.norm(disp_vec, axis=1)
        out["slab_rmsd(Å)"] = float(np.sqrt(np.mean(disp ** 2)))
        out["slab_max_disp(Å)"] = float(np.max(disp))
        z0 = p0[:, 2]
        zmax = float(np.max(z0))
        top = np.where((zmax - z0) <= float(top_z_window))[0]
        if len(top) > 0:
            td = disp[top]
            lift = disp_vec[top, 2]
            out["top_slab_rmsd(Å)"] = float(np.sqrt(np.mean(td ** 2)))
            out["top_slab_max_disp(Å)"] = float(np.max(td))
            out["top_slab_max_lift(Å)"] = float(np.max(lift))
        distorted = (
            (np.isfinite(out["top_slab_rmsd(Å)"]) and out["top_slab_rmsd(Å)"] > 0.35)
            or (np.isfinite(out["top_slab_max_disp(Å)"]) and out["top_slab_max_disp(Å)"] > 0.80)
            or (np.isfinite(out["top_slab_max_lift(Å)"]) and out["top_slab_max_lift(Å)"] > 0.70)
            or (np.isfinite(out["slab_max_disp(Å)"]) and out["slab_max_disp(Å)"] > 1.20)
        )
        out["surface_distorted"] = bool(distorted)
        if distorted:
            out["surface_distortion_note"] = (
                f"top_rmsd={out['top_slab_rmsd(Å)']:.2f};"
                f"top_max_disp={out['top_slab_max_disp(Å)']:.2f};"
                f"top_max_lift={out['top_slab_max_lift(Å)']:.2f}"
            )
    except Exception as e:
        out["surface_distortion_note"] = f"distortion_metric_failed:{type(e).__name__}:{e}"
    return out


def _infer_n_slab_atoms(atoms: Atoms, ads_groups: Optional[list[dict]]) -> int:
    if not ads_groups:
        return len(atoms)
    starts = []
    for g in ads_groups:
        try:
            starts.append(int(g.get("ads_start", len(atoms))))
        except Exception:
            pass
    return int(max(0, min(starts))) if starts else len(atoms)


def _surface_local_free_indices(atoms: Atoms, n_slab_atoms: int, site_xy: Tuple[float, float] | None, *, local_radius: float = 2.80, top_z_window: float = 1.50) -> set[int]:
    if site_xy is None or n_slab_atoms <= 0:
        return set()
    pos = np.asarray(atoms.get_positions()[:n_slab_atoms], dtype=float)
    if pos.size == 0:
        return set()
    zmax = float(np.max(pos[:, 2]))
    out: set[int] = set()
    for i, p in enumerate(pos):
        if (zmax - float(p[2])) > float(top_z_window):
            continue
        try:
            dxy = _mic_xy_distance(atoms.get_cell(), atoms.get_pbc(), site_xy, p[:2])
        except Exception:
            dxy = float(np.linalg.norm(np.asarray(site_xy, dtype=float) - p[:2]))
        if np.isfinite(dxy) and dxy <= float(local_radius):
            out.add(int(i))
    return out


def _build_voc_constraints(atoms: Atoms, ads_groups: Optional[list[dict]], *, policy: str, n_slab_atoms: int, site_xy: Tuple[float, float] | None, anchor_lock_mode: str = "xyz") -> list:
    """Build constraints for VOC proxy relaxation.

    The anchor is locked in x/y/z by default.  This is intentional: SAGE-VOC is
    not a global adsorption search.  It evaluates whether a user-selected local
    H*/OH*/VOC proximity state can be used as a descriptor without allowing the
    molecule to escape into the vacuum or the slab to reconstruct.
    """
    constraints: list = []
    pol = str(policy or "local_flex_proxy").strip().lower()

    if pol in {"placement_only", "single_point_proxy", "frozen_pose_proxy", "axis_locked_proxy"}:
        # VOC pose-constrained descriptor mode:
        # freeze the prepared slab and all adsorbate atoms.  This prevents
        # weakly bound molecular VOC seeds (e.g., CH3CHO*) from rotating away or
        # drifting into the vacuum during BFGS.  The descriptor is therefore a
        # pose-preserving single-point/local proxy, not an adsorption-minimum search.
        fixed = list(range(len(atoms)))
    elif pol in {"rigid_proxy", "short_relax_proxy", "normal_relax_proxy"}:
        # OER-style VOC relax: slab fixed, adsorbate allowed to relax around the locked anchor.
        fixed = list(range(int(n_slab_atoms)))
    elif pol == "local_flex_proxy":
        free = _surface_local_free_indices(atoms, int(n_slab_atoms), site_xy)
        fixed = [i for i in range(int(n_slab_atoms)) if i not in free]
        # If no local surface basin is identified, fall back to rigid slab.
        if len(free) == 0:
            fixed = list(range(int(n_slab_atoms)))
    elif pol == "free_diagnostic":
        fixed = []
        try:
            base_cons = build_relax_constraints(atoms, relaxation_scope="partial", n_fix_layers=2)
            if base_cons:
                constraints.extend(base_cons if isinstance(base_cons, (list, tuple)) else [base_cons])
        except Exception:
            pass
    else:
        fixed = list(range(int(n_slab_atoms)))

    if fixed:
        constraints.append(FixAtoms(indices=[int(i) for i in fixed]))

    if ads_groups and str(anchor_lock_mode).lower() not in {"none", "false", "off"}:
        lock = [True, True, True] if str(anchor_lock_mode).lower() == "xyz" else [True, True, False]
        seen_anchor = set()
        for g in ads_groups:
            for idx in tuple(int(i) for i in g.get("anchor_indices_global", ())):
                if idx < 0 or idx >= len(atoms) or idx in seen_anchor:
                    continue
                seen_anchor.add(idx)
                try:
                    constraints.append(FixCartesian(int(idx), lock))
                except Exception:
                    pass
    return constraints


def _relax_state_once(
    atoms: Atoms,
    *,
    steps: int,
    fmax: float,
    policy: str,
    n_slab_atoms: int,
    site_xy: Tuple[float, float] | None,
    ads_groups: Optional[list[dict]],
    anchor_lock_mode: str = "xyz",
):
    a0 = ensure_pbc3(atoms)
    a = a0.copy()
    a.calc = calc
    constraints = _build_voc_constraints(
        a,
        ads_groups,
        policy=policy,
        n_slab_atoms=int(n_slab_atoms),
        site_xy=site_xy,
        anchor_lock_mode=anchor_lock_mode,
    )
    try:
        if constraints:
            a.set_constraint(constraints)
    except Exception:
        pass

    t0 = time.perf_counter()
    err = ""
    conv = None
    nsteps = 0
    try:
        if int(steps) > 0:
            dyn = BFGS(a, logfile=None)
            dyn.run(fmax=float(fmax), steps=int(steps))
            try:
                conv = bool(dyn.converged())
            except Exception:
                conv = None
            try:
                nsteps = int(dyn.get_number_of_steps()) if hasattr(dyn, "get_number_of_steps") else int(getattr(dyn, "nsteps", 0))
            except Exception:
                nsteps = int(steps)
        E = float(a.get_potential_energy())
    except Exception as e:
        err = str(e)
        E = float("nan")
    elapsed = float(time.perf_counter() - t0)
    try:
        a.set_constraint()
    except Exception:
        pass
    metrics = _slab_distortion_metrics(a0, a, int(n_slab_atoms))
    meta = {
        "converged": conv,
        "n_steps": int(nsteps),
        "elapsed_s": elapsed,
        "error": err,
        "anchor_xy_lock": bool(str(anchor_lock_mode).lower() not in {"none", "false", "off"}),
        "anchor_lock_mode": str(anchor_lock_mode),
        "relax_policy": str(policy),
        "selected_for_descriptor": bool(str(policy).lower() not in {"free_diagnostic", "placement_only"}),
        "diagnostic_only": bool(str(policy).lower() in {"free_diagnostic", "placement_only"}),
        "pose_constrained": bool(str(policy).lower() in {"placement_only", "single_point_proxy", "frozen_pose_proxy", "axis_locked_proxy"}),
        **metrics,
    }
    return a, E, meta


def _relax_state(
    atoms: Atoms,
    *,
    steps: int,
    fmax: float = 0.05,
    relaxation_scope: str = "partial",
    n_fix_layers: int = 2,
    ads_groups: Optional[list[dict]] = None,
    anchor_xy_lock: bool = True,
    voc_relaxation_policy: str = "normal_relax",
    site_xy: Tuple[float, float] | None = None,
):
    """Evaluate a VOC descriptor state with OER-style freedom policies.

    VOC mode deliberately separates geometry checking from relaxation:
      - placement_only: save the placed structure only; no energy descriptor.
      - single_point: no geometry relaxation; one calculator energy evaluation.
      - short_relax: slab fixed, adsorbate anchor locked, short adsorbate-only BFGS.
      - normal_relax: slab fixed, adsorbate anchor locked, normal adsorbate-only BFGS.
      - local_flex/staged/free remain legacy diagnostic policies.

    HER/OER paths do not call this function; this is VOC-only behavior.
    """
    n_slab = _infer_n_slab_atoms(atoms, ads_groups)
    pol = str(voc_relaxation_policy or "single_point").strip().lower()

    # Normalize legacy names into the OER-style VOC policy vocabulary.
    alias = {
        "frozen": "single_point",
        "frozen_pose": "single_point",
        "frozen_pose_proxy": "single_point",
        "single_point_pose": "single_point",
        "pose_proxy": "single_point",
        "axis_locked": "single_point",
        "axis_locked_proxy": "single_point",
        "anchored_axis_proxy": "single_point",
        "rigid": "normal_relax",
        "rigid_proxy": "normal_relax",
        "rigid slab proxy": "normal_relax",
        "local": "local_flex_proxy",
        "local_flex": "local_flex_proxy",
        "local-flex proxy": "local_flex_proxy",
        "free": "free_diagnostic",
        "free relaxation diagnostic only": "free_diagnostic",
    }
    pol = alias.get(pol, pol)

    # SAGE-VOC descriptors are standardized to Normal relax.
    # Legacy/diagnostic policy names are accepted for backward compatibility
    # but are intentionally mapped to the single user-facing descriptor policy.
    pol = "normal_relax"

    if pol in {"placement_only", "placement", "placement_only_proxy"}:
        a0 = ensure_pbc3(atoms)
        a = a0.copy()
        metrics = _slab_distortion_metrics(a0, a, int(n_slab))
        meta = {
            "converged": None,
            "n_steps": 0,
            "elapsed_s": 0.0,
            "error": "",
            "anchor_xy_lock": False,
            "anchor_lock_mode": "none",
            "relax_policy": "placement_only",
            "selected_for_descriptor": False,
            "diagnostic_only": True,
            "pose_constrained": True,
            **metrics,
        }
        return a, float("nan"), meta

    if pol in {"single_point", "single_point_proxy"}:
        return _relax_state_once(
            atoms,
            steps=0,
            fmax=fmax,
            policy="single_point_proxy",
            n_slab_atoms=n_slab,
            site_xy=site_xy,
            ads_groups=ads_groups,
            anchor_lock_mode="none",
        )

    if pol in {"short_relax", "short_relax_proxy"}:
        short_steps = max(1, min(int(steps), 40))
        return _relax_state_once(
            atoms,
            steps=short_steps,
            fmax=fmax,
            policy="short_relax_proxy",
            n_slab_atoms=n_slab,
            site_xy=site_xy,
            ads_groups=ads_groups,
            anchor_lock_mode="xyz",
        )

    if pol in {"normal_relax", "normal_relax_proxy"}:
        return _relax_state_once(
            atoms,
            steps=int(steps),
            fmax=fmax,
            policy="normal_relax_proxy",
            n_slab_atoms=n_slab,
            site_xy=site_xy,
            ads_groups=ads_groups,
            anchor_lock_mode="xyz",
        )

    if pol in {"local_flex_proxy"}:
        return _relax_state_once(atoms, steps=steps, fmax=fmax, policy="local_flex_proxy", n_slab_atoms=n_slab, site_xy=site_xy, ads_groups=ads_groups, anchor_lock_mode="xyz")

    if pol in {"free_diagnostic"}:
        return _relax_state_once(atoms, steps=steps, fmax=fmax, policy="free_diagnostic", n_slab_atoms=n_slab, site_xy=site_xy, ads_groups=ads_groups, anchor_lock_mode="xy")

    # Legacy staged constrained proxy kept as a diagnostic fallback for existing workflows.
    # It is no longer the default VOC mode because weak VOC/OH seeds can desorb or
    # drive surface reconstruction under local-flex relaxation.
    local_a, local_E, local_meta = _relax_state_once(
        atoms,
        steps=steps,
        fmax=fmax,
        policy="local_flex_proxy",
        n_slab_atoms=n_slab,
        site_xy=site_xy,
        ads_groups=ads_groups,
        anchor_lock_mode="xyz",
    )
    if not bool(local_meta.get("surface_distorted", False)) and not str(local_meta.get("error", "")):
        local_meta["fallback_from"] = ""
        local_meta["reconstruction_sensitive"] = False
        return local_a, local_E, local_meta

    rigid_a, rigid_E, rigid_meta = _relax_state_once(
        atoms,
        steps=steps,
        fmax=fmax,
        policy="normal_relax_proxy",
        n_slab_atoms=n_slab,
        site_xy=site_xy,
        ads_groups=ads_groups,
        anchor_lock_mode="xyz",
    )
    rigid_meta["fallback_from"] = "local_flex_proxy"
    rigid_meta["reconstruction_sensitive"] = bool(local_meta.get("surface_distorted", False))
    rigid_meta["fallback_reason"] = str(local_meta.get("surface_distortion_note", "") or local_meta.get("error", "surface_distorted"))
    for k in ["slab_rmsd(Å)", "slab_max_disp(Å)", "top_slab_rmsd(Å)", "top_slab_max_disp(Å)", "top_slab_max_lift(Å)", "surface_distortion_note"]:
        if k in local_meta:
            rigid_meta[f"local_flex_{k}"] = local_meta[k]
    return rigid_a, rigid_E, rigid_meta

def _anchor_xy_after(atoms: Atoms, group: dict) -> Tuple[float, float]:
    ids = tuple(int(i) for i in group.get("anchor_indices_global", ()))
    if not ids:
        return float("nan"), float("nan")
    pos = np.asarray(atoms.get_positions(), dtype=float)
    xy = pos[list(ids), :2].mean(axis=0)
    return float(xy[0]), float(xy[1])


def _ads_group_coords(atoms: Atoms, group: dict) -> Tuple[np.ndarray, list[str]]:
    start = int(group.get("ads_start", -1))
    stop = int(group.get("ads_stop", -1))
    if start < 0 or stop <= start or stop > len(atoms):
        return np.empty((0, 3)), []
    coords = np.asarray(atoms.get_positions()[start:stop], dtype=float)
    syms = atoms.get_chemical_symbols()[start:stop]
    return coords, list(syms)



def _min_surface_distance_for_ads_atoms(slab: Atoms, coords: np.ndarray, syms: list[str], *, heavy_only: bool = True) -> float:
    """Minimum adsorbate-to-slab distance; optionally heavy atoms only.

    This is intentionally chemistry-neutral.  It asks only whether the descriptor
    state remains in contact with the slab, not whether the closest slab atom is
    a cation or oxygen.  Site chemistry is reported as metadata elsewhere.
    """
    try:
        if coords is None or len(coords) == 0 or slab is None or len(slab) == 0:
            return float("nan")
        arr = np.asarray(coords, dtype=float)
        if heavy_only:
            ids = [i for i, s in enumerate(syms or []) if str(s).upper() != "H"]
            if ids:
                arr = arr[ids]
        sp = np.asarray(slab.get_positions(), dtype=float)
        if arr.size == 0 or sp.size == 0:
            return float("nan")
        d = np.linalg.norm(arr[:, None, :] - sp[None, :, :], axis=2)
        return float(np.min(d))
    except Exception:
        return float("nan")



def _nearest_slab_distance_for_points(slab: Atoms, points: np.ndarray) -> float:
    """Minimum distance from selected adsorbate points to any slab atom."""
    try:
        pts = np.asarray(points, dtype=float)
        if pts.size == 0 or slab is None or len(slab) == 0:
            return float("nan")
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        sp = np.asarray(slab.get_positions(), dtype=float)
        d = np.linalg.norm(pts[:, None, :] - sp[None, :, :], axis=2)
        return float(np.min(d))
    except Exception:
        return float("nan")


def _reactive_group_local_indices(coords: np.ndarray, syms: list[str], ads_key: str) -> tuple[list[int], str]:
    """Return the adsorbate-local atom indices that define surface binding.

    This is deliberately *not* a cation-only rule.  It only asks whether the
    chemically relevant part of the requested descriptor state is near the slab.
    Site chemistry remains metadata.  This prevents valid CO* poses from being
    rejected because the closest atom is not the preferred cation label, while
    still rejecting floating CO*/OH*/CH3COO* structures.
    """
    key = clean_adsorbate_label(ads_key)
    syms = [str(s) for s in (syms or [])]

    def ids(sym: str) -> list[int]:
        return [i for i, s in enumerate(syms) if s.upper() == sym.upper()]

    c_idx = ids("C")
    o_idx = ids("O")
    h_idx = ids("H")
    if key == "H":
        return (h_idx or list(range(len(syms)))[:1], "H_atom")
    if key in {"O", "OH", "CH3CH2O", "CH3CH2OH"}:
        return (o_idx or list(range(len(syms))), "O_anchor_group")
    if key == "CO":
        return (c_idx[:1] or list(range(len(syms))), "CO_C_anchor")
    if key == "COOH":
        # COOH* can appear in either O-bound or C-bound near-surface poses during
        # constrained VOC proxy relaxation.  Treat the intact carboxyl C+O group
        # as the reactive contact group; do not require the OH-side O alone to be
        # the closest atom.  Fragmentation is checked separately.
        try:
            c, oxy = _carboxyl_group_indices(np.asarray(coords, dtype=float), syms)
            ids_out = ([] if c is None else [int(c)]) + [int(i) for i in (oxy or [])]
            if ids_out:
                return (ids_out, "COOH_carboxyl_C_O_group")
        except Exception:
            pass
        return ((c_idx + o_idx) or list(range(len(syms))), "COOH_carboxyl_C_O_group")
    if key in {"CH3COO", "CH3COOH"}:
        # Acetate/acetic acid descriptors are governed primarily by the
        # carboxylate O/OH-side group.  A methyl-only contact must not pass QA.
        return (o_idx or c_idx or list(range(len(syms))), "carboxyl_O_group")
    if key == "CH3CHO":
        # Acetaldehyde is a secondary VOC descriptor.  Accept a tilted-bound
        # aldehyde as long as the carbonyl C/O group, not only the methyl group,
        # remains near the oxide surface.  Oxidation/reduction route chemistry is
        # interpreted downstream; QA should only reject floating aldehyde poses.
        if o_idx and c_idx and coords is not None and len(coords):
            pos = np.asarray(coords, dtype=float)
            try:
                best = None
                for oi in o_idx:
                    ds = np.linalg.norm(pos[c_idx] - pos[int(oi)][None, :], axis=1)
                    ci = int(c_idx[int(np.argmin(ds))])
                    dmin = float(np.min(ds))
                    if best is None or dmin < best[0]:
                        best = (dmin, int(oi), ci)
                if best is not None:
                    return ([int(best[1]), int(best[2])], "carbonyl_C_O_group")
            except Exception:
                return ((o_idx[:1] + c_idx[:1]) or list(range(len(syms))), "carbonyl_C_O_group")
        return ((o_idx + c_idx) or list(range(len(syms))), "carbonyl_C_O_group")
    if key == "CH3CO":
        # Acyl radical: require the carbonyl/acyl group, not methyl contact.
        if o_idx and c_idx and coords is not None and len(coords):
            pos = np.asarray(coords, dtype=float)
            oi = o_idx[0]
            try:
                d = np.linalg.norm(pos[c_idx] - pos[oi][None, :], axis=1)
                ci = c_idx[int(np.argmin(d))]
                return ([int(oi), int(ci)], "carbonyl_group")
            except Exception:
                return ([int(oi)] + c_idx[:1], "carbonyl_group")
        return (o_idx or c_idx or list(range(len(syms))), "carbonyl_group")
    # Fallback: all non-H atoms are the reactive heavy group.
    heavy = [i for i, s in enumerate(syms) if s.upper() != "H"]
    return (heavy or list(range(len(syms))), "heavy_atom_group")


def _reactive_group_contact_cutoffs(ads_key: str) -> tuple[float, float]:
    """Species-specific (distance, height) limits for reactive-group contact.

    These are intentionally structural-QA cutoffs, not chemistry-ranking rules.
    C1/C2 oxygenated species are stricter than the generic heavy-atom check so
    gas-phase-like CO*/CH3CHO*/carboxylate poses are not accepted merely because
    some nonreactive atom is near the slab.
    """
    key = clean_adsorbate_label(ads_key)
    if key == "H":
        return 2.80, 2.60
    if key in {"O", "OH"}:
        return 2.45, 2.35
    if key == "CO":
        return 2.35, 2.15
    if key == "CH3CO":
        return 2.45, 2.30
    if key == "COOH":
        # COOH* is allowed to remain valid as an intact carboxyl C/O-bound pose.
        # Use a slightly looser structural contact cutoff than acetate so that
        # normal C-bound COOH* does not get falsely rejected.
        return 2.80, 2.60
    if key in {"CH3COO", "CH3COOH"}:
        return 2.55, 2.35
    if key == "CH3CHO":
        # Secondary aldehyde descriptor: allow tilted carbonyl C/O-bound poses,
        # but still reject genuinely floating CH3CHO*.
        return 2.85, 2.65
    if key in {"CH3CH2O", "CH3CH2OH"}:
        return 2.85, 2.65
    return 2.75, 2.60



def _nearest_atom_distance_to_slab(slab: Atoms, point: np.ndarray) -> float:
    """Minimum distance from one adsorbate atom/point to the slab."""
    try:
        p = np.asarray(point, dtype=float).reshape(1, 3)
        return _nearest_slab_distance_for_points(slab, p)
    except Exception:
        return float("nan")


def _nearest_carbonyl_pair(coords: np.ndarray, syms: list[str]) -> tuple[int | None, int | None]:
    """Return (C_index, O_index) for the C/O pair closest to a carbonyl/acyl group."""
    try:
        pos = np.asarray(coords, dtype=float)
        c_idx = [i for i, s in enumerate(syms or []) if str(s).upper() == "C"]
        o_idx = [i for i, s in enumerate(syms or []) if str(s).upper() == "O"]
        if not c_idx or not o_idx:
            return None, None
        best = None
        for c in c_idx:
            for o in o_idx:
                d = float(np.linalg.norm(pos[int(c)] - pos[int(o)]))
                if best is None or d < best[0]:
                    best = (d, int(c), int(o))
        return (best[1], best[2]) if best is not None else (None, None)
    except Exception:
        return None, None


def _carboxyl_group_indices(coords: np.ndarray, syms: list[str]) -> tuple[int | None, list[int]]:
    """Return carboxyl C and its nearby O atoms for COOH/CH3COO-like species."""
    try:
        pos = np.asarray(coords, dtype=float)
        c_idx = [i for i, s in enumerate(syms or []) if str(s).upper() == "C"]
        o_idx = [i for i, s in enumerate(syms or []) if str(s).upper() == "O"]
        if not c_idx or not o_idx:
            return None, []
        scored = []
        for c in c_idx:
            ds = [(float(np.linalg.norm(pos[int(c)] - pos[int(o)])), int(o)) for o in o_idx]
            close = [o for d, o in ds if d <= 1.95]
            scored.append((len(close), -sum(d for d, _o in ds[:2]), int(c), close))
        scored.sort(reverse=True)
        c = scored[0][2]
        close = scored[0][3]
        if len(close) < 2:
            ds = sorted([(float(np.linalg.norm(pos[int(c)] - pos[int(o)])), int(o)) for o in o_idx])
            close = [o for _d, o in ds[:2]]
        return int(c), [int(o) for o in close]
    except Exception:
        return None, []


def _c_series_pose_metrics(slab: Atoms, coords: np.ndarray, syms: list[str], ads_key: str, support_z: float, *, route_context: str = "") -> dict:
    """C-containing VOC pose QA independent of site chemistry.\n\n    This supplements the generic reactive-group contact check.  It verifies that\n    C-containing descriptor states are still represented by the intended anchor\n    pose: CO* must be C-down, CH3CO* must keep the acyl C near the surface, and\n    carboxylate/carboxylic states must keep the carboxyl group near the surface.\n    It deliberately does not require cation-only contact.\n    """
    key = clean_adsorbate_label(ads_key)
    if key not in {"CO", "CH3CHO", "CH3CO", "COOH", "CH3COO", "CH3COOH"}:
        return {
            "c_series_pose_required": False,
            "c_series_pose_valid": True,
            "c_series_pose_reason": "not_c_series",
            "c_series_pose_mode": "not_c_series",
            "c_anchor_slab_dist(Å)": float("nan"),
            "c_anchor_height_above_support(Å)": float("nan"),
            "c_series_orientation_delta_z(Å)": float("nan"),
        }
    try:
        pos = np.asarray(coords, dtype=float)
        sy = [str(x).upper() for x in (syms or [])]
        if pos.size == 0:
            raise ValueError("empty_adsorbate")
        def h(i):
            return float(pos[int(i), 2] - float(support_z)) if np.isfinite(float(support_z)) else float("nan")
        def d(i):
            return _nearest_atom_distance_to_slab(slab, pos[int(i)])
        valid = False
        reason = ""
        mode = key
        c_dist = float("nan")
        c_height = float("nan")
        dz = float("nan")
        if key == "CO":
            c_idx = [i for i, s in enumerate(sy) if s == "C"]
            o_idx = [i for i, s in enumerate(sy) if s == "O"]
            if not c_idx or not o_idx:
                return {"c_series_pose_required": True, "c_series_pose_valid": False, "c_series_pose_reason": "CO_missing_C_or_O", "c_series_pose_mode": "CO_C_down", "c_anchor_slab_dist(Å)": float("nan"), "c_anchor_height_above_support(Å)": float("nan"), "c_series_orientation_delta_z(Å)": float("nan")}
            c, o = int(c_idx[0]), int(o_idx[0])
            c_dist, c_height = d(c), h(c)
            o_dist = d(o)
            dz = float(pos[o, 2] - pos[c, 2])
            valid = bool(np.isfinite(c_dist) and c_dist <= 2.35 and (not np.isfinite(c_height) or c_height <= 2.15) and dz >= 0.05 and (not np.isfinite(o_dist) or c_dist <= o_dist + 0.25))
            reason = "CO_C_down_surface_bound" if valid else f"invalid_CO_pose(Cd={c_dist:.2f},Ch={c_height:.2f},Oz-Cz={dz:.2f},Od={o_dist:.2f})"
            mode = "CO_C_down"
        elif key == "CH3CHO":
            c, o = _nearest_carbonyl_pair(pos, sy)
            if c is None or o is None:
                mode0 = "aldehyde_carbonyl_reduction_tilted" if str(route_context).lower() == "reduction" else "aldehyde_carbonyl_O_down"
                return {"c_series_pose_required": True, "c_series_pose_valid": False, "c_series_pose_reason": "CH3CHO_missing_carbonyl_pair", "c_series_pose_mode": mode0, "c_anchor_slab_dist(Å)": float("nan"), "c_anchor_height_above_support(Å)": float("nan"), "c_series_orientation_delta_z(Å)": float("nan")}
            cd, od = d(c), d(o)
            ch, oh = h(c), h(o)
            if str(route_context).strip().lower() == "reduction":
                # Reduction CH3CHO*: allow a tilted carbonyl C/O precursor.  The
                # carbonyl group must be near the surface, but the oxidation-only
                # requirement that carbonyl O is the closest/down atom is relaxed.
                c_dist = float(min(cd, od)) if np.isfinite(cd) and np.isfinite(od) else (cd if np.isfinite(cd) else od)
                c_height = float(min(ch, oh)) if np.isfinite(ch) and np.isfinite(oh) else (ch if np.isfinite(ch) else oh)
                dz = float(pos[o, 2] - pos[c, 2])  # positive means O above C / C-down
                max_d = 2.85
                max_h = 2.65
                valid = bool(
                    np.isfinite(c_dist)
                    and c_dist <= max_d
                    and (not np.isfinite(c_height) or c_height <= max_h)
                    and (not np.isfinite(dz) or dz >= -0.85)
                )
                reason = "reduction_aldehyde_carbonyl_group_surface_bound" if valid else f"reduction_aldehyde_carbonyl_detached(Cd={cd:.2f},Od={od:.2f},Ch={ch:.2f},Oh={oh:.2f},O-Cz={dz:.2f},cut={max_d:.2f}/{max_h:.2f})"
                mode = "aldehyde_carbonyl_reduction_tilted"
            else:
                # Oxidation CH3CHO* is only a secondary/accessibility descriptor.
                # Do not reject a tilted-bound aldehyde merely because it is not
                # strict carbonyl-O-down.  Reject only when the carbonyl C/O group
                # is genuinely detached from the surface.
                c_dist = float(min(cd, od)) if np.isfinite(cd) and np.isfinite(od) else (cd if np.isfinite(cd) else od)
                c_height = float(min(ch, oh)) if np.isfinite(ch) and np.isfinite(oh) else (ch if np.isfinite(ch) else oh)
                dz = float(pos[o, 2] - pos[c, 2])
                max_d = 2.85
                max_h = 2.65
                valid = bool(
                    np.isfinite(c_dist)
                    and c_dist <= max_d
                    and (not np.isfinite(c_height) or c_height <= max_h)
                )
                reason = "aldehyde_carbonyl_group_surface_bound" if valid else f"aldehyde_carbonyl_group_detached(Cd={cd:.2f},Od={od:.2f},Ch={ch:.2f},Oh={oh:.2f},O-Cz={dz:.2f},cut={max_d:.2f}/{max_h:.2f})"
                mode = "aldehyde_carbonyl_tilted_bound"
        elif key == "CH3CO":
            c, o = _nearest_carbonyl_pair(pos, sy)
            if c is None or o is None:
                return {"c_series_pose_required": True, "c_series_pose_valid": False, "c_series_pose_reason": f"{key}_missing_carbonyl_pair", "c_series_pose_mode": "carbonyl_group", "c_anchor_slab_dist(Å)": float("nan"), "c_anchor_height_above_support(Å)": float("nan"), "c_series_orientation_delta_z(Å)": float("nan")}
            cd, od = d(c), d(o)
            ch, oh = h(c), h(o)
            c_dist = float(min(cd, od)) if np.isfinite(cd) and np.isfinite(od) else (cd if np.isfinite(cd) else od)
            c_height = float(min(ch, oh)) if np.isfinite(ch) and np.isfinite(oh) else (ch if np.isfinite(ch) else oh)
            dz = float(pos[o, 2] - pos[c, 2])
            max_d = 2.45
            max_h = 2.30
            valid = bool(np.isfinite(c_dist) and c_dist <= max_d and (not np.isfinite(c_height) or c_height <= max_h))
            reason = "carbonyl_group_surface_bound" if valid else f"carbonyl_group_detached(d={c_dist:.2f},h={c_height:.2f},cut={max_d:.2f}/{max_h:.2f})"
            mode = "carbonyl_group"
        else:
            c, oxy = _carboxyl_group_indices(pos, sy)
            if c is None or not oxy:
                return {"c_series_pose_required": True, "c_series_pose_valid": False, "c_series_pose_reason": f"{key}_missing_carboxyl_group", "c_series_pose_mode": "carboxyl_group", "c_anchor_slab_dist(Å)": float("nan"), "c_anchor_height_above_support(Å)": float("nan"), "c_series_orientation_delta_z(Å)": float("nan")}
            ds = [d(i) for i in ([c] + list(oxy))]
            hs = [h(i) for i in ([c] + list(oxy))]
            c_dist = float(np.nanmin(ds)) if ds else float("nan")
            c_height = float(np.nanmin(hs)) if hs else float("nan")
            if key == "COOH":
                max_d, max_h = 2.80, 2.60
                mode = "COOH_carboxyl_C_O_group"
            else:
                max_d, max_h = 2.55, 2.35
                mode = "carboxyl_group"
            valid = bool(np.isfinite(c_dist) and c_dist <= max_d and (not np.isfinite(c_height) or c_height <= max_h))
            reason = "carboxyl_group_surface_bound" if valid else f"carboxyl_group_detached(d={c_dist:.2f},h={c_height:.2f},cut={max_d:.2f}/{max_h:.2f})"
        return {
            "c_series_pose_required": True,
            "c_series_pose_valid": bool(valid),
            "c_series_pose_reason": reason,
            "c_series_pose_mode": mode,
            "c_anchor_slab_dist(Å)": float(c_dist),
            "c_anchor_height_above_support(Å)": float(c_height),
            "c_series_orientation_delta_z(Å)": float(dz),
        }
    except Exception as e:
        return {
            "c_series_pose_required": True,
            "c_series_pose_valid": False,
            "c_series_pose_reason": f"c_series_pose_check_failed:{type(e).__name__}:{e}",
            "c_series_pose_mode": key,
            "c_anchor_slab_dist(Å)": float("nan"),
            "c_anchor_height_above_support(Å)": float("nan"),
            "c_series_orientation_delta_z(Å)": float("nan"),
        }

def _reactive_group_contact_metrics(slab: Atoms, coords: np.ndarray, syms: list[str], ads_key: str, support_z: float) -> dict:
    """Check whether the descriptor's reactive group is actually surface-bound.

    Heavy-atom minimum distance alone can be fooled by methyl/H contacts or by
    a nonreactive end of the molecule touching the slab.  This check looks at
    the species-specific anchor group: C for CO*, O for OH*/O*, carboxyl O group
    for COOH*/CH3COO*, and carbonyl group for CH3CHO*/CH3CO*.
    """
    ids, mode = _reactive_group_local_indices(coords, syms, ads_key)
    pts = np.asarray(coords, dtype=float)
    if pts.size == 0 or not ids:
        return {
            "reactive_group_contact_valid": False,
            "reactive_group_contact_reason": "reactive_group_missing",
            "reactive_group_mode": mode,
            "reactive_group_slab_dist(Å)": float("nan"),
            "reactive_group_height_above_support(Å)": float("nan"),
        }
    valid_ids = [int(i) for i in ids if 0 <= int(i) < len(pts)]
    if not valid_ids:
        return {
            "reactive_group_contact_valid": False,
            "reactive_group_contact_reason": "reactive_group_indices_invalid",
            "reactive_group_mode": mode,
            "reactive_group_slab_dist(Å)": float("nan"),
            "reactive_group_height_above_support(Å)": float("nan"),
        }
    group_pts = pts[valid_ids]
    d_slab = _nearest_slab_distance_for_points(slab, group_pts)
    if np.isfinite(float(support_z)):
        # Use the lowest reactive atom height, not COM.  Monodentate contact is
        # valid for carboxylates, so at least one reactive atom near the surface
        # is enough.
        h = float(np.nanmin(group_pts[:, 2]) - float(support_z))
    else:
        h = float("nan")
    max_d, max_h = _reactive_group_contact_cutoffs(ads_key)
    ok_d = bool(np.isfinite(d_slab) and d_slab <= max_d)
    ok_h = bool((not np.isfinite(h)) or h <= max_h)
    ok = bool(ok_d and ok_h)
    reason = "reactive_group_surface_bound" if ok else f"reactive_group_detached(d={d_slab:.2f}A,h={h:.2f}A,cut={max_d:.2f}/{max_h:.2f})"
    return {
        "reactive_group_contact_valid": ok,
        "reactive_group_contact_reason": reason,
        "reactive_group_mode": mode,
        "reactive_group_slab_dist(Å)": float(d_slab),
        "reactive_group_height_above_support(Å)": float(h),
    }


def _infer_descriptor_internal_bonds(coords: np.ndarray, syms: list[str]) -> tuple[tuple[int, int, float, str], ...]:
    """Infer descriptor-critical internal bonds from the placed template.

    We intentionally track C-C, C-O and O-H only.  C-H bonds are omitted to avoid
    over-rejecting methyl rotations, while COOH -> CO + OH and similar descriptor
    fragmentation is still caught robustly.
    """
    try:
        pos = np.asarray(coords, dtype=float)
        out = []
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                pair = "-".join(sorted([str(syms[i]).upper(), str(syms[j]).upper()]))
                d = float(np.linalg.norm(pos[i] - pos[j]))
                if pair == "C-O" and d <= 1.80:
                    out.append((int(i), int(j), d, pair))
                elif pair == "C-C" and d <= 1.85:
                    out.append((int(i), int(j), d, pair))
                elif pair == "H-O" and d <= 1.25:
                    out.append((int(i), int(j), d, pair))
        return tuple(out)
    except Exception:
        return tuple()


def _descriptor_initial_bonds_broken(coords: np.ndarray, group: dict) -> tuple[bool, str]:
    """Return whether descriptor-critical initial bonds are broken after relax."""
    bonds = group.get("ads_initial_internal_bonds", ()) or ()
    if not bonds:
        return False, ""
    try:
        pos = np.asarray(coords, dtype=float)
        key = clean_adsorbate_label(group.get("ads_key", ""))
        broken = []
        for item in bonds:
            i, j, d0, pair = int(item[0]), int(item[1]), float(item[2]), str(item[3])
            if i < 0 or j < 0 or i >= len(pos) or j >= len(pos):
                broken.append(f"{pair}:{i}-{j}:missing")
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            # Descriptor-state integrity threshold.  Loose enough for constrained
            # relaxation, strict enough to catch COOH -> CO + OH dissociation.
            cut = max(1.65 * float(d0), float(d0) + 0.55)
            if key == "CH3CO":
                # Acyl-like CH3CO* on oxides often reconstructs into eta1/eta2
                # C/O-bound states.  Do not reject solely because the C-O bond is
                # elongated by surface interaction; only reject true splitting.
                if str(pair).upper() == "C-O":
                    cut = max(cut, 2.25)
                elif str(pair).upper() == "C-C":
                    cut = max(cut, 2.05)
            # For the ethanol product proxy, O--H cleavage during relaxation
            # should be treated as dissociative product adsorption, not as true
            # C2-skeleton fragmentation.  True invalid fragmentation for
            # CH3CH2OH* is controlled by C--C/C--O skeleton integrity below.
            if key == "CH3CH2OH" and str(pair).upper() == "H-O":
                continue
            if d > cut:
                broken.append(f"{pair}:{i}-{j}:{d:.2f}>{cut:.2f}")
        return bool(broken), ";".join(broken)
    except Exception as e:
        return True, f"bond_integrity_check_failed:{type(e).__name__}:{e}"


def _adsorbate_buried_below_support(coords: np.ndarray, syms: list[str], support_z: float, *, margin_A: float = 0.35) -> bool:
    """Return True when an adsorbate heavy atom has sunk into the slab cage."""
    try:
        if coords is None or len(coords) == 0 or not np.isfinite(float(support_z)):
            return False
        ids = [i for i, s in enumerate(syms or []) if str(s).upper() != "H"]
        if not ids:
            ids = list(range(len(coords)))
        zmin = float(np.nanmin(np.asarray(coords, dtype=float)[ids, 2]))
        return bool(zmin < (float(support_z) - float(margin_A)))
    except Exception:
        return False


def _voc_internal_broken(coords: np.ndarray, syms: list[str], ads_key: str) -> bool:
    """Molecule-integrity check for VOC descriptor states.

    QA should reject a row when the requested descriptor state no longer exists
    after relaxation, e.g. COOH* dissociating into CO* + OH*.  This function is
    deliberately about internal connectivity only; surface contact is checked
    separately.
    """
    if coords is None or len(coords) == 0:
        return True
    key = clean_adsorbate_label(ads_key)
    syms = [str(s) for s in (syms or [])]

    def ids(sym: str) -> list[int]:
        return [i for i, s in enumerate(syms) if s == sym]

    idx_C = ids("C")
    idx_O = ids("O")
    idx_H = ids("H")

    def dist(i, j) -> float:
        return float(np.linalg.norm(coords[int(i)] - coords[int(j)]))

    def any_pair(a, b, cutoff: float) -> bool:
        return bool(a and b and any(dist(i, j) <= float(cutoff) for i in a for j in b if i != j))

    def count_o_near_c(c: int, cutoff: float = 1.85) -> int:
        return sum(1 for o in idx_O if dist(c, o) <= float(cutoff))

    if key == "H":
        return len(coords) != 1

    if key == "O":
        return len(idx_O) < 1

    if key == "OH":
        return len(idx_O) < 1 or len(idx_H) < 1 or not any_pair(idx_O, idx_H, 1.35)

    if key == "CO":
        return len(idx_C) < 1 or len(idx_O) < 1 or not any_pair(idx_C, idx_O, 1.70)

    if key == "COOH":
        if len(idx_C) < 1 or len(idx_O) < 2 or len(idx_H) < 1:
            return True
        # Keep one C bonded to two O atoms and one O-H bond.  If O-H breaks,
        # the final state is not intact COOH* but a dissociated CO*/OH*-like state.
        has_carboxyl_c = any(count_o_near_c(c, 1.90) >= 2 for c in idx_C)
        has_oh = any_pair(idx_O, idx_H, 1.35)
        return not (has_carboxyl_c and has_oh)

    if key == "CH3CHO":
        if len(idx_C) < 2 or len(idx_O) < 1:
            return True
        cc_ok = any(dist(i, j) <= 1.90 for i in idx_C for j in idx_C if i != j)
        co_ok = any_pair(idx_C, idx_O, 1.75)
        return not (cc_ok and co_ok)

    if key == "CH3CO":
        if len(idx_C) < 2 or len(idx_O) < 1:
            return True
        # CH3CO* is an acyl-like oxidation intermediate.  Surface-bound
        # reconstructed eta1/eta2 acyl states may elongate C-O without true
        # fragmentation.  Require only the heavy-atom CH3-C-O skeleton.
        cc_ok = any(dist(i, j) <= 2.05 for i in idx_C for j in idx_C if i != j)
        co_ok = any_pair(idx_C, idx_O, 2.25)
        return not (cc_ok and co_ok)

    if key in {"CH3COO", "CH3COOH"}:
        if len(idx_C) < 2 or len(idx_O) < 2:
            return True
        # Acetate/acetic acid skeleton: C-C plus one carbon bonded to two O atoms.
        cc_ok = any(dist(i, j) <= 1.95 for i in idx_C for j in idx_C if i != j)
        carboxyl_ok = any(count_o_near_c(c, 1.95) >= 2 for c in idx_C)
        if key == "CH3COOH" and len(idx_H) > 0:
            # Do not force every H assignment, but require at least one O-H if it is intended acid.
            oh_ok = any_pair(idx_O, idx_H, 1.35)
            return not (cc_ok and carboxyl_ok and oh_ok)
        return not (cc_ok and carboxyl_ok)

    if key in {"CH3CH2O", "CH3CH2OH"}:
        if len(idx_C) < 2 or len(idx_O) < 1:
            return True
        cc_ok = any(dist(i, j) <= 1.95 for i in idx_C for j in idx_C if i != j)
        co_ok = any_pair(idx_C, idx_O, 1.95)
        return not (cc_ok and co_ok)

    return False


def _qa_state(slab: Atoms, atoms_rel: Atoms, groups: list[dict], relax_meta: dict, *, initial_site_xy: Tuple[float, float], disp_thresh: float = 1.20) -> dict:
    """Simplified structure-validity QA for VOC descriptor rows.

    Current policy:
      1. The requested adsorbate descriptor must remain internally intact.
      2. The adsorbate must remain in contact with the slab surface.
      3. The adsorbate must not be buried/colliding with the slab.
      4. Severe slab collapse is rejected; mild distortion is retained as a warning.

    Do NOT hard-reject solely because the nearest surface atom is O vs cation,
    or because a requested site label is chemically less favorable.  Those are
    interpretation metadata, not structural validity failures.
    """
    if relax_meta.get("error"):
        return {
            "qa": "crashed",
            "qa_note": str(relax_meta.get("error", "")),
            "geometry_qa_reason": str(relax_meta.get("error", "")),
            "bound_geometry_valid": False,
            "surface_bound": False,
            "adsorbate_fragmented": False,
            "fragmentation_reason": "",
            "ads_lateral_disp(Å)": float("nan"),
            "min_ads_slab_dist(Å)": float("nan"),
            "min_ads_heavy_slab_dist(Å)": float("nan"),
            "anchor_slab_distance(Å)": float("nan"),
            "anchor_site_distance(Å)": float("nan"),
            "reactive_ads_distance(Å)": float("nan"),
            "adsorbate_com_surface_height(Å)": float("nan"),
            "reactive_group_contact_valid": False,
            "reactive_group_contact_reason": "calculation_failed",
            "reactive_group_mode": "",
            "reactive_group_slab_dist(Å)": float("nan"),
            "reactive_group_height_above_support(Å)": float("nan"),
            "c_series_pose_required": False,
            "c_series_pose_valid": False,
            "c_series_pose_reason": "calculation_failed",
            "c_series_pose_mode": "",
            "c_anchor_slab_dist(Å)": float("nan"),
            "c_anchor_height_above_support(Å)": float("nan"),
            "c_series_orientation_delta_z(Å)": float("nan"),
        }

    notes: list[str] = []
    hard_rejects: list[str] = []
    disp_vals: list[float] = []
    min_d_vals: list[float] = []
    min_heavy_vals: list[float] = []
    anchor_slab_vals: list[float] = []
    anchor_site_vals: list[float] = []
    com_height_vals: list[float] = []
    anchor_positions: list[np.ndarray] = []
    fragmented_any = False
    fragmentation_notes: list[str] = []
    reactive_valid_vals: list[bool] = []
    reactive_reason_vals: list[str] = []
    reactive_mode_vals: list[str] = []
    reactive_dist_vals: list[float] = []
    reactive_height_vals: list[float] = []
    c_pose_required_vals: list[bool] = []
    c_pose_valid_vals: list[bool] = []
    c_pose_reason_vals: list[str] = []
    c_pose_mode_vals: list[str] = []
    c_anchor_dist_vals: list[float] = []
    c_anchor_height_vals: list[float] = []
    c_orientation_dz_vals: list[float] = []

    for g in groups:
        coords, syms = _ads_group_coords(atoms_rel, g)
        key = clean_adsorbate_label(g.get("ads_key", ""))
        dmin_all = _nearest_surface_distance(slab, coords)
        dmin_heavy = _min_surface_distance_for_ads_atoms(slab, coords, syms, heavy_only=(key != "H"))
        min_d_vals.append(dmin_all)
        min_heavy_vals.append(dmin_heavy)

        x0, y0 = g.get("anchor_xy_initial", initial_site_xy)
        x1, y1 = _anchor_xy_after(atoms_rel, g)
        disp = _mic_xy_distance(atoms_rel.get_cell(), atoms_rel.get_pbc(), (x0, y0), (x1, y1))
        disp_vals.append(disp)

        anchor_xyz = _anchor_center_xyz(atoms_rel, g)
        if np.all(np.isfinite(anchor_xyz)):
            anchor_positions.append(anchor_xyz)
        a_slab = _anchor_slab_distance(slab, atoms_rel, g)
        a_site = _anchor_site_distance(atoms_rel, g)
        com_h = _adsorbate_com_height_from_support(atoms_rel, g)
        anchor_slab_vals.append(a_slab)
        anchor_site_vals.append(a_site)
        com_height_vals.append(com_h)

        support_z = _safe_float(g.get("support_z_initial", float("nan")))
        broken = _voc_internal_broken(coords, syms, key)
        bonds_broken, bonds_note = _descriptor_initial_bonds_broken(coords, g)
        broken = bool(broken or bonds_broken)
        buried = _adsorbate_buried_below_support(coords, syms, support_z, margin_A=0.35)

        # Surface-bound check.  First require that the species-specific reactive
        # group is near the slab; fall back to heavy-atom contact only as audit
        # metadata.  This catches floating CO*/OH*/CH3COO* while not rejecting
        # valid CO* solely because it is not cation-only.
        rc = _reactive_group_contact_metrics(slab, coords, syms, key, support_z)
        reactive_valid_vals.append(bool(rc.get("reactive_group_contact_valid", False)))
        reactive_reason_vals.append(str(rc.get("reactive_group_contact_reason", "")))
        reactive_mode_vals.append(str(rc.get("reactive_group_mode", "")))
        reactive_dist_vals.append(_safe_float(rc.get("reactive_group_slab_dist(Å)", float("nan"))))
        reactive_height_vals.append(_safe_float(rc.get("reactive_group_height_above_support(Å)", float("nan"))))

        route_context = str(g.get("voc_route_context", ""))
        cp = _c_series_pose_metrics(slab, coords, syms, key, support_z, route_context=route_context)
        c_pose_required_vals.append(bool(cp.get("c_series_pose_required", False)))
        c_pose_valid_vals.append(bool(cp.get("c_series_pose_valid", True)))
        c_pose_reason_vals.append(str(cp.get("c_series_pose_reason", "")))
        c_pose_mode_vals.append(str(cp.get("c_series_pose_mode", "")))
        c_anchor_dist_vals.append(_safe_float(cp.get("c_anchor_slab_dist(Å)", float("nan"))))
        c_anchor_height_vals.append(_safe_float(cp.get("c_anchor_height_above_support(Å)", float("nan"))))
        c_orientation_dz_vals.append(_safe_float(cp.get("c_series_orientation_delta_z(Å)", float("nan"))))

        bound_cut = 3.00
        if key in {"CH3CHO", "CH3CH2O", "CH3CH2OH"}:
            bound_cut = 3.20
        if key == "H":
            bound_cut = 2.80
        heavy_surface_bound = bool(np.isfinite(dmin_heavy) and dmin_heavy <= bound_cut)
        surface_bound = bool(heavy_surface_bound and bool(rc.get("reactive_group_contact_valid", False)))

        if broken:
            fragmented_any = True
            note = f"{key}:internal_descriptor_bond_broken"
            if bonds_note:
                note += f"[{bonds_note}]"
            fragmentation_notes.append(note)
            hard_rejects.append("adsorbate_fragmented")
        if buried:
            hard_rejects.append("buried_adsorbate")
            notes.append(f"{key}:heavy_atom_below_support")
        if np.isfinite(dmin_heavy) and dmin_heavy < 0.65:
            hard_rejects.append("invalid_seed_collision")
            notes.append(f"{key}:heavy_atom_overlap(d={dmin_heavy:.2f}A)")
        if not heavy_surface_bound:
            hard_rejects.append("not_surface_bound")
            notes.append(f"{key}:not_surface_bound(heavy_d={dmin_heavy:.2f}A)")
        elif not bool(rc.get("reactive_group_contact_valid", False)):
            hard_rejects.append("reactive_group_detached")
            notes.append(f"{key}:{rc.get('reactive_group_contact_reason', 'reactive_group_detached')}")
        elif bool(cp.get("c_series_pose_required", False)) and not bool(cp.get("c_series_pose_valid", True)):
            hard_rejects.append("c_series_pose_invalid")
            notes.append(f"{key}:{cp.get('c_series_pose_reason', 'c_series_pose_invalid')}")
        # Lateral migration is metadata unless the adsorbate becomes unbound or fragmented.
        if np.isfinite(disp) and disp > float(disp_thresh):
            notes.append(f"{key}:lateral_disp>{disp_thresh:.2f}A")

    reactive_distance = float("nan")
    if len(anchor_positions) >= 2:
        try:
            dlist = [float(np.linalg.norm(anchor_positions[i] - anchor_positions[j])) for i in range(len(anchor_positions)) for j in range(i + 1, len(anchor_positions))]
            reactive_distance = float(min(dlist)) if dlist else float("nan")
            if np.isfinite(reactive_distance) and reactive_distance > 3.50:
                hard_rejects.append("separated")
                notes.append("coadsorbate_anchor_distance>3.50A")
        except Exception:
            pass
    elif anchor_slab_vals:
        vals = [v for v in anchor_slab_vals if np.isfinite(v)]
        reactive_distance = float(min(vals)) if vals else float("nan")

    # Severe slab collapse is hard reject; mild distortion is warning only.
    top_disp = _safe_float(relax_meta.get("top_slab_max_disp(Å)", float("nan")))
    top_lift = _safe_float(relax_meta.get("top_slab_max_lift(Å)", float("nan")))
    surface_collapsed = bool((np.isfinite(top_disp) and top_disp > 2.00) or (np.isfinite(top_lift) and top_lift > 1.50))
    if surface_collapsed:
        hard_rejects.append("surface_collapsed")
        notes.append(f"severe_surface_collapse(top_disp={top_disp:.2f},top_lift={top_lift:.2f})")

    reject_priority = [
        "adsorbate_fragmented",
        "invalid_seed_collision",
        "buried_adsorbate",
        "reactive_group_detached",
        "c_series_pose_invalid",
        "not_surface_bound",
        "separated",
        "surface_collapsed",
    ]
    qa_final = "ok"
    for q in reject_priority:
        if q in hard_rejects:
            qa_final = q
            break

    pol = str(relax_meta.get("relax_policy", "")).strip().lower()
    policy_ok = {
        "placement_only": "placed_only",
        "single_point_proxy": "ok_single_point_proxy",
        "short_relax_proxy": "ok_short_relax_proxy",
        "normal_relax_proxy": "ok_normal_relax_proxy",
        "frozen_pose_proxy": "ok_frozen_pose_proxy",
        "axis_locked_proxy": "ok_axis_locked_proxy",
        "rigid_proxy": "ok_rigid_proxy",
        "local_flex_proxy": "ok_local_flex_proxy",
        "free_diagnostic": "ok_free_diagnostic",
    }
    if qa_final == "ok":
        qa_final = policy_ok.get(pol, "ok")

    if qa_final in set(policy_ok.values()) | {"ok"} and bool(relax_meta.get("surface_distorted", False)):
        qa_final = "surface_distorted_but_bound"
        notes.append("surface_distorted_but_adsorbate_bound")

    if bool(relax_meta.get("reconstruction_sensitive", False)):
        notes.append("fallback_from_local_flex_surface_distortion")

    all_notes = fragmentation_notes + notes
    qa_note = ";".join(str(x) for x in all_notes if str(x))
    geom_reason = qa_note if qa_note else "bound_intact_geometry_ok"
    ok_set = set(policy_ok.values()) | {"ok", "surface_distorted_but_bound"}

    return {
        "qa": qa_final,
        "qa_note": qa_note,
        "geometry_qa_reason": geom_reason,
        "bound_geometry_valid": bool(qa_final in ok_set),
        "surface_bound": bool(bool(reactive_valid_vals) and all(bool(x) for x in reactive_valid_vals)),
        "adsorbate_fragmented": bool(fragmented_any),
        "fragmentation_reason": ";".join(fragmentation_notes),
        "ads_lateral_disp(Å)": float(np.nanmax(disp_vals)) if disp_vals else float("nan"),
        "min_ads_slab_dist(Å)": float(np.nanmin(min_d_vals)) if min_d_vals else float("nan"),
        "min_ads_heavy_slab_dist(Å)": float(np.nanmin(min_heavy_vals)) if min_heavy_vals else float("nan"),
        "anchor_slab_distance(Å)": float(np.nanmin(anchor_slab_vals)) if anchor_slab_vals else float("nan"),
        "anchor_site_distance(Å)": float(np.nanmax(anchor_site_vals)) if anchor_site_vals else float("nan"),
        "reactive_ads_distance(Å)": reactive_distance,
        "adsorbate_com_surface_height(Å)": float(np.nanmax(com_height_vals)) if com_height_vals else float("nan"),
        "reactive_group_contact_valid": bool(bool(reactive_valid_vals) and all(bool(x) for x in reactive_valid_vals)),
        "reactive_group_contact_reason": ";".join([r for r in reactive_reason_vals if r]),
        "reactive_group_mode": ";".join([m for m in reactive_mode_vals if m]),
        "reactive_group_slab_dist(Å)": float(np.nanmax(reactive_dist_vals)) if reactive_dist_vals else float("nan"),
        "reactive_group_height_above_support(Å)": float(np.nanmax(reactive_height_vals)) if reactive_height_vals else float("nan"),
        "c_series_pose_required": bool(any(bool(x) for x in c_pose_required_vals)),
        "c_series_pose_valid": bool(all(bool(x) for x in c_pose_valid_vals)) if c_pose_valid_vals else True,
        "c_series_pose_reason": ";".join([r for r in c_pose_reason_vals if r]),
        "c_series_pose_mode": ";".join([m for m in c_pose_mode_vals if m]),
        "c_anchor_slab_dist(Å)": float(np.nanmax(c_anchor_dist_vals)) if c_anchor_dist_vals else float("nan"),
        "c_anchor_height_above_support(Å)": float(np.nanmax(c_anchor_height_vals)) if c_anchor_height_vals else float("nan"),
        "c_series_orientation_delta_z(Å)": float(np.nanmin(c_orientation_dz_vals)) if c_orientation_dz_vals else float("nan"),
        "anchor_xy_lock": bool(relax_meta.get("anchor_xy_lock", False)),
        "anchor_lock_mode": str(relax_meta.get("anchor_lock_mode", "")),
        "relax_policy": str(relax_meta.get("relax_policy", "")),
        "selected_for_descriptor": bool(relax_meta.get("selected_for_descriptor", True)),
        "diagnostic_only": bool(relax_meta.get("diagnostic_only", False)),
        "fallback_from": str(relax_meta.get("fallback_from", "")),
        "reconstruction_sensitive": bool(relax_meta.get("reconstruction_sensitive", False)),
        "fallback_reason": str(relax_meta.get("fallback_reason", "")),
        "slab_rmsd(Å)": relax_meta.get("slab_rmsd(Å)", float("nan")),
        "slab_max_disp(Å)": relax_meta.get("slab_max_disp(Å)", float("nan")),
        "top_slab_rmsd(Å)": relax_meta.get("top_slab_rmsd(Å)", float("nan")),
        "top_slab_max_disp(Å)": relax_meta.get("top_slab_max_disp(Å)", float("nan")),
        "top_slab_max_lift(Å)": relax_meta.get("top_slab_max_lift(Å)", float("nan")),
        "surface_distorted": bool(relax_meta.get("surface_distorted", False)),
        "surface_distortion_note": str(relax_meta.get("surface_distortion_note", "")),
        "local_flex_top_slab_max_disp(Å)": relax_meta.get("local_flex_top_slab_max_disp(Å)", float("nan")),
        "local_flex_top_slab_max_lift(Å)": relax_meta.get("local_flex_top_slab_max_lift(Å)", float("nan")),
        "local_flex_surface_distortion_note": str(relax_meta.get("local_flex_surface_distortion_note", "")),
    }


def _mic_3d_distance_for_atoms(atoms: Atoms, p0: np.ndarray, p1: np.ndarray) -> float:
    try:
        v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
        vec, _dist = find_mic(v, atoms.get_cell(), atoms.get_pbc())
        return float(np.linalg.norm(vec))
    except Exception:
        return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)))


def _carbonyl_c_anchor_xyz_from_group(atoms_rel: Atoms, group: dict) -> np.ndarray:
    try:
        coords, syms = _ads_group_coords(atoms_rel, group)
        c_loc, _o_loc = _nearest_carbonyl_pair(coords, list(syms))
        if c_loc is None:
            return _anchor_center_xyz(atoms_rel, group)
        start_i = int(group.get("ads_start", 0))
        return np.asarray(atoms_rel.get_positions()[start_i + int(c_loc)], dtype=float)
    except Exception:
        return _anchor_center_xyz(atoms_rel, group)


def _ech_distance_metrics(atoms_rel: Atoms, groups: list[dict]) -> dict:
    h_groups = [g for g in groups if clean_adsorbate_label(g.get("ads_key", "")) == "H"]
    ch3cho_groups = [g for g in groups if clean_adsorbate_label(g.get("ads_key", "")) == "CH3CHO"]
    out = {
        "ech_c_carbonyl_h_distance_A": float("nan"),
        "ech_h_h_distance_A": float("nan"),
    }
    try:
        if h_groups and ch3cho_groups:
            h_xyz = _anchor_center_xyz(atoms_rel, h_groups[0])
            c_xyz = _carbonyl_c_anchor_xyz_from_group(atoms_rel, ch3cho_groups[0])
            out["ech_c_carbonyl_h_distance_A"] = _mic_3d_distance_for_atoms(atoms_rel, c_xyz, h_xyz)
        if len(h_groups) >= 2:
            h0 = _anchor_center_xyz(atoms_rel, h_groups[0])
            h1 = _anchor_center_xyz(atoms_rel, h_groups[1])
            out["ech_h_h_distance_A"] = _mic_3d_distance_for_atoms(atoms_rel, h0, h1)
    except Exception:
        pass
    return out


def _apply_ech_diagnostic_policy(
    qa_meta: dict,
    comps: Tuple[str, ...],
    groups: list[dict],
    atoms_rel: Atoms,
    *,
    ech_seed_policy: str = "default",
) -> dict:
    """ECH-specific post-QC.

    Fragmentation/collision/burial/slab collapse remain invalid.  For
    H*+CH3CHO*, soft geometry labels such as geometry_invalid or
    adsorbate_floating can still be retained as diagnostic descriptors when
    the molecular CH3CHO identity is intact; this preserves useful
    co-adsorption/proximity outcomes without reviving fragmented structures.
    """
    comps_clean = [clean_adsorbate_label(c) for c in comps]
    is_h_ch3cho = (len(comps_clean) == 2 and set(comps_clean) == {"H", "CH3CHO"})
    is_h_h = (len(comps_clean) == 2 and comps_clean.count("H") == 2)
    if not (is_h_ch3cho or is_h_h):
        return qa_meta

    out = dict(qa_meta or {})
    dist = _ech_distance_metrics(atoms_rel, groups)
    out.update(dist)
    policy = str(ech_seed_policy or "default")
    if policy == "default" and is_h_ch3cho:
        policy = "outer_H"
    if policy == "default" and is_h_h:
        policy = "adjacent_H_pair"

    qa0 = str(out.get("qa", "")).strip().lower()
    frag = bool(out.get("adsorbate_fragmented", False))
    # Hard invalids remain excluded even in ECH diagnostic mode.  Do not include
    # geometry_invalid / adsorbate_floating here for H*+CH3CHO*: those labels can
    # arise from conservative VOC pose checks even when the relaxed coadsorbed
    # CH3CHO molecule remains intact and still provides useful proximity data.
    hard_invalid = {
        "crashed", "invalid", "adsorbate_fragmented",
        "invalid_seed_collision", "buried_adsorbate", "surface_collapsed",
        "bad_seed_high_energy",
    }
    ech_soft_geometry_labels = {
        "geometry_invalid", "adsorbate_floating", "c_series_pose_invalid",
        "reactive_group_detached", "not_surface_bound", "separated", "migrated",
    }

    out.update({
        "ech_state": True,
        "ech_state_type": "H_CH3CHO_coadsorption" if is_h_ch3cho else "H_H_competition",
        "ech_seed_policy": policy,
        "ech_seed_role": _ech_seed_role_for_policy(policy),
        "ech_h_transfer_proximity": False,
        "ech_product_like_collapse": False,
        "ech_h2_like_risk": False,
        "ech_coadsorption_retained": False,
        "ech_salvaged_soft_geometry": bool(is_h_ch3cho and qa0 in ech_soft_geometry_labels),
    })

    if frag or qa0 in hard_invalid:
        if frag or qa0 == "adsorbate_fragmented":
            cls = "invalid_fragmented"
        else:
            cls = f"invalid_{qa0 or 'hard_geometry'}"
        out.update({
            "ech_classification": cls,
            "ech_qc_note": f"hard_invalid:{qa0 or cls}",
            "ech_diagnostic_valid": False,
        })
        return out

    if is_h_h:
        out.update({
            "ech_classification": "disabled_h_h_offset_seed",
            "ech_qc_note": "H*+H* offset seed disabled; use site-pair-based placement before enabling",
            "ech_diagnostic_valid": False,
        })
        return out

    dch = _safe_float(out.get("ech_c_carbonyl_h_distance_A", float("nan")))
    if np.isfinite(dch) and dch < 1.25:
        cls = "product_like_collapse"
        out["ech_product_like_collapse"] = True
        out["ech_h_transfer_proximity"] = True
    elif np.isfinite(dch) and dch <= 2.30:
        cls = "h_transfer_proximity"
        out["ech_h_transfer_proximity"] = True
        out["ech_coadsorption_retained"] = True
    elif np.isfinite(dch) and dch <= 3.50:
        cls = "coadsorption_retained"
        out["ech_coadsorption_retained"] = True
    elif np.isfinite(dch):
        cls = "coadsorbates_separated"
    else:
        cls = "ech_geometry_diagnostic"

    out.update({
        "qa": "ech_diagnostic_valid",
        "bound_geometry_valid": True,
        "ech_classification": cls,
        "ech_qc_note": f"ech_H_CH3CHO_policy:{policy};backend_qa:{qa0 or 'ok'}",
        "ech_diagnostic_valid": True,
    })
    if bool(out.get("ech_salvaged_soft_geometry", False)):
        out["ech_qc_note"] = str(out.get("ech_qc_note", "")) + ";salvaged_soft_geometry"
    return out


def _compute_box_energy_for_template_file(
    template_path: Path,
    calc_obj,
    GROOT: Path,
    *,
    key: str,
    steps: int = 80,
    fmax: float = 0.05,
) -> Tuple[float, str]:
    """Read a resolved CIF template and compute its gas/box reference energy.

    This mirrors the CHE_mode practice of using the same UMA/OCP calculator for
    isolated box references, but it does not rely on relative ref_gas paths after
    the template file has already been resolved.
    """
    mol = read(str(template_path)).copy()
    mol = ensure_pbc3(mol, vac_z=10.0)
    mol.set_cell([15.0, 15.0, 15.0])
    mol.set_pbc(True)
    mol.center()
    mol.calc = calc_obj

    relax_error = ""
    if int(steps) > 0:
        try:
            dyn = BFGS(mol, logfile=None)
            dyn.run(fmax=float(fmax), steps=int(steps))
        except Exception as e:
            # Keep going.  For VOC references, a single-point fallback is more
            # useful than dropping the whole descriptor state as invalid.
            relax_error = f"box_relax_failed:{type(e).__name__}:{e}"

    E = float(mol.get_potential_energy())
    if not np.isfinite(E):
        raise RuntimeError(f"non-finite reference energy for {key} from {template_path}")

    try:
        out_dir = Path(GROOT) / "voc_ads_refs"
        out_dir.mkdir(parents=True, exist_ok=True)
        write(out_dir / f"{key}_box_relaxed.cif", mol)
    except Exception:
        pass

    src = f"computed:{template_path}"
    if relax_error:
        src += f";{relax_error};single_point_fallback_used"
    return E, src


def _load_or_compute_voc_ref_energy(
    ads_key: str,
    calc_obj,
    GROOT: Path,
    ref_dir: str | Path = "ref_gas",
    steps: int = 80,
    fmax: float = 0.05,
    E_H2_prepared: float | None = None,
) -> Tuple[float, str]:
    """Load or compute reference energy for a VOC descriptor component.

    Important correction vs the first VOC patch:
      - H* must follow the CHE/HER convention and use 0.5 * E_H2 from the
        already prepared slab workflow when available.  It should not search for
        a non-existent H_box.cif or recompute H2 through a stale relative path.
      - Molecular VOC/OH references are resolved by template-file existence,
        not by accepting the first ref_gas directory that happens to exist.
    """
    key = clean_adsorbate_label(ads_key)

    # CHE-compatible H reference.  _prepare_slab() already obtained E_H2 using
    # the existing SAGE gas-reference machinery, so reuse it directly.
    if key == "H":
        if E_H2_prepared is not None and np.isfinite(float(E_H2_prepared)):
            return 0.5 * float(E_H2_prepared), "0.5*E_H2_from_prepare_slab"

        # Fallback only if the prepared E_H2 is unavailable.
        try:
            h2_path = resolve_ref_gas_template("H2_box.cif", ref_dir=ref_dir)
            E, src = _compute_box_energy_for_template_file(
                h2_path, calc_obj, GROOT, key="H2", steps=steps, fmax=fmax
            )
            return 0.5 * float(E), "0.5*H2:" + src
        except Exception:
            gas_E, _src = _get_gas_box_energies(
                ["H2"], calc_obj, ref_dir=ref_dir, steps=steps, fmax=fmax, GROOT=GROOT
            )
            return 0.5 * float(gas_E["H2"]), "0.5*H2_from_CHE_mode_fallback"

    cache_path = Path(GROOT) / "voc_ads_refs.json"
    cache: dict = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    if key in cache:
        try:
            cached_E = float(cache[key])
            if np.isfinite(cached_E):
                return cached_E, "cache"
        except Exception:
            pass

    template = VOC_TEMPLATE_FILES.get(key)
    if template is None and key in VOC_ADSORBATES:
        template = VOC_ADSORBATES[key].template
    if template is None:
        raise ValueError(f"Unsupported VOC reference component: {ads_key!r}")

    template_path = resolve_ref_gas_template(str(template), ref_dir=ref_dir)
    E, src = _compute_box_energy_for_template_file(
        template_path, calc_obj, GROOT, key=key, steps=steps, fmax=fmax
    )

    cache[key] = float(E)
    try:
        cache_path.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass
    return float(E), src


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def _build_voc_target_sites(
    mtype: str,
    slab_u_rel: Atoms,
    sites: Iterable[str],
    user_ads_sites: Optional[Mapping[str, object]],
    descriptor_states: Iterable[str] | None = None,
    oxide_voc_site_policy: str = "standard_routed",
) -> List[Dict[str, object]]:
    target_sites: List[Dict[str, object]] = []
    if user_ads_sites:
        for label, site in user_ads_sites.items():
            pos = np.asarray(getattr(site, "position", []), dtype=float)
            if pos.shape[0] < 2:
                continue
            xyz = pos[:3] if pos.shape[0] >= 3 else np.asarray([pos[0], pos[1], np.nan], dtype=float)
            skind = str(getattr(site, "kind", "manual"))
            target_sites.append({
                "site_label": str(label),
                "site_kind": skind,
                "xy": np.asarray(pos[:2], dtype=float),
                "initial_xyz": np.asarray(xyz, dtype=float),
                "surface_indices": tuple(int(i) for i in getattr(site, "surface_indices", ()) or ()),
                "site": site,
                "seed_source": "user_ads_sites",
                "oxide_site_class": "manual",
                "oxide_site_label": str(label),
                "oxide_site_elements": "",
                "oxide_site_policy": "manual_user_ads_sites",
                "site_taxonomy_warning": "manual_site_not_routed_by_oxide_voc_policy",
            })
        return target_sites

    m = str(mtype).lower()
    site_names = [str(s) for s in (sites or ())]
    if m == "metal":
        raw = detect_metal_111_sites(slab_u_rel)
        raw = select_representative_sites(raw, per_kind=2)
        for i, s in enumerate(raw):
            if site_names and str(getattr(s, "kind", "")) not in site_names and not (str(getattr(s, "kind", "")) == "hollow" and any(x in site_names for x in ("fcc", "hcp"))):
                continue
            pos = np.asarray(getattr(s, "position", []), dtype=float)
            target_sites.append({
                "site_label": f"{s.kind}_{i}",
                "site_kind": str(s.kind),
                "xy": np.asarray(pos[:2], dtype=float),
                "initial_xyz": np.asarray(pos[:3], dtype=float),
                "surface_indices": tuple(int(k) for k in (getattr(s, "surface_indices", ()) or ())),
                "site": s,
                "seed_source": "voc_geometry_default",
            })
        return target_sites

    # Default SAGE-VOC oxide mode: keep the legacy geometry family
    # (ontop/bridge/fcc/hollow).  Routed cation-only sites are kept only as an
    # explicit diagnostic option because Co3O4(111)-type polar/corrugated facets
    # can fail when all states are forced onto a single cation-top family.
    site_filter = {s.strip().lower() for s in site_names if str(s).strip()}
    policy_key = str(oxide_voc_site_policy or "").strip().lower()
    use_routed = policy_key in {"fast_routed", "extended_scan", "routed_diagnostic"}

    if not use_routed:
        raw = detect_oxide_surface_sites(slab_u_rel)
        raw = select_representative_sites(raw, per_kind=2)
        for i, s in enumerate(raw):
            sk = str(getattr(s, "kind", "")).lower()
            if site_filter and sk not in site_filter and not (sk == "hollow" and any(x in site_filter for x in ("fcc", "hcp"))):
                continue
            pos = np.asarray(getattr(s, "position", []), dtype=float)
            if pos.shape[0] < 2:
                continue
            target_sites.append({
                "site_label": f"{s.kind}_{i}",
                "site_kind": str(s.kind),
                "xy": np.asarray(pos[:2], dtype=float),
                "initial_xyz": np.asarray(pos[:3] if pos.shape[0] >= 3 else [pos[0], pos[1], np.nan], dtype=float),
                "surface_indices": tuple(int(k) for k in (getattr(s, "surface_indices", ()) or ())),
                "site": s,
                "seed_source": "voc_geometry_default",
                "oxide_site_class": "legacy_geometry",
                "oxide_site_label": f"{s.kind}_{i}",
                "oxide_site_elements": "",
                "oxide_site_policy": "legacy_geometry",
                "site_taxonomy_warning": "",
            })
        return target_sites

    routed = _generate_routed_oxide_voc_sites(slab_u_rel, policy=oxide_voc_site_policy)
    needed_classes: set[str] = set()
    for st in (descriptor_states or ()):  # union of allowed classes for selected route
        needed_classes.update(allowed_oxide_site_classes_for_state(st))
    if not needed_classes:
        needed_classes = {"cation_top", "cation_cation_bridge"}

    for idx, rec0 in enumerate(routed):
        rec = dict(rec0)
        cls = str(rec.get("oxide_site_class", ""))
        lbl = str(rec.get("oxide_site_label", ""))
        if cls not in needed_classes:
            continue
        if site_filter and cls.lower() not in site_filter and lbl.lower() not in site_filter:
            continue
        rec["site_label"] = f"{lbl}_{idx}"
        rec["oxide_site_policy"] = str(oxide_voc_site_policy)
        target_sites.append(rec)

    if not target_sites and routed:
        for idx, rec0 in enumerate(routed[:3]):
            rec = dict(rec0)
            rec["site_label"] = f"{rec.get('oxide_site_label', 'oxide_site')}_{idx}"
            rec["oxide_site_policy"] = str(oxide_voc_site_policy)
            rec["site_taxonomy_warning"] = str(rec.get("site_taxonomy_warning", "")) + ";no_allowed_class_match_fallback"
            target_sites.append(rec)
    return target_sites


def _state_ref_sum(components: Tuple[str, ...], ref_E: Dict[str, float]) -> float:
    total = 0.0
    for c in components:
        if c not in ref_E or not np.isfinite(float(ref_E[c])):
            return float("nan")
        total += float(ref_E[c])
    return float(total)



def _infer_voc_placement_route(descriptor_states: Iterable[str]) -> str:
    """Infer route context only for placement/QC branching.

    This is not a chemistry classifier.  It exists to keep VOCs_oxidation
    untouched while allowing the acetaldehyde reduction route to use a different
    CH3CHO/H placement rule.
    """
    try:
        states = {normalize_voc_state(s) for s in (descriptor_states or ()) if str(s).strip()}
    except Exception:
        states = {str(s) for s in (descriptor_states or ())}
    red = {"H*", "CH3CH2O*", "CH3CH2OH*"}
    ox = {"OH*", "O*", "CH3CO*", "CH3COO*", "CO*", "COOH*", "CH3COOH*"}
    has_red = bool(states & red)
    has_ox = bool(states & ox)
    if has_red and not has_ox:
        return "reduction"
    if has_ox and not has_red:
        return "oxidation"
    return "mixed"

def _run_voc_proxy(
    mtype: str,
    user_slab_cif: str,
    out_root: Path,
    sites: Iterable[str],
    vac_z: float,
    layers: int,
    relax_mode: str,
    user_ads_sites: Optional[Mapping[str, object]],
    target_voc: str = "acetaldehyde",
    descriptor_states: Iterable[str] | None = None,
    ref_dir: str | Path = "ref_gas",
    voc_relaxation_policy: str = "normal_relax",
    oxide_voc_site_policy: str = "standard_routed",
):
    preset = get_voc_preset(target_voc)
    if descriptor_states is None:
        descriptor_states = tuple(preset.get("default_states", ()))
    descriptor_states = tuple(normalize_voc_state(s) for s in descriptor_states if str(s).strip())
    if not descriptor_states:
        descriptor_states = tuple(preset.get("default_states", ()))
    placement_route = _infer_voc_placement_route(descriptor_states)

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

    # VOC mode intentionally uses shorter adsorbate relaxations than full CHE.
    mode_l = str(relax_mode).strip().lower()
    if mode_l == "fast":
        ads_steps = 120
    elif mode_l == "tight":
        ads_steps = 360
    else:
        ads_steps = 220
    fmax = 0.06 if mode_l == "fast" else 0.05

    target_sites = _build_voc_target_sites(
        mtype,
        slab_u_rel,
        sites,
        user_ads_sites,
        descriptor_states=descriptor_states,
        oxide_voc_site_policy=oxide_voc_site_policy,
    )
    if not target_sites:
        raise RuntimeError(f"No target VOC adsorption sites were built for mtype='{mtype}'.")

    # Reference energies. H uses 0.5 H2; all molecular references use box CIFs.
    components_needed = set()
    for st in descriptor_states:
        components_needed.update(state_components(st))
    # Include dependencies for proximity even when base states were not requested.
    for st in descriptor_states:
        comps = state_components(st)
        if len(comps) > 1:
            components_needed.update(comps)
    ref_E: Dict[str, float] = {}
    ref_src: Dict[str, str] = {}
    for comp in sorted(components_needed):
        try:
            Eref, src = _load_or_compute_voc_ref_energy(comp, calc, GROOT, ref_dir=ref_dir, steps=80, fmax=0.05, E_H2_prepared=float(E_H2))
            ref_E[comp] = float(Eref)
            ref_src[comp] = str(src)
        except Exception as e:
            ref_E[comp] = float("nan")
            ref_src[comp] = f"missing_or_failed:{e}"

    rows: List[Dict[str, object]] = []
    per_site_state_energy: Dict[Tuple[str, str], dict] = {}

    # Ensure base single states needed by co-adsorption are computed first.
    eval_states = list(descriptor_states)
    for st in list(descriptor_states):
        comps = state_components(st)
        if len(comps) > 1:
            for c in comps:
                base = f"{c}*"
                if base not in eval_states:
                    eval_states.insert(0, base)

    eval_items: list[tuple[str, str]] = []
    for st in eval_states:
        for pol in _ech_seed_policies_for_state(st):
            eval_items.append((st, pol))

    for site_rec in target_sites:
        site: AdsSite = site_rec["site"]  # type: ignore[assignment]
        site_label = str(site_rec.get("site_label", "site"))
        site_kind = str(site_rec.get("site_kind", getattr(site, "kind", "unknown")))
        site_xy = tuple(float(x) for x in np.asarray(site_rec.get("xy", site.position[:2]), dtype=float)[:2])

        for state, ech_seed_policy in eval_items:
            comps = state_components(state)
            state_label = normalize_voc_state(state)
            if str(mtype).lower() == "oxide" and not _state_allowed_on_oxide_site(state_label, site_rec):
                continue
            state_key = (site_label, state_label) if ech_seed_policy == "default" else (site_label, f"{state_label}::{ech_seed_policy}")
            if state_key in per_site_state_energy:
                continue

            row: Dict[str, object] = {
                "mode": "VOC",
                "target_voc": str(target_voc),
                "target_voc_label": str(preset.get("label", target_voc)),
                "descriptor_state": state_label,
                "adsorbate": state_label,
                "state_type": "proximity" if len(comps) > 1 else "single_adsorbate",
                "ech_seed_policy": "" if ech_seed_policy == "default" else str(ech_seed_policy),
                "ech_seed_role": "" if ech_seed_policy == "default" else _ech_seed_role_for_policy(ech_seed_policy),
                "ech_state": bool(ech_seed_policy != "default"),
                "site_label": site_label,
                "site": site_kind,
                "site_kind": site_kind,
                "requested_site": site_kind,
                "initial_geom_site": site_kind,
                "relaxed_site": site_kind,
                "seed_source": str(site_rec.get("seed_source", "")),
                "oxide_site_class": str(site_rec.get("oxide_site_class", "")),
                "oxide_site_label": str(site_rec.get("oxide_site_label", site_label)),
                "oxide_site_elements": str(site_rec.get("oxide_site_elements", "")),
                "oxide_site_policy": str(site_rec.get("oxide_site_policy", oxide_voc_site_policy if str(mtype).lower() == "oxide" else "")),
                "site_taxonomy_warning": str(site_rec.get("site_taxonomy_warning", "")),
                "MODEL": MODEL_NAME,
                "DEVICE": DEVICE,
                "layers": int(layers),
                "vac_z(Å)": float(vac_z),
                "E_slab_user (eV)": float(E_slab_u),
                "proxy_warning": VOC_PROXY_WARNING,
                "template_warning": "",
                # Explicit CIF path columns are kept for all VOC rows so that
                # the post-run viewer can open rejected/diagnostic structures
                # without relying on filename guessing.
                "structure_cif": "",
                "initial_structure_cif": "",
            }

            seed_suffix = "" if ech_seed_policy == "default" else f"_{_safe_label(ech_seed_policy)}"
            structure_name = f"user_{_safe_label(site_label)}_{_safe_label(state_label)}{seed_suffix}.cif"
            structure_path = UROOT / "sites" / structure_name
            initial_structure_name = f"user_{_safe_label(site_label)}_{_safe_label(state_label)}{seed_suffix}_initial.cif"
            initial_structure_path = UROOT / "sites" / initial_structure_name

            if any((c not in ref_E or not np.isfinite(float(ref_E[c]))) for c in comps):
                row.update({
                    "E_state_user (eV)": float("nan"),
                    "ΔE_proxy (eV)": float("nan"),
                    "ΔE_ads_user (eV)": float("nan"),
                    "ΔE_VOC_ads_proxy (eV)": float("nan"),
                    "ΔE_H_VOC_proximity_proxy (eV)": float("nan"),
                    "ΔE_OH_VOC_proximity_proxy (eV)": float("nan"),
                    "ΔE_proximity_proxy (eV)": float("nan"),
                    "ΔE_raw_proxy_diagnostic (eV)": float("nan"),
                    "ΔE_raw_proximity_diagnostic (eV)": float("nan"),
                    "descriptor_energy_valid": False,
                    "descriptor_energy_mask_note": "missing_reference_template_or_energy",
                    "qa": "invalid",
                    "qa_note": "missing_reference_template_or_energy:" + json.dumps({c: ref_src.get(c, "missing") for c in comps}, ensure_ascii=False),
                    "ref_source": json.dumps({c: ref_src.get(c, "missing") for c in comps}),
                })
                rows.append(row)
                per_site_state_energy[state_key] = row
                continue

            # Metallic VOC/ECH H* must share the HER/CHE H-placement and energy
            # convention.  Do not route metal H* through the oxide surface-OH
            # fallback used for oxide reduction proxies.
            if _is_metal_voc_context(mtype) and len(comps) == 1 and clean_adsorbate_label(comps[0]) == "H":
                try:
                    Au, E_uH, _disp_raw, relax_meta = site_energy_two_stage(
                        slab_u_rel,
                        np.asarray(site_xy, dtype=float),
                        H0S,
                        int(z_steps),
                        int(free_steps),
                        return_meta=True,
                        mtype="metal",
                    )
                    dE_H = float(E_uH - float(E_slab_u) - 0.5 * float(E_H2))
                    dG_H = float(dE_H + float(STANDARD_CHE_CORR))
                    h_pos_final = np.asarray(Au.get_positions()[-1], dtype=float)
                    h_disp = _mic_xy_distance(Au.get_cell(), Au.get_pbc(), site_xy, h_pos_final[:2])
                    structure_path = UROOT / "sites" / f"user_{_safe_label(site_label)}_H_metal_CHE_like.cif"
                    try:
                        write(structure_path, Au)
                    except Exception:
                        structure_path = Path("")
                    row.update({
                        "E_state_user (eV)": float(E_uH),
                        "E_ref_sum (eV)": float(0.5 * float(E_H2)),
                        "ref_source": json.dumps({"H": "0.5*E_H2_from_prepare_slab"}),
                        "ΔE_proxy (eV)": float(dE_H),
                        "ΔE_ads_user (eV)": float(dE_H),
                        "ΔE_H_user (eV)": float(dE_H),
                        "ΔG_H_CHE (eV)": float(dG_H),
                        "ΔG_H_CHE_like (eV)": float(dG_H),
                        "standard_CHE_corr (eV)": float(STANDARD_CHE_CORR),
                        "ΔE_VOC_ads_proxy (eV)": float("nan"),
                        "ΔE_H_VOC_proximity_proxy (eV)": float("nan"),
                        "ΔE_OH_VOC_proximity_proxy (eV)": float("nan"),
                        "ΔE_proximity_proxy (eV)": float("nan"),
                        "ΔE_raw_proxy_diagnostic (eV)": float("nan"),
                        "ΔE_raw_ads_user_diagnostic (eV)": float("nan"),
                        "ΔE_raw_proximity_diagnostic (eV)": float("nan"),
                        "descriptor_energy_valid": True,
                        "descriptor_energy_mask_note": "",
                        "H_descriptor_source": "metal_CHE_HER_like",
                        "H_placement_policy": "CHE_mode.site_energy_two_stage",
                        "reduction_h_placement": "metal_CHE_HER_like_site_energy_two_stage",
                        "anchor_mode": "metal_CHE_HER_like_H_ads",
                        "placement_mode": str((relax_meta or {}).get("placement_mode", "slab_top")),
                        "selected_h0": _safe_float((relax_meta or {}).get("selected_h0")),
                        "h0_candidates": str((relax_meta or {}).get("h0_candidates", "")),
                        "z_relax_n_steps": int((relax_meta or {}).get("z_relax_n_steps", 0)),
                        "fine_relax_n_steps": int((relax_meta or {}).get("fine_relax_n_steps", 0)),
                        "relax_n_steps": int((relax_meta or {}).get("total_relax_n_steps", 0)),
                        "relax_converged": (relax_meta or {}).get("fine_relax_converged", None),
                        "structure_cif": str(Path(structure_path).resolve()) if str(structure_path) else "",
                        "ads_lateral_disp(Å)": float(h_disp),
                        "H_lateral_disp(Å)": float(h_disp),
                        "qa": "ok_metal_che_her_like",
                        "qa_note": "metal_Hstar_evaluated_with_CHE_mode_site_energy_two_stage",
                        "migrated": bool(float(h_disp) > 1.2),
                        "reliability": "reliable",
                    })
                except Exception as e:
                    row.update({
                        "E_state_user (eV)": float("nan"),
                        "ΔE_proxy (eV)": float("nan"),
                        "ΔE_ads_user (eV)": float("nan"),
                        "ΔE_H_user (eV)": float("nan"),
                        "ΔG_H_CHE (eV)": float("nan"),
                        "ΔG_H_CHE_like (eV)": float("nan"),
                        "descriptor_energy_valid": False,
                        "descriptor_energy_mask_note": "metal_H_CHE_like_failed",
                        "H_descriptor_source": "metal_CHE_HER_like",
                        "qa": "crashed",
                        "qa_note": "metal_CHE_HER_like_failed:" + str(e),
                        "relax_error": str(e),
                        "migrated": False,
                    })
                rows.append(row)
                per_site_state_energy[state_key] = row
                continue

            try:
                slab_state, groups = _add_state_to_slab(slab_u_rel, site, state_label, ref_dir=ref_dir, placement_route=placement_route, ech_seed_policy=ech_seed_policy, material_type=str(mtype))
                try:
                    write(initial_structure_path, slab_state)
                    row["initial_structure_cif"] = str(Path(initial_structure_path).resolve())
                except Exception:
                    row["initial_structure_cif"] = ""
                if groups:
                    row["anchor_mode"] = "+".join(str(g.get("anchor_mode", "")) for g in groups)
                    row["template_warning"] = ";".join(str(g.get("template_warning", "")) for g in groups if g.get("template_warning"))
                    row["anchor_height_A"] = "+".join(str(g.get("anchor_height_A", "")) for g in groups)
                    row["anchor_target_z_A"] = "+".join(
                        f"{float(g.get('anchor_target_xyz', (float('nan'), float('nan'), float('nan')))[2]):.3f}"
                        for g in groups
                    )
                    row["support_z_initial_A"] = "+".join(
                        f"{float(g.get('support_z_initial', float('nan'))):.3f}"
                        for g in groups
                    )
                    row["site_support_elements"] = "+".join(str(g.get("site_support_elements", "")) for g in groups)
                    row["site_support_formula"] = "+".join(str(g.get("site_support_formula", "")) for g in groups)
                    row["site_support_cation_count"] = "+".join(str(g.get("site_support_cation_count", "")) for g in groups)
                    row["site_support_anion_count"] = "+".join(str(g.get("site_support_anion_count", "")) for g in groups)
                    row["site_support_is_o_only"] = "+".join(str(g.get("site_support_is_o_only", "")) for g in groups)
                    row["site_support_has_cation"] = "+".join(str(g.get("site_support_has_cation", "")) for g in groups)
                    row["coadsorption_seed_policy"] = "+".join(dict.fromkeys(str(g.get("coadsorption_seed_policy", "")) for g in groups if g.get("coadsorption_seed_policy")))
                    row["coadsorption_role"] = "+".join(dict.fromkeys(str(g.get("coadsorption_role", "")) for g in groups if g.get("coadsorption_role")))
                    row["reduction_h_placement"] = "+".join(dict.fromkeys(str(g.get("reduction_h_placement", "")) for g in groups if g.get("reduction_h_placement")))
                    row["ech_seed_policy"] = ech_seed_policy if ech_seed_policy != "default" else "+".join(str(g.get("ech_seed_policy", "")) for g in groups if g.get("ech_seed_policy"))
                    row["ech_seed_role"] = _ech_seed_role_for_policy(ech_seed_policy) if ech_seed_policy != "default" else "+".join(str(g.get("ech_seed_role", "")) for g in groups if g.get("ech_seed_role"))
                relaxed, E_state, rmeta = _relax_state(
                    slab_state,
                    steps=int(ads_steps),
                    fmax=float(fmax),
                    relaxation_scope="partial",
                    n_fix_layers=2,
                    ads_groups=groups,
                    anchor_xy_lock=True,
                    voc_relaxation_policy=str(voc_relaxation_policy),
                    site_xy=site_xy,
                )
                qa_meta = _qa_state(slab_u_rel, relaxed, groups, rmeta, initial_site_xy=site_xy)
                qa_meta = _apply_ech_diagnostic_policy(qa_meta, comps, groups, relaxed, ech_seed_policy=ech_seed_policy)

                # Energetic proxies.
                E_ref_sum = _state_ref_sum(comps, ref_E)
                dE_proxy = float(E_state - float(E_slab_u) - E_ref_sum) if np.isfinite(E_state) and np.isfinite(E_ref_sum) else float("nan")
                dE_prox = float("nan")
                prox_note = ""
                if len(comps) > 1:
                    base_keys = [(site_label, f"{c}*") for c in comps]
                    base_rows = [per_site_state_energy.get(k) for k in base_keys]
                    if all(_row_can_supply_voc_energy(br) for br in base_rows):
                        base_sum = sum(float(br["E_state_user (eV)"]) for br in base_rows if isinstance(br, dict))
                        dE_prox = float(E_state + float(E_slab_u) - base_sum)
                        prox_note = "E(coads)+E(slab)-sum(E(QA-valid single_states))"
                    else:
                        prox_note = "base_single_state_missing_or_QA_invalid"

                try:
                    write(structure_path, relaxed)
                except Exception:
                    structure_path = Path("")

                energy_payload = _masked_energy_payload(
                    comps=comps,
                    target_key=clean_adsorbate_label(preset.get("target_adsorbate", "CH3CHO")),
                    qa_value=qa_meta.get("qa", ""),
                    dE_proxy_raw=dE_proxy,
                    dE_prox_raw=dE_prox,
                )
                row.update({
                    "E_state_user (eV)": float(E_state),
                    "E_ref_sum (eV)": float(E_ref_sum),
                    "ref_source": json.dumps({c: ref_src.get(c, "") for c in comps}),
                    **energy_payload,
                    "proximity_formula": prox_note,
                    "structure_cif": str(Path(structure_path).resolve()) if str(structure_path) else "",
                    "relax_n_steps": int(rmeta.get("n_steps", 0)),
                    "relax_converged": rmeta.get("converged", None),
                    "relax_elapsed_s": float(rmeta.get("elapsed_s", 0.0)),
                    "relax_error": str(rmeta.get("error", "")),
                    "migrated": bool(str(qa_meta.get("qa", "")).lower() == "migrated"),
                    **qa_meta,
                })
            except Exception as e:
                # If the initial structure was written before relaxation failed,
                # keep it linked for the UI.  No relaxed structure is claimed
                # unless an actual CIF exists at structure_path.
                try:
                    if row.get("initial_structure_cif", "") == "" and 'initial_structure_path' in locals() and Path(initial_structure_path).is_file():
                        row["initial_structure_cif"] = str(Path(initial_structure_path).resolve())
                except Exception:
                    pass
                try:
                    if 'structure_path' in locals() and Path(structure_path).is_file():
                        row["structure_cif"] = str(Path(structure_path).resolve())
                except Exception:
                    pass
                row.update({
                    "E_state_user (eV)": float("nan"),
                    "ΔE_proxy (eV)": float("nan"),
                    "ΔE_ads_user (eV)": float("nan"),
                    "ΔE_VOC_ads_proxy (eV)": float("nan"),
                    "ΔE_H_VOC_proximity_proxy (eV)": float("nan"),
                    "ΔE_OH_VOC_proximity_proxy (eV)": float("nan"),
                    "ΔE_proximity_proxy (eV)": float("nan"),
                    "ΔE_raw_proxy_diagnostic (eV)": float("nan"),
                    "ΔE_raw_proximity_diagnostic (eV)": float("nan"),
                    "descriptor_energy_valid": False,
                    "descriptor_energy_mask_note": "crashed",
                    "qa": "crashed",
                    "qa_note": str(e),
                    "relax_error": str(e),
                    "migrated": False,
                })

            rows.append(row)
            per_site_state_energy[state_key] = row

    # Hide dependency-only states only if the user did not request them and they were added for proximity.
    requested_norm = set(descriptor_states)
    out_rows = [r for r in rows if str(r.get("descriptor_state", "")) in requested_norm]
    df = pd.DataFrame(out_rows)
    if isinstance(df, pd.DataFrame) and df.empty and len(df.columns) == 0:
        df = pd.DataFrame(columns=[
            "mode", "target_voc", "target_voc_label", "descriptor_state", "adsorbate", "state_type",
            "site_label", "site", "site_kind", "oxide_site_class", "oxide_site_label", "oxide_site_elements",
            "oxide_site_policy", "site_taxonomy_warning", "seed_source", "qa", "qa_note", "reliability",
            "ΔE_proxy (eV)", "ΔE_ads_user (eV)", "ΔE_VOC_ads_proxy (eV)",
            "ΔE_H_VOC_proximity_proxy (eV)", "ΔE_OH_VOC_proximity_proxy (eV)", "ΔE_proximity_proxy (eV)",
            "descriptor_energy_valid", "descriptor_energy_mask_note", "proxy_warning",
            "site_support_elements", "site_support_formula", "anchor_slab_distance(Å)",
            "anchor_site_distance(Å)", "bound_geometry_valid", "geometry_qa_reason",
            "ech_seed_policy", "ech_seed_role", "ech_state", "ech_state_type", "ech_classification",
            "ech_c_carbonyl_h_distance_A", "ech_h_h_distance_A", "ech_qc_note"
        ])
    if isinstance(df, pd.DataFrame) and not df.empty:
        # Guard against pyarrow/Streamlit failures from accidental duplicate names.
        df = df.loc[:, ~df.columns.duplicated()].copy()
    out_csv = out_root / "results_sites_voc.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")

    meta = {
        "mode": "VOC",
        "target_voc": str(target_voc),
        "target_voc_label": str(preset.get("label", target_voc)),
        "descriptor_states": list(descriptor_states),
        "interpretation": str(preset.get("interpretation", "")),
        "warning": VOC_PROXY_WARNING,
        "user_slab": str(Path(user_slab_cif).resolve()),
        "relax_mode": str(relax_mode),
        "adsorbate_relax_steps": int(ads_steps),
        "adsorbate_relax_fmax": float(fmax),
        "voc_relaxation_policy": str(voc_relaxation_policy),
        "oxide_voc_site_policy": str(oxide_voc_site_policy),
        "E_slab": float(E_slab_u),
        "reference_energies": ref_E,
        "reference_sources": ref_src,
        "Model": MODEL_NAME,
        "Device": DEVICE,
        "MODEL": MODEL_NAME,
        "DEVICE": DEVICE,
        "warnings": {
            "slab_relax_drop": bool(meta_flags.get("slab_relax_drop", False)),
            "vac_warning": bool(meta_flags.get("vac_warning", False)),
        },
    }
    (out_root / "meta_voc.json").write_text(json.dumps(meta, indent=2))
    return str(out_csv), meta


def run_metal_voc_proxy(
    user_slab_cif: str,
    sites: Iterable[str] = ("fcc", "hcp", "bridge", "ontop"),
    relax_mode: str = "Normal",
    user_ads_sites: Optional[Mapping[str, object]] = None,
    target_voc: str = "acetaldehyde",
    descriptor_states: Iterable[str] | None = None,
    ref_dir: str | Path = "ref_gas",
    voc_relaxation_policy: str = "normal_relax",
    oxide_voc_site_policy: str = "standard_routed",
):
    return _run_voc_proxy(
        "metal",
        user_slab_cif,
        Path("runs_voc_metal") / str(int(time.time())),
        sites,
        vac_z=20.0,
        layers=3,
        relax_mode=relax_mode,
        user_ads_sites=user_ads_sites,
        target_voc=target_voc,
        descriptor_states=descriptor_states,
        ref_dir=ref_dir,
        voc_relaxation_policy=voc_relaxation_policy,
        oxide_voc_site_policy=oxide_voc_site_policy,
    )


def run_oxide_voc_proxy(
    user_slab_cif: str,
    sites: Iterable[str] = ("fcc", "bridge", "ontop"),
    relax_mode: str = "Normal",
    user_ads_sites: Optional[Mapping[str, object]] = None,
    target_voc: str = "acetaldehyde",
    descriptor_states: Iterable[str] | None = None,
    ref_dir: str | Path = "ref_gas",
    voc_relaxation_policy: str = "normal_relax",
    oxide_voc_site_policy: str = "standard_routed",
):
    return _run_voc_proxy(
        "oxide",
        user_slab_cif,
        Path("runs_voc_oxide") / str(int(time.time())),
        sites,
        vac_z=20.0,
        layers=3,
        relax_mode=relax_mode,
        user_ads_sites=user_ads_sites,
        target_voc=target_voc,
        descriptor_states=descriptor_states,
        ref_dir=ref_dir,
        voc_relaxation_policy=voc_relaxation_policy,
        oxide_voc_site_policy=oxide_voc_site_policy,
    )
