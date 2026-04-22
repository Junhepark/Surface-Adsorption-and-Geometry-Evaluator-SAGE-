
from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry import find_mic
from ase.io import write

from ocp_app.core.ads_sites import (
    ANION_SYMBOLS,
    detect_oxide_surface_sites,
    select_representative_sites,
)
from ocp_app.core.anchors.common import (
    H0S,
    MIGRATE_THR,
    calc,
    ensure_pbc3,
    relax_zonly,
    relax_freeH,
    normalize_relaxation_scope,
    site_energy_two_stage,
    site_energy_oh_constrained,
    site_energy_oh_anchoronly, 
)

THREE_STAGE_OXIDE_HER_CAUTION = (
    "Caution: In the current oxide workflow, D1 is treated as a contextual O-top protonation probe, "
    "whereas D2 is the primary reactive-H screening output. D2 follows the main oxide HER (including rigid-initialization handling and seed-height selection) "
    "relaxation setting selected upstream, so changes in relaxation freedom can shift the final "
    "screening value. Both values should be interpreted as screening indicators rather than "
    "quantitatively validated absolute thermodynamic metrics."
)

DESCRIPTOR_D1_DISP_THRESH_A = 1.60
DESCRIPTOR_D2_DISP_THRESH_A = 1.20
DESCRIPTOR_ABS_DE_THRESH_EV = 3.0

D1_CLEAN_HOOK_K = 18.0
D1_PREOH_HOOK_K = 12.0
D1_HOOK_EXTRA_A = 0.12
D1_SHELL_MARGIN_A = 0.20
D1_ANCHOR_MAX_Z_LIFT_A = 0.90
D1_ANCHOR_WARN_Z_LIFT_A = 0.60
D1_ANCHOR_METAL_DIST_ABS_MAX_A = 2.80
D1_ANCHOR_METAL_DIST_RATIO_MAX = 1.35
D1_ANCHOR_METAL_DIST_BUFFER_A = 0.35
D1_RETAINED_NEIGHBORS_MIN = 1


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_scalar(x, default=np.nan) -> float:
    """Convert scalars / 1-element arrays / tuples to a float safely."""
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size == 0:
            return float(default)
        if arr.size == 1:
            return float(arr[0])
        # if a tuple/vector is passed accidentally, use its norm as a stable scalar
        return float(np.linalg.norm(arr))
    except Exception:
        return float(default)


def _series_or_default(df: pd.DataFrame, col: str, default) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _mic_xy_delta(cell, pbc, xy0: np.ndarray, xy1: np.ndarray) -> tuple[np.ndarray, float]:
    xy0 = np.asarray(xy0, dtype=float).reshape(2)
    xy1 = np.asarray(xy1, dtype=float).reshape(2)
    d3 = np.array([xy1[0] - xy0[0], xy1[1] - xy0[1], 0.0], dtype=float)
    vec, _ = find_mic(d3, cell=cell, pbc=[bool(pbc[0]), bool(pbc[1]), False])
    vec = np.asarray(vec, dtype=float)
    return vec[:2], float(np.linalg.norm(vec[:2]))


def _descriptor_h0_scalar(default: float = 1.2) -> float:
    try:
        arr = np.asarray(H0S, dtype=float).reshape(-1)
        if arr.size > 0 and np.isfinite(arr[0]):
            return float(arr[0])
    except Exception:
        pass
    return float(default)


def _safe_label_token(x: object) -> str:
    s = str(x or "pair")
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", s)
    return s[:120] if len(s) > 120 else s


def _nearest_non_h_info(atoms, h_index: int = -1) -> dict[str, object]:
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    if h_index < 0:
        h_index = len(atoms) + int(h_index)
    ref = np.asarray(pos[h_index], dtype=float)
    best = None
    best_key = None
    for i, sym in enumerate(syms):
        if i == h_index or str(sym).upper() == 'H':
            continue
        _dvec, dist = find_mic(np.asarray(pos[i], dtype=float) - ref, atoms.get_cell(), atoms.get_pbc())
        key = (float(dist), int(i))
        if best is None or key < best_key:
            best = {'index': int(i), 'symbol': str(sym), 'distance': float(dist)}
            best_key = key
    if best is None:
        return {'index': -1, 'symbol': 'unknown', 'distance': float('nan')}
    return best


def _classify_h_binding_state(atoms, h_index: int = -1) -> tuple[str, dict[str, object]]:
    near = _nearest_non_h_info(atoms, h_index=h_index)
    sym = str(near.get('symbol', 'unknown'))
    dist = float(near.get('distance', float('nan')))
    if sym in ANION_SYMBOLS and np.isfinite(dist) and dist <= 1.40:
        cls = 'o_bound'
    elif sym not in ANION_SYMBOLS and sym != 'unknown' and np.isfinite(dist) and dist <= 2.40:
        cls = 'metal_adjacent'
    elif np.isfinite(dist):
        cls = 'other'
    else:
        cls = 'unresolved'
    near['binding_class'] = cls
    return cls, near




def _metal_indices(atoms) -> np.ndarray:
    syms = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    return np.asarray([i for i, s in enumerate(syms) if str(s).upper() != 'H' and str(s) not in ANION_SYMBOLS], dtype=int)


def _nearest_metal_neighbors(atoms, anchor_index: int, max_neighbors: int = 2) -> list[dict[str, object]]:
    pos = atoms.get_positions()
    ref = np.asarray(pos[int(anchor_index)], dtype=float)
    rows = []
    for i in _metal_indices(atoms):
        _vec, dist = find_mic(np.asarray(pos[int(i)], dtype=float) - ref, atoms.get_cell(), atoms.get_pbc())
        rows.append({'index': int(i), 'symbol': str(atoms[int(i)].symbol), 'distance': float(dist)})
    rows.sort(key=lambda r: (r['distance'], r['index']))
    return rows[:max(1, int(max_neighbors))]


def _classify_anchor_oh_state(initial_atoms, relaxed_atoms, anchor_index: int, h_index: int = -1) -> dict[str, object]:
    initial_neighbors = _nearest_metal_neighbors(initial_atoms, anchor_index=anchor_index, max_neighbors=2)
    relaxed_neighbors = _nearest_metal_neighbors(relaxed_atoms, anchor_index=anchor_index, max_neighbors=4)

    pos0 = np.asarray(initial_atoms.get_positions()[int(anchor_index)], dtype=float)
    pos1 = np.asarray(relaxed_atoms.get_positions()[int(anchor_index)], dtype=float)
    z_lift = float(pos1[2] - pos0[2])

    retained = 0
    retained_ids = []
    for nn in initial_neighbors:
        ridx = int(nn['index'])
        r0 = float(nn['distance'])
        try:
            _vec, r1 = find_mic(
                np.asarray(relaxed_atoms.get_positions()[ridx], dtype=float) - pos1,
                relaxed_atoms.get_cell(),
                relaxed_atoms.get_pbc(),
            )
            r1 = float(r1)
        except Exception:
            r1 = float('nan')
        cutoff = max(D1_ANCHOR_METAL_DIST_ABS_MAX_A, r0 * D1_ANCHOR_METAL_DIST_RATIO_MAX, r0 + D1_ANCHOR_METAL_DIST_BUFFER_A)
        if np.isfinite(r1) and r1 <= cutoff:
            retained += 1
            retained_ids.append(ridx)

    nearest_relaxed = relaxed_neighbors[0] if relaxed_neighbors else {'index': -1, 'symbol': 'unknown', 'distance': float('nan')}
    h_near = _nearest_non_h_info(relaxed_atoms, h_index=h_index)

    flags = []
    valid = True
    binding_class = 'o_bound'

    nearest_metal_dist = float(nearest_relaxed.get('distance', float('nan')))
    if retained < int(D1_RETAINED_NEIGHBORS_MIN):
        valid = False
        binding_class = 'detached_oh'
        flags.append('detached_oh')
    elif np.isfinite(z_lift) and z_lift > float(D1_ANCHOR_MAX_Z_LIFT_A):
        valid = False
        binding_class = 'detached_oh'
        flags.append('anchor_lift_off')
    elif np.isfinite(z_lift) and z_lift > float(D1_ANCHOR_WARN_Z_LIFT_A):
        binding_class = 'lifted_but_bound'
        flags.append('lifted_but_bound')

    if not np.isfinite(nearest_metal_dist) or nearest_metal_dist > float(D1_ANCHOR_METAL_DIST_ABS_MAX_A):
        valid = False
        binding_class = 'detached_oh'
        if 'detached_oh' not in flags:
            flags.append('detached_oh')

    return {
        'descriptor_valid': bool(valid),
        'binding_class': str(binding_class),
        'qc_flags': ';'.join(flags) if flags else '',
        'anchor_z_lift(Å)': float(z_lift),
        'anchor_retained_m_neighbors': int(retained),
        'anchor_retained_neighbor_ids': ','.join(str(i) for i in retained_ids) if retained_ids else '',
        'anchor_nearest_metal_index': int(nearest_relaxed.get('index', -1)),
        'anchor_nearest_metal_symbol': str(nearest_relaxed.get('symbol', 'unknown')),
        'anchor_nearest_metal_distance(Å)': float(nearest_metal_dist),
        'nearest_symbol': str(h_near.get('symbol', 'unknown')),
        'nearest_index': int(h_near.get('index', -1)),
        'nearest_distance(Å)': float(h_near.get('distance', float('nan'))),
    }


def _rows_same_basin(d1_row: dict[str, object] | None, d2_row: dict[str, object] | None, slab_u_rel=None, xy_tol: float = 0.20) -> bool:
    if not isinstance(d1_row, dict) or not isinstance(d2_row, dict):
        return False
    x1 = _safe_float(d1_row.get('final_h_x(Å)'))
    y1 = _safe_float(d1_row.get('final_h_y(Å)'))
    x2 = _safe_float(d2_row.get('final_h_x(Å)'))
    y2 = _safe_float(d2_row.get('final_h_y(Å)'))
    if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
        return False
    if slab_u_rel is not None:
        _, dxy = _mic_xy_delta(slab_u_rel.get_cell(), slab_u_rel.get_pbc(), np.asarray([x1, y1]), np.asarray([x2, y2]))
    else:
        dxy = float(np.linalg.norm(np.asarray([x1 - x2, y1 - y2], dtype=float)))
    same_binding = str(d1_row.get('binding_class', '')).strip().lower() == str(d2_row.get('binding_class', '')).strip().lower()
    same_symbol = str(d1_row.get('nearest_symbol', '')).strip() == str(d2_row.get('nearest_symbol', '')).strip()
    idx1 = d1_row.get('nearest_index', None)
    idx2 = d2_row.get('nearest_index', None)
    same_index = (idx1 == idx2) and (idx1 is not None)
    return bool(np.isfinite(dxy) and dxy <= float(xy_tol) and same_binding and (same_index or same_symbol))


def _quality_rank(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    stable_s = _series_or_default(work, "site_transition_type", "unknown")
    work["_stable"] = stable_s.astype(str).str.strip().str.lower().eq("stable").astype(int)

    mismatch_s = _series_or_default(work, "placement_mismatch", False)
    if getattr(mismatch_s, "dtype", None) == bool:
        work["_mismatch"] = mismatch_s.fillna(False).astype(int)
    else:
        work["_mismatch"] = pd.to_numeric(mismatch_s, errors="coerce").fillna(0).astype(bool).astype(int)

    disp_s = _series_or_default(work, "H_lateral_disp(Å)", np.nan)
    work["_disp"] = pd.to_numeric(disp_s, errors="coerce")

    de_s = _series_or_default(work, "ΔE_H_user (eV)", np.nan)
    work["_abs_dE"] = np.abs(pd.to_numeric(de_s, errors="coerce"))

    e_col = "ΔG_H (eV)" if "ΔG_H (eV)" in work.columns else ("ΔE_H_user (eV)" if "ΔE_H_user (eV)" in work.columns else None)
    if e_col is None:
        work["_energy"] = np.nan
    else:
        work["_energy"] = pd.to_numeric(work[e_col], errors="coerce")

    return work.sort_values(
        ["_stable", "_mismatch", "_disp", "_abs_dE", "_energy"],
        ascending=[False, True, True, True, True],
        na_position="last",
    )


def _pick_descriptor_seed_row(
    df: pd.DataFrame,
    *,
    energy_col: str,
    disp_thresh: float,
) -> tuple[dict[str, object] | None, str, str]:
    if df is None or df.empty or energy_col not in df.columns:
        return None, "missing", "no candidates"

    work = df.copy()
    work["_energy"] = pd.to_numeric(work[energy_col], errors="coerce")
    disp_s = _series_or_default(work, "H_lateral_disp(Å)", np.nan)
    work["_disp"] = pd.to_numeric(disp_s, errors="coerce")
    de_s = _series_or_default(work, "ΔE_H_user (eV)", np.nan)
    work["_abs_dE"] = np.abs(pd.to_numeric(de_s, errors="coerce"))

    valid_s = _series_or_default(work, "descriptor_valid", True)
    if getattr(valid_s, "dtype", None) == bool:
        work["_valid"] = valid_s.fillna(False)
    else:
        work["_valid"] = valid_s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])

    flags_s = _series_or_default(work, "qc_flags", "")
    work["_flags"] = flags_s.astype(str)

    base = work[
        work["_energy"].notna()
        & work["_disp"].notna()
        & (work["_disp"] <= float(disp_thresh))
        & (work["_valid"])
        & (~work["_flags"].str.contains(r"detached_oh|surface_atom_extraction|water_like_fragment", case=False, na=False))
        & (
            work["_abs_dE"].isna()
            | (work["_abs_dE"] <= float(DESCRIPTOR_ABS_DE_THRESH_EV))
        )
    ]
    ranked = _quality_rank(base)
    if not ranked.empty:
        return ranked.iloc[0].to_dict(), "reliable", ""

    finite_valid = work[work["_energy"].notna() & work["_valid"]].copy()
    ranked = _quality_rank(finite_valid)
    if not ranked.empty:
        disp_val = _safe_float(ranked.iloc[0].get("H_lateral_disp(Å)"))
        warning = f"Using low-confidence fallback seed (disp={disp_val:.3f} Å)."
        return ranked.iloc[0].to_dict(), "fallback_unreliable", warning

    if work[work["_energy"].notna()].shape[0] > 0:
        return None, "missing", "No descriptor-valid candidates remained after D1 QC."

    return None, "missing", "descriptor seed selection failed"


def _build_reactive_h_targets_oxide(slab_u_rel, max_per_kind: int = 2) -> list[dict[str, object]]:
    try:
        auto_sites = detect_oxide_surface_sites(slab_u_rel, max_sites_per_kind=200, z_tol=1.2)
        rep_sites = select_representative_sites(auto_sites, per_kind=max(1, int(max_per_kind)))
    except Exception:
        rep_sites = []
    targets: list[dict[str, object]] = []
    for i, site in enumerate(rep_sites):
        pos = np.asarray(getattr(site, 'position', []), dtype=float)
        if pos.shape[0] < 2:
            continue
        xy = np.asarray(pos[:2], dtype=float)
        xyz = np.asarray(pos[:3], dtype=float) if pos.shape[0] >= 3 else np.asarray([xy[0], xy[1], np.nan], dtype=float)
        targets.append({
            'site_label': f"reactive_{getattr(site, 'kind', 'site')}_{i}",
            'site_kind': str(getattr(site, 'kind', 'unknown')),
            'xy': xy,
            'initial_xyz': xyz,
            'surface_indices': tuple(int(j) for j in (getattr(site, 'surface_indices', ()) or ())),
            'seed_source': 'oxide_reactive_geom',
        })
    return targets


def _evaluate_constrained_oh_descriptor(*, slab_u_rel, E_slab_u: float, E_H2: float, site_seed: dict[str, object],
                                      z_steps: int, free_steps: int, use_net_corr: bool,
                                      out_cif: Path | None = None) -> dict[str, object]:
    label = str(site_seed.get('site_label', 'unknown'))
    kind = str(site_seed.get('site_kind', 'unknown'))
    xy = np.asarray(site_seed.get('xy', [np.nan, np.nan]), dtype=float)
    sidx = tuple(int(i) for i in (site_seed.get('surface_indices') or ()))
    anchor_index = int(sidx[0]) if sidx else -1
    if anchor_index < 0:
        syms = np.asarray(slab_u_rel.get_chemical_symbols(), dtype=object)
        pos = slab_u_rel.get_positions()
        o_idx = np.where(syms == 'O')[0]
        if len(o_idx) == 0:
            raise ValueError('No oxygen atoms available for D1 anchor selection.')
        d2 = [float(np.dot(pos[int(i), :2] - xy, pos[int(i), :2] - xy)) for i in o_idx]
        anchor_index = int(o_idx[int(np.argmin(d2))])

    Au, E_uH, _disp_raw, relax_meta = site_energy_oh_anchoronly(
        slab_u_rel,
        xy,
        anchor_index=int(anchor_index),
        h0s=H0S,
        z_steps=int(z_steps),
        free_steps=int(free_steps),
        return_meta=True,
        shell_margin=D1_SHELL_MARGIN_A,
        hook_k=D1_CLEAN_HOOK_K,
        hook_extra=D1_HOOK_EXTRA_A,
    )
    dE_u = float(E_uH) - float(E_slab_u) - 0.5 * float(E_H2)
    dG_u = float(dE_u + (0.24 if use_net_corr else 0.0))

    h_pos_final = np.asarray(Au.get_positions()[-1], dtype=float)
    _, disp_mic = _mic_xy_delta(slab_u_rel.get_cell(), slab_u_rel.get_pbc(), xy, h_pos_final[:2])

    qc = _classify_anchor_oh_state(slab_u_rel, Au, anchor_index=int(anchor_index), h_index=len(Au)-1)
    if out_cif is not None:
        out_cif.parent.mkdir(parents=True, exist_ok=True)
        write(out_cif, Au)

    row = {
        'site_label': label,
        'site_kind': kind,
        'seed_x': float(xy[0]),
        'seed_y': float(xy[1]),
        'ΔE (eV)': float(dE_u),
        'ΔG (eV)': float(dG_u),
        'H_lateral_disp(Å)': float(disp_mic),
        'binding_class': str(qc.get('binding_class', 'unresolved')),
        'descriptor_valid': bool(qc.get('descriptor_valid', False)),
        'qc_flags': str(qc.get('qc_flags', '')),
        'nearest_symbol': str(qc.get('nearest_symbol', 'unknown')),
        'nearest_index': int(qc.get('nearest_index', anchor_index)),
        'nearest_distance(Å)': float(qc.get('nearest_distance(Å)', float('nan'))),
        'anchor_z_lift(Å)': float(qc.get('anchor_z_lift(Å)', float('nan'))),
        'anchor_retained_m_neighbors': int(qc.get('anchor_retained_m_neighbors', 0)),
        'anchor_retained_neighbor_ids': str(qc.get('anchor_retained_neighbor_ids', '')),
        'anchor_nearest_metal_index': int(qc.get('anchor_nearest_metal_index', -1)),
        'anchor_nearest_metal_symbol': str(qc.get('anchor_nearest_metal_symbol', 'unknown')),
        'anchor_nearest_metal_distance(Å)': float(qc.get('anchor_nearest_metal_distance(Å)', float('nan'))),
        'final_h_x(Å)': float(h_pos_final[0]),
        'final_h_y(Å)': float(h_pos_final[1]),
        'final_h_z(Å)': float(h_pos_final[2]),
        'site_transition_type': (
            'stable' if bool(qc.get('descriptor_valid', False)) and np.isfinite(disp_mic) and disp_mic <= MIGRATE_THR
            else ('lifted_but_bound' if str(qc.get('binding_class', '')) == 'lifted_but_bound' else 'detached_oh')
        ),
        'placement_mismatch': False,
        'migrated': bool(np.isfinite(disp_mic) and disp_mic > MIGRATE_THR),
        'structure_cif': str(out_cif.resolve()) if out_cif is not None and out_cif.exists() else '',
        'relaxation_scope': 'anchor_oh_hookean',
        'n_fix_layers': 0,
        'selected_h0': _safe_float((relax_meta or {}).get('selected_h0')),
        'z_relax_n_steps': int((relax_meta or {}).get('z_relax_n_steps', 0)),
        'z_relax_converged': (relax_meta or {}).get('z_relax_converged', None),
        'z_relax_relaxed_atoms': int((relax_meta or {}).get('z_relax_relaxed_atoms', 0)),
        'fine_relax_n_steps': int((relax_meta or {}).get('fine_relax_n_steps', 0)),
        'fine_relax_converged': (relax_meta or {}).get('fine_relax_converged', None),
        'fine_relax_relaxed_atoms': int((relax_meta or {}).get('fine_relax_relaxed_atoms', 0)),
        'total_relax_n_steps': int((relax_meta or {}).get('total_relax_n_steps', 0)),
        'local_free_count': int((relax_meta or {}).get('local_free_count', 0)),
        'shell_count': int((relax_meta or {}).get('shell_count', 0)),
        'hook_k': _safe_float((relax_meta or {}).get('hook_k', D1_CLEAN_HOOK_K)),
        'hook_extra': _safe_float((relax_meta or {}).get('hook_extra', D1_HOOK_EXTRA_A)),
    }
    return row


def _resolve_anchor_index_from_seed(slab_u_rel, site_seed: dict[str, object]) -> int:
    xy = np.asarray(site_seed.get('xy', [np.nan, np.nan]), dtype=float)
    sidx = tuple(int(i) for i in (site_seed.get('surface_indices') or ()))
    anchor_index = int(sidx[0]) if sidx else -1
    if anchor_index >= 0:
        return int(anchor_index)
    syms = np.asarray(slab_u_rel.get_chemical_symbols(), dtype=object)
    pos = slab_u_rel.get_positions()
    o_idx = np.where(syms == 'O')[0]
    if len(o_idx) == 0:
        raise ValueError('No oxygen atoms available for D1 anchor selection.')
    d2 = [float(np.dot(pos[int(i), :2] - xy, pos[int(i), :2] - xy)) for i in o_idx]
    return int(o_idx[int(np.argmin(d2))])


def _select_background_oh_site(anchor_seed: dict[str, object], d1_targets: list[dict[str, object]], slab_u_rel, min_xy_sep: float = 1.2):
    if not d1_targets:
        return None
    axy = np.asarray(anchor_seed.get('xy', [np.nan, np.nan]), dtype=float).reshape(2)
    alabel = str(anchor_seed.get('site_label', ''))
    best = None
    best_key = None
    for cand in d1_targets:
        clabel = str(cand.get('site_label', ''))
        if clabel == alabel:
            continue
        cxy = np.asarray(cand.get('xy', [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(cxy).all():
            continue
        _, dxy = _mic_xy_delta(slab_u_rel.get_cell(), slab_u_rel.get_pbc(), axy, cxy)
        if np.isfinite(dxy) and dxy < float(min_xy_sep):
            continue
        key = (-float(dxy), clabel)
        if best is None or key < best_key:
            best = cand
            best_key = key
    if best is not None:
        return best
    # fallback: farthest even if close
    for cand in d1_targets:
        clabel = str(cand.get('site_label', ''))
        if clabel == alabel:
            continue
        cxy = np.asarray(cand.get('xy', [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(cxy).all():
            continue
        _, dxy = _mic_xy_delta(slab_u_rel.get_cell(), slab_u_rel.get_pbc(), axy, cxy)
        key = (-float(dxy), clabel)
        if best is None or key < best_key:
            best = cand
            best_key = key
    return best


def _evaluate_prehydroxylated_oh_descriptor(*, slab_u_rel, E_slab_u: float, E_H2: float, site_seed: dict[str, object],
                                           d1_targets: list[dict[str, object]], z_steps: int, free_steps: int, use_net_corr: bool,
                                           out_cif: Path | None = None, bg_out_cif: Path | None = None) -> dict[str, object]:
    label = str(site_seed.get('site_label', 'unknown'))
    kind = str(site_seed.get('site_kind', 'unknown'))
    xy = np.asarray(site_seed.get('xy', [np.nan, np.nan]), dtype=float)
    anchor_index = _resolve_anchor_index_from_seed(slab_u_rel, site_seed)

    bg_seed = _select_background_oh_site(site_seed, d1_targets, slab_u_rel, min_xy_sep=1.2)
    if bg_seed is None:
        row = _evaluate_constrained_oh_descriptor(
            slab_u_rel=slab_u_rel,
            E_slab_u=E_slab_u,
            E_H2=E_H2,
            site_seed=site_seed,
            z_steps=z_steps, free_steps=free_steps,
            use_net_corr=use_net_corr,
            out_cif=out_cif,
        )
        row['D1_model'] = 'clean_fallback'
        row['background_site_label'] = 'NA'
        row['background_structure_cif'] = ''
        row['background_total_relax_n_steps'] = 0
        return row

    bg_xy = np.asarray(bg_seed.get('xy', [np.nan, np.nan]), dtype=float)
    bg_index = _resolve_anchor_index_from_seed(slab_u_rel, bg_seed)

    A_bg, E_bg, _bg_disp, bg_meta = site_energy_oh_anchoronly(
        slab_u_rel,
        bg_xy,
        anchor_index=int(bg_index),
        h0s=H0S,
        z_steps=int(z_steps),
        free_steps=int(free_steps),
        return_meta=True,
        shell_margin=D1_SHELL_MARGIN_A,
        hook_k=D1_PREOH_HOOK_K,
        hook_extra=D1_HOOK_EXTRA_A,
    )
    if bg_out_cif is not None:
        bg_out_cif.parent.mkdir(parents=True, exist_ok=True)
        write(bg_out_cif, A_bg)

    Au, E_uH, _disp_raw, relax_meta = site_energy_oh_anchoronly(
        A_bg,
        xy,
        anchor_index=int(anchor_index),
        h0s=H0S,
        z_steps=int(z_steps),
        free_steps=int(free_steps),
        return_meta=True,
        shell_margin=D1_SHELL_MARGIN_A,
        hook_k=D1_PREOH_HOOK_K,
        hook_extra=D1_HOOK_EXTRA_A,
    )
    dE_u = float(E_uH) - float(E_bg) - 0.5 * float(E_H2)
    dG_u = float(dE_u + (0.24 if use_net_corr else 0.0))

    h_pos_final = np.asarray(Au.get_positions()[-1], dtype=float)
    _, disp_mic = _mic_xy_delta(A_bg.get_cell(), A_bg.get_pbc(), xy, h_pos_final[:2])

    qc = _classify_anchor_oh_state(A_bg, Au, anchor_index=int(anchor_index), h_index=len(Au)-1)
    if out_cif is not None:
        out_cif.parent.mkdir(parents=True, exist_ok=True)
        write(out_cif, Au)

    row = {
        'site_label': label,
        'site_kind': kind,
        'seed_x': float(xy[0]),
        'seed_y': float(xy[1]),
        'ΔE (eV)': float(dE_u),
        'ΔG (eV)': float(dG_u),
        'H_lateral_disp(Å)': float(disp_mic),
        'binding_class': str(qc.get('binding_class', 'unresolved')),
        'descriptor_valid': bool(qc.get('descriptor_valid', False)),
        'qc_flags': str(qc.get('qc_flags', '')),
        'nearest_symbol': str(qc.get('nearest_symbol', 'unknown')),
        'nearest_index': int(qc.get('nearest_index', anchor_index)),
        'nearest_distance(Å)': float(qc.get('nearest_distance(Å)', float('nan'))),
        'anchor_z_lift(Å)': float(qc.get('anchor_z_lift(Å)', float('nan'))),
        'anchor_retained_m_neighbors': int(qc.get('anchor_retained_m_neighbors', 0)),
        'anchor_retained_neighbor_ids': str(qc.get('anchor_retained_neighbor_ids', '')),
        'anchor_nearest_metal_index': int(qc.get('anchor_nearest_metal_index', -1)),
        'anchor_nearest_metal_symbol': str(qc.get('anchor_nearest_metal_symbol', 'unknown')),
        'anchor_nearest_metal_distance(Å)': float(qc.get('anchor_nearest_metal_distance(Å)', float('nan'))),
        'final_h_x(Å)': float(h_pos_final[0]),
        'final_h_y(Å)': float(h_pos_final[1]),
        'final_h_z(Å)': float(h_pos_final[2]),
        'site_transition_type': (
            'stable' if bool(qc.get('descriptor_valid', False)) and np.isfinite(disp_mic) and disp_mic <= MIGRATE_THR
            else ('lifted_but_bound' if str(qc.get('binding_class', '')) == 'lifted_but_bound' else 'detached_oh')
        ),
        'placement_mismatch': False,
        'migrated': bool(np.isfinite(disp_mic) and disp_mic > MIGRATE_THR),
        'structure_cif': str(out_cif.resolve()) if out_cif is not None and out_cif.exists() else '',
        'relaxation_scope': 'anchor_oh_hookean_preOH',
        'n_fix_layers': 0,
        'selected_h0': _safe_float((relax_meta or {}).get('selected_h0')),
        'z_relax_n_steps': int((relax_meta or {}).get('z_relax_n_steps', 0)),
        'z_relax_converged': (relax_meta or {}).get('z_relax_converged', None),
        'z_relax_relaxed_atoms': int((relax_meta or {}).get('z_relax_relaxed_atoms', 0)),
        'fine_relax_n_steps': int((relax_meta or {}).get('fine_relax_n_steps', 0)),
        'fine_relax_converged': (relax_meta or {}).get('fine_relax_converged', None),
        'fine_relax_relaxed_atoms': int((relax_meta or {}).get('fine_relax_relaxed_atoms', 0)),
        'total_relax_n_steps': int((relax_meta or {}).get('total_relax_n_steps', 0)),
        'local_free_count': int((relax_meta or {}).get('local_free_count', 0)),
        'shell_count': int((relax_meta or {}).get('shell_count', 0)),
        'hook_k': _safe_float((relax_meta or {}).get('hook_k', D1_PREOH_HOOK_K)),
        'hook_extra': _safe_float((relax_meta or {}).get('hook_extra', D1_HOOK_EXTRA_A)),
        'background_site_label': str(bg_seed.get('site_label', 'NA')),
        'background_structure_cif': str(bg_out_cif.resolve()) if bg_out_cif is not None and bg_out_cif.exists() else '',
        'background_total_relax_n_steps': int((bg_meta or {}).get('total_relax_n_steps', 0)),
        'D1_model': 'prehydroxylated',
    }
    return row


def _evaluate_single_h_descriptor(*, slab_u_rel, E_slab_u: float, E_H2: float, site_seed: dict[str, object],
                                  z_steps: int, free_steps: int, use_net_corr: bool,
                                  out_cif: Path | None = None,
                                  relaxation_scope: str = "partial",
                                  n_fix_layers: int = 2) -> dict[str, object]:
    label = str(site_seed.get('site_label', 'unknown'))
    kind = str(site_seed.get('site_kind', 'unknown'))
    xy = np.asarray(site_seed.get('xy', [np.nan, np.nan]), dtype=float)

    Au, E_uH, _disp_raw, relax_meta = site_energy_two_stage(
        slab_u_rel,
        xy,
        H0S,
        int(z_steps),
        int(free_steps),
        relaxation_scope=normalize_relaxation_scope(relaxation_scope),
        n_fix_layers=int(n_fix_layers),
        return_meta=True,
    )
    dE_u = float(E_uH) - float(E_slab_u) - 0.5 * float(E_H2)
    dG_u = float(dE_u + (0.24 if use_net_corr else 0.0))

    h_pos_final = np.asarray(Au.get_positions()[-1], dtype=float)
    _, disp_mic = _mic_xy_delta(slab_u_rel.get_cell(), slab_u_rel.get_pbc(), xy, h_pos_final[:2])

    binding_class, near = _classify_h_binding_state(Au, h_index=len(Au)-1)
    if out_cif is not None:
        out_cif.parent.mkdir(parents=True, exist_ok=True)
        write(out_cif, Au)

    row = {
        'site_label': label,
        'site_kind': kind,
        'seed_x': float(xy[0]),
        'seed_y': float(xy[1]),
        'ΔE (eV)': float(dE_u),
        'ΔG (eV)': float(dG_u),
        'H_lateral_disp(Å)': float(disp_mic),
        'binding_class': str(binding_class),
        'nearest_symbol': str(near.get('symbol', 'unknown')),
        'nearest_index': int(near.get('index', -1)),
        'nearest_distance(Å)': float(near.get('distance', float('nan'))),
        'final_h_x(Å)': float(h_pos_final[0]),
        'final_h_y(Å)': float(h_pos_final[1]),
        'final_h_z(Å)': float(h_pos_final[2]),
        'site_transition_type': 'stable' if np.isfinite(disp_mic) and disp_mic <= MIGRATE_THR else 'displaced_same_site',
        'placement_mismatch': False,
        'migrated': bool(np.isfinite(disp_mic) and disp_mic > MIGRATE_THR),
        'structure_cif': str(out_cif.resolve()) if out_cif is not None and out_cif.exists() else '',
        'relaxation_scope': str(normalize_relaxation_scope(relaxation_scope)),
        'n_fix_layers': int(n_fix_layers),
        'selected_h0': _safe_float((relax_meta or {}).get('selected_h0')),
        'z_relax_n_steps': int((relax_meta or {}).get('z_relax_n_steps', 0)),
        'z_relax_converged': (relax_meta or {}).get('z_relax_converged', None),
        'z_relax_relaxed_atoms': int((relax_meta or {}).get('z_relax_relaxed_atoms', 0)),
        'fine_relax_n_steps': int((relax_meta or {}).get('fine_relax_n_steps', 0)),
        'fine_relax_converged': (relax_meta or {}).get('fine_relax_converged', None),
        'fine_relax_relaxed_atoms': int((relax_meta or {}).get('fine_relax_relaxed_atoms', 0)),
        'total_relax_n_steps': int((relax_meta or {}).get('total_relax_n_steps', 0)),
    }
    return row


def _build_two_h_initial(
    slab_u_rel,
    xy1: np.ndarray,
    xy2: np.ndarray,
    z_offset: float | None = None,
):
    A = slab_u_rel.copy()
    z_top = float(np.max(A.get_positions()[:, 2]))
    h0 = _descriptor_h0_scalar() if z_offset is None else float(z_offset)
    A += Atoms(symbols=['H', 'H'], positions=[
        [float(xy1[0]), float(xy1[1]), z_top + h0],
        [float(xy2[0]), float(xy2[1]), z_top + h0],
    ])
    return ensure_pbc3(A, vac_z=20.0)


def _relax_two_h_pair(
    slab_u_rel,
    xy1: np.ndarray,
    xy2: np.ndarray,
    z_steps: int,
    free_steps: int,
    relaxation_scope: str = "partial",
    n_fix_layers: int = 2,
):
    A0 = _build_two_h_initial(slab_u_rel, xy1, xy2, z_offset=None)
    Az, _, z_meta = relax_zonly(A0, steps=int(z_steps), fmax=0.05, return_meta=True)
    Af, E, free_meta = relax_freeH(
        Az,
        steps=int(free_steps),
        fmax=0.03,
        relaxation_scope=normalize_relaxation_scope(relaxation_scope),
        n_fix_layers=int(n_fix_layers),
        return_meta=True,
    )
    meta = {
        'relaxation_scope': str(normalize_relaxation_scope(relaxation_scope)),
        'n_fix_layers': int(n_fix_layers),
        'z_relax_n_steps': int((z_meta or {}).get('n_steps', 0)),
        'z_relax_converged': (z_meta or {}).get('converged', None),
        'z_relax_relaxed_atoms': int((z_meta or {}).get('relaxed_atom_count', 0)),
        'fine_relax_n_steps': int((free_meta or {}).get('n_steps', 0)),
        'fine_relax_converged': (free_meta or {}).get('converged', None),
        'fine_relax_relaxed_atoms': int((free_meta or {}).get('relaxed_atom_count', 0)),
        'total_relax_n_steps': int((z_meta or {}).get('n_steps', 0)) + int((free_meta or {}).get('n_steps', 0)),
    }
    return Af, float(E), meta


def _pair_hh_distance(atoms, i: int, j: int) -> float:
    pos = atoms.get_positions()
    _dvec, dist = find_mic(np.asarray(pos[j], dtype=float) - np.asarray(pos[i], dtype=float), atoms.get_cell(), atoms.get_pbc())
    return float(dist)


def _build_pair_seed_records(best_d2_row: dict[str, object] | None = None, d2_targets: list[dict[str, object]] | None = None,
                             d1_targets: list[dict[str, object]] | None = None, max_pairs: int = 6, **kwargs) -> list[dict[str, object]]:
    # backward-compatible alias
    if best_d2_row is None and "d2_best_row" in kwargs:
        best_d2_row = kwargs.get("d2_best_row")
    if d2_targets is None:
        d2_targets = []
    if d1_targets is None:
        d1_targets = []
    if best_d2_row is None:
        return []
    x0 = _safe_float(best_d2_row.get('seed_x'))
    y0 = _safe_float(best_d2_row.get('seed_y'))
    if not np.isfinite(x0) or not np.isfinite(y0):
        return []
    seeds: list[dict[str, object]] = []
    xy0 = np.asarray([x0, y0], dtype=float)
    for t in d2_targets:
        if str(t.get('site_label')) == str(best_d2_row.get('site_label')):
            continue
        xy2 = np.asarray(t.get('xy', [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(xy2).all():
            continue
        dxy = float(np.linalg.norm(xy0 - xy2))
        if dxy < 0.5:
            continue
        seeds.append({'pair_kind': 'reactive+reactive', 'pair_label': f"{best_d2_row.get('site_label')}+{t.get('site_label')}", 'xy1': xy0.copy(), 'xy2': xy2, 'pair_seed_distance(Å)': dxy})
    for t in d1_targets:
        xy2 = np.asarray(t.get('xy', [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(xy2).all():
            continue
        dxy = float(np.linalg.norm(xy0 - xy2))
        if dxy < 0.5:
            continue
        seeds.append({'pair_kind': 'reactive+Otop', 'pair_label': f"{best_d2_row.get('site_label')}+{t.get('site_label')}", 'xy1': xy0.copy(), 'xy2': xy2, 'pair_seed_distance(Å)': dxy})
    seeds.sort(key=lambda r: (0 if str(r.get('pair_kind')).startswith('reactive+reactive') else 1, _to_scalar(r.get('pair_seed_distance(Å)'), np.inf), str(r.get('pair_label'))))
    dedup = []
    seen = set()
    for rec in seeds:
        xy2 = np.asarray(rec['xy2'], dtype=float).reshape(2)
        key = (rec['pair_kind'], round(float(xy2[0]), 2), round(float(xy2[1]), 2))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(rec)
        if len(dedup) >= max(1, int(max_pairs)):
            break
    return dedup


def _classify_summary(summary: dict[str, object], mode: str) -> str:
    d1 = _safe_float(summary.get('D1_OH (eV)'))
    d2 = _safe_float(summary.get('D2_Hreact (eV)'))
    d3 = _safe_float(summary.get('D3_pair_proxy (eV)'))
    h2_like = bool(summary.get('D3_H2_like_motif', False))
    d3_status = str(summary.get('D3_status', '')).strip().lower()

    if mode == 'D1_OH only (O-top protonation)':
        if not np.isfinite(d1):
            return 'unresolved D1 probe'
        if d1 > 0.3:
            return 'weak O-top protonation tendency'
        if d1 < -1.0:
            return 'strong O-top OH tendency'
        return 'moderate O-top protonation tendency'
    if mode == 'D2_Hreact only (reactive H state)':
        if not np.isfinite(d2):
            return 'no reactive-H screening state found'
        if abs(d2) <= 0.5:
            return 'promising reactive-H regime'
        if d2 < -0.5:
            return 'strong reactive-H stabilization'
        return 'weak reactive-H stabilization'
    if mode == 'D3_pair only (H2 pairing proxy)':
        if d3_status in {'no_pair_seeds', 'no_valid_pair', 'all_candidates_failed', 'skip_same_basin', 'd2_unavailable'}:
            return 'no pairing proxy resolved'
        if not np.isfinite(d3):
            return 'no pairing proxy resolved'
        if h2_like and abs(d3) <= 0.5:
            return 'promising pairing proxy'
        if h2_like and d3 < -0.5:
            return 'paired-H surface trapping'
        if (not h2_like) and d3 < -0.5:
            return 'non-H2-like paired trapping'
        return 'pairing proxy / user review needed'

    # Full 2-stage profile: D2 is the primary screening output; D1 is contextual.
    if np.isfinite(d2):
        if bool(summary.get('D2_same_basin_as_D1', False)):
            return 'D2 reactive-H state overlaps the D1 OH basin'
        if abs(d2) <= 0.5:
            base = 'promising reactive-H regime'
        elif d2 < -0.5:
            base = 'strong reactive-H stabilization'
        else:
            base = 'weak reactive-H stabilization'
        if np.isfinite(d1) and d1 > 0.3:
            return f'{base} | weak O-top protonation tendency'
        if np.isfinite(d1) and d1 < -1.0:
            return f'{base} | strong O-top OH tendency'
        if np.isfinite(d1):
            return f'{base} | moderate O-top protonation tendency'
        return base

    if np.isfinite(d1):
        if d1 > 0.3:
            return 'D2 unresolved | weak O-top protonation tendency'
        if d1 < -1.0:
            return 'D2 unresolved | strong O-top OH tendency'
        return 'D2 unresolved | moderate O-top protonation tendency'
    return 'unresolved oxide descriptor profile'


def run_oxide_descriptor_profile(*, slab_u_rel, E_slab_u: float, E_H2: float,
                                 d1_rows_df: pd.DataFrame, d1_targets: list[dict[str, object]],
                                 out_root: Path, z_steps: int, free_steps: int,
                                 use_net_corr: bool, descriptor_mode: str = 'Full 2-stage profile (recommended)',
                                 max_reactive_per_kind: int = 2, pair_limit: int = 6,
                                 relaxation_scope: str = 'rigid', n_fix_layers: int = 2) -> dict[str, object]:
    stage_dir = Path(out_root) / 'three_stage'
    stage_dir.mkdir(parents=True, exist_ok=True)

    mode = str(descriptor_mode)
    if mode not in {'D2_Hreact only (reactive H state)', 'Full 2-stage profile (recommended)'}:
        mode = 'Full 2-stage profile (recommended)'
    resolved_scope = normalize_relaxation_scope(relaxation_scope)
    summary: dict[str, object] = {
        'descriptor_mode': mode,
        'caution': THREE_STAGE_OXIDE_HER_CAUTION,
        'relaxation_scope': resolved_scope,
        'n_fix_layers': int(n_fix_layers),
    }

    defaults = {
        'D1_OH (eV)': np.nan, 'D1_site_label': 'NA', 'D1_structure_cif': '',
        'D1_seed_quality': 'missing', 'D1_seed_warning': '', 'D1_seed_disp(Å)': np.nan,
        'D1_binding_class': 'NA', 'D1_relaxation_scope': '', 'D1_total_relax_n_steps': 0, 'D1_fine_relax_relaxed_atoms': 0,
        'D1_clean_OH (eV)': np.nan, 'D1_preOH_OH (eV)': np.nan, 'ΔD1_preOH-clean (eV)': np.nan,
        'D1_background_site_label': 'NA', 'D1_background_structure_cif': '', 'D1_model': 'NA',
        'D2_Hreact (eV)': np.nan, 'D2_site_label': 'NA', 'D2_binding_class': 'NA',
        'D2_structure_cif': '', 'D2_seed_quality': 'missing', 'D2_seed_warning': '', 'D2_seed_disp(Å)': np.nan,
        'D2_relaxation_scope': '', 'D2_total_relax_n_steps': 0, 'D2_fine_relax_relaxed_atoms': 0,
        'D2_same_basin_as_D1': False, 'D2_basin_note': '',
        'D2_candidates_csv': '',
        'D3_pair_proxy (eV)': np.nan, 'D3_pair_label': 'NA',
        'D3_H2_like_motif': False, 'D3_final_HH_distance(Å)': np.nan, 'D3_structure_cif': '',
        'D3_relaxation_scope': '', 'D3_total_relax_n_steps': 0, 'D3_fine_relax_relaxed_atoms': 0,
        'D3_candidates_csv': '', 'D3_status': 'disabled_by_design', 'D3_pair_seed_count': 0,
        'D3_valid_pair_count': 0, 'error': '',
    }
    summary.update(defaults)

    # Legacy D3 modes are intentionally disabled in the current app build.
    if mode == 'Full 3-stage profile (experimental)':
        mode = 'Full 2-stage profile (recommended)'
        summary['descriptor_mode'] = mode
    if mode == 'D3_pair only (H2 pairing proxy)':
        mode = 'D2_Hreact only (reactive H state)'
        summary['descriptor_mode'] = mode
        summary['caution'] = (str(summary.get('caution', '')).strip() + ' D3 is disabled in the user-facing app build.').strip()

    need_d1 = mode in {'D1_OH only (O-top protonation)', 'Full 2-stage profile (recommended)'}
    need_d2 = mode in {'D2_Hreact only (reactive H state)', 'Full 2-stage profile (recommended)'}
    need_d3 = False

    d1_best = None
    d2_best = None
    d2_targets: list[dict[str, object]] = []
    d1_scope = 'anchor_oh_hookean_preOH'
    d2_scope = 'partial' if resolved_scope == 'rigid' else resolved_scope
    d3_scope = d2_scope

    if need_d1:
        try:
            d1_clean_rows = []
            d1_pre_rows = []
            for i, site_seed in enumerate(d1_targets or []):
                clean_cif = stage_dir / f"D1clean_{i}_{_safe_label_token(site_seed.get('site_label','site'))}.cif"
                preoh_bg_cif = stage_dir / f"D1bg_{i}_{_safe_label_token(site_seed.get('site_label','site'))}.cif"
                preoh_cif = stage_dir / f"D1_{i}_{_safe_label_token(site_seed.get('site_label','site'))}.cif"
                clean_row = _evaluate_constrained_oh_descriptor(
                    slab_u_rel=slab_u_rel,
                    E_slab_u=E_slab_u,
                    E_H2=E_H2,
                    site_seed=site_seed,
                    z_steps=z_steps, free_steps=free_steps,
                    use_net_corr=use_net_corr,
                    out_cif=clean_cif,
                )
                clean_row['stage'] = 'D1_clean'
                d1_clean_rows.append(clean_row)

                pre_row = _evaluate_prehydroxylated_oh_descriptor(
                    slab_u_rel=slab_u_rel,
                    E_slab_u=E_slab_u,
                    E_H2=E_H2,
                    site_seed=site_seed,
                    d1_targets=d1_targets or [],
                    z_steps=z_steps, free_steps=free_steps,
                    use_net_corr=use_net_corr,
                    out_cif=preoh_cif,
                    bg_out_cif=preoh_bg_cif,
                )
                pre_row['stage'] = 'D1_preOH'
                d1_pre_rows.append(pre_row)

            d1_clean_df = pd.DataFrame(d1_clean_rows)
            d1_pre_df = pd.DataFrame(d1_pre_rows)
            d1_clean_csv = stage_dir / 'D1_clean_candidates.csv'
            d1_csv = stage_dir / 'D1_candidates.csv'
            clean_best = None
            pre_best = None
            if not d1_clean_df.empty:
                d1_clean_df.to_csv(d1_clean_csv, index=False)
                clean_best, q_clean, w_clean = _pick_descriptor_seed_row(d1_clean_df, energy_col='ΔG (eV)', disp_thresh=DESCRIPTOR_D1_DISP_THRESH_A)
                if clean_best is not None:
                    summary['D1_clean_OH (eV)'] = _safe_float(clean_best.get('ΔG (eV)'))
            if not d1_pre_df.empty:
                d1_pre_df.to_csv(d1_csv, index=False)
                summary['D1_candidates_csv'] = str(d1_csv.resolve())
                pre_best, q_pre, w_pre = _pick_descriptor_seed_row(d1_pre_df, energy_col='ΔG (eV)', disp_thresh=DESCRIPTOR_D1_DISP_THRESH_A)
                summary['D1_seed_quality'] = q_pre
                summary['D1_seed_warning'] = w_pre
                if pre_best is not None:
                    summary['D1_preOH_OH (eV)'] = _safe_float(pre_best.get('ΔG (eV)'))
                    summary['D1_background_site_label'] = str(pre_best.get('background_site_label', 'NA'))
                    summary['D1_background_structure_cif'] = str(pre_best.get('background_structure_cif', ''))
                    summary['D1_model'] = str(pre_best.get('D1_model', 'prehydroxylated'))
            d1_best = None
            if isinstance(pre_best, dict) and bool(pre_best.get('descriptor_valid', True)):
                d1_best = pre_best
            elif isinstance(clean_best, dict) and bool(clean_best.get('descriptor_valid', True)):
                d1_best = clean_best
            if d1_best is not None:
                summary['D1_OH (eV)'] = _safe_float(d1_best.get('ΔG (eV)'))
                summary['D1_site_label'] = str(d1_best.get('site_label', 'unknown'))
                summary['D1_structure_cif'] = str(d1_best.get('structure_cif', ''))
                summary['D1_seed_disp(Å)'] = _safe_float(d1_best.get('H_lateral_disp(Å)'))
                summary['D1_binding_class'] = str(d1_best.get('binding_class', 'unknown'))
                summary['D1_relaxation_scope'] = str(d1_best.get('relaxation_scope', d1_scope))
                summary['D1_total_relax_n_steps'] = int(_safe_float(d1_best.get('total_relax_n_steps'), 0.0))
                summary['D1_fine_relax_relaxed_atoms'] = int(_safe_float(d1_best.get('fine_relax_relaxed_atoms'), 0.0))
            else:
                summary['D1_seed_quality'] = 'missing'
                summary['D1_seed_warning'] = 'No D1 O-top targets were evaluated.'
            if np.isfinite(_safe_float(summary.get('D1_clean_OH (eV)'))) and np.isfinite(_safe_float(summary.get('D1_preOH_OH (eV)'))):
                summary['ΔD1_preOH-clean (eV)'] = float(_safe_float(summary.get('D1_preOH_OH (eV)')) - _safe_float(summary.get('D1_clean_OH (eV)')))
        except Exception as e:
            summary['D1_error'] = str(e)

    if need_d2:
        try:
            # Reuse the main oxide HER screening results as the D2/basic-H descriptor.
            d2_df = (d1_rows_df.copy() if isinstance(d1_rows_df, pd.DataFrame) else pd.DataFrame())
            d2_csv = stage_dir / 'D2_candidates.csv'
            if not d2_df.empty:
                d2_df.to_csv(d2_csv, index=False)
                summary['D2_candidates_csv'] = str(d2_csv.resolve())

            energy_col = 'ΔG_H (eV)' if 'ΔG_H (eV)' in d2_df.columns else ('ΔE_H_user (eV)' if 'ΔE_H_user (eV)' in d2_df.columns else None)
            if not d2_df.empty and energy_col is not None:
                d2_best, q, w = _pick_descriptor_seed_row(d2_df, energy_col=energy_col, disp_thresh=DESCRIPTOR_D2_DISP_THRESH_A)
                summary['D2_seed_quality'] = q
                summary['D2_seed_warning'] = w
                if d2_best is not None:
                    summary['D2_Hreact (eV)'] = _safe_float(d2_best.get(energy_col))
                    summary['D2_site_label'] = str(d2_best.get('site_label', d2_best.get('site', 'unknown')))
                    summary['D2_binding_class'] = str(d2_best.get('final_site_kind', d2_best.get('binding_class', d2_best.get('site_kind', 'unknown'))))
                    summary['D2_structure_cif'] = str(d2_best.get('structure_cif', ''))
                    summary['D2_seed_disp(Å)'] = _safe_float(d2_best.get('H_lateral_disp(Å)'))
                    summary['D2_relaxation_scope'] = str(d2_best.get('relaxation_scope', d2_best.get('site_tracking_mode', resolved_scope)))
                    summary['D2_total_relax_n_steps'] = int(_safe_float(d2_best.get('total_relax_n_steps', d2_best.get('z_relax_n_steps', 0.0)), 0.0))
                    summary['D2_fine_relax_relaxed_atoms'] = int(_safe_float(d2_best.get('fine_relax_relaxed_atoms', 0.0), 0.0))
                    if _rows_same_basin(d1_best, d2_best, slab_u_rel=slab_u_rel):
                        summary['D2_same_basin_as_D1'] = True
                        summary['D2_basin_note'] = 'D2 main HER row matches the same final basin as D1.'
            elif energy_col is None:
                summary['D2_seed_quality'] = 'missing'
                summary['D2_seed_warning'] = 'Main HER screening rows did not contain a usable energy column.'
        except Exception as e:
            summary['D2_error'] = str(e)

    # if need_d3:
    #     Legacy D3 / H2-pairing-proxy logic is intentionally disabled.
    #     The code skeleton is retained only for future redevelopment.
    if not need_d3:
        summary['D3_status'] = str(summary.get('D3_status', 'disabled_by_design') or 'disabled_by_design')

    d1 = _safe_float(summary.get('D1_OH (eV)'))
    d2 = _safe_float(summary.get('D2_Hreact (eV)'))
    d3 = _safe_float(summary.get('D3_pair_proxy (eV)'))
    summary['Δ12 (eV)'] = float(d2 - d1) if np.isfinite(d1) and np.isfinite(d2) else float('nan')
    summary['Δ23 (eV)'] = float(d3 - d2) if np.isfinite(d2) and np.isfinite(d3) else float('nan')
    summary['classification'] = _classify_summary(summary, mode)

    warnings_accum = []
    for k in ('D1_seed_warning', 'D2_seed_warning'):
        if str(summary.get(k, '')).strip():
            warnings_accum.append(str(summary.get(k)))
    if warnings_accum:
        summary['caution'] = (str(summary.get('caution', '')).strip() + ' ' + ' '.join(warnings_accum)).strip()

    summary_name = {
        'D1_OH only (O-top protonation)': 'oxide_descriptor_summary_D1.csv',
        'D2_Hreact only (reactive H state)': 'oxide_descriptor_summary_D2.csv',
        # 'D3_pair only (H2 pairing proxy)': 'oxide_descriptor_summary_D3.csv',
    }.get(mode, 'oxide_two_stage_summary.csv')
    summary_csv = stage_dir / summary_name
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    summary['summary_csv'] = str(summary_csv.resolve())
    return summary
