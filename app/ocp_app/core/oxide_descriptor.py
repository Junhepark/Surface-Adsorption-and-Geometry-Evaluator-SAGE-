
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
    site_energy_oh_anchoronly,
    site_energy_oh_constrained,
)

THREE_STAGE_OXIDE_HER_CAUTION = (
    "Caution: The O–H and reactive-H stages are evaluated using the OCP-based "
    "relaxed-state workflow, whereas the H₂ pairing stage is only an approximate "
    "release proxy rather than an explicit reaction barrier. The final-stage result "
    "should therefore be used as a supportive screening indicator, not as a "
    "quantitatively validated kinetic metric."
)

DESCRIPTOR_D1_DISP_THRESH_A = 1.60
DESCRIPTOR_D2_DISP_THRESH_A = 1.20
DESCRIPTOR_ABS_DE_THRESH_EV = 3.0


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


def _compute_anchor_oh_distance(atoms, anchor_index: int, h_index: int = -1) -> float:
    try:
        pos = atoms.get_positions()
        if h_index < 0:
            h_index = len(atoms) + int(h_index)
        dvec = np.asarray(pos[int(anchor_index)], dtype=float) - np.asarray(pos[int(h_index)], dtype=float)
        _vec, dist = find_mic(dvec, atoms.get_cell(), atoms.get_pbc())
        return float(dist)
    except Exception:
        return float('nan')


def _infer_binding_class_from_basic_row(row: dict[str, object]) -> str:
    p = str(row.get('structure_cif', '') or '').strip()
    if p:
        try:
            at = read(p)
            cls, _near = _classify_h_binding_state(at, h_index=len(at)-1)
            return str(cls)
        except Exception:
            pass
    relaxed = str(row.get('relaxed_site', row.get('final_site_kind', ''))).strip().lower()
    if 'anion' in relaxed or 'o_' in relaxed:
        return 'o_bound'
    if any(tok in relaxed for tok in ('ontop', 'bridge', 'fcc', 'hcp', 'metal')):
        return 'metal_adjacent'
    return 'unresolved'


def _build_d2_from_basic_screening(df: pd.DataFrame, *, stage_dir: Path) -> tuple[dict[str, object] | None, str, str, str]:
    if df is None or df.empty:
        return None, 'missing', 'no basic HER screening rows', ''
    work = df.copy()
    energy_col = None
    for c in ('ΔG_H(U,pH) (eV)', 'ΔG_H (eV)', 'ΔE_H_user (eV)'):
        if c in work.columns:
            energy_col = c
            break
    if energy_col is None:
        return None, 'missing', 'no D2 energy column found', ''
    csv_path = stage_dir / 'D2_candidates.csv'
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        work.to_csv(csv_path, index=False)
    except Exception:
        csv_path = Path('')
    best, q, w = _pick_descriptor_seed_row(work, energy_col=energy_col, disp_thresh=DESCRIPTOR_D2_DISP_THRESH_A)
    if best is None:
        return None, q, w, str(csv_path.resolve()) if str(csv_path) else ''
    row = dict(best)
    row['ΔG (eV)'] = _safe_float(best.get(energy_col))
    row['binding_class'] = _infer_binding_class_from_basic_row(best)
    row['relaxation_scope'] = str(best.get('her_relaxation_scope', best.get('relaxation_scope', best.get('fine_relax_scope', 'basic_her'))))
    row['total_relax_n_steps'] = int(_safe_float(best.get('total_relax_n_steps', best.get('fine_relax_n_steps', 0)), 0.0))
    row['fine_relax_relaxed_atoms'] = int(_safe_float(best.get('fine_relax_relaxed_atoms', best.get('relaxed_atom_count', 0)), 0.0))
    return row, q, w, str(csv_path.resolve()) if str(csv_path) else ''


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

    base = work[
        work["_energy"].notna()
        & work["_disp"].notna()
        & (work["_disp"] <= float(disp_thresh))
        & (
            work["_abs_dE"].isna()
            | (work["_abs_dE"] <= float(DESCRIPTOR_ABS_DE_THRESH_EV))
        )
    ]
    ranked = _quality_rank(base)
    if not ranked.empty:
        return ranked.iloc[0].to_dict(), "reliable", ""

    finite = work[work["_energy"].notna()].copy()
    ranked = _quality_rank(finite)
    if not ranked.empty:
        disp_val = _safe_float(ranked.iloc[0].get("H_lateral_disp(Å)"))
        warning = f"Using low-confidence fallback seed (disp={disp_val:.3f} Å)."
        return ranked.iloc[0].to_dict(), "fallback_unreliable", warning

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

    Au, E_uH, _disp_raw, relax_meta = site_energy_oh_anchoronly(
        slab_u_rel,
        xy,
        anchor_index=anchor_index,
        h0s=H0S,
        z_steps=int(z_steps),
        free_steps=int(free_steps),
        return_meta=True,
    )
    dE_u = float(E_uH) - float(E_slab_u) - 0.5 * float(E_H2)
    dG_u = float(dE_u + (0.24 if use_net_corr else 0.0))

    h_pos_final = np.asarray(Au.get_positions()[-1], dtype=float)
    _, disp_mic = _mic_xy_delta(slab_u_rel.get_cell(), slab_u_rel.get_pbc(), xy, h_pos_final[:2])
    ho_dist = _compute_anchor_oh_distance(Au, anchor_index=anchor_index, h_index=len(Au)-1) if anchor_index >= 0 else float('nan')

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
        'binding_class': 'o_bound',
        'nearest_symbol': 'O',
        'nearest_index': int(anchor_index),
        'nearest_distance(Å)': float(ho_dist),
        'final_h_x(Å)': float(h_pos_final[0]),
        'final_h_y(Å)': float(h_pos_final[1]),
        'final_h_z(Å)': float(h_pos_final[2]),
        'site_transition_type': 'stable' if np.isfinite(disp_mic) and disp_mic <= MIGRATE_THR else 'displaced_same_site',
        'placement_mismatch': False,
        'migrated': bool(np.isfinite(disp_mic) and disp_mic > MIGRATE_THR),
        'structure_cif': str(out_cif.resolve()) if out_cif is not None and out_cif.exists() else '',
        'relaxation_scope': 'anchor_oh_only',
        'n_fix_layers': 0,
        'selected_h0': _safe_float((relax_meta or {}).get('selected_h0')),
        'z_relax_n_steps': int((relax_meta or {}).get('z_relax_n_steps', 0)),
        'z_relax_converged': (relax_meta or {}).get('z_relax_converged', None),
        'z_relax_relaxed_atoms': int((relax_meta or {}).get('z_relax_relaxed_atoms', 0)),
        'fine_relax_n_steps': int((relax_meta or {}).get('fine_relax_n_steps', 0)),
        'fine_relax_converged': (relax_meta or {}).get('fine_relax_converged', None),
        'fine_relax_relaxed_atoms': int((relax_meta or {}).get('fine_relax_relaxed_atoms', 0)),
        'total_relax_n_steps': int((relax_meta or {}).get('total_relax_n_steps', 0)),
        'local_free_count': int((relax_meta or {}).get('local_free_count', 1)),
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

    if mode == 'D2_Hreact only (reactive H state)':
        if not np.isfinite(d2):
            return 'no D2 / basic HER result found'
        if d2 > 0.3:
            return 'weak H adsorption'
        if d2 < -0.5:
            return 'strong H stabilization'
        return 'moderate H adsorption'

    if not np.isfinite(d1) and not np.isfinite(d2):
        return 'unresolved'
    if np.isfinite(d1) and np.isfinite(d2):
        if d1 > 0.3 and d2 > 0.3:
            return 'poor proton acceptor'
        if d1 < 0.0 and d2 > 0.3:
            return 'OH-forming but poor HER surface'
        if d1 > 0.3 and d2 <= 0.3:
            return 'reactive-H favored over hydroxylation'
        return 'intermediate / user review needed'
    if np.isfinite(d1):
        return 'D1 only resolved'
    return 'D2 only resolved'


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

    need_d1 = mode in {'Full 2-stage profile (recommended)'}
    need_d2 = mode in {'D2_Hreact only (reactive H state)', 'Full 2-stage profile (recommended)'}
    need_d3 = False

    d1_best = None
    d2_best = None
    d2_targets: list[dict[str, object]] = []
    d1_scope = 'anchor_oh_only'
    d2_scope = 'basic_her_screening'
    d3_scope = d2_scope

    if need_d1:
        try:
            d1_rows = []
            for i, site_seed in enumerate(d1_targets or []):
                out_cif = stage_dir / f"D1_{i}_{_safe_label_token(site_seed.get('site_label','site'))}.cif"
                row = _evaluate_constrained_oh_descriptor(
                    slab_u_rel=slab_u_rel,
                    E_slab_u=E_slab_u,
                    E_H2=E_H2,
                    site_seed=site_seed,
                    z_steps=z_steps,
                    free_steps=free_steps,
                    use_net_corr=use_net_corr,
                    out_cif=out_cif,
                )
                row['stage'] = 'D1_OH'
                d1_rows.append(row)
            d1_df = pd.DataFrame(d1_rows)
            d1_csv = stage_dir / 'D1_candidates.csv'
            if not d1_df.empty:
                d1_df.to_csv(d1_csv, index=False)
                summary['D1_candidates_csv'] = str(d1_csv.resolve())
                d1_best, q, w = _pick_descriptor_seed_row(d1_df, energy_col='ΔG (eV)', disp_thresh=DESCRIPTOR_D1_DISP_THRESH_A)
                summary['D1_seed_quality'] = q
                summary['D1_seed_warning'] = w
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
        except Exception as e:
            summary['D1_error'] = str(e)

    if need_d2:
        try:
            basic_df = d1_rows_df.copy() if isinstance(d1_rows_df, pd.DataFrame) else pd.DataFrame()
            d2_best, q, w, d2_csv_path = _build_d2_from_basic_screening(basic_df, stage_dir=stage_dir)
            summary['D2_seed_quality'] = q
            summary['D2_seed_warning'] = w
            summary['D2_candidates_csv'] = d2_csv_path
            if d2_best is not None:
                summary['D2_Hreact (eV)'] = _safe_float(d2_best.get('ΔG (eV)'))
                summary['D2_site_label'] = str(d2_best.get('site_label', 'unknown'))
                summary['D2_binding_class'] = str(d2_best.get('binding_class', 'unknown'))
                summary['D2_structure_cif'] = str(d2_best.get('structure_cif', ''))
                summary['D2_seed_disp(Å)'] = _safe_float(d2_best.get('H_lateral_disp(Å)'))
                summary['D2_relaxation_scope'] = str(d2_best.get('relaxation_scope', d2_scope))
                summary['D2_total_relax_n_steps'] = int(_safe_float(d2_best.get('total_relax_n_steps'), 0.0))
                summary['D2_fine_relax_relaxed_atoms'] = int(_safe_float(d2_best.get('fine_relax_relaxed_atoms'), 0.0))
                if _rows_same_basin(d1_best, d2_best, slab_u_rel=slab_u_rel):
                    summary['D2_same_basin_as_D1'] = True
                    summary['D2_basin_note'] = 'D2 overlaps the same final basin as D1.'
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
