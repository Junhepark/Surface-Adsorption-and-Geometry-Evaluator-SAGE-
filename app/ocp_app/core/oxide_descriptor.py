
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.io import write
from ase.optimize import BFGS

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
    site_energy_two_stage,
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


def _evaluate_single_h_descriptor(*, slab_u_rel, E_slab_u: float, E_H2: float, site_seed: dict[str, object],
                                  z_steps: int, free_steps: int, use_net_corr: bool,
                                  out_cif: Path | None = None) -> dict[str, object]:
    label = str(site_seed.get('site_label', 'unknown'))
    kind = str(site_seed.get('site_kind', 'unknown'))
    xy = np.asarray(site_seed.get('xy', [np.nan, np.nan]), dtype=float)

    Au, E_uH, _disp_raw = site_energy_two_stage(slab_u_rel, xy, H0S, int(z_steps), int(free_steps))
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
        'nearest_distance(Å)': float(near.get('distance', float('nan'))),
        'site_transition_type': 'stable' if np.isfinite(disp_mic) and disp_mic <= MIGRATE_THR else 'displaced_same_site',
        'placement_mismatch': False,
        'migrated': bool(np.isfinite(disp_mic) and disp_mic > MIGRATE_THR),
        'structure_cif': str(out_cif.resolve()) if out_cif is not None and out_cif.exists() else '',
    }
    return row


def _build_two_h_initial(slab_u_rel, xy1: np.ndarray, xy2: np.ndarray, z_offset: float = H0S):
    A = slab_u_rel.copy()
    z_top = float(np.max(A.get_positions()[:, 2]))
    A += Atoms(symbols=['H', 'H'], positions=[
        [float(xy1[0]), float(xy1[1]), z_top + float(z_offset)],
        [float(xy2[0]), float(xy2[1]), z_top + float(z_offset)],
    ])
    return ensure_pbc3(A, vac_z=20.0)


def _relax_two_h_pair(slab_u_rel, xy1: np.ndarray, xy2: np.ndarray, z_steps: int, free_steps: int):
    A = _build_two_h_initial(slab_u_rel, xy1, xy2, z_offset=H0S)
    A.calc = calc
    slab_n = len(slab_u_rel)
    A.set_constraint(FixAtoms(indices=list(range(slab_n))))
    dyn = BFGS(A, logfile=None)
    dyn.run(fmax=0.03, steps=max(20, int(z_steps) + int(free_steps)))
    E = float(A.get_potential_energy())
    return A, E


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

    if mode == 'D1_OH only (O-top protonation)':
        if not np.isfinite(d1):
            return 'unresolved'
        if d1 > 0.3:
            return 'poor proton acceptor'
        if d1 < -1.0:
            return 'strong OH-forming surface'
        return 'moderate O-top protonation'
    if mode == 'D2_Hreact only (reactive H state)':
        if not np.isfinite(d2):
            return 'no reactive H state found'
        if abs(d2) <= 0.5:
            return 'promising reactive-H regime'
        if d2 < -0.5:
            return 'strong reactive-H stabilization'
        return 'weak reactive-H stabilization'
    if mode == 'D3_pair only (H2 pairing proxy)':
        if not np.isfinite(d3):
            return 'no pairing proxy resolved'
        if h2_like and abs(d3) <= 0.5:
            return 'promising pairing proxy'
        if h2_like and d3 < -0.5:
            return 'paired-H surface trapping'
        if (not h2_like) and d3 < -0.5:
            return 'non-H2-like paired trapping'
        return 'pairing proxy / user review needed'

    if not np.isfinite(d1):
        return 'unresolved'
    if d1 > 0.3:
        return 'poor proton acceptor'
    if not np.isfinite(d2):
        return 'OH-trap surface'
    if np.isfinite(d3) and h2_like and abs(d3) <= 0.5:
        return 'promising three-stage profile'
    if np.isfinite(d3) and (not h2_like) and d3 < -0.5:
        return 'paired-H surface trapping'
    return 'intermediate / user review needed'


def run_oxide_descriptor_profile(*, slab_u_rel, E_slab_u: float, E_H2: float,
                                 d1_rows_df: pd.DataFrame, d1_targets: list[dict[str, object]],
                                 out_root: Path, z_steps: int, free_steps: int,
                                 use_net_corr: bool, descriptor_mode: str = 'Full 3-stage profile (experimental)',
                                 max_reactive_per_kind: int = 2, pair_limit: int = 6) -> dict[str, object]:
    stage_dir = Path(out_root) / 'three_stage'
    stage_dir.mkdir(parents=True, exist_ok=True)

    mode = str(descriptor_mode)
    summary: dict[str, object] = {'descriptor_mode': mode, 'caution': THREE_STAGE_OXIDE_HER_CAUTION}

    defaults = {
        'D1_OH (eV)': np.nan, 'D1_site_label': 'NA', 'D1_structure_cif': '',
        'D1_seed_quality': 'missing', 'D1_seed_warning': '', 'D1_seed_disp(Å)': np.nan,
        'D2_Hreact (eV)': np.nan, 'D2_site_label': 'NA', 'D2_binding_class': 'NA',
        'D2_structure_cif': '', 'D2_seed_quality': 'missing', 'D2_seed_warning': '', 'D2_seed_disp(Å)': np.nan,
        'D2_candidates_csv': '', 'D3_pair_proxy (eV)': np.nan, 'D3_pair_label': 'NA',
        'D3_H2_like_motif': False, 'D3_final_HH_distance(Å)': np.nan, 'D3_structure_cif': '',
        'D3_candidates_csv': '', 'error': '',
    }
    summary.update(defaults)

    need_d1 = mode in {'D1_OH only (O-top protonation)', 'Full 3-stage profile (experimental)'}
    need_d2 = mode in {'D2_Hreact only (reactive H state)', 'D3_pair only (H2 pairing proxy)', 'Full 3-stage profile (experimental)'}
    need_d3 = mode in {'D3_pair only (H2 pairing proxy)', 'Full 3-stage profile (experimental)'}

    d1_best = None
    d2_best = None
    d2_targets: list[dict[str, object]] = []

    if need_d1:
        try:
            d1_df = d1_rows_df.copy() if isinstance(d1_rows_df, pd.DataFrame) else pd.DataFrame()
            d1_energy_col = 'ΔG_H (eV)' if 'ΔG_H (eV)' in d1_df.columns else ('ΔE_H_user (eV)' if 'ΔE_H_user (eV)' in d1_df.columns else None)
            if d1_energy_col is None:
                raise ValueError("No D1 energy column found.")
            d1_best, q, w = _pick_descriptor_seed_row(d1_df, energy_col=d1_energy_col, disp_thresh=DESCRIPTOR_D1_DISP_THRESH_A)
            summary['D1_seed_quality'] = q
            summary['D1_seed_warning'] = w
            if d1_best is not None:
                summary['D1_OH (eV)'] = _safe_float(d1_best.get(d1_energy_col))
                summary['D1_site_label'] = str(d1_best.get('site_label', 'unknown'))
                summary['D1_structure_cif'] = str(d1_best.get('structure_cif', ''))
                summary['D1_seed_disp(Å)'] = _safe_float(d1_best.get('H_lateral_disp(Å)'))
        except Exception as e:
            summary['D1_error'] = str(e)

    if need_d2:
        try:
            d2_targets = _build_reactive_h_targets_oxide(slab_u_rel, max_per_kind=max_reactive_per_kind)
            d2_rows = []
            for i, site_seed in enumerate(d2_targets):
                out_cif = stage_dir / f"D2_{i}_{site_seed.get('site_label','site')}.cif"
                row = _evaluate_single_h_descriptor(
                    slab_u_rel=slab_u_rel, E_slab_u=E_slab_u, E_H2=E_H2,
                    site_seed=site_seed, z_steps=z_steps, free_steps=free_steps,
                    use_net_corr=use_net_corr, out_cif=out_cif,
                )
                row['stage'] = 'D2_Hreact'
                row['is_reactive_candidate'] = bool(str(row.get('binding_class')) == 'metal_adjacent')
                d2_rows.append(row)
            d2_df = pd.DataFrame(d2_rows)
            d2_csv = stage_dir / 'D2_candidates.csv'
            if not d2_df.empty:
                d2_df.to_csv(d2_csv, index=False)
                summary['D2_candidates_csv'] = str(d2_csv.resolve())

            if not d2_df.empty:
                pool = d2_df[d2_df['is_reactive_candidate'].astype(bool)] if 'is_reactive_candidate' in d2_df.columns else d2_df
                if pool.empty:
                    pool = d2_df.copy()
                d2_best, q, w = _pick_descriptor_seed_row(pool, energy_col='ΔG (eV)', disp_thresh=DESCRIPTOR_D2_DISP_THRESH_A)
                summary['D2_seed_quality'] = q
                summary['D2_seed_warning'] = w
                if d2_best is not None:
                    summary['D2_Hreact (eV)'] = _safe_float(d2_best.get('ΔG (eV)'))
                    summary['D2_site_label'] = str(d2_best.get('site_label', 'unknown'))
                    summary['D2_binding_class'] = str(d2_best.get('binding_class', 'unknown'))
                    summary['D2_structure_cif'] = str(d2_best.get('structure_cif', ''))
                    summary['D2_seed_disp(Å)'] = _safe_float(d2_best.get('H_lateral_disp(Å)'))
        except Exception as e:
            summary['D2_error'] = str(e)

    if need_d3:
        try:
            pair_seeds = _build_pair_seed_records(
                d2_best_row=d2_best,
                d2_targets=d2_targets,
                d1_targets=d1_targets,
                max_pairs=pair_limit,
            )
            pair_rows = []
            for i, ps in enumerate(pair_seeds):
                try:
                    xy1 = np.asarray(ps['xy1'], dtype=float).reshape(2)
                    xy2 = np.asarray(ps['xy2'], dtype=float).reshape(2)
                    pair_seed_dist = float(np.linalg.norm(xy1 - xy2))
                    out_cif = stage_dir / f"D3_{i}_{ps.get('pair_label','pair')}".replace(os.sep, '_')
                    out_cif = out_cif.with_suffix('.cif')
                    A2, E2 = _relax_two_h_pair(slab_u_rel, xy1, xy2, z_steps=z_steps, free_steps=free_steps)
                    hh = _to_scalar(_pair_hh_distance(A2, len(A2)-2, len(A2)-1))
                    h2_like = bool(np.isfinite(hh) and hh <= 1.15)
                    write(out_cif, A2)
                    pair_rows.append({
                        'pair_label': str(ps.get('pair_label', f'pair_{i}')),
                        'pair_kind': str(ps.get('pair_kind', 'pair')),
                        'pair_seed_distance(Å)': pair_seed_dist,
                        'ΔE_pair_proxy (eV)': _to_scalar(float(E2) - float(E_slab_u) - float(E_H2)),
                        'final_HH_distance(Å)': hh,
                        'H2_like_motif': bool(h2_like),
                        'structure_cif': str(out_cif.resolve()) if out_cif.exists() else '',
                    })
                except Exception as e:
                    pair_rows.append({
                        'pair_label': str(ps.get('pair_label', f'pair_{i}')),
                        'pair_kind': str(ps.get('pair_kind', 'pair')),
                        'pair_error': str(e),
                    })
                    continue

            d3_df = pd.DataFrame(pair_rows)
            d3_csv = stage_dir / 'D3_pair_candidates.csv'
            if not d3_df.empty:
                for c in ['pair_seed_distance(Å)', 'ΔE_pair_proxy (eV)', 'final_HH_distance(Å)']:
                    if c in d3_df.columns:
                        d3_df[c] = pd.to_numeric(d3_df[c], errors='coerce')
                d3_df.to_csv(d3_csv, index=False)
                summary['D3_candidates_csv'] = str(d3_csv.resolve())

            valid_d3 = d3_df.copy()
            if 'pair_error' in valid_d3.columns:
                valid_d3 = valid_d3[valid_d3['pair_error'].isna()]
            if not valid_d3.empty:
                d3_rank = valid_d3.copy()
                d3_rank['_h2_sort'] = (~d3_rank['H2_like_motif'].astype(bool)).astype(int)
                d3_rank = d3_rank.sort_values(['_h2_sort', 'final_HH_distance(Å)', 'ΔE_pair_proxy (eV)'], ascending=[True, True, True], na_position='last')
                d3_best = d3_rank.iloc[0].to_dict()
                summary['D3_pair_proxy (eV)'] = _safe_float(d3_best.get('ΔE_pair_proxy (eV)'))
                summary['D3_pair_label'] = str(d3_best.get('pair_label', 'unknown'))
                summary['D3_H2_like_motif'] = bool(d3_best.get('H2_like_motif', False))
                summary['D3_final_HH_distance(Å)'] = _safe_float(d3_best.get('final_HH_distance(Å)'))
                summary['D3_structure_cif'] = str(d3_best.get('structure_cif', ''))
            elif not d3_df.empty:
                summary['D3_error'] = 'All D3 pair candidates failed; inspect D3_pair_candidates.csv for pair_error.'
        except Exception as e:
            summary['D3_error'] = str(e)


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
        'D3_pair only (H2 pairing proxy)': 'oxide_descriptor_summary_D3.csv',
    }.get(mode, 'oxide_three_stage_summary.csv')
    summary_csv = stage_dir / summary_name
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    summary['summary_csv'] = str(summary_csv.resolve())
    return summary
