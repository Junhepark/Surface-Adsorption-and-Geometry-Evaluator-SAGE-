# ocp_app/core/anchors/oxide_interface_her.py
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
from ase.geometry import find_mic

from ocp_app.core.ads_sites import ANION_SYMBOLS


METAL_SYMBOLS_DEFAULT = {"Cu", "Ni", "Co", "Fe", "Mn", "Pt", "Pd", "Rh", "Ru", "Ir", "Ag", "Au"}


def _is_metal_symbol(sym: str) -> bool:
    s = str(sym)
    return s.upper() != "H" and s not in ANION_SYMBOLS


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _pbc_distance(atoms, i: int, j: int) -> float:
    pos = atoms.get_positions()
    _vec, dist = find_mic(
        np.asarray(pos[int(j)], dtype=float) - np.asarray(pos[int(i)], dtype=float),
        atoms.get_cell(),
        atoms.get_pbc(),
    )
    return float(dist)


def _pbc_vector_from_point(atoms, point_xyz, atom_index: int):
    pos = atoms.get_positions()
    vec, dist = find_mic(
        np.asarray(pos[int(atom_index)], dtype=float) - np.asarray(point_xyz, dtype=float),
        atoms.get_cell(),
        atoms.get_pbc(),
    )
    return np.asarray(vec, dtype=float), float(dist)


def _frac_xy(atoms, xy) -> tuple[float, float]:
    cell = np.asarray(atoms.get_cell(), dtype=float)
    a = cell[0, :2]
    b = cell[1, :2]
    A = np.column_stack([a, b])
    xy = np.asarray(xy, dtype=float).reshape(2)
    try:
        f = np.linalg.solve(A, xy)
        return float(f[0] % 1.0), float(f[1] % 1.0)
    except Exception:
        ax = max(float(np.linalg.norm(a)), 1.0)
        by = max(float(np.linalg.norm(b)), 1.0)
        return float((xy[0] / ax) % 1.0), float((xy[1] / by) % 1.0)


def _inside_fractional_margin(atoms, xy, margin: float) -> bool:
    if margin <= 0:
        return True
    try:
        fx, fy = _frac_xy(atoms, xy)
        m = float(margin)
        return bool(m < fx < 1.0 - m and m < fy < 1.0 - m)
    except Exception:
        return True


def bridge_pair_label(symbols: Iterable[str]) -> str:
    vals = [str(x) for x in symbols if str(x)]
    if not vals:
        return "NA"
    vals = sorted(vals)
    return "-".join(vals)


def local_metal_environment(atoms, center_xyz, cutoff: float = 3.5) -> dict[str, object]:
    """Return local metal composition around a point.

    The output is intentionally flat so it can be merged directly into a D2 row.
    """
    syms = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    rows = []
    for i, sym in enumerate(syms):
        if not _is_metal_symbol(str(sym)):
            continue
        _vec, dist = _pbc_vector_from_point(atoms, center_xyz, int(i))
        if np.isfinite(dist) and float(dist) <= float(cutoff):
            rows.append((int(i), str(sym), float(dist)))

    rows.sort(key=lambda r: (r[2], r[0]))
    counts = Counter(sym for _, sym, _ in rows)
    n = int(sum(counts.values()))
    out = {
        "local_metal_cutoff(Å)": float(cutoff),
        "local_metal_count": n,
        "local_metal_indices": ",".join(str(i) for i, _, _ in rows),
        "local_metal_symbols": ",".join(sym for _, sym, _ in rows),
    }
    for key in ("Cu", "Ni", "Co", "Fe", "Mn"):
        out[f"local_{key}_count"] = int(counts.get(key, 0))
        out[f"local_{key}_fraction"] = float(counts.get(key, 0) / n) if n > 0 else float("nan")
    return out


def surface_metal_indices(
    atoms,
    *,
    z_window: float = 3.0,
    include_symbols: set[str] | None = None,
) -> list[int]:
    """Return near-top metal/cation indices for interface-aware D2 site generation."""
    syms = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    pos = np.asarray(atoms.get_positions(), dtype=float)
    metal_idx = []
    for i, sym in enumerate(syms):
        s = str(sym)
        if not _is_metal_symbol(s):
            continue
        if include_symbols is not None and s not in set(include_symbols):
            continue
        metal_idx.append(int(i))
    if not metal_idx:
        return []

    z = pos[np.asarray(metal_idx, dtype=int), 2]
    zmax = float(np.max(z))
    return [int(i) for i in metal_idx if pos[int(i), 2] >= zmax - float(z_window)]


def _pair_center_mic(atoms, i: int, j: int) -> np.ndarray:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    p1 = np.asarray(pos[int(i)], dtype=float)
    vec, _dist = find_mic(
        np.asarray(pos[int(j)], dtype=float) - p1,
        atoms.get_cell(),
        atoms.get_pbc(),
    )
    return p1 + 0.5 * np.asarray(vec, dtype=float)


def generate_interface_metal_bridge_targets(
    atoms,
    *,
    max_per_pair: int = 4,
    z_window: float = 3.0,
    min_mm_dist: float = 2.0,
    max_mm_dist: float = 3.8,
    max_pair_dz: float = 2.0,
    local_cutoff: float = 3.5,
    frac_margin: float = 0.05,
    prefer_mixed_pairs: bool = True,
) -> list[dict[str, object]]:
    """Generate D2 metal-bridge H* targets resolved by local metal pair identity.

    This is intended for mixed oxide / oxide-interface models such as CuO/NiO or
    CuO/Co3O4.  It returns dict records, not AdsSite objects, so it can be used
    directly by oxide_descriptor.py without changing the generic ads_sites.py API.
    """
    syms = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    pos = np.asarray(atoms.get_positions(), dtype=float)
    idx = surface_metal_indices(atoms, z_window=float(z_window))
    if len(idx) < 2:
        return []

    records: list[dict[str, object]] = []
    for a, i in enumerate(idx):
        for j in idx[a + 1:]:
            i = int(i)
            j = int(j)
            if abs(float(pos[i, 2] - pos[j, 2])) > float(max_pair_dz):
                continue
            mm = _pbc_distance(atoms, i, j)
            if not (float(min_mm_dist) <= mm <= float(max_mm_dist)):
                continue

            center = _pair_center_mic(atoms, i, j)
            if not _inside_fractional_margin(atoms, center[:2], frac_margin):
                continue

            s1, s2 = str(syms[i]), str(syms[j])
            pair = bridge_pair_label([s1, s2])
            mixed = bool(s1 != s2)
            local = local_metal_environment(atoms, center, cutoff=float(local_cutoff))
            local_short = {f"initial_{k}": v for k, v in local.items()}

            rec = {
                "site_label": f"D2_interface_bridge_{pair}_{i}_{j}",
                "site_kind": "metal_bridge",
                "raw_site_kind": "interface_bridge",
                "xy": np.asarray(center[:2], dtype=float),
                "initial_xyz": np.asarray(center[:3], dtype=float),
                "surface_indices": (int(i), int(j)),
                "surface_metal_symbols": f"{s1},{s2}",
                "seed_source": "oxide_interface_metal_bridge",
                "D2_policy": "interface_metal_bridge_resolved_Hstar",
                "initial_bridge_pair": pair,
                "initial_bridge_metal_1": s1,
                "initial_bridge_metal_2": s2,
                "initial_bridge_index_1": int(i),
                "initial_bridge_index_2": int(j),
                "initial_bridge_indices": f"{int(i)},{int(j)}",
                "initial_M1_M2_distance(Å)": float(mm),
                "initial_pair_dz(Å)": float(abs(pos[i, 2] - pos[j, 2])),
                "initial_interface_like": mixed,
            }
            rec.update(local_short)
            records.append(rec)

    if not records:
        return []

    # Prefer mixed-interface bridges, then central sites, then shorter M-M distances.
    def sort_key(rec: dict[str, object]):
        pair = str(rec.get("initial_bridge_pair", ""))
        mixed_rank = 0 if (bool(rec.get("initial_interface_like", False)) and prefer_mixed_pairs) else 1
        fx, fy = _frac_xy(atoms, np.asarray(rec.get("xy"), dtype=float))
        center_d2 = (fx - 0.5) ** 2 + (fy - 0.5) ** 2
        return (mixed_rank, pair, center_d2, _safe_float(rec.get("initial_M1_M2_distance(Å)"), 999.0))

    records.sort(key=sort_key)

    # Limit number of sites per bridge pair.
    out: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    seen = set()
    for rec in records:
        pair = str(rec.get("initial_bridge_pair", "NA"))
        if counts[pair] >= int(max_per_pair):
            continue
        xy = np.asarray(rec.get("xy"), dtype=float).reshape(2)
        fx, fy = _frac_xy(atoms, xy)
        key = (pair, round(fx, 3), round(fy, 3))
        if key in seen:
            continue
        seen.add(key)
        counts[pair] += 1
        # Make label compact and deterministic after limiting.
        rec = dict(rec)
        rec["site_label"] = f"D2_{pair}_bridge_{counts[pair]}"
        out.append(rec)

    return out


def classify_final_h_metal_environment(
    atoms_with_h,
    *,
    h_index: int = -1,
    metal_cutoff: float = 2.45,
    local_cutoff: float = 3.5,
) -> dict[str, object]:
    """Classify relaxed H environment by the nearest metal pair.

    This complements the existing nearest-metal / nearest-anion D2 QC.
    """
    if h_index < 0:
        h_index = len(atoms_with_h) + int(h_index)

    syms = np.asarray(atoms_with_h.get_chemical_symbols(), dtype=object)
    pos = np.asarray(atoms_with_h.get_positions(), dtype=float)
    hpos = np.asarray(pos[int(h_index)], dtype=float)

    metal_rows = []
    for i, sym in enumerate(syms):
        if i == int(h_index) or not _is_metal_symbol(str(sym)):
            continue
        _vec, dist = _pbc_vector_from_point(atoms_with_h, hpos, int(i))
        metal_rows.append({"index": int(i), "symbol": str(sym), "distance": float(dist)})

    metal_rows.sort(key=lambda r: (r["distance"], r["index"]))
    close = [r for r in metal_rows if np.isfinite(r["distance"]) and r["distance"] <= float(metal_cutoff)]

    if len(close) >= 2:
        m1, m2 = close[0], close[1]
        pair = bridge_pair_label([m1["symbol"], m2["symbol"]])
        kind = "metal_bridge" if len(close) == 2 else "metal_hollow_or_multimetal"
        idx1, idx2 = int(m1["index"]), int(m2["index"])
        mm = _pbc_distance(atoms_with_h, idx1, idx2)
    elif len(close) == 1:
        m1 = close[0]
        m2 = {"index": -1, "symbol": "NA", "distance": float("nan")}
        pair = str(m1["symbol"])
        kind = "metal_top"
        idx1, idx2 = int(m1["index"]), -1
        mm = float("nan")
    else:
        m1 = metal_rows[0] if metal_rows else {"index": -1, "symbol": "unknown", "distance": float("nan")}
        m2 = metal_rows[1] if len(metal_rows) > 1 else {"index": -1, "symbol": "NA", "distance": float("nan")}
        pair = "unresolved"
        kind = "unresolved"
        idx1, idx2 = int(m1.get("index", -1)), int(m2.get("index", -1))
        mm = float("nan")

    local = local_metal_environment(atoms_with_h, hpos, cutoff=float(local_cutoff))
    out = {
        "final_bridge_pair": pair,
        "final_bridge_kind": kind,
        "final_bridge_metal_1": str(m1.get("symbol", "unknown")),
        "final_bridge_metal_2": str(m2.get("symbol", "NA")),
        "final_bridge_index_1": int(idx1),
        "final_bridge_index_2": int(idx2),
        "final_bridge_indices": f"{idx1},{idx2}" if idx2 >= 0 else str(idx1),
        "final_M1_M2_distance(Å)": float(mm),
        "final_H_M1_distance(Å)": float(m1.get("distance", float("nan"))),
        "final_H_M2_distance(Å)": float(m2.get("distance", float("nan"))),
        "final_interface_like": bool(str(m1.get("symbol", "")) != str(m2.get("symbol", "")) and idx2 >= 0),
    }
    out.update({f"final_{k}": v for k, v in local.items()})
    return out


def summarize_d2_bridge_distribution(
    d2_df: pd.DataFrame,
    *,
    energy_col: str = "ΔG_H (eV)",
    pair_col: str = "final_bridge_pair",
    validity_col: str = "D2_descriptor_valid",
    near_zero_window: float = 0.30,
) -> pd.DataFrame:
    """Summarize D2 ΔG_H distribution by bridge-pair class."""
    if d2_df is None or d2_df.empty:
        return pd.DataFrame()

    df = d2_df.copy()
    if energy_col not in df.columns:
        energy_col = "ΔG (eV)" if "ΔG (eV)" in df.columns else None
    if energy_col is None:
        return pd.DataFrame()

    if pair_col not in df.columns:
        pair_col = "initial_bridge_pair" if "initial_bridge_pair" in df.columns else None
    if pair_col is None:
        return pd.DataFrame()

    valid = pd.Series([True] * len(df), index=df.index)
    if validity_col in df.columns:
        s = df[validity_col]
        if getattr(s, "dtype", None) == bool:
            valid = s.fillna(False).astype(bool)
        else:
            valid = s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])

    df["_energy"] = pd.to_numeric(df[energy_col], errors="coerce")
    df["_pair"] = df[pair_col].astype(str).fillna("NA")
    df["_valid"] = valid & df["_energy"].notna()
    df_valid = df[df["_valid"]].copy()
    if df_valid.empty:
        return pd.DataFrame()

    rows = []
    for pair, g in df_valid.groupby("_pair", dropna=False):
        vals = pd.to_numeric(g["_energy"], errors="coerce").dropna()
        if vals.empty:
            continue
        near = vals.abs() <= float(near_zero_window)
        best_idx = vals.abs().idxmin()
        rows.append({
            "bridge_pair": str(pair),
            "N_valid_sites": int(len(vals)),
            "mean_ΔG_H(eV)": float(vals.mean()),
            "median_ΔG_H(eV)": float(vals.median()),
            "min_ΔG_H(eV)": float(vals.min()),
            "max_ΔG_H(eV)": float(vals.max()),
            "best_abs_ΔG_H(eV)": float(vals.abs().min()),
            "best_ΔG_H(eV)": float(vals.loc[best_idx]),
            "best_site_label": str(g.loc[best_idx].get("site_label", "")),
            f"near_zero_fraction_|ΔG|<={float(near_zero_window):.2f}eV": float(near.mean()),
            "near_zero_site_count": int(near.sum()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["near_zero_site_count", f"near_zero_fraction_|ΔG|<={float(near_zero_window):.2f}eV", "best_abs_ΔG_H(eV)"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def bridge_distribution_summary_for_global_row(dist_df: pd.DataFrame, near_zero_window: float = 0.30) -> dict[str, object]:
    """Flatten the bridge distribution table into scalar summary columns."""
    out: dict[str, object] = {}
    if dist_df is None or dist_df.empty:
        out["D2_primary_bridge_pair"] = "NA"
        return out

    near_col = f"near_zero_fraction_|ΔG|<={float(near_zero_window):.2f}eV"
    out["D2_primary_bridge_pair"] = str(dist_df.iloc[0].get("bridge_pair", "NA"))
    out["D2_bridge_pair_classes"] = ",".join(str(x) for x in dist_df["bridge_pair"].tolist())
    for _, row in dist_df.iterrows():
        pair = str(row.get("bridge_pair", "NA")).replace("-", "_")
        out[f"D2_{pair}_site_count"] = int(_safe_float(row.get("N_valid_sites", 0), 0))
        out[f"D2_{pair}_best_ΔG_H(eV)"] = _safe_float(row.get("best_ΔG_H(eV)"))
        out[f"D2_{pair}_best_abs_ΔG_H(eV)"] = _safe_float(row.get("best_abs_ΔG_H(eV)"))
        out[f"D2_{pair}_near_zero_fraction"] = _safe_float(row.get(near_col))
    return out
