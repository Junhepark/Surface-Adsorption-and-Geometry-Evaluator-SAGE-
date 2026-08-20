from __future__ import annotations

"""Deterministic state-energy post-processing for SAGE-VOC.

The module does not predict products, infer reaction selectivity, or calculate
step energies/barriers. It selects one QA-valid minimum adsorption-energy proxy
per registered state and maps those numbers onto the route layout defined in
``voc_registry.py``.
"""

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ocp_app.core.voc_registry import get_voc_preset, get_voc_route, normalize_voc_state


DEFAULT_ENERGY_SCALE = {
    "vmin_eV": -2.5,
    "vcenter_eV": 0.0,
    "vmax_eV": 2.5,
    "clipped": True,
}

VALID_QA_LABELS = {
    "ok",
    "ok_single_point_proxy",
    "ok_short_relax_proxy",
    "ok_normal_relax_proxy",
    "ok_local_flex_proxy",
    "ok_rigid_proxy",
    "ok_frozen_pose_proxy",
    "ok_axis_locked_proxy",
    "surface_distorted_but_bound",
    "ok_metal_che_her_like",
}


def _boolish(value: object, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return bool(default)
    try:
        if pd.isna(value):
            return bool(default)
    except Exception:
        pass
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "valid", "ok", "reliable"}:
        return True
    if text in {"false", "0", "no", "n", "invalid", "bad", "unreliable", "", "none", "nan"}:
        return False
    return bool(default)


def _energy_column(df: pd.DataFrame) -> str | None:
    for col in ("ΔE_proxy (eV)", "ΔE_ads_user (eV)", "ΔG_ads (eV)"):
        if col in df.columns:
            return col
    return None


def _state_column(df: pd.DataFrame) -> str | None:
    for col in ("pathway_state_id", "descriptor_state", "adsorbate", "state_label"):
        if col in df.columns:
            return col
    return None


def _valid_candidate_mask(df: pd.DataFrame, energy_col: str) -> pd.Series:
    energy = pd.to_numeric(df[energy_col], errors="coerce")
    valid = pd.Series(np.isfinite(energy), index=df.index)

    if "descriptor_energy_valid" in df.columns:
        valid &= df["descriptor_energy_valid"].map(lambda x: _boolish(x, default=False))
    elif "reliability" in df.columns:
        valid &= df["reliability"].astype(str).str.strip().str.lower().eq("reliable")
    elif "qa" in df.columns:
        valid &= df["qa"].astype(str).str.strip().str.lower().isin(VALID_QA_LABELS)

    if "diagnostic_only" in df.columns:
        valid &= ~df["diagnostic_only"].map(lambda x: _boolish(x, default=False))
    return valid


def _filter_context(df: pd.DataFrame, voc_key: str | None, route_key: str | None) -> pd.DataFrame:
    work = df.copy()
    if voc_key and "target_voc" in work.columns:
        work = work[
            work["target_voc"].astype(str).str.strip().str.lower().eq(str(voc_key).strip().lower())
        ].copy()
    if route_key and "voc_route" in work.columns:
        routed = work["voc_route"].astype(str).str.strip().str.lower()
        # Preserve legacy rows that predate explicit route metadata.
        work = work[
            routed.eq(str(route_key).strip().lower()) | routed.isin({"", "none", "nan"})
        ].copy()
    return work


def select_voc_state_minima(
    df: pd.DataFrame,
    *,
    voc_key: str | None = None,
    route_key: str | None = None,
) -> pd.DataFrame:
    """Select the minimum finite QA-valid ΔE for each registered VOC state."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    work = _filter_context(df, voc_key, route_key)
    energy_col = _energy_column(work)
    state_col = _state_column(work)
    if energy_col is None or state_col is None or work.empty:
        return pd.DataFrame()

    work = work.copy()
    work["_pathway_state"] = work[state_col].map(normalize_voc_state)
    work["_selected_energy"] = pd.to_numeric(work[energy_col], errors="coerce")
    work = work[_valid_candidate_mask(work, energy_col)].copy()
    work = work[work["_pathway_state"].astype(str).ne("")].copy()
    if work.empty:
        return pd.DataFrame()

    # Stable deterministic tie-breakers after the primary minimum-energy rule.
    for col, default in (
        ("retry_count", 0.0),
        ("top_slab_max_disp(Å)", np.inf),
        ("ads_lateral_disp(Å)", np.inf),
    ):
        if col not in work.columns:
            work[col] = default
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)
    if "site_label" not in work.columns:
        work["site_label"] = ""

    work = work.sort_values(
        [
            "_pathway_state",
            "_selected_energy",
            "retry_count",
            "top_slab_max_disp(Å)",
            "ads_lateral_disp(Å)",
            "site_label",
        ],
        kind="mergesort",
    )
    selected = work.groupby("_pathway_state", sort=False, as_index=False).head(1).copy()
    selected["pathway_state_id"] = selected["_pathway_state"]
    selected["selected_energy_eV"] = selected["_selected_energy"]
    selected["selected_energy_column"] = energy_col
    selected["selection_rule"] = "minimum finite QA-valid ΔE per descriptor state"
    return selected.drop(columns=["_pathway_state", "_selected_energy"], errors="ignore").reset_index(drop=True)


def _json_scalar(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _selected_row_payload(row: Mapping[str, object]) -> dict[str, object]:
    keys = [
        "pathway_state_id",
        "selected_energy_eV",
        "selected_energy_column",
        "selection_rule",
        "qa",
        "qa_note",
        "reliability",
        "site_label",
        "site",
        "site_kind",
        "relaxed_site",
        "initial_structure_cif",
        "structure_cif",
        "MODEL",
        "DEVICE",
        "relax_n_steps",
        "relax_converged",
        "ads_lateral_disp(Å)",
        "top_slab_max_disp(Å)",
    ]
    return {key: _json_scalar(row.get(key)) for key in keys if key in row}


def build_voc_pathway_summary(
    df: pd.DataFrame,
    *,
    voc_key: str,
    route_key: str,
    energy_scale: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Build a JSON-safe state-energy map without product interpretation."""
    preset = get_voc_preset(voc_key)
    route = get_voc_route(voc_key, route_key)
    route_key = str(route.get("key", route_key))

    scale = dict(DEFAULT_ENERGY_SCALE)
    if energy_scale:
        for key in ("vmin_eV", "vcenter_eV", "vmax_eV"):
            if key in energy_scale:
                scale[key] = float(energy_scale[key])
    if not float(scale["vmin_eV"]) < float(scale["vcenter_eV"]) < float(scale["vmax_eV"]):
        raise ValueError("VOC energy scale must satisfy vmin < vcenter < vmax.")

    context_df = _filter_context(
        df if isinstance(df, pd.DataFrame) else pd.DataFrame(), voc_key, route_key
    )
    selected = select_voc_state_minima(context_df, voc_key=voc_key, route_key=route_key)
    selected_by_state: dict[str, dict[str, object]] = {}
    if not selected.empty:
        for _, row in selected.iterrows():
            payload = _selected_row_payload(dict(row))
            selected_by_state[str(payload.get("pathway_state_id", ""))] = payload

    raw_state_counts: dict[str, int] = {}
    state_col = _state_column(context_df) if isinstance(context_df, pd.DataFrame) else None
    if state_col and not context_df.empty:
        for state, count in context_df[state_col].map(normalize_voc_state).value_counts().items():
            raw_state_counts[str(state)] = int(count)

    nodes: list[dict[str, object]] = []
    valid_count = 0
    for node_spec in route.get("nodes", []):
        node = dict(node_spec)
        node_id = str(node.get("id", ""))
        state = normalize_voc_state(str(node.get("state", node_id)))
        node["state"] = state
        node["attempt_count"] = int(raw_state_counts.get(state, 0))

        selected_row = selected_by_state.get(state)
        if selected_row:
            node.update(selected_row)
            node["energy_available"] = True
            node["qa_valid_minimum"] = True
            valid_count += 1
        else:
            node["selected_energy_eV"] = None
            node["energy_available"] = False
            node["qa_valid_minimum"] = False
        nodes.append(node)

    edges: list[dict[str, object]] = []
    for edge_spec in route.get("edges", []):
        edge = dict(edge_spec)
        edge["basis"] = (
            "Registry-defined layout only; the arrow is not assigned a step energy, "
            "barrier, reversibility, or product probability."
        )
        edges.append(edge)

    return {
        "schema_version": 2,
        "mode": "VOC_STATE_ENERGY_MAP",
        "voc_key": str(voc_key),
        "voc_label": str(preset.get("label", voc_key)),
        "route_key": route_key,
        "route_label": str(route.get("label", route_key)),
        "route_description": str(route.get("description", "")),
        "selection_rule": "minimum finite QA-valid ΔE per descriptor state",
        "energy_scale": scale,
        "coverage": {
            "registered_nodes": int(len(nodes)),
            "nodes_with_qa_valid_energy": int(valid_count),
            "fraction": float(valid_count / len(nodes)) if nodes else 0.0,
        },
        "lane_labels": dict(route.get("lane_labels", {})),
        "nodes": nodes,
        "edges": edges,
        # Kept empty for backward-compatible consumers. SAGE does not infer products here.
        "products": [],
        "selected_states": list(selected_by_state.values()),
        "warning": (
            "Node numbers are selected QA-valid ΔE_proxy values. Color encodes only the numerical value: "
            "blue = more negative, white = near 0 eV, red = more positive; gray = no QA-valid value. "
            "The map does not predict products, selectivity, reaction-step energies, barriers, or CO2 formation."
        ),
    }


def pathway_summary_to_frame(summary: Mapping[str, object]) -> pd.DataFrame:
    """Flatten nodes and registry edges into one exportable audit table."""
    rows: list[dict[str, object]] = []
    base = {
        "voc_key": summary.get("voc_key"),
        "route_key": summary.get("route_key"),
        "selection_rule": summary.get("selection_rule"),
    }
    for node in summary.get("nodes", []) or []:
        rows.append({**base, "record_type": "node", **dict(node)})
    for edge in summary.get("edges", []) or []:
        rows.append({**base, "record_type": "edge", **dict(edge)})
    return pd.DataFrame(rows)


def write_voc_pathway_summary(
    summary: Mapping[str, object], output_dir: str | Path
) -> tuple[Path, Path]:
    """Write deterministic JSON and CSV summaries beside the VOC result CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results_voc_state_energy_map.json"
    csv_path = out / "results_voc_state_energy_map.csv"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    pathway_summary_to_frame(summary).to_csv(csv_path, index=False)
    return csv_path, json_path
