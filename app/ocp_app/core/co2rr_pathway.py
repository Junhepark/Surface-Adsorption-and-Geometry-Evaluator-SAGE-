"""Site-consistent CO2RR reaction-network post-processing for SAGE.

Independent adsorbate relaxations are converted to CHE edge energies. The
primary table uses one physical seed site at a time; a path assembled from
unrelated global minima is retained only as a lower-bound diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ocp_app.core.co2rr_registry import (
    CO2RR_PRODUCTS,
    CO2RR_WARNING,
    clean_co2rr_adsorbate,
    co2rr_state_label,
    get_co2rr_preset,
    normalize_co2rr_state,
)


def _as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if np.isfinite(result) else None


def _energy_info(df: pd.DataFrame) -> tuple[str | None, str, bool]:
    """Select the best available cumulative state-energy column."""
    level = "electronic_energy_only"
    if "CO2RR_thermochemistry_level" in df.columns:
        declared = {
            str(value).strip()
            for value in df["CO2RR_thermochemistry_level"].dropna().tolist()
            if str(value).strip()
        }
        if "corrected_CHE" in declared:
            level = "corrected_CHE"
        elif "partial_ZPE_only" in declared:
            level = "partial_ZPE_only"
    elif "thermochemical_correction_applied" in df.columns and bool(
        _as_bool_series(df["thermochemical_correction_applied"]).any()
    ):
        level = "corrected_CHE"
    corrected = level != "electronic_energy_only"
    if corrected:
        for col in ("ΔG_ads (eV)", "ΔG_ads", "CO2RR_state_energy (eV)"):
            if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
                return col, level, True
    for col in ("ΔE_ads_user (eV)", "ΔE_ads_user", "CO2RR_state_energy (eV)"):
        if col in df.columns:
            return col, "electronic_energy_only", False
    return None, "unavailable", False


def _valid_mask(df: pd.DataFrame, *, strict_site: bool = False) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    if "reliability" in df.columns:
        mask &= df["reliability"].astype(str).str.lower().eq("reliable")
    if "qa" in df.columns:
        accepted = ["ok", "bound_relaxed"] if strict_site else [
            "ok", "migrated", "bound_relaxed", "bound_migrated"
        ]
        mask &= df["qa"].astype(str).str.lower().isin(accepted)
    for col in ("broken", "crashed", "desorbed"):
        if col in df.columns:
            mask &= ~_as_bool_series(df[col])
    if strict_site:
        for col in ("migrated_actual", "migrated"):
            if col in df.columns:
                mask &= ~_as_bool_series(df[col])
    return mask


def _site_key(row: Mapping[str, Any]) -> str:
    """Return a stable seed-basin key shared by binding variants."""
    for col in ("binding_base_site_label", "base_site_label", "site_label"):
        value = str(row.get(col, "") or "").strip()
        if value and value.lower() not in {"nan", "none"}:
            return value
    return ""


def _selected_row(
    row: Mapping[str, Any], energy: float, e_col: str, level: str,
    corrected: bool, attempts: int,
) -> dict[str, Any]:
    site_label = str(row.get("site_label", ""))
    base_site_label = _site_key(row) or site_label
    raw_state = str(row.get("_ads_clean", row.get("adsorbate", "")))
    return {
        "pathway_state_id": normalize_co2rr_state(raw_state),
        "pathway_state_label": co2rr_state_label(raw_state),
        "selected_energy_eV": float(energy),
        "energy_column": str(e_col),
        "thermochemistry_level": level,
        "thermochemical_correction_applied": corrected,
        "site_label": site_label,
        "base_site_label": base_site_label,
        "site_key": base_site_label,
        "relaxed_site": str(row.get("relaxed_site", row.get("final_site_kind", ""))),
        "qa": str(row.get("qa", "")),
        "attempt_count": int(attempts),
        "initial_surface_support_indices": str(row.get("initial_surface_support_indices", "")),
        "initial_structure_cif": str(row.get("initial_structure_cif", "")),
        "structure_cif": str(row.get("structure_cif", "")),
    }


def _prepare_rows(
    df: pd.DataFrame, *, strict_site: bool,
) -> tuple[pd.DataFrame, str | None, str, bool, dict[str, int]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), None, "unavailable", False, {}
    e_col, level, corrected = _energy_info(df)
    if e_col is None or "adsorbate" not in df.columns:
        return pd.DataFrame(), None, level, corrected, {}
    work = df.copy()
    work["_ads_clean"] = work["adsorbate"].map(clean_co2rr_adsorbate)
    work["_energy"] = pd.to_numeric(work[e_col], errors="coerce")
    work["_site_key"] = work.apply(lambda row: _site_key(row), axis=1)
    attempts = work.groupby("_ads_clean", dropna=False).size().to_dict()
    valid = work.loc[
        _valid_mask(work, strict_site=strict_site) & np.isfinite(work["_energy"])
    ].copy()
    return valid, e_col, level, corrected, attempts


def select_co2rr_state_minima(
    df: pd.DataFrame,
    pathway_key: str = "competitive_c1",
    states: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Select QA-valid global minima for diagnostics, irrespective of site."""
    preset = get_co2rr_preset(pathway_key)
    selected_states = list(states) if states is not None else list(preset.get("states", []))
    valid, e_col, level, corrected, attempts = _prepare_rows(df, strict_site=False)
    if e_col is None:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for state in selected_states:
        key = clean_co2rr_adsorbate(state)
        sub = valid.loc[valid["_ads_clean"].eq(key)]
        if sub.empty:
            continue
        row = sub.loc[sub["_energy"].idxmin()]
        rows.append(_selected_row(
            row, float(row["_energy"]), e_col, level, corrected,
            int(attempts.get(key, 0)),
        ))
    return pd.DataFrame(rows)


def select_co2rr_state_minima_by_site(
    df: pd.DataFrame,
    pathway_key: str = "competitive_c1",
    states: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Select one non-migrated minimum per chemical state and seed site."""
    preset = get_co2rr_preset(pathway_key)
    selected_states = list(states) if states is not None else list(preset.get("states", []))
    selected = {clean_co2rr_adsorbate(state) for state in selected_states}
    valid, e_col, level, corrected, attempts = _prepare_rows(df, strict_site=True)
    if e_col is None or valid.empty:
        return pd.DataFrame()
    valid = valid.loc[valid["_ads_clean"].isin(selected) & valid["_site_key"].ne("")]
    rows: list[dict[str, Any]] = []
    for (site, key), sub in valid.groupby(["_site_key", "_ads_clean"], sort=True):
        row = sub.loc[sub["_energy"].idxmin()]
        picked = _selected_row(
            row, float(row["_energy"]), e_col, level, corrected,
            int(attempts.get(key, 0)),
        )
        picked["site_key"] = str(site)
        rows.append(picked)
    return pd.DataFrame(rows)


def _normalize_edge(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        source = str(raw.get("from", raw.get("source", "")))
        target = str(raw.get("to", raw.get("target", "")))
        edge_type = str(raw.get("edge_type", "pcet"))
        potential_sensitive = bool(raw.get("potential_sensitive", edge_type == "pcet"))
        return {
            "from": source,
            "to": target,
            "label": str(raw.get("label", f"{source} -> {target}")),
            "n_pe": int(raw.get("n_pe", 1 if potential_sensitive else 0)),
            "edge_type": edge_type,
            "potential_sensitive": potential_sensitive,
            "required_site_count": int(raw.get("required_site_count", 1)),
        }
    values = list(raw)
    source, target = str(values[0]), str(values[1])
    return {
        "from": source,
        "to": target,
        "label": str(values[2]) if len(values) > 2 else f"{source} -> {target}",
        "n_pe": 1,
        "edge_type": "pcet",
        "potential_sensitive": True,
        "required_site_count": 1,
    }


def _endpoint_records(
    product_keys: Sequence[str],
    product_state_energies: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    supplied = dict(product_state_energies or {})
    records: dict[str, dict[str, Any]] = {}
    for product_key in product_keys:
        spec = CO2RR_PRODUCTS.get(str(product_key))
        if spec is None:
            continue
        raw = supplied.get(product_key, supplied.get(spec.state_id, {}))
        if not isinstance(raw, Mapping):
            raw = {"free_energy_eV": raw}
        free_e = _finite(raw.get("free_energy_eV"))
        elec_e = _finite(raw.get("electronic_energy_eV"))
        use_e = free_e if free_e is not None else elec_e
        available = bool(raw.get("available", use_e is not None)) and use_e is not None
        records[spec.state_id] = {
            "id": spec.state_id,
            "label": spec.label,
            "selected_energy_eV": float(use_e) if available else None,
            "energy_available": available,
            "thermochemistry_level": str(raw.get(
                "thermochemistry_level",
                "corrected_CHE" if free_e is not None and raw.get("correction_applied")
                else "electronic_energy_only",
            )),
            "product_key": product_key,
            "warning": str(raw.get("warning", spec.warning)),
            "gas_reference_source": str(raw.get("gas_reference_source", "")),
        }
    return records


def _enumerate_paths(
    edges: Sequence[Mapping[str, Any]], start: str, target: str,
) -> list[list[dict[str, Any]]]:
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        adjacency.setdefault(str(edge["from"]), []).append(dict(edge))
    paths: list[list[dict[str, Any]]] = []

    def visit(node: str, path: list[dict[str, Any]], seen: set[str]) -> None:
        if node == target:
            paths.append(list(path))
            return
        for edge in adjacency.get(node, []):
            nxt = str(edge["to"])
            if nxt not in seen:
                visit(nxt, path + [edge], seen | {nxt})

    visit(start, [], {start})
    return paths


def _path_metrics(
    path: Sequence[Mapping[str, Any]],
    energy_map: Mapping[str, float | None],
    potential_V: float,
) -> dict[str, Any]:
    edge_rows: list[dict[str, Any]] = []
    missing_states: set[str] = set()
    for edge in path:
        source, target = str(edge["from"]), str(edge["to"])
        g0, g1 = _finite(energy_map.get(source)), _finite(energy_map.get(target))
        available = g0 is not None and g1 is not None
        if available:
            dg0 = float(g1 - g0)
            n_pe = int(edge.get("n_pe", 0))
            dgu = (
                float(dg0 + n_pe * float(potential_V))
                if bool(edge.get("potential_sensitive")) else dg0
            )
        else:
            dg0, dgu = None, None
            if g0 is None:
                missing_states.add(source)
            if g1 is None:
                missing_states.add(target)
        edge_rows.append({
            **dict(edge), "delta_G_0_eV": dg0,
            "delta_G_at_U_eV": dgu, "energy_available": available,
        })

    complete = bool(edge_rows) and all(bool(edge["energy_available"]) for edge in edge_rows)
    pcet = [
        edge for edge in edge_rows
        if edge["energy_available"] and bool(edge.get("potential_sensitive"))
        and int(edge.get("n_pe", 0)) > 0
    ]
    chemical = [
        edge for edge in edge_rows
        if edge["energy_available"] and not bool(edge.get("potential_sensitive"))
    ]
    if complete and pcet:
        ratios = [float(edge["delta_G_0_eV"]) / int(edge["n_pe"]) for edge in pcet]
        pds = pcet[int(np.argmax(ratios))]
        pds_per_e = float(max(ratios))
        limiting = -pds_per_e
    else:
        pds, pds_per_e, limiting = None, None, None
    chemical_max = max((float(edge["delta_G_0_eV"]) for edge in chemical), default=0.0)
    finite_u = [
        float(edge["delta_G_at_U_eV"])
        for edge in edge_rows if edge["delta_G_at_U_eV"] is not None
    ]
    return {
        "complete": complete,
        "available_edge_count": int(sum(bool(edge["energy_available"]) for edge in edge_rows)),
        "total_edge_count": int(len(edge_rows)),
        "missing_states": sorted(missing_states),
        "edges": edge_rows,
        "pds": "" if pds is None else str(pds["label"]),
        "pds_delta_G_0_eV": None if pds is None else float(pds["delta_G_0_eV"]),
        "pds_n_pe": None if pds is None else int(pds["n_pe"]),
        "pds_delta_G_per_e_eV": pds_per_e,
        "limiting_potential_V": limiting,
        "chemical_bottleneck_eV": float(max(0.0, chemical_max)),
        "max_delta_G_at_U_eV": max(finite_u) if complete and finite_u else None,
    }


def _suffix(
    path: Sequence[Mapping[str, Any]], start_state: str,
) -> list[dict[str, Any]]:
    for idx, edge in enumerate(path):
        if str(edge.get("from")) == str(start_state):
            return [dict(item) for item in path[idx:]]
    return []


def _core_before_endpoint(
    path: Sequence[Mapping[str, Any]], endpoint_state: str,
) -> list[dict[str, Any]]:
    return [dict(edge) for edge in path if str(edge.get("to")) != str(endpoint_state)]


def _metric_fields(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_complete": bool(metrics.get("complete")),
        f"{prefix}_PDS": str(metrics.get("pds", "")),
        f"{prefix}_PDS_delta_G_0_eV": metrics.get("pds_delta_G_0_eV"),
        f"{prefix}_PDS_delta_G_per_e_eV": metrics.get("pds_delta_G_per_e_eV"),
        f"{prefix}_limiting_potential_V": metrics.get("limiting_potential_V"),
        f"{prefix}_max_delta_G_at_U_eV": metrics.get("max_delta_G_at_U_eV"),
        f"{prefix}_chemical_bottleneck_eV": metrics.get("chemical_bottleneck_eV"),
    }


def _screening_bottleneck(metrics: Mapping[str, Any]) -> tuple[str, float | None, str]:
    """Compare PCET ΔG/n and chemical ΔG without calling chemistry a PDS."""
    if not bool(metrics.get("complete")):
        return "", None, "incomplete"
    candidates: list[tuple[float, str, str]] = []
    for edge in metrics.get("edges", []) or []:
        dg = _finite(edge.get("delta_G_0_eV"))
        if dg is None:
            continue
        n_pe = int(edge.get("n_pe", 0))
        if bool(edge.get("potential_sensitive")) and n_pe > 0:
            candidates.append((float(dg / n_pe), str(edge.get("label", "")), "PCET_PDS"))
        else:
            candidates.append((float(dg), str(edge.get("label", "")), "chemical_step"))
    if not candidates:
        return "", None, "none"
    score, label, kind = max(candidates, key=lambda item: item[0])
    return label, float(score), kind


def _path_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    overall = record.get("overall_metrics", {})
    branch = record.get("branch_core_metrics", {})
    overall_max = _finite(overall.get("max_delta_G_at_U_eV"))
    branch_max = _finite(branch.get("max_delta_G_at_U_eV"))
    return (
        0 if overall.get("complete") else 1,
        0 if branch.get("complete") else 1,
        len(overall.get("missing_states", [])),
        -int(overall.get("available_edge_count", 0)),
        overall_max if overall_max is not None else float("inf"),
        branch_max if branch_max is not None else float("inf"),
    )


def _mechanistic_branch(product_key: str, path_node_ids: Sequence[str]) -> str:
    nodes = set(str(node) for node in path_node_ids)
    key = str(product_key).lower()
    if key == "methane":
        entry = "COH" if "COH*" in nodes else "CHO"
        removal = (
            "CH2OH_deoxygenation" if "CH2OH*" in nodes
            else "CH_direct_deoxygenation"
        )
        return f"{entry}_CHOH_{removal}"
    if key == "methanol":
        return "COH_CHOH_CH2OH" if "COH*" in nodes else "CHO_CHOH_CH2OH"
    if key == "formaldehyde":
        return "CHO_to_HCHO_release"
    if key == "formate":
        return "HCOO_bidentate" if "HCOO*" in nodes else "OCHO_monodentate"
    if key == "co":
        return "COOH_to_CO_desorption"
    return "unclassified"


def _record_path(
    product_key: str,
    product_label: str,
    product_state: str,
    branch_start: str,
    path_index: int,
    path: Sequence[Mapping[str, Any]],
    energy_map: Mapping[str, float | None],
    potential_V: float,
    *,
    analysis_scope: str,
    site_key: str,
) -> dict[str, Any]:
    node_ids = ["CO2"] + [str(edge["to"]) for edge in path]
    overall = _path_metrics(path, energy_map, potential_V)
    post_co_path = _suffix(path, "CO*")
    post_co = (
        _path_metrics(post_co_path, energy_map, potential_V)
        if post_co_path else _path_metrics([], energy_map, potential_V)
    )
    post_co_core_path = _core_before_endpoint(post_co_path, product_state)
    post_co_core = (
        _path_metrics(post_co_core_path, energy_map, potential_V)
        if post_co_core_path else _path_metrics([], energy_map, potential_V)
    )
    branch_path = _suffix(path, branch_start)
    branch = (
        _path_metrics(branch_path, energy_map, potential_V)
        if branch_path else _path_metrics([], energy_map, potential_V)
    )
    branch_core_path = _core_before_endpoint(branch_path, product_state)
    branch_core = (
        _path_metrics(branch_core_path, energy_map, potential_V)
        if branch_core_path else _path_metrics([], energy_map, potential_V)
    )
    endpoint_complete = _finite(energy_map.get(product_state)) is not None
    return {
        "product_key": product_key,
        "product": product_label,
        "path_index": int(path_index),
        "path": " -> ".join(node_ids),
        "path_node_ids": node_ids,
        "mechanistic_branch": _mechanistic_branch(product_key, node_ids),
        "analysis_scope": analysis_scope,
        "site_consistency": (
            "single_site" if analysis_scope == "site_consistent"
            else "global_minimum_diagnostic"
        ),
        "site_key": site_key,
        "endpoint_complete": endpoint_complete,
        "overall_metrics": overall,
        "post_co_metrics": post_co,
        "post_co_core_metrics": post_co_core,
        "branch_metrics": branch,
        "branch_core_metrics": branch_core,
        **overall,
        **_metric_fields("post_CO", post_co),
        **_metric_fields("post_CO_adsorbed_core", post_co_core),
        **_metric_fields("branch", branch),
        **_metric_fields("branch_adsorbed_core", branch_core),
    }


def build_co2rr_pathway_summary(
    df: pd.DataFrame,
    pathway_key: str = "competitive_c1",
    states: list[str] | tuple[str, ...] | None = None,
    *,
    product_state_energies: Mapping[str, Any] | None = None,
    potential_V: float = 0.0,
) -> dict[str, Any]:
    preset = get_co2rr_preset(pathway_key)
    selected_states = list(states) if states is not None else list(preset.get("states", []))
    product_keys = list(preset.get("products", []))
    minima = select_co2rr_state_minima(df, pathway_key=pathway_key, states=selected_states)
    site_minima = select_co2rr_state_minima_by_site(
        df, pathway_key=pathway_key, states=selected_states,
    )
    minima_map = {
        str(row["pathway_state_id"]): dict(row)
        for _, row in minima.iterrows()
    } if not minima.empty else {}

    attempts: dict[str, int] = {}
    if isinstance(df, pd.DataFrame) and not df.empty and "adsorbate" in df.columns:
        attempts = (
            df.assign(_ads=df["adsorbate"].map(clean_co2rr_adsorbate))
            .groupby("_ads").size().to_dict()
        )

    nodes: list[dict[str, Any]] = [{
        "id": "CO2", "label": "CO2 + CHE reservoirs",
        "selected_energy_eV": 0.0, "energy_available": True,
        "node_type": "source", "analysis_scope": "global_minimum_diagnostic",
    }]
    for state in selected_states:
        state_id = normalize_co2rr_state(state)
        hit = minima_map.get(state_id)
        nodes.append({
            "id": state_id,
            "label": co2rr_state_label(state_id),
            "selected_energy_eV": None if hit is None else float(hit["selected_energy_eV"]),
            "energy_available": hit is not None,
            "node_type": "adsorbate",
            "analysis_scope": "global_minimum_diagnostic",
            "attempt_count": int(attempts.get(clean_co2rr_adsorbate(state_id), 0)),
            "site_label": "" if hit is None else str(hit.get("site_label", "")),
            "base_site_label": "" if hit is None else str(hit.get("base_site_label", "")),
            "relaxed_site": "" if hit is None else str(hit.get("relaxed_site", "")),
            "qa": "" if hit is None else str(hit.get("qa", "")),
            "structure_cif": "" if hit is None else str(hit.get("structure_cif", "")),
            "thermochemistry_level": (
                "unavailable" if hit is None
                else str(hit.get("thermochemistry_level", "electronic_energy_only"))
            ),
        })

    endpoint_map = _endpoint_records(product_keys, product_state_energies)
    nodes.extend(
        {**endpoint, "node_type": "product", "analysis_scope": "shared_endpoint"}
        for endpoint in endpoint_map.values()
    )
    endpoint_energy_map = {
        state: _finite(record.get("selected_energy_eV"))
        for state, record in endpoint_map.items()
    }
    global_energy_map = {
        "CO2": 0.0,
        **{
            state: _finite(row.get("selected_energy_eV"))
            for state, row in minima_map.items()
        },
        **endpoint_energy_map,
    }

    site_maps: dict[str, dict[str, float | None]] = {}
    site_rows: list[dict[str, Any]] = []
    if not site_minima.empty:
        for site, sub in site_minima.groupby("site_key", sort=True):
            energy_map = {"CO2": 0.0, **endpoint_energy_map}
            for _, row in sub.iterrows():
                state_id = str(row["pathway_state_id"])
                energy_map[state_id] = _finite(row.get("selected_energy_eV"))
                site_rows.append(dict(row))
            site_maps[str(site)] = energy_map

    raw_edges = [_normalize_edge(edge) for edge in preset.get("edges", [])]
    allowed_ids = (
        {normalize_co2rr_state(state) for state in selected_states}
        | {"CO2"} | set(endpoint_map)
    )
    edges = [
        edge for edge in raw_edges
        if edge["from"] in allowed_ids and edge["to"] in allowed_ids
    ]

    path_records: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []
    for product_key in product_keys:
        spec = CO2RR_PRODUCTS.get(product_key)
        if spec is None:
            continue
        paths = _enumerate_paths(edges, "CO2", spec.state_id)
        global_candidates = [
            _record_path(
                product_key, spec.label, spec.state_id, spec.branch_start_state,
                idx, path, global_energy_map, float(potential_V),
                analysis_scope="global_minimum_diagnostic", site_key="",
            )
            for idx, path in enumerate(paths)
        ]
        site_candidates = [
            _record_path(
                product_key, spec.label, spec.state_id, spec.branch_start_state,
                idx, path, energy_map, float(potential_V),
                analysis_scope="site_consistent", site_key=site,
            )
            for site, energy_map in site_maps.items()
            for idx, path in enumerate(paths)
        ]
        path_records.extend(global_candidates)
        path_records.extend(site_candidates)
        best_global = min(global_candidates, key=_path_sort_key) if global_candidates else None
        best_site = min(site_candidates, key=_path_sort_key) if site_candidates else None
        best = best_site or best_global
        if best is None:
            best = {
                "path": "", "mechanistic_branch": "unavailable", "site_key": "",
                "site_consistency": "unavailable", "endpoint_complete": False,
                "overall_metrics": {"complete": False, "missing_states": [spec.state_id]},
                "post_co_metrics": {}, "post_co_core_metrics": {},
                "branch_metrics": {}, "branch_core_metrics": {},
            }
        overall = best.get("overall_metrics", {})
        post_co = best.get("post_co_metrics", {})
        post_co_core = best.get("post_co_core_metrics", {})
        branch = best.get("branch_metrics", {})
        branch_core = best.get("branch_core_metrics", {})
        endpoint = endpoint_map.get(spec.state_id, {})
        ranking_eligible = bool(best_site is not None and overall.get("complete"))
        screening_metrics = branch if branch.get("complete") else branch_core
        screening_label, screening_value, screening_kind = _screening_bottleneck(
            screening_metrics
        )
        product_rows.append({
            "product_key": product_key,
            "product": spec.label,
            "best_path": best.get("path", ""),
            "best_branch": best.get("mechanistic_branch", "unavailable"),
            "site_key": best.get("site_key", ""),
            "site_consistency": best.get("site_consistency", "unavailable"),
            "candidate_path_count": int(len(site_candidates)),
            "complete_candidate_path_count": int(sum(
                bool(record.get("overall_metrics", {}).get("complete"))
                for record in site_candidates
            )),
            "path_complete": bool(overall.get("complete")),
            "endpoint_complete": bool(best.get("endpoint_complete")),
            "ranking_eligible": ranking_eligible,
            "overall_PDS": overall.get("pds", ""),
            "overall_PDS_delta_G_0_eV": overall.get("pds_delta_G_0_eV"),
            "overall_limiting_potential_V_vs_RHE": overall.get("limiting_potential_V"),
            "overall_max_delta_G_at_U_eV": overall.get("max_delta_G_at_U_eV"),
            "post_CO_complete": bool(post_co.get("complete")),
            "post_CO_PDS": post_co.get("pds", ""),
            "post_CO_PDS_delta_G_0_eV": post_co.get("pds_delta_G_0_eV"),
            "post_CO_limiting_potential_V_vs_RHE": post_co.get("limiting_potential_V"),
            "post_CO_adsorbed_core_complete": bool(post_co_core.get("complete")),
            "post_CO_adsorbed_core_PDS": post_co_core.get("pds", ""),
            "post_CO_adsorbed_core_PDS_delta_G_0_eV": post_co_core.get("pds_delta_G_0_eV"),
            "branch_start_state": spec.branch_start_state,
            "branch_complete": bool(branch.get("complete")),
            "branch_PDS": branch.get("pds", ""),
            "branch_PDS_delta_G_0_eV": branch.get("pds_delta_G_0_eV"),
            "branch_limiting_potential_V_vs_RHE": branch.get("limiting_potential_V"),
            "branch_adsorbed_core_complete": bool(branch_core.get("complete")),
            "branch_adsorbed_core_PDS": branch_core.get("pds", ""),
            "branch_adsorbed_core_PDS_delta_G_0_eV": branch_core.get("pds_delta_G_0_eV"),
            "branch_screening_scope": (
                "full_branch_with_endpoint" if branch.get("complete")
                else "adsorbed_core_only" if branch_core.get("complete")
                else "incomplete"
            ),
            "branch_screening_bottleneck": screening_label,
            "branch_screening_bottleneck_eV": screening_value,
            "branch_screening_bottleneck_type": screening_kind,
            "branch_screening_PDS": (
                branch.get("pds", "") if branch.get("complete")
                else branch_core.get("pds", "") if branch_core.get("complete")
                else ""
            ),
            "branch_screening_PDS_delta_G_0_eV": (
                branch.get("pds_delta_G_0_eV") if branch.get("complete")
                else branch_core.get("pds_delta_G_0_eV")
                if branch_core.get("complete") else None
            ),
            "chemical_bottleneck_eV": overall.get("chemical_bottleneck_eV"),
            "applied_potential_V_vs_RHE": float(potential_V),
            "missing_states": ";".join(overall.get("missing_states", [])),
            "confidence": (
                "incomplete_endpoint_or_state" if not ranking_eligible
                else "corrected_site_consistent"
                if str(endpoint.get("thermochemistry_level")) == "corrected_CHE"
                else "provisional_site_consistent_electronic_energy"
            ),
            "endpoint_warning": str(endpoint.get("warning", spec.warning)),
            "global_minimum_diagnostic_path": (
                "" if best_global is None else best_global.get("path", "")
            ),
            "global_minimum_diagnostic_complete": bool(
                best_global and best_global.get("overall_metrics", {}).get("complete")
            ),
            "global_minimum_diagnostic_overall_PDS_eV": (
                None if best_global is None
                else best_global.get("overall_metrics", {}).get("pds_delta_G_0_eV")
            ),
        })

    ranked = sorted(
        [row for row in product_rows if row["ranking_eligible"]],
        key=lambda row: (
            float(row["branch_screening_bottleneck_eV"])
            if row["branch_screening_bottleneck_eV"] is not None else float("inf"),
            float(row["post_CO_PDS_delta_G_0_eV"])
            if row["post_CO_PDS_delta_G_0_eV"] is not None else float("inf"),
            float(row["overall_max_delta_G_at_U_eV"])
            if row["overall_max_delta_G_at_U_eV"] is not None else float("inf"),
        ),
    )
    rank_map = {row["product_key"]: idx + 1 for idx, row in enumerate(ranked)}
    for row in product_rows:
        row["rank"] = rank_map.get(row["product_key"])
    product_rows.sort(
        key=lambda row: (row["rank"] is None, row["rank"] or 10**6, row["product_key"])
    )

    flat_edges: list[dict[str, Any]] = []
    for record in path_records:
        for edge in record.get("edges", []):
            flat_edges.append({
                **dict(edge),
                "product_key": record.get("product_key"),
                "path_index": record.get("path_index"),
                "analysis_scope": record.get("analysis_scope"),
                "site_key": record.get("site_key"),
            })

    return {
        "mode": "CO2RR_REACTION_NETWORK",
        "pathway_key": str(pathway_key),
        "pathway_label": str(preset.get("label", pathway_key)),
        "description": str(preset.get("description", "")),
        "potential_V_vs_RHE": float(potential_V),
        "nodes": nodes,
        "site_resolved_minima": site_rows,
        "edges": flat_edges,
        "paths": path_records,
        "product_summary": product_rows,
        "warning": CO2RR_WARNING,
        "ranking_basis": (
            "Primary ranking uses non-migrated, exact single-site paths. "
            "Product-branch PDS is compared first, then post-CO PDS and the overall uphill step. "
            "Global-minimum mixed-site paths are diagnostics only. Missing gas endpoints remain "
            "incomplete and are never silently estimated from an absent CIF."
        ),
        "energy_scale": {"vmin_eV": -2.5, "vcenter_eV": 0.0, "vmax_eV": 2.5},
    }


def pathway_summary_to_frame(summary: Mapping[str, Any] | None) -> pd.DataFrame:
    if not isinstance(summary, Mapping):
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "pathway": summary.get("pathway_key", ""),
            "pathway_label": summary.get("pathway_label", ""),
            **dict(node),
        }
        for node in summary.get("nodes", []) or []
    ])


def edge_summary_to_frame(summary: Mapping[str, Any] | None) -> pd.DataFrame:
    if not isinstance(summary, Mapping):
        return pd.DataFrame()
    return pd.DataFrame([dict(edge) for edge in summary.get("edges", []) or []])


def product_summary_to_frame(summary: Mapping[str, Any] | None) -> pd.DataFrame:
    if not isinstance(summary, Mapping):
        return pd.DataFrame()
    return pd.DataFrame([
        dict(row) for row in summary.get("product_summary", []) or []
    ])


def write_co2rr_pathway_summary(
    summary: Mapping[str, Any], out_dir: str | Path,
):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    state_csv = root / "results_co2rr_state_energy_map.csv"
    site_csv = root / "results_co2rr_site_resolved_state_map.csv"
    edge_csv = root / "results_co2rr_edge_free_energies.csv"
    product_csv = root / "results_co2rr_product_favorability.csv"
    json_path = root / "results_co2rr_reaction_network.json"
    pathway_summary_to_frame(summary).to_csv(
        state_csv, index=False, float_format="%.6f",
    )
    pd.DataFrame(summary.get("site_resolved_minima", []) or []).to_csv(
        site_csv, index=False, float_format="%.6f",
    )
    edge_summary_to_frame(summary).to_csv(
        edge_csv, index=False, float_format="%.6f",
    )
    product_summary_to_frame(summary).to_csv(
        product_csv, index=False, float_format="%.6f",
    )
    payload = dict(summary)
    payload["output_files"] = {
        "state_csv": str(state_csv.resolve()),
        "site_resolved_state_csv": str(site_csv.resolve()),
        "edge_csv": str(edge_csv.resolve()),
        "product_csv": str(product_csv.resolve()),
        "json": str(json_path.resolve()),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return state_csv, json_path
