import streamlit as st
from pathlib import Path
import pandas as pd
from io import BytesIO, StringIO
import zipfile
import re
import uuid
from datetime import datetime, timezone
import os
import json
import tempfile
import inspect
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.data import chemical_symbols
import py3Dmol
import streamlit.components.v1 as components

from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor

from ocp_app.core import seeds
from ocp_app.core.anchors.CHE_mode import (
    run_metal_che,
    run_oxide_che,
    run_metal_co2rr_che,
    run_oxide_co2rr_che,
    run_metal_oer_che,    # OER competition support
    run_oxide_oer_che,    # OER competition support
)
from ocp_app.core.anchors.voc_mode import (
    run_metal_voc_proxy,
    run_oxide_voc_proxy,
)
from ocp_app.core.voc_registry import VOC_PRESETS, get_voc_preset
from ocp_app.core.co2rr_registry import (
    CO2RR_PATHWAY_ORDER,
    CO2RR_WARNING,
    all_co2rr_states,
    co2rr_site_allowed,
    co2rr_state_label,
    get_co2rr_preset,
)
from ocp_app.core.co2rr_pathway import (
    build_co2rr_pathway_summary,
    edge_summary_to_frame as co2rr_edge_summary_to_frame,
    pathway_summary_to_frame as co2rr_pathway_summary_to_frame,
    product_summary_to_frame as co2rr_product_summary_to_frame,
    write_co2rr_pathway_summary,
)
from ocp_app.core.voc_pathway import (
    select_voc_state_minima,
    build_voc_pathway_summary,
    pathway_summary_to_frame,
    write_voc_pathway_summary,
)
from ocp_app.ui.voc_pathway_view import (
    render_voc_pathway,
    render_pathway_support_table,
)
from ocp_app.core.cifgen import (
    BulkSource,
    BulkSpec,
    RatioTuneSpec,
    generate_bulk,
    scale_xy_and_tune_ratio,
)
from ocp_app.core import run_history as rh
from ocp_app.core.state import (
    _init_state,
    _clear_ml_cache,
    _atoms_signature,
    _reset_prepared_from_working,
    _ensure_prepared_uptodate,
    _push_prepared_update,
    _jsonable,
    normalize_mp_id,
    infer_default_tune_elements,
    ensure_tune_defaults_from_structure,
)
from ocp_app.core.conditioning import (
    _cluster_z_layers,
    _suggest_conditioning_params,
    _get_conditioned_slab,
)
from ocp_app.core.anchors.oxide_her import (
    _oxide_o_based_ads_position_compat,
    _pbc_min_image_xy_distance_sq,
    _top_surface_o_indices,
    _generate_oxide_her_oanchor_sites,
    _project_single_oxide_her_site_to_otop,
    _project_oxide_her_sites_to_otop,
)
from ocp_app.core.structure_ops import (
    atoms_to_cif_string,
    atoms_to_cif_bytes,
    atoms_to_xyz_string,
    _recenter_slab_z_into_cell,
    add_vacuum_z,
    set_pbc_z,
    repeat_xy,
    _surface_xy_lengths,
    _suggest_minimal_xy_repeat,
    slab_thickness_z,
    suggest_active_region_crop,
    crop_top_slab_window,
)
from ocp_app.ui.viewers import _render_min_dist_panel, show_atoms_3d
try:
    from ocp_app.core.slab_reduction import get_slab_reduction_presets, reduce_slab_symmetrically
except Exception:
    get_slab_reduction_presets = None
    reduce_slab_symmetrically = None

from ocp_app.core.oxide_surface_rules import (
    infer_oxide_family_from_atoms,
    _classify_surface_exposure,
    _flip_slab_z_keep_cell,
    _normalize_oxide_candidate_top_surface,
    _top_surface_o_anchor_sites_with_spacing,
    _build_oxide_oh_terminated_candidate,
    _expand_oxide_surface_state_candidates,
    _oxide_candidate_rank_key,
    _pick_best_oxide_slab_candidate,
    _oxide_mode_keep_candidate,
    _normalize_oxide_candidate_oer_top_surface,
    _oxide_oer_candidate_rank_key,
    _oxide_oer_cation_metrics,
)
from ocp_app.core.surface_families import _infer_interface_surface_family
from ocp_app.core.slabify import (
    _DEFAULT_SLAB_MIN_THICKNESS,
    _DEFAULT_SLAB_MAX_CANDIDATES,
    _pick_best_slab_candidate_auto,
    _pick_best_slab_candidate,
    _miller_sort_key,
    _format_hkl_label,
    _reduce_hkl_by_gcd,
    _enumerate_low_index_millers,
    _recommended_step2_facets,
    _facet_choices_for_scope,
    _vacuum_target_from_ui,
    _normalize_voc_oxide_candidate_top_surface,
    _voc_oxide_candidate_rank_key,
    slabify_from_bulk,
)
from ocp_app.core.postprocess import (
    split_reliable_unreliable,
    _normalize_text_series,
    co2rr_apply_qa_policy,
    co2rr_split_by_qa,
    oxygen_apply_qa_policy,
    oxygen_split_by_qa,
    voc_apply_qa_policy,
    voc_split_by_qa,
    voc_split_candidates_diagnostics_rejected,
    co2rr_dedupe_candidates,
    build_compact_table,
    annotate_site_transitions,
    summarize_site_transitions,
    _make_ml_screen_key,
    _build_ml_compact_df,
)
from ocp_app.core.preview import (
    build_adsorbate_preview_slab,
    export_zip_of_struct_map,
)
from ocp_app.core.reporting import (
    build_llm_payload,
    call_llm_interpreter,
)
from ocp_app.core.co2rr_air_summary import (
    build_co2rr_air_summary,
    co2rr_air_summary_to_frame,
    annotate_co2rr_air_summary,
)
from ocp_app.core.structure_check import validate_structure
from ocp_app.core.structure_engineering import (
    analyze_parent_slab,
    build_adatom_candidate_at_site,
    build_substitution_candidate_at_index,
    build_vacancy_candidate_at_index,
    candidate_summary_records,
    detect_selectable_adatom_sites,
    enumerate_adatom_candidates,
    enumerate_substitution_candidates,
    enumerate_vacancy_candidates,
    export_engineered_candidates_zip,
    structure_content_signature,
    substitution_radius_diagnostics,
    infer_structure_material_class,
    oxidation_state_options,
    suggested_substitution_oxidation_states,
    substitution_geometry_diagnostics,
)
from ocp_app.ui.structure_picker import (
    eligible_atom_indices,
    render_adatom_site_picker,
    render_atom_picker,
)
from ocp_app.core.ads_sites import _oxide_o_based_ads_position
from ocp_app.core.ads_sites import oxide_surface_seed_position, expand_oxide_channels_for_adsorbate, ANION_SYMBOLS, detect_oxide_oer_cation_sites
from ocp_app.core.ads_sites import (
    detect_metal_111_sites,
    detect_oxide_surface_sites,
    select_representative_sites,
    AdsSite,
    generate_slab_ads_series,
    generate_candidate_sites,
)

# ML screening
HAS_ADSORML = True
ADSORML_IMPORT_ERR = None
try:
    from ocp_app.core.adsorbml_lite_screening import (
        ScreeningSettings,
        screen_sites_adsorbml_lite,
        union_topk_sites,
        relax_slab_chgnet,
    )
except Exception as e:
    HAS_ADSORML = False
    ADSORML_IMPORT_ERR = str(e)

from collections import Counter
from functools import reduce
from math import gcd

# Optional slabify (pymatgen surface)
HAS_SLABIFY = True
SLABIFY_IMPORT_ERR = None
try:
    from pymatgen.core.surface import SlabGenerator
    try:
        from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
    except Exception:
        get_symmetrically_distinct_miller_indices = None
except Exception as e:
    HAS_SLABIFY = False
    SLABIFY_IMPORT_ERR = str(e)
    get_symmetrically_distinct_miller_indices = None

# ---------------- App config ----------------
st.set_page_config(page_title="OCP App (HAPLAB)", layout="wide")
st.title("Surface Adsorption and Geometry Evaluator(SAGE) — HER / CO₂RR / OER (HAPLAB v1.0)")

# Surface Adsorption and Geometry Evaluator(SAGE) — HER / CO₂RR / OER / VOCs (HAPLAB v1.2-voc-alpha

R_PH = 0.0591  # eV per pH
GLOBAL_SEED = 42
RATIO_SUM = 10

CO2RR_MIGRATION_DISP_THRESH_A = 0.8  # Å; adsorbate lateral displacement threshold to flag migration


# ---------------- Session State ----------------

_init_state()

# Initialize session-only run history UI/state (cleared when app refreshes/closes).
# run_history.py (core module) is expected to be session_state-only.
try:
    # Preferred API (provided by the run_history.py we drafted)
    if hasattr(rh, "ensure_history_state"):
        rh.ensure_history_state(max_items=10)
    # Backward-compat API (if you later rename)
    elif hasattr(rh, "init_history_state"):
        rh.init_history_state(max_items=10)
except Exception:
    # Never block the main app if history is unavailable.
    pass


# ---- Convenience helpers (mp-id normalize, tuning defaults, min-distance panel) ----

def _safe_float(val, default=np.nan):
    try:
        return float(val)
    except Exception:
        return float(default)


def _read_result_csv_safely(csv_path, *, context: str = "result"):
    """Read a SAGE CSV without allowing empty/incomplete files to crash the UI.

    Returns
    -------
    (dataframe, diagnostic)
        dataframe is None when the file is missing, zero-byte, headerless, or
        unreadable. A header-only CSV returns an empty DataFrame with a
        diagnostic so the caller can report that no calculation rows survived.
    """
    diagnostic = {
        "context": str(context),
        "csv_path": "" if csv_path is None else str(csv_path),
        "exists": False,
        "is_file": False,
        "size_bytes": None,
        "status": "unknown",
        "error": None,
    }

    if csv_path is None or not str(csv_path).strip():
        diagnostic["status"] = "missing_path"
        diagnostic["error"] = "The calculation backend did not return a CSV path."
        return None, diagnostic

    try:
        p = Path(str(csv_path)).expanduser()
        diagnostic["csv_path"] = str(p)
        diagnostic["exists"] = bool(p.exists())
        diagnostic["is_file"] = bool(p.is_file())
        if not p.is_file():
            diagnostic["status"] = "missing_file"
            diagnostic["error"] = "The returned CSV path does not point to a file."
            return None, diagnostic

        size = int(p.stat().st_size)
        diagnostic["size_bytes"] = size
        if size <= 0:
            diagnostic["status"] = "zero_byte_file"
            diagnostic["error"] = "The result CSV exists but contains zero bytes."
            return None, diagnostic

        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError as exc:
            diagnostic["status"] = "no_columns"
            diagnostic["error"] = str(exc) or "No columns to parse from the result CSV."
            return None, diagnostic
        except pd.errors.ParserError as exc:
            diagnostic["status"] = "parser_error"
            diagnostic["error"] = str(exc)
            return None, diagnostic
        except Exception as exc:
            diagnostic["status"] = "read_error"
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            return None, diagnostic

        diagnostic["n_rows"] = int(len(df))
        diagnostic["n_columns"] = int(len(df.columns))
        diagnostic["columns"] = [str(c) for c in df.columns]

        if len(df.columns) == 0:
            diagnostic["status"] = "no_columns"
            diagnostic["error"] = "The CSV was read but no columns were present."
            return None, diagnostic

        if df.empty:
            diagnostic["status"] = "header_only"
            diagnostic["error"] = (
                "The CSV contains column headers but no result rows. "
                "All candidate attempts may have failed or been filtered out."
            )
            return df, diagnostic

        diagnostic["status"] = "ok"
        return df, diagnostic

    except Exception as exc:
        diagnostic["status"] = "inspection_error"
        diagnostic["error"] = f"{type(exc).__name__}: {exc}"
        return None, diagnostic


def _render_empty_result_diagnostic(csv_diag, *, meta=None):
    """Render a concise diagnostic panel for missing or empty calculation CSVs."""
    diag = dict(csv_diag or {})
    status = str(diag.get("status", "unknown"))
    path_text = str(diag.get("csv_path", "") or "(not returned)")
    size_text = diag.get("size_bytes")
    size_label = "unknown" if size_text is None else f"{int(size_text)} bytes"

    st.error(
        "The calculation backend did not produce a readable result table. "
        "The app stopped before post-processing instead of raising a pandas EmptyDataError."
    )
    st.write(f"- CSV status: **{status}**")
    st.write(f"- CSV path: `{path_text}`")
    st.write(f"- File size: **{size_label}**")
    if diag.get("error"):
        st.write(f"- Diagnostic: {diag.get('error')}")

    st.info(
        "This usually means that no candidate row was written: every site may have "
        "failed during placement/relaxation, all rows may have been rejected upstream, "
        "the run may have been interrupted, or a stale output file may have been reused."
    )

    with st.expander("Show backend metadata and CSV diagnostic", expanded=False):
        st.json({
            "csv_diagnostic": _jsonable(diag),
            "backend_meta": _jsonable(meta) if isinstance(meta, dict) else meta,
        })


def _coerce_bool_series(s: pd.Series, default: bool = False) -> pd.Series:
    """Coerce mixed bool/string/numeric Series to boolean values."""
    if s is None:
        return pd.Series(dtype=bool)

    def _one(x):
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return bool(default)
        xs = str(x).strip().lower()
        if xs in {"true", "1", "yes", "y", "valid", "ok"}:
            return True
        if xs in {"false", "0", "no", "n", "invalid", "bad", "nan", "none", ""}:
            return False
        try:
            return bool(int(float(xs)))
        except Exception:
            return bool(default)

    return s.map(_one).astype(bool)

def _oer_site_adsorbate_compact(df: pd.DataFrame) -> pd.DataFrame:
    """Compact OER site-level rows without destroying OH/O/OOH triplets.

    CO2RR-style deduplication can pick the single best O, OH, and OOH across
    different sites, which breaks OER thermodynamic interpretation.  For OER,
    keep one lowest-energy height per (oer_base_site_label, adsorbate).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "adsorbate" not in work.columns:
        return work
    if "oer_base_site_label" not in work.columns:
        if "site_label" in work.columns:
            work["oer_base_site_label"] = work["site_label"].astype(str).str.replace(r":h[0-9.]+$", "", regex=True)
        else:
            work["oer_base_site_label"] = "oer_site"
    e_col = "ΔG_ads (eV)" if "ΔG_ads (eV)" in work.columns else ("ΔE_ads_user (eV)" if "ΔE_ads_user (eV)" in work.columns else None)
    if e_col is None:
        return work
    work["_oer_energy"] = pd.to_numeric(work[e_col], errors="coerce")
    work = work[np.isfinite(work["_oer_energy"])].copy()
    if work.empty:
        return work.drop(columns=["_oer_energy"], errors="ignore")
    rows = []
    ads_order = {"OH": 0, "O": 1, "OOH": 2}
    for (_base, _ads), sub in work.groupby(["oer_base_site_label", work["adsorbate"].astype(str).str.upper()], dropna=False):
        # Match backend summary policy: lowest valid ΔG per site/intermediate.
        ridx = sub["_oer_energy"].idxmin()
        rows.append(work.loc[ridx])
    out = pd.DataFrame(rows).drop(columns=["_oer_energy"], errors="ignore")
    if not out.empty:
        out["_ads_order"] = out["adsorbate"].astype(str).str.upper().map(lambda x: ads_order.get(x, 9))
        out = out.sort_values(["oer_base_site_label", "_ads_order", "site_label"], kind="mergesort").drop(columns=["_ads_order"], errors="ignore").reset_index(drop=True)
    return out


def _split_oxide_d2_primary_reliability(df: pd.DataFrame):
    """Reliability split for oxide D2-primary HER rows.

    D2 intentionally allows lateral relaxation within the local metal-cation
    basin. Therefore H_lateral_disp(Å) is retained as audit metadata but is not
    used as an automatic rejection criterion here. The reliable/unreliable split
    is based on the D2 final-state validity produced by oxide_descriptor.py.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df, df

    out = df.copy()
    valid_mask = pd.Series(False, index=out.index)

    # Preferred source: explicit descriptor validity from oxide_descriptor.py.
    if "D2_descriptor_valid" in out.columns:
        valid_mask = _coerce_bool_series(out["D2_descriptor_valid"], default=False)
        valid_mask.index = out.index
    else:
        # Fallback for older result files: infer from final metal-centered state.
        valid_labels = {
            "metal_top", "metal_bridge", "metal_hollow", "metal_adjacent",
            "top", "bridge", "hollow", "fcc", "hcp",
        }
        for col in ("D2_binding_class", "binding_class", "final_site_kind", "relaxed_site"):
            if col in out.columns:
                vals = out[col].astype(str).str.strip().str.lower()
                valid_mask |= vals.isin(valid_labels)

    # Hard invalid labels stay invalid even if a fallback label is ambiguous.
    invalid_labels = {
        "o_bound", "o-h_migrated", "oh_migrated", "anion", "anion_ontop",
        "desorbed", "subsurface", "slab_collapsed", "unresolved", "other",
    }
    for col in ("D2_binding_class", "binding_class", "final_site_kind", "relaxed_site", "migration_destination"):
        if col in out.columns:
            vals = out[col].astype(str).str.strip().str.lower()
            valid_mask &= ~vals.isin(invalid_labels)

    # Keep extreme-energy blowups out of the reliable table, but do not reject
    # merely because H_lateral_disp exceeds the legacy HER displacement cutoff.
    for ecol in ("ΔE_H_user (eV)", "ΔG_H (eV)"):
        if ecol in out.columns:
            ev = pd.to_numeric(out[ecol], errors="coerce")
            valid_mask &= ~(ev.abs() > 10.0)

    df_rel = out[valid_mask].copy()
    df_unrel = out[~valid_mask].copy()
    return df_rel, df_unrel


def _pick_oxide_her_representatives(df_rel: pd.DataFrame):
    """Return representative oxide-HER sites from reliable rows.

    Definitions:
      - occupied representative: most stabilized reliable H* site (minimum ΔG_H)
      - HER-optimal representative: reliable site with ΔG_H closest to 0 eV

    Non-duplicate rows are preferred when the column is available.
    """
    if not isinstance(df_rel, pd.DataFrame) or df_rel.empty:
        return {}

    energy_col = None
    for cand in ("ΔG_H(U,pH) (eV)", "ΔG_H (eV)"):
        if cand in df_rel.columns:
            energy_col = cand
            break
    if energy_col is None:
        return {}

    work = df_rel.copy()
    if "is_duplicate" in work.columns:
        try:
            nondup = work[pd.to_numeric(work["is_duplicate"], errors="coerce").fillna(0).astype(int) == 0].copy()
            if not nondup.empty:
                work = nondup
        except Exception:
            pass

    work["_rep_energy"] = pd.to_numeric(work[energy_col], errors="coerce")
    work = work[np.isfinite(work["_rep_energy"])].copy()
    if work.empty:
        return {}

    def _pack(row):
        row = row.copy()
        return {
            "site_label": str(row.get("site_label", "NA")),
            "relaxed_site": str(row.get("relaxed_site", row.get("final_site_kind", "NA"))),
            "binding_class": str(row.get("binding_class", row.get("D2_binding_class", ""))),
            "energy": float(row.get("_rep_energy", np.nan)),
            "row": row,
            "energy_col": energy_col,
        }

    idx_occ = work["_rep_energy"].idxmin()
    idx_opt = work["_rep_energy"].abs().idxmin()

    return {
        "occupied": _pack(work.loc[idx_occ]),
        "her_optimal": _pack(work.loc[idx_opt]),
        "n_candidates": int(len(work)),
        "energy_col": energy_col,
    }

def _normalize_hkl_sign(hkl):
    vals = [int(v) for v in tuple(hkl)]
    for v in vals:
        if v != 0:
            if v < 0:
                vals = [-x for x in vals]
            break
    return tuple(vals)


def _canonical_hkl_family(hkl):
    try:
        red = tuple(int(v) for v in _reduce_hkl_by_gcd(tuple(int(x) for x in hkl)))
    except Exception:
        red = tuple(int(x) for x in hkl)
    return _normalize_hkl_sign(red)


def _augment_oxide_low_index_facets(facet_choices, scope_label):
    base = []
    seen = set()
    for hkl in (facet_choices or []):
        can = _canonical_hkl_family(hkl)
        if can not in seen:
            seen.add(can)
            base.append(can)

    scope_s = str(scope_label or "")
    force = []

    # Always expose the c-axis low-index family so literature facets such as
    # (002)/(003) can be selected via the reduced (001) family in the UI.
    if scope_s in {"Recommended oxide facets", "Recommended facets", "Low-index facets (up to 1)", "Extended facets (up to 2)"}:
        force.extend([(0, 0, 1)])

    if scope_s in {"Low-index facets (up to 1)", "Extended facets (up to 2)"}:
        force.extend([
            (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ])

    if scope_s == "Extended facets (up to 2)":
        force.extend([
            (1, 0, 2), (0, 1, 2), (1, 1, 2),
            (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 2, 0),
            (2, 1, 1), (1, 2, 1),
        ])

    for hkl in force:
        can = _canonical_hkl_family(hkl)
        if can not in seen:
            seen.add(can)
            base.append(can)

    return base


def _format_facet_label_with_alias(hkl):
    can = _canonical_hkl_family(hkl)
    label = _format_hkl_label(can)
    if can == (0, 0, 1):
        return f"{label} [00l family; literature 002/003 alias]"
    return label


def _viewer_safe_label(s: object) -> str:
    """Mirror voc_mode._safe_label for post-run CIF lookup in the UI."""
    out = str(s or "").replace("*", "star").replace("+", "__plus__")
    out = out.replace("/", "_").replace(" ", "_").replace(":", "__")
    return out




def _viewer_clean_ads_token(val: object) -> str:
    s = str(val or "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    return s.replace("*", "").replace("+", "__plus__").replace("/", "_").replace(" ", "_").replace(":", "__")


def _viewer_row_expected_state_tokens(row: dict) -> list[str]:
    """Return exact-ish state tokens expected in a relaxed CIF basename.

    This prevents the relaxed-structure viewer from opening a fallback H* CIF
    when a VOC product row such as CH3CH2OH* has no usable explicit path.
    """
    vals = []
    for col in ("descriptor_state", "adsorbate", "state_label"):
        v = str(row.get(col, "") or "").strip()
        if v and v.lower() not in {"nan", "none", "null"}:
            vals.append(v)
    out = []
    for v in vals:
        safe = _viewer_safe_label(v)
        clean = _viewer_clean_ads_token(v)
        for t in (safe, clean):
            t = str(t or "").strip().lower()
            if t and t not in out:
                out.append(t)
    return out


def _viewer_path_matches_row_state(path_obj, row: dict) -> bool:
    """Guard against cross-linked relaxed CIFs in VOC viewer.

    Older/fallback path guessing can otherwise match user_<site>_H.cif for a
    CH3CH2OH* row because the row shares the same site_label.  For VOC rows,
    require the CIF basename to contain the requested adsorbate/state token.
    """
    try:
        name = Path(path_obj).name.lower()
    except Exception:
        return False

    mode = str(row.get("mode", "VOC") or "VOC").strip().upper()
    has_voc_cols = any(k in row for k in ("descriptor_state", "adsorbate", "target_voc"))
    if mode != "VOC" and not has_voc_cols:
        return True

    tokens = _viewer_row_expected_state_tokens(row)
    if not tokens:
        return True

    # H* is a special single-atom descriptor.  Do not allow it to satisfy or be
    # selected for carbon-containing VOC product/intermediate rows.
    ads_clean = _viewer_clean_ads_token(row.get("adsorbate", row.get("descriptor_state", ""))).lower()
    if ads_clean == "h":
        return ("_h_" in name) or name.endswith("_h.cif") or ("hstar" in name) or ("h_metal_che_like" in name)

    # For normal VOC species, prefer exact safe token with 'star'.  This avoids
    # CH3CH2O matching CH3CH2OH by substring.
    for tok in tokens:
        if tok.endswith("star") and tok in name:
            return True

    # Fallback to delimiter-aware clean token.
    for tok in tokens:
        if tok.endswith("star"):
            continue
        if re.search(r"(^|_)" + re.escape(tok) + r"(_|\.|$)", name):
            return True

    return False

def _resolve_relaxed_structure_path(row: pd.Series | dict, csv_path: str | Path | None = None):
    """Best-effort resolver for post-run relaxed structure CIF paths.

    Priority:
      1) explicit row['structure_cif'] when present
      2) robust relative/basename fallback against the current CSV folder
      3) VOC/HER filename patterns generated by backend _safe_label()

    This resolver must be tolerant of older CSVs because rejected VOC rows from
    earlier builds may not include structure_cif even though the CIF was saved.
    """
    row = row if isinstance(row, dict) else dict(row)

    root = None
    if csv_path:
        try:
            root = Path(str(csv_path)).expanduser().resolve().parent
        except Exception:
            try:
                root = Path(str(csv_path)).expanduser().parent
            except Exception:
                root = None

    candidates = []

    def _add_candidate(p):
        if p is None:
            return
        try:
            ps = str(p).strip()
        except Exception:
            return
        if not ps or ps.lower() in {"nan", "none", "null"}:
            return
        try:
            pp = Path(ps).expanduser()
        except Exception:
            return
        candidates.append(pp)
        if root is not None:
            # path relative to CSV root
            if not pp.is_absolute():
                candidates.append(root / pp)
            # basename fallback handles absolute paths produced in a temporary
            # run directory when only the exported CSV was moved/copied.
            candidates.append(root / pp.name)
            candidates.append(root / "sample" / "sites" / pp.name)

    # Explicit backend columns first.
    for col in ("structure_cif", "relaxed_structure_cif", "final_structure_cif", "cif_path"):
        if col in row:
            _add_candidate(row.get(col))

    site_label = str(row.get("site_label", "")).strip()
    site_variants = []
    if site_label:
        site_variants.extend([site_label, site_label.replace(":", "__"), _viewer_safe_label(site_label)])
    # de-duplicate preserving order
    site_variants = list(dict.fromkeys([x for x in site_variants if x]))

    state_vals = []
    for col in ("descriptor_state", "adsorbate", "state_label"):
        val = str(row.get(col, "")).strip()
        if val:
            state_vals.append(val)
            state_vals.append(val.replace("*", ""))
            state_vals.append(val.upper().replace("*", ""))
            state_vals.append(_viewer_safe_label(val))
    state_vals = list(dict.fromkeys([x for x in state_vals if x and x.lower() not in {"nan", "none"}]))

    seed_policy = str(row.get("ech_seed_policy", "")).strip()
    seed_suffixes = [""]
    if seed_policy and seed_policy.lower() not in {"nan", "none", "default"}:
        seed_suffixes.append("_" + _viewer_safe_label(seed_policy))

    if root is not None and site_variants:
        sites_dir = root / "sample" / "sites"
        for site in site_variants:
            for state in state_vals:
                for suff in seed_suffixes:
                    candidates.append(sites_dir / f"user_{site}_{state}{suff}.cif")
            candidates.append(sites_dir / f"user_{site}_H.cif")
            candidates.append(sites_dir / f"user_{site}_H_metal_CHE_like.cif")

        # Last-resort glob fallback for legacy/sanitized names.
        # Keep this state-specific; never fall back to all files for a site,
        # because that can display H* for CH3CH2OH*/CH3CH2O* rows.
        try:
            if sites_dir.is_dir():
                expected_tokens = _viewer_row_expected_state_tokens(row)
                for site in site_variants:
                    for state in expected_tokens:
                        candidates.extend(sorted(sites_dir.glob(f"user_{site}_{state}*.cif")))
                        if state.endswith("star"):
                            candidates.extend(sorted(sites_dir.glob(f"user_{site}_{state.replace('star','')}*.cif")))
        except Exception:
            pass

    seen = set()
    for p in candidates:
        try:
            pp = Path(p)
            key = str(pp)
            if key in seen:
                continue
            seen.add(key)
            if pp.is_file() and _viewer_path_matches_row_state(pp, row):
                return pp
        except Exception:
            pass

    return None


def _format_relaxed_view_option(row: pd.Series | dict, is_her: bool = True) -> str:
    row = row if isinstance(row, dict) else dict(row)
    site_label = str(row.get("site_label", "?"))
    relaxed_site = str(row.get("relaxed_site", row.get("final_site_kind", "?")))
    reliability = str(row.get("reliability", ""))
    if is_her:
        dg = _safe_float(row.get("ΔG_H(U,pH) (eV)", row.get("ΔG_H (eV)", np.nan)))
        return f"{site_label} | final={relaxed_site} | ΔG_H(U,pH)={dg:.3f} eV | {reliability}"
    ads = str(row.get("adsorbate", "?"))
    qa = str(row.get("qa", reliability or ""))
    if str(row.get("mode", "")).upper() == "CO2RR":
        value = _safe_float(row.get("ΔE_ads_user (eV)", np.nan))
        label = "ΔE_ads"
    else:
        value = _safe_float(row.get("ΔG_ads (eV)", row.get("ΔE_proxy (eV)", row.get("ΔE_ads_user (eV)", np.nan))))
        label = "ΔG_ads" if "ΔG_ads (eV)" in row else ("ΔE_proxy" if "ΔE_proxy (eV)" in row else "ΔE_ads")
    return f"{ads} | {site_label} | final={relaxed_site} | {label}={value:.3f} eV | {qa}"

def _cluster_z_layers_simple(atoms, tol: float = 0.8):
    pos = np.asarray(atoms.get_positions(), dtype=float)
    if pos.size == 0:
        return []
    order = np.argsort(pos[:, 2])
    layers = []
    current = [int(order[0])]
    z_ref = float(pos[order[0], 2])
    for idx in order[1:]:
        idx = int(idx)
        z_val = float(pos[idx, 2])
        if abs(z_val - z_ref) <= float(tol):
            current.append(idx)
            z_ref = float(np.mean(pos[current, 2]))
        else:
            layers.append(sorted(current))
            current = [idx]
            z_ref = z_val
    if current:
        layers.append(sorted(current))
    return layers


def _apply_top_free_layer_constraint(atoms, top_free_layers: int = 2, layer_tol: float = 0.8):
    a = atoms.copy()
    layers = _cluster_z_layers_simple(a, tol=float(layer_tol))
    n_layers = len(layers)
    n_free_layers = max(1, min(int(top_free_layers), max(1, n_layers)))
    free_idx = set()
    for layer in layers[-n_free_layers:]:
        free_idx.update(int(i) for i in layer)
    fixed_idx = [int(i) for i in range(len(a)) if i not in free_idx]
    if fixed_idx:
        a.set_constraint(FixAtoms(indices=fixed_idx))
    meta = {
        "n_layers": int(n_layers),
        "free_top_layers": int(n_free_layers),
        "fixed_atoms": int(len(fixed_idx)),
        "free_atoms": int(len(a) - len(fixed_idx)),
        "fixed_indices": fixed_idx,
    }
    return a, meta


def _extract_relaxed_atoms_from_result(result):
    if hasattr(result, "get_positions"):
        return result
    if isinstance(result, (list, tuple)):
        for item in result:
            if hasattr(item, "get_positions"):
                return item
    if isinstance(result, dict):
        for key in ("atoms", "relaxed_atoms", "slab_relaxed", "slab", "structure"):
            item = result.get(key)
            if hasattr(item, "get_positions"):
                return item
    return None


def _run_chgnet_slab_relax_adaptive(atoms, *, fmax: float, max_steps: int, seed: Optional[int] = None):
    if (not HAS_ADSORML) or ("relax_slab_chgnet" not in globals()):
        raise RuntimeError(f"CHGNet slab relax unavailable: {ADSORML_IMPORT_ERR or 'not imported'}")
    fn = relax_slab_chgnet
    sig = inspect.signature(fn)
    kwargs = {}
    if "fmax" in sig.parameters:
        kwargs["fmax"] = float(fmax)
    if "max_steps" in sig.parameters:
        kwargs["max_steps"] = int(max_steps)
    elif "steps" in sig.parameters:
        kwargs["steps"] = int(max_steps)
    elif "nsteps" in sig.parameters:
        kwargs["nsteps"] = int(max_steps)
    if seed is not None and "seed" in sig.parameters:
        kwargs["seed"] = int(seed)
    try:
        result = fn(atoms.copy(), **kwargs)
    except TypeError:
        result = fn(atoms.copy())
    relaxed = _extract_relaxed_atoms_from_result(result)
    if relaxed is None:
        raise RuntimeError("Could not extract relaxed Atoms from relax_slab_chgnet(...) result.")
    relaxed = relaxed.copy()
    try:
        relaxed.set_constraint()
    except Exception:
        pass
    return relaxed


def _get_oxide_her_constrained_prerelaxed_slab(
    atoms, *, enable: bool, top_free_layers: int, layer_tol: float, fmax: float, max_steps: int, seed: Optional[int] = None
):
    if not bool(enable):
        return atoms, None
    prepared, meta = _apply_top_free_layer_constraint(atoms, top_free_layers=int(top_free_layers), layer_tol=float(layer_tol))
    relaxed = _run_chgnet_slab_relax_adaptive(prepared, fmax=float(fmax), max_steps=int(max_steps), seed=seed)
    meta = dict(meta)
    meta.update({
        "enabled": True,
        "fmax": float(fmax),
        "max_steps": int(max_steps),
        "seed": None if seed is None else int(seed),
    })
    return relaxed, meta



def _counter_to_compact_formula(counter: Counter | dict | None) -> str:
    """Format an element counter as a compact formula-like string for QC display."""
    if not counter:
        return ""
    items = []
    for el in sorted(counter.keys()):
        try:
            n = int(counter[el])
        except Exception:
            n = counter[el]
        items.append(f"{el}{n}")
    return " ".join(items)


def _common_like_bottom_fix_indices(atoms, n_fix_layers: int = 2, z_tol: float = 0.25) -> list[int]:
    """Replicate common.py partial-relaxation bottom-layer fixing logic for UI/QC audit."""
    if atoms is None or len(atoms) == 0:
        return []
    z = np.asarray(atoms.get_positions()[:, 2], dtype=float)
    zs = np.unique(np.round(z, 3))
    zs.sort()
    cuts = zs[: min(int(n_fix_layers), len(zs))]
    fixed = []
    for i, atom in enumerate(atoms):
        if atom.symbol == "H":
            continue
        if np.any(np.abs(float(atom.position[2]) - cuts) < float(z_tol)):
            fixed.append(int(i))
    return fixed


def _oxide_her_pre_run_audit(atoms, *, n_fix_layers: int = 2, z_tol: float = 0.25, layer_tol: float = 0.8) -> dict:
    """Audit the exact slab that will enter oxide HER D1/D2 descriptor calculations.

    This is intentionally non-invasive: it does not slabify or modify uploaded
    Miller-index slabs. It only exposes the vacuum, z-layer, top/bottom
    composition, and D2-partial fixed/relaxed atom counts so upload-CIF and
    MP-generated slabs can be checked under the same guardrail.
    """
    audit = {"status": "not_available", "hard_errors": [], "warnings": []}
    if atoms is None:
        audit["hard_errors"].append("No prepared slab is available.")
        return audit
    a = atoms.copy()
    audit["status"] = "ok"
    audit["n_atoms"] = int(len(a))
    try:
        audit["formula"] = str(a.get_chemical_formula())
    except Exception:
        audit["formula"] = ""
    try:
        cell = a.get_cell()
        audit["cell_z_A"] = float(cell.lengths()[2])
    except Exception:
        audit["cell_z_A"] = float("nan")
    try:
        z = np.asarray(a.get_positions()[:, 2], dtype=float)
        audit["z_min_A"] = float(np.min(z))
        audit["z_max_A"] = float(np.max(z))
        audit["slab_thickness_z_A"] = float(np.max(z) - np.min(z))
        audit["estimated_vacuum_z_A"] = float(audit["cell_z_A"] - audit["slab_thickness_z_A"])
    except Exception:
        audit["z_min_A"] = audit["z_max_A"] = audit["slab_thickness_z_A"] = audit["estimated_vacuum_z_A"] = float("nan")
    try:
        rep = validate_structure(a, target_area=70.0)
        audit["validate_vacuum_z_A"] = float(getattr(rep, "vacuum_z", np.nan))
    except Exception as e:
        audit["validate_vacuum_z_A"] = float("nan")
        audit["warnings"].append(f"validate_structure failed: {e}")
    try:
        pbc = tuple(bool(x) for x in a.get_pbc())
        audit["pbc"] = str(pbc)
        audit["pbc_z"] = bool(pbc[2])
    except Exception:
        audit["pbc"] = "unknown"
        audit["pbc_z"] = True
    try:
        layers = _cluster_z_layers_simple(a, tol=float(layer_tol))
        audit["n_z_layers_clustered"] = int(len(layers))
        if layers:
            bottom_layer = layers[0]
            top_layer = layers[-1]
            audit["bottom_layer_atom_count"] = int(len(bottom_layer))
            audit["top_layer_atom_count"] = int(len(top_layer))
            audit["bottom_layer_formula"] = _counter_to_compact_formula(Counter(a[i].symbol for i in bottom_layer))
            audit["top_layer_formula"] = _counter_to_compact_formula(Counter(a[i].symbol for i in top_layer))
        else:
            audit["bottom_layer_atom_count"] = 0
            audit["top_layer_atom_count"] = 0
            audit["bottom_layer_formula"] = ""
            audit["top_layer_formula"] = ""
    except Exception as e:
        audit["n_z_layers_clustered"] = 0
        audit["warnings"].append(f"z-layer clustering failed: {e}")
    fixed_idx = _common_like_bottom_fix_indices(a, n_fix_layers=int(n_fix_layers), z_tol=float(z_tol))
    audit["D2_partial_n_fix_layers"] = int(n_fix_layers)
    audit["D2_partial_fixed_atom_count"] = int(len(fixed_idx))
    audit["D2_partial_relaxed_atom_count"] = int(len(a) - len(fixed_idx))
    audit["D2_partial_fixed_formula"] = _counter_to_compact_formula(Counter(a[i].symbol for i in fixed_idx))
    audit["D2_partial_policy"] = "bottom_n_layers_fixed; upper_slab_and_H_relaxed"
    vac_eff = audit.get("validate_vacuum_z_A", audit.get("estimated_vacuum_z_A", float("nan")))
    if np.isfinite(vac_eff):
        if bool(audit.get("pbc_z", True)) and float(vac_eff) < 8.0:
            audit["hard_errors"].append(f"Vacuum is too small for adsorption screening (vacuum_z={vac_eff:.2f} Å).")
        elif float(vac_eff) < 15.0:
            audit["warnings"].append(f"Vacuum is below the recommended upload-slab audit threshold (vacuum_z={vac_eff:.2f} Å).")
    if int(audit.get("D2_partial_fixed_atom_count", 0)) <= 0:
        audit["hard_errors"].append("D2 partial relaxation would fix zero slab atoms; check z orientation/layering.")
    if int(audit.get("D2_partial_relaxed_atom_count", 0)) <= 0:
        audit["hard_errors"].append("D2 partial relaxation would leave no atoms free; check n_fix_layers/layering.")
    if int(audit.get("n_z_layers_clustered", 0)) < max(3, int(n_fix_layers) + 1):
        audit["warnings"].append("Few z-layers were detected; bottom-layer fixing may be poorly defined.")
    return audit


def _render_oxide_her_pre_run_audit(audit: dict, *, expanded: bool = False):
    """Render the oxide HER D1/D2 input-slab audit in Streamlit."""
    if not isinstance(audit, dict) or not audit:
        return
    with st.expander("Oxide HER D1/D2 pre-run slab audit", expanded=bool(expanded)):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Atoms", str(audit.get("n_atoms", "NA")))
        c2.metric("Vacuum z", f"{_safe_float(audit.get('validate_vacuum_z_A', audit.get('estimated_vacuum_z_A', np.nan))):.2f} Å")
        c3.metric("D2 fixed atoms", str(audit.get("D2_partial_fixed_atom_count", "NA")))
        c4.metric("D2 relaxed atoms", str(audit.get("D2_partial_relaxed_atom_count", "NA")))
        st.write(f"- Formula: **{audit.get('formula', '')}**")
        st.write(f"- Top layer: **{audit.get('top_layer_formula', '')}**")
        st.write(f"- Bottom layer: **{audit.get('bottom_layer_formula', '')}**")
        st.write(f"- D2 policy: **{audit.get('D2_partial_policy', '')}**")
        if audit.get("warnings"):
            st.warning("; ".join(str(x) for x in audit.get("warnings", [])))
        if audit.get("hard_errors"):
            st.error("; ".join(str(x) for x in audit.get("hard_errors", [])))


def _append_oxide_her_audit_columns(df: pd.DataFrame, audit: dict) -> pd.DataFrame:
    """Attach scalar pre-run audit metadata to the result table."""
    if not isinstance(df, pd.DataFrame) or not isinstance(audit, dict) or not audit:
        return df
    out = df.copy()
    keys = [
        "formula", "cell_z_A", "slab_thickness_z_A", "estimated_vacuum_z_A",
        "validate_vacuum_z_A", "pbc", "pbc_z", "n_z_layers_clustered",
        "top_layer_formula", "bottom_layer_formula", "D2_partial_n_fix_layers",
        "D2_partial_fixed_atom_count", "D2_partial_relaxed_atom_count",
        "D2_partial_fixed_formula", "D2_partial_policy",
    ]
    for key in keys:
        if key in audit:
            out[f"pre_run_{key}"] = audit.get(key)
    out["pre_run_audit_warnings"] = "; ".join(str(x) for x in audit.get("warnings", []))
    out["pre_run_audit_hard_errors"] = "; ".join(str(x) for x in audit.get("hard_errors", []))
    return out


def _surface_fraction_from_meta(meta: dict, side: str):
    if meta is None:
        return None
    for key in (
        f"surface_O_fraction_{side}",
        f"surface_0_fraction_{side}",
        f"surface_o_fraction_{side}",
        f"surface_fraction_{side}",
    ):
        if key in meta and meta.get(key) is not None:
            try:
                return float(meta.get(key))
            except Exception:
                continue
    return None


def _annotate_step2_slab_symmetry(meta: Optional[dict], *, frac_tol_strict: float = 0.03, frac_tol_loose: float = 0.12):
    """Lightweight top/bottom symmetry annotation for Step 2 QC/UI.

    This does not attempt a crystallographic symmetry analysis.
    It only evaluates whether top/bottom surface terminations look similar enough
    to treat the slab as symmetric for screening purposes.
    """
    md = dict(meta or {})
    top_exp = str(md.get("top_exposure", "unknown") or "unknown").strip().lower()
    bot_exp = str(md.get("bottom_exposure", "unknown") or "unknown").strip().lower()
    f_top = _surface_fraction_from_meta(md, "top")
    f_bot = _surface_fraction_from_meta(md, "bottom")
    asym_flag = bool(md.get("top_bottom_asymmetric", False))

    reasons = []
    frac_flag = "unknown"
    if (f_top is not None) and (f_bot is not None):
        df = abs(float(f_top) - float(f_bot))
        if df <= float(frac_tol_strict):
            frac_flag = "match"
            reasons.append(f"surface fraction match (Δ={df:.3f})")
        elif df <= float(frac_tol_loose):
            frac_flag = "close"
            reasons.append(f"surface fraction close (Δ={df:.3f})")
        else:
            frac_flag = "mismatch"
            reasons.append(f"surface fraction mismatch (Δ={df:.3f})")
    else:
        reasons.append("surface fraction unavailable")

    if top_exp != "unknown" and bot_exp != "unknown":
        if top_exp == bot_exp:
            exp_flag = "match"
            reasons.append(f"exposure match ({top_exp})")
        else:
            exp_flag = "mismatch"
            reasons.append(f"exposure mismatch ({top_exp} vs {bot_exp})")
    else:
        exp_flag = "unknown"
        reasons.append("exposure unavailable")

    if asym_flag or exp_flag == "mismatch" or frac_flag == "mismatch":
        slab_symmetry = "asymmetric"
    elif exp_flag == "match" and frac_flag == "match":
        slab_symmetry = "symmetric"
    else:
        slab_symmetry = "quasi-symmetric"

    md["surface_fraction_top"] = f_top
    md["surface_fraction_bottom"] = f_bot
    md["slab_symmetry"] = slab_symmetry
    md["slab_symmetry_basis"] = "; ".join(reasons)
    return md


def _oxide_plausibility_rank_key(meta: Optional[dict]):
    """Rank oxide slab candidates by plausibility for Step 2 slab screening.

    Goal:
      - prefer slabs that are usable / reference-like
      - prefer symmetric terminations over quasi-symmetric / asymmetric
      - use the existing oxide_rank_key only as a late tie-breaker

    This is intentionally different from activity-oriented ranking.
    """
    md = dict(meta or {})

    usability = str(md.get("slab_usability", "") or "").strip().lower()
    symmetry = str(md.get("slab_symmetry", "") or "").strip().lower()
    validity = str(md.get("rule_validity", "") or "").strip().lower()
    role = str(md.get("rule_role", "") or "").strip().lower()
    diagnostics = str(md.get("surface_diagnostics_status", "") or "").strip().lower()

    def _usability_rank(x: str) -> int:
        if any(tok in x for tok in ("reference_ready", "reference", "usable")) and "exploratory" not in x:
            return 0
        if "usable" in x:
            return 1
        if "exploratory" in x:
            return 2
        if any(tok in x for tok in ("invalid", "unusable", "reject", "fail")):
            return 3
        return 4

    def _symmetry_rank(x: str) -> int:
        if x == "symmetric":
            return 0
        if x == "quasi-symmetric":
            return 1
        if x == "asymmetric":
            return 2
        return 3

    def _validity_rank(x: str) -> int:
        if x in {"ok", "valid", "pass"}:
            return 0
        if x in {"warn", "warning"}:
            return 1
        if x in {"fail", "invalid", "reject"}:
            return 2
        return 3

    def _role_rank(x: str) -> int:
        if "reference" in x:
            return 0
        if any(tok in x for tok in ("secondary", "support", "supplementary")):
            return 1
        if any(tok in x for tok in ("exploratory", "advanced", "polar")):
            return 2
        return 3

    def _diag_rank(x: str) -> int:
        if x in {"ok", "pass", "good"}:
            return 0
        if x in {"warn", "warning"}:
            return 1
        if x in {"fail", "bad", "invalid"}:
            return 2
        return 3

    return (
        _usability_rank(usability),
        _symmetry_rank(symmetry),
        _validity_rank(validity),
        _role_rank(role),
        _diag_rank(diagnostics),
        md.get("oxide_rank_key", (999, 999, 999, 999)),
    )



# ---------------- Sidebar: global options ----------------
with st.sidebar:
    with st.expander("0) Credentials & LLM", expanded=False):
        st.text_input("Materials Project API key (MP_API_KEY)", type="password", key="mp_api_key")
        st.text_input("OpenAI API key (OPENAI_API_KEY)", type="password", key="openai_api_key")
        MODEL_OPTIONS = [
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-5",
            "gpt-5.2",
        ]
        if "llm_model" not in st.session_state or st.session_state.get("llm_model") not in MODEL_OPTIONS:
            st.session_state["llm_model"] = "gpt-4o-mini"
        st.selectbox(
            "LLM model",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(st.session_state["llm_model"]),
            key="llm_model",
        )
        st.checkbox("Enable LLM interpretation", key="llm_enabled")
        st.caption("Keys are kept in session_state only. Prefer environment variables for long-term use.")

    st.header("Global settings")


    mode = st.radio("Material type", ["Metal (CHE)", "Oxide (CHE)"], horizontal=False)
    mtype = "metal" if "Metal" in mode else "oxide"

    reaction_mode = st.radio(
        "Reaction mode",
        ["HER (ΔG_H)", "CO₂RR (ΔE descriptor)", "OER competition (OOH/O/OH)", "VOCs (ΔE_proxy)"],
        horizontal=False,
        help=(
            "HER: H* adsorption free energy | CO₂RR: CO/formate/methanol/methane C1 intermediate presets | "
            "OER: OOH*/O*/OH* oxygen-intermediate CHE descriptors | "
            "VOCs: target-VOC adsorption and H*/OH* co-adsorption proximity proxies"
        ),
    )
    is_her = reaction_mode.startswith("HER")
    is_oer = reaction_mode.startswith("OER")
    is_voc = reaction_mode.startswith("VOCs")
    is_oxygen = is_oer

    relax_mode = st.selectbox(
        "Relaxation level (OCP)",
        ["Fast", "Normal", "Tight"],
        index=1,
        help="Fast: 300 steps, Normal: 600 steps, Tight: 900 steps (Applied to BOTH Slab and Adsorbate)",
    )

    st.divider()

    if mtype == "metal":
        default_sites = ["fcc", "hcp", "bridge", "ontop"]
    else:
        default_sites = ["fcc", "bridge", "ontop"]

    site_preset = tuple(default_sites)

    # VOC-mode defaults are defined up front to keep Streamlit reruns stable.
    voc_key = str(st.session_state.get("voc_target", "acetaldehyde"))
    voc_states = []
    voc_relaxation_policy = "normal_relax"
    st.session_state["voc_relaxation_policy"] = voc_relaxation_policy
    oxide_voc_site_policy = str(st.session_state.get("oxide_voc_site_policy", "geometry_representative"))

    # CO2RR-air competition controls are intentionally scoped to the CO2RR branch only.
    # They do not modify HER, OER, or VOC calculation branches.
    co2rr_air_enabled = False
    co2rr_air_oxygen_ads = []
    co2rr_air_include_her = True
    co2rr_air_oer_relaxation_mode = "short_relax"
    co2rr_pathway_key = str(st.session_state.get("co2rr_pathway", "competitive_c1"))
    co2rr_include_her = bool(st.session_state.get("co2rr_include_her", True))
    co2rr_potential_V = float(st.session_state.get("co2rr_potential_V", 0.0))

    if is_her:
        co2_ads = []
        orr_ads = []
    elif is_oxygen:
        co2_ads = []
        orr_ads = st.multiselect(
            "OER intermediates",
            ["OOH*", "O*", "OH*"],
            default=["OOH*", "O*", "OH*"],
            help="OER oxygen-intermediate pathway. OER summaries require valid, surface-bound OH*, O*, and OOH* rows on the same site/channel.",
        )
        if is_oer:
            oer_relaxation_mode = st.selectbox(
                "OER diagnostic relaxation mode",
                ["placement_only", "single_point", "short_relax", "normal_relax"],
                index=2,
                help=(
                    "OER oxide diagnostics only. placement_only/single_point save the initial adsorbate geometry without adsorbate relaxation; "
                    "short_relax uses short adsorbate-only relaxation; normal_relax uses the standard adsorbate relaxation. HER is unaffected."
                ),
                key="oer_relaxation_mode",
            )
            oer_manual_cation_indices_text = st.text_input(
                "OER manual cation indices (optional)",
                value="",
                help="Comma/space-separated atom indices for manual Ir_cus/cation targeting. Empty = automatic exposed-cation detector. Used only for oxide OER.",
                key="oer_manual_cation_indices_text",
            )
        else:
            oer_relaxation_mode = "short_relax"
            oer_manual_cation_indices_text = ""
    elif is_voc:
        co2_ads = []
        orr_ads = []
        voc_key = st.selectbox(
            "Target VOC",
            list(VOC_PRESETS.keys()),
            index=list(VOC_PRESETS.keys()).index(st.session_state.get("voc_target", "acetaldehyde")) if st.session_state.get("voc_target", "acetaldehyde") in VOC_PRESETS else 0,
            format_func=lambda k: VOC_PRESETS[k]["label"],
            key="voc_target",
            help="SAGE-VOC currently ships with an acetaldehyde preset; the registry is designed for additional VOCs later.",
        )
        _voc_preset = get_voc_preset(voc_key)
        _voc_routes = dict(_voc_preset.get("routes", {}))
        _route_order = ("reduction", "oxidation")
        _route_keys = [k for k in _route_order if k in _voc_routes] or list(_voc_routes.keys()) or ["oxidation"]
        _default_route = str(_voc_preset.get("default_route", "oxidation"))
        if _default_route not in _route_keys:
            _default_route = _route_keys[0]
        _prev_route = st.session_state.get("_voc_route_prev", None)
        voc_route = st.selectbox(
            "Acetaldehyde route",
            _route_keys,
            index=_route_keys.index(st.session_state.get("voc_route", _default_route)) if st.session_state.get("voc_route", _default_route) in _route_keys else _route_keys.index(_default_route),
            format_func=lambda k: _voc_routes.get(k, {}).get("label", str(k)),
            key="voc_route",
            help=(
                "Direct reduction = CH3CHO-to-ethanol direct electroreduction proxies. "
                "Oxidation = aldehyde activation + acetate/deep-oxidation proxies. "
                "ECH/co-adsorption route is disabled in this stable branch."
            ),
        )
        _route_default_states = list(_voc_routes.get(voc_route, {}).get("states", _voc_preset.get("default_states", [])))
        _voc_state_options = list(dict.fromkeys(list(_route_default_states) + list(_voc_preset.get("optional_states", []))))
        _current_voc_states = list(st.session_state.get("voc_states", []))
        # Keep descriptor-state selection route-scoped.  This prevents old
        # session values from legacy routes from leaking into the active
        # direct-reduction or oxidation routes after code updates/reruns.
        if (
            _prev_route != voc_route
            or not _current_voc_states
            or any(str(x) not in set(_voc_state_options) for x in _current_voc_states)
        ):
            st.session_state["_voc_route_prev"] = voc_route
            st.session_state["voc_states"] = list(_route_default_states)
        voc_states = st.multiselect(
            "VOC descriptor states",
            _voc_state_options,
            default=[s for s in list(st.session_state.get("voc_states", _route_default_states)) if s in _voc_state_options],
            help=(
                "Direct reduction default: H*, CH3CHO*, CH3CH2O*, CH3CH2OH*. "
                "Oxidation default: OH*, CH3CHO*, CH3CO*, CH3COO*, CO*, COOH*. "
                "O* is handled in OER mode; CH3COOH* remains optional. "
                "ECH/co-adsorption states such as H*+CH3CHO* and H*+H* are disabled."
            ),
            key="voc_states",
        )
        # VOC descriptors are standardized to Normal relax.
        # This removes ambiguous single-point/short-relax descriptor variants from the user workflow.
        voc_relaxation_policy = "normal_relax"
        st.session_state["voc_relaxation_policy"] = voc_relaxation_policy
        st.caption("VOC relaxation policy is fixed to Normal relax (slab fixed, anchor locked).")
        st.caption(str(_voc_preset.get("warning", "")))
        if mtype == "oxide":
            oxide_voc_site_policy = st.selectbox(
                "Oxide VOC site policy",
                ["geometry_representative", "fast_routed", "extended_scan"],
                index=0,
                format_func=lambda x: {
                    "geometry_representative": "Geometry representative sites (default: Step-3 ontop/bridge/fcc)",
                    "fast_routed": "Routed cation/anion sites (diagnostic)",
                    "extended_scan": "Extended routed site scan (diagnostic)",
                }.get(str(x), str(x)),
                key="oxide_voc_site_policy",
                help="Default keeps Step-3 ontop/bridge/fcc/hollow sites. Routed cation-only modes are diagnostic options for oxide-specific tests.",
            )
    else:
        voc_relaxation_policy = "normal_relax"
        _pathway_keys = [k for k in CO2RR_PATHWAY_ORDER]
        _prev_pathway = st.session_state.get("_co2rr_pathway_prev", None)
        co2rr_pathway_key = st.selectbox(
            "CO₂RR product-targeted preset",
            _pathway_keys,
            index=_pathway_keys.index(st.session_state.get("co2rr_pathway", "competitive_c1")) if st.session_state.get("co2rr_pathway", "competitive_c1") in _pathway_keys else 0,
            format_func=lambda k: get_co2rr_preset(k).get("label", str(k)),
            key="co2rr_pathway",
            help=(
                "Competitive C1 evaluates shared intermediates once and compares CO, formate, "
                "formaldehyde, methanol, and methane thermodynamic pathways. Neutral CH2O is "
                "treated as HCHO(g), not forced onto the surface as CH2O*. Product presets "
                "remain available for smaller diagnostic runs."
            ),
        )
        _co2rr_preset = get_co2rr_preset(co2rr_pathway_key)
        _co2rr_default_states = list(_co2rr_preset.get("states", ["COOH*", "CO*"]))
        _co2rr_options = all_co2rr_states()
        _current_co2rr_states = list(st.session_state.get("co2rr_ads", []))
        if (
            _prev_pathway != co2rr_pathway_key
            or not _current_co2rr_states
            or any(str(x) not in set(_co2rr_options) for x in _current_co2rr_states)
        ):
            st.session_state["_co2rr_pathway_prev"] = co2rr_pathway_key
            st.session_state["co2rr_ads"] = list(_co2rr_default_states)
        co2_ads = st.multiselect(
            "CO₂RR intermediates",
            _co2rr_options,
            default=[s for s in st.session_state.get("co2rr_ads", _co2rr_default_states) if s in _co2rr_options],
            format_func=co2rr_state_label,
            key="co2rr_ads",
        )
        st.caption(str(_co2rr_preset.get("description", "")))
        st.caption(CO2RR_WARNING)
        co2rr_potential_V = float(st.number_input(
            "CO₂RR analysis potential (V vs RHE)",
            min_value=-2.50,
            max_value=1.00,
            value=float(st.session_state.get("co2rr_potential_V", 0.0)),
            step=0.05,
            key="co2rr_potential_V",
            help=(
                "Updates ΔG(U)=ΔG(0)+n·U for reduction PCET edges. "
                "The limiting potential itself is calculated from the U=0 edge energies."
            ),
        ))
        co2rr_include_her = st.checkbox(
            "Include HER competition guardrail (H*)",
            value=bool(st.session_state.get("co2rr_include_her", True)),
            key="co2rr_include_her",
            help="Runs the existing single-site HER guardrail separately from the carbon-intermediate preset.",
        )
        orr_ads = []
        co2rr_air_enabled = st.checkbox(
            "Append CO₂RR-air competition summary",
            value=False,
            key="co2rr_air_enabled",
            help=(
                "CO₂RR-only add-on. Runs the existing OER oxygen-intermediate engine in a separate "
                "output folder and combines those rows with the CO₂RR HER guardrail for ORR/HER risk."
            ),
        )
        if co2rr_air_enabled:
            co2rr_air_oxygen_ads = st.multiselect(
                "ORR/O₂ competition intermediates",
                ["OOH*", "O*", "OH*"],
                default=["OOH*", "O*", "OH*"],
                key="co2rr_air_oxygen_ads",
                help="Screening-only oxygen-affinity indicators for air-fed or dilute-CO₂ CO₂RR.",
            )
            co2rr_air_include_her = st.checkbox(
                "Include HER guardrail in CO₂RR-air summary",
                value=bool(co2rr_include_her),
                key="co2rr_air_include_her",
            )
            co2rr_air_oer_relaxation_mode = st.selectbox(
                "CO₂RR-air oxygen relaxation mode",
                ["single_point", "short_relax", "normal_relax"],
                index=1,
                key="co2rr_air_oer_relaxation_mode",
                help="Applied only to the auxiliary oxygen-intermediate run launched from CO₂RR mode.",
            )



    if not is_oer:
        oer_relaxation_mode = "short_relax"
        oer_manual_cation_indices_text = ""

    surfactant_class = "none"
    surfactant_prerelax_slab = False
    # Surfactant scenario controls are intentionally placed in Step 3 (Site selection),
    # because the module is a structural conditioning option (slab pre-relaxation / site enumeration)
    # rather than a thermodynamic CHE correction.

    st.divider()
    st.markdown("#### Run history (session-only)")
    st.caption("Closing or refreshing this app clears this history.")
    try:
        rh.render_history_sidebar(max_items=10)
    except Exception as e:
        st.caption(f"Run history unavailable: {e}")

# ---------------- STEP 1: Load structure ----------------
st.markdown("## 1) Load structure")

slab_source_mode = st.radio(
    "Choose slab source",
    ["Upload CIF", "Generate from MP bulk (cifgen)"],
    horizontal=True,
    key="slab_source_mode",
)

prev_mode = st.session_state.get("_slab_source_mode_prev", None)
if prev_mode is None:
    st.session_state["_slab_source_mode_prev"] = slab_source_mode
elif prev_mode != slab_source_mode:
    st.session_state["_upload_sig"] = None
    st.session_state["_slab_source_mode_prev"] = slab_source_mode
    st.session_state["atoms_loaded"] = None
    st.session_state["atoms_tuned"] = None
    st.session_state["ratio_tune_meta"] = None
    _reset_prepared_from_working()

colL, colR = st.columns([1.2, 1.0])

with colL:
    if slab_source_mode == "Upload CIF":
        slab_file = st.file_uploader("Upload CIF (slab or bulk)", type=["cif"], key="upload_cif")
        if slab_file is not None:
            file_sig = (slab_file.name, slab_file.size)
            if st.session_state.get("_upload_sig") != file_sig:
                st.session_state["_upload_sig"] = file_sig
                try:
                    slab_file.seek(0)
                except Exception:
                    pass
                atoms_loaded = read(slab_file, format="cif")
                st.session_state["atoms_loaded"] = atoms_loaded
                st.session_state["atoms_tuned"] = None
                st.session_state["ratio_tune_meta"] = None
                _reset_prepared_from_working()
                st.success(f"Loaded: {atoms_loaded.get_chemical_formula()} | atoms={len(atoms_loaded)}")

    elif slab_source_mode == "Generate from MP bulk (cifgen)":
        st.markdown("#### Bulk CIF Generator (Materials Project)")
        mp_id_raw = st.text_input("Bulk mp-id (number only is OK)", "19009", key="mp_id_raw")
        mp_id = normalize_mp_id(mp_id_raw)
        st.caption(f"Resolved mp-id: `{mp_id}`")

        if st.button("Fetch bulk from MP", type="primary", key="btn_fetch_mp"):
            try:
                if not mp_id:
                    raise ValueError("mp-id is empty.")
                bulk_src = BulkSource(kind="mp-id", ref=mp_id, label="host", api_key=(st.session_state.get("mp_api_key") or None))
                bulk_spec = BulkSpec(bulk_source=bulk_src)
                atoms_loaded = generate_bulk(bulk_spec)
                st.session_state["atoms_loaded"] = atoms_loaded
                st.session_state["loaded_mp_id"] = mp_id
                st.session_state["atoms_tuned"] = None
                st.session_state["ratio_tune_meta"] = None
                _reset_prepared_from_working()
                st.success(f"Fetched: {atoms_loaded.get_chemical_formula()} | atoms={len(atoms_loaded)}")
            except Exception as e:
                st.error(f"CIF generation failed: {e}")


with colR:
    working = st.session_state.get("atoms_tuned") or st.session_state.get("atoms_loaded")
    if working is None:
        st.info("Load a CIF first.")
    else:
        st.markdown("#### Preview (Working)")
        show_atoms_3d(working, height=360, width=520, tag="working")
        st.download_button(
            "Download WORKING CIF",
            atoms_to_cif_bytes(working, symprec=0.1),
            file_name="working_structure.cif",
            mime="chemical/x-cif",
            key="dl_working_cif",
        )


# ---------------- STEP 2: Surface setup ----------------
st.markdown("## 2) Surface setup")

working = st.session_state.get("atoms_tuned") or st.session_state.get("atoms_loaded")
if working is None:
    st.info("Load a structure first (Step 1).")
else:
    _ensure_prepared_uptodate()
    prepared = st.session_state.get("atoms_prepared")

    surface_route = st.radio(
        "Surface route",
        ["Use current structure", "Slabify from bulk"],
        horizontal=True,
        key="surface_route_mode",
        help="Choose either the current structure directly or first split the bulk into a slab/facet.",
    )

    route_prev = st.session_state.get("_surface_route_prev", None)
    route_stage_default = 0 if surface_route == "Slabify from bulk" else 1
    if route_prev != surface_route:
        st.session_state["_surface_route_prev"] = surface_route
        st.session_state["surface_setup_stage"] = route_stage_default
        if surface_route != "Slabify from bulk":
            try:
                st.session_state["slab_reduction_base_atoms"] = prepared.copy()
            except Exception:
                pass
    surface_setup_stage = int(st.session_state.get("surface_setup_stage", route_stage_default))
    if surface_route != "Slabify from bulk" and surface_setup_stage < 1:
        surface_setup_stage = 1
        st.session_state["surface_setup_stage"] = 1

    stage_labels = {
        0: "2-1. Select slab",
        1: "2-2. Set vacuum",
        2: "2-3. Expand XY supercell",
        3: "2-4. Reduce slab thickness",
        4: "2-5. Review prepared slab",
        5: "2-6. Surface engineering",
    }
    stage_short_labels = {
        0: "2-1",
        1: "2-2",
        2: "2-3",
        3: "2-4",
        4: "2-5",
        5: "2-6",
    }
    visible_stage_ids = [0, 1, 2, 3, 4, 5] if surface_route == "Slabify from bulk" else [1, 2, 3, 4, 5]
    max_revisit_stage = max(visible_stage_ids[0], min(int(surface_setup_stage), max(visible_stage_ids)))

    st.caption(f"Surface-setup wizard: **{stage_labels.get(surface_setup_stage, '2-1. Select slab')}**")
    nav_cols = st.columns(len(visible_stage_ids))
    for _col, _sid in zip(nav_cols, visible_stage_ids):
        _is_current = int(surface_setup_stage) == int(_sid)
        _is_accessible = int(_sid) <= int(max_revisit_stage)
        _prefix = "●" if _is_current else "○"
        with _col:
            if st.button(
                f"{_prefix} {stage_short_labels.get(_sid, _sid)}",
                key=f"btn_surface_stage_nav_{surface_route}_{_sid}",
                disabled=not _is_accessible,
                use_container_width=True,
            ):
                st.session_state["surface_setup_stage"] = int(_sid)
                st.rerun()
    st.caption("Click a completed stage to revisit it. Forward jumps to incomplete stages are disabled.")

    if surface_route == "Slabify from bulk" and surface_setup_stage == 0:
        if mtype == "oxide":
            st.markdown("### Oxide surface builder")
            st.caption("For oxides, clean slab candidates are ranked with family-aware validity rules. The app distinguishes conservative reference surfaces from exploratory or advanced clean facets.")
        else:
            st.markdown("### Metal slabify")
            st.caption("Facet splitting is handled here. After selecting a slab candidate, vacuum / supercell / QC are applied in the common panel below.")

        if not HAS_SLABIFY:
            st.info(f"SlabGenerator not available: {SLABIFY_IMPORT_ERR}")
        else:
            if mtype == "oxide":
                facet_scope = st.selectbox(
                    "Facet set",
                    ["Recommended oxide facets", "Low-index facets (up to 1)", "Extended facets (up to 2)"],
                    index=0,
                    key="slab_facet_scope",
                    help="Recommended oxide facets are conservative and bias toward less problematic low-index surfaces.",
                )
                facet_scope_for_calc = "Recommended facets" if facet_scope == "Recommended oxide facets" else facet_scope
                facet_choices = _augment_oxide_low_index_facets(
                    _facet_choices_for_scope(prepared, facet_scope_for_calc),
                    facet_scope,
                )
                facet_labels = [_format_facet_label_with_alias(hkl) for hkl in facet_choices]
                if bool(globals().get("is_oer", False)):
                    oxide_surface_mode = "OER AEM cation surface preference"
                    st.info("OER mode: slab candidates will be ranked by cation exposure / Ir_cus-like suitability, not by HER-style O-rich top exposure.")
                else:
                    oxide_surface_mode = "Exploratory any clean termination"
                st.caption("Low-index c-axis facets are exposed through the reduced (001) family. Literature 00l labels such as (002) are treated as the same selectable family.")
                oxide_hydrox_mode = "Clean only"
                _oxide_info = infer_oxide_family_from_atoms(prepared)
                _oxide_family = _infer_interface_surface_family(prepared)
                _cs = _oxide_info.get("crystal_system")
                _sg = _oxide_info.get("spacegroup_symbol")
                if _oxide_family == "cubic_AO":
                    st.caption("Cubic AO guidance: prefer (100) first, keep (110) as secondary, and treat (111) as a polar / advanced clean facet.")
                elif _oxide_family == "rutile_AO2":
                    st.caption("Rutile AO2 guidance: use (110) as the primary clean reference facet. Keep (100) and (101) as non-reference exploratory facets.")
                elif _oxide_family == "anatase_AO2":
                    st.caption("Anatase AO2 guidance: use (101) as the primary clean reference facet. Keep (001) as secondary / higher-energy and (100) as exploratory.")
                elif str(_oxide_family).startswith(("monoclinic_", "orthorhombic_", "triclinic_")):
                    st.caption(f"Low-symmetry oxide guidance: {_oxide_family} ({_cs}, {_sg}) is treated as exploratory in clean-surface mode unless facet-specific validation is available.")
                elif str(_oxide_family).endswith("_ABO3"):
                    st.caption("ABO3 oxide guidance: clean surfaces are termination-dependent, so all candidates are exploratory unless termination is resolved separately.")
                elif str(_oxide_family).endswith("_AB2O4"):
                    st.caption("AB2O4 oxide guidance: clean surface preference depends on cation distribution, so candidates are treated as exploratory.")
            else:
                facet_scope = st.selectbox(
                    "Facet set",
                    ["Recommended facets", "Low-index facets (up to 1)", "Extended facets (up to 2)"],
                    index=0,
                    key="slab_facet_scope",
                    help="Recommended = compact preset. Low-index / Extended = broader Miller-index coverage without manual typing.",
                )
                facet_choices = _facet_choices_for_scope(prepared, facet_scope)
                facet_labels = [_format_hkl_label(hkl) for hkl in facet_choices]
                oxide_surface_mode = None

            if not facet_choices:
                st.warning("No facet candidates were generated from the current structure.")
            else:
                sel_label = st.selectbox(
                    "Facet",
                    facet_labels,
                    index=0,
                    key="slab_facet_choice",
                )
                sel_hkl = facet_choices[facet_labels.index(sel_label)]
                st.caption(f"Selected Miller index: {sel_hkl}")

                colS1, colS2 = st.columns([1.2, 0.8])
                with colS1:
                    slab_vac_choice = st.selectbox(
                        "Target vacuum for generated slab",
                        ["20 Å", "30 Å (recommended)", "40 Å", "Custom"],
                        index=1,
                        key="slab_gen_vac_choice",
                    )
                with colS2:
                    slab_vac_custom = None
                    if slab_vac_choice == "Custom":
                        slab_vac_custom = st.number_input(
                            "Custom vacuum (Å)",
                            min_value=8.0,
                            max_value=80.0,
                            value=30.0,
                            step=1.0,
                            key="slab_gen_vac_custom",
                        )
                slab_target_vac = _vacuum_target_from_ui(slab_vac_choice, slab_vac_custom)

                colG1, colG2 = st.columns(2)
                with colG1:
                    if st.button("Generate slab candidates", key="btn_slabify_gen"):
                        try:
                            cand_atoms, cand_meta = slabify_from_bulk(
                                prepared,
                                miller=tuple(int(x) for x in sel_hkl),
                                min_slab_size=float(_DEFAULT_SLAB_MIN_THICKNESS),
                                min_vacuum_size=float(slab_target_vac),
                                max_candidates=int(_DEFAULT_SLAB_MAX_CANDIDATES),
                            )
                            if mtype == "oxide":
                                norm_atoms, norm_meta = [], []
                                mode_pref = str(oxide_surface_mode or "Reference clean surface")
                                rejected = 0
                                for a_i, m_i in zip(cand_atoms, cand_meta):
                                    if bool(globals().get("is_oer", False)):
                                        a_n, m_n = _normalize_oxide_candidate_oer_top_surface(a_i, m_i, z_window=1.8)
                                        m_n["oxide_surface_mode"] = mode_pref
                                        m_n["hydroxylation_mode"] = "Clean only"
                                        m_n = _annotate_step2_slab_symmetry(m_n)
                                        m_n["oxide_rank_key"] = _oxide_oer_candidate_rank_key(m_n)
                                        m_n["oxide_plausibility_rank_key"] = _oxide_oer_candidate_rank_key(m_n)
                                        # Do not silently discard OER-not-suitable slabs here.
                                        # Keep them visible so the user can see why a termination is poor.
                                        norm_atoms.append(a_n)
                                        norm_meta.append(m_n)
                                    elif bool(globals().get("is_voc", False)):
                                        # VOC-specific termination handling:
                                        # keep the existing HER/OER logic untouched, but for VOCs
                                        # choose the slab orientation whose top side exposes more
                                        # cation/mixed adsorption basins for CH3CHO*/OH*/acetate-like species.
                                        _voc_route_for_slab = str(st.session_state.get("voc_route", "oxidation"))
                                        a_n, m_n = _normalize_voc_oxide_candidate_top_surface(
                                            a_i,
                                            m_i,
                                            route=_voc_route_for_slab,
                                            z_window=1.8,
                                        )
                                        m_n["oxide_surface_mode"] = "VOC OER-like cation-accessible surface"
                                        m_n["hydroxylation_mode"] = "Clean only"
                                        m_n = _annotate_step2_slab_symmetry(m_n)
                                        m_n["oxide_rank_key"] = _voc_oxide_candidate_rank_key(m_n)
                                        m_n["oxide_plausibility_rank_key"] = _voc_oxide_candidate_rank_key(m_n)
                                        norm_atoms.append(a_n)
                                        norm_meta.append(m_n)
                                    else:
                                        a_n, m_n = _normalize_oxide_candidate_top_surface(a_i, m_i, z_window=1.8)
                                        m_n["oxide_surface_mode"] = mode_pref
                                        m_n["hydroxylation_mode"] = "Clean only"
                                        keep = _oxide_mode_keep_candidate(m_n, mode_pref)
                                        if keep:
                                            m_n = _annotate_step2_slab_symmetry(m_n)
                                            m_n["oxide_rank_key"] = _oxide_candidate_rank_key(m_n)
                                            m_n["oxide_plausibility_rank_key"] = _oxide_plausibility_rank_key(m_n)
                                            norm_atoms.append(a_n)
                                            norm_meta.append(m_n)
                                        else:
                                            rejected += 1
                                cand_atoms, cand_meta = norm_atoms, norm_meta
                                if rejected:
                                    st.info(f"Filtered out {rejected} oxide candidate(s) that failed the current clean-surface selection mode.")
                                if bool(globals().get("is_oer", False)):
                                    st.caption("OER mode kept all generated terminations but ranked them by OER_AEM_cation suitability. Use OER-not-suitable only as diagnostic, not as a benchmark surface.")
                                elif bool(globals().get("is_voc", False)):
                                    st.caption("VOC mode uses an OER-like cation-accessible oxide slab processing policy and ranks symmetric slabs first. HER/OER calculation branches themselves are not modified.")
                                if not cand_atoms:
                                    raise ValueError("No oxide slab candidates remained after family-aware clean-surface filtering. Try another facet or switch to a less restrictive clean-surface mode.")
                                paired_ranked = sorted(
                                    zip(cand_atoms, cand_meta),
                                    key=lambda x: x[1].get("oxide_plausibility_rank_key", (999, 999, 999, 999, 999, (999, 999, 999, 999))),
                                )
                                cand_atoms = [a for a, _m in paired_ranked]
                                cand_meta = [dict(_m) for _a, _m in paired_ranked]
                            st.session_state["slabify_candidates_atoms"] = cand_atoms
                            st.session_state["slabify_candidates_meta"] = cand_meta
                            st.success(
                                f"Generated {len(cand_atoms)} slab candidate(s) for {_format_hkl_label(sel_hkl)} | min slab thickness = {_DEFAULT_SLAB_MIN_THICKNESS:.1f} Å | target vacuum = {slab_target_vac:.1f} Å."
                            )
                        except Exception as e:
                            st.error(f"Slabify failed: {e}")
                with colG2:
                    if st.button("Clear slab candidates", key="btn_slabify_clear"):
                        st.session_state["slabify_candidates_atoms"] = None
                        st.session_state["slabify_candidates_meta"] = None
                        st.info("Cleared slab candidates.")

                cand_atoms = st.session_state.get("slabify_candidates_atoms") or []
                cand_meta = st.session_state.get("slabify_candidates_meta") or []
                if cand_meta:
                    if mtype == "oxide":
                        cand_meta = [_annotate_step2_slab_symmetry(m) for m in cand_meta]
                        st.session_state["slabify_candidates_meta"] = cand_meta
                    df_cands = pd.DataFrame(cand_meta)
                    if mtype == "oxide":
                        if bool(globals().get("is_oer", False)):
                            st.warning(
                                "OER mode: review `oer_slab_suitability`, exposed cation metrics, and O-crowding before selecting an oxide slab. HER-style O-rich slabs can be poor OER AEM surfaces."
                            )
                            basic_cols = [c for c in [
                                "idx", "miller", "oer_slab_suitability", "oer_slab_score", "oer_best_cation_symbol", "oer_best_cation_index", "oer_best_cation_coordination", "oer_best_o_crowding_min_OO_A", "top_exposure", "surface_O_fraction_top", "slab_symmetry", "n_atoms", "vacuum_z"
                            ] if c in df_cands.columns]
                        elif bool(globals().get("is_voc", False)):
                            st.warning(
                                "VOC mode: review `voc_slab_suitability`, cation accessibility, and O-only basin proxies before selecting an oxide slab. O-rich terminations can be poor CH3CHO*/OH* descriptor surfaces."
                            )
                            basic_cols = [c for c in [
                                "idx", "miller", "voc_slab_suitability", "voc_surface_score", "voc_top_window_formula", "voc_top_window_cation_count", "voc_surface_O_fraction_top_window", "voc_metal_oxygen_bridge_proxy_count", "voc_oxygen_oxygen_bridge_proxy_count", "top_exposure", "slab_symmetry", "n_atoms", "vacuum_z"
                            ] if c in df_cands.columns]
                        else:
                            st.warning(
                                "Please review `slab_symmetry`, `surface_O_fraction_top`, `slab_usability`, and `facet_warning` before selecting an oxide slab."
                            )
                            basic_cols = [c for c in [
                                "idx", "miller", "top_exposure", "surface_O_fraction_top", "slab_symmetry", "slab_usability", "n_atoms", "vacuum_z"
                            ] if c in df_cands.columns]
                        st.dataframe(df_cands[basic_cols], use_container_width=True)
                        with st.expander("Show detailed slab candidate metadata", expanded=False):
                            detail_cols = [c for c in [
                                "idx", "miller", "surface_family", "crystal_system", "spacegroup_symbol", "spacegroup_number", "voc_surface_policy", "voc_route", "voc_slab_suitability", "voc_surface_score", "voc_surface_warning", "voc_top_layer_formula", "voc_top_window_formula", "voc_top_layer_cation_count", "voc_top_window_cation_count", "voc_surface_O_fraction_top_layer", "voc_surface_O_fraction_top_window", "voc_metal_metal_bridge_proxy_count", "voc_metal_oxygen_bridge_proxy_count", "voc_oxygen_oxygen_bridge_proxy_count", "voc_metal_containing_hollow_proxy_count", "voc_oxygen_only_hollow_proxy_count", "flipped_for_voc_surface", "oer_slab_suitability", "oer_slab_score", "oer_slab_warning", "oer_best_cation_index", "oer_best_cation_symbol", "oer_best_cation_coordination", "oer_best_cation_z_depth_A", "oer_best_open_direction_z", "oer_best_o_crowding_min_OO_A", "oer_candidate_cation_indices", "flipped_for_oer_cation_exposure", "rule_validity", "rule_role", "surface_diagnostics_status", "slab_usability", "oxide_validity", "oxide_role", "top_exposure", "bottom_exposure", "surface_O_fraction_top",
                                "surface_O_fraction_bottom", "surface_fraction_top", "surface_fraction_bottom", "slab_symmetry", "slab_symmetry_basis", "top_bottom_asymmetric", "flipped_for_oxide_top_exposure", "oxide_top_surface_ok", "facet_warning", "n_atoms",
                                "vacuum_z", "recommend_repeat", "slab_usability_reason", "oxide_rule_notes", "surface_diagnostics_notes", "issues"
                            ] if c in df_cands.columns]
                            st.dataframe(df_cands[detail_cols], use_container_width=True)
                        auto_idx = 0
                        auto_meta = _annotate_step2_slab_symmetry(cand_meta[auto_idx])
                        if bool(globals().get("is_oer", False)):
                            st.caption(
                                f"Auto-selected OER oxide candidate: #{auto_idx} | OER={auto_meta.get('oer_slab_suitability')} | "
                                f"score={_safe_float(auto_meta.get('oer_slab_score', np.nan)):.2f} | best={auto_meta.get('oer_best_cation_symbol')}{auto_meta.get('oer_best_cation_index')} | "
                                f"crowding_OO={_safe_float(auto_meta.get('oer_best_o_crowding_min_OO_A', np.nan)):.2f} Å | top={auto_meta.get('top_exposure')} | atoms={auto_meta.get('n_atoms')}"
                            )
                        elif bool(globals().get("is_voc", False)):
                            st.caption(
                                f"Auto-selected VOC oxide candidate: #{auto_idx} | VOC={auto_meta.get('voc_slab_suitability')} | "
                                f"score={_safe_float(auto_meta.get('voc_surface_score', np.nan)):.2f} | window={auto_meta.get('voc_top_window_formula')} | "
                                f"cations={auto_meta.get('voc_top_window_cation_count')} | Ofrac={_safe_float(auto_meta.get('voc_surface_O_fraction_top_window', np.nan)):.2f} | "
                                f"top={auto_meta.get('top_exposure')} | atoms={auto_meta.get('n_atoms')}"
                            )
                        else:
                            st.caption(
                                f"Auto-selected oxide candidate: #{auto_idx} | symmetry={auto_meta.get('slab_symmetry')} | usability={auto_meta.get('slab_usability')} | "
                                f"rule={auto_meta.get('rule_validity')}/{auto_meta.get('rule_role')} | top={auto_meta.get('top_exposure')} | atoms={auto_meta.get('n_atoms')}"
                            )
                    else:
                        basic_cols = [c for c in ["idx", "miller", "n_atoms", "vacuum_z", "formula"] if c in df_cands.columns]
                        st.dataframe(df_cands[basic_cols], use_container_width=True)
                        with st.expander("Show detailed slab candidate metadata", expanded=False):
                            st.dataframe(df_cands, use_container_width=True)
                        auto_idx = 0

                    sel_idx = st.selectbox(
                        "Select slab candidate",
                        list(range(len(cand_atoms))),
                        index=int(auto_idx),
                        format_func=lambda i: (
                            (
                                f"#{i} | OER={cand_meta[i].get('oer_slab_suitability')} | score={_safe_float(cand_meta[i].get('oer_slab_score', np.nan)):.2f} | "
                                f"best={cand_meta[i].get('oer_best_cation_symbol')}{cand_meta[i].get('oer_best_cation_index')} | top={cand_meta[i].get('top_exposure')} | vac_z={cand_meta[i].get('vacuum_z', np.nan):.2f} Å | atoms={cand_meta[i].get('n_atoms')} | {cand_meta[i].get('formula')}"
                            ) if (mtype == "oxide" and bool(globals().get("is_oer", False))) else (
                                f"#{i} | VOC={cand_meta[i].get('voc_slab_suitability')} | score={_safe_float(cand_meta[i].get('voc_surface_score', np.nan)):.2f} | "
                                f"window={cand_meta[i].get('voc_top_window_formula')} | cations={cand_meta[i].get('voc_top_window_cation_count')} | "
                                f"top={cand_meta[i].get('top_exposure')} | vac_z={cand_meta[i].get('vacuum_z', np.nan):.2f} Å | atoms={cand_meta[i].get('n_atoms')} | {cand_meta[i].get('formula')}"
                            ) if (mtype == "oxide" and bool(globals().get("is_voc", False))) else (
                                f"#{i} | {cand_meta[i].get('slab_symmetry', '?')} | {cand_meta[i].get('slab_usability')} | rule={cand_meta[i].get('rule_validity')}/{cand_meta[i].get('rule_role')} | top={cand_meta[i].get('top_exposure')} | vac_z={cand_meta[i].get('vacuum_z', np.nan):.2f} Å | atoms={cand_meta[i].get('n_atoms')} | {cand_meta[i].get('formula')}"
                            ) if mtype == "oxide" else
                            f"#{i} | vac_z={cand_meta[i].get('vacuum_z', np.nan):.2f} Å | atoms={cand_meta[i].get('n_atoms')} | {cand_meta[i].get('formula')}"
                        ),
                        key="slabify_sel_idx",
                    )
                    show_atoms_3d(cand_atoms[sel_idx], height=360, width=700, tag=f"slab_cand_{sel_idx}")

                    if st.button("Use selected slab", type="primary", key="btn_slabify_apply"):
                        _push_prepared_update(
                            cand_atoms[sel_idx],
                            "slabify_apply",
                            {
                                "candidate_idx": int(sel_idx),
                                "miller": cand_meta[sel_idx].get("miller"),
                                "route": "slabify_from_bulk_oxide" if mtype == "oxide" else "slabify_from_bulk",
                                "top_exposure": cand_meta[sel_idx].get("top_exposure"),
                                "bottom_exposure": cand_meta[sel_idx].get("bottom_exposure"),
                                "flipped_for_oxide_top_exposure": cand_meta[sel_idx].get("flipped_for_oxide_top_exposure"),
                                "oxide_top_surface_ok": cand_meta[sel_idx].get("oxide_top_surface_ok"),
                                "rule_validity": cand_meta[sel_idx].get("rule_validity"),
                                "rule_role": cand_meta[sel_idx].get("rule_role"),
                                "surface_diagnostics_status": cand_meta[sel_idx].get("surface_diagnostics_status"),
                                "slab_usability": cand_meta[sel_idx].get("slab_usability"),
                                "oxide_validity": cand_meta[sel_idx].get("oxide_validity"),
                                "oxide_role": cand_meta[sel_idx].get("oxide_role"),
                                "slab_symmetry": cand_meta[sel_idx].get("slab_symmetry"),
                                "slab_symmetry_basis": cand_meta[sel_idx].get("slab_symmetry_basis"),
                                "oxide_surface_role": cand_meta[sel_idx].get("oxide_surface_role"),
                                "oer_slab_suitability": cand_meta[sel_idx].get("oer_slab_suitability"),
                                "oer_slab_score": cand_meta[sel_idx].get("oer_slab_score"),
                                "oer_best_cation_index": cand_meta[sel_idx].get("oer_best_cation_index"),
                                "oer_best_cation_symbol": cand_meta[sel_idx].get("oer_best_cation_symbol"),
                                "oer_best_cation_coordination": cand_meta[sel_idx].get("oer_best_cation_coordination"),
                                "oer_best_o_crowding_min_OO_A": cand_meta[sel_idx].get("oer_best_o_crowding_min_OO_A"),
                                "oer_slab_warning": cand_meta[sel_idx].get("oer_slab_warning"),
                                "voc_surface_policy": cand_meta[sel_idx].get("voc_surface_policy"),
                                "voc_route": cand_meta[sel_idx].get("voc_route"),
                                "voc_slab_suitability": cand_meta[sel_idx].get("voc_slab_suitability"),
                                "voc_surface_score": cand_meta[sel_idx].get("voc_surface_score"),
                                "voc_surface_warning": cand_meta[sel_idx].get("voc_surface_warning"),
                                "voc_top_layer_formula": cand_meta[sel_idx].get("voc_top_layer_formula"),
                                "voc_top_window_formula": cand_meta[sel_idx].get("voc_top_window_formula"),
                                "voc_top_window_cation_count": cand_meta[sel_idx].get("voc_top_window_cation_count"),
                                "voc_surface_O_fraction_top_window": cand_meta[sel_idx].get("voc_surface_O_fraction_top_window"),
                                "flipped_for_voc_surface": cand_meta[sel_idx].get("flipped_for_voc_surface"),
                            },
                        )
                        try:
                            st.session_state["slab_reduction_base_atoms"] = cand_atoms[sel_idx].copy()
                        except Exception:
                            pass
                        st.session_state["surface_setup_stage"] = 1
                        st.success("Selected slab applied. Continue to vacuum setup.")
                        st.rerun()

        prepared = st.session_state.get("atoms_prepared")

    colA, colB = st.columns([1.15, 0.85])

    with colA:
        rep = validate_structure(prepared, target_area=70.0)
        vac_z = float(getattr(rep, "vacuum_z", 0.0))
        pbc = tuple(bool(x) for x in prepared.get_pbc())

        st.markdown("### Structure check (current active structure)")
        st.write(f"- Atoms: **{getattr(rep, 'n_atoms', len(prepared))}**")
        st.write(f"- Vacuum_z: **{vac_z:.2f} Å**")
        st.write(f"- PBC: **{pbc}**")

        st.markdown("#### Min distances by element pair")
        _render_min_dist_panel(rep)

        if mtype == "oxide":
            fam = infer_oxide_family_from_atoms(prepared)
            if fam["family"] != "unknown":
                st.info(
                    f"Detected Oxide: {fam['family']} ({fam['reduced_formula']}) | crystal={fam.get('crystal_system')} | sg={fam.get('spacegroup_symbol')}"
                )
            surf_meta = _classify_surface_exposure(prepared, z_window=1.8)
            prep_meta = _normalize_oxide_candidate_top_surface(prepared, {"surface_family": _infer_interface_surface_family(prepared), "miller": None}, z_window=1.8)[1]
            prep_meta = _annotate_step2_slab_symmetry({**prep_meta, **surf_meta})
            with st.expander("Show oxide surface diagnostics", expanded=False):
                st.write(f"- Surface exposure (top/bottom): **{prep_meta['top_exposure']} / {prep_meta['bottom_exposure']}**")
                st.write(f"- Surface O fraction (top): **{prep_meta['surface_O_fraction_top']:.2f}**")
                st.write(f"- Slab symmetry: **{prep_meta.get('slab_symmetry', 'unknown')}**")
                st.write(f"- Rule validity: **{prep_meta.get('rule_validity', 'warn')}** | role: **{prep_meta.get('rule_role', 'exploratory')}**")
                st.write(f"- Surface diagnostics: **{prep_meta.get('surface_diagnostics_status', 'warn')}** | slab usability: **{prep_meta.get('slab_usability', 'exploratory_only')}**")
                if prep_meta.get('slab_usability_reason'):
                    st.caption(f"Slab usability: {prep_meta.get('slab_usability_reason')}")
                if prep_meta.get('oxide_rule_notes'):
                    for _note in prep_meta.get('oxide_rule_notes', [])[:2]:
                        st.caption(f"Oxide rule: {_note}")
                if prep_meta.get('surface_diagnostics_notes'):
                    for _note in prep_meta.get('surface_diagnostics_notes', [])[:2]:
                        st.caption(f"Surface diagnostic: {_note}")
                if prep_meta.get('slab_symmetry') == 'asymmetric':
                    st.caption("Oxide note: top/bottom terminations are asymmetric. Interpret clean-slab HER outputs cautiously.")
                elif prep_meta.get('slab_symmetry') == 'quasi-symmetric':
                    st.caption("Oxide note: top/bottom terminations are only quasi-symmetric. Treat representative-surface claims cautiously.")
                if bool(globals().get("is_oer", False)):
                    try:
                        oer_qc = _oxide_oer_cation_metrics(prepared)
                        st.markdown("**OER AEM cation-site slab QC**")
                        st.write(f"- OER slab suitability: **{oer_qc.get('oer_slab_suitability', 'unknown')}**")
                        st.write(f"- Top cation count: **{oer_qc.get('oer_top_cation_count', 'NA')}** | top anion count: **{oer_qc.get('oer_top_anion_count', 'NA')}**")
                        st.write(f"- Top cation symbols: **{oer_qc.get('oer_top_cation_symbols', '')}**")
                        st.write(f"- Candidate cation indices: **{oer_qc.get('oer_candidate_cation_indices', '')}**")
                        if oer_qc.get('oer_slab_warning'):
                            st.warning(str(oer_qc.get('oer_slab_warning')))
                    except Exception as e:
                        st.warning(f"OER slab QC failed: {e}")

        bulk_like = (vac_z < 10.0) and bool(prepared.get_pbc()[2])
        if bulk_like:
            st.warning(
                "BULK-like detected. Surface sites become ill-defined and many candidates may collapse/collide.\n\n"
                "Recommended: add sufficient vacuum (e.g., 30 Å) or use the slabify route."
            )

        rec = getattr(rep, "recommend_repeat", None)
        if rec:
            nx, ny, nz = rec
            if int(nz) > 1:
                st.info(f"Recommend repeat (raw): {nx}×{ny}×{nz} (forcing nz=1 is recommended for surfaces).")
            else:
                st.info(f"Recommend repeat: {nx}×{ny}×{nz}")

        if getattr(rep, "issues", None):
            st.caption("Structure issues")
            for msg in rep.issues:
                st.write(f"- {msg}")

        with st.expander("Prepared history (what changed?)", expanded=False):
            st.json(_jsonable(st.session_state.get("prepared_history") or []))

    with colB:
        st.markdown("### Preview and common tools")
        show_atoms_3d(prepared, height=360, width=520, tag="prepared")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset prepared", key="btn_reset_prepared"):
                _reset_prepared_from_working()
                try:
                    st.session_state["slab_reduction_base_atoms"] = st.session_state.get("atoms_prepared").copy()
                except Exception:
                    pass
                st.session_state["surface_setup_stage"] = (0 if surface_route == "Slabify from bulk" else 1)
                st.success("Prepared reset to working.")
                st.rerun()

        with c2:
            st.download_button(
                "Download PREPARED CIF",
                atoms_to_cif_bytes(prepared, symprec=0.1),
                file_name="prepared_structure.cif",
                mime="chemical/x-cif",
                key="dl_prepared_cif",
            )

        st.markdown("---")
        status_lines = []
        status_lines.append(f"Slab selected: **{'yes' if (surface_route != 'Slabify from bulk' or surface_setup_stage >= 1) else 'no'}**")
        status_lines.append(f"Vacuum reviewed: **{'yes' if surface_setup_stage >= 2 else 'no'}**")
        status_lines.append(f"XY supercell reviewed: **{'yes' if surface_setup_stage >= 3 else 'no'}**")
        status_lines.append(f"Slab reduction reviewed: **{'yes' if surface_setup_stage >= 4 else 'no'}**")
        status_lines.append(f"Surface engineering reviewed: **{'yes' if surface_setup_stage >= 5 else 'no'}**")
        st.caption(" | ".join(status_lines))

        if surface_setup_stage == 1:
            st.markdown("#### 2-2. Set vacuum")
            st.caption("Recommended next step: set the z vacuum before XY supercell expansion.")
            vac_choice = st.selectbox(
                "Target vacuum_z",
                ["20 Å", "30 Å (recommended)", "40 Å", "Custom"],
                index=1,
                key="common_vac_choice",
            )
            vac_custom = None
            if vac_choice == "Custom":
                vac_custom = st.number_input(
                    "Custom total vacuum_z (Å)",
                    min_value=8.0,
                    max_value=80.0,
                    value=30.0,
                    step=1.0,
                    key="common_vac_custom",
                )
            keep_pbc_z = st.checkbox("Keep pbc_z=True", value=True, key="vac_keep_pbc_z")
            target_vac = _vacuum_target_from_ui(vac_choice, vac_custom)
            v1, v2 = st.columns(2)
            with v1:
                if st.button("Apply vacuum and continue", key="btn_apply_vacuum_common"):
                    a2 = add_vacuum_z(prepared, total_vacuum_z=float(target_vac), keep_pbc_z=bool(keep_pbc_z))
                    _push_prepared_update(a2, "add_vacuum", {"total_vacuum_z": float(target_vac), "keep_pbc_z": bool(keep_pbc_z)})
                    st.session_state["surface_setup_stage"] = 2
                    st.success(f"Vacuum set to {float(target_vac):.1f} Å.")
                    st.rerun()
            with v2:
                if st.button("Skip to XY supercell →", key="btn_skip_to_xy"):
                    st.session_state["surface_setup_stage"] = 2
                    st.rerun()
            if surface_route == "Slabify from bulk":
                if st.button("← Back to slab selection", key="btn_back_stage1"):
                    st.session_state["surface_setup_stage"] = 0
                    st.rerun()

        elif surface_setup_stage == 2:
            st.markdown("#### 2-3. Expand XY supercell")
            st.caption("Recommended next step: adjust the XY supercell first, then reduce slab thickness from that XY-expanded base slab.")
            if surface_route == "Slabify from bulk":
                st.caption("For the slabify route, keep 1×1 by default and expand only when the in-plane slab is too small.")
                a_len, b_len = _surface_xy_lengths(prepared)
                nx_auto, ny_auto = _suggest_minimal_xy_repeat(prepared, min_length_a=8.0, min_length_b=8.0, max_repeat=3)
                st.write(f"- In-plane lengths: **a = {a_len:.2f} Å**, **b = {b_len:.2f} Å**")
                st.write(f"- Minimal suggested repeat: **{nx_auto}×{ny_auto}×1**")

                colR1, colR2, colR3 = st.columns(3)
                with colR1:
                    if st.button("Keep 1×1 and continue", key="btn_rep_keep_111"):
                        try:
                            st.session_state["slab_reduction_base_atoms"] = prepared.copy()
                        except Exception:
                            pass
                        st.session_state["surface_setup_stage"] = 3
                        st.info("Kept current slab at 1×1.")
                        st.rerun()
                with colR2:
                    if st.button("Apply minimal repeat", key="btn_rep_auto_minimal"):
                        if int(nx_auto) == 1 and int(ny_auto) == 1:
                            try:
                                st.session_state["slab_reduction_base_atoms"] = prepared.copy()
                            except Exception:
                                pass
                            st.session_state["surface_setup_stage"] = 3
                            st.info("Current slab already satisfies the minimal in-plane size target.")
                            st.rerun()
                        else:
                            a2 = repeat_xy(prepared, int(nx_auto), int(ny_auto))
                            _push_prepared_update(a2, "repeat_xy_minimal", {"nx": int(nx_auto), "ny": int(ny_auto), "from": "slabify_minimal_xy"})
                            try:
                                st.session_state["slab_reduction_base_atoms"] = a2.copy()
                            except Exception:
                                pass
                            st.session_state["surface_setup_stage"] = 3
                            st.success(f"Applied minimal repeat: {int(nx_auto)}×{int(ny_auto)}×1.")
                            st.rerun()
                with colR3:
                    if st.button("Apply 2×2×1", key="btn_rep_221_slabify"):
                        a2 = repeat_xy(prepared, 2, 2)
                        _push_prepared_update(a2, "repeat_xy", {"nx": 2, "ny": 2, "from": "slabify_manual"})
                        try:
                            st.session_state["slab_reduction_base_atoms"] = a2.copy()
                        except Exception:
                            pass
                        st.session_state["surface_setup_stage"] = 3
                        st.success("Applied 2×2×1.")
                        st.rerun()
            else:
                st.caption("Uploaded or current slabs may be kept at their existing XY size. Expand only when the in-plane cell is too small for the intended screening.")
                colR0, colR1, colR2, colR3 = st.columns(4)
                with colR0:
                    if st.button("Keep current XY and continue", key="btn_rep_keep_current_upload"):
                        try:
                            st.session_state["slab_reduction_base_atoms"] = prepared.copy()
                        except Exception:
                            pass
                        st.session_state["surface_setup_stage"] = 3
                        st.info("Kept current XY supercell without expansion.")
                        st.rerun()
                with colR1:
                    if st.button("2×2×1", key="btn_rep_221"):
                        a2 = repeat_xy(prepared, 2, 2)
                        _push_prepared_update(a2, "repeat_xy", {"nx": 2, "ny": 2})
                        try:
                            st.session_state["slab_reduction_base_atoms"] = a2.copy()
                        except Exception:
                            pass
                        st.session_state["surface_setup_stage"] = 3
                        st.success("Applied 2×2×1.")
                        st.rerun()
                with colR2:
                    if st.button("4×4×1", key="btn_rep_441"):
                        a2 = repeat_xy(prepared, 4, 4)
                        _push_prepared_update(a2, "repeat_xy", {"nx": 4, "ny": 4})
                        try:
                            st.session_state["slab_reduction_base_atoms"] = a2.copy()
                        except Exception:
                            pass
                        st.session_state["surface_setup_stage"] = 3
                        st.success("Applied 4×4×1.")
                        st.rerun()
                with colR3:
                    if st.button("Use auto recommendation", key="btn_rep_auto"):
                        rec = getattr(rep, "recommend_repeat", None)
                        if not rec:
                            st.warning("No repeat recommendation available. Use 'Keep current XY and continue' if the uploaded slab size is intentional.")
                        else:
                            nx, ny, _nz = rec
                            if int(nx) == 1 and int(ny) == 1:
                                try:
                                    st.session_state["slab_reduction_base_atoms"] = prepared.copy()
                                except Exception:
                                    pass
                                st.session_state["surface_setup_stage"] = 3
                                st.info("Auto recommendation is 1×1×1. Kept current XY supercell.")
                                st.rerun()
                            else:
                                a2 = repeat_xy(prepared, int(nx), int(ny))
                                _push_prepared_update(a2, "repeat_xy_auto", {"nx": int(nx), "ny": int(ny), "from": "validate_structure"})
                                try:
                                    st.session_state["slab_reduction_base_atoms"] = a2.copy()
                                except Exception:
                                    pass
                                st.session_state["surface_setup_stage"] = 3
                                st.success(f"Applied {int(nx)}×{int(ny)}×1.")
                                st.rerun()
            if st.button("← Back to vacuum", key="btn_back_stage2"):
                st.session_state["surface_setup_stage"] = 1
                st.rerun()

        elif surface_setup_stage == 3:
            st.markdown("#### 2-4. Reduce slab thickness")
            st.caption("Recommended next step: reduce slab thickness from the final XY supercell, then review the prepared slab.")
            if reduce_slab_symmetrically is None or get_slab_reduction_presets is None:
                st.caption("slab_reduction.py is not available in the current app path.")
            else:
                reduction_base = st.session_state.get("slab_reduction_base_atoms") or prepared
                base_layers = len(_cluster_z_layers_simple(reduction_base, tol=0.8))
                st.write(f"- Reduction base z-layers: **{base_layers}**")
                reduction_presets = get_slab_reduction_presets()
                reduction_label_to_level = {
                    "None (keep current thickness)": "None",
                    "Medium (recommended)": "Medium",
                    "Large": "Large",
                    "Small (aggressive)": "Small",
                    "Custom": "Custom",
                }
                if mtype == "metal":
                    reduction_options = [
                        "None (keep current thickness)",
                        "Medium (recommended)",
                        "Large",
                        "Small (aggressive)",
                        "Custom",
                    ]
                    reduction_default_index = 0
                else:
                    reduction_options = [
                        "Medium (recommended)",
                        "Large",
                        "Small (aggressive)",
                        "Custom",
                    ]
                    reduction_default_index = 0
                reduction_mode_label = st.selectbox(
                    "Slab reduction preset",
                    reduction_options,
                    index=reduction_default_index,
                    key="step2_slab_reduction_level",
                    help="The reduction target is defined by preserved z-layer count on the XY-expanded base slab.",
                )
                reduction_mode = reduction_label_to_level.get(str(reduction_mode_label), "Medium")
                target_preserved_layers = None
                if reduction_mode == "None":
                    st.caption("Keep the current XY-expanded slab thickness without z-layer reduction.")
                elif reduction_mode in reduction_presets:
                    preset_meta = reduction_presets.get(reduction_mode, {})
                    st.caption(str(preset_meta.get("description", "")))
                    target_preserved_layers = int(preset_meta.get("target_preserved_layers", 4))
                    st.write(f"- Target preserved layers: **{target_preserved_layers}**")
                    if reduction_mode == "Small":
                        st.warning("Small is aggressive. Oxide terminations may become unstable or fragmented.")
                elif reduction_mode == "Custom":
                    target_preserved_layers = int(st.number_input(
                        "Custom preserved z-layer count",
                        min_value=2,
                        max_value=max(2, int(base_layers)),
                        value=min(max(2, int(base_layers)), 4),
                        step=1,
                        key="step2_custom_preserved_layers",
                    ))
                    st.caption("Custom uses the preserved z-layer count on the current XY-expanded base slab. Parity is adjusted automatically when needed.")

                r1, r2, r3 = st.columns(3)
                with r1:
                    if st.button("Apply slab reduction", key="btn_apply_slab_reduction"):
                        try:
                            if reduction_mode == "None":
                                a2 = reduction_base.copy()
                                reduction_meta = {
                                    "reduced": False,
                                    "reduction_level": "None",
                                    "reason": "Metal preset selected: keep current slab thickness.",
                                    "original_layer_count": int(base_layers),
                                    "kept_layer_count": int(base_layers),
                                    "original_atoms": int(len(reduction_base)),
                                    "reduced_atoms": int(len(reduction_base)),
                                    "original_thickness_A": float(slab_thickness_z(reduction_base)),
                                    "reduced_thickness_A": float(slab_thickness_z(reduction_base)),
                                }
                                _push_prepared_update(a2, "slab_reduction_none", reduction_meta)
                                st.info("No slab reduction applied. Current thickness was preserved.")
                            else:
                                a2, reduction_meta = reduce_slab_symmetrically(
                                    reduction_base,
                                    level=(None if reduction_mode == "Custom" else str(reduction_mode)),
                                    target_preserved_layers=int(target_preserved_layers) if target_preserved_layers is not None else None,
                                    keep_pbc_z=True,
                                )
                                _push_prepared_update(a2, "slab_reduction", reduction_meta)
                                if bool(reduction_meta.get("reduced", False)):
                                    st.success(
                                        f"Applied {reduction_meta.get('reduction_level')} slab reduction: layers "
                                        f"{reduction_meta.get('original_layer_count')} → {reduction_meta.get('kept_layer_count')}, atoms "
                                        f"{reduction_meta.get('original_atoms')} → {reduction_meta.get('reduced_atoms')}, thickness "
                                        f"{reduction_meta.get('original_thickness_A', float('nan')):.2f} → {reduction_meta.get('reduced_thickness_A', float('nan')):.2f} Å."
                                    )
                                else:
                                    st.info(str(reduction_meta.get("reason", "No reduction was needed.")))
                            st.session_state["surface_setup_stage"] = 4
                            st.rerun()
                        except Exception as e:
                            st.error(f"Slab reduction failed: {e}")
                with r2:
                    if st.button("Skip to review →", key="btn_skip_to_review"):
                        st.session_state["surface_setup_stage"] = 4
                        st.rerun()
                with r3:
                    if st.button("← Back to XY supercell", key="btn_back_stage3"):
                        st.session_state["surface_setup_stage"] = 2
                        st.rerun()


        elif surface_setup_stage == 4:
            st.markdown("#### 2-5. Review prepared slab")
            st.success("The parent slab is prepared. Surface engineering is optional.")
            st.write(f"- Atoms: **{len(prepared)}**")
            st.write(f"- Vacuum_z: **{float(getattr(rep, 'vacuum_z', 0.0)):.2f} Å**")
            if mtype == "oxide":
                surf_meta_review = _annotate_step2_slab_symmetry(_classify_surface_exposure(prepared, z_window=1.8))
                st.write(f"- Slab symmetry: **{surf_meta_review.get('slab_symmetry', 'unknown')}**")
                st.caption(str(surf_meta_review.get('slab_symmetry_basis', '')))
            cdone1, cdone2 = st.columns(2)
            with cdone1:
                if st.button("Continue to surface engineering →", key="btn_step2_to_engineering"):
                    st.session_state["surface_engineering_base_atoms"] = prepared.copy()
                    st.session_state["surface_engineering_base_signature"] = structure_content_signature(prepared)
                    st.session_state["surface_engineering_applied_signature"] = None
                    st.session_state["surface_engineering_candidates"] = []
                    st.session_state["surface_engineering_selected_index"] = 0
                    st.session_state["surface_engineering_candidate_select"] = 0
                    st.session_state["surface_engineering_manual_atom_index"] = None
                    st.session_state["surface_engineering_manual_site_index"] = None
                    st.session_state["surface_setup_stage"] = 5
                    st.rerun()
            with cdone2:
                if st.button("← Back to slab reduction", key="btn_back_stage4"):
                    st.session_state["surface_setup_stage"] = 3
                    st.rerun()

        else:
            st.markdown("#### 2-6. Surface engineering")
            st.caption(
                "Generate deterministic single-substitution, single-vacancy, or single-adatom candidates "
                "from the prepared parent slab. This stage performs structural generation and geometry checks only; "
                "it does not claim synthesis feasibility or thermodynamic stability."
            )

            current_sig = structure_content_signature(prepared)
            base_atoms = st.session_state.get("surface_engineering_base_atoms")
            base_sig = st.session_state.get("surface_engineering_base_signature")
            applied_sig = st.session_state.get("surface_engineering_applied_signature")
            if base_atoms is None or (current_sig not in {base_sig, applied_sig}):
                st.session_state["surface_engineering_base_atoms"] = prepared.copy()
                st.session_state["surface_engineering_base_signature"] = current_sig
                st.session_state["surface_engineering_applied_signature"] = None
                st.session_state["surface_engineering_candidates"] = []
                st.session_state["surface_engineering_selected_index"] = 0
                st.session_state["surface_engineering_candidate_select"] = 0
                base_atoms = prepared.copy()
                base_sig = current_sig
            else:
                base_atoms = base_atoms.copy()

            base_analysis = analyze_parent_slab(base_atoms)
            base_symbols = sorted(set(base_atoms.get_chemical_symbols()))
            st.write(
                f"- Parent: **{base_atoms.get_chemical_formula()}**, "
                f"**{len(base_atoms)} atoms**, "
                f"**{len(base_analysis.get('layers', []))} z-layers**"
            )
            if applied_sig and current_sig == applied_sig:
                st.info("An engineered candidate is currently applied. New candidates are still generated from the preserved parent slab.")

            operation = st.radio(
                "Structure operation",
                ["None", "Single substitution", "Single vacancy", "Single adatom"],
                horizontal=True,
                key="surface_engineering_operation",
            )
            operation_prev = st.session_state.get("_surface_engineering_operation_prev")
            if operation_prev != operation:
                st.session_state["_surface_engineering_operation_prev"] = operation
                st.session_state["surface_engineering_candidates"] = []
                st.session_state["surface_engineering_selected_index"] = 0
                st.session_state["surface_engineering_candidate_select"] = 0
                st.session_state["surface_engineering_manual_atom_index"] = None
                st.session_state["surface_engineering_manual_site_index"] = None

            selection_policy = None
            if operation != "None":
                selection_policy = st.radio(
                    "Position selection",
                    ["Automatic distinct candidates", "Manual position"],
                    horizontal=True,
                    key="surface_engineering_selection_policy",
                    help=(
                        "Automatic mode keeps one representative from each local-environment orbit. "
                        "Manual mode lets you click an exact parent atom or adsorption-site marker."
                    ),
                )
                policy_prev = st.session_state.get("_surface_engineering_selection_policy_prev")
                if policy_prev != selection_policy:
                    st.session_state["_surface_engineering_selection_policy_prev"] = selection_policy
                    st.session_state["surface_engineering_candidates"] = []
                    st.session_state["surface_engineering_selected_index"] = 0
                    st.session_state["surface_engineering_candidate_select"] = 0
                    st.session_state["surface_engineering_manual_atom_index"] = None
                    st.session_state["surface_engineering_manual_site_index"] = None

            candidates = st.session_state.get("surface_engineering_candidates", []) or []
            generation_error = None

            if operation == "Single substitution":
                c1, c2, c3 = st.columns(3)
                with c1:
                    host = st.selectbox("Host element", base_symbols, key="eng_sub_host")
                with c2:
                    dopant_options = [s for s in chemical_symbols[1:] if s and s != host]
                    preferred_dopant = "Ag" if str(host) != "Ag" else "Cu"
                    dopant_default = dopant_options.index(preferred_dopant) if preferred_dopant in dopant_options else 0
                    dopant = st.selectbox(
                        "Dopant element",
                        dopant_options,
                        index=dopant_default,
                        key="eng_sub_dopant",
                    )
                with c3:
                    depth = st.selectbox(
                        "Depth",
                        ["surface", "subsurface", "surface+subsurface"],
                        key="eng_sub_depth",
                    )

                material_class = infer_structure_material_class(base_atoms)
                geometry_diag = substitution_geometry_diagnostics(
                    base_atoms,
                    host=str(host),
                    dopant=str(dopant),
                )
                host_oxidation_state = None
                dopant_oxidation_state = None
                shared_ligand_weight = 0.50

                with st.expander("Local geometry initialization", expanded=True):
                    if material_class == "oxide":
                        suggestions = suggested_substitution_oxidation_states(
                            base_atoms,
                            host=str(host),
                            dopant=str(dopant),
                            host_role=str(geometry_diag.get("host_role", "cation")),
                            dopant_role=str(geometry_diag.get("dopant_role", "cation")),
                        )
                        st.write("Model: **oxide coordination-polyhedron initialization**")
                        st.caption(
                            "Only the directly coordinated opposite-sublattice atoms are adjusted. "
                            "For cation substitution, first-shell anion ligands move; shared ligands are damped. "
                            "For anion substitution, neighboring cations move more conservatively."
                        )
                        st.write(
                            f"Host role: **{geometry_diag.get('host_role', 'unknown')}** · "
                            f"Dopant role: **{geometry_diag.get('dopant_role', 'unknown')}**"
                        )
                        if bool(geometry_diag.get("cross_sublattice", False)):
                            st.error(
                                "Cross-sublattice substitution was selected. Automatic distance adjustment "
                                "will be disabled for this candidate and the host lattice positions retained."
                            )

                        inferred = dict(suggestions.get("inference", {}) or {})
                        if bool(inferred.get("ambiguous", False)):
                            st.warning(
                                "Oxidation-state inference is ambiguous for this composition. "
                                "Review the selected oxidation states manually."
                            )
                        elif inferred.get("assignments"):
                            st.caption(
                                "Best-effort charge-neutral assignment: "
                                + ", ".join(
                                    f"{el}{int(ox):+d}"
                                    for el, ox in sorted(inferred.get("assignments", {}).items())
                                )
                            )

                        host_options = list(suggestions.get("host_options", []) or oxidation_state_options(str(host)))
                        dopant_options_ox = list(suggestions.get("dopant_options", []) or oxidation_state_options(str(dopant)))
                        suggested_host = suggestions.get("host_oxidation_state")
                        suggested_dopant = suggestions.get("dopant_oxidation_state")
                        if suggested_host not in host_options and suggested_host is not None:
                            host_options = [int(suggested_host)] + host_options
                        if suggested_dopant not in dopant_options_ox and suggested_dopant is not None:
                            dopant_options_ox = [int(suggested_dopant)] + dopant_options_ox

                        ox1, ox2 = st.columns(2)
                        with ox1:
                            host_default = host_options.index(int(suggested_host)) if suggested_host in host_options else 0
                            host_oxidation_state = st.selectbox(
                                "Host oxidation state",
                                host_options,
                                index=host_default,
                                format_func=lambda x: f"{int(x):+d}",
                                key="eng_sub_host_oxidation",
                            )
                        with ox2:
                            dopant_default_ox = (
                                dopant_options_ox.index(int(suggested_dopant))
                                if suggested_dopant in dopant_options_ox else 0
                            )
                            dopant_oxidation_state = st.selectbox(
                                "Dopant oxidation state",
                                dopant_options_ox,
                                index=dopant_default_ox,
                                format_func=lambda x: f"{int(x):+d}",
                                key="eng_sub_dopant_oxidation",
                            )

                        if int(host_oxidation_state) != int(dopant_oxidation_state):
                            st.warning(
                                f"Charge mismatch: {host}{int(host_oxidation_state):+d} → "
                                f"{dopant}{int(dopant_oxidation_state):+d}. "
                                "This generator does not automatically add vacancies, co-dopants, or polarons."
                            )

                        local_adjustment = st.checkbox(
                            "Apply oxide polyhedron initialization",
                            value=True,
                            key="eng_sub_local_adjustment",
                            help="This is a starting-geometry adjustment, not an energy relaxation.",
                        )
                        g1, g2, g3 = st.columns(3)
                        with g1:
                            adjustment_strength = st.slider(
                                "Polyhedron adjustment strength",
                                min_value=0.0,
                                max_value=0.60,
                                value=0.25,
                                step=0.05,
                                key="eng_sub_adjustment_strength",
                                disabled=not local_adjustment,
                            )
                        with g2:
                            shared_ligand_weight = st.slider(
                                "Shared-ligand weight",
                                min_value=0.10,
                                max_value=1.00,
                                value=0.50,
                                step=0.05,
                                key="eng_sub_shared_ligand_weight",
                                disabled=not local_adjustment,
                            )
                        with g3:
                            max_local_displacement_A = st.number_input(
                                "Max displacement / ligand (Å)",
                                min_value=0.02,
                                max_value=0.30,
                                value=0.12,
                                step=0.02,
                                key="eng_sub_max_local_displacement",
                                disabled=not local_adjustment,
                            )
                        adjustment_shells = 1
                        st.caption(
                            "Default: first coordination polyhedron only, 25% of the ionic-radius difference, "
                            "shared-ligand damping 0.50, maximum displacement 0.12 Å."
                        )
                    else:
                        radius_diag = substitution_radius_diagnostics(str(host), str(dopant))
                        st.write("Model: **metallic local-neighbor initialization**")
                        st.caption(
                            "The dopant remains at the host lattice site. First- and second-shell metal atoms "
                            "are shifted using bounded covalent-radius ratios."
                        )
                        signed_pct = 100.0 * float(radius_diag["signed_radius_mismatch_fraction"])
                        direction = (
                            "local expansion" if signed_pct > 0
                            else ("local contraction" if signed_pct < 0 else "no radius-driven change")
                        )
                        st.write(
                            f"Host radius: **{float(radius_diag['host_radius_A']):.2f} Å** · "
                            f"Dopant radius: **{float(radius_diag['dopant_radius_A']):.2f} Å** · "
                            f"Signed mismatch: **{signed_pct:+.1f}%** ({direction})"
                        )
                        local_adjustment = st.checkbox(
                            "Apply metallic radius-guided initialization",
                            value=True,
                            key="eng_sub_local_adjustment",
                        )
                        g1, g2, g3 = st.columns(3)
                        with g1:
                            adjustment_strength = st.slider(
                                "Adjustment strength",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.50,
                                step=0.05,
                                key="eng_sub_adjustment_strength",
                                disabled=not local_adjustment,
                            )
                        with g2:
                            adjustment_shells = st.selectbox(
                                "Adjusted neighbor shells",
                                [1, 2],
                                index=1,
                                key="eng_sub_adjustment_shells",
                                disabled=not local_adjustment,
                            )
                        with g3:
                            max_local_displacement_A = st.number_input(
                                "Max displacement / atom (Å)",
                                min_value=0.05,
                                max_value=0.50,
                                value=0.20,
                                step=0.05,
                                key="eng_sub_max_local_displacement",
                                disabled=not local_adjustment,
                            )
                        st.caption(
                            "Default: 50% of the radius-predicted change, two shells, "
                            "maximum 0.20 Å displacement per neighboring atom."
                        )

                if selection_policy == "Manual position":
                    eligible = eligible_atom_indices(
                        base_analysis,
                        element=str(host),
                        depth=str(depth),
                    )
                    selected_atom = render_atom_picker(
                        base_atoms,
                        base_analysis,
                        selectable_indices=eligible,
                        selected_index=st.session_state.get("surface_engineering_manual_atom_index"),
                        key=f"eng_sub_picker_{base_sig}_{host}_{depth}",
                        title=f"Click the {host} atom to replace with {dopant}",
                    )
                    st.session_state["surface_engineering_manual_atom_index"] = selected_atom
                    if selected_atom is not None:
                        env = base_analysis["environment_by_index"][int(selected_atom)]
                        st.info(
                            f"Selected parent atom #{selected_atom}: {host}, "
                            f"{env.depth_class}, layer {env.layer_id}, CN={env.coordination_number}."
                        )
                    if st.button(
                        "Generate substitution at selected atom",
                        key="btn_eng_generate_sub_manual",
                        disabled=(selected_atom is None),
                    ):
                        try:
                            candidate = build_substitution_candidate_at_index(
                                base_atoms,
                                host=str(host),
                                dopant=str(dopant),
                                target_index=int(selected_atom),
                                depth=str(depth),
                                apply_local_adjustment=bool(local_adjustment),
                                adjustment_strength=float(adjustment_strength),
                                adjustment_shells=int(adjustment_shells),
                                max_local_displacement_A=float(max_local_displacement_A),
                                protect_bottom_layers=1,
                                host_oxidation_state=host_oxidation_state,
                                dopant_oxidation_state=dopant_oxidation_state,
                                shared_ligand_weight=float(shared_ligand_weight),
                            )
                            candidates = [candidate]
                            st.session_state["surface_engineering_candidates"] = candidates
                            st.session_state["surface_engineering_selected_index"] = 0
                            st.session_state["surface_engineering_candidate_select"] = 0
                        except Exception as exc:
                            generation_error = str(exc)
                elif st.button("Generate substitution candidates", key="btn_eng_generate_sub"):
                    try:
                        candidates = enumerate_substitution_candidates(
                            base_atoms,
                            host=str(host),
                            dopant=str(dopant),
                            depth=str(depth),
                            max_candidates=20,
                            apply_local_adjustment=bool(local_adjustment),
                            adjustment_strength=float(adjustment_strength),
                            adjustment_shells=int(adjustment_shells),
                            max_local_displacement_A=float(max_local_displacement_A),
                            protect_bottom_layers=1,
                            host_oxidation_state=host_oxidation_state,
                            dopant_oxidation_state=dopant_oxidation_state,
                            shared_ligand_weight=float(shared_ligand_weight),
                        )
                        st.session_state["surface_engineering_candidates"] = candidates
                        st.session_state["surface_engineering_selected_index"] = 0
                        st.session_state["surface_engineering_candidate_select"] = 0
                    except Exception as exc:
                        generation_error = str(exc)

            elif operation == "Single vacancy":
                c1, c2 = st.columns(2)
                with c1:
                    vacancy_element = st.selectbox("Removed element", base_symbols, key="eng_vac_element")
                with c2:
                    vacancy_depth = st.selectbox(
                        "Depth",
                        ["surface", "subsurface", "surface+subsurface"],
                        key="eng_vac_depth",
                    )

                if selection_policy == "Manual position":
                    eligible = eligible_atom_indices(
                        base_analysis,
                        element=str(vacancy_element),
                        depth=str(vacancy_depth),
                    )
                    selected_atom = render_atom_picker(
                        base_atoms,
                        base_analysis,
                        selectable_indices=eligible,
                        selected_index=st.session_state.get("surface_engineering_manual_atom_index"),
                        key=f"eng_vac_picker_{base_sig}_{vacancy_element}_{vacancy_depth}",
                        title=f"Click the {vacancy_element} atom to remove",
                    )
                    st.session_state["surface_engineering_manual_atom_index"] = selected_atom
                    if selected_atom is not None:
                        env = base_analysis["environment_by_index"][int(selected_atom)]
                        st.info(
                            f"Selected parent atom #{selected_atom}: {vacancy_element}, "
                            f"{env.depth_class}, layer {env.layer_id}, CN={env.coordination_number}."
                        )
                    if st.button(
                        "Generate vacancy at selected atom",
                        key="btn_eng_generate_vac_manual",
                        disabled=(selected_atom is None),
                    ):
                        try:
                            candidate = build_vacancy_candidate_at_index(
                                base_atoms,
                                element=str(vacancy_element),
                                target_index=int(selected_atom),
                                depth=str(vacancy_depth),
                            )
                            candidates = [candidate]
                            st.session_state["surface_engineering_candidates"] = candidates
                            st.session_state["surface_engineering_selected_index"] = 0
                            st.session_state["surface_engineering_candidate_select"] = 0
                        except Exception as exc:
                            generation_error = str(exc)
                elif st.button("Generate vacancy candidates", key="btn_eng_generate_vac"):
                    try:
                        candidates = enumerate_vacancy_candidates(
                            base_atoms,
                            element=str(vacancy_element),
                            depth=str(vacancy_depth),
                            max_candidates=20,
                        )
                        st.session_state["surface_engineering_candidates"] = candidates
                        st.session_state["surface_engineering_selected_index"] = 0
                        st.session_state["surface_engineering_candidate_select"] = 0
                    except Exception as exc:
                        generation_error = str(exc)

            elif operation == "Single adatom":
                c1, c2, c3 = st.columns(3)
                with c1:
                    adatom_options = chemical_symbols[1:]
                    adatom_default = adatom_options.index("Cu") if "Cu" in adatom_options else 0
                    adatom = st.selectbox(
                        "Adatom element",
                        adatom_options,
                        index=adatom_default,
                        key="eng_ad_element",
                    )
                with c2:
                    site_kinds = st.multiselect(
                        "Initial site classes",
                        ["ontop", "bridge", "hollow"],
                        default=["ontop", "bridge", "hollow"],
                        key="eng_ad_site_kinds",
                    )
                with c3:
                    distance_scale = st.number_input(
                        "Initial distance scale",
                        min_value=0.80,
                        max_value=1.30,
                        value=1.00,
                        step=0.05,
                        key="eng_ad_distance_scale",
                    )

                if selection_policy == "Manual position":
                    try:
                        manual_sites = detect_selectable_adatom_sites(
                            base_atoms,
                            site_kinds=tuple(site_kinds),
                            max_sites_per_kind=200,
                        )
                    except Exception as exc:
                        manual_sites = []
                        generation_error = str(exc)
                    selected_site_index = render_adatom_site_picker(
                        base_atoms,
                        manual_sites,
                        selected_site_index=st.session_state.get("surface_engineering_manual_site_index"),
                        key=f"eng_ad_picker_{base_sig}_{'_'.join(site_kinds)}",
                    )
                    st.session_state["surface_engineering_manual_site_index"] = selected_site_index
                    if selected_site_index is not None and manual_sites:
                        site = manual_sites[int(selected_site_index)]
                        kind = "hollow" if str(site.kind).lower() in {"fcc", "hcp"} else str(site.kind)
                        st.info(
                            f"Selected site {selected_site_index}: {kind}, "
                            f"support parent atoms={tuple(int(i) for i in site.surface_indices)}."
                        )
                    if st.button(
                        "Generate adatom at selected site",
                        key="btn_eng_generate_ad_manual",
                        disabled=(selected_site_index is None or not manual_sites),
                    ):
                        try:
                            site = manual_sites[int(selected_site_index)]
                            candidate = build_adatom_candidate_at_site(
                                base_atoms,
                                adatom=str(adatom),
                                site=site,
                                distance_scale=float(distance_scale),
                                site_index=int(selected_site_index),
                            )
                            candidates = [candidate]
                            st.session_state["surface_engineering_candidates"] = candidates
                            st.session_state["surface_engineering_selected_index"] = 0
                            st.session_state["surface_engineering_candidate_select"] = 0
                        except Exception as exc:
                            generation_error = str(exc)
                elif st.button("Generate adatom candidates", key="btn_eng_generate_ad"):
                    try:
                        candidates = enumerate_adatom_candidates(
                            base_atoms,
                            adatom=str(adatom),
                            site_kinds=tuple(site_kinds),
                            distance_scale=float(distance_scale),
                            max_candidates=20,
                        )
                        st.session_state["surface_engineering_candidates"] = candidates
                        st.session_state["surface_engineering_selected_index"] = 0
                        st.session_state["surface_engineering_candidate_select"] = 0
                    except Exception as exc:
                        generation_error = str(exc)

            else:
                st.info("No structural modification selected. The prepared parent slab remains active.")


            if generation_error:
                st.error(f"Candidate generation failed: {generation_error}")

            candidates = st.session_state.get("surface_engineering_candidates", []) or []
            if candidates:
                summary_df = pd.DataFrame(candidate_summary_records(candidates))
                st.markdown("##### Generated candidates")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                selected_index = st.selectbox(
                    "Candidate preview",
                    options=list(range(len(candidates))),
                    index=min(
                        int(st.session_state.get("surface_engineering_selected_index", 0)),
                        max(len(candidates) - 1, 0),
                    ),
                    format_func=lambda i: candidates[int(i)].label,
                    key="surface_engineering_candidate_select",
                )
                st.session_state["surface_engineering_selected_index"] = int(selected_index)
                selected = candidates[int(selected_index)]

                p1, p2 = st.columns([1.2, 0.8])
                with p1:
                    show_atoms_3d(
                        selected.atoms,
                        height=390,
                        width=720,
                        tag=f"surface_engineering_{selected.candidate_id}",
                    )
                with p2:
                    st.write(f"- Candidate ID: `{selected.candidate_id}`")
                    st.write(f"- Validation: **{selected.validation.get('status', 'unknown')}**")
                    st.write(f"- Formula: **{selected.atoms.get_chemical_formula()}**")
                    st.write(
                        f"- Minimum distance: **{_safe_float(selected.validation.get('minimum_distance_A')):.3f} Å**"
                    )
                    local_meta = dict(selected.recipe.parameters.get("local_geometry_adjustment", {}) or {})
                    if selected.recipe.operation == "single_substitution" and local_meta:
                        st.write(
                            f"- Material/init model: **{local_meta.get('material_class', 'unknown')} / "
                            f"{local_meta.get('method', 'unknown')}**"
                        )
                        mismatch = local_meta.get("signed_radius_mismatch_fraction")
                        if mismatch is not None:
                            st.write(f"- Radius mismatch: **{100.0 * float(mismatch):+.1f}%**")
                        if local_meta.get("host_oxidation_state") is not None:
                            st.write(
                                f"- Oxidation states: **host {float(local_meta.get('host_oxidation_state')):+g} → "
                                f"dopant {float(local_meta.get('dopant_oxidation_state')):+g}** · "
                                f"coordination: **{int(local_meta.get('coordination_number', 0) or 0)}**"
                            )
                        st.write(
                            f"- Moved first-shell atoms: **{int(local_meta.get('n_moved_atoms', 0) or 0)}** · "
                            f"max displacement: **{_safe_float(local_meta.get('max_applied_displacement_A'), 0.0):.3f} Å**"
                        )
                        d_before = local_meta.get("mean_first_shell_distance_before_A")
                        d_after = local_meta.get("mean_first_shell_distance_after_A")
                        if d_before is not None and d_after is not None:
                            st.write(
                                f"- Mean first-shell distance: **{float(d_before):.3f} → {float(d_after):.3f} Å**"
                            )
                        if local_meta.get("initialization_warning"):
                            st.warning(str(local_meta.get("initialization_warning")))
                    for err in selected.validation.get("errors", []) or []:
                        st.error(str(err))
                    for warn in selected.validation.get("warnings", []) or []:
                        st.warning(str(warn))

                zip_bytes = export_engineered_candidates_zip(candidates)
                a1, a2, a3 = st.columns(3)
                with a1:
                    if st.button(
                        "Apply selected candidate",
                        key="btn_eng_apply_selected",
                        disabled=(selected.validation.get("status") == "reject"),
                    ):
                        _push_prepared_update(
                            selected.atoms.copy(),
                            "surface_engineering",
                            {
                                "candidate_id": selected.candidate_id,
                                "label": selected.label,
                                "recipe": selected.recipe.as_dict(),
                                "validation": selected.validation,
                            },
                        )
                        st.session_state["surface_engineering_applied_signature"] = structure_content_signature(selected.atoms)
                        st.session_state["surface_engineering_applied_candidate_id"] = selected.candidate_id
                        st.success(f"Applied {selected.label}.")
                        st.rerun()
                with a2:
                    st.download_button(
                        "Download candidate ZIP",
                        data=zip_bytes,
                        file_name="SAGE_surface_engineering_candidates.zip",
                        mime="application/zip",
                        key="btn_eng_download_zip",
                    )
                with a3:
                    if st.button("Clear candidates", key="btn_eng_clear_candidates"):
                        st.session_state["surface_engineering_candidates"] = []
                        st.session_state["surface_engineering_selected_index"] = 0
                        st.rerun()

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("Reset to parent slab", key="btn_eng_reset_parent"):
                    parent_copy = st.session_state.get("surface_engineering_base_atoms")
                    if parent_copy is not None:
                        _push_prepared_update(
                            parent_copy.copy(),
                            "surface_engineering_reset",
                            {"source": "preserved_parent_slab"},
                        )
                        st.session_state["surface_engineering_applied_signature"] = structure_content_signature(parent_copy)
                        st.session_state["surface_engineering_applied_candidate_id"] = None
                        st.rerun()
            with b2:
                if st.button("Surface engineering reviewed", key="btn_eng_reviewed"):
                    st.success("Step 2 is complete. Continue to Step 3 below.")
            with b3:
                if st.button("← Back to slab review", key="btn_back_stage5"):
                    st.session_state["surface_setup_stage"] = 4
                    st.rerun()

        if bulk_like and surface_route == "Use current structure":
            if st.button("Temporary workaround: set pbc_z=False", key="btn_pbc_false_tmp"):
                a2 = set_pbc_z(prepared, False)
                _push_prepared_update(a2, "set_pbc_z", {"pbc_z": False})
                st.success("pbc_z=False applied (temporary).")
                st.rerun()

# ---------------- STEP 3: Site selection (Geometry / ML) ----------------
st.markdown("## 3) Site selection")

working = st.session_state.get("atoms_tuned") or st.session_state.get("atoms_loaded")
if working is None:
    st.info("Load a structure first.")
else:
    _ensure_prepared_uptodate()
    atoms_for_sites = st.session_state.get("atoms_prepared")

    if int(st.session_state.get("surface_setup_stage", 0)) < 5:
        st.warning(
            "Step 2 wizard is not fully reviewed yet. Recommended order: slab selection → vacuum → "
            "XY supercell → slab reduction → surface engineering review."
        )

    # --- Surfactant-class scenario (structural conditioning) ---
    # Placed here (Step 3) because this feature changes the *structure* used for site enumeration / adsorption,
    # not a CHE correction term.
    if not is_her:
        st.markdown("### Surfactant-class surface conditioning (scenario)")

        _surf_ui = st.selectbox(
            "Surfactant class",
            ["None", "Cationic (CTAB/CTAC)", "Anionic (SDS)", "Nonionic (Triton X-100)"],
            index=0,
            key="surfactant_class_ui",
            help=(
                "Scenario proxy: conditions the slab into nearby surface states using CHGNet (slab-only) "
                "and then evaluates adsorption energetics downstream. This does NOT model explicit surfactant/EDL/solvent/potential."
            ),
        )
        _surf_map = {
            "None": "none",
            "Cationic (CTAB/CTAC)": "cationic",
            "Anionic (SDS)": "anionic",
            "Nonionic (Triton X-100)": "nonionic",
        }
        surfactant_class = _surf_map.get(_surf_ui, "none")

        surfactant_prerelax_slab = st.checkbox(
            "Apply CHGNet slab pre-relaxation",
            value=False,
            key="surfactant_prerelax_slab_ui",
            help="If enabled, the slab is pre-relaxed (slab-only) before site detection / ML screening and before the OCP run.",
        )
    else:
        # HER mode: surfactant conditioning is disabled by design (does not represent HER experiments).
        surfactant_class = "none"
        surfactant_prerelax_slab = False

    # Default conditioning params (used only if pre-relax is enabled)
    cond_top_z_tol = 2.0
    cond_jiggle_amp = 0.05
    cond_fmax = 0.05
    cond_max_steps = 200
    cond_seed_ui = 0
    cond_seed = None

    # Oxide HER constrained CHGNet preconditioning is intentionally disabled.
    # The current oxide HER workflow controls freedom through the relaxation scope
    # (rigid / partial / full), so the legacy slab preconditioning UI is hidden.
    her_constrained_prerelax = False
    her_constrained_top_free_layers = int(st.session_state.get("her_constrained_top_free_layers", 2))
    her_constrained_layer_tol = float(st.session_state.get("her_constrained_layer_tol", 0.8))
    her_constrained_fmax = float(st.session_state.get("her_constrained_fmax", 0.05))
    her_constrained_max_steps = int(st.session_state.get("her_constrained_max_steps", 80))
    her_constrained_seed_ui = int(st.session_state.get("her_constrained_seed_ui", 0))
    her_constrained_seed = None

    # Conditioning parameter UI only when the feature is enabled (CO2RR only)
    if (not is_her) and bool(surfactant_prerelax_slab):
        with st.expander("Conditioning parameters", expanded=False):
            auto_params = st.checkbox(
                "Auto set conditioning parameters (recommended)",
                value=True,
                key="cond_auto_params",
                help="Automatically pick a top-layer window and jiggle amplitude based on the slab's top-layer spacing.",
            )
            auto_profile = "Safe"
            if auto_params:
                auto_profile = st.selectbox(
                    "Auto profile",
                    ["Safe", "Explore (stronger perturbation)"],
                    index=0,
                    key="cond_auto_profile",
                    help="Safe: conservative perturbation. Explore: larger jiggle and more steps to sample nearby surface states.",
                )
                sugg = _suggest_conditioning_params(
                    atoms_for_sites,
                    mtype=str(mtype),
                    surfactant_class=str(surfactant_class),
                    profile="explore" if auto_profile.startswith("Explore") else "safe",
                )
                cond_jiggle_amp = float(sugg["jiggle_amp"])
                cond_top_z_tol = float(sugg["top_z_tol"])
                cond_fmax = float(sugg["fmax"])
                cond_max_steps = int(sugg["max_steps"])
                st.caption(f"Auto: {sugg['rationale']}")
            else:
                cond_jiggle_amp = st.slider("Jiggle amplitude (Å)", 0.0, 0.20, float(cond_jiggle_amp), 0.01, key="cond_jiggle_amp")
                cond_top_z_tol = st.slider("Top-layer window (Å)", 0.5, 5.0, float(cond_top_z_tol), 0.5, key="cond_top_z_tol")
                cond_fmax = st.number_input("CHGNet relax fmax", min_value=0.01, max_value=0.20, value=float(cond_fmax), step=0.01, key="cond_fmax")
                cond_max_steps = st.number_input("CHGNet max steps", min_value=50, max_value=1000, value=int(cond_max_steps), step=50, key="cond_max_steps")

            cond_seed_ui = st.number_input("Seed (0 = auto)", min_value=0, max_value=2**31-1, value=int(cond_seed_ui), step=1, key="cond_seed")

        cond_seed = None if int(cond_seed_ui) == 0 else int(cond_seed_ui)

    # Build the effective slab used for site detection (conditioned or original)
    atoms_for_sites_eff = atoms_for_sites
    slab_prerelax_meta_ui = None
    if is_her and (mtype == "oxide") and bool(her_constrained_prerelax):
        try:
            atoms_for_sites_eff, slab_prerelax_meta_ui = _get_oxide_her_constrained_prerelaxed_slab(
                atoms_for_sites,
                enable=bool(her_constrained_prerelax),
                top_free_layers=int(her_constrained_top_free_layers),
                layer_tol=float(her_constrained_layer_tol),
                fmax=float(her_constrained_fmax),
                max_steps=int(her_constrained_max_steps),
                seed=her_constrained_seed,
            )
            if slab_prerelax_meta_ui:
                st.caption(
                    f"Constrained CHGNet slab pre-relax applied (free top layers={slab_prerelax_meta_ui.get('free_top_layers')}, "
                    f"fixed atoms={slab_prerelax_meta_ui.get('fixed_atoms')}, fmax={float(her_constrained_fmax):.2f}, steps={int(her_constrained_max_steps)})."
                )
        except Exception as _e:
            st.warning(f"Constrained CHGNet slab pre-relax (preview) skipped due to error: {_e}")
            atoms_for_sites_eff, slab_prerelax_meta_ui = atoms_for_sites, None
    elif (not is_her) and bool(surfactant_prerelax_slab):
        atoms_for_sites_eff, slab_prerelax_meta_ui = _get_conditioned_slab(
            atoms_for_sites,
            is_her=bool(is_her),
            surfactant_class=str(surfactant_class),
            enable=bool(surfactant_prerelax_slab),
            top_z_tol=float(cond_top_z_tol),
            jiggle_amp=float(cond_jiggle_amp),
            fmax=float(cond_fmax),
            max_steps=int(cond_max_steps),
            seed=cond_seed,
        )
        if slab_prerelax_meta_ui:
            st.caption(f"Surface conditioning applied (class={surfactant_class}, jiggle={cond_jiggle_amp:.2f} Å).")
    st.markdown("### A. Site generation mode")
    site_generation_mode = st.radio(
        "Choose how Step 3 should generate candidate sites",
        ["Geometry auto-detection", "ML-assisted screening"],
        index=0,
        horizontal=True,
        key="site_generation_mode",
        help=(
            "Geometry auto-detection generates representative sites directly from the prepared slab. "
            "ML-assisted screening first generates geometry candidates, then ranks candidates with AdsorbML-lite."
        ),
    )

    use_auto_sites = True
    ml_enabled = site_generation_mode == "ML-assisted screening"
    site_selection_method = "ML screening (AdsorbML-lite)" if ml_enabled else "Geometry (representative)"
    rep_site_map = None

    st.markdown("### B. Site selection details")
    if site_generation_mode == "Geometry auto-detection":
        st.caption("Representative geometry-based sites are generated from the prepared slab below.")
    else:
        st.caption("AdsorbML-lite screening uses geometry-generated seeds first, then ranks candidates with ML.")

    if not ml_enabled:
        st.markdown("### Geometry representative sites")
        max_rep = st.slider("Max representative sites per kind", 1, 3, 2, key="max_sites_kind")

        try:
            if mtype == "metal":
                auto_sites = detect_metal_111_sites(atoms_for_sites_eff)
                rep_sites = select_representative_sites(auto_sites, per_kind=int(max_rep))
            else:
                if is_her:
                    rep_sites = _generate_oxide_her_oanchor_sites(atoms_for_sites_eff, max_sites=max(1, int(max_rep) * 2), z_window=2.2, min_xy_sep=1.5)
                    rep_sites = _project_oxide_her_sites_to_otop(atoms_for_sites_eff, rep_sites, dz=1.0, extra_z=0.0)
                elif is_oer:
                    rep_sites = detect_oxide_oer_cation_sites(
                        atoms_for_sites_eff,
                        max_sites=int(max_rep),
                    )
                else:
                    auto_sites = detect_oxide_surface_sites(atoms_for_sites_eff)
                    rep_sites = select_representative_sites(auto_sites, per_kind=int(max_rep))

            rep_site_map = {f"{s.kind}_{i}": s for i, s in enumerate(rep_sites)}

            st.dataframe(
                pd.DataFrame([{"label": k, "kind": v.kind, "x": float(v.position[0]), "y": float(v.position[1])} for k, v in rep_site_map.items()]),
                use_container_width=True,
            )

            if rep_sites:
                st.markdown("#### 3D Preview (Geometry seeds)")
                if is_her:
                    preview_ads_options = ["H*"]
                elif is_oxygen:
                    preview_ads_options = orr_ads if orr_ads else ["OOH*", "O*", "OH*"]
                elif is_voc:
                    preview_ads_options = voc_states if voc_states else ["CH3CHO*", "H*", "CH3CH2O*", "CH3CH2OH*"]
                else:
                    preview_ads_options = co2_ads if co2_ads else ["COOH*", "CO*"]
                preview_ads = st.selectbox("Preview adsorbate", preview_ads_options, index=0, key="preview_ads_geom")

                preview_sites = list(rep_sites)
                if not (is_her or is_oxygen or is_voc):
                    preview_sites = [
                        s for s in rep_sites
                        if co2rr_site_allowed(preview_ads, getattr(s, "kind", "ontop"))
                    ]

                slabs_ads = []
                if not preview_sites:
                    st.warning(
                        f"No compatible geometry site is available for {preview_ads}. "
                        "Bidentate HCOO* requires an explicit bridge/cation-pair site."
                    )
                elif is_her:
                    if mtype == "metal":
                        slabs_ads = generate_slab_ads_series(atoms_for_sites_eff, preview_sites, symbol="H", dz=0.0, mode="default")
                    else:
                        preview_sites = _project_oxide_her_sites_to_otop(atoms_for_sites_eff, preview_sites, dz=1.0, extra_z=0.0)
                        slabs_ads = generate_slab_ads_series(atoms_for_sites_eff, preview_sites, symbol="H", dz=0.0, mode="default")
                    export_ads_label = "H"
                else:
                    export_ads_label = preview_ads.replace("*", "")
                    for s in preview_sites:
                        slabs_ads.append(build_adsorbate_preview_slab(atoms_for_sites_eff, s, preview_ads, dz=1.8, ref_dir="ref_gas"))

                if preview_sites and slabs_ads:
                    site_labels = [f"{getattr(s, 'kind', 'site')}_{i}" for i, s in enumerate(preview_sites)]
                    selected_label = st.selectbox("Select site to view", site_labels, index=0, key="geom_view_site_label")
                    idx = site_labels.index(selected_label)
                    show_atoms_3d(slabs_ads[idx], height=420, width=900, tag=f"geom_seed_{idx}")

                    if st.button("Export preview CIFs (zip)", key="btn_export_previews"):
                        zip_buf = BytesIO()
                        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                            for i, (s, ads_slab) in enumerate(zip(preview_sites, slabs_ads)):
                                zf.writestr(f"{s.kind}_{i}_{export_ads_label}.cif", atoms_to_cif_bytes(ads_slab, symprec=0.1))
                        zip_buf.seek(0)
                        st.download_button(
                            "Download preview_sites.zip",
                            zip_buf,
                            "preview_sites.zip",
                            "application/zip",
                            key="dl_preview_zip",
                        )
        except Exception as e:
            st.error(f"Auto detection failed: {e}")

    if use_auto_sites and ml_enabled:
        st.markdown("### ML screening (simplified)")

        if not HAS_ADSORML:
            st.error(f"ML screening module import failed: {ADSORML_IMPORT_ERR}")
        else:
            preset = st.selectbox("Pre-relax preset", ["Fast", "Normal", "Tight"], index=1, key="ml_preset")
            preset_map = {
                "Fast":   {"fmax": 0.08, "max_steps": 100, "relax_ads_only": True},
                "Normal": {"fmax": 0.05, "max_steps": 150, "relax_ads_only": True},
                "Tight":  {"fmax": 0.03, "max_steps": 300, "relax_ads_only": True},
            }
            p = preset_map[preset]

            col1, col2, col3 = st.columns(3)
            with col1:
                geom_per_kind = st.number_input("Geometry seeds per kind", 1, 5, 2, key="ml_geom_per_kind_simple")
            with col2:
                probe_level = st.selectbox("Probe level (random sites)", ["Low", "Medium", "High"], index=1, key="ml_probe_level")
                probe_map = {"Low": 8, "Medium": 16, "High": 32}
                n_random = probe_map[probe_level]
            with col3:
                top_k = st.number_input("Top-k", 1, 30, 6, key="ml_topk_simple")

            adv_settings = {
                "co2rr_clearance": 1.2,
                "union_max": 10,
                "oxide_anchor_mode": "cation",
                "oxide_anchor_height": 1.8,
                "xy_bin": 0.25,
            }

            with st.expander("Additional ML settings", expanded=False):
                adv_settings["co2rr_clearance"] = float(st.number_input("CO2RR clearance (Å)", 0.8, 6.0, 1.2, step=0.1, key="ml_clr_adv"))
                adv_settings["union_max"] = int(st.number_input("Union max sites (CO2RR)", 1, 30, 10, key="ml_union_adv"))
                adv_settings["xy_bin"] = float(st.number_input("Union xy_bin (Å)", 0.05, 1.0, 0.25, step=0.05, key="ml_xybin_adv"))
                if (mtype == "oxide") and (not is_her):
                    adv_settings["oxide_anchor_mode"] = st.selectbox("Oxide CO2RR anchor mode", ["cation", "anion_o"], index=0, key="ml_ox_anchor_adv")
                    adv_settings["oxide_anchor_height"] = float(st.number_input("O-anchor height (Å)", 1.2, 3.5, 1.8, step=0.1, key="ml_ox_anchor_h_adv"))

            rep_eff = validate_structure(atoms_for_sites, target_area=70.0)
            vac_z = float(getattr(rep_eff, "vacuum_z", 0.0))
            bulk_like = (vac_z < 10.0) and bool(atoms_for_sites.get_pbc()[2])
            allow_bulk_like = False
            if bulk_like:
                st.warning("Prepared structure is BULK-like. Recommended: fix vacuum/slabify in Step 2.")
                allow_bulk_like = st.checkbox("Allow ML screening anyway (not recommended)", value=False, key="ml_allow_bulk_like")

            def _run_ml_screening():
                if bulk_like and (not allow_bulk_like):
                    st.error("ML screening blocked: BULK-like. Fix vacuum/slabify in Step 2 or allow explicitly.")
                    return False

                atoms_for_sites_screen, slab_prerelax_meta = _get_conditioned_slab(
                    atoms_for_sites,
                    is_her=bool(is_her),
                    surfactant_class=str(surfactant_class),
                    enable=bool(surfactant_prerelax_slab),
                    top_z_tol=float(cond_top_z_tol),
                    jiggle_amp=float(cond_jiggle_amp),
                    fmax=float(cond_fmax),
                    max_steps=int(cond_max_steps),
                    seed=cond_seed,
                )

                sig = _atoms_signature(atoms_for_sites_screen)
                active_ads_for_key = (orr_ads if is_oxygen else (voc_states if is_voc else co2_ads))
                key = _make_ml_screen_key(
                    sig,
                    mtype,
                    reaction_mode,
                    active_ads_for_key,
                    preset,
                    int(top_k),
                    int(geom_per_kind),
                    probe_level,
                    adv_settings,
                    str(surfactant_class),
                    bool(surfactant_prerelax_slab),
                )

                if st.session_state.get("ml_screen_key") == key and st.session_state.get("ml_union_site_map") is not None:
                    return True

                if mtype == "oxide" and is_her:
                    cand_sites = _generate_oxide_her_oanchor_sites(
                        atoms_for_sites_screen,
                        max_sites=max(4, int(geom_per_kind) + int(n_random)),
                        z_window=2.2,
                        min_xy_sep=1.2,
                    )
                    cand_sites = _project_oxide_her_sites_to_otop(atoms_for_sites_screen, cand_sites, dz=1.0, extra_z=0.0)
                elif mtype == "oxide" and is_oer:
                    cand_sites = detect_oxide_oer_cation_sites(
                        atoms_for_sites_screen,
                        max_sites=max(1, int(geom_per_kind)),
                    )
                else:
                    cand_sites = generate_candidate_sites(
                        atoms_for_sites_screen,
                        mtype=mtype,
                        geom_per_kind=int(geom_per_kind),
                        n_random=int(n_random),
                        rng_seed=GLOBAL_SEED,
                        random_kind="fcc",
                        reaction_mode=("OER" if is_oer else "CO2RR"),
                    )
                if not cand_sites:
                    st.error("No candidate sites generated.")
                    return False

                settings = ScreeningSettings(
                    relax_ads_only=bool(p["relax_ads_only"]),
                    fmax=float(p["fmax"]),
                    max_steps=int(p["max_steps"]),
                    co2rr_clearance=float(adv_settings["co2rr_clearance"]),
                    oxide_anchor_mode=str(adv_settings["oxide_anchor_mode"]),
                    oxide_anchor_height=float(adv_settings["oxide_anchor_height"]),
                    surfactant_class=str(surfactant_class),
                )

                pbar = st.progress(0)
                status = st.empty()

                def _cb(i, n, msg):
                    pbar.progress(int(100 * i / max(n, 1)))
                    status.write(f"{msg}: {i}/{n}")

                try:
                    if is_her:
                        by_ads, raw_by_ads, stats_by_ads = screen_sites_adsorbml_lite(
                            atoms_for_sites_screen,
                            cand_sites,
                            reaction="HER",
                            mtype=mtype,
                            adsorbates=["H*"],
                            top_k=int(top_k),
                            settings=settings,
                            progress_cb=_cb,
                            ref_dir="ref_gas",
                            return_raw=True,
                        )
                    else:
                        _active_ads = orr_ads if is_oxygen else co2_ads
                        if not _active_ads:
                            _ads_label = "OER" if is_oxygen else "CO2RR"
                            st.error(f"Select at least one {_ads_label} intermediate (sidebar).")
                            return False
                        _screen_reaction = "OER" if is_oer else "CO2RR"
                        by_ads, raw_by_ads, stats_by_ads = screen_sites_adsorbml_lite(
                            atoms_for_sites_screen,
                            cand_sites,
                            reaction=_screen_reaction,
                            mtype=mtype,
                            adsorbates=list(_active_ads),
                            top_k=int(top_k),
                            settings=settings,
                            progress_cb=_cb,
                            ref_dir="ref_gas",
                            return_raw=True,
                        )

                    site_map, struct_map, union_items = union_topk_sites(
                        by_ads,
                        union_max_sites=int(adv_settings["union_max"]) if (not is_her) else int(top_k),
                        xy_bin=float(adv_settings["xy_bin"]),
                    )

                    if mtype == "oxide" and is_her and site_map:
                        site_map = _project_oxide_her_sites_to_otop(atoms_for_sites_screen, site_map, dz=1.0, extra_z=0.0)
                        struct_map = {}

                    union_labels = list(site_map.keys()) if site_map else []
                    compact_df = _build_ml_compact_df(union_items, union_labels)

                    rows = []
                    for ads_k, items in (raw_by_ads or {}).items():
                        for r in items:
                            rows.append({
                                "adsorbate": getattr(r, "adsorbate", ads_k),
                                "kind": getattr(r, "kind", "?"),
                                "label": getattr(r, "label", "?"),
                                "anchor_mode": getattr(r, "anchor_mode", ""),
                                "surfactant_class": str(surfactant_class),
                                "valid": bool(getattr(r, "valid", False)),
                                "reason": getattr(r, "reason", ""),
                                "E_pre_total (eV)": getattr(r, "energy", np.nan),
                                "E_pre_per_atom (eV)": getattr(r, "e_per_atom", np.nan),
                                "dmin_ads-surf (Å)": getattr(r, "dmin", np.nan),
                                "lateral_disp (Å)": getattr(r, "lateral_disp", np.nan),
                                "converged": getattr(r, "converged", True),
                            })
                    debug_df = pd.DataFrame(rows) if rows else pd.DataFrame()

                    st.session_state["ml_screen_key"] = key
                    st.session_state["ml_union_site_map"] = site_map
                    st.session_state["ml_union_struct_map"] = struct_map
                    st.session_state["ml_union_items"] = union_items
                    st.session_state["ml_compact_df"] = compact_df
                    st.session_state["ml_debug_df"] = debug_df
                    st.session_state["ml_debug_stats"] = stats_by_ads
                    return True
                except Exception as e:
                    st.error(f"ML screening failed: {e}")
                    return False
                finally:
                    pbar.empty()
                    status.empty()

            colA, colB = st.columns(2)
            with colA:
                if st.button("Run ML screening", type="primary", key="btn_ml_run"):
                    ok = _run_ml_screening()
                    if ok:
                        st.success("ML screening complete.")
            with colB:
                if st.button("Clear ML cache", key="btn_ml_clear"):
                    _clear_ml_cache()
                    st.info("ML cache cleared.")

            if st.session_state.get("ml_union_site_map") is not None:
                st.markdown("#### ML-selected sites (compact)")
                dfc = st.session_state.get("ml_compact_df")
                if dfc is None or dfc.empty:
                    st.warning("No ML-selected sites produced.")
                else:
                    st.dataframe(dfc, use_container_width=True)

                cX1, cX2 = st.columns(2)
                with cX1:
                    dd = st.session_state.get("ml_debug_df")
                    if dd is not None and (not dd.empty):
                        st.download_button(
                            "Download ML debug CSV",
                            dd.to_csv(index=False).encode("utf-8"),
                            "ml_screening_debug.csv",
                            "text/csv",
                            key="dl_ml_debug_csv",
                        )
                with cX2:
                    struct_map = st.session_state.get("ml_union_struct_map") or {}
                    if struct_map:
                        zip_buf = export_zip_of_struct_map(struct_map, symprec=0.1)
                        st.download_button(
                            "Download ML top-k seeds (zip)",
                            zip_buf,
                            "ml_topk_seeds.zip",
                            "application/zip",
                            key="dl_ml_zip",
                        )

                site_map = st.session_state.get("ml_union_site_map") or {}
                struct_map = st.session_state.get("ml_union_struct_map") or {}
                keys = list(site_map.keys())

                if keys:
                    sel = st.selectbox("Preview ML site", keys, key="ml_preview_key")
                    if sel in struct_map and not (mtype == "oxide" and is_her):
                        show_atoms_3d(struct_map[sel], height=420, width=900, tag=f"ml_{sel}")
                    else:
                        s = site_map[sel]
                        if is_her:
                            if mtype == "oxide":
                                s_use = _project_single_oxide_her_site_to_otop(atoms_for_sites_eff, s, dz=1.0, extra_z=0.0)
                                atoms_prev = generate_slab_ads_series(atoms_for_sites_eff, [s_use], symbol="H", mode="default")[0]
                            else:
                                atoms_prev = generate_slab_ads_series(atoms_for_sites_eff, [s], symbol="H", mode="default")[0]
                        else:
                            if is_oxygen:
                                ads0 = (orr_ads[0] if orr_ads else "OOH*")
                            elif is_voc:
                                ads0 = (voc_states[0] if voc_states else "CH3CHO*")
                            else:
                                ads0 = (co2_ads[0] if co2_ads else "COOH*")
                            atoms_prev = build_adsorbate_preview_slab(atoms_for_sites_eff, s, ads0, dz=1.8, ref_dir="ref_gas")
                        show_atoms_3d(atoms_prev, height=420, width=900, tag=f"ml_fallback_{sel}")

# ---------------- STEP 4: Run calculation ----------------
st.markdown("## 4) Run calculation")

# Always define to avoid NameError during reruns (e.g., clicking history items)
atoms_for_calc = None

working = st.session_state.get("atoms_tuned") or st.session_state.get("atoms_loaded")
if working is None:
    st.info("Load a structure first.")
else:
    _ensure_prepared_uptodate()
    atoms_for_calc = st.session_state.get("atoms_prepared")

    U_input = float(st.session_state.get("U_input", 0.0))
    pH_input = float(st.session_state.get("pH_input", 0.0))

    # HER thermochemistry mode
    thermo_mode = "CHE correction (fast screening)"
    zpe_target_mode = "Best-ranked by CHE"
    zpe_target_label = None
    local_zpe_cutoff = 2.5
    local_zpe_max_neighbors = 3

    # HER relaxation policy defaults.
    # Metal HER path is preserved as-is by the current app UI call path.
    # Oxide descriptor mode uses an internal fixed policy:
    #   D1 = constrained/rigid O-site OH descriptor handling
    #   D2_Hreact = partial metal-cation-centered H* relaxation
    her_relaxation_scope = "partial" if (is_her and mtype == "metal") else "rigid"
    her_n_fix_layers = 2

    if is_her and (mtype == "oxide"):
        st.markdown("### HER thermochemistry / ZPE correction")
        thermo_mode = st.selectbox(
            "HER thermochemistry mode",
            [
                "CHE correction (fast screening)",
                "Local ZPE correction (selected structure)",
                "Local ZPE correction (all structures)",
            ],
            index=0,
            key="her_thermo_mode",
        )
        zpe_target_mode = "Best-ranked by CHE"
        zpe_target_label = None
        local_zpe_cutoff = 2.5
        local_zpe_max_neighbors = 3


    # Oxide HER descriptor mode
    oxide_descriptor_mode = "Full 2-stage profile (recommended)"
    oxide_descriptor_max_reactive_per_kind = 2
    oxide_descriptor_pair_limit = 6
    if is_her and (mtype == "oxide"):
        st.markdown("### Oxide HER descriptor mode")
        oxide_descriptor_mode = st.selectbox(
            "Descriptor mode",
            [
                "D2_Hreact only (reactive H state)",
                "Full 2-stage profile (recommended)",
            ],
            index=1,
            key="oxide_descriptor_mode",
            help=(
                "D2 computes only the reactive-H-state descriptor. "
                "Full 2-stage profile computes D1 and D2 together."
            ),
        )
        needs_reactive = oxide_descriptor_mode in {
            "D2_Hreact only (reactive H state)",
            "Full 2-stage profile (recommended)",
        }
        oxide_descriptor_max_reactive_per_kind = 2
        c3a, c3b = st.columns(2)
        with c3a:
            st.caption("Reactive-H seed count is fixed to the current app default.")
        with c3b:
            oxide_descriptor_pair_limit = st.number_input(
                "Pairing seed limit (disabled)",
                min_value=2, max_value=12, value=6, step=1,
                key="oxide_descriptor_pair_limit",
                disabled=True,
            )
        st.caption("D3 / H₂ pairing proxy is disabled in the current app build. The code skeleton is retained only as commented legacy logic.")

        # Oxide HER descriptor relaxation is intentionally not user-selectable.
        # A single UI-level relaxation control cannot describe both descriptor stages:
        #   D1 = constrained/rigid O-site OH descriptor handling
        #   D2_Hreact = partial metal-cation-centered H* relaxation
        # Metal HER behavior is handled by the metal branch and is not changed here.
        her_relaxation_scope = "rigid"
        her_n_fix_layers = 2
        st.caption(
            "Oxide HER descriptor relaxation policy is fixed internally: "
            "D1 uses constrained/rigid O-site OH handling; "
            "D2_Hreact uses partial metal-cation-centered H* relaxation. "
            "The metal HER workflow is unchanged."
        )

    with st.expander("Advanced electrochemical correction", expanded=False):
        colU, colpH = st.columns(2)
        with colU:
            U_input = st.number_input(
                "Potential U (V)",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.05,
                key="U_input",
            )
        with colpH:
            pH_input = st.number_input(
                "pH",
                min_value=0.0,
                max_value=14.0,
                value=0.0,
                step=0.1,
                key="pH_input",
            )

    if (not is_her):
        st.caption(
            "Note: U/pH correction is applied only for HER. CO₂RR reports reaction-referenced electronic-energy "
            "descriptors (ΔE_ads_user) without ZPE, entropy, solvation, or potential corrections. "
            "OER writes a step-wise oxygen-intermediate summary. VOC mode reports UMA/OCP ΔE_proxy and "
            "co-adsorption proximity proxies, not electrochemical ΔG."
        )


    # H*/HER guardrail was removed from non-HER workflows.
    # CO2RR and VOC runs now execute only their selected descriptor states.

    # Auto-selected sites for calculation (applies to both HER/CO2RR)
    use_auto_sites_for_calc = st.checkbox(
        "Use auto-selected sites for calculation",
        value=True,
        key="use_auto_sites_for_calc",
    )

    final_user_sites = None
    if use_auto_sites_for_calc:
        if st.session_state.get("ml_union_site_map") is not None and not (is_oer and mtype == "oxide"):
            st.info("Auto sites source: ML screening union-sites (from Step 3).")
            final_user_sites = st.session_state.get("ml_union_site_map")
        elif st.session_state.get("ml_union_site_map") is not None and (is_oer and mtype == "oxide"):
            st.info("Auto sites source: OER cation detector. Legacy ML union-sites are ignored for oxide-OER taxonomy consistency.")
            try:
                rep_sites = detect_oxide_oer_cation_sites(atoms_for_calc, max_sites=max(1, int(st.session_state.get("max_sites_kind", 2))))
                final_user_sites = {f"{s.kind}_{i}": s for i, s in enumerate(rep_sites)} if rep_sites else None
            except Exception as e:
                st.error(f"OER cation auto-sites failed: {e}")
                final_user_sites = None
        else:
            st.info("Auto sites source: Geometry representative (from Step 3).")
            try:
                if atoms_for_calc is None:
                    raise ValueError("Prepared structure is not available. Please check Step 1–3.")
                per_kind = int(st.session_state.get("max_sites_kind", 2))
                if mtype == "metal":
                    auto_sites = detect_metal_111_sites(atoms_for_calc)
                    rep_sites = select_representative_sites(auto_sites, per_kind=per_kind)
                    rep_site_map_for_calc = {f"{s.kind}_{i}": s for i, s in enumerate(rep_sites)}
                    final_user_sites = rep_site_map_for_calc
                else:
                    if is_her:
                        rep_sites = _generate_oxide_her_oanchor_sites(
                            atoms_for_calc,
                            max_sites=max(1, int(per_kind) * 2),
                            z_window=2.2,
                            min_xy_sep=1.5,
                        )
                        rep_site_map_for_calc = {f"{s.kind}_{i}": s for i, s in enumerate(rep_sites)}
                        final_user_sites = _project_oxide_her_sites_to_otop(
                            atoms_for_calc,
                            rep_site_map_for_calc,
                            dz=1.0,
                            extra_z=0.0,
                        ) if rep_site_map_for_calc else None
                    elif is_oer:
                        rep_sites = detect_oxide_oer_cation_sites(
                            atoms_for_calc,
                            max_sites=max(1, int(st.session_state.get("max_sites_kind", 2))),
                        )
                        rep_site_map_for_calc = {f"{s.kind}_{i}": s for i, s in enumerate(rep_sites)}
                        final_user_sites = rep_site_map_for_calc
                    else:
                        auto_sites = detect_oxide_surface_sites(atoms_for_calc)
                        rep_sites = select_representative_sites(auto_sites, per_kind=per_kind)
                        rep_site_map_for_calc = {f"{s.kind}_{i}": s for i, s in enumerate(rep_sites)}
                        final_user_sites = rep_site_map_for_calc
            except Exception as e:
                st.error(f"Geometry auto-sites failed: {e}")
                final_user_sites = None

    # Basic structure diagnostics (guard against bulk-like input)
    if atoms_for_calc is not None:
        rep_eff = validate_structure(atoms_for_calc, target_area=70.0)
        vac_z = float(getattr(rep_eff, "vacuum_z", 0.0))
        if (vac_z < 10.0) and bool(atoms_for_calc.get_pbc()[2]):
            st.warning(
                f"Prepared structure is BULK-like (vacuum_z={vac_z:.2f} Å, pbc_z=True). "
                "Adsorption sites may collapse and results may be unreliable."
            )
    else:
        st.warning("No prepared structure available yet. Complete Step 1–3 first.")

    oxide_her_input_audit = None
    if atoms_for_calc is not None and is_her and (mtype == "oxide"):
        oxide_her_input_audit = _oxide_her_pre_run_audit(
            atoms_for_calc,
            n_fix_layers=int(her_n_fix_layers),
        )
        _render_oxide_her_pre_run_audit(oxide_her_input_audit, expanded=False)

    if st.button("Run Calculation", type="primary", key="btn_run_calc"):
        if atoms_for_calc is None:
            st.error("No prepared structure available.")
            st.stop()

        if is_her and (mtype == "oxide"):
            setup_stage = int(st.session_state.get("surface_setup_stage", 0))
            if setup_stage < 5:
                st.error(
                    "Oxide HER D1/D2 calculation requires a reviewed prepared slab. "
                    "Complete Step 2 through vacuum, XY-size review, slab reduction, and the optional surface-engineering review before running. "
                    "This does not force slabify or XY expansion; uploaded Miller-index slabs can be kept at the current XY size after review."
                )
                st.stop()
            oxide_her_input_audit = _oxide_her_pre_run_audit(
                atoms_for_calc,
                n_fix_layers=int(her_n_fix_layers),
            )
            if oxide_her_input_audit.get("hard_errors"):
                st.error(
                    "Prepared oxide slab failed the D1/D2 pre-run audit: "
                    + "; ".join(str(x) for x in oxide_her_input_audit.get("hard_errors", []))
                )
                st.stop()

        seeds.fix_all(GLOBAL_SEED)

        uploads = Path("uploads")
        uploads.mkdir(parents=True, exist_ok=True)
        atoms_for_calc_run, slab_prerelax_meta_calc = atoms_for_calc, None
        try:
            if is_her and (mtype == "oxide") and bool(her_constrained_prerelax):
                atoms_for_calc_run, slab_prerelax_meta_calc = _get_oxide_her_constrained_prerelaxed_slab(
                    atoms_for_calc,
                    enable=bool(her_constrained_prerelax),
                    top_free_layers=int(her_constrained_top_free_layers),
                    layer_tol=float(her_constrained_layer_tol),
                    fmax=float(her_constrained_fmax),
                    max_steps=int(her_constrained_max_steps),
                    seed=her_constrained_seed,
                )
            else:
                atoms_for_calc_run, slab_prerelax_meta_calc = _get_conditioned_slab(
                    atoms_for_calc,
                    is_her=bool(is_her),
                    surfactant_class=str(surfactant_class),
                    enable=bool(surfactant_prerelax_slab),
                    top_z_tol=float(cond_top_z_tol),
                    jiggle_amp=float(cond_jiggle_amp),
                    fmax=float(cond_fmax),
                    max_steps=int(cond_max_steps),
                    seed=cond_seed,
                )
        except Exception as _e:
            st.warning(f"CHGNet slab pre-relax (calc) skipped due to error: {_e}")
            atoms_for_calc_run, slab_prerelax_meta_calc = atoms_for_calc, None

        slab_path = uploads / "slab.cif"
        write(slab_path, atoms_for_calc_run, format="cif")

        oxide_her_calc_audit = None
        co2rr_air_summary = None
        co2rr_air_summary_csv = None
        voc_pathway_summary = None
        voc_pathway_summary_csv = None
        voc_pathway_summary_json = None
        co2rr_pathway_summary = None
        co2rr_pathway_summary_csv = None
        co2rr_pathway_summary_json = None
        co2rr_air_oxygen_csv = None
        co2rr_air_oxygen_meta = None
        if is_her and (mtype == "oxide"):
            oxide_her_calc_audit = _oxide_her_pre_run_audit(
                atoms_for_calc_run,
                n_fix_layers=int(her_n_fix_layers),
            )
            if oxide_her_calc_audit.get("hard_errors"):
                st.error(
                    "Final oxide slab failed the D1/D2 audit after optional pre-relax/conditioning: "
                    + "; ".join(str(x) for x in oxide_her_calc_audit.get("hard_errors", []))
                )
                st.stop()

        if is_her and (mtype == "oxide") and final_user_sites:
            try:
                final_user_sites = _project_oxide_her_sites_to_otop(
                    atoms_for_calc_run,
                    final_user_sites,
                    dz=1.0,
                    extra_z=0.0,
                )
            except Exception as _e:
                st.warning(f"Final oxide HER O-top normalization skipped: {_e}")

        manual_sites = tuple(site_preset)

        with st.spinner("Calculating... (Slab & Adsorbate steps synchronized)"):
            if is_her:
                if mtype == "metal":
                    csv_path, meta = run_metal_che(
                        str(slab_path),
                        sites=manual_sites,
                        relax_mode=relax_mode,
                        user_ads_sites=final_user_sites if final_user_sites else None,
                        use_net_corr=True,
                        thermo_mode=thermo_mode,
                        zpe_target_mode=zpe_target_mode,
                        zpe_target_label=zpe_target_label,
                        local_zpe_cutoff=float(local_zpe_cutoff),
                        local_zpe_max_neighbors=int(local_zpe_max_neighbors),
                        her_relaxation_scope="partial",
                    )
                else:
                    csv_path, meta = run_oxide_che(
                        str(slab_path),
                        sites=manual_sites,
                        relax_mode=relax_mode,
                        user_ads_sites=final_user_sites if final_user_sites else None,
                        use_che_shift=True,
                        thermo_mode=thermo_mode,
                        zpe_target_mode=zpe_target_mode,
                        zpe_target_label=zpe_target_label,
                        local_zpe_cutoff=float(local_zpe_cutoff),
                        local_zpe_max_neighbors=int(local_zpe_max_neighbors),
                        oxide_descriptor_mode=str(oxide_descriptor_mode),
                        oxide_descriptor_max_reactive_per_kind=int(oxide_descriptor_max_reactive_per_kind),
                        oxide_descriptor_pair_limit=int(oxide_descriptor_pair_limit),
                        her_relaxation_scope=str(her_relaxation_scope),
                        her_n_fix_layers=int(her_n_fix_layers),
                    )
            elif is_oxygen:
                # ── OER oxygen-intermediate branch ─────────────────
                _orr_adspecies = tuple(orr_ads) if orr_ads else ("OOH*", "O*", "OH*")
                _orr_U = 0.0
                if is_oer:
                    if mtype == "metal":
                        csv_path, meta = run_metal_oer_che(
                            str(slab_path),
                            sites=manual_sites,
                            relax_mode=relax_mode,
                            user_ads_sites=final_user_sites if final_user_sites else None,
                            adspecies=_orr_adspecies,
                            orr_u=_orr_U,
                            oer_relaxation_mode=str(st.session_state.get("oer_relaxation_mode", "short_relax")),
                        )
                    else:
                        _manual_text = str(st.session_state.get("oer_manual_cation_indices_text", "") or "")
                        _manual_indices = []
                        for _tok in re.split(r"[,\s]+", _manual_text.strip()):
                            if not _tok:
                                continue
                            try:
                                _manual_indices.append(int(_tok))
                            except Exception:
                                pass
                        # Use the existing Geometry representative sites slider as the only
                        # OER site-count control. If Step 3 produced explicit oer_cation
                        # sites, pass their count to the backend; otherwise fall back to
                        # max_sites_kind. This keeps OH/O/OOH triplets on the same selected
                        # cation sites instead of using a separate OER-only cap.
                        _oer_site_cap = len(final_user_sites) if isinstance(final_user_sites, dict) and final_user_sites else int(st.session_state.get("max_sites_kind", 2))
                        _oer_sites_for_detector = tuple(f"oer_cation_{i}" for i in range(max(1, int(_oer_site_cap))))
                        csv_path, meta = run_oxide_oer_che(
                            str(slab_path),
                            sites=_oer_sites_for_detector,
                            relax_mode=relax_mode,
                            user_ads_sites=final_user_sites if final_user_sites else None,
                            adspecies=_orr_adspecies,
                            orr_u=_orr_U,
                            oer_relaxation_mode=str(st.session_state.get("oer_relaxation_mode", "short_relax")),
                            oer_manual_cation_indices=tuple(_manual_indices) if _manual_indices else None,
                        )
            elif is_voc:
                # ── VOC proxy branch ─────────────────────────────────────
                _voc_key = str(st.session_state.get("voc_target", voc_key or "acetaldehyde"))
                _voc_route = str(st.session_state.get("voc_route", get_voc_preset(_voc_key).get("default_route", "reduction")))
                _voc_states = tuple(st.session_state.get("voc_states", voc_states or []))
                if not _voc_states:
                    _voc_states = tuple(get_voc_preset(_voc_key).get("default_states", []))

                if mtype == "metal":
                    csv_path, meta = run_metal_voc_proxy(
                        str(slab_path),
                        sites=manual_sites,
                        relax_mode=relax_mode,
                        user_ads_sites=final_user_sites if final_user_sites else None,
                        target_voc=_voc_key,
                        voc_route=_voc_route,
                        descriptor_states=_voc_states,
                        voc_relaxation_policy="normal_relax",
                        oxide_voc_site_policy=str(st.session_state.get("oxide_voc_site_policy", oxide_voc_site_policy)),
                    )
                else:
                    csv_path, meta = run_oxide_voc_proxy(
                        str(slab_path),
                        sites=manual_sites,
                        relax_mode=relax_mode,
                        user_ads_sites=final_user_sites if final_user_sites else None,
                        target_voc=_voc_key,
                        voc_route=_voc_route,
                        descriptor_states=_voc_states,
                        voc_relaxation_policy="normal_relax",
                        oxide_voc_site_policy=str(st.session_state.get("oxide_voc_site_policy", oxide_voc_site_policy)),
                    )
            else:
                # ── CO2RR branch ───────────────────────────────────────
                if not co2_ads:
                    co2_ads = ["COOH*", "CO*"]
                adspecies = tuple(co2_ads)

                _co2rr_air_on = bool(co2rr_air_enabled)
                _co2rr_air_her_guard = bool(co2rr_include_her or (_co2rr_air_on and co2rr_air_include_her))

                if mtype == "metal":
                    csv_path, meta = run_metal_co2rr_che(
                        str(slab_path),
                        sites=manual_sites,
                        relax_mode=relax_mode,
                        user_ads_sites=final_user_sites if final_user_sites else None,
                        adspecies=adspecies,
                        her_guardrail=_co2rr_air_her_guard,
                    )
                else:
                    csv_path, meta = run_oxide_co2rr_che(
                        str(slab_path),
                        sites=manual_sites,
                        relax_mode=relax_mode,
                        user_ads_sites=final_user_sites if final_user_sites else None,
                        adspecies=adspecies,
                        her_guardrail=_co2rr_air_her_guard,
                    )

                # CO2RR-air add-on: auxiliary oxygen-intermediate run.
                # This deliberately calls the existing OER engine from inside CO2RR mode and
                # stores the output separately; the standalone OER branch above is unchanged.
                if _co2rr_air_on:
                    _oxygen_adspecies = tuple(co2rr_air_oxygen_ads) if co2rr_air_oxygen_ads else ("OOH*", "O*", "OH*")
                    _oxygen_out_root = Path(csv_path).resolve().parent / "co2rr_air_oxygen"
                    try:
                        if mtype == "metal":
                            co2rr_air_oxygen_csv, co2rr_air_oxygen_meta = run_metal_oer_che(
                                str(slab_path),
                                out_root=_oxygen_out_root,
                                sites=manual_sites,
                                relax_mode=relax_mode,
                                user_ads_sites=final_user_sites if final_user_sites else None,
                                adspecies=_oxygen_adspecies,
                                orr_u=0.0,
                                oer_relaxation_mode=str(co2rr_air_oer_relaxation_mode),
                            )
                        else:
                            _air_site_cap = len(final_user_sites) if isinstance(final_user_sites, dict) and final_user_sites else int(st.session_state.get("max_sites_kind", 2))
                            _air_oer_sites = tuple(f"co2rr_air_oer_cation_{i}" for i in range(max(1, int(_air_site_cap))))
                            co2rr_air_oxygen_csv, co2rr_air_oxygen_meta = run_oxide_oer_che(
                                str(slab_path),
                                out_root=_oxygen_out_root,
                                sites=_air_oer_sites,
                                relax_mode=relax_mode,
                                user_ads_sites=final_user_sites if final_user_sites else None,
                                adspecies=_oxygen_adspecies,
                                orr_u=0.0,
                                oer_relaxation_mode=str(co2rr_air_oer_relaxation_mode),
                            )
                    except Exception as _e:
                        co2rr_air_oxygen_csv = None
                        co2rr_air_oxygen_meta = {"error": str(_e), "mode": "CO2RR_AIR_OXYGEN_AUX"}
                        st.warning(f"CO₂RR-air auxiliary oxygen run failed: {_e}")

        df, _result_csv_diag = _read_result_csv_safely(
            csv_path,
            context="primary calculation result",
        )
        if df is None:
            _render_empty_result_diagnostic(_result_csv_diag, meta=meta)
            st.stop()
        if isinstance(df, pd.DataFrame) and df.empty:
            _render_empty_result_diagnostic(_result_csv_diag, meta=meta)
            st.stop()

        st.success("Calculation Complete!")
        if (
            isinstance(meta, dict)
            and int(meta.get("CO2RR_PLACEMENT_FAILURE_COUNT", 0) or 0) > 0
        ):
            _placement_failure_count = int(
                meta.get("CO2RR_PLACEMENT_FAILURE_COUNT", 0) or 0
            )
            st.warning(
                f"Skipped {_placement_failure_count} geometrically inaccessible "
                "CO₂RR adsorbate/site seed(s). The remaining sites, binding "
                "variants, and product branches were calculated normally."
            )

        # annotate surfactant scenario into the result table
        if isinstance(df, pd.DataFrame) and (not df.empty):
            df["surfactant_class"] = str(surfactant_class)
            df["surfactant_chgnet_prerelax_slab"] = bool(surfactant_prerelax_slab)
            if is_her and (mtype == "oxide") and isinstance(oxide_her_calc_audit, dict):
                df = _append_oxide_her_audit_columns(df, oxide_her_calc_audit)

        mode_label = "HER" if is_her else ("OER" if is_oer else ("VOC" if is_voc else "CO2RR"))

        if is_her and "ΔG_H (eV)" in df.columns:
            df["ΔG_H(U,pH) (eV)"] = df["ΔG_H (eV)"] - float(U_input) - R_PH * float(pH_input)


        if is_her:
            # HER: annotate site transitions for UI/debugging.
            df = annotate_site_transitions(df, disp_thresh=CO2RR_MIGRATION_DISP_THRESH_A)

            if (
                str(mtype).strip().lower() == "oxide"
                and isinstance(meta, dict)
                and bool(meta.get("OXIDE_DESCRIPTOR_PRIMARY_RESULTS", False))
            ):
                # Oxide D2-primary mode: D2 is allowed to relax within the
                # metal-cation basin. Do not demote a valid D2 row solely due
                # to the legacy H_lateral_disp cutoff.
                df_rel, df_unrel = _split_oxide_d2_primary_reliability(df)
            else:
                # Legacy HER reliability split for metals and non-D2 HER modes.
                df_rel, df_unrel = split_reliable_unreliable(df)

            df["reliability"] = "unreliable"
            if df_rel is not None:
                df.loc[df_rel.index, "reliability"] = "reliable"
            migration_summary = summarize_site_transitions(df)
        else:
            # CO2RR / OER: QA-driven policy.
            # Oxygen-intermediate modes use stricter binding/channel validity:
            # plain migrated oxygen rows are not accepted unless they remain
            # surface-bound and valid_for_oer_summary=True.
            if is_oxygen:
                df = oxygen_apply_qa_policy(df, disp_thresh=CO2RR_MIGRATION_DISP_THRESH_A)
            elif is_voc:
                df = voc_apply_qa_policy(df, disp_thresh=1.20)
            else:
                df = co2rr_apply_qa_policy(df, disp_thresh=CO2RR_MIGRATION_DISP_THRESH_A)
            df = annotate_site_transitions(df, disp_thresh=CO2RR_MIGRATION_DISP_THRESH_A)
            df_diag = pd.DataFrame()
            if is_oxygen:
                df_keep, df_reject = oxygen_split_by_qa(df)
            elif is_voc:
                df_keep, df_diag, df_reject = voc_split_candidates_diagnostics_rejected(df)
            else:
                df_keep, df_reject = co2rr_split_by_qa(df)

            # Set reliability consistent with QA policy.  VOC/ECH diagnostic
            # rows are neither ordinary candidates nor rejected failures.
            df["reliability"] = "unreliable"
            if isinstance(df_keep, pd.DataFrame) and (not df_keep.empty):
                df.loc[df_keep.index, "reliability"] = "reliable"
            if bool(is_voc) and isinstance(df_diag, pd.DataFrame) and (not df_diag.empty):
                df.loc[df_diag.index, "reliability"] = "diagnostic_valid"

            # Backwards-compatible names for downstream UI blocks
            df_rel, df_unrel = df_keep, df_reject
            migration_summary = summarize_site_transitions(df)

            if is_voc:
                try:
                    _summary_voc_key = str(st.session_state.get("voc_target", "acetaldehyde"))
                    _summary_route_key = str(st.session_state.get("voc_route", get_voc_preset(_summary_voc_key).get("default_route", "reduction")))
                    voc_pathway_summary = build_voc_pathway_summary(
                        df,
                        voc_key=_summary_voc_key,
                        route_key=_summary_route_key,
                    )
                    voc_pathway_summary_csv, voc_pathway_summary_json = write_voc_pathway_summary(
                        voc_pathway_summary,
                        Path(csv_path).resolve().parent,
                    )
                except Exception as _e:
                    voc_pathway_summary = None
                    voc_pathway_summary_csv = None
                    voc_pathway_summary_json = None
                    st.warning(f"VOC pathway summary could not be generated: {_e}")

            if (not is_oxygen) and (not is_voc):
                try:
                    _summary_co2rr_key = str(st.session_state.get("co2rr_pathway", "competitive_c1"))
                    _summary_co2rr_states = list(st.session_state.get("co2rr_ads", co2_ads or []))
                    co2rr_pathway_summary = build_co2rr_pathway_summary(
                        df,
                        pathway_key=_summary_co2rr_key,
                        states=_summary_co2rr_states,
                        product_state_energies=(
                            meta.get("CO2RR_PRODUCT_STATE_ENERGIES", {})
                            if isinstance(meta, dict)
                            else {}
                        ),
                        potential_V=float(st.session_state.get("co2rr_potential_V", 0.0)),
                    )
                    co2rr_pathway_summary_csv, co2rr_pathway_summary_json = write_co2rr_pathway_summary(
                        co2rr_pathway_summary,
                        Path(csv_path).resolve().parent,
                    )
                except Exception as _e:
                    co2rr_pathway_summary = None
                    co2rr_pathway_summary_csv = None
                    co2rr_pathway_summary_json = None
                    st.warning(f"CO₂RR state-energy summary could not be generated: {_e}")

            # CO2RR-air summary is a CO2RR-only post-processing add-on.
            # It consumes the already-produced CO2RR rows, an optional auxiliary
            # oxygen-intermediate CSV, and optional CO2RR HER guardrail metadata.
            if (not is_oxygen) and (not is_voc) and bool(co2rr_air_enabled):
                _oxygen_df = None
                if co2rr_air_oxygen_csv:
                    try:
                        _oxygen_df = pd.read_csv(co2rr_air_oxygen_csv)
                        _oxygen_df = oxygen_apply_qa_policy(_oxygen_df, disp_thresh=CO2RR_MIGRATION_DISP_THRESH_A)
                        _oxygen_keep, _oxygen_reject = oxygen_split_by_qa(_oxygen_df)
                    except Exception as _e:
                        _oxygen_df = None
                        if isinstance(meta, dict):
                            meta = dict(meta)
                            meta["CO2RR_AIR_OXYGEN_POSTPROCESS_ERROR"] = str(_e)
                _her_guard = meta.get("HER_GUARDRAIL") if isinstance(meta, dict) else None
                co2rr_air_summary = build_co2rr_air_summary(
                    df,
                    oxygen_df=_oxygen_df,
                    her_guard=_her_guard,
                )
                df = annotate_co2rr_air_summary(df, co2rr_air_summary)
                df_rel = annotate_co2rr_air_summary(df_rel, co2rr_air_summary)
                df_unrel = annotate_co2rr_air_summary(df_unrel, co2rr_air_summary)
                try:
                    co2rr_air_summary_csv = Path(csv_path).resolve().parent / "results_co2rr_air_summary.csv"
                    co2rr_air_summary_to_frame(co2rr_air_summary).to_csv(co2rr_air_summary_csv, index=False, float_format="%.6f")
                except Exception:
                    co2rr_air_summary_csv = None

        try:
            if isinstance(df, pd.DataFrame):
                df.to_csv(csv_path, index=False)
        except Exception as _e:
            st.warning(f"Could not write annotated result CSV: {_e}")

        # Persist results for rendering even after rerun (e.g., toggling UI options)
        if isinstance(meta, dict):
            meta = dict(meta)
            meta["SURFACTANT_CLASS"] = str(surfactant_class)
            meta["SURFACTANT_CHGNET_PRERELAX_SLAB"] = bool(surfactant_prerelax_slab)
            if is_her and (mtype == "oxide") and isinstance(oxide_her_calc_audit, dict):
                meta["OXIDE_HER_PRE_RUN_AUDIT"] = oxide_her_calc_audit
            if migration_summary is not None:
                meta["MIGRATION_SUMMARY"] = migration_summary
            if voc_pathway_summary is not None:
                meta["VOC_PATHWAY_SUMMARY"] = dict(voc_pathway_summary)
                meta["VOC_PATHWAY_SUMMARY_CSV"] = str(voc_pathway_summary_csv) if voc_pathway_summary_csv is not None else None
                meta["VOC_PATHWAY_SUMMARY_JSON"] = str(voc_pathway_summary_json) if voc_pathway_summary_json is not None else None
            if co2rr_pathway_summary is not None:
                meta["CO2RR_PATHWAY_SUMMARY"] = dict(co2rr_pathway_summary)
                meta["CO2RR_PATHWAY_SUMMARY_CSV"] = str(co2rr_pathway_summary_csv) if co2rr_pathway_summary_csv is not None else None
                meta["CO2RR_PATHWAY_SUMMARY_JSON"] = str(co2rr_pathway_summary_json) if co2rr_pathway_summary_json is not None else None
                meta["CO2RR_PATHWAY_KEY"] = str(co2rr_pathway_key)
            if co2rr_air_summary is not None:
                meta["CO2RR_AIR_SUMMARY"] = dict(co2rr_air_summary)
                meta["CO2RR_AIR_SUMMARY_CSV"] = str(co2rr_air_summary_csv) if co2rr_air_summary_csv is not None else None
                meta["CO2RR_AIR_OXYGEN_CSV"] = str(co2rr_air_oxygen_csv) if co2rr_air_oxygen_csv is not None else None
                meta["CO2RR_AIR_OXYGEN_META"] = co2rr_air_oxygen_meta

        st.session_state["last_run"] = {
            "is_her": bool(is_her),
            "is_oer": bool(is_oer),
            "is_voc": bool(is_voc),
            "is_orr": False,
            "is_oxygen": bool(is_oxygen),
            "is_co2rr_air": bool(co2rr_air_enabled) if ((not is_her) and (not is_oer) and (not is_voc)) else False,
            "mtype": str(mtype),
            "reaction_mode": str(reaction_mode),
            "mode_label": str(mode_label),
            "csv_path": str(csv_path),
            "meta": meta,
            "df": df,
            "df_rel": df_rel,
            "df_unrel": df_unrel,
            "df_diag": df_diag if bool(is_voc) else pd.DataFrame(),
            "voc_pathway_summary": voc_pathway_summary if bool(is_voc) else None,
            "co2rr_pathway_summary": co2rr_pathway_summary if ((not is_her) and (not is_oer) and (not is_voc)) else None,
            "co2rr_pathway_key": str(co2rr_pathway_key),
            "U_input": float(U_input),
            "pH_input": float(pH_input),
        }

        # Add to session-only run history (max 10; cleared on refresh/close)
        try:
            model_name = ""
            device_name = ""
            if isinstance(meta, dict):
                model_name = str(meta.get("MODEL", meta.get("model", "")) or "")
                device_name = str(meta.get("DEVICE", meta.get("device", "")) or "")

            label = f"{atoms_for_calc.get_chemical_formula()} (n={len(atoms_for_calc)})"
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            csv_name = f"{mode_label}_{mtype}_{relax_mode}.csv".replace(" ", "_")

            # Preferred API (from the run_history.py module we drafted)
            if hasattr(rh, "make_history_record_from_last_run") and hasattr(rh, "add_history_record"):
                hr = rh.make_history_record_from_last_run(
                    run_id=uuid.uuid4().hex[:10],
                    last_run=st.session_state.get("last_run") or {},
                    label=label,
                    relax_mode=str(relax_mode),
                    model=model_name,
                    device=device_name,
                    df=df,
                    csv_bytes=csv_bytes,
                    csv_name=csv_name,
                    prepared_cif_bytes=None,
                    prepared_cif_name=None,
                )
                rh.add_history_record(hr, max_items=10)

            # Backward/alternate API (if you decide to keep dict-based records)
            elif hasattr(rh, "add_record"):
                rec = {
                    "run_id": uuid.uuid4().hex[:10],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "label": label,
                    "reaction_mode": str(reaction_mode),
                    "mtype": str(mtype),
                    "relax_mode": str(relax_mode),
                    "model": model_name,
                    "device": device_name,
                    "note": "",
                    "pinned": False,
                    "csv_name": csv_name,
                    "csv_bytes": csv_bytes,
                    "meta": meta,
                }
                rh.add_record(rec, max_items=10, select=True)
        except Exception:
            pass
            
            


# ---------------- Results (persistent, from last run) ----------------
last_run = st.session_state.get("last_run", None)
if last_run is not None:
    df = last_run.get("df")
    df_rel = last_run.get("df_rel")
    df_unrel = last_run.get("df_unrel")
    df_diag = last_run.get("df_diag") if bool(last_run.get("is_voc")) else pd.DataFrame()
    meta = last_run.get("meta") or {}
    voc_pathway_summary = last_run.get("voc_pathway_summary") or (meta.get("VOC_PATHWAY_SUMMARY") if isinstance(meta, dict) else None)
    co2rr_pathway_summary = last_run.get("co2rr_pathway_summary") or (meta.get("CO2RR_PATHWAY_SUMMARY") if isinstance(meta, dict) else None)
    mode_label = last_run.get("mode_label", "HER" if last_run.get("is_her") else ("VOC" if last_run.get("is_voc") else "CO2RR"))
    U_disp = float(last_run.get("U_input", 0.0))
    pH_disp = float(last_run.get("pH_input", 0.0))

    # --- Run history notes (session-only) ---
    try:
        with st.expander("Run history (selected)", expanded=False):
            rh.render_selected_run_details()
    except Exception:
        pass
    # --- Lightweight warnings (do not gate rendering) ---
    if bool(last_run.get("is_her")) and isinstance(df, pd.DataFrame) and ("is_duplicate" in df.columns):
        try:
            n_dups = int(pd.to_numeric(df["is_duplicate"], errors="coerce").fillna(0).astype(int).sum())
            if n_dups > 0:
                st.warning(f"{n_dups} sites converged to duplicates. Check 'is_duplicate'.")
        except Exception:
            pass

    if (not bool(last_run.get("is_her"))) and isinstance(df, pd.DataFrame) and ("ΔE_ads_user (eV)" in df.columns):
        try:
            n_blow = int((pd.to_numeric(df["ΔE_ads_user (eV)"], errors="coerce").abs() > 50.0).sum())
            if n_blow > 0:
                st.warning(f"{n_blow} {mode_label} points show |ΔE_ads_user| > 50 eV (likely bad placement/unstable relax).")
        except Exception:
            pass

    # --- Main results tables (ALWAYS) ---
    if bool(last_run.get("is_her")):
        if str(last_run.get("mtype", "")) == "oxide" and isinstance(meta, dict) and bool(meta.get("OXIDE_DESCRIPTOR_PRIMARY_RESULTS", False)):
            st.markdown("### Results (Reliable) — Oxide D2 primary")
            st.caption("This table shows the selected metal-cation-centered D2 H* representative. Legacy O-anchor / anion-ontop HER rows are not displayed or written as primary results in this mode.")
        else:
            st.markdown("### Results (Reliable)")
        if isinstance(df_rel, pd.DataFrame):
            st.dataframe(build_compact_table(df_rel, mode_label), use_container_width=True)
        if isinstance(df_unrel, pd.DataFrame) and (not df_unrel.empty):
            with st.expander("Show Unreliable / Unstable Sites", expanded=False):
                st.dataframe(build_compact_table(df_unrel, mode_label), use_container_width=True)
        if isinstance(meta, dict) and meta.get("MIGRATION_SUMMARY"):
            mig_summary = meta.get("MIGRATION_SUMMARY") or {}
            with st.expander("Migration metadata", expanded=False):
                st.write(f"- migrated rows: **{int(mig_summary.get('n_migrated', 0))}**")
                mig_paths = mig_summary.get("paths") or []
                if mig_paths:
                    st.dataframe(pd.DataFrame(mig_paths), use_container_width=True)
                elif isinstance(df, pd.DataFrame) and ("migration_path" in df.columns):
                    cols = [c for c in ["site_label", "requested_site", "initial_geom_site", "relaxed_site", "placement_mismatch", "migrated_actual", "migration_destination", "migration_path", "actual_migration_path", "site_transition_type", "ΔG_H(U,pH) (eV)", "ΔG_H (eV)", "ΔE_H_user (eV)", "H_lateral_disp(Å)", "migrated"] if c in df.columns]
                    st.dataframe(df[cols], use_container_width=True)
    else:
        df_keep = df_rel if isinstance(df_rel, pd.DataFrame) else pd.DataFrame()
        df_reject = df_unrel if isinstance(df_unrel, pd.DataFrame) else pd.DataFrame()

        if bool(last_run.get("is_oer")):
            df_dedup = _oer_site_adsorbate_compact(df_keep) if isinstance(df_keep, pd.DataFrame) else pd.DataFrame()
        elif bool(last_run.get("is_voc")):
            try:
                _render_voc_key = str((meta or {}).get("target_voc", "acetaldehyde"))
                _render_route_key = str((meta or {}).get("voc_route", "reduction"))
                df_dedup = select_voc_state_minima(
                    df_keep,
                    voc_key=_render_voc_key,
                    route_key=_render_route_key,
                )
            except Exception:
                df_dedup = pd.DataFrame()
        else:
            df_dedup = co2rr_dedupe_candidates(df_keep) if isinstance(df_keep, pd.DataFrame) else pd.DataFrame()

        qa_counts = df["qa"].value_counts(dropna=False) if (isinstance(df, pd.DataFrame) and ("qa" in df.columns)) else pd.Series(dtype=int)

        if bool(last_run.get("is_voc")):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total runs", int(len(df)) if isinstance(df, pd.DataFrame) else 0)
            c2.metric("Candidates", int(len(df_keep)) if isinstance(df_keep, pd.DataFrame) else 0)
            c3.metric("Diagnostic", int(len(df_diag)) if isinstance(df_diag, pd.DataFrame) else 0)
            c4.metric("Rejected", int(len(df_reject)) if isinstance(df_reject, pd.DataFrame) else 0)
            c5.metric("State minima", int(len(df_dedup)) if isinstance(df_dedup, pd.DataFrame) else 0)
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total runs", int(len(df)) if isinstance(df, pd.DataFrame) else 0)
            c2.metric("Candidates (QA-valid)", int(len(df_keep)) if isinstance(df_keep, pd.DataFrame) else 0)
            c3.metric("Rejected (QA-invalid)", int(len(df_reject)) if isinstance(df_reject, pd.DataFrame) else 0)
            c4.metric("Compact rows" if bool(last_run.get("is_oer")) else "Unique minima (dedup)", int(len(df_dedup)) if isinstance(df_dedup, pd.DataFrame) else 0)

        if bool(last_run.get("is_voc")) and isinstance(voc_pathway_summary, dict):
            st.markdown("### VOC state energy map")
            st.caption(str(voc_pathway_summary.get("warning", "")))
            render_voc_pathway(voc_pathway_summary)
            with st.expander("State-energy details", expanded=False):
                render_pathway_support_table(voc_pathway_summary)

            _pathway_frame = pathway_summary_to_frame(voc_pathway_summary)
            _dlp1, _dlp2 = st.columns(2)
            with _dlp1:
                st.download_button(
                    "Download VOC state-energy map CSV",
                    _pathway_frame.to_csv(index=False).encode("utf-8"),
                    "voc_state_energy_map.csv",
                    "text/csv",
                    key="dl_voc_state_energy_map_csv",
                )
            with _dlp2:
                st.download_button(
                    "Download VOC state-energy map JSON",
                    json.dumps(voc_pathway_summary, ensure_ascii=False, indent=2).encode("utf-8"),
                    "voc_state_energy_map.json",
                    "application/json",
                    key="dl_voc_state_energy_map_json",
                )

        if (not bool(last_run.get("is_oer"))) and (not bool(last_run.get("is_voc"))) and isinstance(co2rr_pathway_summary, dict):
            st.markdown("### CO₂RR thermodynamic product favorability")
            st.caption(str(co2rr_pathway_summary.get("warning", "")))
            st.caption(str(co2rr_pathway_summary.get("ranking_basis", "")))
            _co2rr_product_df = co2rr_product_summary_to_frame(co2rr_pathway_summary)
            _co2rr_edge_df = co2rr_edge_summary_to_frame(co2rr_pathway_summary)
            _co2rr_map_df = co2rr_pathway_summary_to_frame(co2rr_pathway_summary)
            if isinstance(_co2rr_product_df, pd.DataFrame) and not _co2rr_product_df.empty:
                _priority_cols = [
                    "rank", "product", "site_key", "site_consistency",
                    "endpoint_complete", "path_complete", "ranking_eligible",
                    "overall_PDS", "overall_PDS_delta_G_0_eV",
                    "post_CO_PDS", "post_CO_PDS_delta_G_0_eV",
                    "post_CO_adsorbed_core_PDS",
                    "post_CO_adsorbed_core_PDS_delta_G_0_eV",
                    "branch_start_state", "branch_screening_scope",
                    "branch_screening_bottleneck",
                    "branch_screening_bottleneck_eV",
                    "branch_screening_bottleneck_type",
                    "branch_screening_PDS", "branch_screening_PDS_delta_G_0_eV",
                    "branch_PDS", "branch_PDS_delta_G_0_eV",
                    "branch_adsorbed_core_PDS",
                    "branch_adsorbed_core_PDS_delta_G_0_eV",
                    "missing_states", "confidence",
                ]
                _ordered_cols = [
                    col for col in _priority_cols if col in _co2rr_product_df.columns
                ] + [
                    col for col in _co2rr_product_df.columns if col not in _priority_cols
                ]
                st.dataframe(
                    _co2rr_product_df[_ordered_cols], use_container_width=True
                )
                _methanol_rows = (
                    _co2rr_product_df.loc[
                        _co2rr_product_df["product_key"].astype(str).eq("methanol")
                    ]
                    if "product_key" in _co2rr_product_df.columns
                    else pd.DataFrame()
                )
                if (
                    not _methanol_rows.empty
                    and "endpoint_complete" in _methanol_rows.columns
                    and not bool(_methanol_rows["endpoint_complete"].iloc[0])
                ):
                    st.warning(
                        "CH3OH endpoint is incomplete: add CH3OH to "
                        "ref_gas/thermo_CO2RR.json or provide ref_gas/CH3OH_box.cif. "
                        "Until then, only the site-consistent adsorbed branch up to CH2OH* "
                        "is reported; methanol is not included in the final product ranking."
                    )
            else:
                st.info(
                    "No site-consistent product pathway is available. Check missing intermediates, "
                    "per-site coverage, and explicit product gas references in the state/edge tables."
                )
            with st.expander("CO₂RR elementary edge ΔG and PDS support", expanded=False):
                if isinstance(_co2rr_edge_df, pd.DataFrame):
                    st.dataframe(_co2rr_edge_df, use_container_width=True)
            with st.expander("CO₂RR cumulative state energies", expanded=False):
                if isinstance(_co2rr_map_df, pd.DataFrame):
                    st.dataframe(_co2rr_map_df, use_container_width=True)
            _cmap1, _cmap2, _cmap3, _cmap4 = st.columns(4)
            with _cmap1:
                st.download_button(
                    "Product table CSV",
                    _co2rr_product_df.to_csv(index=False).encode("utf-8"),
                    "co2rr_product_favorability.csv",
                    "text/csv",
                    key="dl_co2rr_product_favorability_csv",
                )
            with _cmap2:
                st.download_button(
                    "Edge ΔG CSV",
                    _co2rr_edge_df.to_csv(index=False).encode("utf-8"),
                    "co2rr_edge_free_energies.csv",
                    "text/csv",
                    key="dl_co2rr_edge_free_energies_csv",
                )
            with _cmap3:
                st.download_button(
                    "State map CSV",
                    _co2rr_map_df.to_csv(index=False).encode("utf-8"),
                    "co2rr_state_energy_map.csv",
                    "text/csv",
                    key="dl_co2rr_state_energy_map_csv",
                )
            with _cmap4:
                st.download_button(
                    "Network JSON",
                    json.dumps(co2rr_pathway_summary, ensure_ascii=False, indent=2).encode("utf-8"),
                    "co2rr_reaction_network.json",
                    "application/json",
                    key="dl_co2rr_reaction_network_json",
                )

        if bool(last_run.get("is_oer")):
            st.markdown("### OER site-level candidate rows")
            st.caption("One lowest-energy height is retained per selected oer_cation site and intermediate. Use the OER step summary for η_OER interpretation.")
        else:
            st.markdown("### VOC state minima" if bool(last_run.get("is_voc")) else ("### CO₂RR relaxed minima" if not bool(last_run.get("is_oer")) else "### Candidates"))
        if isinstance(df_dedup, pd.DataFrame):
            st.dataframe(build_compact_table(df_dedup, mode_label), use_container_width=True)

        with st.expander("Show all candidate attempts (including duplicates)", expanded=False):
            if isinstance(df_keep, pd.DataFrame):
                st.dataframe(build_compact_table(df_keep, mode_label), use_container_width=True)

        if bool(last_run.get("is_voc")) and isinstance(df_diag, pd.DataFrame) and (not df_diag.empty):
            with st.expander("Show ECH diagnostic attempts", expanded=False):
                st.caption("These rows are descriptor diagnostics, not ordinary ranking candidates and not QA-invalid rejected rows.")
                st.dataframe(build_compact_table(df_diag, mode_label), use_container_width=True)

        if isinstance(df_reject, pd.DataFrame) and (not df_reject.empty):
            with st.expander("Show rejected attempts (qa-based)", expanded=False):
                st.dataframe(build_compact_table(df_reject, mode_label), use_container_width=True)

        if qa_counts is not None and (not qa_counts.empty):
            with st.expander("QA breakdown", expanded=False):
                st.dataframe(qa_counts.rename_axis("qa").reset_index(name="count"), use_container_width=True)

        if isinstance(meta, dict) and meta.get("CO2RR_AIR_SUMMARY"):
            st.markdown("### CO₂RR-air competition summary")
            _air_sum = co2rr_air_summary_to_frame(meta.get("CO2RR_AIR_SUMMARY"))
            _air_cols = [
                c for c in [
                    "co2rr_pathway_preference",
                    "her_competition_risk",
                    "orr_competition_risk",
                    "co_poisoning_risk",
                    "air_tolerance_index",
                    "ΔG_COOH_best (eV)",
                    "ΔG_OCHO_best (eV)",
                    "ΔG_CO_best (eV)",
                    "ΔG_H_guardrail (eV)",
                    "ΔG_OOH_best (eV)",
                    "ΔG_O_best (eV)",
                    "ΔG_OH_best (eV)",
                ] if c in _air_sum.columns
            ]
            if _air_cols:
                st.dataframe(_air_sum[_air_cols], use_container_width=True)
            else:
                st.dataframe(_air_sum, use_container_width=True)
            st.caption(
                "This is a screening-only CO₂RR add-on for air-fed/dilute-CO₂ conditions. "
                "It does not replace kinetic ORR modeling."
            )
            _air_csv = meta.get("CO2RR_AIR_SUMMARY_CSV")
            if _air_csv:
                try:
                    _air_path = Path(str(_air_csv))
                    if _air_path.is_file():
                        st.download_button(
                            "Download CO₂RR-air summary CSV",
                            _air_path.read_bytes(),
                            _air_path.name,
                            "text/csv",
                            key="dl_co2rr_air_summary",
                        )
                except Exception:
                    pass

        # OER oxygen-intermediate summary, if produced by CHE_mode.
        try:
            _csvp = Path(str(last_run.get("csv_path", "")))
            _summary_path = _csvp.parent / "results_oer_competition_summary.csv"
            if not _summary_path.is_file():
                _summary_path = None
            if _summary_path is not None:
                st.markdown("### OER step summary")
                _sumdf = pd.read_csv(_summary_path)
                if "results_oer_competition_summary" in str(_summary_path):
                    st.caption("OER mode uses standard OER AEM H2O/H2 references. Recommended η uses explicit OOH only when the OOH−OH scaling sanity check passes; otherwise it uses the ΔG*OOH = ΔG*OH + 3.20 eV scaling proxy.")
                    _summary_cols = [
                        c for c in [
                            "OER_site_rank_by_recommended_eta", "OER_representative_site",
                            "oer_base_site_label", "site_label", "site",
                            "eta_OER_recommended (V)", "eta_OER (V)",
                            "eta_OER_explicit (V)", "eta_OER_scaling_proxy (V)",
                            "OER_summary_source", "explicit_OOH_confidence",
                            "benchmark_consistency_label",
                            "ΔG_OOH_minus_ΔG_OH (eV)",
                            "OOH_OH_scaling_deviation_from_3p20 (eV)",
                            "OER_PDS", "OER_recommendation_basis",
                        ] if c in _sumdf.columns
                    ]
                    if _summary_cols:
                        st.dataframe(_sumdf[_summary_cols], use_container_width=True)
                        with st.expander("Show full OER summary table", expanded=False):
                            st.dataframe(_sumdf, use_container_width=True)
                    else:
                        st.dataframe(_sumdf, use_container_width=True)

                st.download_button(
                    "Download OER summary CSV",
                    _sumdf.to_csv(index=False).encode("utf-8"),
                    _summary_path.name,
                    "text/csv",
                    key=f"dl_oxygen_summary_{_summary_path.name}",
                )
        except Exception:
            pass

        if isinstance(meta, dict) and meta.get("MIGRATION_SUMMARY"):
            mig_summary = meta.get("MIGRATION_SUMMARY") or {}
            with st.expander("Migration metadata", expanded=False):
                st.write(f"- migrated rows: **{int(mig_summary.get('n_migrated', 0))}**")
                mig_paths = mig_summary.get("paths") or []
                if mig_paths:
                    st.dataframe(pd.DataFrame(mig_paths), use_container_width=True)
                elif isinstance(df, pd.DataFrame) and ("migration_path" in df.columns):
                    cols = [c for c in ["adsorbate", "site_label", "requested_site", "initial_geom_site", "relaxed_site", "placement_mismatch", "migrated_actual", "migration_destination", "migration_path", "actual_migration_path", "site_transition_type", "ΔE_ads_user (eV)", "ΔG_ads (eV)", "ads_lateral_disp(Å)", "qa", "migrated"] if c in df.columns]
                    st.dataframe(df[cols], use_container_width=True)

        cdl1, cdl2 = st.columns(2)
        with cdl1:
            st.download_button(
                "Download OER site-level candidates CSV" if bool(last_run.get("is_oer")) else "Download candidates (dedup) CSV",
                df_dedup.to_csv(index=False).encode("utf-8") if isinstance(df_dedup, pd.DataFrame) else b"",
                (f"{str(mode_label).lower()}_site_level_candidates.csv" if bool(last_run.get("is_oer")) else f"{str(mode_label).lower()}_candidates_dedup.csv"),
                "text/csv",
                key="dl_co2rr_candidates_dedup",
            )
        with cdl2:
            st.download_button(
                "Download candidates (all) CSV",
                df_keep.to_csv(index=False).encode("utf-8") if isinstance(df_keep, pd.DataFrame) else b"",
                f"{str(mode_label).lower()}_candidates_all.csv",
                "text/csv",
                key="dl_co2rr_candidates_all",
            )

    # --- Oxide HER representative sites + selected descriptor profile ---
    if bool(last_run.get("is_her")) and str(last_run.get("mtype", "")) == "oxide" and isinstance(meta, dict) and meta.get("OXIDE_DESCRIPTOR_SUMMARY"):
        _ods = meta.get("OXIDE_DESCRIPTOR_SUMMARY") or {}
        _mode = str(meta.get("OXIDE_DESCRIPTOR_MODE", _ods.get("descriptor_mode", "Basic HER screening")))
        _rep = _pick_oxide_her_representatives(df_rel)
        _occ = _rep.get("occupied") if isinstance(_rep, dict) else None
        _opt = _rep.get("her_optimal") if isinstance(_rep, dict) else None
        _is_d2_only_mode = str(_mode).strip().startswith("D2_Hreact only")
        # D2 is now selected inside oxide_descriptor.py from independently
        # generated surface-metal-cation-centered H* candidates.  Do not overwrite
        # it with the main HER representative row, because the main HER table can
        # still contain O-top/anionic candidates used for basic screening.
        _align_d2_to_representative = False
        _display_ods = dict(_ods)
        _display_ods["descriptor_mode"] = _mode
        _d2_primary_results = bool(meta.get("OXIDE_DESCRIPTOR_PRIMARY_RESULTS", False))

        if _d2_primary_results:
            st.markdown("### Oxide D2 primary descriptor and selected profile")
            st.caption(
                "The primary HER result table above is now the selected D2 metal-cation-centered H* representative. "
                "Legacy O-anchor / anion-ontop HER rows are skipped in this mode."
            )
        else:
            st.markdown("### Oxide HER representative sites and selected profile")
        if _occ and not _d2_primary_results:
            st.caption(
                "Representative occupied site = most stabilized reliable H* site (minimum ΔG_H among reliable, non-duplicate candidates). "
                "This reflects the site expected to fill first under the occupancy-first interpretation."
            )
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Representative occupied site", str(_occ.get("site_label", "NA")))
            rc2.metric("Representative occupied ΔG_H (eV)", f"{float(_occ.get('energy', np.nan)):.4f}" if np.isfinite(_safe_float(_occ.get("energy"))) else "n/a")
            rc3.metric("Relaxed site", str(_occ.get("relaxed_site", "NA")))
            rc4.metric("Reliable candidate count", str(int(_rep.get("n_candidates", 0))))

            _rep_rows = [{
                "representative_type": "occupied-first representative",
                "rule": "minimum reliable ΔG_H",
                "site_label": _occ.get("site_label", "NA"),
                "relaxed_site": _occ.get("relaxed_site", "NA"),
                "ΔG_H (eV)": _occ.get("energy", np.nan),
                "binding_class": _occ.get("binding_class", ""),
            }]
            if _opt:
                _rep_rows.append({
                    "representative_type": "HER-optimal reference",
                    "rule": "minimum |ΔG_H| among reliable sites",
                    "site_label": _opt.get("site_label", "NA"),
                    "relaxed_site": _opt.get("relaxed_site", "NA"),
                    "ΔG_H (eV)": _opt.get("energy", np.nan),
                    "binding_class": _opt.get("binding_class", ""),
                })
            st.dataframe(pd.DataFrame(_rep_rows), use_container_width=True)

        if _is_d2_only_mode:
            st.markdown("#### Selected D2 descriptor result")
            st.caption(
                "In D2-only mode, the displayed D2 value is selected from valid surface-metal-cation-centered H* candidates using min(|ΔG_H|)."
            )
        else:
            st.markdown("#### Selected 2-stage descriptor profile")
            st.caption(
                "In full 2-stage mode, D1 remains the O-site protonation descriptor, while D2 is selected independently from valid surface-metal-cation-centered H* candidates using min(|ΔG_H|)."
            )
        # if _mode in {"D3_pair only (H2 pairing proxy)", "Full 3-stage profile (experimental)"}:
        #     st.warning(str(meta.get("OXIDE_DESCRIPTOR_CAUTION", _ods.get("caution", "The H₂ pairing stage is an approximate release proxy rather than an explicit barrier."))))

        summary_cols = [
            "descriptor_mode",
            "D1_OH (eV)", "D1_clean_OH (eV)", "D1_preOH_OH (eV)", "ΔD1_preOH-clean (eV)", "D2_Hreact (eV)",
            "Δ12 (eV)", "classification",
            "D1_site_label", "D1_binding_class", "D1_background_site_label", "D1_model",
            "D2_site_label", "D2_binding_class", "D2_final_site_kind", "D2_abs_Hreact (eV)", "D2_selection_rule",
            "D2_initial_bridge_pair", "D2_final_bridge_pair", "D2_primary_bridge_pair", "D2_bridge_pair_classes",
            "D2_initial_interface_like", "D2_final_interface_like",
            "D2_initial_local_Cu_fraction", "D2_final_local_Cu_fraction",
            "D2_target_count", "D2_nearest_metal_symbol", "D2_nearest_metal_distance(Å)",
            "D2_nearest_anion_symbol", "D2_nearest_anion_distance(Å)", "D2_qc_flags",
            "D2_same_basin_as_D1", "D2_basin_note",
            # "D3_pair_proxy (eV)", "Δ23 (eV)",
            # "D3_H2_like_motif", "D3_final_HH_distance(Å)",
            # "D3_pair_label", "D3_status", "D3_pair_seed_count", "D3_valid_pair_count",
        ]
        _summary_df = pd.DataFrame([{k: _display_ods.get(k, np.nan) for k in summary_cols}])
        st.dataframe(_summary_df, use_container_width=True)

        _bridge_csv = (
            _display_ods.get("D2_bridge_distribution_csv")
            or meta.get("OXIDE_DESCRIPTOR_D2_BRIDGE_DISTRIBUTION_CSV", "")
        )
        if _bridge_csv and Path(str(_bridge_csv)).is_file():
            st.markdown("#### D2 bridge-type-resolved HER descriptor distribution")
            st.caption(
                "Bridge classes are assigned from the final relaxed H* local metal environment. "
                "The near-zero fraction counts valid D2 sites with |ΔG_H| ≤ 0.30 eV."
            )
            try:
                _bridge_df = pd.read_csv(str(_bridge_csv))
                _bridge_metric_cols = st.columns(4)
                _bridge_metric_cols[0].metric(
                    "Primary bridge pair",
                    str(_display_ods.get("D2_primary_bridge_pair", "NA")),
                )
                _bridge_metric_cols[1].metric(
                    "Bridge classes",
                    str(_display_ods.get("D2_bridge_pair_classes", "NA")),
                )
                if "N_valid_sites" in _bridge_df.columns:
                    _bridge_metric_cols[2].metric(
                        "Valid D2 bridge sites",
                        str(int(pd.to_numeric(_bridge_df["N_valid_sites"], errors="coerce").fillna(0).sum())),
                    )
                else:
                    _bridge_metric_cols[2].metric("Valid D2 bridge sites", "NA")
                if "best_abs_ΔG_H(eV)" in _bridge_df.columns and len(_bridge_df) > 0:
                    _best_abs = pd.to_numeric(_bridge_df["best_abs_ΔG_H(eV)"], errors="coerce").min()
                    _bridge_metric_cols[3].metric(
                        "Best |ΔG_H|",
                        f"{float(_best_abs):.4f} eV" if np.isfinite(_safe_float(_best_abs)) else "NA",
                    )
                else:
                    _bridge_metric_cols[3].metric("Best |ΔG_H|", "NA")

                st.dataframe(_bridge_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download D2 bridge distribution CSV",
                    _bridge_df.to_csv(index=False).encode("utf-8"),
                    "D2_bridge_distribution.csv",
                    "text/csv",
                    key=f"dl_d2_bridge_distribution_{uuid.uuid4().hex[:8]}",
                )
            except Exception as _e:
                st.info(f"Could not load D2 bridge distribution table: {_e}")
        elif str(_display_ods.get("D2_bridge_distribution_error", "")).strip():
            st.info(f"D2 bridge distribution summary was not generated: {_display_ods.get('D2_bridge_distribution_error')}")

        _profile_points = []
        _d1 = _safe_float(_display_ods.get("D1_OH (eV)"))
        _d2 = _safe_float(_display_ods.get("D2_Hreact (eV)"))
        if (not _d2_primary_results) and (not _align_d2_to_representative) and _occ and np.isfinite(_safe_float(_occ.get("energy"))) and np.isfinite(_d2):
            _delta_rep = abs(float(_occ.get("energy")) - _d2)
            if _delta_rep > 1e-8:
                st.info(
                    f"Selected profile D2 = {_d2:.4f} eV, while the representative occupied site = {float(_occ.get('energy')):.4f} eV "
                    f"({_occ.get('site_label', 'NA')})."
                )
        # _d3 = _safe_float(_ods.get("D3_pair_proxy (eV)"))
        if np.isfinite(_d1):
            _profile_points.append({"Stage": "Pre-hydroxylated O–H formation", "Energy (eV)": _d1})
        if np.isfinite(_d2):
            _profile_points.append({"Stage": "Reactive H state", "Energy (eV)": _d2})
        # if np.isfinite(_d3):
        #     _profile_points.append({"Stage": "H₂ pairing proxy", "Energy (eV)": _d3})
        if _profile_points:
            _profile_df = pd.DataFrame(_profile_points)
            st.line_chart(_profile_df.set_index("Stage"))

        def _render_descriptor_stage_viewer(stage_key: str, title: str, energy_key: str, label_key: str, extra_items: list[tuple[str, str]] | None = None):
            stage_path = _display_ods.get(f"{stage_key}_structure_cif", "")
            stage_label = str(_display_ods.get(label_key, "NA"))
            stage_energy = _safe_float(_display_ods.get(energy_key, np.nan))
            with st.expander(title, expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("label", stage_label)
                c2.metric("energy (eV)", f"{stage_energy:.4f}" if np.isfinite(stage_energy) else "n/a")
                if extra_items:
                    extra_key, extra_label = extra_items[0]
                    extra_val = _display_ods.get(extra_key, "")
                    c3.metric(extra_label, str(extra_val) if str(extra_val).strip() else "n/a")
                else:
                    c3.metric("structure", "available" if stage_path and Path(str(stage_path)).is_file() else "missing")
                if extra_items and len(extra_items) > 1:
                    extra_df = pd.DataFrame([{lbl: _display_ods.get(key, np.nan) for key, lbl in extra_items[1:]}])
                    st.dataframe(extra_df, use_container_width=True)
                stage_meta_keys = [
                    (f"{stage_key}_relaxation_scope", "scope"),
                    (f"{stage_key}_total_relax_n_steps", "total steps"),
                    (f"{stage_key}_fine_relax_relaxed_atoms", "relaxed atoms"),
                ]
                stage_meta_row = {lbl: _display_ods.get(key, np.nan) for key, lbl in stage_meta_keys if key in _display_ods}
                if stage_meta_row:
                    st.dataframe(pd.DataFrame([stage_meta_row]), use_container_width=True)
                if stage_path and Path(str(stage_path)).is_file():
                    try:
                        _at_stage = read(str(stage_path))
                        show_atoms_3d(_at_stage, height=420, width=880, tag=f"descriptor_{stage_key}_{uuid.uuid4().hex[:8]}")
                        st.download_button(
                            f"Download {stage_key} CIF",
                            Path(stage_path).read_bytes(),
                            Path(stage_path).name,
                            "chemical/x-cif",
                            key=f"dl_descriptor_{stage_key}_{uuid.uuid4().hex[:8]}",
                        )
                    except Exception as _e:
                        st.info(f"Could not render {stage_key} CIF: {_e}")
                else:
                    st.info(f"{stage_key} CIF is not available for this run.")

        st.markdown("#### Descriptor stage structures")
        _render_descriptor_stage_viewer(
            "D1",
            "D1 structure — pre-hydroxylated O–H formation",
            "D1_OH (eV)",
            "D1_site_label",
            extra_items=[("D1_binding_class", "binding class"), ("D1_background_site_label", "background OH site"), ("D1_model", "model"), ("D1_seed_quality", "seed quality"), ("D1_seed_disp(Å)", "seed disp (Å)"), ("D1_seed_warning", "seed warning")],
        )
        _render_descriptor_stage_viewer(
            "D2",
            "D2 structure — reactive H state",
            "D2_Hreact (eV)",
            "D2_site_label",
            extra_items=[
                ("D2_binding_class", "binding class"),
                ("D2_final_site_kind", "final site kind"),
                ("D2_initial_bridge_pair", "initial bridge"),
                ("D2_final_bridge_pair", "final bridge"),
                ("D2_primary_bridge_pair", "primary bridge"),
                ("D2_selection_rule", "selection rule"),
                ("D2_initial_local_Cu_fraction", "initial local Cu fraction"),
                ("D2_final_local_Cu_fraction", "final local Cu fraction"),
                ("D2_nearest_metal_symbol", "nearest metal"),
                ("D2_nearest_metal_distance(Å)", "nearest metal dist (Å)"),
                ("D2_qc_flags", "QC flags"),
                ("D2_same_basin_as_D1", "same basin as D1"),
                ("D2_basin_note", "basin note"),
                ("D2_seed_disp(Å)", "seed disp (Å)"),
            ],
        )
        # _render_descriptor_stage_viewer(
        #     "D3",
        #     "D3 structure — H₂ pairing proxy",
        #     "D3_pair_proxy (eV)",
        #     "D3_pair_label",
        #     extra_items=[("D3_H2_like_motif", "H2-like motif"), ("D3_final_HH_distance(Å)", "final HH dist (Å)"), ("D3_status", "status")],
        # )

        with st.expander("Show descriptor candidate tables", expanded=False):
            _d2_csv = _display_ods.get("D2_candidates_csv") or meta.get("OXIDE_DESCRIPTOR_D2_CANDIDATES_CSV", "")
            _d2_bridge_csv = (
                _display_ods.get("D2_bridge_distribution_csv")
                or meta.get("OXIDE_DESCRIPTOR_D2_BRIDGE_DISTRIBUTION_CSV", "")
            )
            # _d3_csv = _ods.get("D3_candidates_csv") or meta.get("OXIDE_DESCRIPTOR_D3_CANDIDATES_CSV", "")
            _d1_csv = _display_ods.get("D1_candidates_csv", "")
            if _d1_csv and Path(str(_d1_csv)).is_file():
                st.markdown("#### D1 pre-hydroxylated candidates")
                try:
                    st.dataframe(pd.read_csv(str(_d1_csv)), use_container_width=True)
                except Exception as _e:
                    st.info(f"Could not load D1 candidate table: {_e}")
            if _d2_csv and Path(str(_d2_csv)).is_file():
                st.markdown("#### D2 reactive-H candidates")
                try:
                    st.dataframe(pd.read_csv(str(_d2_csv)), use_container_width=True)
                except Exception as _e:
                    st.info(f"Could not load D2 candidate table: {_e}")
            if _d2_bridge_csv and Path(str(_d2_bridge_csv)).is_file():
                st.markdown("#### D2 bridge distribution")
                try:
                    st.dataframe(pd.read_csv(str(_d2_bridge_csv)), use_container_width=True, hide_index=True)
                except Exception as _e:
                    st.info(f"Could not load D2 bridge distribution table: {_e}")
            # if _d3_csv and Path(str(_d3_csv)).is_file():
            #     st.markdown("#### D3 H₂ pairing proxy candidates")
            #     try:
            #         st.dataframe(pd.read_csv(str(_d3_csv)), use_container_width=True)
            #     except Exception as _e:
            #         st.info(f"Could not load D3 candidate table: {_e}")

    # --- Relaxed post-run structure viewer ---
    viewer_frames = {}
    df_dedup = locals().get("df_dedup", pd.DataFrame())
    df_keep = locals().get("df_keep", pd.DataFrame())
    df_reject = locals().get("df_reject", pd.DataFrame())
    df_diag = locals().get("df_diag", pd.DataFrame())

    if bool(last_run.get("is_her")):
        if isinstance(df_rel, pd.DataFrame) and (not df_rel.empty):
            viewer_frames["Reliable results"] = df_rel
        if isinstance(df_unrel, pd.DataFrame) and (not df_unrel.empty):
            viewer_frames["Unreliable / unstable"] = df_unrel
        if (not viewer_frames) and isinstance(df, pd.DataFrame) and (not df.empty):
            viewer_frames["All results"] = df
    else:
        if isinstance(df_dedup, pd.DataFrame) and (not df_dedup.empty):
            viewer_frames["Candidates (dedup)"] = df_dedup
        if isinstance(df_keep, pd.DataFrame) and (not df_keep.empty):
            viewer_frames["All candidate attempts"] = df_keep
        if bool(last_run.get("is_voc")) and isinstance(df_diag, pd.DataFrame) and (not df_diag.empty):
            viewer_frames["Diagnostic attempts"] = df_diag
        if isinstance(df_reject, pd.DataFrame) and (not df_reject.empty):
            viewer_frames["Rejected attempts"] = df_reject
        if (not viewer_frames) and isinstance(df, pd.DataFrame) and (not df.empty):
            viewer_frames["All results"] = df

    if viewer_frames:
        with st.expander("Relaxed post-run structure viewer", expanded=False):
            viewer_source = st.selectbox(
                "Result group",
                list(viewer_frames.keys()),
                index=0,
                key="relaxed_view_source",
            )
            viewer_df = viewer_frames.get(viewer_source, pd.DataFrame()).reset_index(drop=True)

            if isinstance(viewer_df, pd.DataFrame) and (not viewer_df.empty):
                option_labels = [_format_relaxed_view_option(r, is_her=bool(last_run.get("is_her"))) for _, r in viewer_df.iterrows()]
                viewer_idx = st.selectbox(
                    "Select relaxed structure",
                    list(range(len(option_labels))),
                    format_func=lambda i: option_labels[i],
                    index=0,
                    key=f"relaxed_view_idx_{viewer_source}",
                )

                viewer_row = viewer_df.iloc[int(viewer_idx)]
                _has_initial_cif = bool(str(viewer_row.get("initial_structure_cif", "") or "").strip())
                _structure_view_mode = "relaxed"
                if _has_initial_cif:
                    _structure_view_mode = st.radio(
                        "Structure snapshot",
                        ["initial", "relaxed"],
                        index=1,
                        horizontal=True,
                        help="For OER diagnostics, initial shows the placed adsorbate before adsorbate relaxation; relaxed shows the post-run structure.",
                        key=f"structure_snapshot_{viewer_source}_{viewer_idx}",
                    )
                if _structure_view_mode == "initial" and _has_initial_cif:
                    viewer_path = Path(str(viewer_row.get("initial_structure_cif"))).expanduser()
                else:
                    viewer_path = _resolve_relaxed_structure_path(viewer_row, csv_path=last_run.get("csv_path"))

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("site_label", str(viewer_row.get("site_label", "?")))
                m2.metric("initial", str(viewer_row.get("initial_site_kind", viewer_row.get("requested_site", "?"))))
                m3.metric("final", str(viewer_row.get("final_site_kind", viewer_row.get("relaxed_site", "?"))))
                disp_view = _safe_float(viewer_row.get("H_lateral_disp(Å)", viewer_row.get("ads_lateral_disp(Å)", np.nan)))
                m4.metric("disp (Å)", f"{disp_view:.2f}" if np.isfinite(disp_view) else "n/a")

                thermo_mode_run = str(meta.get("thermo_mode", "CHE correction (fast screening)")) if isinstance(meta, dict) else "CHE correction (fast screening)"
                extra_order = [
                    "migrated", "reliability", "qa", "ΔG_H(U,pH) (eV)", "ΔG_H (eV)",
                    "ΔG_H_CHE (eV)", "ΔG_H_local (eV)", "local_thermo_corr (eV)",
                    "ΔE_ads_user (eV)", "ΔG_ads (eV)", "ΔE_H_user (eV)",
                    "oer_diagnostic_mode", "oer_diagnostic_active", "oer_ads_relax_steps_used", "oer_ads_relax_fmax_used",
                    "oxygen_bound_to_cation", "oxygen_anchor_target_cation_dist(Å)", "oxygen_anchor_anion_dist(Å)",
                    "valid_for_oer_summary", "oer_state_class", "oer_state_note",
                    "site_transition_type", "migrated_actual", "placement_mismatch", "site_family",
                    "selected_h0", "her_relaxation_scope", "z_relax_n_steps", "fine_relax_n_steps", "fine_relax_relaxed_atoms", "total_relax_n_steps",
                    "zpe_scope", "zpe_selected_atoms", "zpe_warning"
                ]
                if bool(last_run.get("is_her")) and thermo_mode_run.startswith("Local ZPE correction"):
                    extra_order = [c for c in extra_order if c != "ΔG_H_CHE (eV)"]
                extra_cols = [c for c in extra_order if c in viewer_row.index]
                if extra_cols:
                    st.dataframe(pd.DataFrame([viewer_row[extra_cols].to_dict()]), use_container_width=True)

                if viewer_path is not None and Path(viewer_path).is_file():
                    try:
                        at_view = read(str(viewer_path))
                        show_atoms_3d(at_view, height=460, width=900, tag=f"relaxed_view_{viewer_source}_{viewer_idx}")
                        st.download_button(
                            f"Download selected {_structure_view_mode} CIF",
                            Path(viewer_path).read_bytes(),
                            Path(viewer_path).name,
                            "chemical/x-cif",
                            key=f"dl_relaxed_cif_{viewer_source}_{viewer_idx}",
                        )
                    except Exception as e:
                        st.warning(f"Could not render relaxed CIF: {e}")
                else:
                    st.info("Relaxed CIF path could not be resolved for this row, or the available CIF basename does not match the selected adsorbate/state. This prevents accidental cross-linking such as opening H* for a CH3CH2OH* row. Re-run with the latest VOC exporter if the row should have a relaxed CIF.")

    # download full results
    try:
        st.download_button(
            "Download CSV Results",
            df.to_csv(index=False).encode("utf-8") if isinstance(df, pd.DataFrame) else b"",
            "che_results.csv" if bool(last_run.get("is_her")) else "co2rr_results.csv",
            "text/csv",
            key="dl_results_csv",
        )
    except Exception:
        pass


    # ---------------- LLM interpretation ----------------
    if st.session_state.get("llm_enabled", False):
        with st.expander("LLM interpretation (post-run)", expanded=False):
            st.caption(
                "Generates a structured interpretation based on the current 'last run' table and (if available) Materials Project metadata."
            )
            if st.button("Generate interpretation", key="btn_llm_interpret"):
                try:
                    payload = build_llm_payload(last_run, mp_api_key=st.session_state.get("mp_api_key") or None)
                    out = call_llm_interpreter(payload, api_key=st.session_state.get("openai_api_key") or "", model_name=st.session_state.get("llm_model", "gpt-4o-mini"))
                    # Display token usage / estimated cost (best effort)
                    usage = out.get("_usage") if isinstance(out, dict) else None
                    cost = out.get("_cost_estimate_usd") if isinstance(out, dict) else None
                    model_name = out.get("_model") if isinstance(out, dict) else None
                    if isinstance(usage, dict):
                        in_toks = usage.get("input_tokens", usage.get("prompt_tokens"))
                        out_toks = usage.get("output_tokens", usage.get("completion_tokens"))
                        total = usage.get("total_tokens")
                        msg = f"LLM usage ({model_name}): input={in_toks}, output={out_toks}"
                        if total is not None:
                            msg += f", total={total}"
                        if cost is not None:
                            try:
                                msg += f" | est. cost=${float(cost):.6f}"
                            except Exception:
                                pass
                        st.caption(msg)

                    st.session_state["llm_last_out"] = out
                    st.success("LLM interpretation generated.")
                except Exception as e:
                    st.error(str(e))

            out = st.session_state.get("llm_last_out")
            if isinstance(out, dict):
                st.markdown("#### Executive summary")
                st.write(out.get("executive_summary", ""))

                st.markdown("#### QC findings")
                for x in (out.get("qc_findings") or []):
                    st.write(f"- {x}")

                st.markdown("#### Site interpretation")
                for x in (out.get("site_interpretation") or []):
                    st.write(f"- {x}")

                st.markdown("#### Recommended for paper (two-track)")
                rec = out.get("recommended_for_paper") or {}
                if isinstance(rec, dict):
                    primary = rec.get("primary_track_to_report", "")
                    primary_display = primary
                    strong_tr = rec.get("strong_binding_track") if isinstance(rec.get("strong_binding_track"), dict) else {}
                    if primary == "strong_binding" and isinstance(strong_tr, dict):
                        crit = str(strong_tr.get("criterion", "")).strip()
                        if crit == "min_dE_all_positive":
                            primary_display = "least_unfavorable (min ΔE among stable; all ΔE_ads ≥ 0)"
                    if primary_display:
                        st.write(f"- **Primary track**: {primary_display}")
                
                    def _render_track(title: str, tr):
                        st.write(f"**{title}**")
                        if not isinstance(tr, dict):
                            st.write(tr)
                            return
                        st.write(f"- site_label: {tr.get('site_label')}")
                        st.write(f"- value: {tr.get('value')} (column: {tr.get('column_used')}, criterion: {tr.get('criterion')})")
                        st.write(f"- selection_rule: {tr.get('selection_rule','')}")
                        st.write(f"- evidence_note: {tr.get('evidence_note','')}")
                
                    strong_title = "Strong binding track"
                    strong_tr2 = rec.get("strong_binding_track")
                    if isinstance(strong_tr2, dict) and str(strong_tr2.get("criterion", "")).strip() == "min_dE_all_positive":
                        strong_title = "Least-unfavorable track (min ΔE among stable; all ΔE_ads ≥ 0)"
                    balanced_title = "Balanced binding track"
                    bal_tr2 = rec.get("balanced_binding_track")
                    if isinstance(bal_tr2, dict) and str(bal_tr2.get("criterion", "")).strip().startswith("N/A"):
                        balanced_title = "Balanced binding track (N/A — no exothermic candidates)"
                    _render_track(strong_title, rec.get("strong_binding_track"))
                    _render_track(balanced_title, rec.get("balanced_binding_track"))
                
                    if rec.get("how_to_report"):
                        st.write(f"- **how_to_report**: {rec.get('how_to_report')}")
                    if rec.get("suggested_paper_text"):
                        st.write(f"- **suggested_paper_text**: {rec.get('suggested_paper_text')}")
                else:
                    st.write(rec)

                st.markdown("#### Next actions")
                for x in (out.get("next_actions") or []):
                    st.write(f"- {x}")

                st.markdown("#### Uncertainties")
                for x in (out.get("uncertainties") or []):
                    st.write(f"- {x}")
