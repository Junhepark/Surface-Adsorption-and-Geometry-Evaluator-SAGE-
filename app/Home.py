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
    run_metal_orr_che,    # ORR support
    run_oxide_orr_che,    # ORR support
)
try:
    from ocp_app.core.gas_refs import make_orr_templates as _make_orr_templates
except Exception:
    _make_orr_templates = None

# Auto-generate ORR intermediate template CIFs (only if missing)
if _make_orr_templates is not None:
    try:
        _make_orr_templates("ref_gas")
    except Exception:
        pass
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
)
from ocp_app.core.surface_families import (
    INTERFACE_FACET_PRESETS,
    THICKNESS_ALLOCATION_OPTIONS,
    _infer_interface_surface_family,
    _get_interface_facet_options,
    _recommended_interface_facet_labels,
    _facet_labels_for_mode,
)
from ocp_app.core.interface_builder import (
    _allocation_to_thicknesses,
    _inplane_metrics,
    _auto_max_xy_repeat,
    _find_best_xy_repeat_pair,
    _scale_film_to_substrate_xy,
    _auto_initial_gap,
    _stack_interface_pair,
    _registry_candidates_auto,
    _geometry_prefilter_interface_work,
    _select_best_slab_for_facet,
    _build_interface_candidates_from_bulks,
)
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
    slabify_from_bulk,
)
from ocp_app.core.postprocess import (
    split_reliable_unreliable,
    _normalize_text_series,
    co2rr_apply_qa_policy,
    co2rr_split_by_qa,
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
from ocp_app.core.structure_check import validate_structure
from ocp_app.core.ads_sites import _oxide_o_based_ads_position
from ocp_app.core.ads_sites import oxide_surface_seed_position, expand_oxide_channels_for_adsorbate, ANION_SYMBOLS
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
st.title("Surface Adsorption and Geometry Evaluator(SAGE) — HER / CO₂RR / ORR (HAPLAB v1.0)")

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


def _resolve_relaxed_structure_path(row: pd.Series | dict, csv_path: str | Path | None = None):
    """Best-effort resolver for post-run relaxed structure CIF paths.

    Priority:
      1) explicit row['structure_cif'] when present
      2) infer from csv output root: <root>/sample/sites/user_<site_label>_H.cif
      3) infer adsorbate-specific path: <root>/sample/sites/user_<site_label>_<ADS>.cif
    """
    row = row if isinstance(row, dict) else dict(row)
    explicit = row.get("structure_cif", None)
    if explicit:
        p = Path(str(explicit)).expanduser()
        if p.is_file():
            return p

    root = None
    if csv_path:
        try:
            root = Path(str(csv_path)).expanduser().resolve().parent
        except Exception:
            try:
                root = Path(str(csv_path)).expanduser().parent
            except Exception:
                root = None

    site_label = str(row.get("site_label", "")).strip()
    site_label_file = site_label.replace(":", "__") if site_label else ""
    ads = str(row.get("adsorbate", "")).replace("*", "").strip().upper()

    candidates = []
    if root is not None and site_label_file:
        if ads:
            candidates.append(root / "sample" / "sites" / f"user_{site_label_file}_{ads}.cif")
        candidates.append(root / "sample" / "sites" / f"user_{site_label_file}_H.cif")

    for p in candidates:
        try:
            if Path(p).is_file():
                return Path(p)
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
    dg = _safe_float(row.get("ΔG_ads (eV)", np.nan))
    return f"{ads} | {site_label} | final={relaxed_site} | ΔG_ads={dg:.3f} eV | {qa}"

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
        ["HER (ΔG_H)", "CO₂RR (ΔG_ads)", "ORR (ΔG_ads)"],
        horizontal=False,
        help="HER: H* adsorption free energy | CO₂RR: COOH*/CO*/HCOO*/OCHO* | ORR: OOH*/O*/OH* (Norskov 4e⁻ CHE)",
    )
    is_her = reaction_mode.startswith("HER")
    is_orr = reaction_mode.startswith("ORR")

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

    site_preset = st.multiselect("Sites (Manual mode)", default_sites, default=default_sites)

    if is_her:
        co2_ads = []
        orr_ads = []
    elif is_orr:
        co2_ads = []
        orr_ads = st.multiselect(
            "ORR intermediates",
            ["OOH*", "O*", "OH*"],
            default=["OOH*", "O*", "OH*"],
            help="Norskov 4-electron pathway: OOH* → O* → OH* → H₂O",
        )
        orr_U = st.number_input(
            "Applied potential U (V vs RHE)",
            min_value=-2.0, max_value=2.0, value=0.0, step=0.05,
            help="0.0 = equilibrium potential (1.23 V vs SHE). More negative = higher reduction overpotential.",
            key="orr_U_input",
        )
    else:
        co2_ads = st.multiselect(
            "CO₂RR intermediates",
            ["COOH*", "CO*", "HCOO*", "OCHO*"],
            default=["COOH*", "CO*", "HCOO*", "OCHO*"],
        )
        orr_ads = []


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

    if surface_route == "Slabify from bulk":
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
                facet_choices = _facet_choices_for_scope(prepared, facet_scope_for_calc)
                facet_labels = [_format_hkl_label(hkl) for hkl in facet_choices]
                oxide_surface_mode = st.selectbox(
                    "Oxide clean-surface selection mode",
                    ["Reference clean surface", "O-dominant surface preference", "Exploratory any clean termination"],
                    index=0,
                    key="oxide_surface_mode",
                    help="Reference clean surface keeps conservative family-aware candidates. O-dominant preference still favors O-rich or mixed tops. Exploratory any clean termination keeps non-rejected clean candidates.",
                )
                st.caption(
                    "O-dominant preference does not enforce a pure O-top surface. "
                    "Please check `surface_O_fraction_top` and `facet_warning` before using this slab."
                )
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
                                    a_n, m_n = _normalize_oxide_candidate_top_surface(a_i, m_i, z_window=1.8)
                                    m_n["oxide_surface_mode"] = mode_pref
                                    m_n["hydroxylation_mode"] = "Clean only"
                                    keep = _oxide_mode_keep_candidate(m_n, mode_pref)
                                    if keep:
                                        m_n["oxide_rank_key"] = _oxide_candidate_rank_key(m_n)
                                        norm_atoms.append(a_n)
                                        norm_meta.append(m_n)
                                    else:
                                        rejected += 1
                                cand_atoms, cand_meta = norm_atoms, norm_meta
                                if rejected:
                                    st.info(f"Filtered out {rejected} oxide candidate(s) that failed the current clean-surface selection mode.")
                                if not cand_atoms:
                                    raise ValueError("No oxide slab candidates remained after family-aware clean-surface filtering. Try another facet or switch to a less restrictive clean-surface mode.")
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
                    df_cands = pd.DataFrame(cand_meta)
                    if mtype == "oxide":
                        st.warning(
                            "Please review `surface_O_fraction_top`, `surface_family`, and `facet_warning` before selecting an oxide slab."
                        )
                        pref_cols = [c for c in [
                            "idx", "miller", "surface_family", "crystal_system", "spacegroup_symbol", "spacegroup_number", "rule_validity", "rule_role", "surface_diagnostics_status", "slab_usability", "oxide_validity", "oxide_role", "top_exposure", "bottom_exposure", "surface_O_fraction_top",
                            "surface_O_fraction_bottom", "top_bottom_asymmetric", "flipped_for_oxide_top_exposure", "oxide_top_surface_ok", "facet_warning", "n_atoms",
                            "vacuum_z", "recommend_repeat", "slab_usability_reason", "oxide_rule_notes", "surface_diagnostics_notes", "issues"
                        ] if c in df_cands.columns]
                        st.dataframe(df_cands[pref_cols], use_container_width=True)
                        auto_atoms, auto_meta = _pick_best_oxide_slab_candidate(cand_atoms, cand_meta)
                        auto_idx = next((i for i, m in enumerate(cand_meta) if m.get("idx") == auto_meta.get("idx")), 0)
                        st.caption(f"Auto-selected oxide candidate: #{auto_idx} | usability={auto_meta.get('slab_usability')} | rule={auto_meta.get('rule_validity')}/{auto_meta.get('rule_role')} | top={auto_meta.get('top_exposure')} | atoms={auto_meta.get('n_atoms')}")
                    else:
                        st.dataframe(df_cands, use_container_width=True)
                        auto_idx = 0

                    sel_idx = st.selectbox(
                        "Select slab candidate",
                        list(range(len(cand_atoms))),
                        index=int(auto_idx),
                        format_func=lambda i: (
                            f"#{i} | {cand_meta[i].get('slab_usability')} | rule={cand_meta[i].get('rule_validity')}/{cand_meta[i].get('rule_role')} | top={cand_meta[i].get('top_exposure')} | vac_z={cand_meta[i].get('vacuum_z', np.nan):.2f} Å | atoms={cand_meta[i].get('n_atoms')} | {cand_meta[i].get('formula')}"
                            if mtype == "oxide" else
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
                            },
                        )
                        st.success("Selected slab applied. Continue with vacuum / supercell / QC below.")
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

        _render_min_dist_panel(rep)

        if mtype == "oxide":
            fam = infer_oxide_family_from_atoms(prepared)
            if fam["family"] != "unknown":
                st.info(
                    f"Detected Oxide: {fam['family']} ({fam['reduced_formula']}) | crystal={fam.get('crystal_system')} | sg={fam.get('spacegroup_symbol')}"
                )
            surf_meta = _classify_surface_exposure(prepared, z_window=1.8)
            st.write(f"- Surface exposure (top/bottom): **{surf_meta['top_exposure']} / {surf_meta['bottom_exposure']}**")
            st.write(f"- Surface O fraction (top): **{surf_meta['surface_O_fraction_top']:.2f}**")
            prep_meta = _normalize_oxide_candidate_top_surface(prepared, {"surface_family": _infer_interface_surface_family(prepared), "miller": None}, z_window=1.8)[1]
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
            if surf_meta['top_bottom_asymmetric']:
                st.caption("Oxide note: top/bottom terminations are asymmetric. Interpret clean-slab HER outputs cautiously.")

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
        st.markdown("#### Vacuum")
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
        if st.button("Apply vacuum", key="btn_apply_vacuum_common"):
            a2 = add_vacuum_z(prepared, total_vacuum_z=float(target_vac), keep_pbc_z=bool(keep_pbc_z))
            _push_prepared_update(a2, "add_vacuum", {"total_vacuum_z": float(target_vac), "keep_pbc_z": bool(keep_pbc_z)})
            st.success(f"Vacuum set to {float(target_vac):.1f} Å.")
            st.rerun()

        st.markdown("#### Supercell (XY only)")
        if surface_route == "Slabify from bulk":
            st.caption("For the slabify route, keep 1×1 by default and expand only when the in-plane slab is too small.")
            a_len, b_len = _surface_xy_lengths(prepared)
            nx_auto, ny_auto = _suggest_minimal_xy_repeat(prepared, min_length_a=8.0, min_length_b=8.0, max_repeat=3)
            st.write(f"- In-plane lengths: **a = {a_len:.2f} Å**, **b = {b_len:.2f} Å**")
            st.write(f"- Minimal suggested repeat: **{nx_auto}×{ny_auto}×1**")

            colR1, colR2, colR3 = st.columns(3)
            with colR1:
                if st.button("Keep 1×1", key="btn_rep_keep_111"):
                    st.info("Kept current slab at 1×1.")
            with colR2:
                if st.button("Apply minimal repeat", key="btn_rep_auto_minimal"):
                    if int(nx_auto) == 1 and int(ny_auto) == 1:
                        st.info("Current slab already satisfies the minimal in-plane size target.")
                    else:
                        a2 = repeat_xy(prepared, int(nx_auto), int(ny_auto))
                        _push_prepared_update(a2, "repeat_xy_minimal", {"nx": int(nx_auto), "ny": int(ny_auto), "from": "slabify_minimal_xy"})
                        st.success(f"Applied minimal repeat: {int(nx_auto)}×{int(ny_auto)}×1.")
                        st.rerun()
            with colR3:
                if st.button("Apply 2×2×1", key="btn_rep_221_slabify"):
                    a2 = repeat_xy(prepared, 2, 2)
                    _push_prepared_update(a2, "repeat_xy", {"nx": 2, "ny": 2, "from": "slabify_manual"})
                    st.success("Applied 2×2×1.")
                    st.rerun()
        else:
            colR1, colR2, colR3 = st.columns(3)
            with colR1:
                if st.button("2×2×1", key="btn_rep_221"):
                    a2 = repeat_xy(prepared, 2, 2)
                    _push_prepared_update(a2, "repeat_xy", {"nx": 2, "ny": 2})
                    st.success("Applied 2×2×1.")
                    st.rerun()
            with colR2:
                if st.button("4×4×1", key="btn_rep_441"):
                    a2 = repeat_xy(prepared, 4, 4)
                    _push_prepared_update(a2, "repeat_xy", {"nx": 4, "ny": 4})
                    st.success("Applied 4×4×1.")
                    st.rerun()
            with colR3:
                if st.button("Use auto recommendation", key="btn_rep_auto"):
                    rec = getattr(rep, "recommend_repeat", None)
                    if not rec:
                        st.warning("No repeat recommendation available.")
                    else:
                        nx, ny, _nz = rec
                        a2 = repeat_xy(prepared, int(nx), int(ny))
                        _push_prepared_update(a2, "repeat_xy_auto", {"nx": int(nx), "ny": int(ny), "from": "validate_structure"})
                        st.success(f"Applied {int(nx)}×{int(ny)}×1.")
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

    st.markdown("### Structure preview and optional active-region z crop")
    prev3_col, crop3_col = st.columns([1.45, 0.85])
    with prev3_col:
        show_atoms_3d(atoms_for_sites, height=420, width=900, tag="step3_prepared_preview")
    with crop3_col:
        rep_step3 = validate_structure(atoms_for_sites, target_area=70.0)
        vac_z_step3 = float(getattr(rep_step3, "vacuum_z", 0.0))
        crop_diag_step3 = suggest_active_region_crop(
            atoms_for_sites,
            thickness_threshold_A=12.0,
            atoms_threshold=160,
            default_window_A=10.0,
        )
        slab_thk_step3 = float(crop_diag_step3.get("slab_thickness_A", float("nan")))
        st.write(f"- Atoms: **{len(atoms_for_sites)}**")
        st.write(f"- Vacuum_z: **{vac_z_step3:.2f} Å**")
        st.write(
            f"- Slab thickness: **{slab_thk_step3:.2f} Å**"
            if np.isfinite(slab_thk_step3)
            else "- Slab thickness: **n/a**"
        )

        if (vac_z_step3 < 10.0) and bool(atoms_for_sites.get_pbc()[2]):
            st.warning(
                f"Prepared structure is BULK-like (vacuum_z={vac_z_step3:.2f} Å, pbc_z=True). "
                "Adsorption sites may collapse and results may be unreliable."
            )

        auto_crop_default_step3 = bool(crop_diag_step3.get("recommend_crop", False))
        if auto_crop_default_step3:
            st.warning(
                "The current slab is deep along z and may destabilize adsorption relaxation. "
                "Crop the upper active region before site generation / screening?"
            )
        else:
            st.caption(
                "You can still crop manually if you only need surface ΔG_H/ΔG_ads rather than deep subsurface effects."
            )

        use_step3_crop = st.checkbox(
            "Use active-region z crop",
            value=auto_crop_default_step3,
            key="use_step3_active_crop",
        )

        if use_step3_crop:
            crop_keep_top_window_step3 = st.number_input(
                "Keep top active window (Å)",
                min_value=6.0,
                max_value=20.0,
                value=float(crop_diag_step3.get("suggested_window_A", 10.0)),
                step=0.5,
                key="crop_keep_top_window_step3_A",
            )
            crop_target_vac_step3 = st.number_input(
                "Target vacuum after crop (Å)",
                min_value=12.0,
                max_value=60.0,
                value=max(20.0, float(vac_z_step3) if np.isfinite(vac_z_step3) else 30.0),
                step=1.0,
                key="crop_target_vacuum_step3_A",
            )
            crop_min_layers_step3 = st.number_input(
                "Minimum preserved z-layers",
                min_value=2,
                max_value=12,
                value=4,
                step=1,
                key="crop_min_layers_step3",
            )
            if st.button("Apply z crop to prepared structure", key="btn_apply_step3_crop"):
                try:
                    a2, crop_meta = crop_top_slab_window(
                        atoms_for_sites,
                        keep_top_window_A=float(crop_keep_top_window_step3),
                        target_vacuum_z=float(crop_target_vac_step3),
                        keep_pbc_z=True,
                        min_layers=int(crop_min_layers_step3),
                        min_atoms=16,
                        layer_tol=0.8,
                    )
                    _push_prepared_update(a2, "crop_top_window_step3", crop_meta)
                    if bool(crop_meta.get("cropped", False)):
                        st.success(
                            f"Applied active-region z crop: kept {crop_meta.get('kept_atoms')} atoms, "
                            f"removed {crop_meta.get('removed_atoms')} atoms, thickness "
                            f"{crop_meta.get('original_thickness_A', float('nan')):.2f} → {crop_meta.get('cropped_thickness_A', float('nan')):.2f} Å."
                        )
                    else:
                        st.info("No effective z crop was needed. Vacuum was normalized and the prepared structure was refreshed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Active-region z crop failed: {e}")

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

    # Defaults for oxide HER constrained CHGNet preconditioning
    her_constrained_prerelax = False
    her_constrained_top_free_layers = int(st.session_state.get("her_constrained_top_free_layers", 2))
    her_constrained_layer_tol = float(st.session_state.get("her_constrained_layer_tol", 0.8))
    her_constrained_fmax = float(st.session_state.get("her_constrained_fmax", 0.05))
    her_constrained_max_steps = int(st.session_state.get("her_constrained_max_steps", 80))
    her_constrained_seed_ui = int(st.session_state.get("her_constrained_seed_ui", 0))
    her_constrained_seed = None

    if is_her and (mtype == "oxide"):
        st.markdown("### Oxide HER constrained CHGNet preconditioning (experimental)")
        her_constrained_prerelax = st.checkbox(
            "Apply constrained CHGNet slab pre-relaxation",
            value=bool(st.session_state.get("her_constrained_prerelax_ui", False)),
            key="her_constrained_prerelax_ui",
            help=(
                "Precondition the clean oxide slab before O-top / reactive-H probing. "
                "Only the top z-layers remain free; lower layers are fixed to preserve bulk-like support."
            ),
        )
        if bool(her_constrained_prerelax):
            if not HAS_ADSORML:
                st.warning(f"CHGNet slab pre-relax unavailable: {ADSORML_IMPORT_ERR}")
            with st.expander("Constrained CHGNet parameters", expanded=False):
                her_constrained_top_free_layers = st.number_input(
                    "Free top z-layers", min_value=1, max_value=4, value=int(her_constrained_top_free_layers), step=1, key="her_constrained_top_free_layers"
                )
                her_constrained_layer_tol = st.number_input(
                    "z-layer clustering tolerance (Å)", min_value=0.3, max_value=1.5, value=float(her_constrained_layer_tol), step=0.1, key="her_constrained_layer_tol"
                )
                her_constrained_fmax = st.number_input(
                    "CHGNet relax fmax", min_value=0.01, max_value=0.20, value=float(her_constrained_fmax), step=0.01, key="her_constrained_fmax"
                )
                her_constrained_max_steps = st.number_input(
                    "CHGNet max steps", min_value=20, max_value=300, value=int(her_constrained_max_steps), step=10, key="her_constrained_max_steps"
                )
                her_constrained_seed_ui = st.number_input(
                    "Seed (0 = auto)", min_value=0, max_value=2**31-1, value=int(her_constrained_seed_ui), step=1, key="her_constrained_seed_ui"
                )
                st.caption("Recommended for oxide HER: use 1–2 free top layers and keep the lower slab fixed. This is a slab preconditioning step, not the final HER metric.")
            her_constrained_seed = None if int(her_constrained_seed_ui) == 0 else int(her_constrained_seed_ui)

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
    use_auto_sites = st.checkbox("Use auto-detected sites (recommended)", value=True, key="use_auto_sites")
    site_selection_method = st.selectbox(
        "Site selection method",
        ["Geometry (representative)", "ML screening (AdsorbML-lite)"],
        index=0,
        key="site_sel_method",
    )

    ml_enabled = site_selection_method.startswith("ML")
    rep_site_map = None

    if use_auto_sites and (not ml_enabled):
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
                elif is_orr:
                    preview_ads_options = orr_ads if orr_ads else ["OOH*", "O*", "OH*"]
                else:
                    preview_ads_options = co2_ads if co2_ads else ["COOH*", "CO*"]
                preview_ads = st.selectbox("Preview adsorbate", preview_ads_options, index=0, key="preview_ads_geom")

                slabs_ads = []
                if is_her:
                    if mtype == "metal":
                        slabs_ads = generate_slab_ads_series(atoms_for_sites_eff, rep_sites, symbol="H", dz=0.0, mode="default")
                    else:
                        rep_sites = _project_oxide_her_sites_to_otop(atoms_for_sites_eff, rep_sites, dz=1.0, extra_z=0.0)
                        slabs_ads = generate_slab_ads_series(atoms_for_sites_eff, rep_sites, symbol="H", dz=0.0, mode="default")
                    export_ads_label = "H"
                else:
                    export_ads_label = preview_ads.replace("*", "")
                    for s in rep_sites:
                        slabs_ads.append(build_adsorbate_preview_slab(atoms_for_sites_eff, s, preview_ads, dz=1.8, ref_dir="ref_gas"))

                idx = st.selectbox("Select site to view", list(range(len(rep_sites))), index=0, key="geom_view_idx")
                show_atoms_3d(slabs_ads[idx], height=420, width=900, tag=f"geom_seed_{idx}")

                if st.button("Export preview CIFs (zip)", key="btn_export_previews"):
                    zip_buf = BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for i, (s, ads_slab) in enumerate(zip(rep_sites, slabs_ads)):
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
                key = _make_ml_screen_key(
                    sig,
                    mtype,
                    reaction_mode,
                    co2_ads,
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
                else:
                    cand_sites = generate_candidate_sites(
                        atoms_for_sites_screen,
                        mtype=mtype,
                        geom_per_kind=int(geom_per_kind),
                        n_random=int(n_random),
                        rng_seed=GLOBAL_SEED,
                        random_kind="fcc",
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
                        _active_ads = orr_ads if is_orr else co2_ads
                        if not _active_ads:
                            _ads_label = "ORR" if is_orr else "CO2RR"
                            st.error(f"Select at least one {_ads_label} intermediate (sidebar).")
                            return False
                        by_ads, raw_by_ads, stats_by_ads = screen_sites_adsorbml_lite(
                            atoms_for_sites_screen,
                            cand_sites,
                            reaction="ORR" if is_orr else "CO2RR",
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
                            if is_orr:
                                ads0 = (orr_ads[0] if orr_ads else "OOH*")
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

    st.markdown("### Electrochemical conditions")
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

    # HER thermochemistry mode
    thermo_mode = "CHE correction (fast screening)"
    zpe_target_mode = "Best-ranked by CHE"
    zpe_target_label = None
    local_zpe_cutoff = 2.5
    local_zpe_max_neighbors = 3

    if is_her:
        st.markdown("### HER thermochemistry")
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

        ctm1, ctm2, ctm3 = st.columns(3)
        with ctm1:
            zpe_target_mode = st.selectbox(
                "Selected structure rule",
                ["Best-ranked by CHE", "User-selected site label"],
                index=0,
                key="zpe_target_mode",
                disabled=(thermo_mode != "Local ZPE correction (selected structure)"),
            )
        with ctm2:
            local_zpe_cutoff = st.number_input(
                "Local ZPE neighbor cutoff (Å)",
                min_value=1.5,
                max_value=4.0,
                value=2.5,
                step=0.1,
                key="local_zpe_cutoff",
                disabled=(thermo_mode == "CHE correction (fast screening)"),
            )
        with ctm3:
            local_zpe_max_neighbors = st.number_input(
                "Max local neighbors",
                min_value=1,
                max_value=8,
                value=3,
                step=1,
                key="local_zpe_max_neighbors",
                disabled=(thermo_mode == "CHE correction (fast screening)"),
            )

        if thermo_mode == "Local ZPE correction (selected structure)" and zpe_target_mode == "User-selected site label":
            zpe_target_label = st.text_input(
                "Target site label",
                value="",
                key="zpe_target_label",
                help="Example: ontop_0, bridge_1",
            ).strip() or None


    # Oxide HER descriptor mode
    oxide_descriptor_mode = "Basic HER screening"
    oxide_descriptor_max_reactive_per_kind = 2
    oxide_descriptor_pair_limit = 6
    if is_her and (mtype == "oxide"):
        with st.expander("Oxide HER descriptor mode", expanded=False):
            oxide_descriptor_mode = st.selectbox(
                "Descriptor mode",
                [
                    "Basic HER screening",
                    "D1_OH only (O-top protonation)",
                    "D2_Hreact only (reactive H state)",
                    "D3_pair only (H2 pairing proxy)",
                    "Full 3-stage profile (experimental)",
                ],
                index=0,
                key="oxide_descriptor_mode",
                help=(
                    "Basic HER screening keeps the legacy oxide HER workflow. "
                    "D1 computes only the O-top protonation descriptor. "
                    "D2 computes only the reactive-H-state descriptor. "
                    "D3 computes only the H2 pairing proxy (with internal precursor generation). "
                    "Full 3-stage profile computes D1, D2, and D3 together."
                ),
            )
            needs_reactive = oxide_descriptor_mode in {
                "D2_Hreact only (reactive H state)",
                "D3_pair only (H2 pairing proxy)",
                "Full 3-stage profile (experimental)",
            }
            needs_pair = oxide_descriptor_mode in {
                "D3_pair only (H2 pairing proxy)",
                "Full 3-stage profile (experimental)",
            }
            c3a, c3b = st.columns(2)
            with c3a:
                oxide_descriptor_max_reactive_per_kind = st.number_input(
                    "Reactive-H seeds per kind",
                    min_value=1, max_value=4, value=2, step=1,
                    key="oxide_descriptor_max_reactive_per_kind",
                    disabled=(not needs_reactive),
                )
            with c3b:
                oxide_descriptor_pair_limit = st.number_input(
                    "Pairing seed limit",
                    min_value=2, max_value=12, value=6, step=1,
                    key="oxide_descriptor_pair_limit",
                    disabled=(not needs_pair),
                )
            if oxide_descriptor_mode in {
                "D3_pair only (H2 pairing proxy)",
                "Full 3-stage profile (experimental)",
            }:
                st.caption("The H₂ pairing stage is treated as an approximate release proxy rather than an explicit barrier.")

    if (not is_her):
        st.caption("Note: U/pH correction is applied only for HER. CO₂RR is reported as descriptor ΔG_ads. ORR additionally writes a step-wise summary file (results_orr_summary.csv) using the entered U for one-electron CHE shifts.")


    # Defaults (so variables exist regardless of mode)
    co2rr_her_guardrail = False
    her_site_pref = "ontop"
    her_use_net_corr = True

    # Optional companion HER guardrail for CO2RR (single-site, capped steps)
    if (not is_her):
        st.markdown("### Optional companion: HER guardrail")
        cHG1, cHG2, cHG3 = st.columns([1.6, 1.0, 1.0])
        with cHG1:
            co2rr_her_guardrail = st.checkbox(
                "Also compute HER guardrail (single site, capped steps)",
                value=True,
                help="Adds one H* relaxation on a preferred site. Designed to avoid compute blow-up.",
                key="co2rr_her_guardrail",
            )
        with cHG2:
            her_site_pref = st.selectbox(
                "HER site preference",
                ["ontop", "bridge", "fcc", "hcp"],
                index=0,
                disabled=(not co2rr_her_guardrail),
                key="co2rr_her_site_pref",
            )
        with cHG3:
            her_use_net_corr = st.checkbox(
                "Apply NET correction (0.24 eV)",
                value=True,
                disabled=(not co2rr_her_guardrail),
                key="co2rr_her_use_net_corr",
            )

    # Auto-selected sites for calculation (applies to both HER/CO2RR)
    use_auto_sites_for_calc = st.checkbox(
        "Use auto-selected sites for calculation",
        value=True,
        key="use_auto_sites_for_calc",
    )

    final_user_sites = None
    if use_auto_sites_for_calc:
        if st.session_state.get("ml_union_site_map") is not None:
            st.info("Auto sites source: ML screening union-sites (from Step 3).")
            final_user_sites = st.session_state.get("ml_union_site_map")
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

    if st.button("Run Calculation", type="primary", key="btn_run_calc"):
        if atoms_for_calc is None:
            st.error("No prepared structure available.")
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
                    )
            elif is_orr:
                # ── ORR branch ─────────────────────────────────────────
                _orr_adspecies = tuple(orr_ads) if orr_ads else ("OOH*", "O*", "OH*")
                _orr_U = float(st.session_state.get("orr_U_input", 0.0))
                if mtype == "metal":
                    csv_path, meta = run_metal_orr_che(
                        str(slab_path),
                        sites=manual_sites,
                        relax_mode=relax_mode,
                        user_ads_sites=final_user_sites if final_user_sites else None,
                        adspecies=_orr_adspecies,
                        orr_u=_orr_U,
                    )
                else:
                    csv_path, meta = run_oxide_orr_che(
                        str(slab_path),
                        sites=manual_sites,
                        relax_mode=relax_mode,
                        user_ads_sites=final_user_sites if final_user_sites else None,
                        adspecies=_orr_adspecies,
                        orr_u=_orr_U,
                    )
            else:
                # ── CO2RR branch ───────────────────────────────────────
                if not co2_ads:
                    co2_ads = ["COOH*", "CO*"]
                adspecies = tuple(co2_ads)

                if mtype == "metal":
                    csv_path, meta = run_metal_co2rr_che(
                        str(slab_path),
                        sites=manual_sites,
                        relax_mode=relax_mode,
                        user_ads_sites=final_user_sites if final_user_sites else None,
                        adspecies=adspecies,
                        her_guardrail=bool(co2rr_her_guardrail),
                        her_site_preference=str(her_site_pref),
                        her_use_net_corr=bool(her_use_net_corr),
                    )
                else:
                    csv_path, meta = run_oxide_co2rr_che(
                        str(slab_path),
                        sites=manual_sites,
                        relax_mode=relax_mode,
                        user_ads_sites=final_user_sites if final_user_sites else None,
                        adspecies=adspecies,
                        her_guardrail=bool(co2rr_her_guardrail),
                        her_site_preference=str(her_site_pref),
                        her_use_net_corr=bool(her_use_net_corr),
                    )

        st.success("Calculation Complete!")

        df = pd.read_csv(csv_path)
        # annotate surfactant scenario into the result table
        if isinstance(df, pd.DataFrame) and (not df.empty):
            df["surfactant_class"] = str(surfactant_class)
            df["surfactant_chgnet_prerelax_slab"] = bool(surfactant_prerelax_slab)

        mode_label = "HER" if is_her else ("ORR" if is_orr else "CO2RR")

        if is_her and "ΔG_H (eV)" in df.columns:
            df["ΔG_H(U,pH) (eV)"] = df["ΔG_H (eV)"] - float(U_input) - R_PH * float(pH_input)


        if is_her:
            # HER: keep legacy reliability split, but also annotate site transitions for UI/debugging.
            df = annotate_site_transitions(df, disp_thresh=CO2RR_MIGRATION_DISP_THRESH_A)
            df_rel, df_unrel = split_reliable_unreliable(df)
            df["reliability"] = "unreliable"
            if df_rel is not None:
                df.loc[df_rel.index, "reliability"] = "reliable"
            migration_summary = summarize_site_transitions(df)
        else:
            # CO2RR / ORR: QA-driven policy (migrated is NOT an auto-reject)
            df = co2rr_apply_qa_policy(df, disp_thresh=CO2RR_MIGRATION_DISP_THRESH_A)
            df = annotate_site_transitions(df, disp_thresh=CO2RR_MIGRATION_DISP_THRESH_A)
            df_keep, df_reject = co2rr_split_by_qa(df)

            # Set reliability consistent with QA policy
            df["reliability"] = "unreliable"
            df.loc[df_keep.index, "reliability"] = "reliable"

            # Backwards-compatible names for downstream UI blocks
            df_rel, df_unrel = df_keep, df_reject
            migration_summary = summarize_site_transitions(df)

        # Persist results for rendering even after rerun (e.g., toggling UI options)
        if isinstance(meta, dict):
            meta = dict(meta)
            meta["SURFACTANT_CLASS"] = str(surfactant_class)
            meta["SURFACTANT_CHGNET_PRERELAX_SLAB"] = bool(surfactant_prerelax_slab)
            if migration_summary is not None:
                meta["MIGRATION_SUMMARY"] = migration_summary

        st.session_state["last_run"] = {
            "is_her": bool(is_her),
            "mtype": str(mtype),
            "reaction_mode": str(reaction_mode),
            "mode_label": str(mode_label),
            "csv_path": str(csv_path),
            "meta": meta,
            "df": df,
            "df_rel": df_rel,
            "df_unrel": df_unrel,
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
    meta = last_run.get("meta") or {}
    mode_label = last_run.get("mode_label", "HER" if last_run.get("is_her") else "CO2RR")
    U_disp = float(last_run.get("U_input", 0.0))
    pH_disp = float(last_run.get("pH_input", 0.0))

    # --- Run history notes (session-only) ---
    try:
        with st.expander("Run history (selected)", expanded=False):
            rh.render_selected_run_details()
    except Exception:
        pass

    # --- Optional: HER guardrail summary (CO2RR companion) ---
    her_guard = None
    try:
        her_guard = meta.get("HER_GUARDRAIL") if isinstance(meta, dict) else None
    except Exception:
        her_guard = None

    if her_guard is not None and isinstance(her_guard, dict):
        st.markdown("### HER guardrail (single-site)")
        try:
            dG = float(her_guard.get("ΔG_H (eV)", np.nan))
        except Exception:
            dG = np.nan

        dG_uph = dG - U_disp - R_PH * pH_disp if np.isfinite(dG) else np.nan

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Site", f"{her_guard.get('site','?')} / {her_guard.get('site_label','?')}")
        g2.metric("ΔG_H (eV)", f"{dG:.3f}" if np.isfinite(dG) else "n/a")
        g3.metric("ΔG_H(U,pH) (eV)", f"{dG_uph:.3f}" if np.isfinite(dG_uph) else "n/a")
        try:
            disp_val = float(her_guard.get("H_lateral_disp(Å)", np.nan))
        except Exception:
            disp_val = np.nan
        g4.metric("H lateral disp (Å)", f"{disp_val:.2f}" if np.isfinite(disp_val) else "n/a")

        # download HER guardrail row
        try:
            hg_df = pd.DataFrame([her_guard])
            st.download_button(
                "Download HER guardrail CSV",
                hg_df.to_csv(index=False).encode("utf-8"),
                "her_guardrail.csv",
                "text/csv",
                key="dl_her_guardrail_csv",
            )
        except Exception:
            pass

        # optional structure preview if cif exists (best-effort)
        try:
            cifp = her_guard.get("structure_cif", None)
            if cifp and Path(str(cifp)).is_file():
                with st.expander("Preview HER guardrail structure", expanded=False):
                    at_hg = read(str(cifp))
                    show_atoms_3d(at_hg, height=420, width=900, tag="her_guardrail")
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
                st.warning(f"{n_blow} CO2RR points show |ΔE_ads_user| > 50 eV (likely bad placement/unstable relax).")
        except Exception:
            pass

    # --- Main results tables (ALWAYS) ---
    if bool(last_run.get("is_her")):
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

        df_dedup = co2rr_dedupe_candidates(df_keep) if isinstance(df_keep, pd.DataFrame) else pd.DataFrame()

        qa_counts = df["qa"].value_counts(dropna=False) if (isinstance(df, pd.DataFrame) and ("qa" in df.columns)) else pd.Series(dtype=int)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total runs", int(len(df)) if isinstance(df, pd.DataFrame) else 0)
        c2.metric("Candidates (qa=ok/migrated)", int(len(df_keep)) if isinstance(df_keep, pd.DataFrame) else 0)
        c3.metric("Rejected (qa!=ok/migrated)", int(len(df_reject)) if isinstance(df_reject, pd.DataFrame) else 0)
        c4.metric("Unique minima (dedup)", int(len(df_dedup)) if isinstance(df_dedup, pd.DataFrame) else 0)

        st.markdown("### Candidates (Deduplicated by relaxed minimum)")
        if isinstance(df_dedup, pd.DataFrame):
            st.dataframe(build_compact_table(df_dedup, mode_label), use_container_width=True)

        with st.expander("Show all candidate attempts (including duplicates)", expanded=False):
            if isinstance(df_keep, pd.DataFrame):
                st.dataframe(build_compact_table(df_keep, mode_label), use_container_width=True)

        if isinstance(df_reject, pd.DataFrame) and (not df_reject.empty):
            with st.expander("Show rejected attempts (qa-based)", expanded=False):
                st.dataframe(build_compact_table(df_reject, mode_label), use_container_width=True)

        if qa_counts is not None and (not qa_counts.empty):
            with st.expander("QA breakdown", expanded=False):
                st.dataframe(qa_counts.rename_axis("qa").reset_index(name="count"), use_container_width=True)

        if isinstance(meta, dict) and meta.get("MIGRATION_SUMMARY"):
            mig_summary = meta.get("MIGRATION_SUMMARY") or {}
            with st.expander("Migration metadata", expanded=False):
                st.write(f"- migrated rows: **{int(mig_summary.get('n_migrated', 0))}**")
                mig_paths = mig_summary.get("paths") or []
                if mig_paths:
                    st.dataframe(pd.DataFrame(mig_paths), use_container_width=True)
                elif isinstance(df, pd.DataFrame) and ("migration_path" in df.columns):
                    cols = [c for c in ["adsorbate", "site_label", "requested_site", "initial_geom_site", "relaxed_site", "placement_mismatch", "migrated_actual", "migration_destination", "migration_path", "actual_migration_path", "site_transition_type", "ΔG_ads (eV)", "ΔE_ads_user (eV)", "ads_lateral_disp(Å)", "qa", "migrated"] if c in df.columns]
                    st.dataframe(df[cols], use_container_width=True)

        cdl1, cdl2 = st.columns(2)
        with cdl1:
            st.download_button(
                "Download candidates (dedup) CSV",
                df_dedup.to_csv(index=False).encode("utf-8") if isinstance(df_dedup, pd.DataFrame) else b"",
                "co2rr_candidates_dedup.csv",
                "text/csv",
                key="dl_co2rr_candidates_dedup",
            )
        with cdl2:
            st.download_button(
                "Download candidates (all) CSV",
                df_keep.to_csv(index=False).encode("utf-8") if isinstance(df_keep, pd.DataFrame) else b"",
                "co2rr_candidates_all.csv",
                "text/csv",
                key="dl_co2rr_candidates_all",
            )

    # --- Oxide HER descriptor profile / summary ---
    if bool(last_run.get("is_her")) and str(last_run.get("mtype", "")) == "oxide" and isinstance(meta, dict) and meta.get("OXIDE_DESCRIPTOR_SUMMARY"):
        _ods = meta.get("OXIDE_DESCRIPTOR_SUMMARY") or {}
        _mode = str(meta.get("OXIDE_DESCRIPTOR_MODE", _ods.get("descriptor_mode", "Basic HER screening")))
        st.markdown("### Oxide HER descriptor summary")
        if _mode in {"D3_pair only (H2 pairing proxy)", "Full 3-stage profile (experimental)"}:
            st.warning(str(meta.get("OXIDE_DESCRIPTOR_CAUTION", _ods.get("caution", "The H₂ pairing stage is an approximate release proxy rather than an explicit barrier."))))

        summary_cols = [
            "descriptor_mode",
            "D1_OH (eV)", "D2_Hreact (eV)", "D3_pair_proxy (eV)",
            "Δ12 (eV)", "Δ23 (eV)", "classification",
            "D3_H2_like_motif", "D3_final_HH_distance(Å)",
            "D1_site_label", "D2_site_label", "D3_pair_label",
        ]
        _summary_df = pd.DataFrame([{k: _ods.get(k, np.nan) for k in summary_cols}])
        st.dataframe(_summary_df, use_container_width=True)

        _profile_points = []
        _d1 = _safe_float(_ods.get("D1_OH (eV)"))
        _d2 = _safe_float(_ods.get("D2_Hreact (eV)"))
        _d3 = _safe_float(_ods.get("D3_pair_proxy (eV)"))
        if np.isfinite(_d1):
            _profile_points.append({"Stage": "O–H formation", "Energy (eV)": _d1})
        if np.isfinite(_d2):
            _profile_points.append({"Stage": "Reactive H state", "Energy (eV)": _d2})
        if np.isfinite(_d3):
            _profile_points.append({"Stage": "H₂ pairing proxy", "Energy (eV)": _d3})
        if _profile_points:
            _profile_df = pd.DataFrame(_profile_points)
            st.line_chart(_profile_df.set_index("Stage"))

        with st.expander("Show descriptor candidate tables", expanded=False):
            _d2_csv = _ods.get("D2_candidates_csv") or meta.get("OXIDE_DESCRIPTOR_D2_CANDIDATES_CSV", "")
            _d3_csv = _ods.get("D3_candidates_csv") or meta.get("OXIDE_DESCRIPTOR_D3_CANDIDATES_CSV", "")
            if _d2_csv and Path(str(_d2_csv)).is_file():
                st.markdown("#### D2 reactive-H candidates")
                try:
                    st.dataframe(pd.read_csv(str(_d2_csv)), use_container_width=True)
                except Exception as _e:
                    st.info(f"Could not load D2 candidate table: {_e}")
            if _d3_csv and Path(str(_d3_csv)).is_file():
                st.markdown("#### D3 H₂ pairing proxy candidates")
                try:
                    st.dataframe(pd.read_csv(str(_d3_csv)), use_container_width=True)
                except Exception as _e:
                    st.info(f"Could not load D3 candidate table: {_e}")

    # --- Relaxed post-run structure viewer ---
    viewer_frames = {}
    df_dedup = locals().get("df_dedup", pd.DataFrame())
    df_keep = locals().get("df_keep", pd.DataFrame())
    df_reject = locals().get("df_reject", pd.DataFrame())

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
                viewer_path = _resolve_relaxed_structure_path(viewer_row, csv_path=last_run.get("csv_path"))

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("site_label", str(viewer_row.get("site_label", "?")))
                m2.metric("initial", str(viewer_row.get("initial_site_kind", viewer_row.get("requested_site", "?"))))
                m3.metric("final", str(viewer_row.get("final_site_kind", viewer_row.get("relaxed_site", "?"))))
                disp_view = _safe_float(viewer_row.get("H_lateral_disp(Å)", viewer_row.get("ads_lateral_disp(Å)", np.nan)))
                m4.metric("disp (Å)", f"{disp_view:.2f}" if np.isfinite(disp_view) else "n/a")

                extra_cols = [c for c in [
                    "migrated", "reliability", "qa", "ΔG_H(U,pH) (eV)", "ΔG_H (eV)",
                    "ΔG_H_CHE (eV)", "ΔG_H_local (eV)", "local_thermo_corr (eV)",
                    "ΔG_ads (eV)", "ΔE_H_user (eV)", "ΔE_ads_user (eV)",
                    "zpe_scope", "zpe_selected_atoms", "zpe_warning"
                ] if c in viewer_row.index]
                if extra_cols:
                    st.dataframe(pd.DataFrame([viewer_row[extra_cols].to_dict()]), use_container_width=True)

                if viewer_path is not None and Path(viewer_path).is_file():
                    try:
                        at_view = read(str(viewer_path))
                        show_atoms_3d(at_view, height=460, width=900, tag=f"relaxed_view_{viewer_source}_{viewer_idx}")
                        st.download_button(
                            "Download selected relaxed CIF",
                            Path(viewer_path).read_bytes(),
                            Path(viewer_path).name,
                            "chemical/x-cif",
                            key=f"dl_relaxed_cif_{viewer_source}_{viewer_idx}",
                        )
                    except Exception as e:
                        st.warning(f"Could not render relaxed CIF: {e}")
                else:
                    st.info("Relaxed CIF path could not be resolved for this row. If needed, add 'structure_cif' to the exported result rows in CHE_mode.")

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
