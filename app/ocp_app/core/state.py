import os
import re
from collections import Counter

import numpy as np
import streamlit as st

_NON_METALS = {
    "H", "He",
    "B", "C", "N", "O", "F", "Ne",
    "Si", "P", "S", "Cl", "Ar",
    "Br", "I",
}

def _init_state():
    defaults = {
        # Pipeline structures
        "atoms_loaded": None,          # Step 1 output (uploaded or MP bulk)
        "atoms_tuned": None,           # optional surface composition tuning
        "ratio_tune_meta": None,
        "atoms_prepared": None,        # Step 2 output
        "prepared_source_sig": None,
        "prepared_history": [],

        # Mode tracking
        "_slab_source_mode_prev": None,
        "_upload_sig": None,

        # Credentials / LLM
        "mp_api_key": os.environ.get("MP_API_KEY", ""),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "llm_model": "gpt-4o-mini",
        "llm_enabled": True,
        "llm_last_out": None,
        "loaded_mp_id": None,


        # Slabify candidates (expert)
        "slabify_candidates_atoms": None,
        "slabify_candidates_meta": None,

        # Interface builder candidates (basic/expert)
        "interface_candidates_atoms": None,
        "interface_candidates_meta": None,

        # ML cache
        "ml_screen_key": None,
        "ml_union_site_map": None,
        "ml_union_struct_map": None,
        "ml_union_items": None,
        "ml_compact_df": None,
        "ml_debug_df": None,
        "ml_debug_stats": None,

        # Tuning defaults tracking
        # Last run outputs (for persistent rendering)
        "last_run": None,
        "_tune_defaults_sig": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _clear_ml_cache():
    for k in [
        "ml_screen_key",
        "ml_union_site_map",
        "ml_union_struct_map",
        "ml_union_items",
        "ml_compact_df",
        "ml_debug_df",
        "ml_debug_stats",
    ]:
        st.session_state[k] = None

def _atoms_signature(atoms) -> tuple:
    try:
        cell = atoms.get_cell()
        lengths = tuple(np.round(cell.lengths(), 4).tolist())
        angles = tuple(np.round(cell.angles(), 3).tolist())
    except Exception:
        lengths = ("?", "?", "?")
        angles = ("?", "?", "?")
    return (len(atoms), atoms.get_chemical_formula(), lengths, angles)

def _reset_prepared_from_working():
    working = st.session_state.get("atoms_tuned") or st.session_state.get("atoms_loaded")
    if working is None:
        st.session_state["atoms_prepared"] = None
        st.session_state["prepared_source_sig"] = None
        st.session_state["prepared_history"] = []
        _clear_ml_cache()
        st.session_state["slabify_candidates_atoms"] = None
        st.session_state["slabify_candidates_meta"] = None
        st.session_state["interface_candidates_atoms"] = None
        st.session_state["interface_candidates_meta"] = None
        return

    sig = _atoms_signature(working)
    st.session_state["atoms_prepared"] = working.copy()
    st.session_state["prepared_source_sig"] = sig
    st.session_state["prepared_history"] = [{"action": "reset_from_working", "sig": sig}]
    _clear_ml_cache()
    st.session_state["slabify_candidates_atoms"] = None
    st.session_state["slabify_candidates_meta"] = None
    st.session_state["interface_candidates_atoms"] = None
    st.session_state["interface_candidates_meta"] = None

def _ensure_prepared_uptodate():
    working = st.session_state.get("atoms_tuned") or st.session_state.get("atoms_loaded")
    if working is None:
        return
    sig = _atoms_signature(working)
    if st.session_state.get("atoms_prepared") is None or st.session_state.get("prepared_source_sig") != sig:
        _reset_prepared_from_working()

def _push_prepared_update(atoms_new, action: str, meta: dict | None = None):
    st.session_state["atoms_prepared"] = atoms_new
    st.session_state["prepared_history"] = (st.session_state.get("prepared_history") or []) + [{
        "action": action,
        **(meta or {}),
    }]
    _clear_ml_cache()
    st.session_state["slabify_candidates_atoms"] = None
    st.session_state["slabify_candidates_meta"] = None
    st.session_state["interface_candidates_atoms"] = None
    st.session_state["interface_candidates_meta"] = None

def _jsonable(x):
    if isinstance(x, (np.integer, np.int64, np.int32)):
        return int(x)
    if isinstance(x, (np.floating, np.float64, np.float32)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return x

def normalize_mp_id(raw: str) -> str:
    """Normalize MP-ID variants ('19009'/'mp19009'/'mp-19009') to 'mp-19009'."""
    s = (raw or "").strip()
    if not s:
        return ""
    m = re.match(r"^(?:mp[-_]?){0,1}(\d+)$", s, flags=re.IGNORECASE)
    if m:
        return f"mp-{m.group(1)}"
    return s

def infer_default_tune_elements(atoms) -> tuple[str, str]:
    """Element1 = most abundant metal, Element2 = second most (default 'Cu')"""
    syms = [s for s in atoms.get_chemical_symbols() if s not in ("O", "H")]
    if not syms:
        return ("Ni", "Cu")

    counts = Counter(syms)
    metal_counts = Counter({el: n for el, n in counts.items() if el not in _NON_METALS})
    if not metal_counts:
        metal_counts = counts  # fallback

    el1 = metal_counts.most_common(1)[0][0]
    el2 = next((el for el, _ in metal_counts.most_common() if el != el1), "Cu")
    return (str(el1), str(el2))

def ensure_tune_defaults_from_structure(atoms):
    """Auto-update default element selections when the structure changes"""
    sig = _atoms_signature(atoms)
    if st.session_state.get("_tune_defaults_sig") != sig:
        el1, el2 = infer_default_tune_elements(atoms)
        st.session_state["_tune_defaults_sig"] = sig

        # Sync basic/expert defaults
        st.session_state["rt_el1_basic"] = el1
        st.session_state["rt_el2_basic"] = el2
        st.session_state["rt_el1_adv"] = el1
        st.session_state["rt_el2_adv"] = el2

