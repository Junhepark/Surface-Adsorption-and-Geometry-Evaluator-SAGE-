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
from ase.io import read, write
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
except Exception as e:
    HAS_SLABIFY = False
    SLABIFY_IMPORT_ERR = str(e)

# ---------------- App config ----------------
st.set_page_config(page_title="SAGE App (HAPLAB)", layout="wide")
st.title("Surface Adsorption and Geometry Evaluator — HER / CO₂RR / ORR (HAPLAB v1.0)")

R_PH = 0.0591  # eV per pH
GLOBAL_SEED = 42
RATIO_SUM = 10

CO2RR_MIGRATION_DISP_THRESH_A = 0.8  # Å; adsorbate lateral displacement threshold to flag migration

# Approximate API pricing (USD per 1M tokens). Used ONLY for rough in-app cost estimates.
# Update as needed if OpenAI pricing changes.
PRICING_USD_PER_1M = {
    # gpt-4o mini (standard)
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # gpt-5 / gpt-5.2 (standard)
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
}

def _estimate_cost_usd(model_name: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Rough cost estimate for a single call (standard processing)."""
    if input_tokens is None or output_tokens is None:
        return None
    if not model_name:
        return None
    m = model_name.lower()
    rate = None
    # Match by prefix for dated model names
    for prefix, r in PRICING_USD_PER_1M.items():
        if m.startswith(prefix):
            rate = r
            break
    if rate is None:
        return None
    return (float(input_tokens) * float(rate["input"]) + float(output_tokens) * float(rate["output"])) / 1_000_000.0


ADS_TEMPLATE_FILES = {
    "CO":   "CO_box.cif",
    "COOH": "COOH_box.cif",
    "HCOO": "HCOO_box.cif",
    "OCHO": "OCHO_box.cif",
    # ORR intermediates
    "O":    "O_box.cif",
    "OH":   "OH_box.cif",
    "OOH":  "OOH_box.cif",
}

# ---------------- Session State ----------------
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


def _cluster_z_layers(z_vals: np.ndarray, tol: float = 0.35):
    """Cluster z-values into layers. Returns list of (z_center, indices_in_original_array)."""
    if z_vals.size == 0:
        return []
    order = np.argsort(z_vals)
    z_sorted = z_vals[order]
    clusters = []
    start = 0
    for i in range(1, len(z_sorted)):
        if (z_sorted[i] - z_sorted[i - 1]) > tol:
            clusters.append(order[start:i])
            start = i
    clusters.append(order[start:len(z_sorted)])
    out = []
    for idxs in clusters:
        zc = float(np.mean(z_vals[idxs]))
        out.append((zc, idxs))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def _suggest_conditioning_params(atoms, *, mtype: str, surfactant_class: str, profile: str = "safe"):
    """Heuristic auto-tuning for CHGNet slab conditioning parameters."""
    mtype = (mtype or "").lower()
    cls = (surfactant_class or "none").lower()
    prof = (profile or "safe").lower()

    pos = atoms.get_positions()
    z = pos[:, 2]
    zmax = float(np.max(z))
    top_region = (z > (zmax - 8.0))
    z_top = z[top_region]
    layers = _cluster_z_layers(z_top, tol=0.35)

    dz = 2.0
    if len(layers) >= 2:
        dz = max(0.8, float(layers[0][0] - layers[1][0]))

    top_z_tol = float(np.clip(dz + 1.2, 2.0, 4.5))

    def _n_layers_in_window(win):
        zwin = z_top[z_top > (float(np.max(z_top)) - win)]
        return len(_cluster_z_layers(zwin, tol=0.35))

    if _n_layers_in_window(top_z_tol) < 2:
        top_z_tol = float(np.clip(top_z_tol + dz, 2.0, 5.0))

    is_oxide_like = (mtype == "oxide") or ("O" in set(atoms.get_chemical_symbols()) and len(set(atoms.get_chemical_symbols())) >= 2)
    if is_oxide_like:
        idx_win = np.where(z > (zmax - top_z_tol))[0]
        syms = set(atoms.get_chemical_symbols()[i] for i in idx_win)
        has_o = ("O" in syms)
        has_non_o = any(s != "O" for s in syms)
        if not (has_o and has_non_o):
            top_z_tol = float(np.clip(top_z_tol + dz, 2.0, 5.0))

    if prof.startswith("explore"):
        base = 0.06 if is_oxide_like else 0.05
        jiggle_amp = base + 0.02 * max(0.0, (top_z_tol - 2.5))
        if cls == "nonionic":
            jiggle_amp += 0.005
        jiggle_amp = float(np.clip(jiggle_amp, 0.04, 0.12))
        max_steps = 400
    else:
        jiggle_amp = 0.05 if is_oxide_like else 0.04
        max_steps = 250

    fmax = 0.05
    rationale = f"auto(profile={prof}, dz≈{dz:.2f} Å, top_window={top_z_tol:.2f} Å, jiggle={jiggle_amp:.2f} Å)"
    return {"top_z_tol": float(top_z_tol), "jiggle_amp": float(jiggle_amp), "fmax": float(fmax), "max_steps": int(max_steps), "rationale": rationale}


def _get_conditioned_slab(atoms, *, is_her: bool, surfactant_class: str, enable: bool, top_z_tol: float = 2.0, jiggle_amp: float = 0.05, fmax: float = 0.05, max_steps: int = 200, seed: Optional[int] = None):
    """Return a CHGNet-conditioned (slab-only) structure for the given surfactant scenario.

    This is a *scenario proxy* for interfacial conditioning: it does not model explicit surfactant,
    solvent, EDL, or potential. It is used to perturb/relax the slab into nearby surface states
    and then evaluate adsorption energetics downstream with the OCP model.

    Caching: keyed by a lightweight structure signature + surfactant_class.
    """
    try:
        cls = str(surfactant_class or "none").lower()
    except Exception:
        cls = "none"

    if bool(is_her) or (not bool(enable)) or (cls in ("none", "", "null")):
        return atoms, None

    if not HAS_ADSORML:
        return atoms, None

    sig = _atoms_signature(atoms)
    cache = st.session_state.setdefault("slab_condition_cache", {})
    key = (sig, cls, float(top_z_tol), float(jiggle_amp), float(fmax), int(max_steps), int(seed) if seed is not None else None)

    hit = cache.get(key)
    if isinstance(hit, dict) and ("atoms" in hit):
        return hit["atoms"], hit.get("meta")

    # Compute and cache
    atoms2, meta = relax_slab_chgnet(atoms, surfactant_class=cls, top_z_tol=float(top_z_tol), jiggle_amp=float(jiggle_amp), seed=seed, fmax=float(fmax), max_steps=int(max_steps), device="auto")
    atoms2 = _recenter_slab_z_into_cell(atoms2, margin=1.0)
    cache[key] = {"atoms": atoms2, "meta": meta}
    return atoms2, meta


def _reset_prepared_from_working():
    working = st.session_state.get("atoms_tuned") or st.session_state.get("atoms_loaded")
    if working is None:
        st.session_state["atoms_prepared"] = None
        st.session_state["prepared_source_sig"] = None
        st.session_state["prepared_history"] = []
        _clear_ml_cache()
        st.session_state["slabify_candidates_atoms"] = None
        st.session_state["slabify_candidates_meta"] = None
        return

    sig = _atoms_signature(working)
    st.session_state["atoms_prepared"] = working.copy()
    st.session_state["prepared_source_sig"] = sig
    st.session_state["prepared_history"] = [{"action": "reset_from_working", "sig": sig}]
    _clear_ml_cache()
    st.session_state["slabify_candidates_atoms"] = None
    st.session_state["slabify_candidates_meta"] = None

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



def _oxide_o_based_ads_position_compat(atoms, site, dz: float = 1.0, extra_z: float = 0.0):
    """Call _oxide_o_based_ads_position with best-effort kwarg compatibility.

    Different ocp_app versions have used different parameter names:
      - dz_h_oh, dz_h, or dz
    This helper inspects the callable signature and only passes supported kwargs.

    Returns
    -------
    (x, y, z): tuple[float, float, float]
    """
    try:
        params = inspect.signature(_oxide_o_based_ads_position).parameters
    except Exception:
        params = {}

    kwargs = {}
    if "dz_h_oh" in params:
        kwargs["dz_h_oh"] = float(dz)
    elif "dz_h" in params:
        kwargs["dz_h"] = float(dz)
    elif "dz" in params:
        kwargs["dz"] = float(dz)

    if "extra_z" in params:
        kwargs["extra_z"] = float(extra_z)

    out = _oxide_o_based_ads_position(atoms, site, **kwargs) if kwargs else _oxide_o_based_ads_position(atoms, site)
    # Be tolerant to older return formats
    try:
        x, y, z = out
    except Exception:
        raise RuntimeError(f"_oxide_o_based_ads_position returned unexpected value: {out!r}")
    return float(x), float(y), float(z)

def infer_oxide_family_from_atoms(atoms):
    symbols = [s for s in atoms.get_chemical_symbols() if s != "H"]
    counts = Counter(symbols)
    O = counts.pop("O", 0)
    if O == 0 or len(counts) == 0:
        return {"family": "unknown", "reduced_formula": None}

    nums = list(counts.values()) + [O]
    g = reduce(gcd, nums) if nums else 1
    if g == 0:
        g = 1

    red_counts = {el: n // g for el, n in counts.items()}
    O_red = O // g
    cation_sum = sum(red_counts.values())

    parts = [f"{el}{n if n > 1 else ''}" for el, n in sorted(red_counts.items())]
    if O_red > 0:
        parts.append(f"O{O_red if O_red > 1 else ''}")
    reduced_formula = "".join(parts)

    family = "unknown"
    if O_red == 1 and cation_sum == 1:
        family = "rocksalt_AO"
    elif O_red == 2 and cation_sum == 1:
        family = "rutile_AO2"
    elif O_red == 3 and cation_sum == 2:
        family = "perovskite_ABO3"
    elif O_red == 4 and cation_sum == 3:
        family = "spinel_AB2O4"

    return {"family": family, "reduced_formula": reduced_formula}

# ---- Convenience helpers (mp-id normalize, tuning defaults, min-distance panel) ----
_NON_METALS = {
    "H", "He",
    "B", "C", "N", "O", "F", "Ne",
    "Si", "P", "S", "Cl", "Ar",
    "Br", "I",
}

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

def _render_min_dist_panel(rep):
    """
    Based on validate_structure() results.
     Displays only OK/WARNING/CRITICAL levels (internal indices hidden).
    """
    ng = getattr(rep, "nearest_global", None) or {}
    nbp = getattr(rep, "nearest_by_pair", None) or []

    if (not ng) or (not nbp):
        st.info("Min interatomic distance: not available.")
        return

    g_pair = str(ng.get("pair", ""))
    g_dmin = ng.get("d_min", None)
    if g_dmin is None:
        st.info("Min interatomic distance: not available.")
        return

    g_flag = "ok"
    for p in nbp:
        if getattr(p, "pair", "") == g_pair:
            g_flag = str(getattr(p, "flag", "ok"))
            break

    level = g_flag.upper()
    line = (
        f"Min interatomic distance: **{float(g_dmin):.3f} Å** (**{g_pair}**)  \n"
        f"Level: **{level}** (source: validate_structure)"
    )

    if g_flag.lower().startswith("crit"):
        st.error(line)
    elif g_flag.lower().startswith("warn"):
        st.warning(line)
    else:
        st.success(line)

    rows = []
    for p in nbp:
        rows.append({
            "pair": getattr(p, "pair", ""),
            "d_min (Å)": float(getattr(p, "d_min", np.nan)),
            "d_mean (Å)": float(getattr(p, "d_mean", np.nan)),
            "n_bonds": int(getattr(p, "n_bonds", 0)),
            "level": str(getattr(p, "flag", "ok")).upper(),
        })
    df_pair = pd.DataFrame(rows).sort_values("d_min (Å)")

    with st.expander("Min distances by element pair (OK/WARNING/CRITICAL)", expanded=False):
        st.dataframe(df_pair, use_container_width=True)

def atoms_to_cif_string(atoms, symprec: float = 0.1) -> str:
    """Serialize ASE Atoms to CIF.

    Primary path: pymatgen CifWriter (keeps legacy behavior).
    Fallback path: ASE CIF writer to avoid spglib symmetry failures on slabs/relaxed structures.
    """
    try:
        struct = AseAtomsAdaptor.get_structure(atoms)
        return str(CifWriter(struct, symprec=symprec))
    except Exception:
        # Fallback: write P1 CIF via ASE (no symmetry search); use a temp file since ASE CIF writer expects a real file handle.
        fd, tmp_path = tempfile.mkstemp(prefix="ocpapp_", suffix=".cif")
        os.close(fd)
        try:
            write(tmp_path, atoms, format="cif")
            return Path(tmp_path).read_text(encoding="utf-8", errors="replace")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def atoms_to_cif_bytes(atoms, symprec: float = 0.1) -> bytes:
    return atoms_to_cif_string(atoms, symprec=symprec).encode("utf-8")

def atoms_to_xyz_string(atoms) -> str:
    buf = StringIO()
    write(buf, atoms, format="xyz")
    xyz_str = buf.getvalue()
    buf.close()
    return xyz_str


def _recenter_slab_z_into_cell(atoms, margin: float = 1.0):
    """Shift atoms along z so the slab sits well inside the unit cell (avoid z-wrapping artifacts).

    Only applies when the cell vector c is (approximately) aligned with z.
    Does not change the cell size.
    """
    a = atoms.copy()
    try:
        cell = a.get_cell()
        # Require c-vector ~ z-axis (orthorhombic/near-orthorhombic slabs)
        if abs(cell[2, 0]) > 1e-3 or abs(cell[2, 1]) > 1e-3:
            return a
        Lz = float(cell[2, 2])
        if not np.isfinite(Lz) or Lz <= 0:
            return a

        z = a.get_positions()[:, 2]
        zmin = float(np.min(z))
        zmax = float(np.max(z))
        zc = 0.5 * (zmin + zmax)

        # Center slab around Lz/2
        shift = (0.5 * Lz) - zc
        pos = a.get_positions()
        pos[:, 2] += shift
        a.set_positions(pos)

        # Enforce a margin to avoid any atoms slightly crossing boundaries after jiggle/relax.
        z = a.get_positions()[:, 2]
        if float(np.min(z)) < float(margin):
            pos = a.get_positions()
            pos[:, 2] += (float(margin) - float(np.min(z)))
            a.set_positions(pos)
        z = a.get_positions()[:, 2]
        if float(np.max(z)) > (Lz - float(margin)):
            pos = a.get_positions()
            pos[:, 2] -= (float(np.max(z)) - (Lz - float(margin)))
            a.set_positions(pos)
    except Exception:
        return a
    return a

def show_atoms_3d(atoms, height=420, width=700, tag="view"):
    atoms = atoms.copy()
    try:
        atoms.wrap()
    except Exception:
        pass

    xyz_str = atoms_to_xyz_string(atoms)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_str, "xyz")
    view.setStyle({"stick": {}, "sphere": {"radius": 0.3}})
    view.zoomTo()

    html = view._make_html()

    new_id = f"mol_{tag}_{uuid.uuid4().hex}"
    m = re.search(r'id="([^"]+)"', html)
    if m:
        old_id = m.group(1)
        html = html.replace(f'id="{old_id}"', f'id="{new_id}"', 1)
        html = html.replace(old_id, new_id)

    components.html(html, height=height, width=width)


def add_vacuum_z(atoms, total_vacuum_z: float = 30.0, keep_pbc_z: bool = True):
    a = atoms.copy()
    try:
        a.wrap()
    except Exception:
        pass
    pbc = list(a.get_pbc())
    if len(pbc) != 3:
        pbc = [True, True, True]
    pbc[2] = bool(keep_pbc_z)
    a.set_pbc(pbc)
    a.center(vacuum=float(total_vacuum_z) / 2.0, axis=2)
    return a

def set_pbc_z(atoms, pbc_z: bool):
    a = atoms.copy()
    pbc = list(a.get_pbc())
    if len(pbc) != 3:
        pbc = [True, True, True]
    pbc[2] = bool(pbc_z)
    a.set_pbc(pbc)
    return a

def repeat_xy(atoms, nx: int, ny: int):
    a = atoms.copy()
    return a.repeat((int(nx), int(ny), 1))

def slabify_from_bulk(atoms, miller=(1, 1, 1), min_slab_size=12.0, min_vacuum_size=30.0, max_candidates=6):
    if not HAS_SLABIFY:
        raise RuntimeError(f"pymatgen SlabGenerator not available: {SLABIFY_IMPORT_ERR}")

    struct = AseAtomsAdaptor.get_structure(atoms)
    gen = SlabGenerator(
        initial_structure=struct,
        miller_index=tuple(int(x) for x in miller),
        min_slab_size=float(min_slab_size),
        min_vacuum_size=float(min_vacuum_size),
        center_slab=True,
        in_unit_planes=True,
        primitive=True,
        reorient_lattice=True,
    )
    slabs = gen.get_slabs(symmetrize=False)[: int(max_candidates)]

    cand_atoms = []
    cand_meta = []
    for i, s in enumerate(slabs):
        a = AseAtomsAdaptor.get_atoms(s)
        a = set_pbc_z(a, True)
        rep = validate_structure(a, target_area=70.0)
        cand_atoms.append(a)
        cand_meta.append({
            "idx": i,
            "miller": tuple(miller),
            "n_atoms": len(a),
            "formula": a.get_chemical_formula(),
            "vacuum_z": float(getattr(rep, "vacuum_z", np.nan)),
            "recommend_repeat": getattr(rep, "recommend_repeat", None),
            "issues": getattr(rep, "issues", []),
        })
    return cand_atoms, cand_meta

def split_reliable_unreliable(df, dE_thresh=3.0, disp_thresh=0.8):
    if df is None or df.empty:
        return df, df

    mask = pd.Series(False, index=df.index)

    if "ΔE_H_user (eV)" in df.columns:
        mask |= df["ΔE_H_user (eV)"].abs() > dE_thresh
    if "ΔE_ads_user (eV)" in df.columns:
        mask |= df["ΔE_ads_user (eV)"].abs() > dE_thresh

    if "H_lateral_disp(Å)" in df.columns:
        mask |= df["H_lateral_disp(Å)"] > disp_thresh
    if "ads_lateral_disp(Å)" in df.columns:
        mask |= df["ads_lateral_disp(Å)"] > disp_thresh

    return df[~mask].copy(), df[mask].copy()


def _normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def co2rr_apply_qa_policy(df: pd.DataFrame, disp_thresh: float = 0.8) -> pd.DataFrame:
    """Ensure CO2RR QA-related columns exist and are normalized.

    Policy:
      - 'qa' is the authoritative filter key when present (ok/migrated kept; others rejected).
      - If 'qa' is missing, infer a conservative 'qa' from displacement and energy blow-ups.
      - 'migrated' is treated as metadata (NOT an automatic reject) once 'qa' exists.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Normalize / infer qa
    if "qa" in out.columns:
        out["qa"] = _normalize_text_series(out["qa"])
    else:
        qa = pd.Series("ok", index=out.index, dtype="object")

        # Energy blow-ups -> crashed
        ecols = [c for c in ["ΔE_ads_user (eV)", "ΔE_ads_user", "ΔG_ads (eV)", "ΔG_ads"] if c in out.columns]
        if "ΔE_ads_user (eV)" in out.columns:
            e = pd.to_numeric(out["ΔE_ads_user (eV)"], errors="coerce")
            qa[(~np.isfinite(e)) | (e.abs() > 50.0)] = "crashed"

        # Large lateral displacement -> migrated (metadata; still keep unless also crashed)
        if "ads_lateral_disp(Å)" in out.columns:
            disp = pd.to_numeric(out["ads_lateral_disp(Å)"], errors="coerce").fillna(0.0)
            qa[(disp > float(disp_thresh)) & (qa == "ok")] = "migrated"

        out["qa"] = qa

    # Infer migrated if missing (for UI display only)
    if "migrated" not in out.columns:
        if "ads_lateral_disp(Å)" in out.columns:
            disp = pd.to_numeric(out["ads_lateral_disp(Å)"], errors="coerce").fillna(0.0)
            out["migrated"] = disp > float(disp_thresh)
        else:
            out["migrated"] = False

    # Ensure relaxed_site exists (fallback to 'site' if absent)
    if "relaxed_site" not in out.columns:
        if "site" in out.columns:
            out["relaxed_site"] = out["site"]
        else:
            out["relaxed_site"] = "unknown"

    return out

def co2rr_split_by_qa(df: pd.DataFrame):
    """Split CO2RR results into candidates vs rejected, using qa as the only reject criterion."""
    if df is None or df.empty:
        return df, df

    qa = _normalize_text_series(df["qa"]) if "qa" in df.columns else pd.Series("ok", index=df.index)
    keep_mask = qa.isin(["ok", "migrated"])
    return df[keep_mask].copy(), df[~keep_mask].copy()

def co2rr_dedupe_candidates(df_keep: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate CO2RR candidates by (adsorbate, relaxed_site) keeping the best (lowest) energy row."""
    if df_keep is None or df_keep.empty:
        return df_keep

    # Choose ranking column
    rank_cols = ["ΔG_ads (eV)", "ΔE_ads_user (eV)", "ΔG_ads", "ΔE_ads_user"]
    rank_col = next((c for c in rank_cols if c in df_keep.columns), None)
    if rank_col is None:
        return df_keep

    # Coerce to numeric for ranking
    tmp = df_keep.copy()
    tmp[rank_col] = pd.to_numeric(tmp[rank_col], errors="coerce")

    key_cols = []
    if "adsorbate" in tmp.columns:
        key_cols.append("adsorbate")
    if "relaxed_site" in tmp.columns:
        key_cols.append("relaxed_site")
    elif "site" in tmp.columns:
        key_cols.append("site")

    if len(key_cols) < 2:
        return tmp

    idx = tmp.groupby(key_cols, dropna=False)[rank_col].idxmin()
    dedup = tmp.loc[idx].copy()

    # Mark duplicates vs representative
    dedup["is_representative"] = True
    tmp["is_representative"] = tmp.index.isin(dedup.index)

    # Preserve original ordering roughly by adsorbate then energy
    dedup = dedup.sort_values(by=key_cols + [rank_col], kind="mergesort").reset_index(drop=True)
    return dedup

def build_compact_table(df, mode: str):
    if df is None or df.empty:
        return df

    if mode == "HER":
        cols = [
            "site_label",
            "site",
            "ΔG_H(U,pH) (eV)",
            "ΔG_H (eV)",
            "ΔE_H_user (eV)",
            "H_lateral_disp(Å)",
            "is_duplicate",
            "reliability",
        ]
    else:
        # CO2RR (and other adsorbate modes)
        cols = [
            "site_label",
            "site",               # start site
            "relaxed_site",       # reclassified site after relaxation (if available)
            "oxide_seed_mode",
            "surface_channel",
            "adsorbate",
            "qa",                 # ok / migrated / desorbed / broken / crashed / unstable
            "migrated",           # boolean (if available)
            "ΔG_ads (eV)",
            "ΔE_ads_user (eV)",
            "ΔE_raw(slab+ads - slab) (eV)",
            "E_ref_reagents (eV)",
            "G_correction (eV)",
            "ref_rxn",
            "ads_lateral_disp(Å)",
            "ads_relax_elapsed_s",
            "ads_relax_n_steps",
            "ads_relax_converged",
            "is_duplicate",
            "reliability",
        ]
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols]

def _load_ads_template_preview(ads: str, ref_dir: str | Path = "ref_gas"):
    ads_clean = ads.replace("*", "").upper()
    if ads_clean not in ADS_TEMPLATE_FILES:
        raise ValueError(f"Unsupported adsorbate '{ads}' for preview")

    cif_path = Path(ref_dir) / ADS_TEMPLATE_FILES[ads_clean]
    if not cif_path.is_file():
        raise FileNotFoundError(f"Adsorbate template not found: {cif_path}")

    a = read(cif_path).copy()
    symbols = a.get_chemical_symbols()

    # Anchor priority: C (CO2RR) → O (ORR: OH*, OOH*, O*) → atom 0
    anchor_idx = None
    for i, s in enumerate(symbols):
        if s == "C":
            anchor_idx = i
            break
    if anchor_idx is None:
        for i, s in enumerate(symbols):
            if s == "O":
                anchor_idx = i
                break
    if anchor_idx is None:
        anchor_idx = 0

    pos = a.get_positions()
    anchor_pos = pos[anchor_idx].copy()
    pos -= anchor_pos
    pos[:, 2] = np.abs(pos[:, 2])
    a.set_positions(pos)
    return a

def _build_adsorbate_preview_slab(slab_atoms, site, ads: str, dz: float = 1.8, ref_dir: str | Path = "ref_gas"):
    slab = slab_atoms.copy()
    z_top = float(slab.get_positions()[:, 2].max())
    ads_clean = ads.replace("*", "").upper()

    ads_atoms = _load_ads_template_preview(ads, ref_dir=ref_dir)

    is_oxide_like = any(s in ANION_SYMBOLS for s in slab.get_chemical_symbols()) and any(s not in ANION_SYMBOLS for s in slab.get_chemical_symbols())
    if is_oxide_like and ads_clean in ("O", "OH", "OOH"):
        channel = expand_oxide_channels_for_adsorbate(ads_clean)[0]
        x0, y0, z0, _surface_channel = oxide_surface_seed_position(slab, site, ads_clean, channel=channel)
        base = np.array([x0, y0, z0], dtype=float)
    else:
        z_min = float(ads_atoms.get_positions()[:, 2].min())
        base_z = z_top + float(dz) - z_min
        xy = np.asarray(site.position[:2], dtype=float)
        base = np.array([xy[0], xy[1], base_z], dtype=float)

    ads_atoms.set_cell(slab.get_cell())
    ads_atoms.set_pbc(slab.get_pbc())
    ads_atoms.translate(base)

    return slab + ads_atoms

def _make_ml_screen_key(sig, mtype, reaction_mode, co2_ads, preset, top_k, geom_per_kind, probe_level, adv_settings: dict, surfactant_class: str = "none", surfactant_prerelax_slab: bool = False):
    return (
        sig,
        str(mtype),
        str(reaction_mode),
        tuple(co2_ads or []),
        str(preset),
        int(top_k),
        int(geom_per_kind),
        str(probe_level),
        str(surfactant_class),
        bool(surfactant_prerelax_slab),
        tuple(sorted((adv_settings or {}).items())),
    )

def _build_ml_compact_df(union_items, union_labels):
    rows = []
    union_items = union_items or []
    union_labels = union_labels or [f"ML_{i}" for i in range(len(union_items))]
    for lbl, r in zip(union_labels, union_items):
        rows.append({
            "label": lbl,
            "adsorbate": getattr(r, "adsorbate", "?"),
            "kind": getattr(r, "kind", "?"),
            "status": "selected",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def _export_zip_of_struct_map(struct_map: dict, symprec: float = 0.1) -> BytesIO:
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, atoms in (struct_map or {}).items():
            zf.writestr(f"{k}.cif", atoms_to_cif_bytes(atoms, symprec=symprec))
    zip_buf.seek(0)
    return zip_buf

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

    expert_mode = st.checkbox("I'm an expert (show advanced options)", value=False)

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

    else:
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

# Optional: surface composition tuning (MP / expert)
loaded_base = st.session_state.get("atoms_loaded")
working = st.session_state.get("atoms_tuned") or loaded_base
if loaded_base is not None:
    st.markdown("### Optional: Surface composition tuning (ratio)")

    # Auto-set default elements based on structure
    ensure_tune_defaults_from_structure(loaded_base)

    # Prevent cumulative growth: always apply tuning on atoms_loaded (not on already tuned)
    if st.session_state.get("atoms_tuned") is not None:
        cA, cB = st.columns([1, 1])
        with cA:
            st.info("Tuning is already applied. Re-applying will overwrite tuning (base = atoms_loaded).")
        with cB:
            if st.button("Clear tuning (revert to loaded)", key="btn_clear_tuning"):
                st.session_state["atoms_tuned"] = None
                st.session_state["ratio_tune_meta"] = None
                _reset_prepared_from_working()
                st.success("Tuning cleared.")
                st.rerun()

    if not expert_mode:
        do_tune = st.checkbox("Apply surface composition tuning (Yes/No)", value=False, key="tune_basic_yesno")
        if do_tune:
            colA, colB, colC = st.columns(3)
            with colA:
                el1 = st.text_input("Element 1", st.session_state.get("rt_el1_basic", "Ni"), key="rt_el1_basic")
                n1 = st.number_input(
                    f"Ratio (Element 1) (sum={RATIO_SUM})",
                    min_value=0,
                    max_value=RATIO_SUM,
                    value=min(6, RATIO_SUM),
                    step=1,
                    key="rt_n1_basic",
                )
            with colB:
                el2 = st.text_input(
                    "Element 2",
                    st.session_state.get("rt_el2_basic", "Cu"),
                    key="rt_el2_basic",
                    help="Enter the metal element for substitution (e.g. Ni, Cu, Al, Mg).",
                )
                n2 = int(RATIO_SUM - int(n1))
                st.write(f"Ratio (Element 2): **{n2}** (auto = {RATIO_SUM} - Ratio(Element 1))")
            with colC:
                layers_from_top = st.number_input("Top layers", 1, 4, 1, key="rt_layers_basic")

            if st.button("Apply tuning", type="primary", key="btn_apply_tune_basic"):
                try:
                    if not str(el2).strip():
                        st.error("Element 2 is empty. Please enter a metal symbol (e.g., Cu).")
                        st.stop()

                    spec = RatioTuneSpec(
                        target_ratio={str(el1): int(n1), str(el2): int(n2)},
                        layers_from_top=int(layers_from_top),
                        layer_tol=0.30,
                        auto_scale_xy=True,
                        max_xy=6,
                        min_sites=20,
                        exact_divisible=True,
                        candidate_elements=[str(el1), str(el2)],
                        rng_seed=GLOBAL_SEED,
                        max_atoms_after_repeat=200,
                        prefer_square_xy=True,
                    )
                    tuned_atoms, meta_out = scale_xy_and_tune_ratio(loaded_base, spec)
                    st.session_state["atoms_tuned"] = tuned_atoms
                    st.session_state["ratio_tune_meta"] = meta_out
                    _reset_prepared_from_working()
                    st.success("Tuning applied (basic). Prepared structure reset.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Tuning failed: {e}")
    else:
        with st.expander("Advanced ratio tuning (expert)", expanded=False):
            colA, colB, colC = st.columns(3)
            with colA:
                layers_from_top = st.number_input("Layers from top", 1, 6, 1, key="rt_layers_adv")
                layer_tol = st.number_input("Layer tol (Å)", 0.05, 1.00, 0.30, step=0.05, key="rt_tol_adv")
            with colB:
                el1 = st.text_input("Element 1", st.session_state.get("rt_el1_adv", "Ni"), key="rt_el1_adv")
                n1 = st.number_input(
                    f"Ratio (Element 1) (sum={RATIO_SUM})",
                    min_value=0,
                    max_value=RATIO_SUM,
                    value=min(6, RATIO_SUM),
                    step=1,
                    key="rt_n1_adv",
                )
                el2 = st.text_input(
                    "Element 2",
                    st.session_state.get("rt_el2_adv", "Cu"),
                    key="rt_el2_adv",
                    help="Enter the metal element for substitution (e.g. Ni, Cu, Al, Mg).",
                )
                n2 = int(RATIO_SUM - int(n1))
                st.write(f"Ratio (Element 2): **{n2}** (auto = {RATIO_SUM} - Ratio(Element 1))")
            with colC:
                max_xy = st.number_input("Max XY repeat", 1, 12, 6, key="rt_maxxy_adv")
                min_sites = st.number_input("Min candidate sites", 1, 500, 20, key="rt_minsites_adv")
                exact_divisible = st.checkbox("Exact divisible (prefer)", value=True, key="rt_exact_adv")

            cand_str = st.text_input(
                "Candidate elements (comma-separated, optional)",
                value="",
                help="Leave empty to use all elements in top layers except O and H.",
                key="rt_cands_adv",
            )
            cand_list = [c.strip() for c in cand_str.split(",") if c.strip()] or None

            max_atoms_cap = st.number_input("Max atoms after repeat", 50, 800, 200, step=10, key="rt_atomcap_adv")
            prefer_square = st.checkbox("Prefer square XY", value=True, key="rt_square_adv")

            if st.button("Apply tuning (expert)", type="primary", key="btn_apply_tune_adv"):
                try:
                    if not str(el2).strip():
                        st.error("Element 2 is empty. Please enter a metal symbol (e.g., Cu).")
                        st.stop()

                    spec = RatioTuneSpec(
                        target_ratio={str(el1): int(n1), str(el2): int(n2)},
                        layers_from_top=int(layers_from_top),
                        layer_tol=float(layer_tol),
                        auto_scale_xy=True,
                        max_xy=int(max_xy),
                        min_sites=int(min_sites),
                        exact_divisible=bool(exact_divisible),
                        candidate_elements=cand_list,
                        rng_seed=GLOBAL_SEED,
                        max_atoms_after_repeat=int(max_atoms_cap),
                        prefer_square_xy=bool(prefer_square),
                    )
                    tuned_atoms, meta_out = scale_xy_and_tune_ratio(loaded_base, spec)
                    st.session_state["atoms_tuned"] = tuned_atoms
                    st.session_state["ratio_tune_meta"] = meta_out
                    _reset_prepared_from_working()
                    st.success("Tuning applied (expert). Prepared structure reset.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Tuning failed: {e}")

    with st.expander("Tuning meta (optional)", expanded=False):
        meta = st.session_state.get("ratio_tune_meta", None)
        if meta is None:
            st.write("No tuning meta.")
        else:
            st.json(_jsonable(meta))

# ---------------- STEP 2: Prepare surface ----------------
st.markdown("## 2) Prepare surface (vacuum / repeat / slabify)")

working = st.session_state.get("atoms_tuned") or st.session_state.get("atoms_loaded")
if working is None:
    st.info("Load a structure first (Step 1).")
else:
    _ensure_prepared_uptodate()
    prepared = st.session_state.get("atoms_prepared")

    colA, colB = st.columns([1.15, 0.85])

    with colA:
        rep = validate_structure(prepared, target_area=70.0)
        vac_z = float(getattr(rep, "vacuum_z", 0.0))
        pbc = tuple(bool(x) for x in prepared.get_pbc())

        st.markdown("### Structure check (Prepared)")
        st.write(f"- Atoms: **{getattr(rep, 'n_atoms', len(prepared))}**")
        st.write(f"- Vacuum_z: **{vac_z:.2f} Å**")
        st.write(f"- PBC: **{pbc}**")

        # OK/WARNING/CRITICAL + pair table (no i/j)
        _render_min_dist_panel(rep)

        if mtype == "oxide":
            fam = infer_oxide_family_from_atoms(prepared)
            if fam["family"] != "unknown":
                st.info(f"Detected Oxide: {fam['family']} ({fam['reduced_formula']})")

        bulk_like = (vac_z < 10.0) and bool(prepared.get_pbc()[2])
        if bulk_like:
            st.warning(
                "BULK-like detected. Surface sites become ill-defined and many candidates may collapse/collide.\n\n"
                "Recommended: add sufficient vacuum (e.g., 30 Å) or slabify."
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
        st.markdown("### Quick fixes (Basic)")
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

        if vac_z < 15.0 and bool(prepared.get_pbc()[2]):
            st.info("Vacuum looks small. Recommended: **30 Å** total vacuum.")
            if st.button("Add vacuum (30 Å)", type="primary", key="btn_add_vac_30"):
                a2 = add_vacuum_z(prepared, total_vacuum_z=30.0, keep_pbc_z=True)
                _push_prepared_update(a2, "add_vacuum", {"total_vacuum_z": 30.0, "keep_pbc_z": True})
                st.success("Vacuum added.")
                st.rerun()

        st.markdown("#### Supercell (XY only)")
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

        if bulk_like:
            if st.button("Temporary workaround: set pbc_z=False", key="btn_pbc_false_tmp"):
                a2 = set_pbc_z(prepared, False)
                _push_prepared_update(a2, "set_pbc_z", {"pbc_z": False})
                st.success("pbc_z=False applied (temporary).")
                st.rerun()

    if expert_mode:
        with st.expander("Advanced surface tools (expert)", expanded=False):
            st.markdown("#### Vacuum (custom)")
            colV1, colV2, colV3 = st.columns(3)
            with colV1:
                vac_custom = st.number_input("Total vacuum_z (Å)", 8.0, 60.0, 30.0, step=1.0, key="vac_custom")
            with colV2:
                keep_pbc_z = st.checkbox("Keep pbc_z=True", value=True, key="vac_keep_pbc_z_adv")
            with colV3:
                if st.button("Apply vacuum (custom)", key="btn_apply_vac_custom"):
                    a2 = add_vacuum_z(prepared, total_vacuum_z=float(vac_custom), keep_pbc_z=bool(keep_pbc_z))
                    _push_prepared_update(a2, "add_vacuum", {"total_vacuum_z": float(vac_custom), "keep_pbc_z": bool(keep_pbc_z)})
                    st.success("Custom vacuum applied.")
                    st.rerun()

            st.markdown("---")
            st.markdown("#### Bulk → Slabify (pymatgen SlabGenerator)")
            if not HAS_SLABIFY:
                st.info(f"SlabGenerator not available: {SLABIFY_IMPORT_ERR}")
            else:
                colS1, colS2, colS3, colS4 = st.columns(4)
                with colS1:
                    h = st.number_input("h", -5, 5, 1, step=1, key="slab_h")
                with colS2:
                    k = st.number_input("k", -5, 5, 1, step=1, key="slab_k")
                with colS3:
                    l = st.number_input("l", -5, 5, 1, step=1, key="slab_l")
                with colS4:
                    max_cands = st.number_input("Max candidates", 1, 12, 6, step=1, key="slab_max_cands")

                colS5, colS6 = st.columns(2)
                with colS5:
                    min_slab = st.number_input("Min slab thickness (Å)", 6.0, 40.0, 12.0, step=1.0, key="slab_min_slab")
                with colS6:
                    min_vac = st.number_input("Min vacuum (Å)", 8.0, 80.0, 30.0, step=1.0, key="slab_min_vac")

                colG1, colG2 = st.columns(2)
                with colG1:
                    if st.button("Generate slab candidates", key="btn_slabify_gen"):
                        try:
                            cand_atoms, cand_meta = slabify_from_bulk(
                                prepared,
                                miller=(int(h), int(k), int(l)),
                                min_slab_size=float(min_slab),
                                min_vacuum_size=float(min_vac),
                                max_candidates=int(max_cands),
                            )
                            st.session_state["slabify_candidates_atoms"] = cand_atoms
                            st.session_state["slabify_candidates_meta"] = cand_meta
                            st.success(f"Generated {len(cand_atoms)} slab candidates.")
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
                    df_meta = pd.DataFrame(cand_meta)
                    st.dataframe(df_meta, use_container_width=True)
                    sel_idx = st.selectbox(
                        "Select slab candidate",
                        list(range(len(cand_atoms))),
                        index=0,
                        format_func=lambda i: f"#{i} | vac_z={cand_meta[i].get('vacuum_z', np.nan):.2f} Å | atoms={cand_meta[i].get('n_atoms')} | {cand_meta[i].get('formula')}",
                        key="slabify_sel_idx",
                    )
                    show_atoms_3d(cand_atoms[sel_idx], height=360, width=700, tag=f"slab_cand_{sel_idx}")

                    if st.button("Use selected slab (replace prepared)", type="primary", key="btn_slabify_apply"):
                        _push_prepared_update(cand_atoms[sel_idx], "slabify_apply", {"candidate_idx": int(sel_idx), "miller": cand_meta[sel_idx].get("miller")})
                        st.success("Selected slab applied.")
                        st.rerun()

# ---------------- STEP 3: Site selection (Geometry / ML) ----------------
st.markdown("## 3) Site selection")

working = st.session_state.get("atoms_tuned") or st.session_state.get("atoms_loaded")
if working is None:
    st.info("Load a structure first.")
else:
    _ensure_prepared_uptodate()
    atoms_for_sites = st.session_state.get("atoms_prepared")

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
    if (not is_her) and bool(surfactant_prerelax_slab):
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
                    mode2 = "default" if mtype == "metal" else "oxide_o"
                    slabs_ads = generate_slab_ads_series(atoms_for_sites_eff, rep_sites, symbol="H", dz=0.0, mode=mode2)
                    export_ads_label = "H"
                else:
                    export_ads_label = preview_ads.replace("*", "")
                    for s in rep_sites:
                        slabs_ads.append(_build_adsorbate_preview_slab(atoms_for_sites_eff, s, preview_ads, dz=1.8, ref_dir="ref_gas"))

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

            if expert_mode:
                with st.expander("Advanced ML settings (expert)", expanded=False):
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
                        zip_buf = _export_zip_of_struct_map(struct_map, symprec=0.1)
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
                    if sel in struct_map:
                        show_atoms_3d(struct_map[sel], height=420, width=900, tag=f"ml_{sel}")
                    else:
                        s = site_map[sel]
                        if is_her:
                            mode2 = "default" if mtype == "metal" else "oxide_o"
                            atoms_prev = generate_slab_ads_series(atoms_for_sites_eff, [s], symbol="H", mode=mode2)[0]
                        else:
                            if is_orr:
                                ads0 = (orr_ads[0] if orr_ads else "OOH*")
                            else:
                                ads0 = (co2_ads[0] if co2_ads else "COOH*")
                            atoms_prev = _build_adsorbate_preview_slab(atoms_for_sites_eff, s, ads0, dz=1.8, ref_dir="ref_gas")
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
                else:
                    auto_sites = detect_oxide_surface_sites(atoms_for_calc)

                rep_sites = select_representative_sites(auto_sites, per_kind=per_kind)
                rep_site_map_for_calc = {f"{s.kind}_{i}": s for i, s in enumerate(rep_sites)}

                # For oxide-HER, shift positions to O-top-based adsorption positions (consistent with your oxide logic)
                if mtype == "oxide" and is_her and rep_site_map_for_calc:
                    shifted = {}
                    for label, site in rep_site_map_for_calc.items():
                        new_x, new_y, new_z = _oxide_o_based_ads_position_compat(
                            atoms_for_calc,
                            site,
                            dz=1.0,
                            extra_z=0.0,
                        )
                        shifted[label] = AdsSite(
                            kind=site.kind,
                            position=(new_x, new_y, new_z),
                            surface_indices=site.surface_indices,
                        )
                    final_user_sites = shifted
                else:
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
                    )
                else:
                    csv_path, meta = run_oxide_che(
                        str(slab_path),
                        sites=manual_sites,
                        relax_mode=relax_mode,
                        user_ads_sites=final_user_sites if final_user_sites else None,
                        use_che_shift=True,
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
            # Legacy HER reliability split (energy/displacement-based)
            df_rel, df_unrel = split_reliable_unreliable(df)
            df["reliability"] = "unreliable"
            if df_rel is not None:
                df.loc[df_rel.index, "reliability"] = "reliable"
        else:
            # CO2RR / ORR: QA-driven policy (migrated is NOT an auto-reject)
            df = co2rr_apply_qa_policy(df)
            df_keep, df_reject = co2rr_split_by_qa(df)

            # Set reliability consistent with QA policy
            df["reliability"] = "unreliable"
            df.loc[df_keep.index, "reliability"] = "reliable"

            # Backwards-compatible names for downstream UI blocks
            df_rel, df_unrel = df_keep, df_reject

        # Persist results for rendering even after rerun (e.g., toggling UI options)
        if isinstance(meta, dict):
            meta = dict(meta)
            meta["SURFACTANT_CLASS"] = str(surfactant_class)
            meta["SURFACTANT_CHGNET_PRERELAX_SLAB"] = bool(surfactant_prerelax_slab)

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
            
            




# ---------------- LLM interpretation helpers ----------------
def _fetch_mp_meta(mp_id: str, api_key: str | None):
    """Fetch lightweight Materials Project metadata to help interpret structural realism."""
    if not mp_id:
        return None
    api_key = api_key or None
    # Prefer mp-api, fall back to legacy pymatgen MPRester if available.
    try:
        from mp_api.client import MPRester  # type: ignore
        with MPRester(api_key=api_key) as mpr:
            docs = mpr.summary.search(material_ids=[mp_id])
            if not docs:
                return {"mp_id": mp_id}
            d = docs[0]
            get = (lambda k: getattr(d, k, None)) if not isinstance(d, dict) else d.get
            return {
                "mp_id": mp_id,
                "formula_pretty": get("formula_pretty"),
                "energy_above_hull": get("energy_above_hull"),
                "formation_energy_per_atom": get("formation_energy_per_atom"),
                "band_gap": get("band_gap"),
                "crystal_system": get("crystal_system"),
                "spacegroup_symbol": (get("symmetry") or {}).get("symbol") if isinstance(get("symmetry"), dict) else None,
            }
    except Exception:
        pass
    try:
        from pymatgen.ext.matproj import MPRester  # type: ignore
        with MPRester(api_key) as mpr:
            docs = mpr.query(criteria={"material_id": mp_id},
                             properties=["pretty_formula", "e_above_hull", "formation_energy_per_atom", "band_gap", "spacegroup"])
            if not docs:
                return {"mp_id": mp_id}
            d = docs[0]
            return {
                "mp_id": mp_id,
                "formula_pretty": d.get("pretty_formula"),
                "energy_above_hull": d.get("e_above_hull"),
                "formation_energy_per_atom": d.get("formation_energy_per_atom"),
                "band_gap": d.get("band_gap"),
                "spacegroup_symbol": (d.get("spacegroup") or {}).get("symbol") if isinstance(d.get("spacegroup"), dict) else None,
            }
    except Exception:
        return {"mp_id": mp_id}


def _pick_representative_sites_her(df: pd.DataFrame):
    # Prefer CHE-shifted column if present
    col = None
    for c in ["ΔG_H(U,pH) (eV)", "dG_H(U,pH) (eV)", "ΔG_H (eV)", "dG_H (eV)"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        return {}

    g = pd.to_numeric(df[col], errors="coerce")
    g_valid = g.dropna()
    if g_valid.empty:
        return {}

    def _site_label(i):
        if "site_label" in df.columns:
            return str(df.loc[i, "site_label"])
        if "site" in df.columns:
            return str(df.loc[i, "site"])
        return str(i)

    i_act = (g_valid.abs()).idxmin()
    i_min = g_valid.idxmin()

    return {
        "activity_best": {"site_label": _site_label(i_act), "value": float(g.loc[i_act]), "criterion": "min_abs_dG"},
        "thermo_best": {"site_label": _site_label(i_min), "value": float(g.loc[i_min]), "criterion": "most_negative_dG"},
        "column_used": col,
    }



def _pick_representative_sites_co2rr(df: pd.DataFrame):
    # Prefer user-reported adsorption energy column
    col = None
    for c in ["ΔE_ads_user (eV)", "dE_ads_user (eV)", "ΔE_ads (eV)", "dE_ads (eV)"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        return {}

    e = pd.to_numeric(df[col], errors="coerce")

    # Stable = non-migrating when available (+ exclude obvious failed rows when possible)
    stable_mask = pd.Series([True] * len(df), index=df.index)

    if "migrated" in df.columns:
        stable_mask &= (pd.to_numeric(df["migrated"], errors="coerce").fillna(0) == 0)

    if "reliability" in df.columns:
        stable_mask &= ~df["reliability"].astype(str).str.lower().isin(["failed", "error", "nan"])

    if "qa" in df.columns:
        stable_mask &= ~df["qa"].astype(str).str.lower().str.contains("fail|invalid|error|nan", regex=True, na=False)

    e_stable = e[stable_mask].dropna()
    if e_stable.empty:
        # Fallback: use any non-NaN values if stability filter removed everything
        e_stable = e.dropna()
        if e_stable.empty:
            return {"column_used": col, "stable_filter": "none (no valid values)"}
        stable_mask = pd.Series([True] * len(df), index=df.index)  # reset to "all"

    def _site_label(i):
        if "site_label" in df.columns:
            return str(df.loc[i, "site_label"])
        if "site" in df.columns:
            return str(df.loc[i, "site"])
        return str(i)

    # Regime diagnostics (important for wording)
    all_positive = bool((e_stable >= 0).all())
    min_val = float(e_stable.min())

    # Strong track:
    # - If any exothermic (negative) exists, "strong" = most negative
    # - If all are >=0, do NOT call it strong adsorption; it's "least endothermic / least unfavorable"
    i_min = e_stable.idxmin()
    strong_criterion = "most_negative_dE" if not all_positive else "min_dE_all_positive"
    strong_note = (
        "Exothermic adsorption exists (ΔE<0 present); 'strong' wording allowed."
        if not all_positive
        else "All stable sites have ΔE>=0; DO NOT call this 'strong adsorption'. Report as least-unfavorable/least-endothermic and discuss only relative ordering."
    )

    strong = {
        "site_label": _site_label(i_min),
        "value": float(e.loc[i_min]),
        "criterion": strong_criterion,
        "terminology_note": strong_note,
    }

    # Balanced track (more conservative for CO2RR):
    # - Only define balanced among exothermic candidates (ΔE<0). If none, set N/A.
    e_exo = e_stable[e_stable < 0]
    if not e_exo.empty:
        i_bal = (e_exo.abs()).idxmin()
        balanced = {
            "site_label": _site_label(i_bal),
            "value": float(e.loc[i_bal]),
            "criterion": "min_abs_dE_among_exothermic",
            "terminology_note": "Balanced binding defined only among ΔE<0 candidates to avoid weak-adsorption misinterpretation.",
        }
    else:
        balanced = {
            "site_label": None,
            "value": None,
            "criterion": "N/A_no_exothermic_adsorption",
            "terminology_note": "No ΔE<0 candidates among stable sites; balanced track is not defined under this conservative rule.",
        }

    return {
        "strong_binding_best": strong,
        "balanced_binding_best": balanced,
        "column_used": col,
        "stable_filter": "non-migrating (migrated==0) + non-failed rows when available",
        "binding_regime": {
            "all_positive_stable": all_positive,
            "min_dE_stable": min_val,
        },
    }


def _build_llm_payload(last_run: dict):
    df = last_run.get("df")
    meta = last_run.get("meta") or {}

    payload = {
        "run_meta": {
            "reaction_mode": str(last_run.get("reaction_mode", "")),
            "material_type": str(last_run.get("mtype", "")),
            "mode_label": str(last_run.get("mode_label", "")),
            "model": str(meta.get("MODEL", meta.get("model", ""))),
            "device": str(meta.get("DEVICE", meta.get("device", ""))),
            "U": float(last_run.get("U_input", 0.0)),
            "pH": float(last_run.get("pH_input", 14.0)),
        },
        "mp_meta": None,
        "definitions": {
            "ΔE_ads_definition": None,
            "ΔE_ads_sign_convention": None,
            "ΔG_H_definition": None,
            "ΔG_H_sign_convention": None,
            "migration_definition": f"Flag as migrated if ads_lateral_disp(Å) > {CO2RR_MIGRATION_DISP_THRESH_A} Å (lateral displacement of the adsorbate anchor between initial and relaxed positions).",
            "migration_disp_threshold_A": CO2RR_MIGRATION_DISP_THRESH_A,
            "available_distance_fields": [],
            "distance_metrics_note": "If z/min-distance diagnostics are absent, do not infer adsorption/desorption; report as unavailable.",
        },
        "qc_flags": {},
        "rules": {},
        "table_preview": [],
    }

    mp_id = st.session_state.get("loaded_mp_id")
    payload["mp_meta"] = _fetch_mp_meta(mp_id, st.session_state.get("mp_api_key") or None) if mp_id else None

    # --- Definitions / conventions (minimize LLM hallucinations) ---
    is_her = bool(last_run.get("is_her"))
    if is_her:
        payload["definitions"]["ΔG_H_definition"] = "ΔG_H is computed from H* adsorption with CHE; ΔG_H(U,pH) = ΔG_H − U − 0.0591·pH (eV)."
        payload["definitions"]["ΔG_H_sign_convention"] = "More negative ΔG_H indicates stronger H* binding; activity commonly correlates with |ΔG_H| near 0 eV."
    else:
        payload["definitions"]["ΔE_ads_definition"] = "ΔE_ads = E(slab+ads) − E(slab) − E(ref_adsorbate). (eV)"
        payload["definitions"]["ΔE_ads_sign_convention"] = "More negative ΔE_ads means stronger (more exothermic) binding under this reference; positive implies endothermic adsorption."

    if isinstance(df, pd.DataFrame):
        dist_candidates = [
            "ads_z_min(Å)", "ads_z_max(Å)", "ads_z(Å)", "ads_height(Å)",
            "slab_z_max(Å)", "z_top(Å)",
            "min_surf_dist(Å)", "ads_min_surf_dst(Å)", "ads_surf_min_dist(Å)",
            "ads_z_clearance(Å)",
        ]
        payload["definitions"]["available_distance_fields"] = [c for c in dist_candidates if c in df.columns]
        # Optional derived clearance if columns exist
        if ("ads_z_min(Å)" in df.columns) and (("slab_z_max(Å)" in df.columns) or ("z_top(Å)" in df.columns)) and ("ads_z_clearance(Å)" not in df.columns):
            try:
                slab_col = "slab_z_max(Å)" if "slab_z_max(Å)" in df.columns else "z_top(Å)"
                df["ads_z_clearance(Å)"] = pd.to_numeric(df["ads_z_min(Å)"], errors="coerce") - pd.to_numeric(df[slab_col], errors="coerce")
            except Exception:
                pass

    if isinstance(df, pd.DataFrame) and (not df.empty):
        if bool(last_run.get("is_her")):
            payload["rules"] = _pick_representative_sites_her(df)
        else:
            payload["rules"] = _pick_representative_sites_co2rr(df)

        # QC
        if "migrated" in df.columns:
            payload["qc_flags"]["n_migrated"] = int(pd.to_numeric(df["migrated"], errors="coerce").fillna(0).sum())
        if "is_duplicate" in df.columns:
            payload["qc_flags"]["n_duplicate"] = int(pd.to_numeric(df["is_duplicate"], errors="coerce").fillna(0).sum())

        # Preview rows (avoid huge payload)
        cols_priority = [
            "site_label", "site", "site_kind", "relaxed_site", "reliability", "qa",
            "migrated", "is_duplicate",
            # Displacement / migration diagnostics (if present)
            "ads_lateral_disp(Å)", "H_lateral_disp(Å)",
            # Key energies
            "ΔG_H(U,pH) (eV)", "ΔG_H (eV)", "dG_H(U,pH) (eV)", "dG_H (eV)",
            "ΔE_ads_user (eV)", "E_slab_user (eV)", "E_slab+ads_user (eV)",
            # Distance / adsorption plausibility diagnostics (if present)
            "ads_z_min(Å)", "ads_z_clearance(Å)", "min_surf_dist(Å)", "ads_min_surf_dist(Å)",
        ]
        cols = [c for c in cols_priority if c in df.columns]
        preview = (df[cols].head(40) if cols else df.head(40)).copy()
        payload["table_preview"] = preview.fillna("").to_dict(orient="records")

    return payload


def _call_llm_interpreter(payload: dict):
    """Call OpenAI to produce a structured, paper-oriented interpretation of the last run.

    Notes
    -----
    - Uses Structured Outputs via `responses.parse` with a Pydantic schema.
    - IMPORTANT: Avoid free-form `dict`/`Any` fields in the schema when `strict` is enabled.
      OpenAI's Structured Outputs requires closed object schemas (no additional properties).
    """
    import json

    api_key = (st.session_state.get("openai_api_key") or "").strip()
    if not api_key:
        raise RuntimeError("OpenAI API key is empty (set it in 0) Credentials & LLM).")

    try:
        from openai import OpenAI  # type: ignore
        from pydantic import BaseModel, Field, ConfigDict
    except Exception as e:
        raise RuntimeError("Missing dependency. Install with: pip install openai pydantic") from e

    class PaperTrack(BaseModel):
        model_config = ConfigDict(extra="forbid")

        track_name: str = Field(..., description="Name of the reporting track (e.g., 'strong_binding', 'balanced_binding').")
        criterion: str = Field(..., description="Criterion used (e.g., most_negative_dE, min_abs_dE, most_negative_dG, min_abs_dG).")
        column_used: str = Field(..., description="Which numeric column was used to choose this track.")
        site_label: str | None = Field(..., description="Site label selected for this track. Use null if unavailable.")
        value: float | None = Field(..., description="Selected metric value for this track. Use null if unavailable.")
        selection_rule: str = Field(..., description="Explicit rule describing how this value was chosen.")
        evidence_note: str = Field(..., description="Short note tying the selection to payload evidence and QC filters (e.g., non-migrating).")

    class RecommendedForPaper(BaseModel):
        model_config = ConfigDict(extra="forbid")

        strong_binding_track: PaperTrack = Field(..., description="Track emphasizing strongest binding (most negative energy) among stable sites.")
        balanced_binding_track: PaperTrack = Field(..., description="Track emphasizing balanced binding (closest to 0) among stable sites.")
        primary_track_to_report: str = Field(
            ...,
            description="Which track should be the primary headline in the paper: 'strong_binding' or 'balanced_binding'.",
        )
        how_to_report: str = Field(
            ...,
            description="Guidance on how to report both tracks without over-claiming activity; include sign-convention caveats.",
        )
        suggested_paper_text: str = Field(
            ...,
            description="1–4 sentences suitable for Methods/Results that report both tracks and justify the rules.",
        )

    class Interpretation(BaseModel):
        model_config = ConfigDict(extra="forbid")

        executive_summary: str = Field(..., description="High-level summary (3–6 sentences).")

        qc_findings: list[str] = Field(
            ...,
            description="Quality-control findings (migration, duplicates, site collapse, non-physical artifacts).",
        )
        site_interpretation: list[str] = Field(
            ...,
            description=(
                "Interpretation of site behavior: why different sites look similar/different; "
                "why bridge/fcc can collapse to top; oxide vs metal behavior; surface area/termination effects."
            ),
        )
        recommended_for_paper: RecommendedForPaper = Field(
            ...,
            description="Two-track recommendation for representative value(s) and how to report them.",
        )
        next_actions: list[str] = Field(
            ...,
            description="Concrete next actions (checks, reruns, DFT spot-checks, structure realism checks).",
        )
        uncertainties: list[str] = Field(
            ...,
            description="What cannot be concluded from the payload; assumptions; missing inputs.",
        )

    client = OpenAI(api_key=api_key)
    
    developer_instructions = """
You are a scientific assistant that interprets electrocatalysis screening results (HER or CO2RR).

Use ONLY the provided JSON payload as evidence.
Do not use outside knowledge, unstated assumptions, or inferred values not supported by the payload.

Your task is to return a JSON object only, following the required schema exactly.

PRIORITY OF EVIDENCE
1. payload["definitions"] (highest priority)
2. payload["rules"]
3. per-site QC / diagnostics / flags
4. numeric site values
5. mp_meta thermodynamic indicators
If any source conflicts with a lower-priority source, follow the higher-priority source and note the conflict in uncertainties.

GENERAL DECISION RULES
- A site is "stable" only if it is not excluded by payload["definitions"] or payload["rules"].
- Exclude from recommendation any site flagged as migrated, collapsed, duplicate, or non-physical according to payload definitions/rules.
- If distance or geometry diagnostics required for a stability decision are missing, do not assume stability; mark the site as uncertain.
- If no stable sites remain after exclusions, set all recommended track fields to null and explain why.
- If only one stable site remains, both tracks may point to the same site if it satisfies both selection rules.
- If multiple sites are tied, break ties deterministically in this order:
  (1) site with fewer QC concerns,
  (2) site with more complete diagnostics,
  (3) lexicographically smaller site_label.

QC INTERPRETATION TASKS
1. Flag QC issues:
   - migration
   - site collapse
   - duplicates
   - non-physical artifacts
   - missing diagnostics that prevent firm interpretation
2. Explain likely causes using only payload-supported materials interpretations, such as:
   - surface termination effects
   - limited site diversity
   - oxide vs metal anchoring differences
   - slab/model limitations
Do not assert causes not grounded in the payload.

RECOMMENDATION RULES
You MUST populate recommended_for_paper with two tracks.

For HER:
- strong_binding_track = most negative ΔG_H among stable sites
- balanced_binding_track = stable site with minimum |ΔG_H|

For CO2RR:
- If any stable site has ΔE_ads < 0:
  - strong_binding_track = most negative ΔE_ads among stable sites
  - balanced_binding_track = stable site with minimum |ΔE_ads| among sites with ΔE_ads < 0
- If all stable sites have ΔE_ads >= 0:
  - report the minimum ΔE_ads site as the least-unfavorable / least-endothermic case
  - set balanced_binding_track.site_label = null
  - set balanced_binding_track.value = null
  - explain that no exothermic stable candidate exists under this reference definition
  - do not describe this as strong adsorption

REPORTING RULES
- Prefer payload["rules"] when selecting representative values.
- Use exact sign conventions from payload["definitions"].
- State explicitly whether a recommended site is non-migrating under the stated migration criterion.
- Do not use:
  "reliable", "optimal", "high efficiency", "low efficiency", "potential for CO2RR"
  unless such claims are explicitly supported by payload fields that justify them.
- Do not use "favorable" unless the sign convention and comparison set are explicitly stated.
- If ΔE_ads is positive, state that adsorption is endothermic/weak under this reference and discuss only relative ordering.
- If mp_meta contains energy_above_hull or formation energy, describe them only as thermodynamic indicators, typically near 0 K and without kinetic/processing context.

OUTPUT SCHEMA
Return JSON only with exactly these top-level keys:
{
  "mode": null,
  "qc_summary": {
    "status": null,
    "issues": [],
    "uncertainties": []
  },
  "site_assessment": [],
  "recommended_for_paper": {
    "strong_binding_track": {
      "site_label": null,
      "value": null,
      "evidence_note": "",
      "uncertainties": []
    },
    "balanced_binding_track": {
      "site_label": null,
      "value": null,
      "evidence_note": "",
      "uncertainties": []
    }
  },
  "materials_interpretation": [],
  "next_actions": []
}

SITE_ASSESSMENT ITEM SCHEMA
Each item in site_assessment must be:
{
  "site_label": null,
  "included_in_selection": null,
  "exclusion_reason": null,
  "qc_flags": [],
  "value": null,
  "evidence_note": "",
  "uncertainties": []
}

NEXT ACTIONS
Propose only payload-consistent validation actions, such as:
- alternative termination or polymorph
- slab thickness / repeat size check
- additional site diversity
- DFT spot-check
- duplicate filtering / geometry verification

Do not output markdown. Do not output prose outside JSON.
"""
    
    resp = client.responses.parse(
        model=st.session_state.get("llm_model", "gpt-4o-mini-2024-07-18"),
        input=[
            {"role": "system", "content": developer_instructions},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text_format=Interpretation,
        temperature=0.2,
        store=False,
    )
    out = resp.output_parsed.model_dump()

    # Attach best-effort usage + cost estimate for transparency
    model_name = str(st.session_state.get("llm_model", "gpt-4o-mini-2024-07-18"))
    out["_model"] = model_name
    usage_obj = getattr(resp, "usage", None)
    usage_dict = None
    if usage_obj is not None:
        try:
            usage_dict = usage_obj.model_dump()  # pydantic
        except Exception:
            try:
                usage_dict = dict(usage_obj)
            except Exception:
                usage_dict = {"raw": str(usage_obj)}
    if usage_dict is not None:
        out["_usage"] = usage_dict
        # SDKs may use either input_tokens/output_tokens or prompt_tokens/completion_tokens
        in_toks = usage_dict.get("input_tokens", usage_dict.get("prompt_tokens"))
        out_toks = usage_dict.get("output_tokens", usage_dict.get("completion_tokens"))
        try:
            in_toks_i = int(in_toks) if in_toks is not None else None
        except Exception:
            in_toks_i = None
        try:
            out_toks_i = int(out_toks) if out_toks is not None else None
        except Exception:
            out_toks_i = None
        out["_cost_estimate_usd"] = _estimate_cost_usd(model_name, in_toks_i, out_toks_i)

    return out


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
                    payload = _build_llm_payload(last_run)
                    out = _call_llm_interpreter(payload)
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
