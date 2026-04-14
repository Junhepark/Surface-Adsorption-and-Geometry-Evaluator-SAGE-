from typing import Optional

import numpy as np
import streamlit as st

from ocp_app.core.structure_ops import _recenter_slab_z_into_cell

HAS_ADSORML = True
ADSORML_IMPORT_ERR = None
try:
    from ocp_app.core.adsorbml_lite_screening import relax_slab_chgnet
except Exception as e:
    HAS_ADSORML = False
    ADSORML_IMPORT_ERR = str(e)

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

