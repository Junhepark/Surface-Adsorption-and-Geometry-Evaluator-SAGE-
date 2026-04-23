# ocp_app/core/anchors/common.py
from __future__ import annotations

import time

import numpy as np
import torch
from ase import Atoms
from ase.build import add_adsorbate
from ase.constraints import FixAtoms, FixCartesian, Hookean
from ase.optimize import BFGS
from ase.geometry import find_mic

from fairchem.core import pretrained_mlip, FAIRChemCalculator

# ---------------- Config / UMA init ----------------
MODEL_NAME = "uma-s-1p1"   # or "uma-m-1p1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0)
torch.manual_seed(0)

_predictor = pretrained_mlip.get_predict_unit(MODEL_NAME, device=DEVICE)
calc = FAIRChemCalculator(_predictor, task_name="oc20")

# ---- Thermochemistry (298 K) ----
ZPE_CORR = 0.04
TDS_CORR = 0.20
NET_CORR = ZPE_CORR + TDS_CORR 

# ---- Thresholds / parameters (shared) ----
H0S = (1.0, 1.2, 1.4)   # Initial H height scan values (Angstrom)
RIGID_OXIDE_H0S = (0.65, 0.80, 0.95, 1.10)  # Lower initial heights for rigid oxide O-top seeds
MIGRATE_THR = 0.30       # Lateral displacement flag threshold (Angstrom)
VAC_WARN_MIN = 20.0      # Vacuum warning threshold (Angstrom)
UNUSUAL_DDELTA = 1.00    # |dE_user - dE_anchor(site)| warning threshold (eV)


# ---------------- Shared utilities ----------------
def ensure_pbc3(a, vac_z=None):
    """Enforce 3D PBC; optionally reset z-vacuum."""
    a = a.copy()
    a.set_pbc([True, True, True])
    if vac_z is not None:
        a.center(axis=2, vacuum=float(vac_z))
    a.wrap(eps=1e-9)
    return a


def layer_indices(at, n=3, tol=0.25):
    """Return index arrays for the top n layers, ordered from top to bottom."""
    z = at.get_positions()[:, 2]
    zuniq = np.unique(np.round(z, 3))
    zuniq.sort()
    topz = zuniq[-n:]
    layers = [np.where(np.isclose(z, zval, atol=tol))[0] for zval in topz[::-1]]
    return layers  # [top, second, third, ...] (up to n)


def first_layer_min_distance(a, *, include_h_in_top_layer: bool = False):
    """Minimum distance between H and the top-layer atoms.

    Parameters
    ----------
    include_h_in_top_layer
        When True, use the legacy metal-benchmark behavior and measure against
        the raw top-layer indices returned by ``layer_indices``. When False,
        ignore top-layer H atoms and fall back to the highest non-H atoms.
    """
    pos = a.get_positions()
    h_idx = [i for i, _ in enumerate(a) if a[i].symbol == "H"]
    if not h_idx:
        return None
    hpos = pos[h_idx[0]]
    if bool(include_h_in_top_layer):
        top_idx = layer_indices(a, n=1)[0]
    else:
        top_idx = [int(i) for i in layer_indices(a, n=1)[0] if a[int(i)].symbol != "H"]
        if not top_idx:
            non_h = [i for i, atom in enumerate(a) if atom.symbol != "H"]
            if not non_h:
                return None
            z = pos[non_h, 2]
            zmax = float(np.max(z))
            top_idx = [int(non_h[k]) for k, zv in enumerate(z) if abs(float(zv) - zmax) <= 0.35]
            if not top_idx:
                return None
    dxyz = np.linalg.norm(pos[top_idx] - hpos, axis=1)
    return float(dxyz.min())


def put_H(at, xy, height=1.20, min_clearance=0.9, base_z=None, *, mtype: str | None = None):
    """Place an H adsorbate on the slab at the given xy position.

    Metal surfaces keep the exact legacy benchmark placement rule so that
    historical metal HER values are reproduced. Oxides retain the newer
    anchor-z placement and non-H top-layer clearance logic.
    """
    mtype_s = str(mtype or "").strip().lower()
    if mtype_s == "metal":
        return _put_H_metal_legacy(at, xy, height=height, min_clearance=min_clearance)

    a = ensure_pbc3(at)
    if base_z is None or not np.isfinite(float(base_z)):
        add_adsorbate(
            a,
            "H",
            height=float(height),
            position=(float(xy[0]), float(xy[1])),
        )
    else:
        h = Atoms(
            "H",
            positions=[[float(xy[0]), float(xy[1]), float(base_z) + float(height)]],
            cell=a.get_cell(),
            pbc=a.get_pbc(),
        )
        a += h
    a = ensure_pbc3(a)
    md = first_layer_min_distance(a, include_h_in_top_layer=False)
    if md is not None and md < float(min_clearance):
        for i, atom in enumerate(a):
            if atom.symbol == "H":
                p = np.asarray(atom.position, dtype=float)
                p[2] += (float(min_clearance) - float(md)) + 0.10
                atom.position = p
                break
    a = ensure_pbc3(a)
    return a


def make_bottom_fix_mask(at, n_fix_layers=2, z_tol=0.25):
    """Create a boolean mask that fixes the bottom n_fix_layers layers."""
    z = at.get_positions()[:, 2]
    zs = np.unique(np.round(z, 3))
    cuts = zs[: min(int(n_fix_layers), len(zs))]
    mask = []
    for atom in at:
        if atom.symbol == "H":
            mask.append(False)
        else:
            mask.append(np.any(np.abs(atom.position[2] - cuts) < z_tol))
    return np.array(mask, bool)


def make_non_h_fix_mask(at):
    """Create a boolean mask that fixes every non-H atom."""
    return np.array([atom.symbol != "H" for atom in at], dtype=bool)


def recommend_h0s_for_relaxation(relaxation_scope: str | None, *, mtype: str | None = None, site_kind: str | None = None, default_h0s=H0S):
    """Return scope-aware initial H heights.

    Rigid oxide O-top / anion seeds often need lower initial heights because the
    slab cannot adapt to capture an overly high H seed.  This helper preserves
    the legacy H0S values for all other modes.
    """
    scope = normalize_relaxation_scope(relaxation_scope, default="partial") if relaxation_scope is not None else "partial"
    mtype_s = str(mtype or "").strip().lower()
    kind_s = str(site_kind or "").strip().lower()
    if scope == "rigid" and mtype_s == "oxide":
        if any(tok in kind_s for tok in ("o_top", "anion", "ontop")):
            return tuple(float(x) for x in RIGID_OXIDE_H0S)
    return tuple(float(x) for x in np.asarray(default_h0s, dtype=float).reshape(-1))


def default_min_clearance_for_scope(relaxation_scope: str | None, *, mtype: str | None = None) -> float:
    scope = normalize_relaxation_scope(relaxation_scope, default="partial") if relaxation_scope is not None else "partial"
    if scope == "rigid" and str(mtype or "").strip().lower() == "oxide":
        return 0.75
    return 0.90


def normalize_relaxation_scope(scope: str | None, *, default: str = "partial") -> str:
    """Normalize relaxation scope labels."""
    s = str(scope or default).strip().lower()
    aliases = {
        "freeze": "rigid",
        "fixed": "rigid",
        "h-only": "rigid",
        "h_only": "rigid",
        "top": "partial",
        "top_only": "partial",
        "bottom-fixed": "partial",
        "bottom_fixed": "partial",
        "all": "full",
        "free": "full",
    }
    s = aliases.get(s, s)
    if s not in {"rigid", "partial", "full"}:
        raise ValueError(f"Unknown relaxation_scope='{scope}'")
    return s


def build_relax_constraints(at, relaxation_scope: str = "partial", n_fix_layers: int = 2):
    """Build slab constraints for rigid / partial / full relaxation scopes."""
    scope = normalize_relaxation_scope(relaxation_scope)
    if scope == "rigid":
        mask = make_non_h_fix_mask(at)
        return [FixAtoms(mask=mask)] if mask.any() else []
    if scope == "partial":
        mask = make_bottom_fix_mask(at, n_fix_layers=int(n_fix_layers))
        return [FixAtoms(mask=mask)] if mask.any() else []
    return []


def _minimum_image_distance(at, i: int, j: int) -> float:
    pos = at.get_positions()
    dvec = np.asarray(pos[int(j)], dtype=float) - np.asarray(pos[int(i)], dtype=float)
    vec, dist = find_mic(dvec, at.get_cell(), at.get_pbc())
    return float(dist)


def build_anchor_oh_constraints(at, anchor_index: int, h_index: int = -1, shell_margin: float = 0.20, hook_k: float = 5.0, hook_extra: float = 0.20):
    """Build D1 local-OH constraints with soft O–metal support.

    D1 semantics:
      - H stays on the selected O-top anchor (x,y fixed; z free)
      - anchor O can respond locally to form O–H (xyz free)
      - the rest of the slab remains fixed
      - anchor O is softly restrained to its first-shell metal neighbors using
        Hookean springs so that detached-OH artifacts are suppressed.
    """
    a = ensure_pbc3(at)
    n = len(a)
    anchor_index = int(anchor_index)
    if anchor_index < 0 or anchor_index >= n:
        raise IndexError(f"anchor_index out of range: {anchor_index}")
    if h_index < 0:
        h_index = n + int(h_index)
    h_index = int(h_index)
    if h_index < 0 or h_index >= n or a[h_index].symbol != 'H':
        raise IndexError(f"h_index does not point to an H atom: {h_index}")

    # Identify first-shell metal neighbors of the anchor O in the initial slab.
    metal_candidates = [i for i, atom in enumerate(a) if i not in {anchor_index, h_index} and atom.symbol not in {'H', 'O'}]
    shell_indices = []
    shell_r0 = {}
    if metal_candidates:
        dists = [(i, _minimum_image_distance(a, anchor_index, i)) for i in metal_candidates]
        if dists:
            dmin = min(d for _, d in dists)
            cutoff = float(dmin) + float(shell_margin)
            shell_indices = [int(i) for i, d in dists if d <= cutoff]
            shell_r0 = {int(i): float(d) for i, d in dists if d <= cutoff}

    fixed_idx = [i for i in range(n) if i not in {anchor_index, h_index}]
    constraints = []
    if fixed_idx:
        constraints.append(FixAtoms(indices=fixed_idx))
    # Keep H above the selected O anchor: lateral fixed, z free.
    constraints.append(FixCartesian(h_index, [True, True, False]))
    # Softly preserve O–metal support while allowing explicit O–H formation.
    for i in shell_indices:
        rt = float(shell_r0.get(int(i), _minimum_image_distance(a, anchor_index, i))) + float(hook_extra)
        constraints.append(Hookean(a1=anchor_index, a2=int(i), rt=rt, k=float(hook_k)))

    meta = {
        'anchor_index': int(anchor_index),
        'h_index': int(h_index),
        'local_free_indices': [int(anchor_index), int(h_index)],
        'local_free_count': 2,
        'shell_indices': [int(i) for i in shell_indices],
        'shell_count': int(len(shell_indices)),
        'shell_margin': float(shell_margin),
        'hook_k': float(hook_k),
        'hook_extra': float(hook_extra),
        'fixed_indices': fixed_idx,
    }
    return constraints, meta

def _constraint_atom_dof_status(at):
    """Return per-atom fully_fixed flags after current ASE constraints are applied."""
    n = len(at)
    fixed_xyz = np.zeros((n, 3), dtype=bool)
    constraints = at.constraints if isinstance(at.constraints, (list, tuple)) else ([at.constraints] if at.constraints else [])
    for cons in constraints:
        if cons is None:
            continue
        if hasattr(cons, "get_indices"):
            try:
                idx = np.asarray(cons.get_indices(), dtype=int).reshape(-1)
            except Exception:
                idx = np.array([], dtype=int)
        else:
            idx = np.array([], dtype=int)
        cname = cons.__class__.__name__
        if cname == "FixAtoms":
            if idx.size > 0:
                fixed_xyz[idx, :] = True
            continue
        if cname == "FixCartesian":
            mask = None
            for attr in ("mask", "index_mask"):
                if hasattr(cons, attr):
                    try:
                        mask = np.asarray(getattr(cons, attr), dtype=bool).reshape(3)
                        break
                    except Exception:
                        pass
            if mask is None:
                d = getattr(cons, "todict", lambda: {})()
                kwargs = d.get("kwargs", {}) if isinstance(d, dict) else {}
                if "mask" in kwargs:
                    try:
                        mask = np.asarray(kwargs.get("mask"), dtype=bool).reshape(3)
                    except Exception:
                        mask = None
            if idx.size > 0 and mask is not None:
                fixed_xyz[idx[0], :] |= mask
    fully_fixed = fixed_xyz.all(axis=1)
    return fully_fixed


def _relaxation_meta(at, *, relaxation_scope: str, n_fix_layers: int, elapsed_s: float, n_steps: int, converged, error=None):
    fully_fixed = _constraint_atom_dof_status(at)
    fixed_atom_count = int(np.count_nonzero(fully_fixed))
    relaxed_atom_count = int(len(at) - fixed_atom_count)
    return {
        "relaxation_scope": str(relaxation_scope),
        "n_fix_layers": int(n_fix_layers),
        "fixed_atom_count": fixed_atom_count,
        "relaxed_atom_count": relaxed_atom_count,
        "n_steps": int(n_steps),
        "converged": converged,
        "elapsed_s": float(elapsed_s),
        "error": "" if error is None else str(error),
    }


def _run_bfgs_with_meta(at, *, fmax: float, steps: int):
    if int(steps) <= 0:
        return 0, None, 0.0, None
    dyn = BFGS(at, logfile=None)
    t0 = time.perf_counter()
    err = None
    conv = None
    try:
        dyn.run(fmax=float(fmax), steps=int(steps))
        try:
            conv = bool(dyn.converged())
        except Exception:
            conv = None
    except Exception as e:
        err = str(e)
    elapsed = float(time.perf_counter() - t0)
    n_steps = 0
    try:
        if hasattr(dyn, "get_number_of_steps"):
            n_steps = int(dyn.get_number_of_steps())
        elif hasattr(dyn, "nsteps"):
            n_steps = int(getattr(dyn, "nsteps"))
    except Exception:
        n_steps = 0
    return n_steps, conv, elapsed, err


def _put_H_metal_legacy(at, xy, height=1.20, min_clearance=0.9):
    """Legacy metal H placement used by historical benchmark tables."""
    a = ensure_pbc3(at)
    add_adsorbate(
        a,
        "H",
        height=float(height),
        position=(float(xy[0]), float(xy[1])),
    )
    a = ensure_pbc3(a)
    md = first_layer_min_distance(a, include_h_in_top_layer=True)
    if md is not None and md < min_clearance:
        for i, atom in enumerate(a):
            if atom.symbol == "H":
                p = atom.position
                p[2] += (min_clearance - md) + 0.2
                atom.position = p
                break
    return a


def _relax_zonly_metal_legacy(atoms, steps=None, fmax=0.05, return_meta: bool = False):
    """Exact legacy metal z-only relaxation."""
    a = ensure_pbc3(atoms)
    a.calc = calc
    cons = [FixAtoms(mask=make_bottom_fix_mask(a, n_fix_layers=2))]
    for i, atom in enumerate(a):
        if atom.symbol == "H":
            cons.append(FixCartesian(i, [True, True, False]))
    a.set_constraint(cons)
    nsteps = 0 if steps is None else int(steps)
    if nsteps > 0:
        BFGS(a, logfile=None).run(fmax=fmax, steps=nsteps)
    E = a.get_potential_energy()
    meta = {
        "relaxation_scope": "metal_legacy_zonly",
        "n_fix_layers": 2,
        "fixed_atom_count": int(np.count_nonzero(make_bottom_fix_mask(a, n_fix_layers=2))),
        "relaxed_atom_count": int(len(a) - np.count_nonzero(make_bottom_fix_mask(a, n_fix_layers=2))),
        "n_steps": int(nsteps),
        "converged": None,
        "elapsed_s": 0.0,
        "error": "",
    }
    if return_meta:
        return a, E, meta
    return a, E


def _relax_freeH_metal_legacy(atoms, steps=None, fmax=0.03, return_meta: bool = False):
    """Exact legacy metal partial relaxation (bottom layers fixed only)."""
    a = ensure_pbc3(atoms)
    a.calc = calc
    a.set_constraint([FixAtoms(mask=make_bottom_fix_mask(a, n_fix_layers=2))])
    nsteps = 0 if steps is None else int(steps)
    if nsteps > 0:
        BFGS(a, logfile=None).run(fmax=fmax, steps=nsteps)
    E = a.get_potential_energy()
    meta = {
        "relaxation_scope": "metal_legacy_partial",
        "n_fix_layers": 2,
        "fixed_atom_count": int(np.count_nonzero(make_bottom_fix_mask(a, n_fix_layers=2))),
        "relaxed_atom_count": int(len(a) - np.count_nonzero(make_bottom_fix_mask(a, n_fix_layers=2))),
        "n_steps": int(nsteps),
        "converged": None,
        "elapsed_s": 0.0,
        "error": "",
    }
    if return_meta:
        return a, E, meta
    return a, E


def _site_energy_two_stage_metal_legacy(at_relaxed, xy, h0s=H0S, z_steps=None, free_steps=None, return_meta: bool = False):
    """Exact legacy metal two-stage relaxation path."""
    if z_steps is None or free_steps is None:
        raise ValueError("site_energy_two_stage: z_steps and free_steps must be specified.")

    best = {"E": None, "atoms": None, "disp_xy": None, "meta": None}
    xy = np.array(xy, float)
    for h0 in h0s:
        A0 = _put_H_metal_legacy(at_relaxed, xy, height=h0)
        Az, _, z_meta = _relax_zonly_metal_legacy(A0, steps=z_steps, fmax=0.05, return_meta=True)
        Af, Ef, free_meta = _relax_freeH_metal_legacy(Az, steps=free_steps, fmax=0.03, return_meta=True)
        hi = [i for i, atom in enumerate(Af) if atom.symbol == "H"][0]
        disp_xy = float(np.linalg.norm(Af[hi].position[:2] - xy))
        trial_meta = {
            "selected_h0": float(h0),
            "h0_candidates": ";".join(f"{float(v):.2f}" for v in np.asarray(h0s, dtype=float).reshape(-1)),
            "placement_mode": "slab_top",
            "z_relax_n_steps": int((z_meta or {}).get("n_steps", 0)),
            "z_relax_converged": (z_meta or {}).get("converged", None),
            "z_relax_elapsed_s": float((z_meta or {}).get("elapsed_s", 0.0)),
            "z_relax_relaxed_atoms": int((z_meta or {}).get("relaxed_atom_count", 0)),
            "z_relax_fixed_atoms": int((z_meta or {}).get("fixed_atom_count", 0)),
            "fine_relax_scope": "metal_legacy_partial",
            "fine_n_fix_layers": 2,
            "fine_relax_n_steps": int((free_meta or {}).get("n_steps", 0)),
            "fine_relax_converged": (free_meta or {}).get("converged", None),
            "fine_relax_elapsed_s": float((free_meta or {}).get("elapsed_s", 0.0)),
            "fine_relax_relaxed_atoms": int((free_meta or {}).get("relaxed_atom_count", 0)),
            "fine_relax_fixed_atoms": int((free_meta or {}).get("fixed_atom_count", 0)),
            "total_relax_n_steps": int((z_meta or {}).get("n_steps", 0)) + int((free_meta or {}).get("n_steps", 0)),
        }
        if best["E"] is None or Ef < best["E"]:
            best.update({"E": Ef, "atoms": Af, "disp_xy": disp_xy, "meta": trial_meta})
    if return_meta:
        return best["atoms"], best["E"], best["disp_xy"], (best["meta"] or {})
    return best["atoms"], best["E"], best["disp_xy"]


def relax_zonly(atoms, steps=None, fmax=0.05, return_meta: bool = False):
    """
    Relax H in z-direction only while fixing all slab atoms.
    If steps is None or 0, evaluate energy without geometry change.
    """
    a = ensure_pbc3(atoms)
    a.calc = calc
    cons = []
    mask_non_h = make_non_h_fix_mask(a)
    if mask_non_h.any():
        cons.append(FixAtoms(mask=mask_non_h))
    for i, atom in enumerate(a):
        if atom.symbol == "H":
            cons.append(FixCartesian(i, [True, True, False]))
    a.set_constraint(cons)
    nsteps = 0 if steps is None else int(steps)
    n_done, conv, elapsed, err = _run_bfgs_with_meta(a, fmax=float(fmax), steps=nsteps)
    E = a.get_potential_energy()
    meta = _relaxation_meta(
        a,
        relaxation_scope="rigid_zonly",
        n_fix_layers=int(max(0, len(layer_indices(a, n=2)))) if len(a) > 0 else 0,
        elapsed_s=elapsed,
        n_steps=n_done,
        converged=conv,
        error=err,
    )
    if return_meta:
        return a, E, meta
    return a, E


def relax_freeH(atoms, steps=None, fmax=0.03, relaxation_scope: str = "partial", n_fix_layers: int = 2, return_meta: bool = False):
    """
    Relax an H/slab structure using a selected slab-constraint scope.

    Parameters
    ----------
    relaxation_scope : {'rigid', 'partial', 'full'}
        rigid   -> fix all non-H slab atoms; relax H only
        partial -> fix bottom n_fix_layers layers; relax upper slab + H
        full    -> relax all atoms freely
    n_fix_layers : int
        Number of bottom layers fixed when relaxation_scope='partial'.
    """
    a = ensure_pbc3(atoms)
    a.calc = calc
    scope = normalize_relaxation_scope(relaxation_scope)
    cons = build_relax_constraints(a, relaxation_scope=scope, n_fix_layers=int(n_fix_layers))
    a.set_constraint(cons)
    nsteps = 0 if steps is None else int(steps)
    n_done, conv, elapsed, err = _run_bfgs_with_meta(a, fmax=float(fmax), steps=nsteps)
    E = a.get_potential_energy()
    meta = _relaxation_meta(
        a,
        relaxation_scope=scope,
        n_fix_layers=int(n_fix_layers),
        elapsed_s=elapsed,
        n_steps=n_done,
        converged=conv,
        error=err,
    )
    if return_meta:
        return a, E, meta
    return a, E


def site_energy_two_stage(
    at_relaxed,
    xy,
    h0s=H0S,
    z_steps=None,
    free_steps=None,
    relaxation_scope: str = "partial",
    n_fix_layers: int = 2,
    return_meta: bool = False,
    anchor_xyz=None,
    mtype: str | None = None,
    site_kind: str | None = None,
    min_clearance: float | None = None,
):
    """
    Two-stage relaxation: z-only (coarse) then scope-controlled fine relax.
    For metal surfaces, the exact legacy benchmark path is preserved.
    """
    if str(mtype or "").strip().lower() == "metal":
        return _site_energy_two_stage_metal_legacy(
            at_relaxed,
            xy,
            h0s=tuple(float(x) for x in np.asarray(h0s, dtype=float).reshape(-1)),
            z_steps=z_steps,
            free_steps=free_steps,
            return_meta=return_meta,
        )

    if z_steps is None or free_steps is None:
        raise ValueError("site_energy_two_stage: z_steps and free_steps must be specified.")

    scope = normalize_relaxation_scope(relaxation_scope)
    best = {"E": None, "atoms": None, "disp_xy": None, "meta": None}
    xy = np.array(xy, float)
    h0s_eff = recommend_h0s_for_relaxation(scope, mtype=mtype, site_kind=site_kind, default_h0s=h0s)
    if min_clearance is None:
        min_clearance = default_min_clearance_for_scope(scope, mtype=mtype)
    base_z = None
    if anchor_xyz is not None:
        try:
            _arr = np.asarray(anchor_xyz, dtype=float).reshape(-1)
            if _arr.size >= 3 and np.isfinite(_arr[2]):
                base_z = float(_arr[2])
        except Exception:
            base_z = None
    for h0 in h0s_eff:
        A0 = put_H(at_relaxed, xy, height=h0, min_clearance=float(min_clearance), base_z=base_z if scope == "rigid" else None, mtype=mtype)
        Az, _, z_meta = relax_zonly(A0, steps=z_steps, fmax=0.05, return_meta=True)
        Af, Ef, free_meta = relax_freeH(
            Az,
            steps=free_steps,
            fmax=0.03,
            relaxation_scope=scope,
            n_fix_layers=int(n_fix_layers),
            return_meta=True,
        )
        hi = [i for i, atom in enumerate(Af) if atom.symbol == "H"][0]
        disp_xy = float(np.linalg.norm(Af[hi].position[:2] - xy))
        trial_meta = {
            "selected_h0": float(h0),
            "h0_candidates": ";".join(f"{float(v):.2f}" for v in h0s_eff),
            "placement_mode": "anchor_z" if (scope == "rigid" and base_z is not None) else "slab_top",
            "z_relax_n_steps": int((z_meta or {}).get("n_steps", 0)),
            "z_relax_converged": (z_meta or {}).get("converged", None),
            "z_relax_elapsed_s": float((z_meta or {}).get("elapsed_s", 0.0)),
            "z_relax_relaxed_atoms": int((z_meta or {}).get("relaxed_atom_count", 0)),
            "z_relax_fixed_atoms": int((z_meta or {}).get("fixed_atom_count", 0)),
            "fine_relax_scope": str((free_meta or {}).get("relaxation_scope", scope)),
            "fine_n_fix_layers": int((free_meta or {}).get("n_fix_layers", int(n_fix_layers))),
            "fine_relax_n_steps": int((free_meta or {}).get("n_steps", 0)),
            "fine_relax_converged": (free_meta or {}).get("converged", None),
            "fine_relax_elapsed_s": float((free_meta or {}).get("elapsed_s", 0.0)),
            "fine_relax_relaxed_atoms": int((free_meta or {}).get("relaxed_atom_count", 0)),
            "fine_relax_fixed_atoms": int((free_meta or {}).get("fixed_atom_count", 0)),
            "total_relax_n_steps": int((z_meta or {}).get("n_steps", 0)) + int((free_meta or {}).get("n_steps", 0)),
        }
        if best["E"] is None or Ef < best["E"]:
            best.update({"E": Ef, "atoms": Af, "disp_xy": disp_xy, "meta": trial_meta})
    if return_meta:
        return best["atoms"], best["E"], best["disp_xy"], (best["meta"] or {})
    return best["atoms"], best["E"], best["disp_xy"]


def relax_anchor_oh(atoms, anchor_index: int, steps=None, fmax=0.03, return_meta: bool = False, shell_margin: float = 0.20, hook_k: float = 10.0, hook_extra: float = 0.20):
    """Relax D1 local hydroxylation with H z-only and Hookean-supported anchor O."""
    a = ensure_pbc3(atoms)
    a.calc = calc
    cons, local_meta = build_anchor_oh_constraints(
        a,
        anchor_index=int(anchor_index),
        h_index=len(a) - 1,
        shell_margin=float(shell_margin),
        hook_k=float(hook_k),
        hook_extra=float(hook_extra),
    )
    a.set_constraint(cons)
    nsteps = 0 if steps is None else int(steps)
    n_done, conv, elapsed, err = _run_bfgs_with_meta(a, fmax=float(fmax), steps=nsteps)
    E = a.get_potential_energy()
    meta = _relaxation_meta(
        a,
        relaxation_scope='anchor_oh_hookean',
        n_fix_layers=0,
        elapsed_s=elapsed,
        n_steps=n_done,
        converged=conv,
        error=err,
    )
    meta.update(local_meta or {})
    if return_meta:
        return a, E, meta
    return a, E


def site_energy_oh_anchoronly(
    at_relaxed,
    xy,
    anchor_index: int,
    h0s=H0S,
    z_steps=None,
    free_steps=None,
    return_meta: bool = False,
    shell_margin: float = 0.20,
    hook_k: float = 10.0,
    hook_extra: float = 0.20,
):
    """D1 hydroxylation descriptor with two-step logic.

    Step 1: slab fixed, H z-only search on each O-top seed.
    Step 2: on the selected O anchor, relax the explicit OH state with
            Hookean O–metal support so the OH remains surface-bound.
    """
    if z_steps is None or free_steps is None:
        raise ValueError('site_energy_oh_anchoronly: z_steps and free_steps must be specified.')

    best = {'E': None, 'atoms': None, 'disp_xy': None, 'meta': None}
    xy = np.array(xy, float)
    for h0 in h0s:
        A0 = put_H(at_relaxed, xy, height=h0)
        Az, _, z_meta = relax_zonly(A0, steps=z_steps, fmax=0.05, return_meta=True)
        Af, E, local_meta = relax_anchor_oh(
            Az,
            anchor_index=int(anchor_index),
            steps=free_steps,
            fmax=0.03,
            return_meta=True,
            shell_margin=float(shell_margin),
            hook_k=float(hook_k),
            hook_extra=float(hook_extra),
        )
        hi = [i for i, atom in enumerate(Af) if atom.symbol == 'H'][0]
        disp_xy = float(np.linalg.norm(Af[hi].position[:2] - xy))
        trial_meta = {
            'selected_h0': float(h0),
            'anchor_index': int(anchor_index),
            'local_free_count': int((local_meta or {}).get('local_free_count', 0)),
            'shell_count': int((local_meta or {}).get('shell_count', 0)),
            'hook_k': float((local_meta or {}).get('hook_k', 0.0)),
            'hook_extra': float((local_meta or {}).get('hook_extra', 0.0)),
            'z_relax_n_steps': int((z_meta or {}).get('n_steps', 0)),
            'z_relax_converged': (z_meta or {}).get('converged', None),
            'z_relax_elapsed_s': float((z_meta or {}).get('elapsed_s', 0.0)),
            'z_relax_relaxed_atoms': int((z_meta or {}).get('relaxed_atom_count', 0)),
            'z_relax_fixed_atoms': int((z_meta or {}).get('fixed_atom_count', 0)),
            'fine_relax_scope': 'anchor_oh_hookean',
            'fine_n_fix_layers': 0,
            'fine_relax_n_steps': int((local_meta or {}).get('n_steps', 0)),
            'fine_relax_converged': (local_meta or {}).get('converged', None),
            'fine_relax_elapsed_s': float((local_meta or {}).get('elapsed_s', 0.0)),
            'fine_relax_relaxed_atoms': int((local_meta or {}).get('relaxed_atom_count', 0)),
            'fine_relax_fixed_atoms': int((local_meta or {}).get('fixed_atom_count', 0)),
            'total_relax_n_steps': int((z_meta or {}).get('n_steps', 0)) + int((local_meta or {}).get('n_steps', 0)),
        }
        if best['E'] is None or E < best['E']:
            best.update({'E': E, 'atoms': Af, 'disp_xy': disp_xy, 'meta': trial_meta})
    if return_meta:
        return best['atoms'], best['E'], best['disp_xy'], (best['meta'] or {})
    return best['atoms'], best['E'], best['disp_xy']


def site_energy_oh_constrained(
    at_relaxed,
    xy,
    h0s=H0S,
    z_steps=None,
    return_meta: bool = False,
):
    """Legacy z-only constrained OH descriptor kept for backward compatibility."""
    if z_steps is None:
        raise ValueError("site_energy_oh_constrained: z_steps must be specified.")

    best = {"E": None, "atoms": None, "disp_xy": None, "meta": None}
    xy = np.array(xy, float)
    for h0 in h0s:
        A0 = put_H(at_relaxed, xy, height=h0)
        Az, E, z_meta = relax_zonly(A0, steps=z_steps, fmax=0.05, return_meta=True)
        hi = [i for i, atom in enumerate(Az) if atom.symbol == "H"][0]
        disp_xy = float(np.linalg.norm(Az[hi].position[:2] - xy))
        trial_meta = {
            "selected_h0": float(h0),
            "h0_candidates": ";".join(f"{float(v):.2f}" for v in h0s_eff),
            "placement_mode": "anchor_z" if (scope == "rigid" and base_z is not None) else "slab_top",
            "z_relax_n_steps": int((z_meta or {}).get("n_steps", 0)),
            "z_relax_converged": (z_meta or {}).get("converged", None),
            "z_relax_elapsed_s": float((z_meta or {}).get("elapsed_s", 0.0)),
            "z_relax_relaxed_atoms": int((z_meta or {}).get("relaxed_atom_count", 0)),
            "z_relax_fixed_atoms": int((z_meta or {}).get("fixed_atom_count", 0)),
            "fine_relax_scope": "constrained_oh_zonly",
            "fine_n_fix_layers": 0,
            "fine_relax_n_steps": 0,
            "fine_relax_converged": True,
            "fine_relax_elapsed_s": 0.0,
            "fine_relax_relaxed_atoms": 0,
            "fine_relax_fixed_atoms": int(len(Az) - 1),
            "total_relax_n_steps": int((z_meta or {}).get("n_steps", 0)),
        }
        if best["E"] is None or E < best["E"]:
            best.update({"E": E, "atoms": Az, "disp_xy": disp_xy, "meta": trial_meta})
    if return_meta:
        return best["atoms"], best["E"], best["disp_xy"], (best["meta"] or {})
    return best["atoms"], best["E"], best["disp_xy"]


def factor_near_square(k: int):
    """Decompose integer k into near-square factors nx * ny."""
    r = int(np.floor(np.sqrt(k)))
    for nx in range(r, 0, -1):
        if k % nx == 0:
            return nx, k // nx
    nx = max(1, int(round(np.sqrt(k))))
    ny = max(1, int(round(k / nx)))
    return nx, ny


__all__ = [
    "MODEL_NAME", "DEVICE", "ZPE_CORR", "TDS_CORR", "NET_CORR",
    "H0S", "MIGRATE_THR", "VAC_WARN_MIN", "UNUSUAL_DDELTA",
    "calc",
    "ensure_pbc3", "layer_indices", "first_layer_min_distance",
    "put_H", "make_bottom_fix_mask", "make_non_h_fix_mask",
    "normalize_relaxation_scope", "build_relax_constraints",
    "relax_zonly", "relax_freeH",
    "site_energy_two_stage", "relax_anchor_oh", "site_energy_oh_anchoronly", "site_energy_oh_constrained", "factor_near_square",
]