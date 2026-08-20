from __future__ import annotations

from typing import Dict

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from ocp_app.core.structure_check import COVALENT_RADII

from .environment import analyze_parent_slab


def element_covalent_radius(symbol: str) -> float:
    sym = str(symbol)
    if sym in COVALENT_RADII:
        return float(COVALENT_RADII[sym])
    z = atomic_numbers.get(sym)
    if z is None:
        raise ValueError(f"Unknown element symbol: {sym}")
    value = float(covalent_radii[int(z)])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"No usable covalent radius is available for {sym}.")
    return value


def substitution_radius_diagnostics(host: str, dopant: str) -> Dict[str, float]:
    r_host = element_covalent_radius(str(host))
    r_dopant = element_covalent_radius(str(dopant))
    signed = (r_dopant - r_host) / max(r_host, 1e-12)
    return {
        "host_radius_A": float(r_host),
        "dopant_radius_A": float(r_dopant),
        "signed_radius_mismatch_fraction": float(signed),
        "absolute_radius_mismatch_fraction": float(abs(signed)),
        "radius_source": "covalent_radius",
    }


def _mic_vector(atoms: Atoms, i: int, j: int) -> np.ndarray:
    try:
        return np.asarray(atoms.get_distance(int(i), int(j), mic=True, vector=True), dtype=float)
    except Exception:
        return np.asarray(atoms.positions[int(j)] - atoms.positions[int(i)], dtype=float)


def _pair_distance(atoms: Atoms, i: int, j: int) -> float:
    try:
        return float(atoms.get_distance(int(i), int(j), mic=True))
    except Exception:
        return float(np.linalg.norm(atoms.positions[int(j)] - atoms.positions[int(i)]))


def _protected_bottom_indices(analysis: Dict[str, object], protect_bottom_layers: int) -> set[int]:
    layers = list(analysis.get("layers", []) or [])
    n = max(0, int(protect_bottom_layers))
    protected = set()
    for layer in layers[max(0, len(layers) - n):]:
        protected.update(int(i) for i in layer)
    return protected


def metallic_radius_guided_substitution_initialization(
    atoms: Atoms,
    *,
    target_index: int,
    host: str,
    dopant: str,
    strength: float = 0.50,
    shells: int = 2,
    second_shell_weight: float = 0.35,
    second_shell_factor: float = 1.65,
    max_pair_scale_change: float = 0.12,
    max_displacement_A: float = 0.20,
    protect_bottom_layers: int = 1,
) -> tuple[Atoms, Dict[str, object]]:
    """Bounded metallic-neighbor initialization for a substitutional alloy."""
    if atoms is None or len(atoms) == 0:
        raise ValueError("Parent structure is empty.")
    idx = int(target_index)
    if idx < 0 or idx >= len(atoms):
        raise IndexError(f"Target atom index {idx} is outside the parent structure.")
    if str(atoms[idx].symbol) != str(host):
        raise ValueError(f"Target atom {idx} is {atoms[idx].symbol}, not {host}.")

    strength = float(np.clip(float(strength), 0.0, 1.0))
    shells = 1 if int(shells) <= 1 else 2
    analysis = analyze_parent_slab(atoms)
    first_shell = set(int(i) for i in analysis.get("neighbor_map", {}).get(idx, ()))
    protected = _protected_bottom_indices(analysis, int(protect_bottom_layers))

    distances = {
        int(j): _pair_distance(atoms, idx, int(j))
        for j in range(len(atoms))
        if int(j) != idx
    }
    finite = [d for d in distances.values() if np.isfinite(d) and d > 1e-8]
    nearest = min(finite) if finite else float("nan")

    if not first_shell and np.isfinite(nearest):
        first_shell = {
            int(j) for j, d in distances.items()
            if d <= float(nearest) * 1.20
        }

    second_cutoff = float(second_shell_factor) * float(nearest) if np.isfinite(nearest) else 0.0
    second_shell = {
        int(j) for j, d in distances.items()
        if int(j) not in first_shell and d <= second_cutoff
    } if shells >= 2 else set()

    r_host = element_covalent_radius(str(host))
    r_dopant = element_covalent_radius(str(dopant))
    child = atoms.copy()
    child[idx].symbol = str(dopant)
    dopant_position = np.asarray(atoms.positions[idx], dtype=float).copy()

    records = []
    moved = []
    skipped = []
    capped = []

    for j in sorted(first_shell | second_shell):
        if int(j) in protected:
            skipped.append(int(j))
            continue
        vec = _mic_vector(atoms, idx, int(j))
        d_before = float(np.linalg.norm(vec))
        if not np.isfinite(d_before) or d_before <= 1e-8:
            continue
        r_neighbor = element_covalent_radius(str(atoms[int(j)].symbol))
        raw_scale = (r_dopant + r_neighbor) / max(r_host + r_neighbor, 1e-12)
        bounded_scale = float(np.clip(raw_scale, 1.0 - max_pair_scale_change, 1.0 + max_pair_scale_change))
        shell = "first" if int(j) in first_shell else "second"
        shell_weight = 1.0 if shell == "first" else float(second_shell_weight)
        delta = float(strength) * shell_weight * d_before * (bounded_scale - 1.0)
        if abs(delta) > float(max_displacement_A):
            delta = float(np.sign(delta) * float(max_displacement_A))
            capped.append(int(j))
        displacement = vec / d_before * delta
        child.positions[int(j)] += displacement
        d_after = _pair_distance(child, idx, int(j))
        if float(np.linalg.norm(displacement)) > 1e-10:
            moved.append(int(j))
        records.append({
            "atom_index": int(j),
            "symbol": str(atoms[int(j)].symbol),
            "shell": shell,
            "distance_before_A": d_before,
            "distance_after_A": d_after,
            "displacement_A": float(np.linalg.norm(displacement)),
            "raw_pair_scale": float(raw_scale),
            "bounded_pair_scale": float(bounded_scale),
        })

    child.positions[idx] = dopant_position
    try:
        child.wrap()
    except Exception:
        pass

    first_records = [r for r in records if r["shell"] == "first"]
    before = [float(r["distance_before_A"]) for r in first_records]
    after = [float(r["distance_after_A"]) for r in first_records]
    disps = [float(r["displacement_A"]) for r in records]
    diagnostics = substitution_radius_diagnostics(str(host), str(dopant))

    return child, {
        "material_class": "metal",
        "method": "metallic_radius_guided_local_initialization",
        "applied": bool(moved),
        **diagnostics,
        "strength": strength,
        "shells": shells,
        "second_shell_weight": float(second_shell_weight),
        "max_displacement_limit_A": float(max_displacement_A),
        "protect_bottom_layers": int(protect_bottom_layers),
        "first_shell_parent_indices": tuple(sorted(first_shell)),
        "second_shell_parent_indices": tuple(sorted(second_shell)),
        "moved_parent_indices": tuple(sorted(moved)),
        "protected_skipped_indices": tuple(sorted(skipped)),
        "capped_displacement_indices": tuple(sorted(capped)),
        "n_first_shell": len(first_shell),
        "n_second_shell": len(second_shell),
        "n_moved_atoms": len(moved),
        "max_applied_displacement_A": max(disps) if disps else 0.0,
        "mean_applied_displacement_A": float(np.mean(disps)) if disps else 0.0,
        "mean_first_shell_distance_before_A": float(np.mean(before)) if before else None,
        "mean_first_shell_distance_after_A": float(np.mean(after)) if after else None,
        "movement_records": records,
    }


# Backward-compatible alias used by the immediately preceding patch.
radius_guided_substitution_initialization = metallic_radius_guided_substitution_initialization
