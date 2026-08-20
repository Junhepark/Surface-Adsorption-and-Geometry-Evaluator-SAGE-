from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from ase import Atoms

from .environment import analyze_parent_slab
from .ionic_radii import ionic_radius
from .material import species_role


def _mic_vector(atoms: Atoms, i: int, j: int) -> np.ndarray:
    try:
        return np.asarray(atoms.get_distance(int(i), int(j), mic=True, vector=True), dtype=float)
    except Exception:
        return np.asarray(atoms.positions[int(j)] - atoms.positions[int(i)], dtype=float)


def _distance(atoms: Atoms, i: int, j: int) -> float:
    try:
        return float(atoms.get_distance(int(i), int(j), mic=True))
    except Exception:
        return float(np.linalg.norm(atoms.positions[int(j)] - atoms.positions[int(i)]))


def _protected_bottom_indices(analysis: Dict[str, object], protect_bottom_layers: int) -> set[int]:
    layers = list(analysis.get("layers", []) or [])
    n = max(0, int(protect_bottom_layers))
    out = set()
    for layer in layers[max(0, len(layers) - n):]:
        out.update(int(i) for i in layer)
    return out


def _opposite_sublattice_candidates(
    atoms: Atoms,
    *,
    target_index: int,
    target_role: str,
) -> List[tuple[int, float]]:
    candidates = []
    for j, symbol in enumerate(atoms.get_chemical_symbols()):
        if int(j) == int(target_index):
            continue
        role = species_role(str(symbol), atoms)
        if target_role == "cation":
            keep = role == "anion"
        elif target_role == "anion":
            keep = role == "cation"
        else:
            keep = False
        if not keep:
            continue
        d = _distance(atoms, int(target_index), int(j))
        if np.isfinite(d) and 0.5 < d <= 3.6:
            candidates.append((int(j), float(d)))
    return sorted(candidates, key=lambda x: (x[1], x[0]))


def _first_coordination_shell(
    atoms: Atoms,
    *,
    target_index: int,
    target_role: str,
) -> List[int]:
    candidates = _opposite_sublattice_candidates(
        atoms,
        target_index=int(target_index),
        target_role=str(target_role),
    )
    if not candidates:
        return []

    d0 = float(candidates[0][1])
    cutoff = min(3.4, max(d0 * 1.25, d0 + 0.40))
    selected: List[int] = []
    previous = None
    for idx, d in candidates:
        if d > cutoff:
            break
        if previous is not None and len(selected) >= 2 and (d - previous) > 0.35:
            break
        selected.append(int(idx))
        previous = float(d)
    return selected


def _coordination_number_for_radius(
    atoms: Atoms,
    *,
    target_index: int,
    target_role: str,
) -> int:
    return max(
        1,
        len(_first_coordination_shell(
            atoms,
            target_index=int(target_index),
            target_role=str(target_role),
        )),
    )


def _ligand_sharing_count(
    atoms: Atoms,
    *,
    ligand_index: int,
    central_role: str,
) -> int:
    opposite_role = "cation" if central_role == "cation" else "anion"
    return len(_first_coordination_shell(
        atoms,
        target_index=int(ligand_index),
        target_role=opposite_role,
    ))


def oxide_polyhedron_substitution_initialization(
    atoms: Atoms,
    *,
    target_index: int,
    host: str,
    dopant: str,
    host_oxidation_state: int | float,
    dopant_oxidation_state: int | float,
    strength: float = 0.25,
    shared_ligand_weight: float = 0.50,
    anion_substitution_weight: float = 0.35,
    max_displacement_A: float = 0.12,
    protect_bottom_layers: int = 1,
) -> tuple[Atoms, Dict[str, object]]:
    """Initialize an oxide substitution by adjusting one coordination polyhedron.

    Cation substitution:
        Move directly coordinated anion ligands only.
        Shared ligands are damped.

    Anion substitution:
        Move directly coordinated cations only, with an additional damping
        because each cation participates in several polyhedra.

    The dopant stays exactly on the parent lattice site. Cell vectors and all
    atoms outside the first opposite-sublattice coordination shell are retained.
    """
    if atoms is None or len(atoms) == 0:
        raise ValueError("Parent structure is empty.")
    idx = int(target_index)
    if idx < 0 or idx >= len(atoms):
        raise IndexError(f"Target atom index {idx} is outside the parent structure.")
    if str(atoms[idx].symbol) != str(host):
        raise ValueError(f"Target atom {idx} is {atoms[idx].symbol}, not {host}.")

    target_role = species_role(str(host), atoms)
    dopant_role = species_role(str(dopant), atoms)
    child = atoms.copy()
    child[idx].symbol = str(dopant)
    center = np.asarray(atoms.positions[idx], dtype=float).copy()

    analysis = analyze_parent_slab(atoms)
    protected = _protected_bottom_indices(analysis, int(protect_bottom_layers))
    ligands = _first_coordination_shell(
        atoms,
        target_index=idx,
        target_role=target_role,
    )
    cn = max(1, len(ligands))

    host_radius = ionic_radius(str(host), host_oxidation_state, cn)
    dopant_radius = ionic_radius(str(dopant), dopant_oxidation_state, cn)
    r_host = host_radius.get("radius_A")
    r_dopant = dopant_radius.get("radius_A")

    metadata: Dict[str, object] = {
        "material_class": "oxide",
        "method": "oxide_coordination_polyhedron_initialization",
        "applied": False,
        "host_role": target_role,
        "dopant_role": dopant_role,
        "cross_sublattice": bool(target_role != dopant_role),
        "host_oxidation_state": float(host_oxidation_state),
        "dopant_oxidation_state": float(dopant_oxidation_state),
        "charge_mismatch": float(dopant_oxidation_state) - float(host_oxidation_state),
        "coordination_number": int(cn),
        "host_ionic_radius": host_radius,
        "dopant_ionic_radius": dopant_radius,
        "radius_source": (
            f"{host_radius.get('source')}|{dopant_radius.get('source')}"
        ),
        "signed_radius_mismatch_fraction": None,
        "absolute_radius_mismatch_fraction": None,
        "strength": float(strength),
        "shared_ligand_weight": float(shared_ligand_weight),
        "anion_substitution_weight": float(anion_substitution_weight),
        "max_displacement_limit_A": float(max_displacement_A),
        "protect_bottom_layers": int(protect_bottom_layers),
        "first_shell_parent_indices": tuple(int(i) for i in ligands),
        "n_first_shell": int(len(ligands)),
        "n_moved_atoms": 0,
        "moved_parent_indices": (),
        "protected_skipped_indices": (),
        "capped_displacement_indices": (),
        "max_applied_displacement_A": 0.0,
        "mean_applied_displacement_A": 0.0,
        "mean_first_shell_distance_before_A": None,
        "mean_first_shell_distance_after_A": None,
        "movement_records": [],
        "initialization_warning": None,
    }

    if target_role not in {"cation", "anion"}:
        metadata["initialization_warning"] = "Target sublattice role could not be classified."
        child.positions[idx] = center
        return child, metadata

    if target_role != dopant_role:
        metadata["initialization_warning"] = (
            "Cross-sublattice substitution detected; geometry was kept at the host lattice positions."
        )
        child.positions[idx] = center
        return child, metadata

    if r_host is None or r_dopant is None:
        metadata["initialization_warning"] = (
            "An ionic radius was unavailable for the selected oxidation-state/coordination model; "
            "geometry was kept at the host lattice positions."
        )
        child.positions[idx] = center
        return child, metadata

    r_host = float(r_host)
    r_dopant = float(r_dopant)
    signed_mismatch = (r_dopant - r_host) / max(r_host, 1e-12)
    metadata["signed_radius_mismatch_fraction"] = float(signed_mismatch)
    metadata["absolute_radius_mismatch_fraction"] = float(abs(signed_mismatch))

    strength = float(np.clip(float(strength), 0.0, 1.0))
    shared_ligand_weight = float(np.clip(float(shared_ligand_weight), 0.0, 1.0))
    anion_substitution_weight = float(np.clip(float(anion_substitution_weight), 0.0, 1.0))
    max_displacement_A = max(0.0, float(max_displacement_A))

    records = []
    moved = []
    skipped = []
    capped = []
    before_values = []
    after_values = []
    displacements = []

    ionic_delta = float(r_dopant - r_host)
    for j in ligands:
        if int(j) in protected:
            skipped.append(int(j))
            continue

        vec = _mic_vector(atoms, idx, int(j))
        d_before = float(np.linalg.norm(vec))
        if not np.isfinite(d_before) or d_before <= 1e-8:
            continue

        sharing_count = _ligand_sharing_count(
            atoms,
            ligand_index=int(j),
            central_role=target_role,
        )
        if target_role == "cation":
            topology_weight = 1.0 if sharing_count <= 1 else shared_ligand_weight
        else:
            topology_weight = anion_substitution_weight

        delta = strength * topology_weight * ionic_delta
        if abs(delta) > max_displacement_A:
            delta = float(np.sign(delta) * max_displacement_A)
            capped.append(int(j))

        displacement = vec / d_before * delta
        child.positions[int(j)] += displacement
        d_after = _distance(child, idx, int(j))
        disp = float(np.linalg.norm(displacement))
        if disp > 1e-10:
            moved.append(int(j))

        before_values.append(d_before)
        after_values.append(d_after)
        displacements.append(disp)
        records.append({
            "atom_index": int(j),
            "symbol": str(atoms[int(j)].symbol),
            "role": species_role(str(atoms[int(j)].symbol), atoms),
            "distance_before_A": d_before,
            "distance_after_A": d_after,
            "ionic_radius_delta_A": ionic_delta,
            "topology_weight": float(topology_weight),
            "sharing_count": int(sharing_count),
            "displacement_A": disp,
        })

    child.positions[idx] = center
    try:
        child.wrap()
    except Exception:
        pass

    metadata.update({
        "applied": bool(moved),
        "moved_parent_indices": tuple(sorted(moved)),
        "protected_skipped_indices": tuple(sorted(skipped)),
        "capped_displacement_indices": tuple(sorted(capped)),
        "n_moved_atoms": int(len(moved)),
        "max_applied_displacement_A": max(displacements) if displacements else 0.0,
        "mean_applied_displacement_A": float(np.mean(displacements)) if displacements else 0.0,
        "mean_first_shell_distance_before_A": float(np.mean(before_values)) if before_values else None,
        "mean_first_shell_distance_after_A": float(np.mean(after_values)) if after_values else None,
        "movement_records": records,
    })
    return child, metadata
