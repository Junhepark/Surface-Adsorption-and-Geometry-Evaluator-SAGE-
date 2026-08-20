from __future__ import annotations

from typing import List, Sequence

from ase import Atoms
from ase.data import atomic_numbers

from .environment import analyze_parent_slab, structure_content_signature, select_indices
from .equivalence import group_equivalent_atom_indices, choose_center_preferred_atom_index
from .models import EngineeredStructure, StructureRecipe
from .validation import validate_engineered_structure


def _build_vacancy_candidate(
    atoms: Atoms,
    *,
    element: str,
    target_index: int,
    selection_mode: str,
    orbit_id: int | None = None,
    equivalent_parent_indices: Sequence[int] = (),
    requested_depth: str | None = None,
) -> EngineeredStructure:
    if str(element) not in atomic_numbers:
        raise ValueError("Vacancy element must be a valid element symbol.")
    idx = int(target_index)
    if idx < 0 or idx >= len(atoms):
        raise IndexError(f"Target atom index {idx} is outside the parent structure.")
    if atoms[idx].symbol != str(element):
        raise ValueError(
            f"Selected atom {idx} is {atoms[idx].symbol}, not the requested vacancy element {element}."
        )

    analysis = analyze_parent_slab(atoms)
    env = analysis["environment_by_index"][idx]
    if requested_depth and requested_depth not in {"all", "surface+subsurface"}:
        if env.depth_class != str(requested_depth):
            raise ValueError(
                f"Selected atom {idx} is {env.depth_class}, not {requested_depth}."
            )
    elif requested_depth == "surface+subsurface" and env.depth_class not in {"surface", "subsurface"}:
        raise ValueError(f"Selected atom {idx} is not in the surface/subsurface region.")

    eligible = select_indices(analysis, element=str(element), depth=str(requested_depth or env.depth_class))
    denominator = max(len(eligible), 1)
    eff = 1.0 / float(denominator)
    parent_neighbors = tuple(int(i) for i in analysis["neighbor_map"].get(idx, ()))

    child = atoms.copy()
    del child[idx]
    parent_to_child = {}
    for pidx in range(len(atoms)):
        if pidx == idx:
            parent_to_child[int(pidx)] = None
        elif pidx < idx:
            parent_to_child[int(pidx)] = int(pidx)
        else:
            parent_to_child[int(pidx)] = int(pidx - 1)

    parameters = {
        "parent_structure_signature": structure_content_signature(atoms),
        "element": str(element),
        "depth": str(env.depth_class),
        "requested_depth": str(requested_depth or env.depth_class),
        "selection_mode": str(selection_mode),
        "selected_parent_index": idx,
        "parent_neighbor_indices": parent_neighbors,
        "effective_fraction": float(eff),
    }
    if orbit_id is not None:
        parameters["orbit_id"] = int(orbit_id)
    if equivalent_parent_indices:
        parameters["equivalent_parent_indices"] = tuple(int(i) for i in equivalent_parent_indices)

    recipe = StructureRecipe(
        operation="single_vacancy",
        parent_formula=atoms.get_chemical_formula(),
        parent_n_atoms=len(atoms),
        target_indices=(idx,),
        target_environment=env.as_dict(),
        removed_elements=(str(element),),
        parameters=parameters,
    )
    mode_tag = "manual" if str(selection_mode) == "manual" else f"orbit{int(orbit_id or 0)}"
    cid = f"vac_{element}_{env.depth_class}_{mode_tag}_{recipe.stable_id()}"
    validation = validate_engineered_structure(
        child,
        parent_atoms=atoms,
        operation="vacancy",
        modified_indices=(),
        effective_fraction=eff,
    )
    label_suffix = f"atom {idx}" if str(selection_mode) == "manual" else f"orbit {int(orbit_id or 0)}"
    return EngineeredStructure(
        atoms=child,
        recipe=recipe,
        candidate_id=cid,
        label=f"V_{element} | {env.depth_class} | CN {env.coordination_number} | {label_suffix}",
        validation=validation,
        atom_provenance={
            "parent_to_child": parent_to_child,
            "removed_parent_index": idx,
            "removed_parent_neighbors": parent_neighbors,
        },
    )


def build_vacancy_candidate_at_index(
    atoms: Atoms,
    *,
    element: str,
    target_index: int,
    depth: str | None = None,
) -> EngineeredStructure:
    """Build one user-selected vacancy candidate at an exact parent atom index."""
    return _build_vacancy_candidate(
        atoms,
        element=element,
        target_index=int(target_index),
        selection_mode="manual",
        requested_depth=depth,
    )


def enumerate_vacancy_candidates(
    atoms: Atoms,
    *,
    element: str,
    depth: str = "surface",
    max_candidates: int = 20,
) -> List[EngineeredStructure]:
    if str(element) not in atomic_numbers:
        raise ValueError("Vacancy element must be a valid element symbol.")

    analysis = analyze_parent_slab(atoms)
    eligible = select_indices(analysis, element=str(element), depth=str(depth))
    if not eligible:
        raise ValueError(f"No {element} atoms found in depth selection '{depth}'.")

    groups = group_equivalent_atom_indices(analysis["environment_by_index"], eligible)
    out: List[EngineeredStructure] = []
    for orbit_id, orbit in enumerate(groups[: int(max_candidates)]):
        idx = choose_center_preferred_atom_index(atoms, orbit)
        out.append(
            _build_vacancy_candidate(
                atoms,
                element=str(element),
                target_index=idx,
                selection_mode="automatic",
                orbit_id=int(orbit_id),
                equivalent_parent_indices=orbit,
                requested_depth=str(depth),
            )
        )
    return out
