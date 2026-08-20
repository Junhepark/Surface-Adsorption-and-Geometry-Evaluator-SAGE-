from __future__ import annotations

from typing import List, Sequence

from ase import Atoms
from ase.data import atomic_numbers

from .environment import analyze_parent_slab, structure_content_signature, select_indices
from .equivalence import group_equivalent_atom_indices, choose_center_preferred_atom_index
from .ionic_radii import suggested_substitution_oxidation_states
from .local_geometry import (
    metallic_radius_guided_substitution_initialization,
    substitution_radius_diagnostics,
)
from .material import (
    infer_structure_material_class,
    substitution_sublattice_compatibility,
)
from .models import EngineeredStructure, StructureRecipe
from .oxide_local_geometry import oxide_polyhedron_substitution_initialization
from .validation import validate_engineered_structure


def substitution_geometry_diagnostics(
    atoms: Atoms,
    *,
    host: str,
    dopant: str,
    host_oxidation_state: int | float | None = None,
    dopant_oxidation_state: int | float | None = None,
) -> dict:
    compatibility = substitution_sublattice_compatibility(
        atoms,
        host=str(host),
        dopant=str(dopant),
    )
    material_class = str(compatibility["material_class"])
    if material_class == "metal":
        return {
            **compatibility,
            "geometry_model": "metallic_radius_guided",
            **substitution_radius_diagnostics(str(host), str(dopant)),
        }

    suggestions = suggested_substitution_oxidation_states(
        atoms,
        host=str(host),
        dopant=str(dopant),
        host_role=str(compatibility["host_role"]),
        dopant_role=str(compatibility["dopant_role"]),
    )
    if host_oxidation_state is None:
        host_oxidation_state = suggestions.get("host_oxidation_state")
    if dopant_oxidation_state is None:
        dopant_oxidation_state = suggestions.get("dopant_oxidation_state")
    return {
        **compatibility,
        "geometry_model": "oxide_coordination_polyhedron",
        "host_oxidation_state": host_oxidation_state,
        "dopant_oxidation_state": dopant_oxidation_state,
        "oxidation_state_suggestions": suggestions,
    }


def _initialize_substitution(
    atoms: Atoms,
    *,
    target_index: int,
    host: str,
    dopant: str,
    apply_local_adjustment: bool,
    adjustment_strength: float,
    adjustment_shells: int,
    max_local_displacement_A: float,
    protect_bottom_layers: int,
    host_oxidation_state: int | float | None,
    dopant_oxidation_state: int | float | None,
    shared_ligand_weight: float,
) -> tuple[Atoms, dict]:
    material_class = infer_structure_material_class(atoms)

    if not apply_local_adjustment:
        child = atoms.copy()
        child[int(target_index)].symbol = str(dopant)
        return child, {
            "material_class": material_class,
            "method": "keep_host_lattice_positions",
            "applied": False,
            "strength": 0.0,
            "n_moved_atoms": 0,
            "max_applied_displacement_A": 0.0,
            "mean_applied_displacement_A": 0.0,
            "mean_first_shell_distance_before_A": None,
            "mean_first_shell_distance_after_A": None,
            "movement_records": [],
        }

    if material_class == "oxide":
        compatibility = substitution_sublattice_compatibility(
            atoms,
            host=str(host),
            dopant=str(dopant),
        )
        suggestions = suggested_substitution_oxidation_states(
            atoms,
            host=str(host),
            dopant=str(dopant),
            host_role=str(compatibility["host_role"]),
            dopant_role=str(compatibility["dopant_role"]),
        )
        if host_oxidation_state is None:
            host_oxidation_state = suggestions.get("host_oxidation_state")
        if dopant_oxidation_state is None:
            dopant_oxidation_state = suggestions.get("dopant_oxidation_state")

        child = atoms.copy()
        child[int(target_index)].symbol = str(dopant)
        if host_oxidation_state is None or dopant_oxidation_state is None:
            return child, {
                "material_class": "oxide",
                "method": "oxide_coordination_polyhedron_initialization",
                "applied": False,
                "host_role": compatibility.get("host_role"),
                "dopant_role": compatibility.get("dopant_role"),
                "cross_sublattice": compatibility.get("cross_sublattice"),
                "host_oxidation_state": host_oxidation_state,
                "dopant_oxidation_state": dopant_oxidation_state,
                "initialization_warning": (
                    "Oxidation states could not be resolved; host-lattice positions were retained."
                ),
                "n_moved_atoms": 0,
                "max_applied_displacement_A": 0.0,
                "mean_first_shell_distance_before_A": None,
                "mean_first_shell_distance_after_A": None,
                "movement_records": [],
            }

        return oxide_polyhedron_substitution_initialization(
            atoms,
            target_index=int(target_index),
            host=str(host),
            dopant=str(dopant),
            host_oxidation_state=float(host_oxidation_state),
            dopant_oxidation_state=float(dopant_oxidation_state),
            strength=float(adjustment_strength),
            shared_ligand_weight=float(shared_ligand_weight),
            max_displacement_A=float(max_local_displacement_A),
            protect_bottom_layers=int(protect_bottom_layers),
        )

    return metallic_radius_guided_substitution_initialization(
        atoms,
        target_index=int(target_index),
        host=str(host),
        dopant=str(dopant),
        strength=float(adjustment_strength),
        shells=int(adjustment_shells),
        max_displacement_A=float(max_local_displacement_A),
        protect_bottom_layers=int(protect_bottom_layers),
    )


def _build_substitution_candidate(
    atoms: Atoms,
    *,
    host: str,
    dopant: str,
    target_index: int,
    selection_mode: str,
    orbit_id: int | None = None,
    equivalent_parent_indices: Sequence[int] = (),
    requested_depth: str | None = None,
    apply_local_adjustment: bool = True,
    adjustment_strength: float = 0.50,
    adjustment_shells: int = 2,
    max_local_displacement_A: float = 0.20,
    protect_bottom_layers: int = 1,
    host_oxidation_state: int | float | None = None,
    dopant_oxidation_state: int | float | None = None,
    shared_ligand_weight: float = 0.50,
) -> EngineeredStructure:
    if str(host) not in atomic_numbers or str(dopant) not in atomic_numbers:
        raise ValueError("Host and dopant must be valid element symbols.")
    if str(host) == str(dopant):
        raise ValueError("Host and dopant must be different elements.")
    idx = int(target_index)
    if idx < 0 or idx >= len(atoms):
        raise IndexError(f"Target atom index {idx} is outside the parent structure.")
    if atoms[idx].symbol != str(host):
        raise ValueError(
            f"Selected atom {idx} is {atoms[idx].symbol}, not the requested host {host}."
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

    eligible = select_indices(
        analysis,
        element=str(host),
        depth=str(requested_depth or env.depth_class),
    )
    eff = 1.0 / float(max(len(eligible), 1))

    child, adjustment_meta = _initialize_substitution(
        atoms,
        target_index=idx,
        host=str(host),
        dopant=str(dopant),
        apply_local_adjustment=bool(apply_local_adjustment),
        adjustment_strength=float(adjustment_strength),
        adjustment_shells=int(adjustment_shells),
        max_local_displacement_A=float(max_local_displacement_A),
        protect_bottom_layers=int(protect_bottom_layers),
        host_oxidation_state=host_oxidation_state,
        dopant_oxidation_state=dopant_oxidation_state,
        shared_ligand_weight=float(shared_ligand_weight),
    )

    material_class = str(adjustment_meta.get(
        "material_class",
        infer_structure_material_class(atoms),
    ))
    parameters = {
        "parent_structure_signature": structure_content_signature(atoms),
        "host": str(host),
        "dopant": str(dopant),
        "depth": str(env.depth_class),
        "requested_depth": str(requested_depth or env.depth_class),
        "selection_mode": str(selection_mode),
        "selected_parent_index": idx,
        "effective_fraction": float(eff),
        "material_class": material_class,
        "local_geometry_adjustment": adjustment_meta,
    }
    if orbit_id is not None:
        parameters["orbit_id"] = int(orbit_id)
    if equivalent_parent_indices:
        parameters["equivalent_parent_indices"] = tuple(int(i) for i in equivalent_parent_indices)

    recipe = StructureRecipe(
        operation="single_substitution",
        parent_formula=atoms.get_chemical_formula(),
        parent_n_atoms=len(atoms),
        target_indices=(idx,),
        target_environment=env.as_dict(),
        added_elements=(str(dopant),),
        removed_elements=(str(host),),
        parameters=parameters,
    )
    mode_tag = "manual" if str(selection_mode) == "manual" else f"orbit{int(orbit_id or 0)}"
    cid = f"sub_{host}_to_{dopant}_{env.depth_class}_{mode_tag}_{recipe.stable_id()}"
    validation = validate_engineered_structure(
        child,
        parent_atoms=atoms,
        operation="substitution",
        modified_indices=(idx,),
        effective_fraction=eff,
    )

    warnings = validation.setdefault("warnings", [])
    warning = adjustment_meta.get("initialization_warning")
    if warning:
        warnings.append(str(warning))
    if bool(adjustment_meta.get("cross_sublattice", False)):
        warnings.append("cross_sublattice_substitution_requires_manual_validation")
    charge_mismatch = adjustment_meta.get("charge_mismatch")
    if charge_mismatch is not None and abs(float(charge_mismatch)) > 1e-8:
        warnings.append(
            f"charge_mismatch={float(charge_mismatch):+.1f}; explicit_charge_compensation_not_generated"
        )
    if adjustment_meta.get("capped_displacement_indices"):
        warnings.append("local_geometry_initialization_reached_displacement_cap")
    if warnings and validation.get("status") == "pass":
        validation["status"] = "warn"

    label_suffix = (
        f"atom {idx}"
        if str(selection_mode) == "manual"
        else f"orbit {int(orbit_id or 0)} | atom {idx}"
    )
    init_tag = (
        "oxide-polyhedron"
        if material_class == "oxide"
        else "metal-radius"
    )
    if not adjustment_meta.get("applied"):
        init_tag += "-retained"

    return EngineeredStructure(
        atoms=child,
        recipe=recipe,
        candidate_id=cid,
        label=f"{host}→{dopant} | {env.depth_class} | {label_suffix} | {init_tag}",
        validation=validation,
        atom_provenance={"parent_to_child": {int(i): int(i) for i in range(len(atoms))}},
    )


def build_substitution_candidate_at_index(
    atoms: Atoms,
    *,
    host: str,
    dopant: str,
    target_index: int,
    depth: str | None = None,
    apply_local_adjustment: bool = True,
    adjustment_strength: float = 0.50,
    adjustment_shells: int = 2,
    max_local_displacement_A: float = 0.20,
    protect_bottom_layers: int = 1,
    host_oxidation_state: int | float | None = None,
    dopant_oxidation_state: int | float | None = None,
    shared_ligand_weight: float = 0.50,
) -> EngineeredStructure:
    return _build_substitution_candidate(
        atoms,
        host=host,
        dopant=dopant,
        target_index=int(target_index),
        selection_mode="manual",
        requested_depth=depth,
        apply_local_adjustment=apply_local_adjustment,
        adjustment_strength=adjustment_strength,
        adjustment_shells=adjustment_shells,
        max_local_displacement_A=max_local_displacement_A,
        protect_bottom_layers=protect_bottom_layers,
        host_oxidation_state=host_oxidation_state,
        dopant_oxidation_state=dopant_oxidation_state,
        shared_ligand_weight=shared_ligand_weight,
    )


def enumerate_substitution_candidates(
    atoms: Atoms,
    *,
    host: str,
    dopant: str,
    depth: str = "surface",
    max_candidates: int = 20,
    apply_local_adjustment: bool = True,
    adjustment_strength: float = 0.50,
    adjustment_shells: int = 2,
    max_local_displacement_A: float = 0.20,
    protect_bottom_layers: int = 1,
    host_oxidation_state: int | float | None = None,
    dopant_oxidation_state: int | float | None = None,
    shared_ligand_weight: float = 0.50,
) -> List[EngineeredStructure]:
    if str(host) not in atomic_numbers or str(dopant) not in atomic_numbers:
        raise ValueError("Host and dopant must be valid element symbols.")
    if str(host) == str(dopant):
        raise ValueError("Host and dopant must be different elements.")

    analysis = analyze_parent_slab(atoms)
    eligible = select_indices(analysis, element=str(host), depth=str(depth))
    if not eligible:
        raise ValueError(f"No {host} atoms found in depth selection '{depth}'.")

    groups = group_equivalent_atom_indices(analysis["environment_by_index"], eligible)
    out: List[EngineeredStructure] = []
    for orbit_id, orbit in enumerate(groups[: int(max_candidates)]):
        idx = choose_center_preferred_atom_index(atoms, orbit)
        out.append(
            _build_substitution_candidate(
                atoms,
                host=str(host),
                dopant=str(dopant),
                target_index=idx,
                selection_mode="automatic",
                orbit_id=int(orbit_id),
                equivalent_parent_indices=orbit,
                requested_depth=str(depth),
                apply_local_adjustment=apply_local_adjustment,
                adjustment_strength=adjustment_strength,
                adjustment_shells=adjustment_shells,
                max_local_displacement_A=max_local_displacement_A,
                protect_bottom_layers=protect_bottom_layers,
                host_oxidation_state=host_oxidation_state,
                dopant_oxidation_state=dopant_oxidation_state,
                shared_ligand_weight=shared_ligand_weight,
            )
        )
    return out
