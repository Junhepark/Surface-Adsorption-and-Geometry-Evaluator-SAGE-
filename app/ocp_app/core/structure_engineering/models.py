from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha1
import json
from typing import Any, Dict, Tuple

from ase import Atoms


@dataclass(frozen=True)
class AtomEnvironment:
    index: int
    symbol: str
    layer_id: int
    depth_class: str
    species_class: str
    exposed: bool
    coordination_number: int
    neighbor_counts: Tuple[Tuple[str, int], ...]
    neighbor_distance_bins: Tuple[float, ...]
    environment_key: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructureRecipe:
    operation: str
    parent_formula: str
    parent_n_atoms: int
    target_indices: Tuple[int, ...] = ()
    target_environment: Dict[str, Any] = field(default_factory=dict)
    added_elements: Tuple[str, ...] = ()
    removed_elements: Tuple[str, ...] = ()
    parameters: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def stable_id(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, default=str).encode("utf-8")
        return sha1(payload).hexdigest()[:12]


@dataclass
class EngineeredStructure:
    atoms: Atoms
    recipe: StructureRecipe
    candidate_id: str
    label: str
    validation: Dict[str, Any]
    atom_provenance: Dict[str, Any] = field(default_factory=dict)

    def summary_record(self) -> Dict[str, Any]:
        v = dict(self.validation or {})
        p = dict(self.recipe.parameters or {})
        local = dict(p.get("local_geometry_adjustment", {}) or {})
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "operation": self.recipe.operation,
            "formula": self.atoms.get_chemical_formula(),
            "n_atoms": len(self.atoms),
            "target_indices": ",".join(str(i) for i in self.recipe.target_indices),
            "depth": p.get("depth", ""),
            "site_kind": p.get("site_kind", ""),
            "effective_fraction": p.get("effective_fraction", None),
            "material_class": p.get("material_class", local.get("material_class", "")),
            "local_initialization": local.get("method", ""),
            "host_oxidation_state": local.get("host_oxidation_state", None),
            "dopant_oxidation_state": local.get("dopant_oxidation_state", None),
            "charge_mismatch": local.get("charge_mismatch", None),
            "coordination_number": local.get("coordination_number", None),
            "radius_mismatch_signed": local.get("signed_radius_mismatch_fraction", None),
            "local_moved_atoms": local.get("n_moved_atoms", None),
            "local_max_displacement_A": local.get("max_applied_displacement_A", None),
            "first_shell_distance_before_A": local.get("mean_first_shell_distance_before_A", None),
            "first_shell_distance_after_A": local.get("mean_first_shell_distance_after_A", None),
            "validation_status": v.get("status", "unknown"),
            "n_errors": len(v.get("errors", []) or []),
            "n_warnings": len(v.get("warnings", []) or []),
            "minimum_distance_A": v.get("minimum_distance_A", None),
        }
