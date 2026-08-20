from __future__ import annotations

from typing import Dict, Iterable

from ase import Atoms

try:
    from ocp_app.core.ads_sites import ANION_SYMBOLS as _ANION_SYMBOLS
except Exception:
    _ANION_SYMBOLS = {"O", "S", "Se", "Te", "N", "P", "F", "Cl", "Br", "I"}

ANION_SYMBOLS = frozenset(str(x) for x in _ANION_SYMBOLS)


def infer_structure_material_class(atoms: Atoms) -> str:
    """Return 'oxide' for O-containing cation/anion solids, otherwise 'metal'.

    This intentionally mirrors SAGE's existing high-level metal/oxide split:
    oxygen must coexist with at least one non-H, non-O element.
    """
    if atoms is None or len(atoms) == 0:
        return "unknown"
    symbols = set(str(s) for s in atoms.get_chemical_symbols())
    has_oxygen = "O" in symbols
    has_cation_candidate = any(s not in {"O", "H"} for s in symbols)
    return "oxide" if has_oxygen and has_cation_candidate else "metal"


def species_role(symbol: str, atoms: Atoms | None = None) -> str:
    sym = str(symbol)
    if sym in ANION_SYMBOLS:
        return "anion"
    if atoms is not None and infer_structure_material_class(atoms) == "oxide":
        return "cation"
    return "metal"


def substitution_sublattice_compatibility(
    atoms: Atoms,
    *,
    host: str,
    dopant: str,
) -> Dict[str, object]:
    material_class = infer_structure_material_class(atoms)
    host_role = species_role(str(host), atoms)
    dopant_role = species_role(str(dopant), atoms)
    cross_sublattice = (
        material_class == "oxide"
        and host_role in {"cation", "anion"}
        and dopant_role in {"cation", "anion"}
        and host_role != dopant_role
    )
    return {
        "material_class": material_class,
        "host_role": host_role,
        "dopant_role": dopant_role,
        "cross_sublattice": bool(cross_sublattice),
    }
