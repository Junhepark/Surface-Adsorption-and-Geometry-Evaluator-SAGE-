from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms

from ocp_app.core.structure_ops import add_vacuum_z, _recenter_slab_z_into_cell


_SLAB_REDUCTION_PRESETS: Dict[str, dict] = {
    "Large": {
        "target_preserved_layers": 6,
        "target_vacuum_z": 30.0,
        "description": "Large preserves 6 z-layers from the base slab. Conservative preset for thicker slabs.",
    },
    "Medium": {
        "target_preserved_layers": 4,
        "target_vacuum_z": 30.0,
        "description": "Medium preserves 4 z-layers from the base slab. Recommended default preset.",
    },
    "Small": {
        "target_preserved_layers": 2,
        "target_vacuum_z": 30.0,
        "description": "Small preserves 2 z-layers from the base slab. Aggressive preset; oxide terminations may become unstable or fragmented.",
    },
}


def get_slab_reduction_presets() -> Dict[str, dict]:
    return {k: dict(v) for k, v in _SLAB_REDUCTION_PRESETS.items()}


def _cluster_z_layers_simple(atoms: Atoms, tol: float = 0.8) -> List[List[int]]:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    if pos.size == 0:
        return []
    order = np.argsort(pos[:, 2])
    layers: List[List[int]] = []
    current = [int(order[0])]
    z_ref = float(pos[order[0], 2])
    for idx in order[1:]:
        idx = int(idx)
        z_val = float(pos[idx, 2])
        if abs(z_val - z_ref) <= float(tol):
            current.append(idx)
            z_ref = float(np.mean(pos[current, 2]))
        else:
            layers.append(sorted(current))
            current = [idx]
            z_ref = z_val
    if current:
        layers.append(sorted(current))
    return layers


def _slab_thickness_z(atoms: Atoms) -> float:
    if len(atoms) == 0:
        return float("nan")
    z = np.asarray(atoms.get_positions(), dtype=float)[:, 2]
    return float(np.max(z) - np.min(z))


def _resolve_target_preserved_layers(
    n_layers: int,
    requested_layers: int,
    *,
    min_preserved_layers: int,
) -> tuple[int, dict]:
    """Resolve the requested preserved layer count to a feasible symmetric target.

    Symmetric top/bottom pair removal preserves the parity of the original layer count.
    When the requested layer count has the wrong parity, bias upward (less aggressive)
    to the nearest feasible value, then downward if needed.
    """
    n_layers = max(0, int(n_layers))
    requested = max(1, int(requested_layers))
    min_keep = max(1, int(min_preserved_layers))

    feasible = [k for k in range(min_keep, n_layers + 1) if (n_layers - k) % 2 == 0]
    if not feasible:
        feasible = [n_layers]

    if requested in feasible:
        return int(requested), {
            "requested_preserved_layers": int(requested),
            "effective_target_preserved_layers": int(requested),
            "parity_adjusted": False,
        }

    higher = [k for k in feasible if k >= requested]
    lower = [k for k in feasible if k <= requested]
    if higher:
        chosen = min(higher)
    else:
        chosen = max(lower)

    return int(chosen), {
        "requested_preserved_layers": int(requested),
        "effective_target_preserved_layers": int(chosen),
        "parity_adjusted": int(chosen) != int(requested),
    }


def _reduce_to_target_layers(
    atoms: Atoms,
    *,
    target_preserved_layers: int,
    layer_tol: float,
    min_atoms: int,
    min_preserved_layers: int,
) -> Tuple[Atoms, dict]:
    layers = _cluster_z_layers_simple(atoms, tol=float(layer_tol))
    n_layers = len(layers)
    target_keep, target_meta = _resolve_target_preserved_layers(
        n_layers,
        int(target_preserved_layers),
        min_preserved_layers=int(min_preserved_layers),
    )

    if target_keep >= n_layers:
        return atoms.copy(), {
            "removed_layer_pairs": 0,
            "kept_layer_count": int(n_layers),
            "original_layer_count": int(n_layers),
            **target_meta,
        }

    remove_pairs = max(0, (n_layers - int(target_keep)) // 2)
    keep_layers = layers[remove_pairs : n_layers - remove_pairs]
    keep_indices = sorted({int(i) for layer in keep_layers for i in layer})
    if len(keep_indices) < int(min_atoms) or len(keep_layers) < int(min_preserved_layers):
        return atoms.copy(), {
            "removed_layer_pairs": 0,
            "kept_layer_count": int(n_layers),
            "original_layer_count": int(n_layers),
            **target_meta,
        }

    reduced = atoms[keep_indices].copy()
    return reduced, {
        "removed_layer_pairs": int(remove_pairs),
        "kept_layer_count": int(len(keep_layers)),
        "original_layer_count": int(n_layers),
        **target_meta,
    }


def reduce_slab_symmetrically(
    atoms: Atoms,
    *,
    level: str | None = None,
    target_preserved_layers: int | None = None,
    keep_pbc_z: bool = True,
    min_preserved_layers: int = 2,
    layer_tol: float = 0.8,
    min_atoms: int = 8,
    target_vacuum_z: float | None = None,
):
    if level is None and target_preserved_layers is None:
        raise ValueError("Either level or target_preserved_layers must be provided.")
    if level is not None and level != "Custom" and level not in _SLAB_REDUCTION_PRESETS:
        raise ValueError(f"Unknown slab reduction level: {level}")

    cfg = dict(_SLAB_REDUCTION_PRESETS.get(level or "", {}))
    requested_preserved_layers = (
        int(target_preserved_layers)
        if target_preserved_layers is not None
        else int(cfg.get("target_preserved_layers", min_preserved_layers))
    )

    original = atoms.copy()
    original_layers = _cluster_z_layers_simple(original, tol=float(layer_tol))
    n_layers = len(original_layers)
    original_thickness = _slab_thickness_z(original)
    original_atoms = len(original)

    reduced, layer_meta = _reduce_to_target_layers(
        original,
        target_preserved_layers=int(requested_preserved_layers),
        layer_tol=float(layer_tol),
        min_atoms=int(min_atoms),
        min_preserved_layers=int(min_preserved_layers),
    )

    final_vac = float(target_vacuum_z) if target_vacuum_z is not None else float(cfg.get("target_vacuum_z", 30.0))

    if len(reduced) == len(original):
        meta = {
            "reduction_level": str(level or "Custom"),
            "reduced": False,
            "symmetry_preserved": True,
            "reason": f"No balanced layer reduction was possible. Requested preserved layers={requested_preserved_layers}, effective target={layer_meta.get('effective_target_preserved_layers')}, original layers={layer_meta.get('original_layer_count')}, min_atoms={int(min_atoms)}.",
            "original_atoms": int(original_atoms),
            "reduced_atoms": int(original_atoms),
            "original_thickness_A": float(original_thickness),
            "reduced_thickness_A": float(original_thickness),
            "final_vacuum_z": float(original.cell[2, 2]) if original.cell.shape == (3, 3) else float("nan"),
            "target_preserved_layers": int(requested_preserved_layers),
            **layer_meta,
        }
        return original, meta

    reduced = add_vacuum_z(
        reduced,
        total_vacuum_z=float(final_vac),
        keep_pbc_z=bool(keep_pbc_z),
    )
    reduced = _recenter_slab_z_into_cell(reduced)
    reduced_thickness = _slab_thickness_z(reduced)

    meta = {
        "reduction_level": str(level or "Custom"),
        "reduced": True,
        "symmetry_preserved": True,
        "reason": "Balanced top/bottom slab reduction applied from the XY-expanded base slab using the preserved-layer target.",
        "original_atoms": int(original_atoms),
        "reduced_atoms": int(len(reduced)),
        "removed_atoms": int(original_atoms - len(reduced)),
        "original_thickness_A": float(original_thickness),
        "reduced_thickness_A": float(reduced_thickness),
        "final_vacuum_z": float(reduced.cell[2, 2]) if reduced.cell.shape == (3, 3) else float("nan"),
        "layer_tol": float(layer_tol),
        "min_preserved_layers": int(min_preserved_layers),
        "min_atoms": int(min_atoms),
        "target_preserved_layers": int(requested_preserved_layers),
        **layer_meta,
    }
    return reduced, meta
