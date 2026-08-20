from __future__ import annotations

from collections import Counter
from hashlib import sha1
import json
from typing import Dict, Iterable, List, Sequence

import numpy as np
from ase import Atoms

from ocp_app.core.ads_sites import ANION_SYMBOLS
from ocp_app.core.structure_check import _radius

from .models import AtomEnvironment



def structure_content_signature(atoms: Atoms, decimals: int = 4) -> str:
    """Hash symbols, cell, PBC, and coordinates for cache invalidation."""
    if atoms is None:
        return "none"
    payload = {
        "symbols": tuple(atoms.get_chemical_symbols()),
        "cell": np.round(np.asarray(atoms.get_cell(), dtype=float), int(decimals)).tolist(),
        "pbc": tuple(bool(x) for x in atoms.get_pbc()),
        "positions": np.round(np.asarray(atoms.get_positions(), dtype=float), int(decimals)).tolist(),
    }
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def cluster_z_layers(atoms: Atoms, z_tol: float = 0.8) -> List[List[int]]:
    """Cluster slab atoms into approximate z layers, ordered top to bottom."""
    if atoms is None or len(atoms) == 0:
        return []
    z = np.asarray(atoms.get_positions()[:, 2], dtype=float)
    order = np.argsort(z)[::-1]
    layers: List[List[int]] = []
    refs: List[float] = []
    for idx in order.tolist():
        zi = float(z[idx])
        if not layers or abs(zi - refs[-1]) > float(z_tol):
            layers.append([int(idx)])
            refs.append(zi)
        else:
            layers[-1].append(int(idx))
            refs[-1] = float(np.mean(z[layers[-1]]))
    return [sorted(layer) for layer in layers]


def depth_class_for_layer(layer_id: int) -> str:
    if int(layer_id) == 0:
        return "surface"
    if int(layer_id) == 1:
        return "subsurface"
    return "bulk_like"


def species_class(symbol: str, all_symbols: Sequence[str]) -> str:
    sym = str(symbol)
    has_anion = any(str(s) in ANION_SYMBOLS for s in all_symbols)
    has_cation = any(str(s) not in ANION_SYMBOLS for s in all_symbols)
    if has_anion and has_cation:
        return "anion" if sym in ANION_SYMBOLS else "cation"
    return "metal" if sym not in ANION_SYMBOLS else "anion"


def _neighbor_environment(
    atoms: Atoms,
    index: int,
    cutoff_scale: float = 1.25,
    max_distance: float = 4.2,
) -> tuple[int, tuple[tuple[str, int], ...], tuple[float, ...], tuple[int, ...]]:
    symbols = atoms.get_chemical_symbols()
    sym_i = symbols[int(index)]
    neigh: List[tuple[int, str, float]] = []
    for j, sym_j in enumerate(symbols):
        if int(j) == int(index):
            continue
        try:
            d = float(atoms.get_distance(int(index), int(j), mic=True))
        except Exception:
            d = float(np.linalg.norm(atoms.positions[int(index)] - atoms.positions[int(j)]))
        cutoff = min(float(max_distance), float(cutoff_scale) * (_radius(sym_i) + _radius(sym_j)))
        if d <= cutoff:
            neigh.append((int(j), str(sym_j), float(d)))
    counts = tuple(sorted(Counter(s for _j, s, _d in neigh).items()))
    distance_bins = tuple(sorted(round(float(d), 1) for _j, _s, d in neigh))
    indices = tuple(sorted(int(j) for j, _s, _d in neigh))
    return len(neigh), counts, distance_bins, indices


def analyze_parent_slab(
    atoms: Atoms,
    z_tol: float = 0.8,
    cutoff_scale: float = 1.25,
) -> Dict[str, object]:
    """Return deterministic per-atom environments for an upper slab surface."""
    if atoms is None or len(atoms) == 0:
        raise ValueError("Parent structure is empty.")

    layers = cluster_z_layers(atoms, z_tol=float(z_tol))
    layer_of: Dict[int, int] = {}
    for lid, layer in enumerate(layers):
        for idx in layer:
            layer_of[int(idx)] = int(lid)

    symbols = atoms.get_chemical_symbols()
    envs: List[AtomEnvironment] = []
    neighbor_map: Dict[int, tuple[int, ...]] = {}
    for idx, sym in enumerate(symbols):
        lid = int(layer_of.get(int(idx), len(layers)))
        depth = depth_class_for_layer(lid)
        cn, counts, dbins, neigh_idx = _neighbor_environment(
            atoms,
            int(idx),
            cutoff_scale=float(cutoff_scale),
        )
        neighbor_map[int(idx)] = neigh_idx
        key_payload = {
            "symbol": str(sym),
            "layer_id": lid,
            "depth": depth,
            "species_class": species_class(str(sym), symbols),
            "coordination_number": int(cn),
            "neighbor_counts": counts,
            "distance_bins": dbins,
        }
        key = json.dumps(key_payload, sort_keys=True, default=str)
        envs.append(
            AtomEnvironment(
                index=int(idx),
                symbol=str(sym),
                layer_id=lid,
                depth_class=depth,
                species_class=species_class(str(sym), symbols),
                exposed=(lid == 0),
                coordination_number=int(cn),
                neighbor_counts=counts,
                neighbor_distance_bins=dbins,
                environment_key=key,
            )
        )

    return {
        "layers": layers,
        "environments": envs,
        "environment_by_index": {e.index: e for e in envs},
        "neighbor_map": neighbor_map,
        "top_indices": tuple(layers[0]) if layers else (),
        "subsurface_indices": tuple(layers[1]) if len(layers) > 1 else (),
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
    }


def select_indices(
    analysis: Dict[str, object],
    *,
    element: str | None = None,
    depth: str = "surface",
) -> List[int]:
    envs: Iterable[AtomEnvironment] = analysis.get("environments", [])  # type: ignore[assignment]
    allowed_depths = {
        "surface": {"surface"},
        "subsurface": {"subsurface"},
        "surface+subsurface": {"surface", "subsurface"},
        "all": {"surface", "subsurface", "bulk_like"},
    }.get(str(depth), {str(depth)})
    out = []
    for env in envs:
        if element is not None and str(env.symbol) != str(element):
            continue
        if env.depth_class not in allowed_depths:
            continue
        out.append(int(env.index))
    return out
