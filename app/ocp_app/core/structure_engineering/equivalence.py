from __future__ import annotations

from collections import defaultdict
import json
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from ase import Atoms

from ocp_app.core.ads_sites import AdsSite

from .models import AtomEnvironment


def group_equivalent_atom_indices(
    environment_by_index: Dict[int, AtomEnvironment],
    indices: Iterable[int],
) -> List[List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx in indices:
        env = environment_by_index[int(idx)]
        groups[str(env.environment_key)].append(int(idx))
    return [sorted(v) for _k, v in sorted(groups.items(), key=lambda kv: min(kv[1]))]


def _fractional_xy_from_cartesian(atoms: Atoms, position) -> tuple[float, float]:
    """Return wrapped in-plane fractional coordinates for an atom/site position."""
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pos = np.asarray(position, dtype=float)
    if pos.shape[0] == 2:
        pos = np.array([float(pos[0]), float(pos[1]), 0.0], dtype=float)
    try:
        frac = np.linalg.solve(cell.T, pos)
        return float(frac[0] % 1.0), float(frac[1] % 1.0)
    except Exception:
        a = max(float(np.linalg.norm(cell[0])), 1e-8)
        b = max(float(np.linalg.norm(cell[1])), 1e-8)
        return float((pos[0] / a) % 1.0), float((pos[1] / b) % 1.0)


def _center_distance_score(atoms: Atoms, position) -> tuple[float, float]:
    """Score closeness to the displayed unit-cell center.

    The first term is the real-space in-plane distance from fractional
    (0.5, 0.5).  The second term favors positions farther from boundaries.
    """
    fx, fy = _fractional_xy_from_cartesian(atoms, position)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    delta = (fx - 0.5) * cell[0] + (fy - 0.5) * cell[1]
    dist = float(np.linalg.norm(delta))
    edge_margin = float(min(fx, 1.0 - fx, fy, 1.0 - fy))
    return dist, -edge_margin


def choose_center_preferred_atom_index(atoms: Atoms, indices: Sequence[int]) -> int:
    """Choose the symmetry-equivalent atom nearest the unit-cell center."""
    if not indices:
        raise ValueError("Cannot choose a representative from an empty atom group.")
    return int(
        min(
            (int(i) for i in indices),
            key=lambda i: (
                *_center_distance_score(atoms, atoms.positions[int(i)]),
                int(i),
            ),
        )
    )


def choose_center_preferred_site(atoms: Atoms, sites: Sequence[AdsSite]) -> AdsSite:
    """Choose the equivalent adsorption site nearest the unit-cell center."""
    if not sites:
        raise ValueError("Cannot choose a representative from an empty site group.")
    return min(
        sites,
        key=lambda site: (
            *_center_distance_score(atoms, site.position),
            tuple(int(i) for i in site.surface_indices),
        ),
    )


def canonical_site_kind(kind: str) -> str:
    value = str(kind).lower().strip()
    return "hollow" if value in {"fcc", "hcp"} else value


def site_environment_key(
    site: AdsSite,
    atoms: Atoms,
    environment_by_index: Dict[int, AtomEnvironment],
) -> str:
    support = tuple(int(i) for i in site.surface_indices)
    support_symbols = tuple(sorted(atoms[int(i)].symbol for i in support))
    support_env = tuple(sorted(environment_by_index[int(i)].environment_key for i in support))
    payload = {
        "kind": canonical_site_kind(str(site.kind)),
        "support_symbols": support_symbols,
        "support_env": support_env,
        "support_size": len(support),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def group_equivalent_sites(
    sites: Sequence[AdsSite],
    atoms: Atoms,
    environment_by_index: Dict[int, AtomEnvironment],
) -> List[List[AdsSite]]:
    groups: Dict[str, List[AdsSite]] = defaultdict(list)
    for site in sites:
        if not site.surface_indices:
            continue
        groups[site_environment_key(site, atoms, environment_by_index)].append(site)
    return [v for _k, v in sorted(groups.items(), key=lambda kv: kv[0])]
