# ocp_app/core/structure_check.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


# Covalent bond radii (Å) for commonly used elements
# Approximate values based on Cordero et al. 2008
COVALENT_RADII: Dict[str, float] = {
    # p-block
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,

    # s-block
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "K": 2.03,
    "Ca": 1.76,

    # 3d transition metals
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,

    # 4d transition metals
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,

    # 5d transition metals
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
}

# Element pairs for which long-distance warnings are suppressed
# (e.g. anion-anion pairs in oxides/halides)
LONG_DISTANCE_IGNORE = {
    "O-O",
    "N-N",
    "F-F",
    "Cl-Cl",
    "Br-Br",
    "I-I",
    "S-S",
}


def _radius(sym: str) -> float:
    """Return covalent radius for the given symbol; defaults to 1.20 Å if unknown."""
    return COVALENT_RADII.get(sym, 1.20)


@dataclass
class PairDistanceStats:
    pair: str
    d_min: float
    d_mean: float
    n_bonds: int
    flag: str  # "ok" | "warning" | "critical"


@dataclass
class StructureReport:
    n_atoms: int
    cell_lengths: Tuple[float, float, float]
    vacuum_z: float
    area_xy: float
    recommend_repeat: Optional[Tuple[int, int, int]]
    issues: List[str]
    nearest_global: Dict
    nearest_by_pair: List[PairDistanceStats]

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["nearest_by_pair"] = [asdict(p) for p in self.nearest_by_pair]
        return d


def validate_structure(
    atoms: Atoms,
    target_area: float = 70.0,          # Target slab surface area (Å²)
    min_vacuum: Optional[float] = None, # If None, no vacuum issue is raised
) -> StructureReport:
    """Generate a structural sanity / distance statistics report for the uploaded structure."""

    issues: List[str] = []

    n_atoms = len(atoms)
    cell = atoms.get_cell()
    a, b, c = cell.lengths()
    cell_lengths = (float(a), float(b), float(c))

    # ---- PBC / cell sanity ----
    if not all(atoms.get_pbc()):
        issues.append("PBC is not True in all directions (non-3D periodic cell).")

    vol = atoms.get_volume()
    if vol < 10.0:
        issues.append(f"Cell volume is very small: {vol:.2f} Å³.")

    # ---- Slab thickness / vacuum estimation ----
    z = atoms.positions[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    thickness = z_max - z_min
    vacuum_z = float(c - thickness)

    # In this app the OCP pipeline resets vacuum to ~30 Å anyway, so
    # vacuum_z is reported as informational only and not flagged by default.
    if (min_vacuum is not None) and (vacuum_z < min_vacuum):
        issues.append(
            f"Vacuum along z is small ({vacuum_z:.2f} Å). "
            f"Recommended ≥ {min_vacuum:.1f} Å."
        )

    # ---- XY area & supercell recommendation ----
    area_xy = float(np.linalg.norm(np.cross(cell[0], cell[1])))
    recommend_repeat: Optional[Tuple[int, int, int]] = None
    if area_xy < target_area:
        scale = np.sqrt(target_area / max(area_xy, 1e-6))
        nx = int(np.ceil(scale))
        ny = int(np.ceil(scale))
        if nx * ny > 1:
            recommend_repeat = (nx, ny, 1)
            issues.append(
                f"Surface area is small ({area_xy:.1f} Å²). "
                f"Consider repeating to {nx}×{ny} in plane."
            )

    # ---- Nearest-distance / coordination analysis ----
    # Use a generous 4 Å cutoff
    cutoffs = [4.0] * n_atoms
    i_idx, j_idx, dists = neighbor_list("ijd", atoms, cutoffs)

    if len(dists) == 0:
        issues.append("No neighbors found within 4 Å. Structure may be isolated.")
        return StructureReport(
            n_atoms=n_atoms,
            cell_lengths=cell_lengths,
            vacuum_z=vacuum_z,
            area_xy=area_xy,
            recommend_repeat=recommend_repeat,
            issues=issues,
            nearest_global={},
            nearest_by_pair=[],
        )

    # Global minimum distance
    k_min = int(np.argmin(dists))
    gi, gj, gd = int(i_idx[k_min]), int(j_idx[k_min]), float(dists[k_min])
    sym_i = atoms[gi].symbol
    sym_j = atoms[gj].symbol
    pair_label = f"{sym_i}-{sym_j}"

    nearest_global = {
        "d_min": gd,
        "i": gi,
        "j": gj,
        "pair": pair_label,
    }

    # Per-pair statistics and coordination counting
    pair_data: Dict[str, List[float]] = {}
    pair_thresh: Dict[str, float] = {}
    coord_counts = np.zeros(n_atoms, dtype=int)

    for ii, jj, dd in zip(i_idx, j_idx, dists):
        i = int(ii)
        j = int(jj)
        s1 = atoms[i].symbol
        s2 = atoms[j].symbol

        if s1 <= s2:
            pair = f"{s1}-{s2}"
        else:
            pair = f"{s2}-{s1}"

        pair_data.setdefault(pair, []).append(float(dd))

        r1 = _radius(s1)
        r2 = _radius(s2)
        ref = r1 + r2
        pair_thresh[pair] = ref

        # Count as "neighbor" if within 1.25× reference bond length
        if dd < 1.25 * ref:
            coord_counts[i] += 1
            coord_counts[j] += 1

    nearest_by_pair: List[PairDistanceStats] = []

    for pair, arr in pair_data.items():
        arr_np = np.asarray(arr)
        d_min = float(arr_np.min())
        d_mean = float(arr_np.mean())
        ref = pair_thresh[pair]
        ratio = d_min / ref if ref > 0 else 1.0

        if ratio < 0.60:
            # Too short → critical
            flag = "critical"
            issues.append(
                f"Very short {pair} distance: d_min={d_min:.2f} Å "
                f"(expected ≈ {ref:.2f} Å)."
            )
        elif ratio < 0.80:
            # Slightly short → warning
            flag = "warning"
        elif ratio > 1.50:
            # Too long → warning (but suppress for anion-anion exception pairs)
            if pair in LONG_DISTANCE_IGNORE:
                # Keep in table but do not raise a warning
                flag = "ok"
            else:
                flag = "warning"
                issues.append(
                    f"Unusually long {pair} distance: d_min={d_min:.2f} Å "
                    f"(expected ≈ {ref:.2f} Å). Structure may be too sparse "
                    "or contain stretched bonds."
                )
        else:
            flag = "ok"

        nearest_by_pair.append(
            PairDistanceStats(
                pair=pair,
                d_min=d_min,
                d_mean=d_mean,
                n_bonds=len(arr),
                flag=flag,
            )
        )

    # Warn about isolated atoms (coordination = 0)
    isolated_idx = [i for i, c in enumerate(coord_counts) if c == 0]
    if isolated_idx:
        n_iso = len(isolated_idx)
        sample = ", ".join(map(str, isolated_idx[:5]))
        issues.append(
            f"{n_iso} atoms have no neighbors within ~bond length "
            f"(examples: indices {sample}). "
            "Structure may contain isolated fragments or be too sparse."
        )

    return StructureReport(
        n_atoms=n_atoms,
        cell_lengths=cell_lengths,
        vacuum_z=vacuum_z,
        area_xy=area_xy,
        recommend_repeat=recommend_repeat,
        issues=issues,
        nearest_global=nearest_global,
        nearest_by_pair=nearest_by_pair,
    )
