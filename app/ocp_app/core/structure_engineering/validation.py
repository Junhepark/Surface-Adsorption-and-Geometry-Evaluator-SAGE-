from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from ase import Atoms

from ocp_app.core.structure_check import _radius, validate_structure


def _minimum_pair_ratio(atoms: Atoms) -> tuple[float, float, tuple[int, int] | None]:
    if atoms is None or len(atoms) < 2:
        return float("inf"), float("inf"), None
    symbols = atoms.get_chemical_symbols()
    best_ratio = float("inf")
    best_d = float("inf")
    best_pair = None
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            try:
                d = float(atoms.get_distance(i, j, mic=True))
            except Exception:
                d = float(np.linalg.norm(atoms.positions[i] - atoms.positions[j]))
            denom = max(_radius(symbols[i]) + _radius(symbols[j]), 1e-8)
            ratio = d / denom
            if ratio < best_ratio:
                best_ratio = float(ratio)
                best_d = float(d)
                best_pair = (int(i), int(j))
    return best_ratio, best_d, best_pair


def validate_engineered_structure(
    atoms: Atoms,
    *,
    parent_atoms: Atoms,
    operation: str,
    modified_indices: Iterable[int] = (),
    effective_fraction: float | None = None,
    min_vacuum: float = 8.0,
) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []

    try:
        report = validate_structure(atoms, target_area=70.0, min_vacuum=float(min_vacuum))
        report_dict = report.as_dict()
    except Exception as exc:
        report_dict = {}
        errors.append(f"validate_structure_failed: {exc}")

    ratio, dmin, pair = _minimum_pair_ratio(atoms)
    if not np.isfinite(ratio):
        errors.append("minimum_pair_distance_unavailable")
    elif ratio < 0.55:
        errors.append(f"atomic_overlap_ratio={ratio:.3f}")
    elif ratio < 0.70:
        warnings.append(f"short_contact_ratio={ratio:.3f}")

    if atoms is None or len(atoms) == 0:
        errors.append("empty_structure")
    if abs(float(atoms.get_volume())) < 10.0:
        errors.append("invalid_or_tiny_cell")

    try:
        vac = float(report_dict.get("vacuum_z", float("nan")))
        if np.isfinite(vac) and vac < float(min_vacuum):
            warnings.append(f"small_vacuum_z={vac:.2f}A")
    except Exception:
        pass

    try:
        cell = np.asarray(atoms.get_cell(), dtype=float)
        min_xy = min(float(np.linalg.norm(cell[0])), float(np.linalg.norm(cell[1])))
        if min_xy < 5.0:
            warnings.append(f"short_periodic_image_spacing_xy={min_xy:.2f}A")
    except Exception:
        pass

    if effective_fraction is not None:
        f = float(effective_fraction)
        if f >= 0.25:
            warnings.append(f"high_effective_fraction={f:.3f}")
        elif f >= 0.125:
            warnings.append(f"moderate_effective_fraction={f:.3f}")

    op = str(operation)
    if op == "adatom":
        try:
            parent_count = len(parent_atoms)
            parent_top = float(np.max(atoms.positions[:parent_count, 2]))
            ad_idx = max(int(i) for i in modified_indices)
            ad_z = float(atoms.positions[ad_idx, 2])
            if ad_z <= parent_top:
                errors.append("adatom_not_above_parent_surface")
        except Exception:
            warnings.append("adatom_height_check_unavailable")

    for issue in report_dict.get("issues", []) or []:
        issue_s = str(issue)
        if "very small" in issue_s.lower():
            errors.append(issue_s)
        else:
            warnings.append(issue_s)

    status = "reject" if errors else ("warn" if warnings else "pass")
    return {
        "status": status,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "minimum_distance_A": None if not np.isfinite(dmin) else float(dmin),
        "minimum_distance_ratio": None if not np.isfinite(ratio) else float(ratio),
        "minimum_pair_indices": pair,
        "structure_report": report_dict,
    }
