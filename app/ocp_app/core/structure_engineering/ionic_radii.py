from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Dict, Iterable, List, Sequence

import numpy as np
from ase import Atoms

try:
    from pymatgen.core import Element, Species
except Exception:
    try:
        from pymatgen.core.periodic_table import Element, Species
    except Exception:
        Element = None
        Species = None


_CN_ROMAN = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
    11: "XI",
    12: "XII",
}


def _finite_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) and out > 0.0 else None


def oxidation_state_options(symbol: str) -> List[int]:
    """Return deterministic integer oxidation-state options from pymatgen."""
    sym = str(symbol)
    values: List[int] = []
    if Element is not None:
        try:
            el = Element(sym)
            ordered = list(getattr(el, "common_oxidation_states", ()) or ())
            ordered += list(getattr(el, "oxidation_states", ()) or ())
            for value in ordered:
                fv = float(value)
                iv = int(round(fv))
                if abs(fv - iv) <= 1e-8 and iv not in values:
                    values.append(iv)
        except Exception:
            pass

    fallback = {
        "O": [-2, -1],
        "F": [-1],
        "Cl": [-1, 1, 3, 5, 7],
        "Br": [-1, 1, 3, 5, 7],
        "I": [-1, 1, 3, 5, 7],
        "S": [-2, 4, 6],
        "Se": [-2, 4, 6],
        "N": [-3, 3, 5],
        "P": [-3, 3, 5],
        "H": [1, -1],
    }
    for value in fallback.get(sym, []):
        if int(value) not in values:
            values.append(int(value))

    if not values:
        values = list(range(-4, 9))
    return values


def preferred_oxidation_state(symbol: str, *, role: str | None = None) -> int | None:
    options = oxidation_state_options(str(symbol))
    if not options:
        return None
    if str(symbol) == "O" and -2 in options:
        return -2
    if role == "anion":
        negative = [x for x in options if x < 0]
        if negative:
            return negative[0]
    if role in {"cation", "metal"}:
        positive = [x for x in options if x > 0]
        if positive:
            return positive[0]
    nonzero = [x for x in options if x != 0]
    return nonzero[0] if nonzero else options[0]


def ionic_radius(
    symbol: str,
    oxidation_state: int | float,
    coordination_number: int,
) -> Dict[str, object]:
    """Get an ionic radius, preferring Shannon radius for oxidation state + CN.

    The function is deliberately fail-soft. If a coordination-specific Shannon
    value is unavailable, it falls back to a species ionic radius and then an
    element-average ionic radius. It never falls back to a covalent radius for
    an oxide initialization.
    """
    sym = str(symbol)
    ox = float(oxidation_state)
    cn = max(1, min(12, int(round(coordination_number))))
    roman = _CN_ROMAN.get(cn, str(cn))

    result: Dict[str, object] = {
        "symbol": sym,
        "oxidation_state": ox,
        "coordination_number": cn,
        "coordination_label": roman,
        "radius_A": None,
        "source": "unavailable",
        "spin": None,
    }

    if Species is not None:
        try:
            sp = Species(sym, ox)
            for spin in ("", "High Spin", "Low Spin"):
                try:
                    value = sp.get_shannon_radius(
                        roman,
                        spin=spin,
                        radius_type="ionic",
                    )
                    radius = _finite_float(value)
                    if radius is not None:
                        result.update({
                            "radius_A": radius,
                            "source": "pymatgen_shannon_ionic",
                            "spin": spin or "unspecified",
                        })
                        return result
                except Exception:
                    continue

            try:
                radius = _finite_float(getattr(sp, "ionic_radius", None))
                if radius is not None:
                    result.update({
                        "radius_A": radius,
                        "source": "pymatgen_species_ionic",
                    })
                    return result
            except Exception:
                pass
        except Exception:
            pass

    if Element is not None:
        try:
            el = Element(sym)
            radius = _finite_float(getattr(el, "average_ionic_radius", None))
            if radius is not None:
                result.update({
                    "radius_A": radius,
                    "source": "pymatgen_average_ionic",
                })
                return result
        except Exception:
            pass

    return result


def _cation_counts(atoms: Atoms) -> Dict[str, int]:
    counts = Counter(
        str(s)
        for s in atoms.get_chemical_symbols()
        if str(s) not in {"O", "H"}
    )
    return {k: int(v) for k, v in counts.items()}


def infer_oxide_oxidation_states(atoms: Atoms) -> Dict[str, object]:
    """Best-effort charge-neutral single-state assignment per cation element.

    The model cannot represent mixed valence within one element. Ambiguity is
    therefore reported rather than silently resolved.
    """
    symbols = [str(s) for s in atoms.get_chemical_symbols()]
    n_oxygen = int(sum(1 for s in symbols if s == "O"))
    cations = _cation_counts(atoms)
    if n_oxygen <= 0 or not cations:
        return {
            "status": "not_oxide",
            "assignments": {},
            "alternatives": [],
            "residual_charge": None,
            "ambiguous": True,
        }

    target_charge = float(2 * n_oxygen)
    names = sorted(cations)
    option_lists: List[List[int]] = []
    for sym in names:
        opts = [x for x in oxidation_state_options(sym) if x > 0]
        if not opts:
            opts = [1, 2, 3, 4, 5, 6]
        option_lists.append(opts[:8])

    solutions = []
    for combo in product(*option_lists):
        total = sum(float(cations[sym]) * float(ox) for sym, ox in zip(names, combo))
        residual = abs(total - target_charge)
        common_penalty = 0.0
        magnitude_penalty = 0.0
        for sym, ox in zip(names, combo):
            preferred = preferred_oxidation_state(sym, role="cation")
            if preferred is not None:
                common_penalty += 0.05 * abs(float(ox) - float(preferred))
            magnitude_penalty += 0.001 * abs(float(ox))
        score = residual + common_penalty + magnitude_penalty
        solutions.append((score, residual, combo))

    solutions.sort(key=lambda x: (x[0], x[1], x[2]))
    best_score, best_residual, best_combo = solutions[0]
    alternatives = []
    for score, residual, combo in solutions[:10]:
        if residual <= max(0.25, best_residual + 0.01):
            alternatives.append({
                "assignments": {sym: int(ox) for sym, ox in zip(names, combo)},
                "residual_charge": float(residual),
                "score": float(score),
            })

    best = {sym: int(ox) for sym, ox in zip(names, best_combo)}
    equally_good = [
        alt for alt in alternatives
        if abs(float(alt["residual_charge"]) - float(best_residual)) < 1e-8
    ]
    ambiguous = len(equally_good) > 1 or best_residual > 0.25

    return {
        "status": "ok" if best_residual <= 0.25 else "unresolved",
        "assignments": best,
        "alternatives": alternatives,
        "residual_charge": float(best_residual),
        "ambiguous": bool(ambiguous),
        "model": "single_oxidation_state_per_cation_element",
        "target_positive_charge": target_charge,
    }


def suggested_substitution_oxidation_states(
    atoms: Atoms,
    *,
    host: str,
    dopant: str,
    host_role: str,
    dopant_role: str,
) -> Dict[str, object]:
    inferred = infer_oxide_oxidation_states(atoms)
    assignments = dict(inferred.get("assignments", {}) or {})

    if host_role == "anion":
        host_ox = preferred_oxidation_state(str(host), role="anion")
    else:
        host_ox = assignments.get(str(host))
        if host_ox is None:
            host_ox = preferred_oxidation_state(str(host), role="cation")

    dopant_options = oxidation_state_options(str(dopant))
    if host_ox in dopant_options:
        dopant_ox = int(host_ox)
    else:
        dopant_ox = preferred_oxidation_state(
            str(dopant),
            role=("anion" if dopant_role == "anion" else "cation"),
        )

    return {
        "host_oxidation_state": host_ox,
        "dopant_oxidation_state": dopant_ox,
        "host_options": oxidation_state_options(str(host)),
        "dopant_options": dopant_options,
        "inference": inferred,
    }
