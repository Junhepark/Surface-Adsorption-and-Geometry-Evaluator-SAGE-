from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import logging
from collections import Counter
import math

import numpy as np
from ase import Atoms

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Optional Materials Project clients (mp-api / legacy pymatgen)
# ----------------------------------------------------------------------
try:
    from mp_api.client import MPRester as _MPResterNew  # type: ignore
except Exception:
    _MPResterNew = None  # type: ignore

try:
    from pymatgen.ext.matproj import MPRester as _MPResterOld  # type: ignore
except Exception:
    _MPResterOld = None  # type: ignore


def _get_mp_client(api_key: Optional[str] = None):
    """
    Create a Materials Project client.
    Prefers mp-api if available; falls back to pymatgen.ext.matproj.MPRester.
    """
    if _MPResterNew is not None:
        return _MPResterNew(api_key=api_key) if api_key else _MPResterNew()
    if _MPResterOld is not None:
        return _MPResterOld(api_key) if api_key else _MPResterOld()
    raise RuntimeError(
        "Neither mp-api nor pymatgen.ext.matproj MPRester is available. "
        "Install `pymatgen[matproj]` (Py3.10) or a Python-3.11-compatible `mp-api`."
    )


def _standardize_to_conventional(
    struct: Structure,
    *,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
) -> Structure:
    """
    Standardize structure to match the conventional cell as closely as
    possible to the 'Download CIF' format on the MP website:
      1) refined_structure
      2) conventional_standard_structure

    Returns the original structure on failure.
    """
    try:
        sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
        refined = sga.get_refined_structure()
        sga2 = SpacegroupAnalyzer(refined, symprec=symprec, angle_tolerance=angle_tolerance)
        conv = sga2.get_conventional_standard_structure()
        return conv
    except Exception as e:
        logger.warning(f"Conventional standardization failed; returning original structure. ({e})")
        return struct


def _mpr_get_structure_by_material_id(
    mpr,
    mp_id: str,
    *,
    final: bool,
    conventional_unit_cell: bool,
) -> Optional[Structure]:
    """
    Retrieve structure by material ID, covering both mp-api and legacy
    MPRester interfaces.
    """
    fn = getattr(mpr, "get_structure_by_material_id", None)
    if callable(fn):
        try:
            return fn(mp_id, final=final, conventional_unit_cell=conventional_unit_cell)
        except TypeError:
            try:
                return fn(mp_id, final=final)
            except TypeError:
                return fn(mp_id)

    summary = getattr(mpr, "summary", None)
    if summary is not None and hasattr(summary, "search"):
        docs = summary.search(material_ids=[mp_id])
        if not docs:
            return None
        doc0 = docs[0]
        try:
            return doc0.structure  # type: ignore[attr-defined]
        except AttributeError:
            return doc0["structure"]  # type: ignore[index]

    return None


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------
@dataclass
class BulkSource:
    kind: Literal["mp-id", "cif", "ase", "pmg"]
    ref: Union[str, Path, Atoms, Structure]
    label: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class BulkSpec:
    bulk_source: BulkSource

    mp_final: bool = True
    mp_conventional_unit_cell: bool = True

    mp_symprec: float = 0.1
    mp_angle_tolerance: float = 5.0


@dataclass
class DopingSpec:
    """
    (Deprecated) Legacy compatibility.  Use RatioTuneSpec for new code.
    """
    target_ratio: Dict[str, int]
    layers_from_top: int = 1
    mode: Literal["uniform", "cluster"] = "uniform"
    layer_tol: float = 0.3
    rng_seed: int = 0


@dataclass
class RatioTuneSpec:
    """
    Automatic XY repeat search + top-layer metal ratio tuning (substitution).

    Additional safeguards:
      - max_atoms_after_repeat: upper limit on total atoms after repeat
      - prefer_square_xy: prefer nx ≈ ny and aspect ≈ 1
    """
    target_ratio: Dict[str, int]
    layers_from_top: int = 1
    layer_tol: float = 0.3

    auto_scale_xy: bool = True
    max_xy: int = 6
    min_sites: int = 20
    exact_divisible: bool = True

    candidate_elements: Optional[List[str]] = None
    exclude_elements: Tuple[str, ...] = ("O", "H")

    rng_seed: int = 0

    max_atoms_after_repeat: int = 200
    prefer_square_xy: bool = True


# ----------------------------------------------------------------------
# Bulk structure loading
# ----------------------------------------------------------------------
def load_bulk_structure(
    src: BulkSource,
    *,
    mp_final: bool = True,
    mp_conventional_unit_cell: bool = True,
    mp_symprec: float = 0.1,
    mp_angle_tolerance: float = 5.0,
) -> Structure:
    if src.kind == "mp-id":
        mp_id = str(src.ref)
        client = _get_mp_client(api_key=src.api_key)

        try:
            with client as mpr:
                struct = _mpr_get_structure_by_material_id(
                    mpr,
                    mp_id,
                    final=mp_final,
                    conventional_unit_cell=mp_conventional_unit_cell,
                )
        except TypeError:
            struct = _mpr_get_structure_by_material_id(
                client,
                mp_id,
                final=mp_final,
                conventional_unit_cell=mp_conventional_unit_cell,
            )

        try:
            client.close()  # type: ignore[attr-defined]
        except Exception:
            pass

        if struct is None:
            raise ValueError(f"No structure found for mp-id='{mp_id}'")

        if mp_conventional_unit_cell:
            struct = _standardize_to_conventional(
                struct,
                symprec=mp_symprec,
                angle_tolerance=mp_angle_tolerance,
            )

        logger.info(
            f"Loaded MP structure: mp-id={mp_id}, n_atoms={len(struct)}, "
            f"formula={struct.composition.reduced_formula}"
        )
        return struct

    if src.kind == "cif":
        path = Path(src.ref)
        if not path.is_file():
            raise FileNotFoundError(f"CIF not found: {path}")
        return Structure.from_file(path)

    if src.kind == "ase":
        if not isinstance(src.ref, Atoms):
            raise TypeError("BulkSource(kind='ase') requires ref to be ase.Atoms.")
        return AseAtomsAdaptor.get_structure(src.ref)

    if src.kind == "pmg":
        if not isinstance(src.ref, Structure):
            raise TypeError("BulkSource(kind='pmg') requires ref to be pymatgen Structure.")
        return src.ref

    raise ValueError(f"Unknown BulkSource kind: {src.kind}")


def generate_bulk(spec: BulkSpec) -> Atoms:
    struct = load_bulk_structure(
        spec.bulk_source,
        mp_final=spec.mp_final,
        mp_conventional_unit_cell=spec.mp_conventional_unit_cell,
        mp_symprec=spec.mp_symprec,
        mp_angle_tolerance=spec.mp_angle_tolerance,
    )
    return AseAtomsAdaptor.get_atoms(struct)


# ----------------------------------------------------------------------
# Layer analysis
# ----------------------------------------------------------------------
def _compute_layer_indices(atoms: Atoms, tol: float = 0.3) -> Dict[int, np.ndarray]:
    pos = atoms.get_positions()
    if pos.size == 0:
        return {}

    z = pos[:, 2]
    order = np.argsort(z)
    z_sorted = z[order]

    layers: Dict[int, List[int]] = {}
    current_layer = [int(order[0])]
    z_ref = float(z_sorted[0])
    layer_id = 0

    for idx, zi in zip(order[1:], z_sorted[1:]):
        if abs(float(zi) - z_ref) <= tol:
            current_layer.append(int(idx))
        else:
            layers[layer_id] = current_layer
            layer_id += 1
            current_layer = [int(idx)]
            z_ref = float(zi)

    layers[layer_id] = current_layer
    return {lid: np.array(idxs, dtype=int) for lid, idxs in layers.items()}


def _top_layer_indices(atoms: Atoms, layers_from_top: int, tol: float) -> np.ndarray:
    layer_map = _compute_layer_indices(atoms, tol=tol)
    if not layer_map:
        return np.array([], dtype=int)

    max_layer_id = max(layer_map.keys())
    idxs: List[int] = []
    for i in range(int(layers_from_top)):
        lid = max_layer_id - i
        if lid in layer_map:
            idxs.extend(layer_map[lid].tolist())

    if not idxs:
        return np.array([], dtype=int)
    return np.array(sorted(set(idxs)), dtype=int)


# ----------------------------------------------------------------------
# (Deprecated) Simple doping utilities
# ----------------------------------------------------------------------
def _compute_target_counts(total_sites: int, target_ratio: Dict[str, int], *, exact: bool) -> Dict[str, int]:
    ratio_sum = int(sum(target_ratio.values()))
    if ratio_sum <= 0 or total_sites <= 0:
        return {}

    if exact and (total_sites % ratio_sum == 0):
        mult = total_sites // ratio_sum
        return {el: int(n) * mult for el, n in target_ratio.items()}

    target_counts_float = {el: total_sites * (n / ratio_sum) for el, n in target_ratio.items()}
    target_counts = {el: int(round(v)) for el, v in target_counts_float.items()}

    diff = total_sites - sum(target_counts.values())
    if diff != 0:
        main_el = max(target_ratio.items(), key=lambda kv: kv[1])[0]
        target_counts[main_el] = target_counts.get(main_el, 0) + diff

    return target_counts


def apply_layer_doping(atoms: Atoms, spec: DopingSpec) -> Atoms:
    doped = atoms.copy()
    symbols = np.array(doped.get_chemical_symbols(), dtype=object)

    top_idx = _top_layer_indices(doped, spec.layers_from_top, spec.layer_tol)
    if len(top_idx) == 0:
        logger.warning("No layers detected; skipping doping.")
        return doped

    total_sites = int(len(top_idx))
    ratio_sum = int(sum(spec.target_ratio.values()))
    if ratio_sum <= 0 or total_sites <= 0:
        logger.warning("Invalid target_ratio or no sites in target layers; skipping.")
        return doped

    target_counts = _compute_target_counts(total_sites, spec.target_ratio, exact=False)
    layer_symbols = symbols[top_idx]
    current_counts = Counter(layer_symbols.tolist())

    to_increase: Dict[str, int] = {}
    to_decrease: Dict[str, int] = {}

    all_keys = set(current_counts.keys()) | set(target_counts.keys())
    for el in all_keys:
        cur_n = int(current_counts.get(el, 0))
        tgt_n = int(target_counts.get(el, 0))
        if tgt_n > cur_n:
            to_increase[el] = tgt_n - cur_n
        elif cur_n > tgt_n:
            to_decrease[el] = cur_n - tgt_n

    if not to_increase and not to_decrease:
        return doped

    rng = np.random.default_rng(int(spec.rng_seed))
    dec_indices_by_el: Dict[str, np.ndarray] = {}
    for el, dec_n in to_decrease.items():
        if dec_n <= 0:
            continue
        mask = (layer_symbols == el)
        dec_indices_by_el[el] = top_idx[mask]

    for el_add, add_n in to_increase.items():
        remaining = int(add_n)
        donors_sorted = sorted(dec_indices_by_el.items(), key=lambda kv: len(kv[1]), reverse=True)

        for el_dec, pool in donors_sorted:
            if remaining <= 0:
                break
            if len(pool) == 0:
                continue

            choose_n = min(remaining, len(pool))
            chosen = rng.choice(pool, size=choose_n, replace=False)
            symbols[chosen] = el_add
            dec_indices_by_el[el_dec] = np.setdiff1d(pool, chosen, assume_unique=False)
            remaining -= choose_n

    doped.set_chemical_symbols(symbols.tolist())
    return doped


# ----------------------------------------------------------------------
# XY scale + ratio tuning
# ----------------------------------------------------------------------
def _xy_aspect_after_repeat(atoms: Atoms, nx: int, ny: int) -> float:
    cell = atoms.get_cell()
    a_len, b_len = float(cell.lengths()[0]), float(cell.lengths()[1])
    if b_len <= 1e-12:
        return float("inf")
    return (nx * a_len) / (ny * b_len)


def _xy_repeat_score(
    atoms: Atoms,
    nx: int,
    ny: int,
    *,
    dist_to_divisible: int,
    prefer_square_xy: bool,
) -> Dict[str, Any]:
    aspect = _xy_aspect_after_repeat(atoms, nx, ny)
    if aspect > 0 and np.isfinite(aspect):
        aspect_term = abs(math.log(aspect))
    else:
        aspect_term = 10.0

    area_term = float(nx * ny)
    aniso_term = float(abs(nx - ny))

    # weights
    dist_weight = 2.5
    w_aspect = 6.0
    w_aniso = 0.6 if prefer_square_xy else 0.0

    score = (dist_weight * float(dist_to_divisible)) + area_term + (w_aspect * float(aspect_term)) + (w_aniso * aniso_term)

    return {
        "score": float(score),
        "nx": int(nx),
        "ny": int(ny),
        "area_term": float(area_term),
        "aspect": float(aspect),
        "aspect_term": float(aspect_term),
        "aniso_term": float(aniso_term),
        "dist": int(dist_to_divisible),
    }


def _pick_repeat_xy_for_ratio(
    atoms: Atoms,
    base_sites: int,
    ratio_sum: int,
    *,
    base_n_atoms: int,
    max_atoms_after_repeat: int,
    max_xy: int,
    min_sites: int,
    exact_divisible: bool,
    prefer_square_xy: bool,
) -> Tuple[int, int, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "base_sites": int(base_sites),
        "ratio_sum": int(ratio_sum),
        "base_n_atoms": int(base_n_atoms),
        "max_atoms_after_repeat": int(max_atoms_after_repeat),
        "max_xy": int(max_xy),
        "min_sites": int(min_sites),
        "exact_divisible": bool(exact_divisible),
        "prefer_square_xy": bool(prefer_square_xy),
    }

    if base_sites <= 0 or ratio_sum <= 0:
        meta["reason"] = "invalid_base_sites_or_ratio_sum"
        return 1, 1, meta

    best: Optional[Dict[str, Any]] = None
    ranked: List[Dict[str, Any]] = []

    for nx in range(1, int(max_xy) + 1):
        for ny in range(1, int(max_xy) + 1):
            total_atoms = int(base_n_atoms) * int(nx) * int(ny)
            if total_atoms > int(max_atoms_after_repeat):
                continue

            total_sites = int(base_sites) * int(nx) * int(ny)
            if total_sites < int(min_sites):
                continue

            if exact_divisible:
                if total_sites % int(ratio_sum) != 0:
                    continue
                dist = 0
            else:
                rem = total_sites % int(ratio_sum)
                dist = int(min(rem, int(ratio_sum) - rem))

            sc = _xy_repeat_score(
                atoms,
                int(nx),
                int(ny),
                dist_to_divisible=int(dist),
                prefer_square_xy=bool(prefer_square_xy),
            )
            sc["total_sites"] = int(total_sites)
            sc["total_atoms"] = int(total_atoms)
            ranked.append(sc)

            if best is None:
                best = sc
            else:
                cur_key = (sc["score"], sc["area_term"], sc["aniso_term"], max(sc["nx"], sc["ny"]), sc["nx"], sc["ny"])
                best_key = (best["score"], best["area_term"], best["aniso_term"], max(best["nx"], best["ny"]), best["nx"], best["ny"])
                if cur_key < best_key:
                    best = sc

    if best is None:
        meta["reason"] = "no_repeat_found_under_constraints"
        return 1, 1, meta

    meta["chosen"] = {
        "nx": int(best["nx"]),
        "ny": int(best["ny"]),
        "total_sites": int(best["total_sites"]),
        "total_atoms": int(best["total_atoms"]),
        "aspect": float(best["aspect"]),
        "score": float(best["score"]),
        "terms": {
            "area_term": float(best["area_term"]),
            "aspect_term": float(best["aspect_term"]),
            "aniso_term": float(best["aniso_term"]),
            "dist": int(best["dist"]),
        },
    }

    ranked_sorted = sorted(
        ranked,
        key=lambda d: (d["score"], d["area_term"], d["aniso_term"], max(d["nx"], d["ny"]), d["nx"], d["ny"]),
    )[:10]
    meta["top10"] = [
        {
            "nx": int(d["nx"]),
            "ny": int(d["ny"]),
            "total_sites": int(d["total_sites"]),
            "total_atoms": int(d["total_atoms"]),
            "aspect": float(d["aspect"]),
            "dist": int(d["dist"]),
            "score": float(d["score"]),
        }
        for d in ranked_sorted
    ]

    return int(best["nx"]), int(best["ny"]), meta


def _candidate_mask(
    symbols: np.ndarray,
    *,
    candidate_elements: Optional[List[str]],
    exclude_elements: Tuple[str, ...],
) -> np.ndarray:
    if candidate_elements is not None:
        cand_arr = np.array(candidate_elements, dtype=object)
        return np.isin(symbols, cand_arr)
    ex_arr = np.array(list(exclude_elements), dtype=object)
    return ~np.isin(symbols, ex_arr)


def scale_xy_and_tune_ratio(atoms: Atoms, spec: RatioTuneSpec) -> Tuple[Atoms, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "target_ratio": dict(spec.target_ratio),
        "layers_from_top": int(spec.layers_from_top),
        "layer_tol": float(spec.layer_tol),
        "auto_scale_xy": bool(spec.auto_scale_xy),
        "exact_divisible": bool(spec.exact_divisible),
        "max_xy": int(spec.max_xy),
        "min_sites": int(spec.min_sites),
        "candidate_elements": list(spec.candidate_elements) if spec.candidate_elements else None,
        "exclude_elements": list(spec.exclude_elements),
        "rng_seed": int(spec.rng_seed),
        "max_atoms_after_repeat": int(spec.max_atoms_after_repeat),
        "prefer_square_xy": bool(spec.prefer_square_xy),
        "warnings": [],
    }

    if not spec.target_ratio or int(sum(spec.target_ratio.values())) <= 0:
        meta["error"] = "invalid_target_ratio"
        return atoms.copy(), meta

    work = atoms.copy()

    top_idx = _top_layer_indices(work, spec.layers_from_top, spec.layer_tol)
    if len(top_idx) == 0:
        meta["error"] = "no_layers_detected"
        return work, meta

    symbols = np.array(work.get_chemical_symbols(), dtype=object)
    meta["present_elements_in_top"] = sorted(set(symbols[top_idx].tolist()))

    cand_mask_all = _candidate_mask(
        symbols,
        candidate_elements=spec.candidate_elements,
        exclude_elements=spec.exclude_elements,
    )
    cand_idx_base = top_idx[cand_mask_all[top_idx]]
    base_sites = int(len(cand_idx_base))
    meta["base_candidate_sites"] = base_sites

    ratio_sum = int(sum(spec.target_ratio.values()))

    if spec.candidate_elements is not None and base_sites == 0:
        meta["error"] = "candidate_elements_not_found_in_top_layers"
        return work, meta

    # 2) auto-scale XY (aspect-aware + atom-cap)
    nx = ny = 1
    if spec.auto_scale_xy and base_sites > 0:
        nx, ny, rep_meta = _pick_repeat_xy_for_ratio(
            work,
            base_sites,
            ratio_sum,
            base_n_atoms=len(work),
            max_atoms_after_repeat=int(spec.max_atoms_after_repeat),
            max_xy=int(spec.max_xy),
            min_sites=int(spec.min_sites),
            exact_divisible=bool(spec.exact_divisible),
            prefer_square_xy=bool(spec.prefer_square_xy),
        )

        # Exact divisible repeat not found → try approximate fallback
        if (nx, ny) == (1, 1) and spec.exact_divisible:
            nx2, ny2, rep_meta2 = _pick_repeat_xy_for_ratio(
                work,
                base_sites,
                ratio_sum,
                base_n_atoms=len(work),
                max_atoms_after_repeat=int(spec.max_atoms_after_repeat),
                max_xy=int(spec.max_xy),
                min_sites=int(spec.min_sites),
                exact_divisible=False,
                prefer_square_xy=bool(spec.prefer_square_xy),
            )
            if (nx2, ny2) != (1, 1):
                meta["warnings"].append("Exact divisible repeat not found; using approximate repeat.")
                nx, ny = nx2, ny2
                rep_meta["fallback_approx"] = rep_meta2.get("chosen", rep_meta2)

        meta["repeat_search"] = rep_meta

        if (nx, ny) != (1, 1):
            work = work.repeat((nx, ny, 1))
            meta["applied_repeat_xy"] = (int(nx), int(ny))

    # 3) Recompute candidate sites after repeat
    top_idx = _top_layer_indices(work, spec.layers_from_top, spec.layer_tol)
    symbols = np.array(work.get_chemical_symbols(), dtype=object)

    cand_mask_all = _candidate_mask(
        symbols,
        candidate_elements=spec.candidate_elements,
        exclude_elements=spec.exclude_elements,
    )
    cand_idx = top_idx[cand_mask_all[top_idx]]
    total_sites = int(len(cand_idx))
    meta["final_candidate_sites"] = total_sites
    meta["final_n_atoms"] = int(len(work))

    if total_sites <= 0:
        meta["error"] = "no_candidate_sites_after_repeat"
        return work, meta

    # 4) Target vs current counts
    cur_counts = Counter(symbols[cand_idx].tolist())
    target_counts = _compute_target_counts(total_sites, spec.target_ratio, exact=bool(spec.exact_divisible))

    meta["current_counts"] = dict(cur_counts)
    meta["target_counts"] = dict(target_counts)

    to_increase: Dict[str, int] = {}
    to_decrease: Dict[str, int] = {}

    all_keys = set(cur_counts.keys()) | set(target_counts.keys())
    for el in all_keys:
        cur_n = int(cur_counts.get(el, 0))
        tgt_n = int(target_counts.get(el, 0))
        if tgt_n > cur_n:
            to_increase[el] = tgt_n - cur_n
        elif cur_n > tgt_n:
            to_decrease[el] = cur_n - tgt_n

    if not to_increase and not to_decrease:
        meta["note"] = "already_matches_target"
        meta["final_counts"] = dict(cur_counts)
        if not meta["warnings"]:
            meta.pop("warnings", None)
        return work, meta

    rng = np.random.default_rng(int(spec.rng_seed))

    # Donor (excess) index pool
    dec_indices_by_el: Dict[str, np.ndarray] = {}
    for el, dec_n in to_decrease.items():
        if dec_n <= 0:
            continue
        mask_el = (symbols[cand_idx] == el)
        dec_indices_by_el[el] = cand_idx[mask_el]

    # 6) Perform substitution
    for el_add, add_n in sorted(to_increase.items(), key=lambda kv: kv[0]):
        remaining = int(add_n)
        if remaining <= 0:
            continue

        donors_sorted = sorted(dec_indices_by_el.items(), key=lambda kv: len(kv[1]), reverse=True)

        for el_dec, pool in donors_sorted:
            if remaining <= 0:
                break
            if len(pool) == 0:
                continue

            choose_n = min(remaining, len(pool))
            chosen = rng.choice(pool, size=choose_n, replace=False)
            symbols[chosen] = el_add

            dec_indices_by_el[el_dec] = np.setdiff1d(pool, chosen, assume_unique=False)
            remaining -= choose_n

        if remaining > 0:
            meta["warnings"].append(
                f"Not enough donor sites to create {el_add}: requested={add_n}, created={add_n-remaining}"
            )

    work.set_chemical_symbols(symbols.tolist())

    final_symbols = np.array(work.get_chemical_symbols(), dtype=object)
    final_counts = Counter(final_symbols[cand_idx].tolist())
    meta["final_counts"] = dict(final_counts)

    if not meta["warnings"]:
        meta.pop("warnings", None)

    return work, meta


# ----------------------------------------------------------------------
# CIF export
# ----------------------------------------------------------------------
def atoms_to_cif_bytes(atoms: Atoms, *, symprec: float = 0.1) -> bytes:
    struct = AseAtomsAdaptor.get_structure(atoms)
    cw = CifWriter(struct, symprec=symprec)
    return str(cw).encode("utf-8")


def save_atoms_to_cif(atoms: Atoms, path: Union[str, Path], *, symprec: float = 0.1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    struct = AseAtomsAdaptor.get_structure(atoms)
    CifWriter(struct, symprec=symprec).write_file(str(path))
