from __future__ import annotations

from typing import List, Sequence

import numpy as np
from ase import Atom, Atoms
from ase.data import atomic_numbers

from ocp_app.core.ads_sites import (
    ANION_SYMBOLS,
    AdsSite,
    detect_metal_111_sites,
    detect_oxide_surface_sites,
)
from ocp_app.core.structure_check import _radius
from ocp_app.core.structure_ops import _recenter_slab_z_into_cell

from .environment import analyze_parent_slab, structure_content_signature
from .equivalence import canonical_site_kind, group_equivalent_sites, choose_center_preferred_site
from .models import EngineeredStructure, StructureRecipe
from .validation import validate_engineered_structure


def infer_surface_type(atoms: Atoms) -> str:
    symbols = atoms.get_chemical_symbols()
    has_anion = any(s in ANION_SYMBOLS for s in symbols)
    has_cation = any(s not in ANION_SYMBOLS for s in symbols)
    return "oxide" if has_anion and has_cation else "metal"


def detect_selectable_adatom_sites(
    atoms: Atoms,
    *,
    site_kinds: Sequence[str] = ("ontop", "bridge", "hollow"),
    max_sites_per_kind: int = 200,
) -> List[AdsSite]:
    """Return every detected upper-surface site allowed for manual adatom selection."""
    mtype = infer_surface_type(atoms)
    if mtype == "oxide":
        raw_sites = detect_oxide_surface_sites(atoms, max_sites_per_kind=int(max_sites_per_kind))
    else:
        raw_sites = detect_metal_111_sites(atoms, max_sites_per_kind=int(max_sites_per_kind))
    allowed = {canonical_site_kind(k) for k in site_kinds}
    sites = [
        s for s in raw_sites
        if canonical_site_kind(str(s.kind)) in allowed and bool(s.surface_indices)
    ]
    sites.sort(
        key=lambda s: (
            {"ontop": 0, "bridge": 1, "hollow": 2}.get(canonical_site_kind(str(s.kind)), 9),
            round(float(s.position[0]), 6),
            round(float(s.position[1]), 6),
            tuple(int(i) for i in s.surface_indices),
        )
    )
    return sites


def _initial_adatom_position(
    atoms: Atoms,
    site: AdsSite,
    adatom: str,
    distance_scale: float,
) -> np.ndarray:
    support = tuple(int(i) for i in site.surface_indices)
    if not support:
        raise ValueError("Adatom placement requires explicit support atom indices.")

    support_pos = np.asarray(atoms.positions[list(support)], dtype=float)
    support_symbols = [atoms[int(i)].symbol for i in support]
    xy = np.asarray(site.position[:2], dtype=float)
    support_z = float(np.max(support_pos[:, 2]))
    target_d = float(distance_scale) * (
        _radius(str(adatom)) + float(np.mean([_radius(s) for s in support_symbols]))
    )
    radial = np.linalg.norm(support_pos[:, :2] - xy[None, :], axis=1)
    radial_eff = float(np.mean(radial)) if radial.size else 0.0
    height = float(np.sqrt(max(target_d * target_d - radial_eff * radial_eff, (0.45 * target_d) ** 2)))
    pos = np.array([float(xy[0]), float(xy[1]), support_z + height], dtype=float)

    for _ in range(40):
        ok = True
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            d = float(np.linalg.norm(pos - np.asarray(atoms.positions[int(j)], dtype=float)))
            min_allowed = 0.72 * (_radius(str(adatom)) + _radius(str(sym)))
            if d < min_allowed:
                ok = False
                break
        if ok:
            break
        pos[2] += 0.10
    return pos


def build_adatom_candidate_at_site(
    atoms: Atoms,
    *,
    adatom: str,
    site: AdsSite,
    distance_scale: float = 1.0,
    site_index: int | None = None,
) -> EngineeredStructure:
    """Build one user-selected adatom candidate at an exact detected AdsSite."""
    if str(adatom) not in atomic_numbers:
        raise ValueError("Adatom must be a valid element symbol.")
    support = tuple(int(i) for i in site.surface_indices)
    if not support or any(i < 0 or i >= len(atoms) for i in support):
        raise ValueError("Selected adatom site has invalid support atom indices.")

    analysis = analyze_parent_slab(atoms)
    kind = canonical_site_kind(str(site.kind))
    position = _initial_adatom_position(atoms, site, str(adatom), float(distance_scale))
    child = atoms.copy()
    child.append(Atom(str(adatom), position=position))
    child = _recenter_slab_z_into_cell(child, margin=1.0)
    ad_idx = len(child) - 1
    support_symbols = tuple(atoms[int(i)].symbol for i in support)
    env_payload = {
        "site_kind": kind,
        "support_indices": support,
        "support_symbols": support_symbols,
        "support_environments": tuple(
            analysis["environment_by_index"][int(i)].as_dict() for i in support
        ),
    }
    eff = 1.0 / float(max(len(analysis.get("top_indices", ())), 1))
    parameters = {
        "parent_structure_signature": structure_content_signature(atoms),
        "adatom": str(adatom),
        "site_kind": kind,
        "selection_mode": "manual",
        "selected_site_index": None if site_index is None else int(site_index),
        "support_indices": support,
        "support_symbols": support_symbols,
        "initial_position_A": tuple(float(x) for x in position),
        "distance_scale": float(distance_scale),
        "effective_fraction": float(eff),
    }
    recipe = StructureRecipe(
        operation="single_adatom",
        parent_formula=atoms.get_chemical_formula(),
        parent_n_atoms=len(atoms),
        target_indices=support,
        target_environment=env_payload,
        added_elements=(str(adatom),),
        parameters=parameters,
    )
    site_tag = f"site{int(site_index)}" if site_index is not None else "manual"
    cid = f"ad_{adatom}_{kind}_{site_tag}_{recipe.stable_id()}"
    validation = validate_engineered_structure(
        child,
        parent_atoms=atoms,
        operation="adatom",
        modified_indices=(ad_idx,),
        effective_fraction=eff,
    )
    return EngineeredStructure(
        atoms=child,
        recipe=recipe,
        candidate_id=cid,
        label=f"{adatom} adatom | {kind} | {','.join(support_symbols)} | manual {site_tag}",
        validation=validation,
        atom_provenance={
            "parent_to_child": {int(i): int(i) for i in range(len(atoms))},
            "added_child_index": int(ad_idx),
            "support_parent_indices": support,
        },
    )


def _build_automatic_adatom_candidate(
    atoms: Atoms,
    *,
    adatom: str,
    site: AdsSite,
    orbit_id: int,
    equivalent_site_count: int,
    distance_scale: float,
) -> EngineeredStructure:
    support = tuple(int(i) for i in site.surface_indices)
    analysis = analyze_parent_slab(atoms)
    kind = canonical_site_kind(str(site.kind))
    position = _initial_adatom_position(atoms, site, str(adatom), float(distance_scale))
    child = atoms.copy()
    child.append(Atom(str(adatom), position=position))
    child = _recenter_slab_z_into_cell(child, margin=1.0)
    ad_idx = len(child) - 1
    support_symbols = tuple(atoms[int(i)].symbol for i in support)
    env_payload = {
        "site_kind": kind,
        "support_indices": support,
        "support_symbols": support_symbols,
        "support_environments": tuple(
            analysis["environment_by_index"][int(i)].as_dict() for i in support
        ),
    }
    eff = 1.0 / float(max(len(analysis.get("top_indices", ())), 1))
    recipe = StructureRecipe(
        operation="single_adatom",
        parent_formula=atoms.get_chemical_formula(),
        parent_n_atoms=len(atoms),
        target_indices=support,
        target_environment=env_payload,
        added_elements=(str(adatom),),
        parameters={
            "parent_structure_signature": structure_content_signature(atoms),
            "adatom": str(adatom),
            "site_kind": kind,
            "selection_mode": "automatic",
            "site_orbit_id": int(orbit_id),
            "equivalent_site_count": int(equivalent_site_count),
            "support_indices": support,
            "support_symbols": support_symbols,
            "initial_position_A": tuple(float(x) for x in position),
            "distance_scale": float(distance_scale),
            "effective_fraction": float(eff),
        },
    )
    cid = f"ad_{adatom}_{kind}_orbit{orbit_id}_{recipe.stable_id()}"
    validation = validate_engineered_structure(
        child,
        parent_atoms=atoms,
        operation="adatom",
        modified_indices=(ad_idx,),
        effective_fraction=eff,
    )
    return EngineeredStructure(
        atoms=child,
        recipe=recipe,
        candidate_id=cid,
        label=f"{adatom} adatom | {kind} | {','.join(support_symbols)} | orbit {orbit_id}",
        validation=validation,
        atom_provenance={
            "parent_to_child": {int(i): int(i) for i in range(len(atoms))},
            "added_child_index": int(ad_idx),
            "support_parent_indices": support,
        },
    )


def enumerate_adatom_candidates(
    atoms: Atoms,
    *,
    adatom: str,
    site_kinds: Sequence[str] = ("ontop", "bridge", "hollow"),
    distance_scale: float = 1.0,
    max_candidates: int = 20,
) -> List[EngineeredStructure]:
    if str(adatom) not in atomic_numbers:
        raise ValueError("Adatom must be a valid element symbol.")

    sites = detect_selectable_adatom_sites(
        atoms,
        site_kinds=site_kinds,
        max_sites_per_kind=200,
    )
    if not sites:
        raise ValueError("No requested adatom sites were detected on the upper surface.")

    analysis = analyze_parent_slab(atoms)
    groups = group_equivalent_sites(sites, atoms, analysis["environment_by_index"])
    out: List[EngineeredStructure] = []
    for orbit_id, group in enumerate(groups[: int(max_candidates)]):
        site = choose_center_preferred_site(atoms, group)
        out.append(
            _build_automatic_adatom_candidate(
                atoms,
                adatom=str(adatom),
                site=site,
                orbit_id=int(orbit_id),
                equivalent_site_count=int(len(group)),
                distance_scale=float(distance_scale),
            )
        )
    return out
