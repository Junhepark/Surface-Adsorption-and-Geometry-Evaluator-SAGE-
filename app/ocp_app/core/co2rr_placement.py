"""Surface-aware placement engine for SAGE CO2RR adsorbates.

This module maps registry-defined molecular anchors to explicit surface support
atoms.  It is shared by preview, AdsorbML-lite screening, and the final CHE
workflow so all three paths start from the same geometry.

The generated structures are initial seeds, not universal relaxed minima.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms

from ocp_app.core.co2rr_geometry import orient_co2rr_template
from ocp_app.core.co2rr_registry import (
    CO2RRBindingVariant,
    co2rr_site_allowed,
    get_co2rr_adsorbate_spec,
    get_co2rr_binding_variant,
    normalize_co2rr_site_kind,
)

_EPS = 1.0e-10
# Elements treated as framework anions when an oxide-like surface is detected.
_ANION_SYMBOLS = frozenset({"O", "S", "SE", "TE", "F", "CL", "BR", "I", "N", "P"})


@dataclass(frozen=True)
class CO2RRPlacementResult:
    adsorbate_atoms: Atoms
    molecular_anchor_indices: tuple[int, ...]
    surface_support_indices: tuple[int, ...]
    target_anchor_xyz: tuple[float, float, float]
    binding_mode: str
    surface_anchor_family: str
    target_bond_length_A: float
    achieved_support_distances_A: tuple[float, ...]
    minimum_adsorbate_slab_distance_A: float
    azimuth_deg: float
    seed_valid: bool = True
    seed_validation: dict[str, object] | None = None
    binding_variant_key: str = "default"
    binding_variant_label: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "binding_variant_key": self.binding_variant_key,
            "binding_variant_label": self.binding_variant_label,
            "binding_mode": self.binding_mode,
            "surface_anchor_family": self.surface_anchor_family,
            "molecular_anchor_indices": list(self.molecular_anchor_indices),
            "surface_support_indices": list(self.surface_support_indices),
            "target_anchor_xyz": list(self.target_anchor_xyz),
            "target_bond_length_A": float(self.target_bond_length_A),
            "achieved_support_distances_A": list(self.achieved_support_distances_A),
            "minimum_adsorbate_slab_distance_A": float(self.minimum_adsorbate_slab_distance_A),
            "azimuth_deg": float(self.azimuth_deg),
            "seed_valid": bool(self.seed_valid),
            "seed_validation": dict(self.seed_validation or {}),
        }


def _clean_kind(kind: object) -> str:
    return normalize_co2rr_site_kind(kind)


def _is_oxide_like(slab: Atoms, mtype: str | None = None) -> bool:
    if str(mtype or "").strip().lower() == "oxide":
        return True
    if str(mtype or "").strip().lower() == "metal":
        return False
    syms = [str(s).upper() for s in slab.get_chemical_symbols()]
    return ("O" in syms) and any(s not in _ANION_SYMBOLS for s in syms)


def _support_element_mask(slab: Atoms, mtype: str | None = None) -> np.ndarray:
    syms = [str(s).upper() for s in slab.get_chemical_symbols()]
    if _is_oxide_like(slab, mtype=mtype):
        return np.asarray([s not in _ANION_SYMBOLS for s in syms], dtype=bool)
    return np.ones(len(syms), dtype=bool)


def _mic_delta(slab: Atoms, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Return minimum-image p1-p0, wrapping x/y only."""
    d = np.asarray(p1, dtype=float).reshape(3) - np.asarray(p0, dtype=float).reshape(3)
    try:
        cell = np.asarray(slab.get_cell(), dtype=float)
        frac = np.linalg.solve(cell.T, d)
        pbc = np.asarray(slab.get_pbc(), dtype=bool)
        for ax in (0, 1):
            if bool(pbc[ax]):
                frac[ax] -= np.round(frac[ax])
        return frac @ cell
    except Exception:
        return d


def _unwrap_positions(slab: Atoms, indices: Sequence[int]) -> np.ndarray:
    pos = np.asarray(slab.get_positions(), dtype=float)
    ids = [int(i) for i in indices]
    if not ids:
        return np.empty((0, 3), dtype=float)
    ref = pos[ids[0]].copy()
    out = [ref]
    for i in ids[1:]:
        out.append(ref + _mic_delta(slab, ref, pos[i]))
    return np.asarray(out, dtype=float)


def _desired_support_count(site_kind: str, surface_anchor_family: str) -> int:
    if str(surface_anchor_family) == "cation_pair":
        return 2
    kind = _clean_kind(site_kind)
    if kind == "bridge":
        return 2
    if kind == "hollow":
        return 3
    return 1


def _site_position(site, slab: Atoms) -> np.ndarray:
    try:
        p = np.asarray(getattr(site, "position"), dtype=float).reshape(3)
        if np.isfinite(p).all():
            return p
    except Exception:
        pass
    pos = np.asarray(slab.get_positions(), dtype=float)
    return np.asarray([pos[:, 0].mean(), pos[:, 1].mean(), pos[:, 2].max()], dtype=float)


def _valid_surface_indices(site, slab: Atoms, support_mask: np.ndarray) -> list[int]:
    out: list[int] = []
    try:
        raw = tuple(int(i) for i in (getattr(site, "surface_indices", ()) or ()))
    except Exception:
        raw = tuple()
    for i in raw:
        if 0 <= i < len(slab) and bool(support_mask[i]) and i not in out:
            out.append(i)
    return out


def _top_support_candidates(
    slab: Atoms,
    site_xyz: np.ndarray,
    support_mask: np.ndarray,
    *,
    radial_window_A: float = 4.5,
    exposure_tol_A: float = 0.70,
) -> list[int]:
    """Return locally exposed metal/cation atoms near the requested site.

    The previous implementation accepted support atoms up to 3.5 A below the
    uppermost layer.  For bidentate HCOO* this could pair one top-layer atom
    with a subsurface atom, placing the O--O anchor plane inside the slab.
    Support atoms are now restricted to the locally exposed layer.
    """
    pos = np.asarray(slab.get_positions(), dtype=float)
    ids = np.where(support_mask)[0].astype(int).tolist()
    if not ids:
        return []

    local = []
    for i in ids:
        d = _mic_delta(slab, site_xyz, pos[i])
        dxy = float(np.linalg.norm(d[:2]))
        if dxy <= float(radial_window_A):
            local.append((int(i), dxy))
    if not local:
        local = [(int(i), float(np.linalg.norm(_mic_delta(slab, site_xyz, pos[i])[:2]))) for i in ids]

    local_top_z = max(float(pos[i, 2]) for i, _ in local)
    exposed = [(i, dxy) for i, dxy in local if local_top_z - float(pos[i, 2]) <= float(exposure_tol_A)]
    if not exposed:
        exposed = local

    scored = [
        (float(dxy), float(local_top_z - pos[i, 2]), int(i))
        for i, dxy in exposed
    ]
    scored.sort()
    return [i for _dxy, _dz, i in scored]


def _choose_pair(
    slab: Atoms,
    candidates: Sequence[int],
    site_xyz: np.ndarray,
    *,
    midpoint_tol_A: float = 0.90,
    max_z_mismatch_A: float = 0.18,
) -> tuple[int, int]:
    """Choose a nearest-neighbour, coplanar support pair centered on the site.

    HCOO* is not allowed to invent an arbitrary pair around an ontop/hollow
    basin.  Only exposed pairs whose midpoint coincides with the requested
    bridge/cation-pair site are accepted.
    """
    pos = np.asarray(slab.get_positions(), dtype=float)
    ids = [int(i) for i in candidates]
    raw: list[tuple[float, float, float, int, int]] = []
    for a_i, i in enumerate(ids):
        for j in ids[a_i + 1:]:
            pair = _unwrap_positions(slab, (i, j))
            dz = abs(float(pair[0, 2] - pair[1, 2]))
            if dz > float(max_z_mismatch_A):
                continue
            dv = pair[1] - pair[0]
            sep = float(np.linalg.norm(dv[:2]))
            if sep < 1.40 or sep > 5.00:
                continue
            mid = pair.mean(axis=0)
            dmid = _mic_delta(slab, site_xyz, mid)
            midpoint_error = float(np.linalg.norm(dmid[:2]))
            if midpoint_error > float(midpoint_tol_A):
                continue
            raw.append((sep, midpoint_error, dz, int(i), int(j)))

    if not raw:
        raise ValueError(
            "HCOO* requires an exposed, approximately coplanar bridge/cation-pair "
            "whose midpoint matches the requested site; no valid pair was found."
        )

    # Keep only the local nearest-neighbour shell. This avoids selecting a long
    # diagonal pair merely because its midpoint happens to lie near the site.
    nn_sep = min(x[0] for x in raw)
    nn_tol = max(0.25, 0.12 * float(nn_sep))
    shell = [x for x in raw if x[0] <= float(nn_sep) + float(nn_tol)]
    shell.sort(key=lambda x: (x[1], x[2], abs(x[0] - nn_sep), -float((pos[x[3], 2] + pos[x[4], 2]) / 2.0)))
    best = shell[0]
    return int(best[3]), int(best[4])


def select_surface_support_indices(
    slab: Atoms,
    site,
    adsorbate: str,
    *,
    mtype: str | None = None,
    binding_variant: CO2RRBindingVariant | None = None,
) -> tuple[int, ...]:
    """Map a requested site basin to explicit metal/cation support atoms."""
    spec = get_co2rr_adsorbate_spec(adsorbate)
    variant = binding_variant or get_co2rr_binding_variant(adsorbate)
    site_kind = _clean_kind(getattr(site, "kind", "ontop"))
    if not co2rr_site_allowed(
        adsorbate, site_kind, binding_variant=variant
    ):
        raise ValueError(
            f"{variant.label} cannot be seeded on site kind {site_kind!r}; "
            f"allowed site kinds are {tuple(variant.allowed_site_kinds)!r}."
        )
    support_mask = _support_element_mask(slab, mtype=mtype)
    site_xyz = _site_position(site, slab)
    kind = _clean_kind(getattr(site, "kind", "ontop"))
    need = _desired_support_count(kind, variant.surface_anchor_family)

    exposed = _top_support_candidates(slab, site_xyz, support_mask)
    exposed_set = set(int(i) for i in exposed)
    # surface_indices may include subsurface atoms for hollow/bridge taxonomy;
    # retain only indices belonging to the locally exposed support layer.
    valid = [i for i in _valid_surface_indices(site, slab, support_mask) if i in exposed_set]
    candidates = valid + [i for i in exposed if i not in valid]
    if not candidates:
        raise ValueError("No metal/cation surface support atoms are available for CO2RR placement")

    if variant.surface_anchor_family == "cation_pair":
        # Prefer the explicit bridge-defining pair carried by the site object,
        # but subject it to the same strict coplanar/midpoint validation.
        used_explicit_pair = len(valid) >= 2
        pair_pool = valid[:2] if used_explicit_pair else candidates[:20]
        try:
            return tuple(_choose_pair(slab, pair_pool, site_xyz))
        except ValueError:
            if used_explicit_pair:
                return tuple(_choose_pair(slab, candidates[:20], site_xyz))
            raise

    if len(valid) >= need:
        return tuple(valid[:need])
    return tuple(candidates[:need])


def _support_centroid(slab: Atoms, support_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    support = _unwrap_positions(slab, support_indices)
    if support.size == 0:
        raise ValueError("Empty surface support set")
    return support.mean(axis=0), support


def _solve_single_anchor_z(support: np.ndarray, target_xy: np.ndarray, bond_length_A: float) -> float:
    """Find z giving the closest common anchor-support bond length."""
    z0 = float(np.max(support[:, 2]) + 0.45)
    z1 = float(np.max(support[:, 2]) + max(4.0, bond_length_A + 2.0))
    grid = np.linspace(z0, z1, 601)
    dxy2 = np.sum((support[:, :2] - np.asarray(target_xy, dtype=float)[None, :]) ** 2, axis=1)
    dist = np.sqrt(dxy2[None, :] + (grid[:, None] - support[None, :, 2]) ** 2)
    objective = np.mean((dist - float(bond_length_A)) ** 2, axis=1)
    return float(grid[int(np.argmin(objective))])


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad)); s = float(np.sin(angle_rad))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rotation_from_to(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    a = np.asarray(v_from, dtype=float).reshape(3)
    b = np.asarray(v_to, dtype=float).reshape(3)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        return np.eye(3)
    a /= na; b /= nb
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.eye(3)
    if c < -1.0 + 1e-10:
        axis = np.cross(a, np.asarray([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-8:
            axis = np.cross(a, np.asarray([0.0, 1.0, 0.0]))
        axis /= max(float(np.linalg.norm(axis)), _EPS)
        x, y, z = axis
        K = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return np.eye(3) + 2.0 * (K @ K)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    K = np.asarray([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + (K @ K) * ((1.0 - c) / max(s * s, _EPS))


def _pairwise_min_distance(slab: Atoms, ads: Atoms, *, exclude_ads: Iterable[int] = ()) -> float:
    sp = np.asarray(slab.get_positions(), dtype=float)
    ap = np.asarray(ads.get_positions(), dtype=float)
    excluded = set(int(i) for i in exclude_ads)
    vals = []
    for ai, pa in enumerate(ap):
        if ai in excluded:
            continue
        for ps in sp:
            vals.append(float(np.linalg.norm(_mic_delta(slab, ps, pa))))
    return float(min(vals)) if vals else float("nan")


def _anchor_to_non_support_min_distance(
    slab: Atoms,
    ads: Atoms,
    anchor_indices: Sequence[int],
    support_indices: Sequence[int],
) -> float:
    """Minimum distance from molecular anchors to unintended slab atoms."""
    sp = np.asarray(slab.get_positions(), dtype=float)
    ap = np.asarray(ads.get_positions(), dtype=float)
    supports = set(int(i) for i in support_indices)
    vals = []
    for ai in anchor_indices:
        pa = ap[int(ai)]
        for si, ps in enumerate(sp):
            if int(si) in supports:
                continue
            vals.append(float(np.linalg.norm(_mic_delta(slab, ps, pa))))
    return float(min(vals)) if vals else float("nan")


def _support_distances(slab: Atoms, ads: Atoms, anchor_indices: Sequence[int], support_indices: Sequence[int]) -> tuple[float, ...]:
    ap = np.asarray(ads.get_positions(), dtype=float)
    sp = np.asarray(slab.get_positions(), dtype=float)
    if len(anchor_indices) == 1:
        a = ap[int(anchor_indices[0])]
        return tuple(float(np.linalg.norm(_mic_delta(slab, sp[int(i)], a))) for i in support_indices)
    out = []
    for ai, si in zip(anchor_indices, support_indices):
        out.append(float(np.linalg.norm(_mic_delta(slab, sp[int(si)], ap[int(ai)]))))
    return tuple(out)


def _score_candidate(slab: Atoms, ads: Atoms, anchor_indices: Sequence[int], support_indices: Sequence[int], d0: float) -> tuple[float, float]:
    non_anchor_min = _pairwise_min_distance(slab, ads, exclude_ads=anchor_indices)
    dists = _support_distances(slab, ads, anchor_indices, support_indices)
    bond_penalty = float(np.mean([(d - d0) ** 2 for d in dists])) if dists else 0.0
    score = (non_anchor_min if np.isfinite(non_anchor_min) else -100.0) - 0.4 * bond_penalty
    return float(score), float(non_anchor_min)


def _place_single_anchor(
    slab: Atoms,
    ads_template: Atoms,
    anchor_indices: tuple[int, ...],
    support_indices: tuple[int, ...],
    bond_length_A: float,
) -> tuple[Atoms, np.ndarray, float, float]:
    centroid, support = _support_centroid(slab, support_indices)
    target_xy = centroid[:2]
    target_z = _solve_single_anchor_z(support, target_xy, bond_length_A)
    target = np.asarray([target_xy[0], target_xy[1], target_z], dtype=float)

    base_pos = np.asarray(ads_template.get_positions(), dtype=float)
    anchor_center = base_pos[list(anchor_indices)].mean(axis=0)
    best = None
    for deg in range(0, 360, 30):
        R = _rotation_z(np.deg2rad(float(deg)))
        cand = ads_template.copy()
        p = (base_pos - anchor_center) @ R.T + target
        cand.set_positions(p)
        score, min_dist = _score_candidate(slab, cand, anchor_indices, support_indices, bond_length_A)
        if best is None or score > best[0]:
            best = (score, cand, float(deg), min_dist)
    assert best is not None
    return best[1], target, best[2], best[3]


def _place_bidentate_pair(
    slab: Atoms,
    ads_template: Atoms,
    anchor_indices: tuple[int, ...],
    support_indices: tuple[int, ...],
    bond_length_A: float,
) -> tuple[Atoms, np.ndarray, float, float]:
    if len(anchor_indices) != 2 or len(support_indices) != 2:
        raise ValueError("Bidentate placement requires two molecular and two surface anchors")
    support = _unwrap_positions(slab, support_indices)
    p0 = np.asarray(ads_template.get_positions(), dtype=float)
    a0, a1 = [int(i) for i in anchor_indices]
    mol_vec = p0[a1] - p0[a0]
    surf_vec = support[1] - support[0]
    # Align O-O with the surface pair in the xy plane while preserving the
    # outward (+z) molecular orientation established by orient_co2rr_template.
    mol_xy = np.asarray([mol_vec[0], mol_vec[1], 0.0], dtype=float)
    surf_xy = np.asarray([surf_vec[0], surf_vec[1], 0.0], dtype=float)
    R = _rotation_from_to(mol_xy, surf_xy)
    p1 = p0 @ R.T

    mol_sep = float(np.linalg.norm((p1[a1] - p1[a0])[:2]))
    surf_sep = float(np.linalg.norm(surf_xy[:2]))
    u = surf_xy / max(float(np.linalg.norm(surf_xy)), _EPS)
    midpoint = support.mean(axis=0)
    lateral_offset = 0.5 * abs(surf_sep - mol_sep)
    height = float(np.sqrt(max(bond_length_A ** 2 - lateral_offset ** 2, 0.65 ** 2)))
    target_mid = np.asarray([midpoint[0], midpoint[1], float(np.max(support[:, 2]) + height)], dtype=float)
    anchor_mid = p1[[a0, a1]].mean(axis=0)
    p1 = p1 - anchor_mid + target_mid

    # Scan the two O-to-support assignments and a small rigid upward lift.
    # The lift is used only when another top-layer atom crowds the O anchors;
    # it avoids inserting the formate plane into the surface while retaining a
    # near-target M--O distance.
    candidates = []
    for lift in np.linspace(0.0, 0.80, 9):
        lifted_mid = target_mid + np.asarray([0.0, 0.0, float(lift)])
        for deg in (0.0, 180.0):
            Rz = _rotation_z(np.deg2rad(deg))
            cand = ads_template.copy()
            p = (p1 - target_mid) @ Rz.T + lifted_mid
            cand.set_positions(p)
            non_anchor_min = _pairwise_min_distance(slab, cand, exclude_ads=anchor_indices)
            anchor_other_min = _anchor_to_non_support_min_distance(
                slab, cand, anchor_indices, support_indices
            )
            dists = _support_distances(slab, cand, anchor_indices, support_indices)
            bond_penalty = float(np.mean([(d - bond_length_A) ** 2 for d in dists])) if dists else 0.0
            collision_penalty = 0.0
            if np.isfinite(non_anchor_min) and non_anchor_min < 1.20:
                collision_penalty += 30.0 * (1.20 - non_anchor_min) ** 2
            if np.isfinite(anchor_other_min) and anchor_other_min < 1.35:
                collision_penalty += 30.0 * (1.35 - anchor_other_min) ** 2
            score = -8.0 * bond_penalty - collision_penalty - 0.15 * float(lift)
            metrics = _hcoo_seed_metrics(
                slab, cand, anchor_indices, support_indices
            )
            safe = bool(metrics.get("valid", False))
            candidates.append((safe, score, cand, deg, non_anchor_min, lifted_mid, float(lift), metrics))

    safe_candidates = [x for x in candidates if x[0]]
    if not safe_candidates:
        best_diag = max(candidates, key=lambda x: x[1])[7] if candidates else {}
        raise ValueError(
            "No physically valid bidentate HCOO* seed was generated for the selected "
            f"bridge/cation pair. Best diagnostic: {best_diag}"
        )
    # Prefer the smallest rigid lift, then the closest symmetric M--O geometry.
    best = max(safe_candidates, key=lambda x: (-x[6], x[1]))
    return best[2], best[5], float(best[3]), float(best[4])



def _hcoo_seed_metrics(
    slab: Atoms,
    ads: Atoms,
    anchor_indices: Sequence[int],
    support_indices: Sequence[int],
) -> dict[str, object]:
    """Validate the initial bidentate formate seed before relaxation."""
    if len(anchor_indices) != 2 or len(support_indices) != 2:
        return {"valid": False, "reason": "HCOO_requires_two_O_and_two_support_atoms"}
    support = _unwrap_positions(slab, support_indices)
    ap = np.asarray(ads.get_positions(), dtype=float)
    anchors = ap[np.asarray(anchor_indices, dtype=int)]
    dists = _support_distances(slab, ads, anchor_indices, support_indices)
    symbols = [str(x).upper() for x in ads.get_chemical_symbols()]
    c_idx = [i for i, x in enumerate(symbols) if x == "C"]
    h_idx = [i for i, x in enumerate(symbols) if x == "H"]
    o_mean_z = float(np.mean(anchors[:, 2]))
    c_z = float(ap[c_idx[0], 2]) if c_idx else float("nan")
    h_z = float(ap[h_idx[0], 2]) if h_idx else float("nan")
    non_anchor_min = _pairwise_min_distance(slab, ads, exclude_ads=anchor_indices)
    anchor_other_min = _anchor_to_non_support_min_distance(
        slab, ads, anchor_indices, support_indices
    )
    support_z_spread = float(np.ptp(support[:, 2]))
    oxygen_z_spread = float(np.ptp(anchors[:, 2]))
    bond_asymmetry = float(abs(dists[0] - dists[1])) if len(dists) == 2 else float("inf")
    checks = {
        "support_pair_coplanar": support_z_spread <= 0.18,
        "oxygen_pair_coplanar": oxygen_z_spread <= 0.18,
        "mo_distances_in_range": bool(len(dists) == 2 and all(1.75 <= float(d) <= 2.45 for d in dists)),
        "mo_distance_symmetric": bond_asymmetry <= 0.20,
        "carbon_above_oxygen_plane": bool(np.isfinite(c_z) and c_z >= o_mean_z + 0.10),
        "hydrogen_vacuum_facing": bool(np.isfinite(h_z) and np.isfinite(c_z) and h_z >= c_z - 0.05),
        "non_anchor_clearance": bool(not np.isfinite(non_anchor_min) or non_anchor_min >= 1.20),
        "unintended_anchor_clearance": bool(not np.isfinite(anchor_other_min) or anchor_other_min >= 1.35),
    }
    valid = bool(all(checks.values()))
    return {
        "valid": valid,
        "checks": checks,
        "support_z_spread_A": support_z_spread,
        "oxygen_z_spread_A": oxygen_z_spread,
        "MO_distances_A": [float(x) for x in dists],
        "MO_distance_asymmetry_A": bond_asymmetry,
        "carbon_minus_oxygen_plane_A": float(c_z - o_mean_z) if np.isfinite(c_z) else None,
        "hydrogen_minus_carbon_A": float(h_z - c_z) if np.isfinite(h_z) and np.isfinite(c_z) else None,
        "non_anchor_min_slab_distance_A": float(non_anchor_min),
        "anchor_to_non_support_min_distance_A": float(anchor_other_min),
    }

def place_co2rr_adsorbate(
    slab: Atoms,
    site,
    adsorbate: str,
    template_atoms: Atoms,
    *,
    mtype: str | None = None,
    binding_variant: CO2RRBindingVariant | None = None,
) -> CO2RRPlacementResult:
    """Place one CO2RR adsorbate using explicit molecular/surface anchors."""
    spec = get_co2rr_adsorbate_spec(adsorbate)
    variant = binding_variant or get_co2rr_binding_variant(adsorbate)
    oriented, molecular_anchors, _anchor_label = orient_co2rr_template(
        template_atoms,
        adsorbate,
        anchor_mode_override=variant.anchor_mode,
    )
    support_indices = select_surface_support_indices(
        slab,
        site,
        adsorbate,
        mtype=mtype,
        binding_variant=variant,
    )

    if variant.surface_anchor_family == "cation_pair":
        ads, target, azimuth, min_dist = _place_bidentate_pair(
            slab, oriented, molecular_anchors, support_indices, float(variant.target_bond_length_A)
        )
    else:
        ads, target, azimuth, min_dist = _place_single_anchor(
            slab, oriented, molecular_anchors, support_indices, float(variant.target_bond_length_A)
        )

    ads.set_cell(slab.get_cell())
    ads.set_pbc(slab.get_pbc())
    distances = _support_distances(slab, ads, molecular_anchors, support_indices)
    seed_validation = (
        _hcoo_seed_metrics(slab, ads, molecular_anchors, support_indices)
        if str(spec.key).upper() == "HCOO"
        else {"valid": True, "checks": {}}
    )
    if not bool(seed_validation.get("valid", False)):
        raise ValueError(f"Invalid {spec.label} seed: {seed_validation}")
    ads.info = dict(getattr(ads, "info", {}) or {})
    ads.info.update({
        "co2rr_binding_variant_key": str(variant.key),
        "co2rr_binding_variant_label": str(variant.label),
        "co2rr_binding_mode": str(variant.binding_mode),
        "co2rr_surface_anchor_family": str(variant.surface_anchor_family),
        "co2rr_surface_support_indices": list(support_indices),
        "co2rr_target_bond_length_A": float(variant.target_bond_length_A),
        "co2rr_support_distances_A": [float(x) for x in distances],
        "co2rr_azimuth_deg": float(azimuth),
        "co2rr_seed_valid": bool(seed_validation.get("valid", False)),
        "co2rr_seed_validation": dict(seed_validation),
    })
    return CO2RRPlacementResult(
        adsorbate_atoms=ads,
        molecular_anchor_indices=tuple(int(i) for i in molecular_anchors),
        surface_support_indices=tuple(int(i) for i in support_indices),
        target_anchor_xyz=tuple(float(x) for x in target),
        binding_mode=str(variant.binding_mode),
        surface_anchor_family=str(variant.surface_anchor_family),
        target_bond_length_A=float(variant.target_bond_length_A),
        achieved_support_distances_A=tuple(float(x) for x in distances),
        minimum_adsorbate_slab_distance_A=float(min_dist),
        azimuth_deg=float(azimuth),
        seed_valid=bool(seed_validation.get("valid", False)),
        seed_validation=dict(seed_validation),
        binding_variant_key=str(variant.key),
        binding_variant_label=str(variant.label),
    )
