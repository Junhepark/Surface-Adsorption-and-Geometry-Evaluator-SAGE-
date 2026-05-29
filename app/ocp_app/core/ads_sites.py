# ocp_app/core/ads_sites.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple, Dict

import numpy as np
from ase import Atoms, Atom

# Site types: ontop / bridge / hollow (legacy aliases fcc / hcp kept for compatibility)
AdsKind = Literal["ontop", "bridge", "hollow", "fcc", "hcp", "cation", "oer_cation", "anion_ontop"]


@dataclass
class AdsSite:
    kind: AdsKind
    position: Tuple[float, float, float]
    surface_indices: Tuple[int, ...]  # Surface atom indices forming this site (1/2/3)


# Elements treated as anions on oxide surfaces
ANION_SYMBOLS = {"O", "N", "F", "Cl", "S", "Br", "I"}


def _is_oxide_like(atoms: Atoms) -> bool:
    """
    Classify structure as 'oxide-like' if both anion (from ANION_SYMBOLS)
    and non-anion species are present (e.g. NiO, CuO, LaNiO3).
    """
    symbols = atoms.get_chemical_symbols()
    has_anion = any(s in ANION_SYMBOLS for s in symbols)
    has_cation = any(s not in ANION_SYMBOLS for s in symbols)
    return has_anion and has_cation


def _top_layer_indices_z(atoms: Atoms, z_tol: float = 0.8) -> np.ndarray:
    """Return atom indices belonging to the topmost layer by z-coordinate."""
    pos = atoms.positions
    if len(pos) == 0:
        return np.array([], dtype=int)
    z = pos[:, 2]
    z_max = float(z.max())
    top_mask = (z_max - z) < z_tol
    return np.where(top_mask)[0]


def _top_layer_anion_indices_z(atoms: Atoms, z_tol: float = 0.8) -> np.ndarray:
    """
    Return indices of anion-like atoms (from ANION_SYMBOLS) that belong to
    the topmost layer by z-coordinate.  Used for O-top adsorbate placement
    on oxide surfaces.
    """
    pos = atoms.positions
    if len(pos) == 0:
        return np.array([], dtype=int)

    symbols = atoms.get_chemical_symbols()
    anion_mask = np.array([s in ANION_SYMBOLS for s in symbols])
    anion_idx = np.where(anion_mask)[0]
    if len(anion_idx) == 0:
        return np.array([], dtype=int)

    z = pos[anion_idx, 2]
    z_max = float(z.max())
    top_mask = (z_max - z) < z_tol
    return anion_idx[top_mask]


def _oxide_o_based_ads_position(
    slab: Atoms,
    site: AdsSite,
    h_oh: float = 1.0,
    extra_z: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Generate placement coordinates for H* on an oxide surface by
    positioning it above the nearest surface anion (preferably O).

    Procedure:
      1) Among top-layer anions, prefer O atoms.
      2) Find the anion closest in (x, y) to site.position.
      3) Place the adsorbate at (anion_x, anion_y, anion_z + h_oh + extra_z).

    Falls back to site.position + extra_z if no suitable anion is found.
    """
    pos = slab.positions
    symbols = slab.get_chemical_symbols()

    top_anion_idx = _top_layer_anion_indices_z(slab, z_tol=0.8)

    # Priority: filter for O only; fall back to all anions if none
    o_idx = [i for i in top_anion_idx if symbols[i] == "O"]
    candidates = o_idx if o_idx else list(top_anion_idx)

    # No anions at all → fall back to metal-like logic: site.position + extra_z
    if not candidates:
        x, y, z = site.position
        return float(x), float(y), float(z + extra_z)

    target_xy = np.array(site.position[:2], dtype=float)
    cand_xy = pos[candidates][:, :2]

    d2 = np.sum((cand_xy - target_xy) ** 2, axis=1)
    best_idx = candidates[int(np.argmin(d2))]
    o_pos = pos[best_idx].copy()

    h_pos_z = float(o_pos[2] + h_oh + extra_z)
    return float(o_pos[0]), float(o_pos[1]), h_pos_z


def _z_layer_groups(pos: np.ndarray, z_tol: float = 0.6) -> List[np.ndarray]:
    """Cluster atoms into z-layers ordered from top to bottom."""
    if pos.size == 0:
        return []
    z = np.asarray(pos[:, 2], dtype=float)
    order = np.argsort(z)[::-1]
    groups: List[List[int]] = []
    z_refs: List[float] = []
    for idx in order.tolist():
        zi = float(z[idx])
        if (not groups) or abs(zi - z_refs[-1]) > float(z_tol):
            groups.append([int(idx)])
            z_refs.append(zi)
        else:
            groups[-1].append(int(idx))
    return [np.asarray(g, dtype=int) for g in groups]


def _build_sites_from_top_indices(
    pos: np.ndarray,
    top_idx: np.ndarray,
    h_ontop: float = 1.0,
    h_bridge: float = 1.1,
    h_hollow: float = 1.2,
    second_idx: np.ndarray | None = None,
    hcp_match_frac: float = 0.35,
    *,
    hollow_kind: str = "fcc",
    split_hollow: bool = False,
) -> List[AdsSite]:
    """Generate ontop/bridge/hollow-type AdsSite objects from a set of top-layer indices.

    Parameters
    ----------
    hollow_kind
        Label to assign to 3-fold hollow sites when split_hollow=False.
    split_hollow
        If True, try to split hollow sites into fcc/hcp using second-layer registry.
        If False, all hollow sites are labeled with hollow_kind.
    """
    if len(top_idx) == 0:
        raise ValueError("Top-layer index set is empty in _build_sites_from_top_indices.")

    top_pos = pos[top_idx]
    xy = top_pos[:, :2]
    z_surf = float(top_pos[:, 2].mean())
    n_top = len(top_idx)

    # --- 1) Pairwise distances → edge / triangle candidates ---
    diff = xy[:, None, :] - xy[None, :, :]
    dmat = np.sqrt((diff ** 2).sum(axis=-1))
    dmat += np.eye(n_top) * 1e6

    d_min = float(dmat.min())
    if not np.isfinite(d_min) or d_min <= 0.0:
        raise ValueError("Failed to determine nearest-neighbor distance on top layer.")

    cut_edge = 1.25 * d_min

    edges = set()
    for i in range(n_top):
        for j in range(i + 1, n_top):
            if dmat[i, j] < cut_edge:
                edges.add((i, j))

    triangles: List[Tuple[int, int, int]] = []
    for i in range(n_top):
        for j in range(i + 1, n_top):
            for k in range(j + 1, n_top):
                if (i, j) in edges and (i, k) in edges and (j, k) in edges:
                    triangles.append((i, j, k))

    if not triangles:
        cut_tri = 1.35 * d_min
        for i in range(n_top):
            for j in range(i + 1, n_top):
                for k in range(j + 1, n_top):
                    dij = dmat[i, j]
                    dik = dmat[i, k]
                    djk = dmat[j, k]
                    dmax = max(dij, dik, djk)
                    dmin_loc = min(dij, dik, djk)
                    if dmax < cut_tri and dmax / max(dmin_loc, 1e-8) < 1.15:
                        triangles.append((i, j, k))

    ads_sites: List[AdsSite] = []

    # ontop
    for idx in top_idx:
        x, y, _ = pos[idx]
        ads_sites.append(
            AdsSite(
                kind="ontop",
                position=(float(x), float(y), float(z_surf + h_ontop)),
                surface_indices=(int(idx),),
            )
        )

    # bridge
    for i_local, j_local in sorted(edges):
        ia = top_idx[i_local]
        ib = top_idx[j_local]
        center = 0.5 * (pos[ia] + pos[ib])
        x, y = center[:2]
        ads_sites.append(
            AdsSite(
                kind="bridge",
                position=(float(x), float(y), float(z_surf + h_bridge)),
                surface_indices=(int(ia), int(ib)),
            )
        )
    second_xy = None
    if second_idx is not None and len(second_idx) > 0:
        second_xy = pos[np.asarray(second_idx, dtype=int), :2]

    hcp_tol = float(max(0.15, hcp_match_frac * d_min))

    # hollow (single "hollow" label by default; optional fcc/hcp split kept for compatibility)
    for i_local, j_local, k_local in triangles:
        ia = top_idx[i_local]
        ib = top_idx[j_local]
        ic = top_idx[k_local]
        centroid = (pos[ia] + pos[ib] + pos[ic]) / 3.0
        x, y = centroid[:2]

        kind = str(hollow_kind)
        if split_hollow and second_xy is not None and len(second_xy) > 0:
            d2 = np.linalg.norm(second_xy - np.asarray([x, y])[None, :], axis=1)
            if np.isfinite(d2).any() and float(np.min(d2)) <= hcp_tol:
                kind = "hcp"
            else:
                kind = "fcc"

        ads_sites.append(
            AdsSite(
                kind=kind,
                position=(float(x), float(y), float(z_surf + h_hollow)),
                surface_indices=(int(ia), int(ib), int(ic)),
            )
        )

    return ads_sites


def _dedupe_sites(
    sites: List[AdsSite],
    cell: np.ndarray,
    tol: float = 0.20,
    max_sites_per_kind: int = 50,
    edge_frac_cut: float = 0.02,
) -> List[AdsSite]:
    """
    Deduplicate nearly coincident AdsSite objects per kind using cell-aware
    fractional coordinate binning.

    Sites near the cell boundary (within edge_frac_cut in fractional x/y)
    are also removed as PBC-degenerate.  If all sites are removed, the
    original list is preserved.
    """
    a_vec, b_vec = cell[0], cell[1]

    def to_frac_xy(x: float, y: float) -> Tuple[float, float]:
        ax = np.linalg.norm(a_vec)
        bx = np.linalg.norm(b_vec)
        if ax < 1e-8:
            ax = 1.0
        if bx < 1e-8:
            bx = 1.0
        return x / ax, y / bx

    def frac_unit_xy(x: float, y: float) -> Tuple[float, float]:
        fx, fy = to_frac_xy(x, y)
        fx = fx - np.floor(fx)
        fy = fy - np.floor(fy)
        return fx, fy

    # 1) Dedupe by kind + fractional (fx, fy) bin
    seen: Dict[Tuple[str, int, int], bool] = {}
    deduped: List[AdsSite] = []

    for site in sites:
        x, y, _ = site.position
        fx, fy = to_frac_xy(x, y)
        key = (site.kind, round(fx / tol), round(fy / tol))
        if key in seen:
            continue
        seen[key] = True
        deduped.append(site)

    # 2) Remove sites too close to cell boundary (assumed PBC-degenerate)
    if edge_frac_cut > 0.0:
        filtered: List[AdsSite] = []
        for site in deduped:
            x, y, _ = site.position
            fx, fy = frac_unit_xy(x, y)
            if (
                fx < edge_frac_cut
                or fx > 1.0 - edge_frac_cut
                or fy < edge_frac_cut
                or fy > 1.0 - edge_frac_cut
            ):
                # Too close to cell boundary → skip
                continue
            filtered.append(site)
        # If all were removed, keep the original list
        if filtered:
            deduped = filtered

    # 3) Enforce maximum count per kind
    out: List[AdsSite] = []
    count: Dict[str, int] = {"ontop": 0, "bridge": 0, "hollow": 0, "fcc": 0, "hcp": 0}
    for site in deduped:
        c = count.get(site.kind, 0)
        if c >= max_sites_per_kind:
            continue
        count[site.kind] = c + 1
        out.append(site)

    return out



def _fallback_center_hollow_site(pos: np.ndarray, top_idx: np.ndarray, h_hollow: float = 1.2) -> AdsSite | None:
    """Best-effort fallback hollow seed from the 3 top-layer atoms nearest the in-plane center.

    Used only when no triangle-based hollow site is detected. This is a conservative rescue path
    for close-packed metal slabs where z-layer grouping / edge pruning missed a 3-fold hollow.
    """
    if top_idx is None or len(top_idx) < 3:
        return None
    top_idx = np.asarray(top_idx, dtype=int)
    top_xy = pos[top_idx][:, :2]
    center_xy = np.mean(top_xy, axis=0)
    d = np.linalg.norm(top_xy - center_xy[None, :], axis=1)
    order = np.argsort(d)
    pick = top_idx[order[:3]]
    if len(pick) < 3:
        return None
    centroid = np.mean(pos[pick], axis=0)
    z_surf = float(np.mean(pos[top_idx][:, 2]))
    return AdsSite(
        kind="hollow",
        position=(float(centroid[0]), float(centroid[1]), float(z_surf + h_hollow)),
        surface_indices=tuple(int(i) for i in pick.tolist()),
    )

# ----------------  Metal (111) slab  ----------------

def detect_metal_111_sites(
    atoms: Atoms,
    h_ontop: float = 1.0,
    h_bridge: float = 1.1,
    h_hollow: float = 1.2,
    max_sites_per_kind: int = 50,
) -> List[AdsSite]:
    """Detect ontop/bridge/hollow adsorption sites on a metal (111)-like slab."""
    pos = atoms.positions
    layers = _z_layer_groups(pos, z_tol=0.6)
    if not layers:
        raise ValueError("Could not identify top surface layer for metal slab.")

    top_idx = layers[0]
    raw_sites = _build_sites_from_top_indices(
        pos,
        top_idx,
        h_ontop=h_ontop,
        h_bridge=h_bridge,
        h_hollow=h_hollow,
        hollow_kind="hollow",
        split_hollow=False,
    )

    # Rescue path: if no hollow was detected at all, synthesize one from the 3 top-layer atoms
    # nearest the in-plane center. This keeps the metal branch from collapsing to ontop/bridge only.
    if not any(str(getattr(s, "kind", "")) == "hollow" for s in raw_sites):
        fb = _fallback_center_hollow_site(pos, top_idx, h_hollow=h_hollow)
        if fb is not None:
            raw_sites.append(fb)

    cell = atoms.get_cell()
    return _dedupe_sites(
        raw_sites,
        cell,
        tol=0.20,
        max_sites_per_kind=max_sites_per_kind,
        edge_frac_cut=0.02,
    )


def _top_layer_neighbor_counts(top_xy: np.ndarray) -> np.ndarray:
    if len(top_xy) == 0:
        return np.array([], dtype=int)
    if len(top_xy) == 1:
        return np.array([0], dtype=int)
    diff = top_xy[:, None, :] - top_xy[None, :, :]
    dmat = np.sqrt((diff ** 2).sum(axis=-1))
    dmat += np.eye(len(top_xy)) * 1e6
    d_min = float(np.min(dmat)) if len(top_xy) > 1 else np.inf
    if (not np.isfinite(d_min)) or d_min <= 0.0:
        return np.zeros(len(top_xy), dtype=int)
    cut = 1.25 * d_min
    return np.sum(dmat < cut, axis=1).astype(int)


def _site_frac_xy(site: AdsSite, cell: np.ndarray) -> Tuple[float, float]:
    a_vec = np.asarray(cell[0, :2], dtype=float)
    b_vec = np.asarray(cell[1, :2], dtype=float)
    A = np.column_stack([a_vec, b_vec])
    xy = np.asarray(site.position[:2], dtype=float)
    try:
        frac = np.linalg.solve(A, xy)
        fx, fy = frac[0] - np.floor(frac[0]), frac[1] - np.floor(frac[1])
        return float(fx), float(fy)
    except Exception:
        ax = max(float(np.linalg.norm(a_vec)), 1.0)
        bx = max(float(np.linalg.norm(b_vec)), 1.0)
        return float((xy[0] / ax) % 1.0), float((xy[1] / bx) % 1.0)


def _filter_oxide_sites_terrace_like(
    sites: List[AdsSite],
    pos: np.ndarray,
    top_idx: np.ndarray,
    cell: np.ndarray,
    boundary_frac_cut: float = 0.08,
    min_mean_nn: float = 2.0,
) -> List[AdsSite]:
    if not sites or len(top_idx) == 0:
        return sites
    top_idx = np.asarray(top_idx, dtype=int)
    nn_counts = _top_layer_neighbor_counts(pos[top_idx][:, :2])
    top_lookup = {int(idx): int(nn_counts[i]) for i, idx in enumerate(top_idx.tolist())}
    kept: List[AdsSite] = []
    for s in sites:
        fx, fy = _site_frac_xy(s, cell)
        edge_margin = min(fx, 1.0 - fx, fy, 1.0 - fy)
        if edge_margin < float(boundary_frac_cut):
            continue
        surf_idx = tuple(int(i) for i in (getattr(s, 'surface_indices', ()) or ()))
        if surf_idx:
            support = [top_lookup.get(int(i), 0) for i in surf_idx]
            mean_nn = float(np.mean(support)) if support else 0.0
            if mean_nn < float(min_mean_nn):
                continue
        kept.append(s)
    return kept if kept else sites


# ----------------  Metal oxide slab  ----------------

def detect_oxide_surface_sites(
    atoms: Atoms,
    h_ontop: float = 1.0,
    h_bridge: float = 1.1,
    h_hollow: float = 1.2,
    max_sites_per_kind: int = 50,
    z_tol: float = 0.8,
) -> List[AdsSite]:
    """Detect ontop/bridge/fcc sites on a metal oxide slab using the top cation layer."""
    pos = atoms.positions
    symbols = atoms.get_chemical_symbols()
    if len(pos) == 0:
        raise ValueError("Empty Atoms object in detect_oxide_surface_sites.")

    # Extract cation (non-anion) atoms only
    metal_mask = np.array([s not in ANION_SYMBOLS for s in symbols])
    metal_idx = np.where(metal_mask)[0]
    if len(metal_idx) == 0:
        raise ValueError(
            "No cation-like atoms found (all atoms are in ANION_SYMBOLS). "
            "Cannot detect oxide surface sites."
        )

    z_metal = pos[metal_idx, 2]
    z_max = float(z_metal.max())
    top_metal_mask = (z_max - z_metal) < z_tol
    top_idx = metal_idx[top_metal_mask]
    if len(top_idx) == 0:
        raise ValueError("Could not identify top cation layer for oxide slab.")

    raw_sites = _build_sites_from_top_indices(
        pos,
        top_idx,
        h_ontop=h_ontop,
        h_bridge=h_bridge,
        h_hollow=h_hollow,
    )
    cell = atoms.get_cell()
    deduped = _dedupe_sites(
        raw_sites,
        cell,
        tol=0.20,
        max_sites_per_kind=max_sites_per_kind,
        edge_frac_cut=0.02,
    )
    return _filter_oxide_sites_terrace_like(
        deduped,
        pos,
        top_idx,
        np.asarray(cell, dtype=float),
        boundary_frac_cut=0.08,
        min_mean_nn=2.0,
    )


# ----------------  Representative site selection (optional)  ----------------

def select_representative_sites(
    sites: List[AdsSite],
    per_kind: int = 2,
) -> List[AdsSite]:
    """
    Select 1..per_kind representative sites per kind (ontop/bridge/hollow; legacy fcc/hcp aliases allowed)
    using greedy farthest-point sampling to maximize spatial diversity.
    """
    if not sites or per_kind <= 0:
        return []

    by_kind: Dict[str, List[AdsSite]] = {"ontop": [], "bridge": [], "hollow": [], "fcc": [], "hcp": []}
    for s in sites:
        by_kind.setdefault(s.kind, []).append(s)

    selected: List[AdsSite] = []

    for kind, group in by_kind.items():
        if not group:
            continue

        n_group = len(group)
        if n_group <= per_kind:
            # Fewer sites than requested → use all
            selected.extend(group)
            continue

        # Coordinate array
        xy = np.array([g.position[:2] for g in group])
        center = xy.mean(axis=0)

        # 1) Pick the point closest to center as first representative
        d_center = np.linalg.norm(xy - center, axis=1)
        first_idx = int(d_center.argmin())
        chosen_indices = [first_idx]

        # 2) Greedy farthest-point sampling for remaining representatives
        while len(chosen_indices) < per_kind:
            # Candidate indices not yet selected
            mask = np.ones(n_group, dtype=bool)
            mask[chosen_indices] = False
            cand_indices = np.where(mask)[0]
            if len(cand_indices) == 0:
                break

            # For each candidate, compute min distance to already-selected points
            min_dists = []
            for j in cand_indices:
                dists_to_chosen = np.linalg.norm(xy[j] - xy[chosen_indices], axis=1)
                min_dists.append(float(dists_to_chosen.min()))

            # Select the candidate with the largest min-distance
            next_idx = int(cand_indices[int(np.argmax(min_dists))])
            chosen_indices.append(next_idx)

        for idx in chosen_indices:
            selected.append(group[idx])

    return selected


# -----------------------------------------------------------------------------
# Slab + adsorbate construction utilities
# -----------------------------------------------------------------------------

def add_adsorbate_on_site(
    slab: Atoms,
    site: AdsSite,
    symbol: str = "H",
    dz: float = 0.0,
    mode: Literal["default", "oxide_o"] = "default",
) -> Atoms:
    """
    Place a single adsorbate atom on the slab at the given AdsSite position.

    Parameters
    ----------
    slab : Atoms
        Original slab structure (with PBC and cell).
    site : AdsSite
        Adsorption site obtained from detect_*_sites / select_representative_sites.
        site.position is assumed to be a Cartesian coordinate already offset
        above the surface (z_surf + h_*).
    symbol : str, default "H"
        Chemical symbol of the adsorbate atom to add.
    dz : float, default 0.0
        Additional z-offset above site.position (Angstrom).
    mode : {"default", "oxide_o"}, default "default"
        - "oxide_o" : Force O-top placement rule (only when symbol == "H").
        - "default" : Automatically apply O-top if the slab is oxide-like
                      and symbol == "H"; otherwise use site.position + dz.
    """
    a = slab.copy()

    base_symbols = a.get_chemical_symbols()
    base_count = base_symbols.count(symbol)

    # O-top logic is used only for HER (H*).
    # For other adsorbates (symbol != "H"), always use site.position + dz.
    if symbol == "H":
        use_oxide_o = (mode == "oxide_o") or (
            mode == "default" and _is_oxide_like(a)
        )
    else:
        use_oxide_o = False

    if use_oxide_o:
        x, y, z = _oxide_o_based_ads_position(a, site, h_oh=1.0, extra_z=dz)
        pos = (x, y, z)
    else:
        x, y, z = site.position
        pos = (float(x), float(y), float(z + dz))

    a.append(Atom(symbol, pos))

    new_count = a.get_chemical_symbols().count(symbol)
    if new_count != base_count + 1:
        raise RuntimeError(
            f"Failed to append adsorbate '{symbol}' on site "
            f"{site.kind} at position {site.position}"
        )

    return a


def generate_slab_ads_series(
    slab: Atoms,
    sites: List[AdsSite],
    symbol: str = "H",
    dz: float = 0.0,
    mode: Literal["default", "oxide_o"] = "default",
) -> List[Atoms]:
    """
    Generate slab+adsorbate structures for a list of AdsSite objects.

    Parameters
    ----------
    slab : Atoms
        Original slab structure.
    sites : list[AdsSite]
        Adsorption sites (e.g. from select_representative_sites).
    symbol : str, default "H"
        Chemical symbol of the adsorbate atom.
    dz : float, default 0.0
        Additional z-offset.
    mode : {"default", "oxide_o"}, default "default"
        - "default" : Auto-apply O-top only when slab is oxide-like and symbol == "H".
        - "oxide_o" : Force O-top placement (assumes symbol == "H").

    Returns
    -------
    list[Atoms]
        One slab+adsorbate structure per site.
    """
    if not sites:
        return []

    out: List[Atoms] = []
    base_symbols = slab.get_chemical_symbols()
    base_count = base_symbols.count(symbol)

    for s in sites:
        a = add_adsorbate_on_site(slab, s, symbol=symbol, dz=dz, mode=mode)
        # Safety check: verify adsorbate count
        new_count = a.get_chemical_symbols().count(symbol)
        if new_count != base_count + 1:
            raise RuntimeError(
                f"Generated slab for site kind={s.kind} has unexpected "
                f"'{symbol}' count (base={base_count}, new={new_count})."
            )
        out.append(a)

    return out

# =============================================================================
# NEW: Candidate site generation + unique top-anion snapping for oxide HER
# =============================================================================

def generate_candidate_sites(
    slab_atoms: Atoms,
    mtype: str,
    geom_per_kind: int = 2,
    n_random: int = 12,
    rng_seed: int = 0,
    random_kind: str = "hollow",
    reaction_mode: str | None = None,
) -> list[AdsSite]:
    """
    Generate a richer candidate site list for screening:
      (1) geometry-based seeds (representative sites) per kind
      (2) random (x,y) probes on the surface plane (kind = random_kind (for metal, legacy fcc/hcp are remapped to hollow))

    Parameters
    ----------
    slab_atoms : ase.Atoms
    mtype : {"metal","oxide"}
    geom_per_kind : int
        Number of representative seeds per site kind.
    n_random : int
        Number of random probe sites.
    rng_seed : int
        RNG seed.
    random_kind : str
        Kind label for random probe sites (e.g., "hollow", "bridge", "ontop"). Legacy metal labels fcc/hcp are remapped to hollow.

    Returns
    -------
    list[AdsSite]
        Combined candidate sites (deduplicated by XY binning).
    """
    if slab_atoms is None:
        return []

    # Geometry seeds
    rxn = str(reaction_mode or "").strip().upper()
    if mtype == "metal":
        geom_sites = detect_metal_111_sites(slab_atoms)
        rep = select_representative_sites(geom_sites, per_kind=int(max(1, geom_per_kind))) if geom_sites else []
        cand: list[AdsSite] = list(rep)
    elif rxn == "OER":
        # Oxide OER must not inherit metal-like ontop/bridge/fcc plumbing.
        # Use exposed cation AEM sites only and disable random hollow probes.
        geom_sites = detect_oxide_oer_cation_sites(slab_atoms, max_sites=int(max(1, geom_per_kind)))
        cand = list(geom_sites)
        n_random = 0
    else:
        geom_sites = detect_oxide_surface_sites(slab_atoms)
        rep = select_representative_sites(geom_sites, per_kind=int(max(1, geom_per_kind))) if geom_sites else []
        cand: list[AdsSite] = list(rep)

    # Random probes
    n_random = int(max(0, n_random))
    if n_random > 0:
        rng = np.random.default_rng(int(rng_seed))
        random_kind_eff = str(random_kind)
        if mtype == "metal" and random_kind_eff in {"fcc", "hcp"}:
            random_kind_eff = "hollow"
        cell = np.asarray(slab_atoms.get_cell(), dtype=float)
        a = cell[0, :2]
        b = cell[1, :2]

        # Surface reference z
        pos = slab_atoms.get_positions()
        z_top = float(pos[:, 2].max())
        z_probe = z_top + 0.01  # just above the top plane; actual dz handled later by placement

        for _ in range(n_random):
            u, v = rng.random(), rng.random()
            xy = u * a + v * b
            cand.append(
                AdsSite(
                    kind=str(random_kind_eff),
                    position=(float(xy[0]), float(xy[1]), float(z_probe)),
                    surface_indices=(),
                )
            )

    # Deduplicate in XY bins (conservative): keep earliest occurrence
    # (This matters because many seeds collapse to the same O if you later snap without uniqueness.)
    out: list[AdsSite] = []
    seen = set()
    for s in cand:
        x, y = float(s.position[0]), float(s.position[1])
        key = (s.kind, int(round(x / 0.25)), int(round(y / 0.25)))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)

    return out


def snap_sites_to_unique_top_anions(
    slab_atoms: Atoms,
    sites: list[AdsSite],
    z_tol: float = 0.8,
    prefer: str = "O",
    h_oh: float = 1.0,
    extra_z: float = 0.0,
    allow_reuse_when_exhausted: bool = False,
) -> list[AdsSite]:
    """
    For oxide HER, many different (fcc/bridge/ontop) seeds can snap to the same nearest top anion,
    resulting in identical adsorption initializations and (often) identical final relaxed sites.

    This function 'snaps' each candidate site to the nearest TOP anion (preferably O) with UNIQUE
    assignment (no reuse) whenever possible.

    If the number of sites exceeds available top anions:
      - allow_reuse_when_exhausted=False: the remaining sites keep their original (x,y,z)
      - allow_reuse_when_exhausted=True : reuse remaining anions (falls back to nearest)

    Returns a list of AdsSite in the same order as input.
    """
    if slab_atoms is None or not sites:
        return sites

    # Identify "top anion" indices
    pos = slab_atoms.get_positions()
    sym = slab_atoms.get_chemical_symbols()

    # Prefer explicit anion symbol (e.g., "O"). If not present, fallback to any anion-like symbol list.
    z_max = float(pos[:, 2].max())
    top_mask = (pos[:, 2] >= (z_max - float(z_tol)))

    prefer = str(prefer) if prefer else "O"
    top_idxs = [i for i, ok in enumerate(top_mask) if ok and sym[i] == prefer]

    # Fallback: if no preferred symbol in top layer, try any oxygen in slab (within z_tol),
    # else do nothing.
    if not top_idxs and prefer != "O":
        top_idxs = [i for i, ok in enumerate(top_mask) if ok and sym[i] == "O"]
    if not top_idxs:
        # No top anions found: return unchanged
        return sites

    # Build candidate (xy,z) list
    top_xy = np.asarray(pos[top_idxs, :2], dtype=float)
    top_z = np.asarray(pos[top_idxs, 2], dtype=float)

    used = set()
    snapped: list[AdsSite] = []

    for s in sites:
        xy = np.asarray([float(s.position[0]), float(s.position[1])], dtype=float)

        # Find nearest available anion
        d2 = np.sum((top_xy - xy[None, :]) ** 2, axis=1)
        order = np.argsort(d2)

        chosen_j = None
        for j in order:
            idx = int(top_idxs[int(j)])
            if idx not in used:
                chosen_j = int(j)
                used.add(idx)
                break

        if chosen_j is None:
            if allow_reuse_when_exhausted:
                chosen_j = int(order[0])
            else:
                # Keep original site position (do not force a duplicate snap)
                snapped.append(s)
                continue

        # Create snapped AdsSite at (anion_x, anion_y, anion_z + h_oh + extra_z)
        anion_idx = int(top_idxs[chosen_j])
        x_new, y_new = float(pos[anion_idx, 0]), float(pos[anion_idx, 1])
        z_new = float(pos[anion_idx, 2]) + float(h_oh) + float(extra_z)

        snapped.append(
            AdsSite(
                kind=s.kind,
                position=(x_new, y_new, z_new),
                surface_indices=s.surface_indices,
            )
        )

    return snapped


# =============================================================================
# OER oxide seed helpers
# =============================================================================

def _top_layer_cation_indices_z(atoms: Atoms, z_tol: float = 0.8) -> np.ndarray:
    """
    Return indices of cation-like atoms (not in ANION_SYMBOLS) belonging
    to the topmost layer by z-coordinate.
    """
    pos = atoms.positions
    if len(pos) == 0:
        return np.array([], dtype=int)

    symbols = atoms.get_chemical_symbols()
    cation_mask = np.array([s not in ANION_SYMBOLS for s in symbols])
    cat_idx = np.where(cation_mask)[0]
    if len(cat_idx) == 0:
        return np.array([], dtype=int)

    z = pos[cat_idx, 2]
    z_max = float(z.max())
    top_mask = (z_max - z) < z_tol
    return cat_idx[top_mask]




def _oxide_oer_open_direction_from_cation(
    atoms: Atoms,
    cation_index: int,
    *,
    neighbor_cutoff: float = 2.75,
    upward_bias: float = 1.20,
) -> np.ndarray:
    """Estimate a vacuum-facing open direction for cation-bound OER AEM placement.

    This is used only by the OER oxygen-intermediate branch.  It is not
    used by HER or oxide-HER O-anchor descriptors.
    """
    try:
        pos = np.asarray(atoms.get_positions(), dtype=float)
        sym = atoms.get_chemical_symbols()
        ci = int(cation_index)
        if ci < 0 or ci >= len(pos):
            return np.array([0.0, 0.0, 1.0], dtype=float)
        cpos = pos[ci]
        vec = np.array([0.0, 0.0, float(upward_bias)], dtype=float)
        for j, sj in enumerate(sym):
            if sj not in ANION_SYMBOLS:
                continue
            r = cpos - pos[j]
            d = float(np.linalg.norm(r))
            if not np.isfinite(d) or d < 1e-8 or d > float(neighbor_cutoff):
                continue
            z_weight = 1.0 + max(0.0, 1.2 - abs(float(pos[j, 2]) - float(cpos[2])))
            vec += (r / d) * (z_weight / max(d, 0.8))
        if vec[2] < 0.25:
            vec[2] = 0.25
        n = float(np.linalg.norm(vec))
        if not np.isfinite(n) or n < 1e-10:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return vec / n
    except Exception:
        return np.array([0.0, 0.0, 1.0], dtype=float)

def detect_oxide_oer_cation_sites(
    atoms: Atoms,
    *,
    z_window_from_top: float = 2.30,
    cation_z_tol: float = 1.20,
    max_sites: int = 24,
    o_coord_cutoff: float = 2.45,
) -> List[AdsSite]:
    """Detect exposed cation sites for oxide OER AEM screening.

    This detector is intentionally separate from the oxide-HER O-anchor
    detector.  HER on oxides often needs top lattice-O/protonation sites,
    whereas AEM OER needs cation-bound *OH/*O/*OOH on exposed metal
    cations (e.g. Ir_cus on rutile IrO2(110)).

    The returned AdsSite objects have kind="oer_cation" and surface_indices=(i,)
    where i is the selected cation index.  The z coordinate is only a seed
    height; downstream OER placement reuses the explicit cation index.
    """
    if atoms is None or len(atoms) == 0:
        return []

    pos = np.asarray(atoms.get_positions(), dtype=float)
    sym = atoms.get_chemical_symbols()
    z_top = float(np.max(pos[:, 2]))

    cation_idx = [i for i, el in enumerate(sym) if el not in ANION_SYMBOLS]
    if not cation_idx:
        return []

    # Candidate cations must be close to the top surface OR among the topmost
    # cation layer. This covers rutile (110), where bridging O can sit above
    # the coordinatively unsaturated metal cation.
    z_cat = np.asarray([pos[i, 2] for i in cation_idx], dtype=float)
    z_cat_max = float(np.max(z_cat))
    cand = []
    for i in cation_idx:
        zi = float(pos[i, 2])
        if (z_top - zi <= float(z_window_from_top)) or (z_cat_max - zi <= float(cation_z_tol)):
            cand.append(int(i))
    if not cand:
        cand = [int(i) for i in cation_idx if (z_cat_max - float(pos[i, 2])) <= float(cation_z_tol)]

    anion_idx = [i for i, el in enumerate(sym) if el in ANION_SYMBOLS]

    def _o_coord(i: int) -> int:
        if not anion_idx:
            return 0
        d = np.linalg.norm(pos[np.asarray(anion_idx, dtype=int)] - pos[int(i)][None, :], axis=1)
        return int(np.sum(d <= float(o_coord_cutoff)))

    def _oer_site_score(i: int) -> tuple:
        # Prefer exposed, low-coordination cations with an open O-anchor cone.
        # This ranking is used only for oxide OER AEM cation sites and does not
        # affect oxide-HER O-anchor detection.
        coord = int(_o_coord(i))
        depth = max(0.0, z_top - float(pos[int(i), 2]))
        direction = _oxide_oer_open_direction_from_cation(atoms, int(i))
        probe = pos[int(i)] + direction * 1.95
        anion_idx2 = [j for j, el in enumerate(sym) if el in ANION_SYMBOLS]
        if anion_idx2:
            d_probe = np.linalg.norm(pos[np.asarray(anion_idx2, dtype=int)] - probe[None, :], axis=1)
            min_oo = float(np.min(d_probe))
        else:
            min_oo = float('inf')
        # Convert to negative scores for ascending sort; low coord/depth retained.
        crowd_score = max(0.0, min(1.0, (min_oo - 1.45) / 0.60)) if np.isfinite(min_oo) else 1.0
        open_score = max(0.0, min(1.0, float(direction[2])))
        top_score = max(0.0, min(1.0, 1.0 - depth / max(float(z_window_from_top), 1e-6)))
        combined = 0.35 * crowd_score + 0.25 * open_score + 0.25 * top_score + 0.15 * max(0.0, min(1.0, (6.5 - coord) / 4.0))
        return (-float(combined), coord, depth, int(i))

    # Prefer more exposed / lower-crowding / lower-coordinated cations.
    # This is a heuristic for finding Ir_cus-like sites without changing HER.
    ranked = sorted(cand, key=_oer_site_score)

    sites: List[AdsSite] = []
    seen_xy = set()
    for i in ranked:
        x, y, z = [float(v) for v in pos[int(i)]]
        key = (round(x, 3), round(y, 3))
        if key in seen_xy:
            continue
        seen_xy.add(key)
        sites.append(
            AdsSite(
                kind="oer_cation",
                position=(x, y, z + 1.80),
                surface_indices=(int(i),),
            )
        )
        if len(sites) >= int(max_sites):
            break
    return sites


def oxide_oer_slab_suitability(
    atoms: Atoms,
    *,
    top_window: float = 1.80,
    cation_window_from_top: float = 2.30,
) -> Dict[str, object]:
    """Return non-invasive slab QC metadata for oxide OER AEM screening.

    This is a diagnostic helper only.  It does not change HER or slabify
    behavior.  It flags whether the top surface exposes cations suitable for
    cation-bound *OH/*O/*OOH placement.
    """
    out: Dict[str, object] = {
        "oer_slab_role": "OER_AEM_cation",
        "oer_top_cation_count": 0,
        "oer_top_anion_count": 0,
        "oer_top_cation_symbols": "",
        "oer_surface_O_fraction_top": float("nan"),
        "oer_candidate_cation_indices": "",
        "oer_slab_suitability": "unknown",
        "oer_slab_warning": "",
    }
    if atoms is None or len(atoms) == 0:
        out["oer_slab_suitability"] = "invalid"
        out["oer_slab_warning"] = "empty structure"
        return out
    pos = np.asarray(atoms.get_positions(), dtype=float)
    sym = atoms.get_chemical_symbols()
    z_top = float(np.max(pos[:, 2]))
    top_idx = [i for i in range(len(atoms)) if (z_top - float(pos[i, 2])) <= float(top_window)]
    top_cat = [i for i in top_idx if sym[i] not in ANION_SYMBOLS]
    top_an = [i for i in top_idx if sym[i] in ANION_SYMBOLS]
    top_o = [i for i in top_an if sym[i] == "O"]
    cand_sites = detect_oxide_oer_cation_sites(atoms, z_window_from_top=float(cation_window_from_top))
    cand_idx = [int(s.surface_indices[0]) for s in cand_sites if s.surface_indices]
    out["oer_top_cation_count"] = int(len(top_cat))
    out["oer_top_anion_count"] = int(len(top_an))
    out["oer_top_cation_symbols"] = ",".join(sorted(set(sym[i] for i in top_cat)))
    out["oer_surface_O_fraction_top"] = float(len(top_o) / max(len(top_idx), 1))
    out["oer_candidate_cation_indices"] = ",".join(str(i) for i in cand_idx)
    if len(cand_idx) == 0:
        out["oer_slab_suitability"] = "unsuitable"
        out["oer_slab_warning"] = "No exposed/top-near cation was found for cation-bound OER AEM placement."
    elif len(top_cat) == 0:
        out["oer_slab_suitability"] = "warning_cation_buried"
        out["oer_slab_warning"] = "Topmost layer is anion-rich; cations are below the top layer. Use only if this is expected for the target facet (e.g., rutile 110 with cus cations)."
    elif out["oer_surface_O_fraction_top"] >= 0.85:
        out["oer_slab_suitability"] = "warning_o_rich"
        out["oer_slab_warning"] = "Top surface is O-rich; cation-bound OER AEM placement may be unstable."
    else:
        out["oer_slab_suitability"] = "usable"
    return out


def expand_oxide_channels_for_adsorbate(adsorbate: str, policy: str = "default") -> tuple[str, ...]:
    """
    Determine initial seed-channel policy for oxygen intermediates on oxide
    surfaces.  This function is used by CO2RR/ORR/OER helper branches only;
    HER's H-on-oxide logic remains in add_adsorbate_on_site(..., symbol="H",
    mode="oxide_o") and is intentionally untouched.

    policy
    ------
    default
        oxygen-intermediate exploration.  O* and OOH*
        may also probe mixed bridge sites.
    oer_aem_cation
        Strict AEM benchmark mode.  *OH, *O and *OOH are seeded only on
        surface cation sites.  Lattice-O protonation and bridge_mixed states
        are excluded from the primary OER summary.
    """
    a = str(adsorbate).replace('*', '').upper()
    pol = str(policy or "default").strip().lower()

    if a == 'H':
        return ('anion_o',)  # legacy HER O-top behavior; do not alter

    if pol in {'oer_aem_cation', 'aem_cation', 'strict', 'cation_only'} and a in {'OH', 'O', 'OOH'}:
        return ('oer_cation',)

    if a == 'OH':
        return ('cation',)
    if a in ('O', 'OOH'):
        return ('cation', 'bridge_mixed')
    return ('cation',)


def oxide_surface_seed_position(
    slab: Atoms,
    site: AdsSite,
    adsorbate: str,
    channel: str = 'auto',
    z_tol: float = 0.8,
) -> tuple[float, float, float, str]:
    """
    Compute the initial anchor position for an OER intermediate on an
    oxide surface, dispatched by channel type.

    Returns
    -------
    (x, y, z, surface_channel)
        surface_channel examples:
          - adsorbate_on_cation
          - mixed_bridge
          - adsorbate_on_cation
          - mixed_bridge
          - lattice_protonation_disabled_for_aem
          - fallback_site
    """
    ads_clean = str(adsorbate).replace('*', '').upper()
    pos = slab.get_positions()
    sym = slab.get_chemical_symbols()
    if len(pos) == 0:
        x, y, z = site.position
        return float(x), float(y), float(z), 'fallback_site'

    if channel in (None, '', 'auto'):
        channel = expand_oxide_channels_for_adsorbate(ads_clean)[0]

    target_xy = np.asarray(site.position[:2], dtype=float)
    top_cat = list(_top_layer_cation_indices_z(slab, z_tol=z_tol))
    top_an = list(_top_layer_anion_indices_z(slab, z_tol=z_tol))
    top_o = [i for i in top_an if sym[i] == 'O']
    top_an_pref = top_o if top_o else top_an

    heights = {
        'O': 1.45,
        'OH': 1.80,
        'OOH': 2.05,
        'H': 1.00,
    }
    extra_bridge = {
        'O': 0.05,
        'OH': 0.10,
        'OOH': 0.15,
        'H': 0.00,
    }
    h = float(heights.get(ads_clean, 1.80))

    def _nearest(idx_list: list[int], xy_ref: np.ndarray) -> int | None:
        if not idx_list:
            return None
        xy = pos[np.asarray(idx_list, dtype=int), :2]
        d2 = np.sum((xy - xy_ref[None, :]) ** 2, axis=1)
        return int(idx_list[int(np.argmin(d2))])

    if channel == 'anion_o':
        idx = _nearest(top_an_pref, target_xy)
        if idx is not None:
            # H* keeps the legacy lattice-O protonation coordinate used by HER.
            # For OH*/O*/OOH* this channel is not a valid AEM adsorbate state;
            # it is retained only as an explicitly labelled, non-summary
            # diagnostic/fallback state if a caller requests it manually.
            if ads_clean == 'H':
                return float(pos[idx, 0]), float(pos[idx, 1]), float(pos[idx, 2] + h), 'anion_o'
            label = 'lattice_protonation_disabled_for_aem'
            return float(pos[idx, 0]), float(pos[idx, 1]), float(pos[idx, 2] + h), label

    if channel == 'bridge_mixed':
        ic = _nearest(top_cat, target_xy)
        if ic is not None:
            ia = _nearest(top_an_pref, pos[ic, :2])
            if ia is not None:
                xy_mid = 0.5 * (pos[ic, :2] + pos[ia, :2])
                z0 = max(float(pos[ic, 2]), float(pos[ia, 2])) + h + float(extra_bridge.get(ads_clean, 0.10))
                return float(xy_mid[0]), float(xy_mid[1]), float(z0), 'mixed_bridge'

    if channel in {'cation', 'oer_cation'}:
        # OER AEM cation placement should stay on the explicit cation site
        # when one is supplied by detect_oxide_oer_cation_sites().  This avoids
        # silently snapping back to a neighboring lattice-O/bridge-like site.
        idx = None
        try:
            sidx = tuple(int(i) for i in (getattr(site, 'surface_indices', ()) or ()))
            if sidx and sym[int(sidx[0])] not in ANION_SYMBOLS:
                idx = int(sidx[0])
        except Exception:
            idx = None
        if idx is None:
            idx = _nearest(top_cat, target_xy)
        if idx is not None:
            # For OER AEM, place the adsorbate along a local open direction
            # away from neighboring lattice oxygen atoms, not simply along
            # the global z axis.  This is restricted to OH/O/OOH; HER keeps
            # the legacy anion-O/H placement path above.
            if ads_clean in {'O', 'OH', 'OOH'}:
                direction = _oxide_oer_open_direction_from_cation(slab, int(idx))
                anchor = pos[int(idx)] + direction * h
                return float(anchor[0]), float(anchor[1]), float(anchor[2]), 'adsorbate_on_cation'
            return float(pos[idx, 0]), float(pos[idx, 1]), float(pos[idx, 2] + h), 'adsorbate_on_cation'

    # ordered fallback: cation -> anion -> original site
    idx = _nearest(top_cat, target_xy)
    if idx is not None:
        return float(pos[idx, 0]), float(pos[idx, 1]), float(pos[idx, 2] + h), 'adsorbate_on_cation'
    idx = _nearest(top_an_pref, target_xy)
    if idx is not None:
        if ads_clean == 'H':
            return float(pos[idx, 0]), float(pos[idx, 1]), float(pos[idx, 2] + h), 'anion_o'
        return float(pos[idx, 0]), float(pos[idx, 1]), float(pos[idx, 2] + h), 'lattice_protonation_disabled_for_aem'

    x, y, z = site.position
    return float(x), float(y), float(z), 'fallback_site'
