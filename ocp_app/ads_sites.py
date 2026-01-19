# ocp_app/core/ads_sites.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple, Dict

import numpy as np
from ase import Atoms, Atom

# ontop / bridge / fcc 세 가지로만 사용
AdsKind = Literal["ontop", "bridge", "fcc"]


@dataclass
class AdsSite:
    kind: AdsKind
    position: Tuple[float, float, float]
    surface_indices: Tuple[int, ...]  # 이 사이트를 이루는 표면 원자 인덱스 (1/2/3개)


# 산화물에서 음이온으로 간주할 원소들
ANION_SYMBOLS = {"O", "N", "F", "Cl", "S", "Br", "I"}


def _is_oxide_like(atoms: Atoms) -> bool:
    """
    ANION_SYMBOLS 에 속하는 원소와 그렇지 않은 원소가 동시에 존재하면
    'oxide-like' 로 간주 (예: NiO, CuO, LaNiO3 등).
    """
    symbols = atoms.get_chemical_symbols()
    has_anion = any(s in ANION_SYMBOLS for s in symbols)
    has_cation = any(s not in ANION_SYMBOLS for s in symbols)
    return has_anion and has_cation


def _top_layer_indices_z(atoms: Atoms, z_tol: float = 0.8) -> np.ndarray:
    """z 좌표 기준으로 가장 위쪽 layer에 속한 원자 인덱스."""
    pos = atoms.positions
    if len(pos) == 0:
        return np.array([], dtype=int)
    z = pos[:, 2]
    z_max = float(z.max())
    top_mask = (z_max - z) < z_tol
    return np.where(top_mask)[0]


def _top_layer_anion_indices_z(atoms: Atoms, z_tol: float = 0.8) -> np.ndarray:
    """
    ANION_SYMBOLS 에 속하는 원소들 중,
    z 좌표 기준으로 최상단 layer 에 속한 원자 인덱스만 반환.
    (oxide 계에서 O-top 배치용)
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
    oxide 계에서 H*를 O (또는 다른 anion) 위에 올리기 위한 좌표 생성.

    1) 최상단 anion layer 중에서 (우선 O)
    2) site.position 의 (x, y) 와 가장 가까운 anion 하나를 찾고
    3) 그 anion 위로 h_oh + extra_z 만큼 z 방향으로 올린다.

    적절한 anion 이 없으면, 기존 site.position 위에 extra_z 만큼 올리는 방식으로 fallback.
    """
    pos = slab.positions
    symbols = slab.get_chemical_symbols()

    top_anion_idx = _top_layer_anion_indices_z(slab, z_tol=0.8)

    # 1순위: O만 필터링, 없으면 전체 anion
    o_idx = [i for i in top_anion_idx if symbols[i] == "O"]
    candidates = o_idx if o_idx else list(top_anion_idx)

    # anion 이 전혀 없으면 → 이전 metal 로직과 동일하게 site.position + extra_z
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


def _build_sites_from_top_indices(
    pos: np.ndarray,
    top_idx: np.ndarray,
    h_ontop: float = 1.0,
    h_bridge: float = 1.1,
    h_hollow: float = 1.2,
) -> List[AdsSite]:
    """주어진 top-layer 인덱스들에 대해 ontop/bridge/fcc AdsSite 생성."""
    if len(top_idx) == 0:
        raise ValueError("Top-layer index set is empty in _build_sites_from_top_indices.")

    top_pos = pos[top_idx]
    xy = top_pos[:, :2]
    z_surf = float(top_pos[:, 2].mean())
    n_top = len(top_idx)

    # --- 1) pairwise 거리 → edge / triangle 후보 ---
    diff = xy[:, None, :] - xy[None, :, :]   # (n_top, n_top, 2)
    dmat = np.sqrt((diff ** 2).sum(axis=-1)) # (n_top, n_top)
    dmat += np.eye(n_top) * 1e6              # 자기 자신은 매우 큰 값으로

    d_min = float(dmat.min())
    if not np.isfinite(d_min) or d_min <= 0.0:
        raise ValueError("Failed to determine nearest-neighbor distance on top layer.")

    # 조금 여유 있게 edge cutoff 설정
    cut_edge = 1.6 * d_min

    edges = set()
    for i in range(n_top):
        for j in range(i + 1, n_top):
            if dmat[i, j] < cut_edge:
                edges.add((i, j))

    # 기본 삼각형 탐지: edge 3개가 모두 있는 경우
    triangles: List[Tuple[int, int, int]] = []
    for i in range(n_top):
        for j in range(i + 1, n_top):
            for k in range(j + 1, n_top):
                if (i, j) in edges and (i, k) in edges and (j, k) in edges:
                    triangles.append((i, j, k))

    # fallback: 위에서 하나도 안 잡히면, "길이가 비슷한 삼각형"으로 완화
    if not triangles:
        cut_tri = 1.8 * d_min
        for i in range(n_top):
            for j in range(i + 1, n_top):
                for k in range(j + 1, n_top):
                    dij = dmat[i, j]
                    dik = dmat[i, k]
                    djk = dmat[j, k]
                    dmax = max(dij, dik, djk)
                    dmin_loc = min(dij, dik, djk)
                    if dmax < cut_tri and dmax / max(dmin_loc, 1e-8) < 1.4:
                        triangles.append((i, j, k))

    # --- 2) 실제 AdsSite 리스트 생성 ---
    ads_sites: List[AdsSite] = []

    # 2-1) ontop: 각 top-layer 원자 위
    for idx in top_idx:
        x, y, z = pos[idx]
        ads_sites.append(
            AdsSite(
                kind="ontop",
                position=(float(x), float(y), float(z_surf + h_ontop)),
                surface_indices=(int(idx),),
            )
        )

    # 2-2) bridge: edge 중앙
    for i_local, j_local in edges:
        ia = top_idx[i_local]
        ib = top_idx[j_local]
        pa = pos[ia]
        pb = pos[ib]
        center = 0.5 * (pa + pb)
        x, y = center[:2]
        ads_sites.append(
            AdsSite(
                kind="bridge",
                position=(float(x), float(y), float(z_surf + h_bridge)),
                surface_indices=(int(ia), int(ib)),
            )
        )

    # 2-3) fcc: 삼각형 barycenter
    for i_local, j_local, k_local in triangles:
        ia = top_idx[i_local]
        ib = top_idx[j_local]
        ic = top_idx[k_local]
        p = (pos[ia] + pos[ib] + pos[ic]) / 3.0
        x, y = p[:2]
        ads_sites.append(
            AdsSite(
                kind="fcc",
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
    셀 정보를 이용해 거의 같은 위치인 AdsSite를 kind별로 dedupe.

    추가로, fractional x/y 기준으로 셀 경계(0 또는 1) 주변 edge_frac_cut
    안에 있는 사이트는 degenerate 로 간주하고 기본적으로 제거한다.
    (모두 제거되면 다시 원래 리스트를 사용)
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

    # 1) kind + (fx, fy) 기반 dedupe
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

    # 2) edge 영역 사이트 제거 (경계 근처는 PBC 상 중복으로 가정)
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
                # 셀 경계에 너무 가까운 사이트 → 스킵
                continue
            filtered.append(site)
        # 다 날아갔으면 원래 목록 유지
        if filtered:
            deduped = filtered

    # 3) kind별 최대 개수 제한
    out: List[AdsSite] = []
    count: Dict[str, int] = {"ontop": 0, "bridge": 0, "fcc": 0}
    for site in deduped:
        c = count.get(site.kind, 0)
        if c >= max_sites_per_kind:
            continue
        count[site.kind] = c + 1
        out.append(site)

    return out


# ----------------  금속 (111) 슬랩용  ----------------

def detect_metal_111_sites(
    atoms: Atoms,
    h_ontop: float = 1.0,
    h_bridge: float = 1.1,
    h_hollow: float = 1.2,
    max_sites_per_kind: int = 50,
) -> List[AdsSite]:
    """금속 (111)-like slab에서 ontop/bridge/fcc 사이트 자동 탐색."""
    top_idx = _top_layer_indices_z(atoms, z_tol=0.8)
    if len(top_idx) == 0:
        raise ValueError("Could not identify top surface layer for metal slab.")

    pos = atoms.positions
    raw_sites = _build_sites_from_top_indices(
        pos,
        top_idx,
        h_ontop=h_ontop,
        h_bridge=h_bridge,
        h_hollow=h_hollow,
    )
    cell = atoms.get_cell()
    return _dedupe_sites(
        raw_sites,
        cell,
        tol=0.20,
        max_sites_per_kind=max_sites_per_kind,
        edge_frac_cut=0.02,
    )


# ----------------  금속 산화물 슬랩용  ----------------

def detect_oxide_surface_sites(
    atoms: Atoms,
    h_ontop: float = 1.0,
    h_bridge: float = 1.1,
    h_hollow: float = 1.2,
    max_sites_per_kind: int = 50,
    z_tol: float = 0.8,
) -> List[AdsSite]:
    """일반 금속 산화물 slab에서 cation-top 레이어 기준 ontop/bridge/fcc 탐색."""
    pos = atoms.positions
    symbols = atoms.get_chemical_symbols()
    if len(pos) == 0:
        raise ValueError("Empty Atoms object in detect_oxide_surface_sites.")

    # 금속(cation)만 추출
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
    return _dedupe_sites(
        raw_sites,
        cell,
        tol=0.20,
        max_sites_per_kind=max_sites_per_kind,
        edge_frac_cut=0.02,
    )


# ----------------  대표 사이트 선택 (optional)  ----------------

def select_representative_sites(
    sites: List[AdsSite],
    per_kind: int = 2,
) -> List[AdsSite]:
    """
    ontop/bridge/fcc 별로 대표 사이트 1~per_kind 개 선택.

    per_kind >= 1 인 임의의 정수에 대해 동작하도록
    greedy farthest-point sampling 방식으로 구현.
    """
    if not sites or per_kind <= 0:
        return []

    by_kind: Dict[str, List[AdsSite]] = {"ontop": [], "bridge": [], "fcc": []}
    for s in sites:
        by_kind.setdefault(s.kind, []).append(s)

    selected: List[AdsSite] = []

    for kind, group in by_kind.items():
        if not group:
            continue

        n_group = len(group)
        if n_group <= per_kind:
            # 사이트 수가 충분히 적으면 전부 사용
            selected.extend(group)
            continue

        # 좌표 배열
        xy = np.array([g.position[:2] for g in group])
        center = xy.mean(axis=0)

        # 1) 중심에서 가장 가까운 점을 첫 대표로 선택
        d_center = np.linalg.norm(xy - center, axis=1)
        first_idx = int(d_center.argmin())
        chosen_indices = [first_idx]

        # 2) greedy farthest-point sampling으로 나머지 대표 선택
        while len(chosen_indices) < per_kind:
            # 아직 선택되지 않은 후보 인덱스
            mask = np.ones(n_group, dtype=bool)
            mask[chosen_indices] = False
            cand_indices = np.where(mask)[0]
            if len(cand_indices) == 0:
                break

            # 각 후보에 대해, 이미 선택된 점들까지의 거리 중 최소값을 계산
            min_dists = []
            for j in cand_indices:
                dists_to_chosen = np.linalg.norm(xy[j] - xy[chosen_indices], axis=1)
                min_dists.append(float(dists_to_chosen.min()))

            # 그 최소거리(min_dists)가 가장 큰 후보를 다음 대표로 선택
            next_idx = int(cand_indices[int(np.argmax(min_dists))])
            chosen_indices.append(next_idx)

        for idx in chosen_indices:
            selected.append(group[idx])

    return selected


# -----------------------------------------------------------------------------
# Slab + adsorbate 생성 유틸
# -----------------------------------------------------------------------------

def add_adsorbate_on_site(
    slab: Atoms,
    site: AdsSite,
    symbol: str = "H",
    dz: float = 0.0,
    mode: Literal["default", "oxide_o"] = "default",
) -> Atoms:
    """
    주어진 AdsSite.position 에 adsorbate(기본: H) 하나를 올린 slab+ads 구조를 만든다.

    Parameters
    ----------
    slab : Atoms
        원래 slab 구조 (PBC, cell 포함).
    site : AdsSite
        detect_*_sites / select_representative_sites 로 얻은 흡착 사이트.
        site.position 은 이미 표면 위쪽(z_surf + h_*) 로 설정된 Cartesian 좌표라고 가정.
    symbol : str, default "H"
        추가할 adsorbate 원자 기호.
    dz : float, default 0.0
        site.position 의 z 에 추가로 더 올릴 오프셋 (Å).
    mode : {"default", "oxide_o"}, default "default"
        - "oxide_o" : 무조건 O-top 규칙 적용 (단, symbol == "H"일 때만)
        - "default" : slab 이 oxide-like 이고 symbol == "H"일 때만 O-top 자동 적용,
                      그 외(symbol != "H")에는 기존 cation-top+dz 를 유지.
    """
    a = slab.copy()

    base_symbols = a.get_chemical_symbols()
    base_count = base_symbols.count(symbol)

    # [CHANGED]
    # O-top 로직은 "HER(H*)"용으로만 사용.
    # - symbol == "H" 인 경우에만 oxide_o 로직을 켠다.
    # - CO2RR 등 다른 adsorbate(symbol != "H")에는 항상 site.position+dz 그대로 사용.
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
    여러 AdsSite 리스트에 대해 slab+ads 구조들을 한 번에 생성.

    Parameters
    ----------
    slab : Atoms
        원래 slab 구조.
    sites : list[AdsSite]
        흡착 사이트 리스트 (예: representative sites).
    symbol : str, default "H"
        adsorbate 원자 기호.
    dz : float, default 0.0
        z 방향 추가 오프셋.
    mode : {"default", "oxide_o"}, default "default"
        - "default" : slab 이 oxide-like 이고 symbol == "H"일 때만 O-top 자동 적용
        - "oxide_o" : HER(H*)용으로 O-top 강제 적용 (symbol == "H" 이라는 전제)

    Returns
    -------
    list[Atoms]
        각 사이트에 대해 하나씩 생성된 slab+ads 구조 리스트.
    """
    if not sites:
        return []

    out: List[Atoms] = []
    base_symbols = slab.get_chemical_symbols()
    base_count = base_symbols.count(symbol)

    for s in sites:
        a = add_adsorbate_on_site(slab, s, symbol=symbol, dz=dz, mode=mode)
        # 안전장치: 각 구조의 symbol 개수 확인
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
    random_kind: str = "fcc",
) -> list[AdsSite]:
    """
    Generate a richer candidate site list for screening:
      (1) geometry-based seeds (representative sites) per kind
      (2) random (x,y) probes on the surface plane (kind = random_kind)

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
        Kind label for random probe sites (e.g., "fcc", "bridge", "ontop").

    Returns
    -------
    list[AdsSite]
        Combined candidate sites (deduplicated by XY binning).
    """
    if slab_atoms is None:
        return []

    # Geometry seeds
    if mtype == "metal":
        geom_sites = detect_metal_111_sites(slab_atoms)
    else:
        geom_sites = detect_oxide_surface_sites(slab_atoms)

    rep = select_representative_sites(geom_sites, per_kind=int(max(1, geom_per_kind))) if geom_sites else []
    cand: list[AdsSite] = list(rep)

    # Random probes
    n_random = int(max(0, n_random))
    if n_random > 0:
        rng = np.random.default_rng(int(rng_seed))
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
                    kind=str(random_kind),
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
