import inspect

import numpy as np

from ocp_app.core.ads_sites import AdsSite, _oxide_o_based_ads_position

def _oxide_o_based_ads_position_compat(atoms, site, dz: float = 1.0, extra_z: float = 0.0):
    """Call _oxide_o_based_ads_position with best-effort kwarg compatibility.

    Different ocp_app versions have used different parameter names:
      - dz_h_oh, dz_h, or dz
    This helper inspects the callable signature and only passes supported kwargs.

    Returns
    -------
    (x, y, z): tuple[float, float, float]
    """
    try:
        params = inspect.signature(_oxide_o_based_ads_position).parameters
    except Exception:
        params = {}

    kwargs = {}
    if "dz_h_oh" in params:
        kwargs["dz_h_oh"] = float(dz)
    elif "dz_h" in params:
        kwargs["dz_h"] = float(dz)
    elif "dz" in params:
        kwargs["dz"] = float(dz)

    if "extra_z" in params:
        kwargs["extra_z"] = float(extra_z)

    out = _oxide_o_based_ads_position(atoms, site, **kwargs) if kwargs else _oxide_o_based_ads_position(atoms, site)
    # Be tolerant to older return formats
    try:
        x, y, z = out
    except Exception:
        raise RuntimeError(f"_oxide_o_based_ads_position returned unexpected value: {out!r}")
    return float(x), float(y), float(z)

def _pbc_min_image_xy_distance_sq(atoms, xy_ref, xy_target):
    """2D minimum-image distance squared in the slab plane."""
    try:
        cell = np.array(atoms.get_cell(), dtype=float)
        A = np.column_stack([cell[0, :2], cell[1, :2]])
        invA = np.linalg.inv(A)
        f_ref = invA @ np.asarray(xy_ref, dtype=float)
        f_tgt = invA @ np.asarray(xy_target, dtype=float)
        df = f_tgt - f_ref
        df -= np.round(df)
        dxy = A @ df
        return float(np.dot(dxy, dxy))
    except Exception:
        d = np.asarray(xy_target, dtype=float) - np.asarray(xy_ref, dtype=float)
        return float(np.dot(d, d))

def _top_surface_o_indices(atoms, z_window: float = 2.2, min_expand_to: float = 4.0):
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)
    pos = atoms.get_positions()
    z = pos[:, 2]
    zmax = float(np.max(z))
    o_idx = np.where(syms == 'O')[0]
    if len(o_idx) == 0:
        return []
    for win in (float(z_window), 2.8, 3.4, float(min_expand_to)):
        idx = [int(i) for i in o_idx if z[i] >= (zmax - float(win))]
        if idx:
            return idx
    # fallback: highest oxygen only
    i_best = int(o_idx[np.argmax(z[o_idx])])
    return [i_best]

def _generate_oxide_her_oanchor_sites(atoms, max_sites: int = 6, z_window: float = 2.2, min_xy_sep: float = 1.5):
    """Generate oxide-HER site seeds directly from top-surface oxygen anchors.

    Returns AdsSite objects whose positions sit on top-surface O atoms. The actual H height
    is added later by the common O-top projection helper / run path.
    """
    pos = atoms.get_positions()
    top_o = _top_surface_o_indices(atoms, z_window=float(z_window))
    if not top_o:
        return []

    chosen = []
    for idx in sorted(top_o, key=lambda i: float(pos[i, 2]), reverse=True):
        xy_i = pos[idx, :2]
        too_close = False
        for j in chosen:
            if _pbc_min_image_xy_distance_sq(atoms, xy_i, pos[j, :2]) < float(min_xy_sep) ** 2:
                too_close = True
                break
        if too_close:
            continue
        chosen.append(int(idx))
        if len(chosen) >= int(max_sites):
            break

    sites = []
    for k, idx in enumerate(chosen):
        sites.append(AdsSite(kind='o_top', position=tuple(float(x) for x in pos[idx]), surface_indices=(int(idx),)))
    return sites

def _project_single_oxide_her_site_to_otop(atoms, site, dz: float = 1.0, extra_z: float = 0.0):
    """Project one oxide HER seed to a top-surface oxygen anchor.

    Uses a top-layer O-only search in the slab plane, avoiding buried / bottom-layer oxygen selection.
    """
    pos = atoms.get_positions()
    top_o = _top_surface_o_indices(atoms, z_window=2.2)
    if not top_o:
        # Fallback to legacy helper if no top oxygen could be identified
        new_x, new_y, new_z = _oxide_o_based_ads_position_compat(atoms, site, dz=float(dz), extra_z=float(extra_z))
        return AdsSite(kind='o_top', position=(float(new_x), float(new_y), float(new_z)), surface_indices=getattr(site, 'surface_indices', None))

    # If the incoming site already references a top O, keep that anchor.
    surf_idx = list(getattr(site, 'surface_indices', []) or [])
    anchor_idx = None
    for i in surf_idx:
        try:
            ii = int(i)
        except Exception:
            continue
        if ii in top_o and atoms.get_chemical_symbols()[ii] == 'O':
            anchor_idx = ii
            break

    # Otherwise, choose the nearest top-surface O in the slab plane.
    if anchor_idx is None:
        xy_ref = np.asarray(getattr(site, 'position', (0.0, 0.0, 0.0))[:2], dtype=float)
        anchor_idx = min(top_o, key=lambda i: _pbc_min_image_xy_distance_sq(atoms, xy_ref, pos[int(i), :2]))

    x, y, z = [float(v) for v in pos[int(anchor_idx)]]
    return AdsSite(kind='o_top', position=(x, y, z + float(dz) + float(extra_z)), surface_indices=(int(anchor_idx),))

def _project_oxide_her_sites_to_otop(atoms, sites, dz: float = 1.0, extra_z: float = 0.0):
    if not sites:
        return sites
    if isinstance(sites, dict):
        return {label: _project_single_oxide_her_site_to_otop(atoms, site, dz=dz, extra_z=extra_z) for label, site in sites.items()}
    if isinstance(sites, tuple):
        return tuple(_project_single_oxide_her_site_to_otop(atoms, site, dz=dz, extra_z=extra_z) for site in sites)
    if isinstance(sites, list):
        return [_project_single_oxide_her_site_to_otop(atoms, site, dz=dz, extra_z=extra_z) for site in sites]
    return sites

