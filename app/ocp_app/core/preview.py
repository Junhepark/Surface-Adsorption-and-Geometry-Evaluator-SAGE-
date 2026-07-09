from pathlib import Path
from io import BytesIO, StringIO
import zipfile

from ase.io import read

from ocp_app.core.structure_ops import atoms_to_cif_bytes
from ocp_app.core.ads_sites import oxide_surface_seed_position, expand_oxide_channels_for_adsorbate, ANION_SYMBOLS

ADS_TEMPLATE_FILES = {
    "CO":   "CO_box.cif",
    "COOH": "COOH_box.cif",
    "HCOO": "HCOO_box.cif",
    "OCHO": "OCHO_box.cif",
    # ORR/OER intermediates
    "O":    "O_box.cif",
    "OH":   "OH_box.cif",
    "OOH":  "OOH_box.cif",
    # SAGE-VOC acetaldehyde preset templates
    "CH3CHO":   "CH3CHO_box.cif",
    "CH3CO":    "CH3CO_box.cif",
    "CH3CH2O":  "CH3CH2O_box.cif",
    "CH3COO":   "CH3COO_box.cif",
    "CH3CH2OH": "CH3CH2OH_box.cif",
    "CH3COOH":  "CH3COOH_box.cif",
}


def _candidate_ref_gas_dirs(ref_dir: str | Path = "ref_gas") -> list[Path]:
    """Return candidate ref_gas directories without accepting directory-only hits.

    Important: some deployments have an old cwd/ref_gas that contains H2/CO/OH
    but not the newly added VOC CIF files.  Therefore callers must resolve the
    *requested template file*, not merely the first directory named ref_gas.
    """
    ref_path = Path(ref_dir)
    here = Path(__file__).resolve()
    raw = []

    if ref_path.is_absolute():
        raw.append(ref_path)
    else:
        raw.extend([
            Path.cwd() / ref_path,
            Path('/app') / ref_path,
            Path('/app/ref_gas'),
            here.parent / ref_path,
            here.parents[1] / ref_path,
            here.parents[2] / ref_path,
            here.parents[3] / ref_path,
            here.parents[4] / ref_path,
            ref_path,
        ])

    out: list[Path] = []
    seen: set[str] = set()
    for cand in raw:
        try:
            c = cand.resolve()
        except Exception:
            c = cand
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def resolve_ref_gas_template(template_name: str, ref_dir: str | Path = "ref_gas") -> Path:
    """Resolve an adsorbate template by file existence, matching CHE_mode semantics but safer.

    CHE_mode historically uses Path(ref_dir)/template.  VOC templates may be
    installed in /app/ref_gas while the process cwd can be /home/junhee/projects.
    This function checks every plausible ref_gas directory and returns the first
    candidate that actually contains the requested template.
    """
    checked: list[str] = []
    t = str(template_name)

    # If ref_dir itself is a direct file path, allow it.
    ref_path = Path(ref_dir)
    if ref_path.is_file() and ref_path.name == t:
        return ref_path.resolve()

    for d in _candidate_ref_gas_dirs(ref_dir):
        p = d / t
        checked.append(str(p))
        try:
            if p.is_file():
                return p.resolve()
        except Exception:
            pass

    raise FileNotFoundError(
        "Adsorbate template not found. Checked:\n  - " + "\n  - ".join(checked)
    )


def load_ads_template_preview(ads: str, ref_dir: str | Path = "ref_gas"):
    ads_clean = ads.replace("*", "").upper()
    if ads_clean not in ADS_TEMPLATE_FILES:
        raise ValueError(f"Unsupported adsorbate '{ads}' for preview")

    cif_path = resolve_ref_gas_template(ADS_TEMPLATE_FILES[ads_clean], ref_dir=ref_dir)

    # Return the raw template geometry.  Orientation and anchor normalization
    # are handled later by _orient_preview_adsorbate_upright(...) and
    # _preview_anchor_indices_and_position(...).  Do not force z=abs(z) here:
    # that distorts directional VOC templates such as CH3CHO.
    return read(cif_path).copy()




def _mic_xy_norm_preview(slab, xy0, xy1) -> float:
    import numpy as np
    from ase.geometry import find_mic
    v = np.asarray([float(xy1[0]) - float(xy0[0]), float(xy1[1]) - float(xy0[1]), 0.0], dtype=float)
    try:
        vec, _ = find_mic(v, slab.get_cell(), slab.get_pbc())
        return float(np.linalg.norm(vec[:2]))
    except Exception:
        return float(np.linalg.norm(v[:2]))


def _local_surface_z_for_preview(slab, site, radius: float = 2.8, top_z_window: float = 4.0) -> float:
    import numpy as np
    pos = np.asarray(slab.get_positions(), dtype=float)
    if pos.size == 0:
        return 0.0
    site_xy = np.asarray(site.position[:2], dtype=float)
    zmax = float(np.max(pos[:, 2]))
    near = []
    for p in pos:
        if (zmax - float(p[2])) > float(top_z_window):
            continue
        if _mic_xy_norm_preview(slab, site_xy, p[:2]) <= float(radius):
            near.append(float(p[2]))
    if near:
        return float(max(near))
    try:
        idx = tuple(int(i) for i in getattr(site, "surface_indices", ()) or ())
        if idx:
            zs = [float(pos[i, 2]) for i in idx if 0 <= i < len(pos)]
            if zs:
                return float(max(zs))
    except Exception:
        pass
    return float(zmax)


def _offset_xy_for_preview(slab, site, offset_A: float = 1.8):
    import numpy as np
    xy = np.asarray(site.position[:2], dtype=float)
    try:
        a = np.asarray(slab.get_cell()[0, :2], dtype=float)
        n = float(np.linalg.norm(a))
        if n > 1e-8:
            return xy + (a / n) * float(offset_A)
    except Exception:
        pass
    return xy + np.asarray([float(offset_A), 0.0], dtype=float)



def _select_surface_oxygen_for_reduction_h_preview(
    slab,
    site,
    *,
    prefer_adjacent: bool = False,
    z_window: float = 3.0,
    min_adjacent_xy: float = 1.05,
    max_adjacent_xy: float = 4.25,
):
    """Preview analogue of reduction H_on_exposed_surface_O placement."""
    import numpy as np
    try:
        pos = np.asarray(slab.get_positions(), dtype=float)
        if pos.size == 0:
            return None
        syms = slab.get_chemical_symbols()
        site_xy = np.asarray(site.position[:2], dtype=float)
        zmax = float(np.max(pos[:, 2]))
        raw = []  # (dxy, dz_from_top, index)
        for i, (sym, p) in enumerate(zip(syms, pos)):
            if str(sym) not in ANION_SYMBOLS:
                continue
            dz_from_top = zmax - float(p[2])
            if dz_from_top > float(z_window):
                continue
            dxy = _mic_xy_norm_preview(slab, site_xy, p[:2])
            raw.append((float(dxy), float(dz_from_top), int(i)))
        if not raw:
            return None
        min_o_dz = min(c[1] for c in raw)
        top_o = [c for c in raw if c[1] <= min_o_dz + 0.75] or raw
        if prefer_adjacent:
            adj = [c for c in top_o if float(min_adjacent_xy) <= c[0] <= float(max_adjacent_xy)]
            if adj:
                adj.sort(key=lambda c: (c[1], abs(c[0] - 1.80), c[0]))
                return int(adj[0][2])
        top_o.sort(key=lambda c: (c[1], c[0]))
        return int(top_o[0][2])
    except Exception:
        return None


def _surface_h_position_for_preview(slab, site, *, prefer_adjacent: bool = False, oh_length: float = 0.98):
    """Return H coordinate for reduction-route surface-OH preview."""
    import numpy as np
    oi = _select_surface_oxygen_for_reduction_h_preview(slab, site, prefer_adjacent=prefer_adjacent)
    if oi is None:
        return _preview_target_xyz(slab, site, "H", dz=0.60)
    pos = np.asarray(slab.get_positions(), dtype=float)
    p = pos[int(oi)]
    return np.asarray([float(p[0]), float(p[1]), float(p[2]) + float(oh_length)], dtype=float)




def _rotation_matrix_axis_angle_preview(axis, angle_rad: float):
    import numpy as np
    try:
        a = np.asarray(axis, dtype=float)
        n = float(np.linalg.norm(a))
        if n < 1e-12:
            return np.eye(3)
        a = a / n
        x, y, z = a
        c = float(np.cos(float(angle_rad)))
        s = float(np.sin(float(angle_rad)))
        C = 1.0 - c
        return np.asarray([
            [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
            [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
        ], dtype=float)
    except Exception:
        return np.eye(3)


def _place_preview_reduction_ch3cho(slab, site, dz: float, ref_dir: str | Path):
    """Preview CH3CHO* as a reduction-only tilted carbonyl-C precursor."""
    import numpy as np
    ads_atoms = load_ads_template_preview("CH3CHO", ref_dir=ref_dir)
    syms = ads_atoms.get_chemical_symbols()
    pos = np.asarray(ads_atoms.get_positions(), dtype=float)
    c_idx = [i for i, x in enumerate(syms) if str(x).upper() == "C"]
    o_idx = [i for i, x in enumerate(syms) if str(x).upper() == "O"]
    if not c_idx or not o_idx:
        return _place_preview_component(slab, site, "CH3CHO", dz=dz, ref_dir=ref_dir)
    best = None
    for c in c_idx:
        for o in o_idx:
            dco = float(np.linalg.norm(pos[int(o)] - pos[int(c)]))
            if best is None or dco < best[0]:
                best = (dco, int(c), int(o))
    c = int(best[1]); o = int(best[2])
    anchor = pos[c].copy()
    v = pos[o] - pos[c]
    R1 = _rotation_matrix_from_vectors_preview(v, np.asarray([0.0, 0.0, 1.0], dtype=float))
    p1 = (pos - anchor) @ R1.T + anchor
    R2 = _rotation_matrix_axis_angle_preview([0.0, 1.0, 0.0], np.deg2rad(18.0))
    p2 = (p1 - anchor) @ R2.T + anchor
    ads_atoms.set_positions(p2)

    target = _preview_target_xyz(slab, site, "CH3CHO", dz=1.35)
    anchor_pos = np.asarray(ads_atoms.get_positions()[c], dtype=float)
    ads_atoms.set_cell(slab.get_cell())
    ads_atoms.set_pbc(slab.get_pbc())
    ads_atoms.translate(target - anchor_pos)
    if ads_clean == "CH3CH2OH":
        ads_atoms = _lift_preview_adsorbate_clear_of_slab(
            slab,
            ads_atoms,
            min_dist_A=1.35,
            step_A=0.10,
            max_lift_A=2.50,
        )
    return ads_atoms

def _preview_anchor_indices_and_position(mol, ads: str):
    import numpy as np
    ads_clean = str(ads or "").replace("*", "").upper()
    syms = mol.get_chemical_symbols()
    pos = np.asarray(mol.get_positions(), dtype=float)

    def idxs(sym):
        return [i for i, s in enumerate(syms) if s == sym]

    c_idx = idxs("C")
    o_idx = idxs("O")

    def nearest_c_to_o():
        if not c_idx:
            return None
        if not o_idx:
            return int(c_idx[0])
        d = np.linalg.norm(pos[c_idx][:, None, :] - pos[o_idx][None, :, :], axis=2)
        return int(c_idx[int(np.argmin(np.min(d, axis=1)))])

    def nearest_o_to_c(c=None):
        if not o_idx:
            return None
        if c is None:
            c = nearest_c_to_o()
        if c is None:
            return int(o_idx[0])
        d = np.linalg.norm(pos[o_idx] - pos[int(c)][None, :], axis=1)
        return int(o_idx[int(np.argmin(d))])

    if ads_clean == "H":
        return np.asarray([0.0, 0.0, 0.0], dtype=float), (0,), "H_atom"

    if ads_clean == "CH3CHO":
        oi = nearest_o_to_c()
        if oi is not None:
            return pos[oi].copy(), (int(oi),), "carbonyl_o"

    if ads_clean == "CH3COO" and len(o_idx) >= 2:
        chosen = tuple(int(i) for i in o_idx[:2])
        return pos[list(chosen)].mean(axis=0), chosen, "o_o_midpoint"

    if ads_clean in {"OH", "O", "CH3CH2O", "CH3CH2OH"} and o_idx:
        return pos[o_idx[0]].copy(), (int(o_idx[0]),), "o_atom"

    if ads_clean in {"CO", "COOH", "CH3CO", "CH3COOH"} and c_idx:
        ci = nearest_c_to_o()
        if ci is not None:
            return pos[ci].copy(), (int(ci),), "c_atom"

    for sym in ("C", "O"):
        ids = idxs(sym)
        if ids:
            return pos[ids[0]].copy(), (int(ids[0]),), f"fallback_{sym}"
    return pos[0].copy(), (0,), "atom0"



def _rotation_matrix_from_vectors_preview(v_from, v_to):
    """Return a 3x3 rotation matrix that rotates v_from onto v_to."""
    import numpy as np
    a = np.asarray(v_from, dtype=float)
    b = np.asarray(v_to, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3)
    a = a / na
    b = b / nb
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.eye(3)
    if c < -1.0 + 1e-10:
        # 180-degree rotation around any axis perpendicular to a.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]], dtype=float)
        return np.eye(3) + 2.0 * (K @ K)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    K = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]], dtype=float)
    return np.eye(3) + K + (K @ K) * ((1.0 - c) / max(s * s, 1e-12))


def _preview_direction_position_for_upright(mol, ads: str, anchor_indices):
    """Choose a non-anchor point that should point toward +surface normal."""
    import numpy as np
    ads_clean = str(ads or "").replace("*", "").upper()
    syms = mol.get_chemical_symbols()
    pos = np.asarray(mol.get_positions(), dtype=float)
    anchors = tuple(int(i) for i in (anchor_indices or ()))
    anchor_center = pos[list(anchors)].mean(axis=0) if anchors else pos[0].copy()

    c_idx = [i for i, s in enumerate(syms) if s == "C"]
    o_idx = [i for i, s in enumerate(syms) if s == "O"]
    h_idx = [i for i, s in enumerate(syms) if s == "H"]
    non_anchor = [i for i in range(len(pos)) if i not in set(anchors)]

    def nearest(indices):
        if not indices:
            return None
        d = np.linalg.norm(pos[indices] - anchor_center[None, :], axis=1)
        return int(indices[int(np.argmin(d))])

    if ads_clean == "CH3CHO":
        # O-down aldehyde seed: carbonyl C should point away from the surface.
        ci = nearest(c_idx)
        if ci is not None:
            return pos[ci].copy()

    if ads_clean in {"CO", "CH3CO", "COOH", "CH3COOH"}:
        # C-down seeds: O-containing moiety should remain above the C anchor.
        oi = nearest(o_idx)
        if oi is not None:
            return pos[oi].copy()

    if ads_clean in {"CH3COO"}:
        # Bidentate carboxylate proxy: carboxyl carbon / methyl side points up.
        ci = nearest(c_idx)
        if ci is not None:
            return pos[ci].copy()

    if ads_clean == "CH3CH2OH":
        # Product ethanol should not be seeded like ethoxy with O--C forced
        # straight away from the slab.  Align the O-centered O--C/O--H
        # bisector outward so the hydroxyl H also starts on the vacuum side.
        ci = nearest(c_idx)
        hi = nearest(h_idx)
        if ci is not None and hi is not None:
            vc = pos[int(ci)].copy() - anchor_center
            vh = pos[int(hi)].copy() - anchor_center
            nc = float(np.linalg.norm(vc))
            nh = float(np.linalg.norm(vh))
            if nc > 1e-8 and nh > 1e-8:
                v = vc / nc + vh / nh
                nv = float(np.linalg.norm(v))
                if nv > 1e-8:
                    return anchor_center + v / nv
        if ci is not None:
            return pos[ci].copy()

    if ads_clean == "CH3CH2O":
        ci = nearest(c_idx)
        if ci is not None:
            return pos[ci].copy()

    if non_anchor:
        return pos[non_anchor].mean(axis=0)
    return None


def _orient_preview_adsorbate_upright(mol, ads: str, anchor_indices, normal=(0.0, 0.0, 1.0)):
    """Rotate VOC template so the molecular body points away from the slab.

    This keeps legacy ontop/bridge/fcc site selection intact while preventing
    directional VOCs such as CH3CHO from being translated into the surface cage.
    The anchor atom/group remains the pivot.
    """
    import numpy as np
    anchors = tuple(int(i) for i in (anchor_indices or ()))
    if not anchors or len(mol) <= 1:
        return mol

    pos = np.asarray(mol.get_positions(), dtype=float)
    anchor_center = pos[list(anchors)].mean(axis=0)
    direction_pos = _preview_direction_position_for_upright(mol, ads, anchors)
    if direction_pos is None:
        return mol

    v = np.asarray(direction_pos, dtype=float) - anchor_center
    n = np.asarray(normal, dtype=float)
    if np.linalg.norm(v) < 1e-8 or np.linalg.norm(n) < 1e-8:
        return mol

    R = _rotation_matrix_from_vectors_preview(v, n)
    new_pos = (pos - anchor_center) @ R.T + anchor_center

    # If the overall molecular body is still below the anchor plane, flip the
    # direction vector once.  This is a rigid 180-degree correction, not a
    # per-atom abs(z) distortion.
    rel = new_pos - anchor_center
    body = [i for i in range(len(new_pos)) if i not in set(anchors)]
    if body:
        mean_proj = float(np.mean(rel[body] @ (n / max(np.linalg.norm(n), 1e-12))))
        if mean_proj < 0.0:
            R2 = _rotation_matrix_from_vectors_preview(-n, n)
            new_pos = (new_pos - anchor_center) @ R2.T + anchor_center

    mol.set_positions(new_pos)
    return mol


def _site_support_z_for_preview(slab, site, fallback_z=None):
    import numpy as np
    try:
        pos = np.asarray(slab.get_positions(), dtype=float)
        idx = tuple(int(i) for i in getattr(site, "surface_indices", ()) or ())
        if idx:
            zs = [float(pos[i, 2]) for i in idx if 0 <= i < len(pos)]
            if zs:
                return float(max(zs))
    except Exception:
        pass
    if fallback_z is not None:
        try:
            fz = float(fallback_z)
            if np.isfinite(fz):
                return fz
        except Exception:
            pass
    return _local_surface_z_for_preview(slab, site)


def _preview_height_for_ads(ads: str, default: float = 1.25) -> float:
    """Species-specific initial anchor height above the local support atoms.

    Kept aligned with voc_mode._voc_initial_anchor_height so preview and
    calculation seeds are comparable.
    """
    ads_clean = str(ads or "").replace("*", "").upper()
    if ads_clean == "H":
        return 1.00
    if ads_clean == "OH":
        return 1.15
    if ads_clean in {"CH3COO", "CH3COOH", "COOH"}:
        return 1.25
    if ads_clean in {"O", "CO", "CH3CO"}:
        return 1.15
    if ads_clean == "CH3CH2OH":
        # Product ethanol is seeded as a weak retention/desorption proxy, not as
        # an ethoxy-like two-point chemisorbed adsorbate.
        return 1.80
    if ads_clean in {"CH3CHO", "CH3CH2O"}:
        return 1.10
    try:
        return float(default)
    except Exception:
        return 1.10


def _preview_target_xyz(slab, site, ads: str, dz: float = 1.25, xy_override=None):
    import numpy as np
    xy = np.asarray(xy_override if xy_override is not None else site.position[:2], dtype=float)
    z = _site_support_z_for_preview(slab, site) + float(_preview_height_for_ads(ads, default=dz))
    return np.asarray([float(xy[0]), float(xy[1]), float(z)], dtype=float)



def _nearest_preview_slab_dist(slab, ads_atoms):
    import numpy as np
    try:
        sp = np.asarray(slab.get_positions(), dtype=float)
        ap = np.asarray(ads_atoms.get_positions(), dtype=float)
        if sp.size == 0 or ap.size == 0:
            return float("nan")
        return float(np.min(np.linalg.norm(ap[:, None, :] - sp[None, :, :], axis=2)))
    except Exception:
        return float("nan")


def _lift_preview_adsorbate_clear_of_slab(slab, ads_atoms, *, min_dist_A=1.35, step_A=0.10, max_lift_A=2.50):
    import numpy as np
    out = ads_atoms.copy()
    total = 0.0
    dmin = _nearest_preview_slab_dist(slab, out)
    while np.isfinite(dmin) and dmin < float(min_dist_A) and total + float(step_A) <= float(max_lift_A) + 1e-12:
        out.translate([0.0, 0.0, float(step_A)])
        total += float(step_A)
        dmin = _nearest_preview_slab_dist(slab, out)
    return out


def _place_preview_component(slab, site, ads: str, dz: float, ref_dir: str | Path, xy_override=None):
    ads_clean = ads.replace("*", "").upper()
    ads_atoms = load_ads_template_preview(ads, ref_dir=ref_dir)

    target = _preview_target_xyz(slab, site, ads_clean, dz=dz, xy_override=xy_override)
    _anchor_pos0, anchor_idx, _anchor_mode = _preview_anchor_indices_and_position(ads_atoms, ads_clean)
    ads_atoms = _orient_preview_adsorbate_upright(ads_atoms, ads_clean, anchor_idx, normal=(0.0, 0.0, 1.0))
    anchor_pos, _anchor_idx, _anchor_mode = _preview_anchor_indices_and_position(ads_atoms, ads_clean)

    ads_atoms.set_cell(slab.get_cell())
    ads_atoms.set_pbc(slab.get_pbc())
    ads_atoms.translate(target - anchor_pos)
    if ads_clean == "CH3CH2OH":
        ads_atoms = _lift_preview_adsorbate_clear_of_slab(
            slab,
            ads_atoms,
            min_dist_A=1.35,
            step_A=0.10,
            max_lift_A=2.50,
        )
    return ads_atoms


def build_adsorbate_preview_slab(slab_atoms, site, ads: str, dz: float = 1.8, ref_dir: str | Path = "ref_gas"):
    slab = slab_atoms.copy()
    ads_s = str(ads or "")

    # SAGE-VOC co-adsorption preview, e.g. H*+CH3CHO* or OH*+CH3CHO*.
    if "+" in ads_s:
        parts = [p.strip() for p in ads_s.split("+") if p.strip()]
        # Put the VOC/intermediate on the selected site first; H*/OH* are
        # nearby co-adsorbates.  This prevents H*+CH3CHO* from using H as the
        # primary site occupant.
        def _clean(x):
            return x.replace("*", "").upper()
        primary_i = 0
        for ii, part in enumerate(parts):
            if _clean(part) not in {"H", "OH", "O"}:
                primary_i = ii
                break
        ordered = [parts[primary_i]] + [p for ii, p in enumerate(parts) if ii != primary_i]

        out = slab.copy()
        is_reduction_h_voc = (set(_clean(p) for p in parts) == {"H", "CH3CHO"})
        for i, part in enumerate(ordered):
            xy = None if i == 0 else _offset_xy_for_preview(slab, site, offset_A=1.25 * i)
            if part.replace("*", "").upper() == "H":
                from ase import Atoms
                if is_reduction_h_voc:
                    target = _surface_h_position_for_preview(slab, site, prefer_adjacent=True, oh_length=0.98)
                else:
                    target = _surface_h_position_for_preview(slab, site, prefer_adjacent=(i > 0), oh_length=0.98)
                h = Atoms("H", positions=[tuple(float(x) for x in target)], cell=out.get_cell(), pbc=out.get_pbc())
                out = out + h
            else:
                if is_reduction_h_voc and _clean(part) == "CH3CHO":
                    out = out + _place_preview_reduction_ch3cho(slab, site, dz=1.35, ref_dir=ref_dir)
                else:
                    out = out + _place_preview_component(slab, site, part, dz=dz + 0.10 * i, ref_dir=ref_dir, xy_override=xy)
        return out

    if ads_s.replace("*", "").upper() == "H":
        from ase import Atoms
        target = _surface_h_position_for_preview(slab, site, prefer_adjacent=False, oh_length=0.98)
        return slab + Atoms("H", positions=[tuple(float(x) for x in target)], cell=slab.get_cell(), pbc=slab.get_pbc())

    ads_atoms = _place_preview_component(slab, site, ads_s, dz=dz, ref_dir=ref_dir)
    return slab + ads_atoms


def export_zip_of_struct_map(struct_map: dict, symprec: float = 0.1) -> BytesIO:
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, atoms in (struct_map or {}).items():
            zf.writestr(f"{k}.cif", atoms_to_cif_bytes(atoms, symprec=symprec))
    zip_buf.seek(0)
    return zip_buf
