import os
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

def atoms_to_cif_string(atoms, symprec: float = 0.1) -> str:
    """Serialize ASE Atoms to CIF.

    Primary path: pymatgen CifWriter (keeps legacy behavior).
    Fallback path: ASE CIF writer to avoid spglib symmetry failures on slabs/relaxed structures.
    """
    try:
        struct = AseAtomsAdaptor.get_structure(atoms)
        return str(CifWriter(struct, symprec=symprec))
    except Exception:
        # Fallback: write P1 CIF via ASE (no symmetry search); use a temp file since ASE CIF writer expects a real file handle.
        fd, tmp_path = tempfile.mkstemp(prefix="ocpapp_", suffix=".cif")
        os.close(fd)
        try:
            write(tmp_path, atoms, format="cif")
            return Path(tmp_path).read_text(encoding="utf-8", errors="replace")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def atoms_to_cif_bytes(atoms, symprec: float = 0.1) -> bytes:
    return atoms_to_cif_string(atoms, symprec=symprec).encode("utf-8")

def atoms_to_xyz_string(atoms) -> str:
    buf = StringIO()
    write(buf, atoms, format="xyz")
    xyz_str = buf.getvalue()
    buf.close()
    return xyz_str

def _recenter_slab_z_into_cell(atoms, margin: float = 1.0):
    """Shift atoms along z so the slab sits well inside the unit cell (avoid z-wrapping artifacts).

    Only applies when the cell vector c is (approximately) aligned with z.
    Does not change the cell size.
    """
    a = atoms.copy()
    try:
        cell = a.get_cell()
        # Require c-vector ~ z-axis (orthorhombic/near-orthorhombic slabs)
        if abs(cell[2, 0]) > 1e-3 or abs(cell[2, 1]) > 1e-3:
            return a
        Lz = float(cell[2, 2])
        if not np.isfinite(Lz) or Lz <= 0:
            return a

        z = a.get_positions()[:, 2]
        zmin = float(np.min(z))
        zmax = float(np.max(z))
        zc = 0.5 * (zmin + zmax)

        # Center slab around Lz/2
        shift = (0.5 * Lz) - zc
        pos = a.get_positions()
        pos[:, 2] += shift
        a.set_positions(pos)

        # Enforce a margin to avoid any atoms slightly crossing boundaries after jiggle/relax.
        z = a.get_positions()[:, 2]
        if float(np.min(z)) < float(margin):
            pos = a.get_positions()
            pos[:, 2] += (float(margin) - float(np.min(z)))
            a.set_positions(pos)
        z = a.get_positions()[:, 2]
        if float(np.max(z)) > (Lz - float(margin)):
            pos = a.get_positions()
            pos[:, 2] -= (float(np.max(z)) - (Lz - float(margin)))
            a.set_positions(pos)
    except Exception:
        return a
    return a

def add_vacuum_z(atoms, total_vacuum_z: float = 30.0, keep_pbc_z: bool = True):
    a = atoms.copy()
    try:
        a.wrap()
    except Exception:
        pass
    pbc = list(a.get_pbc())
    if len(pbc) != 3:
        pbc = [True, True, True]
    pbc[2] = bool(keep_pbc_z)
    a.set_pbc(pbc)
    a.center(vacuum=float(total_vacuum_z) / 2.0, axis=2)
    return a

def set_pbc_z(atoms, pbc_z: bool):
    a = atoms.copy()
    pbc = list(a.get_pbc())
    if len(pbc) != 3:
        pbc = [True, True, True]
    pbc[2] = bool(pbc_z)
    a.set_pbc(pbc)
    return a

def slab_thickness_z(atoms) -> float:
    """Return the geometric slab thickness along z (max(z)-min(z)) in Å."""
    try:
        if atoms is None or len(atoms) == 0:
            return float("nan")
        z = np.asarray(atoms.get_positions()[:, 2], dtype=float)
        if z.size == 0:
            return float("nan")
        return float(np.max(z) - np.min(z))
    except Exception:
        return float("nan")


def _cluster_indices_by_z(atoms, tol: float = 0.8):
    """Cluster atom indices into approximate z-layers from top to bottom."""
    if atoms is None or len(atoms) == 0:
        return []
    z = np.asarray(atoms.get_positions()[:, 2], dtype=float)
    order = np.argsort(z)[::-1]
    layers = []
    cur = [int(order[0])]
    cur_ref = float(z[order[0]])
    for idx in order[1:]:
        zi = float(z[idx])
        if abs(zi - cur_ref) <= float(tol):
            cur.append(int(idx))
            cur_ref = float(np.mean(z[cur]))
        else:
            layers.append(sorted(cur))
            cur = [int(idx)]
            cur_ref = zi
    if cur:
        layers.append(sorted(cur))
    return layers


def suggest_active_region_crop(
    atoms,
    thickness_threshold_A: float = 12.0,
    atoms_threshold: int = 160,
    default_window_A: float = 10.0,
):
    """Suggest whether an upper-slab crop is worth offering before adsorption runs."""
    thickness = slab_thickness_z(atoms)
    natoms = int(len(atoms)) if atoms is not None else 0
    recommend = bool(
        (np.isfinite(thickness) and float(thickness) > float(thickness_threshold_A))
        or (natoms > int(atoms_threshold))
    )
    if np.isfinite(thickness):
        suggested_window = float(min(max(default_window_A, 0.45 * float(thickness)), max(default_window_A, 14.0)))
    else:
        suggested_window = float(default_window_A)
    suggested_window = float(max(6.0, min(18.0, suggested_window)))
    return {
        "recommend_crop": recommend,
        "slab_thickness_A": float(thickness) if np.isfinite(thickness) else float("nan"),
        "n_atoms": natoms,
        "suggested_window_A": suggested_window,
        "reason": (
            "deep slab" if (np.isfinite(thickness) and float(thickness) > float(thickness_threshold_A)) else (
                "many atoms" if natoms > int(atoms_threshold) else "manual_only"
            )
        ),
    }


def crop_top_slab_window(
    atoms,
    keep_top_window_A: float = 10.0,
    target_vacuum_z: float = 30.0,
    keep_pbc_z: bool = True,
    min_layers: int = 4,
    min_atoms: int = 16,
    layer_tol: float = 0.8,
):
    """Keep only the top active region of a slab and rebuild z vacuum.

    The crop preserves full approximate z-layers instead of cutting through the
    middle of a layer whenever possible. This is intended as a pragmatic
    pre-screening simplification for surface adsorption calculations.
    """
    if atoms is None or len(atoms) == 0:
        raise ValueError("atoms is empty")

    a = atoms.copy()
    pos = a.get_positions()
    z = np.asarray(pos[:, 2], dtype=float)
    zmax = float(np.max(z))
    zcut = float(zmax - float(keep_top_window_A))

    layers = _cluster_indices_by_z(a, tol=float(layer_tol))
    if not layers:
        raise ValueError("failed to identify z-layers")

    keep = set()
    n_kept_layers = 0
    for layer in layers:
        layer_zmax = float(np.max(z[layer]))
        layer_zmin = float(np.min(z[layer]))
        must_keep = (layer_zmax >= zcut) or (layer_zmin >= zcut) or (n_kept_layers < int(min_layers))
        if must_keep:
            keep.update(int(i) for i in layer)
            n_kept_layers += 1
        elif n_kept_layers >= int(min_layers):
            break

    if len(keep) < int(min_atoms):
        for layer in layers[n_kept_layers:]:
            keep.update(int(i) for i in layer)
            n_kept_layers += 1
            if len(keep) >= int(min_atoms):
                break

    keep_idx = sorted(keep)
    if not keep_idx or len(keep_idx) >= len(a):
        # no effective crop; return a vacuum-normalized copy so downstream still behaves
        out = add_vacuum_z(a, total_vacuum_z=float(target_vacuum_z), keep_pbc_z=bool(keep_pbc_z))
        out = _recenter_slab_z_into_cell(out)
        return out, {
            "cropped": False,
            "keep_top_window_A": float(keep_top_window_A),
            "target_vacuum_z": float(target_vacuum_z),
            "min_layers": int(min_layers),
            "kept_atoms": int(len(a)),
            "removed_atoms": 0,
            "kept_layers": int(n_kept_layers),
            "original_thickness_A": float(slab_thickness_z(a)),
            "cropped_thickness_A": float(slab_thickness_z(out)),
        }

    out = a[keep_idx]
    try:
        out.set_cell(a.get_cell())
    except Exception:
        pass
    try:
        pbc = list(a.get_pbc())
        if len(pbc) != 3:
            pbc = [True, True, True]
        pbc[2] = bool(keep_pbc_z)
        out.set_pbc(pbc)
    except Exception:
        pass

    out = add_vacuum_z(out, total_vacuum_z=float(target_vacuum_z), keep_pbc_z=bool(keep_pbc_z))
    out = _recenter_slab_z_into_cell(out)

    meta = {
        "cropped": True,
        "keep_top_window_A": float(keep_top_window_A),
        "target_vacuum_z": float(target_vacuum_z),
        "min_layers": int(min_layers),
        "kept_atoms": int(len(out)),
        "removed_atoms": int(len(a) - len(out)),
        "kept_layers": int(n_kept_layers),
        "original_thickness_A": float(slab_thickness_z(a)),
        "cropped_thickness_A": float(slab_thickness_z(out)),
    }
    return out, meta

def repeat_xy(atoms, nx: int, ny: int):
    a = atoms.copy()
    return a.repeat((int(nx), int(ny), 1))

def _surface_xy_lengths(atoms):
    """Return in-plane cell lengths (a, b) for the current slab/surface."""
    try:
        cell = atoms.get_cell()
        a_len = float(np.linalg.norm(np.array(cell[0], dtype=float)))
        b_len = float(np.linalg.norm(np.array(cell[1], dtype=float)))
        return a_len, b_len
    except Exception:
        return float("nan"), float("nan")

def _suggest_minimal_xy_repeat(atoms, min_length_a: float = 8.0, min_length_b: float = 8.0, max_repeat: int = 3):
    """Suggest the smallest XY repeat that lifts the in-plane cell lengths above a simple threshold.

    Intended for the slabify route only: keep 1×1 by default and expand only when the slab is too small laterally.
    """
    a_len, b_len = _surface_xy_lengths(atoms)
    if not np.isfinite(a_len) or not np.isfinite(b_len) or a_len <= 0 or b_len <= 0:
        return 1, 1

    nx = max(1, int(np.ceil(float(min_length_a) / a_len)))
    ny = max(1, int(np.ceil(float(min_length_b) / b_len)))
    nx = min(int(max_repeat), nx)
    ny = min(int(max_repeat), ny)
    return int(nx), int(ny)

