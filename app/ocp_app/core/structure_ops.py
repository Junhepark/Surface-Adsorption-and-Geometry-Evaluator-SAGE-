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

