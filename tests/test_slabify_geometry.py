"""Guard the slab coordinate convention used by z-based surface processing.

Fixed bulk fixtures avoid remote MP access, model weights, and relaxation.
"""

import numpy as np
import pytest
from ase.build import bulk
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ocp_app.core.slabify import slabify_from_bulk


@pytest.mark.parametrize("primitive", [False, True], ids=["conventional", "primitive"])
def test_ni_111_surface_normal_and_vacuum(primitive):
    atoms = bulk("Ni", "fcc", a=3.52, cubic=not primitive)
    slabs, metadata = slabify_from_bulk(atoms, miller=(1, 1, 1))
    assert len(slabs) == len(metadata) == 1
    assert len(slabs[0]) == 6
    _assert_z_surface(slabs[0], metadata[0])


def test_rutile_110_surface_normal_and_vacuum_for_both_terminations():
    structure = Structure.from_spacegroup(
        "P4_2/mnm", Lattice.tetragonal(4.49, 3.11),
        ["Ru", "O"], [[0, 0, 0], [0.305, 0.305, 0]],
    )
    slabs, metadata = slabify_from_bulk(
        AseAtomsAdaptor.get_atoms(structure), miller=(1, 1, 0),
    )
    assert len(slabs) == len(metadata) == 2
    for slab, meta in zip(slabs, metadata):
        assert len(slab) == 24
        _assert_z_surface(slab, meta)


def _assert_z_surface(slab, meta):
    cell = np.asarray(slab.cell)
    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(normal[:2], [0, 0], atol=1e-8)
    assert abs(normal[2]) == pytest.approx(1, abs=1e-8)
    projected_vacuum = abs(cell[2] @ normal) - np.ptp(slab.positions @ normal)
    assert projected_vacuum >= 30
    assert meta["vacuum_z"] == pytest.approx(projected_vacuum, abs=1e-8)
