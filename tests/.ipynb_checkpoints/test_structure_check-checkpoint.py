from ase import Atoms

from ocp_app.core.structure_check import validate_structure


def test_validate_structure_returns_report_for_simple_atoms():
    atoms = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 1.0], [1.8, 1.8, 2.0]],
        cell=[5.0, 5.0, 15.0],
        pbc=[True, True, True],
    )

    report = validate_structure(atoms, target_area=10.0)

    assert hasattr(report, "issues")
    assert hasattr(report, "cell_lengths")
    assert hasattr(report, "vacuum_z")
    assert isinstance(report.issues, list)
    assert len(report.cell_lengths) == 3


def test_validate_structure_flags_very_short_distance():
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]],
        cell=[5.0, 5.0, 10.0],
        pbc=[True, True, True],
    )

    report = validate_structure(atoms, target_area=10.0)

    assert any("Very short" in issue for issue in report.issues)