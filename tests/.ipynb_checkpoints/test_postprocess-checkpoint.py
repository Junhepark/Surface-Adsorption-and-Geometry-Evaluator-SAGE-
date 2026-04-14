import pandas as pd

from ocp_app.core.postprocess import _normalize_text_series, annotate_site_transitions


def test_normalize_text_series_basic_behavior():
    s = pd.Series([" Top ", "BRIDGE ", "  ok", "", None])

    out = _normalize_text_series(s)

    assert out.tolist() == ["top", "bridge", "ok", "", "none"]


def test_annotate_site_transitions_marks_requested_change():
    df = pd.DataFrame(
        {
            "requested_site": ["top"],
            "site": ["bridge"],
            "relaxed_site": ["bridge"],
            "H_lateral_disp(Å)": [0.10],
            "qa": ["ok"],
        }
    )

    out = annotate_site_transitions(df)

    assert "migration_path" in out.columns
    assert "site_transition_type" in out.columns
    assert "placement_mismatch" in out.columns
    assert "migrated_requested" in out.columns

    assert bool(out.loc[0, "placement_mismatch"]) is True
    assert bool(out.loc[0, "migrated_requested"]) is True