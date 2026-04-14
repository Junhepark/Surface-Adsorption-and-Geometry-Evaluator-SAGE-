from collections import Counter

from ocp_app.core.ads_sites import AdsSite, select_representative_sites


def test_select_representative_sites_limits_per_kind():
    sites = [
        AdsSite(kind="ontop", position=(0.0, 0.0, 1.0), surface_indices=(0,)),
        AdsSite(kind="ontop", position=(5.0, 0.0, 1.0), surface_indices=(1,)),
        AdsSite(kind="bridge", position=(1.0, 1.0, 1.1), surface_indices=(0, 1)),
        AdsSite(kind="bridge", position=(6.0, 1.0, 1.1), surface_indices=(2, 3)),
        AdsSite(kind="hollow", position=(2.0, 2.0, 1.2), surface_indices=(0, 1, 2)),
    ]

    picked = select_representative_sites(sites, per_kind=1)
    counts = Counter(site.kind for site in picked)

    assert counts["ontop"] <= 1
    assert counts["bridge"] <= 1
    assert counts["hollow"] <= 1
    assert len(picked) == 3