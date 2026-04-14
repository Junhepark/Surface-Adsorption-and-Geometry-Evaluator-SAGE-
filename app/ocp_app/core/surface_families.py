from ocp_app.core.oxide_surface_rules import infer_oxide_family_from_atoms

INTERFACE_FACET_PRESETS = {
    "metal": {
        "Close-packed (111)": (1, 1, 1),
        "Square (100)": (1, 0, 0),
        "Row-like (110)": (1, 1, 0),
    },
    "cubic_AO": {
        "(100) — preferred": (1, 0, 0),
        "(110) — secondary": (1, 1, 0),
        "(111) — polar / advanced": (1, 1, 1),
    },
    "monoclinic_AO": {
        "(100) — exploratory": (1, 0, 0),
        "(010) — exploratory": (0, 1, 0),
        "(001) — exploratory": (0, 0, 1),
    },
    "orthorhombic_AO": {
        "(100) — exploratory": (1, 0, 0),
        "(010) — exploratory": (0, 1, 0),
        "(001) — exploratory": (0, 0, 1),
    },
    "triclinic_AO": {
        "(100) — exploratory": (1, 0, 0),
        "(010) — exploratory": (0, 1, 0),
        "(001) — exploratory": (0, 0, 1),
    },
    "tetragonal_AO": {
        "(001) — exploratory": (0, 0, 1),
        "(100) — exploratory": (1, 0, 0),
        "(110) — exploratory": (1, 1, 0),
    },
    "hexagonal_AO": {
        "(001) — exploratory": (0, 0, 1),
        "(100) — exploratory": (1, 0, 0),
        "(101) — exploratory": (1, 0, 1),
    },
    "trigonal_AO": {
        "(001) — exploratory": (0, 0, 1),
        "(100) — exploratory": (1, 0, 0),
        "(101) — exploratory": (1, 0, 1),
    },
    "rutile_AO2": {
        "(110) — preferred": (1, 1, 0),
        "(100) — non-reference": (1, 0, 0),
        "(101) — non-reference": (1, 0, 1),
    },
    "anatase_AO2": {
        "(101) — preferred": (1, 0, 1),
        "(001) — secondary": (0, 0, 1),
        "(100) — exploratory": (1, 0, 0),
    },
    "tetragonal_AO2": {
        "(001) — exploratory": (0, 0, 1),
        "(100) — exploratory": (1, 0, 0),
        "(110) — exploratory": (1, 1, 0),
    },
    "monoclinic_AO2": {
        "(100) — exploratory": (1, 0, 0),
        "(010) — exploratory": (0, 1, 0),
        "(001) — exploratory": (0, 0, 1),
    },
    "orthorhombic_AO2": {
        "(100) — exploratory": (1, 0, 0),
        "(010) — exploratory": (0, 1, 0),
        "(001) — exploratory": (0, 0, 1),
    },
    "hexagonal_AO2": {
        "(001) — exploratory": (0, 0, 1),
        "(100) — exploratory": (1, 0, 0),
        "(101) — exploratory": (1, 0, 1),
    },
    "trigonal_AO2": {
        "(001) — exploratory": (0, 0, 1),
        "(100) — exploratory": (1, 0, 0),
        "(101) — exploratory": (1, 0, 1),
    },
    "cubic_AO2": {
        "(100) — exploratory": (1, 0, 0),
        "(110) — exploratory": (1, 1, 0),
        "(111) — exploratory": (1, 1, 1),
    },
    "generic_AO": {
        "(100) — exploratory": (1, 0, 0),
        "(110) — exploratory": (1, 1, 0),
        "(111) — exploratory": (1, 1, 1),
    },
    "generic_AO2": {
        "(100) — exploratory": (1, 0, 0),
        "(110) — exploratory": (1, 1, 0),
        "(111) — exploratory": (1, 1, 1),
    },
    "cubic_ABO3": {
        "(100) — termination-dependent": (1, 0, 0),
        "(110) — exploratory": (1, 1, 0),
        "(111) — exploratory": (1, 1, 1),
    },
    "tetragonal_ABO3": {
        "(100) — termination-dependent": (1, 0, 0),
        "(001) — termination-dependent": (0, 0, 1),
        "(110) — exploratory": (1, 1, 0),
    },
    "orthorhombic_ABO3": {
        "(100) — exploratory": (1, 0, 0),
        "(010) — exploratory": (0, 1, 0),
        "(001) — exploratory": (0, 0, 1),
    },
    "monoclinic_ABO3": {
        "(100) — exploratory": (1, 0, 0),
        "(010) — exploratory": (0, 1, 0),
        "(001) — exploratory": (0, 0, 1),
    },
    "generic_ABO3": {
        "(100) — exploratory": (1, 0, 0),
        "(110) — exploratory": (1, 1, 0),
        "(111) — exploratory": (1, 1, 1),
    },
    "cubic_AB2O4": {
        "(100) — exploratory": (1, 0, 0),
        "(110) — exploratory": (1, 1, 0),
        "(111) — exploratory": (1, 1, 1),
    },
    "tetragonal_AB2O4": {
        "(001) — exploratory": (0, 0, 1),
        "(100) — exploratory": (1, 0, 0),
        "(110) — exploratory": (1, 1, 0),
    },
    "orthorhombic_AB2O4": {
        "(100) — exploratory": (1, 0, 0),
        "(010) — exploratory": (0, 1, 0),
        "(001) — exploratory": (0, 0, 1),
    },
    "monoclinic_AB2O4": {
        "(100) — exploratory": (1, 0, 0),
        "(010) — exploratory": (0, 1, 0),
        "(001) — exploratory": (0, 0, 1),
    },
    "generic_AB2O4": {
        "(100) — exploratory": (1, 0, 0),
        "(110) — exploratory": (1, 1, 0),
        "(111) — exploratory": (1, 1, 1),
    },
    "generic": {
        "(111)": (1, 1, 1),
        "(100)": (1, 0, 0),
        "(110)": (1, 1, 0),
    },
}

THICKNESS_ALLOCATION_OPTIONS = ["10:0", "8:2", "7:3", "6:4", "4:6", "3:7", "2:8", "0:10"]


def _infer_interface_surface_family(atoms) -> str:
    syms = set(atoms.get_chemical_symbols())
    if "O" not in syms:
        return "metal"
    fam = infer_oxide_family_from_atoms(atoms).get("family", "unknown")
    return fam if fam not in {"", "unknown", None} else "generic"


def _get_interface_facet_options(atoms):
    fam = _infer_interface_surface_family(atoms)
    return INTERFACE_FACET_PRESETS.get(fam, INTERFACE_FACET_PRESETS["generic"]), fam


def _recommended_interface_facet_labels(fam: str, facet_opts: dict):
    fam = str(fam or "generic")
    if fam == "metal":
        prefs = ["Close-packed (111)", "Square (100)"]
    elif fam == "cubic_AO":
        prefs = ["(100) — preferred", "(110) — secondary"]
    elif fam == "rutile_AO2":
        prefs = ["(110) — preferred", "(100) — non-reference"]
    elif fam == "anatase_AO2":
        prefs = ["(101) — preferred", "(001) — secondary"]
    elif fam.endswith("_AO") and fam != "cubic_AO":
        prefs = list(facet_opts.keys())[:2]
    elif fam.endswith("_AO2") and fam not in {"rutile_AO2", "anatase_AO2"}:
        prefs = list(facet_opts.keys())[:2]
    elif fam.endswith("_ABO3") or fam.endswith("_AB2O4"):
        prefs = list(facet_opts.keys())[:2]
    else:
        prefs = list(facet_opts.keys())[:2]
    out = [k for k in prefs if k in facet_opts]
    return out or list(facet_opts.keys())[:1]


def _facet_labels_for_mode(facet_opts: dict, mode: str, manual_label: str, fam: str):
    mode = str(mode or "Recommended")
    if mode == "Recommended":
        return _recommended_interface_facet_labels(fam, facet_opts)
    if mode == "Explore all (slow)":
        return list(facet_opts.keys())
    return [manual_label]
