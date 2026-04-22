from math import ceil, gcd

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

from ocp_app.core.oxide_surface_rules import _pick_best_oxide_slab_candidate, _classify_surface_exposure, infer_oxide_family_from_atoms
from ocp_app.core.structure_check import validate_structure
from ocp_app.core.structure_ops import set_pbc_z
from ocp_app.core.surface_families import _get_interface_facet_options, _infer_interface_surface_family

HAS_SLABIFY = True
SLABIFY_IMPORT_ERR = None
try:
    from pymatgen.core.surface import SlabGenerator
    try:
        from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
    except Exception:
        get_symmetrically_distinct_miller_indices = None
except Exception as e:
    HAS_SLABIFY = False
    SLABIFY_IMPORT_ERR = str(e)
    get_symmetrically_distinct_miller_indices = None

_DEFAULT_SLAB_MIN_THICKNESS = 12.0
_DEFAULT_SLAB_MAX_CANDIDATES = 6
_DEFAULT_MIN_BULK_ATOMS_BEFORE_SLABIFY = 8
_DEFAULT_TARGET_BULK_ATOMS_BEFORE_SLABIFY = 16
_DEFAULT_MAX_BULK_REPEAT_BEFORE_SLABIFY = 3
_DEFAULT_MIN_AXIS_SPAN_BEFORE_SLABIFY_A = 8.0


def _facet_warning_for_family_miller(surface_family: str, miller) -> str:
    fam = str(surface_family or 'generic')
    hkl = tuple(int(x) for x in miller)
    if fam == 'cubic_AO' and hkl == (1, 1, 1):
        return 'polar / advanced cubic AO facet'
    if fam == 'rutile_AO2' and hkl in {(1, 0, 0), (1, 0, 1)}:
        return 'non-reference rutile facet'
    if fam == 'anatase_AO2' and hkl == (0, 0, 1):
        return 'higher-energy anatase facet'
    if fam in {'monoclinic_AO', 'orthorhombic_AO', 'triclinic_AO', 'monoclinic_AO2', 'orthorhombic_AO2', 'triclinic_AO2'}:
        return 'facet-specific clean reference not yet defined for this low-symmetry oxide'
    if fam.endswith('_ABO3'):
        return 'termination-dependent ABO3 facet'
    if fam.endswith('_AB2O4'):
        return 'cation-distribution-dependent AB2O4 facet'
    return ''


def _pick_best_slab_candidate_auto(cand_atoms, cand_meta, mtype_for_rank: str = "metal"):
    if str(mtype_for_rank).lower() == "oxide":
        return _pick_best_oxide_slab_candidate(cand_atoms, cand_meta)
    return _pick_best_slab_candidate(cand_atoms, cand_meta)


def _pick_best_slab_candidate(cand_atoms, cand_meta):
    if not cand_atoms:
        raise ValueError("No slab candidates generated.")
    best_i = 0
    best_key = None
    for i, meta in enumerate(cand_meta or []):
        issues = meta.get("issues", []) or []
        n_issues = len(issues)
        n_atoms = int(meta.get("n_atoms", len(cand_atoms[i])))
        vac_penalty = abs(float(meta.get("vacuum_z", 0.0)) - 15.0)
        key = (n_issues, n_atoms, vac_penalty)
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return cand_atoms[best_i].copy(), (cand_meta[best_i] if cand_meta else {"idx": best_i})


def _miller_sort_key(hkl):
    h, k, l = [abs(int(x)) for x in hkl]
    return (max(h, k, l), h + k + l, h, k, l)


def _format_hkl_label(hkl):
    h, k, l = [int(x) for x in hkl]
    return f"({h}{k}{l})"


def _reduce_hkl_by_gcd(hkl):
    vals = [abs(int(x)) for x in hkl]
    g = 0
    for v in vals:
        g = gcd(g, v)
    g = max(g, 1)
    return tuple(int(x) // g for x in hkl)


def _enumerate_low_index_millers(max_index: int = 2):
    out = []
    seen = set()
    for h in range(0, int(max_index) + 1):
        for k in range(0, int(max_index) + 1):
            for l in range(0, int(max_index) + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkl = _reduce_hkl_by_gcd((h, k, l))
                if hkl in seen:
                    continue
                seen.add(hkl)
                out.append(hkl)
    out.sort(key=_miller_sort_key)
    return out


def _recommended_step2_facets(atoms):
    facet_opts, _fam = _get_interface_facet_options(atoms)
    out = []
    seen = set()
    for hkl in facet_opts.values():
        hkl = tuple(int(x) for x in hkl)
        if hkl in seen:
            continue
        seen.add(hkl)
        out.append(hkl)
    out.sort(key=_miller_sort_key)
    return out


def _facet_choices_for_scope(atoms, scope_label: str):
    scope = str(scope_label or "Recommended facets")
    if scope == "Recommended facets":
        cands = _recommended_step2_facets(atoms)
    elif scope == "Low-index facets (up to 1)":
        cands = _enumerate_low_index_millers(max_index=1)
    else:
        cands = _enumerate_low_index_millers(max_index=2)

    if HAS_SLABIFY and (get_symmetrically_distinct_miller_indices is not None):
        try:
            struct = AseAtomsAdaptor.get_structure(atoms)
            max_idx = 1 if scope == "Low-index facets (up to 1)" else 2
            if scope == "Recommended facets":
                max_idx = 2
            distinct = {tuple(int(x) for x in hkl) for hkl in get_symmetrically_distinct_miller_indices(struct, int(max_idx))}
            if distinct:
                cands = [hkl for hkl in cands if tuple(hkl) in distinct]
                if scope == "Recommended facets":
                    rec = _recommended_step2_facets(atoms)
                    cands = [hkl for hkl in rec if hkl in distinct] or cands
        except Exception:
            pass

    return cands


def _vacuum_target_from_ui(choice_label: str, custom_value: float | None = None) -> float:
    choice = str(choice_label or "30 Å (recommended)")
    if choice.startswith("20"):
        return 20.0
    if choice.startswith("40"):
        return 40.0
    if choice.startswith("Custom"):
        return float(custom_value if custom_value is not None else 30.0)
    return 30.0




def _suggest_miller_aware_bulk_repeat(
    atoms,
    miller=(1, 1, 1),
    *,
    min_axis_span_A: float = _DEFAULT_MIN_AXIS_SPAN_BEFORE_SLABIFY_A,
    max_repeat: int = _DEFAULT_MAX_BULK_REPEAT_BEFORE_SLABIFY,
):
    """Heuristic pre-slabify repeat suggestion tied to the requested Miller index.

    This is intentionally conservative. It does not guarantee an ideal slab for every
    crystal system, but it reduces the chance that a very short parent cell is asked
    to support a non-trivial Miller cut without enough real-space span.
    """
    hkl = tuple(int(x) for x in miller)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    axis_lengths = [float(np.linalg.norm(cell[i])) for i in range(3)]
    reps = [1, 1, 1]
    for i, (h_i, axis_len) in enumerate(zip(hkl, axis_lengths)):
        if int(h_i) == 0:
            continue
        if axis_len <= 1e-8:
            continue
        reps[i] = int(min(max_repeat, max(1, ceil(float(min_axis_span_A) / float(axis_len)))))
    return tuple(int(x) for x in reps), {
        "bulk_axis_lengths_before_slabify_A": tuple(float(x) for x in axis_lengths),
        "bulk_min_axis_span_target_before_slabify_A": float(min_axis_span_A),
    }



def _expand_small_bulk_before_slabify(
    atoms,
    *,
    miller=(1, 1, 1),
    min_bulk_atoms: int = _DEFAULT_MIN_BULK_ATOMS_BEFORE_SLABIFY,
    target_bulk_atoms: int = _DEFAULT_TARGET_BULK_ATOMS_BEFORE_SLABIFY,
    max_repeat: int = _DEFAULT_MAX_BULK_REPEAT_BEFORE_SLABIFY,
    min_axis_span_A: float = _DEFAULT_MIN_AXIS_SPAN_BEFORE_SLABIFY_A,
):
    """Pre-expand bulk before slabify using bounded isotropic + Miller-aware heuristics.

    Two bounded rules are combined:
      1) isotropic upscaling for ultra-small bulk cells;
      2) axis-wise upscaling when the requested Miller index relies on axes that are too short.

    The final repeat is the element-wise maximum of those two suggestions, capped by
    ``max_repeat`` on each axis.
    """
    n_before = int(len(atoms))
    iso_rep = 1
    if n_before > 0 and n_before < int(min_bulk_atoms):
        iso_rep = int(max(2, ceil((float(target_bulk_atoms) / float(n_before)) ** (1.0 / 3.0))))
        iso_rep = int(min(max_repeat, max(2, iso_rep)))

    miller_rep, miller_meta = _suggest_miller_aware_bulk_repeat(
        atoms,
        miller=miller,
        min_axis_span_A=float(min_axis_span_A),
        max_repeat=int(max_repeat),
    )
    total_rep = tuple(int(max(iso_rep, miller_rep[i])) for i in range(3))

    meta = {
        "bulk_atoms_before_slabify": n_before,
        "bulk_atoms_after_prescale": n_before,
        "bulk_atoms_after_miller_prescale": n_before,
        "bulk_expanded_before_slabify": total_rep != (1, 1, 1),
        "bulk_supercell_before_slabify": total_rep,
        "bulk_isotropic_supercell_before_slabify": (iso_rep, iso_rep, iso_rep),
        "bulk_miller_aware_supercell_before_slabify": tuple(int(x) for x in miller_rep),
        "bulk_requested_miller_for_prescale": tuple(int(x) for x in miller),
        **miller_meta,
    }
    if total_rep == (1, 1, 1):
        return atoms.copy(), meta

    expanded = atoms.repeat(total_rep)
    meta["bulk_atoms_after_prescale"] = int(len(expanded))
    meta["bulk_atoms_after_miller_prescale"] = int(len(expanded))
    return expanded, meta


def slabify_from_bulk(atoms, miller=(1, 1, 1), min_slab_size=12.0, min_vacuum_size=30.0, max_candidates=6):
    if not HAS_SLABIFY:
        raise RuntimeError(f"pymatgen SlabGenerator not available: {SLABIFY_IMPORT_ERR}")

    miller = tuple(int(x) for x in miller)
    atoms_for_slabify, prescale_meta = _expand_small_bulk_before_slabify(atoms, miller=miller)
    struct = AseAtomsAdaptor.get_structure(atoms_for_slabify)
    gen = SlabGenerator(
        initial_structure=struct,
        miller_index=tuple(int(x) for x in miller),
        min_slab_size=float(min_slab_size),
        min_vacuum_size=float(min_vacuum_size),
        center_slab=True,
        in_unit_planes=False,
        primitive=True,
        reorient_lattice=True,
    )
    slabs = gen.get_slabs(symmetrize=False)[: int(max_candidates)]

    cand_atoms = []
    cand_meta = []
    oxide_info = infer_oxide_family_from_atoms(atoms_for_slabify)
    fam = _infer_interface_surface_family(atoms_for_slabify)
    facet_warning = _facet_warning_for_family_miller(fam, miller)

    for i, s in enumerate(slabs):
        a = AseAtomsAdaptor.get_atoms(s)
        a = set_pbc_z(a, True)
        rep = validate_structure(a, target_area=70.0)
        meta = {
            "idx": i,
            "miller": miller,
            "n_atoms": len(a),
            "formula": a.get_chemical_formula(),
            "vacuum_z": float(getattr(rep, "vacuum_z", np.nan)),
            "recommend_repeat": getattr(rep, "recommend_repeat", None),
            "issues": getattr(rep, "issues", []),
            "surface_family": fam,
            "crystal_system": oxide_info.get("crystal_system"),
            "spacegroup_symbol": oxide_info.get("spacegroup_symbol"),
            "spacegroup_number": oxide_info.get("spacegroup_number"),
            "facet_warning": facet_warning,
            **prescale_meta,
        }
        if fam != "metal":
            meta.update(_classify_surface_exposure(a, z_window=1.8))
        cand_atoms.append(a)
        cand_meta.append(meta)
    return cand_atoms, cand_meta
