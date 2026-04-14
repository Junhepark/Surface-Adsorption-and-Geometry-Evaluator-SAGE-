from collections import Counter
from functools import reduce
from math import gcd

import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except Exception:
    SpacegroupAnalyzer = None

from ocp_app.core.anchors.oxide_her import _pbc_min_image_xy_distance_sq, _top_surface_o_indices
from ocp_app.core.structure_check import validate_structure
from ocp_app.core.structure_ops import _recenter_slab_z_into_cell


def _infer_crystal_metadata_from_atoms(atoms):
    crystal_system = "unknown"
    spacegroup_symbol = None
    spacegroup_number = None
    if SpacegroupAnalyzer is None:
        return {
            "crystal_system": crystal_system,
            "spacegroup_symbol": spacegroup_symbol,
            "spacegroup_number": spacegroup_number,
        }
    try:
        struct = AseAtomsAdaptor.get_structure(atoms)
        sga = SpacegroupAnalyzer(struct, symprec=0.1, angle_tolerance=5.0)
        crystal_system = str(sga.get_crystal_system() or "unknown")
        sg_symbol = sga.get_space_group_symbol()
        sg_number = sga.get_space_group_number()
        spacegroup_symbol = str(sg_symbol) if sg_symbol else None
        spacegroup_number = int(sg_number) if sg_number is not None else None
    except Exception:
        pass
    return {
        "crystal_system": crystal_system,
        "spacegroup_symbol": spacegroup_symbol,
        "spacegroup_number": spacegroup_number,
    }


def _family_from_stoich_and_structure(O_red: int, cation_sum: int, crystal_system: str, spacegroup_number=None):
    cs = str(crystal_system or "unknown").lower()
    sg = int(spacegroup_number) if spacegroup_number is not None else None

    if O_red == 1 and cation_sum == 1:
        if cs == "cubic":
            return "cubic_AO"
        if cs == "tetragonal":
            return "tetragonal_AO"
        if cs == "orthorhombic":
            return "orthorhombic_AO"
        if cs == "hexagonal":
            return "hexagonal_AO"
        if cs == "trigonal":
            return "trigonal_AO"
        if cs == "monoclinic":
            return "monoclinic_AO"
        if cs == "triclinic":
            return "triclinic_AO"
        return "generic_AO"

    if O_red == 2 and cation_sum == 1:
        if sg == 136:
            return "rutile_AO2"
        if sg == 141:
            return "anatase_AO2"
        if cs == "tetragonal":
            return "tetragonal_AO2"
        if cs == "orthorhombic":
            return "orthorhombic_AO2"
        if cs == "hexagonal":
            return "hexagonal_AO2"
        if cs == "trigonal":
            return "trigonal_AO2"
        if cs == "monoclinic":
            return "monoclinic_AO2"
        if cs == "triclinic":
            return "triclinic_AO2"
        if cs == "cubic":
            return "cubic_AO2"
        return "generic_AO2"

    if O_red == 3 and cation_sum == 2:
        return f"{cs}_ABO3" if cs not in ("", "unknown") else "generic_ABO3"

    if O_red == 4 and cation_sum == 3:
        return f"{cs}_AB2O4" if cs not in ("", "unknown") else "generic_AB2O4"

    return "unknown"


def infer_oxide_family_from_atoms(atoms):
    symbols = [s for s in atoms.get_chemical_symbols() if s != "H"]
    counts = Counter(symbols)
    O = counts.pop("O", 0)
    crystal_meta = _infer_crystal_metadata_from_atoms(atoms)
    if O == 0 or len(counts) == 0:
        return {
            "family": "unknown",
            "reduced_formula": None,
            **crystal_meta,
        }

    nums = list(counts.values()) + [O]
    g = reduce(gcd, nums) if nums else 1
    if g == 0:
        g = 1

    red_counts = {el: n // g for el, n in counts.items()}
    O_red = O // g
    cation_sum = sum(red_counts.values())

    parts = [f"{el}{n if n > 1 else ''}" for el, n in sorted(red_counts.items())]
    if O_red > 0:
        parts.append(f"O{O_red if O_red > 1 else ''}")
    reduced_formula = "".join(parts)

    family = _family_from_stoich_and_structure(
        O_red=O_red,
        cation_sum=cation_sum,
        crystal_system=crystal_meta.get("crystal_system", "unknown"),
        spacegroup_number=crystal_meta.get("spacegroup_number"),
    )

    return {
        "family": family,
        "reduced_formula": reduced_formula,
        **crystal_meta,
    }


def _normalize_miller_tuple(hkl):
    if hkl is None:
        return None
    try:
        return tuple(int(x) for x in hkl)
    except Exception:
        return None


def _classify_surface_exposure(atoms, z_window: float = 1.8):
    """Best-effort oxide surface exposure classification from top/bottom z-windows."""
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)
    pos = atoms.get_positions()
    z = pos[:, 2]
    zmax = float(np.max(z))
    zmin = float(np.min(z))

    top_idx = np.where(z >= (zmax - float(z_window)))[0]
    bot_idx = np.where(z <= (zmin + float(z_window)))[0]

    def _label(idxs):
        if len(idxs) == 0:
            return "unknown", 0.0
        sy = syms[idxs]
        n_o = int(np.sum(sy == "O"))
        frac_o = float(n_o / max(len(idxs), 1))
        if frac_o >= 0.80:
            lab = "O-rich"
        elif frac_o <= 0.20:
            lab = "metal-rich"
        else:
            lab = "mixed"
        return lab, frac_o

    top_lab, top_frac = _label(top_idx)
    bot_lab, bot_frac = _label(bot_idx)
    asym = (top_lab != bot_lab)
    return {
        "top_exposure": top_lab,
        "bottom_exposure": bot_lab,
        "surface_O_fraction_top": float(top_frac),
        "surface_O_fraction_bottom": float(bot_frac),
        "top_bottom_asymmetric": bool(asym),
    }


def _flip_slab_z_keep_cell(atoms):
    """Flip a slab along z while keeping the same cell, pbc, and vacuum extent."""
    a = atoms.copy()
    pos = a.get_positions().copy()
    z = pos[:, 2]
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    pos[:, 2] = (zmax + zmin) - z
    a.set_positions(pos)
    return _recenter_slab_z_into_cell(a, margin=1.0)


def _oxide_family_facet_profile(surface_family: str, miller=None):
    fam = str(surface_family or "generic")
    hkl = _normalize_miller_tuple(miller)

    validity = "warn"
    role = "exploratory"
    facet_pref = 2
    notes = []

    if fam == "cubic_AO":
        if hkl == (1, 0, 0):
            validity = "pass"
            role = "reference"
            facet_pref = 0
            notes.append("Cubic AO (100) is treated as the primary clean reference facet.")
        elif hkl == (1, 1, 0):
            validity = "pass"
            role = "secondary"
            facet_pref = 1
            notes.append("Cubic AO (110) is retained as a secondary clean facet.")
        elif hkl == (1, 1, 1):
            validity = "warn"
            role = "advanced"
            facet_pref = 3
            notes.append("Cubic AO (111) is treated as a polar / advanced clean facet.")
        else:
            notes.append("Cubic AO facet is outside the conservative reference set; keep as exploratory only.")
    elif fam in {"monoclinic_AO", "triclinic_AO", "orthorhombic_AO"}:
        if hkl in {(1, 0, 0), (0, 1, 0), (0, 0, 1)}:
            facet_pref = 0
        elif hkl in {(1, 1, 0), (1, 0, 1), (0, 1, 1)}:
            facet_pref = 1
        else:
            facet_pref = 2
        validity = "warn"
        role = "exploratory"
        notes.append("Low-symmetry AO oxide: treat clean facets as exploratory unless facet-specific validation is available.")
    elif fam in {"tetragonal_AO", "hexagonal_AO", "trigonal_AO", "generic_AO"}:
        validity = "warn"
        role = "exploratory"
        facet_pref = 1 if hkl in {(1, 0, 0), (0, 0, 1)} else 2
        notes.append("Non-cubic AO oxide: no hard clean reference facet is encoded; keep as exploratory.")
    elif fam == "rutile_AO2":
        if hkl == (1, 1, 0):
            validity = "pass"
            role = "reference"
            facet_pref = 0
            notes.append("Rutile AO2 (110) is treated as the primary clean reference facet.")
        elif hkl == (1, 0, 0):
            validity = "warn"
            role = "exploratory"
            facet_pref = 1
            notes.append("Rutile AO2 (100) is retained as a non-reference exploratory facet.")
        elif hkl == (1, 0, 1):
            validity = "warn"
            role = "exploratory"
            facet_pref = 2
            notes.append("Rutile AO2 (101) is retained as a non-reference exploratory facet.")
        else:
            facet_pref = 3
            notes.append("Rutile AO2 facet is outside the conservative reference set; keep as exploratory only.")
    elif fam == "anatase_AO2":
        if hkl == (1, 0, 1):
            validity = "pass"
            role = "reference"
            facet_pref = 0
            notes.append("Anatase AO2 (101) is treated as the primary clean reference facet.")
        elif hkl == (0, 0, 1):
            validity = "warn"
            role = "secondary"
            facet_pref = 1
            notes.append("Anatase AO2 (001) is retained as a secondary / higher-energy clean facet.")
        elif hkl == (1, 0, 0):
            validity = "warn"
            role = "exploratory"
            facet_pref = 2
            notes.append("Anatase AO2 (100) is treated as exploratory.")
        else:
            facet_pref = 3
            notes.append("Anatase AO2 facet is outside the conservative reference set; keep as exploratory only.")
    elif fam.endswith("_AO2") or fam == "generic_AO2":
        validity = "warn"
        role = "exploratory"
        facet_pref = 1 if hkl in {(1, 0, 0), (0, 0, 1), (1, 1, 0)} else 2
        notes.append("Non-rutile AO2 oxide: keep clean facets as exploratory unless prototype-specific validation is available.")
    elif fam.endswith("_ABO3") or fam == "generic_ABO3":
        validity = "warn"
        role = "exploratory"
        facet_pref = 1 if hkl == (1, 0, 0) else 2
        notes.append("ABO3 oxide clean surfaces are termination-dependent and are kept as exploratory only.")
    elif fam.endswith("_AB2O4") or fam == "generic_AB2O4":
        validity = "warn"
        role = "exploratory"
        facet_pref = 1 if hkl == (1, 0, 0) else 2
        notes.append("AB2O4 oxide clean surfaces depend on cation distribution and are kept as exploratory only.")
    else:
        validity = "warn"
        role = "exploratory"
        facet_pref = 2
        notes.append("Generic oxide rules applied; no family-specific clean reference facet is encoded.")

    return {
        "oxide_validity": validity,
        "oxide_role": role,
        "rule_validity": validity,
        "rule_role": role,
        "oxide_facet_pref": int(facet_pref),
        "oxide_rule_notes": list(notes),
    }


def _collect_surface_diagnostic_notes(meta):
    m = dict(meta or {})
    notes = []
    status = "pass"

    top_exp = str(m.get("top_exposure", "unknown"))
    if top_exp == "metal-rich":
        status = "fail"
        notes.append("Top surface is metal-rich after normalization.")
    elif top_exp == "unknown":
        if status != "fail":
            status = "warn"
        notes.append("Top surface exposure could not be classified cleanly.")

    facet_warning = str(m.get("facet_warning", "") or "")
    if facet_warning:
        if status == "pass":
            status = "warn"
        notes.append(f"Facet warning: {facet_warning}.")

    if bool(m.get("top_bottom_asymmetric", False)):
        if status == "pass":
            status = "warn"
        notes.append("Top and bottom terminations are asymmetric.")

    issues = list(m.get("issues", []) or [])

    hard_issue_patterns = (
        "Very short ",
        "no neighbors found",
        "no neighbors within",
        "isolated fragments",
        "Cell volume is very small",
    )

    advisory_issue_patterns = (
        "Surface area is small",
        "Consider repeating to",
        "Vacuum along z is small",
    )

    has_hard_issues = any(
        any(pat.lower() in str(issue).lower() for pat in hard_issue_patterns)
        for issue in issues
    )

    has_advisory_issues = any(
        any(pat.lower() in str(issue).lower() for pat in advisory_issue_patterns)
        for issue in issues
    )

    if has_hard_issues:
        status = "fail"
        notes.append("General structure QC reported critical issues.")
    elif issues:
        if has_advisory_issues:
            notes.append("General structure QC reported advisory issues (e.g., repeat/vacuum recommendation).")
        else:
            if status == "pass":
                status = "warn"
            notes.append("General structure QC reported non-critical issues.")

    return {
        "surface_diagnostics_status": status,
        "surface_diagnostics_notes": notes,
        "has_critical_issues": bool(has_hard_issues),
        "has_advisory_issues": bool(has_advisory_issues),
    }


def _derive_oxide_slab_usability(meta):
    m = dict(meta or {})
    rule_validity = str(m.get("rule_validity", m.get("oxide_validity", "warn")))
    rule_role = str(m.get("rule_role", m.get("oxide_role", "exploratory")))
    diag_status = str(m.get("surface_diagnostics_status", "warn"))
    has_advisory_issues = bool(m.get("has_advisory_issues", False))

    if rule_validity == "reject" or diag_status == "fail":
        usability = "do_not_use"
        reason = "Rejected by clean-surface rules or by surface diagnostics."
    elif (
        rule_validity == "pass"
        and rule_role in ("reference", "secondary")
        and diag_status == "pass"
    ):
        usability = "reference_use"
        reason = "Conservative clean-surface reference / secondary slab."
    elif (
        rule_validity == "pass"
        and rule_role in ("reference", "secondary")
        and diag_status == "warn"
        and has_advisory_issues
        and "polar" not in str(m.get("facet_warning", "")).lower()
        and not bool(m.get("top_bottom_asymmetric", False))
    ):
        usability = "reference_use"
        reason = "Reference / secondary slab with advisory-only QC notes."
    else:
        usability = "exploratory_only"
        reason = "Use only for exploratory or illustrative clean-surface calculations."

    return {
        "slab_usability": usability,
        "slab_usability_reason": reason,
    }


def _annotate_oxide_candidate_validity(meta):
    m = dict(meta or {})
    fam = str(m.get("surface_family") or "generic")
    if fam in ("unknown", ""):
        fam = "generic"
    hkl = _normalize_miller_tuple(m.get("miller"))

    prof = _oxide_family_facet_profile(fam, hkl)
    rule_notes = list(prof.get("oxide_rule_notes", []))
    rule_validity = str(prof.get("rule_validity", prof.get("oxide_validity", "warn")))
    rule_role = str(prof.get("rule_role", prof.get("oxide_role", "exploratory")))

    m.update(prof)
    m["rule_validity"] = rule_validity
    m["rule_role"] = rule_role

    diag = _collect_surface_diagnostic_notes(m)
    m.update(diag)

    # Backward-compatible summary label for existing UI code.
    final_validity = rule_validity
    final_role = rule_role
    if str(m.get("surface_diagnostics_status", "warn")) == "fail":
        final_validity = "reject"
        final_role = "reject"
    elif str(m.get("surface_diagnostics_status", "warn")) == "warn" and final_validity == "pass":
        final_validity = "warn"
        if final_role == "reference":
            final_role = "secondary"

    m["oxide_validity"] = final_validity
    m["oxide_role"] = final_role
    m["oxide_rule_notes"] = rule_notes
    m["oxide_reject"] = (final_validity == "reject")

    usability = _derive_oxide_slab_usability(m)
    m.update(usability)
    return m


def _oxide_mode_keep_candidate(meta, mode: str):
    m = dict(meta or {})
    usability = str(m.get("slab_usability", "exploratory_only"))
    top_exp = str(m.get("top_exposure", "unknown"))
    mode = str(mode or "Reference clean surface")

    if usability == "do_not_use":
        return False

    if mode == "Reference clean surface":
        return usability == "reference_use"
    if mode == "O-dominant surface preference":
        return (top_exp in ("O-rich", "mixed")) and (usability in ("reference_use", "exploratory_only"))
    return usability in ("reference_use", "exploratory_only")


def _normalize_oxide_candidate_top_surface(atoms, meta=None, z_window: float = 1.8):
    """Normalize oxide slabs toward an O-dominant top surface and attach validity metadata.

    - If the bottom surface is more O-rich than the top, flip the slab.
    - If the top remains metal-rich after normalization, mark as rejected for clean oxide screening.
    - Attach family-aware validity / role metadata for downstream ranking and UI gating.
    """
    a = atoms.copy()
    m = dict(meta or {})
    surf = _classify_surface_exposure(a, z_window=float(z_window))

    exp_score = {"O-rich": 0, "mixed": 1, "metal-rich": 2, "unknown": 3}
    top_score = exp_score.get(str(surf.get("top_exposure", "unknown")), 3)
    bot_score = exp_score.get(str(surf.get("bottom_exposure", "unknown")), 3)

    flipped = False
    if bot_score < top_score:
        a = _flip_slab_z_keep_cell(a)
        flipped = True
        surf = _classify_surface_exposure(a, z_window=float(z_window))

    m.update(surf)
    m["flipped_for_oxide_top_exposure"] = bool(flipped)
    top_exp = str(m.get("top_exposure", "unknown"))
    m["oxide_top_surface_ok"] = top_exp in ("O-rich", "mixed")
    if not m["oxide_top_surface_ok"]:
        m.setdefault("issues", [])
        if "Top surface is metal-rich for oxide clean-slab screening." not in m["issues"]:
            m["issues"] = list(m["issues"]) + ["Top surface is metal-rich for oxide clean-slab screening."]

    m = _annotate_oxide_candidate_validity(m)
    return a, m


def _top_surface_o_anchor_sites_with_spacing(atoms, z_window: float = 2.2, min_xy_sep: float = 1.5):
    """Return top-surface oxygen indices filtered by lateral spacing."""
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
        if not too_close:
            chosen.append(int(idx))
    return chosen


def _build_oxide_oh_terminated_candidate(atoms, coverage: float = 0.25, z_window: float = 2.2, dz_oh: float = 0.98, min_xy_sep: float = 1.5):
    """Generate a simple OH-terminated oxide surface candidate by placing H on top-layer O atoms.

    This is a lightweight surface-state generator for oxide slabs. It does not attempt global optimization
    or explicit reconstruction; it only creates plausible hydroxylated starting models for ranking/testing.
    """
    a = atoms.copy()
    o_idx = _top_surface_o_anchor_sites_with_spacing(a, z_window=float(z_window), min_xy_sep=float(min_xy_sep))
    if not o_idx:
        return None, {"hydroxylation_state": "clean", "oh_added": 0, "target_coverage": float(coverage)}

    n_top = len(o_idx)
    n_add = max(1, int(np.ceil(float(coverage) * n_top)))
    o_pick = o_idx[:n_add]

    pos = a.get_positions()
    h_positions = []
    for idx in o_pick:
        xyz = np.array(pos[idx], dtype=float).copy()
        xyz[2] += float(dz_oh)
        h_positions.append(xyz)

    if not h_positions:
        return None, {"hydroxylation_state": "clean", "oh_added": 0, "target_coverage": float(coverage)}

    h_atoms = Atoms(
        symbols=['H'] * len(h_positions),
        positions=np.array(h_positions, dtype=float),
        cell=a.get_cell(),
        pbc=a.get_pbc(),
    )
    out = a + h_atoms
    out = _recenter_slab_z_into_cell(out, margin=1.0)
    meta = {
        "hydroxylation_state": f"OH-{int(round(float(coverage) * 100))}",
        "oh_added": int(len(h_positions)),
        "target_coverage": float(coverage),
        "oh_anchor_indices": [int(i) for i in o_pick],
    }
    return out, meta


def _expand_oxide_surface_state_candidates(cand_atoms, cand_meta, hydroxylation_mode: str = "Clean only"):
    """Expand clean oxide slab candidates into simple surface-state candidates (clean / hydroxylated)."""
    mode = str(hydroxylation_mode or "Clean only")
    out_atoms, out_meta = [], []
    for a, m in zip(cand_atoms or [], cand_meta or []):
        m0 = dict(m)
        m0.setdefault("hydroxylation_state", "clean")
        out_atoms.append(a)
        out_meta.append(m0)

        if mode in ("Add OH-terminated candidate (0.25 ML)", "Add OH-terminated candidates (0.25/0.50 ML)"):
            a25, meta25 = _build_oxide_oh_terminated_candidate(a, coverage=0.25)
            if a25 is not None:
                m25 = dict(m)
                m25.update(meta25)
                m25["n_atoms"] = len(a25)
                rep25 = validate_structure(a25, target_area=70.0)
                m25["vacuum_z"] = float(getattr(rep25, "vacuum_z", np.nan))
                m25["recommend_repeat"] = getattr(rep25, "recommend_repeat", None)
                m25["issues"] = getattr(rep25, "issues", [])
                m25.update(_classify_surface_exposure(a25, z_window=1.8))
                m25 = _annotate_oxide_candidate_validity(m25)
                out_atoms.append(a25)
                out_meta.append(m25)
        if mode == "Add OH-terminated candidates (0.25/0.50 ML)":
            a50, meta50 = _build_oxide_oh_terminated_candidate(a, coverage=0.50)
            if a50 is not None:
                m50 = dict(m)
                m50.update(meta50)
                m50["n_atoms"] = len(a50)
                rep50 = validate_structure(a50, target_area=70.0)
                m50["vacuum_z"] = float(getattr(rep50, "vacuum_z", np.nan))
                m50["recommend_repeat"] = getattr(rep50, "recommend_repeat", None)
                m50["issues"] = getattr(rep50, "issues", [])
                m50.update(_classify_surface_exposure(a50, z_window=1.8))
                m50 = _annotate_oxide_candidate_validity(m50)
                out_atoms.append(a50)
                out_meta.append(m50)
    return out_atoms, out_meta


def _oxide_candidate_rank_key(meta):
    m = dict(meta or {})
    validity = str(m.get("oxide_validity", "warn"))
    role = str(m.get("oxide_role", "exploratory"))
    usability = str(m.get("slab_usability", "exploratory_only"))
    issues = m.get("issues", []) or []
    n_issues = len(issues)
    n_atoms = int(m.get("n_atoms", 10**9))
    vac_penalty = abs(float(m.get("vacuum_z", 0.0)) - 15.0)
    top_exp = str(m.get("top_exposure", "unknown"))
    mode_pref = str(m.get("oxide_surface_mode", "Reference clean surface") or "Reference clean surface")

    usability_score = {"reference_use": 0, "exploratory_only": 1, "do_not_use": 4}.get(usability, 2)
    validity_score = {"pass": 0, "warn": 1, "reject": 4}.get(validity, 2)
    role_score = {"reference": 0, "secondary": 1, "exploratory": 2, "advanced": 3, "reject": 4}.get(role, 2)
    exp_score = {"O-rich": 0, "mixed": 1, "unknown": 2, "metal-rich": 4}.get(top_exp, 3)

    if mode_pref == "O-dominant surface preference":
        exp_score = {"O-rich": 0, "mixed": 1, "unknown": 3, "metal-rich": 5}.get(top_exp, 3)
    elif mode_pref == "Exploratory any clean termination":
        exp_score = {"O-rich": 0, "mixed": 1, "unknown": 1, "metal-rich": 5}.get(top_exp, 2)

    asym_pen = 1 if bool(m.get("top_bottom_asymmetric", False)) else 0
    polar_pen = 1 if "polar" in str(m.get("facet_warning", "")).lower() else 0
    hydrox = str(m.get("hydroxylation_state", "clean")).lower()
    hydrox_pen = 0
    if hydrox not in ("clean", ""):
        hydrox_pen = 1 if "25" in hydrox else 2
    facet_pref = int(m.get("oxide_facet_pref", 2))
    return (usability_score, validity_score, role_score, facet_pref, exp_score, hydrox_pen, asym_pen, polar_pen, n_issues, n_atoms, vac_penalty)


def _pick_best_oxide_slab_candidate(cand_atoms, cand_meta):
    if not cand_atoms:
        raise ValueError("No oxide slab candidates generated.")
    best_i = 0
    best_key = None
    for i, meta in enumerate(cand_meta or []):
        key = _oxide_candidate_rank_key(meta)
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return cand_atoms[best_i].copy(), (cand_meta[best_i] if cand_meta else {"idx": best_i})
