import numpy as np

from ocp_app.core.surface_families import _infer_interface_surface_family
from ocp_app.core.slabify import slabify_from_bulk, _pick_best_slab_candidate_auto
from ocp_app.core.structure_check import validate_structure
from ocp_app.core.structure_ops import _recenter_slab_z_into_cell, repeat_xy

GLOBAL_SEED = 42

HAS_ADSORML = True
ADSORML_IMPORT_ERR = None
try:
    from ocp_app.core.adsorbml_lite_screening import relax_slab_chgnet
except Exception as e:
    HAS_ADSORML = False
    ADSORML_IMPORT_ERR = str(e)

def _allocation_to_thicknesses(allocation: str, total_thickness: float = 28.0, min_each: float = 6.0):
    a_str, b_str = allocation.split(":")
    a, b = int(a_str), int(b_str)
    s = max(a + b, 1)
    if a == 0:
        t_a = 0.0
    else:
        t_a = max(min_each, total_thickness * (a / s))
    if b == 0:
        t_b = 0.0
    else:
        t_b = max(min_each, total_thickness * (b / s))
    return t_a, t_b

def _inplane_metrics(atoms):
    cell = atoms.get_cell()
    a_vec = np.array(cell[0], dtype=float)
    b_vec = np.array(cell[1], dtype=float)
    a = float(np.linalg.norm(a_vec))
    b = float(np.linalg.norm(b_vec))
    cosg = np.clip(np.dot(a_vec, b_vec) / max(a * b, 1e-8), -1.0, 1.0)
    gamma = float(np.degrees(np.arccos(cosg)))
    return {"a": a, "b": b, "gamma": gamma, "a_vec": a_vec, "b_vec": b_vec}

def _auto_max_xy_repeat(atoms_a, atoms_b) -> int:
    """Conservative automatic repeat cap for fast interface search."""
    n = len(atoms_a) + len(atoms_b)
    if n <= 24:
        return 4
    if n <= 80:
        return 3
    return 2

def _find_best_xy_repeat_pair(atoms_a, atoms_b, max_rep: int = 4, max_strain_each: float = 0.08):
    mA = _inplane_metrics(atoms_a)
    mB = _inplane_metrics(atoms_b)
    best = None
    for na in range(1, max_rep + 1):
        for ma in range(1, max_rep + 1):
            a_len = mA["a"] * na
            b_len = mA["b"] * ma
            area_a = a_len * b_len * np.sin(np.radians(mA["gamma"]))
            for nb in range(1, max_rep + 1):
                for mb in range(1, max_rep + 1):
                    c_len = mB["a"] * nb
                    d_len = mB["b"] * mb
                    area_b = c_len * d_len * np.sin(np.radians(mB["gamma"]))
                    mismatch_a = abs(a_len - c_len) / max(a_len, c_len, 1e-8)
                    mismatch_b = abs(b_len - d_len) / max(b_len, d_len, 1e-8)
                    gamma_pen = abs(mA["gamma"] - mB["gamma"]) / 180.0
                    area_pen = abs(area_a - area_b) / max(area_a, area_b, 1e-8)
                    score = mismatch_a + mismatch_b + gamma_pen + 0.5 * area_pen
                    item = {
                        "repeat_a": (na, ma),
                        "repeat_b": (nb, mb),
                        "mismatch_a": float(mismatch_a),
                        "mismatch_b": float(mismatch_b),
                        "gamma_pen": float(gamma_pen),
                        "area_pen": float(area_pen),
                        "score": float(score),
                    }
                    if max(mismatch_a, mismatch_b) > max_strain_each:
                        continue
                    if best is None or item["score"] < best["score"]:
                        best = item
    if best is None:
        # Fallback: ignore max strain ceiling and pick the least-bad repeat
        for na in range(1, max_rep + 1):
            for ma in range(1, max_rep + 1):
                a_len = mA["a"] * na
                b_len = mA["b"] * ma
                area_a = a_len * b_len * np.sin(np.radians(mA["gamma"]))
                for nb in range(1, max_rep + 1):
                    for mb in range(1, max_rep + 1):
                        c_len = mB["a"] * nb
                        d_len = mB["b"] * mb
                        area_b = c_len * d_len * np.sin(np.radians(mB["gamma"]))
                        mismatch_a = abs(a_len - c_len) / max(a_len, c_len, 1e-8)
                        mismatch_b = abs(b_len - d_len) / max(b_len, d_len, 1e-8)
                        gamma_pen = abs(mA["gamma"] - mB["gamma"]) / 180.0
                        area_pen = abs(area_a - area_b) / max(area_a, area_b, 1e-8)
                        score = mismatch_a + mismatch_b + gamma_pen + 0.5 * area_pen
                        item = {
                            "repeat_a": (na, ma),
                            "repeat_b": (nb, mb),
                            "mismatch_a": float(mismatch_a),
                            "mismatch_b": float(mismatch_b),
                            "gamma_pen": float(gamma_pen),
                            "area_pen": float(area_pen),
                            "score": float(score),
                        }
                        if best is None or item["score"] < best["score"]:
                            best = item
    return best

def _scale_film_to_substrate_xy(film_atoms, substrate_atoms):
    film = film_atoms.copy()
    sub_cell = substrate_atoms.get_cell().copy()
    film_cell = film.get_cell().copy()
    new_cell = film_cell.copy()
    new_cell[0] = sub_cell[0]
    new_cell[1] = sub_cell[1]
    film.set_cell(new_cell, scale_atoms=True)
    return film

def _auto_initial_gap(substrate_atoms, film_atoms) -> float:
    fam_s = _infer_interface_surface_family(substrate_atoms)
    fam_f = _infer_interface_surface_family(film_atoms)
    if fam_s != "metal" and fam_f != "metal":
        return 2.2
    if fam_s == "metal" and fam_f == "metal":
        return 2.4
    return 2.3

def _stack_interface_pair(substrate_atoms, film_atoms, gap: float = 2.5, shift_frac=(0.0, 0.0), pbc_z: bool = True):
    sub = substrate_atoms.copy()
    film = film_atoms.copy()
    cell = sub.get_cell().copy()
    shift_vec = np.array(cell[0]) * float(shift_frac[0]) + np.array(cell[1]) * float(shift_frac[1])

    pos_s = sub.get_positions()
    pos_s[:, 2] -= float(np.min(pos_s[:, 2]))
    sub.set_positions(pos_s)
    top_s = float(np.max(pos_s[:, 2]))

    pos_f = film.get_positions()
    pos_f[:, 2] -= float(np.min(pos_f[:, 2]))
    pos_f += shift_vec
    pos_f[:, 2] += top_s + float(gap)
    film.set_positions(pos_f)

    combo = sub + film
    max_z = float(np.max(combo.get_positions()[:, 2]))
    new_cell = cell.copy()
    if abs(new_cell[2, 0]) < 1e-6 and abs(new_cell[2, 1]) < 1e-6:
        new_cell[2] = np.array([0.0, 0.0, max(max_z + 8.0, float(cell.lengths()[2]))])
    combo.set_cell(new_cell)
    combo.set_pbc([True, True, bool(pbc_z)])
    combo = _recenter_slab_z_into_cell(combo, margin=1.0)
    return combo

def _registry_candidates_auto(n_random: int = 2, seed: int = GLOBAL_SEED):
    """Small registry set for fast screening; CHGNet is called only on a few candidates."""
    rng = np.random.default_rng(int(seed))
    regs = [
        ("centered", (0.0, 0.0)),
        ("shift_xy_half", (0.5, 0.5)),
    ]
    seen = {(round(sx, 3), round(sy, 3)) for _name, (sx, sy) in regs}
    for i in range(int(n_random)):
        sx = float(rng.uniform(0.0, 1.0))
        sy = float(rng.uniform(0.0, 1.0))
        key = (round(sx, 3), round(sy, 3))
        if key in seen:
            continue
        seen.add(key)
        regs.append((f"rand_{i+1}", (sx, sy)))
    return regs

def _geometry_prefilter_interface_work(work, keep_n: int = 8, atom_cap: int = 260):
    """Fast prefilter before CHGNet.

    Ranks interface candidates using only cheap geometric/QC signals and keeps a small subset
    for expensive CHGNet pre-relaxation.
    """
    scored = []
    for combo, meta in (work or []):
        rep = validate_structure(combo, target_area=70.0)
        ng = getattr(rep, "nearest_global", {}) or {}
        dmin = float(ng.get("d_min", np.nan)) if ng else float("nan")
        issues = getattr(rep, "issues", []) or []
        n_issues = len(issues)
        n_atoms = int(meta.get("n_atoms", len(combo)))
        vac_z = float(getattr(rep, "vacuum_z", meta.get("vacuum_z", np.nan)))
        strain = float(meta.get("chosen_strain", 1e9))
        area_pen = float(meta.get("area_pen", 1e9))
        mismatch = float(meta.get("xy_mismatch_a", 1e9)) + float(meta.get("xy_mismatch_b", 1e9))
        vac_pen = 0.0 if np.isnan(vac_z) else max(0.0, 12.0 - vac_z)
        # Penalties: too many atoms, very small dmin, many issues, large strain/mismatch.
        atom_pen = max(0, n_atoms - int(atom_cap)) * 0.01
        dmin_pen = 10.0 if (np.isfinite(dmin) and dmin < 0.65) else (2.0 if (np.isfinite(dmin) and dmin < 0.80) else 0.0)
        score = (
            5.0 * float(n_issues)
            + 20.0 * float(strain)
            + 8.0 * float(mismatch)
            + 2.0 * float(area_pen)
            + 3.0 * float(vac_pen)
            + float(atom_pen)
            + float(dmin_pen)
        )
        meta2 = dict(meta)
        meta2["geom_prefilter_score"] = float(score)
        meta2["geom_prefilter_dmin"] = None if not np.isfinite(dmin) else float(dmin)
        scored.append((combo, meta2))
    scored.sort(key=lambda x: (x[1].get("geom_prefilter_score", 1e99), len(x[1].get("issues", [])), x[1].get("n_atoms", 10**9)))
    return scored[: max(1, int(keep_n))]

def _select_best_slab_for_facet(bulk_atoms, miller, thickness):
    cands, meta = slabify_from_bulk(bulk_atoms, miller=miller, min_slab_size=float(thickness), min_vacuum_size=12.0, max_candidates=6)
    fam = _infer_interface_surface_family(bulk_atoms)
    rank_mode = "oxide" if fam != "metal" else "metal"
    return _pick_best_slab_candidate_auto(cands, meta, rank_mode)

def _build_interface_candidates_from_bulks(
    bulk_a, bulk_b,
    miller_a=(1,1,1), miller_b=(1,1,1),
    allocation="6:4",
    substrate_choice="Auto decide by lower strain",
    max_xy_repeat: int | None = None,
    z_gap: float | None = None,
    prerelax: bool = True,
    prerelax_top_n: int = 3,
    geom_prefilter_keep: int = 8,
    registry_random_n: int = 2,
    prerelax_max_steps: int = 80,
):
    t_a, t_b = _allocation_to_thicknesses(allocation)

    if t_a == 0.0 and t_b == 0.0:
        raise ValueError("Both interface fractions are zero.")

    slab_a = meta_a = None
    slab_b = meta_b = None
    if t_a > 0:
        slab_a, meta_a = _select_best_slab_for_facet(bulk_a, miller_a, t_a)
    if t_b > 0:
        slab_b, meta_b = _select_best_slab_for_facet(bulk_b, miller_b, t_b)

    if slab_a is None or slab_b is None:
        base = (slab_a if slab_a is not None else slab_b).copy()
        label = "A_only_baseline" if slab_a is not None else "B_only_baseline"
        rep = validate_structure(base, target_area=70.0)
        return [base], [{
            "label": label,
            "kind": "single_slab_baseline",
            "miller_a": tuple(miller_a),
            "miller_b": tuple(miller_b),
            "allocation": allocation,
            "n_atoms": len(base),
            "formula": base.get_chemical_formula(),
            "vacuum_z": float(getattr(rep, "vacuum_z", np.nan)),
            "issues": getattr(rep, "issues", []),
        }]

    auto_rep = int(max_xy_repeat) if max_xy_repeat is not None else _auto_max_xy_repeat(slab_a, slab_b)
    match = _find_best_xy_repeat_pair(slab_a, slab_b, max_rep=auto_rep)
    a_rep = repeat_xy(slab_a, *match["repeat_a"])
    b_rep = repeat_xy(slab_b, *match["repeat_b"])

    mA = _inplane_metrics(a_rep)
    mB = _inplane_metrics(b_rep)
    strain_if_a_sub = abs(mA["a"] - mB["a"]) / max(mA["a"], 1e-8) + abs(mA["b"] - mB["b"]) / max(mA["b"], 1e-8)
    strain_if_b_sub = abs(mA["a"] - mB["a"]) / max(mB["a"], 1e-8) + abs(mA["b"] - mB["b"]) / max(mB["b"], 1e-8)

    if substrate_choice == "Auto decide by lower strain":
        a_sub = strain_if_a_sub <= strain_if_b_sub
    elif substrate_choice == "Structure A as substrate":
        a_sub = True
    else:
        a_sub = False

    if a_sub:
        substrate = a_rep.copy()
        film = _scale_film_to_substrate_xy(b_rep, substrate)
        substrate_name, film_name = "A", "B"
        chosen_strain = float(strain_if_a_sub)
    else:
        substrate = b_rep.copy()
        film = _scale_film_to_substrate_xy(a_rep, substrate)
        substrate_name, film_name = "B", "A"
        chosen_strain = float(strain_if_b_sub)

    gap_used = float(z_gap) if z_gap is not None else _auto_initial_gap(substrate, film)
    reg_grid = _registry_candidates_auto(n_random=int(registry_random_n), seed=GLOBAL_SEED)

    work = []
    for reg_name, shift in reg_grid:
        combo = _stack_interface_pair(substrate, film, gap=gap_used, shift_frac=shift, pbc_z=True)
        rep = validate_structure(combo, target_area=70.0)
        meta = {
            "label": f"{film_name}_on_{substrate_name}_{reg_name}",
            "kind": "interface",
            "allocation": allocation,
            "miller_a": tuple(miller_a),
            "miller_b": tuple(miller_b),
            "repeat_a": tuple(match["repeat_a"]),
            "repeat_b": tuple(match["repeat_b"]),
            "substrate": substrate_name,
            "film": film_name,
            "registry": reg_name,
            "xy_mismatch_a": float(match["mismatch_a"]),
            "xy_mismatch_b": float(match["mismatch_b"]),
            "area_pen": float(match.get("area_pen", 0.0)),
            "chosen_strain": chosen_strain,
            "max_xy_repeat_auto": int(auto_rep),
            "gap_auto": gap_used,
            "n_atoms": len(combo),
            "formula": combo.get_chemical_formula(),
            "vacuum_z": float(getattr(rep, "vacuum_z", np.nan)),
            "issues": getattr(rep, "issues", []),
        }
        work.append((combo, meta))

    # Fast geometry prefilter before expensive CHGNet
    work = _geometry_prefilter_interface_work(work, keep_n=int(geom_prefilter_keep), atom_cap=260)

    if prerelax and HAS_ADSORML:
        ranked = []
        for combo, meta in work:
            try:
                combo2, rmeta = relax_slab_chgnet(
                    combo,
                    surfactant_class="none",
                    top_z_tol=2.5,
                    jiggle_amp=0.03,
                    seed=GLOBAL_SEED,
                    fmax=0.06,
                    max_steps=int(prerelax_max_steps),
                    device="auto",
                )
                combo2 = _recenter_slab_z_into_cell(combo2, margin=1.0)
                meta2 = dict(meta)
                meta2["prerelax_E_total"] = rmeta.get("E_total", None)
                meta2["prerelax_converged"] = rmeta.get("converged", None)
                ranked.append((combo2, meta2))
            except Exception as e:
                meta2 = dict(meta)
                meta2["prerelax_error"] = str(e)
                ranked.append((combo, meta2))
        ranked.sort(key=lambda x: (
            x[1].get("prerelax_E_total") is None,
            x[1].get("prerelax_E_total", 1e99),
            x[1].get("geom_prefilter_score", 1e99),
            len(x[1].get("issues", [])),
        ))
        work = ranked[: max(1, int(prerelax_top_n))]
    else:
        work = work[: max(1, int(prerelax_top_n))]

    cand_atoms = [a for a, _m in work]
    cand_meta = [m for _a, m in work]
    return cand_atoms, cand_meta

