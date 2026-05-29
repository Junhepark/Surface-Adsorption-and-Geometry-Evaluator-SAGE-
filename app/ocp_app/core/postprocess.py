import numpy as np
import pandas as pd

def split_reliable_unreliable(df, dE_thresh=3.0, disp_thresh=1.0):
    if df is None or df.empty:
        return df, df

    mask = pd.Series(False, index=df.index)

    if "ΔE_H_user (eV)" in df.columns:
        mask |= df["ΔE_H_user (eV)"].abs() > dE_thresh
    if "ΔE_ads_user (eV)" in df.columns:
        mask |= df["ΔE_ads_user (eV)"].abs() > dE_thresh

    if "H_lateral_disp(Å)" in df.columns:
        mask |= df["H_lateral_disp(Å)"] > disp_thresh
    if "ads_lateral_disp(Å)" in df.columns:
        mask |= df["ads_lateral_disp(Å)"] > disp_thresh

    return df[~mask].copy(), df[mask].copy()

def _normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()



def _string_series(df: pd.DataFrame, col: str, default: str = "unknown") -> pd.Series:
    if col in df.columns:
        s = df[col].copy()
        s = s.where(~s.isna(), default)
        return s.astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def _normalize_site_name_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.lower()
    alias = {
        "top": "ontop",
        "on_top": "ontop",
        "on-top": "ontop",
        "o_top": "o_top",
        "o-top": "o_top",
        "hollow": "hollow",
        "hcp": "hcp",
        "fcc": "fcc",
        "bridge": "bridge",
        "br": "bridge",
        "oer": "oer_cation",
        "cation": "oer_cation",
        "oer_cation": "oer_cation",
        "unknown": "unknown",
        "nan": "unknown",
        "none": "unknown",
        "": "unknown",
    }
    return out.map(lambda x: alias.get(x, x))


def _requested_site_from_label(df: pd.DataFrame) -> pd.Series:
    if "requested_site" in df.columns:
        return _normalize_site_name_series(_string_series(df, "requested_site", default="unknown"))
    if "site_label" in df.columns:
        raw = _string_series(df, "site_label", default="unknown")
        raw_s = raw.astype(str).str.strip().str.lower()
        # OER oxide labels are intentionally prefixed with oer_cation_... .
        # Splitting on the first underscore would produce requested_site='oer',
        # which creates a false placement_mismatch against initial_geom_site='oer_cation'.
        out = pd.Series("unknown", index=df.index, dtype="object")
        mask_oer = raw_s.str.startswith("oer_cation") | raw_s.str.startswith("cation_")
        out.loc[mask_oer] = "oer_cation"
        mask_rest = ~mask_oer
        out.loc[mask_rest] = raw_s.loc[mask_rest].str.split("_").str[0]
        return _normalize_site_name_series(out)
    if "site" in df.columns:
        return _normalize_site_name_series(_string_series(df, "site", default="unknown"))
    return pd.Series(["unknown"] * len(df), index=df.index, dtype="object")


def annotate_site_transitions(df: pd.DataFrame, disp_thresh: float = 0.8) -> pd.DataFrame:
    """Attach site-transition metadata for post-relaxation migration analysis.

    Three-stage model:
      requested_site -> initial_geom_site -> relaxed_site

    Definitions:
      - requested_site: user/requested seed site (best-effort from explicit column or site_label)
      - initial_geom_site: geometric site assigned to the initial placed structure (fallback: site)
      - relaxed_site: geometric site assigned after relaxation

    Notes:
      - migration_path is USER-REQUESTED path (requested -> final), because this is what users
        usually want to audit in the UI.
      - actual_migration_path is GEOMETRIC path (initial_geom -> final), which separates true
        relaxation-induced migration from seed-placement mismatch.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    requested_site = _requested_site_from_label(out)
    initial_geom_site = _normalize_site_name_series(
        _string_series(out, "initial_geom_site", default=("unknown" if "site" not in out.columns else ""))
    )
    if "site" in out.columns:
        site_series = _normalize_site_name_series(_string_series(out, "site", default="unknown"))
        bad_init = initial_geom_site.isin(["", "unknown"])
        initial_geom_site.loc[bad_init] = site_series.loc[bad_init]
    else:
        initial_geom_site = initial_geom_site.where(~initial_geom_site.isin(["", "unknown"]), requested_site)

    final_candidates = [
        "relaxed_site",
        "site_final",
        "final_site",
        "final_label",
        "relaxed_site_kind",
        "site_kind_final",
    ]
    final_col = next((c for c in final_candidates if c in out.columns), None)
    if final_col is None:
        relaxed_site = initial_geom_site.copy()
        out["relaxed_site"] = relaxed_site
    else:
        relaxed_site = _normalize_site_name_series(_string_series(out, final_col, default="unknown"))
        if "relaxed_site" not in out.columns:
            out["relaxed_site"] = relaxed_site
        else:
            out["relaxed_site"] = _normalize_site_name_series(_string_series(out, "relaxed_site", default="unknown"))
            mask_bad = out["relaxed_site"].isin(["", "unknown"])
            out.loc[mask_bad, "relaxed_site"] = relaxed_site.loc[mask_bad]
            relaxed_site = _normalize_site_name_series(out["relaxed_site"])

    # OER oxide taxonomy: cation/oer_cation naming differences are not
    # physical migration. Normalize the requested label to the OER site family
    # when the initial/final classifiers identify an oer_cation basin.
    try:
        oer_like = initial_geom_site.eq("oer_cation") | relaxed_site.eq("oer_cation")
        requested_site.loc[oer_like & requested_site.isin(["oer", "cation", "unknown", ""]) ] = "oer_cation"
    except Exception:
        pass

    # Invalid/final-unknown handling
    final_valid = ~relaxed_site.isin(["", "unknown", "nan", "none"])

    placement_mismatch = final_valid & (requested_site != initial_geom_site)
    migrated_actual = final_valid & (initial_geom_site != relaxed_site)
    migrated_requested = final_valid & (requested_site != relaxed_site)

    migrated = migrated_requested | migrated_actual

    disp_col = None
    for c in ["H_lateral_disp(Å)", "ads_lateral_disp(Å)"]:
        if c in out.columns:
            disp_col = c
            break
    if disp_col is not None:
        disp = pd.to_numeric(out[disp_col], errors="coerce").fillna(0.0)
        migrated = migrated | (disp > float(disp_thresh))

    if "qa" in out.columns:
        qa = _normalize_text_series(out["qa"])
        migrated = migrated | qa.eq("migrated")
        invalid = qa.isin(["desorbed", "broken", "crashed", "invalid"])
    else:
        invalid = pd.Series(False, index=out.index)

    energy_valid = ~invalid

    out["requested_site"] = requested_site
    out["initial_geom_site"] = initial_geom_site
    out["site_initial"] = initial_geom_site
    out["site_final"] = relaxed_site
    out["placement_mismatch"] = placement_mismatch
    out["migrated_actual"] = migrated_actual
    out["migrated_requested"] = migrated_requested
    out["migrated"] = migrated
    out["energy_valid"] = energy_valid
    out["migration_destination"] = np.where(migrated, relaxed_site, requested_site)
    out["migration_path"] = np.where(
        migrated_requested,
        requested_site.astype(str) + " -> " + relaxed_site.astype(str),
        requested_site.astype(str) + " -> " + requested_site.astype(str),
    )
    out["requested_to_final_path"] = out["migration_path"]
    out["actual_migration_path"] = np.where(
        migrated_actual,
        initial_geom_site.astype(str) + " -> " + relaxed_site.astype(str),
        initial_geom_site.astype(str) + " -> " + initial_geom_site.astype(str),
    )

    def _transition_type(row):
        if not bool(row.get("energy_valid", True)):
            return "invalid"
        pm = bool(row.get("placement_mismatch", False))
        ma = bool(row.get("migrated_actual", False))
        mr = bool(row.get("migrated_requested", False))
        if pm and ma:
            return "placement_mismatch+actual_migration"
        if pm and mr:
            return "placement_mismatch_only"
        if ma:
            return "actual_migration"
        if mr:
            return "requested_site_changed"
        return "stable"

    out["site_transition_type"] = out.apply(_transition_type, axis=1)
    return out



def summarize_site_transitions(df: pd.DataFrame) -> dict:
    if df is None or df.empty or ("migration_path" not in df.columns):
        return {"n_migrated": 0, "paths": []}

    tmp = df.copy()
    if "migrated" in tmp.columns:
        mig_mask = pd.to_numeric(tmp["migrated"], errors="coerce").fillna(0).astype(bool)
        tmp = tmp.loc[mig_mask].copy()
    if tmp.empty:
        return {"n_migrated": 0, "paths": []}

    group_cols = [
        c for c in [
            "adsorbate",
            "requested_site",
            "initial_geom_site",
            "relaxed_site",
            "migration_destination",
            "migration_path",
            "actual_migration_path",
            "site_transition_type",
        ] if c in tmp.columns
    ]
    if not group_cols:
        return {"n_migrated": int(len(tmp)), "paths": []}

    counts = tmp.groupby(group_cols, dropna=False).size().reset_index(name="count")
    counts = counts.sort_values(["count"] + group_cols, ascending=[False] + [True] * len(group_cols)).reset_index(drop=True)
    return {"n_migrated": int(len(tmp)), "paths": counts.to_dict(orient="records")}


def co2rr_apply_qa_policy(df: pd.DataFrame, disp_thresh: float = 0.8) -> pd.DataFrame:
    """Ensure CO2RR QA-related columns exist and are normalized.

    Policy:
      - 'qa' is the authoritative filter key when present (ok/migrated kept; others rejected).
      - If 'qa' is missing, infer a conservative 'qa' from displacement and energy blow-ups.
      - 'migrated' is treated as metadata (NOT an automatic reject) once 'qa' exists.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Normalize / infer qa
    if "qa" in out.columns:
        out["qa"] = _normalize_text_series(out["qa"])
    else:
        qa = pd.Series("ok", index=out.index, dtype="object")

        # Energy blow-ups -> crashed
        ecols = [c for c in ["ΔE_ads_user (eV)", "ΔE_ads_user", "ΔG_ads (eV)", "ΔG_ads"] if c in out.columns]
        if "ΔE_ads_user (eV)" in out.columns:
            e = pd.to_numeric(out["ΔE_ads_user (eV)"], errors="coerce")
            qa[(~np.isfinite(e)) | (e.abs() > 50.0)] = "crashed"

        # Large lateral displacement -> migrated (metadata; still keep unless also crashed)
        if "ads_lateral_disp(Å)" in out.columns:
            disp = pd.to_numeric(out["ads_lateral_disp(Å)"], errors="coerce").fillna(0.0)
            qa[(disp > float(disp_thresh)) & (qa == "ok")] = "migrated"

        out["qa"] = qa

    # Infer migrated if missing (for UI display only)
    if "migrated" not in out.columns:
        if "ads_lateral_disp(Å)" in out.columns:
            disp = pd.to_numeric(out["ads_lateral_disp(Å)"], errors="coerce").fillna(0.0)
            out["migrated"] = disp > float(disp_thresh)
        else:
            out["migrated"] = False

    # Ensure relaxed_site exists (fallback to 'site' if absent)
    if "relaxed_site" not in out.columns:
        if "site" in out.columns:
            out["relaxed_site"] = out["site"]
        else:
            out["relaxed_site"] = "unknown"

    return out

def co2rr_split_by_qa(df: pd.DataFrame):
    """Split CO2RR results into candidates vs rejected, using qa as the only reject criterion."""
    if df is None or df.empty:
        return df, df

    qa = _normalize_text_series(df["qa"]) if "qa" in df.columns else pd.Series("ok", index=df.index)
    keep_mask = qa.isin(["ok", "migrated"])
    return df[keep_mask].copy(), df[~keep_mask].copy()


def oxygen_apply_qa_policy(df: pd.DataFrame, disp_thresh: float = 0.8) -> pd.DataFrame:
    """QA policy for OER oxygen-intermediate rows.

    Unlike CO2RR, a plain `migrated` oxygen intermediate is not automatically
    accepted.  It is accepted only when the engine marked it as a bound oxygen
    adsorbate (`valid_for_oer_summary=True`), in which case the normalized
    status is `bound_relaxed` when the OER cation site remains stable, or `bound_migrated` when a genuine site change is detected.
    """
    if df is None or df.empty:
        return df

    out = co2rr_apply_qa_policy(df, disp_thresh=disp_thresh).copy()
    qa = _normalize_text_series(out["qa"]) if "qa" in out.columns else pd.Series("ok", index=out.index)

    if "valid_for_oer_summary" in out.columns:
        valid = out["valid_for_oer_summary"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        valid = qa.eq("ok")

    # Channel-level hard rejects for AEM OER summaries.
    if "surface_channel" in out.columns:
        bad_ch = {
            "protonated_lattice_oxygen",
            "anion_o",
            "lattice_protonation_disabled_for_aem",
            "fallback_site",
        }
        ch = out["surface_channel"].astype(str).str.strip().str.lower()
        qa.loc[ch.isin(bad_ch)] = "invalid_oer_channel"
        valid.loc[ch.isin(bad_ch)] = False

    # Detached/broken/crashed states remain rejected.
    for col in ("oer_state_class", "qa"):
        if col in out.columns:
            s = out[col].astype(str).str.lower()
            bad = s.str.contains("detached|broken|crashed|invalid|fallback", regex=True)
            valid.loc[bad] = False
            qa.loc[bad & qa.isin(["ok", "migrated", "bound_migrated"])] = s.loc[bad]

    mig = out["migrated"].astype(bool) if "migrated" in out.columns else pd.Series(False, index=out.index)

    # v16 OER cation-bound label refinement.  If an oxygen intermediate remains
    # cation-bound and site tracking says the oer_cation basin is stable, do not
    # treat lateral relaxation alone as physical migration.
    def _bool_col(name, default=False):
        if name not in out.columns:
            return pd.Series([default] * len(out), index=out.index)
        return out[name].astype(str).str.lower().isin(["true", "1", "yes"])

    stable_oer_cation = pd.Series(False, index=out.index)
    if len(out) > 0:
        oxygen_bound = _bool_col("oxygen_bound_to_cation", default=False)
        stable_transition = (
            out["site_transition_type"].astype(str).str.lower().eq("stable")
            if "site_transition_type" in out.columns else pd.Series(False, index=out.index)
        )
        actual_migration = _bool_col("migrated_actual", default=False)
        site_family_ok = (
            out["site_family"].astype(str).str.lower().eq("oxide_oer_cation")
            if "site_family" in out.columns else pd.Series(True, index=out.index)
        )
        init_ok = (
            out["initial_geom_site"].astype(str).str.lower().eq("oer_cation")
            if "initial_geom_site" in out.columns else pd.Series(True, index=out.index)
        )
        final_ok = (
            out["relaxed_site"].astype(str).str.lower().eq("oer_cation")
            if "relaxed_site" in out.columns else pd.Series(True, index=out.index)
        )
        stable_oer_cation = valid & oxygen_bound & stable_transition & (~actual_migration) & site_family_ok & init_ok & final_ok

    qa.loc[stable_oer_cation] = "bound_relaxed"
    mig.loc[stable_oer_cation] = False
    qa.loc[(~stable_oer_cation) & mig & valid] = "bound_migrated"
    qa.loc[mig & (~valid) & qa.eq("migrated")] = "migrated_rejected"

    out["migrated"] = mig
    out["qa"] = qa
    out["valid_for_oer_summary"] = valid
    return out


def oxygen_split_by_qa(df: pd.DataFrame):
    """Split OER rows into valid candidates and rejected attempts."""
    if df is None or df.empty:
        return df, df
    out = oxygen_apply_qa_policy(df)
    qa = _normalize_text_series(out["qa"]) if "qa" in out.columns else pd.Series("ok", index=out.index)
    keep = qa.isin(["ok", "bound_relaxed", "bound_migrated"])
    if "valid_for_oer_summary" in out.columns:
        valid = out["valid_for_oer_summary"].astype(str).str.lower().isin(["true", "1", "yes"])
        keep &= valid
    return out[keep].copy(), out[~keep].copy()


def co2rr_dedupe_candidates(df_keep: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate CO2RR candidates by (adsorbate, relaxed_site) keeping the best (lowest) energy row."""
    if df_keep is None or df_keep.empty:
        return df_keep

    # Choose ranking column
    rank_cols = ["ΔG_ads (eV)", "ΔE_ads_user (eV)", "ΔG_ads", "ΔE_ads_user"]
    rank_col = next((c for c in rank_cols if c in df_keep.columns), None)
    if rank_col is None:
        return df_keep

    # Coerce to numeric for ranking
    tmp = df_keep.copy()
    tmp[rank_col] = pd.to_numeric(tmp[rank_col], errors="coerce")

    key_cols = []
    if "adsorbate" in tmp.columns:
        key_cols.append("adsorbate")
    if "relaxed_site" in tmp.columns:
        key_cols.append("relaxed_site")
    elif "site" in tmp.columns:
        key_cols.append("site")

    if len(key_cols) < 2:
        return tmp

    idx = tmp.groupby(key_cols, dropna=False)[rank_col].idxmin()
    dedup = tmp.loc[idx].copy()

    # Mark duplicates vs representative
    dedup["is_representative"] = True
    tmp["is_representative"] = tmp.index.isin(dedup.index)

    # Preserve original ordering roughly by adsorbate then energy
    dedup = dedup.sort_values(by=key_cols + [rank_col], kind="mergesort").reset_index(drop=True)
    return dedup

def build_compact_table(df, mode: str):
    if df is None or df.empty:
        return df

    if mode == "HER":
        cols = [
            "site_label",
            "requested_site",
            "initial_geom_site",
            "relaxed_site",
            "placement_mismatch",
            "migrated_actual",
            "migrated",
            "migration_destination",
            "migration_path",
            "actual_migration_path",
            "site_transition_type",
            "ΔG_H(U,pH) (eV)",
            "ΔG_H (eV)",
            "ΔE_H_user (eV)",
            "H_lateral_disp(Å)",
            "is_duplicate",
            "reliability",
        ]
    else:
        # CO2RR (and other adsorbate modes)
        cols = [
            "site_label",
            "requested_site",
            "initial_geom_site",
            "relaxed_site",
            "placement_mismatch",
            "migrated_actual",
            "migrated",
            "migration_destination",
            "migration_path",
            "actual_migration_path",
            "site_transition_type",
            "oxide_seed_mode",
            "surface_channel",
            "oer_base_site_label",
            "oer_start_height_A",
            "adsorbate",
            "qa",
            "oer_state_class",
            "oer_state_note",
            "valid_for_oer_summary",
            "oxygen_ads_bound",
            "oxygen_bound_to_cation",
            "oxygen_nearest_slab_symbol",
            "oxygen_anchor_slab_dist(Å)",
            "oxygen_anchor_cation_dist(Å)",
            "oxygen_anchor_target_cation_dist(Å)",
            "oxygen_anchor_target_cation_index",
            "oxygen_anchor_target_cation_symbol",
            "oxygen_anchor_anion_dist(Å)",
            "oxygen_internal_bond_ok",
            "oxygen_surface_bond_ok",
            "ΔG_ads (eV)",
            "ΔE_ads_user (eV)",
            "ΔE_raw(slab+ads - slab) (eV)",
            "E_ref_reagents (eV)",
            "G_correction (eV)",
            "ref_rxn",
            "ads_lateral_disp(Å)",
            "ads_relax_elapsed_s",
            "ads_relax_n_steps",
            "ads_relax_converged",
            "is_duplicate",
            "reliability",
        ]
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols]

def _make_ml_screen_key(sig, mtype, reaction_mode, co2_ads, preset, top_k, geom_per_kind, probe_level, adv_settings: dict, surfactant_class: str = "none", surfactant_prerelax_slab: bool = False):
    return (
        sig,
        str(mtype),
        str(reaction_mode),
        tuple(co2_ads or []),
        str(preset),
        int(top_k),
        int(geom_per_kind),
        str(probe_level),
        str(surfactant_class),
        bool(surfactant_prerelax_slab),
        tuple(sorted((adv_settings or {}).items())),
    )

def _build_ml_compact_df(union_items, union_labels):
    rows = []
    union_items = union_items or []
    union_labels = union_labels or [f"ML_{i}" for i in range(len(union_items))]
    for lbl, r in zip(union_labels, union_items):
        rows.append({
            "label": lbl,
            "adsorbate": getattr(r, "adsorbate", "?"),
            "kind": getattr(r, "kind", "?"),
            "status": "selected",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

