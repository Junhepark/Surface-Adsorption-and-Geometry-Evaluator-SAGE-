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


def _boolish_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series([bool(default)] * len(df), index=df.index)
    vals = df[col].astype(str).str.strip().str.lower()
    return vals.isin(["true", "1", "yes", "y", "valid", "ok"])


def _voc_ech_row_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(False, index=df.index)
    if "ech_state" in df.columns:
        mask |= _boolish_series(df, "ech_state", default=False)
    if "ech_seed_policy" in df.columns:
        sp = df["ech_seed_policy"].astype(str).str.strip().str.lower()
        mask |= sp.isin(["outer_h", "near_carbonyl_h", "adjacent_h_pair"])
    if "descriptor_state" in df.columns:
        ds = df["descriptor_state"].astype(str).str.replace(" ", "", regex=False).str.upper()
        mask |= ds.isin(["H*+CH3CHO*", "H*+H*"])
    return mask


def _voc_metal_her_h_mask(df: pd.DataFrame) -> pd.Series:
    """Rows where VOC H* is intentionally evaluated as HER/CHE-like metal H*.

    These rows do not contain a molecular adsorbate heavy atom, so generic VOC
    heavy-atom / reactive-group / C-series QA must never demote them into the
    rejected bucket.  Detection is defensive because cached CSVs may have only
    part of the corrected metadata.
    """
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    idx = df.index
    ads = pd.Series("", index=idx, dtype="object")
    if "adsorbate" in df.columns:
        ads = df["adsorbate"].astype(str)
    elif "descriptor_state" in df.columns:
        ads = df["descriptor_state"].astype(str)
    ads_norm = ads.str.replace(" ", "", regex=False).str.upper()
    h_only = ads_norm.isin(["H", "H*"])

    src = _string_series(df, "H_descriptor_source", default="").str.strip().str.lower() if "H_descriptor_source" in df.columns else pd.Series("", index=idx)
    pol = _string_series(df, "H_placement_policy", default="").str.strip().str.lower() if "H_placement_policy" in df.columns else pd.Series("", index=idx)
    red = _string_series(df, "reduction_h_placement", default="").str.strip().str.lower() if "reduction_h_placement" in df.columns else pd.Series("", index=idx)
    anchor = _string_series(df, "anchor_mode", default="").str.strip().str.lower() if "anchor_mode" in df.columns else pd.Series("", index=idx)

    has_metal_her_meta = (
        src.str.contains("metal_che_her", regex=False)
        | pol.str.contains("che_mode.site_energy_two_stage", regex=False)
        | red.str.contains("metal_che_her", regex=False)
        | anchor.str.contains("metal_che_her", regex=False)
    )

    finite_che = pd.Series(False, index=idx)
    for col in ["ΔG_H_CHE (eV)", "ΔG_H_CHE_like (eV)", "ΔE_H_user (eV)"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            finite_che |= np.isfinite(vals) & (vals.abs() < 10.0)

    return h_only & (has_metal_her_meta | finite_che)


def _voc_h_only_mask(df: pd.DataFrame) -> pd.Series:
    """Any standalone H* row in VOC mode.

    H* has no molecular heavy atom or C/O reactive group.  It must be checked as
    a single-atom H descriptor, not by VOC molecular pose rules.
    """
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    if "adsorbate" in df.columns:
        ads = df["adsorbate"].astype(str)
    elif "descriptor_state" in df.columns:
        ads = df["descriptor_state"].astype(str)
    else:
        return pd.Series(False, index=df.index)
    ads_norm = ads.str.replace(" ", "", regex=False).str.upper()
    return ads_norm.isin(["H", "H*"])


def _voc_h_has_finite_energy(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    finite = pd.Series(False, index=df.index)
    for col in ["ΔG_H_CHE (eV)", "ΔG_H_CHE_like (eV)", "ΔE_H_user (eV)", "ΔE_proxy (eV)", "ΔE_ads_user (eV)"]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            finite |= np.isfinite(v) & (v.abs() < 10.0)
    return finite



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
        "cation_top": "cation_top",
        "anion_top": "anion_top",
        "cation_cation_bridge": "cation_cation_bridge",
        "cation_oxygen_bridge": "cation_oxygen_bridge",
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
        # Reclassify obviously detached rows even if an older backend labeled
        # them only as migrated.  This protects old cached/result CSVs.
        if "min_ads_slab_dist(Å)" in out.columns:
            dmin = pd.to_numeric(out["min_ads_slab_dist(Å)"], errors="coerce")
            out.loc[dmin < 0.90, "qa"] = "invalid"
            out.loc[dmin > 2.75, "qa"] = "desorbed"
        if "reactive_ads_distance(Å)" in out.columns:
            rdist = pd.to_numeric(out["reactive_ads_distance(Å)"], errors="coerce")
            out.loc[rdist > 3.25, "qa"] = "separated"
        for ecol in ("ΔE_proxy (eV)", "ΔE_ads_user (eV)", "ΔE_raw_proxy_diagnostic (eV)"):
            if ecol in out.columns:
                ev = pd.to_numeric(out[ecol], errors="coerce")
                out.loc[(~np.isfinite(ev)) | (ev.abs() > 10.0), "qa"] = "crashed"
        if "surface_distorted" in out.columns:
            sd = out["surface_distorted"].astype(str).str.lower().isin(["true", "1", "yes"])
            out.loc[sd & out["qa"].isin(["ok", "ok_single_point_proxy", "ok_short_relax_proxy", "ok_normal_relax_proxy", "ok_local_flex_proxy", "ok_rigid_proxy", "ok_frozen_pose_proxy", "ok_axis_locked_proxy"]), "qa"] = "surface_distorted"
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


# -----------------------------------------------------------------------------
# SAGE-VOC QA helpers
# -----------------------------------------------------------------------------

def voc_apply_qa_policy(df: pd.DataFrame, disp_thresh: float = 1.2) -> pd.DataFrame:
    """Structure-first QA policy for SAGE-VOC proxy rows.

    Validity is intentionally simple:
      - adsorbate descriptor state must stay internally intact;
      - adsorbate must remain surface-bound;
      - no collision/burial/severe slab collapse;
      - finite descriptor energy within a broad sanity window.

    Site chemistry such as O-only vs cation-containing support is metadata, not
    a hard reject.  This prevents valid CO* poses from being rejected merely
    because the nearest surface atom is not the preferred cation label.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    if "qa" in out.columns:
        out["qa"] = _normalize_text_series(out["qa"])
    else:
        out["qa"] = "ok"

    hard_reject = pd.Series(False, index=out.index)
    reasons = pd.Series("", index=out.index, dtype="object")

    ads_label = _string_series(out, "adsorbate", default="").str.replace(" ", "", regex=False).str.upper() if "adsorbate" in out.columns else pd.Series("", index=out.index)
    metal_her_h_only = _voc_metal_her_h_mask(out)
    h_only = _voc_h_only_mask(out)

    def _append_reason(mask, label):
        nonlocal reasons
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=out.index)
        reasons.loc[mask] = reasons.loc[mask].astype(str).where(reasons.loc[mask].astype(str).eq(""), reasons.loc[mask].astype(str) + ";") + str(label)

    # Standalone H* rows are H-only descriptors, not VOC molecules.
    # Neutralize molecular-geometry columns before generic VOC QA can use
    # missing heavy atoms or missing reactive groups as rejection evidence.
    if h_only.any():
        out.loc[h_only, "selected_for_descriptor"] = True
        out.loc[h_only, "diagnostic_only"] = False
        out.loc[h_only, "adsorbate_fragmented"] = False
        out.loc[h_only, "surface_bound"] = True
        out.loc[h_only, "bound_geometry_valid"] = True
        out.loc[h_only, "reactive_group_contact_valid"] = True
        out.loc[h_only, "c_series_pose_required"] = False
        out.loc[h_only, "c_series_pose_valid"] = True
        out.loc[h_only & _voc_h_has_finite_energy(out), "descriptor_energy_valid"] = True
        for _col in ["min_ads_heavy_slab_dist(Å)", "min_ads_slab_dist(Å)", "reactive_group_slab_dist(Å)", "c_anchor_slab_dist(Å)"]:
            if _col in out.columns:
                out.loc[h_only, _col] = 1.0
        for _col in ["reactive_group_height_above_support(Å)", "c_anchor_height_above_support(Å)"]:
            if _col in out.columns:
                out.loc[h_only, _col] = 0.5

    # 1) Existing explicit hard-fail QA labels from backend.
    backend_hard = {
        "crashed",
        "adsorbate_fragmented",
        "broken",
        "invalid",
        "invalid_seed_collision",
        "buried_adsorbate",
        "not_surface_bound",
        "reactive_group_detached",
        "c_series_pose_invalid",
        "adsorbate_floating",
        "desorbed",
        "separated",
        "surface_collapsed",
        "bad_seed_high_energy",
    }
    m = out["qa"].isin(backend_hard)
    hard_reject |= m
    _append_reason(m, "backend_hard_qa")

    # 2) Fragmentation: this is the most important descriptor-state failure.
    if "adsorbate_fragmented" in out.columns:
        frag = out["adsorbate_fragmented"].astype(str).str.lower().isin(["true", "1", "yes"])
        hard_reject |= frag
        _append_reason(frag, "adsorbate_fragmented")

    # 3) Surface-bound check.  Prefer heavy-atom distance if available.
    if "min_ads_heavy_slab_dist(Å)" in out.columns:
        hd = pd.to_numeric(out["min_ads_heavy_slab_dist(Å)"], errors="coerce")
        unbound = (~np.isfinite(hd)) | (hd > 3.05)
        collision = np.isfinite(hd) & (hd < 0.65)
        hard_reject |= unbound | collision
        _append_reason(unbound, "not_surface_bound")
        _append_reason(collision, "invalid_seed_collision")
    elif "min_ads_slab_dist(Å)" in out.columns:
        dmin = pd.to_numeric(out["min_ads_slab_dist(Å)"], errors="coerce")
        unbound = (~np.isfinite(dmin)) | (dmin > 3.10)
        collision = np.isfinite(dmin) & (dmin < 0.75)
        hard_reject |= unbound | collision
        _append_reason(unbound, "not_surface_bound")
        _append_reason(collision, "invalid_seed_collision")

    # 4) Backend explicit bound flag, if present, is authoritative for failure.
    if "bound_geometry_valid" in out.columns:
        bvalid = out["bound_geometry_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        # Do not use this alone to reject legacy rows that lack the new columns.
        has_new_geom = any(c in out.columns for c in ["adsorbate_fragmented", "min_ads_heavy_slab_dist(Å)", "surface_bound"])
        if has_new_geom:
            hard_reject |= ~bvalid
            _append_reason(~bvalid, "bound_geometry_invalid")
    if "surface_bound" in out.columns:
        sbound = out["surface_bound"].astype(str).str.lower().isin(["true", "1", "yes"])
        hard_reject |= ~sbound
        _append_reason(~sbound, "not_surface_bound_flag")

    # Species-specific reactive group contact.  This catches floating CO*/OH*/
    # CH3COO* cases that can pass a whole-molecule heavy-atom distance check.
    if "reactive_group_contact_valid" in out.columns:
        rg_valid = out["reactive_group_contact_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        hard_reject |= ~rg_valid
        _append_reason(~rg_valid, "reactive_group_detached")
    if "reactive_group_slab_dist(Å)" in out.columns:
        rgd = pd.to_numeric(out["reactive_group_slab_dist(Å)"], errors="coerce")
        # Broad safety net for cached/legacy rows where the boolean flag may be missing or stale.
        rg_far = (~np.isfinite(rgd)) | (rgd > 3.05)
        hard_reject |= rg_far
        _append_reason(rg_far, "reactive_group_far_from_slab")
    if "reactive_group_height_above_support(Å)" in out.columns:
        rgh = pd.to_numeric(out["reactive_group_height_above_support(Å)"], errors="coerce")
        rg_float = np.isfinite(rgh) & (rgh > 3.05)
        hard_reject |= rg_float
        _append_reason(rg_float, "reactive_group_floating")

    # C-containing descriptor pose QA.  This is deliberately independent from
    # cation/O site chemistry: CO* must be C-down and carbonyl/carboxyl groups
    # must remain near the surface, but nearest surface element is not a hard
    # reject.
    if "c_series_pose_valid" in out.columns:
        c_req = out["c_series_pose_required"].astype(str).str.lower().isin(["true", "1", "yes"]) if "c_series_pose_required" in out.columns else pd.Series(True, index=out.index)
        c_valid = out["c_series_pose_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        c_bad = c_req & (~c_valid)
        hard_reject |= c_bad
        _append_reason(c_bad, "c_series_pose_invalid")
    if "c_anchor_slab_dist(Å)" in out.columns:
        cdist = pd.to_numeric(out["c_anchor_slab_dist(Å)"], errors="coerce")
        ads = out["adsorbate"].astype(str).str.upper() if "adsorbate" in out.columns else pd.Series("", index=out.index)
        # Broad cached-row safety net.  Species-specific backend checks are stricter.
        c_far = ads.str.contains("CO", regex=False) & ((~np.isfinite(cdist)) | (cdist > 2.85))
        # CH3CHO is a secondary/accessibility descriptor.  Allow tilted-bound
        # carbonyl C/O contact; reject only clearly detached aldehyde poses.
        ch3cho_far = ads.str.contains("CH3CHO", regex=False) & ((~np.isfinite(cdist)) | (cdist > 2.85))
        hard_reject |= c_far | ch3cho_far
        _append_reason(c_far, "c_anchor_far_from_slab")
        _append_reason(ch3cho_far, "aldehyde_carbonyl_O_far_from_slab")
    if "c_anchor_height_above_support(Å)" in out.columns:
        cheight = pd.to_numeric(out["c_anchor_height_above_support(Å)"], errors="coerce")
        ads = out["adsorbate"].astype(str).str.upper() if "adsorbate" in out.columns else pd.Series("", index=out.index)
        ch3cho_float = ads.str.contains("CH3CHO", regex=False) & np.isfinite(cheight) & (cheight > 2.65)
        hard_reject |= ch3cho_float
        _append_reason(ch3cho_float, "aldehyde_carbonyl_O_floating")

    # 5) Severe slab collapse only.  Mild distortion can remain as warning candidate.
    top_disp = pd.to_numeric(out.get("top_slab_max_disp(Å)", pd.Series(np.nan, index=out.index)), errors="coerce")
    top_lift = pd.to_numeric(out.get("top_slab_max_lift(Å)", pd.Series(np.nan, index=out.index)), errors="coerce")
    collapsed = (np.isfinite(top_disp) & (top_disp > 2.0)) | (np.isfinite(top_lift) & (top_lift > 1.5))
    hard_reject |= collapsed
    _append_reason(collapsed, "surface_collapsed")

    # 6) Energy sanity check only; moderate positive values are not hard reject.
    e_col = "ΔE_proxy (eV)" if "ΔE_proxy (eV)" in out.columns else ("ΔE_ads_user (eV)" if "ΔE_ads_user (eV)" in out.columns else None)
    if e_col is not None:
        ev = pd.to_numeric(out[e_col], errors="coerce")
        blow = (~np.isfinite(ev)) | (ev.abs() > 10.0)
        hard_reject |= blow
        _append_reason(blow, "energy_invalid_or_extreme")

    ok_like = {"ok", "ok_single_point_proxy", "ok_short_relax_proxy", "ok_normal_relax_proxy", "ok_local_flex_proxy", "ok_rigid_proxy", "ok_frozen_pose_proxy", "ok_axis_locked_proxy", "surface_distorted_but_bound", "ok_metal_che_her_like", "ech_diagnostic_valid"}
    pol = _string_series(out, "relax_policy", default="").str.lower() if "relax_policy" in out.columns else pd.Series("", index=out.index)
    default_ok = np.where(pol.eq("normal_relax_proxy"), "ok_normal_relax_proxy", "ok")

    # Assign final QA once, at the end.  This avoids overwriting hard rejects.
    out["qa_note"] = np.where(reasons.astype(str).ne(""), reasons, out.get("qa_note", ""))
    out.loc[hard_reject, "qa"] = "geometry_invalid"
    if "adsorbate_fragmented" in out.columns:
        frag = out["adsorbate_fragmented"].astype(str).str.lower().isin(["true", "1", "yes"])
        out.loc[frag, "qa"] = "adsorbate_fragmented"
    if "min_ads_heavy_slab_dist(Å)" in out.columns:
        hd = pd.to_numeric(out["min_ads_heavy_slab_dist(Å)"], errors="coerce")
        out.loc[(~np.isfinite(hd)) | (hd > 3.05), "qa"] = "not_surface_bound"
        out.loc[np.isfinite(hd) & (hd < 0.65), "qa"] = "invalid_seed_collision"
    if "reactive_group_contact_valid" in out.columns:
        rg_valid = out["reactive_group_contact_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        out.loc[~rg_valid, "qa"] = "reactive_group_detached"
    if "reactive_group_slab_dist(Å)" in out.columns:
        rgd = pd.to_numeric(out["reactive_group_slab_dist(Å)"], errors="coerce")
        out.loc[(~np.isfinite(rgd)) | (rgd > 3.05), "qa"] = "reactive_group_detached"
    if "reactive_group_height_above_support(Å)" in out.columns:
        rgh = pd.to_numeric(out["reactive_group_height_above_support(Å)"], errors="coerce")
        out.loc[np.isfinite(rgh) & (rgh > 3.05), "qa"] = "adsorbate_floating"
    if "c_series_pose_valid" in out.columns:
        c_req = out["c_series_pose_required"].astype(str).str.lower().isin(["true", "1", "yes"]) if "c_series_pose_required" in out.columns else pd.Series(True, index=out.index)
        c_valid = out["c_series_pose_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        out.loc[c_req & (~c_valid), "qa"] = "c_series_pose_invalid"
    if "c_anchor_slab_dist(Å)" in out.columns and "adsorbate" in out.columns:
        cdist = pd.to_numeric(out["c_anchor_slab_dist(Å)"], errors="coerce")
        ads = out["adsorbate"].astype(str).str.upper()
        out.loc[ads.str.contains("CH3CHO", regex=False) & ((~np.isfinite(cdist)) | (cdist > 2.85)), "qa"] = "c_series_pose_invalid"
    if "c_anchor_height_above_support(Å)" in out.columns and "adsorbate" in out.columns:
        cheight = pd.to_numeric(out["c_anchor_height_above_support(Å)"], errors="coerce")
        ads = out["adsorbate"].astype(str).str.upper()
        out.loc[ads.str.contains("CH3CHO", regex=False) & np.isfinite(cheight) & (cheight > 2.65), "qa"] = "adsorbate_floating"

    # ECH diagnostic rows.  Fragmentation/collision/burial/slab collapse remain
    # hard rejected.  For H*+CH3CHO*, geometry_invalid/adsorbate_floating may be
    # salvageable because those labels can come from conservative VOC pose checks
    # even when the relaxed CH3CHO molecule remains intact and proximity metrics
    # are meaningful.
    ech_mask = _voc_ech_row_mask(out)
    ech_hch3cho = pd.Series(False, index=out.index)
    if "ech_state_type" in out.columns:
        ech_hch3cho |= out["ech_state_type"].astype(str).str.strip().str.lower().eq("h_ch3cho_coadsorption")
    if "descriptor_state" in out.columns:
        ech_hch3cho |= out["descriptor_state"].astype(str).str.replace(" ", "", regex=False).str.upper().eq("H*+CH3CHO*")
    if "adsorbate" in out.columns:
        ech_hch3cho |= out["adsorbate"].astype(str).str.replace(" ", "", regex=False).str.upper().eq("H*+CH3CHO*")
    ech_hch3cho &= ech_mask

    hard_labels_general = [
        "crashed", "invalid", "geometry_invalid", "adsorbate_fragmented",
        "invalid_seed_collision", "buried_adsorbate", "surface_collapsed",
        "adsorbate_floating", "bad_seed_high_energy",
    ]
    hard_labels_hch3cho = [
        "crashed", "invalid", "adsorbate_fragmented",
        "invalid_seed_collision", "buried_adsorbate", "surface_collapsed",
        "bad_seed_high_energy",
    ]
    qa_lower = out["qa"].astype(str).str.lower()
    ech_hard = (ech_mask & (~ech_hch3cho) & qa_lower.isin(hard_labels_general)) | (ech_hch3cho & qa_lower.isin(hard_labels_hch3cho))
    if "adsorbate_fragmented" in out.columns:
        ech_hard |= ech_mask & _boolish_series(out, "adsorbate_fragmented", default=False)
    desc_valid_for_ech = _boolish_series(out, "descriptor_energy_valid", default=True) if "descriptor_energy_valid" in out.columns else pd.Series(True, index=out.index)
    ech_soft = ech_mask & (~ech_hard) & desc_valid_for_ech
    if "ech_salvaged_soft_geometry" not in out.columns:
        soft_labels_hch3cho = qa_lower.isin(["geometry_invalid", "adsorbate_floating", "c_series_pose_invalid", "reactive_group_detached", "not_surface_bound", "separated", "migrated"])
        out["ech_salvaged_soft_geometry"] = bool(False)
        out.loc[ech_hch3cho & soft_labels_hch3cho & (~ech_hard), "ech_salvaged_soft_geometry"] = True
    if "ech_classification" in out.columns:
        cls = out["ech_classification"].astype(str).str.strip().str.lower()
        ech_soft &= ~cls.str.startswith("invalid_")
    out.loc[ech_soft, "qa"] = "ech_diagnostic_valid"
    hard_reject = hard_reject & (~ech_soft)

    # Metal H* rows supplied by the CHE/HER-like branch are not molecular VOC
    # adsorbates and must not be rejected by heavy-atom/reactive-group VOC QA.
    # Their reliability is controlled by the HER-like calculation and energy
    # sanity, not by CH3CHO/COx pose checks.
    try:
        h_finite = h_only & _voc_h_has_finite_energy(out)
        if h_finite.any():
            hard_reject.loc[h_finite] = False
            out.loc[h_finite, "descriptor_energy_valid"] = True
            out.loc[h_finite, "selected_for_descriptor"] = True
            out.loc[h_finite, "diagnostic_only"] = False
            out.loc[h_finite, "qa"] = "ok_hstar_proxy"
            if metal_her_h_only.any():
                out.loc[metal_her_h_only & h_finite, "qa"] = "ok_metal_che_her_like"
            out.loc[h_finite, "qa_note"] = out.loc[h_finite, "qa_note"].astype(str).where(
                out.loc[h_finite, "qa_note"].astype(str).str.len() > 0,
                "standalone_Hstar_descriptor_exempt_from_VOC_molecular_QA"
            )
    except Exception:
        pass

    ok_mask = ~hard_reject
    # Preserve explicit ok-like backend labels; otherwise use relax-policy default.
    current_ok = out["qa"].isin(ok_like)
    out.loc[ok_mask & (~current_ok), "qa"] = pd.Series(default_ok, index=out.index).loc[ok_mask & (~current_ok)]

    # Mild surface distortion label only if still valid.
    if "surface_distorted" in out.columns:
        sd = out["surface_distorted"].astype(str).str.lower().isin(["true", "1", "yes"])
        out.loc[ok_mask & sd, "qa"] = "surface_distorted_but_bound"

    if "migrated" not in out.columns:
        if "ads_lateral_disp(Å)" in out.columns:
            disp = pd.to_numeric(out["ads_lateral_disp(Å)"], errors="coerce").fillna(0.0)
            out["migrated"] = disp > float(disp_thresh)
        else:
            out["migrated"] = False

    if "relaxed_site" not in out.columns:
        out["relaxed_site"] = out["site"] if "site" in out.columns else "unknown"

    keep_qa = {"ok", "ok_single_point_proxy", "ok_short_relax_proxy", "ok_normal_relax_proxy", "ok_local_flex_proxy", "ok_rigid_proxy", "ok_frozen_pose_proxy", "ok_axis_locked_proxy", "surface_distorted_but_bound", "ok_metal_che_her_like", "ok_hstar_proxy"}
    selected = out["selected_for_descriptor"].astype(str).str.lower().isin(["true", "1", "yes"]) if "selected_for_descriptor" in out.columns else pd.Series(True, index=out.index)
    diagnostic = out["diagnostic_only"].astype(str).str.lower().isin(["true", "1", "yes"]) if "diagnostic_only" in out.columns else pd.Series(False, index=out.index)
    reliable_mask = out["qa"].isin(keep_qa) & selected & (~diagnostic) & (~hard_reject)
    if "descriptor_energy_valid" in out.columns:
        desc_valid = out["descriptor_energy_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        # For old rows this may be stale, but for current backend it should be respected.
        reliable_mask &= desc_valid
    out["reliability"] = np.where(reliable_mask, "reliable", "unreliable")
    ech_diag = _voc_ech_row_mask(out) & out["qa"].astype(str).str.lower().eq("ech_diagnostic_valid")
    if "descriptor_energy_valid" in out.columns:
        ech_diag &= _boolish_series(out, "descriptor_energy_valid", default=False)
    out.loc[ech_diag, "reliability"] = "diagnostic_valid"
    return out


def voc_split_candidates_diagnostics_rejected(df: pd.DataFrame):
    """Split VOC proxy rows into candidate, ECH-diagnostic, and rejected rows.

    ECH co-adsorption rows such as H*+CH3CHO* are not ordinary ranking
    candidates.  When the backend marks them as ech_diagnostic_valid and the
    hard geometry checks are satisfied, keep them in a separate diagnostic
    bucket rather than counting them as rejected/QA-invalid.
    """
    if df is None or df.empty:
        return df, df, df

    out = voc_apply_qa_policy(df)

    keep_qa = {
        "ok",
        "ok_single_point_proxy",
        "ok_short_relax_proxy",
        "ok_normal_relax_proxy",
        "ok_local_flex_proxy",
        "ok_rigid_proxy",
        "ok_frozen_pose_proxy",
        "ok_axis_locked_proxy",
        "surface_distorted_but_bound",
        "ok_metal_che_her_like",
        "ok_hstar_proxy",
    }

    candidate = out["qa"].isin(keep_qa)
    if "reliability" in out.columns:
        candidate &= out["reliability"].astype(str).str.lower().eq("reliable")
    if "selected_for_descriptor" in out.columns:
        candidate &= out["selected_for_descriptor"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "diagnostic_only" in out.columns:
        candidate &= ~out["diagnostic_only"].astype(str).str.lower().isin(["true", "1", "yes"])

    h_only = _voc_h_only_mask(out)
    if h_only.any():
        candidate |= h_only & _voc_h_has_finite_energy(out)

    diagnostic = _voc_ech_row_mask(out) & out["qa"].astype(str).str.lower().eq("ech_diagnostic_valid")
    if "reliability" in out.columns:
        diagnostic |= out["reliability"].astype(str).str.lower().eq("diagnostic_valid")
    diagnostic &= ~candidate

    rejected = ~(candidate | diagnostic)

    return out[candidate].copy(), out[diagnostic].copy(), out[rejected].copy()


def voc_split_by_qa(df: pd.DataFrame):
    """Backward-compatible two-way VOC split.

    Returns ordinary descriptor candidates and genuinely rejected rows.
    ECH diagnostic rows are intentionally excluded from the rejected side; use
    voc_split_candidates_diagnostics_rejected() when the diagnostic bucket is
    needed explicitly.
    """
    if df is None or df.empty:
        return df, df
    cand, _diag, reject = voc_split_candidates_diagnostics_rejected(df)
    return cand, reject


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
    elif str(mode).upper() == "VOC":
        cols = [
            "site_label",
            "requested_site",
            "initial_geom_site",
            "relaxed_site",
            "migrated",
            "adsorbate",
            "descriptor_state",
            "state_type",
            "target_voc",
            "qa",
            "qa_note",
            "relax_policy",
            "selected_for_descriptor",
            "fallback_from",
            "coadsorption_seed_policy",
            "coadsorption_role",
            "reduction_h_placement",
            "H_descriptor_source",
            "H_placement_policy",
            "ΔG_H_CHE (eV)",
            "ΔG_H_CHE_like (eV)",
            "ΔE_H_user (eV)",
            "standard_CHE_corr (eV)",
            "H_lateral_disp(Å)",
            "ech_seed_policy",
            "ech_seed_role",
            "ech_state_type",
            "ech_classification",
            "ech_c_carbonyl_h_distance_A",
            "ech_h_h_distance_A",
            "ech_coadsorption_retained",
            "ech_h_transfer_proximity",
            "ech_product_like_collapse",
            "ech_h2_like_risk",
            "ech_qc_note",
            "reconstruction_sensitive",
            "top_slab_max_disp(Å)",
            "top_slab_max_lift(Å)",
            "min_ads_slab_dist(Å)",
            "reactive_ads_distance(Å)",
            "ΔE_proxy (eV)",
            "ΔE_VOC_ads_proxy (eV)",
            "ΔE_H_VOC_proximity_proxy (eV)",
            "ΔE_OH_VOC_proximity_proxy (eV)",
            "ΔE_proximity_proxy (eV)",
            "ΔE_ads_user (eV)",
            "E_state_user (eV)",
            "E_ref_sum (eV)",
            "ads_lateral_disp(Å)",
            "min_ads_slab_dist(Å)",
            "template_warning",
            "proxy_warning",
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
    # Streamlit/pyarrow fails when duplicate column names are present.
    # This can happen either because a display list accidentally includes the
    # same column twice, or because a CSV/result DataFrame was produced with
    # duplicated names.  Keep the first occurrence deterministically.
    work = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    existing_cols = []
    seen_cols = set()
    for c in cols:
        if c in work.columns and c not in seen_cols:
            existing_cols.append(c)
            seen_cols.add(c)
    return work[existing_cols]

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

