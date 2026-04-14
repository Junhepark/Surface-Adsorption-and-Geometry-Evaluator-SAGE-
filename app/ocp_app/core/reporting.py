import json
import numpy as np
import pandas as pd

PRICING_USD_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
}


def estimate_cost_usd(model_name: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    if not model_name:
        return None
    m = model_name.lower()
    rate = None
    for prefix, r in PRICING_USD_PER_1M.items():
        if m.startswith(prefix):
            rate = r
            break
    if rate is None:
        return None
    return (float(input_tokens) * float(rate["input"]) + float(output_tokens) * float(rate["output"])) / 1_000_000.0


def fetch_mp_meta(mp_id: str, api_key: str | None):
    if not mp_id:
        return None
    api_key = api_key or None
    try:
        from mp_api.client import MPRester  # type: ignore
        with MPRester(api_key=api_key) as mpr:
            docs = mpr.summary.search(material_ids=[mp_id])
            if not docs:
                return {"mp_id": mp_id}
            d = docs[0]
            get = (lambda k: getattr(d, k, None)) if not isinstance(d, dict) else d.get
            return {
                "mp_id": mp_id,
                "formula_pretty": get("formula_pretty"),
                "energy_above_hull": get("energy_above_hull"),
                "formation_energy_per_atom": get("formation_energy_per_atom"),
                "band_gap": get("band_gap"),
                "crystal_system": get("crystal_system"),
                "spacegroup_symbol": (get("symmetry") or {}).get("symbol") if isinstance(get("symmetry"), dict) else None,
            }
    except Exception:
        pass
    try:
        from pymatgen.ext.matproj import MPRester  # type: ignore
        with MPRester(api_key) as mpr:
            docs = mpr.query(criteria={"material_id": mp_id},
                             properties=["pretty_formula", "e_above_hull", "formation_energy_per_atom", "band_gap", "spacegroup"])
            if not docs:
                return {"mp_id": mp_id}
            d = docs[0]
            return {
                "mp_id": mp_id,
                "formula_pretty": d.get("pretty_formula"),
                "energy_above_hull": d.get("e_above_hull"),
                "formation_energy_per_atom": d.get("formation_energy_per_atom"),
                "band_gap": d.get("band_gap"),
                "spacegroup_symbol": (d.get("spacegroup") or {}).get("symbol") if isinstance(d.get("spacegroup"), dict) else None,
            }
    except Exception:
        return {"mp_id": mp_id}


def pick_representative_sites_her(df: pd.DataFrame):
    col = None
    for c in ["ΔG_H(U,pH) (eV)", "dG_H(U,pH) (eV)", "ΔG_H (eV)", "dG_H (eV)"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        return {}

    g = pd.to_numeric(df[col], errors="coerce")
    g_valid = g.dropna()
    if g_valid.empty:
        return {}

    def _site_label(i):
        if "site_label" in df.columns:
            return str(df.loc[i, "site_label"])
        if "site" in df.columns:
            return str(df.loc[i, "site"])
        return str(i)

    i_act = (g_valid.abs()).idxmin()
    i_min = g_valid.idxmin()

    return {
        "activity_best": {"site_label": _site_label(i_act), "value": float(g.loc[i_act]), "criterion": "min_abs_dG"},
        "thermo_best": {"site_label": _site_label(i_min), "value": float(g.loc[i_min]), "criterion": "most_negative_dG"},
        "column_used": col,
    }


def pick_representative_sites_co2rr(df: pd.DataFrame):
    col = None
    for c in ["ΔE_ads_user (eV)", "dE_ads_user (eV)", "ΔE_ads (eV)", "dE_ads (eV)"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        return {}

    e = pd.to_numeric(df[col], errors="coerce")
    stable_mask = pd.Series([True] * len(df), index=df.index)

    if "migrated" in df.columns:
        stable_mask &= (pd.to_numeric(df["migrated"], errors="coerce").fillna(0) == 0)
    if "reliability" in df.columns:
        stable_mask &= ~df["reliability"].astype(str).str.lower().isin(["failed", "error", "nan"])
    if "qa" in df.columns:
        stable_mask &= ~df["qa"].astype(str).str.lower().str.contains("fail|invalid|error|nan", regex=True, na=False)

    e_stable = e[stable_mask].dropna()
    if e_stable.empty:
        e_stable = e.dropna()
        if e_stable.empty:
            return {"column_used": col, "stable_filter": "none (no valid values)"}
        stable_mask = pd.Series([True] * len(df), index=df.index)

    def _site_label(i):
        if "site_label" in df.columns:
            return str(df.loc[i, "site_label"])
        if "site" in df.columns:
            return str(df.loc[i, "site"])
        return str(i)

    all_positive = bool((e_stable >= 0).all())
    i_min = e_stable.idxmin()
    strong_criterion = "most_negative_dE" if not all_positive else "min_dE_all_positive"
    strong_note = (
        "Exothermic adsorption exists (ΔE<0 present); 'strong' wording allowed."
        if not all_positive
        else "All stable sites have ΔE>=0; DO NOT call this 'strong adsorption'. Report as least-unfavorable/least-endothermic and discuss only relative ordering."
    )
    strong = {
        "site_label": _site_label(i_min),
        "value": float(e.loc[i_min]),
        "criterion": strong_criterion,
        "terminology_note": strong_note,
    }

    e_exo = e_stable[e_stable < 0]
    if not e_exo.empty:
        i_bal = (e_exo.abs()).idxmin()
        balanced = {
            "site_label": _site_label(i_bal),
            "value": float(e.loc[i_bal]),
            "criterion": "min_abs_dE_among_exothermic",
            "terminology_note": "Balanced binding defined only among ΔE<0 candidates to avoid weak-adsorption misinterpretation.",
        }
    else:
        balanced = {
            "site_label": None,
            "value": None,
            "criterion": "N/A_no_exothermic_adsorption",
            "terminology_note": "No ΔE<0 candidates among stable sites; balanced track is not defined under this conservative rule.",
        }

    return {
        "strong_binding_best": strong,
        "balanced_binding_best": balanced,
        "column_used": col,
        "stable_filter": "non-migrating (migrated==0) + non-failed rows when available",
        "binding_regime": {
            "all_stable_nonnegative": bool(all_positive),
            "min_stable_value": float(e_stable.min()),
            "n_stable": int(e_stable.shape[0]),
        },
    }


def build_llm_payload(last_run: dict, mp_api_key: str | None = None):
    df = last_run.get("df")
    meta = last_run.get("meta") or {}

    payload = {
        "run_meta": {
            "reaction_mode": str(last_run.get("reaction_mode", "")),
            "material_type": str(last_run.get("mtype", "")),
            "mode_label": str(last_run.get("mode_label", "")),
            "model": str(meta.get("MODEL", meta.get("model", ""))),
            "device": str(meta.get("DEVICE", meta.get("device", ""))),
            "U": float(last_run.get("U_input", 0.0)),
            "pH": float(last_run.get("pH_input", 14.0)),
        },
        "mp_meta": None,
        "definitions": {
            "sign_convention": {},
            "filters": {},
        },
        "rules": {},
        "qc_flags": {},
        "site_rows": [],
    }

    mp_id = None
    try:
        mp_id = str(last_run.get("loaded_mp_id") or meta.get("MP_ID") or "").strip()
    except Exception:
        mp_id = None
    payload["mp_meta"] = fetch_mp_meta(mp_id, mp_api_key) if mp_id else None

    is_her = bool(last_run.get("is_her"))
    if is_her:
        payload["definitions"]["sign_convention"] = {
            "column": "ΔG_H(U,pH) (eV) if present, else ΔG_H (eV)",
            "meaning": "Hydrogen adsorption free energy descriptor. Values near 0 eV are often discussed as balanced binding; more negative means stronger H binding; more positive means weaker H binding.",
        }
        payload["definitions"]["filters"] = {
            "reliable_rows": "Use rows from df_rel (energy/displacement-based reliable split).",
            "unreliable_rows": "Rows in df_unrel are not used for representative recommendations.",
        }
    else:
        payload["definitions"]["sign_convention"] = {
            "column": "ΔE_ads_user (eV) if present, else ΔG_ads (eV)",
            "meaning": "Adsorption descriptor relative to the app's chosen reference. Negative means exothermic/stronger binding under that reference; positive means endothermic/weaker binding.",
        }
        payload["definitions"]["filters"] = {
            "candidates": "Use qa in {ok, migrated} as candidate rows; rows with qa outside that set are rejected.",
            "stable_subset_for_recommendation": "For representative recommendations, further prefer non-migrating rows (migrated==0) and exclude obviously failed/invalid rows when possible.",
            "migrated_policy": "A migrated row may remain a candidate in UI tables, but migrated rows are excluded from representative recommendations unless no stable rows remain.",
        }

    if isinstance(df, pd.DataFrame) and not df.empty:
        if bool(last_run.get("is_her")):
            payload["rules"] = pick_representative_sites_her(df)
        else:
            payload["rules"] = pick_representative_sites_co2rr(df)

        if "is_duplicate" in df.columns:
            payload["qc_flags"]["n_duplicates"] = int(pd.to_numeric(df["is_duplicate"], errors="coerce").fillna(0).sum())
        if "migrated" in df.columns:
            payload["qc_flags"]["n_migrated"] = int(pd.to_numeric(df["migrated"], errors="coerce").fillna(0).sum())
        if "qa" in df.columns:
            payload["qc_flags"]["qa_counts"] = df["qa"].astype(str).value_counts(dropna=False).to_dict()

        keep_cols = [
            "site_label", "site", "site_kind", "relaxed_site", "migration_destination", "migration_path",
            "reliability", "qa", "migrated", "is_duplicate",
            "ads_lateral_disp(Å)", "H_lateral_disp(Å)",
            "ΔG_H(U,pH) (eV)", "ΔG_H (eV)", "ΔE_H_user (eV)",
            "ΔE_ads_user (eV)", "ΔG_ads (eV)",
            "surfactant_class", "surfactant_chgnet_prerelax_slab",
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        payload["site_rows"] = df[keep_cols].to_dict(orient="records")
        if isinstance(meta, dict) and meta.get("MIGRATION_SUMMARY") is not None:
            payload["migration_summary"] = meta.get("MIGRATION_SUMMARY")

    return payload


def call_llm_interpreter(payload: dict, *, api_key: str, model_name: str = "gpt-4o-mini"):
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("OpenAI API key is empty (set it in 0) Credentials & LLM).")

    try:
        from openai import OpenAI  # type: ignore
        from pydantic import BaseModel, Field, ConfigDict
    except Exception as e:
        raise RuntimeError("Missing dependency. Install with: pip install openai pydantic") from e

    class PaperTrack(BaseModel):
        model_config = ConfigDict(extra="forbid")
        track_name: str = Field(..., description="Name of the reporting track (e.g., 'strong_binding', 'balanced_binding').")
        criterion: str = Field(..., description="Criterion used (e.g., most_negative_dE, min_abs_dE, most_negative_dG, min_abs_dG).")
        column_used: str = Field(..., description="Which numeric column was used to choose this track.")
        site_label: str | None = Field(..., description="Site label selected for this track. Use null if unavailable.")
        value: float | None = Field(..., description="Selected metric value for this track. Use null if unavailable.")
        selection_rule: str = Field(..., description="Explicit rule describing how this value was chosen.")
        evidence_note: str = Field(..., description="Short note tying the selection to payload evidence and QC filters (e.g., non-migrating).")

    class RecommendedForPaper(BaseModel):
        model_config = ConfigDict(extra="forbid")
        strong_binding_track: PaperTrack = Field(..., description="Track emphasizing strongest binding (most negative energy) among stable sites.")
        balanced_binding_track: PaperTrack = Field(..., description="Track emphasizing balanced binding (closest to 0) among stable sites.")
        primary_track_to_report: str = Field(..., description="Which track should be the primary headline in the paper: 'strong_binding' or 'balanced_binding'.")
        how_to_report: str = Field(..., description="Guidance on how to report both tracks without over-claiming activity; include sign-convention caveats.")
        suggested_paper_text: str = Field(..., description="1–4 sentences suitable for Methods/Results that report both tracks and justify the rules.")

    class Interpretation(BaseModel):
        model_config = ConfigDict(extra="forbid")
        executive_summary: str = Field(..., description="High-level summary (3–6 sentences).")
        qc_findings: list[str] = Field(..., description="Quality-control findings (migration, duplicates, site collapse, non-physical artifacts).")
        site_interpretation: list[str] = Field(..., description=("Interpretation of site behavior: why different sites look similar/different; why bridge/fcc can collapse to top; oxide vs metal behavior; surface area/termination effects."))
        recommended_for_paper: RecommendedForPaper = Field(..., description="Two-track recommendation for representative value(s) and how to report them.")
        next_actions: list[str] = Field(..., description="Concrete next actions (checks, reruns, DFT spot-checks, structure realism checks).")
        uncertainties: list[str] = Field(..., description="What cannot be concluded from the payload; assumptions; missing inputs.")

    client = OpenAI(api_key=api_key)

    developer_instructions = """
You are a scientific assistant that interprets electrocatalysis screening results (HER or CO2RR).

Use ONLY the provided JSON payload as evidence.
Do not use outside knowledge, unstated assumptions, or inferred values not supported by the payload.

Your task is to return a JSON object only, following the required schema exactly.

PRIORITY OF EVIDENCE
1. payload["definitions"] (highest priority)
2. payload["rules"]
3. per-site QC / diagnostics / flags
4. numeric site values
5. mp_meta thermodynamic indicators
If any source conflicts with a lower-priority source, follow the higher-priority source and note the conflict in uncertainties.

GENERAL DECISION RULES
- A site is "stable" only if it is not excluded by payload["definitions"] or payload["rules"].
- Exclude from recommendation any site flagged as migrated, collapsed, duplicate, or non-physical according to payload definitions/rules.
- If distance or geometry diagnostics required for a stability decision are missing, do not assume stability; mark the site as uncertain.
- If no stable sites remain after exclusions, set all recommended track fields to null and explain why.
- If only one stable site remains, both tracks may point to the same site if it satisfies both selection rules.
- If multiple sites are tied, break ties deterministically in this order:
  (1) site with fewer QC concerns,
  (2) site with more complete diagnostics,
  (3) lexicographically smaller site_label.

QC INTERPRETATION TASKS
1. Flag QC issues:
   - migration
   - site collapse
   - duplicates
   - non-physical artifacts
   - missing diagnostics that prevent firm interpretation
2. Explain likely causes using only payload-supported materials interpretations, such as:
   - surface termination effects
   - limited site diversity
   - oxide vs metal anchoring differences
   - slab/model limitations
Do not assert causes not grounded in the payload.

RECOMMENDATION RULES
You MUST populate recommended_for_paper with two tracks.

For HER:
- strong_binding_track = most negative ΔG_H among stable sites
- balanced_binding_track = stable site with minimum |ΔG_H|

For CO2RR:
- If any stable site has ΔE_ads < 0:
  - strong_binding_track = most negative ΔE_ads among stable sites
  - balanced_binding_track = stable site with minimum |ΔE_ads| among sites with ΔE_ads < 0
- If all stable sites have ΔE_ads >= 0:
  - report the minimum ΔE_ads site as the least-unfavorable / least-endothermic case
  - set balanced_binding_track.site_label = null
  - set balanced_binding_track.value = null
  - explain that no exothermic stable candidate exists under this reference definition
  - do not describe this as strong adsorption

REPORTING RULES
- Prefer payload["rules"] when selecting representative values.
- Use exact sign conventions from payload["definitions"].
- State explicitly whether a recommended site is non-migrating under the stated migration criterion.
- Do not use:
  "reliable", "optimal", "high efficiency", "low efficiency", "potential for CO2RR"
  unless such claims are explicitly supported by payload fields that justify them.
- Do not use "favorable" unless the sign convention and comparison set are explicitly stated.
- If ΔE_ads is positive, state that adsorption is endothermic/weak under this reference and discuss only relative ordering.
- If mp_meta contains energy_above_hull or formation energy, describe them only as thermodynamic indicators, typically near 0 K and without kinetic/processing context.

OUTPUT SCHEMA
Return JSON only with exactly these top-level keys:
{
  "mode": null,
  "qc_summary": {
    "status": null,
    "issues": [],
    "uncertainties": []
  },
  "site_assessment": [],
  "recommended_for_paper": {
    "strong_binding_track": {
      "site_label": null,
      "value": null,
      "evidence_note": "",
      "uncertainties": []
    },
    "balanced_binding_track": {
      "site_label": null,
      "value": null,
      "evidence_note": "",
      "uncertainties": []
    }
  },
  "materials_interpretation": [],
  "next_actions": []
}

SITE_ASSESSMENT ITEM SCHEMA
Each item in site_assessment must be:
{
  "site_label": null,
  "included_in_selection": null,
  "exclusion_reason": null,
  "qc_flags": [],
  "value": null,
  "evidence_note": "",
  "uncertainties": []
}

NEXT ACTIONS
Propose only payload-consistent validation actions, such as:
- alternative termination or polymorph
- slab thickness / repeat size check
- additional site diversity
- DFT spot-check
- duplicate filtering / geometry verification

Do not output markdown. Do not output prose outside JSON.
"""

    resp = client.responses.parse(
        model=model_name,
        input=[
            {"role": "system", "content": developer_instructions},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text_format=Interpretation,
        temperature=0.2,
        store=False,
    )
    out = resp.output_parsed.model_dump()
    out["_model"] = model_name
    usage_obj = getattr(resp, "usage", None)
    usage_dict = None
    if usage_obj is not None:
        try:
            usage_dict = usage_obj.model_dump()
        except Exception:
            try:
                usage_dict = dict(usage_obj)
            except Exception:
                usage_dict = {"raw": str(usage_obj)}
    if usage_dict is not None:
        out["_usage"] = usage_dict
        in_toks = usage_dict.get("input_tokens", usage_dict.get("prompt_tokens"))
        out_toks = usage_dict.get("output_tokens", usage_dict.get("completion_tokens"))
        try:
            in_toks_i = int(in_toks) if in_toks is not None else None
        except Exception:
            in_toks_i = None
        try:
            out_toks_i = int(out_toks) if out_toks is not None else None
        except Exception:
            out_toks_i = None
        out["_cost_estimate_usd"] = estimate_cost_usd(model_name, in_toks_i, out_toks_i)
    return out
