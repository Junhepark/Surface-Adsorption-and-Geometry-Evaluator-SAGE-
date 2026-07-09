from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


CO2RR_AIR_CO2_ADS = {"COOH", "CO", "OCHO", "HCOO"}
CO2RR_AIR_OXYGEN_ADS = {"OOH", "O", "OH"}


def _clean_adsorbate(x: Any) -> str:
    return str(x or "").replace("*", "").strip().upper()


def _risk_from_score(score: float) -> str:
    try:
        v = float(score)
    except Exception:
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    if v >= 0.67:
        return "high"
    if v >= 0.34:
        return "medium"
    return "low"


def _energy_col(df: pd.DataFrame) -> str | None:
    for c in ("ΔG_ads (eV)", "ΔE_ads_user (eV)", "ΔE_proxy (eV)", "ΔG_ads", "ΔE_ads_user"):
        if c in df.columns:
            return c
    return None


def _valid_rows(df: pd.DataFrame, *, family: str = "co2rr") -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "adsorbate" in out.columns:
        out["_ads_clean"] = out["adsorbate"].map(_clean_adsorbate)
    else:
        out["_ads_clean"] = ""

    if "reliability" in out.columns:
        rel = out["reliability"].astype(str).str.lower()
        out = out.loc[rel.eq("reliable")].copy()

    if "qa" in out.columns:
        qa = out["qa"].astype(str).str.lower()
        if family == "oxygen":
            keep = qa.isin(["ok", "bound_relaxed", "bound_migrated"])
        else:
            keep = qa.isin(["ok", "migrated"])
        out = out.loc[keep].copy()

    if family == "oxygen" and "valid_for_oer_summary" in out.columns:
        valid = out["valid_for_oer_summary"].astype(str).str.lower().isin(["true", "1", "yes"])
        out = out.loc[valid].copy()

    ecol = _energy_col(out)
    if ecol is not None:
        out[ecol] = pd.to_numeric(out[ecol], errors="coerce")
        out = out.loc[np.isfinite(out[ecol])].copy()
    return out


def _best_ads_energy(df: pd.DataFrame, ads: str) -> float:
    if df is None or df.empty:
        return float("nan")
    work = df.copy()
    if "_ads_clean" not in work.columns:
        work["_ads_clean"] = work.get("adsorbate", pd.Series("", index=work.index)).map(_clean_adsorbate)
    sub = work.loc[work["_ads_clean"].eq(str(ads).upper())].copy()
    if sub.empty:
        return float("nan")
    ecol = _energy_col(sub)
    if ecol is None:
        return float("nan")
    vals = pd.to_numeric(sub[ecol], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return float("nan")
    return float(vals.min())


def _best_ads_site(df: pd.DataFrame, ads: str) -> str:
    if df is None or df.empty:
        return ""
    work = df.copy()
    if "_ads_clean" not in work.columns:
        work["_ads_clean"] = work.get("adsorbate", pd.Series("", index=work.index)).map(_clean_adsorbate)
    sub = work.loc[work["_ads_clean"].eq(str(ads).upper())].copy()
    ecol = _energy_col(sub)
    if sub.empty or ecol is None:
        return ""
    sub[ecol] = pd.to_numeric(sub[ecol], errors="coerce")
    sub = sub.loc[np.isfinite(sub[ecol])].copy()
    if sub.empty:
        return ""
    row = sub.loc[sub[ecol].idxmin()]
    for c in ("relaxed_site", "site_label", "site"):
        if c in row and str(row.get(c, "")).strip():
            return str(row.get(c))
    return ""


def _extract_her_delta_g(her_guard: Mapping[str, Any] | None) -> float:
    if not isinstance(her_guard, Mapping):
        return float("nan")
    for c in ("ΔG_H (eV)", "deltaG_H_eV", "dG_H_eV"):
        if c in her_guard:
            try:
                return float(her_guard[c])
            except Exception:
                pass
    return float("nan")


def _classify_pathway(dg_cooh: float, dg_ocho: float, dg_hcoo: float) -> tuple[str, str]:
    vals = {
        "CO": dg_cooh,
        "formate": dg_ocho,
    }
    if np.isfinite(dg_hcoo):
        # HCOO* is kept as a secondary formate proxy if present.
        vals["formate"] = min(vals["formate"], dg_hcoo) if np.isfinite(vals["formate"]) else dg_hcoo

    finite = {k: v for k, v in vals.items() if np.isfinite(v)}
    if len(finite) < 2:
        return "uncertain", "missing COOH*/OCHO*-family descriptors"

    delta = float(finite["formate"] - finite["CO"])
    if delta > 0.10:
        return "CO", f"COOH* proxy is lower than OCHO*/HCOO* by {delta:.3f} eV"
    if delta < -0.10:
        return "formate", f"OCHO*/HCOO* proxy is lower than COOH* by {abs(delta):.3f} eV"
    return "mixed_or_uncertain", f"COOH* and OCHO*/HCOO* proxies differ by only {abs(delta):.3f} eV"


def _classify_her_risk(dg_h: float) -> tuple[str, float, str]:
    if not np.isfinite(dg_h):
        return "unknown", float("nan"), "HER guardrail not available"
    a = abs(float(dg_h))
    # ΔG_H close to zero is good for HER, therefore bad for CO2RR selectivity.
    if a <= 0.25:
        return "high", 1.0, f"|ΔG_H|={a:.3f} eV; HER-competitive"
    if a <= 0.55:
        return "medium", 0.5, f"|ΔG_H|={a:.3f} eV; moderate HER competition"
    return "low", 0.0, f"|ΔG_H|={a:.3f} eV; weaker HER competition"


def _classify_orr_risk(dg_ooh: float, dg_o: float, dg_oh: float) -> tuple[str, float, str]:
    vals = [v for v in (dg_ooh, dg_o, dg_oh) if np.isfinite(v)]
    if not vals:
        return "unknown", float("nan"), "oxygen-intermediate descriptors not available"
    min_o = float(min(vals))
    # This is a screening-only oxygen-affinity risk index.  Strongly stabilized
    # oxygen intermediates imply O2 sensitivity under air-fed dilute CO2.
    if min_o <= 0.0:
        return "high", 1.0, f"best oxygen-intermediate ΔG={min_o:.3f} eV; O2-sensitive"
    if min_o <= 0.80:
        return "medium", 0.5, f"best oxygen-intermediate ΔG={min_o:.3f} eV; moderate O2 sensitivity"
    return "low", 0.0, f"best oxygen-intermediate ΔG={min_o:.3f} eV; lower O2 sensitivity"


def build_co2rr_air_summary(
    co2rr_df: pd.DataFrame,
    oxygen_df: pd.DataFrame | None = None,
    her_guard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a one-row CO2RR-air competition summary.

    This helper is deliberately post-processing only.  It does not call or alter
    HER, OER, VOC, or CO2RR calculation functions.  It consumes already-produced
    CO2RR rows, optional OER/oxygen rows, and optional CO2RR HER guardrail metadata.
    """
    co2 = _valid_rows(co2rr_df, family="co2rr")
    oxy = _valid_rows(oxygen_df, family="oxygen") if oxygen_df is not None else pd.DataFrame()

    dg_cooh = _best_ads_energy(co2, "COOH")
    dg_co = _best_ads_energy(co2, "CO")
    dg_ocho = _best_ads_energy(co2, "OCHO")
    dg_hcoo = _best_ads_energy(co2, "HCOO")

    dg_ooh = _best_ads_energy(oxy, "OOH")
    dg_o = _best_ads_energy(oxy, "O")
    dg_oh = _best_ads_energy(oxy, "OH")
    dg_h = _extract_her_delta_g(her_guard)

    pathway, pathway_basis = _classify_pathway(dg_cooh, dg_ocho, dg_hcoo)
    her_risk, her_score, her_basis = _classify_her_risk(dg_h)
    orr_risk, orr_score, orr_basis = _classify_orr_risk(dg_ooh, dg_o, dg_oh)

    if orr_risk == "high":
        air_tol = "O2-sensitive"
    elif her_risk == "high":
        air_tol = "HER-limited"
    elif pathway in {"CO", "formate"} and orr_risk in {"low", "medium"} and her_risk in {"low", "medium"}:
        air_tol = "conditionally_air_compatible"
    else:
        air_tol = "uncertain"

    co_poisoning_risk = "unknown"
    co_poisoning_basis = "CO* descriptor unavailable"
    if np.isfinite(dg_co):
        if dg_co <= -1.00:
            co_poisoning_risk = "high"
        elif dg_co <= -0.50:
            co_poisoning_risk = "medium"
        else:
            co_poisoning_risk = "low"
        co_poisoning_basis = f"best CO* ΔG={dg_co:.3f} eV"

    return {
        "co2rr_pathway_preference": pathway,
        "co2rr_pathway_basis": pathway_basis,
        "her_competition_risk": her_risk,
        "her_competition_score": her_score,
        "her_competition_basis": her_basis,
        "orr_competition_risk": orr_risk,
        "orr_competition_score": orr_score,
        "orr_competition_basis": orr_basis,
        "co_poisoning_risk": co_poisoning_risk,
        "co_poisoning_basis": co_poisoning_basis,
        "air_tolerance_index": air_tol,
        "ΔG_COOH_best (eV)": dg_cooh,
        "ΔG_CO_best (eV)": dg_co,
        "ΔG_OCHO_best (eV)": dg_ocho,
        "ΔG_HCOO_best (eV)": dg_hcoo,
        "ΔG_H_guardrail (eV)": dg_h,
        "ΔG_OOH_best (eV)": dg_ooh,
        "ΔG_O_best (eV)": dg_o,
        "ΔG_OH_best (eV)": dg_oh,
        "COOH_best_site": _best_ads_site(co2, "COOH"),
        "CO_best_site": _best_ads_site(co2, "CO"),
        "OCHO_best_site": _best_ads_site(co2, "OCHO"),
        "HCOO_best_site": _best_ads_site(co2, "HCOO"),
        "OOH_best_site": _best_ads_site(oxy, "OOH"),
        "O_best_site": _best_ads_site(oxy, "O"),
        "OH_best_site": _best_ads_site(oxy, "OH"),
        "n_co2rr_valid_rows": int(len(co2)),
        "n_oxygen_valid_rows": int(len(oxy)),
        "screening_scope": "CO2RR + HER guardrail + ORR/OER oxygen-intermediate competition",
        "screening_note": (
            "Heuristic air-fed/dilute-CO2 screening index.  It is not a kinetic ORR model "
            "and should be benchmarked against trend-level metal/product classes."
        ),
    }


def co2rr_air_summary_to_frame(summary: Mapping[str, Any] | None) -> pd.DataFrame:
    if not isinstance(summary, Mapping) or not summary:
        return pd.DataFrame()
    return pd.DataFrame([dict(summary)])


def annotate_co2rr_air_summary(df: pd.DataFrame, summary: Mapping[str, Any] | None) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or not isinstance(summary, Mapping):
        return df
    out = df.copy()
    for key in (
        "co2rr_pathway_preference",
        "her_competition_risk",
        "orr_competition_risk",
        "co_poisoning_risk",
        "air_tolerance_index",
    ):
        out[key] = summary.get(key, "")
    return out
