# run_history.py
# Session-only run history + sidebar UI (Streamlit)
# - Stores up to N recent runs (default 10) in st.session_state
# - Supports memo/star/tag highlight
# - Provides simple "Run details" rendering and CSV download from stored bytes
#
# Design choices
# - "Session-only": uses st.session_state only (no localStorage, no disk persistence by default)
# - No API keys or secrets are stored
# - Payload kept compact: summary + csv_bytes (+ optional prepared_cif_bytes if you provide it)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# ---------------------------
# State keys (session_state)
# ---------------------------
HISTORY_KEY = "run_history_items"
HISTORY_SELECTED_KEY = "run_history_selected_id"
HISTORY_MAX_KEY = "run_history_max_items"


# ---------------------------
# Minimal record schema
# ---------------------------
@dataclass
class HistoryRecord:
    run_id: str
    created_at_iso: str  # ISO8601 string
    label: str           # e.g., "NiO | atoms=210" or custom
    mode_label: str      # "HER" / "CO2RR"
    mtype: str           # "metal" / "oxide" (or others)
    relax_mode: str      # "Fast"/"Normal"/"Tight"
    model: str           # meta.MODEL
    device: str          # meta.DEVICE

    summary_line: str    # 1-line summary for list UI
    warnings: Dict[str, Any]

    # Optional artifacts (session-only)
    csv_name: Optional[str] = None
    csv_bytes: Optional[bytes] = None

    prepared_cif_name: Optional[str] = None
    prepared_cif_bytes: Optional[bytes] = None

    # User annotations
    starred: bool = False
    tag: str = "none"    # "none" | "red" | "yellow" | "green" | "blue" etc.
    memo: str = ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_history_state(max_items: int = 10) -> None:
    """Initialize session_state keys for history."""
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []  # list[dict] (HistoryRecord serialized)
    if HISTORY_SELECTED_KEY not in st.session_state:
        st.session_state[HISTORY_SELECTED_KEY] = None
    if HISTORY_MAX_KEY not in st.session_state:
        st.session_state[HISTORY_MAX_KEY] = int(max_items)


def _coerce_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_str(x, default: str = "") -> str:
    try:
        s = str(x)
        return s if s is not None else default
    except Exception:
        return default


def _pick_best_metric_from_df(df: pd.DataFrame, is_her: bool) -> Tuple[str, Optional[float], str]:
    """
    Returns (metric_name, metric_value, site_label_string)
    - HER: prefer ΔG_H(U,pH) then ΔG_H
    - CO2RR: prefer ΔG_ads then ΔE_ads_user
    """
    if df is None or df.empty:
        return ("n/a", None, "n/a")

    def _site(i) -> str:
        if "site_label" in df.columns:
            return _safe_str(df.loc[i, "site_label"])
        if "site" in df.columns:
            return _safe_str(df.loc[i, "site"])
        return _safe_str(i)

    if is_her:
        for col in ["ΔG_H(U,pH) (eV)", "ΔG_H (eV)", "dG_H(U,pH) (eV)", "dG_H (eV)"]:
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce")
                if v.notna().any():
                    idx = (v.abs()).idxmin()
                    return (col, float(v.loc[idx]), _site(idx))
        return ("ΔG_H", None, "n/a")

    # CO2RR
    for col in ["ΔG_ads (eV)", "ΔE_ads_user (eV)", "ΔG_ads", "ΔE_ads_user"]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            if v.notna().any():
                # If QA exists, keep only ok/migrated (consistent with your policy)
                mask = pd.Series(True, index=df.index)
                if "qa" in df.columns:
                    qa = df["qa"].astype(str).str.strip().str.lower()
                    mask &= qa.isin(["ok", "migrated"])
                vv = v[mask].dropna()
                if vv.empty:
                    vv = v.dropna()
                idx = vv.idxmin()
                return (col, float(v.loc[idx]), _site(idx))
    return ("ΔE_ads_user", None, "n/a")


def _build_warnings_from_df(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"n_total": 0}

    out: Dict[str, Any] = {"n_total": int(len(df))}
    if "migrated" in df.columns:
        out["n_migrated"] = int(pd.to_numeric(df["migrated"], errors="coerce").fillna(0).sum())
    if "is_duplicate" in df.columns:
        out["n_duplicate"] = int(pd.to_numeric(df["is_duplicate"], errors="coerce").fillna(0).sum())
    if "reliability" in df.columns:
        rel = df["reliability"].astype(str).str.lower()
        out["n_unreliable"] = int((~rel.eq("reliable")).sum())
    if "qa" in df.columns:
        qa = df["qa"].astype(str).str.lower().value_counts(dropna=False).to_dict()
        out["qa_counts"] = qa
    if "qc_flags" in df.columns:
        flags = df["qc_flags"].astype(str).fillna("")
        out["n_qc_flagged"] = int((flags.str.strip() != "").sum())
    if "migration_type" in df.columns:
        mt = df["migration_type"].astype(str).str.lower()
        out["migration_type_counts"] = mt.value_counts(dropna=False).to_dict()
    if "final_site_kind" in df.columns:
        fk = df["final_site_kind"].astype(str).str.lower()
        out["final_site_kind_counts"] = fk.value_counts(dropna=False).to_dict()
    return out


def _serialize_record(r: HistoryRecord) -> Dict[str, Any]:
    return {
        "run_id": r.run_id,
        "created_at_iso": r.created_at_iso,
        "label": r.label,
        "mode_label": r.mode_label,
        "mtype": r.mtype,
        "relax_mode": r.relax_mode,
        "model": r.model,
        "device": r.device,
        "summary_line": r.summary_line,
        "warnings": r.warnings,
        "csv_name": r.csv_name,
        "csv_bytes": r.csv_bytes,
        "prepared_cif_name": r.prepared_cif_name,
        "prepared_cif_bytes": r.prepared_cif_bytes,
        "starred": bool(r.starred),
        "tag": r.tag,
        "memo": r.memo,
    }


def _deserialize_record(d: Dict[str, Any]) -> HistoryRecord:
    return HistoryRecord(
        run_id=_safe_str(d.get("run_id")),
        created_at_iso=_safe_str(d.get("created_at_iso")),
        label=_safe_str(d.get("label")),
        mode_label=_safe_str(d.get("mode_label")),
        mtype=_safe_str(d.get("mtype")),
        relax_mode=_safe_str(d.get("relax_mode")),
        model=_safe_str(d.get("model")),
        device=_safe_str(d.get("device")),
        summary_line=_safe_str(d.get("summary_line")),
        warnings=d.get("warnings") if isinstance(d.get("warnings"), dict) else {},
        csv_name=d.get("csv_name"),
        csv_bytes=d.get("csv_bytes"),
        prepared_cif_name=d.get("prepared_cif_name"),
        prepared_cif_bytes=d.get("prepared_cif_bytes"),
        starred=bool(d.get("starred", False)),
        tag=_safe_str(d.get("tag"), "none"),
        memo=_safe_str(d.get("memo"), ""),
    )


def list_history() -> List[HistoryRecord]:
    ensure_history_state()
    items = st.session_state.get(HISTORY_KEY) or []
    out: List[HistoryRecord] = []
    for d in items:
        if isinstance(d, dict):
            out.append(_deserialize_record(d))
    return out


def get_selected_run_id() -> Optional[str]:
    ensure_history_state()
    v = st.session_state.get(HISTORY_SELECTED_KEY)
    return str(v) if v else None


def select_run(run_id: Optional[str]) -> None:
    ensure_history_state()
    st.session_state[HISTORY_SELECTED_KEY] = run_id


def clear_history() -> None:
    ensure_history_state()
    st.session_state[HISTORY_KEY] = []
    st.session_state[HISTORY_SELECTED_KEY] = None


def delete_run(run_id: str) -> None:
    ensure_history_state()
    items = st.session_state.get(HISTORY_KEY) or []
    items2 = [d for d in items if isinstance(d, dict) and str(d.get("run_id")) != str(run_id)]
    st.session_state[HISTORY_KEY] = items2
    if st.session_state.get(HISTORY_SELECTED_KEY) == run_id:
        st.session_state[HISTORY_SELECTED_KEY] = None


def update_run_annotation(run_id: str, *, memo: Optional[str] = None, starred: Optional[bool] = None, tag: Optional[str] = None) -> None:
    ensure_history_state()
    items = st.session_state.get(HISTORY_KEY) or []
    for d in items:
        if not isinstance(d, dict):
            continue
        if str(d.get("run_id")) != str(run_id):
            continue
        if memo is not None:
            d["memo"] = str(memo)
        if starred is not None:
            d["starred"] = bool(starred)
        if tag is not None:
            d["tag"] = str(tag)
        break
    st.session_state[HISTORY_KEY] = items


def add_history_record(r: HistoryRecord, max_items: Optional[int] = None) -> None:
    """
    Insert record to the front (most recent first), enforce max length.
    """
    ensure_history_state()
    max_keep = int(max_items) if max_items is not None else int(st.session_state.get(HISTORY_MAX_KEY, 10))

    items = st.session_state.get(HISTORY_KEY) or []

    # De-dup by run_id (if any)
    items = [d for d in items if isinstance(d, dict) and str(d.get("run_id")) != str(r.run_id)]

    items.insert(0, _serialize_record(r))
    items = items[:max_keep]

    st.session_state[HISTORY_KEY] = items
    st.session_state[HISTORY_SELECTED_KEY] = r.run_id


def make_history_record_from_last_run(
    *,
    run_id: str,
    last_run: Dict[str, Any],
    label: str,
    relax_mode: str,
    model: str,
    device: str,
    df: Optional[pd.DataFrame],
    csv_bytes: Optional[bytes] = None,
    csv_name: Optional[str] = None,
    prepared_cif_bytes: Optional[bytes] = None,
    prepared_cif_name: Optional[str] = None,
) -> HistoryRecord:
    """
    Build a HistoryRecord from the "last_run" object in home.py.

    Notes:
    - You should pass model/device extracted from your meta dict
    - You can optionally attach csv_bytes and prepared_cif_bytes (session-only)
    """
    is_her = bool(last_run.get("is_her", False))
    mode_label = _safe_str(last_run.get("mode_label"), "HER" if is_her else "CO2RR")
    mtype = _safe_str(last_run.get("mtype", ""))

    metric_name, metric_val, site_lbl = _pick_best_metric_from_df(df if isinstance(df, pd.DataFrame) else pd.DataFrame(), is_her=is_her)
    warn = _build_warnings_from_df(df if isinstance(df, pd.DataFrame) else None)

    # compact warning badges
    badges = []
    if "n_migrated" in warn and _coerce_int(warn["n_migrated"]) > 0:
        badges.append(f"migr={_coerce_int(warn['n_migrated'])}")
    if "n_duplicate" in warn and _coerce_int(warn["n_duplicate"]) > 0:
        badges.append(f"dup={_coerce_int(warn['n_duplicate'])}")
    if "n_unreliable" in warn and _coerce_int(warn["n_unreliable"]) > 0:
        badges.append(f"unrel={_coerce_int(warn['n_unreliable'])}")
    if "n_qc_flagged" in warn and _coerce_int(warn["n_qc_flagged"]) > 0:
        badges.append(f"qc={_coerce_int(warn['n_qc_flagged'])}")
    badge_str = (", ".join(badges)) if badges else "ok"

    val_str = "n/a" if metric_val is None else f"{metric_val:.3f}"
    summary = f"{mode_label} | best: {site_lbl} | {metric_name}={val_str} | {badge_str}"

    return HistoryRecord(
        run_id=run_id,
        created_at_iso=_utc_now_iso(),
        label=label,
        mode_label=mode_label,
        mtype=mtype,
        relax_mode=relax_mode,
        model=model,
        device=device,
        summary_line=summary,
        warnings=warn,
        csv_name=csv_name,
        csv_bytes=csv_bytes,
        prepared_cif_name=prepared_cif_name,
        prepared_cif_bytes=prepared_cif_bytes,
        starred=False,
        tag="none",
        memo="",
    )


# ---------------------------
# UI renderers
# ---------------------------
_TAG_OPTIONS = ["none", "red", "yellow", "green", "blue", "purple"]


def render_history_sidebar(*, title: str = "History (session-only)", max_items: int = 10) -> None:
    """
    Render a compact history list in sidebar.

    Intended usage:
        with st.sidebar:
            render_history_sidebar()
    """
    ensure_history_state(max_items=max_items)

    st.caption("Recent runs (max 10). Click to open details.")
    st.caption("This history is stored only for the current session and will be cleared on session reset/app restart.")

    items = list_history()
    if not items:
        st.caption("No runs yet.")
        return

    # Clear button
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Clear", key="hist_clear_btn"):
            clear_history()
            st.rerun()
    with colB:
        # placeholder for future batch controls
        st.write("")

    selected = get_selected_run_id()
    for i, r in enumerate(items):
        star = "★" if r.starred else " "
        tag = f"[{r.tag}]" if r.tag and r.tag != "none" else ""
        btn_label = f"{star} {tag} {r.label} — {r.summary_line}"

        if st.button(btn_label, key=f"hist_item_{i}_{r.run_id}"):
            select_run(r.run_id)
            st.rerun()


def render_selected_run_details(*, show_downloads: bool = True) -> Optional[HistoryRecord]:
    """
    Render details panel for the selected run.
    Return the selected HistoryRecord (or None).
    """
    ensure_history_state()
    rid = get_selected_run_id()
    if not rid:
        st.info("Select a run from the sidebar history.")
        return None

    items = list_history()
    rec = next((x for x in items if x.run_id == rid), None)
    if rec is None:
        st.warning("Selected run not found (history may have been cleared).")
        return None

    st.markdown("### Run details")
    cols = st.columns([1.2, 1.0, 0.8, 0.8])
    cols[0].write(f"**Label**: {rec.label}")
    cols[1].write(f"**Mode**: {rec.mode_label} / {rec.mtype}")
    cols[2].write(f"**Relax**: {rec.relax_mode}")
    cols[3].write(f"**When**: {rec.created_at_iso}")

    st.write(f"**Summary**: {rec.summary_line}")

    with st.expander("Warnings / QC", expanded=False):
        st.json(rec.warnings or {})

    # Annotations
    st.markdown("#### Annotations")
    c1, c2, c3 = st.columns([0.8, 1.0, 2.2])
    with c1:
        star_new = st.checkbox("Star", value=bool(rec.starred), key=f"hist_star_{rec.run_id}")
    with c2:
        tag_new = st.selectbox("Tag", _TAG_OPTIONS, index=_TAG_OPTIONS.index(rec.tag) if rec.tag in _TAG_OPTIONS else 0, key=f"hist_tag_{rec.run_id}")
    with c3:
        memo_new = st.text_input("Memo", value=rec.memo or "", key=f"hist_memo_{rec.run_id}")

    # Apply annotation updates
    if st.button("Save annotations", key=f"hist_save_{rec.run_id}"):
        update_run_annotation(rec.run_id, memo=memo_new, starred=star_new, tag=tag_new)
        st.success("Saved.")
        st.rerun()

    # Download artifacts (if stored)
    if show_downloads:
        st.markdown("#### Session artifacts")
        d1, d2, d3 = st.columns([1, 1, 1])
        with d1:
            if rec.csv_bytes:
                st.download_button(
                    "Download CSV (stored)",
                    data=rec.csv_bytes,
                    file_name=rec.csv_name or "results.csv",
                    mime="text/csv",
                    key=f"hist_dl_csv_{rec.run_id}",
                )
            else:
                st.caption("CSV not stored.")
        with d2:
            if rec.prepared_cif_bytes:
                st.download_button(
                    "Download Prepared CIF (stored)",
                    data=rec.prepared_cif_bytes,
                    file_name=rec.prepared_cif_name or "prepared_structure.cif",
                    mime="chemical/x-cif",
                    key=f"hist_dl_cif_{rec.run_id}",
                )
            else:
                st.caption("CIF not stored.")
        with d3:
            if st.button("Delete this run", key=f"hist_del_{rec.run_id}"):
                delete_run(rec.run_id)
                st.rerun()

    return rec
