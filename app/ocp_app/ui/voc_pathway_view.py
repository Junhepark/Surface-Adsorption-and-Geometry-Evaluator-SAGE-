from __future__ import annotations

"""Streamlit renderer for the SAGE-VOC numerical state-energy map."""

from html import escape
from typing import Mapping

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from ocp_app.core.voc_pathway import pathway_summary_to_frame


_BLUE = (33, 102, 172)
_WHITE = (247, 247, 247)
_RED = (178, 24, 43)
_GRAY = "#E5E7EB"
_GRAY_BORDER = "#94A3B8"


def _fmt_energy(value: object) -> str:
    try:
        val = float(value)
        return f"{val:.3f} eV" if np.isfinite(val) else "N/A"
    except Exception:
        return "N/A"


def _mix_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, float(t)))
    return tuple(int(round((1.0 - t) * x + t * y)) for x, y in zip(a, b))


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, int(v))):02X}" for v in rgb)


def _energy_color(value: object, scale: Mapping[str, object]) -> tuple[str, str, str, bool]:
    """Return strip, border, text colors and clipping status for the B-W-R scale."""
    try:
        val = float(value)
        if not np.isfinite(val):
            raise ValueError
    except Exception:
        return _GRAY, _GRAY_BORDER, "#0F172A", False

    vmin = float(scale.get("vmin_eV", -2.5))
    center = float(scale.get("vcenter_eV", 0.0))
    vmax = float(scale.get("vmax_eV", 2.5))
    clipped = min(vmax, max(vmin, val))
    was_clipped = bool(val < vmin or val > vmax)

    if clipped <= center:
        t = (clipped - vmin) / max(center - vmin, 1e-12)
        rgb = _mix_rgb(_BLUE, _WHITE, t)
    else:
        t = (clipped - center) / max(vmax - center, 1e-12)
        rgb = _mix_rgb(_WHITE, _RED, t)

    fill = _rgb_hex(rgb)
    # Darker border from the same hue family.
    border = _rgb_hex(_mix_rgb(rgb, (15, 23, 42), 0.22))
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    text = "#0F172A"
    return fill, border, text, was_clipped


def render_voc_pathway(summary: Mapping[str, object], *, height: int | None = None) -> None:
    nodes = [dict(x) for x in (summary.get("nodes", []) or [])]
    edges = [dict(x) for x in (summary.get("edges", []) or [])]
    if not nodes:
        st.info("No registered VOC states are available for this route.")
        return

    scale = dict(summary.get("energy_scale", {}) or {})
    vmin = float(scale.get("vmin_eV", -2.5))
    center = float(scale.get("vcenter_eV", 0.0))
    vmax = float(scale.get("vmax_eV", 2.5))

    max_level = max(int(n.get("level", 0)) for n in nodes)
    lanes = sorted({int(n.get("lane", 0)) for n in nodes})
    lane_index = {lane: i for i, lane in enumerate(lanes)}
    lane_labels_raw = summary.get("lane_labels", {}) or {}
    lane_labels = {int(k): str(v) for k, v in lane_labels_raw.items()}

    node_w, node_h = 176, 72
    x_gap, y_gap = 226, 132
    margin_x, margin_y = 34, 58
    width = margin_x * 2 + node_w + max_level * x_gap
    svg_height = margin_y * 2 + node_h + max(0, len(lanes) - 1) * y_gap
    height = int(height or min(590, max(240, svg_height + 90)))

    positions: dict[str, tuple[float, float]] = {}
    for node in nodes:
        x = margin_x + int(node.get("level", 0)) * x_gap
        y = margin_y + lane_index[int(node.get("lane", 0))] * y_gap
        positions[str(node.get("id", ""))] = (x, y)

    lane_svg: list[str] = []
    for lane in lanes:
        y = margin_y + lane_index[lane] * y_gap - 18
        label = escape(lane_labels.get(lane, ""))
        if label:
            lane_svg.append(
                f'<text x="{margin_x}" y="{y}" font-size="12" font-weight="700" fill="#475569">{label}</text>'
            )

    edge_svg: list[str] = []
    for edge in edges:
        # Sequential state links are intentionally not drawn: the displayed
        # values are independent state-level adsorption proxies, not reaction-step energies.
        # Keep only auxiliary H*/OH* context as a weak, arrow-free dashed guide.
        edge_type = str(edge.get("edge_type", ""))
        if edge_type != "co_reactant_support":
            continue
        src = positions.get(str(edge.get("from", "")))
        dst = positions.get(str(edge.get("to", "")))
        if src is None or dst is None:
            continue
        x1, y1 = src[0] + node_w / 2, src[1] + node_h
        x2, y2 = dst[0] + node_w / 2, dst[1]
        mid_y = (y1 + y2) / 2
        path = f"M {x1:.1f} {y1:.1f} C {x1:.1f} {mid_y:.1f}, {x2:.1f} {mid_y:.1f}, {x2:.1f} {y2:.1f}"
        edge_svg.append(
            f'<path d="{path}" fill="none" stroke="#94A3B8" stroke-width="1.4" stroke-dasharray="5 5"/>'
        )

    node_svg: list[str] = []
    for node in nodes:
        node_id = str(node.get("id", ""))
        x, y = positions[node_id]
        energy_value = node.get("selected_energy_eV")
        strip, border, fg, was_clipped = _energy_color(energy_value, scale)
        label = escape(str(node.get("label", node_id)))
        energy = escape(_fmt_energy(energy_value))
        clipped_label = "clipped" if was_clipped else ""

        node_svg.append(
            f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="10" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.3"/>'
        )
        node_svg.append(
            f'<path d="M {x+10} {y} H {x+node_w-10} Q {x+node_w} {y} {x+node_w} {y+10} V {y+12} H {x} V {y+10} Q {x} {y} {x+10} {y} Z" fill="{strip}"/>'
        )
        node_svg.append(
            f'<text x="{x + node_w/2}" y="{y + 34}" text-anchor="middle" font-size="15" font-weight="700" fill="{fg}">{label}</text>'
        )
        node_svg.append(
            f'<text x="{x + node_w/2}" y="{y + 56}" text-anchor="middle" font-size="12.5" font-weight="600" fill="{fg}">{energy}</text>'
        )
        if clipped_label:
            node_svg.append(
                f'<text x="{x + node_w - 9}" y="{y + node_h - 7}" text-anchor="end" font-size="9.5" font-weight="600" fill="#64748B">{clipped_label}</text>'
            )

    title = escape(str(summary.get("route_label", summary.get("route_key", "VOC state energy map"))))
    description = escape(str(summary.get("route_description", "")))
    html = f"""
    <div style="font-family:Arial,sans-serif;border:1px solid #E2E8F0;border-radius:10px;background:#FFFFFF;overflow-x:auto;padding:8px 8px 6px 8px;">
      <div style="font-weight:700;color:#0F172A;padding:4px 8px 0 8px;">{title}</div>
      <div style="font-size:11px;color:#64748B;padding:2px 8px 0 8px;">{description}</div>
      <svg width="{width}" height="{svg_height}" viewBox="0 0 {width} {svg_height}" role="img" aria-label="VOC state energy map">
        {''.join(lane_svg)}
        {''.join(edge_svg)}
        {''.join(node_svg)}
      </svg>
      <div style="padding:4px 14px 2px 14px;">
        <div style="height:12px;border-radius:6px;border:1px solid #CBD5E1;background:linear-gradient(90deg,#2166AC 0%,#F7F7F7 50%,#B2182B 100%);"></div>
        <div style="display:flex;justify-content:space-between;font-size:10.5px;color:#475569;margin-top:3px;">
          <span>{vmin:.1f} eV</span><span>{center:.1f} eV</span><span>{vmax:+.1f} eV</span>
        </div>
      </div>
      <div style="font-size:11px;color:#64748B;padding:4px 8px 8px 8px;">
        Blue → white → red represents the selected QA-valid ΔE_proxy value only. Gray means no QA-valid numerical value. Values outside the displayed range are clipped and labeled. Dashed auxiliary links provide context only; no reaction-step energies or product predictions are implied.
      </div>
    </div>
    """
    components.html(html, height=height, scrolling=True)


def render_pathway_support_table(summary: Mapping[str, object]) -> None:
    frame = pathway_summary_to_frame(summary)
    if frame.empty:
        st.info("No VOC state-energy audit rows are available.")
        return
    preferred = [
        "record_type",
        "id",
        "label",
        "from",
        "to",
        "state",
        "selected_energy_eV",
        "selected_energy_column",
        "energy_available",
        "attempt_count",
        "site_label",
        "qa",
        "qa_note",
        "initial_structure_cif",
        "structure_cif",
    ]
    cols = [c for c in preferred if c in frame.columns]
    st.dataframe(frame[cols] if cols else frame, use_container_width=True)
