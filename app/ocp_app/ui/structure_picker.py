from __future__ import annotations

import inspect
from typing import Dict, Iterable, List, Sequence

import numpy as np
import streamlit as st
from ase import Atoms

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    go = None
    HAS_PLOTLY = False

from ocp_app.core.ads_sites import AdsSite


_DEPTH_ORDER = {"surface": 0, "subsurface": 1, "bulk_like": 2}
_SITE_COLORS = {"ontop": "#1f77b4", "bridge": "#ff7f0e", "hollow": "#2ca02c"}


def _canonical_site_kind(kind: str) -> str:
    value = str(kind).strip().lower()
    return "hollow" if value in {"fcc", "hcp"} else value


def _project_fractional_xy(atoms: Atoms, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project wrapped fractional xy coordinates to an in-plane Cartesian plot."""
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pos = np.asarray(positions, dtype=float)
    try:
        frac = np.linalg.solve(cell.T, pos.T).T
    except Exception:
        frac = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
        if len(frac) != len(pos):
            raise ValueError("Failed to obtain fractional coordinates for the structure picker.")
    frac[:, 0] %= 1.0
    frac[:, 1] %= 1.0

    avec = cell[0]
    bvec = cell[1]
    a_len = max(float(np.linalg.norm(avec)), 1e-12)
    bx = float(np.dot(bvec, avec) / a_len)
    by_sq = max(float(np.dot(bvec, bvec) - bx * bx), 0.0)
    by = float(np.sqrt(by_sq))
    x = frac[:, 0] * a_len + frac[:, 1] * bx
    y = frac[:, 1] * by
    return x, y, frac


def _cell_outline(atoms: Atoms) -> tuple[list[float], list[float]]:
    cell = np.asarray(atoms.get_cell(), dtype=float)
    a_len = max(float(np.linalg.norm(cell[0])), 1e-12)
    bx = float(np.dot(cell[1], cell[0]) / a_len)
    by = float(np.sqrt(max(float(np.dot(cell[1], cell[1]) - bx * bx), 0.0)))
    return [0.0, a_len, a_len + bx, bx, 0.0], [0.0, 0.0, by, by, 0.0]


def _selection_points(event) -> list:
    if event is None:
        return []
    try:
        selection = event.selection
    except Exception:
        try:
            selection = event.get("selection", {})
        except Exception:
            selection = {}
    if selection is None:
        return []
    try:
        points = selection.points
    except Exception:
        try:
            points = selection.get("points", [])
        except Exception:
            points = []
    return list(points or [])


def _point_value(point, key, default=None):
    try:
        return getattr(point, key)
    except Exception:
        try:
            return point.get(key, default)
        except Exception:
            return default


def _point_customdata(point):
    return _point_value(point, "customdata", None)


def _coerce_selected_index(data):
    """Normalize Streamlit/Plotly customdata to a single integer index.

    Depending on Streamlit and Plotly versions, one-point customdata may arrive
    as an int, float, numpy scalar, one-item list/tuple, or mapping.
    """
    if data is None:
        return None
    if isinstance(data, dict):
        for key in ("index", "atom_index", "site_index", "value"):
            if key in data:
                return _coerce_selected_index(data[key])
        return None
    if isinstance(data, (list, tuple, np.ndarray)):
        if len(data) == 0:
            return None
        return _coerce_selected_index(data[0])
    try:
        return int(data)
    except Exception:
        return None


def _plotly_select(fig, *, key: str):
    """Render native Streamlit Plotly point selection when supported."""
    try:
        params = inspect.signature(st.plotly_chart).parameters
    except Exception:
        params = {}
    if "on_select" in params:
        return st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="points",
        ), True
    st.plotly_chart(fig, use_container_width=True, key=key)
    return None, False


def eligible_atom_indices(
    analysis: Dict[str, object],
    *,
    element: str,
    depth: str,
) -> List[int]:
    allowed = {
        "surface": {"surface"},
        "subsurface": {"subsurface"},
        "surface+subsurface": {"surface", "subsurface"},
        "all": {"surface", "subsurface", "bulk_like"},
    }.get(str(depth), {str(depth)})
    out = []
    for env in analysis.get("environments", []) or []:
        if str(env.symbol) == str(element) and str(env.depth_class) in allowed:
            out.append(int(env.index))
    return out


def render_atom_picker(
    atoms: Atoms,
    analysis: Dict[str, object],
    *,
    selectable_indices: Sequence[int],
    selected_index: int | None,
    key: str,
    title: str = "Click an atom",
) -> int | None:
    selectable = sorted(set(int(i) for i in selectable_indices))
    if not selectable:
        st.warning("No atoms satisfy the current element/depth filter.")
        return None

    env_by_idx = analysis.get("environment_by_index", {}) or {}
    selected = int(selected_index) if selected_index is not None and int(selected_index) in selectable else None

    if not HAS_PLOTLY:
        return st.selectbox(
            "Selectable atom index",
            selectable,
            index=(selectable.index(selected) if selected in selectable else 0),
            format_func=lambda i: (
                f"#{i} {atoms[int(i)].symbol} | "
                f"{getattr(env_by_idx.get(int(i)), 'depth_class', '?')} | "
                f"CN={getattr(env_by_idx.get(int(i)), 'coordination_number', '?')}"
            ),
            key=f"{key}_fallback",
        )

    pos = np.asarray(atoms.get_positions(), dtype=float)
    x, y, frac = _project_fractional_xy(atoms, pos)
    all_idx = list(range(len(atoms)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker={"size": 9, "color": "#c5c9d3", "opacity": 0.42, "line": {"width": 0}},
        customdata=[int(i) for i in all_idx],
        text=[f"#{i} {atoms[i].symbol}" for i in all_idx],
        hovertemplate="%{text}<extra>background atom</extra>",
        name="all atoms",
        showlegend=False,
    ))

    sx = [float(x[i]) for i in selectable]
    sy = [float(y[i]) for i in selectable]
    custom = []
    hover = []
    for i in selectable:
        env = env_by_idx.get(int(i))
        custom.append([int(i)])
        hover.append(
            f"atom #{i}<br>element={atoms[i].symbol}<br>"
            f"depth={getattr(env, 'depth_class', '?')}<br>"
            f"layer={getattr(env, 'layer_id', '?')}<br>"
            f"CN={getattr(env, 'coordination_number', '?')}<br>"
            f"frac=({frac[i,0]:.3f}, {frac[i,1]:.3f})"
        )
    fig.add_trace(go.Scatter(
        x=sx,
        y=sy,
        mode="markers+text",
        marker={"size": 15, "color": "#d62728", "opacity": 0.90, "line": {"width": 1.0, "color": "white"}},
        text=[str(i) for i in selectable],
        textposition="top center",
        customdata=[int(i) for i in selectable],
        hovertext=hover,
        hovertemplate="%{hovertext}<extra>selectable atom</extra>",
        name="selectable",
    ))
    if selected is not None:
        fig.add_trace(go.Scatter(
            x=[float(x[selected])],
            y=[float(y[selected])],
            mode="markers",
            marker={"size": 25, "color": "rgba(0,0,0,0)", "line": {"width": 4, "color": "#111111"}, "symbol": "circle-open"},
            hoverinfo="skip",
            showlegend=False,
        ))
    ox, oy = _cell_outline(atoms)
    fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines", line={"color": "#555", "width": 1.5, "dash": "dot"}, hoverinfo="skip", showlegend=False))
    fig.update_layout(
        title=title,
        clickmode="event+select",
        dragmode=False,
        height=520,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis={"title": "surface x (Å)", "scaleanchor": "y", "scaleratio": 1, "showgrid": False},
        yaxis={"title": "surface y (Å)", "showgrid": False},
        legend={"orientation": "h"},
    )
    st.caption("Click one colored point once. Do not drag a selection box.")
    event, clickable = _plotly_select(fig, key=f"{key}_plot")
    for point in reversed(_selection_points(event)):
        idx = _coerce_selected_index(_point_customdata(point))
        if idx in selectable:
            selected = int(idx)
            break

        # Fallback for Plotly versions that omit customdata.  The selectable
        # atoms are trace 1 and point_index refers to the position in that trace.
        curve_number = _point_value(point, "curve_number", _point_value(point, "curveNumber", None))
        point_index = _point_value(
            point,
            "point_index",
            _point_value(point, "pointIndex", _point_value(point, "point_number", None)),
        )
        try:
            if int(curve_number) == 1 and 0 <= int(point_index) < len(selectable):
                selected = int(selectable[int(point_index)])
                break
        except Exception:
            pass
    if not clickable:
        st.caption("This Streamlit version does not expose Plotly click events; use the atom-index selector below.")
        selected = st.selectbox(
            "Selectable atom index",
            selectable,
            index=(selectable.index(selected) if selected in selectable else 0),
            format_func=lambda i: f"#{i} {atoms[int(i)].symbol} | {getattr(env_by_idx.get(int(i)), 'depth_class', '?')}",
            key=f"{key}_legacy_select",
        )
    return selected


def render_adatom_site_picker(
    atoms: Atoms,
    sites: Sequence[AdsSite],
    *,
    selected_site_index: int | None,
    key: str,
) -> int | None:
    if not sites:
        st.warning("No adatom sites satisfy the selected site classes.")
        return None
    selected = int(selected_site_index) if selected_site_index is not None and 0 <= int(selected_site_index) < len(sites) else None

    if not HAS_PLOTLY:
        return st.selectbox(
            "Selectable site",
            list(range(len(sites))),
            index=(selected if selected is not None else 0),
            format_func=lambda i: f"site {i} | {_canonical_site_kind(sites[i].kind)} | support={sites[i].surface_indices}",
            key=f"{key}_fallback",
        )

    pos = np.asarray(atoms.get_positions(), dtype=float)
    x_atom, y_atom, _frac_atom = _project_fractional_xy(atoms, pos)
    site_pos = np.asarray([s.position for s in sites], dtype=float)
    x_site, y_site, frac_site = _project_fractional_xy(atoms, site_pos)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_atom,
        y=y_atom,
        mode="markers",
        marker={"size": 9, "color": "#aeb4bf", "opacity": 0.50},
        text=[f"#{i} {atoms[i].symbol}" for i in range(len(atoms))],
        hovertemplate="%{text}<extra>surface atom</extra>",
        showlegend=False,
    ))

    for kind in ("ontop", "bridge", "hollow"):
        ids = [i for i, s in enumerate(sites) if _canonical_site_kind(s.kind) == kind]
        if not ids:
            continue
        fig.add_trace(go.Scatter(
            x=[float(x_site[i]) for i in ids],
            y=[float(y_site[i]) for i in ids],
            mode="markers+text",
            marker={"size": 15, "color": _SITE_COLORS[kind], "opacity": 0.88, "line": {"width": 1, "color": "white"}},
            text=[str(i) for i in ids],
            textposition="top center",
            customdata=[int(i) for i in ids],
            hovertext=[
                f"site {i}<br>kind={kind}<br>support={sites[i].surface_indices}<br>"
                f"frac=({frac_site[i,0]:.3f}, {frac_site[i,1]:.3f})"
                for i in ids
            ],
            hovertemplate="%{hovertext}<extra>selectable site</extra>",
            name=kind,
        ))
    if selected is not None:
        fig.add_trace(go.Scatter(
            x=[float(x_site[selected])],
            y=[float(y_site[selected])],
            mode="markers",
            marker={"size": 27, "color": "rgba(0,0,0,0)", "line": {"width": 4, "color": "#111111"}, "symbol": "circle-open"},
            hoverinfo="skip",
            showlegend=False,
        ))
    ox, oy = _cell_outline(atoms)
    fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines", line={"color": "#555", "width": 1.5, "dash": "dot"}, hoverinfo="skip", showlegend=False))
    fig.update_layout(
        title="Click an ontop, bridge, or hollow marker",
        clickmode="event+select",
        dragmode=False,
        height=540,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis={"title": "surface x (Å)", "scaleanchor": "y", "scaleratio": 1, "showgrid": False},
        yaxis={"title": "surface y (Å)", "showgrid": False},
        legend={"orientation": "h"},
    )
    st.caption("Click one colored point once. Do not drag a selection box.")
    event, clickable = _plotly_select(fig, key=f"{key}_plot")
    for point in reversed(_selection_points(event)):
        idx = _coerce_selected_index(_point_customdata(point))
        if idx is not None and 0 <= int(idx) < len(sites):
            selected = int(idx)
            break
    if not clickable:
        st.caption("This Streamlit version does not expose Plotly click events; use the site selector below.")
        selected = st.selectbox(
            "Selectable site",
            list(range(len(sites))),
            index=(selected if selected is not None else 0),
            format_func=lambda i: f"site {i} | {_canonical_site_kind(sites[i].kind)} | support={sites[i].surface_indices}",
            key=f"{key}_legacy_select",
        )
    return selected
