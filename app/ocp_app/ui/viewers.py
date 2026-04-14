import re
import uuid

import numpy as np
import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components

from ocp_app.core.structure_ops import atoms_to_xyz_string

def _render_min_dist_panel(rep):
    """
    Based on validate_structure() results.
     Displays only OK/WARNING/CRITICAL levels (internal indices hidden).
    """
    ng = getattr(rep, "nearest_global", None) or {}
    nbp = getattr(rep, "nearest_by_pair", None) or []

    if (not ng) or (not nbp):
        st.info("Min interatomic distance: not available.")
        return

    g_pair = str(ng.get("pair", ""))
    g_dmin = ng.get("d_min", None)
    if g_dmin is None:
        st.info("Min interatomic distance: not available.")
        return

    g_flag = "ok"
    for p in nbp:
        if getattr(p, "pair", "") == g_pair:
            g_flag = str(getattr(p, "flag", "ok"))
            break

    level = g_flag.upper()
    line = (
        f"Min interatomic distance: **{float(g_dmin):.3f} Å** (**{g_pair}**)  \n"
        f"Level: **{level}** (source: validate_structure)"
    )

    if g_flag.lower().startswith("crit"):
        st.error(line)
    elif g_flag.lower().startswith("warn"):
        st.warning(line)
    else:
        st.success(line)

    rows = []
    for p in nbp:
        rows.append({
            "pair": getattr(p, "pair", ""),
            "d_min (Å)": float(getattr(p, "d_min", np.nan)),
            "d_mean (Å)": float(getattr(p, "d_mean", np.nan)),
            "n_bonds": int(getattr(p, "n_bonds", 0)),
            "level": str(getattr(p, "flag", "ok")).upper(),
        })
    df_pair = pd.DataFrame(rows).sort_values("d_min (Å)")

    with st.expander("Min distances by element pair (OK/WARNING/CRITICAL)", expanded=False):
        st.dataframe(df_pair, use_container_width=True)

def show_atoms_3d(atoms, height=420, width=700, tag="view"):
    atoms = atoms.copy()
    try:
        atoms.wrap()
    except Exception:
        pass

    xyz_str = atoms_to_xyz_string(atoms)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_str, "xyz")
    view.setStyle({"stick": {}, "sphere": {"radius": 0.3}})
    view.zoomTo()

    html = view._make_html()

    new_id = f"mol_{tag}_{uuid.uuid4().hex}"
    m = re.search(r'id="([^"]+)"', html)
    if m:
        old_id = m.group(1)
        html = html.replace(f'id="{old_id}"', f'id="{new_id}"', 1)
        html = html.replace(old_id, new_id)

    components.html(html, height=height, width=width)

