"""Rigid-body orientation helpers for SAGE CO2RR adsorbate templates.

The helpers in this module define *initial seed geometries*.  They do not claim
that the same adsorption geometry is the relaxed minimum on every surface.
Their purpose is narrower: keep the registry-defined anchor at the surface and
place the remaining molecular body in the vacuum-facing hemisphere without
folding or distorting internal bond geometry.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from ase import Atoms

from ocp_app.core.co2rr_registry import (
    get_co2rr_adsorbate_spec,
    is_supported_co2rr_adsorbate,
)

_EPS = 1.0e-10


def _unit(v: np.ndarray, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < _EPS:
        a = np.asarray(fallback, dtype=float).reshape(3)
        n = float(np.linalg.norm(a))
    return a / max(n, _EPS)


def _indices(symbols: Iterable[str], symbol: str) -> list[int]:
    return [i for i, s in enumerate(symbols) if str(s).upper() == str(symbol).upper()]


def co2rr_anchor_indices(
    atoms: Atoms,
    adsorbate: str,
    anchor_mode_override: str | None = None,
) -> tuple[tuple[int, ...], str]:
    """Return registry-defined anchor atom indices and a human-readable mode."""
    key = str(adsorbate or "").replace("*", "").strip().upper()
    if not is_supported_co2rr_adsorbate(key):
        raise KeyError(f"Unsupported CO2RR adsorbate: {adsorbate!r}")

    spec = get_co2rr_adsorbate_spec(key)
    symbols = atoms.get_chemical_symbols()
    c_idx = _indices(symbols, "C")
    o_idx = _indices(symbols, "O")
    mode = str(anchor_mode_override or spec.anchor_mode)

    if mode == "o_o_midpoint":
        if len(o_idx) < 2:
            raise ValueError(f"{key} requires two O atoms for bidentate anchoring")
        return (int(o_idx[0]), int(o_idx[1])), "O,O midpoint"

    if mode in {"o_atom", "o_min_z"}:
        if not o_idx:
            raise ValueError(f"{key} requires an O atom for O anchoring")
        if mode == "o_min_z" and len(o_idx) > 1:
            pos = np.asarray(atoms.get_positions(), dtype=float)
            oi = int(o_idx[int(np.argmin(pos[np.asarray(o_idx, dtype=int), 2]))])
        else:
            oi = int(o_idx[0])
        return (oi,), "O atom"

    if not c_idx:
        raise ValueError(f"{key} requires a C atom for C anchoring")
    return (int(c_idx[0]),), "C atom"


def _body_indices(n_atoms: int, anchor_indices: tuple[int, ...]) -> list[int]:
    anchors = set(int(i) for i in anchor_indices)
    return [i for i in range(int(n_atoms)) if i not in anchors]


def _mean_unit_vectors(pos: np.ndarray, anchor: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    vecs = []
    for i in indices:
        v = np.asarray(pos[int(i)] - anchor, dtype=float)
        n = float(np.linalg.norm(v))
        if np.isfinite(n) and n > _EPS:
            vecs.append(v / n)
    if not vecs:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return np.sum(np.asarray(vecs, dtype=float), axis=0)


def _choose_outward_and_secondary(
    pos: np.ndarray,
    symbols: list[str],
    key: str,
    anchor_indices: tuple[int, ...],
    anchor: np.ndarray,
    anchor_mode_override: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a vacuum-facing body vector and an azimuth reference.

    The outward vector uses the *whole molecular body* rather than a single
    nearest atom.  This is essential for planar radicals such as COOH and CHO:
    aligning only one C--O bond can rotate the other substituents toward the
    slab even though the selected anchor itself is correct.
    """
    body = _body_indices(len(pos), anchor_indices)
    c_idx = _indices(symbols, "C")
    o_idx = _indices(symbols, "O")
    h_idx = _indices(symbols, "H")
    mode = str(anchor_mode_override or get_co2rr_adsorbate_spec(key).anchor_mode)

    # Bidentate formate: O--O is parallel to the surface; C/H point to vacuum.
    if mode == "o_o_midpoint" and len(anchor_indices) >= 2:
        outward = _mean_unit_vectors(pos, anchor, [i for i in body if i in c_idx or i in h_idx])
        secondary = pos[int(anchor_indices[1])] - pos[int(anchor_indices[0])]
        return outward, secondary

    # Monodentate/O-bound species: the carbon-containing body points outward.
    if mode in {"o_atom", "o_min_z"}:
        preferred = [i for i in body if i in c_idx]
        outward = _mean_unit_vectors(pos, anchor, preferred or body)
        # Use another heavy atom first to fix the molecular plane, then H.
        secondary_candidates = [i for i in body if i in o_idx and i not in anchor_indices]
        secondary_candidates += [i for i in body if i in h_idx]
        secondary = pos[secondary_candidates[0]] - anchor if secondary_candidates else pos[body[0]] - anchor
        return outward, secondary

    # COOH*: use the O--C--O angular bisector as the outward direction.
    # Including the acidic H in the body-average tilts the carboxyl group toward
    # the OH side and can place the second O nearly parallel to the surface.
    # The two C--O bonds define the chemically relevant carboxyl orientation;
    # O--O fixes the in-plane azimuth without distorting the internal geometry.
    if key == "COOH" and len(o_idx) >= 2:
        v1 = _unit(pos[int(o_idx[0])] - anchor)
        v2 = _unit(pos[int(o_idx[1])] - anchor)
        outward = v1 + v2
        if float(np.linalg.norm(outward)) < _EPS:
            outward = _mean_unit_vectors(pos, anchor, body)
        secondary = pos[int(o_idx[1])] - pos[int(o_idx[0])]
        return outward, secondary

    # Other C-bound intermediates: orient the complete non-anchor body into
    # vacuum. Unit-vector averaging prevents one long bond from dominating.
    outward = _mean_unit_vectors(pos, anchor, body)
    heavy = [i for i in body if symbols[i] in {"C", "O"}]
    candidates = heavy + [i for i in body if i in h_idx]
    secondary = pos[candidates[0]] - anchor if candidates else np.asarray([1.0, 0.0, 0.0])
    return outward, secondary


def _orthonormal_frame(outward: np.ndarray, secondary: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = _unit(outward)
    sec = np.asarray(secondary, dtype=float).reshape(3)
    sec = sec - float(np.dot(sec, z_axis)) * z_axis
    if float(np.linalg.norm(sec)) < _EPS:
        # Pick the Cartesian axis least parallel to z.
        axes = [np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])]
        seed = min(axes, key=lambda a: abs(float(np.dot(a, z_axis))))
        sec = seed - float(np.dot(seed, z_axis)) * z_axis
    x_axis = _unit(sec, fallback=(1.0, 0.0, 0.0))
    y_axis = _unit(np.cross(z_axis, x_axis), fallback=(0.0, 1.0, 0.0))
    # Re-orthogonalize x to suppress accumulated numerical error.
    x_axis = _unit(np.cross(y_axis, z_axis), fallback=(1.0, 0.0, 0.0))
    return x_axis, y_axis, z_axis


def orient_co2rr_template(
    atoms: Atoms,
    adsorbate: str,
    anchor_mode_override: str | None = None,
) -> tuple[Atoms, tuple[int, ...], str]:
    """Return a rigidly oriented template with its anchor centered at the origin.

    No per-atom ``abs(z)`` operation is used.  Internal distances and angles are
    preserved exactly; only translation and one rigid rotation are applied.
    """
    key = str(adsorbate or "").replace("*", "").strip().upper()
    out = atoms.copy()
    anchor_indices, anchor_label = co2rr_anchor_indices(
        out, key, anchor_mode_override=anchor_mode_override
    )
    pos = np.asarray(out.get_positions(), dtype=float)
    symbols = out.get_chemical_symbols()
    anchor = pos[list(anchor_indices)].mean(axis=0)

    outward, secondary = _choose_outward_and_secondary(
        pos, symbols, key, anchor_indices, anchor,
        anchor_mode_override=anchor_mode_override,
    )
    x_axis, y_axis, z_axis = _orthonormal_frame(outward, secondary)

    rel = pos - anchor
    new_pos = np.column_stack((rel @ x_axis, rel @ y_axis, rel @ z_axis))

    # For multi-anchor species, remove tiny numerical anchor-z offsets as a
    # group.  Do not move individual atoms or alter bond geometry.
    anchor_center_new = new_pos[list(anchor_indices)].mean(axis=0)
    new_pos = new_pos - anchor_center_new
    out.set_positions(new_pos)

    out.info = dict(getattr(out, "info", {}) or {})
    out.info["co2rr_anchor_indices"] = list(anchor_indices)
    out.info["co2rr_anchor_mode"] = anchor_label
    out.info["co2rr_anchor_mode_key"] = str(
        anchor_mode_override or get_co2rr_adsorbate_spec(key).anchor_mode
    )
    out.info["co2rr_orientation"] = "rigid_body_vacuum_facing"
    return out, anchor_indices, anchor_label


def co2rr_orientation_metrics(atoms: Atoms, adsorbate: str) -> dict[str, float | str | bool]:
    """Return lightweight seed-orientation diagnostics for tests/UI."""
    key = str(adsorbate or "").replace("*", "").strip().upper()
    anchors, label = co2rr_anchor_indices(atoms, key)
    pos = np.asarray(atoms.get_positions(), dtype=float)
    body = _body_indices(len(pos), anchors)
    anchor_z = float(pos[list(anchors), 2].mean())
    body_z = pos[np.asarray(body, dtype=int), 2] if body else np.asarray([], dtype=float)
    return {
        "anchor_mode": label,
        "anchor_z_A": anchor_z,
        "body_min_z_A": float(np.min(body_z)) if body_z.size else float("nan"),
        "body_mean_z_A": float(np.mean(body_z)) if body_z.size else float("nan"),
        "vacuum_facing": bool(body_z.size and float(np.mean(body_z)) > anchor_z + 1.0e-6),
    }
