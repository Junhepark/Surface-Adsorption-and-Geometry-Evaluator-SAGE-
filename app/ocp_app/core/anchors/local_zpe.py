from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from ase import Atoms
from ase.thermochemistry import HarmonicThermo
from ase.vibrations import Vibrations

from ocp_app.core.anchors.common import calc, ensure_pbc3


def _top_surface_indices(atoms: Atoms, n_slab_atoms: int, z_window: float = 2.0) -> np.ndarray:
    pos = atoms.get_positions()[:n_slab_atoms]
    if len(pos) == 0:
        return np.array([], dtype=int)
    z = pos[:, 2]
    zmax = float(np.max(z))
    idx = np.where((zmax - z) <= float(z_window))[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(z))], dtype=int)
    return idx.astype(int)


def _select_local_vibration_indices(
    atoms: Atoms,
    n_slab_atoms: int,
    cutoff: float = 2.5,
    max_neighbors: int = 3,
    top_z_window: float = 2.0,
) -> Dict[str, object]:
    syms = atoms.get_chemical_symbols()
    h_candidates = [i for i, s in enumerate(syms) if s == "H"]
    if not h_candidates:
        raise ValueError("No H atom found in slab+ads structure.")

    h_index = int(h_candidates[-1])
    pos = atoms.get_positions()
    h_pos = np.asarray(pos[h_index], dtype=float)

    top_idx = _top_surface_indices(atoms, n_slab_atoms=n_slab_atoms, z_window=top_z_window)
    if top_idx.size == 0:
        raise ValueError("No slab atoms available for local vibration selection.")

    d = np.linalg.norm(pos[top_idx] - h_pos[None, :], axis=1)
    order = np.argsort(d)
    top_sorted = top_idx[order]
    d_sorted = d[order]

    selected_neighbors: List[int] = [int(i) for i, dist in zip(top_sorted.tolist(), d_sorted.tolist()) if float(dist) <= float(cutoff)]
    if not selected_neighbors:
        selected_neighbors = [int(top_sorted[0])]

    selected_neighbors = selected_neighbors[: int(max_neighbors)]
    selected = [h_index] + selected_neighbors

    return {
        "h_index": h_index,
        "neighbor_indices": selected_neighbors,
        "selected_indices": selected,
        "n_vib_atoms": len(selected),
    }


def compute_local_h_thermo_correction(
    slab_ads_atoms: Atoms,
    n_slab_atoms: int,
    temperature: float = 298.15,
    cutoff: float = 2.5,
    max_neighbors: int = 3,
    top_z_window: float = 2.0,
    delta: float = 0.01,
) -> Dict[str, object]:
    """
    Local vibrational thermochemistry for adsorbed H on a slab.

    Returns
    -------
    dict with keys:
        delta_zpe_eV
        delta_ts_eV
        local_corr_eV   # ΔZPE - TΔS
        selected_indices
        neighbor_indices
        h_index
        n_vib_atoms
        warnings
    """
    atoms = ensure_pbc3(slab_ads_atoms)
    atoms = atoms.copy()
    atoms.calc = calc

    meta = _select_local_vibration_indices(
        atoms,
        n_slab_atoms=int(n_slab_atoms),
        cutoff=float(cutoff),
        max_neighbors=int(max_neighbors),
        top_z_window=float(top_z_window),
    )
    selected = list(meta["selected_indices"])

    tmpdir = Path(tempfile.mkdtemp(prefix="sage_local_zpe_"))
    vib_name = str(tmpdir / "vib")
    warnings: List[str] = []
    try:
        vib = Vibrations(atoms, indices=selected, name=vib_name, delta=float(delta))
        vib.run()
        vib_energies = vib.get_energies()
        vib_energies = [complex(x) for x in vib_energies if x is not None]

        real_energies: List[float] = []
        imag_count = 0
        for e in vib_energies:
            if abs(e.imag) > 1e-8:
                imag_count += 1
            if e.real > 1e-8:
                real_energies.append(float(e.real))

        if not real_energies:
            raise RuntimeError("No positive vibrational energies were produced.")

        thermo = HarmonicThermo(vib_energies=real_energies, potentialenergy=0.0)
        zpe = float(thermo.get_ZPE_correction())
        entropy = float(thermo.get_entropy(temperature=float(temperature)))
        delta_ts = float(float(temperature) * entropy)
        local_corr = float(zpe - delta_ts)

        if imag_count > 0:
            warnings.append(f"Ignored {imag_count} imaginary/complex vibrational mode(s).")

        out = {
            "delta_zpe_eV": zpe,
            "delta_ts_eV": delta_ts,
            "local_corr_eV": local_corr,
            "selected_indices": selected,
            "neighbor_indices": list(meta["neighbor_indices"]),
            "h_index": int(meta["h_index"]),
            "n_vib_atoms": int(meta["n_vib_atoms"]),
            "warnings": warnings,
        }
        return out
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
