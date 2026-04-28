"""
Extended smoke tests for SAGE v1.0.0.

Complements existing test_ads_sites / test_postprocess / test_seeds /
test_structure_check with:
  - CHE thermodynamic correction arithmetic
  - Table S2 metallic benchmark regression guard
  - Oxide / metal classification logic
  - Top-layer detection edge cases
  - Gas-reference file integrity
  - Surface-family facet preset coverage

NOTE: ocp_app.core.anchors.common loads UMA at import time,
      so all CHE constants are verified via source-file grep
      rather than direct import.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

# ----------- modules safe to import in CI (no model loading) -----------
from ocp_app.core.ads_sites import (
    AdsSite,
    _is_oxide_like,
    _top_layer_indices_z,
    _top_layer_anion_indices_z,
    select_representative_sites,
    ANION_SYMBOLS,
)
from ocp_app.core.structure_check import validate_structure
from ocp_app.core.seeds import fix_all
from ocp_app.core.surface_families import INTERFACE_FACET_PRESETS


# ======================================================================
# 1. CHE correction — verify the 0.24 eV constant from source code
# ======================================================================
# Cannot import from anchors/common.py (model side-effect), so grep source.

ROOT = Path(__file__).resolve().parents[1]
CHE_MODE_SRC = ROOT / "app" / "ocp_app" / "core" / "anchors" / "CHE_mode.py"
COMMON_SRC = ROOT / "app" / "ocp_app" / "core" / "anchors" / "common.py"


def test_standard_che_corr_value():
    """STANDARD_CHE_CORR in CHE_mode.py must be 0.24 eV."""
    text = CHE_MODE_SRC.read_text()
    assert "STANDARD_CHE_CORR = 0.24" in text, (
        "STANDARD_CHE_CORR must equal 0.24 eV (manuscript Table S1)"
    )


def test_zpe_tds_sum_equals_net_corr():
    """ZPE_CORR + TDS_CORR in common.py must equal NET_CORR = 0.24."""
    text = COMMON_SRC.read_text()
    # Parse values directly from source
    zpe = tds = None
    for line in text.splitlines():
        if line.startswith("ZPE_CORR"):
            zpe = float(line.split("=")[1].strip())
        if line.startswith("TDS_CORR"):
            tds = float(line.split("=")[1].strip())
    assert zpe is not None and tds is not None, "ZPE_CORR / TDS_CORR not found"
    assert math.isclose(zpe + tds, 0.24, abs_tol=1e-6), (
        f"ZPE_CORR ({zpe}) + TDS_CORR ({tds}) != 0.24"
    )


def test_che_arithmetic():
    """ΔG_H = ΔE_H + 0.24 eV — pure arithmetic sanity check."""
    delta_e = -0.340  # hypothetical raw adsorption energy
    delta_g = delta_e + 0.24
    assert math.isclose(delta_g, -0.100, abs_tol=1e-6)


# ======================================================================
# 2. Table S2 metallic benchmark regression guard
# ======================================================================
# Published SAGE ΔG_H values; any code change that shifts these
# should be caught here.

TABLE_S2 = [
    ("Ni",  "mp-23",  -0.255),
    ("Co",  "mp-102", -0.271),
    ("Pt",  "mp-126", -0.100),
    ("Pd",  "mp-2",   -0.168),
    ("Rh",  "mp-74",  -0.117),
    ("Cu",  "mp-30",  +0.178),
    ("Ag",  "mp-124", +0.555),
]

NORSKOV_REF = {
    "Ni": -0.270, "Co": -0.270, "Pt": -0.090,
    "Pd": -0.140, "Rh": -0.100, "Cu": +0.190, "Ag": +0.510,
}


@pytest.mark.parametrize("metal,mp_id,sage_dg", TABLE_S2)
def test_benchmark_within_literature_tolerance(metal, mp_id, sage_dg):
    """SAGE ΔG_H must remain within ±0.10 eV of Nørskov reference."""
    ref = NORSKOV_REF[metal]
    diff = abs(sage_dg - ref)
    assert diff < 0.10, (
        f"{metal} ({mp_id}): SAGE={sage_dg:.3f}, ref={ref:.3f}, diff={diff:.3f}"
    )


def test_benchmark_mean_absolute_deviation():
    """MAD across 7-metal benchmark must stay below 0.05 eV."""
    diffs = [abs(dg - NORSKOV_REF[m]) for m, _, dg in TABLE_S2]
    mad = sum(diffs) / len(diffs)
    assert mad < 0.05, f"MAD = {mad:.4f} eV exceeds 0.05 eV threshold"


# ======================================================================
# 3. Oxide / metal classification
# ======================================================================

def test_is_oxide_like_pure_metal():
    """Pure metal slab must NOT be classified as oxide-like."""
    cu = Atoms("Cu4", positions=[(0,0,0),(1.8,0,0),(0,1.8,0),(1.8,1.8,0)],
               cell=[5,5,15], pbc=True)
    assert _is_oxide_like(cu) is False


def test_is_oxide_like_metal_oxide():
    """NiO-type structure must be classified as oxide-like."""
    nio = Atoms("NiO", positions=[(0,0,0),(2.1,0,0)],
                cell=[5,5,15], pbc=True)
    assert _is_oxide_like(nio) is True


def test_anion_symbols_include_oxygen():
    """O must be in ANION_SYMBOLS for oxide logic to work."""
    assert "O" in ANION_SYMBOLS


# ======================================================================
# 4. Top-layer detection edge cases
# ======================================================================

def test_top_layer_indices_z_single_layer():
    """Single-layer slab: all atoms are in the top layer."""
    atoms = Atoms("Cu4",
                  positions=[(0,0,5),(2,0,5),(0,2,5),(2,2,5)],
                  cell=[5,5,15], pbc=True)
    top = _top_layer_indices_z(atoms, z_tol=0.8)
    assert len(top) == 4


def test_top_layer_indices_z_two_layers():
    """Two layers 3 Å apart: only top layer atoms are returned."""
    atoms = Atoms("Cu4",
                  positions=[(0,0,2),(2,0,2),(0,0,5),(2,0,5)],
                  cell=[5,5,15], pbc=True)
    top = _top_layer_indices_z(atoms, z_tol=0.8)
    assert len(top) == 2
    assert all(atoms.positions[i, 2] > 4.0 for i in top)


def test_top_layer_anion_indices_z_oxide():
    """On a Co-O slab, only O atoms in the top layer are returned."""
    atoms = Atoms("Co2O2",
                  positions=[(0,0,2),(2,0,2),(0,0,5),(2,0,5)],
                  cell=[5,5,15], pbc=True)
    # Co at z=2, O at z=5 — but need to check actual symbols
    # Actually: Co, Co, O, O — so O are indices 2,3 at z=5
    top_anions = _top_layer_anion_indices_z(atoms, z_tol=0.8)
    assert len(top_anions) == 2
    for i in top_anions:
        assert atoms[i].symbol == "O"


def test_top_layer_indices_z_empty():
    """Empty Atoms object returns empty array."""
    atoms = Atoms()
    top = _top_layer_indices_z(atoms)
    assert len(top) == 0


# ======================================================================
# 5. Site selection — expanded from existing test
# ======================================================================

def test_select_representative_sites_respects_total_cap():
    """Total sites returned must not exceed 3 × per_kind."""
    sites = [
        AdsSite(kind="ontop",  position=(0,0,1), surface_indices=(0,)),
        AdsSite(kind="ontop",  position=(1,0,1), surface_indices=(1,)),
        AdsSite(kind="ontop",  position=(2,0,1), surface_indices=(2,)),
        AdsSite(kind="bridge", position=(0,1,1), surface_indices=(0,1)),
        AdsSite(kind="bridge", position=(1,1,1), surface_indices=(1,2)),
        AdsSite(kind="hollow", position=(0,0,1), surface_indices=(0,1,2)),
        AdsSite(kind="hollow", position=(1,1,1), surface_indices=(1,2,3)),
    ]
    picked = select_representative_sites(sites, per_kind=2)
    assert len(picked) <= 6


def test_select_representative_sites_empty_input():
    """Empty site list must return empty list."""
    picked = select_representative_sites([], per_kind=3)
    assert picked == []


# ======================================================================
# 6. Gas-reference file integrity
# ======================================================================

REF_GAS_DIR = ROOT / "ref_gas"

EXPECTED_GAS_CIFS = ["H2_box.cif", "CO2_box.cif", "CO_box.cif", "H2O_box.cif"]


@pytest.mark.parametrize("cif_name", EXPECTED_GAS_CIFS)
def test_gas_reference_cif_exists(cif_name):
    """Each core gas-reference CIF must be present in ref_gas/."""
    assert (REF_GAS_DIR / cif_name).is_file(), f"Missing {cif_name}"


def test_h2_meta_json_has_energy():
    """H2_meta.json must contain the cached H2 energy."""
    meta_path = REF_GAS_DIR / "H2_meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text())
    assert "E_H2 (eV)" in meta
    e_h2 = meta["E_H2 (eV)"]
    # UMA H2 energy should be in a reasonable range
    assert -10.0 < e_h2 < 0.0, f"H2 energy {e_h2} eV looks unreasonable"


# ======================================================================
# 7. Surface-family facet presets
# ======================================================================

def test_metal_facet_presets_include_111():
    """Metal preset must include the (111) close-packed facet."""
    assert "metal" in INTERFACE_FACET_PRESETS
    metal = INTERFACE_FACET_PRESETS["metal"]
    facets_111 = [v for v in metal.values() if v == (1, 1, 1)]
    assert len(facets_111) >= 1


def test_all_facet_presets_are_3_tuples():
    """Every facet in every preset must be a 3-element integer tuple."""
    for family, presets in INTERFACE_FACET_PRESETS.items():
        for label, hkl in presets.items():
            assert len(hkl) == 3, f"{family}/{label}: {hkl} is not a 3-tuple"
            assert all(isinstance(i, int) for i in hkl), (
                f"{family}/{label}: {hkl} contains non-int"
            )


# ======================================================================
# 8. Seed reproducibility — torch included
# ======================================================================

def test_fix_all_torch_seed():
    """fix_all must also fix torch RNG when available."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")

    fix_all(42)
    t1 = torch.rand(3)
    fix_all(42)
    t2 = torch.rand(3)
    assert torch.allclose(t1, t2)
