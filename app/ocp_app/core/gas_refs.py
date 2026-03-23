# ocp_app/core/gas_refs.py

from __future__ import annotations

import json
from pathlib import Path

from ase.build import molecule
from ase.io import read, write
from ase.optimize import BFGS


def _ensure_pbc3_gas(a, vac_z: float = 10.0):
    """Simple PBC helper for gas-phase molecules in a periodic box."""
    a = a.copy()
    a.set_pbc([True, True, True])
    a.center(vacuum=float(vac_z))
    return a


def get_gas_ref(
    calc,
    species: str,
    ref_dir: str | Path = "ref_gas",
    steps: int = 200,
    fmax: float = 0.03,
    vac_z: float = 10.0,
):
    """
    Return a gas-phase reference structure and energy (species_box.cif + E_species).

    species: 'H2', 'CO2', 'CO', 'H2O' (case-insensitive)

    Lookup order:
      - If ref_dir/{species}_box.cif and {species}_meta.json both exist,
        reuse the cached structure and energy.
      - Otherwise:
        * Use {species}_box.cif if present, or build via ASE molecule().
        * Relax once, store the structure and energy, then return.
    """
    species = species.upper()
    if species not in ("H2", "CO2", "CO", "H2O"):
        raise ValueError(f"Unsupported gas species '{species}'")

    ref_dir = Path(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)

    cif_path = ref_dir / f"{species}_box.cif"
    meta_path = ref_dir / f"{species}_meta.json"
    meta_key = f"E_{species} (eV)"

    # 1) Both CIF and meta exist → reuse directly
    if cif_path.is_file() and meta_path.is_file():
        a = read(cif_path)
        meta = json.loads(meta_path.read_text())
        if meta_key in meta:
            E = float(meta[meta_key])
            return a, E

    # 2) Meta missing → compute energy once and save
    #    - Use existing CIF if available
    #    - Otherwise build from ASE molecule()
    if cif_path.is_file():
        a = read(cif_path)
    else:
        # ASE molecule() supports H2, CO2, CO, H2O directly
        a = molecule(species)

    a = _ensure_pbc3_gas(a, vac_z=vac_z)
    a.calc = calc

    if steps > 0:
        BFGS(a, logfile=None).run(fmax=fmax, steps=steps)

    E = float(a.get_potential_energy())

    # Save structure + meta (same format as H2 reference)
    write(cif_path, a)
    meta = {
        meta_key: E,
        "vac_z(Å)": float(vac_z),
        "steps": int(steps),
        "fmax": float(fmax),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return a, E


def get_h2_ref(
    calc,
    ref_dir: str | Path = "ref_gas",
    steps: int = 200,
    fmax: float = 0.03,
    vac_z: float = 10.0,
):
    """
    Return the canonical H2 reference (H2_box.cif + E_H2).

    Reuses cached files if available; otherwise relaxes once and saves.
    This is a legacy-interface wrapper that calls get_gas_ref('H2') internally.
    """
    return get_gas_ref(
        calc,
        "H2",
        ref_dir=ref_dir,
        steps=steps,
        fmax=fmax,
        vac_z=vac_z,
    )

# =====================================================================
# ORR intermediate template CIF generation
# =====================================================================

from ase import Atoms as _Atoms


def make_orr_templates(ref_dir: str | Path = "ref_gas") -> None:
    """
    Generate ORR intermediate template CIF files in ref_gas/.
    Written in the same {ads}_box.cif format read by _load_ads_template().

    Generated files:
      - O_box.cif   : single O atom (anchor = O, z=0 reference)
      - OH_box.cif  : O(anchor)-H  (O-H ≈ 0.97 Å)
      - OOH_box.cif : O(anchor)-O-H (O-O ≈ 1.21 Å, O-H ≈ 0.97 Å)

    Existing files are not overwritten (cache-first policy).
    Should be called once at app startup or during setup.
    """
    ref_dir = Path(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)

    # ── O* ────────────────────────────────────────────────────────────
    o_path = ref_dir / "O_box.cif"
    if not o_path.is_file():
        a = _Atoms("O", positions=[(0.0, 0.0, 0.0)])
        a.set_cell([10.0, 10.0, 10.0])
        a.set_pbc(True)
        a.center()
        write(o_path, a)

    # ── OH* ───────────────────────────────────────────────────────────
    # O is the anchor (slab side), H above (+z)
    oh_path = ref_dir / "OH_box.cif"
    if not oh_path.is_file():
        a = _Atoms(
            "OH",
            positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.97)],
        )
        a.set_cell([10.0, 10.0, 10.0])
        a.set_pbc(True)
        a.center()
        write(oh_path, a)

    # ── OOH* ──────────────────────────────────────────────────────────
    # Lower O is the anchor (slab side), upper O with H on top
    ooh_path = ref_dir / "OOH_box.cif"
    if not ooh_path.is_file():
        a = _Atoms(
            "OOH",
            positions=[
                (0.0, 0.0, 0.00),   # O anchor
                (0.0, 0.0, 1.21),   # O  (O-O ≈ 1.21 Å)
                (0.97, 0.0, 1.21),  # H  (O-H ≈ 0.97 Å, lateral)
            ],
        )
        a.set_cell([10.0, 10.0, 10.0])
        a.set_pbc(True)
        a.center()
        write(ooh_path, a)
