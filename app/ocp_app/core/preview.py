from pathlib import Path
from io import BytesIO, StringIO
import zipfile

from ase.io import read

from ocp_app.core.structure_ops import atoms_to_cif_bytes
from ocp_app.core.ads_sites import oxide_surface_seed_position, expand_oxide_channels_for_adsorbate, ANION_SYMBOLS

ADS_TEMPLATE_FILES = {
    "CO":   "CO_box.cif",
    "COOH": "COOH_box.cif",
    "HCOO": "HCOO_box.cif",
    "OCHO": "OCHO_box.cif",
    # ORR intermediates
    "O":    "O_box.cif",
    "OH":   "OH_box.cif",
    "OOH":  "OOH_box.cif",
}


def load_ads_template_preview(ads: str, ref_dir: str | Path = "ref_gas"):
    ads_clean = ads.replace("*", "").upper()
    if ads_clean not in ADS_TEMPLATE_FILES:
        raise ValueError(f"Unsupported adsorbate '{ads}' for preview")

    cif_path = Path(ref_dir) / ADS_TEMPLATE_FILES[ads_clean]
    if not cif_path.is_file():
        raise FileNotFoundError(f"Adsorbate template not found: {cif_path}")

    a = read(cif_path).copy()
    symbols = a.get_chemical_symbols()

    # Anchor priority: C (CO2RR) → O (ORR: OH*, OOH*, O*) → atom 0
    anchor_idx = None
    for i, s in enumerate(symbols):
        if s == "C":
            anchor_idx = i
            break
    if anchor_idx is None:
        for i, s in enumerate(symbols):
            if s == "O":
                anchor_idx = i
                break
    if anchor_idx is None:
        anchor_idx = 0

    pos = a.get_positions()
    anchor_pos = pos[anchor_idx].copy()
    pos -= anchor_pos
    pos[:, 2] = abs(pos[:, 2])
    a.set_positions(pos)
    return a


def build_adsorbate_preview_slab(slab_atoms, site, ads: str, dz: float = 1.8, ref_dir: str | Path = "ref_gas"):
    slab = slab_atoms.copy()
    z_top = float(slab.get_positions()[:, 2].max())
    ads_clean = ads.replace("*", "").upper()

    ads_atoms = load_ads_template_preview(ads, ref_dir=ref_dir)

    is_oxide_like = any(s in ANION_SYMBOLS for s in slab.get_chemical_symbols()) and any(s not in ANION_SYMBOLS for s in slab.get_chemical_symbols())
    if is_oxide_like and ads_clean in ("O", "OH", "OOH"):
        channel = expand_oxide_channels_for_adsorbate(ads_clean)[0]
        x0, y0, z0, _surface_channel = oxide_surface_seed_position(slab, site, ads_clean, channel=channel)
        base = [x0, y0, z0]
    else:
        z_min = float(ads_atoms.get_positions()[:, 2].min())
        base_z = z_top + float(dz) - z_min
        xy = site.position[:2]
        base = [float(xy[0]), float(xy[1]), float(base_z)]

    ads_atoms.set_cell(slab.get_cell())
    ads_atoms.set_pbc(slab.get_pbc())
    ads_atoms.translate(base)

    return slab + ads_atoms


def export_zip_of_struct_map(struct_map: dict, symprec: float = 0.1) -> BytesIO:
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, atoms in (struct_map or {}).items():
            zf.writestr(f"{k}.cif", atoms_to_cif_bytes(atoms, symprec=symprec))
    zip_buf.seek(0)
    return zip_buf
