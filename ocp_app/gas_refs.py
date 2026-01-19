# ocp_app/core/gas_refs.py

from __future__ import annotations

import json
from pathlib import Path

from ase.build import molecule
from ase.io import read, write
from ase.optimize import BFGS


def _ensure_pbc3_gas(a, vac_z: float = 10.0):
    """가스 상자용 간단 PBC helper."""
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
    전역 gas 참조 (species_box.cif + E_species)를 반환.

    species: 'H2', 'CO2', 'CO', 'H2O' (대소문자 무시)

    - ref_dir/{species}_box.cif, {species}_meta.json 이 있으면 그대로 사용
    - meta가 없으면:
        * {species}_box.cif가 있으면 그 구조를 사용
        * 없으면 ASE의 molecule(...) 로 새로 생성
      → 한 번 relax해서 에너지 저장 후 사용
    """
    species = species.upper()
    if species not in ("H2", "CO2", "CO", "H2O"):
        raise ValueError(f"Unsupported gas species '{species}'")

    ref_dir = Path(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)

    cif_path = ref_dir / f"{species}_box.cif"
    meta_path = ref_dir / f"{species}_meta.json"
    meta_key = f"E_{species} (eV)"

    # 1) CIF + meta 둘 다 있으면 그대로 사용
    if cif_path.is_file() and meta_path.is_file():
        a = read(cif_path)
        meta = json.loads(meta_path.read_text())
        if meta_key in meta:
            E = float(meta[meta_key])
            return a, E

    # 2) meta가 없으면 에너지 한 번 계산해서 저장
    #    - CIF가 있으면 그 구조 사용
    #    - 없으면 molecule(...)로 생성
    if cif_path.is_file():
        a = read(cif_path)
    else:
        # ASE 분자 이름 그대로 사용
        # H2, CO2, CO, H2O 는 ASE에서 바로 생성 가능
        a = molecule(species)

    a = _ensure_pbc3_gas(a, vac_z=vac_z)
    a.calc = calc

    if steps > 0:
        BFGS(a, logfile=None).run(fmax=fmax, steps=steps)

    E = float(a.get_potential_energy())

    # 구조 + meta 저장 (기존 H2 방식과 동일 포맷)
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
    전역 H2 참조 (canonical H2_box.cif + E_H2)를 반환.

    - ref_dir/H2_box.cif, H2_meta.json 이 있으면 그대로 사용
    - 없으면 한 번만 relax 해서 파일/에너지 저장 후 사용

    (기존 인터페이스 유지용 래퍼. 내부적으로 get_gas_ref('H2') 호출)
    """
    return get_gas_ref(
        calc,
        "H2",
        ref_dir=ref_dir,
        steps=steps,
        fmax=fmax,
        vac_z=vac_z,
    )