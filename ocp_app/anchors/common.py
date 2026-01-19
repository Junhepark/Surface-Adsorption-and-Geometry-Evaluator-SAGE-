# ocp_app/core/anchors/common.py
from __future__ import annotations

import numpy as np
import torch
from ase.build import add_adsorbate
from ase.constraints import FixAtoms, FixCartesian
from ase.optimize import BFGS

from fairchem.core import pretrained_mlip, FAIRChemCalculator

# ---------------- Config / UMA init ----------------
MODEL_NAME = "uma-s-1p1"   # 또는 "uma-m-1p1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0)
torch.manual_seed(0)

_predictor = pretrained_mlip.get_predict_unit(MODEL_NAME, device=DEVICE)
calc = FAIRChemCalculator(_predictor, task_name="oc20")

# ---- thermochem (298 K) ----
ZPE_CORR = 0.04
TDS_CORR = 0.20
NET_CORR = ZPE_CORR - TDS_CORR   # ≈ -0.16 eV (절대 ΔG 산출옵션에서만 사용)

# ---- thresholds / params (공통) ----
H0S = (1.0, 1.2, 1.4)   # Å, 초기 높이 스캔
MIGRATE_THR = 0.30      # Å, H 평면 이동 플래그 기준
VAC_WARN_MIN = 20.0     # Å, 진공 경고
UNUSUAL_DDELTA = 1.00   # eV, |ΔE_user - ΔE_anchor(site)| 경고


# ---------------- 공통 utils ----------------
def ensure_pbc3(a, vac_z=None):
    """3차원 PBC를 강제하고, 필요시 z-방향 진공을 재설정."""
    a = a.copy()
    a.set_pbc([True, True, True])
    if vac_z is not None:
        a.center(axis=2, vacuum=float(vac_z))
    a.wrap(eps=1e-9)
    return a


def layer_indices(at, n=3, tol=0.25):
    """상단 n개 레이어의 인덱스 리스트를 위에서부터 반환."""
    z = at.get_positions()[:, 2]
    zuniq = np.unique(np.round(z, 3))
    zuniq.sort()
    topz = zuniq[-n:]
    layers = [np.where(np.isclose(z, zval, atol=tol))[0] for zval in topz[::-1]]
    return layers  # [top, second, third] (최대 n개)


def first_layer_min_distance(a):
    """H와 top layer atom들 사이 최소 거리."""
    top_idx = layer_indices(a, n=1)[0]
    pos = a.get_positions()
    h_idx = [i for i, _ in enumerate(a) if a[i].symbol == "H"]
    if not h_idx:
        return None
    hpos = pos[h_idx[0]]
    dxyz = np.linalg.norm(pos[top_idx] - hpos, axis=1)
    return float(dxyz.min())


def put_H(at, xy, height=1.20, min_clearance=0.9):
    """주어진 slab(at) 위 xy 위치에 H adsorbate 추가."""
    a = ensure_pbc3(at)
    add_adsorbate(
        a,
        "H",
        height=float(height),
        position=(float(xy[0]), float(xy[1])),
    )
    a = ensure_pbc3(a)
    md = first_layer_min_distance(a)
    if md is not None and md < min_clearance:
        for i, atom in enumerate(a):
            if atom.symbol == "H":
                p = atom.position
                p[2] += (min_clearance - md) + 0.2
                atom.position = p
                break
    return a


def make_bottom_fix_mask(at, n_fix_layers=2, z_tol=0.25):
    """하부 n_fix_layers 레이어만 고정하는 mask."""
    z = at.get_positions()[:, 2]
    zs = np.unique(np.round(z, 3))
    cuts = zs[: min(n_fix_layers, len(zs))]
    mask = []
    for atom in at:
        if atom.symbol == "H":
            mask.append(False)
        else:
            mask.append(np.any(np.abs(atom.position[2] - cuts) < z_tol))
    return np.array(mask, bool)


def relax_zonly(atoms, steps=None, fmax=0.05):
    """
    H는 z만 풀고, slab 하부는 fix.
    steps가 None이거나 0이면 기하를 바꾸지 않고 에너지만 평가.
    """
    a = ensure_pbc3(atoms)
    a.calc = calc
    cons = [FixAtoms(mask=make_bottom_fix_mask(a, n_fix_layers=2))]
    for i, atom in enumerate(a):
        if atom.symbol == "H":
            cons.append(FixCartesian(i, [True, True, False]))
    a.set_constraint(cons)
    nsteps = 0 if steps is None else int(steps)
    if nsteps > 0:
        BFGS(a, logfile=None).run(fmax=fmax, steps=nsteps)
    return a, a.get_potential_energy()


def relax_freeH(atoms, steps=None, fmax=0.03):
    """
    하부 레이어만 fix, 나머지 free relax (H 포함).
    steps가 None이거나 0이면 기하를 바꾸지 않고 에너지만 평가.
    """
    a = ensure_pbc3(atoms)
    a.calc = calc
    a.set_constraint([FixAtoms(mask=make_bottom_fix_mask(a, n_fix_layers=2))])
    nsteps = 0 if steps is None else int(steps)
    if nsteps > 0:
        BFGS(a, logfile=None).run(fmax=fmax, steps=nsteps)
    return a, a.get_potential_energy()


def site_energy_two_stage(at_relaxed, xy, h0s=H0S,
                          z_steps=None, free_steps=None):
    """
    z-only(짧게) → full relax(정밀), 여러 h0 중 최저 선택.
    반환: (최종 구조, 최종 에너지, H 평면 이동량)

    z_steps, free_steps는 각 단계에서 허용할 최대 스텝 수.
    """
    if z_steps is None or free_steps is None:
        raise ValueError("site_energy_two_stage: z_steps와 free_steps를 명시해야 합니다.")

    best = {"E": None, "atoms": None, "disp_xy": None}
    xy = np.array(xy, float)
    for h0 in h0s:
        A0 = put_H(at_relaxed, xy, height=h0)
        Az, _ = relax_zonly(A0, steps=z_steps, fmax=0.05)
        Af, Ef = relax_freeH(Az, steps=free_steps, fmax=0.03)
        hi = [i for i, atom in enumerate(Af) if atom.symbol == "H"][0]
        disp_xy = float(np.linalg.norm(Af[hi].position[:2] - xy))
        if best["E"] is None or Ef < best["E"]:
            best.update({"E": Ef, "atoms": Af, "disp_xy": disp_xy})
    return best["atoms"], best["E"], best["disp_xy"]


def factor_near_square(k: int):
    """정수 k를 근사 정사각형 nx×ny로 분해."""
    r = int(np.floor(np.sqrt(k)))
    for nx in range(r, 0, -1):
        if k % nx == 0:
            return nx, k // nx
    nx = max(1, int(round(np.sqrt(k))))
    ny = max(1, int(round(k / nx)))
    return nx, ny


__all__ = [
    "MODEL_NAME", "DEVICE", "ZPE_CORR", "TDS_CORR", "NET_CORR",
    "H0S", "MIGRATE_THR", "VAC_WARN_MIN", "UNUSUAL_DDELTA",
    "calc",
    "ensure_pbc3", "layer_indices", "first_layer_min_distance",
    "put_H", "make_bottom_fix_mask",
    "relax_zonly", "relax_freeH",
    "site_energy_two_stage", "factor_near_square",
]
