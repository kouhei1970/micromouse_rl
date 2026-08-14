#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-G の無害性の検査 — **L_c = 0 なら現行の円弧接続とビット単位で一致する**。

`clothoid_path.build_clothoid_path` は `diag_path.build_diagonal_path` の**写し**に
曲がり角の作り方だけを差し替えたものである（`competition/reference_interp.py` と
同じ作り）。**写しが元からずれると気づけない**ので、ここで機械的に検査する。

検査は 3 本:

1. **`L_c = 0` で 20 迷路すべての全配列がビット単位で一致する**（無害性の本体）
2. **`L_c > 0` でも 45° 以外の曲がり角は動かない**（裁定 (a) の帰属）
3. **クロソイドの幾何が閉じる**（`corner_samples` の閉合検査が働くこと）

    .venv/bin/python -m pytest tests/test_clothoid_path.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal",
          REPO_ROOT / "experiments" / "exp_015_time_optimal_route"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from clothoid_path import (build_clothoid_path, clothoid_tangent,  # noqa: E402
                           corner_samples, max_clothoid_len)
from competition.baseline_slalom_e1_tr import load_time_model  # noqa: E402
from competition.route_planner import value_field  # noqa: E402
from diag_path import build_diagonal_path  # noqa: E402
from diagonal_model import DiagonalGridModel, cell_center_node, descend  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from route_model import connects_true, load_maze  # noqa: E402
from run_016b import cut_segment, longest_diagonal_run  # noqa: E402

R_ARC_M = 0.060
MAZE_DIR = REPO_ROOT / "competition" / "mazes" / "design_v4"


def _segments():
    """016-B/016-C と**同じ選び方**で、調整用迷路 20 件の部分経路を返す。"""
    a, b = load_time_model()
    out = []
    for f in sorted(MAZE_DIR.glob("maze_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
        v, h, start, goals = load_maze(str(f))
        conn = connects_true(v, h)
        model = DiagonalGridModel(a, b, r=1.0)
        field = value_field([tuple(g) for g in goals], 16, 16, conn, model)
        p = descend(field, model, cell_center_node(tuple(start)), "N", 16, 16, conn)
        s0, e0 = longest_diagonal_run(p["dirs"])
        i, j = cut_segment(p["nodes"], p["dirs"], s0, e0)
        out.append((f.stem, p["nodes"][i:j + 1], p["dirs"][i:j]))
    return out


SEGMENTS = _segments()
CELL = RobotParams().cell_size


def test_lc_zero_is_bit_identical():
    """**L_c = 0 なら現行とビット単位で一致**（無害性の本体）。"""
    assert len(SEGMENTS) == 20, f"調整用迷路が 20 件でない: {len(SEGMENTS)}"
    for name, nodes, dirs in SEGMENTS:
        pa, ka, ia = build_diagonal_path(nodes, dirs, CELL, R_ARC_M)
        pb, kb, ib = build_clothoid_path(nodes, dirs, CELL, R_ARC_M, L_c=0.0)
        assert np.array_equal(ka, kb), f"{name}: 区間の印が違う"
        assert np.array_equal(ia, ib), f"{name}: 区間の添字が違う"
        for fld in ("s", "x", "y", "heading", "curvature"):
            va, vb = np.asarray(getattr(pa, fld)), np.asarray(getattr(pb, fld))
            assert va.shape == vb.shape, f"{name}.{fld}: 長さが違う {va.shape} {vb.shape}"
            assert np.array_equal(va, vb), (
                f"{name}.{fld}: ビット一致しない（最大差 {np.max(np.abs(va - vb)):.3e}）")


def test_only_45deg_corners_change():
    """**L_c > 0 でも 45° 以外の曲がり角は動かない**（裁定 (a) の帰属）。"""
    n_touched = n_45 = n_90 = 0
    for name, nodes, dirs in SEGMENTS:
        rep = {}
        build_clothoid_path(nodes, dirs, CELL, R_ARC_M, L_c=0.020, report=rep)
        for c in rep["corners"]:
            if c["turn_deg"] == 45:
                n_45 += 1
                assert c["L_c_m"] > 0.0, f"{name}: 45° にクロソイドが入っていない"
                n_touched += 1
            else:
                n_90 += 1
                assert c["L_c_m"] == 0.0, (
                    f"{name}: 45° 以外（{c['turn_deg']}°）にクロソイドが入った")
    assert n_45 == 80 and n_90 == 35, f"曲がり角の数が診断と違う: 45°={n_45} 90°={n_90}"
    assert n_touched == 80


@pytest.mark.parametrize("L_c", [0.005, 0.010, 0.020, 0.030, 0.040])
def test_corner_geometry_closes(L_c):
    """**クロソイド接続の閉合**（符号・積分の誤りを捕まえる検査そのものの確認）。"""
    R, theta = R_ARC_M, math.radians(45.0)
    for sgn in (+1.0, -1.0):
        for yaw_in in (0.0, math.pi / 4, -2.0):
            u = np.array([math.cos(yaw_in), math.sin(yaw_in)])
            P = np.array([0.3, -0.2])
            # 例外が出なければ閉合している（corner_samples が中で検査する）
            xs, ys, hs, ks, T_s = corner_samples(P, u, yaw_in, sgn * theta, R, L_c, 0.005)
            assert len(xs) > 3
            # 曲率は 0 から始まり、最大 1/R に達し、0 へ戻る
            assert abs(ks[0]) < 1e-9, "入口の曲率が 0 でない"
            assert abs(abs(ks).max() - 1.0 / R) < 1e-6, "曲率の最大が 1/R でない"
            # 接線長は現行の円弧より必ず長い（クロソイドを挟むぶん）
            assert T_s > R * math.tan(theta / 2.0)


def test_constraints_are_respected():
    """**制約 (1) 接線長 ≤ 余地・(2) 2τ ≤ θ** を破る L_c は縮められる。"""
    R, theta, room = R_ARC_M, math.radians(45.0), 0.06364
    lc = max_clothoid_len(R, theta, room)
    assert lc <= R * theta + 1e-12, "制約 (2) を破っている"
    assert clothoid_tangent(lc, R, theta)[0] <= room + 1e-9, "制約 (1) を破っている"
    # 過大な要求は縮められて記録される
    name, nodes, dirs = SEGMENTS[0]
    rep = {}
    build_clothoid_path(nodes, dirs, CELL, R_ARC_M, L_c=0.500, report=rep)
    c45 = [c for c in rep["corners"] if c["turn_deg"] == 45]
    assert c45, "45° の曲がり角が無い"
    assert all(c["shrunk"] for c in c45), "過大な L_c が縮められていない"
    assert all(c["L_c_m"] <= R_ARC_M * theta + 1e-12 for c in c45)
