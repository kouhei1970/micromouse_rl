#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-H1d の計装が**走行を 1 ビットも変えない**ことを確かめる（教授指示 2026-08-15）。

`run_016h1_diag.make_probed` は `_wheel_targets_to_voltage` を包んで指令角速度と電圧を
記録するが、**返り値は親のものを素通しする**。したがって同じ迷路を
「包んだ方策」と「包んでいない方策」で走らせたとき、**軌跡は完全に一致するはず**である。

**「記録のみのつもりの変更が挙動を変える」事故の予防**であり、
計装を測定に使う前に通す（従来の流儀）。

    .venv/bin/python -m pytest tests/test_h1d_probe_harmless.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "exp_016_diagonal"))
sys.path.insert(0, str(ROOT))

from competition.baseline_slalom_diag_cal import L_C_CLOTHOID_M      # noqa: E402
from competition.baseline_slalom_e1_tr import load_time_model        # noqa: E402
from competition.route_planner import value_field                    # noqa: E402
from mouse.mjcf import build_maze_robot_xml                          # noqa: E402
from mouse.params import RobotParams                                 # noqa: E402
from mouse.sim import MouseSim                                       # noqa: E402

import run_016f0_ladder                                              # noqa: E402
import run_016g_ladder                                               # noqa: E402
from diagonal_model import (DELTA8, DiagonalGridModel,               # noqa: E402
                            cell_center_node, descend)
from route_model import connects_true, load_maze                     # noqa: E402
from run_016b import cut_segment, longest_diagonal_run               # noqa: E402
from run_016c import R_ARC_M                                         # noqa: E402
from run_016h1_diag import make_probed                               # noqa: E402

MAZE = ROOT / "competition" / "mazes" / "design_v4" / "maze_41003.npz"
V_DIAG = 0.60          # 飽和が起きうる高速側で確かめる（無害性が最も疑わしい条件）


def _segment(params):
    v, h, start, goals = load_maze(str(MAZE))
    conn = connects_true(v, h)
    a, b = load_time_model()
    field = value_field([tuple(g) for g in goals], 16, 16, conn,
                        DiagonalGridModel(a, b, r=1.0))
    p = descend(field, DiagonalGridModel(a, b, r=1.0),
                cell_center_node(tuple(start)), "N", 16, 16, conn)
    s0, e0 = longest_diagonal_run(p["dirs"])
    i, j = cut_segment(p["nodes"], p["dirs"], s0, e0)
    xml = MAZE.with_suffix(".xml")
    if not xml.exists():
        build_maze_robot_xml(v, h, str(xml), model_name="m_harmless", params=params)
    return xml, p["nodes"][i:j + 1], p["dirs"][i:j], v, h


def _trajectory(policy_cls, params, xml, nodes, dirs, v_walls, h_walls, max_s=40.0):
    """1 迷路を走らせ、毎ティックの (x, y, yaw, 電圧 L, 電圧 R) を返す。"""
    builder = run_016g_ladder.make_builder(L_C_CLOTHOID_M)
    path, kinds, _ = builder(nodes, dirs, params.cell_size, R_ARC_M)
    sim = MouseSim(str(xml), params=params)
    sim.full_reset(cell=(nodes[0][0] // 2, nodes[0][1] // 2),
                   heading_deg=math.degrees(math.atan2(DELTA8[dirs[0]][1],
                                                       DELTA8[dirs[0]][0])))
    pol = policy_cls(path, np.where(kinds == "straight", 1e9, V_DIAG))
    pol.bind_sim(sim)
    pol.bind_maze(v_walls, h_walls)
    pol.on_maze_start(dict(width=16, height=16))
    traj = []
    for _ in range(int(max_s / params.control_dt)):
        vl, vr = pol.act(sim.observation())
        x, y, yaw = sim.privileged_pose()
        traj.append((x, y, yaw, float(vl), float(vr)))
        out = sim.step_control(vl, vr)
        if out.get("collision") or pol.finished:
            break
    return np.asarray(traj, dtype=float)


def test_probe_does_not_change_the_run():
    """包んだ方策と包んでいない方策で、軌跡と電圧が **bit 一致** すること。"""
    params = RobotParams()
    xml, nodes, dirs, v_walls, h_walls = _segment(params)
    base_cls = run_016f0_ladder.make_policy_class(k_acc_ff=1.0, ref_interp=True, safety=0.75)

    plain = _trajectory(base_cls, params, xml, nodes, dirs, v_walls, h_walls)
    probed = _trajectory(make_probed(base_cls), params, xml, nodes, dirs, v_walls, h_walls)

    assert plain.shape == probed.shape, f"ティック数が違う {plain.shape} 対 {probed.shape}"
    assert len(plain) > 100, f"標本が少なすぎて検査にならない（{len(plain)} ティック）"
    # 浮動小数の許容を置かない。**完全一致**でなければ計装が挙動を変えている
    assert np.array_equal(plain, probed), (
        "計装の有無で走行が変わった（最大差 "
        f"{np.max(np.abs(plain - probed)):.3e}）"
    )
