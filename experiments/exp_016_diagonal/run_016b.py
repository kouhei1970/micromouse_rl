#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-B 本体 — **1 本の斜め区間だけを走り、横偏差と方位偏差を対で計測する**。

カード `card_016b.md` の主判定 A（幾何のみで判定）と副次測定 B（45° 旋回の費用）を
**排他的に分けて**出す。**B の値は A の判定に一切影響しない。**

--------------------------------------------------------------------------
作り
--------------------------------------------------------------------------
- **面の選定は機械的**（恣意的に選ばない）: 016-A の経路の中で
  **斜めが連続する最長の区間**を持つ面を選ぶ
- 走行は「直進の助走 → 円弧 → **斜めの連続区間** → 円弧 → 直進の抜け」だけ。
  **全面の統合は 016-D**
- 制御は `competition/baseline_slalom.py` の実装を**そのまま**使う
  （Stanley・車輪 PI・`build_speed_profile`）。**制御は 1 行も変えない**
- **計装は走行後**（軌跡から $e_y$・$\\psi$ を計算する）。**走行中には何も計算しない**
  ＝計装が挙動を変えようがない

使い方:
    .venv/bin/python experiments/exp_016_diagonal/run_016b.py [--v-ceil 0.5]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal",
          REPO_ROOT / "experiments" / "exp_015_time_optimal_route"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mujoco  # noqa: E402

from competition.baseline_slalom import SlalomPolicy, build_speed_profile  # noqa: E402
from common.seed_bands import (assert_seeds_allowed,  # noqa: E402
                                describe_seeds)
from competition.baseline_slalom_e1_tr import load_time_model  # noqa: E402
from competition.route_planner import value_field  # noqa: E402
from mouse.mjcf import build_maze_robot_xml  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

from diag_path import (build_diagonal_path, heading_deviation,  # noqa: E402
                       lateral_deviation)
from diagonal_model import (DELTA8, DIAGONALS, DiagonalGridModel,  # noqa: E402
                            cell_center_node, descend, node_kind, node_xy)
from geometry import body_extent_exact, diagonal_clearance, git_rev  # noqa: E402
from route_model import connects_true, load_maze  # noqa: E402

R_ARC_M = 0.060                     # カード §4-4（教授承認 2026-08-14）
LEAD_CELLS = 2                      # 助走・抜けに使う直進の区画数


# ==========================================================================
# 面と区間の選定（**機械的に**）
# ==========================================================================
def longest_diagonal_run(dirs):
    """方位列の中で**斜めが連続する最長の区間** [start, end) を返す。"""
    best = (0, 0)
    i = 0
    while i < len(dirs):
        if dirs[i] not in DIAGONALS:
            i += 1
            continue
        j = i
        while j < len(dirs) and dirs[j] in DIAGONALS and dirs[j] == dirs[i]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    return best


def pick_face(maze_dir, a, b):
    """設計帯の全面から、斜めの最長連続区間が最も長い面を選ぶ。"""
    best = None
    for f in sorted((REPO_ROOT / maze_dir).glob("maze_*.npz"),
                    key=lambda p: int(p.stem.split("_")[1])):
        v, h, start, goals = load_maze(str(f))
        conn = connects_true(v, h)
        model = DiagonalGridModel(a, b, r=1.0)
        field = value_field([tuple(g) for g in goals], 16, 16, conn, model)
        p = descend(field, model, cell_center_node(tuple(start)), "N", 16, 16, conn)
        s, e = longest_diagonal_run(p["dirs"])
        if best is None or (e - s) > best[1]:
            best = (f, e - s, p, (s, e), v, h)
    return best


def cut_segment(nodes, dirs, s, e, lead=LEAD_CELLS):
    """斜めの区間 [s,e) の前後に直進の助走・抜けを付けた部分経路を切り出す。

    **助走の先頭は必ず区画中心**にする（`full_reset` が区画中心にしか置けないため）。
    """
    i = s
    n_lead = 0
    while i > 0 and n_lead < lead * 2:
        i -= 1
        if dirs[i] in DIAGONALS:
            i += 1
            break
        n_lead += 1
    while i > 0 and node_kind(nodes[i]) != "C":
        i -= 1
    j = e
    n_out = 0
    while j < len(dirs) and n_out < lead * 2:
        if dirs[j] in DIAGONALS:
            break
        j += 1
        n_out += 1
    while j < len(dirs) and node_kind(nodes[j]) != "C":
        j += 1
    return i, j


# ==========================================================================
# 走行の harness（**制御は親のまま**）
# ==========================================================================
class OneSegmentPolicy(SlalomPolicy):
    """与えられた参照経路を 1 回だけ張って走る最小の方策。

    `_replan` と `_on_path_complete` だけを上書きする。
    **`_do_drive_control`・`_advance_cursor`・車輪の制御は親のまま。**
    """

    name = "016-B one diagonal segment"

    def __init__(self, ref_path, v_ceil, **kw):
        super().__init__(**kw)
        self._ref = ref_path
        self._v_ceil_req = float(v_ceil)
        self.finished = False

    def _replan(self, x, y, yaw):
        if self._ref is None:
            self._state = "IDLE"
            return
        path = self._ref
        v_ceil = min(self._v_ceil_req, self.v_cap)
        path.speed = build_speed_profile(path.s, path.curvature, v_ceil,
                                          self.a_lat, self.a_max, self.v_creep, True)
        self._path = path
        self._cursor = 0
        self._path_ticks = 0
        self._path_end_reason = "goal"
        self._state = "DRIVE"

    def _on_path_complete(self, x, y, yaw):
        self.finished = True
        self._state = "IDLE"
        self._path = None
        self._v_setpoint = 0.0


def drive(xml_path, params, ref_path, v_ceil, start_cell, heading_deg, v_walls, h_walls,
          max_s=30.0):
    """1 区間を走らせ、制御周期ごとの (t, x, y, yaw, v) を返す（**記録だけ**）。"""
    sim = MouseSim(str(xml_path), params=params)
    sim.full_reset(cell=start_cell, heading_deg=heading_deg)
    pol = OneSegmentPolicy(ref_path, v_ceil)
    pol.bind_sim(sim)
    pol.bind_maze(v_walls, h_walls)
    pol.on_maze_start(dict(width=16, height=16))
    rec, collided = [], False
    for _ in range(int(max_s / params.control_dt)):
        obs = sim.observation()
        vl, vr = pol.act(obs)
        out = sim.step_control(vl, vr)
        x, y, yaw = sim.privileged_pose()
        v, _w = sim.privileged_velocity()
        rec.append((sim.sim_time, x, y, yaw, v))
        if out.get("collision"):
            collided = True
            break
        if pol.finished:
            break
    return np.array(rec, dtype=float), collided, pol


# ==========================================================================
# 計装（**走行後**に軌跡から計算する）
# ==========================================================================
def measure(rec, path, kinds, idxs, nodes, dirs, cell, free_m, L, W):
    """各時刻の $e_y$・$\\psi$・余裕 $m$ を計算する。"""
    px, py = path.x, path.y
    out = []
    for (t, x, y, yaw, v) in rec:
        k = int(np.argmin((px - x) ** 2 + (py - y) ** 2))
        kind, si = str(kinds[k]), int(idxs[k])
        si = min(si, len(dirs) - 1)
        e_y = lateral_deviation(x, y, nodes[si], nodes[si + 1], cell)
        psi = heading_deviation(yaw, dirs[si])
        w = (L / 2.0) * abs(math.sin(math.radians(psi))) + \
            (W / 2.0) * abs(math.cos(math.radians(psi)))
        out.append(dict(t=t, x=x, y=y, yaw=yaw, v=v, kind=kind, seg=si,
                        e_y_m=e_y, psi_deg=psi, margin_m=free_m - w - e_y,
                        margin_ignoring_psi_m=free_m - W / 2.0 - e_y))
    return out


def measure_b45(samples, path, kinds):
    """副次測定: 45° 遷移 1 回あたりの余分な時間 [s]（**A の判定に影響しない**）。"""
    arc_t, ref_v = {}, []
    for s in samples:
        if s["kind"] in ("straight", "diagonal"):
            ref_v.append(s["v"])
    v_ref = float(np.median(ref_v)) if ref_v else 0.0
    for s in samples:
        if s["kind"] == "arc":
            arc_t.setdefault(s["seg"], []).append(s["t"])
    arc_len = {}
    for k, kd in enumerate(kinds):
        if str(kd) == "arc":
            arc_len.setdefault(int(k), 0.0)
    res = []
    for seg, ts in arc_t.items():
        dt = max(ts) - min(ts)
        n = sum(1 for kd, si in zip(kinds, range(len(kinds))) if str(kd) == "arc")
        res.append(dict(seg=seg, dt_s=dt))
    total = sum(r["dt_s"] for r in res)
    return dict(v_ref_mps=v_ref, n_arcs=len(res), total_arc_time_s=total,
                per_arc=res,
                note=("遷移の余分な時間は 016-B の結果の節で「同じ弧長を v_ref で走った"
                      "場合との差」として算出する。ここでは素の滞在時間を残す"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--v-ceil", type=float, nargs="*", default=[0.6],
                    help="斜め区間の指令速度上限 [m/s]（P3' 用に複数指定できる）")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016b" / "run.json"))
    args = ap.parse_args()

    # **帯の安全弁**（裁定 R40 条件 4）。本スクリプトは**設計帯専用**であり、
    # 凍結帯の seed が混ざったら走らせない（purpose='validate' 固定）。
    _seeds = [int(q.stem.split('_')[1])
              for q in sorted((REPO_ROOT / args.maze_dir).glob('maze_*.npz'))]
    print(describe_seeds(_seeds, 'competition'))
    assert_seeds_allowed(_seeds, namespace='competition', purpose='validate')

    a, b = load_time_model()
    params = RobotParams()
    body = body_extent_exact()
    L, W = body["length_m"], body["width_m"]
    free, _ = diagonal_clearance(params.cell_size, 0.012)

    f, run_len, p, (s0, e0), v_walls, h_walls = pick_face(args.maze_dir, a, b)
    i, j = cut_segment(p["nodes"], p["dirs"], s0, e0)
    nodes, dirs = p["nodes"][i:j + 1], p["dirs"][i:j]
    print(f"面（機械的に選定）: {f.stem}／**斜めの最長連続 {run_len} 歩**"
          f"／切り出し区間 {i}〜{j}（{len(dirs)} 手）")
    start_node = nodes[0]
    start_cell = (start_node[0] // 2, start_node[1] // 2)
    heading_deg = math.degrees(math.atan2(DELTA8[dirs[0]][1], DELTA8[dirs[0]][0]))
    print(f"開始: 区画 {start_cell}・方位 {heading_deg:.0f}°／機体 {L*1000:.0f}×{W*1000:.0f} mm"
          f"／片側自由幅 {free*1000:.3f} mm")

    xml_path = f.with_suffix(".xml")
    if not xml_path.exists():
        build_maze_robot_xml(v_walls, h_walls, str(xml_path),
                             model_name=f"micromouse_016b_{f.stem}", params=params)

    results = []
    for v_ceil in args.v_ceil:
        path, kinds, idxs = build_diagonal_path(nodes, dirs, params.cell_size, R_ARC_M)
        rec, collided, pol = drive(xml_path, params, path, v_ceil, start_cell,
                                    heading_deg, v_walls, h_walls)
        sm = measure(rec, path, kinds, idxs, nodes, dirs, params.cell_size, free, L, W)
        diag = [s for s in sm if s["kind"] == "diagonal"]
        strt = [s for s in sm if s["kind"] == "straight"]
        if not diag:
            print(f"  v_ceil={v_ceil}: **斜め区間に入れなかった → 判定不能**")
            results.append(dict(v_ceil=v_ceil, verdict="判定不能", collided=collided))
            continue
        m_min = min(s["margin_m"] for s in diag)
        e_max = max(s["e_y_m"] for s in diag)
        psi_max = max(s["psi_deg"] for s in diag)
        e_max_str = max((s["e_y_m"] for s in strt), default=float("nan"))
        m_ign = min(s["margin_ignoring_psi_m"] for s in diag)
        verdict = "判定不能" if collided else ("A-成立" if m_min > 0 else "A-不成立")
        print(f"  v_ceil={v_ceil:.2f} m/s → **{verdict}**"
              f"／余裕の最小 {m_min*1000:+.2f} mm"
              f"／e_y 最大 {e_max*1000:.2f} mm（直進区間 {e_max_str*1000:.2f} mm）"
              f"／ψ 最大 {psi_max:.2f}°"
              f"／ψ を無視した余裕 {m_ign*1000:+.2f} mm"
              + ("／**衝突**" if collided else ""))
        results.append(dict(v_ceil=v_ceil, verdict=verdict, collided=collided,
                            margin_min_m=m_min, e_y_max_diag_m=e_max,
                            e_y_max_straight_m=e_max_str, psi_max_deg=psi_max,
                            margin_ignoring_psi_min_m=m_ign,
                            n_ticks=len(sm), n_diag_ticks=len(diag),
                            b45=measure_b45(sm, path, kinds),
                            samples=sm))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(git_rev=git_rev(), maze=f.stem, longest_diag_run=run_len,
                   seg_from=i, seg_to=j, R_arc_m=R_ARC_M,
                   body_length_m=L, body_width_m=W, free_half_width_m=free,
                   results=results),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
