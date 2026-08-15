#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""実機タイムと本モデルの理想タイムの差を、**経路の取り方**と**実効摩擦係数**に分解する。

2019 年の全日本クラシック決勝（迷路 `16MM2019CX`）で、実機 1 位は **4.143 s**（主催者の公開記録）。
本モデルの理想タイム（斜めなし・現行の幾何・$\\mu$=1.0）は **14.094 s** で、実機の方が 3.4 倍速い。
差の主因として次の 2 つが考えられるので、2×2（実際は 2×4）の表で分離する。

  1. **経路の取り方** — 斜め走行を使うと経路そのものが短くなる
  2. **実効摩擦係数** — 実機の上位はサクションファンで路面へ吸い付く

⚠️ 斜め区間の速度上限は**摩擦円だけ**で置く（現行の $v_\\text{斜め}$ = 0.45 m/s は使わない）。
   理想タイムは下界でなければならないため。
⚠️ **実機と本モデルは車体仕様が違う**ので、逆算した $\\mu$ は「本モデルの幾何で実機タイムを
   出すのに必要な値」であって、実機の摩擦係数そのものではない。**参考値**である。
⚠️ その年の競技で実際に使われた迷路が本 npz と同一であることは**未確認**。

    .venv/bin/python research_notes/scripts/check_ideal_time_decomposition.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (ROOT, ROOT / "research_notes" / "scripts", ROOT / "experiments" / "exp_016_diagonal"):
    sys.path.insert(0, str(p))

from competition.baseline_slalom import (_DELTA, build_reference_path,          # noqa: E402
                                         fit_corner_radii, goal_cells)
from competition.route_planner import value_field                               # noqa: E402
from diag_path import build_diagonal_path                                       # noqa: E402
from diagonal_model import DiagonalGridModel, cell_center_node                  # noqa: E402
from competition.baseline_slalom_diag import B45_S, R_SPEED_RATIO                # noqa: E402
from check_physical_limits_and_ideal_lap import (CELL, F_fric, F_stall, M_eff,  # noqa: E402
                                                 V_TOP, c_eff, eta, g, mu_c)
from check_contest_ideal_times import PREFERRED_R, shortest_route               # noqa: E402

MAZE = ROOT / "competition" / "reference_mazes" / "contest" / "contest_16MM2019CX.npz"
REAL_TIME_S = 4.143      # 2019 全日本クラシック 1 位「華金」（主催者の公開記録）
R_STRAIGHT = 0.090       # 直進 ↔ 直進の 90° コーナー（016-D と同じ）


def limits(mu):
    """実効摩擦係数 mu に対する 前後の最大加減速 と 横加速度の上限 [m/s²]。"""
    return mu * eta * g - mu_c * (1 - eta) * g, (mu * eta + mu_c * (1 - eta)) * g


def min_time(s_arr, curv, a_tr, a_lat):
    """最小時間の速度プロファイル（始点・終点とも静止・摩擦円つき）。"""
    n = len(s_arr)
    v = np.minimum(V_TOP, np.where(np.abs(curv) > 1e-9,
                                   np.sqrt(a_lat / np.maximum(np.abs(curv), 1e-9)), V_TOP))
    v[0] = v[-1] = 0.0
    for i in range(n - 1):
        ds = s_arr[i + 1] - s_arr[i]
        ay = v[i] ** 2 * abs(curv[i])
        ax = max(min(a_tr * np.sqrt(max(1 - (ay / a_lat) ** 2, 0.0)),
                     (F_stall - c_eff * v[i] - F_fric) / M_eff), 0.0)
        v[i + 1] = min(v[i + 1], np.sqrt(max(v[i] ** 2 + 2 * ax * ds, 0.0)))
    for i in range(n - 1, 0, -1):
        ds = s_arr[i] - s_arr[i - 1]
        ay = v[i] ** 2 * abs(curv[i])
        ax = a_tr * np.sqrt(max(1 - (ay / a_lat) ** 2, 0.0))
        v[i - 1] = min(v[i - 1], np.sqrt(max(v[i] ** 2 + 2 * ax * ds, 0.0)))
    return sum((s_arr[i + 1] - s_arr[i]) / max(0.5 * (v[i] + v[i + 1]), 1e-9)
               for i in range(n - 1))


def paths(z):
    """(斜めなし, 斜めあり) の (弧長, 曲率) を返す。"""
    v_walls, h_walls = z["v_walls"], z["h_walls"]
    M = v_walls.shape[1]
    start = (int(z["start_x"]), int(z["start_y"]))
    goals = set(zip(z["goals_x"].tolist(), z["goals_y"].tolist()))

    def connects(x, y, nx, ny):
        b = (v_walls[x + 1, y] if nx == x + 1 else v_walls[x, y] if nx == x - 1 else
             h_walls[x, y + 1] if ny == y + 1 else h_walls[x, y])
        return int(b) != 1

    # --- 斜めなし: 区画中心を結ぶ折れ線 ＋ 円弧（現行の計画器の幾何） ---
    route = shortest_route(v_walls, h_walls, start, goals, M=M)
    dirs = [next(k for k, d in _DELTA.items() if (b[0] - a[0], b[1] - a[1]) == d)
            for a, b in zip(route, route[1:])]
    wps = [((x + 0.5) * CELL, (y + 0.5) * CELL) for x, y in route]
    p0 = build_reference_path(route, dirs, dirs[0],
                              fit_corner_radii(wps, PREFERRED_R), CELL, stop_at_end=True)

    # --- 斜めあり: 半区画格子の最短（斜めを含む）経路 ---
    # 費用モデルの係数は実測の回帰から引く（ハードコードしない）。
    # `research_notes/data/time_model_l0c_design.json` は L0-c 専用の a・b。
    import json
    tm = json.load(open(ROOT / "research_notes" / "data" / "time_model_l0c_design.json",
                        encoding="utf-8"))
    model = DiagonalGridModel(tm["a"], tm["b"], r=R_SPEED_RATIO, turn_unit_45=B45_S)
    field = value_field(goal_cells(M, M), M, M, connects, model)
    node, d_in = cell_center_node(start), "N"
    nodes, ddirs = [node], []
    for _ in range(4 * M * M):
        if field.states.get((node, d_in), float("inf")) <= 1e-9:
            break
        best = None
        for d_out, nxt, st, w in model.successors(node, d_in, M, M, connects):
            val = field.states.get(st)
            if val is not None and (best is None or w + val < best[0] - 1e-12):
                best = (w + val, d_out, nxt)
        if best is None:
            break
        nodes.append(best[2])
        ddirs.append(best[1])
        node, d_in = best[2], best[1]
    p1, _kind, _idx = build_diagonal_path(nodes, ddirs, CELL, PREFERRED_R,
                                          stop_at_end=True, r_straight=R_STRAIGHT)
    return ((np.asarray(p0.s, float), np.asarray(p0.curvature, float)),
            (np.asarray(p1.s, float), np.asarray(p1.curvature, float)))


def main():
    z = np.load(MAZE)
    (s0, k0), (s1, k1) = paths(z)
    print(f"迷路 16MM2019CX  経路長: 斜めなし {s0[-1]:.3f} m ／ 斜めあり {s1[-1]:.3f} m "
          f"（{(s1[-1]/s0[-1]-1)*100:+.1f} %）\n")

    print("=== 理想タイム [s] — 経路の取り方 × 実効摩擦係数 ===")
    print(f"{'μ':>5}  {'斜めなし':>10s}  {'斜めあり':>10s}  {'斜めの利得':>10s}")
    for mu in (1.0, 2.0, 3.0, 4.0):
        a_tr, a_lat = limits(mu)
        t0, t1 = min_time(s0, k0, a_tr, a_lat), min_time(s1, k1, a_tr, a_lat)
        print(f"{mu:5.1f}  {t0:10.3f}  {t1:10.3f}  {(t1/t0-1)*100:+9.1f} %")

    print(f"\n=== 逆算: 実機 1 位 {REAL_TIME_S:.3f} s を本モデルの幾何で出すのに要る μ ===")
    for label, (s, k) in (("斜めなし", (s0, k0)), ("斜めあり", (s1, k1))):
        lo, hi = 0.2, 200.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            a_tr, a_lat = limits(mid)
            if min_time(s, k, a_tr, a_lat) > REAL_TIME_S:
                lo = mid
            else:
                hi = mid
        a_tr, a_lat = limits(hi)
        t = min_time(s, k, a_tr, a_lat)
        ok = abs(t - REAL_TIME_S) < 0.02
        print(f"  {label}: μ = {hi:7.2f}"
              + (f"（そのとき {t:.3f} s・前後 {a_tr:.1f} / 横 {a_lat:.1f} m/s²）" if ok
                 else f"  🔴 到達不能 — μ を上げても {t:.3f} s 止まり（最高速 {V_TOP:.2f} m/s の壁）"))


if __name__ == "__main__":
    main()
