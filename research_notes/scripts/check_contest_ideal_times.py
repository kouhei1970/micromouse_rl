#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大会実迷路 42 面の**理想タイム**を計算する（走行なし・純粋な計算）。

`note_025` 第 0 章の物差し（車両の物理限界からの最小時間速度プロファイル）を、
実際の大会で使われた迷路に当てる。「**この迷路の理論限界は何秒か**」が出る。

条件は `check_physical_limits_and_ideal_lap.py` の **(a)** と同じ:
**経路は現行の計画器の幾何のまま（区画中心を結ぶ折れ線 ＋ 円弧）、速度だけ理論限界。**
したがって出る値は「現行の経路の取り方における下界」であって、経路も理論限界に取れば
さらに短くなる（サーキットでは 8.7 % 短かった）。

    .venv/bin/python research_notes/scripts/check_contest_ideal_times.py
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_notes" / "scripts"))

from competition.baseline_slalom import (build_reference_path, fit_corner_radii,  # noqa: E402
                                         _DELTA)
from check_physical_limits_and_ideal_lap import (A_LAT, A_TR, F_stall, F_fric,    # noqa: E402
                                                 M_eff, V_TOP, c_eff, CELL)

CONTEST = ROOT / "competition" / "reference_mazes" / "contest"
PREFERRED_R = 0.060   # m 現行の計画器が使う円弧半径（016 系の既定）


def shortest_route(v_walls, h_walls, start, goals, M=16):
    """壁を考慮した幅優先探索で、区画数最短の経路（区画列）を 1 本返す。"""
    prev = {start: None}
    q = deque([start])
    goal_hit = None
    while q:
        x, y = q.popleft()
        if (x, y) in goals:
            goal_hit = (x, y)
            break
        for d, (dx, dy) in _DELTA.items():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < M and 0 <= ny < M) or (nx, ny) in prev:
                continue
            blocked = (v_walls[x + 1, y] if dx == 1 else
                       v_walls[x, y] if dx == -1 else
                       h_walls[x, y + 1] if dy == 1 else h_walls[x, y])
            if int(blocked) == 1:
                continue
            prev[(nx, ny)] = (x, y)
            q.append((nx, ny))
    if goal_hit is None:
        return None
    route = []
    cur = goal_hit
    while cur is not None:
        route.append(cur)
        cur = prev[cur]
    return route[::-1]


def min_time(s_arr, curv, ds_default=0.005):
    """最小時間の速度プロファイル（始点・終点とも静止）。摩擦円つき。"""
    n = len(s_arr)
    v = np.minimum(V_TOP, np.where(np.abs(curv) > 1e-9,
                                   np.sqrt(A_LAT / np.maximum(np.abs(curv), 1e-9)), V_TOP))
    v[0] = 0.0
    v[-1] = 0.0
    for i in range(n - 1):                       # 前進パス（加速）
        ds = s_arr[i + 1] - s_arr[i]
        ay = v[i] ** 2 * abs(curv[i])
        ax_cap = A_TR * np.sqrt(max(1 - (ay / A_LAT) ** 2, 0.0))
        ax_mot = (F_stall - c_eff * v[i] - F_fric) / M_eff
        ax = max(min(ax_cap, ax_mot), 0.0)
        v[i + 1] = min(v[i + 1], np.sqrt(max(v[i] ** 2 + 2 * ax * ds, 0.0)))
    for i in range(n - 1, 0, -1):                # 後退パス（減速）
        ds = s_arr[i] - s_arr[i - 1]
        ay = v[i] ** 2 * abs(curv[i])
        ax_cap = A_TR * np.sqrt(max(1 - (ay / A_LAT) ** 2, 0.0))
        v[i - 1] = min(v[i - 1], np.sqrt(max(v[i] ** 2 + 2 * ax_cap * ds, 0.0)))
    T = 0.0
    for i in range(n - 1):
        T += (s_arr[i + 1] - s_arr[i]) / max(0.5 * (v[i] + v[i + 1]), 1e-9)
    return T, float(v.max())


def ideal_time(npz_path):
    d = np.load(npz_path)
    v_walls, h_walls = d["v_walls"], d["h_walls"]
    M = v_walls.shape[1]
    start = (int(d["start_x"]), int(d["start_y"]))
    goals = set(zip(d["goals_x"].tolist(), d["goals_y"].tolist()))
    route = shortest_route(v_walls, h_walls, start, goals, M=M)
    if route is None or len(route) < 3:
        return None
    dirs = []
    for (x0, y0), (x1, y1) in zip(route, route[1:]):
        dirs.append(next(k for k, (dx, dy) in _DELTA.items() if (x1 - x0, y1 - y0) == (dx, dy)))
    wps = [((x + 0.5) * CELL, (y + 0.5) * CELL) for x, y in route]
    radii = fit_corner_radii(wps, PREFERRED_R)
    path = build_reference_path(route, dirs, dirs[0], radii, CELL, stop_at_end=True)
    s = np.asarray(path.s, dtype=float)
    curv = np.asarray(path.curvature, dtype=float)
    T, vmax = min_time(s, curv)
    n_turn = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)
    return dict(cells=len(route) - 1, turns=n_turn, length=float(s[-1]), T=T, vmax=vmax)


def main():
    rows = []
    for p in sorted(CONTEST.glob("contest_*.npz")):
        name = p.stem.replace("contest_", "")
        try:
            r = ideal_time(p)
        except Exception as e:                    # 経路が引けない面は理由つきで落とす
            print(f"  [skip] {name}: {type(e).__name__}: {e}")
            continue
        if r is None:
            print(f"  [skip] {name}: 経路が引けない")
            continue
        r["name"] = name
        rows.append(r)
    rows.sort(key=lambda r: r["T"])
    print(f"{'迷路':22s} {'移動':>4s} {'旋回':>4s} {'経路長[m]':>9s} {'理想[s]':>8s} {'最高[m/s]':>9s}")
    for r in rows:
        print(f"{r['name']:22s} {r['cells']:4d} {r['turns']:4d} {r['length']:9.3f} "
              f"{r['T']:8.3f} {r['vmax']:9.3f}")
    T = np.array([r["T"] for r in rows])
    cx = [r for r in rows if r["name"].endswith("CX")]
    print(f"\n全 {len(rows)} 面: 理想タイム 中央値 {np.median(T):.3f} s ／ "
          f"最小 {T.min():.3f} ／ 最大 {T.max():.3f}")
    if cx:
        Tc = np.array([r["T"] for r in cx])
        print(f"全日本決勝 CX {len(cx)} 面: 中央値 {np.median(Tc):.3f} s ／ "
              f"最小 {Tc.min():.3f} ／ 最大 {Tc.max():.3f}")
    out = ROOT / "research_notes" / "scripts" / "contest_ideal_times.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n書き出し: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
