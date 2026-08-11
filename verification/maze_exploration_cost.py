#!/usr/bin/env python3
"""迷路が「探索の遠回り」をどれだけ要求するかを、物理シミュレーション抜きで測る。

作成: 2026-08-11 准教授セッション（独立検証担当）
動機: 報告 001 §7.3 で、同一走行方式 L0-c の経路比 L_探索/L_最短 が
      評価帯 1.016 / 是正前帯 1.219 / 大会実迷路 1.566 と大きく違うことが分かった。
      これが「迷路の性質」なのか「ベースライン実装の性質」なのかを分けるため、
      迷路の壁配列だけから決まる量として測り直す。

測る量: **初回探索の遠回り率** = （未知の迷路を足立法で初めてゴールまで走ったときの区画数） / D_0

  足立法（flood-fill）の標準形を、壁の既知/未知を区別して実装する:
    - 機体は現在区画に来た時点でその区画の 4 辺の壁を観測する（既知にする）
    - 未知の壁は「通れる」とみなす（楽観地図）。楽観地図上でゴールまでの距離を毎歩計算し直し、
      距離が最小になる隣接区画へ 1 区画進む
    - ゴール区画に入った時点で終了
  これは competition/ のベースライン実装を参照せず、古典アルゴリズムの定義から書いたもの。

注意: 同点の分岐（距離が同じ隣接区画が複数ある）の解き方で経路長は変わる。
      ここでは実機の定石にならい「直進 > 右折 > 左折 > 後退」を優先する。
      **絶対値は同点処理に依存するが、全帯に同一の規則を適用しているので帯間の比較は成立する。**
      感度確認のため、別規則（固定方位順 北>東>南>西）での値も併記する。
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
N = 16
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # 北, 東, 南, 西


def load(p: Path):
    d = np.load(p, allow_pickle=True)
    m = {"v": d["v_walls"], "h": d["h_walls"]}
    if "start_x" in d.files:
        m["start"] = (int(d["start_x"]), int(d["start_y"]))
        m["goals"] = [tuple(g) for g in zip(d["goals_x"].tolist(), d["goals_y"].tolist())]
    else:
        m["goals"] = [(7, 7), (8, 7), (7, 8), (8, 8)]
        corners = [(0, 0), (N - 1, 0), (0, N - 1), (N - 1, N - 1)]
        cand = [c for c in corners if n_open_true(m, *c) == 1]
        m["start"] = cand[0] if len(cand) == 1 else (0, 0)
    return m


def true_wall(m, x, y, d) -> int:
    """区画 (x,y) の方向 d 側に壁があるか（真の迷路）。"""
    if d == 0:
        return int(m["h"][x, y + 1])
    if d == 1:
        return int(m["v"][x + 1, y])
    if d == 2:
        return int(m["h"][x, y])
    return int(m["v"][x, y])


def n_open_true(m, x, y) -> int:
    return sum(1 for d in range(4) if not true_wall(m, x, y, d))


def flood(walls_known, walls_is_wall, goals):
    """楽観地図（未知の壁は通れるとみなす）でゴールからの距離を計算する。

    walls_known[x][y][d]   : その辺を観測済みか
    walls_is_wall[x][y][d] : 観測済みの辺が壁か
    """
    dist = [[10**6] * N for _ in range(N)]
    q = deque()
    for gx, gy in goals:
        dist[gx][gy] = 0
        q.append((gx, gy))
    while q:
        x, y = q.popleft()
        for d, (dx, dy) in enumerate(DIRS):
            # 楽観: 未観測の辺は通れる
            if walls_known[x][y][d] and walls_is_wall[x][y][d]:
                continue
            nx, ny = x + dx, y + dy
            if not (0 <= nx < N and 0 <= ny < N):
                continue
            if dist[nx][ny] > dist[x][y] + 1:
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))
    return dist


def explore_first_run(m, tiebreak: str = "straight") -> int | None:
    """未知の迷路を足立法で初めてゴールへ走ったときの移動区画数を返す。"""
    known = [[[False] * 4 for _ in range(N)] for _ in range(N)]
    is_wall = [[[False] * 4 for _ in range(N)] for _ in range(N)]
    goals = set(m["goals"])

    def observe(x, y):
        """現在区画の 4 辺を観測する（隣接区画側の同じ辺も既知にする）。"""
        for d, (dx, dy) in enumerate(DIRS):
            w = true_wall(m, x, y, d)
            known[x][y][d] = True
            is_wall[x][y][d] = bool(w)
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:
                od = (d + 2) % 4
                known[nx][ny][od] = True
                is_wall[nx][ny][od] = bool(w)

    x, y = m["start"]
    heading = 0                      # 北向きで出発（スタートは 3 方向が壁）
    steps = 0
    for _ in range(4 * N * N * 4):   # 打ち切り（無限ループ保護）
        if (x, y) in goals:
            return steps
        observe(x, y)
        dist = flood(known, is_wall, m["goals"])
        best, best_dir = None, None
        # 同点処理: 直進 > 右折 > 左折 > 後退（実機の定石）／ または固定方位順
        order = ([heading, (heading + 1) % 4, (heading + 3) % 4, (heading + 2) % 4]
                 if tiebreak == "straight" else [0, 1, 2, 3])
        for d in order:
            if known[x][y][d] and is_wall[x][y][d]:
                continue
            dx, dy = DIRS[d]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < N and 0 <= ny < N):
                continue
            if best is None or dist[nx][ny] < best:
                best, best_dir = dist[nx][ny], d
        if best_dir is None:
            return None
        heading = best_dir
        x, y = x + DIRS[best_dir][0], y + DIRS[best_dir][1]
        steps += 1
    return None


def d_true(m) -> int:
    dist = [[-1] * N for _ in range(N)]
    q = deque()
    for gx, gy in m["goals"]:
        dist[gx][gy] = 0
        q.append((gx, gy))
    while q:
        x, y = q.popleft()
        for d, (dx, dy) in enumerate(DIRS):
            if true_wall(m, x, y, d):
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and dist[nx][ny] < 0:
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))
    sx, sy = m["start"]
    return dist[sx][sy]


def main() -> None:
    bands = sys.argv[1:] or ["contest_reference", "eval", "validation", "eval_v2_short"]
    out: dict[str, list[dict]] = {}
    print(f"{'帯':<20}{'n':>4}{'D0中央値':>10}{'遠回り率 中央値':>18}{'[Q1, Q3]':>20}{'min':>8}{'max':>8}"
          f"{'別同点規則':>12}")
    for band in bands:
        d = REPO / "competition" / "mazes" / band
        if not d.exists():
            continue
        rows = []
        for p in sorted(d.glob("*.npz")):
            m = load(p)
            d0 = d_true(m)
            s1 = explore_first_run(m, "straight")
            s2 = explore_first_run(m, "fixed")
            rows.append({"maze": p.stem, "d_true": d0, "explore_cells": s1,
                         "detour": (s1 / d0) if (s1 and d0) else None,
                         "detour_fixed": (s2 / d0) if (s2 and d0) else None})
        out[band] = rows
        r = [x["detour"] for x in rows if x["detour"]]
        rf = [x["detour_fixed"] for x in rows if x["detour_fixed"]]
        d0s = [x["d_true"] for x in rows]
        qs = st.quantiles(r, n=4) if len(r) >= 4 else [float("nan")] * 3
        print(f"{band:<20}{len(rows):>4}{st.median(d0s):>10.1f}{st.median(r):>18.3f}"
              f"{f'[{qs[0]:.3f}, {qs[2]:.3f}]':>20}{min(r):>8.3f}{max(r):>8.3f}"
              f"{st.median(rf):>12.3f}")
    (REPO / "verification" / "out" / "maze_exploration_cost.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
