#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""距離最適と時間最適はずれるか — 古典ベースラインの原理的な限界の検証。

**仮説（教授、2026-08-11）**: 足立法が最適化しているのは**距離**であって**時間**ではない。
L0-a は区画ごとに停止するので旋回は時間コストを持つ。距離が同じでも旋回の多い経路は
遅い。したがって「距離最適だが旋回が多い経路」が「1 区画長いが直線的な経路」より
遅くなることは原理的に起こりうる。

**反証条件を先に定める**（結論を先に置かない）:
  仮説が偽なら → 最短距離の経路の中で旋回回数を最小にしても、また距離を 1〜数区画
  伸ばして旋回を減らしても、**推定所要時間は縮まない**（差がゼロ）。
  仮説が真なら → 時間最適経路の推定所要時間が距離最適経路より**有意に短い**面が
  存在し、その差は旋回回数の差で説明できる。

**時間モデルはハードコードしない。**実測（exp_007 の最速走行タイム）から
  時間 = a * (移動区画数) + b * (旋回回数)
を最小二乗で当てはめて a, b を求め、そのモデルの上で 2 つの経路を比較する。
モデルの当てはまり（決定係数と残差）も出す — 当てはまらないモデルの上での
比較は無意味なので、**当てはまりが悪ければ結論を出さない**。

探索は状態 (区画, 進行方向) 上の Dijkstra で行う（旋回コストを表現するには
区画だけでは足りない）。

使い方:
    .venv/bin/python research_notes/scripts/check_distance_vs_time_optimal.py \
        --results competition/results/exp007/eval/adachi_classical_20260811_070512 \
        --maze-dir competition/mazes/eval
"""
import argparse
import glob
import heapq
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "competition" / "reference_mazes"))

from compare_generated_vs_contest import cells_open, load_contest, load_generated  # noqa: E402

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))     # 北・東・南・西
N = 16


def dijkstra_time(v, h, start, goals, turn_cost):
    """状態 (区画, 進行方向) 上で「移動 1 + 旋回 turn_cost」を最小化する。

    返り値: (最小コスト, 移動区画数, 旋回回数)。開始時の向きは自由とする
    （スタート区画は 3 方向壁なので実質 1 方向しかない）。
    """
    best = {}
    pq = []
    for d in range(4):
        heapq.heappush(pq, (0.0, start, d, 0, 0))
    while pq:
        cost, cell, d, nmove, nturn = heapq.heappop(pq)
        if cell in goals:
            return cost, nmove, nturn
        if best.get((cell, d), float("inf")) <= cost:
            continue
        best[(cell, d)] = cost
        for nd in range(4):
            dx, dy = DIRS[nd]
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nxt[0] < N and 0 <= nxt[1] < N):
                continue
            if not cells_open(v, h, cell, nxt):
                continue
            # 旋回は 90 度単位。180 度は 2 回分として数える
            t = min((nd - d) % 4, (d - nd) % 4)
            heapq.heappush(pq, (cost + 1.0 + turn_cost * t, nxt, nd, nmove + 1, nturn + t))
    return float("inf"), -1, -1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="exp_007 の結果ディレクトリ（L0-a）")
    ap.add_argument("--maze-dir", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.results, "maze_*.json"))):
        j = json.load(open(f, encoding="utf-8"))
        if j.get("best_time") is None:
            continue
        npz = os.path.join(REPO_ROOT, args.maze_dir, j["maze_id"] + ".npz")
        z = np.load(npz)
        loader = load_contest if "goals_x" in z else load_generated
        v, h, s, g, _ = loader(npz)
        goals = set(g)
        # 距離最適（旋回コスト 0）— 同じ距離なら旋回が少ない方を選ぶよう微小な重みを置く
        _, nmove_d, nturn_d = dijkstra_time(v, h, s, goals, turn_cost=1e-6)
        rows.append(dict(maze_id=j["maze_id"], t=j["best_time"],
                         nmove=nmove_d, nturn=nturn_d, v=v, h=h, s=s, goals=goals))

    # --- 時間モデル t = a*移動 + b*旋回 を最小二乗で当てはめる（ハードコードしない）
    A = np.array([[r["nmove"], r["nturn"]] for r in rows], dtype=float)
    y = np.array([r["t"] for r in rows], dtype=float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = A @ coef
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    print(f"時間モデルの当てはめ（n={len(rows)} 面、実測の最速走行タイムから導出）")
    print(f"  時間 [s] = {a:.4f} × 移動区画数 + {b:.4f} × 旋回回数")
    print(f"  決定係数 R² = {r2:.4f}、残差の最大 {np.max(np.abs(resid)):.2f} s、"
          f"RMS {np.sqrt(np.mean(resid**2)):.2f} s")
    if r2 < 0.95:
        print("  ** 当てはまりが不十分。このモデルの上での比較は行わない **")
        return 1
    turn_cost = b / a          # 旋回 1 回は移動 turn_cost 区画ぶんの時間

    print(f"\n旋回 1 回のコスト = 移動 {turn_cost:.3f} 区画ぶん")
    print("\n距離最適経路 vs 時間最適経路（同一モデル上での推定所要時間）")
    print(f"{'面':<12}{'距離最適':>22}{'時間最適':>22}{'推定時間 距離最適':>18}"
          f"{'時間最適':>10}{'短縮':>9}{'短縮率':>8}")
    print(f"{'':<12}{'区画/旋回':>22}{'区画/旋回':>22}")
    gains = []
    for r in rows:
        _, nm_t, nt_t = dijkstra_time(r["v"], r["h"], r["s"], r["goals"], turn_cost=turn_cost)
        t_dist = a * r["nmove"] + b * r["nturn"]
        t_time = a * nm_t + b * nt_t
        gain = t_dist - t_time
        gains.append(gain / t_dist)
        flag = " ←" if gain > 0.05 else ""
        left = f"{r['nmove']} / {r['nturn']}"
        right = f"{nm_t} / {nt_t}"
        print(f"{r['maze_id']:<12}{left:>22}{right:>22}{t_dist:>18.2f}{t_time:>10.2f}"
              f"{gain:>9.2f}{gain/t_dist*100:>7.2f}%{flag}")
    g = np.array(gains)
    print(f"\n短縮率: 中央値 {np.median(g)*100:.2f}%、最大 {g.max()*100:.2f}%、"
          f"改善する面 {int((g > 1e-9).sum())}/{len(g)}")
    print("\n判定:")
    if (g > 1e-9).sum() == 0:
        print("  時間最適経路は距離最適経路と一致する → **仮説は支持されない**")
    else:
        print(f"  時間最適経路の方が速い面が {int((g>1e-9).sum())} 面ある → "
              f"**仮説を支持する**（最大 {g.max()*100:.2f}% の短縮余地）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
