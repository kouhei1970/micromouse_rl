"""測定 F（裁定 R37 の受注）— 8 近傍 Dijkstra と Fast Marching 法（FMM）の差。

事前登録: `AUDIT_006` §4-nonies (b)（**本スクリプトを書く前にコミット済み**）。
用途は R37 の誤差条文の**実測点**であり、「どちらが正しいか」を決めるものではない
（場の定義は R37 で 8 近傍に確定している）。

構成:
  1. **校正試験（既知解）**: 障害物の無い自由空間に点源を置き、
     厳密なユークリッド距離との比を**方位の関数**として出す。
     答えが解析的に分かっているので、両解法の異方性を**非循環に**測れる。
  2. **迷路での測定**: seed 7000・現行マスク・6 mm 格子で $1/\rho$ を両解法で計算する。

FMM は自前実装（`.venv` に scipy が無い）。1 次精度の Godunov 上流差分:
  |∇T| = 1 を T=0（始点集合）から解く。各点で x 方向・y 方向の上流値を
  Tx, Ty とすると、|Tx − Ty| < h のとき
      T = (Tx + Ty + sqrt(2h² − (Tx − Ty)²)) / 2
  そうでなければ T = min(Tx, Ty) + h。
"""
import heapq
import json
import math
import sys
from types import SimpleNamespace

REPO_ROOT = "/Users/kouhei/tmp/github/micromouse_rl"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, f"{REPO_ROOT}/verification")

import numpy as np

from audit_r33_config_space import CS, MODE, impl_rule_mask
from mouse.maze6_env import _GEO_CLEARANCE, _GEO_GRID_H, _GEO_GRID_N
from mouse.maze6_gen import GOAL_CELLS, generate_maze, shortest_distances

FAR, TRIAL, KNOWN = 0, 1, 2


def dijkstra8(mask, h, sources):
    """8 近傍 Dijkstra（実装と同じ計量）。sources は (i, j) の一覧。"""
    nx, ny = mask.shape
    dist = np.full((nx, ny), math.inf)
    heap = []
    for (i, j) in sources:
        if mask[i, j]:
            dist[i, j] = 0.0
            heapq.heappush(heap, (0.0, i, j))
    d2 = h * math.sqrt(2.0)
    nbrs = ((1, 0, h), (-1, 0, h), (0, 1, h), (0, -1, h),
            (1, 1, d2), (1, -1, d2), (-1, 1, d2), (-1, -1, d2))
    while heap:
        du, i, j = heapq.heappop(heap)
        if du > dist[i, j]:
            continue
        for di, dj, w in nbrs:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and mask[a, b] and du + w < dist[a, b]:
                dist[a, b] = du + w
                heapq.heappush(heap, (du + w, a, b))
    return dist


def fmm(mask, h, sources):
    """Fast Marching 法（1 次精度・Godunov 上流差分）。等方に近い解法。"""
    nx, ny = mask.shape
    T = np.full((nx, ny), math.inf)
    state = np.full((nx, ny), FAR, dtype=np.int8)
    heap = []
    for (i, j) in sources:
        if mask[i, j]:
            T[i, j] = 0.0
            state[i, j] = TRIAL
            heapq.heappush(heap, (0.0, i, j))

    def solve(i, j):
        """近傍の KNOWN 値から (i, j) の到着時刻を解く。"""
        best = []
        for di, dj in ((1, 0), (-1, 0)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and state[a, b] == KNOWN:
                best.append(T[a, b])
        tx = min(best) if best else math.inf
        best = []
        for di, dj in ((0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and state[a, b] == KNOWN:
                best.append(T[a, b])
        ty = min(best) if best else math.inf
        if math.isinf(tx) and math.isinf(ty):
            return math.inf
        if math.isinf(ty):
            return tx + h
        if math.isinf(tx):
            return ty + h
        if abs(tx - ty) < h:
            return 0.5 * (tx + ty + math.sqrt(2.0 * h * h - (tx - ty) ** 2))
        return min(tx, ty) + h

    while heap:
        tu, i, j = heapq.heappop(heap)
        if state[i, j] == KNOWN or tu > T[i, j]:
            continue
        state[i, j] = KNOWN
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and mask[a, b] and state[a, b] != KNOWN:
                t = solve(a, b)
                if t < T[a, b]:
                    T[a, b] = t
                    state[a, b] = TRIAL
                    heapq.heappush(heap, (t, a, b))
    return T


def calibration(n=241, h=0.006):
    """【校正試験・既知解】自由空間の点源。厳密なユークリッド距離との比を方位別に。"""
    mask = np.ones((n, n), dtype=bool)
    c = n // 2
    d8 = dijkstra8(mask, h, [(c, c)])
    df = fmm(mask, h, [(c, c)])
    rows = []
    r_px = c - 5                                   # 端の影響を避ける半径
    for deg in range(0, 46, 1):
        th = math.radians(deg)
        i = c + int(round(r_px * math.cos(th)))
        j = c + int(round(r_px * math.sin(th)))
        exact = h * math.hypot(i - c, j - c)       # 厳密なユークリッド距離
        rows.append({"deg": deg, "exact": exact,
                     "d8_ratio": d8[i, j] / exact, "fmm_ratio": df[i, j] / exact})
    return rows


def main():
    out = {}
    print("=" * 78)
    print("1. 校正試験（自由空間の点源・答えは解析的に既知）")
    print("=" * 78)
    rows = calibration()
    print(f"{'方位':>5} {'8 近傍 / 厳密':>14} {'FMM / 厳密':>14}")
    for r in rows:
        if r["deg"] % 5 == 0:
            print(f"{r['deg']:>4}° {r['d8_ratio']:>14.5f} {r['fmm_ratio']:>14.5f}")
    d8r = [r["d8_ratio"] for r in rows]
    fmr = [r["fmm_ratio"] for r in rows]
    i8 = int(np.argmax(d8r))
    ifm = int(np.argmax([abs(v - 1.0) for v in fmr]))
    print(f"\n8 近傍: 最大 {max(d8r):.5f} (方位 {rows[i8]['deg']}°)  "
          f"理論値 {math.sqrt(4 - 2 * math.sqrt(2)):.5f} @22.5°  "
          f"0°={d8r[0]:.5f}  45°={d8r[45]:.5f}")
    print(f"FMM   : 最大偏差 {abs(fmr[ifm]-1.0)*100:.3f}% (方位 {rows[ifm]['deg']}°)  "
          f"範囲 {min(fmr):.5f}〜{max(fmr):.5f}")
    out["calibration"] = rows

    print()
    print("=" * 78)
    print("2. 迷路での 1/ρ（seed 7000・現行マスク・6 mm 格子）")
    print("=" * 78)
    # ⚠️ マスクは実装を呼ばず、**私が再実装した現行規則**（区画ごとの軸分離・
    # 閉じた境界線から _GEO_CLEARANCE）で作る。理由: 測定中に共有ツリーの
    # mouse/maze6_env.py が学生B の R34 実装で書き換わり、実装を呼ぶと
    # 「作りかけのマスク」を測ることになるため。**両解法に同じマスクを渡すことが
    # 本測定の要件**であり、どちらのマスクかは計量の比較には効かない。
    h = _GEO_GRID_H
    xs = np.arange(_GEO_GRID_N) * h
    res = []
    for seed in (7000, 7001, 7002, 7003, 7004):
        m = generate_maze(seed, mode=MODE)
        mask = impl_rule_mask(xs, xs, m["v_walls"], m["h_walls"], _GEO_CLEARANCE)
        gx = sorted({c[0] for c in GOAL_CELLS})
        gy = sorted({c[1] for c in GOAL_CELLS})
        idx = np.arange(_GEO_GRID_N)
        cell = np.minimum((idx * h / CS).astype(int), 5)
        src = [(int(i), int(j)) for i in idx[np.isin(cell, gx)]
               for j in idx[np.isin(cell, gy)] if mask[i, j]]
        dmap = shortest_distances(m["v_walls"], m["h_walls"])
        start = tuple(int(v) for v in m["start"])
        d0 = int(dmap[start])
        si = int(round((start[0] + 0.5) * CS / h))
        sj = int(round((start[1] + 0.5) * CS / h))
        g8 = dijkstra8(mask, h, src)[si, sj] / (CS * d0)
        gf = fmm(mask, h, src)[si, sj] / (CS * d0)
        res.append({"seed": seed, "D0": d0, "inv_rho_d8": float(g8), "inv_rho_fmm": float(gf),
                    "excess_pct": float((g8 / gf - 1.0) * 100)})
        print(f"seed {seed}: D0={d0:>2}  8 近傍 {g8:.4f}   FMM {gf:.4f}   "
              f"8 近傍の超過 {(g8/gf-1)*100:+.2f}%")
    ex = [r["excess_pct"] for r in res]
    print(f"\n8 近傍の超過: 中央値 {np.median(ex):+.2f}%  範囲 {min(ex):+.2f}〜{max(ex):+.2f}%")
    out["maze"] = res

    path = f"{REPO_ROOT}/verification/out/r37_metric.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"書き出し: {path}")


if __name__ == "__main__":
    main()
