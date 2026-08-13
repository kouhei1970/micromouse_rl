"""便 2 の判定 — (L) 刻み不変性と (C-1) 中心値のずれを**独立実装で再現する**。

学生B の報告値（design.md）:
  (L) 最大比 **1.3107 / 1.3883**（刻み 1.5 / 0.375 mm）・超過 **0.659 / 0.206 mm**（5 面）
  (C-1) **spec −0.36816 m / field −0.39499 m**（n=640）

**独立性の作り**: 場（測地距離場）も**自前で構成する** — 障害物は `mouse/mjcf.py` が
生成する箱、マスクは表面からの厳密なユークリッド距離、Dijkstra は自前の 8 近傍、
帯への延長も自前。**実装からは `_geo_field` も `_geo_allowed_mask()` も読まない。**
読むのは迷路の壁配列（`generate_maze`）と定数（格子・$w_\text{lat}$）だけである。

走査の手続きは design.md の (L) の条文と学生B の測定に合わせる（**到達可能な点対のみ**・
軸 2 種＋斜め 2 種・全区画）。**手続きを合わせないと数値が比較できない**ため。
"""
import heapq
import json
import math
import sys
import time

REPO_ROOT = "/Users/kouhei/tmp/github/micromouse_rl"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, f"{REPO_ROOT}/verification")

import numpy as np

from audit_r33_config_space import CS, MODE, W_LAT, obstacle_rects
from mouse.maze6_env import _GEO_GRID_H, _GEO_GRID_N
from mouse.maze6_gen import GOAL_CELLS, SIZE, generate_maze, shortest_distances

H = _GEO_GRID_H
N = _GEO_GRID_N
DILATE = W_LAT - H * math.sqrt(2.0)          # 解く領域の離隔 0.03151
LEVELS = (120, 480)                          # 区画一辺 0.18 を N 分割 → 1.5 / 0.375 mm
SEEDS_L = list(range(7000, 7005))            # (L) は 5 面
SEEDS_C1 = list(range(7000, 7020))           # (C-1) は 20 面


def boxes_of(seed):
    """(cx, cy, hx, hy) 形式の障害物の箱（mjcf が生成するものと同じ幾何）。"""
    m = generate_maze(seed, mode=MODE)
    out = []
    for (x0, x1, y0, y1) in obstacle_rects(m["v_walls"], m["h_walls"], True, False):
        out.append(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2))
    return m, np.asarray(out)


def dist_to_boxes(X, Y, boxes):
    """各点から最も近い障害物までの距離。箱は軸並行なので厳密。"""
    d2 = np.full(X.shape, np.inf)
    for cx, cy, hx, hy in boxes:
        dx = np.maximum(np.abs(X - cx) - hx, 0.0)
        dy = np.maximum(np.abs(Y - cy) - hy, 0.0)
        np.minimum(d2, dx * dx + dy * dy, out=d2)
    return np.sqrt(d2)


def build_field(boxes):
    """測地距離場を自前で構成する（マスク → 8 近傍 Dijkstra → 帯への延長）。"""
    pos = np.arange(N) * H
    X = pos[:, None] * np.ones((1, N))
    Y = np.ones((N, 1)) * pos[None, :]
    dmin = dist_to_boxes(X, Y, boxes)
    solve = dmin >= DILATE                       # 解く領域（1 格子セルぶん膨張）
    cell = np.minimum((pos / CS).astype(int), SIZE - 1)
    gx = sorted({c[0] for c in GOAL_CELLS})
    gy = sorted({c[1] for c in GOAL_CELLS})
    in_g = np.isin(cell, gx)[:, None] & np.isin(cell, gy)[None, :]

    dist = np.full((N, N), np.inf)
    heap = []
    for i, j in zip(*np.where(in_g & solve)):
        dist[i, j] = 0.0
        heap.append((0.0, int(i), int(j)))
    heapq.heapify(heap)
    d2 = H * math.sqrt(2.0)
    nb = ((1, 0, H), (-1, 0, H), (0, 1, H), (0, -1, H),
          (1, 1, d2), (1, -1, d2), (-1, 1, d2), (-1, -1, d2))

    def run(allowed):
        while heap:
            du, i, j = heapq.heappop(heap)
            if du > dist[i, j]:
                continue
            for di, dj, w in nb:
                a, b = i + di, j + dj
                if 0 <= a < N and 0 <= b < N and allowed[a, b] and du + w < dist[a, b]:
                    dist[a, b] = du + w
                    heapq.heappush(heap, (du + w, a, b))

    run(solve)                                   # 第 1 段: 解く領域だけ
    heap = [(float(dist[i, j]), int(i), int(j))
            for i, j in zip(*np.where(np.isfinite(dist)))]
    heapq.heapify(heap)
    outside = ~solve
    run(outside)                                 # 第 2 段: 帯へ延長（解けた値は変えない）
    return dist


def bilinear(field, xs, ys):
    i0 = np.clip(np.floor(xs / H).astype(np.int64), 0, N - 2)
    j0 = np.clip(np.floor(ys / H).astype(np.int64), 0, N - 2)
    tx = xs / H - i0
    ty = ys / H - j0
    return ((1 - tx) * (1 - ty) * field[i0, j0] + tx * (1 - ty) * field[i0 + 1, j0]
            + (1 - tx) * ty * field[i0, j0 + 1] + tx * ty * field[i0 + 1, j0 + 1])


def measure_L():
    print("=" * 78)
    print(f"(L) 刻み不変性 — 独立実装（{len(SEEDS_L)} 面・到達可能な点対のみ）")
    print("=" * 78)
    out = []
    for Nd in LEVELS:
        step = CS / Nd
        diag = step * math.sqrt(2.0)
        best_r, best_e, n_pairs = 0.0, -1e9, 0
        at_r = at_e = None
        t0 = time.time()
        for seed in SEEDS_L:
            m, boxes = boxes_of(seed)
            field = build_field(boxes)
            n_side = SIZE * Nd + 1
            chunk = max(1, int(1_500_000 / n_side))
            yj = np.arange(n_side) * step
            for i0 in range(0, n_side - 1, chunk):
                i1 = min(i0 + chunk + 1, n_side)
                xi = np.arange(i0, i1) * step
                X = xi[:, None] * np.ones((1, n_side))
                Y = np.ones((len(xi), 1)) * yj[None, :]
                # 箱の枝刈り（この塊から w_lat 以内に無い箱は到達判定に効かない）
                lo_x, hi_x = xi[0] - W_LAT, xi[-1] + W_LAT
                sel = boxes[(boxes[:, 0] + boxes[:, 2] >= lo_x)
                            & (boxes[:, 0] - boxes[:, 2] <= hi_x)]
                R = dist_to_boxes(X, Y, sel if len(sel) else boxes) >= W_LAT
                G = bilinear(field, X, Y)
                for di, dj, dist_ in ((1, 0, step), (0, 1, step),
                                      (1, 1, diag), (1, -1, diag)):
                    a = (slice(0, len(xi) - di if di else None),
                         slice(max(0, -dj), (n_side - dj) if dj > 0 else None))
                    b = (slice(di, None) if di else slice(0, None),
                         slice(max(0, dj), (n_side + dj) if dj < 0 else None))
                    ok = R[a] & R[b]
                    if not ok.any():
                        continue
                    d = np.abs(G[b] - G[a])
                    n_pairs += int(ok.sum())
                    r = np.where(ok, d / dist_, 0.0)
                    e = np.where(ok, d - dist_, -1e9)
                    k = int(np.argmax(r))
                    if r.flat[k] > best_r:
                        best_r = float(r.flat[k])
                        at_r = (seed, (di, dj))
                    k = int(np.argmax(e))
                    if e.flat[k] > best_e:
                        best_e = float(e.flat[k])
                        at_e = (seed, (di, dj))
        print(f"  刻み {step*1000:.3f} mm: 最大比 {best_r:.4f} {at_r}  "
              f"最大超過 {best_e*1000:.3f} mm {at_e}  "
              f"（点対 {n_pairs:,} / {time.time()-t0:.0f} s）")
        out.append({"step_mm": step * 1000, "max_ratio": best_r,
                    "max_excess_mm": best_e * 1000, "n_pairs": n_pairs})
    return out


def measure_C1():
    print()
    print("=" * 78)
    print("(C-1) 区画中心の値と d·cs の差 — 独立実装（20 面 × 非ゴール 32 区画）")
    print("=" * 78)
    res = {}
    for label, clear in (("spec", W_LAT), ("field", DILATE)):
        diffs = []
        for seed in SEEDS_C1:
            m, boxes = boxes_of(seed)
            if label == "spec":
                # 仕様マスク（膨張なし）で解いた場
                saved = globals()["DILATE"]
                globals()["DILATE"] = W_LAT
                field = build_field(boxes)
                globals()["DILATE"] = saved
            else:
                field = build_field(boxes)
            dmap = shortest_distances(m["v_walls"], m["h_walls"])
            for cx in range(SIZE):
                for cy in range(SIZE):
                    if (cx, cy) in GOAL_CELLS:
                        continue
                    i = int(round((cx + 0.5) * CS / H))
                    j = int(round((cy + 0.5) * CS / H))
                    diffs.append(float(field[i, j]) - int(dmap[(cx, cy)]) * CS)
        a = np.array(diffs)
        res[label] = {"median": float(np.median(a)), "mean": float(a.mean()),
                      "min": float(a.min()), "max": float(a.max()), "n": len(a)}
        print(f"  C-1_{label:<5}: 中央値 {np.median(a):.5f} m  平均 {a.mean():.5f}  "
              f"範囲 {a.min():.5f}〜{a.max():.5f}  n={len(a)}")
    return res


def main():
    out = {"L": measure_L(), "C1": measure_C1()}
    print()
    print("=" * 78)
    print("学生B の報告値との突き合わせ")
    print("=" * 78)
    ref_L = {1.5: (1.3107, 0.659), 0.375: (1.3883, 0.206)}
    for r in out["L"]:
        k = round(r["step_mm"], 3)
        ref = ref_L.get(k)
        if ref:
            print(f"  刻み {k} mm: 比 {r['max_ratio']:.4f} 対 {ref[0]}"
                  f"（差 {r['max_ratio']-ref[0]:+.4f}） / "
                  f"超過 {r['max_excess_mm']:.3f} 対 {ref[1]} mm"
                  f"（差 {r['max_excess_mm']-ref[1]:+.3f}）")
    for k, ref in (("spec", -0.36816), ("field", -0.39499)):
        v = out["C1"][k]["median"]
        print(f"  C-1_{k}: {v:.5f} 対 {ref}（差 {v-ref:+.5f}）")
    with open(f"{REPO_ROOT}/verification/out/bin2_L_C1.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
