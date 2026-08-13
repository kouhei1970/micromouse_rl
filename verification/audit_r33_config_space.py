"""裁定 R33 の突き合わせ — 実装の禁止帯と、准教授の配置空間測定の禁止帯が同一かを直接検定する。

教授の裁定 R33（2026-08-13）:
  実装 `_geo_allowed_mask()` は壁を「厚みゼロの線（区画境界線＝壁中心線）」として扱い、
  壁厚 12 mm の帯はどこにも描いていない。したがって _GEO_CLEARANCE = 0.0460 に
  二重計上は無い。

本スクリプトが答える 2 問（教授の確認依頼）:
  ① 私の配置空間測定（壁帯を描き、壁面から 0.0400 で侵食）の禁止帯が、
     実装の禁止帯（中心線 ±0.0460）と**同一集合か**
  ② `AUDIT_011` §5-ter の表の 0.0460 行（1/ρ = 0.7224）は「壁帯＋壁面から 0.0460」
     ＝ 禁止帯 中心線 ±0.0520 の値か（再現できるか）

検定の作り:
  - 実装のマスクは **実装のコードをそのまま呼ぶ**（`Maze6Env._geo_allowed_mask`）。
    MuJoCo を建てずに済むよう、maze と params だけを持つ shim に束縛して呼ぶ。
  - 私のマスクは **物理モデル（mouse/mjcf.py）が実際に置く箱**から作る:
    壁 = 中心線 ±t_w/2・長手は柱の間（cs/2 - POST/2 の半長）、柱 = 格子点に 12 mm 角。
    侵食は矩形への厳密なユークリッド距離（≥ R）で行う（画像処理の近似を使わない）。
  - 1/ρ = g(start) / (0.18·D_0)。8 近傍 Dijkstra・格子 7.5 mm（AUDIT_011 と同じ）。
"""
import heapq
import json
import math
import sys
from types import SimpleNamespace

REPO_ROOT = "/Users/kouhei/tmp/github/micromouse_rl"
sys.path.insert(0, REPO_ROOT)

import numpy as np

from mouse.maze6_env import _GEO_CLEARANCE, _GEO_GRID_H, _GEO_GRID_N, Maze6Env
from mouse.maze6_gen import GOAL_CELLS, SIZE, generate_maze, shortest_distances
from mouse.mjcf import POST_SIZE, WALL_THICKNESS

CS = 0.18                      # 区画一辺 [m]
W_LAT = 0.0400                 # 機体の真の最外半幅（裁定 R26）[m]
SEEDS = list(range(1, 21))     # AUDIT_011 §2-2 と同じ学習迷路 6×6・loop
MODE = "loop"


# ----------------------------------------------------------------------
# 物理モデルの障害物（mouse/mjcf.py が実際に置く箱と同じ）
# ----------------------------------------------------------------------
def obstacle_rects(v_walls, h_walls, with_posts=True, full_length=False):
    """(x0, x1, y0, y1) の矩形の一覧を返す。mjcf.build_maze_robot_xml と同じ幾何。

    full_length=True は「壁を区画境界いっぱいの帯として描く」素朴な作り
    （柱の位置に隙間を空けない）。私の旧測定がどちらの作りだったかを判別するため。
    """
    rects = []
    tw2 = WALL_THICKNESS / 2
    half_long = CS / 2 if full_length else CS / 2 - POST_SIZE / 2
    if with_posts:
        p2 = POST_SIZE / 2
        for x in range(SIZE + 1):
            for y in range(SIZE + 1):
                px, py = x * CS, y * CS
                rects.append((px - p2, px + p2, py - p2, py + p2))
    for x in range(v_walls.shape[0]):
        for y in range(v_walls.shape[1]):
            if v_walls[x, y] == 1:
                px, py = x * CS, y * CS + CS / 2
                rects.append((px - tw2, px + tw2, py - half_long, py + half_long))
    for x in range(h_walls.shape[0]):
        for y in range(h_walls.shape[1]):
            if h_walls[x, y] == 1:
                px, py = x * CS + CS / 2, y * CS
                rects.append((px - half_long, px + half_long, py - tw2, py + tw2))
    return rects


def erode_mask(xs, ys, rects, radius):
    """格子点 (xs[i], ys[j]) が全矩形から radius 以上離れているか（配置空間の自由空間）。

    矩形への厳密なユークリッド距離を使う（角では円弧になる ＝ 円板による侵食）。
    """
    X = xs[:, None]
    Y = ys[None, :]
    ok = np.ones((len(xs), len(ys)), dtype=bool)
    for (x0, x1, y0, y1) in rects:
        dx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
        if radius <= 0.0:
            # 侵食なし（点の測地）: 矩形の**内部**だけを禁止する。
            # dx=dy=0 は矩形の内部か辺上を意味するので、これで内部が落ちる。
            ok &= ~((dx == 0.0) & (dy == 0.0))
        else:
            ok &= (dx * dx + dy * dy) >= radius * radius
    return ok


def impl_mask(maze, n=None, h=None):
    """実装の `_geo_allowed_mask()` をそのまま呼ぶ（MuJoCo は建てない）。"""
    shim = SimpleNamespace(maze=maze, params=SimpleNamespace(cell_size=CS))
    return Maze6Env._geo_allowed_mask(shim)


def impl_rule_mask(xs, ys, v_walls, h_walls, clearance):
    """実装と**同じ規則**（区画ごとの軸分離・閉じた境界線から clearance）を任意格子で。

    実装のマスクは 6 mm 格子に固定なので、7.5 mm 格子で 1/ρ を測るために規則だけを写す。
    """
    from mouse.maze6_gen import cells_open
    nx, ny = len(xs), len(ys)
    cellx = np.minimum((xs / CS).astype(int), SIZE - 1)
    celly = np.minimum((ys / CS).astype(int), SIZE - 1)
    lo_x, hi_x = xs - cellx * CS, (cellx + 1) * CS - xs
    lo_y, hi_y = ys - celly * CS, (celly + 1) * CS - ys
    ok = np.ones((nx, ny), dtype=bool)
    for cx in range(SIZE):
        ix = np.flatnonzero(cellx == cx)
        for cy in range(SIZE):
            iy = np.flatnonzero(celly == cy)
            if len(ix) == 0 or len(iy) == 0:
                continue
            c = (cx, cy)
            bx_lo = not (cx > 0 and cells_open(v_walls, h_walls, c, (cx - 1, cy)))
            bx_hi = not (cx < SIZE - 1 and cells_open(v_walls, h_walls, c, (cx + 1, cy)))
            by_lo = not (cy > 0 and cells_open(v_walls, h_walls, c, (cx, cy - 1)))
            by_hi = not (cy < SIZE - 1 and cells_open(v_walls, h_walls, c, (cx, cy + 1)))
            ox = np.ones(len(ix), dtype=bool)
            oy = np.ones(len(iy), dtype=bool)
            if bx_lo:
                ox &= lo_x[ix] >= clearance
            if bx_hi:
                ox &= hi_x[ix] >= clearance
            if by_lo:
                oy &= lo_y[iy] >= clearance
            if by_hi:
                oy &= hi_y[iy] >= clearance
            ok[np.ix_(ix, iy)] = ox[:, None] & oy[None, :]
    return ok


# ----------------------------------------------------------------------
# 8 近傍 Dijkstra（ゴール 2×2 の内部の格子点を距離 0 の始点集合とする）
# ----------------------------------------------------------------------
def geodesic_from_goal(mask, h):
    nx, ny = mask.shape
    INF = math.inf
    dist = np.full((nx, ny), INF)
    heap = []
    gx = sorted({c[0] for c in GOAL_CELLS})
    gy = sorted({c[1] for c in GOAL_CELLS})
    xi = np.arange(nx)
    cellx = np.minimum((xi * h / CS).astype(int), SIZE - 1)
    in_gx = np.isin(cellx, gx)
    in_gy = np.isin(np.minimum((np.arange(ny) * h / CS).astype(int), SIZE - 1), gy)
    for i in np.flatnonzero(in_gx):
        for j in np.flatnonzero(in_gy):
            if mask[i, j]:
                dist[i, j] = 0.0
                heapq.heappush(heap, (0.0, int(i), int(j)))
    d1, d2 = h, h * math.sqrt(2.0)
    nbrs = ((1, 0, d1), (-1, 0, d1), (0, 1, d1), (0, -1, d1),
            (1, 1, d2), (1, -1, d2), (-1, 1, d2), (-1, -1, d2))
    while heap:
        du, i, j = heapq.heappop(heap)
        if du > dist[i, j]:
            continue
        for di, dj, w in nbrs:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and mask[a, b]:
                nd = du + w
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    heapq.heappush(heap, (nd, a, b))
    return dist


def inv_rho(mask, h, start_cell, d0):
    dist = geodesic_from_goal(mask, h)
    sx = (start_cell[0] + 0.5) * CS
    sy = (start_cell[1] + 0.5) * CS
    i, j = int(round(sx / h)), int(round(sy / h))
    g = dist[i, j]
    if not math.isfinite(g):
        return None
    return g / (CS * d0)


# ======================================================================
def main():
    out = {"clearance_impl": _GEO_CLEARANCE, "w_lat": W_LAT,
           "wall_thickness": WALL_THICKNESS, "post_size": POST_SIZE}

    # ------------------------------------------------------------------
    # 【問①】実装のマスクと、私の配置空間マスクの一致検定（実装と同じ 6 mm 格子）
    # ------------------------------------------------------------------
    h6 = _GEO_GRID_H
    xs6 = np.arange(_GEO_GRID_N) * h6
    print("=" * 78)
    print(f"【問①】マスクの一致検定  格子 {h6*1000:.1f} mm × {_GEO_GRID_N}^2  seed {SEEDS[0]}–{SEEDS[-1]}")
    print("=" * 78)
    variants = {
        # 壁帯＋柱（＝物理モデルそのもの）、壁面から 0.0400
        "mine_post_R0400": dict(posts=True, full=False, R=W_LAT),
        # 壁を境界いっぱいの帯で描き柱なし（＝柱の効果を落とした素朴な作り）
        "mine_full_R0400": dict(posts=False, full=True, R=W_LAT),
        # 壁は柱の間だけ・柱なし（＝格子点に 12 mm の穴が開く。分解用）
        "mine_nopost_R0400": dict(posts=False, full=False, R=W_LAT),
    }
    agg = {k: dict(diff=0, mine_only=0, impl_only=0, total=0) for k in variants}
    per_seed = {}
    for seed in SEEDS:
        m = generate_maze(seed, mode=MODE)
        im = impl_mask(m)
        rec = {}
        for key, cfg in variants.items():
            rects = obstacle_rects(m["v_walls"], m["h_walls"],
                                   with_posts=cfg["posts"], full_length=cfg["full"])
            mm = erode_mask(xs6, xs6, rects, cfg["R"])
            mine_only = int(np.sum(mm & ~im))
            impl_only = int(np.sum(im & ~mm))
            agg[key]["mine_only"] += mine_only
            agg[key]["impl_only"] += impl_only
            agg[key]["diff"] += mine_only + impl_only
            agg[key]["total"] += im.size
            rec[key] = dict(mine_only=mine_only, impl_only=impl_only,
                            n_impl=int(im.sum()), n_mine=int(mm.sum()))
        per_seed[seed] = rec
    for key, a in agg.items():
        print(f"{key:>18}: 不一致 {a['diff']:>7} / {a['total']} 点 "
              f"({100.0*a['diff']/a['total']:.3f}%)   "
              f"[私だけ許可 {a['mine_only']}, 実装だけ許可 {a['impl_only']}]")
    out["q1_mask_agreement"] = agg
    out["q1_per_seed"] = per_seed

    # 差の機構を分解する: 実装 ⊇ 壁だけ版 ⊇ 柱つき版 という入れ子なら、
    #   (実装 − 壁だけ版) = 壁の端の丸めの差、(壁だけ版 − 柱つき版) = 柱の効果。
    print("\n  差の機構の分解（20 面の合計。入れ子になっているかも検定する）:")
    d_end, d_post, n_nest_fail, n_tot = 0, 0, 0, 0
    for seed in SEEDS:
        m = generate_maze(seed, mode=MODE)
        im = impl_mask(m)
        full = erode_mask(xs6, xs6, obstacle_rects(m["v_walls"], m["h_walls"], False, True), W_LAT)
        post = erode_mask(xs6, xs6, obstacle_rects(m["v_walls"], m["h_walls"], True, False), W_LAT)
        d_end += int(np.sum(im & ~full))
        d_post += int(np.sum(full & ~post))
        n_nest_fail += int(np.sum(~im & full)) + int(np.sum(~full & post))
        n_tot += im.size
    print(f"    実装 − 壁の帯だけ版（壁の端の丸め）: {d_end:>7} 点 ({100.0*d_end/n_tot:.3f}%)")
    print(f"    壁の帯だけ版 − 柱つき版（柱の効果）: {d_post:>7} 点 ({100.0*d_post/n_tot:.3f}%)")
    print(f"    入れ子の破れ（あってはならない）    : {n_nest_fail:>7} 点")
    out["q1_decomposition"] = dict(wall_end=d_end, posts=d_post,
                                   nesting_violations=n_nest_fail, total=n_tot)

    # ------------------------------------------------------------------
    # 【問②】1/ρ の再現（格子 7.5 mm・AUDIT_011 と同じ）
    # ------------------------------------------------------------------
    h = 0.0075
    n = int(round(SIZE * CS / h)) + 1
    xs = np.arange(n) * h
    print()
    print("=" * 78)
    print(f"【問②】1/ρ = g(start)/(0.18·D0) の再現  格子 {h*1000:.1f} mm × {n}^2")
    print("=" * 78)
    cases = [
        ("点の測地・壁帯＋柱（侵食なし）", dict(kind="erode", posts=True, full=False, R=0.0)),
        ("壁帯＋柱・壁面から 0.0400", dict(kind="erode", posts=True, full=False, R=W_LAT)),
        ("壁帯＋柱・壁面から 0.0455", dict(kind="erode", posts=True, full=False, R=0.0455)),
        ("壁帯＋柱・壁面から 0.0460", dict(kind="erode", posts=True, full=False, R=0.0460)),
        ("壁帯いっぱい柱なし・壁面から 0.0400",
         dict(kind="erode", posts=False, full=True, R=W_LAT)),
        ("壁帯いっぱい柱なし・壁面から 0.0460",
         dict(kind="erode", posts=False, full=True, R=0.0460)),
        ("実装の規則・中心線から 0.0460", dict(kind="rule", clearance=_GEO_CLEARANCE)),
        ("実装の規則・中心線から 0.0520", dict(kind="rule", clearance=0.0520)),
    ]
    results = {label: [] for label, _ in cases}
    for seed in SEEDS:
        m = generate_maze(seed, mode=MODE)
        dmap = shortest_distances(m["v_walls"], m["h_walls"])
        start = tuple(m["start"])
        d0 = int(dmap[start])
        for label, cfg in cases:
            if cfg["kind"] == "erode":
                rects = obstacle_rects(m["v_walls"], m["h_walls"],
                                       with_posts=cfg["posts"], full_length=cfg["full"])
                mask = erode_mask(xs, xs, rects, cfg["R"])
            else:
                mask = impl_rule_mask(xs, xs, m["v_walls"], m["h_walls"], cfg["clearance"])
            r = inv_rho(mask, h, start, d0)
            results[label].append(r)
    out["q2_inv_rho"] = {}
    for label, _ in cases:
        vals = [v for v in results[label] if v is not None]
        n_bad = len(results[label]) - len(vals)
        med = float(np.median(vals)) if vals else float("nan")
        print(f"{label:<34}: 中央値 {med:.4f}  (n={len(vals)}"
              + (f", 到達不能 {n_bad} 面" if n_bad else "") + ")")
        out["q2_inv_rho"][label] = dict(median=med, n=len(vals), unreachable=n_bad,
                                        values=[None if v is None else float(v)
                                                for v in results[label]])

    # ------------------------------------------------------------------
    # 【予測】実装そのものを走らせた 1/ρ（学生B の再測定値の予測。事前に出す）
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("【予測】実装の _compute_geodesic_field() を直接呼んだ 1/ρ（6 mm 格子）")
    print("=" * 78)
    vals = []
    for seed in SEEDS:
        m = generate_maze(seed, mode=MODE)
        shim = SimpleNamespace(maze=m, params=SimpleNamespace(cell_size=CS))
        shim._geo_allowed_mask = lambda s=shim: Maze6Env._geo_allowed_mask(s)
        field = Maze6Env._compute_geodesic_field(shim)
        dmap = shortest_distances(m["v_walls"], m["h_walls"])
        start = tuple(m["start"])
        d0 = int(dmap[start])
        i = int(round((start[0] + 0.5) * CS / h6))
        j = int(round((start[1] + 0.5) * CS / h6))
        vals.append(float(field[i, j]) / (CS * d0))
    med = float(np.median(vals))
    print(f"中央値 {med:.4f}   最小 {min(vals):.4f}  最大 {max(vals):.4f}  (n={len(vals)})")
    out["prediction_impl_inv_rho"] = dict(median=med, values=vals,
                                          note="学習迷路 6x6 loop seed 1-20・始点は区画中心")

    path = f"{REPO_ROOT}/verification/out/r33_config_space.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n書き出し: {path}")


if __name__ == "__main__":
    main()
