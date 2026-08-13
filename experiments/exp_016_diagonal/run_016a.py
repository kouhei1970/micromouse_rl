#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""段階 016-A の本体 — 斜めを含む経路の理想時間と、幾何の干渉検査。

**走行はしない。**カード §2 の 3 つを設計帯 20 面で測る。

1. 斜めを含む状態空間の最小時間（`diagonal_model.DiagonalGridModel`。
   Dijkstra 本体は `competition/route_planner.py` を**無改造**で使う）
2. **速度比 $r$ を振った**短縮率（$r$ は未実測なので 1 つに決め打ちしない）
3. **幾何の干渉検査** — 経路に沿って機体の矩形を掃き、壁・柱の箱との最小距離を出す

**対照は「L0-c+E1T+TR が実際に引く経路」**（裁定 R36-1: 置き換えられる方策が
実際に引く 1 本。`experiments/exp_015_time_optimal_route/route_model.py` を再利用）。

--------------------------------------------------------------------------
干渉検査の作り
--------------------------------------------------------------------------
- **機体**: 長 100.0 × 幅 80.0 mm の矩形（`geometry.py` がモデルから測った値）。
  **中心は回転中心**（車軸中点＝機体原点。`geometry.py` で一致を確認済み）
- **壁・柱の箱**: `competition/baseline_slalom.py` の `_isolated_turn_wall_boxes` と
  **同じ式**（縦壁・横壁は柱のぶん短く、柱は全格子点に立つ）
- **掃引**: 各区間を等間隔に標本化し、さらに**方位が変わる節点では中間の方位も**
  標本化する（その場で回頭する理想化）
- **距離**: 凸多角形どうしの最短距離。重なっていれば負（＝干渉）

使い方:
    .venv/bin/python experiments/exp_016_diagonal/run_016a.py
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal",
          REPO_ROOT / "experiments" / "exp_015_time_optimal_route"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from competition.baseline_slalom_e1_tr import load_time_model  # noqa: E402
from competition.route_planner import StraightGridModel, value_field  # noqa: E402
from mouse.mjcf import POST_SIZE, WALL_THICKNESS  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

from diagonal_model import (DELTA8, DIAGONALS, DiagonalGridModel,  # noqa: E402
                            cell_center_node, descend, node_xy)
from geometry import body_extent_exact, git_rev  # noqa: E402
from route_model import connects_true, load_maze, tr_route  # noqa: E402

SPEED_RATIOS = (1.0, 0.85, 0.7, 0.55)     # カード §2: 1 つに決め打ちしない
SAMPLES_PER_SEG = 12                       # 区間あたりの掃引標本数
SAMPLES_PER_TURN = 6                       # 回頭あたりの方位標本数


# ==========================================================================
# 壁と柱の箱（`_isolated_turn_wall_boxes` と同じ式）
# ==========================================================================
def maze_boxes(v_walls, h_walls, width, height, cell, wall_t, post):
    """(cx, cy, hx, hy) の列を返す（軸並行の矩形）。"""
    half_wt, half_ps = wall_t / 2.0, post / 2.0
    boxes = []
    for i in range(width + 1):
        for j in range(height):
            if int(v_walls[i, j]) == 1:
                boxes.append((i * cell, j * cell + cell / 2.0, half_wt, cell / 2.0 - half_ps))
    for i in range(width):
        for j in range(height + 1):
            if int(h_walls[i, j]) == 1:
                boxes.append((i * cell + cell / 2.0, j * cell, cell / 2.0 - half_ps, half_wt))
    for i in range(width + 1):          # 柱は壁の有無によらず全格子点に立つ
        for j in range(height + 1):
            boxes.append((i * cell, j * cell, half_ps, half_ps))
    return np.array(boxes, dtype=float)


# ==========================================================================
# 凸多角形どうしの最短距離（重なれば負）
# ==========================================================================
def _seg_point_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-18), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


def _polys_overlap(P, Q):
    """分離軸定理（凸多角形）。重なっていれば True。"""
    for poly in (P, Q):
        n = len(poly)
        for k in range(n):
            e = poly[(k + 1) % n] - poly[k]
            ax = np.array([-e[1], e[0]])
            nrm = np.linalg.norm(ax)
            if nrm < 1e-15:
                continue
            ax = ax / nrm
            p0, p1 = P @ ax, Q @ ax
            if p0.max() < p1.min() - 1e-12 or p1.max() < p0.min() - 1e-12:
                return False
    return True


def poly_distance(P, Q):
    """凸多角形 P, Q の最短距離。重なっていれば負（−侵入の目安）。"""
    if _polys_overlap(P, Q):
        # 侵入量そのものより「負である」ことが判定に効くので、
        # 最も浅い分離量の符号を反転して返す（診断用の目安）
        depth = float("inf")
        for poly in (P, Q):
            n = len(poly)
            for k in range(n):
                e = poly[(k + 1) % n] - poly[k]
                ax = np.array([-e[1], e[0]])
                nrm = np.linalg.norm(ax)
                if nrm < 1e-15:
                    continue
                ax = ax / nrm
                p0, p1 = P @ ax, Q @ ax
                depth = min(depth, min(p0.max() - p1.min(), p1.max() - p0.min()))
        return -abs(depth)
    d = float("inf")
    for a, b in ((P, Q), (Q, P)):
        m = len(b)
        for pt in a:
            for k in range(m):
                d = min(d, _seg_point_dist(pt, b[k], b[(k + 1) % m]))
    return d


def body_poly(cx, cy, yaw, length, width):
    """回転中心 (cx,cy)・方位 yaw の機体矩形（4 頂点）。"""
    hl, hw = length / 2.0, width / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        out.append((cx + c * dx - s * dy, cy + s * dx + c * dy))
    return np.array(out, dtype=float)


def box_poly(cx, cy, hx, hy):
    return np.array([(cx - hx, cy - hy), (cx + hx, cy - hy),
                     (cx + hx, cy + hy), (cx - hx, cy + hy)], dtype=float)


def sweep_min_distance(nodes, dirs, cell, boxes, length, width):
    """経路に沿って機体を掃き、壁・柱との最小距離 [m] と最悪の位置を返す。"""
    yaw_of = {d: math.atan2(DELTA8[d][1], DELTA8[d][0]) for d in DELTA8}
    polys = [box_poly(*b) for b in boxes]
    worst, worst_at = float("inf"), None

    def check(cx, cy, yaw, tag):
        nonlocal worst, worst_at
        P = body_poly(cx, cy, yaw, length, width)
        lo, hi = P.min(axis=0) - 0.02, P.max(axis=0) + 0.02
        for b, Q in zip(boxes, polys):
            if b[0] + b[2] < lo[0] or b[0] - b[2] > hi[0]:
                continue
            if b[1] + b[3] < lo[1] or b[1] - b[3] > hi[1]:
                continue
            dd = poly_distance(P, Q)
            if dd < worst:
                worst, worst_at = dd, dict(x=cx, y=cy, yaw=yaw, tag=tag,
                                            box=[float(t) for t in b])

    prev_dir = None
    for k, d in enumerate(dirs):
        p0 = np.array(node_xy(nodes[k], cell))
        p1 = np.array(node_xy(nodes[k + 1], cell))
        yaw = yaw_of[d]
        if prev_dir is not None and prev_dir != d:      # 節点でその場回頭
            y0, y1 = yaw_of[prev_dir], yaw
            dy = math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))
            for t in np.linspace(0.0, 1.0, SAMPLES_PER_TURN):
                check(p0[0], p0[1], y0 + t * dy, f"turn@{k}")
        for t in np.linspace(0.0, 1.0, SAMPLES_PER_SEG):
            q = p0 + t * (p1 - p0)
            check(q[0], q[1], yaw, f"seg{k}:{d}")
        prev_dir = d
    return worst, worst_at


# ==========================================================================
def analyse(path, a, b, cell, wall_t, post, length, width, ratios=SPEED_RATIOS):
    v, h, start, goals = load_maze(path)
    gset = [tuple(g) for g in goals]
    conn = connects_true(v, h)
    N = 16

    ctrl = tr_route(v, h, start, set(gset), StraightGridModel(a, b))
    t_ctrl = a * ctrl["moves"] + b * ctrl["turns"]

    boxes = maze_boxes(v, h, N, N, cell, wall_t, post)
    per_r, geom = {}, None
    for r in ratios:
        model = DiagonalGridModel(a, b, r=r)
        field = value_field(gset, N, N, conn, model)
        s_node = cell_center_node(tuple(start))
        t_dia = field.states.get((s_node, "N"))
        p = descend(field, model, s_node, "N", N, N, conn)
        cells = {(n[0] // 2, n[1] // 2) for n in p["nodes"]}
        per_r[f"{r:g}"] = dict(
            time_s=t_dia, gain=(t_ctrl - t_dia) / t_ctrl if t_ctrl else 0.0,
            n_diag=p["n_diag"], n_half=p["n_half"], n_cells=len(cells),
            n_turn45=sum(1 for x in p["turns_deg"] if x == 45),
            n_turn90plus=sum(1 for x in p["turns_deg"] if x >= 90))
        if r == 1.0:
            d_min, at = sweep_min_distance(p["nodes"], p["dirs"], cell, boxes, length, width)
            geom = dict(min_clearance_m=d_min, worst=at, n_nodes=len(p["nodes"]))
    return dict(maze=Path(path).stem, t_ctrl_s=t_ctrl,
                ctrl_moves=ctrl["moves"], ctrl_turns=ctrl["turns"],
                ctrl_cells=len({tuple(c) for c in ctrl["path"]}),
                per_r=per_r, geometry=geom)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016a" / "results.json"))
    args = ap.parse_args()

    a, b = load_time_model()
    prm = RobotParams()
    body = body_extent_exact()
    L, W = body["length_m"], body["width_m"]
    print(f"時間モデル（実測の回帰）: a={a:.4f} s/区画, b={b:.4f} s/折れ")
    print(f"機体（モデルから厳密に）: 長 {L*1000:.2f} × 幅 {W*1000:.2f} mm")
    print(f"迷路: {args.maze_dir}\n")

    rows = []
    files = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    hdr = "".join(f"{'r=' + f'{r:g}':>10}" for r in SPEED_RATIOS)
    print(f"{'面':<12}{'対照 [s]':>10}{hdr}{'斜め歩':>7}{'区画':>6}{'最小余裕':>10}")
    for f in files:
        r0 = analyse(str(f), a, b, prm.cell_size, WALL_THICKNESS, POST_SIZE, L, W)
        rows.append(r0)
        gains = "".join(f"{r0['per_r'][f'{r:g}']['gain']*100:9.1f}%" for r in SPEED_RATIOS)
        g1 = r0["per_r"]["1"]
        print(f"{r0['maze']:<12}{r0['t_ctrl_s']:>10.3f}{gains}{g1['n_diag']:>7}"
              f"{g1['n_cells']:>6}{r0['geometry']['min_clearance_m']*1000:>9.2f}mm")

    print("\n【まとめ】")
    for r in SPEED_RATIOS:
        g = np.array([x["per_r"][f"{r:g}"]["gain"] for x in rows])
        print(f"  r={r:<5g} 短縮率 中央値 {np.median(g)*100:6.2f}%"
              f"（四分位 {np.percentile(g,25)*100:.2f}〜{np.percentile(g,75)*100:.2f}"
              f"、最小 {g.min()*100:.2f} / 最大 {g.max()*100:.2f}）")
    clr = np.array([x["geometry"]["min_clearance_m"] for x in rows])
    n_bad = int((clr < 0).sum())
    print(f"\n  **干渉（最小余裕が負）: {n_bad} / {len(rows)} 面**")
    print(f"  最小余裕の分布: 中央値 {np.median(clr)*1000:.2f} mm"
          f"／最小 {clr.min()*1000:.2f} mm／最大 {clr.max()*1000:.2f} mm")
    n_diag_faces = sum(1 for x in rows if x["per_r"]["1"]["n_diag"] > 0)
    n_more_cells = sum(1 for x in rows if x["per_r"]["1"]["n_cells"] > x["ctrl_cells"])
    print(f"  斜めを含む経路が引かれた面: **{n_diag_faces} / {len(rows)}**")
    print(f"  経路が通る区画数が対照より増えた面: **{n_more_cells} / {len(rows)}**")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(git_rev=git_rev(), maze_dir=args.maze_dir, a=a, b=b,
                   body_length_m=L, body_width_m=W, speed_ratios=list(SPEED_RATIOS),
                   turn_45_assumption="b/2", rows=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
