#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直進 ↔ 斜めの**接続**を幾何で詰める（016-B の実装前の設計判断）。

**走行はしない。**016-A は方位の変化を「その場回頭」として掃引したが、
実際の走行では**円弧で接続する**（`baseline_slalom.py` の 90° コーナーと同じ考え方）。
**接続の曲率をどう扱うかは古典トラック最重量の実装の分岐点**なので、
**制御を書く前に幾何だけで決める**（教授指示 2026-08-14）。

--------------------------------------------------------------------------
何を測るか
--------------------------------------------------------------------------
016-A が引いた経路の**方位が変わる節点**を、半径 $R$ の円弧で接続し直し、
**機体の矩形を掃いて壁・柱との最小距離**を出す。$R$ を振って比べる。

- 円弧は 2 本の直線に**接する**（既存の `corner_arc_params` と同じ）。
  旋回角 $\\theta$ の接線長は $R\\tan(\\theta/2)$ —
  **45° 接続では $0.414R$、90° では $1.000R$** なので、**45° の方が短い距離で曲がれる**
- 接線長が隣の区間の長さを超える場合は**その場回頭に落とす**（安全弁）。
  **落ちた回数を数えて報告する**（常時発動なら円弧接続は成立していない）

**016-A の掃引（その場回頭・20 面すべてで 15.154 mm）との差が、
接続の設計が持ち込むリスクである。**

⚠️ **距離の打ち切り**: 掃引では機体の外接箱から 20 mm 以上離れた箱を評価しない
（計算量のため）。**したがって「最小余裕」が 20 mm を超える場合は `inf` になる**。
判定に使うのは 15 mm 前後の領域なので実害は無いが、**値が大きい側は信用しないこと**。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/transition_geometry.py
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

from competition.baseline_slalom_e1_tr import load_time_model  # noqa: E402
from competition.route_planner import value_field  # noqa: E402
from mouse.mjcf import POST_SIZE, WALL_THICKNESS  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

from diagonal_model import (DELTA8, DiagonalGridModel, cell_center_node,  # noqa: E402
                            descend, node_xy, turn_deg)
from geometry import body_extent_exact, git_rev  # noqa: E402
from route_model import connects_true, load_maze  # noqa: E402
from run_016a import body_poly, box_poly, maze_boxes, poly_distance  # noqa: E402

# `baseline_slalom.py` の arc_radius_candidates_mm と同じ候補
RADII_MM = (45.0, 60.0, 75.0, 90.0)
ARC_SAMPLES = 10
SEG_SAMPLES = 10


def _yaw(d):
    return math.atan2(DELTA8[d][1], DELTA8[d][0])


def build_swept_poses(nodes, dirs, cell, R):
    """節点列を円弧で接続した姿勢列 [(x, y, yaw, tag), ...] と、その場回頭の回数を返す。

    各区間は「直線（接線長を差し引いた残り）→ 次の節点の手前で円弧」の順に並べる。
    接線長が区間長を超えたらその場回頭へ落とす（安全弁）。
    """
    pts = [np.array(node_xy(n, cell)) for n in nodes]
    seg_len = [float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(len(dirs))]
    tan_len = [0.0] * len(nodes)          # 節点 k で消費する接線長
    theta = [0.0] * len(nodes)
    for k in range(1, len(dirs)):
        th = math.radians(turn_deg(dirs[k - 1], dirs[k]))
        theta[k] = th
        tan_len[k] = R * math.tan(th / 2.0) if th > 0 else 0.0

    pivots = 0
    for k in range(1, len(dirs)):
        if tan_len[k] <= 0:
            continue
        room = min(seg_len[k - 1], seg_len[k]) / 2.0
        if tan_len[k] > room or theta[k] >= math.pi - 1e-9:
            tan_len[k] = 0.0              # 円弧を張れない → その場回頭
            pivots += 1

    poses = []
    for k, d in enumerate(dirs):
        p0, p1 = pts[k], pts[k + 1]
        u = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-12)
        a0 = p0 + u * tan_len[k]          # 直線の始点（前の円弧の出口）
        a1 = p1 - u * tan_len[k + 1] if k + 1 < len(nodes) else p1
        y = _yaw(d)
        n_s = max(int(np.linalg.norm(a1 - a0) / 0.005), SEG_SAMPLES)
        for t in np.linspace(0.0, 1.0, n_s):
            q = a0 + t * (a1 - a0)
            poses.append((q[0], q[1], y, f"seg{k}:{d}"))
        if k + 1 < len(dirs):
            d2 = dirs[k + 1]
            if tan_len[k + 1] > 0:        # 円弧で接続
                u2 = (pts[k + 2] - pts[k + 1])
                u2 = u2 / max(np.linalg.norm(u2), 1e-12)
                s0 = pts[k + 1] - u * tan_len[k + 1]
                y0, y1 = _yaw(d), _yaw(d2)
                dy = math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))
                # 円弧の中心 = 接点から法線方向へ R
                nrm = np.array([-u[1], u[0]]) * (1.0 if dy > 0 else -1.0)
                ctr = s0 + nrm * R
                start_ang = math.atan2(s0[1] - ctr[1], s0[0] - ctr[0])
                for t in np.linspace(0.0, 1.0, ARC_SAMPLES):
                    ang = start_ang + dy * t
                    q = ctr + R * np.array([math.cos(ang), math.sin(ang)])
                    poses.append((q[0], q[1], y0 + dy * t, f"arc@{k+1}"))
            else:                          # その場回頭
                y0, y1 = _yaw(d), _yaw(d2)
                dy = math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))
                for t in np.linspace(0.0, 1.0, ARC_SAMPLES):
                    poses.append((pts[k + 1][0], pts[k + 1][1], y0 + dy * t,
                                  f"pivot@{k+1}"))
    return poses, pivots


def min_clearance(poses, boxes, length, width):
    polys = [box_poly(*b) for b in boxes]
    worst, at = float("inf"), None
    for (x, y, yaw, tag) in poses:
        P = body_poly(x, y, yaw, length, width)
        lo, hi = P.min(axis=0) - 0.02, P.max(axis=0) + 0.02
        for b, Q in zip(boxes, polys):
            if b[0] + b[2] < lo[0] or b[0] - b[2] > hi[0]:
                continue
            if b[1] + b[3] < lo[1] or b[1] - b[3] > hi[1]:
                continue
            d = poly_distance(P, Q)
            if d < worst:
                worst, at = d, dict(x=x, y=y, yaw_deg=math.degrees(yaw), tag=tag)
    return worst, at


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--n-faces", type=int, default=6, help="調べる面数（既定 6）")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016b" / "transition_geometry.json"))
    args = ap.parse_args()

    a, b = load_time_model()
    prm = RobotParams()
    body = body_extent_exact()
    L, W = body["length_m"], body["width_m"]
    N = 16
    print(f"機体 長 {L*1000:.1f} × 幅 {W*1000:.1f} mm／区画 {prm.cell_size*1000:.0f} mm")
    print("接線長 = R·tan(θ/2): **45° 接続 = 0.414R・90° 接続 = 1.000R**\n")

    files = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))[:args.n_faces]
    hdr = "".join(f"{'R=' + f'{r:g}':>12}" for r in RADII_MM)
    print(f"{'面':<12}{hdr}")
    rows = []
    for f in files:
        v, h, start, goals = load_maze(str(f))
        gset = [tuple(g) for g in goals]
        conn = connects_true(v, h)
        model = DiagonalGridModel(a, b, r=1.0)
        field = value_field(gset, N, N, conn, model)
        p = descend(field, model, cell_center_node(tuple(start)), "N", N, N, conn)
        boxes = maze_boxes(v, h, N, N, prm.cell_size, WALL_THICKNESS, POST_SIZE)

        # 比較の基準（その場回頭）は 016-A の run_016a.py が既に出している
        # （20 面すべてで 15.154 mm）。ここでは円弧接続だけを見る。
        cells = {}
        line = f"{f.stem:<12}"
        for R in RADII_MM:
            poses, piv = build_swept_poses(p["nodes"], p["dirs"], prm.cell_size, R / 1000.0)
            d, at = min_clearance(poses, boxes, L, W)
            cells[f"{R:g}"] = dict(min_clearance_m=d, n_pivot_fallback=piv, worst=at)
            line += f"{d*1000:>9.2f}mm" + ("!" if d < 0 else " ") + f"{piv:>1d}"
        rows.append(dict(maze=f.stem, by_radius=cells, n_dirs=len(p["dirs"])))
        print(line)

    print("\n（各欄: 最小余裕 [mm] / 末尾の数字 = 円弧を張れずその場回頭に落ちた節点数。"
          "`!` は干渉）")
    print("\n【まとめ】")
    for R in RADII_MM:
        d = np.array([r["by_radius"][f"{R:g}"]["min_clearance_m"] for r in rows])
        piv = np.array([r["by_radius"][f"{R:g}"]["n_pivot_fallback"] for r in rows])
        nd = np.array([r["n_dirs"] for r in rows])
        print(f"  R={R:>5.0f} mm  最小余裕 中央値 {np.median(d)*1000:6.2f} mm"
              f"／最小 {d.min()*1000:6.2f} mm／**干渉 {(d<0).sum()}/{len(d)} 面**"
              f"／その場回頭 {piv.sum()}（全方位変化 {nd.sum()} 中）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(git_rev=git_rev(), maze_dir=args.maze_dir, radii_mm=list(RADII_MM),
                   body_length_m=L, body_width_m=W, rows=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
