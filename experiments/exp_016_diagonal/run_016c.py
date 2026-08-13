#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-C 本体 — **斜め速度の探索**（設計帯 20 面 × 速度の梯子）。

カード `card_016c.md` の手順を**そのまま**実行する:

- 速度の梯子 §1-1 は**凍結**（これ以外の速度で主判定しない）
- 各段の合否 §1-2 は「**20 面すべて A-成立かつ衝突 0**」
- 主判定 §1-3 は「**合格した最大の速度**」
- **梯子の途中で不合格が出ても上の段は全部試す**（非単調性を見逃さない）

--------------------------------------------------------------------------
016-B からの実装の差分（カード §3）
--------------------------------------------------------------------------
**速度上限を区間ごとに分ける**: 斜め（`diagonal`）と遷移（`arc`）は $v_\\text{斜め}$、
**直進（`straight`）は従来どおり $v_\\text{cap}$**。016-B は経路全体に 1 つの上限を
掛けていたため、直進まで遅くなり**斜めを遅くする代償を分離して測れなかった**。
**制御そのものは 1 行も変えない**（`build_speed_profile` に配列で渡すだけ）。

--------------------------------------------------------------------------
半幅の上界（カード §4-2）
--------------------------------------------------------------------------
$$w_\\text{使用}(\\psi) = \\min\\{\\,w_\\text{矩形}(\\psi),\\ h(\\lceil \\psi \\rceil_{0.25°})\\,\\}$$

**どちらも真の半幅の上界なので、小さい方も上界であり続ける**（丸めの損が構成上 0）。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/run_016c.py
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

from competition.baseline_slalom import build_speed_profile  # noqa: E402
from competition.baseline_slalom_e1_tr import load_time_model  # noqa: E402
from competition.route_planner import value_field  # noqa: E402
from mouse.mjcf import build_maze_robot_xml  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

from diag_path import (build_diagonal_path, heading_deviation,  # noqa: E402
                       lateral_deviation)
from diagonal_model import (DELTA8, DiagonalGridModel, cell_center_node,  # noqa: E402
                            descend)
from geometry import body_extent_exact, diagonal_clearance, git_rev  # noqa: E402
from route_model import connects_true, load_maze  # noqa: E402
from run_016b import OneSegmentPolicy, cut_segment, longest_diagonal_run  # noqa: E402

LADDER = (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)   # カード §1-1（凍結）
R_ARC_M = 0.060
H_FINE = REPO_ROOT / "verification" / "out" / "support_function_fine.json"


def load_h_table(path=H_FINE):
    """准教授の 0.25° 刻みの表（計算に使うのはこちら。5° の表は読み物）。"""
    d = json.load(open(path, encoding="utf-8"))
    # `main_column` は説明文（例 "lateral_half_width_m = h(theta+90deg)"）なので、
    # 列名そのものを取り出す。行に無ければ既定の列名へ落とす。
    col = str(d.get("main_column", "")).split("=")[0].strip()
    if col not in d["rows"][0]:
        col = "lateral_half_width_m"
    th = np.array([float(r["theta_deg"]) for r in d["rows"]])
    hw = np.array([float(r[col]) for r in d["rows"]])
    order = np.argsort(th)
    return th[order], hw[order], float(d["resolution_deg"])


class SegSpeedPolicy(OneSegmentPolicy):
    """区間ごとに速度上限を分ける（カード §3）。**制御は親のまま。**"""

    def __init__(self, ref_path, v_ceil_arr, **kw):
        super().__init__(ref_path, float(np.max(v_ceil_arr)), **kw)
        self._v_ceil_arr = np.asarray(v_ceil_arr, dtype=float)

    def _replan(self, x, y, yaw):
        path = self._ref
        arr = np.minimum(self._v_ceil_arr, self.v_cap)
        path.speed = build_speed_profile(path.s, path.curvature, arr,
                                          self.a_lat, self.a_max, self.v_creep, True)
        self._path = path
        self._cursor = 0
        self._path_ticks = 0
        self._path_end_reason = "goal"
        self._state = "DRIVE"


def run_one(xml_path, params, nodes, dirs, v_diag, v_walls, h_walls,
            th, hw, free, L, W, max_s=40.0):
    """1 面 1 速度を走らせ、判定に要る量を返す。"""
    path, kinds, idxs = build_diagonal_path(nodes, dirs, params.cell_size, R_ARC_M)
    sim = MouseSim(str(xml_path), params=params)
    start_cell = (nodes[0][0] // 2, nodes[0][1] // 2)
    heading = math.degrees(math.atan2(DELTA8[dirs[0]][1], DELTA8[dirs[0]][0]))
    sim.full_reset(cell=start_cell, heading_deg=heading)

    v_arr = np.where(kinds == "straight", 1e9, v_diag)     # 直進は v_cap に委ねる
    pol = SegSpeedPolicy(path, v_arr)
    pol.bind_sim(sim)
    pol.bind_maze(v_walls, h_walls)
    pol.on_maze_start(dict(width=16, height=16))

    rec, collided = [], False
    for _ in range(int(max_s / params.control_dt)):
        vl, vr = pol.act(sim.observation())
        out = sim.step_control(vl, vr)
        x, y, yaw = sim.privileged_pose()
        v, _w = sim.privileged_velocity()
        rec.append((sim.sim_time, x, y, yaw, v))
        if out.get("collision"):
            collided = True
            break
        if pol.finished:
            break
    rec = np.array(rec, dtype=float)

    px, py = path.x, path.y
    m_min, e_max, psi_max, n_diag = float("inf"), 0.0, 0.0, 0
    v_straight, arc_t, arc_len = [], {}, {}
    for (t, x, y, yaw, v) in rec:
        k = int(np.argmin((px - x) ** 2 + (py - y) ** 2))
        kind, si = str(kinds[k]), min(int(idxs[k]), len(dirs) - 1)
        if kind == "straight":
            if v > 0.05:
                v_straight.append(v)
            continue
        if kind == "arc":
            arc_t.setdefault(si, []).append((t, x, y))
            continue
        n_diag += 1
        e_y = lateral_deviation(x, y, nodes[si], nodes[si + 1], params.cell_size)
        psi = heading_deviation(yaw, dirs[si])
        w_rect = (L / 2) * abs(math.sin(math.radians(psi))) + \
                 (W / 2) * abs(math.cos(math.radians(psi)))
        w_h = float(hw[min(int(np.searchsorted(th, psi)), len(hw) - 1)])
        m = free - min(w_rect, w_h) - e_y            # カード §4-2
        m_min = min(m_min, m)
        e_max = max(e_max, e_y)
        psi_max = max(psi_max, psi)

    if n_diag == 0:
        return dict(verdict="判定不能", collided=collided, n_diag_ticks=0)
    verdict = "A-不成立" if (collided or m_min <= 0) else "A-成立"
    v_ref = float(np.median(v_straight)) if v_straight else float("nan")
    b45 = []
    for si, pts in arc_t.items():
        ts = [q[0] for q in pts]
        xs = np.array([q[1] for q in pts])
        ys = np.array([q[2] for q in pts])
        alen = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
        if v_ref > 0:
            b45.append((max(ts) - min(ts)) - alen / v_ref)
    return dict(verdict=verdict, collided=collided, margin_min_m=m_min,
                e_y_max_m=e_max, psi_max_deg=psi_max, n_diag_ticks=n_diag,
                v_straight_med_mps=v_ref, n_arcs=len(b45),
                b45_med_s=float(np.median(b45)) if b45 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016c" / "run.json"))
    args = ap.parse_args()

    a, b = load_time_model()
    params = RobotParams()
    body = body_extent_exact()
    L, W = body["length_m"], body["width_m"]
    free, _ = diagonal_clearance(params.cell_size, 0.012)
    th, hw, res = load_h_table()
    print(f"h(θ) 表: {len(th)} 点・刻み {res}°（計算に使うのはこちら）")
    print(f"機体 {L*1000:.1f}×{W*1000:.1f} mm／片側自由幅 {free*1000:.3f} mm")
    print(f"速度の梯子（凍結）: {list(LADDER)}\n")

    faces = []
    for f in sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                    key=lambda p: int(p.stem.split("_")[1])):
        v, h, start, goals = load_maze(str(f))
        conn = connects_true(v, h)
        model = DiagonalGridModel(a, b, r=1.0)
        field = value_field([tuple(g) for g in goals], 16, 16, conn, model)
        p = descend(field, model, cell_center_node(tuple(start)), "N", 16, 16, conn)
        s0, e0 = longest_diagonal_run(p["dirs"])
        i, j = cut_segment(p["nodes"], p["dirs"], s0, e0)
        xml = f.with_suffix(".xml")
        if not xml.exists():
            build_maze_robot_xml(v, h, str(xml), model_name=f"m_{f.stem}", params=params)
        faces.append(dict(maze=f.stem, xml=xml, nodes=p["nodes"][i:j + 1],
                          dirs=p["dirs"][i:j], run_len=e0 - s0, v=v, h=h))
    print(f"面: {len(faces)}／斜めの最長連続 中央値 {np.median([q['run_len'] for q in faces]):.0f} 歩\n")

    hdr = "".join(f"{v:>7.2f}" for v in LADDER)
    print(f"{'面':<12}{'斜め歩':>6}{hdr}")
    rows = []
    for q in faces:
        line, per = f"{q['maze']:<12}{q['run_len']:>6}", {}
        for v_d in LADDER:
            r = run_one(q["xml"], params, q["nodes"], q["dirs"], v_d, q["v"], q["h"],
                        th, hw, free, L, W)
            per[f"{v_d:g}"] = r
            line += ("     ○" if r["verdict"] == "A-成立" else
                     ("     ×" if r["verdict"] == "A-不成立" else "     ?"))
        rows.append(dict(maze=q["maze"], run_len=q["run_len"], per_speed=per))
        print(line, flush=True)

    print("\n【段ごとの合否】（○ = 20 面すべて A-成立かつ衝突 0）")
    passed = []
    for v_d in LADDER:
        k = f"{v_d:g}"
        ok = all(r["per_speed"][k]["verdict"] == "A-成立" for r in rows)
        nu = sum(1 for r in rows if r["per_speed"][k]["verdict"] == "判定不能")
        nf = sum(1 for r in rows if r["per_speed"][k]["verdict"] == "A-不成立")
        nc = sum(1 for r in rows if r["per_speed"][k].get("collided"))
        print(f"  {v_d:.2f} m/s: {'**合格**' if ok else '不合格'}"
              f"（A-不成立 {nf} 面・判定不能 {nu} 面・衝突 {nc} 面）")
        if ok:
            passed.append(v_d)
    v_max = max(passed) if passed else None
    print(f"\n**主判定: v_斜め^max = "
          f"{f'{v_max:.2f} m/s' if v_max else '合格なし → 斜め走行は成立しない'}**")

    if v_max:
        k = f"{v_max:g}"
        vs = [r["per_speed"][k]["v_straight_med_mps"] for r in rows]
        b45 = [r["per_speed"][k]["b45_med_s"] for r in rows
               if r["per_speed"][k].get("b45_med_s") is not None]
        print(f"  直進区間の実測速度 中央値 {np.median(vs):.3f} m/s"
              f" → **r = {v_max/np.median(vs):.3f}**")
        print(f"  **b45 中央値 {np.median(b45)*1000:+.1f} ms**"
              f"（範囲 {min(b45)*1000:+.1f}〜{max(b45)*1000:+.1f}）")
    # 面ごとの合格速度
    pf = []
    for r in rows:
        ok = [v for v in LADDER if r["per_speed"][f"{v:g}"]["verdict"] == "A-成立"]
        pf.append(max(ok) if ok else 0.0)
    print(f"  面ごとの合格速度: 最小 {min(pf):.2f}／中央値 {np.median(pf):.2f}"
          f"／最大 {max(pf):.2f} m/s（**差 {max(pf)-min(pf):.2f}**）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(git_rev=git_rev(), ladder=list(LADDER), R_arc_m=R_ARC_M,
                   body_length_m=L, body_width_m=W, free_half_width_m=free,
                   h_table=str(H_FINE.relative_to(REPO_ROOT)), h_resolution_deg=res,
                   v_diag_max=v_max, per_face_pass=pf, rows=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
