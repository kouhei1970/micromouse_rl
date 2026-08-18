#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""斜め直線の速度上限を解放したら走行時間がどれだけ縮むかの**探索的測定**。

⚠️ **これは判定を出す実験ではない。事前登録も判定条文も無い規模の見積りである。**
   結論を主張するには、別途実験カードと事前登録（① 判定条文 ② 錨の独立再計算
   ③ 否定対照）が要る（`docs/RESEARCH_PLAN.md` §12-9(c)）。

--------------------------------------------------------------------------
背景（確定した事実）
--------------------------------------------------------------------------
古典方策の速度計画で、経路の区分 `kinds` は "straight"（軸平行の直進）／
"diagonal"（45° の斜め直線）／"arc"（円弧）の 3 種類あるが、速度上限の割り当てが

    v_arr = np.where(kinds == "straight", 1e9, v_diag)

の 1 行になっており、**"diagonal"（斜めの直線）が "arc"（円弧）と同じ上限
v_diag に縛られている**（`run_016h1_diag.py:84`。本番は
`competition/baseline_slalom_diag.py:170` も同型）。実機のマイクロマウスでは
「カーブはゆっくり入って速く出る、斜めの直線はできる限り加速する」が定石であり、
現状の実装はその斜め直線側の加速をしていない。

速度計画そのもの（`competition/baseline_slalom.py` の `build_speed_profile`、
494-568 行）は曲率依存の上限包絡に対して後方パス→前方パスの 2 パスをかける
標準的な時間最適の作りになっており、上限さえ正しく与えれば減速・加速は
自動で入る。つまり直す場所は「上限の割り当て 1 行」である可能性が高い、
という予測をここで探索的に測る。

--------------------------------------------------------------------------
2 群
--------------------------------------------------------------------------
    capped   : 現状どおり — v_arr = np.where(kinds == "straight", 1e9, v_diag)
               （直進だけ解放。斜め直線と円弧はどちらも v_diag に縛られる）
    released : v_arr = np.where(kinds == "arc", v_diag, 1e9)
               （円弧だけ v_diag に縛る。直進と斜め直線は解放し、
                 build_speed_profile の曲率依存の上限包絡と前後 2 パスの
                 加減速制約に絞りを任せる）

--------------------------------------------------------------------------
ハーネスの由来（裁定 R23。定義を写さず import する）
--------------------------------------------------------------------------
本スクリプトは `run_016h1_diag.py` を import して `KIND`・`make_probed`・
`entry_exit` をそのまま再利用する。**`run_016h1_diag.py` は変更しない。**

走行ループ本体（`drive_and_record_grouped`）だけは、`run_016h1_diag.drive_and_record`
（76-124 行）の**写し**である。差分は v_arr の組み立てを群ごとに切り替える
1 箇所だけで、それ以外（記録のタイミング・列・停止条件）は写し元と同一。
記録は必ず `sim.step_control()` の**前**で読む（写し元 102-104 行の教訓を継承）。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/probe_diag_cap.py \
        --group capped --v-diag 0.45
    .venv/bin/python experiments/exp_016_diagonal/probe_diag_cap.py \
        --group released --v-diag 0.45
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for p in (REPO_ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.seed_bands import assert_seeds_allowed, describe_seeds        # noqa: E402
from competition.baseline_slalom_diag_cal import L_C_CLOTHOID_M           # noqa: E402
from competition.baseline_slalom_e1_tr import load_time_model             # noqa: E402
from competition.route_planner import value_field                        # noqa: E402
from mouse.mjcf import build_maze_robot_xml                               # noqa: E402
from mouse.params import RobotParams                                      # noqa: E402
from mouse.sim import MouseSim                                            # noqa: E402

import run_016f0_ladder                                                   # noqa: E402
import run_016g_ladder                                                    # noqa: E402
import run_016h1_diag                                                     # noqa: E402  ← ハーネス本体はここから借りる
from diagonal_model import (DELTA8, DiagonalGridModel, cell_center_node,  # noqa: E402
                            descend)
from geometry import git_rev                                              # noqa: E402
from route_model import connects_true, load_maze                          # noqa: E402
from run_016b import cut_segment, longest_diagonal_run                    # noqa: E402
from run_016c import R_ARC_M                                              # noqa: E402

KIND = run_016h1_diag.KIND
GROUPS = ("capped", "released")


def make_v_arr(kinds, v_diag, group):
    """群ごとの速度上限配列（本実験の唯一の変更点）。"""
    if group == "capped":
        # 現状どおり: run_016h1_diag.py:84 と同じ式
        # （本番は competition/baseline_slalom_diag.py:170 も同型）
        return np.where(kinds == "straight", 1e9, v_diag)
    if group == "released":
        # 円弧だけ v_diag に縛る。直進と斜め直線は解放し、build_speed_profile の
        # 曲率依存の上限包絡と前後 2 パスの加減速制約に絞りを任せる。
        return np.where(kinds == "arc", v_diag, 1e9)
    raise ValueError(f"未知の group: {group!r}（{GROUPS} のいずれか）")


def drive_and_record_grouped(xml_path, params, nodes, dirs, v_diag, group, v_walls, h_walls,
                             policy_cls, builder, max_s=40.0):
    """1 迷路を走らせ、制御周期ごとの記録を返す。**制御には触らない。**

    `run_016h1_diag.drive_and_record`（76-124 行）の写し。差分は v_arr の
    組み立て 1 箇所だけ（`make_v_arr` へ委譲）。それ以外は写し元と同一。
    親が変わったら（写し元 76-124 行）、この写しも追随が要る。
    """
    path, kinds, _idx = builder(nodes, dirs, params.cell_size, R_ARC_M)
    sim = MouseSim(str(xml_path), params=params)
    sim.full_reset(cell=(nodes[0][0] // 2, nodes[0][1] // 2),
                   heading_deg=math.degrees(math.atan2(DELTA8[dirs[0]][1],
                                                       DELTA8[dirs[0]][0])))
    v_arr = make_v_arr(kinds, v_diag, group)
    pol = run_016h1_diag.make_probed(policy_cls)(path, v_arr)
    pol.bind_sim(sim)
    pol.bind_maze(v_walls, h_walls)
    pol.on_maze_start(dict(width=16, height=16))

    px, py = np.asarray(path.x), np.asarray(path.y)
    rec, collided = [], False
    for _ in range(int(max_s / params.control_dt)):
        pol._omega_cmd = float("nan")
        vl, vr = pol.act(sim.observation())
        if not np.isfinite(pol._omega_cmd):        # 電圧生成を通らなかったティック
            if pol.finished:
                break
            out = sim.step_control(vl, vr)
            if out.get("collision"):
                collided = True
                break
            continue
        # 🔴 **指令と同時刻の状態を取る**（写し元 102-104 行の教訓。
        # step_control の**後**で読むと記録が 1 ティック先の値になる）。
        x, y, yaw = sim.privileged_pose()
        v_act, w_act = sim.privileged_velocity()
        cur = int(getattr(pol, "_cursor", 0))
        k = int(np.argmin((px - x) ** 2 + (py - y) ** 2))
        head = float(path.heading[cur])
        e_y = ((x - float(path.x[cur])) * (-math.sin(head))
               + (y - float(path.y[cur])) * math.cos(head))
        rec.append((sim.sim_time, pol._omega_cmd, w_act, e_y,
                    KIND[str(kinds[k])], float(path.s[cur]),
                    float(path.curvature[cur]), v_act, x, y, yaw,
                    pol._volt[0], pol._volt[1]))
        out = sim.step_control(vl, vr)
        if out.get("collision"):
            collided = True
            break
        if pol.finished:
            break
    cols = ("t", "omega_des", "omega_act", "e_y", "kind", "s", "kappa", "v_act",
            "x", "y", "yaw", "volt_l", "volt_r")
    rdict = dict(zip(cols, np.asarray(rec, dtype=float).T))
    # 走行ティック数（= 所要時間 / control_dt）は sim.sim_time から直接測る。
    # len(rec) は「電圧生成を通ったティック」だけを数えるので、INIT/IDLE の
    # 端数ティックが混ざる場合に所要時間そのものとずれうる（`sim.sim_time` は
    # step_control を呼んだ回数 × control_dt を正確に積算している）。
    n_ticks = int(round(sim.sim_time / params.control_dt))
    return rdict, collided, pol, n_ticks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=GROUPS, required=True)
    ap.add_argument("--safety", type=float, default=0.75)
    ap.add_argument("--v-diag", type=float, required=True)
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-mazes", type=int, default=None,
                    help="煙試験専用。全面の本走行では指定しないこと")
    args = ap.parse_args()

    maze_files = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if args.max_mazes is not None:
        maze_files = maze_files[:args.max_mazes]
    seeds = [int(q.stem.split("_")[1]) for q in maze_files]
    print(describe_seeds(seeds, "competition"))
    assert_seeds_allowed(seeds, namespace="competition", purpose="validate")

    params = RobotParams()
    dt = params.control_dt
    a, b = load_time_model()
    policy_cls = run_016f0_ladder.make_policy_class(k_acc_ff=1.0, ref_interp=True,
                                                    safety=args.safety)
    builder = run_016g_ladder.make_builder(L_C_CLOTHOID_M)
    print(f"群={args.group}／採用構成: F0 + F0-b + 安全率 {args.safety:g} + 45° クロソイド "
          f"{L_C_CLOTHOID_M*1000:.4f} mm／速度水準 v_diag={args.v_diag:g} m/s／"
          f"制御周期 {dt*1000:.1f} ms\n")

    rows = []
    for f in maze_files:
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
        r, collided, pol_probe, n_ticks = drive_and_record_grouped(
            xml, params, p["nodes"][i:j + 1], p["dirs"][i:j], args.v_diag, args.group,
            v, h, policy_cls, builder)
        if not r or len(r["t"]) == 0:
            print(f"{f.stem}: 記録なし")
            rows.append(dict(maze=f.stem, collided=bool(collided), n_ticks=int(n_ticks),
                             no_record=True))
            continue

        ee = run_016h1_diag.entry_exit(r, float(getattr(pol_probe, "voltage_limit", 3.0)))
        m_diag = r["kind"] == KIND["diagonal"]
        diag_v_med = float(np.median(r["v_act"][m_diag])) if m_diag.any() else float("nan")
        diag_v_max = float(np.max(r["v_act"][m_diag])) if m_diag.any() else float("nan")
        m_arc = r["kind"] == KIND["arc"]
        arc_v_med = float(np.median(r["v_act"][m_arc])) if m_arc.any() else float("nan")
        ey_entry = [abs(float(x0)) for x0, x1, x2 in ee]
        ey_entry_med = float(np.median(ey_entry)) if ey_entry else float("nan")

        rows.append(dict(
            maze=f.stem, collided=bool(collided), n_ticks=int(n_ticks),
            t_s=float(n_ticks * dt), n_arcs=len(ee), n_diag_ticks=int(m_diag.sum()),
            diag_v_med=diag_v_med, diag_v_max=diag_v_max,
            arc_v_med=arc_v_med, ey_entry_med=ey_entry_med,
            entry_exit=[[float(x0), float(x1), float(x2)] for x0, x1, x2 in ee],
        ))
        print(f"{f.stem}: {'衝突 ' if collided else ''}ticks={n_ticks}"
              f"（{n_ticks*dt:.3f}s）／円弧 {len(ee)} 本／"
              f"斜め直線 v中央値 {diag_v_med:.3f}(最大{diag_v_max:.3f}) m/s／"
              f"円弧 v中央値 {arc_v_med:.3f} m/s／|e_y|入口中央値 {ey_entry_med*1000:.2f} mm",
              flush=True)

    n_collided = sum(1 for row in rows if row["collided"])
    print(f"\n【集計】{len(rows)} 面中 衝突 {n_collided} 面"
          + (f"（{', '.join(row['maze'] for row in rows if row['collided'])}）"
             if n_collided else ""))

    out = Path(args.out or (REPO_ROOT / "outputs" / "exp_016_diagonal" / "probe_diag_cap"
                            / f"{args.group}_sf{args.safety:g}_v{args.v_diag:g}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        git_rev=git_rev(), maze_dir=args.maze_dir, group=args.group, safety=args.safety,
        v_diag=args.v_diag, control_dt=dt, L_c=L_C_CLOTHOID_M,
        note=("探索的測定であり、ここから結論を主張するには実験カードと"
              "事前登録が要る（判定条文なし）。"),
        n_mazes=len(rows), n_collided=n_collided,
        collided_mazes=[row["maze"] for row in rows if row["collided"]],
        rows=rows), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
