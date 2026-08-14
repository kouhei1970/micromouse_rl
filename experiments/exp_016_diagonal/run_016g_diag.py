#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-G の診断 3 件 — **実装より前に測る**（`card_016g.md` §2-2）。

**本スクリプトは走行系の制御を 1 行も変えない。**既存の方策をそのまま走らせて
記録するだけ（D1・D2）と、走らせずに幾何だけを解くもの（D3）である。

--------------------------------------------------------------------------
D1（予測 G1）: 円弧進入で横加速度が実際にステップしているか
--------------------------------------------------------------------------
P6（設計提案書）§2-2 の見立て「曲率が 0 → 1/R とステップで変わるので、
その瞬間に横加速度が v²/R だけ跳ぶ」の**直接の確認**。

⚠️ **参照側だけを測ると同語反復になる。**参照経路の曲率配列は
`diag_path.build_diagonal_path` が直線標本に 0、円弧標本に ±1/R を入れて作るので、
**参照がステップすることは構成上あたりまえ**である。**知りたいのは機体が
実際にステップした横加速度を受けているか**なので、**実測の横加速度
a_lat = |v·ω|（v = 前進速度・ω = ヨー角速度。どちらも privileged_velocity の値）
を主の量とし、参照側は参考として併記する。**

--------------------------------------------------------------------------
D2（予測 G2）: 横偏差がどこで立ち上がるか
--------------------------------------------------------------------------
016-F の E8 は「斜め直線の横偏差は円弧から**持ち込まれて**減衰する」ことを
示したが、**発生源は特定していない**。そこで**参照経路までの横のずれ**を
走行の全区間で測り、**円弧の入口を原点にした弧長**に対して並べ、
**直進側・円弧の中・出口の後**の 3 つの窓のどこで増分が最大になるかを見る。

**横のずれの定義**: 参照経路（折れ線）への最近点までの符号つき距離。
**斜め直線区間では参照経路が斜め走路の中心線そのものなので、016-B/016-C の
`lateral_deviation`（節点 a→b の直線までの距離）と一致する。**

--------------------------------------------------------------------------
D3（予測 G3）: クロソイドの長さの幾何的な上限
--------------------------------------------------------------------------
**走らせない。**現行の円弧接続が使っている接線の余地（`room`）の中に、
クロソイド（曲率が弧長に比例して 0 → 1/R まで増える緩和曲線）を挟んだときの
接線長が収まる最大の L_c を、**フレネル積分を数値積分して**求める。

    τ   = L_c / (2R)                     … クロソイド 1 本ぶんの方位変化 [rad]
    X_s = ∫₀^{L_c} cos( s² / (2·R·L_c) ) ds
    Y_s = ∫₀^{L_c} sin( s² / (2·R·L_c) ) ds
    ΔR  = Y_s − R·(1 − cos τ)            … 円弧が内側へずれる量（シフト）[m]
    k   = X_s − R·sin τ
    T_s = k + (R + ΔR)·tan(θ/2)          … 交点から緩和曲線の始点までの接線長 [m]

制約は 2 つ:
  (1) **T_s ≤ room**（現行の円弧接続と同じ余地に収まること）
  (2) **2τ ≤ θ**（曲率が 1/R に達する前に曲がり終わってはいけない
      ＝ 円弧が残ること。等号は緩和曲線だけで曲がりきる極限）

**ΔR は経路を曲がりの内側へ寄せるので、壁までの余裕も食う。**併せて報告する。

使い方:
    .venv/bin/python -u experiments/exp_016_diagonal/run_016g_diag.py \
        --safety 0.75 --v-diag 0.45
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

import run_016c  # noqa: E402
import run_016f0_ladder  # noqa: E402
from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402
from competition.baseline_slalom_e1_tr import load_time_model  # noqa: E402
from competition.route_planner import value_field  # noqa: E402
from mouse.mjcf import build_maze_robot_xml  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

from diag_path import build_diagonal_path  # noqa: E402
from diagonal_model import (DELTA8, DIAGONALS, DiagonalGridModel,  # noqa: E402
                            cell_center_node, descend, turn_deg)
from geometry import git_rev  # noqa: E402
from route_model import connects_true, load_maze  # noqa: E402
from run_016b import cut_segment, longest_diagonal_run  # noqa: E402

R_ARC_M = run_016c.R_ARC_M


# ==========================================================================
# 共通の道具
# ==========================================================================
def cross_track(px, py, rx, ry):
    """参照の折れ線 (rx, ry) への**符号つき**横のずれ [m] と、その最近点の弧長の添字。

    折れ線の各辺へ射影し、最も近い辺の符号つき距離（左が正）を返す。
    """
    ax, ay = rx[:-1], ry[:-1]
    ux, uy = np.diff(rx), np.diff(ry)
    ln2 = ux * ux + uy * uy
    ln2 = np.where(ln2 < 1e-24, 1e-24, ln2)
    t = ((px - ax) * ux + (py - ay) * uy) / ln2
    t = np.clip(t, 0.0, 1.0)
    qx, qy = ax + t * ux, ay + t * uy
    d2 = (px - qx) ** 2 + (py - qy) ** 2
    j = int(np.argmin(d2))
    # 符号: 辺の進行方向に対して左が正
    s = np.sign(ux[j] * (py - ay[j]) - uy[j] * (px - ax[j]))
    return float(s * math.sqrt(d2[j])), j, float(t[j])


def arc_spans(kinds, s):
    """参照経路の中で円弧が連続する区間 [i0, i1] と、その弧長の範囲を返す。"""
    out, i = [], 0
    n = len(kinds)
    while i < n:
        if kinds[i] != "arc":
            i += 1
            continue
        j = i
        while j + 1 < n and kinds[j + 1] == "arc":
            j += 1
        out.append(dict(i0=i, i1=j, s0=float(s[i]), s1=float(s[j])))
        i = j + 1
    return out


def arc_turn_deg(path, sp):
    """円弧 1 本の旋回角 [deg]（弧長 ÷ 半径。半径は曲率から復元する）。"""
    kap = abs(float(path.curvature[sp["i0"]]))
    if kap < 1e-9:
        return float("nan"), float("nan")
    R = 1.0 / kap
    # 標本は 1 点ぶん手前で切れているので、刻み 1 つぶんを足して弧長にする
    ds = (sp["s1"] - sp["s0"]) / max(1, sp["i1"] - sp["i0"])
    return math.degrees((sp["s1"] - sp["s0"] + ds) * kap), R


# ==========================================================================
# 走行して記録する（**制御には触らない**）
# ==========================================================================
def drive_and_record(xml_path, params, nodes, dirs, v_diag, v_walls, h_walls,
                     policy_cls, max_s=40.0):
    """1 迷路 1 速度水準を走らせ、制御周期ごとの生の記録を返す。"""
    path, kinds, idxs = build_diagonal_path(nodes, dirs, params.cell_size, R_ARC_M)
    sim = MouseSim(str(xml_path), params=params)
    start_cell = (nodes[0][0] // 2, nodes[0][1] // 2)
    heading = math.degrees(math.atan2(DELTA8[dirs[0]][1], DELTA8[dirs[0]][0]))
    sim.full_reset(cell=start_cell, heading_deg=heading)

    v_arr = np.where(kinds == "straight", 1e9, v_diag)
    pol = policy_cls(path, v_arr)
    pol.bind_sim(sim)
    pol.bind_maze(v_walls, h_walls)
    pol.on_maze_start(dict(width=16, height=16))

    rec, collided = [], False
    for _ in range(int(max_s / params.control_dt)):
        vl, vr = pol.act(sim.observation())
        out = sim.step_control(vl, vr)
        x, y, yaw = sim.privileged_pose()
        v, w = sim.privileged_velocity()
        idx = int(getattr(pol, "_cursor", 0))
        pth = getattr(pol, "_path", None)
        kap = float(pth.curvature[idx]) if pth is not None else 0.0
        vpl = float(pth.speed[idx]) if pth is not None else 0.0
        rec.append((sim.sim_time, x, y, yaw, v, w, idx, kap, vpl,
                    float(getattr(pol, "_v_setpoint", 0.0))))
        if out.get("collision"):
            collided = True
            break
        if pol.finished:
            break
    cols = ("t", "x", "y", "yaw", "v", "w", "idx", "kappa", "v_plan", "v_set")
    return dict(zip(cols, np.asarray(rec, dtype=float).T)), collided, path, kinds


# ==========================================================================
# D1（予測 G1）: 横加速度のステップ
# ==========================================================================
def diag_d1(r, path, kinds, dt):
    """円弧進入の 1 ティックあたりの横加速度の最大増分。

    - **実測** a_lat = |v·ω|（主の量）
    - **参照** a_lat = v_plan²·|κ(カーソル)|（参考。構成上ステップする）
    """
    a_act = np.abs(r["v"] * r["w"])
    a_ref = r["v_plan"] ** 2 * np.abs(r["kappa"])
    idx = r["idx"].astype(int)
    kind_t = np.asarray([str(kinds[min(i, len(kinds) - 1)]) for i in idx])

    # 円弧へ入ったティック（直前が円弧でない → 今が円弧）
    enter = np.where((kind_t[1:] == "arc") & (kind_t[:-1] != "arc"))[0] + 1
    out = dict(n_entries=int(len(enter)))
    if len(enter) == 0:
        return out

    d_act, d_ref, a_before, a_after = [], [], [], []
    for e in enter:
        lo, hi = max(1, e - 2), min(len(a_act), e + 3)     # 進入の前後 2 ティック
        d_act.append(float(np.max(np.diff(a_act[lo - 1:hi]))))
        d_ref.append(float(np.max(np.diff(a_ref[lo - 1:hi]))))
        a_before.append(float(a_act[e - 1]))
        a_after.append(float(np.max(a_act[e:min(len(a_act), e + 5)])))
    out.update(
        d_a_lat_act_max=float(np.max(d_act)), d_a_lat_act_med=float(np.median(d_act)),
        d_a_lat_ref_max=float(np.max(d_ref)), d_a_lat_ref_med=float(np.median(d_ref)),
        a_lat_before_med=float(np.median(a_before)),
        a_lat_after_med=float(np.median(a_after)),
        # 参考: 1 ティックの増分を dt で割った「横加加速度」相当 [m/s^3]
        jerk_lat_act_max=float(np.max(d_act)) / dt,
    )
    return out


# ==========================================================================
# D2（予測 G2）: 横偏差の立ち上がりの場所
# ==========================================================================
def diag_d2(r, path, kinds, pre_m=0.030, post_m=0.060):
    """円弧の入口を原点にした弧長に対する、横のずれの増分を 3 つの窓で測る。

    窓: 直線側 [−pre, 0)／円弧の中 [0, L_arc]／出口の後 (L_arc, L_arc+post]
    """
    rx, ry, s = np.asarray(path.x), np.asarray(path.y), np.asarray(path.s)
    e_t, s_t = [], []
    for x, y in zip(r["x"], r["y"]):
        e, j, t = cross_track(x, y, rx, ry)
        e_t.append(e)
        s_t.append(s[j] + t * (s[j + 1] - s[j]))
    e_t, s_t = np.asarray(e_t), np.asarray(s_t)

    spans = arc_spans(kinds, s)
    rows = []
    for sp in spans:
        s0, s1 = sp["s0"], sp["s1"]
        w_pre = (s_t >= s0 - pre_m) & (s_t < s0)
        w_arc = (s_t >= s0) & (s_t <= s1)
        w_post = (s_t > s1) & (s_t <= s1 + post_m)
        if w_arc.sum() < 2:
            continue

        def rise(mask):
            """その窓の中で |e| がどれだけ増えたか（窓に入った時点からの最大増分）。"""
            if mask.sum() < 2:
                return None
            a = np.abs(e_t[mask])
            return float(np.max(a) - a[0])

        deg, R_arc = arc_turn_deg(path, sp)
        rows.append(dict(
            s0=s0, s1=s1, arc_len=s1 - s0, turn_deg=deg, R_m=R_arc,
            turn_bin=(45 if abs(deg - 45) < 8 else (90 if abs(deg - 90) < 8 else round(deg))),
            e_abs_at_entry=float(abs(e_t[w_arc][0])) if w_arc.sum() else None,
            e_abs_at_exit=float(abs(e_t[w_arc][-1])) if w_arc.sum() else None,
            e_abs_max_pre=float(np.max(np.abs(e_t[w_pre]))) if w_pre.sum() else None,
            e_abs_max_arc=float(np.max(np.abs(e_t[w_arc]))),
            e_abs_max_post=float(np.max(np.abs(e_t[w_post]))) if w_post.sum() else None,
            rise_pre=rise(w_pre), rise_arc=rise(w_arc), rise_post=rise(w_post),
            n_pre=int(w_pre.sum()), n_arc=int(w_arc.sum()), n_post=int(w_post.sum())))
    return rows


# ==========================================================================
# D3（予測 G3）: クロソイドの長さの幾何的な上限（**走らせない**）
# ==========================================================================
def clothoid_tangent(L_c, R, theta, n=4001):
    """クロソイド ＋ 円弧 ＋ クロソイド の接線長 T_s とシフト ΔR を返す。

    フレネル積分は台形則で数値積分する（n 点。L_c ≤ 0.1 m・R = 0.06 m の
    範囲では 4001 点で 1e-12 m の精度がある）。
    """
    if L_c <= 0:
        return R * math.tan(theta / 2.0), 0.0, 0.0
    tau = L_c / (2.0 * R)
    u = np.linspace(0.0, L_c, n)
    ang = u * u / (2.0 * R * L_c)
    X_s = float(np.trapezoid(np.cos(ang), u))
    Y_s = float(np.trapezoid(np.sin(ang), u))
    dR = Y_s - R * (1.0 - math.cos(tau))
    k = X_s - R * math.sin(tau)
    T_s = k + (R + dR) * math.tan(theta / 2.0)
    return T_s, dR, tau


def max_clothoid_len(R, theta, room, hi=0.20, tol=1e-7):
    """制約 (1) T_s ≤ room かつ (2) 2τ ≤ θ を満たす最大の L_c [m]。"""
    cap_arc = R * theta                       # 制約 (2): 2·(L_c/2R) ≤ θ
    if clothoid_tangent(0.0, R, theta)[0] > room + 1e-12:
        return 0.0, "現行の円弧すら余地に収まっていない"
    lo, high = 0.0, min(hi, cap_arc)
    if clothoid_tangent(high, R, theta)[0] <= room:
        return high, "制約 (2) 円弧が残る条件で頭打ち"
    for _ in range(80):
        mid = 0.5 * (lo + high)
        if clothoid_tangent(mid, R, theta)[0] <= room:
            lo = mid
        else:
            high = mid
        if high - lo < tol:
            break
    return lo, "制約 (1) 接線長が余地に収まる条件で頭打ち"


def corner_rooms(nodes, dirs, cell_size, R):
    """`build_diagonal_path` と**同じ数え方**で、曲がり角ごとの余地と旋回角を出す。"""
    from diagonal_model import node_xy
    pts = [np.array(node_xy(n, cell_size), dtype=float) for n in nodes]
    m = len(dirs)
    seg_len = [float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(m)]
    leg_of, leg_len, cur = [0] * m, [], 0.0
    for k in range(m):
        cur += seg_len[k]
        leg_of[k] = len(leg_len)
        if k + 1 == m or turn_deg(dirs[k], dirs[k + 1]) > 0:
            leg_len.append(cur)
            cur = 0.0
    out = []
    for k in range(1, m):
        deg = turn_deg(dirs[k - 1], dirs[k])
        if deg <= 0:
            continue
        th = math.radians(deg)
        diagonal_involved = (dirs[k - 1] in DIAGONALS) or (dirs[k] in DIAGONALS)
        room = min(leg_len[leg_of[k - 1]], leg_len[leg_of[k]]) / 2.0
        r_c = float(R)
        if r_c * math.tan(th / 2.0) > room + 1e-12:
            r_c = room / math.tan(th / 2.0)          # 収まる最大まで縮める
        out.append(dict(k=k, turn_deg=deg, room_m=room, R_m=r_c,
                        diagonal_involved=bool(diagonal_involved),
                        tangent_now_m=r_c * math.tan(th / 2.0)))
    return out


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safety", type=float, default=0.75,
                    help="旋回安全率（016-cal の校正値 0.75 が既定）")
    ap.add_argument("--v-diag", type=float, default=0.45,
                    help="斜め・円弧に掛ける速度水準 [m/s]")
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or (REPO_ROOT / "outputs" / "exp_016_diagonal" / "016g"
                            / f"diag_sf{args.safety:g}_v{args.v_diag:g}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)

    # 帯の安全弁（裁定 R40 条件 4）。**調整用迷路（seed 41000〜）専用**
    seeds = [int(q.stem.split("_")[1])
             for q in sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"))]
    print(describe_seeds(seeds, "competition"))
    assert_seeds_allowed(seeds, namespace="competition", purpose="validate")

    a, b = load_time_model()
    params = RobotParams()
    dt = params.control_dt
    policy_cls = run_016f0_ladder.make_policy_class(
        k_acc_ff=1.0, ref_interp=True, safety=args.safety)
    print(f"旋回安全率 {args.safety:g}／速度水準 {args.v_diag:g} m/s"
          f"／制御周期 {dt*1000:.1f} ms\n")

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
                          dirs=p["dirs"][i:j], v=v, h=h))

    # ---------------- D3（走らせない。先に出す）------------------------
    print("=" * 74)
    print("D3（予測 G3）: クロソイドの長さの幾何的な上限 — **走らせない**")
    print("=" * 74)
    d3_rows = []
    for q in faces:
        for c in corner_rooms(q["nodes"], q["dirs"], params.cell_size, R_ARC_M):
            Lc, why = max_clothoid_len(c["R_m"], math.radians(c["turn_deg"]), c["room_m"])
            T_s, dR, tau = clothoid_tangent(Lc, c["R_m"], math.radians(c["turn_deg"]))
            d3_rows.append(dict(maze=q["maze"], **c, L_c_max_m=Lc, reason=why,
                                tangent_new_m=T_s, shift_m=dR, tau_deg=math.degrees(tau)))
    by_turn = {}
    for r in d3_rows:
        key = (r["turn_deg"], "斜め絡み" if r["diagonal_involved"] else "直進のみ")
        by_turn.setdefault(key, []).append(r)
    print(f"{'旋回角':>6} {'種別':<10}{'件数':>5}{'余地[mm]':>10}{'現行接線[mm]':>13}"
          f"{'L_c上限[mm]':>13}{'新接線[mm]':>12}{'シフト[mm]':>12}  頭打ちの理由")
    for key in sorted(by_turn):
        deg, kindname = key
        g = by_turn[key]
        room = np.median([r["room_m"] for r in g]) * 1000
        tn = np.median([r["tangent_now_m"] for r in g]) * 1000
        lc = np.array([r["L_c_max_m"] for r in g]) * 1000
        tnew = np.median([r["tangent_new_m"] for r in g]) * 1000
        sh = np.median([r["shift_m"] for r in g]) * 1000
        why = max(set(r["reason"] for r in g),
                  key=lambda w: sum(1 for r in g if r["reason"] == w))
        n20 = sum(1 for v in lc if v >= 20.0)
        print(f"{deg:>6.0f} {kindname:<10}{len(g):>5}{room:>10.2f}{tn:>13.2f}"
              f"{np.median(lc):>13.2f}{tnew:>12.2f}{sh:>12.3f}  {why}")
        print(f"{'':>6} {'':<10}{'':>5}{'':>10}{'':>13}"
              f"（範囲 {lc.min():.2f}〜{lc.max():.2f}・**20 mm 以上とれるのは {n20}/{len(g)} 箇所**）")

    # ---------------- D1・D2（走らせる）--------------------------------
    print("\n" + "=" * 74)
    print("D1（予測 G1）/ D2（予測 G2）: 走らせて測る")
    print("=" * 74)
    d1_rows, d2_rows = [], []
    for q in faces:
        r, collided, path, kinds = drive_and_record(
            q["xml"], params, q["nodes"], q["dirs"], args.v_diag, q["v"], q["h"], policy_cls)
        d1 = diag_d1(r, path, kinds, dt)
        d1["maze"], d1["collided"] = q["maze"], collided
        d1_rows.append(d1)
        for row in diag_d2(r, path, kinds):
            row["maze"] = q["maze"]
            d2_rows.append(row)
        print(f"  {q['maze']:<12} 円弧進入 {d1['n_entries']:>2} 回"
              f"／横加速度の 1 ティック増分 実測 最大 {d1.get('d_a_lat_act_max', float('nan')):.3f}"
              f"・参照 最大 {d1.get('d_a_lat_ref_max', float('nan')):.3f} m/s²", flush=True)

    print("\n--- D1 まとめ（予測 G1: 円弧進入の 1 ティックあたりの増分 ≥ 2.0 m/s²）---")
    for key, lab in (("d_a_lat_act_max", "実測 |v·ω| の 1 ティック増分 最大"),
                     ("d_a_lat_act_med", "実測 |v·ω| の 1 ティック増分 中央値"),
                     ("d_a_lat_ref_max", "参照 v²κ の 1 ティック増分 最大"),
                     ("d_a_lat_ref_med", "参照 v²κ の 1 ティック増分 中央値")):
        vals = [q[key] for q in d1_rows if key in q]
        print(f"  {lab:<34} 20 迷路の中央値 {np.median(vals):>7.3f}"
              f"（{min(vals):.3f}〜{max(vals):.3f}）m/s²")
    vals = [q["a_lat_before_med"] for q in d1_rows if "a_lat_before_med" in q]
    vals2 = [q["a_lat_after_med"] for q in d1_rows if "a_lat_after_med" in q]
    print(f"  {'円弧の直前 → 円弧の中の横加速度':<32} "
          f"{np.median(vals):.3f} → {np.median(vals2):.3f} m/s²")

    print("\n--- D2 まとめ（予測 G2: 横偏差の立ち上がりは円弧の中で起きる）---")
    print(f"  円弧の通過 {len(d2_rows)} 件（20 迷路の合計）")
    for key, lab in (("rise_pre", "直進側 [-30, 0) mm"),
                     ("rise_arc", "円弧の中 [0, L_arc]"),
                     ("rise_post", "出口の後 (L_arc, +60] mm")):
        vals = [q[key] * 1000 for q in d2_rows if q.get(key) is not None]
        print(f"  {lab:<24} 増分の中央値 {np.median(vals):>7.3f} mm"
              f"（{min(vals):.3f}〜{max(vals):.3f}）n={len(vals)}")
    win = {"直進側": 0, "円弧の中": 0, "出口の後": 0}
    for q in d2_rows:
        c = {"直進側": q.get("rise_pre") or -1, "円弧の中": q.get("rise_arc") or -1,
             "出口の後": q.get("rise_post") or -1}
        win[max(c, key=c.get)] += 1
    print(f"  **増分が最大だった窓の内訳**: " +
          "／".join(f"{k} {v} 件" for k, v in win.items()))
    for key, lab in (("e_abs_at_entry", "円弧の入口での |横のずれ|"),
                     ("e_abs_max_arc", "円弧の中の |横のずれ| 最大"),
                     ("e_abs_at_exit", "円弧の出口での |横のずれ|"),
                     ("e_abs_max_post", "出口の後 60 mm の最大")):
        vals = [q[key] * 1000 for q in d2_rows if q.get(key) is not None]
        print(f"  {lab:<26} 中央値 {np.median(vals):>7.3f} mm"
              f"（{min(vals):.3f}〜{max(vals):.3f}）")

    # ---- 旋回角ごとの内訳（**実装の形を決めるのに要る**）----------------
    print("\n--- D2 の旋回角ごとの内訳（どの曲がり角が横偏差を作っているか）---")
    print(f"  {'旋回角':>6}{'件数':>5}{'円弧長[mm]':>12}{'入口 |e|[mm]':>14}"
          f"{'出口 |e|[mm]':>14}{'円弧の中の増分[mm]':>20}")
    for tb in sorted(set(q["turn_bin"] for q in d2_rows)):
        g = [q for q in d2_rows if q["turn_bin"] == tb]
        al = np.median([q["arc_len"] for q in g]) * 1000
        en = np.median([q["e_abs_at_entry"] for q in g]) * 1000
        ex = np.median([q["e_abs_at_exit"] for q in g]) * 1000
        ri = np.array([q["rise_arc"] for q in g]) * 1000
        print(f"  {tb:>6}{len(g):>5}{al:>12.2f}{en:>14.3f}{ex:>14.3f}"
              f"{np.median(ri):>14.3f}（{ri.min():.2f}〜{ri.max():.2f}）")

    json.dump(dict(git_rev=git_rev(), safety=args.safety, v_diag=args.v_diag,
                   R_arc_m=R_ARC_M, control_dt=dt, maze_dir=args.maze_dir,
                   d1=d1_rows, d2=d2_rows, d3=d3_rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
