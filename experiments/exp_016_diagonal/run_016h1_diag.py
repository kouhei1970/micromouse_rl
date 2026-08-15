#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-H1 の実装前診断 D1・D2 — **制御を 1 行も変えずに、実装してよいかを先に測る**。

カード `card_016h1.md` §3。変更案は「**角加速度の前置補償**」:
参照角速度 $\\omega_\\text{ref} = \\kappa v$ の時間微分を、車輪ループの遅れ
$t_\\text{delay} = 2\\,\\Delta t_\\text{ctrl}$ ぶん先取りして指令へ足す。
**この案が成り立つ前提は 2 つあり、どちらも未測定である。**

  **D1**: 円弧・クロソイド区間で $\\omega_\\text{act}$ が $\\omega_\\text{des}$ に
          どれだけ遅れているか。**遅れが $t_\\text{delay}$ = 0.02 s と同じ桁
          （0.01〜0.04 s）でなければ、補償の前提が違う → 実装せず設計へ戻る。**
  **D2**: 円弧出口の横偏差と、円弧の中の角速度追従誤差の積分の相関。
          **相関が 0 近傍なら、横偏差は角速度の遅れ由来ではない → 別の機構を疑う。**

測り方は既存の診断ハーネスを踏襲する（裁定 R23。定義を写さず import する）:
`run_016f0_ladder.make_policy_class` が現行の採用構成の方策を作り、
`run_016g_ladder.make_builder` が 016-G のクロソイド経路を作る。
**本スクリプトは `_wheel_targets_to_voltage` を包んで記録するだけで、返り値は素通しである。**

    .venv/bin/python experiments/exp_016_diagonal/run_016h1_diag.py
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
from competition.route_planner import value_field                         # noqa: E402
from mouse.mjcf import build_maze_robot_xml                               # noqa: E402
from mouse.params import RobotParams                                      # noqa: E402
from mouse.sim import MouseSim                                            # noqa: E402

import run_016f0_ladder                                                   # noqa: E402
import run_016g_ladder                                                    # noqa: E402
from diagonal_model import (DELTA8, DiagonalGridModel, cell_center_node,  # noqa: E402
                            descend)
from geometry import git_rev                                              # noqa: E402
from route_model import connects_true, load_maze                          # noqa: E402
from run_016b import cut_segment, longest_diagonal_run                    # noqa: E402
from run_016c import R_ARC_M                                              # noqa: E402

KIND = {"straight": 0, "arc": 1, "diagonal": 2}
MAX_LAG_TICKS = 20          # D1 で探す遅れの上限（0.20 s。前提 0.02 s の 10 倍）


def make_probed(policy_cls):
    """`_wheel_targets_to_voltage` を包んで指令角速度を記録するだけの派生クラス。"""

    class Probed(policy_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._omega_cmd = float("nan")

        def _wheel_targets_to_voltage(self, v_cmd, omega_cmd, obs):
            self._omega_cmd = float(omega_cmd)
            return super()._wheel_targets_to_voltage(v_cmd, omega_cmd, obs)

    return Probed


def drive_and_record(xml_path, params, nodes, dirs, v_diag, v_walls, h_walls,
                     policy_cls, builder, max_s=40.0):
    """1 迷路を走らせ、制御周期ごとの記録を返す。**制御には触らない。**"""
    path, kinds, _idx = builder(nodes, dirs, params.cell_size, R_ARC_M)
    sim = MouseSim(str(xml_path), params=params)
    sim.full_reset(cell=(nodes[0][0] // 2, nodes[0][1] // 2),
                   heading_deg=math.degrees(math.atan2(DELTA8[dirs[0]][1],
                                                       DELTA8[dirs[0]][0])))
    pol = make_probed(policy_cls)(path, np.where(kinds == "straight", 1e9, v_diag))
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
        # 🔴 **指令と同時刻の状態を取る**（`run_016f0_diag` と同じ作法）。
        # step_control の**後**で読むと、記録の omega_act が 1 ティック先の値になり、
        # 遅れの推定が 1 ティックぶん詰まる（2026-08-15 に自分のハーネスで踏んだ誤り）。
        x, y, yaw = sim.privileged_pose()
        v_act, w_act = sim.privileged_velocity()
        cur = int(getattr(pol, "_cursor", 0))
        k = int(np.argmin((px - x) ** 2 + (py - y) ** 2))
        head = float(path.heading[cur])
        e_y = ((x - float(path.x[cur])) * (-math.sin(head))
               + (y - float(path.y[cur])) * math.cos(head))
        rec.append((sim.sim_time, pol._omega_cmd, w_act, e_y,
                    KIND[str(kinds[k])], float(path.s[cur]),
                    float(path.curvature[cur]), v_act, x, y, yaw))
        out = sim.step_control(vl, vr)
        if out.get("collision"):
            collided = True
            break
        if pol.finished:
            break
    cols = ("t", "omega_des", "omega_act", "e_y", "kind", "s", "kappa", "v_act",
            "x", "y", "yaw")
    return dict(zip(cols, np.asarray(rec, dtype=float).T)), collided


def d1_lag(r, dt):
    """円弧区間で omega_act が omega_des に遅れているティック数（相互相関の最大点）。"""
    m = r["kind"] == KIND["arc"]
    if m.sum() < 30:
        return None
    a = r["omega_des"][m] - r["omega_des"][m].mean()
    b = r["omega_act"][m] - r["omega_act"][m].mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    best = max(range(MAX_LAG_TICKS + 1),
               key=lambda L: float(np.corrcoef(a[:len(a) - L], b[L:])[0, 1])
               if len(a) - L > 10 else -2.0)
    peak = float(np.corrcoef(a[:len(a) - best], b[best:])[0, 1]) if len(a) - best > 10 else float("nan")
    return dict(lag_ticks=best, lag_s=best * dt, peak_corr=peak,
                corr_at_zero=float(np.corrcoef(a, b)[0, 1]), n=int(m.sum()))


def d1_tau(r, dt, path_kappa=None):
    """一次遅れの時定数 tau を最小二乗で当てる（相互相関の 1 ティック刻みより細かく測る）。

    模型: omega_act[k+1] = omega_act[k] + (dt/tau)·(omega_des[k] − omega_act[k])
    ⇒ 差分 d[k] = omega_act[k+1] − omega_act[k] を 誤差 e[k] = omega_des[k] − omega_act[k]
      へ切片なしで回帰し、傾き g = dt/tau から tau = dt/g を得る。

    層別（カード §3 の要求）: 円弧区間のうち **曲率が変化している標本（クロソイド）** と
    **曲率が一定の標本（定曲率の円弧）** に分ける。
    """
    out = {}
    arc = r["kind"] == KIND["arc"]
    kap = r.get("kappa")
    strata = {"円弧ぜんぶ": arc}
    if kap is not None:
        dk = np.abs(np.gradient(kap))
        thr = 1e-6
        strata["クロソイド"] = arc & (dk > thr)
        strata["定曲率"] = arc & (dk <= thr)
    for name, m in strata.items():
        idx = np.flatnonzero(m)
        idx = idx[idx < len(r["t"]) - 1]
        if len(idx) < 30:
            out[name] = None
            continue
        e = r["omega_des"][idx] - r["omega_act"][idx]
        d = r["omega_act"][idx + 1] - r["omega_act"][idx]
        denom = float(np.dot(e, e))
        if denom < 1e-12:
            out[name] = None
            continue
        g = float(np.dot(e, d) / denom)
        out[name] = dict(tau_s=(dt / g if g > 1e-9 else float("inf")), gain=g, n=int(len(idx)))
    return out


def d2_pairs(r, dt):
    """円弧ごとに（角速度追従誤差の積分, 出口の |e_y|）を返す。"""
    kind = r["kind"]
    out = []
    i = 0
    n = len(kind)
    while i < n:
        if kind[i] != KIND["arc"]:
            i += 1
            continue
        j = i
        while j < n and kind[j] == KIND["arc"]:
            j += 1
        if j - i >= 10 and j < n:
            err = float(np.sum(np.abs(r["omega_des"][i:j] - r["omega_act"][i:j])) * dt)
            out.append((err, abs(float(r["e_y"][j]))))
        i = j
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safety", type=float, default=0.75)
    ap.add_argument("--v-diag", type=float, default=0.45)
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = [int(q.stem.split("_")[1])
             for q in sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"))]
    print(describe_seeds(seeds, "competition"))
    assert_seeds_allowed(seeds, namespace="competition", purpose="validate")

    params = RobotParams()
    dt = params.control_dt
    a, b = load_time_model()
    policy_cls = run_016f0_ladder.make_policy_class(k_acc_ff=1.0, ref_interp=True,
                                                    safety=args.safety)
    builder = run_016g_ladder.make_builder(L_C_CLOTHOID_M)
    print(f"採用構成: F0 + F0-b + 安全率 {args.safety:g} + 45° クロソイド "
          f"{L_C_CLOTHOID_M*1000:.4f} mm／速度水準 {args.v_diag:g} m/s／"
          f"制御周期 {dt*1000:.1f} ms／前提の遅れ {2*dt*1000:.0f} ms\n")

    rows, pairs = [], []
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
        r, collided = drive_and_record(xml, params, p["nodes"][i:j + 1], p["dirs"][i:j],
                                       args.v_diag, v, h, policy_cls, builder)
        if not r or len(r["t"]) == 0:
            print(f"{f.stem}: 記録なし")
            continue
        lag = d1_lag(r, dt)
        tau = d1_tau(r, dt)
        m_arc = r["kind"] == KIND["arc"]
        arc_v = float(np.median(r["v_act"][m_arc])) if m_arc.any() else float("nan")
        _al = np.abs(r["v_act"][m_arc] * r["omega_act"][m_arc]) if m_arc.any() else np.array([np.nan])
        arc_alat = float(np.median(_al))
        arc_alat_p95 = float(np.percentile(_al, 95))
        arc_alat_max = float(np.max(_al))
        arc_v_max = float(np.max(r["v_act"][m_arc])) if m_arc.any() else float("nan")
        # --- 016-H1d 副1: 実効半径 R_act = v/|ω|。**その迷路の p5**（最も曲がれている点） ---
        w = np.abs(r["omega_act"][m_arc]) if m_arc.any() else np.array([np.nan])
        vv = r["v_act"][m_arc] if m_arc.any() else np.array([np.nan])
        ok = w > 1e-3
        r_act_p5 = float(np.percentile(vv[ok] / w[ok], 5)) if ok.any() else float("nan")
        # --- 016-H1d 副2: スリップ角。速度の向き（位置の差分）と姿勢の差 ---
        dx = np.gradient(r["x"]); dy = np.gradient(r["y"])
        beta = np.arctan2(dy, dx) - r["yaw"]
        beta = np.abs((beta + np.pi) % (2 * np.pi) - np.pi)
        moving = r["v_act"] > 0.05
        mb = m_arc & moving
        beta_p95 = float(np.degrees(np.percentile(beta[mb], 95))) if mb.any() else float("nan")
        pr = d2_pairs(r, dt)
        pairs += pr
        rows.append(dict(maze=f.stem, collided=bool(collided), n_ticks=int(len(r["t"])),
                         d1=lag, d1_tau=tau, n_arcs=len(pr),
                         arc_v_med=arc_v, arc_alat_med=arc_alat,
                         arc_alat_p95=arc_alat_p95, arc_alat_max=arc_alat_max,
                         arc_v_max=arc_v_max, r_act_p5=r_act_p5, beta_p95_deg=beta_p95,
                         ey_exit_med=float(np.median([q[1] for q in pr])) if pr else float("nan")))
        print(f"{f.stem}: 円弧 {len(pr)} 本／"
              + (f"遅れ {lag['lag_ticks']} ティック = {lag['lag_s']*1000:.0f} ms "
                 f"(相関 {lag['peak_corr']:.3f})" if lag else "D1 は標本不足"), flush=True)

    print("\n" + "=" * 70)
    lags = [x["d1"]["lag_s"] for x in rows if x["d1"]]
    if lags:
        lo, hi = 0.01, 0.04
        med = float(np.median(lags))
        inside = sum(1 for L in lags if lo <= L <= hi)
        print(f"【D1】遅れの中央値 {med*1000:.1f} ms（{min(lags)*1000:.0f}〜{max(lags)*1000:.0f}）"
              f"／前提 {2*dt*1000:.0f} ms")
        print(f"      0.01〜0.04 s の内側: {inside}/{len(lags)} 迷路"
              f"  ⇒ 予測 H4 = {'成立（実装へ）' if lo <= med <= hi else '不成立（設計へ戻る）'}")
    else:
        print("【D1】判定不能（標本不足）")

    av=[x["arc_v_med"] for x in rows if x.get("arc_v_med")==x.get("arc_v_med")]
    aa=[x["arc_alat_med"] for x in rows if x.get("arc_alat_med")==x.get("arc_alat_med")]
    if av:
        p95 = np.median([x["arc_alat_p95"] for x in rows])
        mx = np.median([x["arc_alat_max"] for x in rows])
        vmx = np.median([x["arc_v_max"] for x in rows])
        print(f"\n【円弧の実測】速度 中央値 {np.median(av):.3f}／最大 {vmx:.3f} m/s（指令 {args.v_diag:g}）")
        print(f"  横加速度 中央値 {np.median(aa):.3f}／p95 {p95:.3f}／最大 {mx:.3f} m/s²"
              f"  = 物理限界 6.200 の {np.median(aa)/6.2*100:.0f} / {p95/6.2*100:.0f} / {mx/6.2*100:.0f} %")
    print("\n【D1-tau】一次遅れの時定数（相互相関より細かい測り方。層別）")
    for name in ("円弧ぜんぶ", "クロソイド", "定曲率"):
        vals = [x["d1_tau"][name]["tau_s"] for x in rows
                if x.get("d1_tau") and x["d1_tau"].get(name)]
        if not vals:
            print(f"  {name}: 標本不足")
            continue
        v = np.array(vals)
        print(f"  {name:10s} tau 中央値 {np.median(v)*1000:6.2f} ms"
              f"（{v.min()*1000:5.2f}〜{v.max()*1000:6.2f}・n={len(v)} 迷路）")

    if len(pairs) >= 10:
        e = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        c = float(np.corrcoef(e, y)[0, 1])
        sl = float(np.polyfit(e, y, 1)[0])
        print(f"\n【D2】円弧 {len(pairs)} 本。角速度追従誤差の積分 対 出口の |e_y|:"
              f" 相関 {c:+.3f}・傾き {sl:+.4f} m/(rad)")
        print(f"      ⇒ {'e_y は角速度の遅れ由来と整合' if abs(c) >= 0.3 else '🔴 相関が弱い — 別の機構を疑う'}")
    else:
        print("\n【D2】判定不能（円弧の標本が足りない）")

    out = Path(args.out or (REPO_ROOT / "outputs" / "exp_016_diagonal" / "016h1"
                            / f"d1d2_sf{args.safety:g}_v{args.v_diag:g}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        git_rev=git_rev(), maze_dir=args.maze_dir, safety=args.safety,
        v_diag=args.v_diag, control_dt=dt, t_delay_assumed=2 * dt,
        L_c=L_C_CLOTHOID_M, rows=rows,
        d2_pairs=[[float(x), float(y)] for x, y in pairs]),
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
