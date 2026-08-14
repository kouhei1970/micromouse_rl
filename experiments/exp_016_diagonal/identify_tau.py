#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""内側ループの時定数 $\tau_w$ を**実測**する（exp_016-F カード §2-1）。

**推定しない。**前方注視の距離 $\tau_\text{la}$ はこの値から決めるので、
$\omega_\text{cmd} \to \omega_z$ の応答を測って一次遅れとして同定する。

**手順**: **その場旋回**（$v$=0）で $\omega_\text{cmd}$ にステップを与え、
ジャイロ（`privileged_velocity()` の $\omega_z$）の立ち上がりから時定数を読む。
**設計帯の 1 面だけを使う**（裁定 R40）。

⚠️ **最初は「前進しながら」測ろうとして失敗した。**旋回しながら前進すると
**0.2 秒ほどで壁に当たり、定常値が指令とまったく無関係になる**
（ω 指令 1.0 → 定常 0.136・2.0 → 0.011・4.0 → −0.063 rad/s）。
**その場旋回なら区画の中に収まる**（機体の外接円の半径 64 mm < 壁まで 84 mm）ので、
**壁に触れずに定常へ達する**。**定常値が指令に一致することを毎回確認する。**

使い方:
    .venv/bin/python experiments/exp_016_diagonal/identify_tau.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from competition.baseline_slalom import SlalomPolicy  # noqa: E402
from competition.reference_interp import ReferenceInterpMixin  # noqa: E402
from competition.velocity_loop import VelocityLoopMixin  # noqa: E402
from competition.evaluator import CompetitionEvaluator  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

from geometry import git_rev  # noqa: E402


def run_step(sim, pol, v_cmd, omega_step, t_pre=0.2, t_step=0.8):
    """前進しながら ω 指令にステップを与え、(t, ω_z) を記録する。"""
    dt = pol.control_dt
    rec = []
    n_pre, n_step = int(t_pre / dt), int(t_step / dt)
    for k in range(n_pre + n_step):
        omega = 0.0 if k < n_pre else omega_step
        vl, vr = pol._wheel_targets_to_voltage(v_cmd, omega, sim.observation())
        sim.step_control(vl, vr)
        _v, wz = sim.privileged_velocity()
        rec.append((sim.sim_time, omega, wz))
    return np.array(rec, dtype=float), n_pre * dt


def fit_tau(t, w, t0, w_final):
    """一次遅れの時定数（**63.2% 到達時刻**で読む）。

    ⚠️ **この推定量の分解能は制御周期そのもの**（10 ms）である。
    τ_w が 1〜4 周期の大きさなので、**分解能が値と同じ桁**になる。
    **カード E1 の許容差 ±20% は、この推定量では原理的に満たせない**
    （±10 ms は 0.015 s に対して ±67%）。→ `fit_tau_ls` を併記する。
    """
    target = 0.632 * w_final
    idx = np.where((t > t0) & (w >= target))[0]
    if not len(idx):
        return float("nan")
    return float(t[idx[0]] - t0)


def fit_tau_ls(t, w, t0, w_final):
    """**同じ量（一次遅れの時定数）を、過渡の全点から最小二乗で推定する**（2026-08-14 追加）。

    一次遅れ w(t) = w_final·(1 − exp(−(t−t0)/τ)) を線形化して
    **y = ln(1 − w/w_final) = −(t−t0)/τ** の傾きから τ を読む。
    **63.2% 到達時刻の推定量と定義は同じで、推定の仕方だけが違う**
    （**量を変えたのではない**。§9-17 の「同じ名前で別の量」を作らないため明記する）。
    """
    m = (t > t0) & (w > 0.05 * w_final) & (w < 0.90 * w_final)
    if m.sum() < 3 or w_final <= 0:
        return float("nan")
    y = np.log(np.clip(1.0 - w[m] / w_final, 1e-9, None))
    x = t[m] - t0
    slope = float(np.polyfit(x, y, 1)[0])
    return float(-1.0 / slope) if slope < 0 else float("nan")


class _StepPolicy(ReferenceInterpMixin, VelocityLoopMixin, SlalomPolicy):
    """**新既定（F0＋F0-b 系）**でも同じ試験を回すための混ぜ込み（2026-08-14 追加）。

    **既定パラメータでは親へそのまま委譲する**ので、旧既定の測定と 1 ビットも変わらない
    （`tests/test_velocity_loop.py`・`tests/test_reference_interp.py` が検査済み）。
    """


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k-acc-ff", type=float, default=0.0,
                    help="F0 の加速度前置補償（新既定は 1.0）")
    ap.add_argument("--ref-interp", action="store_true",
                    help="F0-b の参照の弧長内挿（新既定は有効）")
    ap.add_argument("--maze", default="competition/mazes/design_v4/maze_41003.npz")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016f" / "tau_w.json"))
    args = ap.parse_args()

    params = RobotParams()
    ev = CompetitionEvaluator(maze_dir=str(Path(args.maze).parent), out_dir="/tmp/x")
    xml, seed, v, h, W, H = ev._load_or_build_xml(Path(args.maze))
    print(f"内側ループの時定数の同定（設計帯 {Path(args.maze).stem}）")
    print(f"版: k_acc_ff = {args.k_acc_ff:g}（F0）／ref_interp = {args.ref_interp}（F0-b）")
    print(f"{'v [m/s]':>8}{'ω step [rad/s]':>16}{'ω 定常':>10}{'τ_w 63.2% [s]':>15}"
          f"{'τ_w 最小二乗 [s]':>18}   定常の一致")
    rows = []
    for v_cmd in (0.0,):                 # **その場旋回**（前進すると壁に当たる）
        for omega_step in (1.0, 2.0, 4.0, 6.0):
            sim = MouseSim(str(xml), params=params)
            sim.full_reset(cell=(0, 0), heading_deg=90.0)
            pol = _StepPolicy(k_acc_ff=args.k_acc_ff, ref_interp=args.ref_interp)
            pol.bind_sim(sim)
            pol.bind_maze(v, h)
            pol.on_maze_start(dict(width=16, height=16))
            rec, t0 = run_step(sim, pol, v_cmd, omega_step)
            t, wz = rec[:, 0] - rec[0, 0], rec[:, 2]
            tail = wz[int(len(wz) * 0.85):]
            w_final = float(np.median(tail))
            tau = fit_tau(t, wz, t0, w_final)
            tau_ls = fit_tau_ls(t, wz, t0, w_final)
            ok = abs(w_final - omega_step) / max(omega_step, 1e-9) < 0.15
            rows.append(dict(v=v_cmd, omega_step=omega_step,
                             omega_final=w_final, tau_w_s=tau, tau_w_ls_s=tau_ls,
                             steady_ok=bool(ok)))
            print(f"{v_cmd:>8.2f}{omega_step:>16.1f}{w_final:>10.3f}{tau:>15.4f}"
                  f"{tau_ls:>18.4f}"
                  f"   {'○' if ok else '**× 定常が指令と一致しない → 採用しない**'}")
    taus = np.array([r["tau_w_s"] for r in rows
                     if np.isfinite(r["tau_w_s"]) and r["steady_ok"]])
    if not len(taus):
        print("\n**定常が指令に一致した条件が 1 つも無い。同定は失敗である。**")
        return 1
    taus_ls = np.array([r["tau_w_ls_s"] for r in rows
                        if np.isfinite(r["tau_w_ls_s"]) and r["steady_ok"]])
    print(f"\n**τ_w = {np.median(taus):.4f} s（63.2% 到達）**"
          f"（範囲 {taus.min():.4f}〜{taus.max():.4f}・**採用 n={len(taus)}**／全 {len(rows)}）")
    if len(taus_ls):
        print(f"**τ_w = {np.median(taus_ls):.4f} s（最小二乗・同じ量の別推定）**"
              f"（範囲 {taus_ls.min():.4f}〜{taus_ls.max():.4f}・n={len(taus_ls)}）")
    print("⚠️ 63.2% 到達の推定量は**分解能が制御周期そのもの**（10 ms）である。"
          "**τ_la はより分解能の高い最小二乗の値から決める。**")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(git_rev=git_rev(), maze=str(args.maze), rows=rows,
                   k_acc_ff=float(args.k_acc_ff), ref_interp=bool(args.ref_interp),
                   tau_w_median_s=float(np.median(taus)),
                   tau_w_ls_median_s=(float(np.median(taus_ls)) if len(taus_ls) else None)),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
