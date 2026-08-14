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
    """一次遅れの時定数（**63.2% 到達時刻**で読む）。"""
    target = 0.632 * w_final
    idx = np.where((t > t0) & (w >= target))[0]
    if not len(idx):
        return float("nan")
    return float(t[idx[0]] - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maze", default="competition/mazes/design_v4/maze_41003.npz")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016f" / "tau_w.json"))
    args = ap.parse_args()

    params = RobotParams()
    ev = CompetitionEvaluator(maze_dir=str(Path(args.maze).parent), out_dir="/tmp/x")
    xml, seed, v, h, W, H = ev._load_or_build_xml(Path(args.maze))
    print(f"内側ループの時定数の同定（設計帯 {Path(args.maze).stem}）")
    print(f"{'v [m/s]':>8}{'ω step [rad/s]':>16}{'ω 定常':>10}{'τ_w [s]':>10}   定常の一致")
    rows = []
    for v_cmd in (0.0,):                 # **その場旋回**（前進すると壁に当たる）
        for omega_step in (1.0, 2.0, 4.0, 6.0):
            sim = MouseSim(str(xml), params=params)
            sim.full_reset(cell=(0, 0), heading_deg=90.0)
            pol = SlalomPolicy()
            pol.bind_sim(sim)
            pol.bind_maze(v, h)
            pol.on_maze_start(dict(width=16, height=16))
            rec, t0 = run_step(sim, pol, v_cmd, omega_step)
            t, wz = rec[:, 0] - rec[0, 0], rec[:, 2]
            tail = wz[int(len(wz) * 0.85):]
            w_final = float(np.median(tail))
            tau = fit_tau(t, wz, t0, w_final)
            ok = abs(w_final - omega_step) / max(omega_step, 1e-9) < 0.15
            rows.append(dict(v=v_cmd, omega_step=omega_step,
                             omega_final=w_final, tau_w_s=tau, steady_ok=bool(ok)))
            print(f"{v_cmd:>8.2f}{omega_step:>16.1f}{w_final:>10.3f}{tau:>10.4f}"
                  f"   {'○' if ok else '**× 定常が指令と一致しない → 採用しない**'}")
    taus = np.array([r["tau_w_s"] for r in rows
                     if np.isfinite(r["tau_w_s"]) and r["steady_ok"]])
    if not len(taus):
        print("\n**定常が指令に一致した条件が 1 つも無い。同定は失敗である。**")
        return 1
    print(f"\n**τ_w = {np.median(taus):.4f} s**"
          f"（範囲 {taus.min():.4f}〜{taus.max():.4f}・**採用 n={len(taus)}**／全 {len(rows)}）")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(git_rev=git_rev(), maze=str(args.maze), rows=rows,
                   tau_w_median_s=float(np.median(taus))),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
