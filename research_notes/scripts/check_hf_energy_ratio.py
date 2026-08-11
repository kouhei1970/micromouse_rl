"""
research_notes/scripts/check_hf_energy_ratio.py
===============================================
**符号反転数に代わる一次指標**として、「車輪が追従できない帯域にある指令成分の
エネルギー比」を測る。

## なぜ符号反転数では駄目か（2026-08-11 教授指摘）

平均 0 の対称なガウス方策では、隣接ステップで符号が変わる確率はちょうど 1/2 なので
**反転率は σ によらず 50 回/s**（実測 46.5〜48.4、n=6 で確認済み）。つまり

- 平滑な軌道の周りで **±0.001 の微小な揺らぎ**をしている方策 → 反転 50 回/s
- **±1.0 の全振幅で毎ステップ叩きつけている**方策 → 反転 50 回/s

が**同じ値になる**。符号反転数は**振幅を一切見ていない（scale-free な）指標**である。
実機で駆動系を痛めるのは反転の回数ではなく交流電流の大きさであり、
I²R 損失もギヤの打撃も**振幅の 2 乗**に比例する。

## 測る量

車輪ジョイントの時定数 τ_wheel = (I_w + N²J_rotor)/(N²K_tK_e/R + b) = 20.6 ms
（遮断 7.72 Hz）。**この帯域より上の指令成分は速度変化にならず、損失になる。**

低域通過成分 ā を**物理から決めた α** で作る:

    α = exp(−Δt/τ_wheel) = exp(−10 ms / 20.6 ms) = 0.616
    ā_t = α·ā_(t−1) + (1 − α)·a_t

高周波成分のエネルギー比（無次元。指令の全振幅 1.0 に対する実効値の比）:

    HF比 = sqrt( E‖a_t − ā_t‖² / 2 ) / 1.0        （2 は行動の次元数）

無次元なので方策間・条件間で直接比較できる。

**注**: 案 3 の罰に使う α=0.5 は「舵 10 Hz と振動 37 Hz の対数軸中間」というヒューリスティック
由来。本スクリプトの α=0.616 は**車輪ジョイントの遮断そのもの**から決めており、
別の量である（罰の設計値と評価の物差しは独立でよい）。比較のため両方を出す。

使い方:
    .venv/bin/python research_notes/scripts/check_hf_energy_ratio.py
    .venv/bin/python research_notes/scripts/check_hf_energy_ratio.py --n-trials 5
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from mouse.corridor_env import CorridorEnv  # noqa: E402
from mouse.corridor_eval import DEFAULT_COURSE_DIR, _trial_seed  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

# 完走できた方策のみ（完走走行が 0 本の条件は測っても意味がない）
MODELS = [
    ("k=0",       0, "models/exp_006_control_k0.zip"),
    ("k=0",       1, "models/exp_006c_seed1.zip"),
    ("k=0",       2, "models/exp_006c_seed2.zip"),
    ("k=1e-4",    1, "models/exp_006c_k1e-4_seed1.zip"),
    ("k=1e-4",    2, "models/exp_006c_k1e-4_seed2.zip"),
    ("k=1e-3",    1, "models/exp_006c_k1e-3_seed1.zip"),
    ("案3 k=2e-3", 1, "models/exp_006d_hp_k2e-3_seed1.zip"),
    ("案3 k=2e-3", 2, "models/exp_006d_hp_k2e-3_seed2.zip"),
    ("案3 k=2e-3", 3, "models/exp_006d_hp_k2e-3_seed3.zip"),
    ("案3 k=5e-3", 1, "models/exp_006d_hp_k5e-3_seed1.zip"),
    ("案3 k=5e-3", 2, "models/exp_006d_hp_k5e-3_seed2.zip"),
]


def wheel_tau() -> float:
    """車輪ジョイントの時定数 [s]（docs/MODEL_VERIFICATION_PLAN.md §4.2 の量から）。"""
    p = RobotParams()
    armature = p.gear_ratio ** 2 * p.rotor_inertia
    I_w = 0.5 * p.mass_wheel * p.wheel_radius ** 2
    b_elec = p.gear_ratio ** 2 * p.motor_Kt * p.motor_Ke / p.motor_R
    return (I_w + armature) / (b_elec + p.wheel_damping)


def hf_ratio(actions: np.ndarray, alpha: float) -> float:
    """高周波成分の実効値／全振幅。actions は (T, 2)、値域 [-1, 1]。"""
    bar = np.zeros(2)
    acc = 0.0
    for a in actions:
        bar = alpha * bar + (1.0 - alpha) * a
        d = a - bar
        acc += float(np.dot(d, d))
    return math.sqrt(acc / len(actions) / 2.0)


def sign_flip_rate(actions: np.ndarray, dt: float) -> float:
    """左右平均の符号反転率 [回/s]（既存指標との照合用）。"""
    s = np.sign(actions)
    flips = np.sum(s[1:] * s[:-1] < 0, axis=0)
    return float(np.mean(flips) / (len(actions) * dt))


def rollout(model, env, tseed):
    obs, _ = env.reset(seed=tseed)
    acts, terminated = [], False
    goal = False
    for _ in range(env._max_steps + 1):
        a, _ = model.predict(obs, deterministic=True)
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        acts.append(a)
        obs, _r, term, trunc, info = env.step(a)
        if term or trunc:
            terminated = term
            goal = bool(info.get("goal", False))
            break
    no_contact_complete = bool(goal and not info.get("collision", False))
    return np.array(acts), no_contact_complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=3,
                    help="コースあたり試行数（gate 帯 20 コース）")
    ap.add_argument("--out", type=str, default="outputs/hf_energy_ratio.json")
    args = ap.parse_args()

    p = RobotParams()
    tau = wheel_tau()
    alpha_phys = math.exp(-p.control_dt / tau)
    print(f"車輪ジョイント時定数 τ = {tau * 1000:.1f} ms"
          f"（遮断 {1 / (2 * math.pi * tau):.2f} Hz）")
    print(f"物理から決めた α = exp(−Δt/τ) = {alpha_phys:.3f}")
    print(f"（比較用に案 3 の設計値 α = 0.5 でも出す）\n")

    course_seeds = sorted(int(np.load(pp)["seed"])
                          for pp in Path(DEFAULT_COURSE_DIR).glob("corridor_*.npz"))

    rows = []
    for label, seed, path in MODELS:
        mp = REPO_ROOT / path
        if not mp.exists():
            print(f"[skip] {path} が無い")
            continue
        model = PPO.load(str(mp), device="cpu")
        hf_p, hf_5, flips, n_ok = [], [], [], 0
        for cs in course_seeds:
            env = CorridorEnv(course_dir=DEFAULT_COURSE_DIR, course_seeds=[cs],
                              max_cache=2, gamma=0.995, obs_dist_diff=True)
            for t in range(args.n_trials):
                acts, ok = rollout(model, env, _trial_seed(0, cs, t))
                if not ok:          # **完走走行のみ**を集計する
                    continue
                n_ok += 1
                hf_p.append(hf_ratio(acts, alpha_phys))
                hf_5.append(hf_ratio(acts, 0.5))
                flips.append(sign_flip_rate(acts, p.control_dt))
            env.close()
        if n_ok == 0:
            print(f"[skip] {label} seed={seed}: 完走走行が 0 本")
            continue
        rows.append(dict(label=label, seed=seed, n_success=n_ok,
                         hf_phys=float(np.mean(hf_p)), hf_a05=float(np.mean(hf_5)),
                         flip=float(np.mean(flips))))
        print(f"[done] {label} seed={seed}: 完走 {n_ok} 本, "
              f"HF比(α={alpha_phys:.3f}) {np.mean(hf_p):.4f}, 反転 {np.mean(flips):.1f}",
              flush=True)

    rows.sort(key=lambda r: r["hf_phys"])
    print("\n" + "=" * 88)
    print(f"高周波エネルギー比（gate 帯 20 コース ×{args.n_trials} 試行、**完走走行のみ**）")
    print("=" * 88)
    print(f"{'条件':<14}{'seed':>5}{'完走n':>7}{'HF比(α=0.616)':>16}"
          f"{'HF比(α=0.5)':>14}{'反転[回/s]':>12}")
    for r in rows:
        print(f"{r['label']:<14}{r['seed']:>5}{r['n_success']:>7}"
              f"{r['hf_phys']:>16.4f}{r['hf_a05']:>14.4f}{r['flip']:>12.1f}")

    # 指標を替える価値の判定: 反転数で 2.7 倍だった 2 本が、エネルギー比では何倍か
    def find(label, seed):
        return next((r for r in rows if r["label"] == label and r["seed"] == seed), None)

    a, b = find("k=0", 0), find("k=0", 2)
    if a and b:
        print("\n" + "=" * 88)
        print("指標を替える価値の判定（教授指示）")
        print("=" * 88)
        print(f"  exp_005 相当（k=0 seed0）: 反転 {a['flip']:.1f} 回/s, "
              f"HF比 {a['hf_phys']:.4f}")
        print(f"  最良（k=0 seed2）        : 反転 {b['flip']:.1f} 回/s, "
              f"HF比 {b['hf_phys']:.4f}")
        print(f"  → 反転数の比   {a['flip'] / b['flip']:.2f} 倍")
        print(f"  → エネルギー比 {a['hf_phys'] / b['hf_phys']:.2f} 倍")
        print("\n  比が同程度なら指標を替える価値は小さい。"
              "大きく開くなら反転数が現象を潰していたことになる。")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(tau_wheel_s=tau, alpha_phys=alpha_phys,
                       n_trials_per_course=args.n_trials, rows=rows),
                  f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
