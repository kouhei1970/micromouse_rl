"""
research_notes/scripts/check_motor_current.py
=============================================
**RMS 電流を連続定格と比べる**（2026-08-11 教授指示）。

## なぜ電流か

符号反転数も高周波エネルギー比も「指令の形」の指標であって、**害の実体ではない**。
害の実体は**熱と機械的打撃**である。指令の交流成分は、速度変化を生まないまま電流を
流して $I^2R$ の発熱になる。そして**発熱の限界は仕様書に書いてある** —
モータの連続定格である。

これが決まれば「実機に持っていけない」が**定量的な主張**になる。

## 計算

環境のモータモデル（`docs/MODEL_VERIFICATION_PLAN.md` §4.1）:

    τ_w = (N·K_t/R)·(V − K_e·N·ω_w) − b·ω_w − τ_c·sgn(ω_w)

したがってモータ電流は

    I = (V − K_e·N·ω_w) / R          （V: 印加電圧 [V]、ω_w: 車輪角速度 [rad/s]）

制御周期内で V は一定、ω_w は変化するので、**ステップ前後の ω の平均**を使う
（制御周期 10 ms に対し車輪の時定数 20.6 ms なので、この近似の誤差は小さい）。

**分解**: I_rms² = I_dc² + I_ac²。
- **I_dc**（平均電流）は正味のトルク＝仕事に対応する
- **I_ac**（交流成分）は**速度変化を生まないまま発熱だけを生む分**

## 連続定格

`docs/ROBOT_SPEC.md` §3 の連続定格トルク **1.16 mN·m**（モータ軸）より

    I_cont = 1.16e-3 / K_t = 1.16e-3 / 1.98e-3 = 0.586 A

**注意**: 連続定格は**連続運転**の値である。1 走行は約 2〜3 秒なので、熱時定数が
数十秒〜数分なら短時間の超過は許容される。**超過していた場合は「連続定格の何倍か」と
「走行時間」を併記し、断定しないこと。**

**バックラッシュ・打音は電流では捉えられない。**`docs/ROBOT_SPEC.md` に減速機の
バックラッシュの記載は無く（減速比 N=5 は比としてのみモデル化されている）、
この項目は評価できない。

使い方:
    .venv/bin/python research_notes/scripts/check_motor_current.py
"""
import argparse
import json
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

MODELS = [
    ("k=0",       0, "models/exp_006_control_k0.zip"),       # ＝ exp_005（最悪）
    ("k=0",       1, "models/exp_006c_seed1.zip"),
    ("k=0",       2, "models/exp_006c_seed2.zip"),           # k=0 の最良
    ("k=1e-4",    1, "models/exp_006c_k1e-4_seed1.zip"),
    ("k=1e-4",    2, "models/exp_006c_k1e-4_seed2.zip"),
    ("k=1e-3",    1, "models/exp_006c_k1e-3_seed1.zip"),
    ("案3 k=2e-3", 1, "models/exp_006d_hp_k2e-3_seed1.zip"),
    ("案3 k=2e-3", 2, "models/exp_006d_hp_k2e-3_seed2.zip"),
    ("案3 k=2e-3", 3, "models/exp_006d_hp_k2e-3_seed3.zip"),
    ("案3 k=5e-3", 1, "models/exp_006d_hp_k5e-3_seed1.zip"),
    ("案3 k=5e-3", 2, "models/exp_006d_hp_k5e-3_seed2.zip"),
    ("案3 k=8.7e-3", 1, "models/exp_006d_hp_k8.7e-3_seed1.zip"),
    ("案3 k=8.7e-3", 2, "models/exp_006d_hp_k8.7e-3_seed2.zip"),
    ("案3 k=8.7e-3", 3, "models/exp_006d_hp_k8.7e-3_seed3.zip"),
]

CONT_TORQUE_NM = 1.16e-3      # 連続定格トルク（モータ軸）[N·m]。docs/ROBOT_SPEC.md §3


def wheel_omegas(env):
    s = env.sim
    return (float(s.data.qvel[s._left_wheel_qvel_adr]),
            float(s.data.qvel[s._right_wheel_qvel_adr]))


def rollout_currents(ppo, env, p: RobotParams):
    """1 走行の左右モータ電流の時系列 [A] と、完走可否・走行時間を返す。"""
    obs = env._make_observation()
    cur, info = [], {}
    for _ in range(env._max_steps + 1):
        a, _ = ppo.predict(obs, deterministic=True)
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        v = a * p.voltage_limit
        w0 = np.array(wheel_omegas(env))
        obs, _r, term, trunc, info = env.step(a)
        w1 = np.array(wheel_omegas(env))
        # 制御周期内で V は一定、ω は前後の平均で代表させる
        cur.append((v - p.motor_Ke * p.gear_ratio * 0.5 * (w0 + w1)) / p.motor_R)
        if term or trunc:
            break
    ok = bool(info.get("goal", False) and not info.get("collision", False))
    return np.array(cur), ok, len(cur) * p.control_dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument("--out", type=str, default="outputs/motor_current.json")
    args = ap.parse_args()

    p = RobotParams()
    i_cont = CONT_TORQUE_NM / p.motor_Kt
    print(f"連続定格トルク {CONT_TORQUE_NM * 1e3:.2f} mN·m / K_t {p.motor_Kt:.3e} "
          f"→ **連続定格電流 I_cont = {i_cont:.3f} A**")
    print(f"巻線抵抗 R = {p.motor_R} Ω / 減速比 N = {p.gear_ratio:g} / "
          f"電圧上限 ±{p.voltage_limit} V")
    print(f"（V=±{p.voltage_limit} V・ω=0 のときの拘束電流 = "
          f"{p.voltage_limit / p.motor_R:.2f} A ＝ 定格の "
          f"{p.voltage_limit / p.motor_R / i_cont:.1f} 倍）\n")

    course_seeds = sorted(int(np.load(f)["seed"])
                          for f in Path(DEFAULT_COURSE_DIR).glob("corridor_*.npz"))

    rows = []
    for label, seed, path in MODELS:
        mp = REPO_ROOT / path
        if not mp.exists():
            continue
        ppo = PPO.load(str(mp), device="cpu")
        rms, dc, ac, durs = [], [], [], []
        for cs in course_seeds:
            env = CorridorEnv(course_dir=DEFAULT_COURSE_DIR, course_seeds=[cs],
                              max_cache=2, gamma=0.995, obs_dist_diff=True)
            for t in range(args.n_trials):
                env.reset(seed=_trial_seed(0, cs, t))
                cur, ok, dur = rollout_currents(ppo, env, p)
                if not ok:                       # **完走走行のみ**
                    continue
                # 左右をまとめて 1 本の指標にする（両モータの平均）
                r = np.sqrt((cur ** 2).mean(axis=0))       # (2,)
                d = cur.mean(axis=0)                        # (2,)
                rms.append(r.mean())
                dc.append(np.abs(d).mean())
                ac.append(np.sqrt(np.maximum(r ** 2 - d ** 2, 0.0)).mean())
                durs.append(dur)
            env.close()
        if not rms:
            print(f"[skip] {label} seed={seed}: 完走走行が 0 本")
            continue
        rows.append(dict(label=label, seed=seed, n=len(rms),
                         i_rms=float(np.mean(rms)), i_dc=float(np.mean(dc)),
                         i_ac=float(np.mean(ac)), duration_s=float(np.mean(durs)),
                         ratio=float(np.mean(rms)) / i_cont,
                         p_loss_w=float(np.mean(rms)) ** 2 * p.motor_R * 2))
        print(f"[done] {label} seed={seed}: I_rms {np.mean(rms):.3f} A "
              f"({np.mean(rms) / i_cont:.2f}×定格), I_ac {np.mean(ac):.3f} A", flush=True)

    rows.sort(key=lambda r: -r["i_rms"])
    print("\n" + "=" * 100)
    print(f"モータ電流（gate 帯 20 コース ×{args.n_trials} 試行、**完走走行のみ**、左右平均）")
    print("=" * 100)
    print(f"{'条件':<14}{'seed':>5}{'走行n':>7}{'走行時間[s]':>12}"
          f"{'I_rms[A]':>11}{'定格比':>9}{'I_dc[A]':>10}{'I_ac[A]':>10}"
          f"{'交流の寄与':>12}{'I²R[W]':>9}")
    for r in rows:
        ac_share = r["i_ac"] ** 2 / max(r["i_rms"] ** 2, 1e-12)
        print(f"{r['label']:<14}{r['seed']:>5}{r['n']:>7}{r['duration_s']:>12.2f}"
              f"{r['i_rms']:>11.3f}{r['ratio']:>9.2f}{r['i_dc']:>10.3f}"
              f"{r['i_ac']:>10.3f}{ac_share:>12.0%}{r['p_loss_w']:>9.2f}")

    print(f"\n  連続定格 I_cont = {i_cont:.3f} A（1 走行は約 "
          f"{np.mean([r['duration_s'] for r in rows]):.1f} s であり、"
          f"**連続運転ではない**）")
    print("  I_rms² = I_dc² + I_ac²。I_ac は速度変化を生まないまま発熱だけを生む分。")
    print("  I²R は左右 2 個ぶんの銅損の合計。")

    # ---- 熱の判定条件（値を推測せず、条件式だけ置く） -------------------
    print("\n" + "=" * 100)
    print("熱の判定 — ⚠️ 巻線の熱時定数 τ_th が仕様書に無いので、条件式だけ置く")
    print("=" * 100)
    print("  一次遅れの温度上昇: ΔT(t)/ΔT_cont = (I_rms/I_cont)²·(1 − e^(−t/τ_th))")
    print("  連続定格は「ΔT_cont に達する電流」なので、1 走行で ΔT_cont を超えない条件は")
    print("    (I_rms/I_cont)²·(1 − e^(−t/τ_th)) ≤ 1")
    print("  これを τ_th について解くと、**必要な熱時定数の下限**が出る:\n")
    print(f"{'条件':<14}{'seed':>5}{'定格比':>9}{'走行[s]':>9}"
          f"{'必要な τ_th [s]':>17}")
    for r in rows:
        ratio2 = r["ratio"] ** 2
        if ratio2 <= 1.0:
            need = "不要（定格内）"
            print(f"{r['label']:<14}{r['seed']:>5}{r['ratio']:>9.2f}"
                  f"{r['duration_s']:>9.2f}{need:>17}")
            continue
        x = 1.0 / ratio2                    # = 1 − e^(−t/τ)
        tau_need = r["duration_s"] / (-np.log(1.0 - x))
        print(f"{r['label']:<14}{r['seed']:>5}{r['ratio']:>9.2f}"
              f"{r['duration_s']:>9.2f}{tau_need:>17.1f}")
    print("\n  読み方: 実際の τ_th がこの値**以上**なら、その方策は 1 走行では焼けない。")
    print("  **τ_th は docs/ROBOT_SPEC.md にも docs/MODEL_VERIFICATION_PLAN.md にも無い。**")
    print("  データシート（EN_1717_SR_DFF.pdf）が手に入ったら、この 1 列と比べるだけで判定できる。")
    print("\n  ⚠️ 本計算が答えていないこと（将来課題）:")
    print("  - 公式ルールの持ち時間は **420 秒（7 分）・最大 5 走行**")
    print("    （docs/RESEARCH_PLAN.md §2。NTF クラシックマウス競技規定 3-6）。")
    print("    **電流が流れるのは最速走行の間だけではない。**探索走行・追加探索・帰還の")
    print("    すべてで流れ、学生A の測定では L0-a の内訳が探索 141 s ／追加探索 184 s ／")
    print("    帰還 38 s と**走行以外が大半**である。")
    print("    → 正しい問いは「1 走行で焼けないか」ではなく")
    print("      **「持ち時間 420 秒を通して焼けないか」**である可能性が高い（M3 以降で効く）。")
    print("  - 電流は**制御周期（10 ms）での代表値**であり、物理サブステップ 20 分割での")
    print("    積分ではない。**平滑化しているぶん真の RMS はこれより大きい**（変動を均すと")
    print("    2 乗平均は下がる）ので、**本表は過小評価側**であり結論の向きは変わらない。")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(i_cont_a=i_cont, cont_torque_nm=CONT_TORQUE_NM,
                       n_trials_per_course=args.n_trials, rows=rows),
                  f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
