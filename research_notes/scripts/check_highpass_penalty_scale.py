"""
research_notes/scripts/check_highpass_penalty_scale.py
======================================================
案 3（`−k‖a_t − ā_t‖²`）の**罰の大きさ**を exp_005 の方策で実測する。

`design_highpass.md` §4 は α=0.5 で ‖a−ā‖² の平均 = 0.483 と記録しているが、
これは前任者が学習を回さずにフィルタを後適用して求めた値で、**ā の規約
（ā_t が a_t を取り込んだ後の値かどうか）が実装と一致しているか**が確認されていない。

    実装（mouse/corridor_env.py）: ā_t = α·ā_(t−1) + (1−α)·a_t  →  a_t − ā_t = α·(a_t − ā_(t−1))

規約が 1 つずれると罰が α² = 1/4 倍（または 4 倍）変わり、k の選定がそのままずれる。
本スクリプトは**実装と同じ規約**で測り直し、design_highpass.md の値と照合する。

k の選定はこの値に直接依存する（taskB_return_ordering.md §4 の閾値 c を罰の平均で割る）。

使い方:
    .venv/bin/python research_notes/scripts/check_highpass_penalty_scale.py
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from mouse.corridor_env import CorridorEnv  # noqa: E402
from mouse.corridor_eval import VALIDATION_COURSE_DIR  # noqa: E402

MODEL = REPO_ROOT / "models/exp_005_collision_penalty.zip"
COURSE_SEEDS = [5000, 5001, 5002]     # design_highpass.md と同じ「3 コース」
ALPHAS = [0.3, 0.5, 0.8]
DESIGN_VALUES = {0.3: 0.222, 0.5: 0.483, 0.8: 0.918}   # design_highpass.md §4
DESIGN_DA2 = 3.873                                      # 同 §4（‖Δa‖² の平均）


def rollout_actions(model, course_seed: int, max_steps: int = 2000):
    """1 コースを決定論的に走らせ、行動列を返す。"""
    env = CorridorEnv(course_dir=VALIDATION_COURSE_DIR, course_seeds=[course_seed],
                      gamma=0.995, obs_dist_diff=True, potential_offset=True,
                      collision_penalty=-1.0)
    obs, _ = env.reset(seed=0)
    acts = []
    for _ in range(max_steps):
        a, _ = model.predict(obs, deterministic=True)
        acts.append(np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0))
        obs, _r, term, trunc, _info = env.step(a)
        if term or trunc:
            break
    env.close()
    return np.array(acts)


def mean_da2(acts: np.ndarray) -> float:
    """E‖a_t − a_(t−1)‖²。reset 直後の前回行動は 0（実装と同じ）。"""
    prev = np.zeros(2)
    tot = 0.0
    for a in acts:
        d = a - prev
        tot += float(np.dot(d, d))
        prev = a
    return tot / len(acts)


def mean_hp2(acts: np.ndarray, alpha: float) -> float:
    """E‖a_t − ā_t‖²。ā_t = α·ā_(t−1) + (1−α)·a_t、ā_(−1) = 0（実装と同じ規約）。"""
    bar = np.zeros(2)
    tot = 0.0
    for a in acts:
        bar = alpha * bar + (1.0 - alpha) * a
        hp = a - bar
        tot += float(np.dot(hp, hp))
    return tot / len(acts)


def mean_hp2_prev_convention(acts: np.ndarray, alpha: float) -> float:
    """対立規約: 罰に ā_(t−1)（a_t を取り込む**前**）を使った場合。

    a_t − ā_(t−1) は上の 1/α 倍になるので、罰は 1/α² 倍（α=0.5 なら 4 倍）。
    どちらの規約で 0.483 が出たのかを判別するために併記する。
    """
    bar = np.zeros(2)
    tot = 0.0
    for a in acts:
        hp = a - bar
        tot += float(np.dot(hp, hp))
        bar = alpha * bar + (1.0 - alpha) * a
    return tot / len(acts)


def main():
    if not MODEL.exists():
        print(f"[ERROR] {MODEL} がありません")
        return 1
    model = PPO.load(str(MODEL), device="cpu")

    all_acts = []
    print("=== exp_005 の方策を検証帯 3 コースで決定論的に走らせる ===")
    for cs in COURSE_SEEDS:
        acts = rollout_actions(model, cs)
        all_acts.append(acts)
        print(f"  course_seed={cs}: {len(acts)} ステップ")
    acts = np.concatenate(all_acts, axis=0)
    n = len(acts)
    print(f"  合計 {n} ステップ（design_highpass.md は 780 ステップと記載）\n")

    da2 = mean_da2(acts)
    print("=== ‖Δa‖²（exp_006 が罰していた量） ===")
    print(f"  実測 {da2:.3f} / design_highpass.md {DESIGN_DA2:.3f} "
          f"（比 {da2 / DESIGN_DA2:.3f}）\n")

    print("=== ‖a − ā‖²（案 3 が罰する量）— 規約 2 通りを併記 ===")
    print(f"{'α':>6}{'実装の規約':>14}{'対立規約':>12}{'design_highpass':>18}"
          f"{'実装/設計':>12}")
    for alpha in ALPHAS:
        impl = mean_hp2(acts, alpha)
        prev = mean_hp2_prev_convention(acts, alpha)
        des = DESIGN_VALUES[alpha]
        print(f"{alpha:>6.1f}{impl:>14.3f}{prev:>12.3f}{des:>18.3f}{impl / des:>12.3f}")

    print("\n=== k の選定（taskB_return_ordering.md の閾値 c から逆算） ===")
    # 閾値 c は衝突位置・速度の仮定で動く。採用仮定（衝突 2.5 m・1.6 m/s）と
    # 最悪仮定（同 2.0 m/s）の 2 つを載せる。
    for label, c_th in (("採用仮定（衝突 2.5 m・1.6 m/s）", 5.96e-3),
                        ("最悪仮定（衝突 2.5 m・2.0 m/s）", 1.76e-3)):
        hp = mean_hp2(acts, 0.5)
        print(f"  {label}: 順序が崩れる k = {c_th / hp:.2e}"
              f"（第一候補 2.0e-3 に対する余裕 {c_th / hp / 2.0e-3:.1f} 倍）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
