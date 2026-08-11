"""
research_notes/scripts/check_m2_return_ordering.py
==================================================
**M2-0 の割引後総収益を、実際のエピソード長で再計算する**（2026-08-11）。

`docs/RESEARCH_PLAN.md` §9-11 の規律
（**報酬をスケールの違う課題へ転用するときは総収益を再計算する**）の最初の適用。

## なぜ必要か

`experiments/m2_design.md` の検算は **2 つの前提**の上にある:

1. D₀ = 1.8 m・速度 0.96 m/s → **エピソード 188 歩**
2. **行動の滑らかさへの罰が無い**（案 3 は当時まだ存在しなかった）

**どちらも実際とは違う**:

1. 実際の 6x6（loop）は最短 6〜18 歩 ＝ D₀ 1.08〜3.24 m → **112〜337 歩**
2. **M1 で確定した k=8.7e-3・α=0.5 をそのまま持ち込んだ**

**罰は毎ステップ入るので、エピソードが長いほど総量が増える。**M1（廊下、D₀=2.52 m、
263 歩）で余裕 0.066 だった罰が、M2 でどうなるかは**計算し直さないと分からない**。

使い方:
    .venv/bin/python research_notes/scripts/check_m2_return_ordering.py
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GAMMA = 0.995
DT = 0.01
TIME_PENALTY = 0.001
GOAL_BONUS = 1.0
COLLISION_PENALTY = -1.0
VISIT_BONUS = 0.02
CELL = 0.18
N_CELLS_TOTAL = 36
TIME_LIMIT_STEPS = 6000

MEAN_HP2 = 0.483        # ‖a − ā‖² の実測平均（α=0.5、M1 の方策）
K_M1 = 8.7e-3           # M1 で確定した係数


def S(T):
    return (1.0 - GAMMA ** T) / (1.0 - GAMMA)


def visit_discounted(n_cells, steps_per_cell):
    """訪問報酬の割引後総和。i 区画目は i·steps_per_cell 歩目に入ると仮定。"""
    return VISIT_BONUS * sum(GAMMA ** (i * steps_per_cell) for i in range(n_cells))


def main():
    v = 0.96                       # 完走時の平均前進速度 [m/s]（M1 実測）
    steps_per_cell = CELL / (v * DT)      # 18.75 歩/区画

    print("=" * 82)
    print("§0 前提の照合 — m2_design.md の検算は何を仮定していたか")
    print("=" * 82)
    print("  m2_design.md: D₀ = 1.8 m・速度 0.96 m/s → エピソード 188 歩、**罰なし**")
    print("  実際の 6x6(loop): 最短 6〜18 歩 ＝ D₀ 1.08〜3.24 m → 112〜337 歩")
    print(f"  実際に投入した罰: k = {K_M1:g}（α=0.5、M1 で確定）")
    print(f"  → 1 ステップあたりの罰 c = k·E‖a−ā‖² = {K_M1 * MEAN_HP2:.2e}"
          f"（時間罰 {TIME_PENALTY} の {K_M1 * MEAN_HP2 / TIME_PENALTY:.1f} 倍）")
    print(f"  γ = {GAMMA} の実効地平 = 1/(1−γ) = {1/(1-GAMMA):.0f} 歩 = {1/(1-GAMMA)*DT:.0f} s")

    print("\n" + "=" * 82)
    print("§1 罰の有無での比較（最短歩数ごと）")
    print("=" * 82)
    print("  ゴール到達: 最短経路を速度 0.96 m/s で走る。訪問報酬は通った区画ぶん")
    print("  時間切れ: 6000 歩まで走り、全 36 区画を訪問（探索の上限）")
    print("  衝突: 最短の半分まで進んで衝突\n")
    hdr = (f"{'最短[歩]':>9}{'T_goal[歩]':>11}"
           f"{'ゴール(罰なし)':>16}{'ゴール(k=8.7e-3)':>18}"
           f"{'時間切れ(k)':>13}{'衝突(k)':>11}{'余裕(k)':>10}")
    print(hdr)
    rows = []
    for d in (6, 10, 14, 18):
        D0 = d * CELL
        Tg = d * steps_per_cell
        for c, tag in ((0.0, "none"), (K_M1 * MEAN_HP2, "k")):
            g = (GAMMA ** Tg * D0 - (TIME_PENALTY + c) * S(Tg)
                 + GAMMA ** (Tg - 1) * GOAL_BONUS + visit_discounted(d, steps_per_cell))
            if tag == "none":
                g_none = g
            else:
                g_k = g
        # 時間切れ: 6000 歩、全区画訪問、終端 Φ は残り距離ぶん残る（Φ_T = D₀ − d_final·CELL）
        c = K_M1 * MEAN_HP2
        to = (GAMMA ** TIME_LIMIT_STEPS * 0.0 - (TIME_PENALTY + c) * S(TIME_LIMIT_STEPS)
              + visit_discounted(N_CELLS_TOTAL, steps_per_cell))
        Tc = Tg / 2
        col = (GAMMA ** Tc * (D0 / 2) - (TIME_PENALTY + c) * S(Tc)
               + GAMMA ** (Tc - 1) * COLLISION_PENALTY
               + visit_discounted(max(d // 2, 1), steps_per_cell))
        margin = g_k - max(to, col)
        rows.append((d, Tg, g_none, g_k, to, col, margin))
        print(f"{d:>9}{Tg:>11.0f}{g_none:>16.3f}{g_k:>18.3f}{to:>13.3f}{col:>11.3f}"
              f"{margin:>10.3f}")

    print("\n  **罰なしの列が m2_design.md の想定、k の列が実際に回した条件。**")

    print("\n" + "=" * 82)
    print("§2 罰がゴールの価値をどれだけ食うか")
    print("=" * 82)
    print(f"{'最短[歩]':>9}{'罰の総量 c·S(T)':>18}{'ゴール(罰なし)':>16}"
          f"{'ゴール(k)':>12}{'減少率':>9}")
    for d, Tg, g_none, g_k, to, col, margin in rows:
        pen = K_M1 * MEAN_HP2 * S(Tg)
        print(f"{d:>9}{pen:>18.3f}{g_none:>16.3f}{g_k:>12.3f}"
              f"{(1 - g_k / g_none) if g_none else float('nan'):>9.0%}")

    print("\n" + "=" * 82)
    print("§3 順序が崩れない k の上限（M2-0 の実条件）")
    print("=" * 82)
    print("  ゴール > max(時間切れ, 衝突) を保つ最大の k を最短歩数ごとに求める")
    print(f"{'最短[歩]':>9}{'束縛':>12}{'閾値 c':>12}{'閾値 k':>12}"
          f"{'投入した k との比':>18}")
    for d in (6, 10, 14, 18):
        D0 = d * CELL
        Tg = d * steps_per_cell
        lo, hi = 0.0, 0.05
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            g = (GAMMA ** Tg * D0 - (TIME_PENALTY + mid) * S(Tg)
                 + GAMMA ** (Tg - 1) * GOAL_BONUS + visit_discounted(d, steps_per_cell))
            to = (-(TIME_PENALTY + mid) * S(TIME_LIMIT_STEPS)
                  + visit_discounted(N_CELLS_TOTAL, steps_per_cell))
            Tc = Tg / 2
            col = (GAMMA ** Tc * (D0 / 2) - (TIME_PENALTY + mid) * S(Tc)
                   + GAMMA ** (Tc - 1) * COLLISION_PENALTY
                   + visit_discounted(max(d // 2, 1), steps_per_cell))
            if g > max(to, col):
                lo = mid
            else:
                hi = mid
        c_th = lo
        k_th = c_th / MEAN_HP2
        # 束縛がどちらか
        g = (GAMMA ** Tg * D0 - (TIME_PENALTY + c_th) * S(Tg)
             + GAMMA ** (Tg - 1) * GOAL_BONUS + visit_discounted(d, steps_per_cell))
        to = (-(TIME_PENALTY + c_th) * S(TIME_LIMIT_STEPS)
              + visit_discounted(N_CELLS_TOTAL, steps_per_cell))
        Tc = Tg / 2
        col = (GAMMA ** Tc * (D0 / 2) - (TIME_PENALTY + c_th) * S(Tc)
               + GAMMA ** (Tc - 1) * COLLISION_PENALTY
               + visit_discounted(max(d // 2, 1), steps_per_cell))
        bind = "時間切れ" if to >= col else "衝突"
        print(f"{d:>9}{bind:>12}{c_th:>12.2e}{k_th:>12.2e}"
              f"{K_M1 / max(k_th, 1e-12):>17.1f} 倍")

    print("\n" + "=" * 82)
    print("§4 訪問報酬 0.02 の割引後の寄与（m2_design.md は割引前で検算していた）")
    print("=" * 82)
    print(f"  割引前: {VISIT_BONUS} × {N_CELLS_TOTAL} 区画 = "
          f"{VISIT_BONUS * N_CELLS_TOTAL:.2f}（ゴールボーナス 1.0 を下回る、が当時の根拠）")
    print(f"  割引後（{steps_per_cell:.1f} 歩/区画で順に訪問）: "
          f"{visit_discounted(N_CELLS_TOTAL, steps_per_cell):.3f}")
    print(f"  → 実効地平 {1/(1-GAMMA):.0f} 歩に対し 1 区画 {steps_per_cell:.1f} 歩なので、"
          f"**{1/(1-GAMMA)/steps_per_cell:.0f} 区画ぶんしか効かない**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
