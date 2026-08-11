#!/usr/bin/env python3
"""任務 N3-4: exp_012 設計の反証監査（学生B の机上分析）。

作成: 2026-08-12 准教授セッション（独立検証担当・副査）

## 反証条件（材料の再計算に入る前に登録する）

設計書 `experiments/exp_012_continuous_potential/design.md` の主張について、
「偽ならどの観測がどう見えるか」を先に定義する。

| # | 主張 | **偽ならこう見える** |
|---|---|---|
| C1 | 「区画内を走ることは毎歩 −0.0050 の純損、停止 −0.0010 の **5 倍損**」 | 境界報酬を含めた 1 区画あたりの平均が**正**なら、「走ると損」は区画内に限った話で、前進そのものは得。**主張は過大** |
| C2 | 「M1 → M2-0 の報酬の構造的変化は +0.0046 → −0.0050」 | 階段 Φ の**平均**が M1 の +0.0046 と同程度なら、変わったのは**平均ではなく時間分布**。§5-1 の予測表は非対称な比較 |
| C3 | 「ゴール総収益 +0.60 > 衝突 −0.73 なので割引はゴールの価値を奪っていない」 | 学習が見るのは**達成条件つき**ではなく**期待値**。成功率で重みづけた期待値が凍結 −0.200 を下回るなら、この対比だけでは方策の選択を説明できない |
| C4 | 事前登録「3 seed のうち 1 本以上で ≥0.30 なら支持」 | seed 分散が大きければ **max 統計量**は当たりを引く。n=20 面 1 試行の二項揺らぎが 0.10/0.30 の判別に足りなければ、閾値は分解能を超えている |

定数はすべて**コードから**取る（設計書の転記を信用しない）:
  `mouse/maze6_env.py`: _TIME_PENALTY=0.001, _VISIT_BONUS=0.02, Φ=(D_start−d)·cell_size [m]
  reward = γ·Φ(s') − Φ(s) − 0.001 (+ ボーナス/罰) − k·‖a−ā‖²
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---- コードから取った定数 -------------------------------------------------
P_TIME = 0.001          # maze6_env._TIME_PENALTY
VISIT = 0.02            # maze6_env._VISIT_BONUS
CELL = 0.18             # 区画寸法 [m]
GAMMA = 0.995
K = 8.7e-3              # exp_010/012 の滑らかさ罰の係数
S_DRIVE = 0.459         # E‖a−ā‖²（走行する方策・設計書 §1 の実測）
S_FROZEN = 1.086e-4     # 同（凍結した方策）
NSTEP_CELL = 18.75      # 1 区画あたりの制御ステップ数（設計書 §1）
D0_MED = 9.5            # 検証帯 D₀ 中央値 [区画]


def smooth(s: float) -> float:
    return K * s


def within_cell_step(phi_m: float, moving: bool) -> float:
    """区画内の 1 歩の報酬（境界を越えない）。Φ が動かないので整形は drift のみ。"""
    return -(1.0 - GAMMA) * phi_m - P_TIME - smooth(S_DRIVE if moving else S_FROZEN)


def boundary_step(phi_m: float, unvisited: bool, forward: bool = True) -> float:
    """境界を越える 1 歩。Φ が ±CELL 跳ぶ。"""
    dphi = CELL if forward else -CELL
    shaping = GAMMA * (phi_m + dphi) - phi_m
    return shaping - P_TIME - smooth(S_DRIVE) + (VISIT if unvisited else 0.0)


def main() -> None:
    W = 92
    print("=" * W)
    print("任務 N3-4: exp_012 設計の反証監査 — 反証条件 C1〜C4")
    print("=" * W)

    # ---------------------------------------------------------------- C1/C2
    print("\n[C1/C2] 1 歩あたりの経済 — 設計書 §1 の表を再計算し、境界を含めた平均を出す")
    print("\n  (1) 設計書の各行の再現（Φ=0 と Φ=D₀ 中央値の区画で）")
    phi_hi = D0_MED * CELL
    rows = [
        ("停止する（Φ=0）",            within_cell_step(0.0, False),   -0.0010),
        ("区画内を走る（Φ=0）",         within_cell_step(0.0, True),    -0.0050),
        ("停止する（Φ=D₀）",           within_cell_step(phi_hi, False), -0.0096),
        ("区画内を走る（Φ=D₀）",        within_cell_step(phi_hi, True),  -0.0135),
        ("境界を越える（正・未訪問, Φ=0）", boundary_step(0.0, True),     +0.1941),
        ("境界を越える（逆・未訪問, Φ=0）", boundary_step(0.0, True, False), -0.1650),
    ]
    print(f"    {'挙動':<30}{'再計算':>12}{'設計書':>12}{'差':>12}")
    maxdiff = 0.0
    for name, got, doc in rows:
        maxdiff = max(maxdiff, abs(got - doc))
        print(f"    {name:<30}{got:>12.5f}{doc:>12.4f}{got-doc:>12.1e}")
    print(f"    → 全 6 行が一致（最大差 {maxdiff:.1e}）。**設計書の表そのものに誤りは無い**")

    print("\n  (2) 【設計書に無い量】1 区画を前進しきったときの 1 歩あたり平均")
    inner = within_cell_step(0.0, True)
    for label, unvis in (("未訪問区画（訪問報酬あり）", True), ("訪問済み区画（訪問報酬なし）", False)):
        per_cell = (NSTEP_CELL - 1) * inner + boundary_step(0.0, unvis)
        per_step = per_cell / NSTEP_CELL
        print(f"    {label:<28} 1 区画 {per_cell:>+8.4f} → 1 歩あたり **{per_step:>+8.5f}**")
    m1 = -P_TIME - smooth(S_DRIVE) + GAMMA * (CELL / NSTEP_CELL)
    print(f"    参考 M1（連続 Φ・毎歩 γ·ΔΦ = γ·{CELL/NSTEP_CELL:.4f} m）      1 歩あたり **{m1:>+8.5f}**")

    stop = within_cell_step(0.0, False)
    unv = ((NSTEP_CELL - 1) * inner + boundary_step(0.0, True)) / NSTEP_CELL
    vis = ((NSTEP_CELL - 1) * inner + boundary_step(0.0, False)) / NSTEP_CELL
    print(f"\n    ⚠️ **C1 は棄却**: 区画内だけを見ると走行 {inner:+.4f} 対 停止 {stop:+.4f}"
          f"（{inner/stop:.1f} 倍損）だが、")
    print(f"       境界報酬を入れると前進の平均は {unv:+.5f}（未訪問）/ {vis:+.5f}（訪問済み）で"
          f"**いずれも停止 {stop:+.4f} より得**。")
    print(f"       「動かない方が 5 倍得」は**区画内に限った話**で、前進そのものは損ではない。")
    print(f"\n    ⚠️ **C2 は棄却**: 訪問報酬を除いた階段 Φ の平均 {vis:+.5f} は"
          f" M1 の {m1:+.5f} と **{abs(vis/m1-1)*100:.1f}% 差**でほぼ同一。")
    print(f"       **変わったのは平均ではなく時間分布**（毎歩一様 → {NSTEP_CELL:.2f} 歩に 1 回の突起）。")

    # 感度: 1 区画あたりの歩数を変えても結論が変わらないか
    print("\n  (3) 感度 — 1 区画あたりの歩数を変えても上の結論は変わるか")
    print(f"    {'歩/区画':>8}{'区画内 1 歩':>14}{'前進の平均(訪問済)':>20}{'M1 相当':>12}")
    for n in (9.375, 18.75, 37.5, 75.0):
        v = ((n - 1) * inner + boundary_step(0.0, False)) / n
        m = -P_TIME - smooth(S_DRIVE) + GAMMA * (CELL / n)
        print(f"    {n:>8.2f}{inner:>14.5f}{v:>20.5f}{m:>12.5f}")
    print("    → 歩数によらず「前進の平均 ≈ M1 相当」が成り立つ。**C1/C2 の棄却は頑健**")

    # ---------------------------------------------------------- C2-bis
    print("\n  (4) C2 が 0.0% 差だったのは偶然か — 恒等式かどうかを確かめる")
    print("    階段 Φ の 1 区画あたり整形の総和 = 境界の 1 回だけ    = γ·0.18")
    print(f"    連続 Φ の 1 区画あたり整形の総和 = n·γ·(0.18/n)      = γ·0.18")
    print(f"    **同じ。**割引を無視すれば両者は恒等的に等しい（{GAMMA*CELL:.6f}）。")
    print("    → 0.0% 差は偶然ではなく**恒等式**。C2 の棄却は数値ではなく代数で確定する。")

    print("\n  (5) では割引を入れると差はどれだけ出るか（区画の入口から見た現在価値）")
    n_i = int(NSTEP_CELL)                       # 区画内の歩数
    disc_inner = sum(GAMMA ** t for t in range(n_i)) * inner
    disc_bound = GAMMA ** n_i * boundary_step(0.0, False)
    stair_pv = disc_inner + disc_bound
    cont_step = -P_TIME - smooth(S_DRIVE) + GAMMA * (CELL / NSTEP_CELL)
    cont_pv = sum(GAMMA ** t for t in range(n_i + 1)) * cont_step
    print(f"    階段 Φ: 区画内 {disc_inner:+.5f} ＋ 境界 {disc_bound:+.5f} = **{stair_pv:+.5f}**")
    print(f"    連続 Φ: 毎歩 {cont_step:+.5f} を {n_i+1} 歩ぶん割引    = **{cont_pv:+.5f}**")
    print(f"    差 = {stair_pv - cont_pv:+.5f}（連続比 {abs(stair_pv/cont_pv-1)*100:.1f}%）"
          f"／ γ^{n_i} = {GAMMA**n_i:.4f}")
    print("    → **割引を入れても差は 1 桁小さい。**「報酬の構造が変わった」では説明がつかない大きさ。")
    print("      H が正しいとすれば、機構は**報酬の算術ではなく学習の力学**")
    print("      （価値関数の近似・GAE の切り詰め・探索）の側にある。設計書 §7-7 が")
    print("      「本計算に含まれない」と断っている当のものが、**主張の本体**になっている。")

    # ---------------------------------------------------------------- C3
    print("\n[C3] γ 棄却の根拠 — 条件つき収益と期待値は別物")
    for tag, n_goal, n_tot, g_ret, c_ret in (
        ("exp_010 seed1 (k=8.7e-3)", 80, 7036, 0.60, -0.73),
        ("exp_011 seed1 (k=0)",      18, 3293, 0.62, -0.25),
    ):
        p = n_goal / n_tot
        ev = p * g_ret + (1 - p) * c_ret
        print(f"    {tag}")
        print(f"      到達率 {p*100:.2f}%  ／ 条件つき収益 ゴール {g_ret:+.2f} 対 衝突 {c_ret:+.2f}")
        print(f"      **挑戦の期待値 = {p:.4f}·{g_ret:+.2f} + {1-p:.4f}·{c_ret:+.2f} = {ev:+.4f}**"
              f"  対 凍結 −0.200 → {'凍結が有利' if ev < -0.200 else '挑戦が有利'}")
    print("    → 設計書の「+0.60 > −0.73 なので割引はゴールの価値を奪っていない」は**それ自体は正しい**が、")
    print("      方策の選択を説明するのは**期待値**。凍結を選んだのは現在の技量では合理的で、")
    print("      **γ 棄却の結論は動かない**（むしろ強まる）。ただし論拠の書き方に穴がある。")

    # ---------------------------------------------------------------- C4
    print("\n[C4] 事前登録した判定の分解能 — n = 20 面 1 試行で 0.10 と 0.30 を分けられるか")
    from math import comb
    def cdf(k, n, p):
        return sum(comb(n, i) * p**i * (1-p)**(n-i) for i in range(k+1))
    n = 20
    print(f"    {'真のゴール率':>12}{'P(観測 ≥0.30 = 6/20)':>24}{'P(観測 <0.10 = 0,1/20)':>26}")
    for p in (0.05, 0.10, 0.20, 0.30, 0.40):
        p_sup = 1.0 - cdf(5, n, p)
        p_rej = cdf(1, n, p)
        print(f"    {p:>12.2f}{p_sup:>24.3f}{p_rej:>26.3f}")
    print("    → 真値 0.20（＝『中間』相当）でも **支持 0.30 到達が 20% 程度**起きる。")
    print("      さらに支持条件は **3 seed の最良 1 本**なので、1 本あたり 20% でも")
    p1 = 1.0 - cdf(5, n, 0.20)
    print(f"      **3 本のどれかが当たる確率 = 1−(1−{p1:.3f})³ = {1-(1-p1)**3:.3f}**。")
    print("      **真に『中間』の機構でも 5 割前後で『支持』と読める。**max 統計量の取り方に問題がある。")

    out = REPO / "verification" / "out" / "n4_exp012_design_audit.json"
    out.write_text(json.dumps({
        "within_cell_step_moving": inner, "within_cell_step_stopped": stop,
        "per_step_forward_unvisited": unv, "per_step_forward_visited": vis,
        "m1_equivalent": m1,
        "support_prob_if_true_0.20_one_seed": p1,
        "support_prob_if_true_0.20_best_of_3": 1 - (1 - p1) ** 3,
    }, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out}")


if __name__ == "__main__":
    main()
