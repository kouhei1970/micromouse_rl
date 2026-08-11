"""
research_notes/scripts/check_return_ordering.py
===============================================
タスク B: 行動の滑らかさへの罰を入れたときの**割引後の総収益の順序**を計算する。

`handover/student_b.md` に「新しい報酬項を入れる前に必ず割引後の総収益を代表条件で
計算し、ゴール > 探索 > 滞留 > 衝突 の順序を確かめる」とある手順を、exp_006（‖Δa‖²）と
案 3（‖a − ā‖²）について実施する。

## 報酬式（mouse/corridor_env.py 438〜448 行、potential_offset=True）

    r_t = γ·Φ(s_t) − Φ(s_{t−1}) − 0.001
          + 1.0                （ゴール到達。goal_reached が優先）
          + collision_penalty  （衝突・転倒。既定 −1.0）
          − k·‖a_t − a_{t−1}‖²

    Φ = D₀ − D（D は残り経路長 [m]、D₀ は経路全長）。開始時 D=D₀ なので Φ_0 = 0。

## 割引後の総収益（t=1 から始めて γ^{t−1} で割り引く）

ポテンシャル項は telescoping して γ^T·Φ_T − Φ_0 = γ^T·Φ_T。したがって

    G = γ^T·Φ_T − (0.001 + c)·S(T) + （終端ボーナス/罰）·γ^{T−1}
    S(T) = (1 − γ^T)/(1 − γ)      … 割引後の「実効ステップ数」
    c = k·E‖Δa‖²                   … 1 ステップあたりの滑らかさ罰の平均

**要点**: 罰は毎ステップほぼ一定量で入るので実効的な時間罰の増加と同じであり、
エピソードが長い条件（ゴール到達）ほど多く受ける。ゴールと衝突の総収益の差が
c·(S(T_goal) − S(T_collide)) だけ縮む。

使い方:
    .venv/bin/python research_notes/scripts/check_return_ordering.py
"""
import numpy as np

GAMMA = 0.995
DT = 0.01              # 制御周期 [s]
TIME_PENALTY = 0.001
GOAL_BONUS = 1.0
COLLISION_PENALTY = -1.0

# 実測の罰の大きさ（exp_005 の方策 3 コース 780 ステップ）
MEAN_DA2 = 3.873       # ‖a_t − a_(t−1)‖² の平均（exp_006 が罰していた量）
MEAN_HP2_A05 = 0.483   # ‖a_t − ā_t‖² の平均、α=0.5（案 3 が罰する量）


def S(T: float) -> float:
    """割引後の実効ステップ数 (1 − γ^T)/(1 − γ)。

    T は連続値で扱う（整数へ丸めると dG/dx が丸め由来のギザギザを持つため）。
    """
    return (1.0 - GAMMA ** T) / (1.0 - GAMMA)


def G_goal(T: float, D0: float, c: float) -> float:
    """ゴール到達（T ステップで D0 [m] を走り切る）の割引後総収益。"""
    return (GAMMA ** T) * D0 - (TIME_PENALTY + c) * S(T) + (GAMMA ** (T - 1)) * GOAL_BONUS


def G_collide(T: float, x: float, c: float) -> float:
    """距離 x [m] 進んだ地点で衝突（T ステップ）の割引後総収益。Φ_T = x。"""
    return (GAMMA ** T) * x - (TIME_PENALTY + c) * S(T) + (GAMMA ** (T - 1)) * COLLISION_PENALTY


def G_timeout(T: float, x: float, c: float) -> float:
    """時間切れ（T ステップ、到達距離 x [m]）。終端ボーナス・罰なし。

    滞留（その場に留まる）は x=0 の特別な場合。
    """
    return (GAMMA ** T) * x - (TIME_PENALTY + c) * S(T)


# ---------------------------------------------------------------------------
# §0 モデルの検証: exp_005 の card.md に記載の既知値を再現できるか
# ---------------------------------------------------------------------------
def verify_against_exp005():
    print("=" * 78)
    print("§0 モデル検証: exp_005 card.md の既知値を再現できるか（c=0）")
    print("=" * 78)
    D0 = 2.52
    T_MAX = 6000  # 時間切れ上限（滞留の代表値。card.md の −0.200 を再現する長さ）
    cases = [
        ("ゴール（300 歩 ＝ 0.84 m/s）", G_goal(300, D0, 0.0), 0.627),
        ("1.6 m 進んで衝突（100 歩 ＝ 1.6 m/s）", G_collide(100, 1.6, 0.0), 0.285),
        ("滞留・時間切れ（x=0）", G_timeout(T_MAX, 0.0, 0.0), -0.200),
    ]
    print(f"{'条件':<38}{'本計算':>10}{'card.md':>10}{'差':>10}")
    for name, got, want in cases:
        print(f"{name:<38}{got:>10.3f}{want:>10.3f}{got - want:>10.3f}")
    print("\n→ 一致すれば以降の計算は exp_004/005 と同じ土俵に乗っている。\n")


# ---------------------------------------------------------------------------
# §1 代表条件（仮定を明示する）
# ---------------------------------------------------------------------------
D0 = 2.52                 # 経路全長 [m]（検証帯の中央値。exp_005 card.md と同じ）
V_GOAL = 0.96             # 完走時の平均前進速度 [m/s]（exp_005 実測）
V_CRASH = 1.60            # 壁へ直進して突っ込むときの速度 [m/s]（exp_005 card.md の想定）
T_TIMEOUT = 6000          # 時間切れ上限 [ステップ]

T_GOAL = D0 / (V_GOAL * DT)                          # 262.5 歩
# 衝突する位置 [m]（複数点振る）。2.5 m は「ゴール直前で衝突」＝最も得な衝突の極限。
CRASH_POSITIONS = [0.3, 0.6, 1.26, 1.6, 2.0, 2.5]


def crash_steps(x: float, v: float) -> float:
    return max(1.0, x / (v * DT))


def main():
    verify_against_exp005()

    print("=" * 78)
    print("§1 置いた仮定（変えれば閾値も動く）")
    print("=" * 78)
    print(f"  γ = {GAMMA}, Δt = {DT} s, 時間罰 = {TIME_PENALTY}, "
          f"ゴール +{GOAL_BONUS}, 衝突 {COLLISION_PENALTY}")
    print(f"  経路全長 D₀ = {D0} m（検証帯の中央値）")
    print(f"  ゴール到達: 平均 {V_GOAL} m/s → T = {T_GOAL} 歩, S = {S(T_GOAL):.1f}")
    print(f"  衝突: 壁へ直進する想定で {V_CRASH} m/s（速いほどエピソードが短く罰を受けにくい）")
    print(f"  時間切れ・滞留: T = {T_TIMEOUT} 歩（S = {S(T_TIMEOUT):.1f} ＝ ほぼ飽和 200）")
    print(f"  罰の実測平均: ‖Δa‖² = {MEAN_DA2}（exp_006） / "
          f"‖a−ā‖² = {MEAN_HP2_A05}（案 3, α=0.5）")

    print("\n" + "=" * 78)
    print("§2 衝突位置ごとの実効ステップ数の差 ΔS = S(T_goal) − S(T_crash)")
    print("=" * 78)
    print(f"{'衝突位置 x [m]':>14}{'T_crash [歩]':>14}{'S(T_crash)':>12}"
          f"{'ΔS':>10}{'G_collide (c=0)':>18}")
    for x in CRASH_POSITIONS:
        Tc = crash_steps(x, V_CRASH)
        print(f"{x:>14.2f}{Tc:>14.1f}{S(Tc):>12.1f}{S(T_GOAL) - S(Tc):>10.1f}"
              f"{G_collide(Tc, x, 0.0):>18.3f}")

    g0 = G_goal(T_GOAL, D0, 0.0)
    print(f"\n  ゴールの総収益（c=0）= {g0:.3f}")
    best_x = max(CRASH_POSITIONS, key=lambda x: G_collide(crash_steps(x, V_CRASH), x, 0.0))
    Tb = crash_steps(best_x, V_CRASH)
    print(f"  最も得な衝突 = x={best_x} m（{G_collide(Tb, best_x, 0.0):.3f}）"
          f" → 余裕 = {g0 - G_collide(Tb, best_x, 0.0):.3f}")

    # -------------------------------------------------------------------
    # §3 k を振って順序が崩れる閾値を求める
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("§3 exp_006（‖Δa‖²、平均 3.873）: k を振ったときの総収益")
    print("=" * 78)
    ks_da = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 1.8e-3, 3e-3, 1e-2]
    _sweep(ks_da, MEAN_DA2)

    print("\n" + "=" * 78)
    print("§4 案 3（‖a−ā‖²、α=0.5、平均 0.483）: k を振ったときの総収益")
    print("=" * 78)
    ks_hp = [0.0, 1e-3, 2.0e-3, 5e-3, 8.7e-3, 1.4e-2, 3e-2]
    _sweep(ks_hp, MEAN_HP2_A05)

    # -------------------------------------------------------------------
    # §5 閾値を解析的に求める
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("§5 順序が崩れる k の閾値（解析解）")
    print("=" * 78)
    print("  条件: G_goal(c) = G_collide(c)  ⇔  c·ΔS = （c=0 での余裕）")
    print(f"{'衝突位置 x [m]':>14}{'ΔS':>10}{'余裕(c=0)':>12}{'閾値 c':>12}"
          f"{'k (‖Δa‖²)':>14}{'k (案3)':>12}")
    worst = None
    for x in CRASH_POSITIONS:
        Tc = crash_steps(x, V_CRASH)
        dS = S(T_GOAL) - S(Tc)
        margin = g0 - G_collide(Tc, x, 0.0)
        if dS <= 0:
            continue
        c_th = margin / dS
        k_da, k_hp = c_th / MEAN_DA2, c_th / MEAN_HP2_A05
        print(f"{x:>14.2f}{dS:>10.1f}{margin:>12.3f}{c_th:>12.2e}"
              f"{k_da:>14.2e}{k_hp:>12.2e}")
        if worst is None or c_th < worst[1]:
            worst = (x, c_th, k_da, k_hp)
    print(f"\n  **最も厳しい衝突位置 x={worst[0]} m**: 閾値 c = {worst[1]:.2e}")
    print(f"    → exp_006（‖Δa‖²）: k ≥ {worst[2]:.2e} で順序が崩れる")
    print(f"    → 案 3（‖a−ā‖², α=0.5）: k ≥ {worst[3]:.2e} で順序が崩れる")

    # -------------------------------------------------------------------
    # §6 「もっと進んでから衝突する」方向の学習勾配 dG/dx
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("§6 学習勾配 dG/dx: 『もっと進んでから衝突する』は得か")
    print("=" * 78)
    print("  ゴールに到達できない段階の学習は「衝突までの距離 x を伸ばす」ことで進む。")
    print("  dG_collide/dx > 0 なら伸ばす方向に勾配がある。負なら**早く衝突する方が得**。")
    print(f"\n{'x [m]':>8}" + "".join(f"{f'k={k:g}':>12}" for k in ks_da))
    xs = np.arange(0.2, 2.6, 0.2)
    for x in xs:
        row = f"{x:>8.1f}"
        for k in ks_da:
            c = k * MEAN_DA2
            h = 0.02
            Tp, Tm = crash_steps(x + h, V_CRASH), crash_steps(x - h, V_CRASH)
            d = (G_collide(Tp, x + h, c) - G_collide(Tm, x - h, c)) / (2 * h)
            row += f"{d:>12.3f}"
        print(row)
    print("\n  （負値 = その距離では『もっと手前で衝突する』方が総収益が高い＝学習が後退する）")

    # -------------------------------------------------------------------
    # §6-bis 束縛点は方策の能力に依存する（2026-08-11 追加）
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("§6-bis 「最も得な衝突」は方策の能力に依存する — 静的な最悪ケースが保守的な理由")
    print("=" * 78)
    print("  §5 は衝突位置を x=0.3〜2.5 m の全域から選んで最も得なものを束縛にした。")
    print("  しかし**その位置まで走れない方策は、そこで衝突できない**。")
    print("  学習初期は遠くまで走れないので束縛は始点近くの衝突で決まり、")
    print("  能力が上がるにつれ束縛点が奥へ移動する。")
    print("  **静的な最悪ケース（x=2.5 m）は、どの時点の方策にも対応しない仮想的な条件**")
    print("  であり、必然的に保守的（安全側）になる。\n")
    print("  到達できる最大距離 x_max ごとに、束縛となる衝突位置と閾値を出す:")
    print(f"{'x_max [m]':>10}{'束縛となる衝突 x':>18}{'余裕(c=0)':>12}{'閾値 c':>12}"
          f"{'k (‖Δa‖²)':>14}{'k (案3)':>12}")
    for x_max in (0.6, 1.26, 1.6, 2.0, 2.5):
        cand = [x for x in CRASH_POSITIONS if x <= x_max + 1e-9]
        best_c, best_x = None, None
        for x in cand:
            Tc = crash_steps(x, V_CRASH)
            dS = S(T_GOAL) - S(Tc)
            if dS <= 0:
                continue
            c_th = (g0 - G_collide(Tc, x, 0.0)) / dS
            if best_c is None or c_th < best_c:
                best_c, best_x = c_th, x
        margin = g0 - G_collide(crash_steps(best_x, V_CRASH), best_x, 0.0)
        print(f"{x_max:>10.2f}{best_x:>18.2f}{margin:>12.3f}{best_c:>12.2e}"
              f"{best_c / MEAN_DA2:>14.2e}{best_c / MEAN_HP2_A05:>12.2e}")
    print("\n  閾値は x_max とともに単調に下がる ＝ **能力が上がるほど束縛が厳しくなる**。")
    print("  ただし x_max → D₀ ではゴールへ到達できる（ゴール 0.799 > 衝突 2.5 m の 0.574）")
    print("  ので、**その領域の方策はもう衝突を選ばない**。したがって")
    print("  **計算で出るのは「安全側の下界」であって、真の崖の位置ではない。**")
    print("  閾値を下回る k は安全に使えるが、超えたからといって壊れるとは限らない。")

    # -------------------------------------------------------------------
    # §7 仮定の感度: 衝突時の速度を変えると閾値はどれだけ動くか
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("§7 感度解析: 衝突時の速度 V_CRASH を変えたときの閾値")
    print("=" * 78)
    print("  V_CRASH が大きいほど衝突エピソードが短く、罰を受けにくい＝ゴールが不利になる。")
    print(f"{'V_CRASH [m/s]':>14}{'最も得な衝突 x':>16}{'余裕(c=0)':>12}{'閾値 c':>12}"
          f"{'k (‖Δa‖²)':>14}{'k (案3)':>12}")
    for v_crash in (0.96, 1.20, 1.60, 2.00):
        best_c, best_xx = None, None
        for x in CRASH_POSITIONS:
            Tc = crash_steps(x, v_crash)
            dS = S(T_GOAL) - S(Tc)
            if dS <= 0:
                continue
            c_th = (g0 - G_collide(Tc, x, 0.0)) / dS
            if best_c is None or c_th < best_c:
                best_c, best_xx = c_th, x
        margin = g0 - G_collide(crash_steps(best_xx, v_crash), best_xx, 0.0)
        print(f"{v_crash:>14.2f}{best_xx:>16.2f}{margin:>12.3f}{best_c:>12.2e}"
              f"{best_c / MEAN_DA2:>14.2e}{best_c / MEAN_HP2_A05:>12.2e}")
    return 0


def _sweep(ks, mean_pen):
    print(f"{'k':>10}{'c=k·罰':>12}{'ゴール':>10}"
          + "".join(f"{f'衝突{x}m':>10}" for x in CRASH_POSITIONS)
          + f"{'滞留':>10}{'余裕':>10}{'順序':>8}")
    for k in ks:
        c = k * mean_pen
        gg = G_goal(T_GOAL, D0, c)
        gcs = [G_collide(crash_steps(x, V_CRASH), x, c) for x in CRASH_POSITIONS]
        gt = G_timeout(T_TIMEOUT, 0.0, c)
        margin = gg - max(gcs)
        ok = "OK" if margin > 0 else "**崩壊**"
        print(f"{k:>10.1e}{c:>12.2e}{gg:>10.3f}"
              + "".join(f"{v:>10.3f}" for v in gcs)
              + f"{gt:>10.3f}{margin:>10.3f}{ok:>8}")


if __name__ == "__main__":
    raise SystemExit(main())
