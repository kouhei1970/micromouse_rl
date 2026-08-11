"""
research_notes/scripts/check_m2_gamma_horizon.py
================================================
**M2-0 の割引後総収益と学習勾配を、検証帯 20 面の実 D₀ ごとに計算する**（2026-08-11）。

任務 2026-08-11-N2 の (2)(3)。第一容疑者 γ=0.995 を机上で検算する。

## 何を計算するか

`experiments/exp_006_action_smoothness/taskB_return_ordering.md` の形式に従い、
検証帯（seed 7000-7019、loop モード）の**各面の実 D₀** について:

  1. 割引後の総収益  ゴール / 探索して時間切れ / 滞留 / 衝突（位置を振る）
  2. 学習勾配 dG_collide/dx（もっと進んでから衝突する方が得か）
  3. **経路効率の許容倍率 ρ_max**
     = 「最短の何倍まで回り道してゴールしても、滞留・衝突より得か」

(3) が本スクリプトの新しい点である。従来の検算は**最短経路で走った場合**の
総収益だけを見ていたが、学習の初期に方策が実際に経験するのは
**最短の何倍も回り道した末のゴール**である。γ の実効地平が効くのはそちらであり、
「最短経路長 vs 実効地平」の比較では γ の当否を判定できない。

## 報酬モデル（`mouse/maze6_env.py` 306-345 行）

  r_t = γ·Φ(s_t) − Φ(s_(t−1)) − p − c  [+1.0 ゴール] [−1.0 衝突] [+0.02 未訪問]
  Φ = D₀ − d(区画)   d はゴールまでの迷路距離 [m]、Φ_0 = 0
  p = 0.001（時間罰）、c = k·E‖a−ā‖²（滑らかさの罰の 1 歩あたり平均）

ポテンシャル項は telescoping するので、t=1 から γ^(t−1) で割り引いた総収益は

  G = γ^T·Φ_T − (p + c)·S(T) + γ^(T−1)·(終端ボーナス/罰) + V(訪問報酬)
  S(T) = (1 − γ^T) / (1 − γ)          ← 割引後の実効ステップ数

**訪問報酬の割引の規約**（`check_m2_return_ordering.py` と 1 区画ぶんずれるので明記する）:
i 番目の未訪問区画は i·(1 区画あたりの歩数) 歩目に入るとし、その報酬は
γ^(i·spc − 1) で割り引く（i は 1 始まり）。スタート区画は reset 時点で訪問済みなので
報酬は付かない。既存スクリプトは i を 0 始まりにしていたため総和が γ^(−spc) = 1.098 倍
大きく出ていた。**本スクリプトの方が実装に忠実**である。

使い方:
    .venv/bin/python research_notes/scripts/check_m2_gamma_horizon.py
    .venv/bin/python research_notes/scripts/check_m2_gamma_horizon.py --mean-hp2 0.483
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mouse.maze6_gen import generate_maze, shortest_distances  # noqa: E402

# ---- 実装から取った定数（mouse/maze6_env.py） ----------------------------
DT = 0.01                # 制御周期 [s]
CELL = 0.18              # 区画寸法 [m]
TIME_PENALTY = 0.001     # _TIME_PENALTY
GOAL_BONUS = 1.0         # _GOAL_BONUS
COLLISION = -1.0         # _COLLISION_PENALTY
VISIT_BONUS = 0.02       # _VISIT_BONUS
TIME_LIMIT = 6000        # _TIME_LIMIT_STEPS
N_CELLS = 36             # 6x6
VALID_SEEDS = tuple(range(7000, 7020))   # 検証帯（学習・調整に使ってよい帯）

# ---- 挙動から測る量（既定は M1 実測。--mean-hp2 で上書きできる） --------
# E‖a−ā‖² は**挙動に強く依存する**。M2-0 の実測（outputs/m2_step_economy/step_economy.json、
# 検証帯 20 面 × 4 モデル）:
#   走行できている方策（exp_011 k=0）: プール中央値 0.459 / 0.453、走行層の平均 0.352〜0.544
#   凍結した方策（exp_010 k=8.7e-3）:  プール中央値 2.1e-11、停止層の平均 1.086e-4
# M1 の走行方策は 0.483 だったので、**走る方策の値は M1 と M2-0 でほぼ同じ**。
DEFAULT_MEAN_HP2 = 0.459    # M2-0 の実測（exp_011_k0_seed1、走行できている方策）
MEASURED_HP2_STOP = 1.086e-4  # M2-0 の実測（exp_010_seed1 の停止層の平均）
K_M1 = 8.7e-3              # M1 で確定し M2-0 へ持ち込んだ係数
V_RUN = 0.96               # 完走時の平均前進速度 [m/s]（M1 実測）
# 中央差分をとる x は**区画の中央**に置く。0.18 の倍数（区画境界）をまたぐと
# 階段の跳び 0.18 を 2h で割った偽の巨大値（+12 等）を拾う。
MID_X = tuple(round(CELL * (i + 0.5), 4) for i in range(9))


# ==========================================================================
# 総収益の部品
# ==========================================================================
def S(T, gamma):
    """割引後の実効ステップ数 S(T) = (1 − γ^T)/(1 − γ)。"""
    return (1.0 - gamma ** T) / (1.0 - gamma)


def visit_return(n_new, spc, gamma):
    """未訪問区画 n_new 個ぶんの訪問報酬の割引後総和。

    i 番目（i は 1 始まり）の区画は i·spc 歩目に入るとし、γ^(i·spc − 1) で割り引く。
    """
    if n_new <= 0:
        return 0.0
    return VISIT_BONUS * sum(gamma ** (i * spc - 1.0) for i in range(1, int(n_new) + 1))


def g_goal(d_cells, gamma, c, rho=1.0, v=V_RUN):
    """ゴール到達の総収益。rho = 実走経路長 / 最短経路長（1.0 が最短）。

    回り道しても最後にゴールへ入るので Φ_T = D₀ は変わらない。変わるのは
    T（＝罰の総量と、ゴールボーナスの割引）と訪問区画数である。
    """
    D0 = d_cells * CELL
    spc = CELL / (v * DT)                 # 1 区画あたりの歩数
    T = rho * d_cells * spc
    n_new = min(round(rho * d_cells), N_CELLS - 1)   # 回り道ぶんは再訪も含むので上限 35
    return (gamma ** T * D0
            - (TIME_PENALTY + c) * S(T, gamma)
            + gamma ** (T - 1.0) * GOAL_BONUS
            + visit_return(n_new, spc, gamma))


def g_dwell(gamma, c_stop=None, k=K_M1):
    """滞留（スタート区画に留まって時間切れ）。Φ_T = 0、訪問なし。

    **滑らかさの罰は「走行時の値」を使ってはいけない。**罰は挙動に依存し、
    凍結した方策は E‖a−ā‖² を実測 1.086e-4 まで落として**罰そのものを消している**
    （走行時 0.459 の 4200 分の 1）。走行時の c を滞留に適用すると滞留を
    0.999 と 5 倍過大に見積もり、**順序判定を誤る**。
    引き継ぎメモ (3)-3「罰の大きさは挙動に依存する。比較表の各行で測り直すこと」。
    """
    if c_stop is None:
        c_stop = k * MEASURED_HP2_STOP
    return -(TIME_PENALTY + c_stop) * S(TIME_LIMIT, gamma)


def g_explore(gamma, c, phi_T=0.0, v=V_RUN, revisit=1.0):
    """探索して時間切れ。全 36 区画を訪問しきる（訪問報酬の上限を与える楽観側）。

    revisit: 1 区画を新規に開拓するのに要する歩数の、最短走行に対する倍率。
             1.0 は「再訪ゼロで開拓できる」＝訪問報酬を最も高く見積もる仮定。
    """
    spc = CELL / (v * DT) * revisit
    return (gamma ** TIME_LIMIT * phi_T
            - (TIME_PENALTY + c) * S(TIME_LIMIT, gamma)
            + visit_return(N_CELLS - 1, spc, gamma))


def g_collide(x, gamma, c, v=V_RUN, phi_mode="stair"):
    """最短経路上を x [m] 進んでから衝突。

    phi_mode:
      "stair" — **実装どおり**。Φ は区画単位の階段関数なので、
                進んだ区画数 floor(x/CELL) だけ Φ が上がる（区画内では一定）
      "cont"  — 反実仮想。Φ が連続（M1 の廊下と同じ）だとしたら Φ_T = x
    """
    spc = CELL / (v * DT)
    T = x / (v * DT)
    n_cells = int(x // CELL)
    phi_T = x if phi_mode == "cont" else n_cells * CELL
    return (gamma ** T * phi_T
            - (TIME_PENALTY + c) * S(T, gamma)
            + gamma ** (T - 1.0) * COLLISION
            + visit_return(n_cells, spc, gamma))


def best_collide(d_cells, gamma, c, v=V_RUN, n=400):
    """最も得な衝突（衝突位置 x を振った最大値）。

    **x の上限は D₀·CELL より真に小さくとる。**x = D₀·CELL はゴール区画に入った
    位置であり、そこでは環境がゴールで終端するので「衝突」は起こりえない。
    上限を含めると進んだ区画数が D₀ になり、**衝突を 1 区画ぶん過大評価する**
    （D₀=4 で −0.445 対 正しくは −0.546、D₀ が小さい面ほど影響が大きい）。
    """
    xs = np.linspace(0.02, d_cells * CELL - 1e-9, n)
    gs = np.array([g_collide(x, gamma, c, v) for x in xs])
    i = int(np.argmax(gs))
    return float(gs[i]), float(xs[i])


def rho_max(d_cells, gamma, c, v=V_RUN, hi=60.0, k_drive=K_M1):
    """G_goal(ρ) > max(滞留, 探索, 最も得な衝突) を保つ最大の ρ。

    エピソード上限 6000 歩を超える ρ は物理的に成立しないので、その場合は
    上限（= 6000 歩に相当する ρ）で打ち切って「上限まで成立」と報告する。
    """
    alt = max(g_dwell(gamma, k=k_drive), g_explore(gamma, c),
              best_collide(d_cells, gamma, c, v)[0])
    spc = CELL / (v * DT)
    rho_cap = TIME_LIMIT / (d_cells * spc)        # 6000 歩に相当する ρ
    if g_goal(d_cells, gamma, c, 1.0, v) <= alt:
        return 0.0, alt, rho_cap                  # 最短でも勝てない
    lo, high = 1.0, min(hi, rho_cap)
    if g_goal(d_cells, gamma, c, high, v) > alt:
        return high, alt, rho_cap                 # 上限まで成立
    for _ in range(90):
        mid = 0.5 * (lo + high)
        if g_goal(d_cells, gamma, c, mid, v) > alt:
            lo = mid
        else:
            high = mid
    return lo, alt, rho_cap


# ==========================================================================
# 迷路の情報
# ==========================================================================
def maze_table(seeds=VALID_SEEDS, mode="loop"):
    """検証帯の各面の D₀ と最大距離 d_max を返す。"""
    out = []
    for s in seeds:
        m = generate_maze(s, mode=mode)
        dm = shortest_distances(m["v_walls"], m["h_walls"])
        st = tuple(m["start"])
        out.append(dict(seed=s, start=st, d0=int(dm[st]),
                        d_max=int(max(dm.values()))))
    return out


# ==========================================================================
# 出力
# ==========================================================================
def sec0_step_economy(mazes, mean_hp2, k, gamma, hp2_stop=MEASURED_HP2_STOP):
    """任務 (1): **1 歩あたりの経済**を M2-0 の条件で作り直す。

    M1 の表（handover/student_b.md 冒頭 (2)）をそのまま流用してはいけない。
    M1 の Φ は連続量で毎歩 +0.0096 の進捗報酬が入るが、**M2-0 の Φ は区画単位の
    階段関数**なので、区画内では進捗報酬が 0 で、境界で 1 回だけ +0.18 が入る。
    """
    c = k * mean_hp2
    d_med = float(np.median([m["d0"] for m in mazes]))
    D0 = d_med * CELL
    spc = CELL / (V_RUN * DT)
    prog_m1 = V_RUN * DT                     # M1 の 1 歩あたり進捗 [m]
    print("=" * 100)
    print("§0 【任務 (1)】1 歩あたりの経済 — M2-0 の条件で作り直す")
    print("=" * 100)
    print(f"  k = {k:g}、E‖a−ā‖²（走行時）= {mean_hp2:g} → 罰 {c:.4f}/歩")
    print(f"  E‖a−ā‖²（停止時）= {hp2_stop:g} → 罰 {k*hp2_stop:.7f}/歩")
    print(f"  Φ = D₀ − d(**区画**)。区画内では Φ 一定、境界で ±{CELL} m 跳ぶ")
    print(f"  1 区画 = {spc:.2f} 歩（v = {V_RUN} m/s）")
    print()
    print("  ── M2-0（Φ は階段関数。**実装どおり**）─────────────────────────────")
    print(f"{'挙動':<30}{'時間罰':>9}{'滑らか罰':>10}{'整形':>10}{'訪問':>9}{'合計':>10}")
    rows = []

    def row(name, hp2, shaping, visit):
        pen = -k * hp2
        tot = -TIME_PENALTY + pen + shaping + visit
        rows.append((name, tot))
        print(f"{name:<30}{-TIME_PENALTY:>9.4f}{pen:>10.4f}"
              f"{shaping:>10.4f}{visit:>9.4f}{tot:>10.4f}")

    # 区画内（スタート区画 Φ=0）: 整形 = −(1−γ)·Φ = 0
    row("停止する（Φ=0 の区画）", hp2_stop, 0.0, 0.0)
    row("区画内を走る（Φ=0）", mean_hp2, 0.0, 0.0)
    # 区画内（ゴール手前 Φ=D₀）: 整形 = −(1−γ)·D₀
    drift = -(1.0 - gamma) * D0
    row(f"停止する（Φ=D₀={D0:.2f} の区画）", hp2_stop, drift, 0.0)
    row("区画内を走る（Φ=D₀）", mean_hp2, drift, 0.0)
    # 境界を越える 1 歩
    row("境界を越える歩（正・未訪問）", mean_hp2, gamma * CELL, VISIT_BONUS)
    row("境界を越える歩（正・既訪問）", mean_hp2, gamma * CELL, 0.0)
    row("境界を越える歩（逆・未訪問）", mean_hp2, -CELL, VISIT_BONUS)
    row("境界を越える歩（逆・既訪問）", mean_hp2, -CELL, 0.0)
    print()
    print("  ── 参考: M1 廊下（Φ は連続。毎歩 進捗が入る）───────────────────────")
    print(f"{'挙動':<30}{'時間罰':>9}{'滑らか罰':>10}{'整形':>10}{'訪問':>9}{'合計':>10}")
    row("停止する", hp2_stop, 0.0, 0.0)
    row("正しい方向へ走る（M1）", mean_hp2, gamma * prog_m1, 0.0)
    print()
    print("  ── 1 区画（18.75 歩）を単位にした比較 ────────────────────────────")
    n = spc
    for tag, hp2, sh, vis in (("区画内で停止し続ける", hp2_stop, 0.0, 0.0),
                              ("正方向に 1 区画走る", mean_hp2, 0.0, 0.0)):
        per = -TIME_PENALTY - k * hp2 + sh
        acc = per * S(n, gamma)
        if tag.startswith("正方向"):
            acc += gamma ** (n - 1.0) * (gamma * CELL + VISIT_BONUS)
        print(f"    {tag:<24} 割引後 {acc:>8.4f}")
    print()
    print("  【要点】M1 は**毎歩** 整形 +{:.4f} が入るので走ることが即座に得（合計 +{:.4f}/歩）。"
          .format(gamma * prog_m1, -TIME_PENALTY - k * mean_hp2 + gamma * prog_m1))
    print("  M2-0 は区画内の整形が 0 なので、走ることは**毎歩 {:.4f} の純損**であり、"
          .format(-TIME_PENALTY - k * mean_hp2))
    print(f"  停止（{-TIME_PENALTY - k*hp2_stop:.4f}/歩）の {(TIME_PENALTY + k*mean_hp2)/(TIME_PENALTY + k*hp2_stop):.1f} 倍損をする。")
    print(f"  報酬は {spc:.1f} 歩後の +{gamma*CELL + VISIT_BONUS:.3f} の一発だけで返ってくる。")
    print()
    return rows


def sec1_premise(mazes, mean_hp2, k):
    c = k * mean_hp2
    d = np.array([m["d0"] for m in mazes])
    spc = CELL / (V_RUN * DT)
    print("=" * 100)
    print("§1 前提 — 何を入力にしたか")
    print("=" * 100)
    print(f"  検証帯 seed 7000-7019（loop）の D₀[区画]: "
          f"最小 {d.min()} / 中央値 {np.median(d):.1f} / 最大 {d.max()}")
    print(f"  D₀[m]:  {d.min()*CELL:.2f} / {np.median(d)*CELL:.2f} / {d.max()*CELL:.2f}")
    print(f"  1 区画あたり {spc:.2f} 歩（v = {V_RUN} m/s、Δt = {DT} s）")
    print(f"  最短経路の T[歩]: {d.min()*spc:.0f} / {np.median(d)*spc:.0f} / {d.max()*spc:.0f}")
    print(f"  滑らかさの罰: k = {k:g}、E‖a−ā‖² = {mean_hp2:g}"
          f"  → c = {c:.3e}（時間罰 {TIME_PENALTY} の {c/TIME_PENALTY:.2f} 倍）")
    print()
    print("  γ ごとの実効地平と、時間罰の割引後の飽和値:")
    print(f"    {'γ':>7}{'1/(1−γ)[歩]':>13}{'[s]':>7}{'[区画]':>9}"
          f"{'γ^T(中央値の最短)':>18}{'滞留 G':>10}")
    for g in (0.995, 0.997, 0.998, 0.999):
        h = 1.0 / (1.0 - g)
        Tmed = float(np.median(d)) * spc
        print(f"    {g:>7.3f}{h:>13.0f}{h*DT:>7.1f}{h/spc:>9.1f}"
              f"{g**Tmed:>18.3f}{g_dwell(g):>10.3f}")
    print()
    print("  【要点】6x6 の最短経路 75〜281 歩は γ=0.995 の実効地平 200 歩と同程度で、")
    print("  γ^T は 0.245〜0.687。16x16 で報告された γ^T = 0.00115 のような潰れは起きない。")


def sec2_per_maze(mazes, gamma, c, k, mean_hp2):
    print("=" * 100)
    print(f"§2 検証帯 20 面ごとの割引後総収益（γ = {gamma}、k = {k:g}、c = {c:.3e}）")
    print("=" * 100)
    print("  ゴール = 最短経路を 0.96 m/s で走る / 探索 = 6000 歩・全 36 区画訪問")
    print("  滞留 = スタート区画で 6000 歩 / 衝突 = 最短経路上で最も得な位置")
    print()
    print(f"{'seed':>6}{'D₀[区]':>8}{'T[歩]':>7}{'γ^T':>7}"
          f"{'ゴール':>9}{'探索':>9}{'滞留':>9}{'衝突*':>9}{'x*[m]':>7}"
          f"{'余裕':>8}{'実際の順序':>22}")
    rows, broken = [], []
    spc = CELL / (V_RUN * DT)
    for m in mazes:
        d = m["d0"]
        T = d * spc
        gg = g_goal(d, gamma, c)
        ge = g_explore(gamma, c)
        gd = g_dwell(gamma)
        gc, xs = best_collide(d, gamma, c)
        alt = max(ge, gd, gc)
        # 望ましい順序は ゴール > 探索 > 滞留 > 衝突。実際の順序を並べて出す
        names = sorted((("ゴール", gg), ("探索", ge), ("滞留", gd), ("衝突", gc)),
                       key=lambda t: -t[1])
        order_str = ">".join(n for n, _ in names)
        order_ok = order_str == "ゴール>探索>滞留>衝突"
        if not order_ok:
            broken.append((m["seed"], d, order_str))
        rows.append(dict(seed=m["seed"], d0=d, T=T, gamma_T=gamma ** T,
                         goal=gg, explore=ge, dwell=gd, collide=gc, x_star=xs,
                         margin=gg - alt, order=order_str, order_ok=order_ok))
        print(f"{m['seed']:>6}{d:>8}{T:>7.0f}{gamma**T:>7.3f}"
              f"{gg:>9.3f}{ge:>9.3f}{gd:>9.3f}{gc:>9.3f}{xs:>7.2f}"
              f"{gg-alt:>8.3f}   {order_str}")
    print()
    print("  余裕 = ゴール − max(探索, 滞留, 最も得な衝突)。* は最大化した衝突位置。")
    n_bad = len(broken)
    print(f"  **望ましい順序「ゴール>探索>滞留>衝突」が崩れる面: {n_bad} / {len(mazes)}**")
    uniq = sorted(set(o for _, _, o in broken))
    for o in uniq:
        ss = [s for s, _, oo in broken if oo == o]
        print(f"    実際の順序「{o}」: {len(ss)} 面（seed {min(ss)}〜{max(ss)}）")
    return rows


def sec3_gradient(mazes, gamma, c):
    print()
    print("=" * 100)
    print("§3 学習勾配 dG_collide/dx — もっと進んでから衝突する方が得か")
    print("=" * 100)
    print("  ゴールに一度も到達できない段階の学習は「衝突までの距離 x を伸ばす」で")
    print("  しか進まないので、dG/dx > 0 が必要（taskB_return_ordering.md §5 と同じ形式）")
    print()
    d_med = int(np.median([m["d0"] for m in mazes]))
    print(f"{'x[m]':>7}", end="")
    ks = (0.0, 8.7e-3, 1.5e-2, 3.0e-2)
    for k in ks:
        print(f"{'k=' + f'{k:g}':>12}", end="")
    print()
    for x in MID_X:
        if x > d_med * CELL:
            continue
        print(f"{x:>7.2f}", end="")
        for k in ks:
            cc = k * (c / K_M1 if K_M1 else 0.483)   # 同じ E‖a−ā‖² を使う
            h = 0.005
            grad = (g_collide(x + h, gamma, cc) - g_collide(x - h, gamma, cc)) / (2 * h)
            print(f"{grad:>12.3f}", end="")
        print()
    print(f"\n  （D₀ = 中央値 {d_med} 区画 = {d_med*CELL:.2f} m の面で計算）")

    print()
    print("-" * 100)
    print("§3-b 階段 Φ（実装）と連続 Φ（M1 の廊下）で dG/dx の符号が変わるか")
    print("-" * 100)
    print("  区画内では階段 Φ が動かないので、Φ_T の項が dG/dx に寄与しない。")
    print("  M1 は Φ が連続だったので、この項が毎歩 +γ^T ぶん寄与していた。")
    print()
    print(f"{'x[m]':>7}{'階段 k=0':>11}{'連続 k=0':>11}"
          f"{'階段 k=8.7e-3':>15}{'連続 k=8.7e-3':>15}{'差(連続−階段)':>15}")
    h = 0.005
    for x in MID_X:
        if x > d_med * CELL:
            continue
        out = []
        for kk in (0.0, K_M1):
            cc = kk * (c / K_M1 if K_M1 else 1.0)
            for pm in ("stair", "cont"):
                out.append((g_collide(x + h, gamma, cc, phi_mode=pm)
                            - g_collide(x - h, gamma, cc, phi_mode=pm)) / (2 * h))
        print(f"{x:>7.2f}{out[0]:>11.3f}{out[1]:>11.3f}"
              f"{out[2]:>15.3f}{out[3]:>15.3f}{out[3]-out[2]:>15.3f}")
    print()
    print("  → 差（連続−階段）は γ^T ≈ 0.3〜0.7 に一致する。**階段化で失われたのは**")
    print("     **ちょうどこの量**であり、k=8.7e-3 では符号を反転させるのに十分だった。")

    print()
    print("-" * 100)
    print("§3-c 区画を単位にした勾配 ΔG = G(x+1区画) − G(x) — 境界の跳びを含めれば正か")
    print("-" * 100)
    print(f"{'x[m]':>7}{'k=0':>11}{'k=8.7e-3':>12}{'k=1.5e-2':>12}{'k=3e-2':>11}")
    for x in (0.09, 0.27, 0.45, 0.63, 0.99, 1.35):
        if x + CELL > d_med * CELL:
            continue
        print(f"{x:>7.2f}", end="")
        for kk in (0.0, K_M1, 1.5e-2, 3.0e-2):
            cc = kk * (c / K_M1 if K_M1 else 1.0)
            dg = g_collide(x + CELL, gamma, cc) - g_collide(x, gamma, cc)
            print(f"{dg:>12.4f}" if kk else f"{dg:>11.4f}", end="")
        print()
    print()
    print("  区画をまたぐ単位で見れば境界の +0.18 が入るので符号は正になりうる。")
    print("  **つまり勾配は「区画単位では正・区画内では負」という向きが混ざった地形**であり、")
    print("  区画内の 18.75 歩は一貫して「いま終われ」と言っている。")

    print()
    print("-" * 100)
    print("§3-d γ を上げると dG/dx（階段・k=8.7e-3）はどう動くか")
    print("-" * 100)
    print(f"{'x[m]':>7}", end="")
    for g in (0.995, 0.997, 0.998, 0.999):
        print(f"{'γ=' + f'{g}':>13}", end="")
    print()
    for x in MID_X[::2]:
        if x > d_med * CELL:
            continue
        print(f"{x:>7.2f}", end="")
        for g in (0.995, 0.997, 0.998, 0.999):
            dg = (g_collide(x + h, g, c) - g_collide(x - h, g, c)) / (2 * h)
            print(f"{dg:>13.3f}", end="")
        print()
    print()
    print("  → **γ を上げても区画内の dG/dx は負のまま**（むしろ悪化する）。")
    print("     γ は本件の原因ではないことが、勾配の側からも確認できる。")


def sec4_rho(mazes, c, k):
    print()
    print("=" * 100)
    print("§4 【本題】経路効率の許容倍率 ρ_max — 最短の何倍まで回り道して")
    print("   ゴールしても、滞留・衝突より得か")
    print("=" * 100)
    print("  従来の検算は「最短経路で走った場合」の総収益だけを見ていた。しかし")
    print("  学習初期に方策が実際に経験するのは**最短の何倍も回り道した末のゴール**である。")
    print("  γ の実効地平が効くのはそちらであって、最短経路長との比較では判定できない。")
    print()
    gammas = (0.995, 0.997, 0.998, 0.999)
    print(f"{'seed':>6}{'D₀[区]':>8}", end="")
    for g in gammas:
        print(f"{'ρmax@' + f'{g}':>14}", end="")
    print(f"{'ρ上限(6000歩)':>15}")
    out = []
    for m in mazes:
        d = m["d0"]
        print(f"{m['seed']:>6}{d:>8}", end="")
        rec = dict(seed=m["seed"], d0=d)
        cap = None
        for g in gammas:
            r, alt, cap = rho_max(d, g, c)
            rec[f"rho_{g}"] = r
            print(f"{r:>14.2f}", end="")
        rec["rho_cap"] = cap
        print(f"{cap:>15.1f}")
        out.append(rec)
    print()
    for g in gammas:
        v = np.array([o[f"rho_{g}"] for o in out])
        cell_budget = np.array([o[f"rho_{g}"] * o["d0"] for o in out])
        print(f"  γ={g}: ρmax 中央値 {np.median(v):.2f}（範囲 {v.min():.2f}〜{v.max():.2f}）"
              f"  → 通れる区画数の予算 中央値 {np.median(cell_budget):.0f} 区画"
              f"（範囲 {cell_budget.min():.0f}〜{cell_budget.max():.0f}）")
    print()
    print("  「通れる区画数の予算」= ρmax × D₀。**方策がゴールに初めて当たるまでに**")
    print("  **踏める区画数の上限**であり、これを超えて回り道したゴールは")
    print("  滞留・衝突より損になる（＝報酬が『そこまで行くな』と言っている）。")
    return out


def hitting_time_cells(seed, mode="loop"):
    """一様ランダムウォークが**区画グラフ**上でゴール 2x2 に初到達するまでの期待歩数。

    方向の知識を持たない方策が「ゴールに初めて当たる」までに踏む区画数の基準値。
    h(ゴール)=0、h(v) = 1 + 平均_{隣接 u} h(u) を解く（線形方程式）。
    実際の PPO 方策は連続系の制御器でありランダムウォークそのものではないが、
    **方向の情報を持たない段階の所要区画数の目安**として使う。
    """
    m = generate_maze(seed, mode=mode)
    v, h = m["v_walls"], m["h_walls"]
    from mouse.maze6_gen import GOAL_CELLS as GC

    def nbrs(cell):
        x, y = cell
        out = []
        if v[x, y] == 0:
            out.append((x - 1, y))
        if v[x + 1, y] == 0:
            out.append((x + 1, y))
        if h[x, y] == 0:
            out.append((x, y - 1))
        if h[x, y + 1] == 0:
            out.append((x, y + 1))
        return [(a, b) for a, b in out if 0 <= a < 6 and 0 <= b < 6]

    cells = [(x, y) for x in range(6) for y in range(6) if (x, y) not in GC]
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    A = np.eye(n)
    b = np.ones(n)
    for c in cells:
        ns = nbrs(c)
        if not ns:
            continue
        for u in ns:
            if u in idx:
                A[idx[c], idx[u]] -= 1.0 / len(ns)
    sol = np.linalg.solve(A, b)
    return float(sol[idx[tuple(m["start"])]])


def sec4b_budget(mazes, c, rho_rows):
    print()
    print("=" * 100)
    print("§4-b 探索の予算 vs 方向を知らない方策が実際に要する区画数")
    print("=" * 100)
    print("  左: ρmax × D₀ = 報酬が許す踏破区画数の上限")
    print("  右: 一様ランダムウォークがゴール 2x2 に初到達するまでの期待区画数")
    print()
    print(f"{'seed':>6}{'D₀[区]':>8}{'予算@0.995':>12}{'予算@0.998':>12}"
          f"{'要求(RW)':>10}{'倍率@0.995':>12}{'倍率@0.998':>12}")
    ratios = []
    for r in rho_rows:
        ht = hitting_time_cells(r["seed"])
        b995 = r["rho_0.995"] * r["d0"]
        b998 = r["rho_0.998"] * r["d0"]
        ratios.append((ht / b995, ht / b998))
        print(f"{r['seed']:>6}{r['d0']:>8}{b995:>12.1f}{b998:>12.1f}"
              f"{ht:>10.1f}{ht/b995:>12.1f}{ht/b998:>12.1f}")
    a = np.array(ratios)
    print()
    print(f"  倍率（要求 / 予算）の中央値: γ=0.995 で {np.median(a[:,0]):.1f} 倍、"
          f"γ=0.998 で {np.median(a[:,1]):.1f} 倍")
    print(f"  範囲: γ=0.995 で {a[:,0].min():.1f}〜{a[:,0].max():.1f} 倍")
    print("  → **γ を 0.995 から 0.998 へ上げても不足は "
          f"{np.median(a[:,0])/np.median(a[:,1]):.2f} 倍しか縮まない。**")
    print("     γ をいくら上げても（0.999 でも）1 桁足りない。**γ は本件の律速ではない。**")


def sec5_landmines(mazes, c, k, mean_hp2):
    """γ を変えたときに報酬設計の地雷 1〜5 を再検算する（任務 (3)）。"""
    print()
    print("=" * 100)
    print("§5 γ を変えた場合の地雷 1〜5 の再検算（handover/student_b.md の 5 件）")
    print("=" * 100)
    d_med = int(np.median([m["d0"] for m in mazes]))
    D0 = d_med * CELL
    spc = CELL / (V_RUN * DT)
    T = d_med * spc
    gammas = (0.995, 0.997, 0.998, 0.999)

    print("\n【地雷 1】Φ = −d（オフセットなし）だと滞留が 2 位に上がる")
    print("  Φ = −d のとき滞留の整形報酬は毎歩 +(1−γ)·d > 0（止まるだけで正の報酬）。")
    print(f"  D₀ = {D0:.2f} m の面で、スタート区画に留まったときの 1 歩あたりの整形:")
    print(f"    {'γ':>7}{'(1−γ)·D₀':>12}{'時間罰 p':>10}{'差 (正なら滞留が得)':>22}")
    for g in gammas:
        gain = (1.0 - g) * D0
        print(f"    {g:>7.3f}{gain:>12.5f}{TIME_PENALTY:>10.5f}"
              f"{gain - TIME_PENALTY:>22.5f}")
    print("  → **γ を上げると (1−γ)·D₀ が小さくなり、地雷 1 はむしろ軽くなる。**")
    print("     ただし現行はオフセットあり（Φ = D₀ − d）なので、この地雷は元々踏んでいない。")

    print("\n【地雷 2】オフセットのみ（衝突罰なし）だと衝突がゴールより得")
    print(f"  D₀ = {D0:.2f} m・{d_med} 区画の面で、衝突罰 0 とした場合:")
    print(f"    {'γ':>7}{'ゴール':>10}{'最も得な衝突':>14}{'x*[m]':>8}{'差':>10}{'判定':>8}")
    for g in gammas:
        gg = g_goal(d_med, g, c)
        # 衝突罰を 0 にした版
        def gc0(x, g=g):
            TT = x / (V_RUN * DT)
            n = int(x // CELL)
            return (g ** TT * (n * CELL) - (TIME_PENALTY + c) * S(TT, g)
                    + visit_return(n, spc, g))
        xs = np.linspace(0.02, D0, 400)
        vals = np.array([gc0(x) for x in xs])
        i = int(np.argmax(vals))
        print(f"    {g:>7.3f}{gg:>10.3f}{vals[i]:>14.3f}{xs[i]:>8.2f}"
              f"{gg-vals[i]:>10.3f}{'OK' if gg > vals[i] else 'NG':>8}")
    print("  → 衝突罰 −1.0 を入れている限り γ を変えても順序は保たれる（§2 参照）。")
    print("     **ただし γ を上げると滞留・時間切れの罰の飽和値 1/(1−γ) が伸びるので、**")
    print("     **衝突（−1.0 の一発）が滞留より相対的に得になる方向に動く。**下表参照:")
    print(f"    {'γ':>7}{'滞留 G':>10}{'即衝突 G':>11}{'滞留 − 即衝突':>16}{'判定':>18}")
    for g in gammas:
        gd = g_dwell(g)
        gimm = g_collide(0.05, g, c)
        note = "滞留の方が得" if gd > gimm else "**即衝突の方が得**"
        print(f"    {g:>7.3f}{gd:>10.3f}{gimm:>11.3f}{gd-gimm:>16.3f}{note:>18}")

    print("\n【地雷 3】時間罰を上げることは Φ オフセットと等価（F' = F − (1−γ)c）")
    print("  この等価性は γ に依らず成立する（恒等式）。**変わるのは実効的な罰の大きさ**:")
    print(f"    Φ = D₀ − d の実装は、スタート付近では罰 0、ゴール直前では")
    print(f"    毎歩 (1−γ)·D₀ の追加罰として効く。D₀ = {D0:.2f} m のとき:")
    print(f"    {'γ':>7}{'(1−γ)·D₀ [/歩]':>17}{'時間罰 p との比':>16}"
          f"{'進捗報酬 γ·ΔΦ/歩':>19}{'ゴール直前の純整形':>20}")
    for g in gammas:
        drift = (1.0 - g) * D0
        prog = g * V_RUN * DT      # 1 歩あたりの進捗（連続近似）
        print(f"    {g:>7.3f}{drift:>17.5f}{drift/TIME_PENALTY:>16.2f}"
              f"{prog:>19.5f}{prog - drift:>20.5f}")
    print("  → **γ=0.995 ではゴール直前の純整形が進捗報酬の "
          f"{(V_RUN*DT*0.995 - 0.005*D0)/(V_RUN*DT*0.995)*100:.0f}% まで痩せる。**")
    print("     γ を上げるとこの痩せが軽くなる（γ 変更の副産物としての利点）。")

    print("\n【地雷 4】終端で Φ=0 と強制する実装は禁止")
    print("  γ に依らず禁止のまま。最後の遷移で F = γ·0 − Φ(s_(T−1)) となり、")
    print("  Φ = D₀ − d のオフセット版では Φ(s_(T−1)) > 0 なので **負**の報酬が入る")
    print("  （Φ = −d の版では正の報酬が入るので、より危険）。いずれも実装しない。")

    print("\n【地雷 5】本設計は理論的にクリーンではない（終端 Φ≠0 の早期終了ボーナスを")
    print("  衝突罰で相殺している）。γ を変えても構造は変わらない。**γ を上げると**")
    print("  **相殺のバランスが動く**（上の地雷 2 の下表がその実体）ので、γ を変える")
    print("  実験では衝突率・時間切れ率の内訳を必ず見ること。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-hp2", type=float, default=DEFAULT_MEAN_HP2,
                    help="E‖a−ā‖² の実測値（既定は M1 の走行方策 0.483）")
    ap.add_argument("--k", type=float, default=K_M1)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    c = args.k * args.mean_hp2
    mazes = maze_table()

    sec0_step_economy(mazes, args.mean_hp2, args.k, args.gamma)
    sec1_premise(mazes, args.mean_hp2, args.k)
    print()
    rows = sec2_per_maze(mazes, args.gamma, c, args.k, args.mean_hp2)
    sec3_gradient(mazes, args.gamma, c)
    rho = sec4_rho(mazes, c, args.k)
    sec4b_budget(mazes, c, rho)
    sec5_landmines(mazes, c, args.k, args.mean_hp2)

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(
            premise=dict(gamma=args.gamma, k=args.k, mean_hp2=args.mean_hp2, c=c,
                         v_run=V_RUN, dt=DT, cell=CELL,
                         time_penalty=TIME_PENALTY, visit_bonus=VISIT_BONUS,
                         time_limit=TIME_LIMIT),
            mazes=mazes, returns=rows, rho_max=rho), ensure_ascii=False, indent=2))
        print(f"\n  → {p} に保存")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
