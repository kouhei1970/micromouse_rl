#!/usr/bin/env python3
"""016-H1e の模型（進入直線の横偏差の 2 次系）を、走行なしで再計算する。

カード `experiments/exp_016_diagonal/card_016h1e.md` §1・§2 の表と、
事前登録 `experiments/exp_016_diagonal/PREREG_016h1e.md` §4 の W1 の模型値を出す。

**走行は一切しない — 純粋な計算だけである。**
定数は `competition/baseline_slalom.py:649` の既定値を目視で書き写したものであり、
コードを import しない（錨は独立に出す、という §12-9 (c) の規約による）。

模型:
    直線区間（曲率 0）で |e_y|, |e_psi| が小さいとき
        de_y/dt    = -v * e_psi
        de_psi/dt  = -omega,  omega = k_psi * e_psi - atan(k_y * e_y / (v + v_eps))
    を線形化して畳むと
        e_y'' + k_psi * e_y' + (v * k_y / (v + v_eps)) * e_y = 0
    となる。zeta > 1（過減衰）なので遅い極が支配し、その時定数を tau とする。
    是正に食う「距離」は L = v * tau である。
"""

import math

# --- 正本から書き写した定数（competition/baseline_slalom.py:649） ---
K_PSI = 12.0
K_Y = 10.0
V_EPS = 0.15

# 否定対照: 分母を凍結する定数（v=0.45 水準の公称値 0.45 + 0.15）
D0_FROZEN = 0.60

# 速度水準（016-H1d 第 2 段と同一）
LEVELS = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def slow_time_constant(wn2: float) -> float:
    """過減衰 2 次系の遅い極の時定数 [s]。振動域なら例外を投げる。"""
    disc = K_PSI * K_PSI - 4.0 * wn2
    if disc <= 0.0:
        raise ValueError(f"振動域に入っている（wn2={wn2:.4f}）。模型の前提が崩れている")
    return 2.0 / (K_PSI - math.sqrt(disc))


def decay_length(v: float, denom: float | None = None) -> tuple[float, float, float]:
    """速度 v での (固有角周波数の 2 乗, 時定数 tau, 減衰長 L=v*tau) を返す。

    denom を与えると分母をその定数に凍結する（否定対照）。
    """
    d = (v + V_EPS) if denom is None else denom
    wn2 = v * K_Y / d
    tau = slow_time_constant(wn2)
    return wn2, tau, v * tau


def linear_fit(xs, ys) -> tuple[float, float]:
    """最小二乗の (傾き, 切片)。numpy に依存しない（環境が無くても走る）。"""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    return a, my - a * mx


def _table(name: str, denom: float | None) -> tuple[float, float]:
    lengths = []
    print(f"\n=== {name} ===")
    print(f"{'v[m/s]':>7} {'wn^2':>9} {'zeta':>7} {'tau[s]':>8} {'L=v*tau[m]':>11}")
    for v in LEVELS:
        wn2, tau, length = decay_length(v, denom)
        lengths.append(length)
        zeta = K_PSI / (2.0 * math.sqrt(wn2))
        print(f"{v:7.2f} {wn2:9.3f} {zeta:7.3f} {tau:8.3f} {length:11.4f}")
    a, c = linear_fit(LEVELS, lengths)
    print(f"  → L(v) の回帰: 傾き a = {a:+.4f} s / 切片 c = {c:.4f} m")
    print(f"  → L(0.70)/L(0.45) = {lengths[-1] / lengths[0]:.4f}")
    return a, c


def main() -> None:
    print("016-H1e 模型の再計算（走行なし）")
    print(f"定数: k_psi={K_PSI}, k_y={K_Y}, v_eps={V_EPS}（competition/baseline_slalom.py:649）")

    a_healthy, _ = _table("健常（分母 = v + v_eps）", None)
    a_frozen, _ = _table(f"否定対照（分母 = {D0_FROZEN} に凍結）", D0_FROZEN)

    print("\n=== 判定に使う模型値 ===")
    print(f"W1 の模型値（健常の傾き）      : a = {a_healthy:.4f} s（許容 ±0.33 = ±30%）")
    print(f"否定対照の模型値（凍結の傾き）  : a = {a_frozen:+.4f} s")
    print(f"分離比                          : {abs(a_healthy / a_frozen):.1f} 倍")
    print("\n注: 時定数 tau は 0.45→0.70 m/s で "
          f"{decay_length(0.45)[1]:.3f}→{decay_length(0.70)[1]:.3f} s（{(decay_length(0.70)[1] / decay_length(0.45)[1] - 1) * 100:+.1f}%）。")
    print("    速度で変わるのは時間ではなく距離である、というのが本カードの主張である。")


if __name__ == "__main__":
    main()
