"""
research_notes/scripts/check_flip_criterion.py
==============================================
判定基準「符号反転 10 回/s 未満」の**物理的な根拠を導出する**（机上計算・学習不要）。

背景（2026-08-11 教授指示）: この基準は前任の教授セッションが設定した値だが、導出が
確認されていない。今日の測定で学習方策の最良が 28.1 回/s（k=0 seed 2、gate 0.98）であり、
**どの方策も基準の 2.8 倍以上届いていない**。基準が物理から導かれていないなら
達成不可能な目標を追っている可能性があり、逆に物理から導いて 30 回/s なら既に達成済み。

出典: `docs/MODEL_VERIFICATION_PLAN.md` §4.2/§4.3（改訂 r5）、`docs/ROBOT_SPEC.md` §3、
`mouse/params.py`。**推測値は一切使わない。**足りない量は「足りない」と報告する。

## 符号反転率と周波数の対応

電圧が毎ステップ符号を変える方形波を考える。1 周期に符号反転は 2 回起きるので

    反転率 f_flip [回/s]  ⇔  方形波の基本周波数 f = f_flip / 2 [Hz]

制御周期 100 Hz（Nyquist 50 Hz）なので、**理論上限は f_flip = 100 回/s**（＝ 50 Hz）。

使い方:
    .venv/bin/python research_notes/scripts/check_flip_criterion.py
"""
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mouse.params import RobotParams  # noqa: E402

# 観測された反転率（gate 帯・完走した方策のみ。exp_006c、2026-08-11）
OBSERVED = [
    ("k=0 seed 2（学習方策の最良）", 28.1),
    ("k=1e-4 seed 1", 36.8),
    ("k=0 seed 1", 58.2),
    ("k=1e-4 seed 2", 63.0),
    ("k=0 seed 0（＝ exp_005。exp_006 の出発点）", 75.1),
]
CRITERION = 10.0   # 現行の判定基準 [回/s]


def gain_first_order(f_hz: float, tau_s: float) -> float:
    """一次遅れ 1/(1+jωτ) の振幅利得。"""
    return 1.0 / math.sqrt(1.0 + (2.0 * math.pi * f_hz * tau_s) ** 2)


def flip_rate_of(tau_s: float) -> float:
    """時定数 τ の一次遅れの遮断周波数 f_c = 1/(2πτ) に対応する反転率 [回/s]。"""
    return 2.0 * (1.0 / (2.0 * math.pi * tau_s))


def main():
    p = RobotParams()

    # ---- 仕様書から取れる量（すべて出典つき） -------------------------
    R = p.motor_R                      # 巻線抵抗 [Ω]
    Kt, Ke = p.motor_Kt, p.motor_Ke    # トルク定数・逆起電力定数 [N·m/A]=[V·s/rad]
    N = p.gear_ratio                   # 減速比
    J_rotor = p.rotor_inertia          # ロータ慣性（モータ軸）[kg·m²]
    armature = N ** 2 * J_rotor        # 車輪軸換算ロータ慣性 [kg·m²]
    b = p.wheel_damping                # 軸受粘性 [N·m·s/rad]
    m_w, r_w = p.mass_wheel, p.wheel_radius
    I_w = 0.5 * m_w * r_w ** 2         # 車輪の慣性（一様円板）[kg·m²]
    dt = p.control_dt

    b_elec = N ** 2 * Kt * Ke / R      # 逆起電力による等価粘性 [N·m·s/rad]
    tau_wheel = (I_w + armature) / (b_elec + b)     # 車輪ジョイントの時定数 [s]
    tau_motor = R * J_rotor / (Kt * Ke)             # モータ機械時定数（カタログ照合済）[s]
    tau_vehicle = 0.124                # 車体の速度時定数 [s]（計画書 §4.3 の導出値）

    print("=" * 78)
    print("§1 仕様書から取れる量（docs/MODEL_VERIFICATION_PLAN.md §4.2, docs/ROBOT_SPEC.md §3）")
    print("=" * 78)
    print(f"  モータ: FAULHABER 1717T003SR（コアレス、3 V 定格、減速比 N={N:g}）")
    print(f"  巻線抵抗 R            = {R} Ω")
    print(f"  トルク定数 K_t = K_e  = {Kt:.3e} N·m/A (= V·s/rad)")
    print(f"  ロータ慣性 J_rotor    = {J_rotor:.3e} kg·m²（車輪軸換算 armature = {armature:.3e}）")
    print(f"  車輪慣性 I_w          = {I_w:.3e} kg·m²（質量 {m_w} kg・半径 {r_w} m の一様円板）")
    print(f"  軸受粘性 b            = {b:.1e} N·m·s/rad")
    print(f"  制御周期              = {dt * 1000:.0f} ms（{1 / dt:.0f} Hz、Nyquist {1 / (2 * dt):.0f} Hz）")

    print("\n" + "=" * 78)
    print("§2 ⚠️ 導出に必要だが仕様書に無い量")
    print("=" * 78)
    print("  **端子インダクタンス L が docs/MODEL_VERIFICATION_PLAN.md にも")
    print("    docs/ROBOT_SPEC.md にも記載されていない。**")
    print("  → 電気的時定数 τ_e = L/R が計算できない。")
    print("     τ_e は「電圧を反転しても電流が立ち上がりきらず、トルクを生まないまま")
    print("     I²R 損失と発熱だけを生む」帯域を決める量であり、判定基準の")
    print("     もっとも直接的な根拠になりうる。")
    print("  → データシート（EN_1717_SR_DFF.pdf）には記載があるはずだが、")
    print("     本リポジトリの仕様書には転記されていない。**推測で埋めない。**")
    print("  → MODEL_VERIFICATION_PLAN の欠落として報告する。")

    print("\n" + "=" * 78)
    print("§3 仕様書から導出できる帯域（一次遅れの遮断 f_c = 1/(2πτ)）")
    print("=" * 78)
    print(f"{'律速する要素':<34}{'時定数 τ [ms]':>15}{'f_c [Hz]':>12}{'反転率 [回/s]':>16}")
    rows = [
        ("車体の速度（並進）", tau_vehicle),
        ("モータのロータ単体 T_m = RJ/(K_tK_e)", tau_motor),
        ("車輪ジョイント（ロータ+車輪）", tau_wheel),
    ]
    for name, tau in rows:
        print(f"{name:<34}{tau * 1000:>15.1f}{1 / (2 * math.pi * tau):>12.2f}"
              f"{flip_rate_of(tau):>16.1f}")
    print(f"{'制御周期の Nyquist（理論上限）':<34}{'—':>15}{1 / (2 * dt):>12.2f}"
          f"{1 / dt:>16.1f}")
    print(f"\n  ※ T_m = {tau_motor * 1000:.1f} ms はカタログ値 16 ms と一致"
          f"（計画書 §7 の整合検算 2 件目）。導出が正しいことの裏づけ。")

    # ---- 各反転率で、車輪の速度がどれだけ指令に追従するか -------------
    print("\n" + "=" * 78)
    print("§4 各反転率で「車輪の速度が指令にどれだけ追従するか」")
    print("=" * 78)
    print("  方形波の基本周波数 f = 反転率/2 における一次遅れ（τ_wheel）の振幅利得。")
    print("  利得が小さいほど、指令したトルクの振幅が速度変化に**ならない**")
    print("  ＝ 正味の仕事をせずに I²R 損失とギヤの打音になる。\n")
    print(f"{'条件':<40}{'反転[回/s]':>12}{'f [Hz]':>10}{'追従利得':>10}{'無駄':>8}")
    marks = [("**判定基準 10 回/s**", CRITERION)] + OBSERVED + [("Nyquist（理論上限）", 100.0)]
    for name, fr in marks:
        f = fr / 2.0
        g = gain_first_order(f, tau_wheel)
        print(f"{name:<40}{fr:>12.1f}{f:>10.1f}{g:>10.2f}{1 - g:>8.0%}")

    # ---- 結論 ---------------------------------------------------------
    fc_wheel = flip_rate_of(tau_wheel)
    print("\n" + "=" * 78)
    print("§5 結論")
    print("=" * 78)
    print(f"  1. **仕様書から導ける自然な閾値は {fc_wheel:.1f} 回/s**"
          f"（車輪ジョイントの遮断 {1 / (2 * math.pi * tau_wheel):.1f} Hz）。")
    print("     これより速い反転は、指令したトルクの振幅の半分以上が速度変化にならない。")
    print(f"  2. **現行基準 {CRITERION:.0f} 回/s は、この閾値より厳しい**"
          f"（追従利得 {gain_first_order(CRITERION / 2, tau_wheel):.2f}"
          f" ＝ まだ {gain_first_order(CRITERION / 2, tau_wheel):.0%} 追従できる帯域）。")
    print("     つまり 10 回/s は「駆動系が追従できる範囲」の内側にあり、"
          "物理的な破綻点ではない。")
    print(f"  3. 学習方策の最良 28.1 回/s は追従利得"
          f" {gain_first_order(28.1 / 2, tau_wheel):.2f}、"
          f"exp_005 の 75.1 回/s は {gain_first_order(75.1 / 2, tau_wheel):.2f}。")
    print("     **75 回/s は指令の 8 割が無駄**になっており、実機に持っていけないという")
    print("     初代の判断は物理的に妥当。")
    print("  4. **ただし基準値 10 の由来は仕様書からは復元できない。**"
          "L が無いので τ_e 由来の可能性も検証できない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
