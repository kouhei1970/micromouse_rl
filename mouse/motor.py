"""
mouse/motor.py
================
DC モータ（Mabuchi FA-130RA-18100 相当、3V定格）+ 減速ギアの電気モデル。

【参照実装であり、実行時のシミュレーションでは使用しない】
実行時は MuJoCo の <general> アクチュエータ（電気項: gainprm/biasprm）+
joint 属性（機械損失項: damping, frictionloss）が同一の式を物理サブステップ毎に
自動計算する（mouse/robot_v2.py 参照。mouse/sim.py の step_control() は
data.ctrl に電圧を書き込むだけで、Python 側のトルク計算ループは持たない）。

本モジュールは検証テスト（tests/test_mouse_v2.py, competition/model_verification.py）が
閉形式の期待値を「実行時にモデル読込値から導出する」ための参照実装として存在する
（docs/MODEL_VERIFICATION_PLAN.md §5 手順4: 数値ハードコード禁止の方針に対応）。

式（同計画書 §4.1）:
    tau_w(V, omega) = (N*Kt/R) * (V - Ke*N*omega) - b*omega - tau_c*sgn(omega)   [N・m]
    無負荷定常角速度: omega_ss(V) = ((N*Kt/R)*V - tau_c) / (N^2*Kt*Ke/R + b)      [rad/s]

V: 印加電圧指令 [V]（±voltage_limit にクランプ）、omega: 車輪角速度 [rad/s]、
N: 減速比、Kt: トルク定数 [N・m/A]、Ke: 逆起電力定数 [V・s/rad]、R: 巻線抵抗 [Ω]、
b: 車輪軸粘性減衰 [N・m・s/rad]、tau_c: 車輪軸乾性摩擦トルク [N・m]。
"""
import numpy as np


def wheel_torque(v_applied: float, omega_wheel: float, params) -> float:
    """電圧指令 v_applied [V] と車輪角速度 omega_wheel [rad/s] から
    車輪軸トルク [N・m] を計算する（参照実装。閉形式期待値の算出専用）。"""
    p = params
    v = float(np.clip(v_applied, -p.voltage_limit, p.voltage_limit))
    # 減速機を介した逆起電力（モータ軸換算の角速度は N * omega_wheel）
    back_emf = p.motor_Ke * p.gear_ratio * omega_wheel
    tau_electrical = p.gear_ratio * (p.motor_Kt / p.motor_R) * (v - back_emf)
    tau_viscous = p.wheel_damping * omega_wheel
    tau_coulomb = p.wheel_frictionloss * np.sign(omega_wheel)
    return float(tau_electrical - tau_viscous - tau_coulomb)


def omega_ss(v_applied: float, params) -> float:
    """定常（無負荷、空中無負荷試験相当）角速度 [rad/s] を計算する（参照実装）。

    tau_w(V, omega_ss) = 0 の解: omega_ss = ((N*Kt/R)*V - tau_c) / (N^2*Kt*Ke/R + b)
    （docs/MODEL_VERIFICATION_PLAN.md §6 Test0 の期待値式そのもの）。
    """
    p = params
    v = float(np.clip(v_applied, -p.voltage_limit, p.voltage_limit))
    numerator = p.gear_ratio * (p.motor_Kt / p.motor_R) * v - p.wheel_frictionloss
    denominator = (p.gear_ratio ** 2) * p.motor_Kt * p.motor_Ke / p.motor_R + p.wheel_damping
    return float(numerator / denominator)


class DCMotorModel:
    """DCMotorModel: wheel_torque/omega_ss の薄いクラスラッパー（後方互換用の参照実装）。

    実行時のトルク計算には使用しない（本モジュールのモジュール docstring 参照）。
    """

    def __init__(self, params):
        self.params = params

    def torque(self, v_applied: float, omega_wheel: float) -> float:
        return wheel_torque(v_applied, omega_wheel, self.params)

    def omega_ss(self, v_applied: float) -> float:
        return omega_ss(v_applied, self.params)
