#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""速度ループの是正（exp_016 段階 **F0**）— **加速度（慣性）の前置補償**。

2026-08-14 新設。**`competition/baseline_slalom.py` は変更しない**
（速度ループは**全方策が共有する最内ループ**なので、基準スナップショットで帰属を守る。
カード `card_016f0.md` §1・§5-1）。

--------------------------------------------------------------------------
何を直すのか（**D0 の実測で決めた。決め打ちではない**）
--------------------------------------------------------------------------
D0（`experiments/exp_016_diagonal/run_016f0_diag.py`・設計帯 20 面 × 梯子 7 速度）で、
超過 v_act − v_cmd を**層K（運動学・滑り）**と**層W（車輪ループの追従誤差）**に分解した:

| 仮説 | 結果 |
|---|---|
| H_ff 前置補償が定常で過大 | **偽**（直進/定常の層W は −0.0020〜−0.0000 m/s） |
| H_kin 運動学の食い違い | **偽**（層K は定常で ≤ 0.001 m/s） |
| **H_lag 過渡の遅れ** | **支持**（超過は加減速の間だけ現れ、層W が 85〜94% を担う） |

**現行の逆モデル前置補償は静的である**（`WheelPI.step`）:

    V_ff = Ke_eff·ω_ref + inv_gain·( b·ω_ref + τc·sgn(ω_ref) )

**定常負荷しか埋めていないので、加減速に要るトルクは PI が全部背負う。**
Kp = 0.05 V/(rad/s) は小さいので、**加減速の間だけ追従が遅れる**。

**量まで合っている**（前向きに計算した値と実測の照合）:

    J_eff · dω/dt に要る電圧 = inv_gain · J_eff · (a_max·安全率 / r) = 0.364 V
    これを P 項だけで作るなら e_ω = 0.364 / Kp = 7.3 rad/s → 7.3·r = 0.098 m/s の遅れ
    実測の直進/加速の遅れ = 0.065〜0.090 m/s（積分が一部を埋めるぶん小さくなる向き）

--------------------------------------------------------------------------
足す項
--------------------------------------------------------------------------
    ΔV = k_acc_ff · inv_gain · J_eff · ( dv_cmd/dt ) / r        … 左右輪に**同じ量**

    J_eff = N²·J_rotor          （= RobotParams.armature）
          + (1/2)·m_wheel·r²    （車輪自身の慣性。円柱）
          + (1/2)·m_total·r²    （**片輪が負う機体の並進慣性**）

**m_total はソースの記載を合算せず、MuJoCo モデルの機体サブツリーから実行時に読む**
（カード §3 の追記・教授条件②。ハードコード禁止）。

**⚠️ 第 3 項は「左右輪が対称に前後加速を負う」というモデル上の仮定である**
（旋回中は成り立たない）。**仮定であることを明示して使い、効いたかどうかは G2 で判定する。**

**前後加速（共通モード）にだけ足し、旋回（差動モード）には足さない。**
**操舵ループ（016-F）とは別のループなので混ぜない**（カード §1）。

--------------------------------------------------------------------------
使い方（**混ぜ込み**で既存の方策へ足す。既存ファイルは触らない）
--------------------------------------------------------------------------
    class MyPolicy(VelocityLoopMixin, SlalomPolicy):
        pass
    p = MyPolicy(k_acc_ff=1.0)

**k_acc_ff = 0.0（既定）なら親へそのまま委譲する**ので、**現行と 1 ビットも変わらない**
（`tests/test_velocity_loop.py` が全走行のビット一致で確認する）。
"""
import mujoco

from competition.baseline_slalom import WheelPI


class WheelPIAccelFF(WheelPI):
    """`WheelPI` に**加速度前置補償の電圧 `v_ff_extra`** を 1 項足しただけの車輪制御器。

    ⚠️ **`step` は親の本体の写しに 1 項を足したものである。**
    親を書き換えれば済む話だが、**`baseline_slalom.py` は基準スナップショットとして
    凍結している**ので写した（カード §5-1）。**写しが親からずれると気づけない**ので、
    `tests/test_velocity_loop.py` が **`v_ff_extra = 0` で親と完全一致すること**を
    無作為の入力列で検査する（親を書き換えたらテストが落ちる）。
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.v_ff_extra = 0.0     # 呼び出し側が step の直前に設定する [V]

    def step(self, omega_ref: float, omega_act: float, dt: float) -> float:
        sgn = 1.0 if omega_ref > 1e-9 else (-1.0 if omega_ref < -1e-9 else 0.0)
        ff = (self.Ke_eff * omega_ref
              + self.inv_gain * (self.damping * omega_ref + self.friction_torque * sgn)
              + self.v_ff_extra)                      # ← 足したのはこの 1 項だけ
        e = omega_ref - omega_act
        unclamped = ff + self.kp * e + self.ki * self.integral
        v = max(-self.voltage_limit, min(self.voltage_limit, unclamped))
        pushing_further = (v >= self.voltage_limit and e > 0.0) or \
                          (v <= -self.voltage_limit and e < 0.0)
        if not pushing_further:
            self.integral = max(-self.int_clamp, min(self.int_clamp, self.integral + e * dt))
        return v


def subtree_mass(model, body_name: str = "mouse") -> float:
    """`body_name` を根とするサブツリーの質量和 [kg]（**モデルから読む**）。"""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    total = 0.0
    for i in range(model.nbody):
        b = i
        while b != 0:
            if b == root:
                total += float(model.body_mass[i])
                break
            b = int(model.body_parentid[b])
    return total


class VelocityLoopMixin:
    """`_wheel_targets_to_voltage` だけを差し替える混ぜ込み。**他は親のまま。**"""

    def __init__(self, *args, k_acc_ff: float = 0.0, **kw):
        super().__init__(*args, **kw)
        self.k_acc_ff = float(k_acc_ff)   # 0 なら現行と同一（親へ委譲する）
        self._prev_v_cmd = None
        self._J_eff = None
        self._pi_accel = None

    # ------------------------------------------------------------------
    def bind_sim(self, sim) -> None:
        super().bind_sim(sim)
        if self.k_acc_ff == 0.0:
            return
        p = sim.params
        r = self.wheel_radius
        m_total = subtree_mass(sim.model, "mouse")
        # J_eff = 回転子（armature） + 車輪自身 + 片輪が負う機体の並進慣性
        self._J_eff = (p.armature
                       + 0.5 * p.mass_wheel * r * r
                       + 0.5 * m_total * r * r)
        self._m_total = m_total
        # 親と同じ引数で作った、加速度項つきの車輪制御器（親の器は使わない）
        int_clamp = self.int_clamp_frac * self.voltage_limit / max(self.ki_wheel, 1e-9)
        self._pi_accel = (
            WheelPIAccelFF(self.Ke_eff, self.inv_gain, self.wheel_damping,
                           self.wheel_frictionloss, self.kp_wheel, self.ki_wheel,
                           self.voltage_limit, int_clamp),
            WheelPIAccelFF(self.Ke_eff, self.inv_gain, self.wheel_damping,
                           self.wheel_frictionloss, self.kp_wheel, self.ki_wheel,
                           self.voltage_limit, int_clamp))

    # ------------------------------------------------------------------
    def _reset_run_state(self):
        super()._reset_run_state()
        self._prev_v_cmd = None
        if getattr(self, "_pi_accel", None) is not None:
            for pi in self._pi_accel:
                pi.reset()

    # ------------------------------------------------------------------
    def _accel_ff_voltage(self, v_cmd: float) -> float:
        """前後加速の前置補償電圧 [V]（左右輪に**同じ量**を足す）。

        dv_cmd/dt は**指令の後退差分**で取る。走行の張り替えや状態遷移で指令が跳ぶと
        微分が発散するので、**モデル量 `a_max_measured` で挟む**（決め打ちの数値は置かない）。

        **なぜ `a_max`（計画の 3.92）ではなく `a_max_measured`（物理の 5.6）で挟むのか**:
        参照は `a_max` で作られているが、**カーソルは実速度で進む**ので
        dv_cmd/dt = (dv/ds)·v_act となり、**実速度が参照を超えている間は a_max を超える**
        （F0 が直そうとしている当の現象）。物理の上限で挟めば、
        **正当な要求を削らずに、張り替えの跳びだけを落とせる**。
        """
        prev = self._prev_v_cmd
        self._prev_v_cmd = v_cmd
        if prev is None or getattr(self, "_state", None) != "DRIVE":
            return 0.0
        dv = (v_cmd - prev) / self.control_dt
        lim = self.a_max_measured
        dv = max(-lim, min(lim, dv))
        return self.k_acc_ff * self.inv_gain * self._J_eff * (dv / self.wheel_radius)

    # ------------------------------------------------------------------
    def _wheel_targets_to_voltage(self, v_cmd: float, omega_cmd: float, obs):
        # **既定（k_acc_ff = 0）は親へそのまま委譲する** — 1 ビットも変わらない
        if self.k_acc_ff == 0.0:
            return super()._wheel_targets_to_voltage(v_cmd, omega_cmd, obs)

        r, tread = self.wheel_radius, self.tread
        omega_l_des = v_cmd / r - omega_cmd * tread / (2.0 * r)
        omega_r_des = v_cmd / r + omega_cmd * tread / (2.0 * r)
        omega_l_act = float(obs[self._i_wheel])
        omega_r_act = float(obs[self._i_wheel + 1])

        dv_volt = self._accel_ff_voltage(v_cmd)
        pi_l, pi_r = self._pi_accel
        pi_l.v_ff_extra = dv_volt
        pi_r.v_ff_extra = dv_volt
        vl = pi_l.step(omega_l_des, omega_l_act, self.control_dt)
        vr = pi_r.step(omega_r_des, omega_r_act, self.control_dt)
        return vl, vr
