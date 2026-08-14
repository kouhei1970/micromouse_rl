#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""追従制御の刷新 第 1 弾 — **前方注視つき 2 自由度化 ＋ レートダンピング**（exp_016-F）。

2026-08-14 新設。**`competition/baseline_slalom.py` は変更しない**
（**基準スナップショット**で帰属を守る。016-F カード §1）。

--------------------------------------------------------------------------
現行則の欠陥（P6 §2-1(2)）と、本モジュールが潰すもの
--------------------------------------------------------------------------
現行:

    ω_ref = κ(s_cursor)·v_ref + k_ψ·e_ψ − arctan( k_y·e_y / (|v| + v_eps) )

| 欠陥 | 中身 | 本モジュール |
|---|---|---|
| **F1** | 曲率 FF が**現在位置**の κ。直線 → 円弧で κ がステップし、ω は τ_w の遅れを持つので**必ず遅れる** | **(a) 前方注視**: **τ_la 先の κ** を使う |
| **F2** | e_ψ に**ダンピングが無い**（P のみ）。ジャイロは観測にあるのに使っていない | **(b) レートダンピング**: −k_r(ω_z − ω_ff) |

改良:

    ω_ff  = κ(s_cursor + v_ref·τ_la)·v_ref
    ω_ref = ω_ff + k_ψ·e_ψ − arctan( k_y·e_y / (|v| + v_eps) ) − k_r·(ω_z − ω_ff)

**ω_ff を基準にレートを引く**ので、**定常旋回では効かず、遅れと振動にだけ効く**
（定常偏差を作らない）。

--------------------------------------------------------------------------
使い方（**混ぜ込み**で既存の方策へ足す）
--------------------------------------------------------------------------
    class MyPolicy(TwoDofControlMixin, SlalomPolicy):
        pass
    p = MyPolicy(tau_la=0.03, k_r=0.2)

**τ_la は推定しない。**`experiments/exp_016_diagonal/identify_tau.py` で
**内側ループの時定数 τ_w を実測**してから決める（016-F カード §2-1）。
"""
import math

import numpy as np


class TwoDofControlMixin:
    """`_do_drive_control` だけを差し替える混ぜ込み。**他は親のまま。**"""

    def __init__(self, *args, tau_la: float = 0.0, k_r: float = 0.0, **kw):
        super().__init__(*args, **kw)
        self.tau_la = float(tau_la)      # 前方注視時間 [s]（0 なら現行と同じ）
        self.k_r = float(k_r)            # レートダンピングの係数（0 なら現行と同じ）

    # ------------------------------------------------------------------
    def _preview_curvature(self, idx: int, v_ref: float) -> float:
        """**τ_la 先**の曲率を返す（弧長で前方注視する）。"""
        path = self._path
        if self.tau_la <= 0.0:
            return float(path.curvature[idx])
        s_ahead = float(path.s[idx]) + v_ref * self.tau_la
        j = int(np.searchsorted(path.s, s_ahead))
        j = min(max(j, 0), len(path.s) - 1)
        return float(path.curvature[j])

    # ------------------------------------------------------------------
    def _do_drive_control(self, obs, x: float, y: float, yaw: float):
        """親と同じ構造のまま、**FF を前方注視へ差し替え、レート項を足す**。"""
        path = self._path
        idx = self._cursor
        px, py = float(path.x[idx]), float(path.y[idx])
        phead = float(path.heading[idx])
        v_target = float(path.speed[idx])

        # 速度指令のレートリミット（親と同一。上げ方向のみ）
        if self._v_setpoint < v_target:
            self._v_setpoint = min(v_target, self._v_setpoint + self.a_max * self.control_dt)
        else:
            self._v_setpoint = v_target
        v_ref = self._v_setpoint

        ux, uy = math.cos(phead), math.sin(phead)
        e_y = (x - px) * (-uy) + (y - py) * ux
        e_psi = math.atan2(math.sin(phead - yaw), math.cos(phead - yaw))

        v_fwd, omega_z = self._sim.privileged_velocity()
        v_for_gain = max(abs(v_fwd), 1e-6)
        lateral_term = math.atan2(self.k_y * e_y, v_for_gain + self.v_eps)

        # (a) 前方注視つきフィードフォワード
        omega_ff = self._preview_curvature(idx, v_ref) * v_ref
        # (b) レートダンピング（ω_ff を基準に引くので定常偏差を作らない）
        rate_term = self.k_r * (omega_z - omega_ff)

        omega_ref = omega_ff + self.k_psi * e_psi - lateral_term - rate_term
        omega_ref = max(-self.turn_omega_limit, min(self.turn_omega_limit, omega_ref))
        return self._wheel_targets_to_voltage(v_ref, omega_ref, obs)
