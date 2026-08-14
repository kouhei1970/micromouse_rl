#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参照速度を**弧長で内挿して読む**（exp_016 段階 **F0-b**）。

2026-08-14 新設。**`competition/baseline_slalom.py` は変更しない**
（速度ループは全方策が共有する最内ループなので、基準スナップショットで帰属を守る。
カード `card_016f0b.md` §3）。

--------------------------------------------------------------------------
何を直すのか（**D1 の実測で決めた。決め打ちではない**）
--------------------------------------------------------------------------
参照の時間微分は定義上こう分解できる:

    d v_cmd / dt = ( dv/ds ) · ( ds_cursor / dt )

**速度計画が保証しているのは左の因子だけ**（$|dv/ds|\\cdot v \\leq a_\\max$。実測でも
空間上の要求減速度の最大は $a_\\max$ ちょうど・超過 0 件）。
**右の因子（カーソルが 1 ティックに進む弧長）は誰も保証していない。**

D1（設計帯 20 面 × 梯子 7 速度 × 2 harness）の判別:

| 仮説 | 判定 | 根拠 |
|---|---|---|
| **H_quant** 参照の空間量子化 | **支持** | 閉じた形の照合（$ds$/走行 = 5.09/3.0 = 1.70 対 実測 1.627）と、交番が始まる速度の一致（$ds/\\Delta t$ = 0.509 m/s） |
| H_jump 境目でカーソルが飛ぶ | **偽** | 境目 p99 ≤ 境目以外 p99 が全速度・両 harness で成立 |
| H_term 終端の残差 | 支持 | 終端が超過の最大の 55〜64% を作る（**本モジュールでは触らない**。裁定 5-1） |

**参照は最悪の 1% のティックで機体の 1.9 倍の速さで進んでいた**（比の p99 = 1.75〜1.96）。
**格子点の値をそのまま読むのをやめれば、この項は構成上消える。**

--------------------------------------------------------------------------
何をするか（**カード §2-1 で実装より前に固定した定義**）
--------------------------------------------------------------------------
    s_proj   = s[i] + (x − p_x)·cos ψ_i + (y − p_y)·sin ψ_i      （i = cursor）
    v_target = sqrt( max(0, interp(s_proj; s, v_plan²)) )

- `s_proj` は `[s[0], s[-1]]` へ挟む
- **`s_proj` は、親が横偏差 e_y に使っている座標系のもう一方の成分**である
  （親: `e_y = (x−p_x)(−sin ψ_i) + (y−p_y) cos ψ_i`）。**同じ枠組みを使う**
- **v ではなく v² を内挿する**: 速度計画は**等減速・等加速の区間で v² が s に厳密に線形**
  （`build_speed_profile` の前後 2 パスが v[i]² = v[i±1]² ± 2·a_max·ds を作る）。
  **v² の線形内挿はその区間で計画の連続形を厳密に再構成する**（dv/ds も厳密）。
  **超過が生じているのはまさに a_max 律速の区間**なので、ここが厳密であることが効く。
  曲率律速の区間は κ が区分定数で v も定数なので、v 内挿と v² 内挿は一致する

**曲率・方位・経路点の位置（＝操舵が使う量）は親のまま**である。
**操舵ループ（016-F）とは別のループなので混ぜない。**

--------------------------------------------------------------------------
使い方（**混ぜ込み**。既存ファイルは触らない）
--------------------------------------------------------------------------
    class MyPolicy(ReferenceInterpMixin, VelocityLoopMixin, SlalomPolicy):
        pass
    p = MyPolicy(ref_interp=True, k_acc_ff=1.0)

**`ref_interp=False`（既定）なら親へそのまま委譲する**ので、**現行と 1 ビットも変わらない**
（`tests/test_reference_interp.py` が全走行のビット一致で確認する）。
"""
import math

import numpy as np

from competition.baseline_slalom import _wrap_pi


class ReferenceInterpMixin:
    """`_do_drive_control` だけを差し替える混ぜ込み。**他は親のまま。**

    ⚠️ **`_do_drive_control` は親の本体の写しに 1 箇所の差し替えを入れたものである。**
    親を書き換えれば済む話だが、**`baseline_slalom.py` は基準スナップショットとして
    凍結している**（カード §3）。**写しが親からずれると気づけない**ので、
    `tests/test_reference_interp.py` が **`ref_interp=False` で親と全走行ビット一致すること**
    を検査する（親を書き換えたらテストが落ちる）。
    `competition/control_2dof.py` の `TwoDofControlMixin` と同じ作りである。
    """

    def __init__(self, *args, ref_interp: bool = False, **kw):
        super().__init__(*args, **kw)
        self.ref_interp = bool(ref_interp)   # False なら親へ委譲（現行と同一）
        self._v_plan_sq = None               # 速度計画の 2 乗（内挿用のキャッシュ）
        self._v_plan_src = None              # キャッシュがどの配列に対するものか

    # ------------------------------------------------------------------
    def _speed_at(self, idx: int, x: float, y: float) -> float:
        """**弧長で内挿した**参照速度 [m/s]（カード §2-1 の定義そのまま）。"""
        path = self._path
        # v² は経路を張り替えるたびに作り直す（同一オブジェクトなら再利用）
        if self._v_plan_src is not path.speed:
            self._v_plan_sq = np.asarray(path.speed, dtype=float) ** 2
            self._v_plan_src = path.speed

        psi = float(path.heading[idx])
        s_proj = (float(path.s[idx])
                  + (x - float(path.x[idx])) * math.cos(psi)
                  + (y - float(path.y[idx])) * math.sin(psi))
        s_proj = min(max(s_proj, float(path.s[0])), float(path.s[-1]))
        v_sq = float(np.interp(s_proj, path.s, self._v_plan_sq))
        return math.sqrt(max(0.0, v_sq))

    # ------------------------------------------------------------------
    def _do_drive_control(self, obs, x: float, y: float, yaw: float):
        # **既定（ref_interp = False）は親へそのまま委譲する** — 1 ビットも変わらない
        if not self.ref_interp:
            return super()._do_drive_control(obs, x, y, yaw)

        path = self._path
        idx = self._cursor
        px, py = float(path.x[idx]), float(path.y[idx])
        phead = float(path.heading[idx])
        pcurv = float(path.curvature[idx])
        v_target = self._speed_at(idx, x, y)      # ← 差し替えたのはここだけ

        # 速度指令のレートリミット（親と同一。上げ方向のみ）
        if self._v_setpoint < v_target:
            self._v_setpoint = min(v_target, self._v_setpoint + self.a_max * self.control_dt)
        else:
            self._v_setpoint = v_target
        v_ref = self._v_setpoint

        ux, uy = math.cos(phead), math.sin(phead)
        rel_x, rel_y = x - px, y - py
        e_y = rel_x * (-uy) + rel_y * ux
        e_psi = _wrap_pi(phead - yaw)   # **親の実装をそのまま呼ぶ**（写しを増やさない）

        v_fwd, _omega_z = self._sim.privileged_velocity()
        v_for_gain = max(abs(v_fwd), 1e-6)
        lateral_term = math.atan2(self.k_y * e_y, v_for_gain + self.v_eps)

        omega_ref = pcurv * v_ref + self.k_psi * e_psi - lateral_term
        omega_ref = max(-self.turn_omega_limit, min(self.turn_omega_limit, omega_ref))
        return self._wheel_targets_to_voltage(v_ref, omega_ref, obs)
