#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0-c + E1T + TR + **F0（速度ループの是正）** — exp_016 段階 F0 の退行確認用。

2026-08-14 新設。**親（`baseline_slalom_e1t_tr.py`）は 1 行も変更しない**
（基準スナップショットで帰属を守る。カード `card_016f0.md` §5-1）。

`competition/velocity_loop.py` の `VelocityLoopMixin` を混ぜ込むだけで、
**経路決定・探索・速度計画・操舵は親のまま**である。差し替わるのは
`_wheel_targets_to_voltage`（＝**速度ループの最内側**）だけ。

用途: **カード §5-1 の基準スナップショット (γ)** の是正後の側。
設計帯 20 面での走行タイムを、是正前（`SlalomE1TTRPolicy`）と**面ごとに対応をとって**
比較する（§9-15）。**これは退行確認の代理測定であって、参照線の更新ではない**
（参照線 15.06 s と M5 の凍結表は動かさない。教授裁定 2026-08-14）。
"""
from competition.baseline_slalom_e1t_tr import SlalomE1TTRPolicy
from competition.velocity_loop import VelocityLoopMixin

# 物理から導いた全量（1.0 = J_eff·dω/dt をそのまま前置補償する）。
# **調整可能な自由度としては使わない**（1 実験 1 変更。カード §3 の (iv)）。
K_ACC_FF_DESIGN = 1.0


class SlalomE1TTRF0Policy(VelocityLoopMixin, SlalomE1TTRPolicy):
    """L0-c+E1T+TR に**速度ループの是正だけ**を足したもの。"""

    name = "L0-c+E1T+TR+F0 (accel feedforward in the wheel velocity loop)"

    def __init__(self, *a, **kw):
        kw.setdefault("k_acc_ff", K_ACC_FF_DESIGN)
        super().__init__(*a, **kw)
