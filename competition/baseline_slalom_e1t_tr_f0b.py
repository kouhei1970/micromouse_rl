#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0-c + E1T + TR + **F0 + F0-b** — exp_016 段階 F0-b の退行確認用。

2026-08-14 新設。**親（`baseline_slalom_e1t_tr.py`）も `baseline_slalom_e1t_tr_f0.py` も
1 行も変更しない**（基準スナップショットで帰属を守る。カード `card_016f0b.md` §3）。

混ぜ込みは 2 つで、**差し替えるメソッドが違う**:

| 段階 | 混ぜ込み | 差し替える先 |
|---|---|---|
| **F0** | `VelocityLoopMixin` | `_wheel_targets_to_voltage`（加速度の前置補償） |
| **F0-b** | `ReferenceInterpMixin` | `_do_drive_control` の**参照の読み方だけ**（弧長内挿） |

**経路決定・探索・速度計画・操舵の則は親のまま**である。

用途: **カード §3 の (γ) 相当**（設計帯 20 面の走行タイム）の F0-b 側。
**対照は F0 適用後**（`SlalomE1TTRF0Policy`）であり、**F0 前ではない**（1 実験 1 変更）。
**これは退行確認の代理測定であって、参照線の更新ではない**
（参照線 15.06 s と M5 の凍結表は動かさない。教授裁定 2026-08-14）。
"""
from competition.baseline_slalom_e1t_tr import SlalomE1TTRPolicy
from competition.reference_interp import ReferenceInterpMixin
from competition.velocity_loop import VelocityLoopMixin

# F0 の設計値（物理から導いた全量）。**調整可能な自由度としては使わない**
K_ACC_FF_DESIGN = 1.0


class SlalomE1TTRF0bPolicy(ReferenceInterpMixin, VelocityLoopMixin, SlalomE1TTRPolicy):
    """L0-c+E1T+TR に **F0（速度ループ）と F0-b（参照の弧長内挿）**を足したもの。"""

    name = "L0-c+E1T+TR+F0+F0b (accel feedforward + arclength-interpolated reference)"

    def __init__(self, *a, **kw):
        kw.setdefault("k_acc_ff", K_ACC_FF_DESIGN)
        kw.setdefault("ref_interp", True)
        super().__init__(*a, **kw)
