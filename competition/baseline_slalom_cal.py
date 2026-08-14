#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**素の L0-c ＋ 校正済みの旋回安全率 0.75** — M5 gate の確定基準表を作る条件。

2026-08-14 新設。**016-cal（旋回安全率の校正実験）の切り替えで、
M5 gate の基準を作る条件が「素の L0-c・安全率 0.70」から
「素の L0-c・**安全率 0.75**」へ動いた**（`card_016cal_switch.md` §7-3）。

--------------------------------------------------------------------------
なぜ新しいファイルを作るのか（作法 1）
--------------------------------------------------------------------------
**`SlalomPolicy.__init__` の既定値 `safety_factor=0.7` は変更しない。**
変えると `baseline_slalom.py` が基準のスナップショットでなくなり、
**過去の全記録をビット単位で再現できなくなる**。
`baseline_slalom_e1t_tr_f0b_cal.py`（新しい既定の走行方策）とまったく同じ作りで、
**継承して安全率の既定だけを差し替える。**

--------------------------------------------------------------------------
何のために要るのか
--------------------------------------------------------------------------
`card_016cal_switch.md` §7-2 の**申し送り**（教授承認 2026-08-14）:

> **(e′)（経路効率の指標 = 計時窓で通過した区画の数 ÷ 真の最短歩数）の照合**を
> `exp_013/run_arm.py` 系で行う（同スクリプトは `n_cells` を記録する）。
> **確保済みの評価用 20 迷路 × 素の L0-c・安全率 0.75 を 1 回。**

**`run_arm.py` は方策を引数なしで組み立てる**（`load_policy` が
`getattr(module, cls)()` を呼ぶ）ので、**安全率を渡す口が無い**。
そこで**既定値として持つクラス**を用意する。
**`run_arm.py` は完了済み実験のスクリプトなので 1 行も変更しない。**

⚠️ **これは走行方策としての「新しい既定」ではない。**
古典トラックの走行の既定は `baseline_slalom_e1t_tr_f0b_cal.py` の
`SlalomE1TTRF0bCalPolicy`（E1T ＋ TR ＋ F0 ＋ F0-b 込み）である。
**本クラスは「M5 gate の基準を作るための素の条件」専用**である。
"""
from competition.baseline_slalom import SlalomPolicy
from competition.baseline_slalom_e1t_tr_f0b_cal import SAFETY_FACTOR_CALIBRATED


class SlalomCalPolicy(SlalomPolicy):
    """素の L0-c（**安全率だけが親と違う**）。

    校正値は `baseline_slalom_e1t_tr_f0b_cal` から**引いて使う**。
    ここに 0.75 を書き写すと、校正値が改訂されたときに 2 か所へ分かれる。
    """

    name = "l0c_slalom (calibrated safety factor 0.75)"

    def __init__(self, *a, **kw):
        kw.setdefault("safety_factor", SAFETY_FACTOR_CALIBRATED)
        super().__init__(*a, **kw)
