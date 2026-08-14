#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**是正 (D) を載せた方策 2 種**（`experiments/exp_016_diagonal/card_016diag_fixD.md` §1-3）。

**教授指示「(D) 単独で判定 → その後 (C) を載せる」に合わせ、切り分けられる形にしてある。**

| クラス | 中身 | 用途 |
|---|---|---|
| **`SlalomDiagCalFixDPolicy`** | (A) ＋ (B) ＋ **(D)** | **(D) 単独の判定**（(C) は載っていない） |
| **`SlalomDiagCalFixCDPolicy`** | (A) ＋ (B) ＋ (C) ＋ **(D)** | **最終形**（P1〜P5 の測り直しに使う） |

- **(A)** 走行をまたぐ「探索済み」の印（`baseline_slalom_diag_cal_fixab.py`）
- **(B)** 斜めへ入る前の向き合わせ（`baseline_slalom_diag_cal_fixb.py`）
- **(C)** 回収の後も先読みの長さを保つ（`baseline_slalom_diag_cal_fixc.py`）
- **(D)** 速度計画の防御 3 点（`terminal_speed_guard.py`）

**作法 1 に従い、既存の方策ファイルは 1 行も変更していない。**

**混ぜ込みは一番外側に置く** — **防御は最後に掛かるべき**だからである
（内側の実装が上限を決めた後で、その上限を頭打ちにする）。
"""
from competition.baseline_slalom_diag_cal_fixab import SlalomDiagCalFixABPolicy
from competition.baseline_slalom_diag_cal_fixc import SlalomDiagCalFixCPolicy
from competition.terminal_speed_guard import TerminalSpeedGuardMixin


class SlalomDiagCalFixDPolicy(TerminalSpeedGuardMixin, SlalomDiagCalFixABPolicy):
    """(A) ＋ (B) ＋ **(D)**。**(C) は載っていない**（(D) 単独の判定用）。"""

    name = "L0-c+DIAG+F0+F0b+cal0.75+clothoid45+alignB+mazeA+guardD"


class SlalomDiagCalFixCDPolicy(TerminalSpeedGuardMixin, SlalomDiagCalFixCPolicy):
    """(A) ＋ (B) ＋ (C) ＋ **(D)** — **最終形**。"""

    name = "L0-c+DIAG+F0+F0b+cal0.75+clothoid45+alignB+mazeA+horizonC+guardD"
