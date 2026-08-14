#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**是正 (D)** — 斜めの分岐が落とした「速度計画の防御 3 点」を戻す
（`experiments/exp_016_diagonal/card_016diag_fixD.md`）。

2026-08-15 新設（教授裁定 = 是正案 (D) の検収合格・実装 GO）。

--------------------------------------------------------------------------
🔴 **これは「写し」である。委譲できるかを先に確かめた**（カード §1-1）
--------------------------------------------------------------------------
**親（`SlalomPolicy._replan`）は防御 3 点を持っているが、
それは `_replan` の【中にインラインで】書かれていて、独立した関数になっていない**
（`baseline_slalom.py:1205-1241`）。
**`build_speed_profile` は上限の配列を受け取るだけで、防御を含まない。**
**切り出すには親を書き換えるしかないが、`baseline_slalom.py` は
基準のスナップショットとして固定されている。**

**⇒ 写す。ただし「何を写したか」を下の表に一覧で残し、
`tests/test_terminal_speed_guard.py` が「写した防御が実際に効いていること」を検査する。**

--------------------------------------------------------------------------
写した防御の一覧（**これがすべて。これ以外は写さない**）
--------------------------------------------------------------------------
| # | 親の実装 | ここでの適用 |
|---|---|---|
| **①** | `end_reason == "continue"` なら `v_ceil_base = v_uncertain`（**配列全体**） | **`stop_at_end` が偽（経路が打ち切られている）なら、上限の配列全体を `v_uncertain` で頭打ちにする** |
| **②** | `v_end = 0.0 if stop_at_end else min(v_ceil_base, v_uncertain)` を終端手前 `cell_size/2` から適用 | **同じ**（`stop_at_end` が真なら `build_speed_profile` が終端 0 を作るので、ここでは打ち切り時だけ効く） |
| **③** | `d_brake = (v_ceil_base² − v_end²)/(2a_max)`。`s_term < d_brake` なら**配列全体を保守的な上限に固定** | **同じ**（**短い経路で全開加速 → 即全開制動の鋸歯を防ぐ**） |

**⚠️ 上限は常に `np.minimum` で「頭打ち」にする。**
**斜め区間・円弧区間の上限（$v_\text{斜め}$ = 0.45）を引き上げることはない。**

**⚠️ 親の `target_mode == "to_start"` の枝は写していない。**
**斜めの分岐は `to_goal` のときしか動かない**（`_use_diag()` の条件）ので、
**到達しない枝である。**

--------------------------------------------------------------------------
どこへ差し込むか — **親の `_replan` を写さない**
--------------------------------------------------------------------------
**斜めの分岐が呼ぶ `build_speed_profile` を、呼び出しの間だけ包んだものへ差し替える。**
**`finally` で必ず戻す**（`baseline_slalom_diag_cal.py` の参照経路の差し替えと同じ思想）。

**差し替えるのは `competition.baseline_slalom_diag` の名前だけ**なので、
**親自身の `_replan`（`competition.baseline_slalom` の名前を使う）には影響しない。**
"""
import sys

import numpy as np

from competition.baseline_slalom_diag import SlalomDiagPolicy


class TerminalSpeedGuardMixin:
    """速度計画の防御 3 点を、斜めの分岐にも効かせる混ぜ込み。"""

    def __init__(self, *args, terminal_guard: bool = True, **kw):
        super().__init__(*args, **kw)
        # False にすると混ぜ込む前とビット単位で同じ（無害性 D3 の確認用）
        self.terminal_guard = bool(terminal_guard)
        self._diag_mod_for_guard = sys.modules[SlalomDiagPolicy.__module__]
        # 報告用（**読むだけ**。挙動に影響しない）
        self.n_guard_capped = 0     # 防御 ① が上限を下げた回数
        self.n_guard_clamped = 0    # 防御 ③ が経路全体を固定した回数

    # ------------------------------------------------------------------
    def _guarded_speed_profile(self, s_arr, curv_arr, v_ceil, a_lat, a_max,
                               v_creep, stop_at_end):
        """親の防御 3 点を適用してから `build_speed_profile` を呼ぶ。"""
        base = self._diag_mod_for_guard
        orig = self._orig_speed_profile
        v = np.array(v_ceil, dtype=float, copy=True)
        if v.ndim == 0:
            v = np.full(len(s_arr), float(v_ceil))
        v_unc = float(self.v_uncertain)
        length = float(s_arr[-1])

        # --- 防御 ①: 打ち切られた経路は、この先が未知なので v_uncertain で頭打ち ---
        if not stop_at_end:
            before = v.copy()
            v = np.minimum(v, v_unc)
            if np.any(v < before - 1e-12):
                self.n_guard_capped += 1

        # --- 防御 ②: 終端で満たすべき速度 ---
        v_base = float(np.max(v)) if v.size else 0.0
        v_end = 0.0 if stop_at_end else min(v_base, v_unc)
        s_term = length if stop_at_end else max(length - self.cell_size / 2.0, 0.0)

        # --- 防御 ③: 落としきる距離が無ければ、経路全体を保守的な上限に固定 ---
        d_brake = max((v_base ** 2 - v_end ** 2) / (2.0 * a_max), 0.0)
        if s_term < d_brake:
            v = np.minimum(v, max(v_end, min(v_base, v_unc)))
            self.n_guard_clamped += 1
        elif not stop_at_end:
            tail = np.asarray(s_arr) >= s_term
            v[tail] = np.minimum(v[tail], v_end)

        return orig(s_arr, curv_arr, v, a_lat, a_max, v_creep, stop_at_end)

    # ------------------------------------------------------------------
    def _replan(self, x: float, y: float, yaw: float) -> None:
        """**親をそのまま呼ぶ。**速度計画の呼び出しだけを包んだものへ差し替える。"""
        if not self.terminal_guard:
            return super()._replan(x, y, yaw)
        mod = self._diag_mod_for_guard
        self._orig_speed_profile = mod.build_speed_profile
        mod.build_speed_profile = self._guarded_speed_profile
        try:
            return super()._replan(x, y, yaw)
        finally:
            mod.build_speed_profile = self._orig_speed_profile
