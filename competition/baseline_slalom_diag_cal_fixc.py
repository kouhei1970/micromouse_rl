#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**是正 (C)** — 係員回収の後も先読みの長さを保つ
（`experiments/exp_016_diagonal/card_016diag_fixC.md`）。

2026-08-15 新設（教授裁定 = 是正案 (ii) の検収合格・実装 GO）。

--------------------------------------------------------------------------
何を直すのか（カード §0）
--------------------------------------------------------------------------
**`_explored_once` が偽のとき、親は先読みを `explore_horizon_cells` = 2 区画に絞る**
（`baseline_slalom.py:1121`）。
**`_explored_once` は係員回収（`on_retrieval` → `_reset_run_state`）で偽に戻る。**

**⇒ 回収の後の走行は、37 点（≈ 0.19 m ＝ 約 1 区画）の短い経路を
繰り返し張りながら走ることになる。**
**短い経路には「その先にある円弧のための減速」が入りようがない**ので、
**速度計画は $v_\text{cap}$ = 2.895 のままになり、機体は計画どおり加速して、
拘束が現れたときには止まれない**（`card_016diag_switch.md` の原因究明 第 2 段）。

--------------------------------------------------------------------------
どう直すのか — **是正 (A) の印を、先読み長の判定にも使う**
--------------------------------------------------------------------------
**是正 (A) が足した `_maze_explored`（`on_maze_start` でだけ偽に戻る迷路水準の印）を、
先読みの長さを決める箇所でも見る。**

**⚠️ 「地図を知っているのに先読みを 2 区画に絞る」のは設計と食い違った状態である。**
**本カードは対症ではなく、本来あるべき状態へ戻す是正である**（教授の整理）。

| 対象 | 扱い |
|---|---|
| 先読み長を決める箇所（親の `_build_chain` の呼び出し） | **`_explored_once` または `_maze_explored`** で判定させる |
| **`_explored_once` そのもの** | **1 ビットも触らない**（判定の一瞬だけ広げ、`finally` で必ず戻す） |
| 対照（斜めなしの新既定） | **回収が起きないので両者は常に同値** → **実効不変**（C2 で実測確認） |

**是正 (A) と同じ手口である**（`baseline_slalom_diag_cal_fixab.py`）。
**親の式を写さないので、親が変わっても追随する。**

--------------------------------------------------------------------------
本カードで直さないもの
--------------------------------------------------------------------------
🔴 **「速度計画が経路の先を見ない（終端速度が $v_\text{cap}$ のまま）」という
構造の穴そのものは直さない。**
**共有最内ループに近く、確定基準表（14.690 s）を支える既定の挙動を変えるため、
影響範囲に見合う手続きで別途行う**（教授裁定 2026-08-15。
`card_016g.md` §9 の申し送り 6「終端速度の防御」）。
"""
from competition.baseline_slalom_diag_cal_fixab import SlalomDiagCalFixABPolicy


class SlalomDiagCalFixCPolicy(SlalomDiagCalFixABPolicy):
    """是正 (A)(B) ＋ **回収の後も先読みの長さを保つ**。"""

    name = "L0-c+E1T+TR+DIAG+F0+F0b+cal0.75+clothoid45+alignB+mazeA+horizonC"

    def __init__(self, *args, keep_horizon: bool = True, **kw):
        super().__init__(*args, **kw)
        # False にすると是正 (A) 版とビット単位で同じ（無害性 C2 の確認用）
        self.keep_horizon = bool(keep_horizon)
        # 報告用（**読むだけ**。挙動に影響しない）
        self.n_horizon_kept = 0    # 印のおかげで先読みが長いままだった回数

    # ------------------------------------------------------------------
    def _replan(self, x: float, y: float, yaw: float) -> None:
        """**親をそのまま呼ぶ。**先読み長の判定の間だけ `_explored_once` を広げる。

        `finally` で必ず戻すので、**`_explored_once` の値は `_replan` の外へ漏れない**。
        **`_use_diag()` の判定は是正 (A) が別途行う**ので、二重に効くことはない
        （どちらも同じ「印との論理和」を見るだけ）。
        """
        if not (self.keep_horizon and self._maze_explored and not self._explored_once):
            return super()._replan(x, y, yaw)
        saved = self._explored_once
        self._explored_once = True
        self.n_horizon_kept += 1
        try:
            return super()._replan(x, y, yaw)
        finally:
            self._explored_once = saved
