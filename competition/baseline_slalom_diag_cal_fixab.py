#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**是正 (A)** — 「この迷路を探索し終えたか」を走行をまたいで保つ
（`experiments/exp_016_diagonal/card_016diag_fixA.md`）。

2026-08-15 新設（教授裁定 = 是正案 (A) の検収合格・実装 GO）。

--------------------------------------------------------------------------
何を直すのか（カード §0）
--------------------------------------------------------------------------
**`_use_diag()` は「この迷路を探索し終えたか」（迷路水準の知識）を尋ねているのに、
`_explored_once` は「この走行でゴールへ到達したか」（走行水準の状態）しか表していない。**

**`_explored_once` は `_reset_run_state()` で偽に戻る**（`baseline_slalom.py:744`。
呼び出し元は `on_maze_start` と **`on_retrieval`**）。
**⇒ 係員回収が起きるたびに「探索し終えた」が忘れられる。**

**是正 (B) で回収は 80 回 → 2 回に減ったが 0 にはなっていない**
（`card_016diag_fixB.md` §6-2。計時走行の中の衝突が 2 件残る）。
**回収が 1 度でも起きれば、その迷路の残りの走行はすべて探索モードに落ちる。**

--------------------------------------------------------------------------
どう直すのか — **印を 1 つ足すだけ。既存の状態は触らない**（カード §1-1）
--------------------------------------------------------------------------
| 対象 | 扱い |
|---|---|
| **`_maze_explored`**（本モジュールが足す。**迷路水準**） | **`on_maze_start` でだけ偽に戻す。`on_retrieval` では戻さない** |
| `_explored_once`（走行水準） | **1 ビットも触らない**（判定の一瞬だけ広げて、`finally` で必ず戻す） |
| **親の先読み長 `max_cells`** | **触らない**（`_explored_once` を見ている） |

> ### 🔴 **`_explored_once` を「迷路水準」に作り替えない**
> **あれは親の先読み長も決めており、意味を変えると
> 対照（斜めなしの新既定）の挙動まで変わる。**
> **本カードは斜めの分岐の条件だけを直す。**
>
> **⇒ 回収の後も「先読み 2 区画・速度上限 `v_uncertain`」は残る。**
> **本カードはそこを直さない**（カード §4 の限定 2）。

--------------------------------------------------------------------------
判定の書き方（**親の式を写さない**）
--------------------------------------------------------------------------
`_use_diag()` は**親の式をそのまま呼ぶ**。
**判定している一瞬だけ `_explored_once` を「迷路水準の印との論理和」に見せ、
`finally` で必ず元へ戻す**（`baseline_slalom_diag_cal.py` の参照経路の差し替えと同じ思想）。

**親の式を写さないので、親が変わっても追随する。**
**`_explored_once` の値そのものは判定の外へ漏れない**ので、
**親の先読み長の計算には一切影響しない。**
"""
from competition.baseline_slalom_diag_cal_fixb import SlalomDiagCalFixBPolicy


class SlalomDiagCalFixABPolicy(SlalomDiagCalFixBPolicy):
    """是正 (B) ＋ **走行をまたぐ「探索済み」の印**。"""

    name = "L0-c+E1T+TR+DIAG+F0+F0b+cal0.75+clothoid45+alignB+mazeA"

    def __init__(self, *args, use_maze_flag: bool = True, **kw):
        super().__init__(*args, **kw)
        # False にすると是正 (B) 版とビット単位で同じ（無害性 A2 の確認用）
        self.use_maze_flag = bool(use_maze_flag)
        self._maze_explored = False
        # 報告用（**読むだけ**。挙動に影響しない）
        self.n_flag_saved_diag = 0    # 印のおかげで斜めの分岐が成立した回数

    # ------------------------------------------------------------------
    def on_maze_start(self, maze_info: dict) -> None:
        """**印を偽に戻すのはここだけ**（`on_retrieval` では戻さない）。"""
        out = super().on_maze_start(maze_info)
        self._maze_explored = False
        self.n_flag_saved_diag = 0
        return out

    # ------------------------------------------------------------------
    def _flip_target_mode(self) -> None:
        """ゴールへ到達した瞬間に印を立てる（親が `_explored_once` を立てるのと同じ場所）。"""
        if self.target_mode == "to_goal":
            self._maze_explored = True
        return super()._flip_target_mode()

    # ------------------------------------------------------------------
    def _use_diag(self) -> bool:
        """**親の式をそのまま呼ぶ。**判定の一瞬だけ印との論理和に見せる。

        `finally` で必ず戻すので、**`_explored_once` の値は判定の外へ漏れない**
        （親の先読み長の計算には一切影響しない）。
        """
        if not self.use_maze_flag:
            return super()._use_diag()
        saved = self._explored_once
        self._explored_once = bool(saved or self._maze_explored)
        try:
            out = super()._use_diag()
        finally:
            self._explored_once = saved
        # 「印が無ければ成立しなかった」回数を数える（**読むだけ**）
        if out and not saved:
            self.n_flag_saved_diag += 1
        return out
