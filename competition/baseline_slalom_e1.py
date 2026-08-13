#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0-c（スラローム走行）に **E1（追加探索）** を統合した版。

2026-08-13 新設（exp_014・レバー ③）。**`competition/baseline_slalom.py`
（E1 なし版）は変更しない** — exp_013 で測った L0-c は本実験の対照であり、
再現できなくなると面ごとの対応差が取れなくなるため。

--------------------------------------------------------------------------
⚠️ 期待できるのは (d) ではなく (e)（exp_014 カード §2）
--------------------------------------------------------------------------
**L0-c は最良走行が 20/20 面で既に真の最短を引けている。**5 走行を繰り返す
うちに地図が育って最短に到達しているためで、**(d) には改善の余地が無い。**
(e') が 1.034 なのは「**初回の**最短走行」を見ているからである。

**E1 の価値は「速くすること」ではなく「1 回で最短に到達すること」**であり、
主効果は **(e) 初回最短走行効率**（現状 1.378）に出る。競技上の意味は
**持ち時間の余裕と走行回数の節約**であって、最速タイムの短縮ではない。

--------------------------------------------------------------------------
`_explored_once` を親のまま（初回ゴールで True）にした理由
--------------------------------------------------------------------------
起票時のカード §3 には「`verify` 中は探索走行モードを維持する」と書いたが、
**コードを読んで誤りと分かったので変更した**（予測は変更していない）。

`_explored_once` が制御するのは**チェーンの先読み長だけ**である
（`explore_horizon_cells`=2 か 迷路全体か。L1121）。**未知壁へ高速で
突っ込まない安全弁は別の場所にあり、区間ごとに効いている** —
末尾区画の出口が未観測なら `end_reason == "continue"` となり、
その区間は `v_uncertain` に固定される（L1206 付近）。
`_build_chain` も `require_known_exit=True` で既知区画しか伸ばさない。

さらに `baseline_slalom.py` L1200-1203 のコメントは
「**E1 の追加探索は既知経路を通って未知の壁へ向かうので、この切り替えの
有無で探索時間が大きく変わる**」と、まさにこの統合を予期して書かれている。
**先読みを短いままにすると、既知区画を無駄に遅く走ることになる。**

--------------------------------------------------------------------------
実装の形
--------------------------------------------------------------------------
`SlalomPolicy` を継承し、**`_target_cells` と `_flip_target_mode` だけ**を
上書きする。L0-c は目標の反転が `_flip_target_mode` の 1 箇所に集約されて
いるので、L0-b より素直に差し替えられる。

使い方:
    .venv/bin/python -m competition.evaluator \\
        --policy competition.baseline_slalom_e1:SlalomE1Policy \\
        --maze-dir competition/mazes/eval --out-dir outputs/exp_014_e1_integration/l0c_e1
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.baseline_slalom import SlalomPolicy, goal_cells  # noqa: E402
from competition.explore_e1 import explore_targets, is_shortest_confirmed  # noqa: E402


class SlalomE1Policy(SlalomPolicy):
    """L0-c + E1。`target_mode` に `verify`（追加探索）を追加する。"""

    name = "L0-c+E1 (slalom with E1 extra exploration)"

    # ------------------------------------------------------------------
    # 目標集合 — `verify` 分岐を足す（baseline_classical.py L245-258 と同じ）
    # ------------------------------------------------------------------
    def _target_cells(self):
        if self.target_mode == "to_goal":
            return goal_cells(self.width, self.height)
        if self.target_mode == "verify":
            t = explore_targets(self.v_walls_known, self.h_walls_known,
                                self.width, self.height, (0, 0),
                                goal_cells(self.width, self.height))
            if t:
                return sorted(t)
            self.target_mode = "to_start"      # 確定済み → 帰路へ
        return [(0, 0)]

    def _shortest_confirmed(self) -> bool:
        """現在の楽観最短経路が真の最短経路として確定しているか（E1）。"""
        return is_shortest_confirmed(self.v_walls_known, self.h_walls_known,
                                     self.width, self.height, (0, 0),
                                     goal_cells(self.width, self.height))

    # ------------------------------------------------------------------
    # 目標の反転 — 2 値反転を 3 値遷移へ差し替える
    #   `to_goal` → （確定なら）`to_start` ／（未確定なら）`verify`
    #   `verify`  → （確定なら）`to_start` ／（未確定なら）`verify` のまま
    #   `to_start`→ `to_goal`
    # ------------------------------------------------------------------
    def _flip_target_mode(self) -> None:
        if self.target_mode == "to_goal":
            # 初回ゴール到達。先読みを迷路全体へ広げる（親と同じ。理由は冒頭）
            self._explored_once = True
            self.target_mode = "to_start" if self._shortest_confirmed() else "verify"
        elif self.target_mode == "verify":
            self.target_mode = "to_start" if self._shortest_confirmed() else "verify"
        else:
            self.target_mode = "to_goal"
