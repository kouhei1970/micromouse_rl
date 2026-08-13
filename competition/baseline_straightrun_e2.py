#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0-b + **E2（予算つき打ち切り）** — 持ち時間を見て追加探索をやめる版。

2026-08-13 新設（exp_017）。**`competition/baseline_straightrun_e1.py`（対照）は
変更しない** — exp_014 で測った L0-b+E1 は本実験の対照であり、再現できなくなると
面ごとの対応差が取れなくなるため（この系列で繰り返し効いている作法）。

--------------------------------------------------------------------------
なぜ要るのか
--------------------------------------------------------------------------
exp_014 で E1 を L0-b へ統合した結果、**(e') は 1.034 → 1.000 に直った一方、
maze_7837 で (b) を失った**（$D$=97。帯で最も長い面）。E1 が帰路を **+60.4 s**
延ばし、走行 2 の開始が 274.4 → **334.8 s** へずれ、**必要 85.9 s に対し
残り 85.2 s。0.6 s 足りずに timeout**（exp_014 カード §6-5）。

**E1 は「確定するまで探索を続ける」無条件の設計なので、持ち時間と正面から
競合する。**この費用は**計時窓の外（走行の隙間）に落ちるため走行タイムには
現れず、持ち時間が尽きた瞬間に「走行の消失」として全損で現れる**（`note_017`）。

**E2 はまさにこれを見て打ち切る。**判断の式は
`competition/baseline_classical_e2.py`（L0-a 用・2026-08-11 の教授裁定）と同一:

    残り持ち時間 < ( D_帰路 + D_最速 ) × s_区画 × (1 + 余裕)

--------------------------------------------------------------------------
再利用の形（コピーしない）— ただし 3 つだけ
--------------------------------------------------------------------------
予算判定の本体（`_seconds_per_cell` / `_known_distance` / `_budget_exhausted`）は
**`AdachiE2Policy` の関数をそのまま借りる**。3 つとも `super()` を呼ばない純粋な
判断ロジックなので、別の親を持つクラスへ移しても意味が変わらない。

**`__init__` / `on_maze_start` / `act` は借りない。**これらは `super()` を
（引数なしで）呼んでおり、引数なし `super()` は**定義されたクラス**を基準に解決
されるため、借りると `AdachiPolicy`（L0-a）側の実装へ飛んでしまう。
**同じ理由で、記録用の数行（経過時間と通過区画数）はここに書き直してある。**

使い方:
    .venv/bin/python -m competition.evaluator \\
        --policy competition.baseline_straightrun_e2:StraightRunE2Policy \\
        --maze-dir competition/mazes/eval --out-dir outputs/exp_017_budget_cutoff_l0b/l0b_e2
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.baseline_classical_e2 import (DEFAULT_MARGIN,  # noqa: E402
                                                DEFAULT_TIME_BUDGET, AdachiE2Policy)
from competition.baseline_straightrun import pos_to_cell  # noqa: E402
from competition.baseline_straightrun_e1 import StraightRunE1Policy  # noqa: E402


class StraightRunE2Policy(StraightRunE1Policy):
    """L0-b + E2。追加探索（`verify`）を予算で打ち切る。"""

    name = "L0-b+E2 (straight-run, budget-aware cutoff)"

    # --- 予算判定の本体は L0-a 用の実装を借りる（コピーしない） ---
    _seconds_per_cell = AdachiE2Policy._seconds_per_cell
    _known_distance = AdachiE2Policy._known_distance
    _budget_exhausted = AdachiE2Policy._budget_exhausted

    def __init__(self, *args, time_budget=None, margin=DEFAULT_MARGIN, **kwargs):
        super().__init__(*args, **kwargs)
        self._time_budget_arg = time_budget
        self._margin = float(margin)
        self.time_budget = DEFAULT_TIME_BUDGET
        self._t0 = None            # 最初に act が呼ばれた時刻 [s]
        self._cells_moved = 0      # 通過した区画数（s_区画 の分母）
        self._last_cell = None
        self._cut_at = None        # 打ち切った時刻 [s]（報告用。None なら打ち切っていない）

    # ------------------------------------------------------------------
    def on_maze_start(self, maze_info: dict) -> None:
        super().on_maze_start(maze_info)
        self.time_budget = float(
            self._time_budget_arg if self._time_budget_arg is not None
            else maze_info.get("time_budget", DEFAULT_TIME_BUDGET))
        self._t0 = None
        self._cells_moved = 0
        self._last_cell = None
        self._cut_at = None

    # ------------------------------------------------------------------
    def act(self, obs):
        # s_区画 の実測に使う量を更新してから本体へ渡す（読むだけで出力は変えない）
        sim = getattr(self, "_sim", None)
        if sim is not None:
            if self._t0 is None:
                self._t0 = sim.sim_time
            x, y, _yaw = sim.privileged_pose()
            cell = pos_to_cell(x, y, self.width, self.height, self.cell_size)
            if self._last_cell is not None and cell != self._last_cell:
                self._cells_moved += 1
            self._last_cell = cell
        return super().act(obs)

    # ------------------------------------------------------------------
    def _target_cells(self):
        """追加探索中に予算が尽きていれば、帰路へ切り替えてから親へ委ねる。"""
        if self.target_mode == "verify" and self._budget_exhausted():
            self.target_mode = "to_start"
            if self._cut_at is None and getattr(self, "_sim", None) is not None:
                self._cut_at = self._sim.sim_time
        return super()._target_cells()
