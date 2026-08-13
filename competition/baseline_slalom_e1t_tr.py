#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0-c + **E1T** + TR — 確定判定も経路選択も「時間」で行う版。

2026-08-13 新設（exp_018・裁定 R36-2）。**`competition/baseline_slalom_e1_tr.py`
（対照 = exp_015 の処理条件）は変更しない** — 対照が再現できなくなると面ごとの
対応差が取れなくなるため（この系列で繰り返し効いている作法）。

--------------------------------------------------------------------------
1 実験 1 変更 — 変えるのは「何を確定とみなすか」だけ
--------------------------------------------------------------------------
親（L0-c+E1+TR）は**歩数**最短が確定するまで追加探索し、走るのは**時間**最短の
経路だった。**確定の基準と経路の基準が食い違っている**ので、時間モデルは
「壁が無いように見える」未探索領域へ引き寄せられ、初回最短走行が遠回りする
（exp_015 カード §6-5 の実測）。

本クラスは**確定判定と探索目標だけ**を時間の基準へ替える
（`competition/explore_e1t.py`）。**経路選択のアルゴリズム・制御ゲイン・
速度プロファイル・軌道生成・ロボットパラメータは親のまま。**

**追加探索の目標へ向かう道順は親のまま（歩数最短）**である点に注意
（`tr_modes` は既定の `("to_goal",)` のまま＝時間最短で走るのは計時される
最短走行だけ）。**替えるのは「どこまで探索を続けるか」であって「探索中の
走り方」ではない。**

使い方:
    .venv/bin/python -m competition.evaluator \\
        --policy competition.baseline_slalom_e1t_tr:SlalomE1TTRPolicy \\
        --maze-dir competition/mazes/eval \\
        --out-dir outputs/exp_018_time_confirmed_exploration/l0c_e1t_tr
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.baseline_slalom import goal_cells  # noqa: E402
from competition.baseline_slalom_e1_tr import SlalomE1TRPolicy  # noqa: E402
from competition.explore_e1t import (explore_targets_time,  # noqa: E402
                                      is_time_shortest_confirmed)


class SlalomE1TTRPolicy(SlalomE1TRPolicy):
    """L0-c + E1T + TR。確定判定・探索目標を時間の基準にする。"""

    name = "L0-c+E1T+TR (time-confirmed exploration, time-optimal route)"

    # ------------------------------------------------------------------
    # 目標集合 — `verify` の中身だけを時間の基準へ替える
    # ------------------------------------------------------------------
    def _target_cells(self):
        if self.target_mode == "to_goal":
            return goal_cells(self.width, self.height)
        if self.target_mode == "verify":
            t = explore_targets_time(self.v_walls_known, self.h_walls_known,
                                     self.width, self.height, (0, 0),
                                     goal_cells(self.width, self.height),
                                     self.cost_model)
            if t:
                return sorted(t)
            self.target_mode = "to_start"      # 確定済み → 帰路へ
        return [(0, 0)]

    # ------------------------------------------------------------------
    # 確定判定 — 歩数ではなく時間で見る（`_flip_target_mode` が親で呼ぶ）
    # ------------------------------------------------------------------
    def _shortest_confirmed(self) -> bool:
        return is_time_shortest_confirmed(self.v_walls_known, self.h_walls_known,
                                          self.width, self.height, (0, 0),
                                          goal_cells(self.width, self.height),
                                          self.cost_model)
