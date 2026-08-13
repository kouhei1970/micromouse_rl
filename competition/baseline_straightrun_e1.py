#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0-b（超信地旋回走行・直進連続）に **E1（追加探索）** を統合した版。

2026-08-13 新設（exp_014・レバー ③）。**`competition/baseline_straightrun.py`
（E1 なし版）は変更しない** — exp_013 で測った L0-b は本実験の対照であり、
再現できなくなると面ごとの対応差が取れなくなるため（E2 のときと同じ流儀。
`baseline_classical_e2.py` の冒頭を参照）。

--------------------------------------------------------------------------
なぜ E1 を足すのか
--------------------------------------------------------------------------
exp_013 の方式間対応差（§9-15 準拠）で、**L0-b と L0-c の (e') は完全に一致し
（対応差の中央値 +0.000・悪い面 0/20）、L0-a(E1) だけが良い**（−0.012）。
3 方式は経路決定ロジックが同一なので、**差は E1 の有無だけ**と考えられる。

E1 =「最短経路が確定するまで追加探索を続ける」（`competition/explore_e1.py`）。
ゴールに着いた時点で最短が確定していなければ、**「開いていたら最短経路に
使われうる未知壁」に隣接する区画**へ寄り道してから帰る。

--------------------------------------------------------------------------
⚠️ 期待できるのは (d) ではなく (e)（exp_014 カード §2）
--------------------------------------------------------------------------
**L0-b は最良走行が 15/20 面で既に真の最短を引けている**（L0-c は 20/20）。
走行を繰り返すうちに地図が育って最短に到達しているためで、**(d) が改善する
余地があるのは残り 5 面だけ**である。**E1 の価値は「速くすること」ではなく
「1 回で最短に到達すること」**であり、主効果は **(e) 初回最短走行効率**に出る。

--------------------------------------------------------------------------
実装の形
--------------------------------------------------------------------------
`StraightRunPolicy` を継承し、**`_target_cells` と `_do_plan` の目標反転だけ**を
上書きする。新規のアルゴリズムは書かない（`explore_e1` をそのまま呼ぶ）。

使い方:
    .venv/bin/python -m competition.evaluator \\
        --policy competition.baseline_straightrun_e1:StraightRunE1Policy \\
        --maze-dir competition/mazes/eval --out-dir outputs/exp_014_e1_integration/l0b_e1
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.baseline_straightrun import StraightRunPolicy, goal_cells, pos_to_cell  # noqa: E402
from competition.explore_e1 import explore_targets, is_shortest_confirmed  # noqa: E402


class StraightRunE1Policy(StraightRunPolicy):
    """L0-b + E1。`target_mode` に `verify`（追加探索）を追加する。"""

    name = "L0-b+E1 (straight-run with E1 extra exploration)"

    # ------------------------------------------------------------------
    # 目標集合 — `verify` 分岐を足す（baseline_classical.py L245-258 と同じ）
    # ------------------------------------------------------------------
    def _target_cells(self):
        if self.target_mode == "to_goal":
            return goal_cells(self.width, self.height)
        if self.target_mode == "verify":
            # 追加探索（E1）:「開いていたら最短経路に使われうる未知壁」に
            # 隣接する区画を目標にする。そこへ行って壁を観測すれば確定する
            t = explore_targets(self.v_walls_known, self.h_walls_known,
                                self.width, self.height, (0, 0),
                                goal_cells(self.width, self.height))
            if t:
                return sorted(t)
            # 確定済み → 帰路へ切り替え
            self.target_mode = "to_start"
        return [(0, 0)]

    def _shortest_confirmed(self) -> bool:
        """現在の楽観最短経路が真の最短経路として確定しているか（E1）。"""
        return is_shortest_confirmed(self.v_walls_known, self.h_walls_known,
                                     self.width, self.height, (0, 0),
                                     goal_cells(self.width, self.height))

    # ------------------------------------------------------------------
    # 目標到達時の遷移 — ゴール到達で「確定していなければ verify へ」
    #
    # 親の `_do_plan` は `to_goal ⇔ to_start` の**2 値反転**しか持たない。
    # そこで親を呼ぶ前に**3 値遷移**をここで済ませ、
    # **「現在区画が目標集合でなくなる」まで回してから**親へ渡す。
    #
    # ⚠️ 回し切るのが必須である。`verify` に切り替えた直後、現在区画が
    # そのまま探索目標になっている場合があり、その状態で親を呼ぶと
    # **親の 2 値反転が発火して `verify` → `to_goal` に飛ぶ**（探索が壊れる）。
    # 抜けた時点で親の `dist_field.get(cur_cell) == 0` は偽になるので、
    # 親側の反転は発火しない。
    # ------------------------------------------------------------------
    def _do_plan(self, x: float, y: float) -> None:
        cur_cell = pos_to_cell(x, y, self.width, self.height, self.cell_size)
        # 状態は 3 つなので有限回で必ず抜ける（安全弁として上限を置く）
        for _ in range(8):
            if self._flood_fill(self._target_cells()).get(cur_cell) != 0:
                break
            if self.target_mode in ("to_goal", "verify"):
                self.target_mode = "to_start" if self._shortest_confirmed() else "verify"
            else:
                self.target_mode = "to_goal"
        return super()._do_plan(x, y)
