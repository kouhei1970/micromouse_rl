#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0-c + E1T + TR + **DIAG** — 斜めを含む経路で最短走行する版（exp_016-D）。

2026-08-14 新設。**`competition/baseline_slalom_e1t_tr.py`（対照）は変更しない**
— exp_018 で測った L0-c+E1T+TR は本実験の対照であり、再現できなくなると
面ごとの対応差が取れなくなるため（この系列で繰り返し効いている作法）。

--------------------------------------------------------------------------
1 実験 1 変更 — 変えるのは「経路の選択肢に斜めを入れるか」だけ
--------------------------------------------------------------------------
**計時される最短走行（`to_goal` かつ探索済み）でのみ**、経路を
`experiments/exp_016_diagonal/diagonal_model.py` の**半区画格子・8 方位**で引き、
軌道を `diag_path.build_diagonal_path`（直線 → 円弧 → 斜め）で生成する。

**探索走行・E1T の追加探索・帰路は親のまま。制御も親のまま**
（Stanley・車輪 PI・`build_speed_profile` に手を入れない）。

--------------------------------------------------------------------------
⚠️ 確定判定は据え置く（016-D カード §1。**既知のずれ**）
--------------------------------------------------------------------------
E1T が確定させるのは**直進の格子の上の時間最短経路**であって、
**斜めを含む経路の壁ではない**。したがって exp_015 と同じ機構で
**初回最短走行が未確認の壁のある側へ引き寄せられうる**。
**これは D6 として予測に登録済みであり、隠していない。**

**安全弁**: 軌道を生成するのは**壁が既知の区画だけ**に限る（`require_known_exit`
と同じ考え方）。未知の区画に当たったらそこで軌道を打ち切り、
**その手前まで走ってから引き直す**（`_path_end_reason = "continue"`）。
**楽観のまま突っ込まない。**

--------------------------------------------------------------------------
凍結した設計値（016-D カード §2。実行後に動かさない）
--------------------------------------------------------------------------
接続の円弧 $R$ = 60 mm ／ 斜め区間の速度上限 0.45 m/s ／ 直進は $v_{cap}$ ／
経路選択の速度比 $r$ = 0.814 ／ 45° 費用 18.1 ms
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from competition.baseline_slalom import build_speed_profile, goal_cells, pos_to_cell  # noqa: E402
from competition.baseline_slalom_e1t_tr import SlalomE1TTRPolicy  # noqa: E402
from competition.route_planner import value_field  # noqa: E402

from diag_path import build_diagonal_path  # noqa: E402
from diagonal_model import (DELTA8, DIAGONALS, DiagonalGridModel,  # noqa: E402
                            cell_center_node, node_kind)

# --- 016-D カード §2 で凍結した設計値 ---
R_ARC_M = 0.060
V_DIAG_MPS = 0.45
R_SPEED_RATIO = 0.814
B45_S = 0.0181


class SlalomDiagPolicy(SlalomE1TTRPolicy):
    """L0-c + E1T + TR + DIAG。最短走行だけ斜めを含む経路で走る。"""

    name = "L0-c+E1T+TR+DIAG (diagonal fast run)"

    def __init__(self, *args, r_speed=R_SPEED_RATIO, b45=B45_S,
                 v_diag=V_DIAG_MPS, arc_radius=R_ARC_M, **kw):
        super().__init__(*args, **kw)
        self.v_diag = float(v_diag)
        self.arc_radius = float(arc_radius)
        # 経路選択の費用モデル（実測値。ハードコードせず引数で受ける）
        self.diag_model = DiagonalGridModel(self.time_a, self.time_b,
                                            r=float(r_speed), turn_unit_45=float(b45))
        self.n_diag_plans = 0        # 斜め経路を引いた回数（報告用）
        self.n_diag_truncated = 0    # 未知の区画で打ち切った回数（報告用）

    # ------------------------------------------------------------------
    def _use_diag(self) -> bool:
        """斜めで走るのは、計時される最短走行だけ。"""
        return bool(self._explored_once) and self.target_mode == "to_goal"

    def _node_committable(self, node) -> bool:
        """その節点へ**軌道を張ってよいか**（＝経路が横切る壁が観測済みで開いているか）。

        **区画中心の節点は無条件で可**（そこに壁は無い）。
        **壁スロットの中点の節点は、その壁が「既知かつ開いている」ときだけ可。**

        ⚠️ 経路の**選択**は親と同じく楽観（未知＝通行可）で行い、
        **軌道化の commitment だけを既知に限る**（`require_known_exit` と同じ考え方）。
        **区画の 4 辺すべてを要求するのは過剰**である — 斜めが横切るのは
        入口と出口の 2 辺だけで、区画の中に障害物は無い。
        """
        k = node_kind(node)
        if k == "C":
            return True
        u, v = node
        if k == "V":
            return int(self.v_walls_known[u // 2, (v - 1) // 2]) == 0
        if k == "H":
            return int(self.h_walls_known[(u - 1) // 2, v // 2]) == 0
        return False

    def _plan_diag(self, start_cell, heading):
        """半区画格子で目標までの経路を引き、**未知の区画の手前で打ち切る**。

        Returns: (nodes, dirs, truncated) — 軌道化できる範囲だけ
        """
        targets = goal_cells(self.width, self.height)
        field = value_field(targets, self.width, self.height,
                            self._connects_known, self.diag_model)
        node = cell_center_node(tuple(start_cell))
        d_in = heading if heading in DELTA8 else "N"
        nodes, dirs = [node], []
        for _ in range(4 * self.width * self.height):
            if field.states.get((node, d_in), float("inf")) <= 1e-9:
                return nodes, dirs, False          # 目標に到達
            best = None
            for d_out, nxt, st, w in self.diag_model.successors(
                    node, d_in, self.width, self.height, self._connects_known):
                v = field.states.get(st)
                if v is None:
                    continue
                if best is None or w + v < best[0] - 1e-12:
                    best = (w + v, d_out, nxt)
            if best is None:
                return nodes, dirs, False
            _c, d_out, nxt = best
            # 横切る壁が「既知かつ開いている」ことを確かめてから軌道化する
            if not self._node_committable(nxt):
                return nodes, dirs, True
            nodes.append(nxt)
            dirs.append(d_out)
            node, d_in = nxt, d_out
        return nodes, dirs, True

    # ------------------------------------------------------------------
    def _replan(self, x: float, y: float, yaw: float) -> None:
        if not self._use_diag():
            return super()._replan(x, y, yaw)

        cur_cell = pos_to_cell(x, y, self.width, self.height, self.cell_size)
        if node_kind(cell_center_node(cur_cell)) != "C":
            return super()._replan(x, y, yaw)

        nodes, dirs, truncated = self._plan_diag(cur_cell, self._heading_dir)
        if len(dirs) < 2:
            return super()._replan(x, y, yaw)      # 斜めにならない → 親に委ねる

        try:
            path, kinds, _idx = build_diagonal_path(
                nodes, dirs, self.cell_size, self.arc_radius,
                stop_at_end=not truncated)
        except ValueError:
            # 接続が張れない配置（半歩 + 90°）。**黙って落とさず親へ委ねる**
            return super()._replan(x, y, yaw)

        self.n_diag_plans += 1
        self.n_diag_truncated += int(truncated)

        v_arr = np.where(kinds == "straight", self.v_cap, self.v_diag)
        v_arr = np.minimum(v_arr, self.v_cap)
        stop_at_end = not truncated
        path.speed = build_speed_profile(path.s, path.curvature, v_arr,
                                          self.a_lat, self.a_max, self.v_creep,
                                          stop_at_end)
        self._path = path
        self._cursor = 0
        self._path_ticks = 0
        self._path_end_reason = "goal" if not truncated else "continue"
        self._state = "DRIVE"
