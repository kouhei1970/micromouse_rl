"""L0-a + E2（予算つき追加探索）— 持ち時間を見て探索を打ち切る古典ベースライン
Adachi + budget-aware exploration: stop verifying before the fast run no longer fits.

2026-08-11 新設。**`competition/baseline_classical.py`（E1 版）は変更しない**
（E1 版は凍結記録として残し、E2 の効果を測る対照にするため）。

## なぜ要るのか

exp_007 の実測（大会実迷路 18 面・L0-a）で、持ち時間 420 s の内訳はこうだった:

    探索走行 141.3 s ／ **E1 追加探索 191.7 s（最大の消費）** ／ 帰路 38.0 s ／ 残り 58.1 s
    最速走行に必要な時間: 77.8 s

**確定には到達している（15/18 面）。ただし確定が持ち時間の 73% 時点と遅すぎて、
残り時間に最速走行が入らない。**18 面中 13 面で最速走行が成立しなかった。

E1 は「最短経路が確定するまで追加探索を続ける」という**無条件の**設計なので、
**「確定の保証」と「持ち時間」が正面から競合する**。実際の競技者は competing しない:
**残り時間で最速走行が 1 本入るかを見て、入らなくなる前に打ち切る。**

## 打ち切り条件（教授裁定 2026-08-11）

**時刻や訪問率ではなく「予算」で決める。**各計画時に次を評価する:

    残り持ち時間 < ( T_帰路 + T_最速 ) × (1 + マージン)
      T_帰路 = 既知地図での「現在地 → スタート」の最短距離 × s_区画
      T_最速 = 既知地図での「スタート → ゴール」の最短距離 × s_区画

成立したら `target_mode` を `verify` から `to_start` へ切り替える
（＝追加探索を打ち切り、既知情報での最良経路に切り替える）。

**s_区画（1 区画あたりの所要時間）はハードコードしない。**その走行で実測した
「経過時間 ÷ 通過区画数」を使う。方策の外から定数を与えずに済み、
「数値のハードコード禁止」（研究計画書 §9）にも合う。走るほど精度が上がる。

距離は**悲観地図**（未知壁 = 壁）で測る。帰路と最速走行は実際に悲観的に計画される
ため（E1 の楽観・悲観の非対称性。`baseline_classical._connects_known` 参照）、
見積りも同じ前提で取るのが整合的。悲観で到達不能なら楽観にフォールバックする。

## 実装の形

`AdachiPolicy` を継承し、**`_target_cells` だけを上書き**する。`_do_plan` が毎計画時に
`_target_cells()` を呼ぶので、そこで予算を評価すれば追加探索の途中でも切り替わる。
**探索・制御・軌道生成には一切触らない**ので、E1 版との差は打ち切り条件だけになる。

使い方:
    .venv/bin/python -m competition.evaluator \\
        --policy competition.baseline_classical_e2:AdachiE2Policy \\
        --maze-dir competition/mazes/eval
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from competition.baseline_classical import AdachiPolicy, goal_cells, pos_to_cell  # noqa: E402
from competition.explore_e1 import bfs_distances  # noqa: E402

# 公式プロトコルの持ち時間 [s]。`maze_info["time_budget"]` があればそちらを優先する
# （評価器が渡してくれる。古い評価器と組み合わせたときのための既定値）。
DEFAULT_TIME_BUDGET = 420.0
# 見積りに対する安全率。実測で「残り 67.3 s に対し最速走行に 77.8 s 必要」＝
# 10.5 s（約 15%）足りなかったので、同じ水準の余裕を置く。
DEFAULT_MARGIN = 0.15


class AdachiE2Policy(AdachiPolicy):
    """L0-a に「予算で追加探索を打ち切る」判断を足したもの（E2）。"""

    name = "l0a_e2_budget"

    def __init__(self, *args, time_budget=None, margin=DEFAULT_MARGIN, **kwargs):
        super().__init__(*args, **kwargs)
        self._time_budget_arg = time_budget
        self._margin = float(margin)
        self._t0 = None            # 最初に act が呼ばれた時刻 [s]
        self._cells_moved = 0      # 通過した区画数（s_区画 の分母）
        self._last_cell = None
        self._cut_at = None        # 打ち切った時刻 [s]（記録用。None なら打ち切っていない）

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
        # s_区画 の実測に使う量を更新してから本体へ渡す
        sim = getattr(self, "_sim", None)
        if sim is not None:
            t = sim.sim_time
            if self._t0 is None:
                self._t0 = t
            x, y, _yaw = sim.privileged_pose()
            cell = pos_to_cell(x, y, self.width, self.height, self.cell_size)
            if self._last_cell is not None and cell != self._last_cell:
                self._cells_moved += 1
            self._last_cell = cell
        return super().act(obs)

    # ------------------------------------------------------------------
    def _seconds_per_cell(self):
        """その走行で実測した 1 区画あたりの所要時間 [s]。まだ測れなければ None。"""
        sim = getattr(self, "_sim", None)
        if sim is None or self._t0 is None or self._cells_moved < 4:
            return None            # 分母が小さいうちは信用しない
        return (sim.sim_time - self._t0) / self._cells_moved

    def _known_distance(self, sources, target):
        """既知地図での sources → target の最短距離。悲観 → 楽観の順に試す。"""
        for pessimistic in (True, False):
            d = bfs_distances(self.v_walls_known, self.h_walls_known,
                              self.width, self.height, sources, pessimistic=pessimistic)
            if target in d:
                return d[target]
        return None

    def _budget_exhausted(self):
        """残り持ち時間で「帰路 + 最速走行」が入らないなら True。"""
        sim = getattr(self, "_sim", None)
        s = self._seconds_per_cell()
        if sim is None or s is None:
            return False
        remaining = self.time_budget - sim.sim_time
        if remaining <= 0:
            return True
        x, y, _yaw = sim.privileged_pose()
        cur = pos_to_cell(x, y, self.width, self.height, self.cell_size)
        start = (0, 0)
        goals = goal_cells(self.width, self.height)
        d_back = self._known_distance([start], cur)                 # 現在地 → スタート
        d_fast = self._known_distance(goals, start)                 # スタート → ゴール
        if d_back is None or d_fast is None:
            return False           # 見積れないときは打ち切らない（E1 の挙動を保つ）
        need = (d_back + d_fast) * s * (1.0 + self._margin)
        return remaining < need

    # ------------------------------------------------------------------
    def _target_cells(self):
        """追加探索中に予算が尽きていれば、帰路へ切り替えてから本体へ委ねる。"""
        if self.target_mode == "verify" and self._budget_exhausted():
            self.target_mode = "to_start"
            if self._cut_at is None and getattr(self, "_sim", None) is not None:
                self._cut_at = self._sim.sim_time
        return super()._target_cells()
