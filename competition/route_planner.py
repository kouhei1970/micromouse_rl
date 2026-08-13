#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""状態空間（区画 × 進行方向）上の**時間最小**経路 — exp_015（レバー ①）の中核。

--------------------------------------------------------------------------
何を解くのか
--------------------------------------------------------------------------
従来の足立法（`_flood_fill`）は**歩数**（移動区画数）を最小化する。しかし
実測の時間モデルは

    時間 [s] = a · (移動区画数) + b · (旋回回数)

であり、**旋回にも時間がかかる**（L0-c の実測で a=0.1554 s/区画・
b=0.1358 s/折れ、`research_notes/data/time_model_l0c_design.json`）。
したがって「歩数は多いが折れが少ない経路」が**時間では速い**ことが起こる。

旋回コストを扱うには、状態を**区画だけ**にしてはいけない。同じ区画でも
「どちらを向いて入ってきたか」で次の一手の費用が変わるからである。そこで
状態を **(区画, 進行方向)** に拡張し、Dijkstra で目標からの最小時間を解く。

--------------------------------------------------------------------------
⚠️ コストモデルは差し替えられる形にしてある（教授裁定 2026-08-13-R29）
--------------------------------------------------------------------------
exp_016（斜め走行）では**状態空間そのものが変わる**（斜め区間を持つため、
節点は区画中心だけでなく区画の境界にも置かれ、進行方向は 8 方位になる）。
そこで本モジュールは Dijkstra 本体（`value_field`）を状態空間の中身から
切り離し、**モデル側が状態・遷移・費用を定義する**契約にした。

モデルが実装すべきもの（`StraightGridModel` が直進格子版の実装例）:

  - `target_states(targets)`   目標に居ることを表す状態の列挙（値 0 の起点）
  - `predecessors(state, width, height, connects)`
                              後ろ向き Dijkstra 用。(前の状態, 辺の費用)
  - `successors(cell, d_in, width, height, connects)`
                              貪欲降下用。(出口方位, 次区画, 次状態, 辺の費用)
  - `cell_of(state)`          その状態がどの区画に居るか

`value_field` は上の 4 つしか呼ばない。**斜め走行の追加はモデルの差し替えで
済み、Dijkstra 本体も方策側の呼び出しも変えなくてよい。**

--------------------------------------------------------------------------
未知壁の扱い（重要）
--------------------------------------------------------------------------
通行可否は呼び出し側から渡される `connects(x, y, nx, ny)` に**全面的に委ねる**。
方策側は `_connects_known`（**未知壁は通行可＝楽観的**）をそのまま渡すこと。
ここを `_flood_fill` と揃えないと、時間最短経路が歩数最短経路と別の地図の
上で引かれることになり、比較が成立しない（exp_015 カード §4-1）。
"""
import heapq

# ==========================================================================
# 4 方位・座標規約（`competition/baseline_slalom.py` と同一。並びは時計回り）
# ==========================================================================
DIRS = ("N", "E", "S", "W")
DELTA = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}

# 費用の同点判定に使う許容値 [s]。浮動小数の加算順序の違いで生じる差
# （~1e-15 s）だけを吸収し、実在する費用差（最小でも 1e-3 s の桁）は
# 同点にしない。
TIE_EPS = 1e-9


def turn_count(d_in: str, d_out: str) -> int:
    """方位 d_in から d_out への**旋回回数**（90° を 1、180° を 2 と数える）。

    裁定 R4 の `n_turns` の定義（区画列の進行方向変化・180° = 2）と同じ数え方で
    あり、時間モデルの回帰に使った折れ数の定義とも一致する。
    """
    i0, i1 = DIRS.index(d_in), DIRS.index(d_out)
    diff = (i1 - i0) % 4
    return min(diff, 4 - diff)


class StraightGridModel:
    """直進格子のコストモデル。**状態 = (区画, その区画へ入ってきた進行方向)**。

    辺は「1 区画ぶんの移動」1 つで、費用は

        a · 1 + b · (入ってきた向きから出て行く向きへの旋回回数)

    である。旋回の費用を**出る側の区画で**払う形にしているので、状態が
    「入ってきた向き」を覚えていれば費用が決まる。

    a, b は必ず実測の回帰結果から渡すこと（**ハードコードしない**。
    exp_015 カード §3）。
    """

    kind = "straight_grid"

    def __init__(self, a: float, b: float):
        self.a = float(a)
        self.b = float(b)

    # --- 費用 ---------------------------------------------------------
    def edge_cost(self, d_in: str, d_out: str) -> float:
        return self.a + self.b * turn_count(d_in, d_out)

    # --- 状態 ---------------------------------------------------------
    @staticmethod
    def cell_of(state):
        return state[0]

    @staticmethod
    def target_states(targets):
        """目標区画に「どの向きで入っていても」到達とみなす（費用 0 の起点）。"""
        for c in targets:
            for d in DIRS:
                yield (tuple(c), d)

    # --- 遷移 ---------------------------------------------------------
    def predecessors(self, state, width: int, height: int, connects):
        """後ろ向き Dijkstra 用。state=(cell, d_out) へ**入って来られる**状態。

        cell へ d_out で入ったということは、直前は cell − δ(d_out) に居て
        d_out 方向へ出たということ。その区画に居たときの向き d_in は 4 通り
        あり、それぞれ旋回回数が違う。
        """
        cell, d_out = state
        dx, dy = DELTA[d_out]
        px, py = cell[0] - dx, cell[1] - dy
        if not (0 <= px < width and 0 <= py < height):
            return
        if not connects(px, py, cell[0], cell[1]):
            return
        for d_in in DIRS:
            yield ((px, py), d_in), self.edge_cost(d_in, d_out)

    def successors(self, cell, d_in: str, width: int, height: int, connects):
        """貪欲降下用。区画 cell に d_in の向きで居るときの一手の候補。"""
        for d_out in DIRS:
            dx, dy = DELTA[d_out]
            nx, ny = cell[0] + dx, cell[1] + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if not connects(cell[0], cell[1], nx, ny):
                continue
            yield d_out, (nx, ny), ((nx, ny), d_out), self.edge_cost(d_in, d_out)


class TimeField(dict):
    """`_flood_fill` の返り値と**差し替え可能**な時間の場。

    dict としては「区画 → その区画に至る最小時間（進行方向は問わない）」を返す。
    目標区画では厳密に 0.0 になるので、親クラスの
    `dist_field.get(cell) == 0`（目標到達の判定）がそのまま働く。
    到達不能な区画はキーを持たない（`.get()` が None を返す）ので、
    親の到達可能性の判定もそのまま働く。

    状態ごとの値は `.states` に持つ（`_select_direction` が使う）。
    """

    def __init__(self, states: dict, model):
        self.states = states
        self.model = model
        per_cell = {}
        for state, v in states.items():
            c = model.cell_of(state)
            if v < per_cell.get(c, float("inf")):
                per_cell[c] = v
        super().__init__(per_cell)


def forward_field(start_states, width: int, height: int, connects, model) -> dict:
    """**前向き** Dijkstra。`start_states` から各状態までの最小時間を返す。

    2026-08-13 追加（exp_018）。`value_field`（後ろ向き）と対で使い、
    「ある未知壁が最適経路に使われうるか」を状態の上で判定する
    （$D_f(s) + c(s \\to s') + D_b(s') = T$ なら、その辺は最適経路上にある）。

    **`value_field` と同じモデル契約の上で動く**（`successors` と `cell_of` しか
    使わない）ので、exp_016 で状態空間を差し替えても両方そのまま使える。

    Args:
        start_states: 出発状態の列（例 `[((0, 0), "N")]`）。**出発時の向きを含む**
    Returns:
        dict: 状態 → 出発状態からの最小時間 [s]
    """
    best = {}
    pq = []
    for st in start_states:
        if st not in best:
            best[st] = 0.0
            heapq.heappush(pq, (0.0, st))
    while pq:
        cost, state = heapq.heappop(pq)
        if cost > best.get(state, float("inf")) + TIE_EPS:
            continue
        cell, d_in = state
        for _d_out, _n, nxt, w in model.successors(cell, d_in, width, height, connects):
            nc = cost + w
            if nc < best.get(nxt, float("inf")) - TIE_EPS:
                best[nxt] = nc
                heapq.heappush(pq, (nc, nxt))
    return best


def value_field(targets, width: int, height: int, connects, model) -> TimeField:
    """目標集合からの**後ろ向き** Dijkstra。

    Args:
        targets: 目標区画の列（複数可＝多始点）
        width, height: 迷路の区画数
        connects: `connects(x, y, nx, ny) -> bool`。隣接 2 区画が通行可能か。
                  **未知壁の扱いはこの関数に委ねる**（方策側は楽観的な
                  `_connects_known` を渡す）
        model: コストモデル（`StraightGridModel` 等）

    Returns:
        TimeField（区画 → 最小時間 [s]、`.states` に状態ごとの値）
    """
    best = {}
    pq = []
    for st in model.target_states(targets):
        if st not in best:
            best[st] = 0.0
            heapq.heappush(pq, (0.0, st))
    while pq:
        cost, state = heapq.heappop(pq)
        if cost > best.get(state, float("inf")) + TIE_EPS:
            continue
        for prev, w in model.predecessors(state, width, height, connects):
            nc = cost + w
            if nc < best.get(prev, float("inf")) - TIE_EPS:
                best[prev] = nc
                heapq.heappush(pq, (nc, prev))
    return TimeField(best, model)
