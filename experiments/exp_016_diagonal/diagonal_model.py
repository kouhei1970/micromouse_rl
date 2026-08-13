#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""斜めを含む費用モデル（exp_016 段階 016-A）。

**`competition/route_planner.py` の契約（裁定 R29）をそのまま満たす**ので、
Dijkstra 本体（`value_field` / `forward_field`）は**無改造**で使える。

--------------------------------------------------------------------------
状態空間 — 半区画の格子と 8 方位
--------------------------------------------------------------------------
節点を**半区画刻み**で置く。区画寸法 $c$ に対し、**2 倍した整数座標**で持つと
すべてが整数演算になる（節点 $(u,v)$ の実座標は $(u \\cdot c/2,\\ v \\cdot c/2)$）:

| 節点 | 2 倍座標 | 実座標 | 意味 |
|---|---|---|---|
| `C(i,j)` | $(2i+1,\\ 2j+1)$ | 区画の中心 | 直進の節点 |
| `V(i,j)` | $(2i,\\ 2j+1)$ | 縦の壁スロットの中点 | 斜めの節点 |
| `H(i,j)` | $(2i+1,\\ 2j)$ | 横の壁スロットの中点 | 斜めの節点 |
| （両方偶数） | $(2i,\\ 2j)$ | **格子点＝柱** | **節点にならない** |

**すべての遷移は 2 倍座標に $(d_x, d_y) \\in \\{-1,0,1\\}^2 \\setminus \\{(0,0)\\}$ を足す**
という同じ形になる:

- **半歩**（$d$ が上下左右）: 区画中心 ↔ 壁スロット中点。距離 $c/2$
- **斜め**（$d$ が斜め）: 壁スロット中点 → 壁スロット中点。距離 $c/\\sqrt2 \\cdot \\frac12 \\cdot 2 = c/\\sqrt2$
  … 正確には $\\sqrt{(c/2)^2 + (c/2)^2} = c/\\sqrt2$

**座標の偶奇が構造を自動的に守る**: 区画中心から斜めに動くと柱（両方偶数）に
なるので**斜めは壁スロット中点からしか出られない**。壁スロット中点から
壁に平行な向きへ動いても柱になるので、**半歩は壁を横切る向きにしか動けない**。
場合分けを書かなくても、**格子の偶奇が「あり得る動き」を決めている。**

--------------------------------------------------------------------------
通れるかどうか
--------------------------------------------------------------------------
**壁スロットの中点は、その壁が開いているときだけ節点として存在する。**
したがって遷移の可否は「行き先の節点が存在するか」だけで決まり、
**壁の判定は節点の存在判定に吸収される**。外周の壁スロットは常に閉じている。

区画の連結は呼び出し側の `connects(x, y, nx, ny)` に委ねる
（方策の `_connects_known` と同じ引数の並び。**未知壁の扱いは呼び出し側の責任**）。

--------------------------------------------------------------------------
費用（**すべて秒**。仮定は明示する）
--------------------------------------------------------------------------
- **半歩**: $a/2$（$a$ = 直進 1 区画の実測時間 0.1554 s）
- **斜め 1 歩**: $\\dfrac{a}{\\sqrt2 \\, r}$（$r = v_\\text{斜め}/v_\\text{直進}$。**未実測なので引数**）
- **旋回**: $b \\cdot \\dfrac{\\Delta}{90°}$（$b$ = 実測の 90° 旋回費用 0.1358 s）

> ### ⚠️ 仮定 2 件（実測ではない）
> 1. **45° 旋回の費用を $b/2$ とした。**実測しているのは 90° の $b$ だけである。
>    斜めの出入りは 45° 旋回なので、**この仮定が短縮率に直接効く**
> 2. **斜め区間の速度比 $r$ は未実測。**そのため $r$ を**引数にして振る**
>    （1.0 / 0.85 / 0.7 / 0.55）。**1 つの値に決め打ちしない**

使い方:
    from experiments.exp_016_diagonal.diagonal_model import DiagonalGridModel
    model = DiagonalGridModel(a, b, r=0.7)
    field = value_field(goal_states, W, H, connects, model)     # route_planner のまま
"""
import math

# 8 方位（2 倍座標での増分）。名前は 4 方位版（route_planner.DIRS）と互換にする。
DIRS8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
DELTA8 = {"N": (0, 1), "NE": (1, 1), "E": (1, 0), "SE": (1, -1),
          "S": (0, -1), "SW": (-1, -1), "W": (-1, 0), "NW": (-1, 1)}
_IDX = {d: i for i, d in enumerate(DIRS8)}
DIAGONALS = frozenset(("NE", "SE", "SW", "NW"))

SQRT1_2 = math.sqrt(0.5)          # 斜め 1 歩の長さ（区画単位）


def turn_deg(d_in: str, d_out: str) -> int:
    """方位 d_in から d_out への最小回転角 [deg]（45° 刻み）。"""
    k = abs(_IDX[d_out] - _IDX[d_in]) % 8
    return min(k, 8 - k) * 45


def node_kind(node):
    """節点の種類を返す（'C' = 区画中心 / 'V' = 縦壁の中点 / 'H' = 横壁の中点）。"""
    u, v = node
    if u % 2 == 1 and v % 2 == 1:
        return "C"
    if u % 2 == 0 and v % 2 == 1:
        return "V"
    if u % 2 == 1 and v % 2 == 0:
        return "H"
    return None                    # 両方偶数 = 柱。節点にならない


def cell_center_node(cell):
    """区画 (i,j) の中心節点。"""
    return (2 * cell[0] + 1, 2 * cell[1] + 1)


def node_xy(node, cell_size):
    """節点の実座標 [m]。"""
    return (node[0] * cell_size / 2.0, node[1] * cell_size / 2.0)


class DiagonalGridModel:
    """斜めを含む費用モデル（`route_planner` の契約を満たす）。

    Args:
        a: 直進 1 区画あたりの時間 [s]（実測の回帰から渡す。ハードコードしない）
        b: 90° 旋回 1 回あたりの時間 [s]（同上）
        r: 斜め区間の速度比 $v_\\text{斜め}/v_\\text{直進}$（**未実測。振る**）
        turn_unit_45: 45° 旋回の費用 [s]。既定は $b/2$（**仮定**）
    """

    kind = "diagonal_half_grid"

    def __init__(self, a: float, b: float, r: float = 1.0, turn_unit_45: float = None):
        self.a = float(a)
        self.b = float(b)
        self.r = float(r)
        self.turn_45 = float(turn_unit_45) if turn_unit_45 is not None else self.b / 2.0

    # --- 費用 ---------------------------------------------------------
    def turn_cost(self, d_in: str, d_out: str) -> float:
        return self.turn_45 * (turn_deg(d_in, d_out) / 45.0)

    def move_cost(self, d_out: str) -> float:
        if d_out in DIAGONALS:
            return self.a * SQRT1_2 / self.r
        return self.a / 2.0

    def edge_cost(self, d_in: str, d_out: str) -> float:
        return self.move_cost(d_out) + self.turn_cost(d_in, d_out)

    # --- 節点の存在 ---------------------------------------------------
    @staticmethod
    def node_exists(node, width, height, connects) -> bool:
        """節点が存在するか。**壁スロットの中点は壁が開いているときだけ存在する。**"""
        k = node_kind(node)
        if k is None:
            return False                      # 柱
        u, v = node
        if k == "C":
            i, j = (u - 1) // 2, (v - 1) // 2
            return 0 <= i < width and 0 <= j < height
        if k == "V":
            i, j = u // 2, (v - 1) // 2
            if not (0 <= j < height) or not (0 <= i <= width):
                return False
            if i == 0 or i == width:
                return False                  # 外周の壁は常に閉じている
            return bool(connects(i - 1, j, i, j))
        i, j = (u - 1) // 2, v // 2           # k == "H"
        if not (0 <= i < width) or not (0 <= j <= height):
            return False
        if j == 0 or j == height:
            return False
        return bool(connects(i, j - 1, i, j))

    # --- 契約（route_planner が呼ぶのはこの 4 つだけ） -----------------
    @staticmethod
    def cell_of(state):
        """状態が属する区画。

        **区画中心の節点では厳密**。壁スロットの中点は 2 つの区画に接するので
        「2 倍座標を 2 で割った側」を返す**便宜的な値**である
        （`TimeField` の区画ごとの辞書はこのモデルでは意味を持たない。
        使うのは `.states` の方）。
        """
        (u, v), _d = state
        return (u // 2, v // 2)

    @staticmethod
    def target_states(targets):
        """目標区画の**中心**に、どの向きで入っていても到達とみなす。"""
        for c in targets:
            node = cell_center_node(tuple(c))
            for d in DIRS8:
                yield (node, d)

    def successors(self, node, d_in: str, width: int, height: int, connects):
        """節点 `node` に向き `d_in` で居るときの一手の候補。

        Yields: (出口方位, 次の節点, 次の状態, 辺の費用)
        """
        for d_out in DIRS8:
            dx, dy = DELTA8[d_out]
            nxt = (node[0] + dx, node[1] + dy)
            if not self.node_exists(nxt, width, height, connects):
                continue
            yield d_out, nxt, (nxt, d_out), self.edge_cost(d_in, d_out)

    def predecessors(self, state, width: int, height: int, connects):
        """後ろ向き Dijkstra 用。`state=(node, d_out)` へ**入って来られる**状態。"""
        node, d_out = state
        dx, dy = DELTA8[d_out]
        prev = (node[0] - dx, node[1] - dy)
        if not self.node_exists(prev, width, height, connects):
            return
        w = self.move_cost(d_out)
        for d_in in DIRS8:
            yield (prev, d_in), w + self.turn_cost(d_in, d_out)


def descend(field, model, start_node, start_dir, width, height, connects, tie_eps=1e-9):
    """時間の場を 1 手ずつ降りて経路（節点列と方位列）を作る。

    Returns: dict(nodes, dirs, cost, n_diag, n_half, turns_deg)
    """
    node, d_in = start_node, start_dir
    nodes, dirs = [node], []
    cost, n_diag, n_half, turns = 0.0, 0, 0, []
    for _ in range(4 * width * height + 16):
        if field.states.get((node, d_in), float("inf")) <= tie_eps:
            break                              # 目標に到達（値 0）
        best = None
        for d_out, nxt, st, w in model.successors(node, d_in, width, height, connects):
            v = field.states.get(st)
            if v is None:
                continue
            tot = w + v
            if best is None or tot < best[0] - tie_eps:
                best = (tot, d_out, nxt, w)
        if best is None:
            break
        _tot, d_out, nxt, w = best
        turns.append(turn_deg(d_in, d_out))
        cost += w
        if d_out in DIAGONALS:
            n_diag += 1
        else:
            n_half += 1
        node, d_in = nxt, d_out
        nodes.append(node)
        dirs.append(d_out)
    return dict(nodes=nodes, dirs=dirs, cost=cost, n_diag=n_diag, n_half=n_half,
                turns_deg=turns)
