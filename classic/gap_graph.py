"""
classic/gap_graph.py
================
経路のグラフのノードを、区画の中心ではなく**柱間の中点**に置いた最短経路の計算
（`research_notes/note_037_probabilistic_localization.md` §15、
`experiments/exp_032_post_gap_graph/PREREG.md` の実装）。

柱間 = 隣り合う 2 本の柱の間。壁が立ちうる場所であり、立っていなければ通れる。
`classic/maze_map.py` の v_walls（形状 (width+1, height)）・h_walls（形状
(width, height+1)）と 1 対 1 に対応させ、柱間 1 つを 1 ノードとする（16x16 の
迷路で 544 ノード）。

1 つの区画の中で、ある柱間から行ける先は 3 通り（`note_037` §15-1 の実測値）:

    向かい合う柱間へまっすぐ         0.18 m
    直交する柱間へ 45°（2 通り）     0.12728 m（= 0.18 / √2）

45° の線は柱と柱のちょうど真ん中を通るので、区画の中心を結ぶ折れ線と違って
実際に走れる。斜めの辺を許すかどうかは `build_gap_graph` の `allow_diagonal`
引数で切り替える（判定条文はこの切り替えだけを変えて比べる。禁じた場合が
`classic/flood.py` の区画グラフと一致することを検査で確かめる）。

🔴 **直交する柱間どうしの辺は、斜めを禁じても消えない**（実際に計算して
確かめて分かったこと）。実機は区画の中心で超信地旋回できるので、斜めを
禁じても「曲がる」動き自体は常にできる。曲がる距離は柱間から区画の中心へ
半区画・中心からもう一方の柱間へ半区画で計 0.18 m となり、直進1回と同じ
距離になる（区画グラフの「曲がっても1区画ぶんの歩数」と同じ勘定）。
`allow_diagonal=True` のときだけ、この辺を柱の間を斜めに突っ切る短い方
（0.12728 m）へ置き換える。経路の折れ線（後述）では、この「中心経由」の
辺だけ区画の中心を経由点として明示的に挟む（斜めに突っ切る辺は挟まない）。

🔴 **走行はしない。純粋な計算だけ。** `classic/flood.py`・`classic/route.py`・
`classic/maze_map.py` はこのモジュールから読むだけで、一切変更しない。

## 出発・ゴールと半区画の辺（PREREG §3 が実装前に決めることを要求）

出発点・ゴールは区画の**中心**（柱間ではない）。中心を柱間グラフへつなぐため、
区画ごとに 1 個の仮想ノード（「区画中心ノード」）を**問い合わせのたびに**動的に
追加する（544 個の実ノードは触らない。使い回す静的な配列とは別枠）。

**決めたこと**: 区画中心ノードは、その区画の 4 辺のうち壁が無い柱間へ**直進のみ**
（半区画 = 0.09 m）でつなぐ。斜めではつながない。理由: 区画の中心は隣接 2 区画の
柱間を結ぶ 45° の線の上には乗らない（中心から直交する柱間までの距離は水平・垂直
成分がそれぞれ半区画ずつで、45° の直線にはならない）。斜めが意味を持つのは
「柱間から柱間」という辺そのものの定義であって、区画の中心はその定義に含まれない。
この決めにより、斜めを禁じたときの柱間グラフの経路長は「出発の半区画 + 柱間を渡る
0.18 m の直進の積み重ね + ゴールの半区画」に一致し、区画グラフの経路長
（0.18 × 区画の歩数）とちょうど一致する（PREREG §3 の等価性の錨）。

ゴールが複数区画（大会規約の中央 2x2）にまたがる場合は、区画ごとに 1 個の仮想
ゴールノードを用意し、ダイクストラ法が最初に確定させたものを採用する
（`classic/flood.py` の多点ゴール BFS と同じ「一番近い 1 つを採る」規約）。

## マイコン実装の予算（PREREG §6・note_037 §13-4）

**RAM**（544 実ノード + 仮想ノード数個。1 迷路ぶん・単精度）:

| 配列 | 型 | サイズ | バイト数 |
|---|---|---|---|
| node_xy（柱間の世界座標） | float32 ×2 | 544×2 | 4352 |
| neighbor_id（隣接ノード番号、最大6本） | int16 | 544×6 | 6528 |
| neighbor_w（辺の距離） | float32 | 544×6 | 13056 |
| degree（次数） | uint8 | 544 | 544 |
| dist（確定距離） | float32 | 544+α | ~2200 |
| prev（経路復元用の1つ前のノード） | int16 | 544+α | ~1100 |
| visited（確定フラグ） | 1bit/ノード | 544+α | ~70 |

合計 約 28 KB。note_037 §13-1 の「推定器と地図で 64 KB 以内」に対し半分以下で
収まる（neighbor_w を辺の種別コード 1 バイト＋定数表 3 個に置き換えれば
約 3.3 KB まで減らせるが、本モジュールは分かりやすさを優先して素直な float32
配列のまま実装した）。

**演算量**（ダイクストラ法 1 回・544+仮想ノード分。ヒープを持たない配列選択法。
理由は次段落）: 展開するノードは高々 549、1 ノードの展開ごとに最大 6 本の隣接辺を
調べるので、辺の緩和は高々 549×6 ≈ 3300 回。「次に確定するノードを選ぶ」線形探索が
支配的で、549 回の探索 × 549 要素 ≈ 30 万回の比較になる。単精度の比較を 1 回 2〜3
サイクルとして見積もると、STM32F405 級（168 MHz）で約 5 ms、H743 級（480 MHz）で
約 2 ms。**1 kHz の制御周期（1 周期 1 ms）には収まらない。**壁地図が更新されたとき
だけ（探索走行中に新しい壁が確定した区画ごと、または最短走行を始める前の 1 回）に
計算すればよく、毎ティックの計算ではないので、複数ティックにまたがって計算するか、
走行を止めている区間（探索完了後・最短走行の直前など）で 1 回計算する運用でよい。
配列選択法（ヒープ不使用）を選んだ理由は、544 ノード程度ではヒープの構築・保守に
余分な配列（ヒープ内の位置を逆引きする表など）が要り、単純な線形探索より複雑な
わりに得が小さいため。辺の長さが実質 2 種類（直進・斜め）しか無いことを使った
バケツ待ち行列（ダイアル法）にすればヒープ無しで O(ノード数+辺数) まで落とせるが、
現状の見積りで用途（走行を止めた区間での 1 回計算）には十分なので採らない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from classic.flood import FloodMode, is_passable
from classic.maze_map import Direction, MazeMap

Cell = Tuple[int, int]

# ==========================================================================
# 幾何定数（note_037 §15-1・PREREG §1。区画寸法 p はクラシック競技の 0.18 m）
# ==========================================================================
CELL_PITCH_M = 0.18  # 区画の一辺 p（docs/COORDINATE_SYSTEM.md §6 と同じ）
STRAIGHT_GAP_DIST_M = CELL_PITCH_M  # 向かい合う柱間へ直進
DIAG_GAP_DIST_M = CELL_PITCH_M / math.sqrt(2.0)  # 直交する柱間へ45°（0.12728...m）
HALF_CELL_DIST_M = CELL_PITCH_M / 2.0  # 区画の中心と柱間を結ぶ半区画（0.09 m）

# 1 つの柱間は最大 2 区画に面し、区画ごとに「向かい合う1本＋直交2本」＝3本の
# 辺を持ちうるので、実ノードの次数は最大 6（PREREG §1）。仮想の区画中心ノード
# （最大4本）もこの枠に収まる。
MAX_DEGREE = 6


# ==========================================================================
# ノード番号と世界座標
# ==========================================================================
def num_v_gaps(width: int, height: int) -> int:
    """縦の柱間（v_walls と同じ形）の総数。"""
    return (width + 1) * height


def num_h_gaps(width: int, height: int) -> int:
    """横の柱間（h_walls と同じ形）の総数。"""
    return width * (height + 1)


def v_gap_id(x: int, y: int, width: int, height: int) -> int:
    """v_walls[x, y] に対応するノード番号（0 <= x <= width, 0 <= y < height）。"""
    return x * height + y


def h_gap_id(x: int, y: int, width: int, height: int) -> int:
    """h_walls[x, y] に対応するノード番号（0 <= x < width, 0 <= y <= height）。
    縦の柱間の後ろに詰める（0 〜 num_v_gaps-1 が縦、その次から横）。"""
    return num_v_gaps(width, height) + x * (height + 1) + y


def v_gap_world_xy(x: int, y: int, p: float = CELL_PITCH_M) -> Tuple[float, float]:
    """縦の柱間 v_walls[x, y] の中点の世界座標 [m]。

    柱 (x, y) と柱 (x, y+1)（同じ列・隣り合う行）のちょうど中間。
    座標の規約は docs/COORDINATE_SYSTEM.md（原点 = 南西の柱の中心、+x = 東、
    +y = 北）に従う。柱は格子点 (p*i, p*j) に立つので、柱間の中点はその
    ちょうど中間点になる。
    """
    return (x * p, y * p + p / 2.0)


def h_gap_world_xy(x: int, y: int, p: float = CELL_PITCH_M) -> Tuple[float, float]:
    """横の柱間 h_walls[x, y] の中点の世界座標 [m]。

    柱 (x, y) と柱 (x+1, y)（同じ行・隣り合う列）のちょうど中間。
    """
    return (x * p + p / 2.0, y * p)


def cell_center_xy(cx: int, cy: int, p: float = CELL_PITCH_M) -> Tuple[float, float]:
    """区画 (cx, cy) の中心の世界座標 [m]（docs/COORDINATE_SYSTEM.md §6 の式）。"""
    return (cx * p + p / 2.0, cy * p + p / 2.0)


def standard_goal_cells(width: int, height: int) -> List[Cell]:
    """中央 2x2 ゴール領域のセル座標（`classic/explorer.py::_goal_cells` と
    同じ規約の複製。`classic/` は `competition/` へ依存させない層構造
    （note_033 §2）のため、既存の複製パターン（`experiments/exp_030_map_quality/
    judge.py::goal_cells`）にならって本モジュールにも短く複製する）。"""
    gx0, gx1 = width // 2 - 1, width // 2
    gy0, gy1 = height // 2 - 1, height // 2
    return [(gx0, gy0), (gx0, gy1), (gx1, gy0), (gx1, gy1)]


# ==========================================================================
# 柱間の通過可否（classic.flood.is_passable にそのまま委ねる。楽観・悲観の
# 扱いを独自実装せず、既存の判定基準と食い違わないようにする）
# ==========================================================================
def _v_gap_passable(maze: MazeMap, x: int, y: int, mode: FloodMode) -> bool:
    """縦の柱間 v_walls[x, y] が通れるか。"""
    if x < maze.width:
        return is_passable(maze, x, y, Direction.W, mode)
    return is_passable(maze, maze.width - 1, y, Direction.E, mode)


def _h_gap_passable(maze: MazeMap, x: int, y: int, mode: FloodMode) -> bool:
    """横の柱間 h_walls[x, y] が通れるか。"""
    if y < maze.height:
        return is_passable(maze, x, y, Direction.S, mode)
    return is_passable(maze, x, maze.height - 1, Direction.N, mode)


# ==========================================================================
# 静的なグラフ（544 ノード。壁地図が変わるたびに作り直す。制御周期ごとではない）
# ==========================================================================
@dataclass
class GapGraph:
    """柱間グラフ本体。ノード番号・世界座標・隣接表は固定長の numpy 配列
    （マイコンでの固定配列の使い回しに対応する形。モジュール docstring 参照）。"""

    width: int
    height: int
    mode: FloodMode
    allow_diagonal: bool
    n_v: int
    n_h: int
    n_nodes: int
    node_xy: np.ndarray  # (n_nodes, 2) float32 [m]
    neighbor_id: np.ndarray  # (n_nodes, MAX_DEGREE) int32、未使用は -1
    neighbor_w: np.ndarray  # (n_nodes, MAX_DEGREE) float32 [m]
    degree: np.ndarray  # (n_nodes,) int32


def build_gap_graph(
    maze: MazeMap,
    mode: FloodMode,
    allow_diagonal: bool,
    straight_dist_m: float = STRAIGHT_GAP_DIST_M,
    diag_dist_m: float = DIAG_GAP_DIST_M,
) -> GapGraph:
    """柱間グラフを組み立てる（544 ノード。区画ごとに最大 6 辺を張る）。

    straight_dist_m / diag_dist_m を既定値から変えられるようにしてあるのは、
    否定対照 N1（PREREG §4: 斜めのコストを直進と同じ 0.18 m にする）を
    グラフの作り方そのものを変えずに再現するため。
    """
    width, height = maze.width, maze.height
    n_v = num_v_gaps(width, height)
    n_h = num_h_gaps(width, height)
    n_nodes = n_v + n_h

    node_xy = np.zeros((n_nodes, 2), dtype=np.float32)
    for x in range(width + 1):
        for y in range(height):
            node_xy[v_gap_id(x, y, width, height)] = v_gap_world_xy(x, y)
    for x in range(width):
        for y in range(height + 1):
            node_xy[h_gap_id(x, y, width, height)] = h_gap_world_xy(x, y)

    neighbor_id = np.full((n_nodes, MAX_DEGREE), -1, dtype=np.int32)
    neighbor_w = np.zeros((n_nodes, MAX_DEGREE), dtype=np.float32)
    degree = np.zeros(n_nodes, dtype=np.int32)

    def _add_edge(a: int, b: int, w: float) -> None:
        # 無向グラフなので両端に足す。次数は高々6なので配列の末尾に詰めるだけ
        # でよい（探索・並べ替えは不要 — 組み込み向けの固定配列を想定した形）。
        da = int(degree[a])
        assert da < MAX_DEGREE, f"ノード{a}の次数がMAX_DEGREE={MAX_DEGREE}を超えた"
        neighbor_id[a, da] = b
        neighbor_w[a, da] = w
        degree[a] = da + 1

        db = int(degree[b])
        assert db < MAX_DEGREE, f"ノード{b}の次数がMAX_DEGREE={MAX_DEGREE}を超えた"
        neighbor_id[b, db] = a
        neighbor_w[b, db] = w
        degree[b] = db + 1

    # 区画ごとに、その4辺の柱間（W/E/S/N）の間へ最大6辺を張る。柱間は隣り合う
    # 2区画でしか共有されないので、この二重ループで各辺はちょうど1回だけ
    # 作られる（重複や food.py 側との整合の崩れは起きない）。
    for cx in range(width):
        for cy in range(height):
            w_id = v_gap_id(cx, cy, width, height)
            e_id = v_gap_id(cx + 1, cy, width, height)
            s_id = h_gap_id(cx, cy, width, height)
            n_id = h_gap_id(cx, cy + 1, width, height)

            w_ok = _v_gap_passable(maze, cx, cy, mode)
            e_ok = _v_gap_passable(maze, cx + 1, cy, mode)
            s_ok = _h_gap_passable(maze, cx, cy, mode)
            n_ok = _h_gap_passable(maze, cx, cy + 1, mode)

            if w_ok and e_ok:
                _add_edge(w_id, e_id, straight_dist_m)
            if s_ok and n_ok:
                _add_edge(s_id, n_id, straight_dist_m)

            # 直交する柱間どうしも必ずつなぐ（実際に計算して確かめて分かったこと
            # — 下記コメント）。斜めを禁じても、実機は区画の中心で超信地旋回
            # できる。区画の中心を通って直交する柱間へ抜ける経路の長さは
            # 半区画+半区画=0.18m で、直進1回と同じ距離になる（区画グラフの
            # 「曲がっても1区画ぶんの歩数」と同じ勘定）。斜めを許すときだけ、
            # この直交の辺を柱の間を斜めに突っ切る短い方（0.12728m）へ置き換える。
            #
            # この「直交の辺を禁止時は消す」のではなく「距離を0.18mへ戻す」
            # という扱いが要る理由: 消してしまうと、区画の中心からしか外へ
            # 出られない区画（例えば出発区画のように2辺が外周壁の区画）を
            # 曲がって抜ける経路が柱間グラフ上から消え、PREREG §3 の等価性の
            # 錨（斜め禁止時に classic/flood.py の区画グラフと一致すること）が
            # 4x4 の手計算迷路で最初から崩れた（NoGapRouteError）。区画の中心で
            # 曲がる動作は斜め探索の有無によらず常に可能なので、これは実装の
            # 誤りだった（斜め禁止 = 45°の近道が使えないだけで、曲がること
            # 自体を禁じるのではない）。
            perp_dist_m = diag_dist_m if allow_diagonal else straight_dist_m
            if w_ok and n_ok:
                _add_edge(w_id, n_id, perp_dist_m)
            if w_ok and s_ok:
                _add_edge(w_id, s_id, perp_dist_m)
            if e_ok and n_ok:
                _add_edge(e_id, n_id, perp_dist_m)
            if e_ok and s_ok:
                _add_edge(e_id, s_id, perp_dist_m)

    return GapGraph(
        width=width, height=height, mode=mode, allow_diagonal=allow_diagonal,
        n_v=n_v, n_h=n_h, n_nodes=n_nodes,
        node_xy=node_xy, neighbor_id=neighbor_id, neighbor_w=neighbor_w, degree=degree,
    )


# ==========================================================================
# 最短経路（ダイクストラ法。区画中心の仮想ノードをその場で付け足す）
# ==========================================================================
class NoGapRouteError(Exception):
    """柱間グラフ上で start からゴールへ到達できない場合に送出する。"""


@dataclass
class GapPath:
    """柱間グラフ上の最短経路（走行ラインの下地）。"""

    distance_m: float
    node_ids: List[int]  # 仮想ノードを含む内部ノード番号列（先頭=出発、末尾=ゴール）
    xy_m: np.ndarray  # (K, 2) float32。出発区画の中心〜ゴール区画の中心の折れ線
    is_diagonal: List[bool] = field(default_factory=list)  # xy_m の各線分(K-1個)が斜めか
    n_expansions: int = 0  # ダイクストラ法で確定させたノードの数（マイコン予算の実測）


def _gap_coords(node_id: int, graph: GapGraph) -> Tuple[bool, int, int]:
    """実ノード番号から (縦の柱間か, x, y) を逆引きする（v_gap_id/h_gap_id の逆写像）。"""
    if node_id < graph.n_v:
        x, y = divmod(node_id, graph.height)
        return True, x, y
    idx = node_id - graph.n_v
    x, y = divmod(idx, graph.height + 1)
    return False, x, y


def _shared_cell(node_a: int, node_b: int, graph: GapGraph) -> Cell:
    """直交する柱間ノード node_a・node_b（一方が縦、他方が横）が共有する
    唯一の区画を返す（曲がり角の中心経由点を求めるのに使う）。"""
    is_va, xa, ya = _gap_coords(node_a, graph)
    is_vb, xb, yb = _gap_coords(node_b, graph)
    if is_va == is_vb:
        raise ValueError("同種の柱間どうしは区画中心を経由しないはず（実装の前提が崩れている）")
    if is_va:
        v_x, v_y, h_x, h_y = xa, ya, xb, yb
    else:
        v_x, v_y, h_x, h_y = xb, yb, xa, ya
    v_cells = {(v_x - 1, v_y), (v_x, v_y)}
    h_cells = {(h_x, h_y - 1), (h_x, h_y)}
    shared = {
        c for c in (v_cells & h_cells)
        if 0 <= c[0] < graph.width and 0 <= c[1] < graph.height
    }
    if len(shared) != 1:
        raise ValueError(f"柱間 {node_a},{node_b} の共有区画が一意に決まりません: {shared}")
    return next(iter(shared))


def _edge_weight(
    graph: GapGraph, extra_edges: Dict[int, List[Tuple[int, float]]], u: int, v: int
) -> float:
    """経路復元でたどった辺 u->v の重みを引き直す（曲がり角かどうかの判定に使う）。"""
    if u < graph.n_nodes:
        deg = int(graph.degree[u])
        for k in range(deg):
            if int(graph.neighbor_id[u, k]) == v:
                return float(graph.neighbor_w[u, k])
    for nb, w in extra_edges.get(u, ()):
        if nb == v:
            return w
    raise ValueError(f"ノード{u}->{v}の辺が見つかりません（実装の前提が崩れている）")


def _link_cell_center(
    maze: MazeMap,
    graph: GapGraph,
    cell: Cell,
    extra_edges: Dict[int, List[Tuple[int, float]]],
    vid: int,
) -> None:
    """区画 cell の仮想中心ノード vid を、開いている境界の柱間へ半区画
    （0.09 m）の直進でつなぐ（モジュール docstring の「決めたこと」参照。
    斜めではつながない）。"""
    cx, cy = cell
    if not maze.in_bounds(cx, cy):
        raise ValueError(f"区画 {cell} は迷路の範囲外です")
    width, height = graph.width, graph.height
    borders = [
        (v_gap_id(cx, cy, width, height), _v_gap_passable(maze, cx, cy, graph.mode)),
        (v_gap_id(cx + 1, cy, width, height), _v_gap_passable(maze, cx + 1, cy, graph.mode)),
        (h_gap_id(cx, cy, width, height), _h_gap_passable(maze, cx, cy, graph.mode)),
        (h_gap_id(cx, cy + 1, width, height), _h_gap_passable(maze, cx, cy + 1, graph.mode)),
    ]
    for gap_id, ok in borders:
        if not ok:
            continue
        extra_edges.setdefault(vid, []).append((gap_id, HALF_CELL_DIST_M))
        extra_edges.setdefault(gap_id, []).append((vid, HALF_CELL_DIST_M))


def shortest_path(graph: GapGraph, maze: MazeMap, start: Cell, goals: Sequence[Cell]) -> GapPath:
    """start からゴール群のいずれかまでの最短経路（柱間グラフ上・ダイクストラ法）。

    ゴールが複数区画にまたがる場合は、区画ごとに仮想ゴールノードを用意し、
    最初に確定した（= 最も近い）ものを採用する（`classic/flood.py` の多点
    ゴール BFS と同じ「最も近い1つ」規約）。到達不能なら NoGapRouteError。
    """
    goals = list(goals)
    if not goals:
        raise ValueError("goals が空です")

    n_nodes = graph.n_nodes
    start_vid = n_nodes
    goal_vids = [n_nodes + 1 + i for i in range(len(goals))]
    n_total = n_nodes + 1 + len(goals)

    # 仮想ノードの辺は「出発1個＋ゴール区画数個」という定数個しか無いので、
    # ここでは Python の dict/list で組んでいるが、組み込みでは
    # 「仮想ノード数（高々5） × 最大4本」の固定長の小さい配列に置き換えられる
    # （544ノード分の静的配列とは別に、この小さい配列だけを問い合わせのたびに
    # 作り直せばよく、動的な確保の量は常に一定で増えない）。
    extra_edges: Dict[int, List[Tuple[int, float]]] = {}
    _link_cell_center(maze, graph, start, extra_edges, start_vid)
    for vid, g in zip(goal_vids, goals):
        _link_cell_center(maze, graph, g, extra_edges, vid)

    def _neighbors(node_id: int):
        if node_id < n_nodes:
            deg = int(graph.degree[node_id])
            for k in range(deg):
                yield int(graph.neighbor_id[node_id, k]), float(graph.neighbor_w[node_id, k])
        for nb, w in extra_edges.get(node_id, ()):
            yield nb, w

    # --- ダイクストラ法（単精度・ヒープ不使用の配列選択法。モジュール
    # docstring の「マイコン実装の予算」参照）。dist/prev/visited は呼び出す
    # たびに確保しているが、サイズは常に n_total（544 + 仮想ノード数）で
    # 一定であり、内側のループでは一切確保しない。
    INF = np.float32(np.inf)
    dist = np.full(n_total, INF, dtype=np.float32)
    prev = np.full(n_total, -1, dtype=np.int32)
    visited = np.zeros(n_total, dtype=bool)
    dist[start_vid] = np.float32(0.0)

    goal_set = set(goal_vids)
    reached_goal = -1
    n_expansions = 0

    for _ in range(n_total):
        masked = np.where(visited, INF, dist)
        u = int(np.argmin(masked))
        if not np.isfinite(masked[u]):
            break  # 残り全て到達不能
        visited[u] = True
        n_expansions += 1
        if u in goal_set:
            reached_goal = u
            break
        du = float(dist[u])
        for nb, w in _neighbors(u):
            if visited[nb]:
                continue
            nd = np.float32(du + w)
            if nd < dist[nb]:
                dist[nb] = nd
                prev[nb] = u

    if reached_goal < 0:
        raise NoGapRouteError(f"{start} からゴール {goals} へ到達できません（mode={graph.mode}）")

    # 経路の復元（仮想ノードを含む）
    node_ids: List[int] = []
    cur = reached_goal
    while cur != -1:
        node_ids.append(cur)
        cur = int(prev[cur])
    node_ids.reverse()

    goal_cell_of = {vid: g for vid, g in zip(goal_vids, goals)}

    def _node_xy(nid: int) -> Tuple[float, float]:
        if nid == start_vid:
            return cell_center_xy(*start)
        if nid in goal_cell_of:
            return cell_center_xy(*goal_cell_of[nid])
        return (float(graph.node_xy[nid, 0]), float(graph.node_xy[nid, 1]))

    # 折れ線を組み立てる。実際に計算して分かったこと: 「斜め禁止時、
    # 直交する柱間どうしを区画の中心経由(0.18m)でつなぐ」辺は、2 つの柱間を
    # 直接結ぶと見た目上は斜め45°の直線になってしまう（柱間の中点どうしの
    # 素の直線距離は、常に 0.12728m — 斜めに切ったときの距離 — だから）。
    # これは辺の重み(0.18m)と矛盾するので、経路の折れ線ではこの辺だけ
    # 区画の中心を経由点として明示的に挟み、実際に曲がる動きを表す
    # （半区画0.09m×2に分解され、辺の重みとちょうど一致する）。斜めに
    # 突っ切る辺（重み0.12728m）は挟まない — 柱間の中点どうしを結ぶ直線が
    # そのまま最短距離になっている。
    raw_xy = [_node_xy(nid) for nid in node_ids]
    xy_points: List[Tuple[float, float]] = [raw_xy[0]]
    for i in range(1, len(node_ids)):
        u, v = node_ids[i - 1], node_ids[i]
        pu, pv = raw_xy[i - 1], raw_xy[i]
        dx, dy = pv[0] - pu[0], pv[1] - pu[1]
        axis_aligned = abs(dx) < 1e-6 or abs(dy) < 1e-6
        if not axis_aligned:
            w = _edge_weight(graph, extra_edges, u, v)
            if math.isclose(w, STRAIGHT_GAP_DIST_M, abs_tol=1e-6):
                xy_points.append(cell_center_xy(*_shared_cell(u, v, graph)))
        xy_points.append(pv)

    xy = np.array(xy_points, dtype=np.float32)
    is_diagonal: List[bool] = []
    for i in range(len(xy_points) - 1):
        dx = float(xy[i + 1, 0]) - float(xy[i, 0])
        dy = float(xy[i + 1, 1]) - float(xy[i, 1])
        is_diagonal.append(abs(dx) > 1e-6 and abs(dy) > 1e-6)

    return GapPath(
        distance_m=float(dist[reached_goal]),
        node_ids=node_ids,
        xy_m=xy,
        is_diagonal=is_diagonal,
        n_expansions=n_expansions,
    )


def polyline_length_m(xy_m: np.ndarray) -> float:
    """折れ線 xy_m（(K,2)）の総延長 [m]。GapPath.distance_m との一致検査に使う
    （辺の重みと世界座標の幾何が食い違っていないことを確かめるための、
    経路長の独立な再計算）。"""
    if len(xy_m) < 2:
        return 0.0
    diffs = np.diff(xy_m.astype(np.float64), axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))
