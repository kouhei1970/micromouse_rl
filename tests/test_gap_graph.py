"""
tests/test_gap_graph.py
================
`classic/gap_graph.py`（経路のグラフのノードを区画の中心ではなく柱間の中点へ
移したモジュール。`experiments/exp_032_post_gap_graph/PREREG.md` の実装）の
単体テスト。pytest で実行する:

    source .venv/bin/activate
    python -m pytest tests/test_gap_graph.py -v

検証内容（PREREG §5・作業指示の「作るもの その2」）:
  1. 4x4 の手計算迷路で、斜めあり／斜めなしの最短経路長を手で数えた値と照合する
  2. 等価性の錨（PREREG §3）: 斜めを禁じたときの柱間グラフの経路長が、既存
     `classic/flood.py` の区画グラフの経路長と一致すること（複数の迷路で）
  3. 折れ線の総長（世界座標から計算）がダイクストラ法の経路長と一致すること
     （辺の重みと幾何が食い違っていないことの確認。コーディネータからの追加指示）
  4. 否定対照 N1: 斜めの辺のコストを直進と同じにすると ρ=1 になる
  5. 否定対照 N2: 柱間を全て壁にすると経路が存在しない
  6. 否定対照 N3: 壁配列を無作為に置き換えると、区画グラフとの一致は保ったまま
     （実装は壁を読んでいる）、距離そのものは置き換えるたびに変わる
  7. 未知(UNKNOWN)の扱い（楽観・悲観）が classic/flood.py と同じ向きに効くこと

🔴 `classic/flood.py`・`classic/route.py`・`classic/maze_map.py` は一切変更しない
（読むだけ）。本ファイルもそれらを変更しない。
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from classic.flood import FloodMode, compute_flood, distance_at
from classic.gap_graph import (
    CELL_PITCH_M,
    DIAG_GAP_DIST_M,
    HALF_CELL_DIST_M,
    STRAIGHT_GAP_DIST_M,
    GapPath,
    NoGapRouteError,
    build_gap_graph,
    polyline_length_m,
    shortest_path,
    standard_goal_cells,
)
from classic.maze_map import MazeMap, WallState

Cell = Tuple[int, int]

DESIGN_V4_DIR = os.path.join(REPO_ROOT, "competition", "mazes", "design_v4")


# ==========================================================================
# テスト用ヘルパ
# ==========================================================================
def _maze_from_truth_walls(v_walls: np.ndarray, h_walls: np.ndarray) -> MazeMap:
    """npz の真の壁配列（0=壁なし・非0=壁あり）から、全壁既知の MazeMap を作る
    （`experiments/exp_030_map_quality/judge.py::load_true_map` と同じ変換）。"""
    width = v_walls.shape[0] - 1
    height = v_walls.shape[1]
    maze = MazeMap(width, height)
    maze.v_walls[:, :] = np.where(v_walls != 0, int(WallState.WALL), int(WallState.OPEN))
    maze.h_walls[:, :] = np.where(h_walls != 0, int(WallState.WALL), int(WallState.OPEN))
    return maze


def _load_design_v4_maze(seed: int) -> MazeMap:
    d = np.load(os.path.join(DESIGN_V4_DIR, f"maze_{seed}.npz"))
    return _maze_from_truth_walls(d["v_walls"], d["h_walls"])


def _cell_graph_distance_m(
    maze: MazeMap, mode: FloodMode, start: Cell, goals: Sequence[Cell]
) -> Optional[float]:
    """既存 `classic/flood.py` の区画グラフから求めた最短経路長 [m]
    （= 0.18 x 区画の歩数）。到達不能なら None。"""
    dist = compute_flood(maze, list(goals), mode)
    steps = distance_at(dist, start)
    return None if steps is None else steps * CELL_PITCH_M


def _full_open_interior(width: int, height: int) -> MazeMap:
    """外周だけ壁・内部は全て開通の迷路（手計算用の最小構成）。"""
    maze = MazeMap(width, height)
    maze.v_walls[1:width, :] = WallState.OPEN
    maze.h_walls[:, 1:height] = WallState.OPEN
    return maze


def _gap_distance_or_none(
    maze: MazeMap, mode: FloodMode, allow_diagonal: bool, start: Cell, goals: Sequence[Cell]
) -> Optional[float]:
    graph = build_gap_graph(maze, mode, allow_diagonal)
    try:
        return shortest_path(graph, maze, start, goals).distance_m
    except NoGapRouteError:
        return None


DESIGN_V4_SAMPLE_SEEDS = [41003, 41006, 41011, 41018, 41020, 41049]


# ==========================================================================
# 0. 幾何定数が note_037 §15-1 の実測値と一致すること
# ==========================================================================
def test_geometry_constants_match_note_037():
    assert STRAIGHT_GAP_DIST_M == pytest.approx(0.18, abs=1e-9)
    assert HALF_CELL_DIST_M == pytest.approx(0.09, abs=1e-9)
    # note_037 §15-1 の表の丸め値（0.18/√2 = 0.1272792...）
    assert DIAG_GAP_DIST_M == pytest.approx(0.12728, abs=1e-5)


# ==========================================================================
# 1. 手計算による照合（PREREG §5。4x4・完全開放・note_030 §3 S0 と同じやり方）
# ==========================================================================
def test_hand_computed_4x4_no_diagonal_matches_manhattan():
    """手計算:
    4x4 の内部を全て開通にした迷路。出発 (0,0)、ゴールは中央2x2
    {(1,1),(1,2),(2,1),(2,2)}（`standard_goal_cells` の規約）。

    斜め禁止 = 区画グラフと同じマンハッタン距離。(0,0)から各ゴール区画までの
    歩数は (1,1)=2, (1,2)=3, (2,1)=3, (2,2)=4 で最小は (1,1) の 2 区画。
    2 区画 x 0.18m = 0.36 m。
    """
    maze = _full_open_interior(4, 4)
    start = (0, 0)
    goals = standard_goal_cells(4, 4)
    assert goals == [(1, 1), (1, 2), (2, 1), (2, 2)]

    graph = build_gap_graph(maze, FloodMode.PESSIMISTIC, allow_diagonal=False)
    path = shortest_path(graph, maze, start, goals)

    assert path.distance_m == pytest.approx(0.36, abs=1e-4)
    assert polyline_length_m(path.xy_m) == pytest.approx(path.distance_m, abs=1e-4)
    assert not any(path.is_diagonal)  # 斜め禁止なので斜めの区間は無いはず


def test_hand_computed_4x4_diagonal_matches_hand_route():
    """手計算（斜め許可）:
    区画(0,0)は西・南が外周壁なので、中心から出られるのは東の柱間 v_walls[1,0]
    と北の柱間 h_walls[0,1] のみ（それぞれ半区画0.09m）。

    区画(1,0)の中では、西柱間=v_walls[1,0]・北柱間=h_walls[1,1]・南柱間
    (=h_walls[1,0])は外周壁で塞がっているので、西と北は「直交」の関係になり
    45°の斜め辺(0.12728m)で直接つながる。h_walls[1,1] はゴール区画(1,1)の
    南柱間そのものなので、そこから半区画(0.09m)でゴール中心へ着く。

        出発中心 --0.09--> v_walls[1,0] --0.12728(斜め)--> h_walls[1,1] --0.09--> ゴール中心

    合計 = 0.09 + 0.12728 + 0.09 = 0.30728 m
    （0(0,0)から(1,1)へ対角を1辺で直接突っ切る 0.18*sqrt(2)=0.25456m は柱の
    真上を通る辺であり、柱間グラフには存在しない＝選ばれない）。
    """
    maze = _full_open_interior(4, 4)
    start = (0, 0)
    goals = standard_goal_cells(4, 4)

    graph = build_gap_graph(maze, FloodMode.PESSIMISTIC, allow_diagonal=True)
    path = shortest_path(graph, maze, start, goals)

    hand_calc = HALF_CELL_DIST_M + DIAG_GAP_DIST_M + HALF_CELL_DIST_M
    assert hand_calc == pytest.approx(0.30728, abs=1e-4)
    assert path.distance_m == pytest.approx(0.30728, abs=1e-4)
    assert path.distance_m == pytest.approx(hand_calc, abs=1e-6)

    # 折れ線の総長がダイクストラ法の距離と一致すること（辺の重みと幾何の食い違いが無いこと）
    assert polyline_length_m(path.xy_m) == pytest.approx(path.distance_m, abs=1e-4)

    # 経路は3区間、うち斜めはちょうど1区間のはず（上記手計算の経路そのもの）
    assert path.is_diagonal == [False, True, False]

    # 斜め許可のほうが斜め禁止より短いはず（0.30728 < 0.36）
    nodiag_path = shortest_path(
        build_gap_graph(maze, FloodMode.PESSIMISTIC, allow_diagonal=False), maze, start, goals
    )
    assert path.distance_m < nodiag_path.distance_m


# ==========================================================================
# 2. 等価性の錨（PREREG §3・合格条件）: 斜め禁止時、既存の区画グラフと一致する
# ==========================================================================
@pytest.mark.parametrize("seed", DESIGN_V4_SAMPLE_SEEDS)
def test_equivalence_anchor_no_diagonal_matches_flood(seed):
    maze = _load_design_v4_maze(seed)
    start = (0, 0)
    goals = standard_goal_cells(maze.width, maze.height)

    d_cell = _cell_graph_distance_m(maze, FloodMode.PESSIMISTIC, start, goals)
    assert d_cell is not None  # design_v4 の迷路は全てゴールへ到達可能なはず

    graph = build_gap_graph(maze, FloodMode.PESSIMISTIC, allow_diagonal=False)
    path = shortest_path(graph, maze, start, goals)

    assert path.distance_m == pytest.approx(d_cell, abs=1e-4)


# ==========================================================================
# 3. 折れ線の総長がダイクストラ法の経路長と一致すること（斜め許可の場合も）
# ==========================================================================
@pytest.mark.parametrize("seed", DESIGN_V4_SAMPLE_SEEDS)
@pytest.mark.parametrize("allow_diagonal", [False, True])
def test_polyline_length_matches_distance(seed, allow_diagonal):
    maze = _load_design_v4_maze(seed)
    start = (0, 0)
    goals = standard_goal_cells(maze.width, maze.height)

    graph = build_gap_graph(maze, FloodMode.PESSIMISTIC, allow_diagonal=allow_diagonal)
    path = shortest_path(graph, maze, start, goals)

    assert polyline_length_m(path.xy_m) == pytest.approx(path.distance_m, abs=1e-4)
    # 折れ線は出発区画の中心から始まりゴール区画の中心で終わる
    from classic.gap_graph import cell_center_xy

    assert tuple(path.xy_m[0]) == pytest.approx(cell_center_xy(*start), abs=1e-5)
    goal_reached = tuple(path.xy_m[-1])
    assert any(
        goal_reached == pytest.approx(cell_center_xy(*g), abs=1e-5) for g in goals
    )


# ==========================================================================
# 4. 否定対照 N1（PREREG §4）: 斜めの辺のコストを直進と同じにすると ρ=1 になる
# ==========================================================================
def test_negative_control_n1_equal_diagonal_cost_forces_rho_one():
    maze = _load_design_v4_maze(41003)
    start = (0, 0)
    goals = standard_goal_cells(maze.width, maze.height)
    mode = FloodMode.PESSIMISTIC

    d_nodiag = shortest_path(
        build_gap_graph(maze, mode, allow_diagonal=False), maze, start, goals
    ).distance_m
    d_diag_normal = shortest_path(
        build_gap_graph(maze, mode, allow_diagonal=True), maze, start, goals
    ).distance_m
    # 通常の斜めコストでは実際に縮む（この迷路・この設定では実測で確かめた）
    assert d_diag_normal < d_nodiag - 1e-6

    d_diag_forced = shortest_path(
        build_gap_graph(maze, mode, allow_diagonal=True, diag_dist_m=STRAIGHT_GAP_DIST_M),
        maze,
        start,
        goals,
    ).distance_m
    # N1: 斜めのコストを直進と同じ(0.18m)にすると、斜め禁止と同じ距離まで戻る（ρ=1）
    assert d_diag_forced == pytest.approx(d_nodiag, abs=1e-4)


# ==========================================================================
# 5. 否定対照 N2（PREREG §4）: 柱間を全て壁にすると経路が存在しない
# ==========================================================================
def test_negative_control_n2_all_gaps_walled_has_no_route():
    maze = MazeMap(6, 6)
    maze.v_walls[:, :] = WallState.WALL
    maze.h_walls[:, :] = WallState.WALL
    start = (0, 0)
    goals = standard_goal_cells(6, 6)

    graph = build_gap_graph(maze, FloodMode.PESSIMISTIC, allow_diagonal=True)
    with pytest.raises(NoGapRouteError):
        shortest_path(graph, maze, start, goals)


# ==========================================================================
# 6. 否定対照 N3（PREREG §4）: 壁配列を無作為に置き換えると、区画グラフとの
#    一致（§3の錨）は崩れないまま（=実装は壁を読んでいる）、距離そのものは
#    置き換えるたびに変わる。崩れなければ／変わらなければ、壁を読んでいない
#    ことを疑う。
#    値は事前に実測して固定した（8x8, P(open)=0.5, 決め打ちのseed）:
#    seed 0 -> 到達不能、seed 2 -> 6歩、seed 7 -> 7歩、seed 13 -> 9歩
#    （`python -c` での実測。本ファイルは「〜のはず」ではなくこの実測値を使う）。
# ==========================================================================
@pytest.mark.parametrize(
    "seed,expected_steps",
    [(0, None), (2, 6), (7, 7), (13, 9)],
)
def test_negative_control_n3_random_walls_change_distance(seed, expected_steps):
    width = height = 8
    start = (0, 0)
    goals = standard_goal_cells(width, height)
    mode = FloodMode.PESSIMISTIC

    rng = np.random.RandomState(seed)
    maze = MazeMap(width, height)
    v_open = rng.rand(*maze.v_walls.shape) < 0.5
    h_open = rng.rand(*maze.h_walls.shape) < 0.5
    maze.v_walls[1:width, :] = np.where(v_open[1:width, :], WallState.OPEN, WallState.WALL)
    maze.h_walls[:, 1:height] = np.where(h_open[:, 1:height], WallState.OPEN, WallState.WALL)

    d_cell = _cell_graph_distance_m(maze, mode, start, goals)
    d_gap = _gap_distance_or_none(maze, mode, allow_diagonal=False, start=start, goals=goals)

    if expected_steps is None:
        assert d_cell is None  # 実測どおり到達不能であることの確認（前提が崩れていないか）
        assert d_gap is None
    else:
        expected_m = expected_steps * CELL_PITCH_M
        assert d_cell == pytest.approx(expected_m, abs=1e-9)
        assert d_gap == pytest.approx(expected_m, abs=1e-4)

    # 既存の区画グラフとの一致（§3の錨）は、無作為に置き換えた壁でも崩れない
    assert (d_cell is None) == (d_gap is None)
    if d_cell is not None:
        assert d_gap == pytest.approx(d_cell, abs=1e-4)


def test_negative_control_n3_distances_actually_vary_with_the_walls():
    """N3 の核心: 壁配列を置き換えるたびに柱間グラフの距離が実際に変わること
    （変わらなければ gap_graph.py が壁を読んでいないバグを疑う）。"""
    width = height = 8
    start = (0, 0)
    goals = standard_goal_cells(width, height)
    mode = FloodMode.PESSIMISTIC

    observed = []
    for seed in (2, 7, 13):
        rng = np.random.RandomState(seed)
        maze = MazeMap(width, height)
        v_open = rng.rand(*maze.v_walls.shape) < 0.5
        h_open = rng.rand(*maze.h_walls.shape) < 0.5
        maze.v_walls[1:width, :] = np.where(v_open[1:width, :], WallState.OPEN, WallState.WALL)
        maze.h_walls[:, 1:height] = np.where(h_open[:, 1:height], WallState.OPEN, WallState.WALL)
        d = _gap_distance_or_none(maze, mode, allow_diagonal=False, start=start, goals=goals)
        assert d is not None
        observed.append(round(d, 6))

    assert len(set(observed)) == len(observed), (
        f"壁配列を変えても柱間グラフの距離が変わらなかった: {observed}"
        "（gap_graph.py が壁を読んでいない可能性がある）"
    )


# ==========================================================================
# 7. 未知(UNKNOWN)の扱い（楽観・悲観）が classic/flood.py と同じ向きに効くこと
# ==========================================================================
def test_unknown_handling_matches_flood_both_modes():
    """探索途中を模した「一部だけ既知」の地図（内部壁の95%を真値で埋め、
    残り5%を未知のままにした。seed=18・design_v4 seed=41003 で実測すると
    楽観25歩・悲観91歩と大きく割れる組み合わせが見つかったので、これを固定する）。
    楽観・悲観のどちらも、gap_graph.py の距離が classic/flood.py の距離と
    完全に一致し、かつ楽観 <= 悲観（未知を通れるとみなす方が短い/同じ）に
    なることを確かめる。"""
    truth = _load_design_v4_maze(41003)
    start = (0, 0)
    goals = standard_goal_cells(truth.width, truth.height)

    rng = np.random.RandomState(18)
    partial = MazeMap(truth.width, truth.height)
    keep_known_v = rng.rand(*partial.v_walls.shape) < 0.95
    keep_known_h = rng.rand(*partial.h_walls.shape) < 0.95
    partial.v_walls[keep_known_v] = truth.v_walls[keep_known_v]
    partial.h_walls[keep_known_h] = truth.h_walls[keep_known_h]

    d_cell_opt = _cell_graph_distance_m(partial, FloodMode.OPTIMISTIC, start, goals)
    d_cell_pess = _cell_graph_distance_m(partial, FloodMode.PESSIMISTIC, start, goals)
    assert d_cell_opt is not None and d_cell_pess is not None
    # 実測値を固定（歩数）。事前の期待ではなく、実際に計算して得た値。
    assert d_cell_opt == pytest.approx(25 * CELL_PITCH_M, abs=1e-9)
    assert d_cell_pess == pytest.approx(91 * CELL_PITCH_M, abs=1e-9)

    d_gap_opt = _gap_distance_or_none(partial, FloodMode.OPTIMISTIC, False, start, goals)
    d_gap_pess = _gap_distance_or_none(partial, FloodMode.PESSIMISTIC, False, start, goals)

    assert d_gap_opt == pytest.approx(d_cell_opt, abs=1e-4)
    assert d_gap_pess == pytest.approx(d_cell_pess, abs=1e-4)
    # 楽観(未知=通れる)のほうが悲観(未知=通れない)以下になる、同じ向き
    assert d_gap_opt <= d_gap_pess + 1e-9


def test_unknown_handling_pessimistic_can_be_unreachable_while_optimistic_is_not():
    """地図がまだ半分しか埋まっていない段階では、悲観（最短走行の判定に使う
    `classic.flood.is_reachable_known` と同じ考え方）は「まだ確実には
    たどり着けない」と判定してよい。楽観は探索用なので通れる。"""
    truth = _load_design_v4_maze(41003)
    start = (0, 0)
    goals = standard_goal_cells(truth.width, truth.height)

    rng = np.random.RandomState(0)
    partial = MazeMap(truth.width, truth.height)
    keep_known_v = rng.rand(*partial.v_walls.shape) < 0.5
    keep_known_h = rng.rand(*partial.h_walls.shape) < 0.5
    partial.v_walls[keep_known_v] = truth.v_walls[keep_known_v]
    partial.h_walls[keep_known_h] = truth.h_walls[keep_known_h]

    d_cell_opt = _cell_graph_distance_m(partial, FloodMode.OPTIMISTIC, start, goals)
    d_cell_pess = _cell_graph_distance_m(partial, FloodMode.PESSIMISTIC, start, goals)
    assert d_cell_opt is not None
    assert d_cell_pess is None  # 実測どおり悲観では未到達

    d_gap_opt = _gap_distance_or_none(partial, FloodMode.OPTIMISTIC, False, start, goals)
    d_gap_pess = _gap_distance_or_none(partial, FloodMode.PESSIMISTIC, False, start, goals)
    assert d_gap_opt == pytest.approx(d_cell_opt, abs=1e-4)
    assert d_gap_pess is None
