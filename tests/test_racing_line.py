"""
tests/test_racing_line.py
================
`classic/racing_line.py`（柱間グラフの折れ線の角を丸めて連続な走行ラインにする
モジュール。`experiments/exp_036_racing_line/PREREG.md` の実装）の単体テスト。

    source .venv/bin/activate
    python -m pytest tests/test_racing_line.py -v

作業指示「作るもの その2」の7項目に対応する（各テストのdocstringに番号を書く）:
  1. 直線だけの折れ線を通すと曲率が0のままであること
  2. 直角1つを丸めたときの曲率と弧長を、手計算と照合する
  3. 45°の頂点（斜めの折れ線）を丸められること
  4. 曲率が連続であること（跳びの最大値を固定する。跳びがds(格子間隔)に
     比例して縮むことも確かめる — 円弧方式（`classic/ideal.py`）のような
     真の不連続なら、どれだけ細かく刻んでも跳びは縮まらない。ここが両者の
     決定的な違いである）
  5. 余裕が正であること（既知の迷路 `design_v4/maze_41003` で最小値を固定する）
  6. 丸めを深くしていくと、どこかで余裕が負になる（探索が効いていることの確認）
  7. 出力が`classic/profile.py`の`min_time()`にそのまま渡せること

🔴 「〜のはず」で検査を書かない（作業指示）: 以下の固定値はすべて、実装した
コードを実際に走らせて得た値をそのまま書き写したものである（値の由来は各
テストのコメントに残す）。手計算で照合できる量（2番の弧長・曲率）は別途
symbolic に確かめる。

`classic/gap_graph.py`・`classic/profile.py`・`classic/geometry.py`・
`classic/maze_map.py`・`classic/racing_line.py` は一切変更しない（読むだけ）。
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pytest

from classic.flood import FloodMode
from classic.gap_graph import (
    GapPath,
    build_gap_graph,
    cell_center_xy,
    shortest_path,
    standard_goal_cells,
)
from classic.maze_map import MazeMap, WallState
from classic.profile import min_time, vehicle_limits
from classic.racing_line import (
    RacingLineError,
    RacingLineOverlapError,
    _corner_geometry,
    _find_corners,
    build_racing_line,
    diagonal_length_m,
    evaluate_clearance,
    find_max_feasible_racing_line,
    max_kappa_jump,
    to_segments,
)

Cell = Tuple[int, int]

DESIGN_V4_DIR = os.path.join(REPO_ROOT, "competition", "mazes", "design_v4")


# ==========================================================================
# テスト用ヘルパ（`tests/test_gap_graph.py`と同じ複製パターン）
# ==========================================================================
def _full_open_interior(width: int, height: int) -> MazeMap:
    """外周だけ壁・内部は全て開通の迷路（`tests/test_gap_graph.py`と同じ複製）。"""
    maze = MazeMap(width, height)
    maze.v_walls[1:width, :] = WallState.OPEN
    maze.h_walls[:, 1:height] = WallState.OPEN
    return maze


def _maze_from_truth_walls(v_walls: np.ndarray, h_walls: np.ndarray) -> MazeMap:
    """npzの真の壁配列から全壁既知のMazeMapを作る（`tests/test_gap_graph.py`と同じ複製）。"""
    width = v_walls.shape[0] - 1
    height = v_walls.shape[1]
    maze = MazeMap(width, height)
    maze.v_walls[:, :] = np.where(v_walls != 0, int(WallState.WALL), int(WallState.OPEN))
    maze.h_walls[:, :] = np.where(h_walls != 0, int(WallState.WALL), int(WallState.OPEN))
    return maze


def _load_design_v4_maze(seed: int) -> MazeMap:
    d = np.load(os.path.join(DESIGN_V4_DIR, f"maze_{seed}.npz"))
    return _maze_from_truth_walls(d["v_walls"], d["h_walls"])


def _narrow_corridor_maze(width: int, height: int, corridor_cells: List[Cell]) -> MazeMap:
    """`corridor_cells`（経路順のセル列）の隣接セル間だけ壁を開け、それ以外は
    すべて壁にする（幅0.18mの正真正銘の1本道を作るためのテスト専用ヘルパ。
    合格条件6の検証には、通路の外周が本物の壁として存在することが要る
    — `classic.geometry.wall_obstacles`が柱だけでなく壁も障害物にするので、
    「内部全開通」の迷路では丸めた経路がどれだけ膨らんでも柱にしか当たらず、
    通路の外壁に当たる状況を作れない）。"""
    maze = MazeMap(width, height)
    maze.v_walls[:, :] = WallState.WALL
    maze.h_walls[:, :] = WallState.WALL
    for (ax, ay), (bx, by) in zip(corridor_cells[:-1], corridor_cells[1:]):
        if ax == bx:
            maze.h_walls[ax, max(ay, by)] = WallState.OPEN
        else:
            maze.v_walls[max(ax, bx), ay] = WallState.OPEN
    return maze


def _make_gap_path(xy: np.ndarray) -> GapPath:
    """`GapPath`を手で組み立てる（テスト専用の合成折れ線。実際の柱間グラフの
    出力ではないが、`racing_line.py`は`xy_m`と`distance_m`しか使わないので
    十分。`node_ids`/`is_diagonal`はここでは使わないダミー値）。"""
    xy = xy.astype(np.float32)
    dist = float(np.sum(np.linalg.norm(np.diff(xy.astype(np.float64), axis=0), axis=1)))
    return GapPath(distance_m=dist, node_ids=[], xy_m=xy, is_diagonal=[False] * (len(xy) - 1))


# ==========================================================================
# 1. 直線だけの折れ線を通すと曲率が0のままであること
# ==========================================================================
def test_straight_only_path_keeps_kappa_zero():
    """4x4迷路、y=0の行だけ東西に開通した1本道（出発(0,0)〜ゴール(3,0)）。
    最短経路は方向転換の無い一直線になるはずで、丸める角が1つも無いので
    走行ラインは曲率0のまま（角の丸め処理自体が走らない）。"""
    maze = MazeMap(4, 4)
    maze.v_walls[:, :] = WallState.WALL
    maze.h_walls[:, :] = WallState.WALL
    maze.v_walls[1:4, 0] = WallState.OPEN  # y=0の行を東西に開通
    start, goals = (0, 0), [(3, 0)]

    graph = build_gap_graph(maze, FloodMode.PESSIMISTIC, allow_diagonal=True)
    path = shortest_path(graph, maze, start, goals)
    assert not any(path.is_diagonal)  # 前提: 本当に直線だけの経路であること

    line = build_racing_line(path, R_m=0.05)
    assert line.corner_count == 0
    assert line.kappa_grid == [0.0]  # 曲率0の1セルで表される（モジュールの設計）
    assert line.kind_grid == ["straight"]
    assert line.total_length_m == pytest.approx(path.distance_m, abs=1e-5)

    # min_time にも問題なく渡せる（検査7の前倒し確認）。経路長0.54mは短く、
    # 実際に計算すると速度は台形に達する前の三角形プロファイル（頭打ちなし）
    # になる（n_triangular=1。「最高速度に達するはず」という思い込みでは
    # なく、実際に計算した結果をそのまま書く）。
    segs = to_segments(line)
    it = min_time(segs, vehicle_limits(), v_start=0.0, v_end=0.0)
    assert it.total > 0.0
    assert it.n_triangular == 1 and it.n_trapezoidal == 0
    assert not it.reached_v_top


# ==========================================================================
# 2. 直角1つを丸めたときの曲率と弧長を、手計算と照合する
# ==========================================================================
def _single_right_angle_corner():
    """4x4・内部全開通迷路で斜めを禁じると、柱間グラフは区画(1,1)の中心を
    経由する90°の「曲がり角」を作る（`classic/gap_graph.py`の「中心経由」の
    仕組み。`tests/test_gap_graph.py::test_hand_computed_4x4_no_diagonal_matches_manhattan`
    と同じ迷路・同じ経路）。実際に計算すると角は1つだけで、旋回角は-90°
    （進入=北、退出=東の右折）になる。"""
    maze = _full_open_interior(4, 4)
    start, goals = (0, 0), standard_goal_cells(4, 4)
    graph = build_gap_graph(maze, FloodMode.PESSIMISTIC, allow_diagonal=False)
    path = shortest_path(graph, maze, start, goals)
    corners = _find_corners(path.xy_m)
    assert len(corners) == 1  # 実際に計算して確かめた前提
    return maze, path, corners[0]


def test_hand_computed_single_right_angle_corner():
    """手計算:

    旋回角 delta = -90° = -pi/2 (右折)。丸めの深さ R=0.02m を選ぶ。

    往復クロソイドの定義（モジュールdocstring）:
        a・L = 1/R                    … 立ち上がり終端の曲率が1/R
        a・L^2 = |delta|              … 立ち上がり半区間で旋回角|delta|/2を使い切る
                                         (a・L・L/2 = (1/R)・L/2 = |delta|/2 と等価)

    この2式から L = |delta|・R, a = 1/(|delta|・R^2)。

        L = (pi/2) * 0.02 = 0.03141592653589793 m   … 片道の弧長
        kappa_max = a*L = 1/R = 50.0 [1/m]            … 頂点での曲率のピーク

    全長は往復（立ち上がり+立ち下がり）で 2*L = 0.06283185307179586 m。
    """
    R = 0.02
    maze, path, corner = _single_right_angle_corner()
    assert corner.delta_rad == pytest.approx(-math.pi / 2.0, abs=1e-9)

    g = _corner_geometry(corner.delta_rad, corner.heading_in, R, ds=0.002)
    L_hand = abs(corner.delta_rad) * R
    kappa_max_hand = 1.0 / R
    assert g.L == pytest.approx(L_hand, abs=1e-12)
    assert g.a * g.L == pytest.approx(kappa_max_hand, abs=1e-9)

    line = build_racing_line(path, R_m=R)
    assert line.corner_count == 1
    corner_total_len = sum(
        line.s_grid[i + 1] - line.s_grid[i]
        for i, k in enumerate(line.kind_grid) if k == "corner"
    )
    assert corner_total_len == pytest.approx(2.0 * L_hand, abs=1e-6)

    # セル代表値はセル中点の曲率なので、ピークそのもの(50.0)よりわずかに
    # 小さい値になる（実際に計算して得た値。以後は変更で値がずれたら気づけるよう固定する）。
    assert line.kappa_max_abs() == pytest.approx(48.4375, abs=1e-9)
    assert line.kappa_max_abs() < kappa_max_hand  # 中点則なのでピーク未満のはず

    min_clear, _ = evaluate_clearance(line, maze)
    assert min_clear == pytest.approx(0.027883576724588667, abs=1e-9)
    assert min_clear > 0.0


# ==========================================================================
# 3. 45°の頂点（斜めの折れ線）を丸められること
# ==========================================================================
def test_45_degree_diagonal_corner_can_be_rounded():
    """同じ4x4・内部全開通迷路で斜めを許すと、経路は45°の頂点を2つ持つ
    折れ線になる（`tests/test_gap_graph.py::test_hand_computed_4x4_diagonal_matches_hand_route`
    と同じ経路: 出発中心→柱間(斜め45°)→柱間→ゴール中心）。**ここを通さない
    検査は意味が無い**（作業指示）— 直角しか丸められない実装は45°で例外を
    出すか、閉じずに位置がずれる。"""
    maze = _full_open_interior(4, 4)
    start, goals = (0, 0), standard_goal_cells(4, 4)
    graph = build_gap_graph(maze, FloodMode.PESSIMISTIC, allow_diagonal=True)
    path = shortest_path(graph, maze, start, goals)
    assert any(path.is_diagonal)  # 前提: 本当に斜めを含む経路であること

    corners = _find_corners(path.xy_m)
    assert [c.delta_rad for c in corners] == pytest.approx([-math.pi / 4.0, -math.pi / 4.0], abs=1e-9)

    R = 0.02
    line = build_racing_line(path, R_m=R)  # 例外を出さずに（閉じて）組み立てられること自体が検査
    assert line.corner_count == 2
    assert len(line.kappa_grid) == 154         # 実測値（変更で値がずれたら気づけるよう固定する）
    assert line.kappa_max_abs() == pytest.approx(46.875, abs=1e-9)  # 実測値

    min_clear, _ = evaluate_clearance(line, maze)
    assert min_clear == pytest.approx(0.015154331461361477, abs=1e-9)
    assert min_clear > 0.0

    # min_timeにもそのまま渡せる
    it = min_time(to_segments(line), vehicle_limits(), v_start=0.0, v_end=0.0)
    assert it.total > 0.0


# ==========================================================================
# 4. 曲率が連続であること（跳びの最大値を固定する。ds依存を確かめる）
# ==========================================================================
def test_curvature_jump_shrinks_with_finer_grid():
    """円弧方式（`classic/geometry.py::turn_path`が使う、直線→円弧の接続点）は
    どれだけ細かく刻んでも曲率の跳びが縮まらない**真の不連続**である。
    往復クロソイドは曲率0から線形に立ち上がるので、格子を細かくするほど
    セル間の跳びは縮む（離散化誤差そのものであり、物理的な不連続ではない）。

    45°の頂点2つを持つ経路（上のテストと同じ）でds=0.002mとds=0.0005m
    （ちょうど1/4）を比べると、跳びの最大値もちょうど1/4になる
    （実際に計算して確かめた値。線形なランプなので厳密に比例する）。"""
    maze = _full_open_interior(4, 4)
    start, goals = (0, 0), standard_goal_cells(4, 4)
    graph = build_gap_graph(maze, FloodMode.PESSIMISTIC, allow_diagonal=True)
    path = shortest_path(graph, maze, start, goals)

    R = 0.02
    line_coarse = build_racing_line(path, R_m=R, ds=0.002)
    line_fine = build_racing_line(path, R_m=R, ds=0.0005)

    jump_coarse = max_kappa_jump(line_coarse)
    jump_fine = max_kappa_jump(line_fine)

    assert jump_coarse == pytest.approx(6.25, abs=1e-9)     # 実測値（変更で値がずれたら気づけるよう固定する）
    assert jump_fine == pytest.approx(1.5625, abs=1e-9)      # 実測値（変更で値がずれたら気づけるよう固定する）
    # ds を1/4にすると跳びもちょうど1/4（線形ランプの離散化誤差なので厳密に比例）。
    # 円弧方式ならこの比は1（跳びの大きさ=1/Rで、格子を細かくしても変わらない）。
    assert jump_coarse / jump_fine == pytest.approx(4.0, rel=1e-6)


# ==========================================================================
# 5. 余裕が正であること（既知の迷路design_v4/maze_41003で最小値を固定する）
# ==========================================================================
_CACHE = {}


def _found_line_for_maze_41003():
    """`find_max_feasible_racing_line`は角ごとの数値積分＋経路全体の掃引余裕を
    30回の二分探索で行うため重い（実測: 約7秒）。テスト間で使い回す
    （`tests/test_ideal.py`の`_result`キャッシュと同じ考え方）。"""
    key = "maze_41003"
    if key not in _CACHE:
        maze = _load_design_v4_maze(41003)
        start, goals = (0, 0), standard_goal_cells(maze.width, maze.height)
        graph = build_gap_graph(maze, FloodMode.PESSIMISTIC, allow_diagonal=True)
        path = shortest_path(graph, maze, start, goals)
        line = find_max_feasible_racing_line(path, maze)
        _CACHE[key] = (maze, path, line)
    return _CACHE[key]


def test_clearance_is_positive_on_a_known_maze():
    maze, _path, line = _found_line_for_maze_41003()
    min_clear, _idx = evaluate_clearance(line, maze)
    assert min_clear == pytest.approx(0.014037256799964214, abs=1e-9)  # 実測値（変更で値がずれたら気づけるよう固定する）
    assert min_clear > 0.0
    assert min_clear >= 0.005 - 1e-9  # PREREG既定のmargin_m以上


def test_min_time_accepts_the_racing_line_segments_directly():
    """検査7: `to_segments()`の出力が`classic.profile.min_time()`にそのまま渡せる。"""
    _maze, _path, line = _found_line_for_maze_41003()
    segs = to_segments(line)
    it = min_time(segs, vehicle_limits(), v_start=0.0, v_end=0.0)
    assert it.total == pytest.approx(14.98740235456902, rel=1e-6)  # 実測値（変更で値がずれたら気づけるよう固定する）
    assert math.isfinite(it.total) and it.total > 0.0
    assert set(it.by_kind.keys()) <= {"straight", "diagonal", "corner"}


# ==========================================================================
# 6. 丸めを深くしていくと、どこかで余裕が負になる（探索が効いていることの確認）
# ==========================================================================
def test_deepening_the_rounding_eventually_makes_clearance_negative():
    """「内部全開通」の迷路は柱にしか当たらないので、この検査には**本物の壁で
    囲われた幅0.18mの1本道**が要る（`_narrow_corridor_maze`のdocstring参照）。
    東へ10区画・北へ10区画のL字の1本道を作り、中心を結ぶ直線+1つの90°角
    という単純な経路（`GapPath`を手で合成）を、丸めの深さRだけ変えて試す。

    直線区間がどちらも約1.8m（10区画分）と長いので、消費長の予算（隣り合う
    角どうしの奪い合い）はこの範囲のRでは効かず、**壁との干渉だけ**でRの
    限界が決まる（`RacingLineOverlapError`が出ないことをまず確認する）。
    """
    N = 11
    corridor = [(i, 0) for i in range(N)] + [(N - 1, j) for j in range(1, N)]
    maze = _narrow_corridor_maze(N + 1, N + 1, corridor)
    xy = np.array([cell_center_xy(0, 0), cell_center_xy(N - 1, 0), cell_center_xy(N - 1, N - 1)])
    path = _make_gap_path(xy)

    line_safe = build_racing_line(path, R_m=0.05)  # 消費長の予算にも余裕がある浅い丸め
    min_clear_safe, _ = evaluate_clearance(line_safe, maze)
    assert min_clear_safe == pytest.approx(0.034000003576278685, abs=1e-6)  # 実測値
    assert min_clear_safe > 0.0

    line_deep = build_racing_line(path, R_m=0.15)  # 深い丸め（例外は出ない=予算はまだ余裕がある）
    min_clear_deep, _ = evaluate_clearance(line_deep, maze)
    assert min_clear_deep == pytest.approx(-0.0046757801333727755, abs=1e-6)  # 実測値
    assert min_clear_deep < 0.0  # 探索が無ければここで壁にめり込む

    # 探索を使うと、margin_m(既定0.005)ちょうどの境界で止まる。
    found = find_max_feasible_racing_line(path, maze)
    min_clear_found, _ = evaluate_clearance(found, maze)
    assert min_clear_found >= 0.005 - 1e-6
    assert min_clear_found == pytest.approx(0.005000000108040812, abs=1e-6)  # 実測値
    assert 0.05 < found.R_m < 0.15  # 安全側と危険側のあいだで見つかったこと


def test_overlap_error_is_a_distinct_failure_mode_from_clearance():
    """`build_racing_line`は壁との干渉を見ない（`evaluate_clearance`が別関数な
    のはこのため）。ここでは短い直線（0.09m）1本の片側だけが角に食われる
    単純な例で、消費長の予算超過（`RacingLineOverlapError`）が壁とは無関係に
    単独で作動することを確かめる（`find_max_feasible_racing_line`が両方を
    区別なく「Rを下げる」条件として扱えることの前提）。"""
    maze, path, _corner = _single_right_angle_corner()
    with pytest.raises(RacingLineOverlapError):
        build_racing_line(path, R_m=0.05)  # T(0.05)=0.0935m > セグメント長0.09m
    # R=0.02（上のテストで使った値）は同じ経路で問題なく通る
    build_racing_line(path, R_m=0.02)


# ==========================================================================
# 7. 出力がclassic/profile.pyのmin_time()にそのまま渡せること（統合確認）
# ==========================================================================
def test_to_segments_output_shape_matches_profile_segment():
    """`to_segments`が返す`Segment`が`classic.profile.Segment`そのもので
    あること（型・フィールド名の整合。`min_time`のシグネチャに合わせて
    あることの直接確認）。"""
    from classic.profile import Segment as ProfileSegment

    _maze, _path, line = _found_line_for_maze_41003()
    segs = to_segments(line)
    assert len(segs) == len(line.kappa_grid)
    assert all(isinstance(s, ProfileSegment) for s in segs)
    assert all(s.length > 0.0 for s in segs)
    total_len = sum(s.length for s in segs)
    assert total_len == pytest.approx(line.total_length_m, rel=1e-9)


# ==========================================================================
# 補足: 副次記録の補助関数が経路と矛盾しない値を返すこと
# ==========================================================================
def test_diagonal_length_is_within_total_length():
    _maze, _path, line = _found_line_for_maze_41003()
    diag_len = diagonal_length_m(line)
    assert 0.0 <= diag_len <= line.total_length_m
