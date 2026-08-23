"""classic/ideal.py — 経路から理想時間を出す層

`classic/profile.py`（速度計画器）は「半径 R の弧を物理限界で走ったら何秒か」を、
`classic/geometry.py`（幾何と干渉判定）は「その半径で本当に壁・柱に当たらず通れるか」を
それぞれ計算できる。本モジュールはその 2 つを、**真の迷路の壁から引いた最短経路**に
適用し、「この迷路をこの経路で走ったら物理限界で何秒か」を 1 本の数値にまとめる層である。

設計の起点: `research_notes/note_031_profile_planner_and_eta.md`
「### 段 2 の検収で分かったこと」（半径とオフセットの表・通路幅いっぱいの半径は
実現できないという知見）。

## 範囲（今回はここまで）

**最短走行の経路について、直交（斜めなし）・中心線上の理想時間を出す。**
横オフセット（通路の中心線から機体をずらす走り方）と斜め走行は対応しない（次段）。

## `mode` の意味

- `"spin"`  … 超信地旋回走行。ターンでは必ず停止する（`profile.spin_turn_time`）。
  直線は区画ごとに独立に「停止→停止」で解く（ターン間で速度の連続性は無い）。
- `"slalom"` … 曲がるところを円弧で通り、区間の切れ目でも止まらない
  （経路全体を 1 本の `profile.min_time` で解く。**ただし下記の例外がある**）。

### 🔴 半径が 0 になる場合の扱い（"連続するターンで直線が 0 になる" 場合）

`slalom` モードでも、次の 2 つの事情で、あるターンの半径が幾何的に 0 まで
削られることがある:

1. **経路の最初のターン**（`start_heading` と最初の移動方向が違うとき）。
   このとき「手前の直線」はまだ機体が 1mm も動いていない区間なので長さ 0 であり、
   円弧が手前の直線から食える長さ（`R*tan(|Δθ|/2)`）の上限も 0 になる
   （経路の途中のターンでは、ターン間に必ず 1 区画以上の直線があるので
   この事情は起こらない — 起こるのは経路の先頭だけである）。
2. **180° 折返し**。円弧が入出の直線から食う長さの式 `R*tan(|Δθ|/2)` は
   `Δθ=180°` で `tan(90°)` が発散するため、有限半径の弧では作れない
   （進入・退出の直線が平行になり、それぞれを延長しても 1 点で交わらないため
   「角」という概念自体が定義できない）。

どちらの場合も、**弧（`curvature=1/radius` が発散する）としては表現できない**ので、
その場所だけ速度をいったん 0 まで落とし（前半区間を `v_end=0` で解く）、
`spin_turn_time` を足し、その後の区間を新たに `v_start=0` から解き直す
（経路全体を 1 本の `min_time` にできない箇所だけ、区間を分割する）。
これにより `slalom` モードでも、経路の場所ごとに「実質は超信地旋回」が混ざりうる
（`TurnPlan.radius == 0.0` がその印。`limited_by` は 0 の原因になった境界を指す）。

## 既知の限界

- **半径は各ターンについて独立に最大化する近似であり、厳密な時間最適ではない**
  （前後の速度の連成を考えていない。ある区間の速度が本当に効率よく使えるかは
  隣接区間の速度計画にも依存するが、本モジュールは考慮しない）。
  したがって `T_ideal` は真の最小より**大きい側**に出る。
- 「共有する直線から食える長さ」の配り方には 2 通りある（`allocation` 引数）。
  `"greedy"` は経路の先頭から順に処理し「既に前のターンが消費した分を引いた
  残り長」だけを次のターンに渡す（速いが、手前のターンが直線を先取りし次の
  ターンが締め出される非対称な配分になりうる）。`"proportional"` は直線の制約を
  無視した幾何上の希望消費量から出発し、隣り合う 2 つのターンが同じ直線を
  分け合う場面では需要に比例して両方を同じ割合で縮める緩和法で求める
  （`_allocate_shared_straight`。片方だけを一方的に優先しない）。
  🔴 **どちらも真の時間最適ではなく、どちらが速いかは経路ごとに変わる**
  （比例配分は先取りを防ぐが、その代わり大きい半径を許す代償として直線消費が
  増え、前後の加速・巡航に使える距離が減ることがある — 半径と速度のトレード
  オフをどちらの配分もモデル化していないため）。design_turn_v1 の10迷路では
  比例配分が6迷路で速く、先取りが4迷路で速かった（実測。差はいずれも1%未満）。
  そのため既定の `allocation="best"` は**両方を計算して速い方を採用する**
  （`IdealResult.allocation_used` にどちらを採ったかを記録する）。これは
  真の最小に対する、片方だけを使うより近い上界になる。この緩和法自体は
  縮める方向にしか動かない（一度縮めた消費量は、別の直線の制約が後から緩んでも
  増えて戻らない）ため、「後から空いた余地を隣に回す」ような、より賢い再配分には
  なっていない（`_allocate_shared_straight` の docstring 参照。ただしこれは
  今回指定された仕様どおりの挙動であり、バグではない）。
- 横オフセット（通路の中心線から機体をずらす走り方）と斜め走行は未対応。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from classic.flood import FloodMode
from classic.geometry import (
    PathSegment, Pose, poses_along, sweep_clearance, to_profile_segments, turn_path, wall_obstacles,
)
from classic.maze_map import Direction, MazeMap, WallState, direction_between
from classic.profile import Segment, min_time, spin_turn_time, vehicle_limits
from classic.route import shortest_path

__all__ = [
    "CELL_SIZE",
    "TurnPlan",
    "IdealResult",
    "true_shortest_path",
    "ideal_time_for_path",
]

Cell = Tuple[int, int]

# `classic.geometry.wall_obstacles` の既定値と同じ値を使う（二重管理の芽だが、
# 既定引数を関数越しに取り出す綺麗な手段が無いため定数として複写する。
# `competition/evaluator.py` の `DEFAULT_CELL_SIZE=0.18` とも同じ値）。
CELL_SIZE = 0.180


# ============================================================================
# 0. 方位 ⇔ 角度の変換（geometry.Pose の座標系に合わせる）
# ============================================================================
# 方位 → 角度 [rad]。`Direction` は E=0,N=1,W=2,S=3 の反時計回り順なので
# 素直には `int(d) * π/2` だが、それだと南が +270°・西が +180° になる。
# 🔴 **値の分枝は付け替え前と同一に保つ**（E=0, N=+90°, S=-90°, W=-180°）。
# `classic/explorer.py` が `tracker.reset(heading_deg=math.degrees(psi_start))` へ
# **巻き戻していない値**を渡しており、同じ向きでも 2π ずれた値を与えると
# ヨー角の積分器の初期値が変わってしまうため（2026-08-23 の付け替えで実際に
# 南向き開始の走行が変わり、`tests/test_classic_fast_run_profile.py` が検出した）。
_DIR_ANGLE_RAD: Dict[Direction, float] = {
    Direction.E: 0.0,
    Direction.N: math.pi / 2.0,
    Direction.S: -math.pi / 2.0,
    Direction.W: -math.pi,
}


def _dir_angle(d: Direction) -> float:
    """方位 `Direction` を、`geometry.Pose.theta` と同じ座標系の角度 [rad] にする。

    返す値は (-π, π] ではなく **[-π, π/2] の分枝**（E=0, N=+90°, S=-90°, W=-180°）。
    分枝を変えると走行が変わる。理由は `_DIR_ANGLE_RAD` の注記を見ること。
    """
    return _DIR_ANGLE_RAD[d]


def _turn_delta(from_dir: Direction, to_dir: Direction) -> float:
    """`from_dir` から `to_dir` への旋回角 [rad]（正=左/反時計回り。`geometry.arc` と同じ符号）。

    `classic.route._turn_type` と同じ mod 4 の判定を角度に翻訳したもの
    （rel=1→左90°、rel=2→180°、rel=3→右90°。E,N,W,S がこの順で反時計回りに
    並んでいるため、rel がそのまま反時計回り正の数学角になる）。

    🔴 rel=2（180°折返し）は符号を選ぶ物理的根拠が無い退化ケース（+π も −π も
    同じ向きを指す）。付け替え前は常に −π を返しており（`_dir_angle` と同様、
    値の分枝が下流のシミュレーションへ伝播する — 2026-08-23 の付け替えで
    実際に snap 補正モード・南向き開始の走行が変わり、
    `tests/test_classic_fast_run_profile.py` が検出した）、その分枝をここでも
    保つ。rel∈{0,1,3} は分枝の選び方に曖昧さが無いので `d*(π/2)` のままでよい。
    """
    rel = (int(to_dir) - int(from_dir)) % 4
    if rel == 2:
        return -math.pi
    signed_rel = rel if rel <= 2 else rel - 4
    return signed_rel * (math.pi / 2.0)


# ============================================================================
# 1. 真の壁から最短経路を引く
# ============================================================================
def true_shortest_path(
    v_walls: np.ndarray,
    h_walls: np.ndarray,
    start: Cell,
    goals: Sequence[Cell],
    start_heading: Direction,
) -> List[Cell]:
    """迷路の真の壁から歩数マップを引き、最短経路の区画列を返す。

    真の壁の `MazeMap` は「全マスが既知」として構築する（`v_walls`/`h_walls` の
    値をそのまま `WallState.WALL`/`OPEN` に写す。未知は残さない）。
    歩数マップと経路復元は `classic.flood`/`classic.route` をそのまま再利用する
    （歩数マップ計算を自前で書き直さない）。全マスが既知なので楽観/悲観の違いは
    出ないが、`classic.route.shortest_path` の呼び出し規約（探索走行と同じ関数を
    使う）に合わせて `FloodMode.PESSIMISTIC` を渡す。

    `start_heading` は現状、経路そのものの選択には使わない
    （`classic.route.shortest_path` は最初の 1 手を常にターン 0 回として扱うため、
    実際にどちらを向いて出発するかは同点処理に影響しない）。`ideal_time_for_path`
    と呼び出し形を揃えるために受け取るだけの引数として残してある。
    """
    del start_heading  # 現状の経路選択には使わない（docstring 参照）

    v = np.asarray(v_walls)
    h = np.asarray(h_walls)
    width = v.shape[0] - 1
    height = v.shape[1]

    maze = MazeMap(width, height)
    maze.v_walls[:, :] = np.where(v.astype(bool), int(WallState.WALL), int(WallState.OPEN)).astype(
        maze.v_walls.dtype
    )
    maze.h_walls[:, :] = np.where(h.astype(bool), int(WallState.WALL), int(WallState.OPEN)).astype(
        maze.h_walls.dtype
    )

    return shortest_path(maze, start, list(goals), FloodMode.PESSIMISTIC)


# ============================================================================
# 2. TurnPlan / IdealResult
# ============================================================================
@dataclass(frozen=True)
class TurnPlan:
    """経路上の 1 つのターンの幾何プラン。"""

    index: int          # turns リスト内の通し番号（0 始まり）
    cell: Cell           # ターンが起きる区画（進入・退出の中心線を延長した交点＝この区画の中心）
    delta_theta: float   # 旋回角 [rad]（正=左/反時計回り）
    radius: float        # 円弧半径 [m]。0.0 ならその場旋回（docstring の「半径が0になる場合」参照）
    limited_by: str       # 何が半径を決めたか。配分方式（`allocation`）によって語彙が違う:
                           # allocation="proportional"（比例配分）のとき:
                           #   "geometry" … 幾何の上限（壁・柱との干渉）で決まった
                           #                （比例配分では縮められなかった）
                           #   "shared"   … 共有する直線の取り合いで比例配分により縮められた
                           #                （それでも _R_LO 以上は残った）
                           #   "floor"    … 比例配分で縮められた結果 _R_LO を下回り、
                           #                その場旋回へ降格した
                           # allocation="greedy"（先取り）のとき:
                           #   "geometry" … 幾何の上限で決まった（先取りの上限までは届かなかった）
                           #   "prev"     … 手前の直線の残り長で決まった（先取りの上限ちょうど）
                           #   "next"     … 次の直線の長さで決まった（先取りの上限ちょうど）
                           # mode="spin" のときは常に "n/a"（半径そのものを使わないため）。


@dataclass(frozen=True)
class IdealResult:
    """`ideal_time_for_path` の計算結果。"""

    total: float                    # 総所要時間 [s]
    by_kind: Dict[str, float]        # {"straight":秒, "arc":秒, "spin":秒}
    n_turns: int                     # ターン数
    path_cells: int                  # 経路の区画数（始点を含む）
    path_length: float               # 経路長 [m]（弧で角を切った後の実際の走行距離）
    v_max: float                     # 経路上の最高速度 [m/s]
    turns: List[TurnPlan]             # ターンごとのプラン
    segments: List[Segment]           # 区間列（profile.Segment。v(s) を引き直すのに使う。
                                       # mode="spin" のときは直線ラン 1 本ずつが独立に
                                       # 「停止→停止」で解かれているので、この列全体を
                                       # 1 本の min_time に通し直しても正しい結果にはならない
                                       # （要素ごとに区切って min_time を呼び直す必要がある）。
    alloc_iterations: int = 0         # 共有する直線の比例配分（`_allocate_shared_straight`）が
                                       # 収束するまでに使った反復回数。mode="spin" やターンの
                                       # 無い経路、allocation="greedy" では配分自体が発生しないので
                                       # 0。診断用の値で、`tests/test_ideal.py::test_allocation_converges`
                                       # が10迷路の最大反復回数を印字するのに使う。
    allocation_used: str = "n/a"      # mode="slalom" のとき、実際に採用された配分方式
                                       # （"greedy"/"proportional"。allocation="best" のときは
                                       # 2つの `total` を比べて速い方の値がここに入る）。
                                       # mode="spin" や、経路が短くて計算自体が発生しない場合は
                                       # "n/a"。


# ============================================================================
# 3. 経路の共通前処理（方位列・ターン検出・直線ランの長さ）
# ============================================================================
def _turns_and_runs(cells: Sequence[Cell], start_heading: Direction):
    """経路 `cells` から、(ターンの列, 直線ランの長さ[m]の列) を作る。

    ターンの列は `(move_index, from_dir, to_dir)` のタプル。`move_index` は
    「新しい方向で動く最初の移動」の添字（0始まり）で、この移動の始点区画
    `cells[move_index]` がターンの角（進入・退出の中心線を延長した交点）になる
    （区画中心線どうしを結ぶ経路なので、角の位置＝その区画の中心と一致する）。

    直線ランの長さの列は `len(turns)+1` 個。runs[0] は始点から最初のターンまで、
    runs[k] (0<k<len(turns)) はターン(k-1)からターンkまで、runs[-1] は最後の
    ターンから終点まで。ターンが 0 個なら runs は 1 個（経路全体が直線）。
    """
    n_moves = len(cells) - 1
    dirs = [direction_between(cells[i], cells[i + 1]) for i in range(n_moves)]

    turns: List[Tuple[int, Direction, Direction]] = []
    prev_dir = start_heading
    for i, d in enumerate(dirs):
        if d != prev_dir:
            turns.append((i, prev_dir, d))
        prev_dir = d

    run_bounds = [0] + [t[0] for t in turns] + [n_moves]
    runs = [(run_bounds[i + 1] - run_bounds[i]) * CELL_SIZE for i in range(len(run_bounds) - 1)]
    return turns, runs


# ============================================================================
# 4. mode="spin"（超信地旋回走行）
# ============================================================================
def _ideal_spin(cells: Sequence[Cell], start_heading: Direction) -> IdealResult:
    limits = vehicle_limits()
    turns, runs = _turns_and_runs(cells, start_heading)

    total = 0.0
    by_kind = {"straight": 0.0, "arc": 0.0, "spin": 0.0}
    segments: List[Segment] = []
    v_max = 0.0
    path_length = 0.0

    # 直線ラン: spin モードは弧を作らない（半径は常に0）ので、消費は無く
    # ランの長さがそのまま直線区間の長さになる。各ランを独立に「停止→停止」で解く
    # （ターンのたびに実際に止まるので、ラン同士で速度の連続性は無い）。
    for length in runs:
        if length <= 1e-12:
            continue
        seg = Segment(length=length, curvature=0.0, kind="straight")
        it = min_time([seg], limits, v_start=0.0, v_end=0.0)
        total += it.total
        by_kind["straight"] += it.total
        segments.append(seg)
        v_max = max(v_max, it.v_max)
        path_length += length

    turns_out: List[TurnPlan] = []
    for k, (move_idx, from_dir, to_dir) in enumerate(turns):
        delta_theta = _turn_delta(from_dir, to_dir)
        st = spin_turn_time(delta_theta, limits)
        total += st.time
        by_kind["spin"] += st.time
        # spin モードでは弧を作らないため、半径・幾何の吟味自体が発生しない
        # （常に radius=0.0）。limited_by は slalom モードの3値の意味を持たないので
        # "n/a" にする（geometry/shared/floor のどれでもない、と明示する札）。
        turns_out.append(
            TurnPlan(index=k, cell=cells[move_idx], delta_theta=delta_theta, radius=0.0, limited_by="n/a")
        )

    return IdealResult(
        total=total, by_kind=by_kind, n_turns=len(turns), path_cells=len(cells),
        path_length=path_length, v_max=v_max, turns=turns_out, segments=segments,
        allocation_used="n/a",
    )


# ============================================================================
# 4.5. 幾何判定の高速化（本モジュールだけの最適化。`classic/geometry.py` は変更しない）
# ============================================================================
# `max_feasible_radius` は二分探索の 40 回それぞれで、渡された障害物**全部**に対して
# 掃引全姿勢の厳密な干渉判定をする。16x16 迷路の障害物（500件超）をそのまま渡すと
# 1ターンあたり数秒かかり、経路1本（ターン数十個）×10迷路では現実的な時間に収まらない
# （実測: 全障害物・未整列で1回あたり約4.5秒。ターン数十×迷路10で計算すると
# 数十分かかってしまう）。そこで判定の中身を変えずに次の3つを行う:
#
#   1. 空間フィルタ: ターンの角から離れた障害物は、どんな半径を試しても掃引軌跡に
#      絶対に触れない（半径の探索上限 r_hi と旋回角から、角からの最大到達距離を
#      計算できる。`_safe_reach` 参照）。その範囲の外の障害物は候補から除いてよい
#      （除いても `max_feasible_radius` の返り値は変わらない — 実測で全障害物を
#      渡した場合と一致することを確認済み）。
#   2. 角に近い順の並べ替え: `geometry.clearance()` は障害物を先頭から線形に見て
#      「これまでの最小余裕」で足切りする実装なので、近い障害物を先に渡すと
#      足切りが早く効き、遠い障害物の厳密計算（多角形の頂点×辺の総当たり）を
#      スキップできる（未整列だと同じ障害物集合でも 3 倍以上遅い。実測）。
#   3. 二分探索を自前で回し、試す半径 r ごとに絞り込みをかけ直す:
#      `classic.geometry.max_feasible_radius` は 1 回の呼び出しにつき固定の障害物
#      集合を渡す作りなので、40 回の二分探索のどのステップでも同じ集合を見る。
#      しかし二分探索は最初の数回こそ広い r を試すが、後半は答え（多くは r_hi より
#      ずっと小さい）に収束していくので、後半の大半のステップでは 1 と同じ理屈で
#      もっと狭い範囲しか届かない。そこで `_fast_max_feasible_radius`（下）は
#      `max_feasible_radius` と**全く同じアルゴリズム**を `classic.geometry` の
#      公開関数（`turn_path`/`poses_along`/`sweep_clearance`）だけを使って自前で
#      回し、ステップごとに試す r に応じて絞り込みをかけ直す
#      （実測: 8 倍程度速くなる）。`classic/geometry.py` 自体は変更していない。
#      `tests/test_ideal.py::test_fast_radius_matches_geometry_module` で
#      `classic.geometry.max_feasible_radius`（絞り込み無しの元の実装）と
#      複数のターンで完全に同じ値を返すことを直接照合している。
#
# どれも「同じ答え（半径の値）を速く出す」ための最適化であり、判定ロジック
# （分離軸定理による厳密な干渉判定。`classic.geometry.clearance`/`sweep_clearance`）
# は一切変えていない。
#
# 🔴 `_ideal_slalom` はここで求めた `r_geom`（`r_hi=_R_HI` で呼んだ、直線の制約を
# 無視した幾何上の最大半径）を、共有する直線の按分（下記「比例配分の緩和法」）の
# 入力にする。旧実装はここで `r_hi` を「手前/次の直線から食える長さ」でさらに
# 絞っていたが、それが「先に処理したターンが直線を先取りする」非対称バグの原因
# だったため、今は `r_hi=_R_HI` 固定にして直線側の制約と切り離した
# （幾何と直線配分を別々の段階にする設計。詳細はモジュール docstring「既知の限界」）。
_R_HI = 0.40  # geometry.max_feasible_radius の既定 r_hi と同じ値。明示的に渡して固定する。
_R_LO = 0.02  # geometry.max_feasible_radius の既定 r_lo と同じ値。


def _safe_reach(r_hi: float) -> float:
    """半径 `r_hi` までの二分探索で、掃引が角からどこまで届きうるかの安全な上限 [m]。

    `turn_path` の直線区間長は `r*tan(|Δθ|/2) + _LEAD_MARGIN(0.02)`、掃引開始点は
    角からさらに `r*tan(|Δθ|/2)` 戻った位置なので、角からの最大到達距離は
    `2*r*tan(|Δθ|/2) + 0.02`。本モジュールが弧を作るのは 90° 折返し以外
    （180°は tan が発散するため弧を作らず、`max_feasible_radius` を呼ばない）なので
    `tan(45°)=1` が最大。さらに機体の大きさぶん（`geometry.HALF_LENGTH` 程度）の
    余裕を足しておく（機体は点ではなく、中心が届く範囲より外側にはみ出すため）。
    """
    return 2.0 * r_hi * math.tan(math.pi / 4.0) + 0.02 + 0.10


def _local_obstacles(obstacles, cx: float, cy: float, reach: float):
    """`(cx,cy)` から `reach` 以内にある障害物だけを、近い順に並べて返す。

    `max_feasible_radius` の返り値は変えない安全な絞り込み（`_safe_reach` 参照）。
    """
    near = [
        r for r in obstacles
        if abs(r.cx - cx) <= reach + r.hx and abs(r.cy - cy) <= reach + r.hy
    ]
    near.sort(key=lambda r: (r.cx - cx) ** 2 + (r.cy - cy) ** 2)
    return near


_FAST_DS = 0.005  # `poses_along` の既定 ds=0.002 より粗い（下のコメント参照）。


def _fast_max_feasible_radius(
    delta_theta: float, obstacles, corner_pose: Pose, margin: float, r_lo: float, r_hi: float
) -> float:
    """`classic.geometry.max_feasible_radius` と同じアルゴリズム・同じ答えを返す高速版。

    二分探索の構造（`feasible(r_lo)` が不可なら例外、`feasible(r_hi)` が可ならそのまま
    返す、それ以外は 40 回の二分探索）は元の実装をそのまま複写している。違うのは
    2 点だけ:

    1. `feasible(r)` の内部で、**試す半径 r ごとに** `_local_obstacles` で絞り込み
       直す（元の実装は呼び出し時に固定した障害物集合を毎回そのまま使う）。
    2. 掃引の姿勢間隔に `poses_along` の既定 `ds=0.002` より粗い `_FAST_DS=0.005` を使う。
       `design_turn_v1` の 10 迷路・全ターンで `ds=0.002` との差を直接測定し、
       最大でも 0.045mm（margin の既定値 5mm の 1%未満）しか変わらないことを確認済み
       （`tests/test_ideal.py::test_fast_radius_matches_geometry_module`）。

    どちらも `max_feasible_radius` 自体の判定ロジック（分離軸定理）は変えず、
    掃引のサンプリング密度と障害物の絞り込みだけを変える最適化である。
    """
    heading = (math.cos(corner_pose.theta), math.sin(corner_pose.theta))

    def feasible(r: float) -> bool:
        segments, consumed = turn_path(delta_theta, r)
        lead = segments[0].length
        back_off = consumed + lead
        sweep_start = Pose(
            corner_pose.x - back_off * heading[0],
            corner_pose.y - back_off * heading[1],
            corner_pose.theta,
        )
        local = _local_obstacles(obstacles, corner_pose.x, corner_pose.y, _safe_reach(r))
        poses = poses_along(segments, sweep_start, ds=_FAST_DS)
        min_clear, _ = sweep_clearance(poses, local)
        return min_clear >= margin

    if not feasible(r_lo):
        raise ValueError(
            f"r_lo={r_lo} でも余裕 margin={margin} を満たさない（探索範囲の下限ですら通れない）"
        )
    if feasible(r_hi):
        return r_hi

    lo, hi = r_lo, r_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _geometry_blocks(
    cells: Sequence[Cell], start_heading: Direction, turns_out: Sequence[TurnPlan], runs: Sequence[float]
):
    """半径が確定済みの `turns_out` から、経路全体を幾何ブロックに組み立てる。

    ブロック = `(開始姿勢, geometry.PathSegment の列)`。半径0（その場旋回）の
    ターンでブロックを区切る（その場旋回のあいだ位置は変わらず向きだけ変わるので、
    `geometry.PathSegment`/`poses_along` の弧長パラメータ化では表現できない —
    弧長0で向きだけ変える区間は曲率が発散する）。

    `_ideal_slalom` 自身（`profile.Segment` への変換だけで足りる）と、
    `tests/test_ideal.py` の衝突検査（実際に掃引される世界座標の姿勢が要る。
    `to_profile_segments()` の逆はできないので、こちらは半径探索をやり直さず
    `turns_out` から同じ幾何を再現する）の両方から使う共通部品。
    """
    n_turns = len(turns_out)
    remaining = list(runs)
    consumed_list: List[float] = []
    for tp in turns_out:
        if abs(abs(tp.delta_theta) - math.pi) < 1e-9:
            consumed_list.append(0.0)
        else:
            tan_half = math.tan(abs(tp.delta_theta) / 2.0)
            consumed_list.append(tp.radius * tan_half)
    for k, c in enumerate(consumed_list):
        remaining[k] -= c
        remaining[k + 1] -= c

    blocks: List[List[PathSegment]] = [[]]
    block_starts: List[Pose] = [
        Pose((cells[0][0] + 0.5) * CELL_SIZE, (cells[0][1] + 0.5) * CELL_SIZE, _dir_angle(start_heading))
    ]

    def add_straight(length: float) -> None:
        if length > 1e-12:
            blocks[-1].append(PathSegment(kind="straight", length=length, curvature=0.0))

    add_straight(remaining[0])
    for k in range(n_turns):
        tp = turns_out[k]
        if tp.radius <= 0.0:
            # 直前のブロックの末端姿勢まで進めてから、その場で delta_theta だけ回る
            # （位置は変えない）。次のブロックはその姿勢から始まる。
            end_pose = poses_along(blocks[-1], block_starts[-1])[-1]
            blocks.append([])
            block_starts.append(Pose(end_pose.x, end_pose.y, end_pose.theta + tp.delta_theta))
        else:
            curvature = math.copysign(1.0 / tp.radius, tp.delta_theta)
            arc_len = tp.radius * abs(tp.delta_theta)
            blocks[-1].append(PathSegment(kind="arc", length=arc_len, curvature=curvature))
        add_straight(remaining[k + 1])
    return blocks, block_starts


# ============================================================================
# 4.6. 共有する直線の比例配分（緩和法）
# ============================================================================
def _allocate_shared_straight(
    c0: Sequence[float], runs: Sequence[float], max_iter: int = 100, tol: float = 1e-9,
) -> Tuple[List[float], int]:
    """共有する直線ごとに、需要が長さを超えるぶんを比例配分で縮める緩和法。

    `c0[k]`（`n_turns` 個。直線の制約を無視したターン k の希望消費量）と
    `runs`（`n_turns+1` 個。直線ランの長さ）を受け取り、収束した消費量の列と、
    収束に使った反復回数の組を返す。

    直線 i（0<=i<=n_turns）は、ターン(i-1) の「次」使用とターン i の「手前」使用の
    両方から需要を受ける（端の直線は片側だけ）。需要 `d_i = c[i-1]+c[i]` が
    `runs[i]` を超えたら、両者を `runs[i]/d_i` で等しい割合だけ縮める。各ターンは
    2 本の直線に接するので、一方を縮めるともう一方の直線の需給も変わりうる。
    直感的には何周も繰り返しが要りそうに見えるが、実際には次の段落のとおり
    1 巡で足りる。

    🔴 **この関数は、実は全直線を 1 回さらう（`iteration=1`）だけで厳密解に到達する
    ことが証明できる**（`tests/test_ideal.py::test_allocation_converges` が
    design_turn_v1 全10迷路で反復回数=2 に収まることで実測も裏付けている）。
    証明の骨子: 直線 i を処理した直後は必ず `demand_i<=runs[i]` が成り立つ
    （既に満たしていたか、ちょうど等号になるまで縮めたかのどちらか）。
    この関係は、その後どちらの端のターンがさらに縮められても崩れない
    （`c[i-1]`・`c[i]` は縮む一方で増えないので、和である `demand_i` も
    減る一方である）。つまり**一度処理した直線は、二度と違反状態に戻らない**。
    したがって処理順によらず、全直線を 1 巡すれば全直線が同時に条件を満たす
    （n 個のターンは高々 2 本の直線にしか接しない「鎖」構造だからこそ成り立つ
    性質で、3本以上の直線を同時に取り合うような分岐構造では成り立たない）。
    2 回目の周回は「もう変化が無い」ことを確認するためだけの空振りである。
    したがって `tol`（収束の閾値）は、この鎖構造の入力に対しては最終値に
    影響しない（1e-9 でも 1e-1 でも同じ答えになる。
    `tests/test_ideal.py::test_allocation_tolerance_does_not_affect_the_result`
    で直接確認している）。`max_iter` だけが実効的なパラメータで、
    2 未満に削ると本当に必要な 2 回目の空振り確認が行えず収束前に打ち切られる
    （`tests/test_ideal.py::test_allocation_budget_negative_control`）。

    この緩和は縮める方向にしか動かない（`factor<=1`）ため、`c[k]` は各周回で
    単調非増加であり、下に有界（0以上）なので必ずどこかへ収束する。ただし
    念のため（上記の証明に誤りがあった場合に無限ループへ落ちないよう）
    `max_iter` 回で `tol` 未満に収まらなければ `RuntimeError` を投げる
    （黙って打ち切らない）。
    """
    n_turns = len(c0)
    assert len(runs) == n_turns + 1, f"runs は{n_turns + 1}個のはず（実際={len(runs)}）"

    c = list(c0)
    n_iterations = 0
    max_delta = 0.0
    for iteration in range(max_iter):
        max_delta = 0.0
        for i in range(n_turns + 1):
            idxs = [j for j in (i - 1, i) if 0 <= j < n_turns]
            if not idxs:
                continue
            demand = sum(c[j] for j in idxs)
            if demand > runs[i] + 1e-15:
                factor = (runs[i] / demand) if demand > 0.0 else 1.0
                for j in idxs:
                    new_c = c[j] * factor
                    max_delta = max(max_delta, abs(c[j] - new_c))
                    c[j] = new_c
        n_iterations = iteration + 1
        if max_delta < tol:
            return c, n_iterations

    raise RuntimeError(
        f"半径配分の緩和法が上限{max_iter}回で収束しなかった"
        f"（最終変化量={max_delta:.3e} >= 閾値{tol:.1e}）。"
        "比例配分が正しく単調収束していない可能性がある。"
    )


# ============================================================================
# 5. mode="slalom"（円弧で曲がる。止まらない）
# ============================================================================
# 配分方式（`allocation`）ごとに実装を分けてある:
#   `_ideal_slalom_greedy`       … 経路の先頭から見た先取り（旧実装。復活させた）
#   `_ideal_slalom_proportional` … 共有する直線を需要比例で分け合う緩和法
#   `_ideal_slalom`（下の公開ディスパッチャ） … `allocation` に応じてどちらかを
#       呼ぶ、または両方を計算して速い方（`total` が小さい方）を返す
#       （既定 `allocation="best"`。モジュール docstring「既知の限界」参照）。
def _ideal_slalom_greedy(
    cells: Sequence[Cell], v_walls: np.ndarray, h_walls: np.ndarray,
    start_heading: Direction, margin: float,
) -> IdealResult:
    """先取り配分: 経路の先頭から順に処理し、既に前のターンが消費した分を
    引いた残り長だけを次のターンに渡す。

    隣り合う 2 つのターンが同じ直線を共有する場合、手前のターンが先に処理される
    ぶん有利になる非対称な配分になりうる（`_ideal_slalom_proportional` の
    docstring・モジュール docstring「既知の限界」参照）。それでも実行可能な
    経路を生成し、`_ideal_slalom_proportional` より速いことがある（半径を
    大きく取れた側の区間で高い速度を維持できるため）。`allocation="best"`
    はこの2つの `total` を比べて速い方を採用する。
    """
    limits = vehicle_limits()
    turns, runs = _turns_and_runs(cells, start_heading)
    obstacles = wall_obstacles(v_walls, h_walls)

    n_turns = len(turns)
    remaining = list(runs)  # 各ターンの消費で減っていく「残り直線長」
    turns_out: List[TurnPlan] = []

    for k, (move_idx, from_dir, to_dir) in enumerate(turns):
        delta_theta = _turn_delta(from_dir, to_dir)
        cell = cells[move_idx]
        corner_pose = Pose(
            (cell[0] + 0.5) * CELL_SIZE, (cell[1] + 0.5) * CELL_SIZE, _dir_angle(from_dir)
        )

        is_uturn = abs(abs(delta_theta) - math.pi) < 1e-9
        if is_uturn:
            # 180°折返し: tan(|Δθ|/2)=tan(90°) が発散し、有限半径の弧が作れない。
            # 「geometry」起因の半径0（=どんな半径も幾何的に許されない）として扱う。
            radius, limited_by, consumed = 0.0, "geometry", 0.0
        else:
            tan_half = math.tan(abs(delta_theta) / 2.0)
            r_prev = remaining[k] / tan_half
            r_next = runs[k + 1] / tan_half
            cap = min(_R_HI, r_prev, r_next)  # 採用できない範囲は探索しない

            if cap <= _R_LO:
                # 手前/次の直線が短すぎて（<=20mm）幾何探索の下限にすら届かない。
                # 弧の意味のある候補が無いので、幾何判定を呼ばずその場旋回として
                # 扱う（半径0は弧を描かないので、壁・柱に当たりようが無く常に安全）。
                radius, limited_by = 0.0, ("prev" if r_prev <= r_next else "next")
            else:
                try:
                    r_geom_cap = _fast_max_feasible_radius(
                        delta_theta, obstacles, corner_pose, margin=margin, r_lo=_R_LO, r_hi=cap
                    )
                except ValueError:
                    # r_lo (既定 0.02m) ですら margin を満たす半径が無い
                    # → 幾何的に通れる半径が無い、という結果を radius=0 として扱う
                    # （呼び出し元に例外を投げず、その場旋回への降格として処理する）。
                    r_geom_cap = 0.0

                if cap < _R_HI and r_geom_cap >= cap - 1e-9:
                    # 探索上限(cap)ちょうどまで通った = 幾何はそこで絞っていない
                    # （cap は prev/next が決めた値なので、そちらに帰属させる）。
                    radius, limited_by = r_geom_cap, ("prev" if r_prev <= r_next else "next")
                else:
                    radius, limited_by = r_geom_cap, "geometry"

            consumed = radius * tan_half

        remaining[k] -= consumed
        remaining[k + 1] -= consumed

        turns_out.append(
            TurnPlan(index=k, cell=cell, delta_theta=delta_theta, radius=radius, limited_by=limited_by)
        )

    geo_blocks, block_starts = _geometry_blocks(cells, start_heading, turns_out, runs)
    del block_starts  # ここでは使わない（`profile.Segment` への変換だけで足りる）

    total = 0.0
    by_kind = {"straight": 0.0, "arc": 0.0, "spin": 0.0}
    all_segments: List[Segment] = []
    v_max = 0.0
    path_length = 0.0

    for geo_block in geo_blocks:
        if not geo_block:
            continue
        block = to_profile_segments(geo_block)
        it = min_time(block, limits, v_start=0.0, v_end=0.0)
        total += it.total
        for kind, secs in it.by_kind.items():
            by_kind[kind] = by_kind.get(kind, 0.0) + secs
        v_max = max(v_max, it.v_max)
        path_length += it.path_length
        all_segments.extend(block)

    for tp in turns_out:
        if tp.radius <= 0.0:
            st = spin_turn_time(tp.delta_theta, limits)
            total += st.time
            by_kind["spin"] += st.time

    return IdealResult(
        total=total, by_kind=by_kind, n_turns=n_turns, path_cells=len(cells),
        path_length=path_length, v_max=v_max, turns=turns_out, segments=all_segments,
        alloc_iterations=0, allocation_used="greedy",
    )


def _ideal_slalom_proportional(
    cells: Sequence[Cell], v_walls: np.ndarray, h_walls: np.ndarray,
    start_heading: Direction, margin: float,
) -> IdealResult:
    """比例配分: 共有する直線を、隣り合うターンの需要に比例して分け合う緩和法。

    `_allocate_shared_straight` の docstring・モジュール docstring
    「既知の限界」参照。
    """
    limits = vehicle_limits()
    turns, runs = _turns_and_runs(cells, start_heading)
    obstacles = wall_obstacles(v_walls, h_walls)

    n_turns = len(turns)

    # --- 1段目: 各ターンの「直線の制約を無視した」幾何上の最大半径 r_geom[k] -------
    # （180°折返しは tan(|Δθ|/2)=tan(90°) が発散し、有限半径の弧が作れないので
    #   r_geom=0 のまま扱う。それ以外は _fast_max_feasible_radius を r_hi=_R_HI
    #   固定で呼ぶ — 直線の長さでは絞らない。上の「幾何判定の高速化」節末尾参照）。
    delta_thetas: List[float] = []
    is_uturn: List[bool] = []
    tan_half: List[float] = [0.0] * n_turns
    r_geom: List[float] = [0.0] * n_turns
    for k, (move_idx, from_dir, to_dir) in enumerate(turns):
        delta_theta = _turn_delta(from_dir, to_dir)
        delta_thetas.append(delta_theta)
        uturn = abs(abs(delta_theta) - math.pi) < 1e-9
        is_uturn.append(uturn)
        if uturn:
            continue
        tan_half[k] = math.tan(abs(delta_theta) / 2.0)
        cell = cells[move_idx]
        corner_pose = Pose(
            (cell[0] + 0.5) * CELL_SIZE, (cell[1] + 0.5) * CELL_SIZE, _dir_angle(from_dir)
        )
        try:
            r_geom[k] = _fast_max_feasible_radius(
                delta_theta, obstacles, corner_pose, margin=margin, r_lo=_R_LO, r_hi=_R_HI
            )
        except ValueError:
            # r_lo (既定 0.02m) ですら margin を満たす半径が無い
            # → 幾何的に通れる半径が無い、という結果を r_geom=0 として扱う
            # （呼び出し元に例外を投げず、その場旋回への降格として処理する）。
            r_geom[k] = 0.0

    # 希望消費量（直線の制約を無視した場合にこのターンが両側の直線から食いたい長さ。
    # `turn_path` の弧は入出の直線から等しい長さを食う幾何なので、ターン k は
    # runs[k]（手前）と runs[k+1]（次）の**両方**から同じ量を引く1つの値である）。
    c: List[float] = [0.0 if is_uturn[k] else r_geom[k] * tan_half[k] for k in range(n_turns)]
    c0 = list(c)  # 比例配分前の値（"geometry"/"shared" の判定に使う）

    # --- 2段目: 共有する直線ごとに、需要が長さを超えるぶんを比例配分で縮める -------
    # （緩和法の中身は `_allocate_shared_straight` 参照。単体でも
    #   `tests/test_ideal.py::test_allocation_converges` などが直接照合する）。
    c, n_alloc_iterations = _allocate_shared_straight(c0, runs)

    # --- 3段目: 収束した c[k] から半径・limited_by を決める -----------------------
    _EPS = 1e-9
    turns_out: List[TurnPlan] = []
    for k, (move_idx, from_dir, to_dir) in enumerate(turns):
        cell = cells[move_idx]
        if is_uturn[k]:
            # 「geometry」起因の半径0（=どんな半径も幾何的に許されない）として扱う。
            radius, limited_by = 0.0, "geometry"
        else:
            radius_raw = c[k] / tan_half[k]
            if radius_raw < _R_LO - _EPS:
                radius = 0.0
                # r_geom 自体が既に _R_LO 未満なら（比例配分に関わらず）幾何起因、
                # 比例配分で削られて初めて _R_LO を割ったのなら共有直線の取り合い起因。
                limited_by = "geometry" if r_geom[k] < _R_LO - _EPS else "floor"
            else:
                radius = radius_raw
                shrunk = c[k] < c0[k] - max(_EPS, _EPS * abs(c0[k]))
                limited_by = "shared" if shrunk else "geometry"

        turns_out.append(
            TurnPlan(index=k, cell=cell, delta_theta=delta_thetas[k], radius=radius, limited_by=limited_by)
        )

    # --- 区間列の組み立て -----------------------------------------------
    # 半径0（forced_spin）のターンだけ、経路をブロックに分割する。ブロック内は
    # 直線→弧→直線→…→弧→直線の1本道で、これをまとめて1回の min_time に通す
    # （両端 v_start=v_end=0。区間の切れ目では止まらない）。ブロックの境目には
    # spin_turn_time を足す（その場旋回のあいだ完全に停止する）。
    # 幾何ブロック（開始姿勢 + geometry.PathSegment列）の組み立ては
    # `_geometry_blocks` に切り出してある（`tests/test_ideal.py` の衝突検査が、
    # 半径探索をやり直さず、確定済みの `turns_out` から同じ幾何を再現するのに使う）。
    geo_blocks, block_starts = _geometry_blocks(cells, start_heading, turns_out, runs)
    del block_starts  # ここでは使わない（`profile.Segment` への変換だけで足りる）

    total = 0.0
    by_kind = {"straight": 0.0, "arc": 0.0, "spin": 0.0}
    all_segments: List[Segment] = []
    v_max = 0.0
    path_length = 0.0

    for geo_block in geo_blocks:
        if not geo_block:
            continue
        block = to_profile_segments(geo_block)
        it = min_time(block, limits, v_start=0.0, v_end=0.0)
        total += it.total
        for kind, secs in it.by_kind.items():
            by_kind[kind] = by_kind.get(kind, 0.0) + secs
        v_max = max(v_max, it.v_max)
        path_length += it.path_length
        all_segments.extend(block)

    for tp in turns_out:
        if tp.radius <= 0.0:
            st = spin_turn_time(tp.delta_theta, limits)
            total += st.time
            by_kind["spin"] += st.time

    return IdealResult(
        total=total, by_kind=by_kind, n_turns=n_turns, path_cells=len(cells),
        path_length=path_length, v_max=v_max, turns=turns_out, segments=all_segments,
        alloc_iterations=n_alloc_iterations, allocation_used="proportional",
    )


_ALLOCATIONS = ("greedy", "proportional", "best")


def _ideal_slalom(
    cells: Sequence[Cell], v_walls: np.ndarray, h_walls: np.ndarray,
    start_heading: Direction, margin: float, allocation: str = "best",
) -> IdealResult:
    """配分方式のディスパッチャ。`allocation` の意味は上のコメント・
    モジュール docstring「既知の限界」を参照。
    """
    if allocation not in _ALLOCATIONS:
        raise ValueError(f"未知の allocation: {allocation!r}（{_ALLOCATIONS} のいずれか）")

    if allocation == "greedy":
        return _ideal_slalom_greedy(cells, v_walls, h_walls, start_heading, margin)
    if allocation == "proportional":
        return _ideal_slalom_proportional(cells, v_walls, h_walls, start_heading, margin)

    # "best": 両方を計算し、total が小さい方（速い方）を返す。
    # 🔴 これは真の時間最適の保証ではない（どちらの配分も独立最大化の近似で
    # あることに変わりない）が、2つの上界のうち近い方を採るので、片方だけを
    # 使うより真の最小に近い上界になる（コーディネータの裁定。モジュール
    # docstring「既知の限界」参照）。
    greedy_res = _ideal_slalom_greedy(cells, v_walls, h_walls, start_heading, margin)
    proportional_res = _ideal_slalom_proportional(cells, v_walls, h_walls, start_heading, margin)
    return greedy_res if greedy_res.total <= proportional_res.total else proportional_res


# ============================================================================
# 6. 公開 API
# ============================================================================
def ideal_time_for_path(
    cells: Sequence[Cell],
    v_walls: np.ndarray,
    h_walls: np.ndarray,
    start_heading: Direction,
    mode: str = "slalom",
    margin: float = 0.005,
    allocation: str = "best",
) -> IdealResult:
    """経路 `cells`（区画列）を物理限界で走った理想時間を計算する。

    `mode="spin"`: その場旋回走行。ターンでは必ず停止する（`allocation` は使わない）。
    `mode="slalom"`: 曲がるところを円弧で通り、区間の切れ目でも止まらない
    （ただし本モジュール docstring の「半径が0になる場合の扱い」に該当するターンは
    その場だけ停止する）。`allocation` で、共有する直線の配り方を選ぶ:
    `"greedy"`（先取り）・`"proportional"`（比例配分）・`"best"`（既定。両方を
    計算して速い方を採る）。いずれもモジュール docstring「既知の限界」参照。

    `cells` が 1 区画だけ（既にゴールにいる）なら、移動は無く `total=0.0` を返す。
    """
    if mode not in ("spin", "slalom"):
        raise ValueError(f"未知の mode: {mode!r}（'spin' か 'slalom' のいずれか）")
    if allocation not in _ALLOCATIONS:
        raise ValueError(f"未知の allocation: {allocation!r}（{_ALLOCATIONS} のいずれか）")

    if len(cells) <= 1:
        return IdealResult(
            total=0.0, by_kind={"straight": 0.0, "arc": 0.0, "spin": 0.0},
            n_turns=0, path_cells=len(cells), path_length=0.0, v_max=0.0,
            turns=[], segments=[], allocation_used="n/a",
        )

    if mode == "spin":
        return _ideal_spin(cells, start_heading)
    return _ideal_slalom(cells, v_walls, h_walls, start_heading, margin, allocation)
