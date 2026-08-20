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

- **半径は各ターンについて独立に最大化する貪欲法であり、厳密な時間最適ではない**
  （前後の速度の連成を考えていない。ある区間の速度が本当に効率よく使えるかは
  隣接区間の速度計画にも依存するが、本モジュールは考慮しない）。
  したがって `T_ideal` は真の最小より**大きい側**に出る。
- 「手前/次の直線から食える長さ」は、ターンを経路の先頭から順番に処理しながら
  **既に前のターンが消費した分を引いた残り長**を使う（隣り合う 2 つのターンが
  同じ直線の両端から独立に満額を要求して食い合う — 直線の長さより多く消費してしまう
  — という不整合を避けるための処理順）。この意味でも「独立に最大化」は
  「経路の先頭から見て貪欲」という意味であり、真に対称な配分ではない。
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
def _dir_angle(d: Direction) -> float:
    """方位 `Direction` を、`geometry.Pose.theta` と同じ座標系の角度 [rad] にする。

    `classic.maze_map._DIR_DELTA` は N=(0,1), E=(1,0), S=(0,-1), W=(-1,0)
    （x=東、y=北）なので、E=0, N=+90°, W=180°, S=-90°（数学の通常の角度）になる。
    """
    return math.pi / 2.0 - int(d) * (math.pi / 2.0)


def _turn_delta(from_dir: Direction, to_dir: Direction) -> float:
    """`from_dir` から `to_dir` への旋回角 [rad]（正=左/反時計回り。`geometry.arc` と同じ符号）。

    `classic.route._turn_type` と同じ mod 4 の判定を角度に翻訳したもの
    （rel=1→右90°、rel=2→180°、rel=3→左90°。N,E,S,W がこの順で時計回りに
    並んでいるため、rel の符号を反転すると反時計回り正の数学角になる）。
    """
    rel = (int(to_dir) - int(from_dir)) % 4
    signed_rel = rel if rel <= 2 else rel - 4
    return -signed_rel * (math.pi / 2.0)


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
    limited_by: str       # "geometry"/"prev"/"next"（何が半径の上限を決めたか）


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
        # "n/a" にする（geometry/prev/next のどれでもない、と明示する札）。
        turns_out.append(
            TurnPlan(index=k, cell=cells[move_idx], delta_theta=delta_theta, radius=0.0, limited_by="n/a")
        )

    return IdealResult(
        total=total, by_kind=by_kind, n_turns=len(turns), path_cells=len(cells),
        path_length=path_length, v_max=v_max, turns=turns_out, segments=segments,
    )


# ============================================================================
# 4.5. 幾何判定の高速化（本モジュールだけの最適化。`classic/geometry.py` は変更しない）
# ============================================================================
# `max_feasible_radius` は二分探索の 40 回それぞれで、渡された障害物**全部**に対して
# 掃引全姿勢の厳密な干渉判定をする。16x16 迷路の障害物（500件超）をそのまま渡すと
# 1ターンあたり数秒かかり、経路1本（ターン数十個）×10迷路では現実的な時間に収まらない
# （実測: 全障害物・未整列で1回あたり約4.5秒。ターン数十×迷路10で計算すると
# 数十分かかってしまう）。そこで判定の中身を変えずに次の2つだけ行う:
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
#   3. 探索上限の絞り込み: 半径は「手前/次の直線から食える長さ」（`prev`/`next`）を
#      超えられない（超えたら円弧が直線からはみ出す）。なので `max_feasible_radius`
#      に渡す `r_hi` を `min(_R_HI, r_prev, r_next)` に絞ってよい — それを超える
#      半径が幾何的に通るかどうかは、どのみち採用できないので知る必要が無い
#      （半径が大きいほど掃引区間が長くなり二分探索の1ステップが重くなるので、
#      これは速度にも効く）。ただし探索上限ちょうどで「通る」と判定された場合、
#      幾何がそこで初めて効いたのか、prev/next が先に効いていて幾何はもっと
#      余裕があったのかは、この絞った探索だけでは区別できない。**その場合は
#      prev/next 起因として扱う**（幾何が cap 未満まで絞り込んだことを示せた
#      場合だけ "geometry" とする）。これは `limited_by` の帰属だけに影響し、
#      採用する半径の値そのものは絞り込みの有無に関わらず常に一致する
#      （down 側の値は変えていない — 上限を「どうせ採用できない範囲」に
#      限っただけなので、真の最小候補は必ず探索範囲に含まれる）。
#   4. 二分探索を自前で回し、試す半径 r ごとに絞り込みをかけ直す:
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
# 5. mode="slalom"（円弧で曲がる。止まらない）
# ============================================================================
def _ideal_slalom(
    cells: Sequence[Cell], v_walls: np.ndarray, h_walls: np.ndarray,
    start_heading: Direction, margin: float,
) -> IdealResult:
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
            cap = min(_R_HI, r_prev, r_next)  # 採用できない範囲は探索しない（上のコメント参照）

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
    )


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
) -> IdealResult:
    """経路 `cells`（区画列）を物理限界で走った理想時間を計算する。

    `mode="spin"`: その場旋回走行。ターンでは必ず停止する。
    `mode="slalom"`: 曲がるところを円弧で通り、区間の切れ目でも止まらない
    （ただし本モジュール docstring の「半径が0になる場合の扱い」に該当するターンは
    その場だけ停止する）。

    `cells` が 1 区画だけ（既にゴールにいる）なら、移動は無く `total=0.0` を返す。
    """
    if mode not in ("spin", "slalom"):
        raise ValueError(f"未知の mode: {mode!r}（'spin' か 'slalom' のいずれか）")

    if len(cells) <= 1:
        return IdealResult(
            total=0.0, by_kind={"straight": 0.0, "arc": 0.0, "spin": 0.0},
            n_turns=0, path_cells=len(cells), path_length=0.0, v_max=0.0,
            turns=[], segments=[],
        )

    if mode == "spin":
        return _ideal_spin(cells, start_heading)
    return _ideal_slalom(cells, v_walls, h_walls, start_heading, margin)
