"""
classic/obs_model.py
================
任意の姿勢で成り立つ距離センサ 4 本（LF・LS・RF・RS）の観測モデル
（`experiments/exp_033_observation_model/PREREG.md`。設計の根拠は
`research_notes/note_037_probabilistic_localization.md` §6・§14-2）。

姿勢 (x, y, θ)（世界座標 [m]・ヨー角 [rad]、`docs/COORDINATE_SYSTEM.md` §7 の規約:
θ=0 が東・反時計回りが正）と壁の配置から、4 本の距離センサが返すはずの距離を予測する。

【区画の中心・軸並行に特殊化しない】
柱を「面（矩形の 4 辺）」として扱い、光線と矩形（AABB＝軸平行境界ボックス）の交差を
標準的なスラブ法で解く。壁も柱と同じ表現（`classic.geometry.Rect`）なので、
壁・柱を区別する特別扱いは要らない。斜めの姿勢では柱の面が直接見える。

【多重反射は解かない】
`mouse/ir_sensor.py`（`note_034` の物理モデル）が扱う反射光の積分とは別物。
本モジュールが予測するのは MuJoCo の `mjSENS_RANGEFINDER`（幾何の光線交差）と
同じ量であり、光の物理は関与しない。

【マイコンに載る設計（note_037 §13）】
  - 光線が当たりうる面は近傍の柱・壁だけに絞る（`_local_obstacles`）。走査する
    格子点・柱間の個数は姿勢によらない固定範囲（`_cells_radius` で決まる正方形の窓）。
    マイコン実装では現在の区画インデックスを起点に配列を直接参照するだけでよく、
    本 Python 実装のように毎回リストへ積む必要はない（§6 予算表に記載）。
  - 姿勢の回転（cosθ・sinθ）は 1 回だけ計算し、4 本のセンサで使い回す。
  - センサごとの光軸の仰角は姿勢に依存しない定数なので、実装では起動時に
    1 回だけ計算しておける（本 Python 実装は分かりやすさのため呼び出しのたびに
    計算しているが、値は姿勢が変わっても変化しない）。
  - 単精度で成り立つ式にする: 交差判定は減算・乗算・除算のみ（`_ray_rect_hit`）。
    平方根は「センサごとの光軸の正規化」でのみ使い、これも姿勢に依存しない
    定数計算に押し込められる。

【センサ site の実効位置（発見・要注意）】
`mouse/params.py` の `sensors` に書かれた `pos` はセンサポッドの**基部**の位置であり、
実際に光線を発する MuJoCo の site はそこから光軸方向へさらに `SENSOR_TIP_OFFSET_M`
（2mm）先にある（`mouse/robot_v2.py:191-195` 「実際のセンサ site（先端 + 0.002）」）。
これを無視すると全センサが系統的に約 2mm 遠くを見ていると予測してしまう
（本実験の準備時に実測で発見。`classic/localization.py` の
`MEASURED_STANDOFF_CORRECTION_M = 0.00194` はこの食い違いを較正値として吸収して
いたものだったと分かった — 同モジュールは `pos` をそのまま使っており、片側だけの
推定にのみ影響する較正だったため気づかれていなかった）。

【壁の配置の受け取り方】
`predict_ranges` の `walls` は次のどちらでもよい:
  - `classic.maze_map.MazeMap`（3 値。`WallState.WALL` のみを「壁あり」として扱う。
    UNKNOWN・OPEN はどちらも「壁なし」として光線を素通しする）
  - `(v_walls, h_walls)` の組（`classic/geometry.py::wall_obstacles` と同じ規約の
    生の真値配列。0 以外を「壁あり」とする）
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from classic.geometry import Rect
from classic.maze_map import MazeMap, WallState
from mouse.params import RobotParams

__all__ = [
    "WALL_THICKNESS_M",
    "POST_SIZE_M",
    "SENSOR_TIP_OFFSET_M",
    "SEARCH_MARGIN_M",
    "Walls",
    "predict_ranges",
    "predict_ranges_with_diagnostics",
    "max_candidates_bound",
]

Walls = Union[MazeMap, Tuple[np.ndarray, np.ndarray]]

# ============================================================================
# 幾何定数（docs/ROBOT_SPEC.md §8.2。mouse/mjcf.py の WALL_THICKNESS/POST_SIZE と
# 同じ値。classic/ パッケージの既存の作法（classic/localization.py 冒頭のコメント
# 参照）に合わせ、mouse.mjcf からは import せず数値をここに明記する）。
# ============================================================================
WALL_THICKNESS_M: float = 0.012
POST_SIZE_M: float = 0.012

# センサ site のオフセット [m]（上記docstring参照。mouse/robot_v2.py:191-195）。
SENSOR_TIP_OFFSET_M: float = 0.002

# 候補探索の半径に足すマージン [m]。cutoff ちょうどで探索窓を切ると、浮動小数点誤差や
# 仰角による水平距離の縮み（1/cos(仰角) 倍のぶん水平距離は cutoff よりわずかに短い）で
# 「cutoff 未満のはずの壁」を検索窓の外に取りこぼしうる。余裕を持たせて確実に含める。
SEARCH_MARGIN_M: float = 0.02


# ============================================================================
# 壁の配置の正規化
# ============================================================================
def _walls_to_bool_arrays(walls: Walls) -> Tuple[np.ndarray, np.ndarray]:
    """`MazeMap` または `(v_walls, h_walls)` の組を、bool 配列の組にする。"""
    if isinstance(walls, MazeMap):
        v = walls.v_walls == int(WallState.WALL)
        h = walls.h_walls == int(WallState.WALL)
        return v, h
    v_walls, h_walls = walls
    return np.asarray(v_walls) != 0, np.asarray(h_walls) != 0


# ============================================================================
# センサ幾何（機体固定座標系）
# ============================================================================
def _parse_vec3(s: str) -> Tuple[float, float, float]:
    parts = [float(v) for v in s.split()]
    if len(parts) != 3:
        raise ValueError(f"3 要素のベクトル文字列ではありません: {s!r}")
    return parts[0], parts[1], parts[2]


def _sensor_local_ray(pos: str, zaxis: str) -> Tuple[float, float, float, float, float]:
    """センサ 1 本の、機体固定座標系での「実効原点(水平成分)」と「水平方向の単位
    ベクトル」、光軸の水平成分の大きさ h（= cos(仰角)）を返す。

    姿勢に依存しない値（センサの取り付けだけで決まる）。マイコン実装では起動時に
    1 回だけ計算して定数として持てる。

    Returns:
        (ox_local, oy_local, dx_local, dy_local, h)
    """
    px, py, _pz = _parse_vec3(pos)
    ax, ay, az = _parse_vec3(zaxis)
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    ux, uy = ax / norm, ay / norm  # uz は水平の交差判定に使わない（下記docstring参照）
    # 実効原点 = ポッド基部(pos) + センサ先端オフセット * 光軸単位ベクトル（水平成分のみ）
    ox_local = px + SENSOR_TIP_OFFSET_M * ux
    oy_local = py + SENSOR_TIP_OFFSET_M * uy
    h = math.sqrt(ux * ux + uy * uy)  # = cos(仰角)。asin/atan2 を経由せず素直に求まる
    dx_local = ux / h
    dy_local = uy / h
    return ox_local, oy_local, dx_local, dy_local, h


def _rotate(vx: float, vy: float, cos_t: float, sin_t: float) -> Tuple[float, float]:
    """機体座標系のベクトル (vx,vy) を世界座標系へ回す（ヨー角 θ の回転）。"""
    return vx * cos_t - vy * sin_t, vx * sin_t + vy * cos_t


# ============================================================================
# 光線と矩形（AABB）の交差 — スラブ法
# ============================================================================
def _ray_rect_hit(ox: float, oy: float, dx: float, dy: float, rect: Rect) -> Optional[float]:
    """原点 (ox,oy)・単位方向 (dx,dy) の光線が、軸平行矩形 `rect` の面のどれかに
    交わる最初の距離（水平距離 [m]）を返す。交わらなければ None。

    柱・壁を「面の集合（矩形の 4 辺）」として扱う標準的なスラブ法（AABB との交差）。
    どの辺に当たるかを明示的に場合分けする必要がなく、乗除算だけで交点までの距離が
    求まる（単精度で成り立つ）。機体は壁・柱に食い込まない前提
    （呼び出し側が `classic/geometry.py` の干渉判定で保証する）ので、原点は常に
    矩形の外側にあり、`t_enter` がそのまま入射距離になる。
    """
    x0, x1 = rect.cx - rect.hx, rect.cx + rect.hx
    y0, y1 = rect.cy - rect.hy, rect.cy + rect.hy

    # 退化した場合（光線が軸に平行）は、原点がその軸方向の帯の内側にあるかどうかを
    # 半開区間 [lo, hi) で判定する（下限は含む・上限は含まない）。
    #
    # 🔴 実測で分かったこと（本実験の準備で発見。数値がそろった姿勢グリッドでのみ
    # 起きる稀な退化ケース。継続調査の結果は正直に書く）:
    #   - ある柱の**上端**にちょうど乗る姿勢: MuJoCo は「ヒット無し」
    #     （境界から ±1e-12 ずらすと直ちに「ヒットあり」に切り替わる不連続）
    #   - 別の柱の**下端**にちょうど乗る姿勢: MuJoCo は「ヒットあり」
    #   半開区間 [lo,hi) はこの 2 例とは一致する。ただし** MuJoCo 自身の境界の
    #   扱いは完全に一貫してはいない**ことも実測で確認した（同じ y 座標が、
    #   ある壁の上端では「ヒットあり」、その真上に隣接する柱の下端では
    #   「ヒット無し」になる例が見つかった。`mju_rayGeom` で単体検査すると
    #   幾何的には両方ヒットするはずの計算になるのに、シーン全体の `mj_ray`
    #   では片方しか拾わない — 内部のブロードフェーズ由来と見られる）。
    #   つまりこの退化ケースには**単純な規約では 100% 一致しない**。
    #   半開区間は実測した数例のうち多数に一致した実務的な選択であり、
    #   厳密解ではない。実姿勢（丸い数値に揃わない連続値）ではこの退化は
    #   測度ゼロで、q_95（主判定量）にも影響しない（exp_033 報告参照）。
    if dx != 0.0:
        tx0, tx1 = (x0 - ox) / dx, (x1 - ox) / dx
        if tx0 > tx1:
            tx0, tx1 = tx1, tx0
    else:
        if ox < x0 or ox >= x1:
            return None
        tx0, tx1 = -math.inf, math.inf

    if dy != 0.0:
        ty0, ty1 = (y0 - oy) / dy, (y1 - oy) / dy
        if ty0 > ty1:
            ty0, ty1 = ty1, ty0
    else:
        if oy < y0 or oy >= y1:
            return None
        ty0, ty1 = -math.inf, math.inf

    t_enter = max(tx0, ty0)
    t_exit = min(tx1, ty1)
    if t_enter > t_exit or t_exit < 0.0 or t_enter < 0.0:
        return None
    return t_enter


# ============================================================================
# 近傍の候補（柱・壁）を絞り込む
# ============================================================================
def _cells_radius(cutoff: float, cell_size: float) -> int:
    """探索窓の半径 [格子点単位]（`_local_obstacles` が走査する範囲を決める）。

    探索半径 `cutoff + SEARCH_MARGIN_M` [m] を、原点がどの位置にあっても格子点の
    窓で覆いきるための値。原点は自分が属する格子(区画)内のどこにでもありうる
    （最大で 1 区画ぶん `cell_size` だけ格子点からずれる）ので、
    ceil((cutoff+margin)/cell_size) にさらに 1 を足す。
    """
    return int(math.ceil((cutoff + SEARCH_MARGIN_M) / cell_size)) + 1


def max_candidates_bound(params: Optional[RobotParams] = None) -> int:
    """`_local_obstacles` が返す候補数の理論上限（柱・縦壁・横壁の合計）。

    格子点の窓は 1 辺 `2*cells_radius+1` の正方形なので、柱・縦壁・横壁それぞれ
    高々 `(2*cells_radius+1)^2` 個（迷路の端では配列境界でさらに減る）。
    """
    params = params or RobotParams()
    r = _cells_radius(params.sensor_cutoff, params.cell_size)
    side = 2 * r + 1
    return 3 * side * side


def _local_obstacles(
    v_walls: np.ndarray,
    h_walls: np.ndarray,
    ox: float,
    oy: float,
    cell_size: float,
    cells_radius: int,
    center_goal: bool,
    include_posts: bool = True,
) -> List[Rect]:
    """センサ原点 (ox,oy) の近傍にある柱・壁だけを `Rect` にする。

    走査は原点が属する格子点 `(i_center, j_center) = floor((ox,oy)/cell_size)` を
    中心に、縦横 ±`cells_radius` の固定範囲に限る（`_cells_radius` の docstring
    参照）。マイコン実装ではこの範囲を配列の直接参照で辿るだけでよく、本 Python
    実装のようにリストへ積む必要はない（§6 予算表に記載）。柱・縦壁・横壁それぞれ
    高々 `(2*cells_radius+1)^2` 個（`max_candidates_bound` が理論上限を返す）。

    `mouse/mjcf.py::build_maze_robot_xml` / `classic/geometry.py::wall_obstacles`
    と同じ Rect の作り方（柱: 半長 `post/2`。縦壁: 半長 `(wall/2, cell/2-post/2)`。
    横壁: 半長 `(cell/2-post/2, wall/2)`）。
    """
    width = v_walls.shape[0] - 1
    height = v_walls.shape[1]

    i_center = int(math.floor(ox / cell_size))
    j_center = int(math.floor(oy / cell_size))
    i_lo, i_hi = i_center - cells_radius, i_center + cells_radius
    j_lo, j_hi = j_center - cells_radius, j_center + cells_radius

    half_post = POST_SIZE_M / 2.0
    half_wall = WALL_THICKNESS_M / 2.0
    half_run = cell_size / 2.0 - half_post

    obstacles: List[Rect] = []

    # 柱: 格子点 i in [0,width], j in [0,height]
    # include_posts=False は否定対照 N3（PREREG §4「柱を無視して壁だけで予測する」）専用。
    if include_posts:
        for i in range(max(i_lo, 0), min(i_hi, width) + 1):
            for j in range(max(j_lo, 0), min(j_hi, height) + 1):
                if center_goal and i == width // 2 and j == height // 2:
                    continue
                obstacles.append(Rect(float(i) * cell_size, float(j) * cell_size, half_post, half_post))

    # 縦壁: i in [0,width]（格子線）, j in [0,height-1]（区画の行）
    for i in range(max(i_lo, 0), min(i_hi, width) + 1):
        for j in range(max(j_lo, 0), min(j_hi, height - 1) + 1):
            if v_walls[i, j]:
                px = float(i) * cell_size
                py = float(j) * cell_size + cell_size / 2.0
                obstacles.append(Rect(px, py, half_wall, half_run))

    # 横壁: i in [0,width-1]（区画の列）, j in [0,height]（格子線）
    for i in range(max(i_lo, 0), min(i_hi, width - 1) + 1):
        for j in range(max(j_lo, 0), min(j_hi, height) + 1):
            if h_walls[i, j]:
                px = float(i) * cell_size + cell_size / 2.0
                py = float(j) * cell_size
                obstacles.append(Rect(px, py, half_run, half_wall))

    return obstacles


# ============================================================================
# 公開 API
# ============================================================================
def predict_ranges_with_diagnostics(
    pose: Tuple[float, float, float],
    walls: Walls,
    params: Optional[RobotParams] = None,
    center_goal: bool = True,
    include_posts: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """`predict_ranges` に加え、センサごとに実際にスラブ法を適用した候補数を返す
    （テスト・マイコン予算の実測に使う診断つき版）。

    `include_posts=False` は否定対照 N3（PREREG §4）専用。柱を丸ごと無視して壁だけで
    予測する、壊れた版を作るためのフラグであり、既定 True（通常の予測）では使わない。

    Returns:
        (ranges, candidate_counts): ranges は (4,) [m]、candidate_counts は (4,) int
        （`params.sensors` と同じ順）。
    """
    params = params or RobotParams()
    x, y, theta = pose
    v_walls, h_walls = _walls_to_bool_arrays(walls)

    cutoff = params.sensor_cutoff
    cell_size = params.cell_size
    radius = cutoff + SEARCH_MARGIN_M
    cells_radius = _cells_radius(cutoff, cell_size)

    cos_t, sin_t = math.cos(theta), math.sin(theta)  # 姿勢の回転は 1 回だけ計算し使い回す

    n = len(params.sensors)
    ranges = np.empty(n, dtype=np.float64)
    counts = np.empty(n, dtype=np.int64)

    for idx, s in enumerate(params.sensors):
        ox_local, oy_local, dx_local, dy_local, h = _sensor_local_ray(s['pos'], s['zaxis'])
        ox_world, oy_world = _rotate(ox_local, oy_local, cos_t, sin_t)
        ox_world += x
        oy_world += y
        dx_world, dy_world = _rotate(dx_local, dy_local, cos_t, sin_t)

        obstacles = _local_obstacles(
            v_walls, h_walls, ox_world, oy_world, cell_size, cells_radius, center_goal,
            include_posts=include_posts,
        )

        best_t: Optional[float] = None
        n_checked = 0
        for rect in obstacles:
            # 探索半径の正方形と矩形の AABB が重ならないものは、スラブ法を適用する
            # までもなく除外できる（近傍だけに絞る、の実体）。
            if abs(rect.cx - ox_world) > radius + rect.hx:
                continue
            if abs(rect.cy - oy_world) > radius + rect.hy:
                continue
            n_checked += 1
            t = _ray_rect_hit(ox_world, oy_world, dx_world, dy_world, rect)
            if t is not None and (best_t is None or t < best_t):
                best_t = t

        counts[idx] = n_checked
        if best_t is None:
            ranges[idx] = cutoff  # 近傍に何も交差物体が無い = 飽和
        else:
            d3 = best_t / h  # 光線に沿った距離は水平距離より 1/cos(仰角) 倍長い
            ranges[idx] = d3 if d3 < cutoff else cutoff

    return ranges, counts


def predict_ranges(
    pose: Tuple[float, float, float],
    walls: Walls,
    params: Optional[RobotParams] = None,
    center_goal: bool = True,
    include_posts: bool = True,
) -> np.ndarray:
    """姿勢 `(x, y, θ)` と壁の配置から、距離センサ 4 本が返すはずの距離を予測する。

    Args:
        pose: (x[m], y[m], theta[rad])。世界座標、`docs/COORDINATE_SYSTEM.md` §7
              の規約（θ=0 が東・反時計回りが正）。任意の姿勢で成り立つ
              （区画の中心・軸並行に特殊化しない）。
        walls: `MazeMap` または `(v_walls, h_walls)`（`classic/geometry.py::
               wall_obstacles` と同じ規約の生の真値配列）。
        params: `mouse.params.RobotParams`（None なら既定値。センサの取り付け位置・
                光軸・cutoff・区画寸法をここから取る）。
        center_goal: True のとき中央 2x2 ゴールの中心柱を生成しない
                     （`mouse/mjcf.py::build_maze_robot_xml` と同じ既定 True）。
        include_posts: False で柱を丸ごと無視する（否定対照 N3 専用。通常は True のまま）。

    Returns:
        (4,) ndarray [m]。並びは `params.sensors` と同じ（既定 LF, LS, RF, RS。
        `mouse.sim.MouseSim.observation()` の ranges 部分と同じ並び）。
    """
    ranges, _counts = predict_ranges_with_diagnostics(
        pose, walls, params, center_goal, include_posts
    )
    return ranges
