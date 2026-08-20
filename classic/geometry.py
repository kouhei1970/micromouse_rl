"""classic/geometry.py — 幾何と干渉判定の層

`classic/profile.py`（速度計画器）は「半径 R の弧を機械限界で走ったら何秒か」を計算できるが、
**その半径で本当に通れるのか**は関知しない。半径を掃引して最適値を探すだけの実装は、
幾何的に通れない半径を「最適」と出しうる（実測: 中心線上の 90° ターンで R=300mm を最適と
出したが、手で確認すると中心線上の実際の上限は約 190mm だった。`README` 相当は無いので
`research_notes/note_031_profile_planner_and_eta.md` を参照）。

本モジュールの主目的は、**機体の外形を経路に沿って掃引し、迷路の真の壁・柱との干渉を
厳密に判定すること**である。サンプリング近似ではなく、機体（矩形）と障害物（軸平行矩形）
という限定された形状を利用し、分離軸定理（SAT）で厳密な符号付き距離を計算する。

作法は `experiments/exp_025_s4_slalom/geometry_anchor.py`（四分円の余裕を独立に計算した
祖先スクリプト）を踏襲する。本モジュールが出す `clearance`/`sweep_clearance` の値は、
同スクリプトの手計算とテストで照合する（`tests/test_geometry.py`）。

作った区間列（`PathSegment`）は `to_profile_segments()` で `classic.profile.Segment` に
変換して `classic.profile.min_time` へそのまま渡せる。

`classic/` パッケージの規約どおり、本モジュールは numpy 以外に依存せず、MuJoCo や
`mouse/` のシミュレータを import しない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, NamedTuple, Sequence, Tuple

from classic.profile import Segment

__all__ = [
    "HALF_WIDTH",
    "HALF_LENGTH",
    "Rect",
    "wall_obstacles",
    "Pose",
    "robot_corners",
    "clearance",
    "sweep_clearance",
    "PathSegment",
    "line",
    "arc",
    "poses_along",
    "to_profile_segments",
    "turn_path",
    "max_feasible_radius",
]


# ============================================================================
# 0. 機体の外形（docs/ROBOT_SPEC.md §2.1）
# ============================================================================
# 機体の最外側半幅（±y）。docs/ROBOT_SPEC.md §2.1「機体の最外側半幅」の確定値（コーナー片
# mein_body2〜5 が最外。車輪外面 0.0395 でもシャーシ半幅 0.030 でもない）。
# mouse/maze6_env.py:155 の _GEO_CLEARANCE = WALL_THICKNESS/2 + 0.040 と同じ 0.040 を使っており、
# 判定側（maze6_env）と本モジュール（幾何の掃引側）とで機体半幅の値がずれていないことを
# ここで保証する（二重管理防止のコメント。値そのものは docs/ROBOT_SPEC.md が正）。
HALF_WIDTH = 0.040    # [m]

# 機体の長手半長（±x）。mouse/params.py の chassis_length=0.100（シャーシ外形の全長）の半分。
HALF_LENGTH = 0.050   # [m]


# ============================================================================
# 1. 迷路の自由空間（真の壁から作る）
# ============================================================================
class Rect(NamedTuple):
    """軸平行の長方形（障害物 1 個ぶん）。中心と半長。"""

    cx: float
    cy: float
    hx: float
    hy: float


def wall_obstacles(
    v_walls,
    h_walls,
    cell_size: float = 0.180,
    wall_thickness: float = 0.012,
    post_size: float = 0.012,
) -> List[Rect]:
    """真の壁配列（`v_walls`/`h_walls`）から、壁と柱を軸平行の長方形の列にする。

    `mouse/mjcf.py::build_maze_robot_xml` の建て方と添字規約に合わせてある
    （同ファイルを参照して確認済み）:

    - `v_walls[x, y]`: 区画 (x-1,y)-(x,y) 間の縦壁。形状 (width+1, height)。
      壁が存在するとき、中心 `(x*cell_size, y*cell_size + cell_size/2)`、
      半長 `(wall_thickness/2, cell_size/2 - post_size/2)`。
    - `h_walls[x, y]`: 区画 (x,y-1)-(x,y) 間の横壁。形状 (width, height+1)。
      壁が存在するとき、中心 `(x*cell_size + cell_size/2, y*cell_size)`、
      半長 `(cell_size/2 - post_size/2, wall_thickness/2)`。
    - **柱は格子点すべてに立つ**（壁の有無に関わらず）。中心 `(x*cell_size, y*cell_size)`、
      半長 `(post_size/2, post_size/2)`。`x in [0, width]`, `y in [0, height]`。

    mjcf 側にある「赤い壁上端」geom は同じ xy 足跡（高さだけが違う）なので、
    2D の干渉判定には寄与せず、ここでは 1 壁につき 1 個の `Rect` で足りる。

    mjcf 側の `center_goal`（16x16 のゴール中央柱を落とす特例）はここでは再現しない
    （本モジュールは迷路サイズ非依存の一般的な建て方だけを担う）。
    """
    width = v_walls.shape[0] - 1
    height = v_walls.shape[1]
    expected_h_shape = (width, height + 1)
    if h_walls.shape != expected_h_shape:
        raise ValueError(
            f"h_walls の shape が v_walls と整合しません: h_walls.shape={h_walls.shape}, "
            f"期待値={expected_h_shape} (v_walls.shape={v_walls.shape} から算出)"
        )

    obstacles: List[Rect] = []
    half_post = post_size / 2.0

    # 柱：格子点すべて（壁の有無に関わらず）。x 外側・y 内側ループは mjcf.py と同じ順序。
    for x in range(width + 1):
        for y in range(height + 1):
            obstacles.append(Rect(x * cell_size, y * cell_size, half_post, half_post))

    # 縦壁（vertical walls）。mjcf.py と同じ x 外側・y 内側ループ。
    half_wall = wall_thickness / 2.0
    half_run = cell_size / 2.0 - half_post
    for x in range(width + 1):
        for y in range(height):
            if v_walls[x, y]:
                px = x * cell_size
                py = y * cell_size + cell_size / 2.0
                obstacles.append(Rect(px, py, half_wall, half_run))

    # 横壁（horizontal walls）。
    for x in range(width):
        for y in range(height + 1):
            if h_walls[x, y]:
                px = x * cell_size + cell_size / 2.0
                py = y * cell_size
                obstacles.append(Rect(px, py, half_run, half_wall))

    return obstacles


# ============================================================================
# 2. 機体の掃引と干渉判定
# ============================================================================
@dataclass(frozen=True)
class Pose:
    """機体重心（車軸中心）の平面姿勢。"""

    x: float
    y: float
    theta: float  # [rad]


def robot_corners(
    pose: Pose, half_width: float = HALF_WIDTH, half_length: float = HALF_LENGTH
) -> List[Tuple[float, float]]:
    """機体の 4 隅の世界座標（反時計回り: 右前・左前・左後・右後の順ではなく、
    +x+y → +x-y → -x-y → -x+y の反時計回り）。"""
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    local = (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    )
    return [(pose.x + lx * c - ly * s, pose.y + lx * s + ly * c) for lx, ly in local]


def _rect_polygon(rect: Rect) -> List[Tuple[float, float]]:
    """`Rect` を反時計回りの 4 頂点にする。"""
    x0, x1 = rect.cx - rect.hx, rect.cx + rect.hx
    y0, y1 = rect.cy - rect.hy, rect.cy + rect.hy
    return [(x1, y1), (x0, y1), (x0, y0), (x1, y0)]


def _project(poly: Sequence[Tuple[float, float]], axis: Tuple[float, float]) -> Tuple[float, float]:
    dots = [px * axis[0] + py * axis[1] for px, py in poly]
    return min(dots), max(dots)


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """点 `(px,py)` と線分 `(ax,ay)-(bx,by)` の最短距離。"""
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom < 1e-15:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def _polygon_distance_disjoint(
    poly_a: Sequence[Tuple[float, float]], poly_b: Sequence[Tuple[float, float]]
) -> float:
    """互いに重ならない 2 つの凸多角形の最短距離（厳密）。

    互いに重ならない凸多角形どうしの最短距離は、必ず一方の頂点と他方の辺（またはその
    延長ではなく線分そのもの）の間で実現される（頂点-頂点も、線分の端点として含まれる）。
    総当たり（頂点 × 相手の辺）で厳密に求まる（サンプリング近似ではない）。
    """
    best = math.inf
    na, nb = len(poly_a), len(poly_b)
    for i in range(nb):
        b1, b2 = poly_b[i], poly_b[(i + 1) % nb]
        for px, py in poly_a:
            d = _point_segment_distance(px, py, b1[0], b1[1], b2[0], b2[1])
            if d < best:
                best = d
    for i in range(na):
        a1, a2 = poly_a[i], poly_a[(i + 1) % na]
        for px, py in poly_b:
            d = _point_segment_distance(px, py, a1[0], a1[1], a2[0], a2[1])
            if d < best:
                best = d
    return best


def _rect_clearance(
    robot_poly: Sequence[Tuple[float, float]],
    robot_axes: Sequence[Tuple[float, float]],
    obs_poly: Sequence[Tuple[float, float]],
) -> float:
    """機体（任意姿勢の矩形）と 1 個の障害物（軸平行矩形）の符号付き距離。

    分離軸定理（SAT）: 2 つの凸多角形が交わらないための必要十分条件は、両者の辺の法線
    のどれかを軸に取ったとき、2 つの図形の射影区間が重ならないことである。矩形は辺の
    法線が実質 2 方向（対辺どうしは平行）しかないので、機体側の 2 軸＋障害物側の 2 軸
    （軸平行なので (1,0),(0,1)）の計 4 軸で判定が尽くせる。

    - どれか 1 軸で射影区間が重ならなければ分離（干渉なし）。この場合、真の最短距離は
      SAT の軸だけでは出ない（対角に離れているとき射影の隙間は下界にしかならない）ので、
      `_polygon_distance_disjoint`（頂点×辺の厳密な最短距離）で正確な値を出す。
    - 全軸で重なれば干渉。SAT の定理により、重なり量が最小になる軸が最小移動距離
      （めり込みの深さ）を厳密に与える。符号を負にして返す。
    """
    axes = (robot_axes[0], robot_axes[1], (1.0, 0.0), (0.0, 1.0))
    overlap_min = math.inf
    for axis in axes:
        a_min, a_max = _project(robot_poly, axis)
        b_min, b_max = _project(obs_poly, axis)
        gap = max(b_min - a_max, a_min - b_max)
        if gap > 0.0:
            return _polygon_distance_disjoint(robot_poly, obs_poly)
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap < overlap_min:
            overlap_min = overlap
    return -overlap_min


_ROBOT_CIRCUMRADIUS = math.hypot(HALF_LENGTH, HALF_WIDTH)


def clearance(pose: Pose, obstacles: Sequence[Rect]) -> float:
    """機体の外形（既定寸法の矩形）と障害物群の最短距離 [m]（負なら干渉）。

    最も近い障害物 1 個との符号付き距離を返す（`obstacles` 全体に対する最小値）。
    """
    if not obstacles:
        return math.inf
    robot_poly = robot_corners(pose)
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    robot_axes = ((c, s), (-s, c))

    best = math.inf
    for obs in obstacles:
        obs_r = math.hypot(obs.hx, obs.hy)
        center_dist = math.hypot(pose.x - obs.cx, pose.y - obs.cy)
        # 枝刈り: 中心間距離から両者の外接円半径を引いた下界が現在の最小値を超えられない
        # 障害物は、厳密な SAT 計算をするまでもなく除外できる（結果は変えない安全な高速化）。
        if center_dist - _ROBOT_CIRCUMRADIUS - obs_r > best:
            continue
        obs_poly = _rect_polygon(obs)
        d = _rect_clearance(robot_poly, robot_axes, obs_poly)
        if d < best:
            best = d
    return best


def sweep_clearance(poses: Sequence[Pose], obstacles: Sequence[Rect]) -> Tuple[float, Pose]:
    """姿勢列 `poses` に沿って掃引したときの最小余裕と、そのときの姿勢。"""
    if not poses:
        raise ValueError("poses は空にできない")
    best = math.inf
    best_pose = poses[0]
    for p in poses:
        d = clearance(p, obstacles)
        if d < best:
            best = d
            best_pose = p
    return best, best_pose


# ============================================================================
# 3. 経路（姿勢の列）と区間列
# ============================================================================
@dataclass(frozen=True)
class PathSegment:
    """経路の 1 区間（直線 or 円弧）。`classic.profile.Segment` へそのまま渡せる形。"""

    kind: str        # "straight" / "arc"（`classic.profile.Segment` と同じ札の約束）
    length: float     # 経路長 [m]
    curvature: float  # 曲率 1/R [1/m]（符号付き。正=左旋回/反時計回り）。直線は 0.0


def line(p0: Pose, length: float) -> PathSegment:
    """直線区間を作る。`p0` はこの型（`length`/`curvature` のみを持つ相対表現）には
    現れないが、`arc()` との呼び出し形を揃え、将来経路の絶対配置を扱う拡張の受け口として
    残してある（`poses_along` が実際の絶対姿勢への展開を担う）。"""
    del p0  # 相対表現なので使わない（呼び出し形を arc() と揃えるためのプレースホルダ）
    return PathSegment(kind="straight", length=length, curvature=0.0)


def arc(p0: Pose, delta_theta: float, radius: float) -> PathSegment:
    """円弧区間を作る。`delta_theta` の符号が旋回方向（正=左/反時計回り）を決める。"""
    del p0  # line() と同じ理由で未使用
    length = radius * abs(delta_theta)
    curvature = 0.0 if radius == 0.0 else math.copysign(1.0 / radius, delta_theta)
    return PathSegment(kind="arc", length=length, curvature=curvature)


def _pose_at(start: Pose, curvature: float, s: float) -> Pose:
    """`start` から曲率 `curvature`（符号付き、0 なら直線）で弧長 `s` だけ進んだ姿勢。

    定曲率運動の閉じた式（クロソイドではなく単一の円弧/直線なので厳密解が存在する）:
        theta(s) = theta0 + curvature*s
        x(s) = x0 + (sin(theta(s)) - sin(theta0)) / curvature
        y(s) = y0 - (cos(theta(s)) - cos(theta0)) / curvature
    """
    if curvature == 0.0:
        return Pose(
            start.x + s * math.cos(start.theta),
            start.y + s * math.sin(start.theta),
            start.theta,
        )
    theta = start.theta + curvature * s
    x = start.x + (math.sin(theta) - math.sin(start.theta)) / curvature
    y = start.y - (math.cos(theta) - math.cos(start.theta)) / curvature
    return Pose(x, y, theta)


def poses_along(
    segments: Sequence[PathSegment], start_pose: Pose, ds: float = 0.002
) -> List[Pose]:
    """区間列を `start_pose` から実際に辿り、掃引用の姿勢列にする（`ds` 間隔）。

    区間ごとに独立に等分割する（区間境界の姿勢を必ず含む。区間長が `ds` に等分できない
    半端は各区間内で均等に配る）。
    """
    if not segments:
        return [start_pose]
    poses = [start_pose]
    pose = start_pose
    for seg in segments:
        if seg.length <= 0.0:
            continue
        n = max(int(math.ceil(seg.length / ds)), 1)
        step = seg.length / n
        for i in range(1, n + 1):
            poses.append(_pose_at(pose, seg.curvature, i * step))
        pose = poses[-1]
    return poses


def to_profile_segments(segments: Sequence[PathSegment]) -> List[Segment]:
    """`PathSegment` 列を `classic.profile.Segment` 列に変換する（`min_time` へ渡す用）。

    `kind` は "straight"/"arc" の札をそのまま流用している（`classic.profile.Segment` の
    既定の約束と同一なので変換は不要）。
    """
    return [Segment(length=s.length, curvature=s.curvature, kind=s.kind) for s in segments]


# ============================================================================
# 4. ターンの幾何と、通れる最大半径
# ============================================================================
# 直線が円弧との接続点の先まで掃引に含まれるようにする余裕（掃引の網羅性のためだけの
# 定数。大きすぎても直線部の余裕は障害物から離れているため最小値には影響しない）。
_LEAD_MARGIN = 0.02  # [m]


def turn_path(
    delta_theta: float, radius: float, entry_offset: float = 0.0
) -> Tuple[List[PathSegment], float]:
    """半径 `radius`・旋回角 `delta_theta` のターンを「直線→円弧→直線」にする。

    戻り値の第 2 要素は円弧が入出の直線から食う長さ
    `radius * tan(|delta_theta|/2)`（`classic.profile.tangent_consumption` と同じ式）。
    直線区間の長さはこの消費長に `_LEAD_MARGIN` を足したもの（掃引が接続点の手前の
    直線部分も十分な長さ含むようにするための実務的な余裕であり、この余裕の大小が
    最小余裕の値を左右することはない — 障害物から離れた直線の途中は最小値の候補にならない）。

    🔴 `entry_offset`（直線を通路の中心線から横へずらす量。`|offset| <= W_C = 0.044`）は
    ここでは**セグメント自体を変えない**。理由: 2 本の直線（進入・退出）を同じ量だけ
    平行移動しても、それに内接する半径 `radius` の円弧の弧長・曲率（円の一部という
    内在的な性質）は変わらない — 変わるのは円弧の中心・接点の**絶対位置**だけである。
    したがって offset の効果は、`poses_along` に渡す `start_pose` を中心線から
    `entry_offset` だけ横にずらして呼び出す側（`max_feasible_radius` の呼び出し元）が
    担う。この引数は呼び出し形を将来の拡張（呼び出し元に応じた消費長の補正など）に
    開けておくために残してあるだけで、現状は妥当性チェック以外の計算には使わない。
    """
    if abs(entry_offset) > 0.044 + 1e-9:
        raise ValueError(
            f"entry_offset={entry_offset} が通路の片側の余地 W_C=0.044 を超えている"
        )
    consumed = radius * math.tan(abs(delta_theta) / 2.0)
    lead = consumed + _LEAD_MARGIN
    dummy = Pose(0.0, 0.0, 0.0)
    segments = [line(dummy, lead), arc(dummy, delta_theta, radius), line(dummy, lead)]
    return segments, consumed


def max_feasible_radius(
    delta_theta: float,
    obstacles: Sequence[Rect],
    start_pose: Pose,
    margin: float = 0.005,
    r_lo: float = 0.02,
    r_hi: float = 0.40,
) -> float:
    """「掃引の最小余裕 >= margin」を満たす最大の半径を二分探索で返す。

    `start_pose` は**直線区間の始点ではなく、ターンの幾何学的な角**（進入側の直線と
    退出側の直線を、それぞれ延長したときに交わる点。`theta` は進入側の向き）を表す。
    これは半径によらない値（通路そのものの位置で決まる物理的な定数）である。

    🔴 なぜ「直線区間の始点」を直接渡す形にしなかったか: `turn_path` の直線区間の長さは
    `radius*tan(|delta_theta|/2) + 余裕`（半径に依存する）。もし `start_pose` を直線の
    始点として固定すると、半径を変えるたびに円弧の始点（＝直線の終点）が動き、
    退出側の直線が本来の通路中心線からずれてしまう（半径ごとに掃引しているターンの
    実体が変わってしまう）。角の位置を固定し、そこから半径ごとの消費長ぶんだけ
    逆算して直線の始点を求めることで、**どの半径を試しても同じ物理的な角を掃引する**
    ことを保証する。

    `entry_offset` を使う場合は、呼び出し元が `start_pose` の位置（角を通路の中心線から
    横へずらした位置）で表現する（`turn_path` 自体はオフセットでセグメントを変えない。
    同関数のコメント参照）。

    半径が大きいほど「通りやすくなる」わけではない（弧が壁・柱に近づいて逆に厳しくなる）
    ことが実測で分かっているため、`feasible(r)` は R について単調とは限らない。それでも
    実務上の使い方（`r_lo` 側は十分小さく必ず feasible、大きい R で infeasible になる）を
    前提に、二分探索で「`r_lo` から見て feasible が続く上限」を返す
    （厳密な単調性の証明はしていないが、`geometry_anchor.py` 系の手計算・実測いずれも
    半径を大きくするほど内側は緩み外側は厳しくなる、という単調な構造を示している）。
    """
    heading = (math.cos(start_pose.theta), math.sin(start_pose.theta))

    def feasible(r: float) -> bool:
        segments, consumed = turn_path(delta_theta, r)
        lead = segments[0].length  # turn_path が決めた直線区間の長さ（消費長＋余裕）
        # 角(start_pose)から接点までが consumed、接点から掃引開始点までが lead。
        # 掃引開始点は角から heading の逆向きに (consumed + lead) だけ戻った点になる。
        back_off = consumed + lead
        sweep_start = Pose(
            start_pose.x - back_off * heading[0],
            start_pose.y - back_off * heading[1],
            start_pose.theta,
        )
        poses = poses_along(segments, sweep_start)
        min_clear, _ = sweep_clearance(poses, obstacles)
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
