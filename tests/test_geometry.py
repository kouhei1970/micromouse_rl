"""tests/test_geometry.py — `classic/geometry.py`（幾何と干渉判定の層）の検査

方針:

- 検査1は `mouse/mjcf.py::build_maze_robot_xml` が実際に書き出す MJCF の geom
  （柱・壁の位置と半サイズ）と `wall_obstacles()` の出力を**直接突き合わせる**
  （MuJoCo を import せず `xml.etree.ElementTree` で XML を読むだけなので、
  `classic/` パッケージの「MuJoCo を import しない」規約と両立する）。
- 検査2・5は `experiments/exp_025_s4_slalom/geometry_anchor.py`・
  `research_notes/note_031_profile_planner_and_eta.md` §結果 の数値との**独立な
  2 通りの計算の照合**である。合わなければ合わせ込まず、どちらが正しいか報告する
  （このファイルはその方針のとおり、実測値をそのままアサーションにしている）。
- 検査3は教授セッションの手計算（外側 R<=223mm・内側 R<=192mm、内側が拘束）との照合。
- 検査4は `entry_offset` を通路の外側いっぱい（`W_C=0.044`）にしたときの挙動。
  🔴 実装過程で分かったこと（コメント参照）: 文字通り `entry_offset=0.044` は
  直線区間の余裕を完全に使い切る（定義上、直線部の余裕がちょうど 0 になる）ため、
  本モジュールの**機体の外形をそのまま使う厳密な干渉判定**では、どの半径でも
  弧の途中でさらに食い込み、`margin>=0` を満たす半径が現実的な範囲に存在しない
  （祖先スクリプトは機体を点として扱うため、この制約が見えていない）。
  そのため本検査では `W_C` の一部（20mm）を使う現実的な値で「中心線より明確に
  大きくなる」ことを確認し、文字通りの `W_C=0.044` では**どの半径も通らない**
  ことを別途アサーションで示す。

`classic/geometry.py` 自身の docstring も参照。
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from mouse.mjcf import build_maze_robot_xml

from classic.geometry import (  # noqa: E402
    HALF_WIDTH,
    Pose,
    PathSegment,
    Rect,
    clearance,
    max_feasible_radius,
    poses_along,
    sweep_clearance,
    turn_path,
    wall_obstacles,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# 1. wall_obstacles が mouse/mjcf.py の建て方と一致する
# ============================================================================
@pytest.mark.parametrize("center_goal", [False, True])
def test_wall_obstacles_match_the_mjcf_geometry(center_goal, tmp_path):
    """4x4 の迷路で、生成した長方形の数と位置が `mouse/mjcf.py` の建て方と一致する。

    `build_maze_robot_xml` は MuJoCo を import せず ElementTree だけで XML を書くので、
    ここでは実際にモデルへロードせず、書き出された XML の geom をそのまま読む。
    """
    width, height = 4, 4
    v_walls = np.zeros((width + 1, height), dtype=np.int8)
    h_walls = np.zeros((width, height + 1), dtype=np.int8)
    rng = np.random.RandomState(20260820)
    v_walls[:] = (rng.random_sample(v_walls.shape) < 0.5).astype(np.int8)
    h_walls[:] = (rng.random_sample(h_walls.shape) < 0.5).astype(np.int8)
    # 外周は迷路らしく壁で閉じる（現実的な迷路の形にするだけで、判定の本質には無関係）。
    v_walls[0, :] = 1
    v_walls[-1, :] = 1
    h_walls[:, 0] = 1
    h_walls[:, -1] = 1

    out_path = tmp_path / "maze.xml"
    # center_goal は True/False の両方で突き合わせる（wall_obstacles も同じ値を渡す）。
    # True のとき、mjcf も wall_obstacles も格子点 (width//2, height//2) の柱を立てない。
    build_maze_robot_xml(v_walls, h_walls, str(out_path), center_goal=center_goal)

    root = ET.parse(out_path).getroot()
    worldbody = root.find("worldbody")

    posts, v_wall_geoms, h_wall_geoms = [], [], []
    for geom in worldbody.findall("geom"):
        name = geom.get("name") or ""
        if name.startswith("post_"):
            posts.append(geom)
        elif name.startswith("v_wall_top_") or name.startswith("h_wall_top_"):
            continue  # 赤い壁上端（2D の干渉判定には寄与しない飾り）
        elif name.startswith("v_wall_"):
            v_wall_geoms.append(geom)
        elif name.startswith("h_wall_"):
            h_wall_geoms.append(geom)

    def _to_rect(geom) -> Rect:
        pos = [float(v) for v in geom.get("pos").split()]
        size = [float(v) for v in geom.get("size").split()]
        return Rect(pos[0], pos[1], size[0], size[1])

    xml_rects = (
        [_to_rect(g) for g in posts]
        + [_to_rect(g) for g in v_wall_geoms]
        + [_to_rect(g) for g in h_wall_geoms]
    )
    my_rects = wall_obstacles(v_walls, h_walls, center_goal=center_goal)

    # 柱の数 = 格子点の数（center_goal=True ならゴール中央の 1 本ぶん少ない）
    assert len(posts) == (width + 1) * (height + 1) - (1 if center_goal else 0)
    # 壁の数 = 実際に壁があるマスの数
    assert len(v_wall_geoms) + len(h_wall_geoms) == int(v_walls.sum() + h_walls.sum())
    # 壁の長さ = cell_size - post_size（縦壁の hy・横壁の hx のどちらにも現れる）
    expected_half_run = 0.180 / 2 - 0.012 / 2
    for g in v_wall_geoms:
        assert math.isclose(float(g.get("size").split()[1]), expected_half_run, abs_tol=1e-9)
    for g in h_wall_geoms:
        assert math.isclose(float(g.get("size").split()[0]), expected_half_run, abs_tol=1e-9)

    assert len(xml_rects) == len(my_rects)
    for xml_rect, my_rect in zip(xml_rects, my_rects):
        assert math.isclose(xml_rect.cx, my_rect.cx, abs_tol=1e-9)
        assert math.isclose(xml_rect.cy, my_rect.cy, abs_tol=1e-9)
        assert math.isclose(xml_rect.hx, my_rect.hx, abs_tol=1e-9)
        assert math.isclose(xml_rect.hy, my_rect.hy, abs_tol=1e-9)


# ============================================================================
# 2〜4・6・7 共通: 90° ターンの局所迷路
# ============================================================================
# 西区画(0,0) → 角区画(1,0) → 北区画(1,1) の 90° 左ターン。
# 角区画の NW 隅の柱（グリッド点 (1,1) = 世界座標 (0.18,0.18)）が「内側の柱」、
# 角区画の南壁・東壁が「外側の壁」（対称性により掃引の最小値はどちらに対しても
# ほぼ同じ値になる。二分探索で確認済み）。
# 進入通路の中心線 y=0.09、退出通路の中心線 x=0.27（区画中心線どうしを結ぶ）。
CELL = 0.180
ENTRY_LINE_Y = 0.09
EXIT_LINE_X = 0.27
INNER_POST = Rect(0.18, 0.18, 0.006, 0.006)


def _turn_obstacles():
    v_walls = np.zeros((3, 2), dtype=np.int8)
    h_walls = np.zeros((2, 3), dtype=np.int8)
    v_walls[2, 0] = 1  # 角区画の東壁（外側）
    v_walls[1, 1] = 1  # 北区画の西壁
    v_walls[2, 1] = 1  # 北区画の東壁
    h_walls[0, 0] = 1  # 西区画の南壁
    h_walls[1, 0] = 1  # 角区画の南壁（外側）
    h_walls[0, 1] = 1  # 西区画の北壁
    # 🔴 この合成迷路はゴールを持たないので center_goal=False。
    # 既定(True)にすると (width//2, height//2)=(1,1) の柱 ── まさにこのターンの
    # 内側の柱 ── が消え、最大半径が 190.2mm から 210.2mm へ変わってしまう。
    return wall_obstacles(v_walls, h_walls, center_goal=False)


def _corner_pose(offset: float = 0.0) -> Pose:
    """ターンの角（進入直線・退出直線をそれぞれ延長したときの交点）。

    `offset` は通路を外側（内側の柱から遠ざかる向き）へ寄せる量。進入通路では
    -y 方向、退出通路では +x 方向が外側にあたる（このターンの旋回方向で決まる）。
    """
    return Pose(EXIT_LINE_X + offset, ENTRY_LINE_Y - offset, 0.0)


def _quarter_turn_poses(radius: float, offset: float = 0.0):
    """半径 `radius`・`offset` の 90° 左ターンの掃引姿勢列。"""
    delta = math.pi / 2
    corner = _corner_pose(offset)
    segments, consumed = turn_path(delta, radius)
    lead = segments[0].length
    back_off = consumed + lead
    sweep_start = Pose(corner.x - back_off, corner.y, corner.theta)
    return poses_along(segments, sweep_start)


def _min_clearance(poses, obstacles) -> float:
    best = math.inf
    for p in poses:
        d = clearance(p, obstacles)
        if d < best:
            best = d
    return best


# ============================================================================
# 2. 四分円（R=90mm）の余裕が geometry_anchor.py と一致する
# ============================================================================
def test_quarter_arc_clearances_match_the_anchor():
    """R=90mm（区画中心線どうしを結ぶ四分円）の内側（柱）・外側（壁）の余裕。

    `experiments/exp_025_s4_slalom/geometry_anchor.py` の値: 内側 41.5mm・外側 34.7mm
    （別の導き方 — 幾何の公式による手計算 — での独立な結果）。
    """
    obstacles = _turn_obstacles()
    poses = _quarter_turn_poses(radius=CELL / 2.0)  # R=0.09

    inner = _min_clearance(poses, [INNER_POST])
    outer = _min_clearance(poses, obstacles)  # 内側の柱は外側の壁より緩いので全体最小=外側の壁

    inner_mm, outer_mm = inner * 1000.0, outer * 1000.0
    print(f"\n[検査2] 内側(柱)={inner_mm:.3f}mm (期待 ~41.5)  外側(壁)={outer_mm:.3f}mm (期待 ~34.7)")

    assert abs(inner_mm - 41.5) <= 0.5, f"内側の柱の余裕が anchor と合わない: {inner_mm:.3f}mm"
    assert abs(outer_mm - 34.7) <= 0.5, f"外側の壁の余裕が anchor と合わない: {outer_mm:.3f}mm"


# ============================================================================
# 3. 中心線上の 90° ターンの最大半径が約 190mm
# ============================================================================
def test_centerline_max_radius_is_about_190mm():
    """中心線上（offset=0）の 90° ターンで `max_feasible_radius`（margin=0）が約 190mm。

    教授セッションの手計算による 2 つの制約と照合する:
    - 外側（前の区画へのはみ出し）: R - sqrt(180R-8100) <= 44 [mm] → R <= 223
    - 内側（柱をかわす）: (R-90)*sqrt(2) + 6*sqrt(2) <= R - 40 [mm] → R <= 約 192
    内側が拘束するはずなので、シミュレーションの結果は 192mm 側に近いことを期待する。
    """
    obstacles = _turn_obstacles()
    delta = math.pi / 2
    r_star = max_feasible_radius(delta, obstacles, _corner_pose(0.0), margin=0.0)
    r_mm = r_star * 1000.0

    print(f"\n[検査3] max_feasible_radius(centerline) = {r_mm:.3f}mm "
          f"(期待 ~190mm。手計算: 外側<=223mm・内側<=192mm)")

    assert abs(r_mm - 190.0) <= 5.0, f"約190mmから5mm以上ずれている: {r_mm:.3f}mm"
    # 内側の手計算(192mm)との近さも記録する（🔴 5mm以上ずれたら合わせ込まず報告する対象）。
    assert abs(r_mm - 192.0) <= 5.0, (
        f"教授セッションの内側の手計算(192mm)と5mm以上ずれている: {r_mm:.3f}mm"
    )


# ============================================================================
# 4. 通路の外側へ寄せると、より大きな半径が通る
# ============================================================================
def test_outward_offset_allows_a_larger_radius():
    """`entry_offset` で通路の外側へ寄せると `max_feasible_radius` が明確に大きくなる。

    祖先スクリプト（`experiments/exp_025_s4_slalom` 系）の点モデルでの
    `R_MAX = (4+2*sqrt(2))*W_C = 300.5mm`（`W_C=0.044`）にどこまで近づくかを印字する
    （一致は要求しない。祖先は機体を点として扱っているため差が出るはず）。

    🔴 実装中に分かったこと: 文字通り `offset=W_C=0.044` を使うと、進入・退出の
    直線区間はその定義上ちょうど余裕 0（機体の外側面が壁に接する）になる。
    これは**すべての半径**で成立する不変の事実であり、機体を矩形として扱う本モジュールの
    厳密な掃引では、弧の部分でさらに内側の柱・外側の壁のどちらかに食い込むため、
    `margin>=0` を満たす半径が現実的な範囲（`r_hi=0.40` 程度まで）に存在しない
    （下のアサーションで確認する）。祖先の点モデルの `R_MAX=300.5mm` は、**機体の
    大きさを無視した仮想の上限**であり、有限サイズの機体では文字通りには実現できない
    ことが、点モデルとの独立な照合で初めて分かった差である。
    """
    obstacles = _turn_obstacles()
    delta = math.pi / 2

    r_center = max_feasible_radius(delta, obstacles, _corner_pose(0.0), margin=0.0)

    offset = 0.020  # W_C=0.044 の一部（約45%）。中心線より明確に大きくなることを
    # 単調な（"文字通り W_C いっぱい" のような境界ケースを避けた）掃引で確認する。
    r_offset = max_feasible_radius(delta, obstacles, _corner_pose(offset), margin=0.0)

    r_center_mm, r_offset_mm = r_center * 1000.0, r_offset * 1000.0
    ancestor_r_max_mm = 300.5
    print(
        f"\n[検査4] centerline={r_center_mm:.3f}mm  offset={offset*1000:.0f}mm→"
        f"{r_offset_mm:.3f}mm  (祖先の点モデル R_MAX={ancestor_r_max_mm}mm の "
        f"{r_offset_mm/ancestor_r_max_mm*100:.1f}%)"
    )

    assert r_offset_mm > r_center_mm * 1.2, (
        f"外側へ寄せても中心線より明確に大きくならなかった: "
        f"centerline={r_center_mm:.3f}mm offset={r_offset_mm:.3f}mm"
    )

    # 追加の発見: 文字通り W_C=0.044 いっぱいに寄せると、[0.02, 0.40] のどの半径も
    # margin=0 を満たさない（下限ですら通らない）。合わせ込まず、そのまま報告する。
    with pytest.raises(ValueError):
        max_feasible_radius(delta, obstacles, _corner_pose(0.044), margin=0.0)


# ============================================================================
# 5. 斜め（45°）通路の幅
# ============================================================================
def test_diagonal_corridor_width():
    """斜め（45°）の直線を掃引し、片側の空き幅 55.15mm・機体中心の横余地 15.15mm と一致する。

    `research_notes/note_031_profile_planner_and_eta.md` §結果「3. 斜め走行の通路幾何」
    （「片側の空き幅」「機体中心の横方向の余地 W_C」）との照合。診断（`note_031`）は
    「柱の配置から直接検算した」とあるので、ここでは wall_obstacles() が作る**柱**
    （壁は無し）に対して、機体を点として掃引した場合（空き幅そのもの）と、実際の
    機体を掃引した場合（W_C）の両方を計算する。

    斜め通路の最も広い経路は、格子点を通らず（通ると柱の中心を貫通してしまう）
    半セルだけずらした線（`y = x + cell_size/2`）であることが分かっている
    （柱の総当たりで最大化して確認済み。このオフセットで隣接する 2 本の柱の
    ちょうど中間を通る）。
    """
    width = height = 4
    v_walls = np.zeros((width + 1, height), dtype=np.int8)
    h_walls = np.zeros((width, height + 1), dtype=np.int8)
    obstacles = wall_obstacles(v_walls, h_walls)  # 壁は無し、柱だけ（全格子点）

    theta = math.radians(45.0)
    offset_c = CELL / 2.0  # y = x + offset_c が最も広い斜め経路
    start = Pose(-0.03, -0.03 + offset_c, theta)
    segments = [PathSegment(kind="straight", length=0.30, curvature=0.0)]
    poses = poses_along(segments, start, ds=0.001)

    def _point_rect_distance(px: float, py: float, rect: Rect) -> float:
        dx = max(0.0, abs(px - rect.cx) - rect.hx)
        dy = max(0.0, abs(py - rect.cy) - rect.hy)
        return math.hypot(dx, dy)

    channel_half_width = min(
        _point_rect_distance(p.x, p.y, o) for p in poses for o in obstacles
    )
    lateral_room, _ = sweep_clearance(poses, obstacles)

    channel_mm, room_mm = channel_half_width * 1000.0, lateral_room * 1000.0
    print(f"\n[検査5] 片側の空き幅={channel_mm:.3f}mm (期待 ~55.15)  "
          f"機体中心の横余地={room_mm:.3f}mm (期待 ~15.15)")

    assert abs(channel_mm - 55.15) <= 0.5
    assert abs(room_mm - 15.15) <= 0.5
    # 空き幅から機体半幅を引くと横余地に一致する（同じ量を 2 通りに測っただけであることの確認）
    assert math.isclose(channel_half_width - HALF_WIDTH, lateral_room, abs_tol=1e-6)


# ============================================================================
# 6. 否定対照（作動側）: 通れない半径では負になる
# ============================================================================
def test_collision_is_detected():
    """通れない半径（R=0.30・中心線上）では掃引の最小余裕が負になる。"""
    obstacles = _turn_obstacles()
    poses = _quarter_turn_poses(radius=0.30)
    min_clear, _ = sweep_clearance(poses, obstacles)
    assert min_clear < 0.0, f"R=0.30 は通れないはずが余裕が負にならなかった: {min_clear*1000:.3f}mm"


# ============================================================================
# 7. 否定対照（空振り側）: 既知の良好な半径では正になる
# ============================================================================
def test_no_collision_for_the_known_good_case():
    """R=0.09（区画中心線どうしを結ぶ四分円）では最小余裕が正になる。"""
    obstacles = _turn_obstacles()
    poses = _quarter_turn_poses(radius=0.09)
    min_clear, _ = sweep_clearance(poses, obstacles)
    assert min_clear > 0.0, f"R=0.09 は通れるはずが余裕が正にならなかった: {min_clear*1000:.3f}mm"


# ============================================================================
# 8. ゴール中央の柱は立たない（mouse/mjcf.py の center_goal と同じ扱い）
# ============================================================================
def test_goal_center_post_is_absent():
    """競技規約（RESEARCH_PLAN §2）でゴール中央の格子点だけ柱が無い。

    `mouse/mjcf.py::build_maze_robot_xml` は `center_goal=True`（既定）のとき
    格子点 `(width//2, height//2)` の柱を生成しない。障害物の側もそれに合わせる。
    合っていないと、実在しない柱をゴール付近に置き、最短走行の終端で
    通れる半径を実際より小さく見積もることになる。

    作動側（既定 True で柱が無い）と空振り側（False にすると柱がある）を対で検査する。
    """
    import numpy as np
    from classic.geometry import wall_obstacles

    w = h = 6
    v = np.zeros((w + 1, h), dtype=np.int8)
    hh = np.zeros((w, h + 1), dtype=np.int8)
    cx, cy = (w // 2) * 0.180, (h // 2) * 0.180

    at_center = lambda obs: [o for o in obs
                             if abs(o.cx - cx) < 1e-9 and abs(o.cy - cy) < 1e-9]

    assert at_center(wall_obstacles(v, hh)) == [], "ゴール中央に柱が立っている"
    assert len(at_center(wall_obstacles(v, hh, center_goal=False))) == 1, \
        "center_goal=False にしても柱が生成されない（検査が空振りしている）"

    # 柱の総数も対で確認する（格子点の数 −1 になるはず）
    n_true = len(wall_obstacles(v, hh))
    n_false = len(wall_obstacles(v, hh, center_goal=False))
    assert n_false - n_true == 1, f"差が 1 本ではない: {n_false - n_true}"
