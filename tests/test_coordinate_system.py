"""座標系の取り決めを機械が見張る検査。

正本は `docs/COORDINATE_SYSTEM.md`（2026-08-23 ユーザ決定）。
本ファイルは、その取り決めが実装で崩れていないことを確かめる。
**取り決めを変えたいときは、まず正本を改訂すること。**ここだけ直してはいけない。
"""

import math

import numpy as np
import pytest

from classic.ideal import CELL_SIZE, _dir_angle, _turn_delta
from classic.maze_map import _DIR_DELTA, Direction


# ============================================================================
# §3・§4 座標軸と右手系
# ============================================================================
def test_axes_are_right_handed():
    """東 × 北 = 上。これが成り立たなければ右手系ではない（正本 §4）。"""
    east = np.array([1.0, 0.0, 0.0])   # +X
    north = np.array([0.0, 1.0, 0.0])  # +Y
    up = np.array([0.0, 0.0, 1.0])     # +Z
    assert np.allclose(np.cross(east, north), up)


def test_gravity_points_down_the_z_axis():
    """Z 軸が上方正である以上、重力は −Z を向く（正本 §4）。"""
    import re
    from pathlib import Path

    xml = Path(__file__).resolve().parents[1] / "assets" / "mouse_v2.xml"
    m = re.search(r'gravity="([^"]+)"', xml.read_text(encoding="utf-8"))
    assert m is not None, "assets/mouse_v2.xml に gravity の指定が無い"
    gx, gy, gz = (float(v) for v in m.group(1).split())
    assert (gx, gy) == (0.0, 0.0)
    assert gz < 0.0


# ============================================================================
# §1・§6 方位と、区画から実座標への変換
# ============================================================================
def test_direction_deltas_match_east_x_north_y():
    """+x = 東、+y = 北（正本 §3）。"""
    assert _DIR_DELTA[Direction.N] == (0, 1)
    assert _DIR_DELTA[Direction.E] == (1, 0)
    assert _DIR_DELTA[Direction.S] == (0, -1)
    assert _DIR_DELTA[Direction.W] == (-1, 0)


def test_direction_code_increases_counterclockwise():
    """方位コードは反時計回りに増える（正本 §10-3・2026-08-23 工学系の流儀へ付け替え）。

    ヨー角も反時計回りが正なので、**この 2 つは同じ向きである**
    （付け替え前は逆向きで、`_dir_angle` に符号反転が必要だった）。
    """
    assert (Direction.E, Direction.N, Direction.W, Direction.S) == (0, 1, 2, 3)
    # 番号 +1 で左（反時計）回り 90°、すなわちヨー角は +90°。
    # `_dir_angle` は分枝を持つ値を返すので、差は 2π で巻き戻して比べる。
    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    for d in (Direction.E, Direction.N, Direction.W, Direction.S):
        turned = Direction((int(d) + 1) % 4)
        delta = wrap(_dir_angle(turned) - _dir_angle(d))
        assert math.isclose(delta, math.pi / 2.0, abs_tol=1e-12), f"{d.name}→{turned.name}"
    # 🔴 `_dir_angle` の**値の分枝**は付け替え前と同一に保つ
    # （E=0, N=+90°, S=-90°, W=-180°）。素直な `d * π/2` にすると
    # 南が +270°・西が +180° になり、`classic/explorer.py` が
    # `tracker.reset(heading_deg=...)` へ巻き戻していない値を渡しているため
    # ヨー角の積分器の初期値が 2π ずれて**走行が変わる**
    # （2026-08-23 の付け替えで実際に起き、fast_run_profile の検査が検出した）。
    expected = {Direction.E: 0.0, Direction.N: 90.0, Direction.S: -90.0, Direction.W: -180.0}
    for d, deg in expected.items():
        assert math.isclose(math.degrees(_dir_angle(d)), deg, abs_tol=1e-9), (
            f"{d.name} の角度の分枝が変わっている（走行が変わる）"
        )


def test_turn_delta_180_uses_the_negative_branch():
    """`_turn_delta` の 180°折返し（rel=2）は、+π と −π のどちらも同じ向きを
    指す退化ケースで、符号を選ぶ物理的根拠が無い。付け替え前は常に −π を
    返していたので、その分枝を固定する（`_dir_angle` と同じ理由。素直に
    `signed_rel * π/2` にすると +π になり、snap 補正モード・南向き開始の
    走行が変わることを実測で確認済み — `test_wall_correction_mode_snap_
    matches_pre_blend_baseline_exactly` 参照）。
    """
    for from_dir, to_dir in [
        (Direction.N, Direction.S), (Direction.S, Direction.N),
        (Direction.E, Direction.W), (Direction.W, Direction.E),
    ]:
        assert _turn_delta(from_dir, to_dir) == -math.pi, (
            f"{from_dir.name}->{to_dir.name} の180°折返しの分枝が変わっている（走行が変わる）"
        )


def test_cell_center_conversion():
    """区画 (x,y) の中心 = (p·x + p/2, p·y + p/2)（正本 §6）。"""
    assert CELL_SIZE == pytest.approx(0.180)
    for cx, cy in [(0, 0), (1, 0), (0, 1), (7, 8), (15, 15)]:
        assert cx * CELL_SIZE + CELL_SIZE / 2 == pytest.approx(cx * 0.18 + 0.09)
        assert cy * CELL_SIZE + CELL_SIZE / 2 == pytest.approx(cy * 0.18 + 0.09)
    # スタート区画の中心
    assert 0 * CELL_SIZE + CELL_SIZE / 2 == pytest.approx(0.09)


def test_origin_is_the_south_west_post_centre():
    """原点は格子点 (0,0)、すなわちスタート区画の南西の柱の中心（正本 §5）。

    柱の中心は格子点 (p·i, p·j) にある。したがって i=j=0 の柱の中心が原点。
    """
    post_x, post_y = 0 * CELL_SIZE, 0 * CELL_SIZE
    assert (post_x, post_y) == (0.0, 0.0)
    # 隣の柱は 1 区画ぶん離れている（柱の間隔＝区画の一辺）
    assert 1 * CELL_SIZE - 0 * CELL_SIZE == pytest.approx(0.18)


# ============================================================================
# §7 ヨー角
# ============================================================================
@pytest.mark.parametrize(
    "direction, expected_deg",
    [(Direction.E, 0.0), (Direction.N, 90.0), (Direction.W, 180.0), (Direction.S, -90.0)],
)
def test_yaw_zero_is_east_and_ccw_is_positive(direction, expected_deg):
    """ψ=0 が東、反時計回りが正、北が +90°（正本 §7）。

    西は実装が −180° を返すが、+180° と同じ向きである。角度そのものではなく
    **単位ベクトルで**比べる（±180° の表現の違いで落ちないため）。
    """
    got, want = _dir_angle(direction), math.radians(expected_deg)
    assert math.cos(got) == pytest.approx(math.cos(want), abs=1e-12)
    assert math.sin(got) == pytest.approx(math.sin(want), abs=1e-12)
    # 進む向き（区画の変位）とも一致すること
    dx, dy = _DIR_DELTA[direction]
    assert math.cos(got) == pytest.approx(float(dx), abs=1e-12)
    assert math.sin(got) == pytest.approx(float(dy), abs=1e-12)


def test_heading_90_faces_north():
    """heading_deg=90 のとき機体前方 (+x_b) は世界の北 (+Y) を向く（正本 §7）。

    `mouse/sim.py` の `reset_to_start` が作るクォータニオンと同じ式で確かめる。
    """
    heading_rad = math.radians(90.0)
    qw, qz = math.cos(heading_rad / 2.0), math.sin(heading_rad / 2.0)
    # z 軸回り回転で機体の +x_b が世界のどちらを向くか
    # R·[1,0,0] = [1-2qz², 2·qw·qz, 0]
    fx = 1.0 - 2.0 * qz * qz
    fy = 2.0 * qw * qz
    assert fx == pytest.approx(0.0, abs=1e-12)
    assert fy == pytest.approx(1.0)  # 北

    # 機体の左 (+y_b) は西 (−X) を向く
    lx = -2.0 * qw * qz
    ly = 1.0 - 2.0 * qz * qz
    assert lx == pytest.approx(-1.0)
    assert ly == pytest.approx(0.0, abs=1e-12)


# ============================================================================
# §7 機体固定座標系（前方 +x_b・左 +y_b・上 +z_b）
# ============================================================================
def test_body_frame_is_right_handed():
    """前 × 左 = 上（正本 §7）。"""
    fwd = np.array([1.0, 0.0, 0.0])
    left = np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    assert np.allclose(np.cross(fwd, left), up)


def test_left_wheel_is_on_positive_y_and_front_caster_on_positive_x():
    """機体固定系で +y_b が左、+x_b が前方であること（正本 §7）。

    左輪の y が正、右輪の y が負。前キャスタの x が正、後キャスタの x が負。
    """
    import re
    from pathlib import Path

    xml = (Path(__file__).resolve().parents[1] / "assets" / "mouse_v2.xml").read_text(encoding="utf-8")

    def body_pos(name: str):
        m = re.search(rf'<body[^>]*name="{name}"[^>]*pos="([^"]+)"', xml)
        if m is None:
            m = re.search(rf'<geom[^>]*name="{name}"[^>]*pos="([^"]+)"', xml)
        assert m is not None, f"{name} が assets/mouse_v2.xml に無い"
        return [float(v) for v in m.group(1).split()]

    assert body_pos("left_wheel")[1] > 0.0, "左輪は +y_b 側にあるべき"
    assert body_pos("right_wheel")[1] < 0.0, "右輪は −y_b 側にあるべき"
    assert body_pos("caster_front")[0] > 0.0, "前キャスタは +x_b 側にあるべき"
    assert body_pos("caster_back")[0] < 0.0, "後キャスタは −x_b 側にあるべき"


def test_side_sensors_are_named_from_the_body_frame():
    """左センサは +y_b 側、右センサは −y_b 側にある（正本 §7）。

    センサの左右は**機体基準**の呼び名であり、迷路の東西ではない。
    """
    from mouse.params import RobotParams

    specs = {s["name"]: s for s in RobotParams().sensors}
    ypos = {k: float(v["pos"].split()[1]) for k, v in specs.items()}
    xpos = {k: float(v["pos"].split()[0]) for k, v in specs.items()}

    for left, right in [("LF", "RF"), ("LS", "RS")]:
        assert ypos[left] > 0.0, f"{left} は +y_b（機体の左）側にあるべき"
        assert ypos[right] < 0.0, f"{right} は −y_b（機体の右）側にあるべき"
        assert ypos[left] == pytest.approx(-ypos[right]), f"{left}/{right} は左右対称であるべき"
        assert xpos[left] > 0.0, f"{left} は前方 (+x_b) 側にあるべき"

    # 側方センサの光軸は機体の横（±y_b）を向く。左は +y、右は −y。
    for left, right in [("LS", "RS")]:
        assert float(specs[left]["zaxis"].split()[1]) > 0.0
        assert float(specs[right]["zaxis"].split()[1]) < 0.0
