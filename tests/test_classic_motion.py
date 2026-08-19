"""
tests/test_classic_motion.py
================
classic/motion.py の検査。**実際にシミュレータを動かして**区画単位の動作
（直進 N 区画・その場旋回）が目標に収束することを確認する。

真値位置（privileged_pose/privileged_velocity）は検査の**答え合わせにのみ**
使う。CellMotionController 自体がこれらを使っていないことは、
実行経路（ソースコード）を機械的に検査して確認する
（note_030 §5-2「実行経路そのものを検査する」に対応）。
"""
import inspect
import math
import os

import numpy as np
import pytest

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

from classic import motion as motion_module
from classic.motion import CellMotionController

MAX_STEPS = 6000  # 安全弁（収束しない場合に無限ループしない）


@pytest.fixture(scope="module")
def params():
    return RobotParams()


@pytest.fixture()
def open_sim(tmp_path, params):
    """内部の壁を全て取り払った 5x5 迷路（外周のみ壁）。直進・旋回とも
    衝突なしで検査できる。"""
    W, H = 5, 5
    v = np.zeros((W + 1, H), dtype=int)
    v[0, :] = 1
    v[W, :] = 1
    h = np.zeros((W, H + 1), dtype=int)
    h[:, 0] = 1
    h[:, H] = 1
    xml_path = os.path.join(str(tmp_path), "open.xml")
    build_maze_robot_xml(v, h, xml_path, model_name="open5x5", params=params)
    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(2, 2), heading_deg=90.0)
    return sim


def _run_until_done(sim, ctrl):
    for step in range(MAX_STEPS):
        obs = sim.observation()
        vl, vr, done = ctrl.update(obs)
        sim.step_control(vl, vr)
        if done:
            return step
    raise AssertionError(f"{MAX_STEPS} ステップ以内に収束しなかった（発散または閾値不良の疑い）")


# ==========================================================================
# 0. 実行経路の検査: privileged_pose/privileged_velocity を使っていないこと
# ==========================================================================
def test_controller_does_not_reference_privileged_state():
    """CellMotionController のソースに privileged_pose / privileged_velocity への
    **実際の属性アクセス**が無いことを AST で機械的に確認する（S1 の必須要件:
    真値位置を使わないこと）。docstring 中の説明文（「使わない」という記述自体）は
    文字列リテラルであって属性アクセスではないため、単純な部分文字列検索では
    誤検出する。AST の Attribute ノードだけを見ることで区別する。"""
    import ast
    src = inspect.getsource(motion_module.CellMotionController)
    tree = ast.parse(src)
    forbidden_attrs = {"privileged_pose", "privileged_velocity"}
    found = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs
    }
    assert not found, f"禁止属性への実際のアクセスを検出: {found}"


# ==========================================================================
# 1. 直進 N 区画が目標距離に収束すること（真値との突合はここでのみ行う）
# ==========================================================================
def test_forward_two_cells_reaches_target_distance_and_holds_heading(open_sim, params):
    sim = open_sim
    ctrl = CellMotionController(params)
    ctrl.reset(heading_deg=90.0)

    x0, y0, yaw0 = sim.privileged_pose()
    ctrl.start_forward(2)
    steps = _run_until_done(sim, ctrl)
    x1, y1, yaw1 = sim.privileged_pose()

    displacement = math.hypot(x1 - x0, y1 - y0)
    target = 2 * params.cell_size
    heading_drift_deg = math.degrees(abs(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))))

    print(f"\n[実測] 直進2区画: 目標={target:.4f}m 実測変位={displacement:.4f}m "
          f"誤差={displacement - target:+.4f}m steps={steps} 方位ずれ={heading_drift_deg:.3f}deg")

    assert abs(displacement - target) < 0.01, "直進の到達距離誤差が 10mm を超えている"
    assert heading_drift_deg < 1.0, "直進中に方位が 1° 以上ずれた（方位保持が効いていない）"


def test_forward_one_cell_reaches_target_distance(open_sim, params):
    sim = open_sim
    ctrl = CellMotionController(params)
    ctrl.reset(heading_deg=90.0)

    x0, y0, _ = sim.privileged_pose()
    ctrl.start_forward(1)
    _run_until_done(sim, ctrl)
    x1, y1, _ = sim.privileged_pose()

    displacement = math.hypot(x1 - x0, y1 - y0)
    target = 1 * params.cell_size
    print(f"\n[実測] 直進1区画: 目標={target:.4f}m 実測変位={displacement:.4f}m")
    assert abs(displacement - target) < 0.01


# ==========================================================================
# 2. その場旋回が目標角度に収束し、位置がほぼ動かないこと
# ==========================================================================
@pytest.mark.parametrize("start_method,expected_delta_deg", [
    ("start_turn_left", 90.0),
    ("start_turn_right", -90.0),
    ("start_turn_180", 180.0),
])
def test_turn_reaches_target_angle_without_translating(open_sim, params, start_method, expected_delta_deg):
    sim = open_sim
    ctrl = CellMotionController(params)
    ctrl.reset(heading_deg=90.0)

    x0, y0, yaw0 = sim.privileged_pose()
    getattr(ctrl, start_method)()
    steps = _run_until_done(sim, ctrl)
    x1, y1, yaw1 = sim.privileged_pose()

    actual_delta_deg = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))
    # 180°は符号の丸めで ±180 の境界に乗りうるので絶対値で比較する
    if abs(expected_delta_deg) >= 179.0:
        angle_err = abs(abs(actual_delta_deg) - abs(expected_delta_deg))
    else:
        angle_err = abs(actual_delta_deg - expected_delta_deg)
    drift = math.hypot(x1 - x0, y1 - y0)

    print(f"\n[実測] {start_method}: 目標={expected_delta_deg:+.1f}deg 実測={actual_delta_deg:+.3f}deg "
          f"誤差={angle_err:.3f}deg steps={steps} 位置ずれ={drift*1000:.3f}mm")

    assert angle_err < 3.0, "旋回の到達角度誤差が 3° を超えている"
    assert drift < 0.01, "旋回中に 10mm 以上並進した（超信地旋回になっていない）"


# ==========================================================================
# 3. stop コマンドは常に即座に完了しゼロ電圧を返すこと
# ==========================================================================
def test_stop_is_immediate_and_zero_voltage(open_sim, params):
    sim = open_sim
    ctrl = CellMotionController(params)
    ctrl.reset(heading_deg=90.0)
    ctrl.start_stop()
    obs = sim.observation()
    vl, vr, done = ctrl.update(obs)
    assert done is True
    assert vl == 0.0
    assert vr == 0.0


# ==========================================================================
# 4. わざと目標を届かない値へ壊すと、収束判定が正しく「未達」を示すこと
#    （空振りしない検査であることの確認: 閾値/目標を壊せば失敗が検出できる）
# ==========================================================================
def test_broken_target_makes_convergence_check_fail(open_sim, params):
    sim = open_sim
    ctrl = CellMotionController(params)
    ctrl.reset(heading_deg=90.0)

    x0, y0, _ = sim.privileged_pose()
    ctrl.start_forward(1)
    _run_until_done(sim, ctrl)
    x1, y1, _ = sim.privileged_pose()
    displacement = math.hypot(x1 - x0, y1 - y0)

    # 正しい目標(1区画=0.18m)との比較は通る
    assert abs(displacement - params.cell_size) < 0.01

    # 目標を意図的に壊す（例: 3区画分を期待してしまう誤り）と、
    # 同じ実測値に対する妥当性検査が実際に落ちることを確認する
    broken_target = 3 * params.cell_size
    with pytest.raises(AssertionError):
        assert abs(displacement - broken_target) < 0.01
