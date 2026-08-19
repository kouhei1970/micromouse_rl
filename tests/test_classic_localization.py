"""
tests/test_classic_localization.py
================
`classic/localization.py`（S2: 壁センサによる区画ごとの位置補正）の検査。
`research_notes/note_030_classical_rebuild_plan.md` §3 S2 の任務指示に
対応する。**実際にシミュレータを動かして**実測する（note_030 §5 の作法）。

構成:
  1. 幾何モデルの実測検証（`wall_standoff`・`estimate_lateral_offset` が
     実際のシミュレータの壁位置を正しく復元できること。AMBIGUOUS/CLEAR
     からは推定しないこと）
  2. 無効化 (`enabled=False`) が完全な no-op であること
     （`sense_walls` すら呼ばない・`CellMotionController(localizer=None)`
     と bit 一致すること）
  3. 実行経路検査（型 B 再発防止）
  4. 否定対照 N1/N2（真値を一切使っていないことの実測）
  5. **ドリフトの実測比較**（本タスクの核心。10 区画の連続直進後の
     横方向ドリフトを補正あり/なしで比較する）
  6. **到達率の再測定**（`competition/mazes/design_v4` の同じ 3 面で
     `CompetitionEvaluator` を回し、補正あり/なしでゴール到達数を比較する）
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import mujoco

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

from competition.evaluator import CompetitionEvaluator
from competition.policy_interface import MousePolicy

from classic.checks import assert_same_callable, negative_control
from classic.explorer import ClassicExplorer
from classic.localization import (
    Localizer,
    estimate_forward_offset,
    estimate_lateral_offset,
    wall_standoff,
)
from classic.motion import CellMotionController
from classic.sensing import WallSensing, sense_walls
from classic.sensing import WallState as SenseWallState
import classic.localization as localization_module
from classic.localization import _sensor_geom

REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGN_V4_DIR = REPO_ROOT / "competition" / "mazes" / "design_v4"

MAX_STEPS = 6000  # 収束しない場合の安全弁


@pytest.fixture(scope="module")
def params():
    return RobotParams()


# ==========================================================================
# ヘルパー（tests/test_classic_sensing.py・tests/test_classic_motion.py と
# 同じ作法。位置ずれ ±15mm・方位ずれ ±4° を直接 qpos へ与える）
# ==========================================================================
W5, H5 = 5, 5
CX, CY = 2, 2  # 境界の影響を避けるため中央寄りのセルを使う


def _walled_cell_maze() -> Tuple[np.ndarray, np.ndarray]:
    """CX,CY を四方すべて壁で囲んだ 5x5 迷路。"""
    v = np.ones((W5 + 1, H5), dtype=int)
    h = np.ones((W5, H5 + 1), dtype=int)
    return v, h


def _build_sim(v, h, tmp_path, label, params) -> MouseSim:
    xml_path = os.path.join(str(tmp_path), f"maze_{label}.xml")
    build_maze_robot_xml(v, h, xml_path, model_name=f"m_{label}", params=params)
    return MouseSim(xml_path, params=params)


def _set_pose(sim: MouseSim, dx: float, dy: float, dheading_deg: float) -> None:
    """区画 (CX,CY) の中心からの位置ずれ dx,dy [m]・方位ずれ [deg]（基準90°=北）
    を直接 qpos へ与える（tests/test_classic_sensing.py の `_set_pose` と同一作法）。"""
    cell = sim.params.cell_size
    cx0 = CX * cell + cell / 2
    cy0 = CY * cell + cell / 2
    qpos_adr = sim.model.jnt_qposadr[sim._root_joint_id]
    x = cx0 + dx
    y = cy0 + dy
    heading = math.radians(90.0 + dheading_deg)
    sim.data.qpos[qpos_adr:qpos_adr + 3] = [x, y, 0.002]
    sim.data.qpos[qpos_adr + 3] = math.cos(heading / 2.0)
    sim.data.qpos[qpos_adr + 4] = 0.0
    sim.data.qpos[qpos_adr + 5] = 0.0
    sim.data.qpos[qpos_adr + 6] = math.sin(heading / 2.0)
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def _run_until_done(sim: MouseSim, ctrl: CellMotionController) -> int:
    for step in range(MAX_STEPS):
        obs = sim.observation()
        vl, vr, done = ctrl.update(obs)
        sim.step_control(vl, vr)
        if done:
            return step
    raise AssertionError(f"{MAX_STEPS} ステップ以内に収束しなかった（発散または閾値不良の疑い）")


def _corridor_walls(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """幅 `width` 区画・高さ `height` 区画の、外周だけ壁のコリドー。"""
    v = np.zeros((width + 1, height), dtype=int)
    v[0, :] = 1
    v[width, :] = 1
    h = np.zeros((width, height + 1), dtype=int)
    h[:, 0] = 1
    h[:, height] = 1
    return v, h


# ==========================================================================
# 1. 幾何モデルの実測検証
# ==========================================================================
@pytest.fixture(scope="module")
def walled_sim(tmp_path_factory, params):
    v, h = _walled_cell_maze()
    tmp = tmp_path_factory.mktemp("wall_geom")
    return _build_sim(v, h, tmp, "walled", params)


def test_wall_standoff_matches_measured_geometry(walled_sim, params):
    """`wall_standoff()` の較正値（`classic/localization.py` docstring・
    `MEASURED_STANDOFF_CORRECTION_M`）を、実際のシミュレータで測り直して
    確認する（`classic/sensing.py` のしきい値検査と同じ「実測に基づく、
    決め打ちではない」作法。同じ数値をこのテストが実行のたびに再現する）。"""
    sim = walled_sim
    _set_pose(sim, 0.0, 0.0, 0.0)  # 区画中心・方位ずれ0
    obs = sim.observation()
    sensing = sense_walls(obs, params)

    # 検査専用に内部ヘルパ(_sensor_geom)を直接使い、独立に standoff を逆算する。
    (_, y_ls, _), (_, dy_ls, _) = _sensor_geom(params, 'LS')
    (x_lf, _, _), (dx_lf, _, _) = _sensor_geom(params, 'LF')

    standoff_from_side = sensing.left_dist * dy_ls + y_ls
    standoff_from_front = sensing.front_dist * dx_lf + x_lf
    standoff_config = wall_standoff(params)

    print(f"\n[実測] wall_standoff: 側方由来={standoff_from_side*1000:.4f}mm "
          f"前方由来={standoff_from_front*1000:.4f}mm 設定値={standoff_config*1000:.4f}mm")

    assert abs(standoff_from_side - standoff_config) < 0.0005, "側方センサから逆算した standoff が設定値と食い違う"
    assert abs(standoff_from_front - standoff_config) < 0.0005, "前方センサから逆算した standoff が設定値と食い違う"
    # 素朴な幾何計算（実測較正なし）との差が実在すること自体も報告する
    # （localization.py の MEASURED_STANDOFF_CORRECTION_M の根拠）。
    naive = params.cell_size / 2.0 - localization_module.WALL_THICKNESS / 2.0
    print(f"[実測] 素朴計算との差: {(naive - standoff_config)*1000:.4f}mm")


@pytest.mark.parametrize("shift_mm", [-20.0, -15.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 15.0, 20.0])
def test_estimate_lateral_offset_recovers_a_known_lateral_shift(walled_sim, params, shift_mm):
    """区画中心から既知の横ずれ（x 方向、heading=90°=北向きなので機体座標の
    -y=右方向にあたる）を与え、`estimate_lateral_offset` がそれを正しい符号・
    妥当な精度で復元することを実測する。"""
    sim = walled_sim
    _set_pose(sim, shift_mm / 1000.0, 0.0, 0.0)
    obs = sim.observation()
    sensing = sense_walls(obs, params)
    est = estimate_lateral_offset(sensing, params)

    # heading=90°(北)のとき、機体座標 +y(左)は世界座標 -x にあたる
    # (mouse/sim.py の座標系。90°回転で body+x->world+y, body+y->world-x)。
    # よって世界 +x への shift は e(左正)としては負になるはず。
    expected_e = -shift_mm / 1000.0
    print(f"\n[実測] shift={shift_mm:+.1f}mm 判定={sensing.left},{sensing.right} "
          f"推定={est} 期待={expected_e*1000:+.3f}mm")

    assert est is not None, "両側とも AMBIGUOUS/CLEAR になり推定できなかった（振り幅が較正前提を超えている）"
    err_mm = (est.offset_m - expected_e) * 1000
    assert abs(err_mm) < 1.0, f"横ずれ推定の誤差が 1mm を超えた: {err_mm:+.3f}mm (source={est.source})"


def test_estimate_lateral_offset_returns_none_when_neither_side_is_confidently_a_wall():
    """AMBIGUOUS・CLEAR な読みからは補正しない（曖昧な読みで位置を動かさない
    ことの検査。純粋なユニットテストとして `WallSensing` を直接組み立てる）。"""
    params = RobotParams()
    both_clear = WallSensing(front=SenseWallState.CLEAR, left=SenseWallState.CLEAR, right=SenseWallState.CLEAR,
                              front_dist=0.25, left_dist=0.20, right_dist=0.20)
    both_ambiguous = WallSensing(front=SenseWallState.CLEAR, left=SenseWallState.AMBIGUOUS,
                                  right=SenseWallState.AMBIGUOUS, front_dist=0.25, left_dist=0.083, right_dist=0.083)
    mixed = WallSensing(front=SenseWallState.CLEAR, left=SenseWallState.AMBIGUOUS, right=SenseWallState.CLEAR,
                         front_dist=0.25, left_dist=0.083, right_dist=0.20)

    assert estimate_lateral_offset(both_clear, params) is None
    assert estimate_lateral_offset(both_ambiguous, params) is None
    assert estimate_lateral_offset(mixed, params) is None


def test_estimate_forward_offset_returns_none_unless_front_is_a_wall():
    """前後位置補正も同じ方針: 前方が WALL と確定していない限り推定しない。"""
    params = RobotParams()
    front_clear = WallSensing(front=SenseWallState.CLEAR, left=SenseWallState.WALL, right=SenseWallState.WALL,
                               front_dist=0.25, left_dist=0.06, right_dist=0.06)
    front_ambiguous = WallSensing(front=SenseWallState.AMBIGUOUS, left=SenseWallState.WALL, right=SenseWallState.WALL,
                                   front_dist=0.10, left_dist=0.06, right_dist=0.06)
    front_wall = WallSensing(front=SenseWallState.WALL, left=SenseWallState.WALL, right=SenseWallState.WALL,
                              front_dist=0.05, left_dist=0.06, right_dist=0.06)

    assert estimate_forward_offset(front_clear, params) is None
    assert estimate_forward_offset(front_ambiguous, params) is None
    assert estimate_forward_offset(front_wall, params) is not None


# ==========================================================================
# 2. 無効化 (enabled=False) は完全な no-op であること
# ==========================================================================
def test_localizer_disabled_never_calls_sense_walls(monkeypatch, params):
    """無効化されている間は `sense_walls` すら呼ばない（`correct_forward_remaining`
    が計算コストゼロで即座に無補正の値を返すこと。呼んだら例外を送出する
    ダミーに差し替えて確認する）。"""
    def _boom(obs, params=None):
        raise AssertionError("Localizer(enabled=False) が sense_walls を呼んだ（無効化が徹底されていない）")

    monkeypatch.setattr(localization_module, "sense_walls", _boom)
    loc = Localizer(params, enabled=False)
    remaining = loc.correct_forward_remaining(0.123, obs=np.zeros(len(params.sensors) + 8))
    assert remaining == 0.123
    assert loc.events == []


def test_localizer_enabled_lateral_bias_is_zero_for_ambiguous_readings(params):
    """曖昧な読みで補正しないことの検査（AMBIGUOUS を渡しても位置が動かない）。
    `enabled=True`（S2 有効）でも、両側とも AMBIGUOUS/CLEAR の読みでは
    バイアスは常に 0.0（=`_target_heading` を一切動かさない）。"""
    loc = Localizer(params, enabled=True)
    ambiguous = WallSensing(front=SenseWallState.CLEAR, left=SenseWallState.AMBIGUOUS,
                             right=SenseWallState.AMBIGUOUS, front_dist=0.25, left_dist=0.083, right_dist=0.083)
    bias = loc.lateral_bias_for_forward(ambiguous, cell=(1, 1), heading=0)
    assert bias == 0.0
    assert loc.events == []


def test_localizer_disabled_lateral_bias_is_always_zero_even_with_confident_walls(params):
    """無効化時は、両側とも確度の高い WALL 判定があっても横位置バイアスは
    常に 0.0（=推定すら行わない）。"""
    loc = Localizer(params, enabled=False)
    sensing = WallSensing(front=SenseWallState.CLEAR, left=SenseWallState.WALL, right=SenseWallState.WALL,
                           front_dist=0.25, left_dist=0.03, right_dist=0.12)  # 明らかに横ずれがある読み
    bias = loc.lateral_bias_for_forward(sensing, cell=(1, 1), heading=0)
    assert bias == 0.0
    assert loc.events == []


def test_cellmotioncontroller_forward_is_bit_identical_with_localizer_none_vs_disabled(tmp_path, params):
    """🔴 設計要求: 無効化したら S1（S2 導入前の推測航法のみの経路）と
    bit 一致するはず。`classic/motion.py` は `localizer=None`（S2 導入前の
    既定そのもの）と `localizer=Localizer(enabled=False)` の 2 通りで
    完全に同じコード経路を通るよう実装してある（if self.localizer is not
    None: の分岐そのものを通らない）。実際に同一のシミュレーション条件で
    2 区画前進させ、車輪電圧の列が bit 一致することを実測で確認する。"""
    W, H = 5, 5
    v = np.zeros((W + 1, H), dtype=int)
    v[0, :] = 1
    v[W, :] = 1
    h = np.zeros((W, H + 1), dtype=int)
    h[:, 0] = 1
    h[:, H] = 1

    def fresh_sim(label):
        xml_path = os.path.join(str(tmp_path), f"open_{label}.xml")
        build_maze_robot_xml(v, h, xml_path, model_name=f"open_{label}", params=params)
        sim = MouseSim(xml_path, params=params)
        sim.full_reset(cell=(2, 2), heading_deg=90.0)
        return sim

    def drive(localizer):
        sim = fresh_sim(f"case_{localizer}")
        ctrl = CellMotionController(params, localizer=localizer)
        ctrl.reset(heading_deg=90.0)
        ctrl.start_forward(2)
        voltages = []
        for _ in range(MAX_STEPS):
            obs = sim.observation()
            vl, vr, done = ctrl.update(obs)
            voltages.append((vl, vr))
            sim.step_control(vl, vr)
            if done:
                break
        else:
            raise AssertionError("収束しなかった")
        return tuple(voltages)

    voltages_none = drive(None)
    voltages_disabled = drive(Localizer(params, enabled=False))

    print(f"\n[実測] localizer=None: {len(voltages_none)}ステップ / "
          f"Localizer(enabled=False): {len(voltages_disabled)}ステップ")
    assert voltages_none == voltages_disabled, "無効化した Localizer が、localizer=None(S2導入前)と bit 一致しない"


# ==========================================================================
# 3. 実行経路検査（型 B 再発防止）
# ==========================================================================
def test_issue_forward_is_classicexplorers_own_implementation():
    assert_same_callable(ClassicExplorer, "_issue_forward", ClassicExplorer)


def test_on_stationary_is_classicexplorers_own_implementation():
    assert_same_callable(ClassicExplorer, "_on_stationary", ClassicExplorer)


# ==========================================================================
# 4. 否定対照 N1/N2: 補正ありの探索方策が真値を一切使っていないことの実測
#    （classic/checks.negative_control で対にして書く。作法は
#    tests/test_classic_motion.py・tests/test_classic_policy.py と同じ）
# ==========================================================================
def _small_turning_maze():
    width, height = 6, 6
    v_walls = np.zeros((width + 1, height), dtype=int)
    v_walls[0, :] = 1
    v_walls[width, :] = 1
    h_walls = np.zeros((width, height + 1), dtype=int)
    h_walls[:, 0] = 1
    h_walls[:, height] = 1
    v_walls[1, 0] = 1
    h_walls[2, 2] = 1
    v_walls[3, 3] = 1
    return v_walls, h_walls, width, height


N1N2_STEPS = 3000


def _drive_explorer_collect_voltages(tmp_path, params, label, corrupt, localization_enabled=True):
    v_walls, h_walls, width, height = _small_turning_maze()
    xml_path = str(tmp_path / f"maze_{label}.xml")
    build_maze_robot_xml(v_walls, h_walls, xml_path, model_name=f"m_{label}", params=params)
    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 0), heading_deg=90.0)
    if corrupt:
        sim.privileged_pose = lambda: (9999.0, -9999.0, 12.345)
        sim.privileged_velocity = lambda: (9999.0, -9999.0)

    explorer = ClassicExplorer(width, height, params=params, localization_enabled=localization_enabled)
    voltages = []
    for _ in range(N1N2_STEPS):
        obs = sim.observation()
        vl, vr, _plan_id = explorer.tick(obs)
        voltages.append((vl, vr))
        sim.step_control(vl, vr)
    return tuple(voltages)


def _drive_dummy_privileged_controller(sim, n_steps=N1N2_STEPS):
    """N2（対照）: 真値を露骨に使う単純なダミー制御（既存テストと同じ作法）。"""
    voltages = []
    for _ in range(n_steps):
        x, _y, _yaw = sim.privileged_pose()
        vl = float(np.clip(x, -1.0, 1.0))
        vr = 0.0
        voltages.append((vl, vr))
        sim.step_control(vl, vr)
    return tuple(voltages)


def test_localization_enabled_explorer_does_not_use_privileged_information(tmp_path):
    """N1: `localization_enabled=True`（S2 有効。既定）の `ClassicExplorer` は、
        `sim.privileged_pose()/privileged_velocity()` を壊しても電圧列が
        bit 一致する。
    N2: 真値を使うダミー制御には同じ壊し方で必ず変化が出る（空振り防止）。

    `classic/localization.py` は `WallSensing`/`RobotParams` だけを受け取り、
    `sim` そのものへは触れない設計である（モジュール docstring 参照）。
    これを実測で裏付ける。"""
    params = RobotParams()

    def run_under_test(broken: bool):
        return _drive_explorer_collect_voltages(tmp_path, params, f"ut_{broken}", corrupt=broken,
                                                  localization_enabled=True)

    def run_control(broken: bool):
        v_walls, h_walls, _w, _h = _small_turning_maze()
        xml_path = str(tmp_path / f"maze_ctrl_{broken}.xml")
        build_maze_robot_xml(v_walls, h_walls, xml_path, model_name=f"m_ctrl_{broken}", params=params)
        sim = MouseSim(xml_path, params=params)
        sim.full_reset(cell=(0, 0), heading_deg=90.0)
        if broken:
            sim.privileged_pose = lambda: (9999.0, -9999.0, 12.345)
        return _drive_dummy_privileged_controller(sim)

    got = negative_control(run_under_test=run_under_test, run_control=run_control)
    print(f"\n[実測] 否定対照(localization有効時の privileged_pose/velocity): {got.verdict}")
    assert got.passed, got.verdict


# ==========================================================================
# 5. ドリフトの実測比較（本タスクの核心）
# ==========================================================================
# 10 区画の連続直進の**前**に、その場旋回を実際に 5 回（450°=90°、正味の
# 向きは変わらない）実行させ、`classic/motion.py` docstring に実測されている
# 現実的な旋回誤差（1回あたり ±0.44〜0.48°）を推測航法へ乗せてから直進させる。
# maze_41038 で実際に観測された「10 区画の連続直進で 26〜28mm のドリフト」は、
# 直前の旋回の小さな誤差がそのまま長い直進の間、一定の目標方位として保持され
# 続けることで生じる（`classic/motion.py` docstring 「外乱の無い対称な
# シミュレータでは…曲がりようがない」を踏まえると、無外乱の直進**単体**では
# ドリフトは生じない。ドリフトの発生源は直進の**前**にある）。
# 本テストは maze_41038 の記録そのものの再生ではなく、**同じ機構**
# （実際の旋回で生じる現実的な誤差 → その後の長い直進でそのまま横ドリフトに
# なる）を、既定ゲインの `CellMotionController` を使って再現し、実測する。
SEED_TURNS = ["start_turn_left"] * 5  # 450°=90°。5 回分の実測旋回誤差を乗せる
N_DRIFT_CELLS = 10


def _drift_experiment(tmp_path, params, localization_enabled: bool):
    width, height = 1, N_DRIFT_CELLS + 2
    v, h = _corridor_walls(width, height)
    xml_path = os.path.join(str(tmp_path), f"drift_{localization_enabled}.xml")
    build_maze_robot_xml(v, h, xml_path, model_name=f"drift_{localization_enabled}", params=params)
    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 1), heading_deg=0.0)

    localizer = Localizer(params, enabled=localization_enabled)
    ctrl = CellMotionController(params, localizer=localizer)
    ctrl.reset(heading_deg=0.0)

    for turn_method in SEED_TURNS:
        getattr(ctrl, turn_method)()
        _run_until_done(sim, ctrl)

    x0, y0, _yaw0 = sim.privileged_pose()  # 答え合わせにのみ真値を使う

    for i in range(N_DRIFT_CELLS):
        obs = sim.observation()
        sensing = sense_walls(obs, params)
        ctrl.start_forward(1)
        bias = localizer.lateral_bias_for_forward(sensing, cell=(0, i), heading=0)
        if bias != 0.0:
            ctrl.bias_target_heading(bias)
        _run_until_done(sim, ctrl)

    x1, y1, _yaw1 = sim.privileged_pose()
    lateral_drift_m = x1 - x0  # heading=北のとき、横ドリフトは世界 x 方向
    forward_travel_m = y1 - y0
    return lateral_drift_m, forward_travel_m, localizer


def test_lateral_correction_significantly_reduces_ten_cell_straight_drift(tmp_path, params):
    """🔴 本タスクの核心。maze_41038 で実測された「10 区画の連続直進で
    26〜28mm の横ドリフト」と同じ機構（旋回の実測誤差→長い直進でそのまま
    ドリフト）を再現し、S2 の横位置補正で**有意に減る**ことを実測する。
    減らないなら、それを正直に報告する（任務指示）。"""
    drift_off, fwd_off, loc_off = _drift_experiment(tmp_path, params, localization_enabled=False)
    drift_on, fwd_on, loc_on = _drift_experiment(tmp_path, params, localization_enabled=True)

    report = (
        f"\n[実測] 10区画連続直進のドリフト比較（旋回誤差の種:{SEED_TURNS}）\n"
        f"  補正なし: 横ドリフト={drift_off*1000:+.3f}mm 前進={fwd_off*1000:.2f}mm "
        f"(補正イベント lateral={sum(1 for e in loc_off.events if e.axis=='lateral')})\n"
        f"  補正あり: 横ドリフト={drift_on*1000:+.3f}mm 前進={fwd_on*1000:.2f}mm\n"
        f"  {loc_on.summary()}\n"
        f"  ドリフト削減率: {(1 - abs(drift_on) / abs(drift_off)) * 100:.1f}%"
    )
    print(report)

    # 前提: 問題が実際に再現できていること（そうでなければ「効いた」とは
    # 主張できない）。maze_41038 の 26〜28mm 台には及ばなくとも、無視できない
    # 規模のドリフトが出ていることを最低条件として確認する。
    assert abs(drift_off) > 0.010, (
        "補正なしのドリフトが 10mm 未満だった。maze_41038 の問題を再現できて"
        "いない疑いがあり、これでは補正の効果を主張できない。\n" + report
    )
    # 本題: 補正ありのドリフトが補正なしより有意に(半分以下に)小さいこと。
    assert abs(drift_on) < abs(drift_off) * 0.5, (
        "横位置補正を有効にしても、10区画直進後のドリフトが有意に減らなかった。\n" + report
    )


def test_disabling_localization_makes_the_drift_reduction_check_fail(tmp_path, params):
    """🔴 空振り防止: 上のテストが求める閾値（補正なしの半分未満）を、
    「補正なし」の測定値どうしに当てはめると必ず落ちることを実測する
    （＝上のテストが本当に補正の効果を見ていて、無条件に通る形になって
    いないことの確認）。"""
    drift_a, _fwd_a, _loc_a = _drift_experiment(tmp_path, params, localization_enabled=False)
    drift_b, _fwd_b, _loc_b = _drift_experiment(tmp_path, params, localization_enabled=False)
    print(f"\n[実測] 補正なし同士の再現性: {drift_a*1000:+.3f}mm vs {drift_b*1000:+.3f}mm")
    # 決定論的なシミュレーションなので同一条件なら本来 bit 一致するはずだが、
    # ここで確認したいのは「補正なしを補正なしと比べても閾値は満たさない」
    # ことなので、緩い等号で十分（同一条件なので実際には一致する）。
    assert not (abs(drift_b) < abs(drift_a) * 0.5), (
        "補正なし同士の比較で『半分未満』条件が満たされてしまった。"
        "上のテストの閾値が意味を持たない（=空振り）疑いがある。"
    )


# ==========================================================================
# 6. 到達率の再測定（design_v4 の同じ3面、補正あり/なし）
# ==========================================================================
# `classic/policy.py` は「触ってよいファイル」に含まれないため
# （`ClassicExplorerPolicy` に localization_enabled を渡す口が無い）、
# ここでは同一構造の最小限のラッパをテスト内に用意する（本体は
# classic/policy.py の ClassicExplorerPolicy と同一。localization_enabled を
# 明示的に切り替えられる点だけが異なる）。
class _ToggleableExplorerPolicy(MousePolicy):
    name = "classic_explorer_toggleable"
    requires_privileged = False

    def __init__(self, params=None, localization_enabled: bool = True) -> None:
        self.params = params if params is not None else RobotParams()
        self.localization_enabled = localization_enabled
        self._explorer = None
        self._plan_ids: List[str] = []

    def on_maze_start(self, maze_info: dict) -> None:
        width = int(maze_info["width"])
        height = int(maze_info["height"])
        self._explorer = ClassicExplorer(width, height, params=self.params,
                                          localization_enabled=self.localization_enabled)
        self._plan_ids = []

    def on_retrieval(self) -> None:
        if self._explorer is not None:
            self._explorer.handle_retrieval()

    def act(self, obs: np.ndarray):
        vl, vr, plan_id = self._explorer.tick(obs)
        self._plan_ids.append(plan_id)
        return vl, vr

    def get_plan_ids(self) -> List[str]:
        return list(self._plan_ids)


MAZE_IDS = ["42134", "41038", "41049"]
# tests/test_classic_policy.py と同じ理由（S1 でも 1 面あたり 200〜400秒
# 程度を要することが実測済み）で、機構の完走を見るのに十分な時間を与える。
TEST_TIME_BUDGET_S = 900.0
MAX_RUNS = 5


@pytest.fixture(scope="module")
def evaluator() -> CompetitionEvaluator:
    return CompetitionEvaluator(
        maze_dir=str(DESIGN_V4_DIR),
        time_budget=TEST_TIME_BUDGET_S,
        max_runs=MAX_RUNS,
    )


@pytest.fixture(scope="module")
def reach_comparison(evaluator):
    """3 面 x (補正あり/なし) を実際に CompetitionEvaluator で走らせ、
    結果をまとめて返す（走行のたびに再実行すると時間がかかりすぎるため
    module スコープで 1 回だけ実行する）。"""
    results = {}
    for localization_enabled in (True, False):
        for maze_id in MAZE_IDS:
            policy = _ToggleableExplorerPolicy(localization_enabled=localization_enabled)
            npz_path = DESIGN_V4_DIR / f"maze_{maze_id}.npz"
            result = evaluator.evaluate_maze(npz_path, policy)
            results[(localization_enabled, maze_id)] = result
    return results


def test_localization_correction_goal_reach_comparison_on_design_v4(reach_comparison):
    """🔴 任務指示 4「到達率の再測定」。同じ 3 面で、補正あり/なしのゴール
    到達数を比較する。**到達率が上がらなかった場合は、上がらなかった事実を
    正直に報告する**（検査を緩めたり真値を使ったりして無理に通さない。
    そのため本テストは「上がるはず」という強い主張はしない。実測した事実
    ——両構成のうち少なくとも一方は実際にゴールへ到達できる——だけを
    機械の生存確認として表明し、残りは実測値の報告に留める）。"""
    lines = []
    n_reached = {True: 0, False: 0}
    for localization_enabled in (True, False):
        for maze_id in MAZE_IDS:
            result = reach_comparison[(localization_enabled, maze_id)]
            reached = bool(result["kpi"]["goal_reached"])
            n_reached[localization_enabled] += int(reached)
            outcomes = [r["outcome"] for r in result["runs"]]
            lines.append(
                f"  localization_enabled={localization_enabled} maze={maze_id}: "
                f"goal_reached={reached} run_outcomes={outcomes}"
            )
    report = (
        "迷路ごとのゴール到達（補正あり/なし比較）:\n" + "\n".join(lines) +
        f"\n合計到達数: 補正あり={n_reached[True]}/3 補正なし={n_reached[False]}/3"
    )
    print("\n[実測]\n" + report)

    assert n_reached[True] + n_reached[False] >= 1, (
        "補正あり/なしのどちらでも 1 面もゴールへ到達しなかった（機構そのものが機能していない疑い）。\n" + report
    )
