"""
tests/test_obs_model.py
================
`classic/obs_model.py`（任意の姿勢で成り立つ距離センサ観測モデル、exp_033）の検査。

方針（タスク指示どおり）: 「〜のはず」で検査を書かない。まず実際に
`predict_ranges()` を計算して値を見てから、その値を固定する（本ファイルの
数値はすべて 2026-08-23 に実際に計算した値）。すべての検査で MuJoCo
（`mouse.sim.MouseSim` の rangefinder 実測）とも突き合わせ、予測値が
シミュレータの実測値と一致することを確認する（`tests/test_classic_localization.py`
と同じ作法: `build_maze_robot_xml` で迷路 XML を作り、`data.qpos` を直接書いて
`mujoco.mj_forward` で任意姿勢を反映する）。
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import mujoco

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

from classic.maze_map import MazeMap, WallState
from classic.obs_model import (
    SEARCH_MARGIN_M,
    _cells_radius,
    _local_obstacles,
    _ray_rect_hit,
    _sensor_local_ray,
    max_candidates_bound,
    predict_ranges,
    predict_ranges_with_diagnostics,
)

ATOL = 1e-6  # MuJoCo の rangefinder は幾何の光線追跡そのものなので、機械精度に近い一致を要求する


@pytest.fixture(scope="module")
def params():
    return RobotParams()


def _build_sim(v_walls, h_walls, tmp_path, label, params) -> MouseSim:
    xml_path = os.path.join(str(tmp_path), f"maze_{label}.xml")
    build_maze_robot_xml(v_walls, h_walls, xml_path, model_name=f"m_{label}",
                          center_goal=False, params=params)
    return MouseSim(xml_path, params=params)


def _set_pose(sim: MouseSim, x: float, y: float, theta_rad: float) -> None:
    """任意姿勢を `data.qpos` へ直接書いて `mj_forward` で反映する
    （`tests/test_classic_localization.py::_set_pose` と同じ作法。
    本タスクで「任意の姿勢をシミュレータに取らせる」ために採用した方法そのもの）。"""
    qpos_adr = sim.model.jnt_qposadr[sim._root_joint_id]
    sim.data.qpos[qpos_adr:qpos_adr + 3] = [x, y, 0.002]
    sim.data.qpos[qpos_adr + 3] = math.cos(theta_rad / 2.0)
    sim.data.qpos[qpos_adr + 4] = 0.0
    sim.data.qpos[qpos_adr + 5] = 0.0
    sim.data.qpos[qpos_adr + 6] = math.sin(theta_rad / 2.0)
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def _walled_cell(width: int, height: int) -> "tuple[np.ndarray, np.ndarray]":
    """width x height の全区画・全壁を「壁あり」にした迷路。"""
    v = np.ones((width + 1, height), dtype=np.uint8)
    h = np.ones((width, height + 1), dtype=np.uint8)
    return v, h


def _open_room(width: int, height: int) -> "tuple[np.ndarray, np.ndarray]":
    """外周だけ壁、内部は壁なし（柱だけが残る）の迷路。"""
    v = np.zeros((width + 1, height), dtype=np.uint8)
    h = np.zeros((width, height + 1), dtype=np.uint8)
    v[0, :] = 1
    v[width, :] = 1
    h[:, 0] = 1
    h[:, height] = 1
    return v, h


# ==========================================================================
# 1. 区画の中心・北向き、四方が壁（手計算と照合）
# ==========================================================================
def test_center_north_four_walls_matches_hand_calc_and_sim(tmp_path, params):
    """1x1（四方すべて壁）の区画中心 (0.09, 0.09)・北向き (theta=90°)。

    手計算（LF センサ、`mouse/params.py` の pos='0.031 0.036 0.01'・zaxis='1 0 0.026'）:
      1. 光軸の正規化: norm = sqrt(1^2+0^2+0.026^2) = 1.0003379...
         単位ベクトル ux=0.99966217, uy=0, uz=0.02599122
      2. センサ site（先端）は基部から光軸方向へ 2mm 先
         （`mouse/robot_v2.py:191-195`）:
         ox_local = 0.031 + 0.002*0.99966217 = 0.03299932
         oy_local = 0.036 + 0.002*0 = 0.036
      3. 水平方向の単位ベクトル: h = sqrt(ux^2+uy^2) = 0.99966217 (=cos(仰角))
         dx_local = ux/h = 1.0, dy_local = 0
      4. 北向き(theta=90°)なので機体座標を90°回転: 世界原点 (0.09-0.036, 0.09+0.03299932)
         = (0.054, 0.12299932)。世界方向 (0,1)（真北）
      5. 北壁の南面（近い面）は y=0.18-0.006=0.174 にある
         （区画 (0,0) の北壁 `h_walls[0,1]`、`classic/geometry.py::wall_obstacles` の
         半長 wall_thickness/2=0.006 を中心 y=1*0.18=0.18 から引いた値）。
         水平距離 = 0.174 - 0.12299932 = 0.05100068
      6. 光線に沿った距離 = 水平距離 / h = 0.05100068 / 0.99966217 = 0.05101791

    この 0.05101791 が実際に `predict_ranges` の出力（LF）と一致することを固定する。
    """
    v, h = _walled_cell(1, 1)
    pose = (0.09, 0.09, math.radians(90.0))

    pred = predict_ranges(pose, (v, h), params)

    # 手計算（上記コメントの step 6）。RF も LF と同じ x 位置・同じ光軸で左右対称なので同値。
    expected_lf_rf = 0.05101791097355683
    expected_ls_rs = 0.06811508984519667
    assert pred == pytest.approx(
        [expected_lf_rf, expected_ls_rs, expected_lf_rf, expected_ls_rs], abs=ATOL
    )

    sim = _build_sim(v, h, tmp_path, "t1", params)
    _set_pose(sim, *pose)
    actual = sim.observation()[:4]
    assert pred == pytest.approx(actual, abs=ATOL)


# ==========================================================================
# 2. 45° を向いたとき（斜め探索の前提）
# ==========================================================================
def test_45_degrees_matches_sim(tmp_path, params):
    """3x3（全区画・全壁あり）の中央区画 (1,1) の中心 (0.27, 0.27)・45° 向き。

    軸並行の特殊化では通らない検査（`_sensor_local_ray` の回転が正しく
    45° でも効いていることを固定する）。"""
    v, h = _walled_cell(3, 3)
    pose = (0.27, 0.27, math.radians(45.0))

    pred = predict_ranges(pose, (v, h), params)

    expected = [0.04981144263322901, 0.04732302849370059,
                0.049811442633229, 0.04732302849370059]
    assert pred == pytest.approx(expected, abs=ATOL)

    sim = _build_sim(v, h, tmp_path, "t2", params)
    _set_pose(sim, *pose)
    actual = sim.observation()[:4]
    assert pred == pytest.approx(actual, abs=ATOL)


# ==========================================================================
# 3. 柱だけが見える姿勢（柱を無視すると値が変わることを固定する）
# ==========================================================================
def test_post_only_visibility_changes_when_posts_ignored(tmp_path, params):
    """5x5・外周だけ壁（内部は柱のみ）。区画 (2,2) の中心よりやや外れた位置・
    300° 向きにすると、内部の柱 1 本だけが LF の視界に入る（他の 3 本は cutoff）。

    `include_posts=False`（否定対照 N3 の実装そのもの）にすると、この柱を
    無視するため LF も cutoff (0.3) に変わる。柱を面として扱う必要が実測で
    確かめられることを、この差分で固定する。"""
    v, h = _open_room(5, 5)
    pose = (0.45, 0.45, math.radians(300.0))

    pred_with_posts = predict_ranges(pose, (v, h), params)
    pred_without_posts = predict_ranges(pose, (v, h), params, include_posts=False)

    expected_with = [0.08480878141510356, 0.3, 0.3, 0.3]
    expected_without = [0.3, 0.3, 0.3, 0.3]
    assert pred_with_posts == pytest.approx(expected_with, abs=ATOL)
    assert pred_without_posts == pytest.approx(expected_without, abs=ATOL)
    # 柱を無視すると LF の予測が real な柱の実測から大きく外れる（差 > 0.2m）。
    assert abs(pred_without_posts[0] - pred_with_posts[0]) > 0.2

    sim = _build_sim(v, h, tmp_path, "t3", params)
    _set_pose(sim, *pose)
    actual = sim.observation()[:4]
    # 実測（MuJoCo）は当然「柱がそこに実在する」世界なので、柱を数えた予測の方と一致する。
    assert pred_with_posts == pytest.approx(actual, abs=ATOL)


# ==========================================================================
# 4. 飽和（cutoff = 0.3 が返る）
# ==========================================================================
def test_saturation_returns_cutoff(tmp_path, params):
    """5x5・外周だけ壁の中央区画 (2,2) の中心・北向き。四方 2 区画以上先まで壁が無く、
    cutoff 0.3m 以内に何も交差物体が無いので 4 本とも 0.3 を返す。"""
    v, h = _open_room(5, 5)
    pose = (0.45, 0.45, math.radians(90.0))

    pred = predict_ranges(pose, (v, h), params)
    assert pred == pytest.approx([0.3, 0.3, 0.3, 0.3], abs=ATOL)

    sim = _build_sim(v, h, tmp_path, "t4", params)
    _set_pose(sim, *pose)
    actual = sim.observation()[:4]
    assert actual == pytest.approx([0.3, 0.3, 0.3, 0.3], abs=ATOL)


# ==========================================================================
# 5. 光線に沿った距離と水平距離の違い
# ==========================================================================
def test_ray_length_differs_from_horizontal_distance(params):
    """テスト1 (LF, center-north) の内部値を白箱で取り出し、
    「水平距離をそのまま返す（誤った）モデル」と「1/cos(仰角)で補正した
    正しいモデル」が異なることを固定する。差は事前調査どおり 0.3m 換算で
    約 0.1mm のオーダー（本ケースは距離が短い ~0.051m なので比例して約 17µm）。"""
    v, h = _walled_cell(1, 1)
    x, y, theta = 0.09, 0.09, math.radians(90.0)

    lf = params.sensors[0]
    assert lf['name'] == 'LF'
    ox_l, oy_l, dx_l, dy_l, elev_cos = _sensor_local_ray(lf['pos'], lf['zaxis'])

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    ox_w = x + ox_l * cos_t - oy_l * sin_t
    oy_w = y + ox_l * sin_t + oy_l * cos_t
    dx_w = dx_l * cos_t - dy_l * sin_t
    dy_w = dx_l * sin_t + dy_l * cos_t

    cells_r = _cells_radius(params.sensor_cutoff, params.cell_size)
    obstacles = _local_obstacles(v, h, ox_w, oy_w, params.cell_size, cells_r, center_goal=True)
    best_t = None
    for rect in obstacles:
        t = _ray_rect_hit(ox_w, oy_w, dx_w, dy_w, rect)
        if t is not None and (best_t is None or t < best_t):
            best_t = t
    assert best_t is not None

    horizontal_only = best_t          # 誤り: 水平距離をそのまま距離として使う
    ray_length = best_t / elev_cos    # 正しい: 光線に沿った距離（本モデルの式）

    assert horizontal_only == pytest.approx(0.05100067565746094, abs=ATOL)
    assert ray_length == pytest.approx(0.051017910973556825, abs=ATOL)
    # 光線に沿った距離は水平距離より長い（1/cos(仰角) > 1 なので）。
    assert ray_length > horizontal_only
    diff = ray_length - horizontal_only
    assert diff == pytest.approx(0.05100067565746094 * (1.0 / elev_cos - 1.0), abs=1e-9)
    # 事前調査の見積り（0.3m で約0.1mm）と桁が整合する: 0.051m では比例して約17µm。
    assert 1e-5 < diff < 3e-5

    # predict_ranges の出力が「正しい」ray_length の方と一致すること（水平距離ではない）。
    pred = predict_ranges((x, y, theta), (v, h), params)
    assert pred[0] == pytest.approx(ray_length, abs=ATOL)
    assert pred[0] != pytest.approx(horizontal_only, abs=1e-7)


# ==========================================================================
# 6. 候補として調べる面の数が上限を超えないこと
# ==========================================================================
@pytest.mark.parametrize("maze_and_pose", [
    ("walled_1x1", (1, 1), (0.09, 0.09, math.radians(90.0))),
    ("walled_3x3_45deg", (3, 3), (0.27, 0.27, math.radians(45.0))),
    ("open_5x5_post", (5, 5), (0.45, 0.45, math.radians(300.0))),
    ("open_5x5_sat", (5, 5), (0.45, 0.45, math.radians(90.0))),
])
def test_candidate_count_never_exceeds_bound(maze_and_pose, params):
    """`max_candidates_bound()` が返す理論上限（既定パラメータで 147）を、
    実際に調べた候補数（柱・縦壁・横壁の合計、`_local_obstacles` の長さ）が
    どの姿勢でも超えないことを固定する。"""
    label, (w, h_dim), pose = maze_and_pose
    if label.startswith("walled"):
        v, h = _walled_cell(w, h_dim)
    else:
        v, h = _open_room(w, h_dim)

    bound = max_candidates_bound(params)
    assert bound == 147  # 既定パラメータ（cutoff=0.3, cell_size=0.18）での理論値を固定する

    cells_r = _cells_radius(params.sensor_cutoff, params.cell_size)
    x, y, theta = pose
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    for s in params.sensors:
        ox_l, oy_l, _dx_l, _dy_l, _elev = _sensor_local_ray(s['pos'], s['zaxis'])
        ox_w = x + ox_l * cos_t - oy_l * sin_t
        oy_w = y + ox_l * sin_t + oy_l * cos_t
        obstacles = _local_obstacles(v, h, ox_w, oy_w, params.cell_size, cells_r, center_goal=True)
        assert len(obstacles) <= bound

    # predict_ranges_with_diagnostics が返す「実際にスラブ法を適用した候補数」
    # （近傍フィルタ後、_local_obstacles の候補数以下になるはず）も上限内。
    _, counts = predict_ranges_with_diagnostics(pose, (v, h), params)
    assert np.all(counts <= bound)


# ==========================================================================
# 補足: MazeMap（3値）入力と生の真値配列入力が同じ結果を返すこと
# ==========================================================================
def test_mazemap_input_matches_raw_array_input(params):
    v, h = _walled_cell(3, 3)
    pose = (0.27, 0.27, math.radians(45.0))

    mm = MazeMap(3, 3)
    mm.v_walls[:, :] = np.where(v != 0, int(WallState.WALL), int(WallState.OPEN))
    mm.h_walls[:, :] = np.where(h != 0, int(WallState.WALL), int(WallState.OPEN))

    pred_raw = predict_ranges(pose, (v, h), params)
    pred_mm = predict_ranges(pose, mm, params)
    assert pred_raw == pytest.approx(pred_mm, abs=1e-12)


# ==========================================================================
# 探索半径のマージンが cutoff ちょうどより広いこと（設計の前提の検査）
# ==========================================================================
def test_search_margin_is_positive():
    assert SEARCH_MARGIN_M > 0.0
