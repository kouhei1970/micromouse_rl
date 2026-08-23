"""
tests/test_wall_belief.py
================
`classic/wall_belief.py`（壁の有無の信念、exp_034）の検査。

書き方は `tests/test_classic_pose.py`・`tests/test_obs_model.py` に合わせる。
「〜のはず」で検査を書かず、まず実際に `WallBelief.update()` を走らせて値を見てから、
その値を固定する（本ファイルの数値はすべて 2026-08-23 に実際に計算した値）。
"""
from __future__ import annotations

import ast
import math
import os

import numpy as np
import pytest

from mouse.params import RobotParams

from classic.maze_map import Direction, WallState
from classic.obs_model import predict_ranges
from classic.wall_belief import (
    L_MAX_DEFAULT,
    R_MAX_DEFAULT,
    T_OPEN_DEFAULT,
    T_WALL_DEFAULT,
    WallBelief,
    declare_state,
    dequantize_log_odds,
    quantize_log_odds,
)


@pytest.fixture(scope="module")
def params():
    return RobotParams()


def _walled_cell(width: int, height: int) -> "tuple[np.ndarray, np.ndarray]":
    """width x height の全区画・全壁を「壁あり」にした迷路（test_obs_model.py と同じ）。"""
    v = np.ones((width + 1, height), dtype=np.uint8)
    h = np.ones((width, height + 1), dtype=np.uint8)
    return v, h


def _open_room(width: int, height: int) -> "tuple[np.ndarray, np.ndarray]":
    """外周だけ壁、内部は壁なしの迷路（test_obs_model.py と同じ）。"""
    v = np.zeros((width + 1, height), dtype=np.uint8)
    h = np.zeros((width, height + 1), dtype=np.uint8)
    v[0, :] = 1
    v[width, :] = 1
    h[:, 0] = 1
    h[:, height] = 1
    return v, h


# ==========================================================================
# 1. 真値 (privileged_pose/privileged_velocity・mouse.sim) を使っていないこと
# ==========================================================================
def test_source_never_accesses_privileged_attributes_or_imports_mouse_sim():
    """静的検査: `classic/wall_belief.py` の AST を走査し、`privileged_pose`／
    `privileged_velocity` への属性アクセスと `mouse.sim` の import が
    1 つも無いことを確認する（`tests/test_classic_pose.py:56` と同じ手法）。"""
    src_path = os.path.join(os.path.dirname(__file__), "..", "classic", "wall_belief.py")
    with open(src_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=src_path)

    forbidden = {"privileged_pose", "privileged_velocity"}
    hits = [node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in forbidden]
    assert hits == [], f"classic/wall_belief.py が真値の属性へアクセスしている: {hits}"

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    assert "mouse.sim" not in imported_modules, \
        "classic/wall_belief.py が mouse.sim を import している"


# ==========================================================================
# 2. 壁がある柱間を正しい姿勢から繰り返し観測すると、対数オッズが増えて WALL と宣言される
# ==========================================================================
def test_repeated_correct_observation_of_a_wall_raises_log_odds_to_wall(params):
    """3x3(全壁あり)・中央区画(1,1)の中心・北向き。北壁(h_walls[1,2])を狙って
    「真の壁配置」からの予測値をそのまま観測として与え続ける。

    [実測] 北向きでは LF・RF の両方がこの北壁を見ており、1 回の観測で
    log_odds が厳密に +1.0（= 2 センサ × R_MAX_DEFAULT=0.5、外れ値対策の
    上限に張り付くほど確信度が高い一致のため）増える。4 回で T_WALL_DEFAULT=2.0
    を超えて WALL と宣言される。
    """
    v, h = _walled_cell(3, 3)
    pose = (0.27, 0.27, math.radians(90.0))
    ranges = predict_ranges(pose, (v, h), params)

    wb = WallBelief(3, 3, params)
    cov = np.zeros((3, 3), dtype=np.float32)

    assert wb.declare_at(1, 1, Direction.N) == WallState.UNKNOWN
    for _ in range(4):
        wb.update(pose, cov, ranges)

    log_odds = wb.log_odds_at(1, 1, Direction.N)
    print(f"\n[実測] 4回観測後の北壁の対数オッズ: {log_odds}")
    assert log_odds == pytest.approx(4.0, abs=1e-3)
    assert wb.declare_at(1, 1, Direction.N) == WallState.WALL


# ==========================================================================
# 3. 矛盾する証拠が来たら信念が戻ること（規則2）
# ==========================================================================
def test_contradicting_evidence_makes_belief_decrease(params):
    """WALL 方向へ積んだ後、開通を示す観測（隣接壁を取り払った配置からの予測値）を
    与えると対数オッズが減ることを固定する。一度 WALL と書いても変えられない
    作りになっていないことの検査。

    [実測] 60 回の一致観測で L_MAX_DEFAULT(12.7) に飽和した状態から、
    開通を示す観測を 1 回与えると 1.0 減る（-1.0 = 2センサ×R_MAX_DEFAULT）。"""
    v, h = _walled_cell(3, 3)
    pose = (0.27, 0.27, math.radians(90.0))
    wall_ranges = predict_ranges(pose, (v, h), params)

    v_open, h_open = _open_room(3, 3)
    open_ranges = predict_ranges(pose, (v_open, h_open), params)

    wb = WallBelief(3, 3, params)
    cov = np.zeros((3, 3), dtype=np.float32)
    for _ in range(60):
        wb.update(pose, cov, wall_ranges)
    saturated = wb.log_odds_at(1, 1, Direction.N)
    print(f"\n[実測] 60回一致観測後(飽和): {saturated}")
    assert saturated == pytest.approx(L_MAX_DEFAULT, abs=1e-3)

    wb.update(pose, cov, open_ranges)
    after = wb.log_odds_at(1, 1, Direction.N)
    print(f"[実測] 矛盾する観測を1回与えた後: {after}")
    assert after < saturated, "矛盾する証拠を与えても対数オッズが減らない(規則2違反)"
    assert after == pytest.approx(saturated - 1.0, abs=1e-3)


# ==========================================================================
# 4. 対数オッズが L_max で頭打ちになること（規則3）
# ==========================================================================
def test_log_odds_saturates_at_l_max(params):
    """一致する観測をいくら重ねても対数オッズが L_MAX_DEFAULT を超えないこと。

    [実測] 1回の観測で+1.0ずつ増えるので13回でL_MAX_DEFAULT=12.7に到達し飽和する。
    30回与えても値は変わらない(12.7のまま)。"""
    v, h = _walled_cell(3, 3)
    pose = (0.27, 0.27, math.radians(90.0))
    ranges = predict_ranges(pose, (v, h), params)

    wb = WallBelief(3, 3, params)
    cov = np.zeros((3, 3), dtype=np.float32)
    for _ in range(30):
        wb.update(pose, cov, ranges)
        assert wb.log_odds_at(1, 1, Direction.N) <= L_MAX_DEFAULT + 1e-6

    final = wb.log_odds_at(1, 1, Direction.N)
    print(f"\n[実測] 30回観測後の対数オッズ: {final} (L_MAX_DEFAULT={L_MAX_DEFAULT})")
    assert final == pytest.approx(L_MAX_DEFAULT, abs=1e-3)


# ==========================================================================
# 5. 1回の観測の寄与が R_max で頭打ちになること（外れ値対策。規則1）
# ==========================================================================
def test_single_observation_contribution_is_capped_at_r_max(params):
    """真値から大きく外れた読みを 1 回与えても、対数オッズの変化量が
    一定の上限（2センサ分 = 2*R_MAX_DEFAULT）を超えないことを固定する。
    どれだけ外れていても同じ上限に張り付くことも確認する（外れ値の大きさに
    比例して信念が際限なく動くことがない）。

    [実測] 北壁の予測(0.0510m)から大きく外れた読み 0.15m・0.29m（いずれも
    cutoff=0.3m未満で打ち切り扱いにはならない）のどちらも、1回の観測での
    変化量は -1.0（= -2*R_MAX_DEFAULT）に張り付く。"""
    v, h = _walled_cell(3, 3)
    pose = (0.27, 0.27, math.radians(90.0))
    true_ranges = predict_ranges(pose, (v, h), params)
    cov = np.zeros((3, 3), dtype=np.float32)

    deltas = []
    for wrong in (0.15, 0.29):
        wb = WallBelief(3, 3, params)
        ranges = true_ranges.copy()
        ranges[0] = wrong  # LF
        ranges[2] = wrong  # RF（この姿勢で北壁を見ているのはLF・RFの2本）
        wb.update(pose, cov, ranges)
        delta = wb.log_odds_at(1, 1, Direction.N)
        print(f"\n[実測] wrong={wrong}m のときの1回の変化量: {delta}")
        deltas.append(delta)

    expected_ceiling = -2.0 * R_MAX_DEFAULT
    for d in deltas:
        assert d == pytest.approx(expected_ceiling, abs=1e-3), (
            f"1回の観測の変化量がR_maxの上限({expected_ceiling})に張り付いていない: {d}"
        )
    # 外れの大きさが違っても同じ上限に張り付く(外れ値の大きさに依存しない)。
    assert deltas[0] == pytest.approx(deltas[1], abs=1e-6)


# ==========================================================================
# 6. 姿勢の共分散を大きくすると、同じ観測の寄与が小さくなること（PREREG §1-1）
# ==========================================================================
def test_larger_pose_covariance_shrinks_the_contribution_of_the_same_observation(params):
    """同じ姿勢・同じ観測でも、共分散 P が大きいほど sigma^2 = sigma_sensor^2 + J^T P J
    が大きくなり、1回の対数尤度比の寄与（clip前）が小さくなることを固定する。

    [実測] var in {0.1, 1.0, 10.0} の等方共分散 diag(var,var,var) で、
    1回の観測後の対数オッズ(=寄与そのもの、初期値0から)が
    0.3236 -> 0.03236 -> 0.0032358 と単調に小さくなる
    （var=0.01以下ではR_max=0.5の上限に張り付いてしまい効果が見えないため、
    この検査ではクリップより十分小さい領域を選んでいる）。"""
    v, h = _walled_cell(3, 3)
    pose = (0.27, 0.27, math.radians(90.0))
    ranges = predict_ranges(pose, (v, h), params)

    contributions = []
    for var in (0.1, 1.0, 10.0):
        wb = WallBelief(3, 3, params)
        cov = np.diag([var, var, var]).astype(np.float32)
        wb.update(pose, cov, ranges)
        c = wb.log_odds_at(1, 1, Direction.N)
        print(f"\n[実測] var={var} のときの1回の寄与: {c}")
        contributions.append(c)

    assert contributions[0] < 2.0 * R_MAX_DEFAULT, "この検査はR_maxで頭打ちにならない領域で行う前提"
    assert contributions[0] > contributions[1] > contributions[2] > 0.0, (
        "共分散を大きくしても寄与が単調に小さくならない(PREREG §1-1違反)"
    )


# ==========================================================================
# 7. 宣言が非対称であること（WALLは低い確信度、OPENは高い確信度を要求する）
# ==========================================================================
def test_declare_is_asymmetric_between_wall_and_open():
    """同じ大きさの対数オッズでも、正側(WALL方向)なら宣言され、
    負側(OPEN方向)なら同じ大きさでは宣言されないことを固定する。

    [実測] T_WALL_DEFAULT=2.0 < T_OPEN_DEFAULT=8.0 なので、|l_w|=3.0 は
    WALL側では宣言されるが、OPEN側では宣言されない(UNKNOWNのまま)。"""
    magnitude = 3.0
    assert T_WALL_DEFAULT < magnitude < T_OPEN_DEFAULT

    assert declare_state(magnitude) == WallState.WALL
    assert declare_state(-magnitude) == WallState.UNKNOWN, (
        "同じ大きさの負の対数オッズでOPENが宣言されてしまった(非対称性が無い)"
    )
    # 開通の宣言には本当に高い確信度が要ることも確認する。
    assert declare_state(-(T_OPEN_DEFAULT + 0.1)) == WallState.OPEN


# ==========================================================================
# 8. 45° を向いた姿勢でも更新できること（斜め探索の前提）
# ==========================================================================
def test_update_works_at_45_degree_pose(params):
    """3x3(全壁あり)・中央区画の中心・45°向き。軸並行に特殊化していれば
    候補が見つからず何も更新されないはずだが、実際には更新されることを固定する。

    [実測] 45°姿勢で1回 update すると、少なくとも1つの柱間の対数オッズが
    0から変化する。"""
    v, h = _walled_cell(3, 3)
    pose = (0.27, 0.27, math.radians(45.0))
    ranges = predict_ranges(pose, (v, h), params)

    wb = WallBelief(3, 3, params)
    cov = np.zeros((3, 3), dtype=np.float32)
    before_v, before_h = wb.log_odds_v.copy(), wb.log_odds_h.copy()
    wb.update(pose, cov, ranges)

    changed = not (np.array_equal(before_v, wb.log_odds_v)
                   and np.array_equal(before_h, wb.log_odds_h))
    n_changed = (np.count_nonzero(wb.log_odds_v != before_v)
                 + np.count_nonzero(wb.log_odds_h != before_h))
    print(f"\n[実測] 45°姿勢での1回のupdateで変化した柱間の数: {n_changed}")
    assert changed, "45°姿勢でupdateしても対数オッズが一切変化しなかった(軸並行への特殊化を疑う)"


# ==========================================================================
# 9. 飽和した読み（0.3m）を渡しても例外にならず、意味のある更新になること
# ==========================================================================
def test_saturated_reading_does_not_raise_and_produces_meaningful_update(params):
    """5x5(外周のみ壁)・中央区画(2,2)の中心・北向き。4方向とも2区画以上先まで
    壁が無いので、実測は cutoff=0.3m ちょうど(飽和)になる。この読みを繰り返し
    与えても例外にならず、北壁方向の柱間が「開通」へ向かって動くことを固定する。

    [実測] 20回与えると対数オッズは -L_MAX_DEFAULT(-12.7) に飽和し、
    declareはOPENになる。"""
    v, h = _open_room(5, 5)
    pose = (0.45, 0.45, math.radians(90.0))
    ranges = predict_ranges(pose, (v, h), params)
    assert ranges[0] == pytest.approx(params.sensor_cutoff), "この検査の前提(飽和読み)が崩れている"

    wb = WallBelief(5, 5, params)
    cov = np.zeros((3, 3), dtype=np.float32)
    for _ in range(20):
        wb.update(pose, cov, ranges)  # 例外にならないこと自体がこの検査の一部

    log_odds = wb.log_odds_at(2, 2, Direction.N)
    print(f"\n[実測] 飽和読みを20回与えた後の北壁の対数オッズ: {log_odds}")
    assert log_odds == pytest.approx(-L_MAX_DEFAULT, abs=1e-3)
    assert wb.declare_at(2, 2, Direction.N) == WallState.OPEN


# ==========================================================================
# 10. 何も観測しなければ全柱間が UNKNOWN のままであること（外周を除く）
# ==========================================================================
def test_no_observation_leaves_all_interior_walls_unknown(params):
    wb = WallBelief(4, 5, params)
    mm = wb.to_maze_map()

    interior_v = mm.v_walls[1:4, :]
    interior_h = mm.h_walls[:, 1:5]
    assert np.all(interior_v == int(WallState.UNKNOWN))
    assert np.all(interior_h == int(WallState.UNKNOWN))

    # 外周は確定(WALL)として初期化されている(classic.maze_map.MazeMap と同じ規約)。
    assert np.all(mm.v_walls[0, :] == int(WallState.WALL))
    assert np.all(mm.v_walls[4, :] == int(WallState.WALL))
    assert np.all(mm.h_walls[:, 0] == int(WallState.WALL))
    assert np.all(mm.h_walls[:, 5] == int(WallState.WALL))


# ==========================================================================
# 11. 対数オッズを1バイトに量子化しても宣言が変わらないこと（マイコン実装の裏づけ）
# ==========================================================================
def test_quantizing_log_odds_to_int8_does_not_change_declaration(params):
    """観測をいくつか積んで多様な対数オッズ（正・負・0）を作った状態から、
    int8 量子化 -> 逆量子化した値で declare() した結果が、量子化前と一致することを固定する。"""
    v, h = _walled_cell(3, 3)
    pose = (0.27, 0.27, math.radians(45.0))
    ranges = predict_ranges(pose, (v, h), params)

    wb = WallBelief(3, 3, params)
    cov = np.zeros((3, 3), dtype=np.float32)
    for _ in range(15):
        wb.update(pose, cov, ranges)

    # 量子化前に declare() した結果(既定しきい値)。
    from classic.wall_belief import _declare_array
    before_v = _declare_array(wb.log_odds_v, T_WALL_DEFAULT, T_OPEN_DEFAULT)
    before_h = _declare_array(wb.log_odds_h, T_WALL_DEFAULT, T_OPEN_DEFAULT)

    qv = quantize_log_odds(wb.log_odds_v)
    qh = quantize_log_odds(wb.log_odds_h)
    assert qv.dtype == np.int8 and qh.dtype == np.int8
    assert np.all(np.abs(qv.astype(np.int64)) <= 127)
    assert np.all(np.abs(qh.astype(np.int64)) <= 127)

    dv = dequantize_log_odds(qv)
    dh = dequantize_log_odds(qh)
    after_v = _declare_array(dv, T_WALL_DEFAULT, T_OPEN_DEFAULT)
    after_h = _declare_array(dh, T_WALL_DEFAULT, T_OPEN_DEFAULT)

    print(f"\n[実測] 量子化前後で宣言が一致するか(v): {np.array_equal(before_v, after_v)}")
    print(f"[実測] 量子化前後で宣言が一致するか(h): {np.array_equal(before_h, after_h)}")
    assert np.array_equal(before_v, after_v), "量子化によってv_wallsの宣言が変わった"
    assert np.array_equal(before_h, after_h), "量子化によってh_wallsの宣言が変わった"

    # 少なくとも1か所はWALLと宣言されている(空振り防止: 全部UNKNOWNなら検査に意味が無い)。
    assert np.any(before_v == int(WallState.WALL)) or np.any(before_h == int(WallState.WALL))
