"""tests/test_profile.py — `classic/profile.py`（マウス本体の速度計画器）の検査

方針:

- テスト 1・4 は祖先スクリプト `research_notes/scripts/check_physical_limits_and_ideal_lap.py`
  を直接 import し、同じ入力から同じ値が出ることを確認する（合わせ込みではなく再現）。
- テスト 2（`test_recorded_inertia_matches_the_model`）は `classic/profile.py` が
  実行時に読まない `I_ZZ` / `I_EFF` / `R_CASTER` の 3 定数について、**このテストの中で**
  MuJoCo モデルから毎回合成し直し、定数と一致することを確認する。定数が実モデルから
  ずれたらこのテストが落ちる（`classic/` 本体は迷路 XML を読まないが、テストは読んでよい）。
- テスト 7 は `classic/checks.py` の `negative_control`（否定対照）を使い、
  「壊したら変わる」側と「無関係な計算は変わらない」側の両方を確かめる。
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research_notes" / "scripts"))
import check_physical_limits_and_ideal_lap as ancestor  # noqa: E402

from classic.checks import negative_control  # noqa: E402
from classic.profile import (  # noqa: E402
    I_EFF,
    I_ZZ,
    R_CASTER,
    Segment,
    lap_time,
    min_time,
    spin_turn_time,
    tangent_consumption,
    turn_segments,
    vehicle_limits,
)

MAZE_XML = str(
    Path(__file__).resolve().parent.parent
    / "competition" / "mazes" / "design_turn_v1" / "maze_41000.xml"
)


# ============================================================================
# 1. 車両の物理限界が祖先スクリプトと一致する
# ============================================================================
def test_vehicle_limits_match_the_ancestor_script():
    lim = vehicle_limits()
    for name in ("F_stall", "F_fric", "V_TOP", "A_TR", "A_LAT"):
        expected = getattr(ancestor, name)
        got = getattr(lim, name)
        assert math.isclose(got, expected, rel_tol=1e-12), (name, got, expected)


# ============================================================================
# 2. その場旋回の慣性定数（I_ZZ/I_EFF/R_CASTER）が実モデルと一致する
# ============================================================================
def _compute_yaw_inertia_from_model(xml_path: str):
    """`classic/profile.py` の docstring に書いた合成手順を、このテストの中で再実行する。

    `classic/` 本体は迷路 XML を実行時に読まない（壁の真値を含むため）。ここはテストなので読んでよい。
    """
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    mouse_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    assert mouse_id != -1

    # mouse ボディ以下の全サブツリー（inclusive）。
    # 注意: MuJoCo の world ボディ(id=0)は body_parentid[0]==0（自己ループ）であり、
    # -1 では終端しない。b を 0（world）まで遡っても mouse_id に届かなければ非子孫。
    subtree = []
    for bid in range(model.nbody):
        b = bid
        while b != 0 and b != mouse_id:
            b = model.body_parentid[b]
        if b == mouse_id:
            subtree.append(bid)

    total_mass = sum(model.body_mass[b] for b in subtree)
    com = np.zeros(3)
    for b in subtree:
        com += model.body_mass[b] * data.xipos[b]
    com /= total_mass

    izz = 0.0
    for b in subtree:
        rot = data.ximat[b].reshape(3, 3)
        i_body = np.diag(model.body_inertia[b])
        i_world = rot @ i_body @ rot.T
        dx = data.xipos[b][0] - com[0]
        dy = data.xipos[b][1] - com[1]
        izz += i_world[2, 2] + model.body_mass[b] * (dx ** 2 + dy ** 2)

    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wheel")
    spin_yy = 0.0
    for b in (left_id, right_id):
        rot = data.ximat[b].reshape(3, 3)
        i_body = np.diag(model.body_inertia[b])
        i_world = rot @ i_body @ rot.T
        spin_yy += i_world[1, 1]

    from mouse.params import RobotParams
    p = RobotParams()
    i_eff = izz + spin_yy * ((p.tread / 2) / p.wheel_radius) ** 2

    caster_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "caster_front")
    r_caster = float(model.geom_pos[caster_id][0])

    return izz, i_eff, r_caster


def test_recorded_inertia_matches_the_model():
    izz, i_eff, r_caster = _compute_yaw_inertia_from_model(MAZE_XML)
    assert izz == pytest.approx(I_ZZ, abs=1e-9)
    assert i_eff == pytest.approx(I_EFF, abs=1e-9)
    assert r_caster == pytest.approx(R_CASTER, abs=1e-9)


# ============================================================================
# 3. その場旋回時間
# ============================================================================
def test_spin_turn_90deg():
    lim = vehicle_limits()

    # キャスタ引きずりを含めない場合（教授セッションの手計算との照合）
    st_no_caster = spin_turn_time(math.pi / 2, lim, tau_caster=0.0)
    assert st_no_caster.regime == "triangular"
    assert st_no_caster.alpha == pytest.approx(225.2, rel=1e-3)
    assert st_no_caster.time == pytest.approx(0.1670, rel=1e-3)

    # 既定（キャスタ込み）。実行して得た値を固定する。
    st_default = spin_turn_time(math.pi / 2, lim)
    assert st_default.regime == "triangular"
    assert st_default.alpha == pytest.approx(210.15121388228903, rel=1e-9)
    assert st_default.time == pytest.approx(0.17291154157212957, rel=1e-9)
    # キャスタの引きずりで既定は無キャスタより遅くなる（トルクが減る分、加速が鈍る）
    assert st_default.time > st_no_caster.time


# ============================================================================
# 4. 祖先の周回ラップタイムを再現する（🔴 最重要の錨）
# ============================================================================
def _build_circuit(R: float, offset: float):
    straight = ancestor.SIDE - 2 * (R - offset)
    segs = []
    for _ in range(4):
        segs.append(Segment(length=straight, curvature=0.0, kind="straight"))
        segs.append(Segment(length=math.pi * R / 2, curvature=1.0 / R, kind="arc"))
    return segs


@pytest.mark.parametrize(
    "R, offset",
    [
        (0.060, 0.0),
        (0.090, 0.0),
        (ancestor.R_MAX, ancestor.W_C),
    ],
)
def test_reproduces_the_ancestor_circuit_lap(R, offset):
    lim = vehicle_limits()
    t_ancestor, _, _, path_len_ancestor, _, _ = ancestor.lap(R, offset)

    segs = _build_circuit(R, offset)
    it = lap_time(segs, lim)

    rel_err = abs(it.total - t_ancestor) / t_ancestor
    assert rel_err < 0.01, (R, offset, it.total, t_ancestor, rel_err)
    assert it.path_length == pytest.approx(path_len_ancestor, rel=1e-6)


# ============================================================================
# 5. v_cap を上げると総時間が単調に減る
# ============================================================================
def test_v_cap_monotonic():
    lim = vehicle_limits()
    segs = [Segment(length=1.0, curvature=0.0, kind="straight")]

    it_012 = min_time(segs, lim, v_start=0.0, v_end=0.0, v_cap=0.12)
    it_05 = min_time(segs, lim, v_start=0.0, v_end=0.0, v_cap=0.5)
    it_none = min_time(segs, lim, v_start=0.0, v_end=0.0, v_cap=None)

    assert it_012.total > it_05.total > it_none.total
    assert it_012.v_max <= 0.12 + 1e-9


# ============================================================================
# 6. 90° ターンは半径が大きいほど速い（同じ始終点）
# ============================================================================
def _ninety_turn_same_endpoints(radius: float, leg: float = 0.3):
    """半径 `radius` の 90° ターンを、始終点の位置を固定して組む。

    直線の長さから円弧が食う分（`tangent_consumption`）を差し引くことで、
    半径を変えても入口・出口の点が動かないようにする。
    """
    tc = tangent_consumption(math.pi / 2, radius)
    assert tc < leg, "leg が短すぎて円弧が直線を食い尽くす"
    lead = leg - tc
    trail = leg - tc
    segs = [Segment(length=lead, curvature=0.0, kind="straight")]
    segs += turn_segments(math.pi / 2, radius)
    segs += [Segment(length=trail, curvature=0.0, kind="straight")]
    return segs


def test_larger_radius_is_faster_for_a_90_degree_turn():
    lim = vehicle_limits()
    times = []
    for R in (0.06, 0.09, 0.30):
        segs = _ninety_turn_same_endpoints(R)
        it = min_time(segs, lim, v_start=0.0, v_end=0.0)
        times.append(it.total)
    assert times[0] > times[1] > times[2], times


# ============================================================================
# 7. 否定対照（classic/checks.py の negative_control）
# ============================================================================
def _straight_segments():
    return [Segment(length=1.0, curvature=0.0, kind="straight")]


def test_negative_control_a_tr_break_changes_min_time():
    """作動側: A_TR を半分に壊すと min_time の総時間が変わる。"""
    lim = vehicle_limits()
    segs = _straight_segments()

    def run_under_test(broken: bool) -> float:
        use = replace(lim, A_TR=lim.A_TR / 2) if broken else lim
        return min_time(segs, use, v_start=0.0, v_end=0.0).total

    def run_control(broken: bool) -> float:
        use = replace(lim, A_TR=lim.A_TR / 2) if broken else lim
        return spin_turn_time(math.pi / 2, use).time

    result = negative_control(
        run_under_test, run_control,
        equal=lambda a, b: math.isclose(a, b, rel_tol=1e-12),
    )
    assert result.changed_when_broken


def test_negative_control_a_tr_break_does_not_affect_spin_turn():
    """空振り側: その場旋回時間は A_TR に依存しないので、壊しても変わらない。"""
    lim = vehicle_limits()
    segs = _straight_segments()

    def run_under_test(broken: bool) -> float:
        use = replace(lim, A_TR=lim.A_TR / 2) if broken else lim
        return min_time(segs, use, v_start=0.0, v_end=0.0).total

    def run_control(broken: bool) -> float:
        use = replace(lim, A_TR=lim.A_TR / 2) if broken else lim
        return spin_turn_time(math.pi / 2, use).time

    result = negative_control(
        run_under_test, run_control,
        equal=lambda a, b: math.isclose(a, b, rel_tol=1e-12),
    )
    assert not result.changed_in_control


# ============================================================================
# 8. 短い直線は三角形、十分長い直線は台形になる
# ============================================================================
def _find_triangular_trapezoidal_boundary(lim, lo=0.01, hi=20.0, iters=60):
    """二分探索で「cruise_len が 0 から正になる」境界の直線長を求める。"""

    def cruise_len(length):
        it = min_time([Segment(length=length, curvature=0.0, kind="straight")],
                       lim, v_start=0.0, v_end=0.0)
        return it.segments[0].cruise_len

    assert cruise_len(lo) == 0.0
    assert cruise_len(hi) > 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cruise_len(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def test_short_straight_is_triangular_and_long_is_trapezoidal():
    lim = vehicle_limits()

    boundary = _find_triangular_trapezoidal_boundary(lim)
    print(f"\n[test_profile] 三角形/台形の境界となる直線長 ~= {boundary:.6f} m "
          f"(V_TOP={lim.V_TOP:.4f} m/s, A_TR={lim.A_TR:.4f} m/s^2 の単純計算 "
          f"V_TOP^2/A_TR={lim.V_TOP**2/lim.A_TR:.4f} m とは一致しない — "
          f"モータ出力が速度とともに落ちるため加速側は A_TR より緩い)")

    # 実行して得た値（.venv/bin/python -m pytest -s で印字を確認済み）。
    # ds=0.0005・_MODE_EPS=1e-9 に依存する経験的な値であり、閉形式の
    # V_TOP**2/A_TR (≈2.6 m) にはならない。
    assert boundary == pytest.approx(8.5445, abs=0.01)

    # 短い直線（1 区画 0.18 m）は三角形（定常速度に届かない）
    it_short = min_time([Segment(length=0.18, curvature=0.0, kind="straight")],
                         lim, v_start=0.0, v_end=0.0)
    sr_short = it_short.segments[0]
    assert sr_short.cruise_len == 0.0
    assert it_short.n_triangular == 1
    assert it_short.n_trapezoidal == 0
    assert not it_short.reached_v_top

    # 十分長い直線（境界の 3 倍）は台形（定常速度区間を持つ）
    long_len = boundary * 3.0
    it_long = min_time([Segment(length=long_len, curvature=0.0, kind="straight")],
                        lim, v_start=0.0, v_end=0.0)
    sr_long = it_long.segments[0]
    assert sr_long.cruise_len > 0.0
    # v_peak は V_TOP に漸近するが、モータ出力が速度とともに指数的に減衰するため
    # 有限長では厳密なビット一致にはならない（十分長くしても最後の数 ULP は残る）。
    # そのため近似一致で検査する。
    assert sr_long.v_peak == pytest.approx(lim.V_TOP, rel=1e-6)
    assert it_long.n_triangular == 0
    assert it_long.n_trapezoidal == 1
    assert it_long.reached_v_top


# ============================================================================
# 9. by_mode の合計が total / path_length に一致する
# ============================================================================
def test_by_mode_sums_to_total():
    lim = vehicle_limits()
    segs = _ninety_turn_same_endpoints(0.09)
    it = min_time(segs, lim, v_start=0.0, v_end=0.0)

    sum_t = sum(t for t, _l in it.by_mode.values())
    sum_l = sum(l for _t, l in it.by_mode.values())

    assert sum_t == pytest.approx(it.total, rel=1e-9)
    assert sum_l == pytest.approx(it.path_length, rel=1e-9)

    # 内訳が退化していない（3 モードとも登場する経路であること）ことも確認する
    assert it.by_mode["accel"][0] > 0.0
    assert it.by_mode["decel"][0] > 0.0
