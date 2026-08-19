"""
tests/test_classic_motion.py
================
classic/motion.py の検査。**実際にシミュレータを動かして**区画単位の動作
（直進 N 区画・その場旋回）が目標に収束することを確認する。

真値位置（privileged_pose/privileged_velocity）は検査の**答え合わせ**、
および否定対照（N1/N2）で「真値を壊しても走行が変わらない」ことを
実測するためだけに使う。

🔴 2026-08-19 是正（検分の変異検査が見つけた 7 件のうち、本ファイルに
関わる (2)(3)(5)(6)）: `classic/checks.py` の再発防止検査は実装した時点
では**どこからも呼ばれていなかった**（note_029 §4「登録簿は働かない。
実装された検査だけが働いた」がこの再構築の初日に再現していた）。
以下の検査は `classic.checks` の関数を実際に呼び出す。
"""
import math
import os

import numpy as np
import pytest

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

from classic import motion as motion_module
from classic.checks import assert_same_callable, negative_control
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
# 0-a. 実行経路の検査: 検査対象が実際に走る実装と同一であること
# ==========================================================================
def test_update_resolves_to_cellmotioncontrollers_own_implementation():
    """型 B（note_029 §4-2「実行経路そのものを検査する」・§4-3「是正は本体に
    入れる。別ファイル・別実装へ分岐させない」）是正 (3): 旧実装は独自の AST
    走査で「実行経路検査」を私家版として持っていた（`classic.checks` という
    正本があるのに使われていなかった＝登録簿のまま）。ここでは
    `classic.checks.assert_same_callable` 一本に統合し、このテストで実際に
    駆動される `update` が `CellMotionController` 自身の実装であって、
    将来 mixin 等へ分岐した写しではないことを機械的に保証する。"""
    assert_same_callable(CellMotionController, "update", CellMotionController)


# ==========================================================================
# 0-b. 否定対照 (N1/N2): privileged_pose を一切使っていないことを実測で示す
# ==========================================================================
def _drive_forward_collect_voltages(sim, params, corrupt_privileged=False):
    """CellMotionController に 2 区画前進させ、車輪電圧列を記録する。
    corrupt_privileged=True のとき、sim.privileged_pose() の戻り値を
    デタラメな値へ壊す（CellMotionController がそれを一切使わない設計
    ならば、電圧列は壊す前と bit 一致するはず）。"""
    if corrupt_privileged:
        sim.privileged_pose = lambda: (9999.0, -9999.0, 12.345)

    ctrl = CellMotionController(params)
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
        raise AssertionError(f"{MAX_STEPS} ステップ以内に収束しなかった")
    return tuple(voltages)


def _drive_dummy_privileged_controller(sim, n_steps=5):
    """N2（対照）: 真値 (privileged_pose) を実際に読んで動く、ごく単純な
    ダミー制御。x 座標をそのまま左車輪電圧に写すだけの、真値へ露骨に
    依存する構成（壊し方が効いていることを確認するための空振り防止）。"""
    voltages = []
    for _ in range(n_steps):
        x, _y, _yaw = sim.privileged_pose()
        vl = float(np.clip(x, -1.0, 1.0))
        vr = 0.0
        voltages.append((vl, vr))
        sim.step_control(vl, vr)
    return tuple(voltages)


def test_controller_does_not_use_privileged_pose(tmp_path, params):
    """是正 (2): 特権情報の否定対照（checks.py の型: 「―」＝「特権情報の
    残留は数値照合では検出できない」。note_029 §3「否定対照: 壊して変わら
    なければ、その経路は使われていない」を実装した検査）。
    tests/test_classic_checks.py 冒頭の説明どおり、数値がたまたま一致する
    ことは「真値を使っていない」証拠にならないため、ここでは
    classic.checks.negative_control を使って**壊して確かめる**。

    N1: CellMotionController を普通に走らせる構成と、sim.privileged_pose()
        の戻り値を意図的に壊した構成とで、車輪電圧の列が bit 一致すること
        （旧 AST 走査が保証していたのと同じ性質を、実測で確かめる）。
    N2: 真値を使うと分かっている簡単な対照（ダミー制御）に同じ壊し方を
        当てると、必ず電圧列が変わること（N1 が「壊し方が効いていない
        だけ」で通っていないことの確認）。
    """
    W, H = 5, 5
    v = np.zeros((W + 1, H), dtype=int)
    v[0, :] = 1
    v[W, :] = 1
    h = np.zeros((W, H + 1), dtype=int)
    h[:, 0] = 1
    h[:, H] = 1

    def fresh_sim(label):
        xml_path = os.path.join(str(tmp_path), f"open_{label}.xml")
        build_maze_robot_xml(v, h, xml_path, model_name=f"open5x5_{label}", params=params)
        sim = MouseSim(xml_path, params=params)
        sim.full_reset(cell=(2, 2), heading_deg=90.0)
        return sim

    def run_under_test(broken: bool):
        sim = fresh_sim(f"ut_{broken}")
        return _drive_forward_collect_voltages(sim, params, corrupt_privileged=broken)

    def run_control(broken: bool):
        sim = fresh_sim(f"ctrl_{broken}")
        if broken:
            sim.privileged_pose = lambda: (9999.0, -9999.0, 12.345)
        return _drive_dummy_privileged_controller(sim)

    got = negative_control(run_under_test=run_under_test, run_control=run_control)
    print(f"\n[実測] 否定対照(privileged_pose): {got.verdict}")
    assert got.passed, got.verdict


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
    # 【2026-08-19 是正】外乱の無い対称なシミュレータでは、kp_heading の値に
    # 関わらずそもそも直進中に方位はずれない（この検査はそれを確認しているに
    # すぎない）。heading hold（差動補正）自体が実際に効いていることの実測は
    # test_default_heading_hold_corrects_an_injected_yaw_estimate_error を見よ。
    assert heading_drift_deg < 1.0, "直進中に方位が 1° 以上ずれた（無外乱走行としても異常）"


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
# 2-a. 旋回1回のティック数が予算内に収まること（探索走行の旋回過多の是正・
#      再発防止検査。note_030 任務指示・classic/motion.py の
#      DEFAULT_OMEGA_SETTLE 直上のコメント参照）
# ==========================================================================
@pytest.mark.parametrize("start_method,tick_budget", [
    ("start_turn_left", 300),
    ("start_turn_right", 300),
    ("start_turn_180", 450),
])
def test_turn_completes_within_a_tick_budget_at_default_gains(open_sim, params, start_method, tick_budget):
    """🔴 2026-08-19 是正の再発防止検査。

    **是正前の実測（design_v4 maze_42134 の実走で計装して確認。詳細は
    classic/motion.py の DEFAULT_OMEGA_SETTLE 直上のコメント）**: 旧既定値
    DEFAULT_OMEGA_SETTLE=0.05 rad/s では、90° 旋回が平均 474〜459 ティック
    （4.7〜4.6秒）、180° 旋回が平均 607 ティック（6.07秒）かかっていた。
    旋回角度自体は早期（約2.0秒）に許容誤差内へ収まっているにもかかわらず、
    角速度がほぼゼロになるのを待つ完了判定のせいで、探索走行の持ち時間の
    半分以上が旋回だけで消費されていた（同一区画内での方位の振動ではなく、
    旋回1回そのものが遅いことが原因であると、計装した実走で切り分け済み）。

    ここでは `CellMotionController` を既定ゲインのまま（override しない）
    走らせ、旋回 1 回が tick_budget 以内に完了することを直接検査する。
    **DEFAULT_OMEGA_SETTLE を 0.05 へ戻すと、90°/180° のいずれも
    tick_budget を超えてこの検査が落ちる**（実際に戻して確認済み）。
    """
    sim = open_sim
    ctrl = CellMotionController(params)  # 既定ゲインのみ（override しない）
    ctrl.reset(heading_deg=90.0)
    getattr(ctrl, start_method)()
    steps = _run_until_done(sim, ctrl)
    print(f"\n[実測] {start_method}: {steps} ティック（予算 {tick_budget}）")
    assert steps <= tick_budget, (
        f"{start_method} の完了に {steps} ティックかかった（予算 {tick_budget} 超過）。"
        "DEFAULT_OMEGA_SETTLE が緩和前の値に戻っていないか確認すること。"
    )


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
# 4. わざと収束判定のしきい値を壊すと、到達距離誤差が実際に拡大すること
#    （空振りしない検査であることの確認: tests/test_classic_sensing.py の
#    test_broken_threshold_makes_classification_wrong と同じ形に揃える。
#    是正 (6): 旧検査は実測値を誤った定数と比べるだけで、CellMotionController
#    の実装がどうであれ必ず通っていた＝何も壊していなかった）
# ==========================================================================
def test_broken_convergence_threshold_makes_displacement_error_grow(open_sim, params, monkeypatch):
    """是正 (6)（note_029 §12-9(c) 「検査は壊れたときに鳴ることを自分で
    確かめる」）。distance_tol・speed_settle を実際に monkeypatch で破壊し、
    完了判定が早まって到達距離誤差が実際に拡大することを確認する。"""
    sim = open_sim

    # 壊す前: 正しい収束判定では目標との誤差が小さいことを確認する（前提）。
    ctrl_ok = CellMotionController(params)
    ctrl_ok.reset(heading_deg=90.0)
    x0, y0, _ = sim.privileged_pose()
    ctrl_ok.start_forward(1)
    _run_until_done(sim, ctrl_ok)
    x1, y1, _ = sim.privileged_pose()
    disp_ok = math.hypot(x1 - x0, y1 - y0)
    err_ok = disp_ok - params.cell_size
    assert abs(err_ok) < 0.01

    # 完了判定のしきい値を実際に破壊する（早期に「到達」と誤判定させる）。
    sim.full_reset(cell=(2, 2), heading_deg=90.0)
    ctrl_broken = CellMotionController(params)
    monkeypatch.setattr(ctrl_broken, "distance_tol", 0.05)
    monkeypatch.setattr(ctrl_broken, "speed_settle", 1.0)
    ctrl_broken.reset(heading_deg=90.0)
    x0b, y0b, _ = sim.privileged_pose()
    ctrl_broken.start_forward(1)
    _run_until_done(sim, ctrl_broken)
    x1b, y1b, _ = sim.privileged_pose()
    disp_broken = math.hypot(x1b - x0b, y1b - y0b)
    err_broken = disp_broken - params.cell_size

    print(f"\n[実測] 収束しきい値の破壊: 正常誤差={err_ok:+.4f}m 破壊後誤差={err_broken:+.4f}m")

    assert abs(err_broken) > 0.02, (
        "収束しきい値を破壊しても到達距離誤差が拡大しなかった。"
        "この検査が収束ロジックを実際に通していない（空振り）疑いがある。"
    )


# ==========================================================================
# 5. 方位保持 (kp_heading) が実際に効いていることの実測
#    是正 (5): DEFAULT_KP_HEADING を 2.0→0.0 に潰しても、無外乱の直進検査は
#    そもそも曲がらないので全通過してしまう（恒真だった）。ここでは走行
#    開始直後に推測航法のヨー推定へ人為的な初期誤差を注入し、既定ゲインが
#    それを実際に打ち消す（＝機体が実際に補正回頭する）ことを実測する。
# ==========================================================================
def test_default_heading_hold_corrects_an_injected_yaw_estimate_error(open_sim, params):
    """恒真検査の是正（note_029 §12-9(c) 「検査は壊れたときに鳴ることを
    自分で確かめる」）。DEFAULT_KP_HEADING を 0 に潰すと、対になる
    test_zero_kp_heading_does_not_correct_the_injected_yaw_error と
    同じ「収束しない」結果になり、本テストは落ちる。"""
    sim = open_sim
    ctrl = CellMotionController(params)  # 既定ゲイン（DEFAULT_KP_HEADING を使う）
    ctrl.reset(heading_deg=90.0)
    ctrl.start_forward(2)
    # 実機でもジャイロのバイアス・キャリブレーション誤差で起こりうる、
    # 走行開始直後の推測航法ヨー推定の人為的な初期誤差（+10°）を注入する。
    ctrl._yaw_est += math.radians(10.0)

    x0, y0, yaw0 = sim.privileged_pose()
    steps = _run_until_done(sim, ctrl)
    x1, y1, yaw1 = sim.privileged_pose()

    heading_err_final_deg = math.degrees(ctrl._wrap(ctrl._target_heading - ctrl._yaw_est))
    actual_yaw_drift_deg = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))

    print(f"\n[実測] heading hold: 注入誤差=+10.000deg 最終推定誤差={heading_err_final_deg:+.3f}deg "
          f"実際の回頭={actual_yaw_drift_deg:+.3f}deg steps={steps}")

    # 推測航法のヨー推定誤差が実際に収束していること。
    assert abs(heading_err_final_deg) < 1.0, "既定ゲインで方位推定誤差が収束しなかった"
    # 収束が「辻褄合わせ」ではなく、実際に機体が補正回頭したことで起きている
    # ことを確認する（真値 privileged_pose は答え合わせにのみ使う）。
    assert abs(actual_yaw_drift_deg) > 5.0, "推定誤差は消えたが、実際には機体が回頭していない"


def test_zero_kp_heading_does_not_correct_the_injected_yaw_error(open_sim, params):
    """空振り防止の対照: kp_heading=0 まで潰すと、同じ注入誤差が収束しない
    ことを確認する（上のテストが恒真ではないことの直接証拠）。"""
    sim = open_sim
    ctrl = CellMotionController(params, kp_heading=0.0)
    ctrl.reset(heading_deg=90.0)
    ctrl.start_forward(2)
    ctrl._yaw_est += math.radians(10.0)

    steps = _run_until_done(sim, ctrl)
    heading_err_final_deg = math.degrees(ctrl._wrap(ctrl._target_heading - ctrl._yaw_est))
    print(f"\n[実測] kp_heading=0: 注入誤差=+10.000deg 最終推定誤差={heading_err_final_deg:+.3f}deg steps={steps}")
    assert abs(heading_err_final_deg) > 5.0, "kp_heading=0 でも誤差が収束した（注入方法が効いていない疑い）"


# ==========================================================================
# 6. S3（最短走行）で足した `cells_completed`・`reanchor_heading` の単体検査
#    （note_030 §3 S3 任務指示 7。tests/test_classic_fast_run.py が実際の
#    最短走行を通しで検査するのに対し、こちらは classic/motion.py だけを
#    見る単体検査）
# ==========================================================================
def test_cells_completed_is_zero_while_idle_turning_or_stopped(open_sim, params):
    """`cells_completed` は FORWARD 実行中以外は常に 0 であること
    （note_030 §3 S3 任務指示: 「FORWARD 以外では 0」）。"""
    sim = open_sim
    ctrl = CellMotionController(params)
    ctrl.reset(heading_deg=90.0)
    assert ctrl.cells_completed == 0, "start_forward を一度も呼んでいない(is_idle)状態で 0 でない"

    ctrl.start_turn_left()
    obs = sim.observation()
    ctrl.update(obs)
    assert ctrl.cells_completed == 0, "旋回中に 0 でない"

    ctrl.start_stop()
    obs = sim.observation()
    ctrl.update(obs)
    assert ctrl.cells_completed == 0, "停止中に 0 でない"


def test_cells_completed_counts_cells_passed_during_a_multi_cell_forward(open_sim, params):
    """`cells_completed` の契約（`int(self._dist_est / self.params.cell_size)`）
    を、実際に 2 区画直進させて実測する。移動が進むにつれ 0→1 と単調に
    増えていくこと。

    🔴 完了時ちょうどに区画数(2)と一致することまでは要求しない
    （実測: 完了判定は `abs(remaining) < distance_tol` であり、収束は
    target_dist の**手前**からの漸近なので、`done=True` になった瞬間の
    `_dist_est` は `target_dist` よりわずかに小さいことがあり、
    `int(dist_est/cell_size)` が区画数-1 のまま止まりうる。これは実装の
    契約どおりの挙動であり、`classic/explorer.py` の `tick()` が
    「途中の区画境界の検出には `cells_completed`、最後の1区画の確定には
    `done` フラグ」と使い分けている理由そのものである — 完了時の最後の
    1区画は `cells_completed` の値に依存せず `_advance_state()` が
    無条件に1区画進める設計なので、この挙動があっても最短走行の区画
    カウントは狂わない）。"""
    sim = open_sim
    ctrl = CellMotionController(params)
    ctrl.reset(heading_deg=90.0)
    ctrl.start_forward(2)
    assert ctrl.cells_completed == 0, "start_forward 直後(まだ動いていない)で 0 でない"

    seen = []
    for _ in range(MAX_STEPS):
        obs = sim.observation()
        vl, vr, done = ctrl.update(obs)
        sim.step_control(vl, vr)
        seen.append(ctrl.cells_completed)
        if done:
            break
    else:
        raise AssertionError(f"{MAX_STEPS} ステップ以内に収束しなかった")

    print(f"\n[実測] cells_completed の遷移(末尾10件): {seen[-10:]} 完了時の値={ctrl.cells_completed} "
          f"完了時の_dist_est={ctrl._dist_est:.6f}m (目標={2 * params.cell_size:.6f}m)")
    assert seen == sorted(seen), "cells_completed が単調に増加していない"
    assert 1 in seen, "2区画直進の途中で cells_completed=1 を一度も観測しなかった"
    assert max(seen) <= 2, "cells_completed が区画数(2)を超えて増えた"


def test_reanchor_heading_overwrites_relative_to_current_yaw_estimate_not_additive(params):
    """`reanchor_heading` の契約（`_target_heading = self._yaw_est + bias_rad`
    への**上書き**）が、既存の `bias_target_heading`（現在の `_target_heading`
    への**加算**）と別物であることを、ヨー推定のドリフトを模して実測する
    （note_030 §3 S3 任務指示・classic/motion.py の reanchor_heading
    docstring）。走行開始後にヨー推定だけがドリフトした状況を人為的に作り、
    2 つのメソッドの結果が実際にドリフト量ぶんだけ食い違うことを確認する
    （シミュレータ不要のユニット検査）。"""
    drift = math.radians(4.0)  # 直進中に推測航法へ蓄積したと想定するドリフト
    bias = math.radians(2.0)   # 横位置補正が返す目標方位バイアス

    ctrl_add = CellMotionController(params)
    ctrl_add.reset(heading_deg=90.0)
    ctrl_add.start_forward(3)  # _target_heading = _yaw_est(この時点の値) にリセットされる
    yaw_before_drift = ctrl_add._yaw_est
    ctrl_add._yaw_est += drift  # ドリフトを注入(_target_heading はまだ古いまま)
    ctrl_add.bias_target_heading(bias)
    expected_add = yaw_before_drift + bias  # ドリフトの影響を受けない(古い基準に加算するだけ)

    ctrl_re = CellMotionController(params)
    ctrl_re.reset(heading_deg=90.0)
    ctrl_re.start_forward(3)
    ctrl_re._yaw_est += drift
    ctrl_re.reanchor_heading(bias)
    expected_re = (yaw_before_drift + drift) + bias  # ドリフト後の"今"を基準に置き直す

    diff_deg = math.degrees(ctrl_re._target_heading - ctrl_add._target_heading)
    print(f"\n[実測] bias_target_heading の結果={math.degrees(ctrl_add._target_heading):.4f}deg "
          f"reanchor_heading の結果={math.degrees(ctrl_re._target_heading):.4f}deg "
          f"差={diff_deg:.4f}deg (注入ドリフト={math.degrees(drift):.4f}deg)")

    assert abs(ctrl_add._target_heading - expected_add) < 1e-12, \
        "bias_target_heading が加算(現在の _target_heading への加算)になっていない"
    assert abs(ctrl_re._target_heading - expected_re) < 1e-12, \
        "reanchor_heading が「現在のヨー推定を基準に置き直す」上書きになっていない"
    assert abs(diff_deg - math.degrees(drift)) < 1e-9, (
        "reanchor_heading と bias_target_heading の差が注入したドリフト量と一致しない"
        "（2つのメソッドが実際には同じ計算をしている空振りの疑い）"
    )


# ==========================================================================
# 7. S4（円弧スラローム）の単体検査（note_030 §8-2・
#    experiments/exp_025_s4_slalom/PREREG.md §12 チェックリスト最終項目）。
#    弧は半区画（0.090m）ぶんの余白が前後左右に要るため、既存の open_sim
#    （5x5・区画(2,2)開始）とは別に、より広い開通路のフィクスチャを使う。
# ==========================================================================
@pytest.fixture()
def open_wide_sim(tmp_path, params):
    """スラローム検査用の広い開通路（9x9・内部壁なし・区画(4,4)開始）。
    区画中心から各壁まで約 0.45m の余白があり、助走(1区画)＋弧(半径0.090m)
    の移動量（進行方向・左右とも 0.3m 未満）を余裕をもって収める。"""
    W, H = 9, 9
    v = np.zeros((W + 1, H), dtype=int)
    v[0, :] = 1
    v[W, :] = 1
    h = np.zeros((W, H + 1), dtype=int)
    h[:, 0] = 1
    h[:, H] = 1
    xml_path = os.path.join(str(tmp_path), "open_wide.xml")
    build_maze_robot_xml(v, h, xml_path, model_name="open9x9", params=params)
    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(4, 4), heading_deg=90.0)
    return sim


def _forward_left_components(x0, y0, yaw0, x1, y1):
    """開始位置(x0,y0)・開始方位 yaw0 を基準に、変位を
    「進行方向成分・左方向成分」に分解する（弧の位置ずれ検査の答え合わせ）。"""
    dx = x1 - x0
    dy = y1 - y0
    forward = dx * math.cos(yaw0) + dy * math.sin(yaw0)
    left = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
    return forward, left


def _run_slalom_to_completion(sim, ctrl):
    """スラローム(SLALOM_LEFT_90/SLALOM_RIGHT_90)を完了まで走らせ、
    (steps, v_final) を返す。v_final は完了を検出したティックの
    「ctrl.update() 呼び出し直前」の実測並進速度（推測航法の v_meas と
    同じ定義: (omega_l+omega_r)/2*r）で、止まらずに曲がったことの直接
    検査に使う。"""
    for step in range(1, MAX_STEPS + 1):
        obs = sim.observation()
        _gyro_z, omega_l, omega_r = ctrl._split_obs(obs)
        v_meas = (omega_l + omega_r) / 2.0 * ctrl.params.wheel_radius
        vl, vr, done = ctrl.update(obs)
        sim.step_control(vl, vr)
        if done:
            return step, v_meas
    raise AssertionError(f"{MAX_STEPS} ステップ以内に弧が完了しなかった")


def test_slalom_left_turns_90_degrees_without_stopping(open_wide_sim, params):
    """S4 の主要検査（左）: 直進で助走した状態から `start_slalom_left()` を
    発行し、止まらずに 90° 曲がって区画中心線どうしを結ぶ四分円（半径
    `DEFAULT_ARC_RADIUS`）を通ることを実測する（所要時間の設計値は
    PREREG.md §9 錨 A1: `R_arc·(π/2)/v_cruise`。ここでは実装の定数から
    その場で計算し、実測と照合する）。"""
    sim = open_wide_sim
    ctrl = CellMotionController(params)
    ctrl.reset(heading_deg=90.0)

    # 直進で助走する（stop_at_end=False で速度を保ったまま完了させる）。
    # PREREG.md §2「弧の前後の直進をそれぞれ R_arc だけ短く発行する」に
    # 合わせ、区画中心(1区画=cell_size)ではなく区画境界(cell_size-R_arc)
    # まで進めてから弧へ入る（区画中心線→区画中心線の四分円という設計を
    # 満たすための助走距離。1区画丸ごと進むと弧が区画中心から始まり、
    # 隅柱に接触することを実測で確認した）。
    run_up = params.cell_size - motion_module.DEFAULT_ARC_RADIUS
    ctrl.start_forward(distance=run_up, stop_at_end=False)
    _run_until_done(sim, ctrl)

    x0, y0, yaw0 = sim.privileged_pose()
    ctrl.start_slalom_left()
    steps, v_final = _run_slalom_to_completion(sim, ctrl)
    x1, y1, yaw1 = sim.privileged_pose()

    yaw_delta_deg = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))
    elapsed = steps * params.control_dt
    expected_elapsed = motion_module.DEFAULT_ARC_RADIUS * math.pi / 2.0 / motion_module.DEFAULT_V_CRUISE
    forward, left = _forward_left_components(x0, y0, yaw0, x1, y1)

    print(f"\n[実測] slalom_left: ヨー変化={yaw_delta_deg:+.3f}deg "
          f"所要時間={elapsed:.4f}s(設計値={expected_elapsed:.4f}s) "
          f"終端速度={v_final:.4f}m/s(v_cruise比={v_final / motion_module.DEFAULT_V_CRUISE:.3f}) "
          f"進行方向変位={forward * 1000:.2f}mm 左方向変位={left * 1000:.2f}mm steps={steps}")

    assert abs(yaw_delta_deg - 90.0) < 3.0, "弧のヨー変化が +90° から 3° 以上ずれている"
    assert abs(elapsed - expected_elapsed) < expected_elapsed * 0.10, (
        f"弧の所要時間が設計値から 10% 以上ずれている(設計値={expected_elapsed:.4f}s 実測={elapsed:.4f}s)"
    )
    assert v_final >= 0.8 * motion_module.DEFAULT_V_CRUISE, "弧の完了時点で並進速度が落ちている(止まっている疑い)"
    assert abs(forward - motion_module.DEFAULT_ARC_RADIUS) < 0.010, (
        "進行方向の変位が弧の半径(DEFAULT_ARC_RADIUS)から 10mm 以上ずれている"
    )
    assert abs(left - motion_module.DEFAULT_ARC_RADIUS) < 0.010, (
        "左方向の変位が弧の半径(DEFAULT_ARC_RADIUS)から 10mm 以上ずれている"
    )


def test_slalom_right_turns_minus_90_degrees_without_stopping(open_wide_sim, params):
    """S4 の主要検査（右）: 上の左旋回版の左右対称。"""
    sim = open_wide_sim
    ctrl = CellMotionController(params)
    ctrl.reset(heading_deg=90.0)

    # 区画境界(cell_size-R_arc)まで助走してから弧へ入る（左旋回版と同じ理由。
    # PREREG.md §2）。
    run_up = params.cell_size - motion_module.DEFAULT_ARC_RADIUS
    ctrl.start_forward(distance=run_up, stop_at_end=False)
    _run_until_done(sim, ctrl)

    x0, y0, yaw0 = sim.privileged_pose()
    ctrl.start_slalom_right()
    steps, v_final = _run_slalom_to_completion(sim, ctrl)
    x1, y1, yaw1 = sim.privileged_pose()

    yaw_delta_deg = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))
    elapsed = steps * params.control_dt
    expected_elapsed = motion_module.DEFAULT_ARC_RADIUS * math.pi / 2.0 / motion_module.DEFAULT_V_CRUISE
    forward, left = _forward_left_components(x0, y0, yaw0, x1, y1)

    print(f"\n[実測] slalom_right: ヨー変化={yaw_delta_deg:+.3f}deg "
          f"所要時間={elapsed:.4f}s(設計値={expected_elapsed:.4f}s) "
          f"終端速度={v_final:.4f}m/s(v_cruise比={v_final / motion_module.DEFAULT_V_CRUISE:.3f}) "
          f"進行方向変位={forward * 1000:.2f}mm 右方向変位={-left * 1000:.2f}mm steps={steps}")

    assert abs(yaw_delta_deg - (-90.0)) < 3.0, "弧のヨー変化が -90° から 3° 以上ずれている"
    assert abs(elapsed - expected_elapsed) < expected_elapsed * 0.10, (
        f"弧の所要時間が設計値から 10% 以上ずれている(設計値={expected_elapsed:.4f}s 実測={elapsed:.4f}s)"
    )
    assert v_final >= 0.8 * motion_module.DEFAULT_V_CRUISE, "弧の完了時点で並進速度が落ちている(止まっている疑い)"
    assert abs(forward - motion_module.DEFAULT_ARC_RADIUS) < 0.010, (
        "進行方向の変位が弧の半径(DEFAULT_ARC_RADIUS)から 10mm 以上ずれている"
    )
    assert abs(left - (-motion_module.DEFAULT_ARC_RADIUS)) < 0.010, (
        "右方向の変位が弧の半径(DEFAULT_ARC_RADIUS)から 10mm 以上ずれている"
    )


def test_forward_with_stop_at_end_false_keeps_moving(tmp_path, params):
    """S4: `start_forward(1, stop_at_end=False)` が完了時に並進速度を
    落とさないこと、かつ既定の `start_forward(1)`（`stop_at_end=True`）は
    従来どおり完了時に速度がしきい値未満へ収束することを、同じ 1 区画直進で
    対にして実測する（`classic/motion.py` の update() FORWARD 分岐にある
    「実装仕様書 §4 からの唯一の逸脱」コメントが対象とする性質そのもの）。"""
    def build_open_sim(label):
        W, H = 5, 5
        v = np.zeros((W + 1, H), dtype=int)
        v[0, :] = 1
        v[W, :] = 1
        h = np.zeros((W, H + 1), dtype=int)
        h[:, 0] = 1
        h[:, H] = 1
        xml_path = os.path.join(str(tmp_path), f"open_{label}.xml")
        build_maze_robot_xml(v, h, xml_path, model_name=f"open5x5_{label}", params=params)
        sim = MouseSim(xml_path, params=params)
        sim.full_reset(cell=(2, 2), heading_deg=90.0)
        return sim

    def run(stop_at_end):
        sim = build_open_sim(f"stop_at_end_{stop_at_end}")
        ctrl = CellMotionController(params)
        ctrl.reset(heading_deg=90.0)
        ctrl.start_forward(1, stop_at_end=stop_at_end)
        for _ in range(MAX_STEPS):
            obs = sim.observation()
            _gyro_z, omega_l, omega_r = ctrl._split_obs(obs)
            v_meas = (omega_l + omega_r) / 2.0 * params.wheel_radius
            vl, vr, done = ctrl.update(obs)
            sim.step_control(vl, vr)
            if done:
                return v_meas
        raise AssertionError(f"{MAX_STEPS} ステップ以内に収束しなかった(stop_at_end={stop_at_end})")

    v_final_rolling = run(stop_at_end=False)
    v_final_stopped = run(stop_at_end=True)

    print(f"\n[実測] start_forward(1, stop_at_end=False) 完了時速度={v_final_rolling:.4f}m/s "
          f"start_forward(1)(既定) 完了時速度={v_final_stopped:.4f}m/s "
          f"(v_cruise={motion_module.DEFAULT_V_CRUISE}m/s speed_settle={motion_module.DEFAULT_SPEED_SETTLE}m/s)")

    assert v_final_rolling >= 0.8 * motion_module.DEFAULT_V_CRUISE, (
        "stop_at_end=False の完了時点で速度が落ちている(止まっている疑い)"
    )
    assert abs(v_final_stopped) < motion_module.DEFAULT_SPEED_SETTLE, (
        "stop_at_end=True(既定)の完了時点で速度がしきい値未満に収束していない"
    )


def test_default_forward_is_unchanged_by_the_slalom_feature(open_sim, params):
    """🔴 絶対条件の直接検査: スラローム機能を追加した後も、既定の
    `start_forward(n)`（`stop_at_end` を渡さない・従来どおりの呼び出し）が、
    本追加より前に記録された実測値（本ファイル冒頭の
    `test_forward_two_cells_reaches_target_distance_and_holds_heading` と
    同条件・`classic/motion.py` モジュール docstring に記録の 0.3562m）を
    1mm の誤差もなく再現すること。"""
    sim = open_sim
    ctrl = CellMotionController(params)
    ctrl.reset(heading_deg=90.0)

    x0, y0, _ = sim.privileged_pose()
    ctrl.start_forward(2)
    _run_until_done(sim, ctrl)
    x1, y1, _ = sim.privileged_pose()

    displacement = math.hypot(x1 - x0, y1 - y0)
    recorded = 0.3562  # classic/motion.py モジュール docstring に記録されている、本追加前の実測値
    print(f"\n[実測] 既定 start_forward(2): 実測変位={displacement:.4f}m 記録値(本追加前)={recorded:.4f}m "
          f"差={displacement - recorded:+.4f}m")
    assert abs(displacement - recorded) < 0.001, (
        "既定の直進到達距離が、スラローム機能追加前に記録された実測値(0.3562m)から 1mm 以上ずれた。"
        "既定経路がスラローム追加の影響を受けている疑いがある。"
    )


def test_slalom_radius_is_actually_used(tmp_path, params):
    """否定対照 N3 の単体版（PREREG.md §10）: `start_slalom_left(radius=...)`
    で半径を既定より小さくすると、ヨー変化は同じ 90° のまま、弧の走行距離
    （`ctrl._dist_est`。弧に入ってから積算し直した弧長そのもの）と所要時間が
    半径の比率どおりに短くなることを実測する（半径が実装の中で本当に
    使われていることの確認。使われていなければ、半径を変えても何も
    変わらないはず）。

    🔴 仕様どおりに書けなかった箇所: 当初の指示は「半径を既定の厳密に半分
    (0.045m) にする」だったが、実測すると `radius=0.045` は本機能では
    完走できない。必要な前置き角速度 `v_cruise/radius = 0.12/0.045 =
    2.667 rad/s` が、既定半径 0.090m 専用に「前置き分は v/R=1.333、
    `DEFAULT_OMEGA_ARC=2.0` に対し 0.667 の余裕」という前提で決めた定数
    `DEFAULT_OMEGA_ARC=2.0` を超えてしまい、`update()` の
    `clip(omega_ff+kp_arc_yaw*err, -omega_arc, omega_arc)` で角速度が
    飽和する。飽和すると実効旋回半径が `v_cruise/omega_arc=0.060m` へ
    張り付き、要求半径(0.045m)から計算した弧長の目標(0.0707m)に
    ヨー角90°未満で到達してしまう（実測: `radius=0.045` でヨー変化は
    約56°どまり）。`DEFAULT_ARC_RADIUS`・`DEFAULT_OMEGA_ARC` とも実装
    仕様書の値をそのまま使う制約の下ではこの組み合わせは物理的に
    実現できないため、ここでは `radius=0.075m`（既定の 5/6。前置き角速度
    1.6 rad/s で `omega_arc=2.0` に対し 0.4 の余裕を残し、実測でヨー変化
    90.0° ちょうどに収束することを確認済み）を使う。半径が実際に効いて
    いることを確認するという N3 単体版の目的は、既定より明確に小さい別の
    半径で走行距離・所要時間が短くなることを確認すれば十分満たされる。"""
    def build_wide_sim(label):
        W, H = 9, 9
        v = np.zeros((W + 1, H), dtype=int)
        v[0, :] = 1
        v[W, :] = 1
        h = np.zeros((W, H + 1), dtype=int)
        h[:, 0] = 1
        h[:, H] = 1
        xml_path = os.path.join(str(tmp_path), f"open_wide_{label}.xml")
        build_maze_robot_xml(v, h, xml_path, model_name=f"open9x9_{label}", params=params)
        sim = MouseSim(xml_path, params=params)
        sim.full_reset(cell=(4, 4), heading_deg=90.0)
        return sim

    def run_arc(radius, label):
        sim = build_wide_sim(label)
        ctrl = CellMotionController(params)
        ctrl.reset(heading_deg=90.0)
        # 区画境界(cell_size-radius)まで助走してから弧へ入る（PREREG.md §2。
        # 半径ごとに正しい助走距離を計算しないと弧が区画中心寄りにずれ、
        # 隅柱に接触する — test_slalom_left/right と同じ理由）。
        run_up = params.cell_size - radius
        ctrl.start_forward(distance=run_up, stop_at_end=False)
        _run_until_done(sim, ctrl)
        x0, y0, yaw0 = sim.privileged_pose()
        ctrl.start_slalom_left(radius=radius)
        for step in range(1, MAX_STEPS + 1):
            obs = sim.observation()
            vl, vr, done = ctrl.update(obs)
            sim.step_control(vl, vr)
            if done:
                x1, y1, yaw1 = sim.privileged_pose()
                yaw_delta_deg = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))
                arc_length = ctrl._dist_est  # 弧に入ってから積算し直した弧長そのもの
                return step, yaw_delta_deg, arc_length
        raise AssertionError(f"{MAX_STEPS} ステップ以内に弧が完了しなかった(radius={radius})")

    r_default = motion_module.DEFAULT_ARC_RADIUS
    r_small = 0.075  # 既定の 5/6（理由は docstring 参照。omega_arc の飽和を避ける）
    ratio = r_small / r_default
    steps_default, yaw_default, len_default = run_arc(r_default, "default")
    steps_small, yaw_small, len_small = run_arc(r_small, "small")

    time_default = steps_default * params.control_dt
    time_small = steps_small * params.control_dt

    print(f"\n[実測] 半径{r_default:.3f}m: steps={steps_default} time={time_default:.4f}s "
          f"弧長={len_default:.4f}m yaw={yaw_default:+.2f}deg / "
          f"半径{r_small:.3f}m(既定比{ratio:.3f}): steps={steps_small} time={time_small:.4f}s "
          f"弧長={len_small:.4f}m yaw={yaw_small:+.2f}deg")

    assert abs(yaw_default - 90.0) < 3.0, "既定半径でのヨー変化が 90° から 3° 以上ずれている"
    assert abs(yaw_small - 90.0) < 3.0, "小さい半径でのヨー変化が 90° から 3° 以上ずれている(半径のみを変えたはず)"
    assert abs(time_small - time_default * ratio) < time_default * ratio * 0.15, (
        "半径を縮めても弧の所要時間が比率どおりに縮まなかった(半径が効いていない疑い)"
    )
    assert abs(len_small - len_default * ratio) < len_default * ratio * 0.15, (
        "半径を縮めても弧長が比率どおりに縮まなかった(半径が効いていない疑い)"
    )
