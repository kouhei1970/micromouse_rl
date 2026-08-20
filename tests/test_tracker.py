"""
tests/test_tracker.py
================
classic/tracker.py の検査。**実際にシミュレータを動かして**、速度プロファイル
`v_ref(s)`・曲率プロファイル `kappa_ref(s)` への追従が実際に機能することを
確認する（`tests/test_classic_motion.py` と同じ作法: 開通路の迷路を作り、
`MouseSim` を直接回し、真値姿勢は検査の中でだけ答え合わせに使う）。

背景（何を直しに行くのか、`classic/tracker.py` モジュール docstring も参照）:
現行のコマンド方式（1 コマンド発行 → 完了待ち → 次）は実測で直線が理想の
88.9%、ターンが 8.7% だった。ターンが壊滅的なのは「角度は合っているのに
角速度がゼロになるのを待つ」整定待ちのため。プロファイル追従は区間の切れ目で
止まる必要が無く、この概念ごと消える。ここではその性質を実測する。
"""
import math
import os

import numpy as np
import pytest

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

from classic.profile import Segment, arc_length, min_time, spin_turn_time, vehicle_limits
from classic.tracker import ProfileTracker, TrackerGains

MAX_STEPS = 6000  # 安全弁（収束しない場合に無限ループしない）


@pytest.fixture(scope="module")
def params():
    return RobotParams()


@pytest.fixture(scope="module")
def limits(params):
    return vehicle_limits(params)


def _build_open_maze(tmp_path, params, label, W, H):
    """内部の壁を全て取り払った W×H 迷路（外周のみ壁）。
    tests/test_classic_motion.py の open_sim/open_wide_sim と同じ作り方。"""
    v = np.zeros((W + 1, H), dtype=int)
    v[0, :] = 1
    v[W, :] = 1
    h = np.zeros((W, H + 1), dtype=int)
    h[:, 0] = 1
    h[:, H] = 1
    xml_path = os.path.join(str(tmp_path), f"open_{label}.xml")
    build_maze_robot_xml(v, h, xml_path, model_name=f"open_{label}", params=params)
    return xml_path


def _open_sim(tmp_path, params, label, W, H, cell, heading_deg=90.0):
    xml_path = _build_open_maze(tmp_path, params, label, W, H)
    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=cell, heading_deg=heading_deg)
    return sim


def _run_until_done(sim, tracker, max_steps=MAX_STEPS):
    for step in range(1, max_steps + 1):
        obs = sim.observation()
        vl, vr, done = tracker.update(obs)
        sim.step_control(vl, vr)
        if done:
            return step
    raise AssertionError(f"{max_steps} ステップ以内に収束しなかった（発散または閾値不良の疑い）")


# ==========================================================================
# 1. 直線 4 区画（停止→停止）の追従
# ==========================================================================
def test_follows_a_straight_plan(tmp_path, params, limits):
    cell = params.cell_size
    segs = [Segment(length=4 * cell, curvature=0.0)]
    ideal = min_time(segs, limits, v_start=0.0, v_end=0.0)

    sim = _open_sim(tmp_path, params, "straight", W=5, H=8, cell=(2, 1), heading_deg=90.0)
    tracker = ProfileTracker(params)
    tracker.load_plan(ideal.s_grid, ideal.v_grid, ideal.kappa_grid, psi_start=math.radians(90.0))
    tracker.reset(heading_deg=90.0)

    x0, y0, yaw0 = sim.privileged_pose()
    steps = _run_until_done(sim, tracker)
    x1, y1, yaw1 = sim.privileged_pose()

    elapsed = steps * params.control_dt
    time_ratio = elapsed / ideal.total

    target = 4 * cell
    displacement = math.hypot(x1 - x0, y1 - y0)
    pos_err_mm = abs(displacement - target) * 1000.0
    yaw_err_deg = math.degrees(abs(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))))

    print(f"\n[実測] 直線4区画: 計画時間={ideal.total:.4f}s 実測時間={elapsed:.4f}s "
          f"実測/計画={time_ratio:.4f} 終端位置誤差={pos_err_mm:.3f}mm 終端方位誤差={yaw_err_deg:.3f}deg "
          f"steps={steps}")

    assert time_ratio <= 1.3, f"実測タイムが計画の1.3倍を超えた(比={time_ratio:.4f})"
    assert pos_err_mm < 20.0, f"終端位置誤差が20mmを超えた({pos_err_mm:.3f}mm)"
    assert yaw_err_deg < 3.0, f"終端方位誤差が3degを超えた({yaw_err_deg:.3f}deg)"


# ==========================================================================
# 2. 四分円（両端で sqrt(A_LAT*R) の速度を保つ）の追従
# ==========================================================================
def test_follows_an_arc_plan(tmp_path, params, limits):
    """四分円(R=90mm)を、両端で sqrt(A_LAT*R) の速度を保ったまま通る計画を
    追従させる。

    🔴 四分円だけを区画中心からいきなり発行すると、経路が迷路のグリッド
    交点にある隅柱（`tests/test_classic_motion.py` のスラローム検査が
    `run_up = cell_size - R_arc` の助走を挟んでいるのと同じ理由。区画中心は
    柱の位置ではないが、区画中心から始まる四分円弧は柱へ寄っていく）に
    接触する（実測で確認済み）。ここでも同じ助走（`cell_size - R` の直線、
    停止からその区画境界まで加速）を計画へ含め、区画境界から四分円へ入る。
    ヨー変化・終端速度は計画全体（助走＋弧）を通した実測で見る
    （助走は直線=無曲率なので、ヨー変化への寄与は理論上ゼロ）。"""
    R = 0.09  # m
    cell = params.cell_size
    v_corner = math.sqrt(limits.A_LAT * R)
    run_up = cell - R
    segs = [
        Segment(length=run_up, curvature=0.0),
        Segment(length=arc_length(math.pi / 2.0, R), curvature=1.0 / R, kind="arc"),
    ]
    ideal = min_time(segs, limits, v_start=0.0, v_end=v_corner)

    sim = _open_sim(tmp_path, params, "arc", W=9, H=9, cell=(4, 4), heading_deg=90.0)
    tracker = ProfileTracker(params)
    tracker.load_plan(ideal.s_grid, ideal.v_grid, ideal.kappa_grid, psi_start=math.radians(90.0))
    tracker.reset(heading_deg=90.0)

    yaw0 = sim.privileged_pose()[2]
    steps = _run_until_done(sim, tracker)
    yaw1 = sim.privileged_pose()[2]

    elapsed = steps * params.control_dt
    time_ratio = elapsed / ideal.total
    yaw_delta_deg = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))

    # 完了ティックの実測並進速度（tracker.update() 直前の観測から計算。
    # tests/test_classic_motion.py の _run_slalom_to_completion と同じ考え方）
    obs = sim.observation()
    _gyro_z, omega_l, omega_r = tracker._split_obs(obs)
    v_final = (omega_l + omega_r) / 2.0 * params.wheel_radius

    print(f"\n[実測] 助走付き四分円: 計画時間={ideal.total:.4f}s 実測時間={elapsed:.4f}s "
          f"実測/計画={time_ratio:.4f} ヨー変化={yaw_delta_deg:+.3f}deg "
          f"終端速度={v_final:.4f}m/s(計画比={v_final / v_corner:.3f}) steps={steps}")

    assert abs(yaw_delta_deg - 90.0) < 5.0, "ヨー変化が90degから5deg以上ずれている"
    assert v_final >= 0.7 * v_corner, "終端速度が計画の70%未満(止まっている疑い)"


# ==========================================================================
# 3. 直線 → 四分円 → 直線（区間の切れ目で速度が落ちないこと）
# ==========================================================================
def test_straight_arc_straight_without_stopping(tmp_path, params, limits):
    cell = params.cell_size
    R = 0.09
    theta = math.pi / 2.0
    segs = [
        Segment(length=2 * cell, curvature=0.0),
        Segment(length=arc_length(theta, R), curvature=1.0 / R, kind="arc"),
        Segment(length=2 * cell, curvature=0.0),
    ]
    ideal = min_time(segs, limits, v_start=0.0, v_end=0.0)

    sim = _open_sim(tmp_path, params, "sas", W=9, H=9, cell=(2, 2), heading_deg=90.0)
    tracker = ProfileTracker(params)
    tracker.load_plan(ideal.s_grid, ideal.v_grid, ideal.kappa_grid, psi_start=math.radians(90.0))
    tracker.reset(heading_deg=90.0)

    # 区間の境界の弧長座標（最初のSegment=直線の終わり、2番目のSegment=弧の終わり）
    boundary_s = [2 * cell, 2 * cell + arc_length(theta, R)]

    history = []  # (s, v_meas, v_ref_at_s)
    for step in range(1, MAX_STEPS + 1):
        obs = sim.observation()
        _gyro_z, omega_l, omega_r = tracker._split_obs(obs)
        v_meas = (omega_l + omega_r) / 2.0 * params.wheel_radius
        s_before = tracker.s
        vl, vr, done = tracker.update(obs)
        sim.step_control(vl, vr)
        history.append((s_before, v_meas, tracker._v_at(s_before)))
        if done:
            break
    else:
        raise AssertionError(f"{MAX_STEPS} ステップ以内に収束しなかった")

    # 各境界について、境界を跨ぐ前後5ティックのうち、境界に最も近い方から
    # 5ティックぶんの実測速度が、その時点のv_refの70%以上であることを見る。
    worst_ratio_per_boundary = []
    for b in boundary_s:
        idx = min(range(len(history)), key=lambda i: abs(history[i][0] - b))
        lo = max(0, idx - 5)
        hi = min(len(history), idx + 6)
        window = history[lo:hi]
        ratios = [v_meas / v_ref for (_s, v_meas, v_ref) in window if v_ref > 1e-6]
        worst = min(ratios) if ratios else float("nan")
        worst_ratio_per_boundary.append(worst)

    print(f"\n[実測] 直線→弧→直線: 計画時間={ideal.total:.4f}s 実測時間={len(history) * params.control_dt:.4f}s "
          f"境界s={boundary_s} 境界近傍の最小(実測/v_ref)比={worst_ratio_per_boundary}")

    for b, ratio in zip(boundary_s, worst_ratio_per_boundary):
        assert ratio >= 0.7, f"境界(s={b:.4f}m)近傍で実測速度がv_refの70%を下回った(比={ratio:.4f})"


# ==========================================================================
# 4. 直線 16 区画（上限なし・高速）。
# ==========================================================================
def test_high_speed_straight(tmp_path, params, limits):
    """🔴 当初の任務指示は「通らなくてもよい、壊れ方を記録する」だったが、
    ユーザ指示（2026-08-20 追加分）で「通すための工夫を試すこと」に改まった。
    境界フォーシング・逆モデル前置き（`classic/tracker.py` モジュール
    docstring 参照）を実装した結果、実測でこの計画（16区画・2.88m・
    v_max=3.72m/s・上限なし）も収束するようになったため、通常の判定
    （実測/計画・終端誤差）で検査する。"""
    cell = params.cell_size
    segs = [Segment(length=16 * cell, curvature=0.0)]
    ideal = min_time(segs, limits, v_start=0.0, v_end=0.0)

    sim = _open_sim(tmp_path, params, "highspeed", W=3, H=24, cell=(1, 2), heading_deg=90.0)
    tracker = ProfileTracker(params)
    tracker.load_plan(ideal.s_grid, ideal.v_grid, ideal.kappa_grid, psi_start=math.radians(90.0))
    tracker.reset(heading_deg=90.0)

    max_steps = max(MAX_STEPS, int(ideal.total / params.control_dt * 5) + 1000)

    x0, y0, yaw0 = sim.privileged_pose()
    collided = False
    max_lateral_mm = 0.0
    steps = None
    for step in range(1, max_steps + 1):
        obs = sim.observation()
        vl, vr, done = tracker.update(obs)
        info = sim.step_control(vl, vr)
        x, y, yaw = sim.privileged_pose()
        # 進行方向(北=y)を基準にした「経路からの横方向ずれ」= x方向の変位
        # （直進計画なので理想の経路はx一定、東西方向のずれをそのまま横ずれとして見る）
        lateral_mm = abs(x - x0) * 1000.0
        max_lateral_mm = max(max_lateral_mm, lateral_mm)
        if info.get("collision"):
            collided = True
        if done:
            steps = step
            break
    else:
        steps = max_steps

    x1, y1, yaw1 = sim.privileged_pose()
    target = 16 * cell
    displacement = math.hypot(x1 - x0, y1 - y0)
    pos_err_mm = abs(displacement - target) * 1000.0
    yaw_err_deg = math.degrees(abs(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))))
    elapsed = steps * params.control_dt
    time_ratio = elapsed / ideal.total if steps < max_steps else float("inf")
    reached = pos_err_mm < 20.0 and yaw_err_deg < 3.0 and steps < max_steps

    print(f"\n[実測] 直線16区画(高速・上限なし): 計画時間={ideal.total:.4f}s v_max(計画)={ideal.v_max:.4f}m/s "
          f"実測時間={elapsed:.4f}s 実測/計画={time_ratio:.4f} steps={steps}/{max_steps} "
          f"終端位置誤差={pos_err_mm:.3f}mm 終端方位誤差={yaw_err_deg:.3f}deg "
          f"最大横ずれ={max_lateral_mm:.3f}mm 衝突={collided} 到達判定={reached}")

    # 通った場合は通常の判定基準（他の直線検査と同じ桁）を課す。念のため
    # 「経路(直進方向の中心線)から44mm以上外れない・衝突しない」という
    # 緩い安全弁も両立させておく（万一 time_ratio が悪化しても、衝突・
    # 大幅な横ずれだけは必ず検出する）。
    assert max_lateral_mm < 44.0, f"経路から44mm以上ずれた(最大横ずれ={max_lateral_mm:.3f}mm)"
    assert not collided, "壁と衝突した"
    assert reached, (
        f"直線16区画(高速)が収束しなかった、または終端誤差が判定基準を超えた"
        f"(位置誤差={pos_err_mm:.3f}mm 方位誤差={yaw_err_deg:.3f}deg steps={steps}/{max_steps})"
    )
    assert time_ratio <= 1.3, f"実測タイムが計画の1.3倍を超えた(比={time_ratio:.4f})"


# ==========================================================================
# 4b. その場旋回（時間最適バンバン軌道。整定待ちをしないこと）
#     ユーザ指示（2026-08-20 追加分・追加2）: 「目標角度へP制御して整定を
#     待つ」のではなく、`load_spin_plan()` の時間最適軌道で旋回する。
# ==========================================================================
@pytest.mark.parametrize("delta_deg", [90.0, 180.0])
def test_spin_turn_needs_no_settling_wait(tmp_path, params, limits, delta_deg):
    """現行のコマンド方式（超信地旋回・角速度がゼロになるのを待つ完了判定）の
    実測は、90°旋回が 2.00s（`classic/motion.py` モジュール docstring 記載の
    実測値）。物理限界（`spin_turn_time`）に対して 2.00/0.173=11.6 倍。
    ここでは `load_spin_plan()` の時間最適バンバン軌道で同じ90°/180°旋回を
    実測し、実測/物理限界の比を印字する（これが一番知りたい数字）。

    🔴 2026-08-20 是正の経緯: 当初は逆モデル前置き（角速度目標のリード補償）
    版で終端角速度が0.2rad/sを満たせず（90°=6.9rad/s、180°=5.7rad/s）、
    これを「車輪速度ループの時定数tau_vがその場旋回の総所要時間と同程度だから
    構造的に無理」と判断したが、**教授セッションの検算でこれは誤りと指摘された**
    （その場90°旋回で1輪に要る力0.291Nに対し停動時に出せる力は2.056N・
    7.1倍の余裕、必要電圧0.904Vに対し電源3.0V・2.1Vの余りがあり、力にも
    電圧にも大きな余裕があった。速度PIが必要な電圧を出していなかっただけ）。
    そこで角速度目標のかさ上げをやめ、計画から必要な電圧を直接計算して足す
    逆動力学（電圧）前置きに切り替えた結果、終端角速度は90°=0.77rad/s・
    180°=0.36rad/s まで改善した（9〜16倍）。**それでも0.2rad/s以下には
    届いていない**（車輪PIゲイン kp_wheel_spin/ki_wheel_spin・kp_psi_spin を
    広く実測スイープしたが、両角度とも角度誤差3°未満を保ったまま0.2rad/s以下
    まで追い込む組み合わせは見つからなかった。詳細ログは本タスクの完了報告を
    参照）。電圧が飽和していれば「頭打ち」の説明が成り立つが、**実測では
    飽和していない**（`tracker.voltage_saturated` は両角度とも False）ため、
    合わせ込まずそのまま記録する。"""
    delta = math.radians(delta_deg)
    sp = spin_turn_time(delta, limits)

    sim = _open_sim(tmp_path, params, f"spin{delta_deg}", W=5, H=5, cell=(2, 2), heading_deg=90.0)
    tracker = ProfileTracker(params)
    tracker.reset(heading_deg=90.0)
    tracker.load_spin_plan(delta)

    yaw0 = sim.privileged_pose()[2]
    steps = _run_until_done(sim, tracker)
    yaw1 = sim.privileged_pose()[2]

    elapsed = steps * params.control_dt
    time_ratio = elapsed / sp.time
    actual_delta_deg = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))
    angle_err_deg = abs(abs(actual_delta_deg) - abs(delta_deg))
    gyro_z, _omega_l, _omega_r = tracker._split_obs(sim.observation())
    saturated = tracker.voltage_saturated

    print(f"\n[実測] その場旋回{delta_deg:.0f}deg: 物理限界時間={sp.time:.4f}s({sp.regime}) "
          f"実測時間={elapsed:.4f}s 実測/物理限界={time_ratio:.4f} "
          f"到達角度={actual_delta_deg:+.3f}deg 誤差={angle_err_deg:.3f}deg "
          f"終端角速度={gyro_z:+.4f}rad/s 終端電圧飽和={saturated} steps={steps}")

    assert time_ratio <= 1.3, f"実測タイムが物理限界の1.3倍を超えた(比={time_ratio:.4f})"
    assert angle_err_deg < 3.0, f"到達角度誤差が3degを超えた({angle_err_deg:.3f}deg)"
    assert abs(gyro_z) <= 0.2, (
        f"終端角速度が0.2rad/sを超えた({gyro_z:+.4f}rad/s、終端電圧飽和={saturated})"
    )


# ==========================================================================
# 4c. 否定対照（対で）: 逆動力学（電圧）前置き(use_voltage_feedforward)を
#     切ると、その場旋回の終端角速度が明確に悪化する
# ==========================================================================
def test_voltage_feedforward_reduces_terminal_rate(tmp_path, params, limits):
    """`use_voltage_feedforward=True`（既定）と `False` とで90°/180°その場
    旋回の終端角速度を比べ、両方の実測値を印字する。"""
    def run(delta_deg, use_voltage_feedforward):
        delta = math.radians(delta_deg)
        sim = _open_sim(tmp_path, params, f"vffctrl_{delta_deg}_{use_voltage_feedforward}",
                         W=5, H=5, cell=(2, 2), heading_deg=90.0)
        tracker = ProfileTracker(
            params, gains=TrackerGains(use_voltage_feedforward=use_voltage_feedforward))
        tracker.reset(heading_deg=90.0)
        tracker.load_spin_plan(delta)
        steps = _run_until_done(sim, tracker)
        gyro_z, _omega_l, _omega_r = tracker._split_obs(sim.observation())
        return gyro_z

    for delta_deg in (90.0, 180.0):
        gyro_with_ff = run(delta_deg, True)
        gyro_without_ff = run(delta_deg, False)
        print(f"\n[実測] その場旋回{delta_deg:.0f}deg 終端角速度: "
              f"前置きあり={gyro_with_ff:+.4f}rad/s 前置きなし={gyro_without_ff:+.4f}rad/s")
        assert abs(gyro_with_ff) < abs(gyro_without_ff), (
            f"{delta_deg:.0f}deg: 逆動力学前置きを入れても終端角速度が悪化していない"
            "(前置きが効いていない疑い)"
        )


# ==========================================================================
# 5. 否定対照（対で）: kp_psi=0 に壊すと方位誤差が明確に悪化する
# ==========================================================================
# 🔴 classic/motion.py の test_default_heading_hold_corrects_an_injected_
# yaw_estimate_error / test_zero_kp_heading_does_not_correct_the_injected_
# yaw_error と同じ理由でこの形にした: 外乱の無い対称なシミュレータでは、
# 直線区間で kappa_ref=0 なら psi_ref は一定で、e_psi はそもそも自然には
# 発生しない（symmetric な左右車輪応答のため）。kp_psi を0に壊しても検査1と
# 数値が変わらない「恒真検査」になってしまう。ここでは検査1と同じ計画・
# 同じ開始条件を使い、開始直後に人為的な+10degのヨー推定誤差を注入した上で、
# 既定ゲイン(kp_psi=DEFAULT_KP_PSI)がそれを実際に補正すること(空振り側)と、
# kp_psi=0に壊すと補正されないこと(作動側)を対で確認する。
#
# 🔴 検査1の迷路(W=5, cell=(2,1))をそのまま使うと、+10deg注入直後の補正回頭
# 中に発生する横ずれ（高速巡航中に方位を戻すため、補正しきるまでの間だけ
# 進行方向がわずかに斜めになる）が、実測で隅柱に接触することを確認した
# （実測: 衝突時の接触力105N、その時点の横ずれ64mm）。検査1は外乱を注入
# しないため横ずれがそもそも起きず問題にならないが、本検査は横ずれを
# 前提とするため、迷路の横幅を広げて隅柱との接触余地を無くす。
def _straight_plan_and_sim(tmp_path, params, limits, label):
    cell = params.cell_size
    segs = [Segment(length=4 * cell, curvature=0.0)]
    ideal = min_time(segs, limits, v_start=0.0, v_end=0.0)
    sim = _open_sim(tmp_path, params, label, W=9, H=8, cell=(4, 1), heading_deg=90.0)
    return ideal, sim


_INJECTED_YAW_ERROR_DEG = 4.0
# 🔴 注入量の根拠（実測、2026-08-20）: classic/motion.py の同種検査は+10degを
# 注入するが、本トラッカーは経路追従の最高速度が classic/motion.py の
# v_cruise(0.12m/s)よりずっと速い(直線4区画計画で最高2m/s級)ため、+10degの
# ままだと補正回頭中の横ずれが大きくなりすぎ、迷路の格子線上に必ず立っている
# 隅柱（迷路の横幅を広げても格子間隔は変わらないので回避できない）に接触
# することを実測で確認した（+5deg以上で接触、+4degでは無接触）。+4degへ
# 落として同じ性質（既定ゲインは収束・kp_psi=0は収束しない）を検査する。


def test_default_kp_psi_corrects_an_injected_yaw_estimate_error(tmp_path, params, limits):
    """`classic/motion.py` の `test_default_heading_hold_corrects_an_injected_
    yaw_estimate_error` と同じ答え合わせの形: 「推測航法のヨー推定誤差
    （目標 psi_ref との差）が収束すること」と「その収束が辻褄合わせではなく
    実際に機体が回頭したことで起きていること（真値ヨーが有意にずれる）」を
    両方見る。**実測タイムそのものではなく方位推定誤差の収束を見る検査**
    なので、真値ヨーの変位が小さいことを求めるのは誤り（それは
    「そもそも回頭しなかった」ことの証拠になってしまう）。"""
    ideal, sim = _straight_plan_and_sim(tmp_path, params, limits, "negctrl_ok")
    tracker = ProfileTracker(params)  # 既定ゲイン(kp_psi=DEFAULT_KP_PSI)
    tracker.load_plan(ideal.s_grid, ideal.v_grid, ideal.kappa_grid, psi_start=math.radians(90.0))
    tracker.reset(heading_deg=90.0)
    tracker._yaw_est += math.radians(_INJECTED_YAW_ERROR_DEG)  # 人為的な初期ヨー推定誤差の注入

    x0, y0, yaw0 = sim.privileged_pose()
    _run_until_done(sim, tracker)
    x1, y1, yaw1 = sim.privileged_pose()

    heading_err_final_deg = math.degrees(tracker._wrap(tracker._psi_at(tracker.s) - tracker.yaw_estimate))
    actual_yaw_drift_deg = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))

    print(f"\n[実測] kp_psi既定(注入+{_INJECTED_YAW_ERROR_DEG:.0f}deg): "
          f"最終推定誤差={heading_err_final_deg:+.3f}deg 実際の回頭={actual_yaw_drift_deg:+.3f}deg")

    assert abs(heading_err_final_deg) < 1.0, "既定のkp_psiで注入したヨー推定誤差が収束しなかった"
    assert abs(actual_yaw_drift_deg) > 2.0, "推定誤差は消えたが、実際には機体が回頭していない"


def test_zero_kp_psi_does_not_correct_the_injected_yaw_error(tmp_path, params, limits):
    """空振り防止の対照: kp_psi=0 まで潰すと、同じ注入誤差（ヨー推定誤差）が
    収束しないことを確認する。"""
    ideal, sim = _straight_plan_and_sim(tmp_path, params, limits, "negctrl_broken")
    tracker = ProfileTracker(params, gains=TrackerGains(kp_psi=0.0))  # 壊す
    tracker.load_plan(ideal.s_grid, ideal.v_grid, ideal.kappa_grid, psi_start=math.radians(90.0))
    tracker.reset(heading_deg=90.0)
    tracker._yaw_est += math.radians(_INJECTED_YAW_ERROR_DEG)

    _run_until_done(sim, tracker)
    heading_err_final_deg = math.degrees(tracker._wrap(tracker._psi_at(tracker.s) - tracker.yaw_estimate))

    print(f"\n[実測] kp_psi=0(注入+{_INJECTED_YAW_ERROR_DEG:.0f}deg): 最終推定誤差={heading_err_final_deg:+.3f}deg")
    assert abs(heading_err_final_deg) > 2.0, "kp_psi=0でも注入したヨー推定誤差が収束した(壊し方が効いていない疑い)"
