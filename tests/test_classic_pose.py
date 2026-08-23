"""
tests/test_classic_pose.py
================
classic/pose.py（連続姿勢の拡張カルマンフィルタ・予測だけ）の検査。

書き方は `tests/test_classic_motion.py` に合わせる（特に
`test_controller_does_not_use_privileged_pose` のような「使っていないことを
確かめる」検査の作り方）。**「〜のはず」で検査を書かず、実際に走らせて
値を見てから、その値を固定する検査にしてある**（本ファイル作成時に
手元で実行して得た実測値は各テストのコメントに残す）。
"""
import ast
import math
import os

import numpy as np
import pytest

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

from classic.checks import negative_control
from classic.motion import CellMotionController
from classic.pose import PoseEstimator

MAX_STEPS = 6000  # 安全弁


@pytest.fixture(scope="module")
def params():
    return RobotParams()


@pytest.fixture()
def open_sim(tmp_path, params):
    """内部の壁を全て取り払った 5x5 迷路（外周のみ壁）。
    tests/test_classic_motion.py の同名フィクスチャと同一構成。"""
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


# ==========================================================================
# 1. 真値 (privileged_pose/privileged_velocity) を使っていないことの検査
# ==========================================================================
def test_source_never_accesses_privileged_attributes():
    """静的検査: `classic/pose.py` の AST を走査し、`privileged_pose`／
    `privileged_velocity` への属性アクセス（メソッド呼び出しの対象）が
    1 つも無いことを確認する。

    単純な文字列 grep ではなく AST の `Attribute` ノードだけを見るのは、
    モジュール docstring 中の説明文（「`privileged_pose()` は一切参照しない」
    という日本語の地の文）を誤検出しないため（`classic/motion.py` の
    docstring も同様に説明のためだけに同じ語を含んでいる）。
    """
    src_path = os.path.join(os.path.dirname(__file__), "..", "classic", "pose.py")
    with open(src_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=src_path)

    forbidden = {"privileged_pose", "privileged_velocity"}
    hits = [node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in forbidden]
    assert hits == [], f"classic/pose.py が真値の属性へアクセスしている: {hits}"

    # import 経路も塞がれていることを確認する（mouse.sim を import すらしない）。
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    assert "mouse.sim" not in imported_modules, "classic/pose.py が mouse.sim を import している"


def _drive_forward_collect_poses(sim, params, corrupt_privileged=False):
    """CellMotionController に 2 区画前進させながら、同じ obs を
    PoseEstimator にも与えて並走させる（note_037 §8 段階 E1 と同じ構成）。
    corrupt_privileged=True のとき sim.privileged_pose() をデタラメへ壊す。
    PoseEstimator がそれを一切使わない設計なら、姿勢の列は壊す前と bit 一致するはず。"""
    if corrupt_privileged:
        sim.privileged_pose = lambda: (9999.0, -9999.0, 12.345)

    ctrl = CellMotionController(params)
    ctrl.reset(heading_deg=90.0)
    ctrl.start_forward(2)
    est = PoseEstimator(params, x=0.09 + 2 * params.cell_size, y=0.09 + 2 * params.cell_size,
                        theta=math.pi / 2)

    poses = []
    for _ in range(MAX_STEPS):
        obs = sim.observation()
        est.predict(obs)
        poses.append(est.pose)
        vl, vr, done = ctrl.update(obs)
        sim.step_control(vl, vr)
        if done:
            break
    else:
        raise AssertionError(f"{MAX_STEPS} ステップ以内に収束しなかった")
    return tuple(poses)


def _drive_dummy_privileged_reader(sim, n_steps=5):
    """N2（対照）: 真値 (privileged_pose) を実際に読む、露骨に依存する構成
    （壊し方が効いていることを確認する空振り防止。tests/test_classic_motion.py
    の `_drive_dummy_privileged_controller` と同じ役割）。"""
    readings = []
    for _ in range(n_steps):
        readings.append(sim.privileged_pose())
        sim.step_control(0.0, 0.0)
    return tuple(readings)


def test_pose_estimator_does_not_use_privileged_pose(tmp_path, params):
    """動的検査（否定対照 N1/N2）: `classic/checks.negative_control` を使い、
    `PoseEstimator` を並走させた構成と `sim.privileged_pose()` を壊した構成とで
    姿勢の列が bit 一致すること（N1）、かつ真値を読むと分かっている対照は
    同じ壊し方で必ず結果が変わること（N2）を確認する。"""
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
        return _drive_forward_collect_poses(sim, params, corrupt_privileged=broken)

    def run_control(broken: bool):
        sim = fresh_sim(f"ctrl_{broken}")
        if broken:
            sim.privileged_pose = lambda: (9999.0, -9999.0, 12.345)
        return _drive_dummy_privileged_reader(sim)

    got = negative_control(run_under_test=run_under_test, run_control=run_control)
    print(f"\n[実測] 否定対照(PoseEstimator・privileged_pose): {got.verdict}")
    assert got.passed, got.verdict


# ==========================================================================
# 2. 滑りが無いときの直進が解析解と一致すること
# ==========================================================================
def test_straight_line_matches_analytic_solution_without_slip(params):
    """車輪角速度一定・ジャイロ 0 を `predict_raw` に N 回与え、
    `ds*N` だけ東（theta=0）へ進むという解析解と一致することを確認する。

    [実測] r=0.0135, dt=0.01, omega=4.0 rad/s, N=250 で
    ds_total = r*omega*dt*N = 0.135m。theta=0 起点なので y は変化しない。
    """
    est = PoseEstimator(params)
    est.reset(x=0.0, y=0.0, theta=0.0)

    omega = 4.0
    N = 250
    for _ in range(N):
        est.predict_raw(omega, omega, 0.0)

    r = params.wheel_radius
    dt = params.control_dt
    ds_total = r * omega * dt * N
    x, y, theta = est.pose

    print(f"\n[実測] 直進解析解: ds_total={ds_total:.6f}m 実測x={x:.6f}m 実測y={y:.6f}m theta={theta:.6f}rad")
    assert abs(x - ds_total) < 1e-4, "直進の到達距離が解析解(ds*N)と一致しない"
    assert abs(y - 0.0) < 1e-4, "直進中に横方向(y)が動いた(theta=0起点のはず)"
    assert abs(theta - 0.0) < 1e-6, "ジャイロ0なのにthetaが変化した"


# ==========================================================================
# 3. ジャイロだけを与えたとき、位置が動かず方位だけ変わること
# ==========================================================================
def test_gyro_only_rotates_in_place_without_translating(params):
    est = PoseEstimator(params)
    est.reset(x=0.09, y=0.09, theta=0.0)

    gyro = 0.8  # rad/s
    N = 120
    for _ in range(N):
        est.predict_raw(0.0, 0.0, gyro)

    dt = params.control_dt
    expected_theta = PoseEstimator._wrap(gyro * dt * N)
    x, y, theta = est.pose

    print(f"\n[実測] ジャイロのみ回転: x={x:.9f} y={y:.9f} theta={theta:.6f} 期待theta={expected_theta:.6f}")
    # 許容誤差は単精度(float32)の丸め(0.09付近で相対誤差 ~1e-7 =絶対 ~1e-8)を
    # 見込む。ds=0が厳密に成り立つ(車輪角速度0)ので、動くとすれば単精度の
    # 丸め誤差のみのはずであり、実測でも 1e-8 台にとどまることを確認済み。
    assert x == pytest.approx(0.09, abs=1e-6), "ジャイロのみ入力で x が動いた(車輪角速度0なのに並進した)"
    assert y == pytest.approx(0.09, abs=1e-6), "ジャイロのみ入力で y が動いた(車輪角速度0なのに並進した)"
    assert abs(theta - expected_theta) < 1e-5, "方位の推定が期待値と一致しない"


# ==========================================================================
# 4. theta が [-pi, pi) に正規化されること(境界を含む)
# ==========================================================================
def test_theta_stays_in_range_across_multiple_full_rotations(params):
    """+2pi をまたいで何周も回しても、毎ステップ theta が [-pi, pi) に収まること。

    [実測] gyro=15.0 rad/s, dt=0.01 で 2000 ステップ回すと
    合計回転角 300 rad ≈ 47.7 周。毎ステップ範囲内に収まることを確認した。"""
    est = PoseEstimator(params)
    est.reset(theta=0.0)

    gyro = 15.0
    for i in range(2000):
        est.predict_raw(0.0, 0.0, gyro)
        _, _, theta = est.pose
        assert -math.pi <= theta < math.pi, f"ステップ{i}でthetaが範囲外: {theta}"


def test_theta_wraps_at_the_exact_boundary():
    """境界(ちょうど ±pi)での振る舞いを固定する検査。
    角度の分枝は既に事故を起こしている(docs/COORDINATE_SYSTEM.md §10-3b)ので、
    ここで PoseEstimator._wrap の境界の選び方を実測して固定する。

    [実測] wrap(+pi) == -pi（ちょうど+piは-pi側に正規化される）。
    wrap(-pi) == -pi（-piはそのまま。下側を含む規約）。
    wrap(-pi - 1e-9) ≈ +pi - 1e-9（-piをわずかに下回ると+pi側の値に連続する）。
    wrap(3*pi) == -pi（+2piの整数倍ずれても同じ境界に落ちる）。
    """
    assert PoseEstimator._wrap(math.pi) == pytest.approx(-math.pi, abs=1e-12)
    assert PoseEstimator._wrap(-math.pi) == pytest.approx(-math.pi, abs=1e-12)
    assert PoseEstimator._wrap(3 * math.pi) == pytest.approx(-math.pi, abs=1e-9)
    assert PoseEstimator._wrap(-math.pi - 1e-9) == pytest.approx(math.pi - 1e-9, abs=1e-9)
    # +pi のすぐ内側は正規化されずそのまま(連続性の確認)。
    assert PoseEstimator._wrap(math.pi - 1e-9) == pytest.approx(math.pi - 1e-9, abs=1e-12)


# ==========================================================================
# 5-6. 否定対照 N1・N2（PREREG.md §4）
# ==========================================================================
def test_negative_control_n1_zero_wheel_speed_keeps_position_fixed(params):
    """N1: 車輪角速度を 0 に固定すると、ジャイロが変化しても推定位置は
    動かない（総移動距離 < 1mm）こと。

    [実測] gyro を正弦波状に変化させながら200ステップ回しても総移動距離は
    ちょうど 0（車輪速度0ならds=0が厳密に成り立つため）。"""
    est = PoseEstimator(params)
    est.reset()
    x0, y0, _ = est.pose

    for i in range(200):
        est.predict_raw(0.0, 0.0, 0.3 * math.sin(i * 0.1))

    x1, y1, _ = est.pose
    dist = math.hypot(x1 - x0, y1 - y0)
    print(f"\n[実測] N1 総移動距離: {dist * 1000:.6f} mm")
    assert dist < 0.001, "車輪角速度0なのに推定位置が1mm以上動いた(N1不合格)"


def test_negative_control_n2_zero_gyro_keeps_heading_fixed_despite_wheel_diff(params):
    """N2: ジャイロを 0 に固定すると、実際には差動回転を指示する車輪入力
    (omega_l != omega_r) を与えても方位が変わらないこと。

    [実測] omega_l=3.0, omega_r=-3.0(その場旋回相当の入力)を200ステップ
    与えても、ジャイロ=0を渡し続ける限り theta は起点(pi/2)から一切変化しない。"""
    est = PoseEstimator(params)
    est.reset()
    _, _, theta0 = est.pose

    for _ in range(200):
        est.predict_raw(3.0, -3.0, 0.0)

    _, _, theta1 = est.pose
    print(f"\n[実測] N2 方位変化: {math.degrees(theta1 - theta0):.9f} deg")
    assert theta1 == pytest.approx(theta0, abs=1e-9), (
        "ジャイロ0固定なのに方位が変化した(車輪の差分から回転を取っている疑い。N2不合格)"
    )


# ==========================================================================
# 7. 観測更新が無い間、共分散の対角成分が単調に増えること
# ==========================================================================
def test_covariance_diagonal_increases_monotonically_without_updates(params):
    """[実測] omega_l=2.0, omega_r=2.1, gyro=0.05 を20ステップ与えると、
    P の対角成分(var_x, var_y, var_theta)は各ステップで単調に増加した
    （観測による更新が無いので、当然どのステップでも減ることはない）。"""
    est = PoseEstimator(params)
    est.reset()

    prev_diag = np.diag(est.covariance).copy()
    assert np.all(prev_diag == 0.0), "初期共分散が0でない(既知の初期姿勢という前提と食い違う)"

    for step in range(20):
        est.predict_raw(2.0, 2.1, 0.05)
        diag = np.diag(est.covariance)
        assert np.all(diag >= prev_diag - 1e-12), (
            f"ステップ{step}で共分散の対角成分が減少した: 前={prev_diag} 今={diag}"
        )
        prev_diag = diag

    print(f"\n[実測] 20ステップ後の共分散対角: {prev_diag}")
    assert np.all(prev_diag > 0.0), "20ステップ後も共分散が0のまま(過程雑音が効いていない疑い)"


# ==========================================================================
# 8. 既定の初期姿勢が区画(0,0)の中心・北向きであること
# ==========================================================================
def test_default_initial_pose_is_start_cell_center_facing_north(params):
    est = PoseEstimator(params)
    x, y, theta = est.pose
    print(f"\n[実測] 既定初期姿勢: x={x} y={y} theta={theta}rad({math.degrees(theta)}deg)")
    assert x == pytest.approx(0.09, abs=1e-6)
    assert y == pytest.approx(0.09, abs=1e-6)
    assert theta == pytest.approx(math.pi / 2, abs=1e-6)
    assert est.cell == (0, 0)


# ==========================================================================
# 9. cell が区画境界で正しく切り替わること(下側を含む規約)
# ==========================================================================
def test_cell_switches_exactly_at_the_cell_boundary(params):
    """[実測] x=0.18 ちょうど(cell_size)で cell が (0,y)→(1,y) に切り替わる
    (下側を含む規約: floor(0.18/0.18)=floor(1.0)=1)。"""
    cs = params.cell_size
    est = PoseEstimator(params)

    est.reset(x=cs - 1e-6, y=0.09, theta=math.pi / 2)
    assert est.cell == (0, 0), "境界のわずか手前でまだ次の区画に切り替わっている"

    est.reset(x=cs, y=0.09, theta=math.pi / 2)
    assert est.cell == (1, 0), "境界ちょうどで区画が切り替わっていない(下側を含む規約)"

    est.reset(x=cs + 1e-6, y=0.09, theta=math.pi / 2)
    assert est.cell == (1, 0), "境界のわずか先で区画が正しくない"


# ==========================================================================
# マイコン実装の予算（note_037 §13-4）: 三角関数の呼び出し回数を実測する
# ==========================================================================
def test_predict_raw_calls_cos_and_sin_exactly_once_each(params, monkeypatch):
    """1 制御周期(predict_raw 1 回)あたりの三角関数呼び出しが cos・sin
    それぞれ 1 回、合計 2 回であることを実測する（note_037 §13-4 の予算の
    直接検査）。`classic.pose` モジュールが参照する `math.cos`/`math.sin`
    を数えるラッパへ差し替えて計測する。"""
    import classic.pose as pose_module

    calls = {"cos": 0, "sin": 0}
    real_cos, real_sin = math.cos, math.sin

    def counting_cos(x):
        calls["cos"] += 1
        return real_cos(x)

    def counting_sin(x):
        calls["sin"] += 1
        return real_sin(x)

    monkeypatch.setattr(pose_module.math, "cos", counting_cos)
    monkeypatch.setattr(pose_module.math, "sin", counting_sin)

    est = PoseEstimator(params)
    est.predict_raw(1.0, 1.1, 0.1)

    print(f"\n[実測] predict_raw 1回あたりの三角関数呼び出し: cos={calls['cos']} sin={calls['sin']}")
    assert calls["cos"] == 1, "cos の呼び出しが1回ではない(マイコン予算の想定と食い違う)"
    assert calls["sin"] == 1, "sin の呼び出しが1回ではない(マイコン予算の想定と食い違う)"


def test_state_and_covariance_ram_footprint_is_48_bytes(params):
    """マイコン実装の予算(note_037 §13-4): 状態(x,y,theta)3個+共分散9個=
    12個の単精度浮動小数点数 = 48バイトであることを、実際の dtype/要素数から
    計数する（ハードコードした数値ではなく、実装の itemsize×要素数から出す）。"""
    est = PoseEstimator(params)
    state_count = 3  # x, y, theta
    cov = est.covariance
    assert cov.dtype == np.float32, "共分散が単精度で持たれていない"
    itemsize = cov.dtype.itemsize
    total_bytes = itemsize * (state_count + cov.size)
    print(f"\n[実測] 状態+共分散の RAM: {total_bytes} バイト (itemsize={itemsize}, "
          f"状態{state_count}個+共分散{cov.size}個)")
    assert total_bytes == 48, f"状態+共分散のRAM見積もりが48バイトではない: {total_bytes}"
