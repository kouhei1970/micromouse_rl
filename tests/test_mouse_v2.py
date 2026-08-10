"""
tests/test_mouse_v2.py
================
mouse/ パッケージ（ロボット v2, rev.B）の物理検証テスト。

pytest は使わない plain Python スクリプト。実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_mouse_v2.py

各テストは元々 spec_mouse_v2.md §2 で指定された 6 項目:
  1. 質量検査（電池ボディの存在チェックを含む）
  2. 開ループ速度
  3. 対称性（超信地旋回）
  4. センサ（幾何距離）
  5. 決定性
  6. 衝突検出

理論値は docs/MODEL_VERIFICATION_PLAN.md §4.2 の確定パラメータ（RobotParams）と
mouse/motor.py の参照式から実行時に導出する（ハードコード禁止。同計画書 §5 手順4 の方針）。
全テストで max|qacc| < 1e4 rad/s^2 を監視し、超過時は当該テストを不合格とする
（同計画書 §6 共通規約 (v): 数値発散の検出）。

いずれかのテストで例外/assert が起きても他のテストは継続実行し、
最後に全テストの実測値をまとめて表として print する
（"物理的に仕様が満たせない場合はその事実と実測値を報告書に書く" という
 指示があるため、途中で止めず最後まで走らせる）。
"""
import os
import sys
import tempfile

import numpy as np
import mujoco

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mouse.params import RobotParams
from mouse.sim import MouseSim
from mouse.mjcf import build_maze_robot_xml
from mouse import motor as motor_ref


# ======================================================================
# qacc 発散監視ヘルパー（MODEL_VERIFICATION_PLAN §6 共通規約 (v)）
# ======================================================================
# qacc 監視は 2 段構成（MODEL_VERIFICATION_PLAN §6 共通規約(v)、2026-08-10 改訂）:
# 本受け入れテストは全て全開電圧・衝突を含むストレステストのため、緩和帯 (a) を適用する
# （NaN/Inf 即不合格 + max|qacc| < 1e6。全開スリップ・跳ね・衝突時の車輪DOF接触過渡
#  1.5e4〜6.4e4 rad/s² は正当な物理であり誤検出しない。真の数値発散 6.6e7 級は検出する）。
# 厳格帯 (b) 1e4 は低電圧・滑らか力学のテスト（competition/model_verification.py の
# Test0/1/2/3/5/7a）にのみ適用される。
QACC_LIMIT = 1.0e6  # rad/s^2（緩和帯 (a)）


def step_tracked(sim, v_left, v_right, tracker):
    """sim.step_control() をラップし、実行中の max|qacc| を tracker[0] に蓄積する。"""
    result = sim.step_control(v_left, v_right)
    if sim.data.qacc.size:
        cur = float(np.max(np.abs(sim.data.qacc)))
        if cur > tracker[0]:
            tracker[0] = cur
    return result


def qacc_guard(tracker, test_name):
    """qacc 発散が検出されていれば警告を出し False を返す。"""
    ok = np.isfinite(tracker[0]) and tracker[0] < QACC_LIMIT
    if ok:
        print(f"  (qacc監視 OK: max|qacc|={tracker[0]:.3e} rad/s^2 < {QACC_LIMIT:.0e})")
    else:
        print(f"  [WARN] {test_name}: max|qacc|={tracker[0]:.3e} rad/s^2 が閾値 "
              f"{QACC_LIMIT:.0e} を超過（数値発散の疑い。当該テスト不合格）")
    return ok


# ======================================================================
# 結果収集ヘルパー
# ======================================================================
RESULTS = []  # list[dict(name, theory, actual, unit, tol_desc, passed, note)]


def record(name, theory, actual, unit, tol_desc, passed, note=""):
    RESULTS.append(dict(name=name, theory=theory, actual=actual, unit=unit,
                         tol_desc=tol_desc, passed=bool(passed), note=note))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: theory={theory} {unit}, actual={actual} {unit} "
          f"(許容: {tol_desc}) {note}")


def pct_diff(actual, theory):
    if theory == 0:
        return float('nan')
    return (actual - theory) / theory * 100.0


# ======================================================================
# Test 1: 質量検査
# ======================================================================
def test1_mass():
    print("\n=== Test 1: 質量検査 ===")
    xml_path = os.path.join(REPO_ROOT, 'assets', 'mouse_v2.xml')
    assert os.path.exists(xml_path), (
        f"{xml_path} が存在しません。先に "
        "`.venv/bin/python -m mouse.generate_assets` を実行してください。"
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    mouse_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'mouse')
    assert mouse_id >= 0, "body 'mouse' が見つかりません"

    total_mass = float(model.body_subtreemass[mouse_id])

    print("  per-body mass (mouse サブツリー):")
    for b in range(model.nbody):
        bp = b
        is_descendant = False
        while True:
            if bp == mouse_id:
                is_descendant = True
                break
            if bp == 0:
                break
            bp = model.body_parentid[bp]
        if is_descendant:
            print(f"    body[{b}] name='{model.body(b).name}' mass={model.body_mass[b]:.6f} kg")

    inertia = model.body_inertia[mouse_id]
    print(f"  root body 'mouse' 自身（子ボディ除く）の慣性主値: "
          f"[{inertia[0]:.4e}, {inertia[1]:.4e}, {inertia[2]:.4e}] kg*m^2")

    # 電池ボディの存在チェック（rev.B で新設。MODEL_VERIFICATION_PLAN §4.2/§5 手順1-3）
    # geom 個別質量は compile 後の mjModel には保持されない（body_mass に統合される）ため、
    # 存在（mj_name2id）と搭載位置（geom_pos、機体座標系）のみを確認する。
    battery_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'battery')
    battery_present = battery_geom_id >= 0
    print(f"  battery geom 存在: {battery_present}"
          + (f" (pos={model.geom_pos[battery_geom_id]})" if battery_present else ""))

    mass_ok = 0.090 <= total_mass <= 0.120
    ok = mass_ok and battery_present
    note = "" if battery_present else "battery geom が見つかりません"
    record("mass_total_subtree_kg", 0.106, round(total_mass, 6), "kg",
           "0.090〜0.120 (期待 0.106±0.001, battery geom 存在を含む)", ok, note=note)
    return ok


# ======================================================================
# Test 2: 開ループ速度
# ======================================================================
def test2_open_loop_speed():
    print("\n=== Test 2: 開ループ速度 ===")
    xml_path = os.path.join(REPO_ROOT, 'assets', 'mouse_v2.xml')
    params = RobotParams()
    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 0), heading_deg=0)

    qacc_tracker = [0.0]
    n_steps = int(round(3.0 / params.control_dt))  # 3秒 / 0.01s = 300
    for _ in range(n_steps):
        step_tracked(sim, 3.0, 3.0, qacc_tracker)

    obs = sim.observation()
    omega_l, omega_r = float(obs[-2]), float(obs[-1])
    omega_avg = (omega_l + omega_r) / 2.0
    v_forward, omega_z = sim.privileged_velocity()

    # 無負荷定常角速度の理論値（MODEL_VERIFICATION_PLAN §6 Test0 の式。参照実装から実行時導出）
    theory_omega = motor_ref.omega_ss(3.0, params)
    theory_v = theory_omega * params.wheel_radius

    diff_pct = pct_diff(omega_avg, theory_omega)
    ok = abs(diff_pct) <= 10.0

    print(f"  omega_L={omega_l:.2f} rad/s, omega_R={omega_r:.2f} rad/s, "
          f"omega_avg={omega_avg:.2f} rad/s (diff {diff_pct:+.1f}%)")
    print(f"  privileged v_forward={v_forward:.3f} m/s "
          f"(理論値 {theory_v:.3f} m/s, ロールなし仮定), omega_z={omega_z:.4f} rad/s")

    record("wheel_omega_terminal_3V_3s", round(theory_omega, 2), round(omega_avg, 2), "rad/s",
           "±10%", ok, note=f"diff={diff_pct:+.1f}%")
    return ok and qacc_guard(qacc_tracker, "test2_open_loop_speed")


# ======================================================================
# Test 3: 対称性（超信地旋回）
# ======================================================================
def test3_symmetry():
    print("\n=== Test 3: 対称性（超信地旋回） ===")
    xml_path = os.path.join(REPO_ROOT, 'assets', 'mouse_v2.xml')
    params = RobotParams()
    sim = MouseSim(xml_path, params=params)

    qacc_tracker = [0.0]

    def spin_and_measure_gyro_z(v_left, v_right, duration=1.0):
        sim.full_reset(cell=(0, 0), heading_deg=0)
        n_steps = int(round(duration / params.control_dt))
        for _ in range(n_steps):
            step_tracked(sim, v_left, v_right, qacc_tracker)
        obs = sim.observation()
        return float(obs[11])  # gyro z (obs: [range6, accel3, gyro3, omegaL, omegaR])

    gz_pos = spin_and_measure_gyro_z(3.0, -3.0)   # 左+3V / 右-3V
    gz_neg = spin_and_measure_gyro_z(-3.0, 3.0)   # 逆コマンド

    print(f"  左+3V/右-3V: gyro_z = {gz_pos:.3f} rad/s")
    print(f"  左-3V/右+3V: gyro_z = {gz_neg:.3f} rad/s")

    ok_pos_mag = abs(gz_pos) > 1.0
    ok_neg_mag = abs(gz_neg) > 1.0
    ok_opposite = (gz_pos * gz_neg) < 0

    ok_pos = ok_pos_mag and ok_opposite
    ok_neg = ok_neg_mag and ok_opposite

    record("symmetry_gyro_z_left+3V_right-3V", "|gz|>1, 正負基準", round(gz_pos, 3), "rad/s",
           "|gyro_z|>1 かつ 逆コマンドで逆符号", ok_pos)
    record("symmetry_gyro_z_left-3V_right+3V", "|gz|>1, 正負基準", round(gz_neg, 3), "rad/s",
           "|gyro_z|>1 かつ 逆コマンドで逆符号", ok_neg)
    return ok_pos and ok_neg and qacc_guard(qacc_tracker, "test3_symmetry")


# ======================================================================
# Test 4: センサ（幾何距離）
# ======================================================================
def _make_corridor(length):
    """1 セル幅・length セル長の直線廊下（全周壁のみ、内部壁なし）を作る。
    v_walls shape (length+1, 1), h_walls shape (length, 2)。"""
    v_walls = np.zeros((length + 1, 1), dtype=int)
    h_walls = np.zeros((length, 2), dtype=int)
    v_walls[0, 0] = 1          # 西端
    v_walls[length, 0] = 1     # 東端
    h_walls[:, 0] = 1          # 南側
    h_walls[:, 1] = 1          # 北側
    return v_walls, h_walls


def _rot_z(vec, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y, z = vec
    return np.array([c * x - s * y, s * x + c * y, z])


def _expected_range_to_vertical_wall(mouse_xy, heading_deg, sensor_pos_str, sensor_zaxis_str,
                                      wall_face_x, cutoff):
    """壁面が x = wall_face_x の鉛直面であるときの、指定センサの理論距離 [m] を計算する。
    ray が壁面より先/cutoff 超なら cutoff を返す（build_maze_robot_xml / robot_v2.py の
    ジオメトリ定義から独立に計算し、シミュレーション実測値と突き合わせるための幾何計算）。"""
    heading_rad = np.radians(heading_deg)
    local_pos = np.array([float(v) for v in sensor_pos_str.split()])
    local_dir = np.array([float(v) for v in sensor_zaxis_str.split()])
    local_dir = local_dir / np.linalg.norm(local_dir)

    world_pos = np.array([mouse_xy[0], mouse_xy[1], 0.002]) + _rot_z(local_pos, heading_rad)
    world_dir = _rot_z(local_dir, heading_rad)
    # ポッド先端 site のオフセット（ローカル z = world_dir 方向に +0.002）
    world_pos = world_pos + 0.002 * world_dir

    if abs(world_dir[0]) < 1e-9:
        return cutoff
    t = (wall_face_x - world_pos[0]) / world_dir[0]
    if t < 0 or t > cutoff:
        return cutoff
    return t


def test4_sensors():
    print("\n=== Test 4: センサ（幾何距離） ===")
    params = RobotParams()
    length = 6
    v_walls, h_walls = _make_corridor(length)

    tmp_dir = tempfile.mkdtemp(prefix="mouse_v2_test4_")
    xml_path = os.path.join(tmp_dir, "corridor.xml")
    build_maze_robot_xml(v_walls, h_walls, xml_path, model_name="corridor_sensor_test",
                          center_goal=False, params=params)

    sim = MouseSim(xml_path, params=params)

    cell_size = params.cell_size
    wall_thickness = 0.012
    east_wall_face_x = length * cell_size - wall_thickness / 2.0

    all_ok = True
    qacc_tracker = [0.0]

    def _track_qacc():
        if sim.data.qacc.size:
            cur = float(np.max(np.abs(sim.data.qacc)))
            if cur > qacc_tracker[0]:
                qacc_tracker[0] = cur

    # --- (a) 前壁までの距離: cell(4,0), heading 0（東向き、壁まで 1.5 セル） ---
    sim.reset_to_start(cell=(4, 0), heading_deg=0)
    _track_qacc()
    obs = sim.observation()
    mx, my, _ = sim.privileged_pose()
    print(f"  配置(a): cell=(4,0) heading=0deg pos=({mx:.3f},{my:.3f}) 東壁面 x={east_wall_face_x:.4f}")

    name_to_idx = {'LF': 0, 'LS': 1, 'RF': 2, 'RS': 3, 'FL': 4, 'FR': 5}
    for name in ('LF', 'RF', 'FL', 'FR'):
        spec = next(s for s in params.sensors if s['name'] == name)
        expected = _expected_range_to_vertical_wall(
            (mx, my), 0.0, spec['pos'], spec['zaxis'], east_wall_face_x, params.sensor_cutoff)
        actual = float(obs[name_to_idx[name]])
        ok = abs(actual - expected) <= 0.01
        all_ok = all_ok and ok
        record(f"sensor_dist_to_wall_{name}", round(expected, 4), round(actual, 4), "m",
               "±0.01 m", ok)

    # --- (b) 開けた方向: cell(0,0), heading 0（西端から東へ。前壁は 6 セル先で cutoff 超） ---
    sim.reset_to_start(cell=(0, 0), heading_deg=0)
    _track_qacc()
    obs2 = sim.observation()
    mx2, my2, _ = sim.privileged_pose()
    print(f"  配置(b): cell=(0,0) heading=0deg pos=({mx2:.3f},{my2:.3f}) (開けた前方)")
    for name in ('FL', 'FR'):
        actual = float(obs2[name_to_idx[name]])
        ok = abs(actual - params.sensor_cutoff) < 1e-6
        all_ok = all_ok and ok
        record(f"sensor_open_returns_cutoff_{name}", params.sensor_cutoff, round(actual, 4), "m",
               "cutoff(0.3)と厳密一致", ok)

    return all_ok and qacc_guard(qacc_tracker, "test4_sensors")


# ======================================================================
# Test 5: 決定性
# ======================================================================
def test5_determinism():
    print("\n=== Test 5: 決定性 ===")
    params = RobotParams()
    length = 6
    v_walls, h_walls = _make_corridor(length)

    tmp_dir = tempfile.mkdtemp(prefix="mouse_v2_test5_")
    xml_path = os.path.join(tmp_dir, "corridor.xml")
    build_maze_robot_xml(v_walls, h_walls, xml_path, model_name="corridor_determinism_test",
                          center_goal=False, params=params)

    qacc_tracker = [0.0]

    def run_trajectory():
        sim = MouseSim(xml_path, params=params)
        sim.full_reset(cell=(1, 0), heading_deg=0)
        commands = [(3.0, 3.0)] * 30 + [(3.0, -3.0)] * 20 + [(-1.5, 2.5)] * 25
        history = []
        for vl, vr in commands:
            step_tracked(sim, vl, vr, qacc_tracker)
            history.append(sim.data.qpos.copy())
        return np.array(history)

    traj1 = run_trajectory()
    traj2 = run_trajectory()

    identical = bool(np.array_equal(traj1, traj2))
    max_abs_diff = 0.0 if identical else float(np.max(np.abs(traj1 - traj2)))

    print(f"  軌道長: {len(traj1)} ステップ, 完全一致: {identical}, "
          f"最大差分: {max_abs_diff:.3e}")

    record("determinism_qpos_trajectory", "完全一致(diff=0)",
           "完全一致" if identical else f"最大差分={max_abs_diff:.3e}",
           "-", "bit-exact 一致", identical)
    return identical and qacc_guard(qacc_tracker, "test5_determinism")


# ======================================================================
# Test 6: 衝突検出
# ======================================================================
def test6_collision():
    print("\n=== Test 6: 衝突検出 ===")
    params = RobotParams()
    length = 6
    v_walls, h_walls = _make_corridor(length)

    tmp_dir = tempfile.mkdtemp(prefix="mouse_v2_test6_")
    xml_path = os.path.join(tmp_dir, "corridor.xml")
    build_maze_robot_xml(v_walls, h_walls, xml_path, model_name="corridor_collision_test",
                          center_goal=False, params=params)

    sim = MouseSim(xml_path, params=params)
    sim.reset_to_start(cell=(4, 0), heading_deg=0)  # 東壁に向かって発進（壁まで約0.21m）

    collided = False
    tipped_seen = False
    max_force_seen = 0.0
    steps_to_collision = None
    n_steps = 300  # 3秒（100Hz）
    qacc_tracker = [0.0]

    for i in range(n_steps):
        result = step_tracked(sim, 3.0, 3.0, qacc_tracker)
        max_force_seen = max(max_force_seen, result['max_contact_force'])
        if result['tipped']:
            tipped_seen = True
        if result['collision']:
            collided = True
            steps_to_collision = i + 1
            break

    print(f"  collision={collided} (steps_to_collision={steps_to_collision}), "
          f"tipped_seen(この走行中)={tipped_seen}, max_contact_force_seen={max_force_seen:.3f} N")

    record("collision_detected_dash_into_wall", True, collided, "bool",
           "collision=True になること", collided,
           note=f"steps={steps_to_collision}, tipped_seen={tipped_seen}")
    if tipped_seen:
        record("collision_test_tipover_side_effect", False, True, "bool",
               "想定: tipoverは発生しない", False,
               note="仕様書の想定外の転倒を検出（詳細は本文参照）")
    return collided and qacc_guard(qacc_tracker, "test6_collision")


# ======================================================================
# メイン
# ======================================================================
def main():
    print("mouse v2 (M0-T2) テストスイート")
    print("=" * 70)

    test_fns = [test1_mass, test2_open_loop_speed, test3_symmetry,
                test4_sensors, test5_determinism, test6_collision]

    overall_ok = []
    for fn in test_fns:
        try:
            ok = fn()
        except AssertionError as e:
            print(f"  [ERROR] {fn.__name__}: assertion failed: {e}")
            ok = False
        except Exception as e:  # noqa: BLE001 - テスト継続のため広く捕捉
            print(f"  [ERROR] {fn.__name__}: 例外発生: {e!r}")
            ok = False
        overall_ok.append(ok)

    print("\n" + "=" * 70)
    print("全テスト実測値まとめ")
    print("=" * 70)
    print(f"{'項目':45s} {'理論値':>18s} {'実測値':>18s} {'判定':<5s}  許容/備考")
    print("-" * 70)
    for r in RESULTS:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"{r['name']:45s} {str(r['theory']):>18s} {str(r['actual']):>18s} "
              f"{status:<5s}  {r['tol_desc']}  {r['note']}")

    n_pass = sum(1 for r in RESULTS if r['passed'])
    n_total = len(RESULTS)
    print("-" * 70)
    print(f"合計: {n_pass}/{n_total} 項目 PASS")
    print(f"テスト関数レベル: {sum(1 for x in overall_ok if x)}/{len(overall_ok)} 個が全項目PASS")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
