"""
tests/test_corridor.py
================
M1（廊下追従・exp_001_corridor）の生成器・環境・報酬の単体テスト。

pytest は使わない plain Python スクリプト（tests/test_mouse_v2.py, tests/test_baseline.py
と同じ流儀）。実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_corridor.py

事前条件: assets/corridor/eval/ に評価コース（seed 3000-3019）が生成済みであること
    .venv/bin/python -m mouse.corridor_gen --split eval --seeds 3000-3019

spec_m1_corridor.md §5 で指定された 6 項目:
  1. 生成器の構造（パス重複なし・端点開通数1/中間2・外周全壁・再現性）
  2. 経路長ポテンシャル（単調減少・終点0・始点(n_cells-1)*0.18）
  3. 報酬の性質（停止時の解析式一致・前進で正・ゴールで+1.0）
  4. 観測（次元 = 距離センサ本数+7・距離[0,1]・reset直後の前ステップ行動(0,0)）
     exp_003 で追加: obs_dist_diff=True のとき 2n+7 次元・reset直後の差分0・
     差分が (d_t − d_(t−1))/0.05 のクリップ値と一致・[-1,1] に収まる
  5. 終了条件（壁接触でterminated+collision、上限ステップでtruncated）
  6. 決定性（同一seed+同一行動列でbit-exact一致）
  7. exp_003 で追加: 学習用コース seed が予約帯（3000-3019, 5000-5019）を
     決定的に読み飛ばす（研究計画書 §9-7）

いずれかのテストで例外/assert が起きても他のテストは継続実行し、最後に全テストの
実測値をまとめて表として print する。
"""
import math
import os
import sys

import mujoco
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mouse.corridor_env import CorridorEnv  # noqa: E402
from mouse.corridor_gen import (  # noqa: E402
    generate_course,
    path_cell_centers,
    path_cumulative_lengths,
    remaining_path_length,
)
from mouse.params import RobotParams  # noqa: E402

EVAL_SEEDS = list(range(3000, 3020))
EVAL_COURSE_DIR = os.path.join(REPO_ROOT, "assets", "corridor", "eval")
GAMMA = 0.995  # PPOのgammaと一致させる値（spec_m1_corridor.md §4）

RESULTS = []  # list[dict(name, expected, actual, passed, note)]


def record(name, expected, actual, passed, note=""):
    RESULTS.append(dict(name=name, expected=expected, actual=actual, passed=bool(passed), note=note))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: expected={expected}, actual={actual} {note}")


def record_exception(name, exc):
    RESULTS.append(dict(name=name, expected="no exception", actual=f"{type(exc).__name__}: {exc}",
                         passed=False, note="例外"))
    print(f"  [FAIL] {name}: 例外発生 {type(exc).__name__}: {exc}")


# ==========================================================================
# 独立再検証ヘルパー（corridor_gen の内部 assert に頼らず、テスト側で
# 独自に v_walls/h_walls/path から構造を検証する）
# ==========================================================================
def _open_degree(v_walls, h_walls, width, height, cx, cy):
    count = 0
    if cx + 1 <= width and v_walls[cx + 1, cy] == 0:
        count += 1
    if cx - 1 >= 0 and v_walls[cx, cy] == 0:
        count += 1
    if cy + 1 <= height and h_walls[cx, cy + 1] == 0:
        count += 1
    if cy - 1 >= 0 and h_walls[cx, cy] == 0:
        count += 1
    return count


def _teleport(env, x, y, heading_deg):
    """env.sim のロボットを任意の連続姿勢へ直接セットする（テスト専用ヘルパ。
    mouse/sim.py は凍結のため、公開APIのmj_name2idでroot jointを解決する）。"""
    sim = env.sim
    root_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'root')
    qpos_adr = sim.model.jnt_qposadr[root_jid]
    heading_rad = math.radians(heading_deg)
    sim.data.qpos[qpos_adr] = x
    sim.data.qpos[qpos_adr + 1] = y
    sim.data.qpos[qpos_adr + 3] = math.cos(heading_rad / 2.0)
    sim.data.qpos[qpos_adr + 4] = 0.0
    sim.data.qpos[qpos_adr + 5] = 0.0
    sim.data.qpos[qpos_adr + 6] = math.sin(heading_rad / 2.0)
    mujoco.mj_forward(sim.model, sim.data)


# ==========================================================================
# テスト1: 生成器の構造
# ==========================================================================
def test_generator_structure():
    print("\n[Test 1] 生成器の構造（seed 3000-3019）")
    all_ok = True
    n_cells_list = []
    n_turns_list = []

    for seed in EVAL_SEEDS:
        course = generate_course(seed)  # 内部 _validate_course も通過している前提
        v_walls, h_walls = course["v_walls"], course["h_walls"]
        path = course["path"]
        width, height = course["width"], course["height"]
        n_cells_list.append(course["n_cells"])
        n_turns_list.append(course["n_turns"])

        dup_ok = len(set(path)) == len(path)
        if not dup_ok:
            all_ok = False
            print(f"  seed={seed}: パス重複あり")

        degree_ok = True
        for i, (cx, cy) in enumerate(path):
            deg = _open_degree(v_walls, h_walls, width, height, cx, cy)
            expected = 1 if (i == 0 or i == len(path) - 1) else 2
            if deg != expected:
                degree_ok = False
                print(f"  seed={seed}: セル{(cx, cy)}(idx{i}) 開通数={deg} 期待={expected}")
        if not degree_ok:
            all_ok = False

        boundary_ok = (np.all(v_walls[0, :] == 1) and np.all(v_walls[width, :] == 1)
                        and np.all(h_walls[:, 0] == 1) and np.all(h_walls[:, height] == 1))
        if not boundary_ok:
            all_ok = False
            print(f"  seed={seed}: 外周に穴あり")

    record("全20コースで パス重複なし・端点1/中間2・外周全壁", True, all_ok, all_ok)

    # 再現性: 同一seedを2回生成して完全一致
    repro_ok = True
    for seed in (EVAL_SEEDS[0], EVAL_SEEDS[len(EVAL_SEEDS) // 2], EVAL_SEEDS[-1]):
        c1 = generate_course(seed)
        c2 = generate_course(seed)
        same = (np.array_equal(c1["v_walls"], c2["v_walls"])
                and np.array_equal(c1["h_walls"], c2["h_walls"])
                and c1["path"] == c2["path"]
                and c1["n_cells"] == c2["n_cells"]
                and c1["n_turns"] == c2["n_turns"])
        if not same:
            repro_ok = False
            print(f"  seed={seed}: 再現性NG")
    record("同一seedの再現性（配列完全一致）", True, repro_ok, repro_ok)

    print(f"  n_cells: min={min(n_cells_list)} max={max(n_cells_list)} "
          f"mean={np.mean(n_cells_list):.1f}")
    print(f"  n_turns: min={min(n_turns_list)} max={max(n_turns_list)} "
          f"mean={np.mean(n_turns_list):.1f}")
    return n_cells_list, n_turns_list


# ==========================================================================
# テスト2: 経路長ポテンシャル
# ==========================================================================
def test_path_potential():
    print("\n[Test 2] 経路長ポテンシャル（seed=3000）")
    params = RobotParams()
    course = generate_course(3000)
    path = course["path"]
    centers = path_cell_centers(path, params.cell_size)
    cum = path_cumulative_lengths(centers)

    remaining = [remaining_path_length(centers, cum, x, y) for x, y in centers]

    monotonic = all(remaining[i] > remaining[i + 1] for i in range(len(remaining) - 1))
    record("各セル中心での残り経路長が単調減少", True, monotonic, monotonic)

    end_ok = abs(remaining[-1] - 0.0) < 1e-6
    record("終点での残り経路長=0", 0.0, remaining[-1], end_ok)

    expected_start = (course["n_cells"] - 1) * params.cell_size
    start_ok = abs(remaining[0] - expected_start) < 1e-6
    record("始点での残り経路長=(n_cells-1)*0.18", expected_start, remaining[0], start_ok)


# ==========================================================================
# テスト3: 報酬の性質
# ==========================================================================
def test_reward_properties():
    print("\n[Test 3] 報酬の性質")

    # --- (a) その場停止: 解析式(gamma*Phi(s')-Phi(s)-0.001)と一致するか ---
    env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA)
    obs, info0 = env.reset(seed=123)
    phi_s = -info0["remaining_m"]
    obs, reward, terminated, truncated, info1 = env.step(np.array([0.0, 0.0], dtype=np.float32))
    phi_sp = -info1["remaining_m"]
    expected = GAMMA * phi_sp - phi_s - 0.001
    if info1["goal"]:
        expected += 1.0
    ok_a = abs(reward - expected) < 1e-8
    record("停止時: 実測報酬が解析式(gamma*Phi'-Phi-0.001)と一致",
           expected, reward, ok_a, f"(Phi_s={phi_s:.6f}, Phi_s'={phi_sp:.6f})")
    env.close()

    # --- (b) 前進すると報酬が正になる ---
    env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA)
    obs, info = env.reset(seed=123)
    max_reward = -np.inf
    got_positive = False
    for _ in range(60):
        obs, reward, terminated, truncated, info = env.step(np.array([0.6, 0.6], dtype=np.float32))
        max_reward = max(max_reward, reward)
        if reward > 0.0:
            got_positive = True
        if terminated or truncated:
            break
    record("前進走行で少なくとも1ステップの報酬が正になる", "> 0", max_reward, got_positive)
    env.close()

    # --- (c) ゴール到達で+1.0が乗る ---
    env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA)
    obs, info0 = env.reset(seed=5)
    course = env.course
    cell_size = env.params.cell_size
    goal_cell = course["path"][-1]
    gx = goal_cell[0] * cell_size + cell_size / 2
    gy = goal_cell[1] * cell_size + cell_size / 2
    heading = 0.0  # 向きは終了判定・報酬に影響しないので何でもよい
    _teleport(env, gx, gy, heading)
    # teleport直後の位置を Phi(s) として使うため、reset直後の info ではなく
    # teleport後の座標から remaining を計算し直す
    from mouse.corridor_gen import remaining_path_length as _rem
    remaining_before = _rem(env._path_centers, env._cum_lengths, gx, gy)
    env._prev_potential = -remaining_before
    obs, reward, terminated, truncated, info1 = env.step(np.array([0.0, 0.0], dtype=np.float32))
    phi_s = -remaining_before
    phi_sp = -info1["remaining_m"]
    expected_c = GAMMA * phi_sp - phi_s - 0.001 + 1.0
    ok_c = terminated and info1["goal"] and abs(reward - expected_c) < 1e-6
    record("ゴール到達: terminated=True, info['goal']=True, 報酬に+1.0",
           expected_c, reward, ok_c,
           f"(terminated={terminated}, goal={info1.get('goal')})")
    env.close()


# ==========================================================================
# テスト4: 観測
# ==========================================================================
def test_observation():
    print("\n[Test 4] 観測")
    env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA)
    obs, info = env.reset(seed=1)

    # 観測次元は仕様（RobotParams.sensors の本数）から導出して照合する。
    # 計画書 r6 でセンサ 6→4 本に変わったため固定値 13 では検査できない。
    n_dist = len(RobotParams().sensors)
    expected_dim = n_dist + 7  # 距離 ×n + ジャイロz + 加速度xy + 車輪2 + 前回行動2
    dim_ok = obs.shape == (expected_dim,)
    record(f"観測次元={expected_dim}（距離{n_dist}本+7）", expected_dim, obs.shape, dim_ok)

    dist = obs[0:n_dist]
    dist_ok = bool(np.all(dist >= 0.0) and np.all(dist <= 1.0))
    record(f"距離成分(obs[0:{n_dist}])が[0,1]", True, dist_ok, dist_ok, f"dist={dist}")

    prev_action_ok = bool(np.allclose(obs[-2:], 0.0))
    record("reset直後の前ステップ行動=(0,0)", (0.0, 0.0), tuple(obs[-2:]), prev_action_ok)

    # 1ステップ後も次元・範囲が保たれるか
    obs2, reward, terminated, truncated, info2 = env.step(np.array([0.3, 0.3], dtype=np.float32))
    dim_ok2 = obs2.shape == (expected_dim,)
    record(f"1ステップ後も観測次元={expected_dim}", expected_dim, obs2.shape, dim_ok2)
    env.close()

    # ---- exp_003: 距離センサの1階差分を加えた観測（obs_dist_diff=True）----
    env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA,
                      obs_dist_diff=True)
    obs, info = env.reset(seed=1)
    expected_dim_d = 2 * n_dist + 7
    dim_ok_d = obs.shape == (expected_dim_d,)
    record(f"差分あり観測次元={expected_dim_d}（距離{n_dist}+差分{n_dist}+7）",
           expected_dim_d, obs.shape, dim_ok_d)

    diff0 = obs[n_dist:2 * n_dist]
    reset_zero_ok = bool(np.allclose(diff0, 0.0))
    record("reset直後の距離差分=0", 0.0, tuple(np.round(diff0, 6)), reset_zero_ok)

    # 差分が「生値の1階差分 / 0.05 をクリップしたもの」と一致するかを、
    # 環境の内部状態に頼らず sim.observation() から独立に再計算して照合する。
    dist_raw_before = np.asarray(env.sim.observation()[0:n_dist], dtype=np.float64)
    obs2, reward, terminated, truncated, info2 = env.step(np.array([0.5, 0.5], dtype=np.float32))
    dist_raw_after = np.asarray(env.sim.observation()[0:n_dist], dtype=np.float64)
    expected_diff = np.clip((dist_raw_after - dist_raw_before) / 0.05, -1.0, 1.0)
    actual_diff = np.asarray(obs2[n_dist:2 * n_dist], dtype=np.float64)
    diff_ok = bool(np.allclose(actual_diff, expected_diff, atol=1e-6))
    record("距離差分が (d_t − d_(t−1))/0.05 のクリップ値と一致",
           tuple(np.round(expected_diff, 5)), tuple(np.round(actual_diff, 5)), diff_ok)

    # 差分成分が [-1,1] に収まる（クリップが効いている）
    in_range = True
    for _ in range(50):
        obs2, reward, terminated, truncated, info2 = env.step(
            np.array([1.0, 1.0], dtype=np.float32))
        d = obs2[n_dist:2 * n_dist]
        if not (np.all(d >= -1.0) and np.all(d <= 1.0)):
            in_range = False
            break
        if terminated or truncated:
            break
    record("距離差分が常に[-1,1]", True, in_range, in_range)

    # 距離成分そのものは差分あり/なしで同一（挿入位置が正しいことの確認）
    env.close()
    env_a = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA)
    env_b = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA,
                        obs_dist_diff=True)
    oa, _ = env_a.reset(seed=1)
    ob, _ = env_b.reset(seed=1)
    same_head = bool(np.allclose(oa[0:n_dist], ob[0:n_dist]))
    same_tail = bool(np.allclose(oa[n_dist:], ob[2 * n_dist:]))
    record("差分の挿入で距離成分・後続成分が変わらない", True, same_head and same_tail,
           same_head and same_tail)
    env_a.close()
    env_b.close()


# ==========================================================================
# テスト5: 終了条件
# ==========================================================================
def test_termination():
    print("\n[Test 5] 終了条件")

    # --- 壁接触: スタートセルは進行方向以外の3方向が壁 → 後退させると即衝突 ---
    env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA)
    obs, info = env.reset(seed=42)
    collided = False
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(np.array([-1.0, -1.0], dtype=np.float32))
        if terminated:
            collided = info["collision"]
            break
    record("後退させると壁接触でterminated かつ info['collision']=True",
           True, collided, collided)
    env.close()

    # --- 時間切れ: _max_steps を強制的に小さくしてtruncatedを発火させる ---
    env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA)
    obs, info = env.reset(seed=7)
    env._max_steps = 2
    truncated_flag = False
    terminated_flag = False
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        terminated_flag = terminated_flag or terminated
        truncated_flag = truncated_flag or truncated
    ok = truncated_flag and not terminated_flag
    record("_max_steps到達でtruncated=True(terminatedは伴わない)",
           True, ok, ok, f"(terminated={terminated_flag}, truncated={truncated_flag})")
    env.close()


# ==========================================================================
# テスト6: 決定性
# ==========================================================================
def test_determinism():
    print("\n[Test 6] 決定性（同一seed + 同一行動列でbit-exact一致）")
    action_seq = [np.array([0.4, 0.35], dtype=np.float32),
                  np.array([0.2, 0.5], dtype=np.float32),
                  np.array([-0.1, 0.3], dtype=np.float32),
                  np.array([0.5, 0.5], dtype=np.float32),
                  np.array([0.0, 0.0], dtype=np.float32)] * 6  # 30ステップ

    def rollout():
        env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA)
        obs, info = env.reset(seed=999)
        obs_list = [obs.copy()]
        reward_list = []
        for a in action_seq:
            obs, reward, terminated, truncated, info = env.step(a)
            obs_list.append(obs.copy())
            reward_list.append(reward)
            if terminated or truncated:
                break
        env.close()
        return obs_list, reward_list

    obs_list_1, reward_list_1 = rollout()
    obs_list_2, reward_list_2 = rollout()

    same_len = len(obs_list_1) == len(obs_list_2)
    obs_ok = same_len and all(np.array_equal(a, b) for a, b in zip(obs_list_1, obs_list_2))
    reward_ok = same_len and all(a == b for a, b in zip(reward_list_1, reward_list_2))

    record("2回のrolloutで観測列がbit-exact一致", True, obs_ok, obs_ok,
           f"(n_steps_1={len(obs_list_1)}, n_steps_2={len(obs_list_2)})")
    record("2回のrolloutで報酬列も完全一致", True, reward_ok, reward_ok)


# ==========================================================================
# テスト7: 予約 seed の読み飛ばし（研究計画書 §9-7 の実装保証）
# ==========================================================================
def test_reserved_seed_skip():
    print("\n[Test 7] 学習用コース seed が予約帯（3000-3019, 5000-5019）を飛ばす")

    # 予約帯の直前から始めて、実際に生成されるコース seed 列を観測する
    env = CorridorEnv(mode="generate", base_seed=2998, gamma=GAMMA)
    seen = []
    for _ in range(4):
        _obs, info = env.reset(seed=0)
        seen.append(int(info["course_seed"]))
    env.close()
    expected = [2998, 2999, 3020, 3021]
    ok = seen == expected
    record("生成 seed 列が gate 帯 3000-3019 を飛ばす", expected, seen, ok)

    # 予約帯に一切入らないこと（帯をまたぐ十分な本数を確認）
    env = CorridorEnv(mode="generate", base_seed=4998, gamma=GAMMA)
    seen2 = []
    for _ in range(4):
        _obs, info = env.reset(seed=0)
        seen2.append(int(info["course_seed"]))
    env.close()
    expected2 = [4998, 4999, 5020, 5021]
    ok2 = seen2 == expected2
    record("生成 seed 列が検証帯 5000-5019 を飛ばす", expected2, seen2, ok2)

    # 読み飛ばし後も seed だけで決まる（再現性が失われていない）
    env = CorridorEnv(mode="generate", base_seed=2998, gamma=GAMMA)
    seen3 = []
    for _ in range(4):
        _obs, info = env.reset(seed=0)
        seen3.append(int(info["course_seed"]))
    env.close()
    ok3 = seen3 == seen
    record("同一 base_seed で seed 列が再現する", seen, seen3, ok3)


# ==========================================================================
# テスト8: ポテンシャルのオフセット（exp_004）
# ==========================================================================
def test_potential_offset():
    print("\n[Test 8] ポテンシャルのオフセット Φ' = D₀ − D（exp_004）")

    def stop_rewards(potential_offset, n_steps=20):
        """その場停止（電圧 0）を n_steps 続けたときの報酬列を返す。"""
        env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA,
                          potential_offset=potential_offset)
        env.reset(seed=123)
        d0 = env._cum_lengths[-1]
        rewards = []
        for _ in range(n_steps):
            _obs, r, term, trunc, _info = env.step(np.array([0.0, 0.0], dtype=np.float32))
            rewards.append(r)
            if term or trunc:
                break
        env.close()
        return np.array(rewards), d0

    r_base, d0 = stop_rewards(False)
    r_off, d0b = stop_rewards(True)

    # (a) 現行式（既定）では滞留の報酬が正 = exp_003 で滞留解に落ちた原因
    base_positive = bool(np.all(r_base > 0.0))
    record("既定(Φ=−D)では滞留の報酬が正（局所解の源）", "> 0",
           f"{r_base.min():.6f}〜{r_base.max():.6f}", base_positive)

    # (b) オフセット版では滞留の報酬が負
    off_negative = bool(np.all(r_off < 0.0))
    record("Φ'=D₀−D では滞留の報酬が負", "< 0",
           f"{r_off.min():.6f}〜{r_off.max():.6f}", off_negative)

    # (c) 2 つの式の報酬差は毎ステップ一律 (γ−1)·D₀（＝時間罰を上げるのと等価）
    expected_diff = (GAMMA - 1.0) * d0
    actual_diff = r_off - r_base
    diff_ok = bool(np.allclose(actual_diff, expected_diff, atol=1e-9)) and abs(d0 - d0b) < 1e-12
    record("報酬差が毎ステップ一律 (γ−1)·D₀ になる", f"{expected_diff:.8f}",
           f"{actual_diff.min():.8f}〜{actual_diff.max():.8f}", diff_ok,
           f"(D₀={d0:.4f} m, 実効時間罰 0.001+{(1-GAMMA)*d0:.4f})")

    # (d) 物理は変わらない（同一 seed・同一行動なら軌跡が一致する）
    def stop_positions(potential_offset):
        env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA,
                          potential_offset=potential_offset)
        env.reset(seed=123)
        poses = []
        for _ in range(20):
            env.step(np.array([0.4, 0.4], dtype=np.float32))
            poses.append(env.sim.privileged_pose())
        env.close()
        return np.array(poses)

    same_traj = bool(np.array_equal(stop_positions(False), stop_positions(True)))
    record("報酬式の切り替えで軌跡（物理）は変わらない", True, same_traj, same_traj)


# ==========================================================================
# テスト9: 衝突罰（exp_005）
# ==========================================================================
def test_collision_penalty():
    print("\n[Test 9] 衝突罰 collision_penalty（exp_005）")

    def crash_run(collision_penalty):
        """後退させて壁に当たるまで走り、最終ステップの報酬と終了様式を返す。"""
        env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA,
                          potential_offset=True, collision_penalty=collision_penalty)
        env.reset(seed=42)
        last_reward, info = None, None
        for _ in range(200):
            _obs, r, term, trunc, info = env.step(np.array([-1.0, -1.0], dtype=np.float32))
            last_reward = r
            if term or trunc:
                break
        env.close()
        return last_reward, info

    r_none, info_none = crash_run(0.0)
    r_pen, info_pen = crash_run(-1.0)

    crashed = bool(info_none.get("collision")) and bool(info_pen.get("collision"))
    record("後退させると衝突で終了する（前提の確認）", True, crashed, crashed)

    diff = r_pen - r_none
    diff_ok = abs(diff - (-1.0)) < 1e-9
    record("衝突時の報酬に collision_penalty がそのまま乗る", -1.0, f"{diff:.9f}", diff_ok,
           f"(罰なし={r_none:.6f}, 罰あり={r_pen:.6f})")

    # 既定は 0.0（従来と同じ）＝ 退行がないこと
    env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA)
    default_ok = env.collision_penalty == 0.0
    env.close()
    record("collision_penalty の既定は 0.0（従来の報酬を再現）", 0.0,
           env.collision_penalty, default_ok)

    # 衝突しなかったステップには罰が乗らない（前進中の報酬が一致する）
    def forward_rewards(collision_penalty, n=10):
        env = CorridorEnv(course_dir=EVAL_COURSE_DIR, course_seeds=[3000], gamma=GAMMA,
                          potential_offset=True, collision_penalty=collision_penalty)
        env.reset(seed=42)
        rs = []
        for _ in range(n):
            _obs, r, term, trunc, _info = env.step(np.array([0.4, 0.4], dtype=np.float32))
            rs.append(r)
            if term or trunc:
                break
        env.close()
        return np.array(rs)

    same = bool(np.array_equal(forward_rewards(0.0), forward_rewards(-1.0)))
    record("衝突しないステップの報酬は罰の有無で変わらない", True, same, same)


# ==========================================================================
# メイン
# ==========================================================================
def main():
    if not os.path.isdir(EVAL_COURSE_DIR):
        print(f"[ERROR] {EVAL_COURSE_DIR} が存在しません。先に以下を実行してください:")
        print("  .venv/bin/python -m mouse.corridor_gen --split eval --seeds 3000-3019")
        return 1

    try:
        test_generator_structure()
    except Exception as e:  # noqa: BLE001
        record_exception("test_generator_structure", e)

    try:
        test_path_potential()
    except Exception as e:  # noqa: BLE001
        record_exception("test_path_potential", e)

    try:
        test_reward_properties()
    except Exception as e:  # noqa: BLE001
        record_exception("test_reward_properties", e)

    try:
        test_observation()
    except Exception as e:  # noqa: BLE001
        record_exception("test_observation", e)

    try:
        test_termination()
    except Exception as e:  # noqa: BLE001
        record_exception("test_termination", e)

    try:
        test_determinism()
    except Exception as e:  # noqa: BLE001
        record_exception("test_determinism", e)

    try:
        test_reserved_seed_skip()
    except Exception as e:  # noqa: BLE001
        record_exception("test_reserved_seed_skip", e)

    try:
        test_potential_offset()
    except Exception as e:  # noqa: BLE001
        record_exception("test_potential_offset", e)

    try:
        test_collision_penalty()
    except Exception as e:  # noqa: BLE001
        record_exception("test_collision_penalty", e)

    print("\n" + "=" * 78)
    print("テスト結果一覧")
    print("=" * 78)
    n_pass = sum(1 for r in RESULTS if r["passed"])
    n_total = len(RESULTS)
    for r in RESULTS:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['name']}")
    print("=" * 78)
    print(f"合計: {n_pass}/{n_total} PASS")
    print("=" * 78)

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
