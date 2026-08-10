"""
tests/test_env_hooks.py
================
mouse/env.py の MouseMazeEnvV2 に実装したセンサ劣化フック（U3 対応、
docs/MODEL_VERIFICATION_PLAN.md §5 手順3）の単体・結合テスト。

pytest は使わない plain Python スクリプト（tests/test_mouse_v2.py と同じ流儀）。
実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_env_hooks.py

検証項目（計画書 §5 手順3 の要求どおり）:
  1. 既定設定（全パラメータ None）で 200 ステップの観測列がフック無し経路
     （独立に構築した MouseSim を env.step() と同じ電圧換算・行動系列で
     直接駆動した結果）と bit-exact 一致すること（np.array_equal）
  2. 量子化のみ有効 → round(x/step)*step の期待どおりの離散値になること
  3. 遅れのみ有効 → ステップ応答が一次遅れ理論値（y[k]=α x[k]+(1-α) y[k-1]、
     α=dt/(τ+dt)。計画書 §5 手順3 要件4 の定義そのもの）と一致すること
  4. むだ時間のみ有効 → k ステップ前の値（コールドスタート中はリセット時
     観測値）と一致すること
  5. ノイズのみ有効 → 同一 seed で再現し、統計量（標本標準偏差）が
     指定値と整合すること
  6. seed 付き reset() の再現性（物理+全フック合成の end-to-end 一致）

いずれかのテストで例外/assert が起きても他のテストは継続実行し、
最後に全テストの結果表をまとめて print する（tests/test_mouse_v2.py と同じ方針）。
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mouse.env import MouseMazeEnvV2, SensorDegradation
from mouse.sim import MouseSim
from mouse.params import RobotParams

XML_PATH = os.path.join(REPO_ROOT, 'assets', 'mouse_v2.xml')


# ======================================================================
# 結果収集ヘルパー（tests/test_mouse_v2.py と同じ流儀）
# ======================================================================
RESULTS = []  # list[dict(name, passed, detail)]


def record(name, passed, detail=""):
    RESULTS.append(dict(name=name, passed=bool(passed), detail=detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}  {detail}")


def _action_sequence(n, seed):
    """テスト用の再現可能な擬似ランダム行動系列（[-1,1]^2）。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 2))


# ======================================================================
# Test 1: 既定設定（全パラメータ None）で 200 ステップ bit-exact 一致
# ======================================================================
def test1_bitexact_default():
    print("\n=== Test 1: 既定設定 bit-exact 一致 ===")
    n_steps = 200
    actions = _action_sequence(n_steps, seed=1)

    # --- env 経路（sensor_degradation 既定 None） ---
    env = MouseMazeEnvV2(XML_PATH, seed=42)
    obs_env, _ = env.reset(seed=42)

    # --- フック無し参照経路: 独立構築した MouseSim を env.step() と同じ
    #     電圧換算 (action * voltage_limit) で直接駆動する ---
    params = RobotParams()
    sim_ref = MouseSim(XML_PATH, params=params, seed=42)
    sim_ref.full_reset()
    obs_ref = sim_ref.observation().astype(np.float32)

    reset_match = bool(np.array_equal(obs_env, obs_ref))
    record("reset_obs_bitexact_vs_raw_sim", reset_match,
           f"env.reset()={obs_env[:3]}... raw_sim={obs_ref[:3]}...")

    all_match = reset_match
    max_diff = 0.0
    for t in range(n_steps):
        a = actions[t]
        obs_env_t, _, term_env, trunc_env, _ = env.step(a)

        v_left = float(np.clip(a[0], -1.0, 1.0)) * params.voltage_limit
        v_right = float(np.clip(a[1], -1.0, 1.0)) * params.voltage_limit
        sim_ref.step_control(v_left, v_right)
        obs_ref_t = sim_ref.observation().astype(np.float32)

        match = np.array_equal(obs_env_t, obs_ref_t)
        all_match = all_match and match
        if not match:
            max_diff = max(max_diff, float(np.max(np.abs(
                obs_env_t.astype(np.float64) - obs_ref_t.astype(np.float64)))))
        if term_env:
            # 衝突/転倒で終了したら参照側もそこで打ち切り（それ以降は比較不能）
            break

    record("step_obs_bitexact_vs_raw_sim_200steps", all_match,
           f"max_diff={max_diff:.3e}" if not all_match else "完全一致")

    # --- SensorDegradation() を明示的に渡した場合（全フィールド None）も
    #     同じく bit-exact であること（is_noop() 経路の確認） ---
    env2 = MouseMazeEnvV2(XML_PATH, seed=42, sensor_degradation=SensorDegradation())
    obs_env2, _ = env2.reset(seed=42)
    explicit_noop_match = bool(np.array_equal(obs_env2, obs_ref))
    record("explicit_noop_SensorDegradation_bitexact", explicit_noop_match)

    return all_match and reset_match and explicit_noop_match


# ======================================================================
# Test 2: 量子化のみ有効
# ======================================================================
def test2_quantize_only():
    print("\n=== Test 2: 量子化のみ有効 ===")
    ok_all = True

    env = MouseMazeEnvV2(XML_PATH, seed=2)
    obs0, _ = env.reset(seed=2)
    obs_dim = obs0.shape[0]

    # チャネルごとに異なる step を指定（0 のチャネルは量子化なし＝素通しの確認を兼ねる）
    step = np.zeros(obs_dim, dtype=np.float64)
    step[0:6] = 0.01     # 距離センサ: 1cm 刻み
    step[6:9] = 0.1       # accel: 0.1 刻み
    step[9:12] = 0.0      # gyro: 量子化なし（素通し確認）
    step[12:14] = 0.5     # wheel omega: 0.5 刻み

    env.sensor_degradation = SensorDegradation(quantize_step=step)

    rng = np.random.default_rng(123)
    raw = rng.uniform(-1.0, 1.0, size=obs_dim)
    quantized = env._apply_sensor_degradation(raw)

    expected = raw.copy()
    mask = step != 0
    expected[mask] = np.round(raw[mask] / step[mask]) * step[mask]

    exact_match = bool(np.array_equal(quantized, expected))
    record("quantize_direct_call_matches_formula", exact_match)
    # step==0 チャネル(gyro)は raw と厳密に変化なし
    passthrough_ok = bool(np.array_equal(quantized[9:12], raw[9:12]))
    record("quantize_step0_channels_passthrough", passthrough_ok)
    ok_all = ok_all and exact_match and passthrough_ok

    # 実機統合: env.step() を数ステップ回し、量子化チャネルが厳密に
    # step の整数倍になっていることを確認（float32 キャストの丸め誤差を考慮）
    all_multiple_ok = True
    for _ in range(10):
        a = np.random.default_rng(0).uniform(-1, 1, size=2)
        obs_t, *_ = env.step(a)
        obs_t64 = obs_t.astype(np.float64)
        for c in range(obs_dim):
            if step[c] == 0:
                continue
            ratio = obs_t64[c] / step[c]
            if abs(ratio - round(ratio)) > 1e-4:
                all_multiple_ok = False
    record("quantize_integration_step_multiples", all_multiple_ok)
    ok_all = ok_all and all_multiple_ok

    return ok_all


# ======================================================================
# Test 3: 一次遅れのみ有効 — ステップ応答が理論式と一致
# ======================================================================
def test3_lag_only():
    print("\n=== Test 3: 一次遅れのみ有効 ===")
    tau = 0.05  # [s]
    env = MouseMazeEnvV2(XML_PATH, seed=3)
    obs0, _ = env.reset(seed=3)
    obs_dim = obs0.shape[0]
    dt = env.params.control_dt

    env.sensor_degradation = SensorDegradation(lag_tau=tau)
    # 遅延初期化を発火させるため一度呼ぶ（reset() 直後の観測を y[-1] にする）
    _ = env._apply_sensor_degradation(obs0.astype(np.float64))
    y0 = env._degradation_state['lag_state'].copy()

    alpha = dt / (tau + dt)
    alpha_expected = env._degradation_state['lag_alpha']
    alpha_match = bool(np.allclose(alpha_expected, alpha))
    record("lag_alpha_formula", alpha_match, f"alpha={alpha:.6f}")

    # ステップ入力: x_ss = y0 + delta（全チャネル一律オフセット）
    x_ss = y0 + 1.0
    n_steps = 60
    y_prev = y0.copy()
    exact_match = True
    max_diff = 0.0
    trajectory = []
    for k in range(n_steps):
        y_expected = alpha * x_ss + (1.0 - alpha) * y_prev
        y_actual = env._apply_sensor_degradation(x_ss.copy())
        d = float(np.max(np.abs(y_actual - y_expected)))
        max_diff = max(max_diff, d)
        if d > 1e-9:
            exact_match = False
        y_prev = y_expected
        trajectory.append(y_actual[0])

    record("lag_stepresponse_matches_recursion_formula", exact_match,
           f"max_diff={max_diff:.3e} (60 steps)")

    # 閉形式 y[k] = x_ss + (y0-x_ss)*(1-alpha)^k との突合せ（同じ再帰の別表現）
    k_check = 20
    closed_form = x_ss[0] + (y0[0] - x_ss[0]) * (1.0 - alpha) ** k_check
    closed_form_match = abs(trajectory[k_check - 1] - closed_form) < 1e-6
    record("lag_closed_form_geometric_decay", closed_form_match,
           f"closed_form={closed_form:.6f}, actual={trajectory[k_check-1]:.6f}")

    # 参考: 連続時間一次遅れ e^{-t/tau} との大まかな整合（後退差分離散化のため
    # 数%〜十数%程度の乖離は理論的に生じる。目安として緩めの許容で確認する）
    t = k_check * dt
    continuous = x_ss[0] - (x_ss[0] - y0[0]) * np.exp(-t / tau)
    rel_err = abs(trajectory[k_check - 1] - continuous) / abs(x_ss[0] - y0[0])
    continuous_reasonable = rel_err < 0.25
    record("lag_vs_continuous_time_sanity", continuous_reasonable,
           f"discrete={trajectory[k_check-1]:.4f} continuous={continuous:.4f} "
           f"rel_err={rel_err*100:.1f}% (許容 25%, 後退差分離散化のため)")

    return alpha_match and exact_match and closed_form_match and continuous_reasonable


# ======================================================================
# Test 4: むだ時間のみ有効 — k ステップ前の値と一致
# ======================================================================
def test4_delay_only():
    print("\n=== Test 4: むだ時間のみ有効 ===")
    env = MouseMazeEnvV2(XML_PATH, seed=4)
    obs0, _ = env.reset(seed=4)
    obs_dim = obs0.shape[0]

    # チャネルごとに異なる delay（0..obs_dim-1）を割り当て、FIFO のチャネル別
    # 独立性を同時に確認する
    delay = np.arange(obs_dim, dtype=np.int64)
    env.sensor_degradation = SensorDegradation(delay_steps=delay)

    # 遅延初期化発火（reset 時観測でバッファをコールドスタート充填）
    y0 = env._apply_sensor_degradation(obs0.astype(np.float64)).copy()
    # delay=0 のチャネル(0番)は即座に現在値が返る一方、delay>0 のチャネルは
    # この最初の呼び出し時点(k=0)ではまだ reset 時観測のままのはず
    reset_fill = env._degradation_state['delay_buffer'][0].copy()

    n_steps = obs_dim + 5
    synthetic_history = []  # synthetic_history[k][c] = 1000*k + c
    outputs = []
    for k in range(n_steps):
        raw_k = np.array([1000.0 * k + c for c in range(obs_dim)], dtype=np.float64)
        synthetic_history.append(raw_k)
        out_k = env._apply_sensor_degradation(raw_k).copy()
        outputs.append(out_k)

    all_ok = True
    n_checked = 0
    for k in range(n_steps):
        for c in range(obs_dim):
            d = int(delay[c])
            if k >= d:
                expected = 1000.0 * (k - d) + c
            else:
                expected = reset_fill[c]
            actual = outputs[k][c]
            n_checked += 1
            if actual != expected:
                all_ok = False

    record("delay_fifo_matches_k_steps_prior", all_ok,
           f"{n_checked} 個のチャネル×ステップの組合せを厳密一致で検証")

    # delay=0 のチャネルは常に現在値そのもの（コールドスタートの影響を受けない）
    zero_delay_ok = all(outputs[k][0] == 1000.0 * k for k in range(n_steps))
    record("delay0_channel_is_passthrough", zero_delay_ok)

    return all_ok and zero_delay_ok


# ======================================================================
# Test 5: ノイズのみ有効 — 再現性 + 統計量整合
# ======================================================================
def test5_noise_only():
    print("\n=== Test 5: ノイズのみ有効 ===")
    sigma_scalar = 0.05
    n_samples = 20000

    def run(seed):
        env = MouseMazeEnvV2(XML_PATH, seed=seed)
        obs0, _ = env.reset(seed=seed)
        obs_dim = obs0.shape[0]
        noise_std = np.full(obs_dim, sigma_scalar)
        noise_std[3] = 0.0  # 1チャネルだけノイズなしにして pass-through を確認
        env.sensor_degradation = SensorDegradation(noise_std=noise_std)
        zeros = np.zeros(obs_dim, dtype=np.float64)
        samples = np.stack([env._apply_sensor_degradation(zeros).copy()
                             for _ in range(n_samples)])
        return samples, obs_dim

    samples_a, obs_dim = run(seed=999)

    # --- 統計量整合: 標本標準偏差が指定 std と一致 ---
    sample_std = samples_a.std(axis=0)
    rel_err = np.abs(sample_std - sigma_scalar) / sigma_scalar
    # channel 3 は sigma=0 指定なので比較対象から除く
    rel_err_active = np.delete(rel_err, 3)
    std_ok = bool(np.all(rel_err_active < 0.08))
    record("noise_sample_std_matches_sigma", std_ok,
           f"mean_rel_err={rel_err_active.mean()*100:.2f}% (n={n_samples}, 許容8%)")

    zero_channel_exact = bool(np.all(samples_a[:, 3] == 0.0))
    record("noise_std0_channel_exact_zero", zero_channel_exact)

    sample_mean_ok = bool(np.all(np.abs(samples_a.mean(axis=0)) < 3 * sigma_scalar / np.sqrt(n_samples) * 5))
    record("noise_sample_mean_near_zero", sample_mean_ok,
           f"max|mean|={np.max(np.abs(samples_a.mean(axis=0))):.4f}")

    # --- 同一 seed での再現性 ---
    samples_b, _ = run(seed=999)
    reproducible = bool(np.array_equal(samples_a, samples_b))
    record("noise_reproducible_same_seed", reproducible)

    # --- 異なる seed では系列が変わること（RNG が実際に機能している確認） ---
    samples_c, _ = run(seed=1000)
    differs = not bool(np.array_equal(samples_a, samples_c))
    record("noise_differs_across_seeds", differs)

    return std_ok and zero_channel_exact and sample_mean_ok and reproducible and differs


# ======================================================================
# Test 6: seed 付き reset() の再現性（物理 + 全フック合成の end-to-end）
# ======================================================================
def test6_seeded_reset_reproducibility():
    print("\n=== Test 6: seed 付き reset() の再現性（全フック合成） ===")
    n_steps = 100
    seed = 777

    def make_env():
        env = MouseMazeEnvV2(XML_PATH, seed=seed)
        obs0, _ = env.reset(seed=seed)
        obs_dim = obs0.shape[0]
        env.sensor_degradation = SensorDegradation(
            quantize_step=0.005,
            lag_tau=0.03,
            delay_steps=np.array([0, 1, 2] + [0] * (obs_dim - 3), dtype=np.int64),
            noise_std=0.01,
        )
        return env

    actions = _action_sequence(n_steps, seed=55)

    def run_trajectory():
        env = make_env()
        obs0, _ = env.reset(seed=seed)  # sensor_degradation 設定後に再度 reset
        traj = [obs0.copy()]
        for t in range(n_steps):
            obs_t, _, term, trunc, _ = env.step(actions[t])
            traj.append(obs_t.copy())
            if term or trunc:
                break
        return np.stack(traj)

    traj1 = run_trajectory()
    traj2 = run_trajectory()

    same_instance_reproducible = bool(np.array_equal(traj1, traj2))
    record("seeded_reset_reproducible_across_calls", same_instance_reproducible,
           f"trajectory length={len(traj1)}")

    return same_instance_reproducible


# ======================================================================
# メイン
# ======================================================================
def main():
    print("mouse/env.py センサ劣化フック (U3) テストスイート")
    print("=" * 70)

    test_fns = [test1_bitexact_default, test2_quantize_only, test3_lag_only,
                test4_delay_only, test5_noise_only, test6_seeded_reset_reproducibility]

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
    print("全テスト結果まとめ")
    print("=" * 70)
    for r in RESULTS:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {r['name']}  {r['detail']}")

    n_pass = sum(1 for r in RESULTS if r['passed'])
    n_total = len(RESULTS)
    print("-" * 70)
    print(f"合計: {n_pass}/{n_total} 項目 PASS")
    print(f"テスト関数レベル: {sum(1 for x in overall_ok if x)}/{len(overall_ok)} 個が全項目PASS")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
