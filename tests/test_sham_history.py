#!/usr/bin/env python3
"""mouse/obs_history.py（観測履歴の連結・exp_021）の sham 引数（にせ履歴・exp_022）の単体テスト。

`ObsHistoryWrapper(env, lags, sham=True)` は、遅れの位置にも**現在の観測を複製**する
（次元・ブロック数は sham=False と同じ、履歴として運ぶ情報だけがゼロになる）。
`tests/test_obs_history.py` に対して、sham 引数固有の振る舞いだけを検査する。

| # | 検査 | 要点 |
|---|---|---|
| S1 | sham=True: 全ブロックが現在の素の観測と bit 一致（reset 直後・k 歩後の両方） | obs_history.py `_stack()` |
| S2 | sham の有無で observation_space.shape が完全に同じ | 同上 |
| S3 | sham=True でも素の env の状態遷移が bit 一致する（乱数を消費しない） | 同上 |
| S4 | sham=True の出力に過去の観測が 1 要素も混ざらない | 同上 |
| S5 | sham の既定は False。sham=False の出力は sham 省略時と bit 一致 | 同上 |
| S6 | sham の有無で、素の env に渡す行動列が同じなら素の env の軌跡が bit 一致 | 同上 |

pytest は使わない plain Python スクリプト（tests/test_obs_history.py と同じ流儀）。
実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_sham_history.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from mouse.maze6_env import Maze6Env  # noqa: E402
from mouse.obs_history import ObsHistoryWrapper  # noqa: E402

# 学習環境 v2 のフラグ（tests/test_obs_history.py の V2 と同一）。
V2 = dict(goal_rule_containment=True, collision_respawn=True, episode_limit_steps=2000)

# テストで使う遅れの組（tests/test_obs_history.py の LAGS と同一。機構の検査に十分な小さい組）。
LAGS = (1, 2, 5, 10)

# 前進し続けて観測を毎歩変化させる方策（tests/test_obs_history.py の各検査と同じ狙い）。
FORWARD = np.array([0.8, 0.8], dtype=np.float64)


def make_env(base_seed: int, **kwargs) -> Maze6Env:
    """学習環境 v2 の Maze6Env を作る（mode='generate'、学習用 seed 帯 8000 以降。
    予約帯 6000-6019・7000-7019・7100-7299 は使わない）。"""
    cfg = dict(V2)
    cfg.update(kwargs)
    return Maze6Env(mode="generate", base_seed=base_seed, gamma=0.995, **cfg)


# ======================================================================
# S1: sham=True の全ブロックが「現在の」素の観測と bit 一致（reset 直後・k 歩後）
# ======================================================================
def s1_sham_blocks_match_current() -> bool:
    print("\n[S1] sham=True: 全ブロックが現在の素の観測と bit 一致（reset 直後・k 歩後）")
    seed = 11
    base_seed = 9010
    K = max(LAGS) + 10

    raw_env = make_env(base_seed)
    wrap_base = make_env(base_seed)
    wrapped = ObsHistoryWrapper(wrap_base, LAGS, sham=True)

    o_raw, _ = raw_env.reset(seed=seed)
    o_wrap, _ = wrapped.reset(seed=seed)
    o_raw = np.asarray(o_raw, dtype=np.float32)

    # 空振り防止: 素の観測の要素が全て同じ値でないこと
    const_obs = bool(np.all(o_raw == o_raw[0]))
    print(f"  素の観測が定数でない（空振り防止）: {not const_obs}"
          f"（最小 {o_raw.min():.4f}・最大 {o_raw.max():.4f}）")

    n = raw_env.observation_space.shape[0]
    n_blocks = 1 + len(LAGS)

    def all_blocks_match(o_wrap_, o_raw_, tag) -> bool:
        oks = [np.array_equal(o_wrap_[i * n:(i + 1) * n], o_raw_) for i in range(n_blocks)]
        print(f"  {tag}: 全 {n_blocks} ブロックが現在の素の観測と bit 一致 = {all(oks)}"
              f"（内訳 {oks}）")
        return all(oks)

    reset_ok = all_blocks_match(o_wrap, o_raw, "reset 直後")

    n_steps_done = 0
    for _t in range(K):
        o_raw, _r1, term1, trunc1, _i1 = raw_env.step(FORWARD)
        o_wrap, _r2, term2, trunc2, _i2 = wrapped.step(FORWARD)
        o_raw = np.asarray(o_raw, dtype=np.float32)
        n_steps_done += 1
        if term1 or trunc1 or term2 or trunc2:
            break
    k_ok = all_blocks_match(o_wrap, o_raw, f"{n_steps_done} 歩後")

    raw_env.close()
    wrapped.close()
    return (not const_obs) and reset_ok and k_ok


# ======================================================================
# S2: sham の有無で observation_space.shape が完全に同じ
# ======================================================================
def s2_shape_unchanged_by_sham() -> bool:
    print("\n[S2] sham の有無で observation_space.shape が完全に同じ")
    env_true = make_env(9020)
    env_false = make_env(9021)
    wrapped_true = ObsHistoryWrapper(env_true, LAGS, sham=True)
    wrapped_false = ObsHistoryWrapper(env_false, LAGS, sham=False)
    shape_true = wrapped_true.observation_space.shape
    shape_false = wrapped_false.observation_space.shape
    print(f"  sham=True  の shape = {shape_true}")
    print(f"  sham=False の shape = {shape_false}")
    ok = shape_true == shape_false
    wrapped_true.close()
    wrapped_false.close()
    return ok


# ======================================================================
# S3: sham=True でも素の env の状態遷移が bit 一致（乱数を消費しない）
# ======================================================================
def s3_sham_true_no_rng_consumption() -> bool:
    print("\n[S3] sham=True でも素の env の状態遷移が bit 一致する（乱数を消費しない）")
    base_seed = 9030
    seed_a, seed_b = 555, 777

    # 空振り防止: 初期擾乱が実際に効いていること（seed を変えると観測が変わる）
    e1 = make_env(base_seed)
    o1, _ = e1.reset(seed=seed_a)
    e2 = make_env(base_seed)
    o2, _ = e2.reset(seed=seed_b)
    seed_effect = not np.array_equal(o1, o2)
    print(f"  seed を変えると観測が変わる（空振り防止）: {seed_effect}"
          f"（差の最大 {float(np.max(np.abs(np.asarray(o1) - np.asarray(o2)))):.5f}）")
    e1.close()
    e2.close()

    raw = make_env(base_seed)
    wrap_base = make_env(base_seed)
    wrapped = ObsHistoryWrapper(wrap_base, LAGS, sham=True)
    o_raw0, _ = raw.reset(seed=seed_a)
    o_wrap0, _ = wrapped.reset(seed=seed_a)
    n = raw.observation_space.shape[0]
    reset_ok = np.array_equal(o_raw0, o_wrap0[:n])
    print(f"  reset 直後: 素の観測が bit 一致 = {reset_ok}")

    # リスポーンを跨いで照合する（tests/test_obs_history.py T7 と同じ狙い。
    # リスポーンは走行中に np_random から擾乱を引き直すので、ここを跨がないと
    # 「歩の途中で乱数を消費していないか」の検査が空振りになる）
    N = 800
    rng = np.random.default_rng(21)   # env の乱数とは独立
    n_mismatch, n_respawn_raw, n_respawn_wrap = 0, 0, 0
    n_done = 0
    for _t in range(N):
        a = np.clip(0.4 + 0.5 * rng.uniform(-1, 1, size=2), -1.0, 1.0)
        o_raw, _r1, term1, trunc1, i1 = raw.step(a)
        o_wrap, _r2, term2, trunc2, i2 = wrapped.step(a)
        n_done += 1
        n_respawn_raw += int(bool(i1.get("respawned", False)))
        n_respawn_wrap += int(bool(i2.get("respawned", False)))
        if not np.array_equal(o_raw, o_wrap[:n]):
            n_mismatch += 1
        if term1 or trunc1 or term2 or trunc2:
            break
    step_ok = (n_mismatch == 0)
    # 空振り防止: リスポーンを実際に跨いだこと
    respawn_ok = (n_respawn_raw >= 1 and n_respawn_raw == n_respawn_wrap)
    print(f"  {n_done} 歩・毎歩照合: 不一致 {n_mismatch} 件 → bit 一致 = {step_ok}")
    print(f"  空振り防止: リスポーン回数 素 {n_respawn_raw} 回 / sham側 {n_respawn_wrap} 回"
          f"（1 以上かつ一致であること = {respawn_ok}）")

    raw.close()
    wrapped.close()
    return seed_effect and reset_ok and step_ok and respawn_ok


# ======================================================================
# S4: sham=True の出力に過去の観測が 1 要素も混ざっていない
# ======================================================================
def s4_no_past_observation_leaks_into_sham() -> bool:
    print("\n[S4] sham=True の出力に過去の観測が 1 要素も混ざっていない")
    seed = 13
    base_seed = 9040
    K = max(LAGS) + 10   # k > max(lags)

    raw_env = make_env(base_seed)
    wrap_base = make_env(base_seed)
    wrapped = ObsHistoryWrapper(wrap_base, LAGS, sham=True)

    o0_raw, _ = raw_env.reset(seed=seed)
    wrapped.reset(seed=seed)
    raw_hist = [np.asarray(o0_raw, dtype=np.float32)]   # raw_hist[t] = t 歩後の素の観測

    o_wrap = None
    for _t in range(K):
        o_raw, _r1, term1, trunc1, _i1 = raw_env.step(FORWARD)
        o_wrap, _r2, term2, trunc2, _i2 = wrapped.step(FORWARD)
        raw_hist.append(np.asarray(o_raw, dtype=np.float32))
        if term1 or trunc1 or term2 or trunc2:
            break
    last_t = len(raw_hist) - 1

    # 空振り防止: 観測が毎歩変化していること（変化しなければ過去と現在が同じになり検査が空振り）
    adj_diffs = [float(np.max(np.abs(raw_hist[i + 1] - raw_hist[i])))
                 for i in range(len(raw_hist) - 1)]
    min_adj_diff = min(adj_diffs) if adj_diffs else 0.0
    print(f"  隣接歩の差の最大値の最小（全 {len(adj_diffs)} 歩中）= {min_adj_diff:.3e}"
          f"（0 なら空振り）")

    n = raw_env.observation_space.shape[0]
    enough_steps = last_t >= max(LAGS)
    no_leak = []
    for j, lag in enumerate(LAGS):
        idx = last_t - lag
        block = o_wrap[(j + 1) * n:(j + 2) * n]
        matches_past = idx >= 0 and np.array_equal(block, raw_hist[idx])
        no_leak.append((not matches_past) if idx >= 0 else True)
        print(f"  遅れ{lag}: ブロック{j + 1} が {lag} 歩前の素の観測と一致してしまっていないか"
              f" → 一致=False であるべき: {matches_past}"
              f"（判定 {'OK' if not matches_past else 'NG'}）")

    raw_env.close()
    wrapped.close()
    return enough_steps and (min_adj_diff > 0.0) and all(no_leak)


# ======================================================================
# S5: sham の既定は False。sham=False の出力は sham を渡さないときと bit 一致
# ======================================================================
def s5_default_is_false_and_matches_explicit_false() -> bool:
    print("\n[S5] sham の既定が False であり、sham=False の出力が省略時と bit 一致")
    seed = 17
    base_seed = 9050
    K = max(LAGS) + 5

    env_default = make_env(base_seed)
    wrapped_default = ObsHistoryWrapper(env_default, LAGS)          # sham を渡さない
    env_explicit = make_env(base_seed)
    wrapped_explicit = ObsHistoryWrapper(env_explicit, LAGS, sham=False)

    default_is_false = (wrapped_default.sham is False)
    print(f"  wrapped.sham（省略時） = {wrapped_default.sham!r}（期待 False）")

    o_def, _ = wrapped_default.reset(seed=seed)
    o_exp, _ = wrapped_explicit.reset(seed=seed)
    reset_ok = np.array_equal(o_def, o_exp)
    print(f"  reset 直後: 省略時 と sham=False が bit 一致 = {reset_ok}")

    step_mismatch = 0
    n_steps = 0
    for _t in range(K):
        o_def, _r1, term1, trunc1, _i1 = wrapped_default.step(FORWARD)
        o_exp, _r2, term2, trunc2, _i2 = wrapped_explicit.step(FORWARD)
        n_steps += 1
        if not np.array_equal(o_def, o_exp):
            step_mismatch += 1
        if term1 or trunc1 or term2 or trunc2:
            break
    step_ok = (step_mismatch == 0)
    print(f"  {n_steps} 歩・毎歩照合: 不一致 {step_mismatch} 件 → bit 一致 = {step_ok}")

    wrapped_default.close()
    wrapped_explicit.close()
    return default_is_false and reset_ok and step_ok


# ======================================================================
# S6: sham の有無で、素の env に渡す行動列が同じなら素の env の軌跡が bit 一致
# ======================================================================
def s6_sham_does_not_alter_env_dynamics() -> bool:
    print("\n[S6] sham の有無で、素の env に渡す行動列が同じなら素の env の軌跡が bit 一致")
    seed = 55
    base_seed = 9060
    K = max(LAGS) + 10

    env_true = make_env(base_seed)
    wrap_true = ObsHistoryWrapper(env_true, LAGS, sham=True)
    env_false = make_env(base_seed)
    wrap_false = ObsHistoryWrapper(env_false, LAGS, sham=False)

    # ブロック0（index 0）は sham の有無に関係なく「現在の素の観測」なので、
    # これを取り出せば素の env の軌跡をラッパ越しに追える。
    o_true, _ = wrap_true.reset(seed=seed)
    o_false, _ = wrap_false.reset(seed=seed)
    n = env_true.observation_space.shape[0]

    traj = [o_true[:n].copy()]
    n_mismatch = 0
    n_steps = 0
    for _t in range(K):
        o_true, _r1, term1, trunc1, _i1 = wrap_true.step(FORWARD)
        o_false, _r2, term2, trunc2, _i2 = wrap_false.step(FORWARD)
        n_steps += 1
        traj.append(o_true[:n].copy())
        if not np.array_equal(o_true[:n], o_false[:n]):
            n_mismatch += 1
        if term1 or trunc1 or term2 or trunc2:
            break

    # 空振り防止: 軌跡が定数でないこと（歩ごとに素の観測＝位置等が変わること）
    diffs = [float(np.max(np.abs(traj[i + 1] - traj[i]))) for i in range(len(traj) - 1)]
    min_diff = min(diffs) if diffs else 0.0
    print(f"  {n_steps} 歩の軌跡: 隣接歩の差の最大値の最小 = {min_diff:.3e}（0 なら空振り）")
    print(f"  素の env の軌跡（ブロック0）不一致 {n_mismatch} / {n_steps} 歩")

    ok = (n_mismatch == 0) and (min_diff > 0.0)

    env_true.close()
    env_false.close()
    return ok


# ======================================================================
def main() -> int:
    print("=" * 78)
    print("mouse/obs_history.py の sham 引数（にせ履歴・exp_022）の単体テスト")
    print("=" * 78)
    tests = [
        ("S1 sham=True: 全ブロックが現在の素の観測と一致", s1_sham_blocks_match_current),
        ("S2 sham の有無で observation_space.shape が同じ", s2_shape_unchanged_by_sham),
        ("S3 sham=True でも素の状態遷移が一致（乱数不消費）", s3_sham_true_no_rng_consumption),
        ("S4 sham=True の出力に過去の観測が混ざらない", s4_no_past_observation_leaks_into_sham),
        ("S5 sham の既定 False・省略時と明示 False が一致", s5_default_is_false_and_matches_explicit_false),
        ("S6 sham の有無で素の env の軌跡が一致", s6_sham_does_not_alter_env_dynamics),
    ]
    results = []
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as exc:  # noqa: BLE001 — 1 項目の失敗で全体を止めない
            import traceback
            print(f"  🔴 例外: {exc!r}")
            print("  " + "\n  ".join(traceback.format_exc().splitlines()[-6:]))
            ok = False
        results.append((name, ok))

    print("\n" + "=" * 78)
    n_ok = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✅ PASS' if ok else '🔴 FAIL'}  {name}")
    print(f"\n  {n_ok} / {len(results)} PASS")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
