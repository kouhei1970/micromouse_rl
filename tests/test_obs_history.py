#!/usr/bin/env python3
"""mouse/obs_history.py（観測履歴の連結・exp_021）の単体テスト。

`ObsHistoryWrapper` と `parse_lags` が仕様どおりに動くかを検査する。

| # | 検査 | 要点 |
|---|---|---|
| T1 | `parse_lags` の解析・整列・異常系。`ObsHistoryWrapper(env, ())` が ValueError | obs_history.py 冒頭のコメント |
| T2 | 観測次元 = 元の次元 × (1 + len(lags))（元の次元は決め打ちしない） | 同上 |
| T3 | reset 直後、全ブロックが素の観測と bit 一致（複製） | 同上 |
| T4 | k 歩進めたあと、遅れ ℓ のブロックが「ℓ 歩前の素の観測」と bit 一致 | 同上 |
| T5 | info（cell_entries・respawned・dist_to_goal）が毎歩そのまま素通し | 同上 |
| T6 | `wrapped.sim` が素の env の `sim` と同一オブジェクト | 同上 |
| T7 | ラッパが `np_random` を消費しない（bit 一致で確認） | 同上 |
| T8 | `wrapped.set_d0_max(4)` が素の env へ届く | 同上 |

pytest は使わない plain Python スクリプト（tests/test_curriculum.py と同じ流儀）。
実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_obs_history.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from mouse.maze6_env import Maze6Env  # noqa: E402
from mouse.obs_history import ObsHistoryWrapper, parse_lags  # noqa: E402

# 学習環境 v2 のフラグ（experiments/exp_012_continuous_potential/train.py の
# ENV_VERSION_FLAGS["v2"] と同一。規約終端＋衝突リスポーン＋上限 2000 歩）。
V2 = dict(goal_rule_containment=True, collision_respawn=True, episode_limit_steps=2000)

# テストで使う遅れの組。DEFAULT_LAGS（最大 128）だと 1 本の検査に 100 歩以上要って
# 重いので、ラッパの機構そのもの（元の値・遅れの対応）を確かめるには十分な小さい組にする。
LAGS = (1, 2, 5, 10)


def make_env(base_seed: int, **kwargs) -> Maze6Env:
    """学習環境 v2 の Maze6Env を作る（mode='generate'、学習用 seed 帯 8000 以降。
    予約帯 6000-6019・7000-7019・7100-7299 は使わない）。"""
    cfg = dict(V2)
    cfg.update(kwargs)
    return Maze6Env(mode="generate", base_seed=base_seed, gamma=0.995, **cfg)


def raises_value_error(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


# ======================================================================
# T1: parse_lags の解析・整列・異常系、および空 lags での ObsHistoryWrapper
# ======================================================================
def t1_parse_lags() -> bool:
    print("\n[T1] parse_lags の解析・整列・異常系")
    checks = [
        ("parse_lags(None) == ()", parse_lags(None) == ()),
        ("parse_lags('') == ()", parse_lags("") == ()),
        ("parse_lags('1,2,4') == (1,2,4)", parse_lags("1,2,4") == (1, 2, 4)),
        ("parse_lags('4,2,1') == (1,2,4)（昇順整列）", parse_lags("4,2,1") == (1, 2, 4)),
        ("重複 '1,1,2' は ValueError", raises_value_error(lambda: parse_lags("1,1,2"))),
        ("0 を含む '0,1' は ValueError", raises_value_error(lambda: parse_lags("0,1"))),
        ("負数を含む '-1,2' は ValueError", raises_value_error(lambda: parse_lags("-1,2"))),
    ]
    env = make_env(8000)   # __init__ のみ（reset しない）なので軽い
    checks.append(("ObsHistoryWrapper(env, ()) は ValueError",
                   raises_value_error(lambda: ObsHistoryWrapper(env, ()))))
    for name, ok in checks:
        print(f"  {'OK' if ok else 'NG'}  {name}")
    return all(ok for _, ok in checks)


# ======================================================================
# T2: 観測次元 = 元の次元 × (1 + len(lags))
# ======================================================================
def t2_obs_dim() -> bool:
    print("\n[T2] 観測次元が 元の次元 × (1 + len(lags)) になる")
    env = make_env(8010)
    n_raw = int(env.observation_space.shape[0])   # 決め打ちせず素の env から取る
    wrapped = ObsHistoryWrapper(env, LAGS)
    n_wrapped = int(wrapped.observation_space.shape[0])
    expect = n_raw * (1 + len(LAGS))
    print(f"  素の観測次元 = {n_raw}")
    print(f"  遅れ = {LAGS}（{len(LAGS)} 個）")
    print(f"  ラップ後の観測次元 = {n_wrapped}（期待 {n_raw}×(1+{len(LAGS)}) = {expect}）")
    return n_wrapped == expect


# ======================================================================
# T3: reset 直後、全ブロックが素の観測と bit 一致（最初の観測を複製）
# ======================================================================
def t3_reset_duplicates_first_obs() -> bool:
    print("\n[T3] reset 直後、全ブロックが素の観測と bit 一致（複製）")
    seed = 4242
    raw_ref = make_env(8020)
    o_ref, _ = raw_ref.reset(seed=seed)
    o_ref = np.asarray(o_ref, dtype=np.float32)

    # 空振り防止: 素の観測の要素が全て同じ値でないこと（定数だと何を比べても一致してしまう）
    const_obs = bool(np.all(o_ref == o_ref[0]))
    print(f"  素の観測が定数でない（空振り防止）: {not const_obs}"
          f"（最小 {o_ref.min():.4f}・最大 {o_ref.max():.4f}）")

    env_for_wrap = make_env(8020)   # 同じ base_seed → 1 本目の面も同じ
    wrapped = ObsHistoryWrapper(env_for_wrap, LAGS)
    o_wrap, _ = wrapped.reset(seed=seed)

    n = raw_ref.observation_space.shape[0]
    n_blocks = 1 + len(LAGS)
    blocks_ok = []
    for i in range(n_blocks):
        block = o_wrap[i * n:(i + 1) * n]
        ok = np.array_equal(block, o_ref)
        blocks_ok.append(ok)
        tag = "現在" if i == 0 else f"遅れ{LAGS[i - 1]}"
        print(f"  ブロック{i}（{tag}）が素の観測と bit 一致 = {ok}")

    raw_ref.close()
    wrapped.close()
    return (not const_obs) and all(blocks_ok)


# ======================================================================
# T4: k 歩進めたあと、遅れ ℓ のブロックが「ℓ 歩前の素の観測」と bit 一致
# ======================================================================
def t4_lagged_blocks_match_history() -> bool:
    print("\n[T4] k 歩進めたあと、遅れ ℓ のブロックが ℓ 歩前の素の観測と bit 一致")
    seed = 7
    base_seed = 8030
    K = max(LAGS) + 10       # 全ての遅れが埋まるだけの余裕を持たせる
    action = np.array([0.8, 0.8], dtype=np.float64)   # 観測が毎歩変化する前進方策

    raw_env = make_env(base_seed)
    wrap_base = make_env(base_seed)   # 同じ base_seed → 同じ面
    wrapped = ObsHistoryWrapper(wrap_base, LAGS)

    o0_raw, _ = raw_env.reset(seed=seed)
    o0_wrap, _ = wrapped.reset(seed=seed)
    raw_hist = [np.asarray(o0_raw, dtype=np.float32)]   # raw_hist[t] = t 歩後の素の観測

    n_steps_done = 0
    for _t in range(K):
        o_raw, _r1, term1, trunc1, _i1 = raw_env.step(action)
        o_wrap, _r2, term2, trunc2, _i2 = wrapped.step(action)
        raw_hist.append(np.asarray(o_raw, dtype=np.float32))
        n_steps_done += 1
        if term1 or trunc1 or term2 or trunc2:
            print(f"  ⚠️ {n_steps_done} 歩でエピソードが終了した（想定外だが継続不能）")
            break

    # 空振り防止: 照合に使う歩で素の観測が実際に歩ごとに変化していること
    adj_diffs = [float(np.max(np.abs(raw_hist[i + 1] - raw_hist[i])))
                 for i in range(len(raw_hist) - 1)]
    min_adj_diff = min(adj_diffs) if adj_diffs else 0.0
    print(f"  隣接歩の差の最大値の最小（全 {len(adj_diffs)} 歩中）= {min_adj_diff:.3e}"
          f"（0 なら空振り）")

    last_t = len(raw_hist) - 1     # 実際に到達した歩数
    ok_static = last_t >= max(LAGS)
    n = raw_env.observation_space.shape[0]
    block0 = o_wrap[0:n]
    ok0 = np.array_equal(block0, raw_hist[last_t])
    print(f"  {last_t} 歩後: ブロック0（現在）が raw_hist[{last_t}] と bit 一致 = {ok0}")
    lag_oks = [ok0]
    for j, lag in enumerate(LAGS):
        idx = last_t - lag
        block = o_wrap[(j + 1) * n:(j + 2) * n]
        ok = idx >= 0 and np.array_equal(block, raw_hist[idx])
        lag_oks.append(ok)
        print(f"  遅れ{lag}: ブロック{j + 1} が raw_hist[{idx}]（{lag} 歩前）と bit 一致 = {ok}")

    raw_env.close()
    wrapped.close()
    return ok_static and (min_adj_diff > 0.0) and all(lag_oks)


# ======================================================================
# T5: info（cell_entries・respawned・dist_to_goal）が毎歩そのまま得られる
# ======================================================================
def t5_info_passthrough() -> bool:
    print("\n[T5] info（cell_entries・respawned・dist_to_goal）が毎歩そのまま素通しされる")
    seed = 123
    base_seed = 8040
    raw_env = make_env(base_seed)
    wrap_base = make_env(base_seed)
    wrapped = ObsHistoryWrapper(wrap_base, LAGS)

    _o_raw, info_raw = raw_env.reset(seed=seed)
    _o_wrap, info_wrap = wrapped.reset(seed=seed)

    # 走行して壁に当てにいく開ループ方策（区画をまたぎつつ、いずれ衝突・リスポーンさせる）
    rng = np.random.default_rng(9)   # env の乱数とは独立

    def drive(_i):
        return np.array([0.9, 0.9]) + 0.05 * rng.uniform(-1, 1, size=2)

    KEYS = ("cell_entries", "respawned", "dist_to_goal")
    mismatch = []
    n_respawn = 0
    max_cell_entries = int(info_raw.get("cell_entries", 0))
    n_steps = 0
    for i in range(1500):
        a = drive(i)
        _o_raw, _r1, term1, trunc1, info_raw = raw_env.step(a)
        _o_wrap, _r2, term2, trunc2, info_wrap = wrapped.step(a)
        n_steps += 1
        for k in KEYS:
            if k not in info_raw or k not in info_wrap:
                mismatch.append((i, k, "欠落"))
                continue
            if info_raw[k] != info_wrap[k]:
                mismatch.append((i, k, (info_raw[k], info_wrap[k])))
        if info_wrap.get("respawned"):
            n_respawn += 1
        max_cell_entries = max(max_cell_entries, int(info_wrap.get("cell_entries", 0)))
        if term1 or trunc1 or term2 or trunc2:
            break

    print(f"  走行 {n_steps} 歩、キー不一致 {len(mismatch)} 件（先頭 3 件: {mismatch[:3]}）")
    print(f"  リスポーン回数 = {n_respawn}（1 以上が必要。空振り防止）")
    print(f"  cell_entries の最大値 = {max_cell_entries}"
          f"（開始時 {info_raw.get('cell_entries')} 以上 2 以上に増えることが必要）")

    ok = (not mismatch) and (n_respawn >= 1) and (max_cell_entries >= 2)
    raw_env.close()
    wrapped.close()
    return ok


# ======================================================================
# T6: wrapped.sim が素の env の sim と同一オブジェクト
# ======================================================================
def t6_sim_identity() -> bool:
    print("\n[T6] wrapped.sim が素の env の sim と同一オブジェクト")
    base = make_env(8050)
    wrapped = ObsHistoryWrapper(base, LAGS)
    wrapped.reset(seed=1)
    same = wrapped.sim is base.sim
    not_none = base.sim is not None
    print(f"  base.sim is not None = {not_none}")
    print(f"  wrapped.sim is base.sim = {same}")
    wrapped.close()
    return same and not_none


# ======================================================================
# T7: ラッパが np_random を消費しない
# ======================================================================
def t7_no_rng_consumption() -> bool:
    print("\n[T7] ラッパが np_random を消費しない（bit 一致で確認）")
    base_seed = 8060
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
    wrapped = ObsHistoryWrapper(wrap_base, LAGS)
    o_raw0, _ = raw.reset(seed=seed_a)
    o_wrap0, _ = wrapped.reset(seed=seed_a)
    n = raw.observation_space.shape[0]
    reset_ok = np.array_equal(o_raw0, o_wrap0[:n])
    print(f"  reset 直後: 素の観測が bit 一致 = {reset_ok}")

    # 🔴 リスポーンを跨いで照合する（学生B が強化した点・2026-08-14）。
    # 20 歩では reset の擾乱しか検査できない。**リスポーンは走行中に np_random から
    # 擾乱を引き直す**ので、ここを跨がないと「歩の途中で乱数を消費していないか」の
    # 検査が空振りになる（作法: テストが通ることと、何かを検査していることは別）。
    N = 800
    rng = np.random.default_rng(11)   # env の乱数とは独立
    n_done, n_mismatch, n_respawn_raw, n_respawn_wrap = 0, 0, 0, 0
    o_raw, o_wrap = o_raw0, o_wrap0
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
    # 空振り防止: リスポーンを実際に跨いだこと（跨いでいなければ検査になっていない）
    respawn_ok = (n_respawn_raw >= 1 and n_respawn_raw == n_respawn_wrap)
    print(f"  {n_done} 歩・毎歩照合: 不一致 {n_mismatch} 件 → bit 一致 = {step_ok}")
    print(f"  空振り防止: リスポーン回数 素 {n_respawn_raw} 回 / ラップ {n_respawn_wrap} 回"
          f"（1 以上かつ一致であること = {respawn_ok}）")

    raw.close()
    wrapped.close()
    return seed_effect and reset_ok and step_ok and respawn_ok


# ======================================================================
# T8: wrapped.set_d0_max(4) が素の env へ届く
# ======================================================================
def t8_set_d0_max_forwarded() -> bool:
    print("\n[T8] wrapped.set_d0_max(4) が素の env へ届く")
    base = make_env(8070)
    wrapped = ObsHistoryWrapper(base, LAGS)
    before = base._d0_max
    wrapped.set_d0_max(4)
    after = base._d0_max
    print(f"  呼び出し前 base._d0_max = {before!r}（既定 None）")
    print(f"  呼び出し後 base._d0_max = {after!r}（期待 4）")
    return after == 4


# ======================================================================
def main() -> int:
    print("=" * 78)
    print("mouse/obs_history.py（観測履歴の連結・exp_021）の単体テスト")
    print("=" * 78)
    tests = [
        ("T1 parse_lags の解析・整列・異常系", t1_parse_lags),
        ("T2 観測次元 = 元の次元 × (1+len(lags))", t2_obs_dim),
        ("T3 reset 直後の全ブロック複製", t3_reset_duplicates_first_obs),
        ("T4 遅れブロックが k 歩前の素の観測と一致", t4_lagged_blocks_match_history),
        ("T5 info の毎歩そのまま素通し", t5_info_passthrough),
        ("T6 sim の同一オブジェクト性", t6_sim_identity),
        ("T7 np_random を消費しない", t7_no_rng_consumption),
        ("T8 set_d0_max の転送", t8_set_d0_max_forwarded),
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
