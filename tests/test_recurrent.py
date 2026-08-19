#!/usr/bin/env python3
"""mouse/recurrent.py（再帰型方策の道具・exp_023）の単体テスト。

`experiments/exp_023_recurrent_policy/card.md` §6-4 の投入前検証のうち、
T-R2〜T-R6 を条文どおりに検査する（T-R1「既存の全テストが通る」は本ファイルの
対象外 — 既存のテストファイルを直接叩けば足りるため、ここでは再実装しない）。

| # | 検査 | 要点 |
|---|---|---|
| T-R2 | 群2（RespawnResetRecurrentPPO＋RespawnFlagCallback）でリスポーンの歩の
        `episode_starts` が1になる。群1（素の`RecurrentPPO`）では同じ歩が0のまま | card.md §6-4 |
| T-R3 | `evaluate_maze6` の `policy_reset_fn` で、隠れ状態が試行ごとにリセットされる | 同上 |
| T-R4 | 同じ行動列を入れたときの `Maze6Env` の応答が方策に依らず bit 一致する | 同上 |
| T-R5 | 方策のパラメータ数が期待どおり（LSTM隠れ32で89,733・対照PPOで72,837） | 同上 |
| T-R6 | `RespawnResetRecurrentPPO` をコールバック無しで使うと、素の `RecurrentPPO` と
        1ロールアウトぶん bit 一致する（部分クラスそのものの無害性） | 同上 |

pytest は使わない plain Python スクリプト（tests/test_obs_history.py と同じ流儀）。
実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_recurrent.py
"""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# sb3_contrib は任意依存（未導入環境では pytest 収集を壊さずスキップする）。
pytest.importorskip("sb3_contrib")

import numpy as np  # noqa: E402
from sb3_contrib import RecurrentPPO  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from mouse.maze6_env import Maze6Env  # noqa: E402
from mouse.maze6_eval import evaluate_maze6  # noqa: E402
from mouse.maze6_gen import generate_maze, save_maze  # noqa: E402
from mouse.obs_history import DEFAULT_LAGS, ObsHistoryWrapper  # noqa: E402
from mouse.recurrent import (RecurrentPolicyFn, RespawnFlagCallback,  # noqa: E402
                             RespawnResetRecurrentPPO)

# 学習環境 v2 のフラグ（tests/test_obs_history.py の V2 と同一。規約終端＋衝突リスポーン＋
# 上限 2000 歩）。
V2 = dict(goal_rule_containment=True, collision_respawn=True, episode_limit_steps=2000)

# 方策の設定（exp_023 カード §6-1・§2-1・教授指示と同一）。
POLICY_KWARGS = dict(net_arch=[128, 128], lstm_hidden_size=32)
PPO_KWARGS = dict(learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
                  gamma=0.995, gae_lambda=0.95, ent_coef=0.0, seed=1, device="cpu")


def make_env(base_seed: int, **kwargs) -> Maze6Env:
    """学習環境 v2 の Maze6Env を作る（mode='generate'、学習用 seed 帯 8000 以降。
    予約帯 6000-6019・7000-7019・7100-7299 は使わない）。"""
    cfg = dict(V2)
    cfg.update(kwargs)
    return Maze6Env(mode="generate", base_seed=base_seed, gamma=0.995, **cfg)


def make_vec_env(base_seed: int, **kwargs) -> DummyVecEnv:
    """学習で実際に使う経路（DummyVecEnv＋Monitor＋ObsHistoryWrapper・153 次元）を再現する。"""
    def _mk():
        return Monitor(ObsHistoryWrapper(make_env(base_seed, **kwargs), DEFAULT_LAGS))
    return DummyVecEnv([_mk])


class InfoRecorder(BaseCallback):
    """毎歩の `info['respawned']` と `dones` を記録するだけの検査用コールバック。

    方策・モデルには一切書き込まない（`RespawnFlagCallback` と違い読み取り専用）ので、
    `RespawnFlagCallback` と併用しても・単独で使っても、収集される rollout には影響しない。
    """

    def __init__(self):
        super().__init__()
        self.respawned = []
        self.dones = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        self.respawned.append(bool(infos[0].get("respawned", False)) if infos else False)
        self.dones.append(bool(dones[0]) if dones is not None else False)
        return True


# ======================================================================
# T-R2: 群2でリスポーンの歩の episode_starts が1になる。群1では同じ歩が0のまま
# ======================================================================
def t_r2_respawn_flag_sets_episode_start() -> bool:
    print("\n[T-R2] 群2（RespawnResetRecurrentPPO＋RespawnFlagCallback）でリスポーンの歩の"
          " episode_starts が1になる。群1（素の RecurrentPPO）では同じ歩が0のまま")
    base_seed = 8100

    # --- 群1: 素の RecurrentPPO（コールバック無し）---
    vec1 = make_vec_env(base_seed)
    model1 = RecurrentPPO("MlpLstmPolicy", vec1, policy_kwargs=POLICY_KWARGS, **PPO_KWARGS)
    rec1 = InfoRecorder()
    model1.learn(total_timesteps=2048, callback=rec1)
    starts1 = np.asarray(model1.rollout_buffer.episode_starts).reshape(-1).astype(bool)
    vec1.close()

    # --- 群2: RespawnResetRecurrentPPO + RespawnFlagCallback ---
    vec2 = make_vec_env(base_seed)
    model2 = RespawnResetRecurrentPPO("MlpLstmPolicy", vec2, policy_kwargs=POLICY_KWARGS,
                                      **PPO_KWARGS)
    rec2 = InfoRecorder()
    model2.learn(total_timesteps=2048, callback=CallbackList([RespawnFlagCallback(), rec2]))
    starts2 = np.asarray(model2.rollout_buffer.episode_starts).reshape(-1).astype(bool)
    vec2.close()

    n1, n2 = len(rec1.respawned), len(rec2.respawned)
    print(f"  群1: 収集歩数 {n1}・リスポーン {sum(rec1.respawned)} 回")
    print(f"  群2: 収集歩数 {n2}・リスポーン {sum(rec2.respawned)} 回")

    # buffer.episode_starts[i] は「i-1 歩目の info で拾ったリスポーン旗（と done）」を反映する
    # （sb3_contrib/ppo_recurrent/ppo_recurrent.py:257-298 の実装を辿って確認済み。
    #  add() は self._last_episode_starts = dones で更新される "前" の値を積むため、
    #  ある歩のリスポーン旗は「次の歩」の episode_starts に現れる）。
    g2_hits = [(j, bool(starts2[j + 1])) for j in range(n2 - 1) if rec2.respawned[j]]
    g2_ok = bool(g2_hits) and all(v for _, v in g2_hits)
    g1_mismatch = [(j, bool(starts1[j + 1]), rec1.dones[j]) for j in range(n1 - 1)
                   if rec1.respawned[j] and bool(starts1[j + 1]) != rec1.dones[j]]
    g1_ok = (len(g1_mismatch) == 0)
    print(f"  群2: リスポーンの次の歩で episode_starts=True になった件数 = "
          f"{sum(1 for _, v in g2_hits if v)} / {len(g2_hits)}（全件一致 = {g2_ok}）")
    print(f"  群1: リスポーンの次の歩の episode_starts が done と食い違った件数 = "
          f"{len(g1_mismatch)}（0 が正常。先頭 3 件 {g1_mismatch[:3]}）")

    ones1, ones2 = int(starts1.sum()), int(starts2.sum())
    print(f"  episode_starts の1の数: 群1 = {ones1}・群2 = {ones2}"
          f"（空振り防止: 群2が群1より真に多いこと = {ones2 > ones1}）")

    respawned_at_all = sum(rec2.respawned) >= 1
    print(f"  空振り防止: 群2の走行でリスポーンが1回以上起きた = {respawned_at_all}")

    return respawned_at_all and (ones2 > ones1) and g2_ok and g1_ok


# ======================================================================
# T-R3: evaluate_maze6 の policy_reset_fn で隠れ状態が試行ごとにリセットされる
# ======================================================================
def t_r3_policy_reset_fn_resets_hidden_state() -> bool:
    print("\n[T-R3] evaluate_maze6 の policy_reset_fn で隠れ状態が試行ごとにリセットされる")
    model_base_seed = 8500   # 方策の観測・行動空間を決めるためだけの使い捨て env
    maze_seed = 8501         # 評価に使う一時迷路の seed（学習用帯・確保済み帯ではない）

    vec = make_vec_env(model_base_seed)
    model = RecurrentPPO("MlpLstmPolicy", vec, policy_kwargs=POLICY_KWARGS, **PPO_KWARGS)
    vec.close()   # 学習はしない。隠れ状態の配管（受け渡し）だけを検査するので未学習の重みで足りる

    def _wrap(e):
        return ObsHistoryWrapper(e, DEFAULT_LAGS)

    env_kwargs = dict(goal_rule_containment=True)   # 評価はリスポーンを入れない（評価規約どおり）

    def _run(policy_fn, reset_fn, maze_dir):
        s = evaluate_maze6(policy_fn, maze_dir=maze_dir, n_trials=1, seed=0, gamma=0.995,
                           maze_mode="loop", keep_traces=False, env_kwargs=env_kwargs,
                           env_wrapper=_wrap, policy_reset_fn=reset_fn)
        return json.dumps(s["per_maze"][0]["trials"][0], sort_keys=True)

    with tempfile.TemporaryDirectory() as td:
        maze = generate_maze(maze_seed, mode="loop")
        save_maze(maze, td)

        # --- reset() あり: 同じ迷路・同じ試行 seed を2回続けて走らせると bit 一致するはず ---
        pf_reset = RecurrentPolicyFn(model, deterministic=True)
        j1 = _run(pf_reset, pf_reset.reset, td)
        j2 = _run(pf_reset, pf_reset.reset, td)
        reset_match = (j1 == j2)
        print(f"  reset() あり: 1回目と2回目の走行結果が bit 一致 = {reset_match}")

        # --- 空振り防止: reset() を渡さないと隠れ状態が持ち越されて不一致になるはず ---
        pf_noreset = RecurrentPolicyFn(model, deterministic=True)
        j3 = _run(pf_noreset, None, td)
        j4 = _run(pf_noreset, None, td)
        noreset_mismatch = (j3 != j4)
        print(f"  reset() なし: 1回目と2回目の走行結果が不一致（空振り防止）= {noreset_mismatch}")

    return reset_match and noreset_mismatch


# ======================================================================
# T-R4: 同じ行動列を入れたときの Maze6Env の応答が方策に依らず bit 一致する
# ======================================================================
def t_r4_env_response_independent_of_policy() -> bool:
    print("\n[T-R4] 同じ行動列・同じ seed で Maze6Env（v2・履歴ラッパ付き）を作り直しても"
          "毎歩の観測・報酬・info が bit 一致する")
    base_seed = 8600
    seed = 21
    n_steps_target = 1500
    rng = np.random.default_rng(9)   # env の乱数とは独立（行動列を1回だけ生成して両走行で使い回す）

    def drive(_i):
        return np.array([0.9, 0.9]) + 0.05 * rng.uniform(-1, 1, size=2)

    actions = [drive(i) for i in range(n_steps_target)]
    const_actions = all(np.array_equal(a, actions[0]) for a in actions)

    def run():
        env = ObsHistoryWrapper(make_env(base_seed), DEFAULT_LAGS)
        obs, info = env.reset(seed=seed)
        obs_hist = [np.asarray(obs, dtype=np.float32)]
        reward_hist, dist_hist, respawn_hist, cell_hist = [], [], [], []
        for a in actions:
            obs, r, term, trunc, info = env.step(a)
            obs_hist.append(np.asarray(obs, dtype=np.float32))
            reward_hist.append(float(r))
            dist_hist.append(info.get("dist_to_goal"))
            respawn_hist.append(bool(info.get("respawned", False)))
            cell_hist.append(info.get("cell_entries"))
            if term or trunc:
                break
        env.close()
        return obs_hist, reward_hist, dist_hist, respawn_hist, cell_hist

    run1 = run()
    run2 = run()

    obs_ok = (len(run1[0]) == len(run2[0])
             and all(np.array_equal(a, b) for a, b in zip(run1[0], run2[0])))
    reward_ok = (run1[1] == run2[1])
    dist_ok = (run1[2] == run2[2])
    respawn_ok = (run1[3] == run2[3])
    cell_ok = (run1[4] == run2[4])

    diffs = [float(np.max(np.abs(run1[0][i + 1] - run1[0][i]))) for i in range(len(run1[0]) - 1)]
    min_diff = min(diffs) if diffs else 0.0
    n_respawn = sum(run1[3])

    print(f"  実行歩数 = {len(run1[1])}（両走行とも同じ歩数で終了 = "
          f"{len(run1[1]) == len(run2[1])}）・リスポーン回数（1回目走行）= {n_respawn}")
    print(f"  空振り防止: 行動列が定数でない = {not const_actions}")
    print(f"  空振り防止: 観測が毎歩変化する（隣接差の最小 {min_diff:.3e}） = {min_diff > 0.0}")
    print(f"  observation bit一致 = {obs_ok}・reward = {reward_ok}・"
          f"dist_to_goal = {dist_ok}・respawned = {respawn_ok}・cell_entries = {cell_ok}")

    return ((not const_actions) and (min_diff > 0.0) and obs_ok and reward_ok
           and dist_ok and respawn_ok and cell_ok)


# ======================================================================
# T-R5: 方策のパラメータ数が期待どおり
# ======================================================================
def t_r5_param_counts() -> bool:
    print("\n[T-R5] 方策のパラメータ数（LSTM 隠れ32で89,733・対照 PPO で72,837）")

    vec_r = make_vec_env(8610)
    model_r = RecurrentPPO("MlpLstmPolicy", vec_r, policy_kwargs=POLICY_KWARGS, **PPO_KWARGS)
    n_r = sum(p.numel() for p in model_r.policy.parameters())
    vec_r.close()

    vec_p = make_vec_env(8611)
    model_p = PPO("MlpPolicy", vec_p, learning_rate=3e-4, n_steps=2048, batch_size=256,
                  n_epochs=10, gamma=0.995, gae_lambda=0.95, ent_coef=0.0,
                  policy_kwargs=dict(net_arch=[128, 128]), seed=1, device="cpu")
    n_p = sum(p.numel() for p in model_p.policy.parameters())
    vec_p.close()

    print(f"  RecurrentPPO（LSTM隠れ32・153次元）パラメータ数 = {n_r:,}（期待 89,733）")
    print(f"  PPO（MlpPolicy・153次元）パラメータ数 = {n_p:,}（期待 72,837）")

    return (n_r == 89733) and (n_p == 72837)


# ======================================================================
# T-R6: 部分クラスの無害性（コールバック無しで素の RecurrentPPO と bit 一致）
# ======================================================================
def t_r6_subclass_harmless_without_callback() -> bool:
    print("\n[T-R6] RespawnResetRecurrentPPO をコールバック無しで使うと、"
          "素の RecurrentPPO と1ロールアウト（2048歩）ぶん bit 一致する")
    base_seed = 8700

    vec1 = make_vec_env(base_seed)
    model1 = RecurrentPPO("MlpLstmPolicy", vec1, policy_kwargs=POLICY_KWARGS, **PPO_KWARGS)
    rec1 = InfoRecorder()
    model1.learn(total_timesteps=2048, callback=rec1)
    vec1.close()

    vec2 = make_vec_env(base_seed)
    model2 = RespawnResetRecurrentPPO("MlpLstmPolicy", vec2, policy_kwargs=POLICY_KWARGS,
                                      **PPO_KWARGS)
    rec2 = InfoRecorder()
    model2.learn(total_timesteps=2048, callback=rec2)   # 🔴 RespawnFlagCallback を意図的に付けない
    vec2.close()

    buf1, buf2 = model1.rollout_buffer, model2.rollout_buffer
    obs_ok = np.array_equal(buf1.observations, buf2.observations)
    act_ok = np.array_equal(buf1.actions, buf2.actions)
    rew_ok = np.array_equal(buf1.rewards, buf2.rewards)
    starts_ok = np.array_equal(buf1.episode_starts, buf2.episode_starts)

    n_respawn1 = sum(rec1.respawned)
    print(f"  空振り防止: この走行でリスポーンが1回以上起きた = {n_respawn1 >= 1}"
          f"（回数 {n_respawn1}）")
    print(f"  observations bit一致 = {obs_ok}・actions = {act_ok}・"
          f"rewards = {rew_ok}・episode_starts = {starts_ok}")

    return (n_respawn1 >= 1) and obs_ok and act_ok and rew_ok and starts_ok


# ======================================================================
def main() -> int:
    print("=" * 78)
    print("mouse/recurrent.py（再帰型方策の道具・exp_023）の投入前検証 T-R2〜T-R6")
    print("=" * 78)
    tests = [
        ("T-R2 群2でリスポーンの歩の episode_starts が1になる", t_r2_respawn_flag_sets_episode_start),
        ("T-R3 policy_reset_fn で隠れ状態が試行ごとにリセットされる",
         t_r3_policy_reset_fn_resets_hidden_state),
        ("T-R4 環境の応答が方策に依らず bit 一致", t_r4_env_response_independent_of_policy),
        ("T-R5 方策のパラメータ数が期待どおり", t_r5_param_counts),
        ("T-R6 部分クラスの無害性（コールバック無しで bit 一致）",
         t_r6_subclass_harmless_without_callback),
    ]
    results = []
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as exc:  # noqa: BLE001 — 1項目の失敗で全体を止めない
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
