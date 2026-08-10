"""
experiments/exp_001_corridor/train.py
================
M1（廊下追従）PPO 学習スクリプト。spec_m1_corridor.md §4 の確定ハイパーパラメータ
をそのまま使う。訓練コースは mode="generate"（毎エピソード mouse.corridor_gen で
新規手続き生成、2026-08-10 指揮側追加指示）を使い、固定プールへの過学習を避ける。

使い方:
    .venv/bin/python experiments/exp_001_corridor/train.py --smoke       # 10万ステップ疎通確認
    .venv/bin/python experiments/exp_001_corridor/train.py               # 本学習 300万ステップ
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from mouse.corridor_env import CorridorEnv  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

# ポテンシャル整形の γ と PPO の gamma は必ず一致させる（理論保証の前提。
# spec_m1_corridor.md §4、RESEARCH_PLAN Track A）。ここが単一の情報源で、
# 環境コンストラクタにも PPO コンストラクタにも同じ値を渡す。
GAMMA = 0.995

# 訓練コースseedの起点（学習用seedは2000以降という憲章の規約）。
TRAIN_BASE_SEED = 2000
# ワーカ間でコースseedが重複しないためのオフセット（2026-08-10 指揮側指示の例をそのまま採用）。
WORKER_SEED_STRIDE = 1_000_000

SMOKE_TOTAL_STEPS = 100_000
FULL_TOTAL_STEPS = 3_000_000


def make_env(rank: int, gamma: float, log_dir: Path):
    """CorridorEnv(mode="generate") を Monitor でラップした env_fn を返す。"""
    def _init():
        env = CorridorEnv(
            mode="generate",
            base_seed=TRAIN_BASE_SEED + rank * WORKER_SEED_STRIDE,
            gamma=gamma,
        )
        env = Monitor(env, filename=str(log_dir / f"env_{rank}"))
        return env
    return _init


class EpisodeStatsCallback(BaseCallback):
    """smoke 運用時の初期挙動観測用コールバック（2026-08-10 指揮側指示 項目3）。

    - 最初の n_track_episodes 件（全ワーカ合算）のエピソード長・報酬を記録する
      （SB3 の Monitor ラッパが info["episode"] = {"r":..,"l":..,"t":..} を
      エピソード終了時にセットするのを拾う）。
    - 行動振動指標（符号反転頻度・行動差分RMS）を全ステップにわたり集計する。
      self.locals["clipped_actions"] は OnPolicyAlgorithm.collect_rollouts が
      実際に env.step() へ渡した行動（アクション空間境界へのクリップ後）であり、
      corridor_eval.py が使う「実際に印加された行動」の定義と一致する。
    """

    def __init__(self, n_envs: int, n_track_episodes: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.n_envs = n_envs
        self.n_track_episodes = n_track_episodes
        self.episode_lengths = []
        self.episode_rewards = []
        self._prev_action = np.zeros((n_envs, 2), dtype=np.float64)
        self._has_prev = np.zeros(n_envs, dtype=bool)
        self.sign_flips = np.zeros(2, dtype=np.int64)
        self.diff_sq_sum = 0.0
        self.diff_count = 0
        self.n_on_step_calls = 0

    def _on_step(self) -> bool:
        actions = self.locals.get("clipped_actions")
        if actions is None:
            actions = self.locals.get("actions")
        actions = np.asarray(actions, dtype=np.float64).reshape(self.n_envs, -1)
        infos = self.locals.get("infos", [])
        self.n_on_step_calls += 1

        for i in range(self.n_envs):
            if self._has_prev[i]:
                diff = actions[i] - self._prev_action[i]
                self.diff_sq_sum += float(np.sum(diff ** 2))
                self.diff_count += 1
                for k in (0, 1):
                    if self._prev_action[i, k] * actions[i, k] < 0.0:
                        self.sign_flips[k] += 1
            self._prev_action[i] = actions[i]
            self._has_prev[i] = True

            ep = infos[i].get("episode") if i < len(infos) else None
            if ep is not None and len(self.episode_lengths) < self.n_track_episodes:
                self.episode_lengths.append(int(ep["l"]))
                self.episode_rewards.append(float(ep["r"]))
        return True

    def summary(self, control_dt: float) -> dict:
        elapsed_s_per_env = self.n_on_step_calls * control_dt
        lengths = np.array(self.episode_lengths, dtype=np.float64)
        diff_rms = float(np.sqrt(self.diff_sq_sum / max(self.diff_count, 1)))
        flip_rate = (self.sign_flips / (self.n_envs * elapsed_s_per_env)
                     if elapsed_s_per_env > 0 else np.zeros(2))

        length_stats = None
        length_hist = None
        if lengths.size:
            length_stats = dict(
                n=int(lengths.size), min=float(lengths.min()), max=float(lengths.max()),
                mean=float(lengths.mean()), median=float(np.median(lengths)),
                std=float(lengths.std()),
            )
            counts, edges = np.histogram(lengths, bins=min(10, max(1, lengths.size)))
            length_hist = dict(
                bin_edges=[float(e) for e in edges],
                counts=[int(c) for c in counts],
            )

        return dict(
            n_episodes_tracked=len(self.episode_lengths),
            episode_length_stats=length_stats,
            episode_length_histogram=length_hist,
            episode_reward_mean_first_tracked=(
                float(np.mean(self.episode_rewards)) if self.episode_rewards else None),
            action_diff_rms=diff_rms,
            sign_flip_rate_left_per_s=float(flip_rate[0]),
            sign_flip_rate_right_per_s=float(flip_rate[1]),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="M1 exp_001_corridor PPO学習")
    parser.add_argument("--total-steps", type=int, default=FULL_TOTAL_STEPS)
    parser.add_argument("--smoke", action="store_true",
                         help="10万ステップ・単一環境で疎通確認（本学習は実行しない）")
    parser.add_argument("--seed", type=int, default=0, help="PPO本体の乱数seed")
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--log-dir", type=str, default="logs/exp_001_corridor")
    parser.add_argument("--model-out", type=str, default="models/exp_001_corridor.zip")
    args = parser.parse_args(argv)

    total_steps = SMOKE_TOTAL_STEPS if args.smoke else args.total_steps
    n_envs = 1 if args.smoke else args.n_envs

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(i, args.gamma, log_dir) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns) if n_envs == 1 else SubprocVecEnv(env_fns)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=args.gamma, gae_lambda=0.95, ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.seed, verbose=1,
    )
    new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    stats_cb = EpisodeStatsCallback(n_envs=n_envs, n_track_episodes=100)
    ckpt_cb = CheckpointCallback(save_freq=max(500_000 // n_envs, 1), save_path=str(log_dir))

    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=[ckpt_cb, stats_cb])
    elapsed = time.time() - t0

    steps_per_sec = total_steps / elapsed if elapsed > 0 else float("nan")
    print(f"[train] total_steps={total_steps} n_envs={n_envs} elapsed={elapsed:.1f}s "
          f"steps/s={steps_per_sec:.1f}")

    stats = stats_cb.summary(control_dt=RobotParams().control_dt)
    print("[train] episode stats (first tracked episodes):")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.smoke:
        stats_path = log_dir / "smoke_episode_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(dict(
                total_steps=total_steps, n_envs=n_envs, elapsed_s=elapsed,
                steps_per_sec=steps_per_sec, **stats,
            ), f, indent=2, ensure_ascii=False)
        print(f"[train] smoke stats saved: {stats_path}")
    else:
        model_out = Path(args.model_out)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_out))
        print(f"[train] model saved: {model_out}")

    vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
