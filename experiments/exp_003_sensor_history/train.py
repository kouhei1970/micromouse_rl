"""
experiments/exp_003_sensor_history/train.py
================
M1 exp_003（観測へのセンサ履歴の追加）PPO 学習スクリプト。

exp_002（距離センサ 4 本の瞬時値のみ、100 万ステップ、壁接触なし完走率 79%）から
**変更点は 1 つだけ**: 観測に距離センサ 4 本の 1 階差分を加える（11 → 15 次元、
CorridorEnv(obs_dist_diff=True)）。報酬式・net_arch・PPO ハイパーパラメータ・
コース生成・gamma は exp_002 と同一。

exp_001/train.py からの差分（教授承認済み。いずれも学習設定の変更ではない）:
 1. obs_dist_diff=True                      ← 本実験の唯一の変更点
 2. 成果物のパスを実験ごとに分離（研究計画書 §9-1。exp_002 が exp_001 の
    ログ・モデルを上書きした事故の再発防止）
 3. 検証帯（seed 5000-5019）での壁接触なし完走率を 5 万ステップごとに記録
    （計測の追加）。学習中の判断に ep_rew_mean を使えないため:
    報酬 r = γΦ(s')−Φ(s)−0.001, Φ = −(残り経路長) は、その場に留まるだけで
    (1−γ)·残り距離 = 0.005×2 m = +0.010/step を生み、時間罰 0.001 を上回る。
    割引後の収益では最適方策は変わらない（γ が PPO と一致するので理論保証は成立）が、
    Monitor が出す ep_rew_mean は**割引なしの総和**なのでエピソード長に強く依存し、
    完走率を映さない（exp_001=99% と exp_002=79% がどちらも 4.7〜5.3 で並ぶ）。
 4. エピソードごとのコース seed を JSONL に記録（研究計画書 §9-2。
    後から任意のエピソードの形状を復元して失敗を再現できるようにする）

使い方:
    .venv/bin/python experiments/exp_003_sensor_history/train.py --smoke   # 疎通確認
    .venv/bin/python experiments/exp_003_sensor_history/train.py           # 本学習 100 万
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
from mouse.corridor_eval import evaluate_corridor, VALIDATION_COURSE_DIR  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

# ポテンシャル整形の γ と PPO の gamma は必ず一致させる（理論保証の前提）。
GAMMA = 0.995

# 訓練コース seed の起点（学習用 seed は 2000 以降）。予約帯（3000-3019 の gate 用、
# 5000-5019 の検証用）は CorridorEnv 側で決定的に読み飛ばされる。
TRAIN_BASE_SEED = 2000
WORKER_SEED_STRIDE = 1_000_000

SMOKE_TOTAL_STEPS = 100_000
FULL_TOTAL_STEPS = 1_000_000     # exp_002 と同一予算（exp_002 は約 16 万で頭打ち）
VALIDATION_EVERY_STEPS = 50_000


def make_env(rank: int, gamma: float, log_dir: Path, potential_offset: bool = False,
              collision_penalty: float = 0.0, action_smooth_penalty: float = 0.0):
    """CorridorEnv(mode="generate", obs_dist_diff=True) を Monitor で包んだ env_fn。"""
    def _init():
        env = CorridorEnv(
            mode="generate",
            base_seed=TRAIN_BASE_SEED + rank * WORKER_SEED_STRIDE,
            gamma=gamma,
            obs_dist_diff=True,
            potential_offset=potential_offset,
            collision_penalty=collision_penalty,
            action_smooth_penalty=action_smooth_penalty,
        )
        env = Monitor(env, filename=str(log_dir / f"env_{rank}"))
        return env
    return _init


class EpisodeStatsCallback(BaseCallback):
    """行動振動指標の集計と、エピソードごとのコース seed の記録。

    - 行動振動: 左右モータ電圧指令の符号反転頻度 [回/s] と行動差分 RMS。
      self.locals["clipped_actions"] は実際に env.step() へ渡された行動であり、
      corridor_eval.py が使う定義と一致する。
    - コース seed: エピソード終了時の info（Monitor が info["episode"] を入れる）
      から course_seed を拾って JSONL へ書く。研究計画書 §9-2。
    """

    def __init__(self, n_envs: int, seed_log_path: Path = None,
                 n_track_episodes: int = 100, verbose: int = 0):
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
        self.n_episodes_total = 0

        self._seed_log_path = Path(seed_log_path) if seed_log_path is not None else None
        self._seed_file = None
        self._seed_written = 0

    def _on_training_start(self) -> None:
        if self._seed_log_path is not None:
            self._seed_file = open(self._seed_log_path, "w", encoding="utf-8")

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

            info = infos[i] if i < len(infos) else {}
            ep = info.get("episode")
            if ep is None:
                continue
            self.n_episodes_total += 1
            if len(self.episode_lengths) < self.n_track_episodes:
                self.episode_lengths.append(int(ep["l"]))
                self.episode_rewards.append(float(ep["r"]))
            if self._seed_file is not None:
                self._seed_file.write(json.dumps(dict(
                    t=int(self.num_timesteps), worker=i,
                    course_seed=int(info.get("course_seed", -1)),
                    n_cells=int(info.get("n_cells", -1)),
                    ep_len=int(ep["l"]), ep_rew=float(ep["r"]),
                    goal=bool(info.get("goal", False)),
                    collision=bool(info.get("collision", False)),
                ), ensure_ascii=False) + "\n")
                self._seed_written += 1
                if self._seed_written % 200 == 0:
                    self._seed_file.flush()
        return True

    def _on_training_end(self) -> None:
        if self._seed_file is not None:
            self._seed_file.close()
            self._seed_file = None

    def summary(self, control_dt: float) -> dict:
        elapsed_s_per_env = self.n_on_step_calls * control_dt
        lengths = np.array(self.episode_lengths, dtype=np.float64)
        diff_rms = float(np.sqrt(self.diff_sq_sum / max(self.diff_count, 1)))
        flip_rate = (self.sign_flips / (self.n_envs * elapsed_s_per_env)
                     if elapsed_s_per_env > 0 else np.zeros(2))

        length_stats = None
        if lengths.size:
            length_stats = dict(
                n=int(lengths.size), min=float(lengths.min()), max=float(lengths.max()),
                mean=float(lengths.mean()), median=float(np.median(lengths)),
                std=float(lengths.std()),
            )

        return dict(
            n_episodes_total=self.n_episodes_total,
            n_episodes_tracked=len(self.episode_lengths),
            episode_length_stats=length_stats,
            episode_reward_mean_first_tracked=(
                float(np.mean(self.episode_rewards)) if self.episode_rewards else None),
            action_diff_rms=diff_rms,
            sign_flip_rate_left_per_s=float(flip_rate[0]),
            sign_flip_rate_right_per_s=float(flip_rate[1]),
        )


class ValidationCallback(BaseCallback):
    """検証帯（seed 5000-5019）20 本 ×1 試行の壁接触なし完走率を定期記録する。

    gate 判定（seed 3000-3019）はここでは**行わない**。研究計画書 §9-7 のとおり、
    日常の判断は検証帯だけで行い、test 帯を見て設定を選ばない。
    """

    def __init__(self, eval_every_steps: int, log_dir: Path, gamma: float,
                 obs_dist_diff: bool, verbose: int = 0):
        super().__init__(verbose)
        self.eval_every_steps = int(eval_every_steps)
        self.log_dir = Path(log_dir)
        self.gamma = float(gamma)
        self.obs_dist_diff = bool(obs_dist_diff)
        self.history = []
        self._next_at = self.eval_every_steps

    def _evaluate(self) -> dict:
        t0 = time.time()

        def policy_fn(obs):
            action, _ = self.model.predict(obs, deterministic=True)
            return action

        summary = evaluate_corridor(
            policy_fn, course_dir=VALIDATION_COURSE_DIR, n_trials=1,
            deterministic=True, seed=0, gamma=self.gamma,
            save_output=False, obs_dist_diff=self.obs_dist_diff,
        )
        rec = dict(
            total_timesteps=int(self.num_timesteps),
            no_contact_completion_rate=summary["no_contact_completion_rate"],
            collision_rate=summary["collision_rate"],
            timeout_rate=summary["timeout_rate"],
            mean_forward_speed_mps=summary["mean_forward_speed_mps"],
            mean_sec_per_cell=summary["mean_sec_per_cell"],
            action_diff_rms_mean=summary["action_diff_rms_mean"],
            sign_flip_rate_left_mean=summary["sign_flip_rate_left_mean"],
            sign_flip_rate_right_mean=summary["sign_flip_rate_right_mean"],
            failed_course_seeds=[pc["course_seed"] for pc in summary["per_course"]
                                 if pc["n_no_contact_complete"] == 0],
            eval_wall_time_s=time.time() - t0,
        )
        self.history.append(rec)

        self.logger.record("validation/no_contact_completion_rate",
                           rec["no_contact_completion_rate"])
        self.logger.record("validation/mean_forward_speed_mps",
                           rec["mean_forward_speed_mps"])
        self.logger.record("validation/sign_flip_rate_mean",
                           0.5 * (rec["sign_flip_rate_left_mean"]
                                  + rec["sign_flip_rate_right_mean"]))
        with open(self.log_dir / "validation_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        print(f"[validation] t={rec['total_timesteps']} "
              f"完走率={rec['no_contact_completion_rate']:.2f} "
              f"速度={rec['mean_forward_speed_mps']:.3f} m/s "
              f"({rec['eval_wall_time_s']:.1f} s)")
        return rec

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_at:
            self._evaluate()
            # 追い越した分をまとめて次へ送る（n_envs 刻みで進むため）
            while self._next_at <= self.num_timesteps:
                self._next_at += self.eval_every_steps
        return True

    def _on_training_end(self) -> None:
        # 最終モデルでの検証帯成績を必ず 1 点残す
        if not self.history or self.history[-1]["total_timesteps"] != self.num_timesteps:
            self._evaluate()


def main(argv=None):
    parser = argparse.ArgumentParser(description="M1 exp_003_sensor_history PPO学習")
    parser.add_argument("--total-steps", type=int, default=FULL_TOTAL_STEPS)
    parser.add_argument("--smoke", action="store_true",
                         help="10万ステップ・単一環境で疎通確認（本学習は実行しない）")
    parser.add_argument("--seed", type=int, default=0, help="PPO本体の乱数seed")
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--log-dir", type=str, default="logs/exp_003_sensor_history")
    parser.add_argument("--model-out", type=str,
                        default="models/exp_003_sensor_history.zip")
    parser.add_argument("--validation-every", type=int, default=VALIDATION_EVERY_STEPS)
    parser.add_argument("--potential-offset", action="store_true",
                        help="exp_004: ポテンシャルを Φ=−D から Φ'=D₀−D へ（滞留の局所解を潰す）")
    parser.add_argument("--collision-penalty", type=float, default=0.0,
                        help="exp_005: 衝突時に元報酬へ加える値（負で罰。例 -1.0）")
    parser.add_argument("--action-smooth-penalty", type=float, default=0.0,
                        help="exp_006: 行動差分への罰の係数 k（−k‖a_t − a_(t−1)‖²）")
    args = parser.parse_args(argv)

    total_steps = SMOKE_TOTAL_STEPS if args.smoke else args.total_steps
    n_envs = 1 if args.smoke else args.n_envs

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(i, args.gamma, log_dir, args.potential_offset,
                        args.collision_penalty, args.action_smooth_penalty)
               for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns) if n_envs == 1 else SubprocVecEnv(env_fns)
    print(f"[train] 観測空間 = {vec_env.observation_space}（距離4 + 差分4 + 7）")
    print(f"[train] ポテンシャル = {'Φ=D₀−D（オフセットあり）' if args.potential_offset else 'Φ=−D（現行）'}")
    print(f"[train] 衝突罰 = {args.collision_penalty}")
    print(f"[train] 行動差分への罰 k = {args.action_smooth_penalty}")

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=args.gamma, gae_lambda=0.95, ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.seed, verbose=1,
    )
    new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    stats_cb = EpisodeStatsCallback(
        n_envs=n_envs, seed_log_path=log_dir / "episode_seeds.jsonl",
        n_track_episodes=100)
    val_cb = ValidationCallback(
        eval_every_steps=args.validation_every, log_dir=log_dir,
        gamma=args.gamma, obs_dist_diff=True)
    ckpt_cb = CheckpointCallback(save_freq=max(200_000 // n_envs, 1),
                                 save_path=str(log_dir))

    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=[ckpt_cb, stats_cb, val_cb])
    elapsed = time.time() - t0

    steps_per_sec = total_steps / elapsed if elapsed > 0 else float("nan")
    print(f"[train] total_steps={total_steps} n_envs={n_envs} elapsed={elapsed:.1f}s "
          f"steps/s={steps_per_sec:.1f}")

    stats = stats_cb.summary(control_dt=RobotParams().control_dt)
    print("[train] episode stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    model_out = Path(args.model_out)
    if args.smoke:
        model_out = model_out.with_name(model_out.stem + "_smoke.zip")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))
    print(f"[train] model saved: {model_out}")

    run_summary = dict(
        experiment="exp_003_sensor_history", smoke=bool(args.smoke),
        total_steps=total_steps, n_envs=n_envs, seed=args.seed, gamma=args.gamma,
        obs_dist_diff=True, potential_offset=bool(args.potential_offset),
        collision_penalty=float(args.collision_penalty),
        action_smooth_penalty=float(args.action_smooth_penalty),
        train_base_seed=TRAIN_BASE_SEED,
        worker_seed_stride=WORKER_SEED_STRIDE,
        elapsed_s=elapsed, steps_per_sec=steps_per_sec,
        validation_history=val_cb.history, **stats,
    )
    summary_path = log_dir / ("smoke_run_summary.json" if args.smoke else "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)
    print(f"[train] run summary saved: {summary_path}")

    vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
