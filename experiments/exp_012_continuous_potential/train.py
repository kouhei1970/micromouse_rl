"""
experiments/exp_012_continuous_potential/train.py
===================================================
exp_012（ポテンシャル Φ の連続化）の PPO 学習スクリプト。

`experiments/exp_010_m2_0/train.py` を土台にし、**exp_010 との差は
Φ の連続化 1 点のみ**（`Maze6Env(..., continuous_potential=True)`）。
1 実験 1 変更の原則（研究計画書 §9）を守り、他のハイパーパラメータは
**exp_010 と完全に同一**にする:

| 項目 | 値 |
|---|---|
| 行動の高周波成分への罰 | α=0.5・k=8.7e-3 |
| 並列環境数 | 1 |
| γ | 0.995 |
| 総ステップ | 200 万 |
| 定期評価の間隔 | 10 万ステップごと |

**背景（詳細は `experiments/exp_012_continuous_potential/design.md`）**:
exp_010/011 の学習ログ実測から、M2-0（迷路）で学習が定着しない主因は
「γ による割引の潰れ」でも「終点の報酬順序」でもなく、**進捗報酬（ポテンシャル
整形）が入るステップの時間分布**だと特定された。区画単位の階段関数 Φ では
進捗報酬が全ステップの 5.3%（区画境界を跨ぐ歩）にしか入らず、残り 94.7% には
「動くな」としか読めない密な負の信号（時間罰＋滑らかさの罰）だけが掛かる。
M1（廊下）は Φ が連続だったため進捗報酬が 100% のステップに一様に入っていた。
本実験は Φ を区画中心・壁開口部中点を結ぶ折れ線への射影による連続量へ置き換え、
この時間分布を M1 と同じ「密」な状態に戻す（`mouse/maze6_env.py` に実装済み。
既定 `continuous_potential=False` では階段版のまま bit 単位で不変）。

事前登録した予測・打ち切り基準・支持/棄却の判定基準は design.md §5・§6 が正本。
**⚠️ design.md §6 の打ち切り基準（検証帯の定期評価で 100 万歩までの 10 点すべてが
ゴール率 < 0.05 なら打ち切る）は、土台にした exp_010 の train.py に実装されておらず、
本スクリプトにも実装していない**（1 実験 1 変更のため、無断で追加せず指示者へ
報告済み）。運用時は `validation_history.json` を人手で確認し、必要なら
学習プロセスを手動で止めること。

**帯の分割**（研究計画書 §9-7）:

| 用途 | maze seed |
|---|---|
| 学習 | 8000 以降（予約帯は `Maze6Env._next_maze_seed()` が決定的に読み飛ばす） |
| 日常判断・チェックポイント選択 | 検証帯 **7000-7019** |
| gate 判定 | 評価帯 **6000-6019**（本スクリプトでは**測らない**） |

使い方:
    .venv/bin/python experiments/exp_012_continuous_potential/train.py --smoke     # 疎通確認
    .venv/bin/python experiments/exp_012_continuous_potential/train.py \
        --action-highpass-penalty 8.7e-3 --seed 1 \
        --log-dir logs/exp_012_cont_phi_seed1 --model-out models/exp_012_cont_phi_seed1.zip
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

from mouse.maze6_env import Maze6Env  # noqa: E402
from mouse.maze6_eval import VALIDATION_MAZE_DIR, evaluate_maze6  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

GAMMA = 0.995
TRAIN_BASE_SEED = 8000          # 学習用 maze seed の起点（予約帯は環境側が読み飛ばす）
WORKER_SEED_STRIDE = 1_000_000

SMOKE_TOTAL_STEPS = 100_000
FULL_TOTAL_STEPS = 2_000_000    # M1 の 2 倍。迷路はエピソードが長く探索も要るため
VALIDATION_EVERY_STEPS = 100_000


def make_env(rank: int, gamma: float, log_dir: Path, args):
    def _init():
        env = Maze6Env(
            mode="generate",
            base_seed=TRAIN_BASE_SEED + rank * WORKER_SEED_STRIDE,
            gamma=gamma,
            maze_mode=args.maze_mode,
            visit_bonus=args.visit_bonus,
            collision_penalty=args.collision_penalty,
            action_smooth_penalty=args.action_smooth_penalty,
            action_highpass_penalty=args.action_highpass_penalty,
            action_highpass_alpha=args.action_highpass_alpha,
            continuous_potential=True,  # exp_012: exp_010 との差はこの 1 点のみ
        )
        return Monitor(env, filename=str(log_dir / f"env_{rank}"))
    return _init


class EpisodeStatsCallback(BaseCallback):
    """行動振動の指標と、エピソードごとの maze seed を記録する。

    M1 と同じ様式。`ep_rew_mean` は性能の判断に使わない（研究計画書 §9）。
    """

    def __init__(self, n_envs: int, seed_log_path: Path = None, n_track: int = 100):
        super().__init__()
        self.n_envs = n_envs
        self._prev_sign = np.zeros((n_envs, 2))
        self._flips = np.zeros((n_envs, 2), dtype=int)
        self._steps = np.zeros(n_envs, dtype=int)
        self._diff_sq = np.zeros(n_envs)
        self._prev_action = np.zeros((n_envs, 2))
        self._ep_flip_rates, self._ep_diff_rms = [], []
        self._n_track = n_track
        self._seed_log_path = Path(seed_log_path) if seed_log_path else None
        self._seed_file = None
        self._written = 0

    def _on_training_start(self) -> None:
        if self._seed_log_path is not None:
            self._seed_file = open(self._seed_log_path, "w", encoding="utf-8")

    def _on_step(self) -> bool:
        acts = np.asarray(self.locals["actions"], dtype=np.float64).reshape(self.n_envs, 2)
        s = np.sign(acts)
        self._flips += (s * self._prev_sign < 0).astype(int)
        self._prev_sign = s
        self._diff_sq += ((acts - self._prev_action) ** 2).sum(axis=1)
        self._prev_action = acts
        self._steps += 1

        dt = RobotParams().control_dt
        for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            if not done:
                continue
            t = max(self._steps[i] * dt, 1e-9)
            self._ep_flip_rates.append(self._flips[i] / t)
            self._ep_diff_rms.append(np.sqrt(self._diff_sq[i] / max(self._steps[i], 1)))
            if len(self._ep_flip_rates) > self._n_track:
                self._ep_flip_rates.pop(0)
                self._ep_diff_rms.pop(0)
            if self._seed_file is not None:
                self._seed_file.write(json.dumps(dict(
                    step=int(self.num_timesteps),
                    maze_seed=int(info.get("maze_seed", -1)),
                    outcome=("goal" if info.get("goal") else
                             "collision" if info.get("collision") else "timeout"),
                    n_visited=int(info.get("n_visited", 0)),
                    odom_error_m=float(info.get("odom_error_m", float("nan"))),
                ), ensure_ascii=False) + "\n")
                self._written += 1
                if self._written % 200 == 0:
                    self._seed_file.flush()
            self._flips[i] = 0
            self._steps[i] = 0
            self._diff_sq[i] = 0.0
            self._prev_sign[i] = 0.0
            self._prev_action[i] = 0.0
        return True

    def _on_training_end(self) -> None:
        if self._seed_file is not None:
            self._seed_file.close()
            self._seed_file = None

    def summary(self) -> dict:
        if not self._ep_flip_rates:
            return dict(action_diff_rms=None, sign_flip_rate_left_per_s=None,
                        sign_flip_rate_right_per_s=None)
        fr = np.mean(np.array(self._ep_flip_rates), axis=0)
        return dict(action_diff_rms=float(np.mean(self._ep_diff_rms)),
                    sign_flip_rate_left_per_s=float(fr[0]),
                    sign_flip_rate_right_per_s=float(fr[1]))


class ValidationCallback(BaseCallback):
    """**検証帯 7000-7019** でのゴール到達率を定期記録する。

    gate 判定（評価帯 6000-6019）はここでは**行わない**（研究計画書 §9-7）。
    """

    def __init__(self, eval_every_steps: int, log_dir: Path, gamma: float,
                 maze_mode: str, n_trials: int = 1):
        super().__init__()
        self.eval_every = int(eval_every_steps)
        self.log_dir = Path(log_dir)
        self.gamma = float(gamma)
        self.maze_mode = maze_mode
        self.n_trials = int(n_trials)
        self.history = []
        self._next_at = self.eval_every

    def _evaluate(self):
        t0 = time.time()
        s = evaluate_maze6(
            lambda o: self.model.predict(o, deterministic=True)[0],
            maze_dir=VALIDATION_MAZE_DIR, n_trials=self.n_trials, seed=0,
            gamma=self.gamma, maze_mode=self.maze_mode)
        rec = dict(total_timesteps=int(self.num_timesteps),
                   goal_rate=s["goal_rate"], collision_rate=s["collision_rate"],
                   timeout_rate=s["timeout_rate"],
                   mean_goal_time_s=s["mean_goal_time_s"],
                   mean_sec_per_cell=s["mean_sec_per_cell"],
                   mean_n_visited=s["mean_n_visited"],
                   mean_odom_error_m=s["mean_odom_error_m"],
                   sign_flip_rate_mean=s["sign_flip_rate_mean"],
                   i_rms_mean=s["i_rms_mean"],
                   failed_maze_seeds=s["failed_maze_seeds"],
                   eval_wall_time_s=time.time() - t0)
        self.history.append(rec)
        self.logger.record("validation/goal_rate", rec["goal_rate"])
        if rec["sign_flip_rate_mean"] is not None:
            self.logger.record("validation/sign_flip_rate_mean", rec["sign_flip_rate_mean"])
        with open(self.log_dir / "validation_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"[validation] t={rec['total_timesteps']} "
              f"ゴール率={rec['goal_rate']:.2f} "
              f"訪問={rec['mean_n_visited'] or float('nan'):.1f} 区画 "
              f"({rec['eval_wall_time_s']:.1f} s)", flush=True)
        return rec

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_at:
            self._evaluate()
            while self._next_at <= self.num_timesteps:
                self._next_at += self.eval_every
        return True

    def _on_training_end(self) -> None:
        if not self.history or self.history[-1]["total_timesteps"] != self.num_timesteps:
            self._evaluate()


def main(argv=None):
    p = argparse.ArgumentParser(description="exp_012（Φ の連続化）PPO 学習")
    p.add_argument("--total-steps", type=int, default=FULL_TOTAL_STEPS)
    p.add_argument("--smoke", action="store_true", help="10 万ステップで疎通確認")
    p.add_argument("--seed", type=int, default=0, help="PPO 本体の乱数 seed")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--n-envs", type=int, default=1,
                   help="並列環境数。exp_008 より並列 1 が実時間で最速")
    p.add_argument("--maze-mode", choices=["loop", "full"], default="loop")
    p.add_argument("--log-dir", type=str, default="logs/exp_012_cont_phi")
    p.add_argument("--model-out", type=str, default="models/exp_012_cont_phi.zip")
    p.add_argument("--validation-every", type=int, default=VALIDATION_EVERY_STEPS)
    p.add_argument("--visit-bonus", type=float, default=0.02)
    p.add_argument("--collision-penalty", type=float, default=-1.0)
    p.add_argument("--action-smooth-penalty", type=float, default=0.0,
                   help="exp_006 の ‖Δa‖² 版（M1 では案 3 に劣る。既定 0）")
    p.add_argument("--action-highpass-penalty", type=float, default=8.7e-3,
                   help="案 3。M1 で 3 基準を達成した値（既定 8.7e-3）")
    p.add_argument("--action-highpass-alpha", type=float, default=0.5)
    p.add_argument("--init-model", type=str, default=None,
                   help="初期重みにする学習済みモデル（微調整）")
    args = p.parse_args(argv)

    total_steps = SMOKE_TOTAL_STEPS if args.smoke else args.total_steps
    n_envs = 1 if args.smoke else args.n_envs
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(i, args.gamma, log_dir, args) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns) if n_envs == 1 else SubprocVecEnv(env_fns)
    print(f"[train] 観測空間 = {vec_env.observation_space}"
          f"（距離4 + 差分4 + ジャイロ1 + 加速度2 + 車輪2 + 前回行動2 + ゴール相対2）")
    print(f"[train] 迷路モード = {args.maze_mode} / 訪問報酬 = {args.visit_bonus} "
          f"/ 衝突罰 = {args.collision_penalty}")
    print(f"[train] 行動の高周波成分への罰 k = {args.action_highpass_penalty}"
          f"（α = {args.action_highpass_alpha}）")
    print(f"[train] 並列環境数 = {n_envs}")

    if args.init_model:
        model = PPO.load(args.init_model, env=vec_env, seed=args.seed, verbose=1)
        print(f"[train] 初期重みを {args.init_model} から読み込み（微調整）")
    else:
        # ハイパーパラメータは M1（exp_003〜006）・exp_010 と同一。1 実験 1 変更を守る
        model = PPO("MlpPolicy", vec_env,
                    learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
                    gamma=args.gamma, gae_lambda=0.95, ent_coef=0.0,
                    policy_kwargs=dict(net_arch=[128, 128]),
                    seed=args.seed, verbose=1)
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))

    stats_cb = EpisodeStatsCallback(n_envs=n_envs,
                                    seed_log_path=log_dir / "episode_seeds.jsonl")
    val_cb = ValidationCallback(eval_every_steps=args.validation_every, log_dir=log_dir,
                                gamma=args.gamma, maze_mode=args.maze_mode)
    ckpt_cb = CheckpointCallback(save_freq=max(400_000 // n_envs, 1),
                                 save_path=str(log_dir))

    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=[ckpt_cb, stats_cb, val_cb])
    elapsed = time.time() - t0
    print(f"[train] total_steps={total_steps} n_envs={n_envs} "
          f"elapsed={elapsed:.1f}s steps/s={total_steps / max(elapsed, 1e-9):.1f}")

    model_out = Path(args.model_out)
    if args.smoke:
        model_out = model_out.with_name(model_out.stem + "_smoke.zip")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))
    print(f"[train] model saved: {model_out}")

    with open(log_dir / ("smoke_run_summary.json" if args.smoke else "run_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(dict(
            experiment="exp_012_cont_phi", smoke=bool(args.smoke), total_steps=total_steps,
            n_envs=n_envs, seed=args.seed, gamma=args.gamma, maze_mode=args.maze_mode,
            visit_bonus=args.visit_bonus, collision_penalty=args.collision_penalty,
            action_smooth_penalty=args.action_smooth_penalty,
            action_highpass_penalty=args.action_highpass_penalty,
            action_highpass_alpha=args.action_highpass_alpha,
            init_model=args.init_model, train_base_seed=TRAIN_BASE_SEED,
            elapsed_s=elapsed, steps_per_sec=total_steps / max(elapsed, 1e-9),
            validation_history=val_cb.history, **stats_cb.summary()),
            f, indent=2, ensure_ascii=False)
    vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
