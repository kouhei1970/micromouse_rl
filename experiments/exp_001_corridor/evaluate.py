"""
experiments/exp_001_corridor/evaluate.py
================
保存済み PPO モデルを読み込み、mouse.corridor_eval.evaluate_corridor（gate 測定
ハーネス、凍結扱い）を実行する薄いラッパ。

使い方:
    .venv/bin/python experiments/exp_001_corridor/evaluate.py \\
        --model models/exp_001_corridor.zip
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import PPO  # noqa: E402

from mouse.corridor_eval import evaluate_corridor, DEFAULT_COURSE_DIR, DEFAULT_N_TRIALS  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="M1 exp_001_corridor 評価実行ラッパ")
    parser.add_argument("--model", type=str, default="models/exp_001_corridor.zip")
    parser.add_argument("--course-dir", type=str, default=DEFAULT_COURSE_DIR)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true",
                         help="指定時のみ確率的方策で評価（既定は決定的）")
    args = parser.parse_args(argv)

    model = PPO.load(args.model)
    deterministic = not args.stochastic

    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    summary = evaluate_corridor(
        policy_fn, course_dir=args.course_dir, n_trials=args.n_trials,
        deterministic=deterministic, seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
