"""
experiments/exp_003_sensor_history/evaluate.py
================
保存済み PPO モデルを読み込み、mouse.corridor_eval.evaluate_corridor（gate 測定
ハーネス）を実行する薄いラッパ。

既定は gate 判定（seed 3000-3019 × 5 試行 = 100 試行）。日常の判断に使う検証帯は
`--split validation`（seed 5000-5019）を指定する。研究計画書 §9-7 のとおり、
gate 帯の成績を見て設定を選び直してはならない。

使い方:
    .venv/bin/python experiments/exp_003_sensor_history/evaluate.py               # gate
    .venv/bin/python experiments/exp_003_sensor_history/evaluate.py --split validation
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import PPO  # noqa: E402

from mouse.corridor_eval import (  # noqa: E402
    evaluate_corridor, DEFAULT_COURSE_DIR, VALIDATION_COURSE_DIR, DEFAULT_N_TRIALS,
)

OUTPUT_NAME = "exp_003_sensor_history"


def main(argv=None):
    parser = argparse.ArgumentParser(description="M1 exp_003_sensor_history 評価実行ラッパ")
    parser.add_argument("--model", type=str, default="models/exp_003_sensor_history.zip")
    parser.add_argument("--split", choices=["eval", "validation"], default="eval",
                        help="eval=gate判定(3000-3019) / validation=日常判断(5000-5019)")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true",
                         help="指定時のみ確率的方策で評価（既定は決定的）")
    parser.add_argument("--no-save", action="store_true", help="outputs/ へ保存しない")
    args = parser.parse_args(argv)

    course_dir = DEFAULT_COURSE_DIR if args.split == "eval" else VALIDATION_COURSE_DIR
    output_name = OUTPUT_NAME if args.split == "eval" else f"{OUTPUT_NAME}_validation"

    model = PPO.load(args.model)
    deterministic = not args.stochastic

    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    summary = evaluate_corridor(
        policy_fn, course_dir=course_dir, n_trials=args.n_trials,
        deterministic=deterministic, seed=args.seed,
        save_output=not args.no_save, output_name=output_name,
        obs_dist_diff=True,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
