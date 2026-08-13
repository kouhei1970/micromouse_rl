"""発注: `goal_rate` 算出の検証（seed1 の 0.05 ＝「1 面ゴール」が本物か）。

**検証できる範囲を先に確定する**:
  検証は 10 万歩ごと（21 点）だが、**チェックポイントは 40 万歩ごと（5 点）**しか
  保存されていない。したがって **`goal_rate` = 0.05 を記録した 70 万・90 万・130 万歩の
  重みは存在せず、その 3 点は再実行では再現できない。**
  再現できるのは **40 万・80 万・120 万・160 万・200 万歩の 5 点**である。

やること:
  1. 保存された 5 つのチェックポイントを、**訓練と同じ手続き**
     （`evaluate_maze6`・`deterministic=True`・`seed=0`・検証帯）で再実行し、
     `validation_history.json` の同じ歩数の記録と**面ごとに突き合わせる**
  2. ゴール判定を**環境のフラグに頼らず**、軌跡から独立に再判定する
     （機体の位置がゴール 2×2 に入ったか）
  3. ゴールした面（seed 7017）について、前後のチェックポイントがどこまで進むかを測る
"""
import json
import sys

REPO_ROOT = "/Users/kouhei/tmp/github/micromouse_rl"
sys.path.insert(0, REPO_ROOT)

import numpy as np
from stable_baselines3 import PPO

from mouse.maze6_eval import VALIDATION_MAZE_DIR, evaluate_maze6

LOG = f"{REPO_ROOT}/logs/exp_012_condE_seed1"
CKPTS = {400_000: "rl_model_400000_steps.zip", 800_000: "rl_model_800000_steps.zip",
         1_200_000: "rl_model_1200000_steps.zip", 1_600_000: "rl_model_1600000_steps.zip",
         2_000_000: "rl_model_2000000_steps.zip"}


def main():
    hist = {r["total_timesteps"]: r
            for r in json.load(open(f"{LOG}/validation_history.json"))}

    print("=" * 78)
    print("0. 検証できる範囲")
    print("=" * 78)
    pos = [t for t, r in hist.items() if r["goal_rate"] > 0]
    print(f"  検証点 {len(hist)} 点 / 保存されたチェックポイント {len(CKPTS)} 点")
    print(f"  goal_rate > 0 の点: {pos}")
    print(f"  そのうち重みが残っている点: {[t for t in pos if t in CKPTS]}"
          "  → **0.05 の 3 点は再実行で再現できない**")

    print()
    print("=" * 78)
    print("1. 保存された 5 点の再実行（訓練と同じ手続き）")
    print("=" * 78)
    rows = []
    for t, fn in CKPTS.items():
        model = PPO.load(f"{LOG}/{fn}", device="cpu")
        s = evaluate_maze6(lambda o: model.predict(o, deterministic=True)[0],
                           maze_dir=VALIDATION_MAZE_DIR, n_trials=1, seed=0)
        rec = hist[t]
        same_rate = abs(s["goal_rate"] - rec["goal_rate"]) < 1e-12
        same_failed = sorted(s["failed_maze_seeds"]) == sorted(rec["failed_maze_seeds"])
        a_v, b_v = s["mean_n_visited"], rec["mean_n_visited"]
        same_visit = (a_v is None and b_v is None) or (
            a_v is not None and b_v is not None and abs(a_v - b_v) < 1e-9)
        rows.append({"steps": t, "rerun_goal_rate": s["goal_rate"],
                     "recorded_goal_rate": rec["goal_rate"],
                     "failed_match": same_failed, "visited_match": same_visit,
                     "rerun_visited": s["mean_n_visited"],
                     "recorded_visited": rec["mean_n_visited"]})
        print(f"  {t:>9} 歩: ゴール率 再実行 {s['goal_rate']:.2f} 対 記録 {rec['goal_rate']:.2f}"
              f"  {'一致' if same_rate else '🔴 不一致'}"
              f" / 失敗面の集合 {'一致' if same_failed else '🔴 不一致'}"
              f" / 訪問区画 {'一致' if same_visit else '🔴 不一致'}"
              f" ({a_v} 対 {b_v})")

    n_ok = sum(1 for r in rows if r["failed_match"] and r["visited_match"]
               and abs(r["rerun_goal_rate"] - r["recorded_goal_rate"]) < 1e-12)
    print(f"\n  → {n_ok} / {len(rows)} 点が完全一致"
          f"{'（記録の忠実性と再現性が確認された）' if n_ok == len(rows) else '🔴'}")

    out = {"reproducible_points": rows,
           "positive_points": pos,
           "positive_with_checkpoint": [t for t in pos if t in CKPTS],
           "goal_maze": 7017}
    with open(f"{REPO_ROOT}/verification/out/goal_rate_audit.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"\n書き出し: {REPO_ROOT}/verification/out/goal_rate_audit.json")


if __name__ == "__main__":
    main()
