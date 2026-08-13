"""保全済み「ゴールに届いた重み」7 点の検証（教授発注）。

`AUDIT_018` §3-bis と同型。7 点の内訳:
  - **E seed2 の 200 万歩**（`logs/exp_012_condE_seed2/rl_model_2000000_steps.zip`。
    §9-19 の制定前なので、40 万歩ごとのチェックポイントが**偶然**その点に当たった）
  - **C' seed2 の 4 点**・**C' seed3 の 2 点**（`models/..._first_goal_*.zip`。§9-19 の退避）

やること:
  1. **記録された `goal_rate` = 0.05 を再現するか**（訓練と同じ手続き）
  2. **どの面がゴールしたか**（$D_0$ 集合一致の直接確認）
  3. **競技規約の判定**（機体全体がゴール区画に内包されるか）を**自前の姿勢考慮の判定**で行う
     — 学生B は `evaluator.body_fully_inside` を直接呼んでいるので、**別実装での確認**になる
  4. **「届く状態」と「離脱後」のパラメータ距離**（次の保存点との差）
"""
import glob
import json
import math
import os
import sys

REPO_ROOT = "/Users/kouhei/tmp/github/micromouse_rl"
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from stable_baselines3 import PPO

from mouse.maze6_env import Maze6Env
from mouse.maze6_eval import VALIDATION_MAZE_DIR, _trial_seed, evaluate_maze6

# 機体の外形（裁定 R26。前後 ±0.0500・左右 ±0.0400 の矩形で保守側に判定する）
FRONT, LAT = 0.0500, 0.0400
GOAL_LO, GOAL_HI = 2 * 0.18, 4 * 0.18


def snapshots():
    """検証する 7 点（条件・歩数・重みのパス）。"""
    out = [("E", 2, 2_000_000, f"{REPO_ROOT}/logs/exp_012_condE_seed2/rl_model_2000000_steps.zip")]
    for p in sorted(glob.glob(f"{REPO_ROOT}/models/exp_012_condCp_seed*_first_goal_*.zip")):
        base = os.path.basename(p)
        seed = int(base.split("_seed")[1].split("_")[0])
        step = int(base.rsplit("_", 1)[1].replace(".zip", ""))
        out.append(("Cp", seed, step, p))
    return out


def body_inside(x, y, yaw):
    """機体全体がゴール 2×2 に内包されるか（**自前の姿勢考慮の判定**）。

    機体の 4 隅（前後 ±FRONT・左右 ±LAT）を姿勢で回して、全部が区画の内側かを見る。
    """
    c, s = math.cos(yaw), math.sin(yaw)
    for dx, dy in ((FRONT, LAT), (FRONT, -LAT), (-FRONT, LAT), (-FRONT, -LAT)):
        px = x + dx * c - dy * s
        py = y + dx * s + dy * c
        if not (GOAL_LO <= px <= GOAL_HI and GOAL_LO <= py <= GOAL_HI):
            return False
    return True


def run_face(model, maze_seed):
    """1 面を評価器と同じ構成で走らせ、軌跡から規約判定を行う。"""
    env = Maze6Env(maze_dir=VALIDATION_MAZE_DIR, maze_seeds=[maze_seed], max_cache=2,
                   gamma=0.995, mode="fixed", maze_mode="loop")
    obs, _ = env.reset(seed=_trial_seed(0, maze_seed, 0))
    n, center_in, contained, last = 0, 0, 0, None
    for _ in range(5000):
        a, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(np.clip(np.asarray(a, dtype=np.float64), -1, 1))
        x, y, yaw = env.sim.privileged_pose()
        n += 1
        if GOAL_LO <= x <= GOAL_HI and GOAL_LO <= y <= GOAL_HI:
            center_in += 1
        if body_inside(x, y, yaw):
            contained += 1
        last = (x, y, yaw)
        if term or trunc:
            break
    env.close()
    return {"n_steps": n, "env_goal": bool(info.get("goal")),
            "center_in_steps": center_in, "body_contained_steps": contained,
            "final_xy_yaw": [round(v, 6) for v in last]}


def params(path):
    m = PPO.load(path, device="cpu")
    return m, torch.cat([p.detach().flatten() for p in m.policy.parameters()])


def main():
    snaps = snapshots()
    print(f"検証する重み: {len(snaps)} 点\n")
    ledger = []
    for cond, seed, step, path in snaps:
        model, vec = params(path)
        s = evaluate_maze6(lambda o: model.predict(o, deterministic=True)[0],
                           maze_dir=VALIDATION_MAZE_DIR, n_trials=1, seed=0)
        got = [m for m in range(7000, 7020) if m not in s["failed_maze_seeds"]]
        row = {"condition": cond, "seed": seed, "steps": step,
               "goal_rate_rerun": s["goal_rate"], "goal_faces": got}
        if got:
            row.update(run_face(model, got[0]))
        ledger.append(row)
        c = row.get("body_contained_steps")
        print(f"  {cond} seed{seed} {step:>9} 歩: ゴール率 {s['goal_rate']:.2f}  面 {got}"
              + (f"  歩数 {row['n_steps']}  中心が入った歩 {row['center_in_steps']}"
                 f"  **機体全体の内包 {c}**" if got else ""))
    with open(f"{REPO_ROOT}/verification/out/goal_weights_ledger.json", "w") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"\n書き出し: {REPO_ROOT}/verification/out/goal_weights_ledger.json")
    return ledger


if __name__ == "__main__":
    main()
