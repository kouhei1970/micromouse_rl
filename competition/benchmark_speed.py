"""
計算速度の実測（M0 タスク4）
Measure simulation throughput of the mouse_v2 16x16 maze environment.

新ロボット（mouse_v2、凍結モデル）+ 16x16 評価迷路環境で、
単一環境および SubprocVecEnv(6)（サブプロセス並列のベクトル化環境。
複数の環境を別プロセスで同時に走らせる stable-baselines3 の仕組み）の
steps/sec を実測する。今後の学習規模見積りに使う。

使い方 / Usage:
    .venv/bin/python competition/benchmark_speed.py [--steps 2000] [--n-envs 6]

計測方法:
- ランダム行動（action_space.sample）で warmup 後、固定ステップ数の壁時計時間を計測
- 「制御 steps/sec」= Gymnasium の step() 回数 / 秒。物理サブステップは 1 制御
  ステップあたり control_dt/physics_dt = 20 回（物理 steps/sec = 制御×20）
- 実時間倍率 = 制御 steps/sec × control_dt（シミュレーション時間が実時間の何倍で進むか）
"""
import argparse
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MAZE_XML = os.path.join(_REPO_ROOT, "competition", "mazes", "eval", "maze_1000.xml")


def make_env():
    from mouse.env import MouseMazeEnvV2
    return MouseMazeEnvV2(MAZE_XML)


def bench_single(n_steps: int) -> float:
    env = make_env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    # warmup（初回 JIT・キャッシュの影響を除く）
    for _ in range(100):
        env.step(rng.uniform(-0.3, 0.3, size=2).astype(np.float32))
    t0 = time.perf_counter()
    for _ in range(n_steps):
        obs, r, term, trunc, info = env.step(rng.uniform(-0.3, 0.3, size=2).astype(np.float32))
        if term or trunc:
            env.reset()
    dt = time.perf_counter() - t0
    env.close()
    return n_steps / dt


def bench_vec(n_steps: int, n_envs: int) -> float:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    vec = SubprocVecEnv([make_env for _ in range(n_envs)])
    vec.reset()
    rng = np.random.default_rng(0)
    for _ in range(50):
        vec.step(rng.uniform(-0.3, 0.3, size=(n_envs, 2)).astype(np.float32))
    t0 = time.perf_counter()
    for _ in range(n_steps):
        vec.step(rng.uniform(-0.3, 0.3, size=(n_envs, 2)).astype(np.float32))
    dt = time.perf_counter() - t0
    vec.close()
    return n_steps * n_envs / dt


def main():
    ap = argparse.ArgumentParser(description="mouse_v2 16x16 環境の steps/sec 実測")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n-envs", type=int, default=6)
    args = ap.parse_args()

    from mouse.params import RobotParams
    p = RobotParams()
    substeps = round(p.control_dt / p.physics_dt)

    print(f"環境: mouse_v2（凍結モデル）+ 16x16 評価迷路 (maze_1000)")
    print(f"制御周期 {1/p.control_dt:.0f} Hz / 物理 {1/p.physics_dt:.0f} Hz（{substeps} サブステップ）\n")

    sps1 = bench_single(args.steps)
    print(f"[単一環境]           制御 {sps1:8.0f} steps/s | 物理 {sps1*substeps:9.0f} steps/s | 実時間比 {sps1*p.control_dt:6.1f}x")

    spsN = bench_vec(args.steps, args.n_envs)
    print(f"[SubprocVecEnv({args.n_envs})]  制御 {spsN:8.0f} steps/s | 物理 {spsN*substeps:9.0f} steps/s | 実時間比 {spsN*p.control_dt:6.1f}x "
          f"| 並列化効率 {spsN/(sps1*args.n_envs)*100:.0f}%")


if __name__ == "__main__":
    main()
