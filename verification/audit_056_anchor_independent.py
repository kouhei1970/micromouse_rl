"""
verification/audit_056_anchor_independent.py
================
AUDIT_056 工程1' の錨 `σ95` を、教授セッションが独立に再計算する
（`docs/RESEARCH_PLAN.md` §12-9 (b)「錨の独立再計算スクリプト」）。

錨の定義（`AUDIT_056` 追記2 →`AUDIT_050` §4-1）:
  同じ 200 姿勢を、**乱数種だけを変えた 2 回の反射 1 回の光線追跡**で走らせたときの、
  満量比の差の絶対値の 95 パーセンタイル。光線本数は 480,000 本（`AUDIT_050` 追記2 の決定）。

`AUDIT_050` の `σ95` = 0.00168 は**旧半値角 5°/5° で測った値**なので流用しない
（`AUDIT_056` 追記1）。本スクリプトは**更新後の半値角 LED 3.0°・PT 6.0°・離隔 6.0mm・
鏡面なし**で測り直す。

使い方（10 分以内に分割して走らせるため、姿勢の範囲を指定できる）:
  .venv/bin/python verification/audit_056_anchor_independent.py --lo 0 --hi 50
  ...
  .venv/bin/python verification/audit_056_anchor_independent.py --summary
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import wall_obstacles          # noqa: E402
from mouse.params import RobotParams                 # noqa: E402
from verification.audit_050_raycast import raycast_response, sensors_from_params  # noqa: E402

# --- 事前登録で固定されている値（AUDIT_050 §2-1・§2-2、AUDIT_056 追記1・追記2） ---
MAZE = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "maze_41001.npz"
N_POSES = 200
POSE_SEED = 20250821
N_RAYS = 480_000
SEED_A, SEED_B = 777001, 777002
I_FULL = 0.8298934
LED_HALF_DEG = 3.0      # 更新後の既定値（note_034 追記14）
PT_HALF_DEG = 6.0
SEPARATION_M = 0.0060
OUT_DIR = REPO_ROOT / "outputs" / "audit_056"


def load_maze():
    d = np.load(MAZE)
    return d


def sample_poses(n, seed, width, height, cell, sensors):
    """AUDIT_050 §2-1 の手順（区画を無作為→±40mm→方位一様→センサ1本）。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        cx = rng.integers(0, width)
        cy = rng.integers(0, height)
        x = (cx + 0.5) * cell + rng.uniform(-0.04, 0.04)
        y = (cy + 0.5) * cell + rng.uniform(-0.04, 0.04)
        theta = rng.uniform(-math.pi, math.pi)
        sensor = sensors[rng.integers(0, len(sensors))]
        out.append((sensor, (float(x), float(y), float(theta))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=N_POSES)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "anchor_independent.json"
    record = json.loads(out_path.read_text()) if out_path.exists() else {}

    if args.summary:
        keys = sorted(record, key=int)
        if len(keys) < N_POSES:
            print(f"未完了: {len(keys)}/{N_POSES} 姿勢")
        diffs = np.array([abs(record[k]["a"] - record[k]["b"]) for k in keys]) / I_FULL
        print(f"[錨の独立再計算] n={len(keys)} 光線 {N_RAYS} 本 種 {SEED_A}/{SEED_B}")
        print(f"  半値角 LED {LED_HALF_DEG}° / PT {PT_HALF_DEG}°、離隔 {SEPARATION_M*1000:.1f}mm、鏡面なし")
        print(f"  σ95 (満量比の差の95パーセンタイル) = {np.percentile(diffs, 95):.5f}")
        print(f"  中央値 = {np.median(diffs):.6f}  最大 = {diffs.max():.5f}")
        return

    npz = load_maze()
    params = RobotParams()
    sensors = sensors_from_params(params)
    cell = float(params.cell_size)
    v_walls, h_walls = npz["v_walls"], npz["h_walls"]
    height, width = h_walls.shape[0] - 1, v_walls.shape[1] - 1
    rects = wall_obstacles(v_walls, h_walls, cell_size=cell)

    poses = sample_poses(N_POSES, POSE_SEED, width, height, cell, sensors)
    for i in range(args.lo, min(args.hi, N_POSES)):
        if str(i) in record:
            continue
        sensor, pose = poses[i]
        vals = {}
        for tag, sd in (("a", SEED_A), ("b", SEED_B)):
            vals[tag] = raycast_response(
                sensor, pose, rects, n_rays=N_RAYS, seed=sd, max_bounces=1,
                led_half_angle_deg=LED_HALF_DEG, pt_half_angle_deg=PT_HALF_DEG,
                separation_m=SEPARATION_M,
            )
        record[str(i)] = vals
        out_path.write_text(json.dumps(record))
        print(f"  {i:3d}: a={vals['a']:.6f} b={vals['b']:.6f} |差|/満量={abs(vals['a']-vals['b'])/I_FULL:.6f}",
              flush=True)


if __name__ == "__main__":
    main()
