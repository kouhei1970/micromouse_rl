"""
verification/audit_056_stage1_independent.py
================
AUDIT_056 工程1'（面積分と反射 1 回の光線追跡の突き合わせ）を、教授セッションが独立に計算する。

`audit_056_anchor_independent.py` が残した一次記録（`outputs/audit_056/anchor_independent.json`）に、
**乱数種 777001 の反射 1 回の光線追跡の値がすでに 200 姿勢ぶん入っている**ので、
同じ 200 姿勢の面積分だけを計算して突き合わせる。

判定量（`AUDIT_056` 追記2 →`AUDIT_050` §4-2 と同一）:
    M1 = |î_光線追跡(反射1回) − î_面積分| の 95 パーセンタイル（満量比）
    M1 / σ95 を  ≤2 / ≤10 / >10  で分割する

否定対照（`AUDIT_050` §6 と同じ形）: 光線追跡の LED 半値角を 3.0° → 7.0° にした版で M1' を出し、
`M1'/σ95 > 10` に入ることを確かめる（検査が作動することの実測）。

使い方:
  .venv/bin/python verification/audit_056_stage1_independent.py            # M1
  .venv/bin/python verification/audit_056_stage1_independent.py --control --lo 0 --hi 200
  .venv/bin/python verification/audit_056_stage1_independent.py --summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import wall_obstacles                      # noqa: E402
from mouse.ir_sensor import IrSensorSpec, SurfaceSpec, response  # noqa: E402
from mouse.params import RobotParams                             # noqa: E402
from verification.audit_050_raycast import raycast_response, sensors_from_params  # noqa: E402
from verification.audit_056_anchor_independent import (          # noqa: E402
    I_FULL, LED_HALF_DEG, MAZE, N_POSES, OUT_DIR, POSE_SEED, PT_HALF_DEG,
    SEED_A, SEPARATION_M, sample_poses,
)

N_RAYS = 480_000
CONTROL_LED_HALF_DEG = 7.0


def build():
    npz = np.load(MAZE)
    params = RobotParams()
    sensors = sensors_from_params(params)
    cell = float(params.cell_size)
    v_walls, h_walls = npz["v_walls"], npz["h_walls"]
    height, width = h_walls.shape[0] - 1, v_walls.shape[1] - 1
    rects = wall_obstacles(v_walls, h_walls, cell_size=cell)
    poses = sample_poses(N_POSES, POSE_SEED, width, height, cell, sensors)
    return rects, poses


def area_integral(sensor, pose, rects):
    """面積分。工程1' は鏡面なしで比べる（光線追跡に鏡面成分が無いため。追記1）。"""
    spec = IrSensorSpec(
        name=sensor.name, pos=tuple(sensor.pos), axis=tuple(sensor.axis),
        separation_m=SEPARATION_M,
        led_half_angle_deg=LED_HALF_DEG, pt_half_angle_deg=PT_HALF_DEG,
    )
    surf = SurfaceSpec(diffuse=0.8, specular=0.0)
    return response(spec, pose, rects, surf, occlusion=True, include_floor=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true", help="否定対照（LED 半値角 7°）を計算する")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=N_POSES)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    anchor = json.loads((OUT_DIR / "anchor_independent.json").read_text())
    ai_path = OUT_DIR / "stage1_area_independent.json"
    ctl_path = OUT_DIR / "stage1_control_independent.json"

    if args.summary:
        ai = json.loads(ai_path.read_text())
        keys = sorted(ai, key=int)
        ray = np.array([anchor[k]["a"] for k in keys])
        area = np.array([ai[k] for k in keys])
        m1 = np.percentile(np.abs(ray - area) / I_FULL, 95)
        sigma95 = np.percentile(
            np.abs(np.array([anchor[k]["a"] - anchor[k]["b"] for k in sorted(anchor, key=int)])) / I_FULL, 95)
        print(f"[工程1' 独立計算] n={len(keys)}  半値角 LED {LED_HALF_DEG}°/PT {PT_HALF_DEG}°・鏡面なし")
        print(f"  σ95 = {sigma95:.5f}（錨・独立再計算）")
        print(f"  M1  = {m1:.5f}   M1/σ95 = {m1/sigma95:.2f}")
        if ctl_path.exists():
            ctl = json.loads(ctl_path.read_text())
            ck = sorted(ctl, key=int)
            m1c = np.percentile(
                np.abs(np.array([ctl[k] for k in ck]) - np.array([ai[k] for k in ck])) / I_FULL, 95)
            print(f"  否定対照（LED 7°）: n={len(ck)}  M1' = {m1c:.5f}  M1'/σ95 = {m1c/sigma95:.1f}")
        return

    rects, poses = build()
    path = ctl_path if args.control else ai_path
    rec = json.loads(path.read_text()) if path.exists() else {}
    for i in range(args.lo, min(args.hi, N_POSES)):
        if str(i) in rec:
            continue
        sensor, pose = poses[i]
        if args.control:
            rec[str(i)] = raycast_response(
                sensor, pose, rects, n_rays=N_RAYS, seed=SEED_A, max_bounces=1,
                led_half_angle_deg=CONTROL_LED_HALF_DEG, pt_half_angle_deg=PT_HALF_DEG,
                separation_m=SEPARATION_M)
        else:
            rec[str(i)] = area_integral(sensor, pose, rects)
        path.write_text(json.dumps(rec))
    print(f"完了: {len(rec)}/{N_POSES} → {path}")


if __name__ == "__main__":
    main()
