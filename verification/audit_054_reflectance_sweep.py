"""
verification/audit_054_reflectance_sweep.py
================
壁の拡散反射率 ρ を実測できないため、**既存データへのパラメータ走査で縛れるか**を調べ、
縛れない場合に **ρ の不確かさが判断にどう効くか** を測る（ユーザ提案・2026-08-21）。

## 使える性質

反射 k 回の経路は必ず ρ の k 乗を持つので、応答は

    総応答(ρ) = Σ_k ρ^k · A_k

と分解できる。`A_k`（幾何だけで決まる係数）を反射率 1.0 で一度計算しておけば、
任意の ρ を解析的に評価できる。光線追跡を ρ ごとに回し直す必要がない。

## 分かったこと（2026-08-21・`research_notes/note_034` 追記13）

1. **既存の実測では ρ を決められない。**測定の場面（壁 1 枚＋床）では多重反射が
   全体の 2〜5% しか占めず、距離によってもほとんど動かないため、形に効かない
2. **ρ は判断に効く。**実迷路での M2（多重反射の寄与の 95 パーセンタイル）は
   ρ=0.5 で 0.0074（帯1）、ρ=0.6 で 0.0128（帯2）と、**ρ≈0.55 で帯が変わる**

使い方:
  .venv/bin/python verification/audit_054_reflectance_sweep.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import Rect, wall_obstacles
from mouse.params import RobotParams
from verification.audit_050_raycast import Sensor, raycast_response, sensors_from_params

I_FULL = 0.8298934
RHOS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# ユーザ提供の実測（mouse/data/ir_measured_front_20260821.csv の抜粋）
DIST_MM = np.array([200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 50], dtype=float)
AD_RIGHT = np.array([431, 529, 670, 859, 1145, 1585, 1851, 2240, 2766, 3227, 3373], dtype=float)


def bounce_coeffs(sensor, pose, rects, *, n_rays, seed, **kw):
    """反射 k 回ぶんの寄与 A_1..A_4 を、反射率 1.0 で計算する。"""
    vals = [raycast_response(sensor, pose, rects, n_rays=n_rays, seed=seed,
                             max_bounces=k, diffuse=1.0, **kw) for k in (1, 2, 3, 4)]
    return [vals[0], vals[1] - vals[0], vals[2] - vals[1], vals[3] - vals[2]]


def front_sensor():
    p = RobotParams()
    s = next(x for x in p.sensors if x["name"] == "LF")
    pos = tuple(float(v) for v in s["pos"].split())
    axis = tuple(float(v) for v in s["zaxis"].split())
    return Sensor(name="LF", pos=pos, axis=axis), pos


def part1_measurement_scene(n_rays=120_000, separation_m=0.0060):
    """実測の場面（壁 1 枚＋床）で、多重反射が全体の何割を占めるかを距離ごとに出す。"""
    sensor, pos = front_sensor()
    print("[1] 実測の場面（壁 1 枚＋床）で多重反射が占める割合（ρ=0.8）")
    rows = []
    for x in DIST_MM:
        wall = [Rect(cx=pos[0] + x / 1000.0 + 0.006, cy=0.0, hx=0.006, hy=0.30)]
        A = bounce_coeffs(sensor, (0.0, 0.0, 0.0), wall, n_rays=n_rays, seed=777001,
                          include_floor=True, led_half_angle_deg=3.0, pt_half_angle_deg=6.0,
                          max_range_m=0.35, wall_height_m=0.05, separation_m=separation_m)
        rows.append(A)
        tot = sum(0.8 ** (k + 1) * A[k] for k in range(4))
        ind = sum(0.8 ** (k + 1) * A[k] for k in range(1, 4))
        print(f"  {x:4.0f}mm: {ind / tot * 100:5.1f}%")
    A = np.array(rows)

    print("\n  ρ を振ったときの、右センサ 50〜200mm への当てはまり（利得は自由）")
    for rho in RHOS:
        m = A @ np.array([rho, rho ** 2, rho ** 3, rho ** 4])
        g = float(np.sum(m * AD_RIGHT) / np.sum(m ** 2))
        r = float(np.sqrt(np.mean(((g * m - AD_RIGHT) / AD_RIGHT) ** 2))) * 100
        print(f"    ρ={rho:.1f}: 残差 {r:5.2f}%")
    print("  → 差が小さい。**既存データでは ρ を決められない**")


def part2_maze_decision(n_poses=80, n_rays=15_000):
    """実迷路で、ρ を変えたときの M2（多重反射の寄与の 95 パーセンタイル）を出す。"""
    p = RobotParams()
    data = np.load(REPO_ROOT / "competition/mazes/design_turn_v1/maze_41001.npz")
    rects = wall_obstacles(data["v_walls"], data["h_walls"], cell_size=p.cell_size)
    width = int(data["v_walls"].shape[0] - 1)
    height = int(data["v_walls"].shape[1])
    cell = p.cell_size
    sensors = sensors_from_params(p)
    rng = np.random.default_rng(20250821)
    poses = []
    for _ in range(200):
        cx = rng.integers(0, width); cy = rng.integers(0, height)
        x = (cx + 0.5) * cell + rng.uniform(-0.04, 0.04)
        y = (cy + 0.5) * cell + rng.uniform(-0.04, 0.04)
        th = rng.uniform(-math.pi, math.pi)
        poses.append((sensors[rng.integers(0, len(sensors))], (x, y, th)))

    A = np.array([bounce_coeffs(sp, po, rects, n_rays=n_rays, seed=777001)
                  for sp, po in poses[:n_poses]])
    print(f"\n[2] 実迷路 {n_poses} 姿勢。ρ ごとの M2（事前登録 §5-1 の主判定量）")
    print("    ρ      M2      中央値   事前登録の分割")
    for rho in RHOS:
        ind = (rho ** 2 * A[:, 1] + rho ** 3 * A[:, 2] + rho ** 4 * A[:, 3]) / I_FULL
        m2 = float(np.percentile(np.abs(ind), 95))
        band = ("帯1（≤0.01 単一反射で足りる）" if m2 <= 0.01
                else "帯2（0.01〜0.05 無視できない）" if m2 <= 0.05
                else "帯3（>0.05 作り直し）")
        print(f"   {rho:4.1f} {m2:.5f} {float(np.median(np.abs(ind))):.5f}   {band}")
    print("  → **ρ ≈ 0.55 で帯が変わる。**ρ を決められないことが、そのまま判断の不確かさになる")


def main():
    print("=" * 78)
    print("壁の拡散反射率 ρ: 既存データで縛れるか / 縛れないと何が変わるか")
    print("=" * 78)
    part1_measurement_scene()
    part2_maze_decision()


if __name__ == "__main__":
    main()
