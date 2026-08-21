"""
verification/audit_052_angle_vs_model.py
================
実機の「壁の角度と距離を振った」実測（`assets/angle_vs_sensor.csv`）と、
物理モデル（`mouse/ir_sensor.py::response()`）を突き合わせる。

`verification/audit_051_measured_vs_model.py`（正対したときの距離依存）の続きで、
`research_notes/note_034_ir_sensor_model.md` の問い C のうち **入射角の依存** を扱う。

測定の条件（ユーザの説明・2026-08-21）:
  - **センサ 1 本ずつ別々に測ったもの**（4 本を同時に測ったのではない）
  - **光軸が壁に垂直に当たる状態を 0 度**とし、そこから**壁の角度**と距離を変えた
  - `range` = 距離 [mm]、`angle` = 壁の角度 [deg]、値は AD 変換の値そのもの

したがって幾何は「光軸に沿った距離を `range` に保ったまま、入射角だけを `angle` にする」
（本ファイルの `axis_fixed_distance()`）。センサを壁の法線から `angle` だけ傾けた位置に置き、
光軸が壁上の同じ点を向くようにすれば、これと同じ配置になる。

当てはめ方:
  センサごとの利得は未知なので、**各センサ・各距離で角度 0 の値で規格化**して形だけを比べる
  （規格化すれば利得は落ちる）。自由パラメータは反射面の性質（鏡面成分ととがり）と
  角度のずれの 3 つだけである。

使い方:
  .venv/bin/python verification/audit_052_angle_vs_model.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import Rect
from mouse.ir_sensor import IrSensorSpec, SurfaceSpec, response

CSV_PATH = REPO_ROOT / "assets" / "angle_vs_sensor.csv"
SENSOR_NAMES = ("LF", "LS", "RS", "RF")

# 壁の手前面の x 座標。センサをこの手前に置く。値そのものに意味はない。
WALL_FACE_X = 0.5
# 壁は十分広く取る（半値角 5° の照射円は 200mm でも直径 35mm 程度）。
WALL = [Rect(cx=WALL_FACE_X + 0.006, cy=0.0, hx=0.006, hy=0.6)]
MOUNT_HEIGHT_M = 0.010


def load_measurement():
    """CSV を {(range_mm, angle_deg): {sensor: ad}} と、range/angle の一覧にして返す。"""
    table = {}
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (int(row["range"]), int(row["angle"]))
            table[key] = {n: int(row[n]) for n in SENSOR_NAMES}
    ranges = sorted({k[0] for k in table}, reverse=True)
    angles = sorted({k[1] for k in table})
    return table, ranges, angles


def axis_fixed_distance(range_m: float, angle_deg: float, surf: SurfaceSpec) -> float:
    """光軸に沿った距離を `range_m` に保ったまま、入射角を `angle_deg` にしたときの応答。

    壁を回すのと等価な配置である（壁上の同じ点を、法線から `angle_deg` 傾いた向きから
    同じ距離で見る）。
    """
    t = math.radians(angle_deg)
    pos = (WALL_FACE_X - range_m * math.cos(t), -range_m * math.sin(t), MOUNT_HEIGHT_M)
    axis = (math.cos(t), math.sin(t), 0.0)
    spec = IrSensorSpec(name="probe", pos=pos, axis=axis)
    return response(spec, (0.0, 0.0, 0.0), WALL, surf, include_floor=True, occlusion=True)


def measured_profile(table, ranges, angles):
    """4 本 × 4 距離を、それぞれの角度 0 の値で規格化して平均した角度の形を返す。"""
    out = []
    for a in angles:
        vals = [table[(rg, a)][n] / table[(rg, 0)][n]
                for rg in ranges for n in SENSOR_NAMES if (rg, a) in table]
        out.append(float(np.mean(vals)))
    return np.array(out)


def model_profile(angles, surf, angle_offset_deg=0.0, range_m=0.180):
    base = axis_fixed_distance(range_m, angle_offset_deg, surf)
    return np.array([axis_fixed_distance(range_m, a + angle_offset_deg, surf) / base
                     for a in angles])


def fit_surface(angles, meas,
                speculars=(0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25),
                shininesses=(15.0, 25.0, 35.0, 50.0, 80.0, 120.0),
                offsets=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0)):
    """鏡面成分・とがり・角度のずれを走査し、残差 RMS が最小の組を返す。"""
    best = None
    for sv in speculars:
        for sh in shininesses:
            for off in offsets:
                surf = SurfaceSpec(diffuse=0.8, specular=sv, shininess=sh)
                vals = model_profile(angles, surf, off)
                rms = float(np.sqrt(np.mean((vals - meas) ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, sv, sh, off, vals)
    return best


def main():
    table, ranges, angles = load_measurement()
    meas = measured_profile(table, ranges, angles)

    print("=" * 74)
    print("問い C のうち 入射角の依存（assets/angle_vs_sensor.csv）")
    print("=" * 74)
    print(f"  距離 {ranges} mm / 角度 {angles} deg / センサ {list(SENSOR_NAMES)}")

    print("\n[実測] 各センサ・各距離で角度 0 で規格化し、4 本 4 距離を平均した形")
    print("  角度 :", " ".join(f"{a:+6d}" for a in angles))
    print("  実測 :", " ".join(f"{v:6.2f}" for v in meas))

    print("\n[鏡面成分なし（現行の既定 specular=0.0）]")
    plain = model_profile(angles, SurfaceSpec(diffuse=0.8, specular=0.0))
    rms_plain = float(np.sqrt(np.mean((plain - meas) ** 2)))
    print("  モデル:", " ".join(f"{v:6.2f}" for v in plain), f"  残差RMS {rms_plain:.4f}")

    print("\n[鏡面成分を入れて当てはめる]")
    rms, sv, sh, off, vals = fit_surface(angles, meas)
    print(f"  最良: 鏡面 {sv} / とがり {sh:.0f} / 角度のずれ {off:+.0f}°  残差RMS {rms:.4f}")
    print("  モデル:", " ".join(f"{v:6.2f}" for v in vals))
    print(f"  → 鏡面成分を入れると残差が {rms_plain / rms:.1f} 倍良くなる")

    print("\n[距離ごとに当てはめて、パラメータが揃うかを見る]")
    for rg in ranges:
        m = np.array([table[(rg, a)][n] / table[(rg, 0)][n]
                      for a in angles for n in SENSOR_NAMES if (rg, a) in table]
                     ).reshape(len(angles), -1).mean(axis=1)
        r2, s2, sh2, o2, _ = fit_surface(angles, m)
        print(f"  range={rg:3d}mm → 鏡面 {s2:4.2f} / とがり {sh2:5.1f} / ずれ {o2:+.0f}°  残差RMS {r2:.4f}")

    print("\n[センサごとに当てはめて、個体差を見る]")
    for n in SENSOR_NAMES:
        m = np.array([np.mean([table[(rg, a)][n] / table[(rg, 0)][n]
                               for rg in ranges if (rg, a) in table]) for a in angles])
        r2, s2, sh2, o2, _ = fit_surface(angles, m)
        print(f"  {n} → 鏡面 {s2:4.2f} / とがり {sh2:5.1f} / ずれ {o2:+.0f}°  残差RMS {r2:.4f}")


if __name__ == "__main__":
    main()
