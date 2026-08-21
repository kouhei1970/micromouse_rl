"""
verification/audit_053_peak_mechanism.py
================
応答が距離に対して非単調になる機構（近距離で LED の照射円が PT の視野に入らなくなる）が、
モデルで再現できているかを調べる。ユーザが実機で確認している現象である（2026-08-21）。

`research_notes/note_034_ir_sensor_model.md` 追記8 の再現手順。

3 つのことを調べる:

1. **否定対照**: LED と PT の離隔をゼロにすると山が消えるか。
   消えれば、山は確かに「照射円と視野円のずれ」から出ていることになる
   （壊して変われば、その経路は使われている）。
2. **機構の見積もり**: 半値角 5° の照射円の半値半径は `d·tan5°`。これが離隔の半分と
   釣り合う距離を手計算で出し、モデルの山の位置と照合する。
3. **既存の実測で山の始まりが見えているか**: `mouse/data/ir_measured_front_20260821.csv`
   の 50〜80mm は 1/d² よりずっと平らである。これを説明する候補を、自由パラメータの数を
   そろえて並べる。

使い方:
  .venv/bin/python verification/audit_053_peak_mechanism.py
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
from mouse.params import RobotParams

CSV_PATH = REPO_ROOT / "mouse" / "data" / "ir_measured_front_20260821.csv"
# note_034 追記7 で実測から求めた反射面の性質。
SURF = SurfaceSpec(diffuse=0.8, specular=0.12, shininess=50.0)
HALF_ANGLE_DEG = 5.0


def front_sensor_pos_axis():
    p = RobotParams()
    s = next(x for x in p.sensors if x["name"] == "LF")
    return (tuple(float(v) for v in s["pos"].split()),
            tuple(float(v) for v in s["zaxis"].split()))


def model_curve(dist_mm, separation_m, pos=None, axis=None):
    """センサから `dist_mm` 前方の正対する壁に対する応答。`dist_mm` は配列でもよい。"""
    if pos is None:
        pos, axis = front_sensor_pos_axis()
    spec = IrSensorSpec(name="LF", pos=pos, axis=axis, separation_m=separation_m)
    arr = np.atleast_1d(np.asarray(dist_mm, dtype=float))
    out = np.array([
        response(spec, (0.0, 0.0, 0.0),
                 [Rect(cx=pos[0] + x / 1000.0 + 0.006, cy=0.0, hx=0.006, hy=0.30)],
                 SURF, include_floor=True, occlusion=True)
        for x in arr])
    return out


def load_measurement():
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("range_mm"):
                continue
            a, b, c = line.split(",")
            rows.append((float(a), float(b), float(c)))
    rows.sort()
    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def rms_pct(pred, y):
    return float(np.sqrt(np.mean(((pred - y) / y) ** 2))) * 100.0


def negative_control():
    """離隔をゼロにすると山が消えることを確かめる。"""
    dd = np.arange(5.0, 120.0, 1.0)
    print("[1] 否定対照: 離隔をゼロにすると山は消えるか")
    for sep, label in ((0.0065, "離隔 6.5mm（実機の値）"), (0.0, "離隔 0（LED と PT を同じ点に）")):
        v = model_curve(dd, sep)
        monotone = bool(np.all(np.diff(v) < 0))
        print(f"  {label}: 山の位置 {dd[int(np.argmax(v))]:.0f}mm / "
              f"距離に対して単調減少か = {monotone}")
    print("  → 離隔を壊すと山が消える。山は照射円と視野円のずれから出ている")


def mechanism_estimate():
    """照射円の半値半径と離隔の釣り合いから、山の位置を手計算で見積もる。"""
    t = math.tan(math.radians(HALF_ANGLE_DEG))
    sep_mm = 6.5
    d_cross = sep_mm / (2.0 * t)
    print(f"\n[2] 機構の見積もり（半値角 {HALF_ANGLE_DEG}°・離隔 {sep_mm}mm）")
    print(f"  照射円の半値半径 = d·tan{HALF_ANGLE_DEG}° = {t:.4f}·d")
    print(f"  これが離隔の半分と釣り合う距離 = {sep_mm}/(2·{t:.4f}) = {d_cross:.1f} mm")
    dd = np.arange(20.0, 90.0, 1.0)
    v = model_curve(dd, 0.0065)
    print(f"  モデルの山の位置 = {dd[int(np.argmax(v))]:.0f} mm（手計算の少し外側）")


def compare_explanations():
    """50〜80mm の平らさを説明する候補を、自由パラメータの数をそろえて並べる。"""
    d_mm, _left, right = load_measurement()
    m_peak = model_curve(d_mm, 0.0065)
    m_flat = model_curve(d_mm, 0.0)

    print("\n[3] 既存の実測（右センサ・50〜200mm）を説明する候補")
    g1 = float(np.sum(m_peak * right) / np.sum(m_peak ** 2))
    g2 = float(np.sum(m_flat * right) / np.sum(m_flat ** 2))
    print(f"  [a] 山あり（離隔6.5mm）・飽和なし   自由度1  残差 {rms_pct(g1 * m_peak, right):5.2f}%")
    print(f"  [b] 山なし（1/d²）・飽和なし        自由度1  残差 {rms_pct(g2 * m_flat, right):5.2f}%")

    def best_saturating(model):
        best = (np.inf, None)
        for ceiling in np.arange(3400.0, 20000.0, 50.0):
            for gain in np.arange(ceiling * 0.2, ceiling * 60.0, ceiling * 0.05):
                r = rms_pct(ceiling * (1.0 - np.exp(-gain * model / ceiling)), right)
                if r < best[0]:
                    best = (r, (ceiling, gain))
        return best

    r3, (c3, _) = best_saturating(m_flat)
    print(f"  [c] 山なし（1/d²）＋軟らかい飽和    自由度2  残差 {r3:5.2f}%（天井 {c3:.0f}）")

    best_sep = (np.inf, None)
    for sep in np.arange(0.0050, 0.0105, 0.0005):
        m = model_curve(d_mm, float(sep))
        g = float(np.sum(m * right) / np.sum(m ** 2))
        r = rms_pct(g * m, right)
        if r < best_sep[0]:
            best_sep = (r, float(sep))
    print(f"  [d] 山あり・離隔も自由・飽和なし    自由度2  残差 {best_sep[0]:5.2f}%"
          f"（離隔 {best_sep[1] * 1000:.1f}mm）")

    i80 = int(np.argmin(np.abs(d_mm - 80.0)))
    i50 = int(np.argmin(np.abs(d_mm - 50.0)))
    print(f"\n  80→50mm での伸び: 実測 {right[i50] / right[i80]:.2f}倍 / "
          f"山あり {m_peak[i50] / m_peak[i80]:.2f}倍 / 山なし(1/d²) {m_flat[i50] / m_flat[i80]:.2f}倍")
    print("  → 山ありは自由度 1 で、飽和は自由度 2 で説明する。**決着はしていない**。")
    print("     両者は 44mm より近い側で正反対を予測する（山あり: 下がる / 飽和: 上がり続ける）")


def main():
    print("=" * 74)
    print("非単調性の機構（近距離で照射円が PT の視野に入らなくなる）")
    print("=" * 74)
    negative_control()
    mechanism_estimate()
    compare_explanations()


if __name__ == "__main__":
    main()
