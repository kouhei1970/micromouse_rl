"""
verification/audit_051_measured_vs_model.py
================
実機の距離センサの実測（`mouse/data/ir_measured_front_20260821.csv`）と、
物理モデル（`mouse/ir_sensor.py::response()`）を突き合わせる。

これは `research_notes/note_034_ir_sensor_model.md` の問い C（定式化が実機と合うか）に
あたる。AUDIT_050（光線追跡との突き合わせ）は問い A（実装が定式化どおりか）であり、
別のものである。

測定の条件は「正面に壁を置き、ロボットの距離をずらして AD 変換の値を記録した」もの。
入射角は 0°、横ずれも 0 とみなす。したがって本スクリプトが確かめられるのは
**正対したときの距離依存の形だけ**である（入射角・横ずれ・遮蔽・多重反射は確かめられない）。

当てはめ方:
  実機の利得（AD 値／モデルの無次元値）は未知なので自由パラメータに取る。
  距離の基準点も未確認なので、距離のずれも自由パラメータに取る。
  **形だけを比べる**（利得と基準点を合わせたうえで、残差が残るかを見る）。

使い方:
  .venv/bin/python verification/audit_051_measured_vs_model.py
"""
from __future__ import annotations

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

# 飽和とみられる領域を当てはめから外す（左右で外す範囲が違う。note_034 追記6）。
FIT_RANGE_RIGHT_MM = (80.0, 200.0)
FIT_RANGE_LEFT_MM = (120.0, 200.0)


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


def front_sensor_spec(separation_m=0.0065, half_angle_deg=5.0):
    """`mouse/params.py` の前方センサ LF を IrSensorSpec にする。"""
    p = RobotParams()
    s = next(x for x in p.sensors if x["name"] == "LF")
    pos = tuple(float(v) for v in s["pos"].split())
    axis = tuple(float(v) for v in s["zaxis"].split())
    return IrSensorSpec(
        name="LF", pos=pos, axis=axis, separation_m=separation_m,
        led_half_angle_deg=half_angle_deg, pt_half_angle_deg=half_angle_deg,
    )


def model_response(dist_mm, spec, *, include_floor=True):
    """センサから光軸方向に `dist_mm` 離れた正対する壁に対する応答。

    壁は迷路規格の厚み 12mm、幅は十分広く取る（半値角 5° の照射円は 200mm でも
    直径 35mm 程度なので、幅 600mm あれば端の影響は出ない）。
    """
    surf = SurfaceSpec()
    x_face = spec.pos[0] + dist_mm / 1000.0
    rect = Rect(cx=x_face + 0.006, cy=0.0, hx=0.006, hy=0.30)
    return response(spec, (0.0, 0.0, 0.0), [rect], surf,
                    include_floor=include_floor, occlusion=True)


def fit_gain_and_offset(d_mm, ad, spec, fit_lo, fit_hi,
                        offsets_mm=np.arange(-30.0, 30.5, 0.5)):
    """利得と距離のずれを当てはめ、(ずれ, 利得, 残差RMS[比]) を返す。"""
    mask = (d_mm >= fit_lo) & (d_mm <= fit_hi)
    best = (None, None, np.inf)
    for off in offsets_mm:
        mv = np.array([model_response(x + off, spec) for x in d_mm[mask]])
        gain = float(np.sum(mv * ad[mask]) / np.sum(mv * mv))
        rms = float(np.sqrt(np.mean(((gain * mv - ad[mask]) / ad[mask]) ** 2)))
        if rms < best[2]:
            best = (float(off), gain, rms)
    return best


def scan_parameters(d_mm, ad, fit_lo, fit_hi):
    """離隔と半値角を振って、形の残差がどこで最小になるかを見る（利得は毎回最適化）。

    実測の形が、ユーザから提供された仕様値（離隔 6.5mm・半値角 5°）を選ぶかどうかを
    確かめるための走査。距離のずれは 0 に固定する（自由度を増やすと縮退するため）。
    """
    mask = (d_mm >= fit_lo) & (d_mm <= fit_hi)

    def rms_for(separation_m, half_angle_deg):
        spec = front_sensor_spec(separation_m, half_angle_deg)
        mv = np.array([model_response(x, spec) for x in d_mm[mask]])
        gain = float(np.sum(mv * ad[mask]) / np.sum(mv * mv))
        return float(np.sqrt(np.mean(((gain * mv - ad[mask]) / ad[mask]) ** 2))) * 100.0

    print("  離隔を振る（半値角 5.0° 固定）:")
    for sep in (0.004, 0.005, 0.006, 0.0065, 0.007, 0.008, 0.010):
        print(f"    離隔 {sep * 1000:4.1f}mm → 残差RMS {rms_for(sep, 5.0):6.2f}%")
    print("  半値角を振る（離隔 6.5mm 固定）:")
    for ha in (3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0):
        print(f"    半値角 {ha:4.1f}° → 残差RMS {rms_for(0.0065, ha):6.2f}%")


def main():
    d_mm, left, right = load_measurement()
    spec = front_sensor_spec()

    print("=" * 72)
    print("問い C: 定式化が実機と合うか（正対したときの距離依存の形だけ）")
    print("=" * 72)

    for name, ad, (lo, hi) in (("右", right, FIT_RANGE_RIGHT_MM),
                               ("左", left, FIT_RANGE_LEFT_MM)):
        off, gain, rms = fit_gain_and_offset(d_mm, ad, spec, lo, hi)
        print(f"\n[{name}センサ] 当てはめ帯 {lo:.0f}-{hi:.0f}mm")
        print(f"  距離のずれ {off:+.1f}mm / 利得 {gain:.1f} AD・単位^-1 / 残差RMS {rms * 100:.2f}%")
        mv = np.array([model_response(x + off, spec) for x in d_mm])
        print("   距離   実測   モデル×利得     比")
        for i in range(0, len(d_mm), 3):
            print(f"  {d_mm[i]:5.0f} {ad[i]:6.0f} {gain * mv[i]:12.0f} {gain * mv[i] / ad[i]:8.3f}")
        peak = max(model_response(x, spec) for x in np.arange(20.0, 90.0, 1.0))
        print(f"  → 応答のピークでの AD 予測 {gain * peak:.0f}（12bit 満量 4095）")

    print("\n" + "=" * 72)
    print("実測の形は、どのパラメータを選ぶか（右センサ・80-200mm・距離のずれ 0 固定）")
    print("=" * 72)
    scan_parameters(d_mm, right, *FIT_RANGE_RIGHT_MM)

    print("\n" + "=" * 72)
    print("未検証の帯（50mm 未満）でモデルが予測する AD 値。右センサの利得で換算")
    print("=" * 72)
    _, gain_r, _ = fit_gain_and_offset(d_mm, right, spec, *FIT_RANGE_RIGHT_MM)
    print("  距離   離隔6.0mm  離隔6.5mm  離隔7.0mm")
    for x in (50, 45, 40, 35, 30, 25, 20, 15, 10, 5):
        vals = [gain_r * model_response(float(x), front_sensor_spec(sep, 5.0))
                for sep in (0.006, 0.0065, 0.007)]
        print(f"  {x:4d}  {vals[0]:9.0f} {vals[1]:10.0f} {vals[2]:10.0f}")


if __name__ == "__main__":
    main()
