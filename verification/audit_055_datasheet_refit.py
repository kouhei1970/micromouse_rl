"""
verification/audit_055_datasheet_refit.py
================
半値角をデータシートの値（LED = SFH 4550 ±3°／PT = ST-1KL3A ±6°。note_034 追記9）に
**固定**した上で、残りのパラメータ（離隔・鏡面成分・とがり・角度のずれ）を
2 つの実測データ（`mouse/data/ir_measured_front_20260821.csv` と `assets/angle_vs_sensor.csv`）
へ当てはめ直し、残差の表を作る。

新しい既定値をいくつにするかは判断しない（教授セッションが決める）。本スクリプトは
測定だけを行い、`mouse/ir_sensor.py` の既定値は一切変更しない
（`IrSensorSpec`/`SurfaceSpec` の値はすべて引数で渡す）。

構成:
  (a) 距離の実測への離隔の当てはめ  … `verification/audit_051_measured_vs_model.py` の
      当てはめ方（利得と距離のずれを自由パラメータに取り、形だけを比べる）を踏襲。
  (b) 角度の実測への鏡面成分の当てはめ … `verification/audit_052_angle_vs_model.py` の
      当てはめ方（センサごと・距離ごとに角度 0 で規格化）を踏襲。
  (c) 交差検証 … (b) の鏡面成分を入れて (a) の残差がどう変わるか、
      (a) の離隔を入れて (b) の残差がどう変わるかを見る。

使い方:
  .venv/bin/python verification/audit_055_datasheet_refit.py
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import Rect
from mouse.ir_sensor import IrSensorSpec, SurfaceSpec, response
from mouse.params import RobotParams

# データシートで確定した半値角（note_034 追記9）。
LED_HALF_DEG = 3.0     # OSRAM SFH 4550
PT_HALF_DEG = 6.0      # KODENSHI ST-1KL3A（typ）
# 旧既定値（比較用。IrSensorSpec のこれまでの既定はこれだった）。
OLD_LED_HALF_DEG = 5.0
OLD_PT_HALF_DEG = 5.0
# IrSensorSpec.separation_m の旧既定値（audit_052 が明示的に渡していなかったため
# 暗黙に使っていた値。(b) の主要な当てはめでもこれをそのまま使う）。
DEFAULT_SEPARATION_M = 0.0065


# ============================================================================
# (a) 距離の実測 ir_measured_front_20260821.csv
# ============================================================================
DIST_CSV = REPO_ROOT / "mouse" / "data" / "ir_measured_front_20260821.csv"
FIT_RANGE_RIGHT_MM = (80.0, 200.0)
FIT_RANGE_LEFT_MM = (120.0, 200.0)
FULL_RANGE_MM = (50.0, 200.0)
# 離隔の走査格子（0.25mm 刻み。指示の 5.0/5.5/6.0/6.5/7.0mm を含む）。
SEPARATIONS_MM = np.arange(5.0, 7.0001, 0.25)
OFFSETS_MM = np.arange(-30.0, 30.5, 0.5)   # audit_051 と同じ格子


def load_dist_measurement():
    rows = []
    with open(DIST_CSV, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("range_mm"):
                continue
            a, b, c = line.split(",")
            rows.append((float(a), float(b), float(c)))
    rows.sort()
    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def front_sensor_spec(separation_m, led_half_deg=LED_HALF_DEG, pt_half_deg=PT_HALF_DEG):
    """`mouse/params.py` の前方センサ LF を IrSensorSpec にする（audit_051 と同じ）。"""
    p = RobotParams()
    s = next(x for x in p.sensors if x["name"] == "LF")
    pos = tuple(float(v) for v in s["pos"].split())
    axis = tuple(float(v) for v in s["zaxis"].split())
    return IrSensorSpec(
        name="LF", pos=pos, axis=axis, separation_m=separation_m,
        led_half_angle_deg=led_half_deg, pt_half_angle_deg=pt_half_deg,
    )


def model_response_dist(dist_mm, spec, surf=None, *, include_floor=True):
    surf = surf if surf is not None else SurfaceSpec()
    x_face = spec.pos[0] + dist_mm / 1000.0
    rect = Rect(cx=x_face + 0.006, cy=0.0, hx=0.006, hy=0.30)
    return response(spec, (0.0, 0.0, 0.0), [rect], surf,
                    include_floor=include_floor, occlusion=True)


def fit_gain_and_offset(d_mm, ad, spec, fit_lo, fit_hi, surf=None,
                         offsets_mm=OFFSETS_MM):
    """利得と距離のずれを当てはめ、(ずれ, 利得, 残差RMS[比]) を返す（audit_051 と同じ）。"""
    mask = (d_mm >= fit_lo) & (d_mm <= fit_hi)
    best = (None, None, np.inf)
    for off in offsets_mm:
        mv = np.array([model_response_dist(x + off, spec, surf) for x in d_mm[mask]])
        gain = float(np.sum(mv * ad[mask]) / np.sum(mv * mv))
        rms = float(np.sqrt(np.mean(((gain * mv - ad[mask]) / ad[mask]) ** 2)))
        if rms < best[2]:
            best = (float(off), gain, rms)
    return best


def rms_over_range(d_mm, ad, spec, lo, hi, off, gain, surf=None):
    mask = (d_mm >= lo) & (d_mm <= hi)
    mv = np.array([model_response_dist(x + off, spec, surf) for x in d_mm[mask]])
    return float(np.sqrt(np.mean(((gain * mv - ad[mask]) / ad[mask]) ** 2))) * 100.0


def peak_position_mm(spec, surf=None, lo=15.0, hi=90.0, step=0.5):
    xs = np.arange(lo, hi + step / 2, step)
    vals = [model_response_dist(x, spec, surf) for x in xs]
    i = int(np.argmax(vals))
    return float(xs[i])


def part_a(d_mm, left, right):
    """半値角 3°/6° 固定で離隔を振り、右・左センサそれぞれの残差表を作る。"""
    print("=" * 78)
    print("(a) 距離の実測 ir_measured_front_20260821.csv への離隔の当てはめ")
    print(f"    半値角固定: LED {LED_HALF_DEG}° / PT {PT_HALF_DEG}°（データシート）")
    print("=" * 78)

    results = {}
    for name, ad, (lo, hi) in (("右", right, FIT_RANGE_RIGHT_MM),
                               ("左", left, FIT_RANGE_LEFT_MM)):
        print(f"\n[{name}センサ] 当てはめ帯 {lo:.0f}-{hi:.0f}mm（遠方帯）／全域 {FULL_RANGE_MM}")
        print("  離隔[mm]  ずれ[mm]  遠方RMS[%]  全域RMS[%]  山の位置[mm]")
        rows = []
        for sep_mm in SEPARATIONS_MM:
            spec = front_sensor_spec(sep_mm / 1000.0)
            off, gain, rms_far = fit_gain_and_offset(d_mm, ad, spec, lo, hi)
            rms_full = rms_over_range(d_mm, ad, spec, *FULL_RANGE_MM, off, gain)
            peak = peak_position_mm(spec)
            rows.append((sep_mm, off, rms_far * 100.0, rms_full, peak, gain))
            print(f"   {sep_mm:6.2f}   {off:+6.1f}    {rms_far*100:8.2f}   {rms_full:8.2f}   {peak:8.1f}")
        best = min(rows, key=lambda r: r[2])
        print(f"  → 遠方RMS 最小: 離隔 {best[0]:.2f}mm（{best[2]:.2f}%）")
        results[name] = {"rows": rows, "best_sep_mm": best[0], "best_off_mm": best[1],
                         "best_gain": best[5]}
    return results


# ============================================================================
# (b) 角度の実測 assets/angle_vs_sensor.csv
# ============================================================================
ANGLE_CSV = REPO_ROOT / "assets" / "angle_vs_sensor.csv"
SENSOR_NAMES = ("LF", "LS", "RS", "RF")
WALL_FACE_X = 0.5
WALL = [Rect(cx=WALL_FACE_X + 0.006, cy=0.0, hx=0.006, hy=0.6)]
MOUNT_HEIGHT_M = 0.010

SPECULARS = tuple(round(x, 2) for x in np.arange(0.0, 0.2001, 0.02))
SHININESSES = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 65.0, 80.0)
ANGLE_OFFSETS = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def load_angle_measurement():
    table = {}
    with open(ANGLE_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (int(row["range"]), int(row["angle"]))
            table[key] = {n: int(row[n]) for n in SENSOR_NAMES}
    ranges = sorted({k[0] for k in table}, reverse=True)
    angles = sorted({k[1] for k in table})
    return table, ranges, angles


def axis_fixed_distance(range_m, angle_deg, surf, *, separation_m=DEFAULT_SEPARATION_M,
                         led_half_deg=LED_HALF_DEG, pt_half_deg=PT_HALF_DEG):
    t = math.radians(angle_deg)
    pos = (WALL_FACE_X - range_m * math.cos(t), -range_m * math.sin(t), MOUNT_HEIGHT_M)
    axis = (math.cos(t), math.sin(t), 0.0)
    spec = IrSensorSpec(name="probe", pos=pos, axis=axis, separation_m=separation_m,
                         led_half_angle_deg=led_half_deg, pt_half_angle_deg=pt_half_deg)
    return response(spec, (0.0, 0.0, 0.0), WALL, surf, include_floor=True, occlusion=True)


def model_profile(angles, surf, angle_offset_deg=0.0, range_m=0.180, **kw):
    base = axis_fixed_distance(range_m, angle_offset_deg, surf, **kw)
    return np.array([axis_fixed_distance(range_m, a + angle_offset_deg, surf, **kw) / base
                     for a in angles])


def fit_surface(angles, meas, speculars=SPECULARS, shininesses=SHININESSES,
                offsets=ANGLE_OFFSETS, **kw):
    best = None
    for sv in speculars:
        for sh in shininesses:
            for off in offsets:
                surf = SurfaceSpec(diffuse=0.8, specular=sv, shininess=sh)
                vals = model_profile(angles, surf, off, **kw)
                rms = float(np.sqrt(np.mean((vals - meas) ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, sv, sh, off, vals)
    return best


def measured_profile_overall(table, ranges, angles):
    out = []
    for a in angles:
        vals = [table[(rg, a)][n] / table[(rg, 0)][n]
                for rg in ranges for n in SENSOR_NAMES if (rg, a) in table]
        out.append(float(np.mean(vals)))
    return np.array(out)


def measured_profile_by_range(table, ranges, angles, rg):
    return np.array([np.mean([table[(rg, a)][n] / table[(rg, 0)][n] for n in SENSOR_NAMES])
                     for a in angles])


def measured_profile_by_sensor(table, ranges, angles, n):
    return np.array([np.mean([table[(rg, a)][n] / table[(rg, 0)][n]
                              for rg in ranges if (rg, a) in table]) for a in angles])


def part_b(table, ranges, angles):
    print("\n" + "=" * 78)
    print("(b) 角度の実測 assets/angle_vs_sensor.csv への鏡面成分の当てはめ")
    print(f"    半値角固定: LED {LED_HALF_DEG}° / PT {PT_HALF_DEG}°（データシート）")
    print(f"    離隔は (b) では既定 {DEFAULT_SEPARATION_M*1000:.2f}mm のまま固定")
    print("=" * 78)

    meas_overall = measured_profile_overall(table, ranges, angles)

    print("\n[鏡面 0（現行の既定）のときの残差 — 比較の基準]")
    plain = model_profile(angles, SurfaceSpec(diffuse=0.8, specular=0.0))
    rms_plain = float(np.sqrt(np.mean((plain - meas_overall) ** 2)))
    print(f"  残差RMS {rms_plain:.4f}")

    print("\n[全体（4距離×4本平均）での最良の組]")
    r0, sv0, sh0, off0, _ = fit_surface(angles, meas_overall)
    print(f"  鏡面 {sv0:.2f} / とがり {sh0:.0f} / 角度のずれ {off0:+.0f}°  残差RMS {r0:.4f}")

    print("\n[距離ごとの最良の組]")
    by_range = {}
    for rg in ranges:
        m = measured_profile_by_range(table, ranges, angles, rg)
        r, sv, sh, off, _ = fit_surface(angles, m)
        by_range[rg] = (r, sv, sh, off)
        print(f"  range={rg:3d}mm → 鏡面 {sv:.2f} / とがり {sh:.0f} / ずれ {off:+.0f}°  残差RMS {r:.4f}")

    print("\n[センサごとの最良の組]")
    by_sensor = {}
    for n in SENSOR_NAMES:
        m = measured_profile_by_sensor(table, ranges, angles, n)
        r, sv, sh, off, _ = fit_surface(angles, m)
        by_sensor[n] = (r, sv, sh, off)
        print(f"  {n} → 鏡面 {sv:.2f} / とがり {sh:.0f} / ずれ {off:+.0f}°  残差RMS {r:.4f}")

    print(f"\n[参考: 旧半値角 {OLD_LED_HALF_DEG}°/{OLD_PT_HALF_DEG}° での最良の組（追記7 の再現確認）]")
    r_old, sv_old, sh_old, off_old, _ = fit_surface(
        angles, meas_overall, led_half_deg=OLD_LED_HALF_DEG, pt_half_deg=OLD_PT_HALF_DEG)
    print(f"  鏡面 {sv_old:.2f} / とがり {sh_old:.0f} / ずれ {off_old:+.0f}°  残差RMS {r_old:.4f}"
         f"  （追記7: 0.12 / 50 / +2° / 0.0070）")

    return {
        "meas_overall": meas_overall,
        "plain_rms": rms_plain,
        "overall_best": (r0, sv0, sh0, off0),
        "by_range": by_range,
        "by_sensor": by_sensor,
        "old_half_angle_best": (r_old, sv_old, sh_old, off_old),
    }


# ============================================================================
# (c) 交差検証
# ============================================================================
def part_c(d_mm, left, right, a_results, b_results, angles, meas_overall):
    print("\n" + "=" * 78)
    print("(c) 交差検証（半値角 3°/6° 固定のもとで）")
    print("=" * 78)

    r0, sv0, sh0, off0 = b_results["overall_best"]

    print(f"\n[c1] (b) の鏡面成分（{sv0:.2f} / とがり{sh0:.0f}）を入れた状態で (a) の残差がどう変わるか")
    print("  センサ  離隔[mm]  鏡面0の遠方RMS[%]  鏡面込みの遠方RMS[%]  鏡面込みの全域RMS[%]  鏡面込みのずれ[mm]")
    c1_rows = {}
    for name, ad, (lo, hi) in (("右", right, FIT_RANGE_RIGHT_MM),
                               ("左", left, FIT_RANGE_LEFT_MM)):
        sep_mm = a_results[name]["best_sep_mm"]
        plain_rms = next(r[2] for r in a_results[name]["rows"] if abs(r[0] - sep_mm) < 1e-9)
        spec = front_sensor_spec(sep_mm / 1000.0)
        surf = SurfaceSpec(diffuse=0.8, specular=sv0, shininess=sh0)
        off, gain, rms_far = fit_gain_and_offset(d_mm, ad, spec, lo, hi, surf=surf)
        rms_full = rms_over_range(d_mm, ad, spec, *FULL_RANGE_MM, off, gain, surf=surf)
        c1_rows[name] = (sep_mm, plain_rms, rms_far * 100.0, rms_full, off)
        print(f"   {name}    {sep_mm:6.2f}    {plain_rms:10.2f}         {rms_far*100:10.2f}"
             f"           {rms_full:10.2f}          {off:+6.1f}")

    print("\n[c2] (a) で選ばれた離隔を入れた状態で (b) の残差がどう変わるか（全体の当てはめをやり直す）")
    print("  センサ由来の離隔  離隔[mm]  鏡面0の残差  最良の組（鏡面/とがり/ずれ）  最良の残差")
    c2_rows = {}
    for name in ("右", "左"):
        sep_m = a_results[name]["best_sep_mm"] / 1000.0
        plain = model_profile(angles, SurfaceSpec(diffuse=0.8, specular=0.0), separation_m=sep_m)
        rms_plain_b = float(np.sqrt(np.mean((plain - meas_overall) ** 2)))
        r, sv, sh, off, _ = fit_surface(angles, meas_overall, separation_m=sep_m)
        c2_rows[name] = (a_results[name]["best_sep_mm"], rms_plain_b, sv, sh, off, r)
        print(f"   {name}センサ由来      {a_results[name]['best_sep_mm']:6.2f}   {rms_plain_b:9.4f}"
             f"    {sv:.2f} / {sh:.0f} / {off:+.0f}°           {r:.4f}")
    print(f"  （参考: (b) の既定離隔 {DEFAULT_SEPARATION_M*1000:.2f}mm での鏡面0残差"
         f" {b_results['plain_rms']:.4f} ／ 最良残差 {b_results['overall_best'][0]:.4f}）")

    return c1_rows, c2_rows


def main():
    t_start = time.time()

    d_mm, left, right = load_dist_measurement()
    t0 = time.time()
    a_results = part_a(d_mm, left, right)
    print(f"\n[経過時間] (a) 完了: {time.time() - t0:.1f}秒")

    t0 = time.time()
    table, ranges, angles = load_angle_measurement()
    b_results = part_b(table, ranges, angles)
    print(f"\n[経過時間] (b) 完了: {time.time() - t0:.1f}秒")

    t0 = time.time()
    part_c(d_mm, left, right, a_results, b_results, angles, b_results["meas_overall"])
    print(f"\n[経過時間] (c) 完了: {time.time() - t0:.1f}秒")

    print(f"\n[総経過時間] {time.time() - t_start:.1f}秒")


if __name__ == "__main__":
    main()
