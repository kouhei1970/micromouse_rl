"""tests/test_ir_sensor.py — `mouse/ir_sensor.py`（IR LED＋PT 距離センサの放射モデル）の検査

背景・仕様は `research_notes/note_034_ir_sensor_model.md` を正とする。ここでの検査方針:

- 数値は本ファイルを実行して得たものだけを書く（合わせ込みはしない）。ピークの距離・
  「同じ応答を与える 2 つの距離」は検査自身が実測して `print()` する（`pytest -s` で見える）。
- 峰の位置は壁の有無・床の有無・格子の細かさで動きうる（note_034 参照）ため、
  「41mm」のような教授セッションの見積もりに合わせ込まず、ここで実測した値
  （既定パラメータ・n_grid=28 で 44mm）を基準として固定する。

共通のセンサ配置: `pos=(0, 0, 0.010)`・`axis=(1, 0, 0)`（機体原点に取付高さ10mm・
光軸+x）。この配置は機体座標の原点にあるため、機体姿勢 `(x, y, theta)` がそのまま
「センサ基準点の world 位置 (x, y) ・世界座標での光軸方位 theta」になる
（回転オフセットを考えなくてよい）。壁は world x=0 の面（法線 -x 側）に置き、
`distance` はセンサ基準点から壁面までの垂直距離、`incidence` はセンサの向き
（＝壁法線からの光軸のずれ）とする。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from classic.geometry import Rect
from mouse.ir_sensor import (
    IrSensorSpec,
    ResponseTable,
    SurfaceSpec,
    adc,
    build_table_from_model,
    load_table,
    lookup,
    response,
    save_table,
)

# ============================================================================
# 共通のセットアップ
# ============================================================================
SENSOR = IrSensorSpec(name="T", pos=(0.0, 0.0, 0.010), axis=(1.0, 0.0, 0.0))
SURF = SurfaceSpec()
WALL_HY = 0.09  # 壁の半長 [m]（片側90mm。180mm区画の壁1枚ぶんに相当する代表値）
WALL = Rect(cx=0.006, cy=0.0, hx=0.006, hy=WALL_HY)


def _flat_wall_response(distance_m: float, incidence_deg: float = 0.0,
                         lateral_m: float = 0.0, sensor: IrSensorSpec = SENSOR,
                         surf: SurfaceSpec = SURF, **kw) -> float:
    """`SENSOR` を正対する壁（world x=0）から `distance_m` 離した位置に置いたときの応答。

    `sensor.pos` が機体原点 (0,0,z) にあるため、姿勢 `(x, y, theta)` は
    そのまま「センサ位置 (x, y=lateral)・光軸の世界方位 theta=incidence」になる
    （`docstring` 冒頭参照）。
    """
    theta = math.radians(incidence_deg)
    pose = (-distance_m, lateral_m, theta)
    return response(sensor, pose, [WALL], surf, **kw)


# ============================================================================
# 1. 距離に対して応答は単調でなく山なりになる
# ============================================================================
def test_response_is_not_monotonic_in_distance():
    """壁に正対したときの応答を 20〜70mm で 1mm 刻みに走らせ、山なりであることを確認する。

    実測（本テストの `print` 出力、既定パラメータ・n_grid=28）: ピークは 44mm、
    値は 0.8337（任意単位）。5mm ではピークの 0.45% 未満。
    """
    distances_mm = np.arange(20, 71, 1)
    values = np.array([_flat_wall_response(d / 1000.0) for d in distances_mm])

    peak_i = int(np.argmax(values))
    peak_mm = int(distances_mm[peak_i])
    peak_v = float(values[peak_i])
    print(f"[response-vs-distance] peak = {peak_mm}mm, value = {peak_v:.6f}")
    for d, v in zip(distances_mm, values):
        print(f"  {d:3d}mm  {v:.6f}")

    # 実測して固定した基準値（本ファイル docstring 参照）。壁・床の有無や格子の細かさで
    # 動きうる値なので、ここは実測した 44mm に対する許容幅として扱う。
    EXPECTED_PEAK_MM = 44
    assert abs(peak_mm - EXPECTED_PEAK_MM) <= 2, (
        f"ピーク位置が実測の基準値から動いた: {peak_mm}mm (基準 {EXPECTED_PEAK_MM}mm)"
    )

    # 山なり: ピークの手前は単調増加、ピークの奥は単調減少（数値誤差ぶんだけ緩める）。
    rising = values[: peak_i + 1]
    falling = values[peak_i:]
    assert np.all(np.diff(rising) >= -1e-9), "ピーク手前が単調増加になっていない"
    assert np.all(np.diff(falling) <= 1e-9), "ピーク奥が単調減少になっていない"
    # ピークの手前と奥の両方が実在する（=「山」であって「棚」ではない）ことも確認する。
    assert peak_i > 0 and peak_i < len(values) - 1

    # 5mm はピークの 1% 未満（近づきすぎると値が下がる、という現象そのものの確認）。
    v5 = _flat_wall_response(0.005)
    ratio = v5 / peak_v
    print(f"[response-vs-distance] 5mm value = {v5:.6f}, ratio to peak = {ratio:.4%}")
    assert ratio < 0.01


# ============================================================================
# 2. 同じ応答を与える距離が 2 つある（強度だけでは距離が決まらない）
# ============================================================================
def test_same_value_at_two_distances():
    """ピークの奥側で 1 点 (`far_d`) を選び、ピークの手前側で同じ応答を与える点を
    二分法で実際に探す。強度だけでは「近いのか遠いのか」が決まらないことの直接の検査。
    """
    far_d = 0.090
    v_far = _flat_wall_response(far_d)

    lo, hi = 0.020, 0.044  # ピーク(44mm)手前の単調増加区間
    v_lo, v_hi = _flat_wall_response(lo), _flat_wall_response(hi)
    assert v_lo < v_far < v_hi, "二分法の前提（区間の両端で target を挟む）が崩れている"

    for _ in range(40):
        mid = (lo + hi) / 2.0
        v_mid = _flat_wall_response(mid)
        if v_mid < v_far:
            lo = mid
        else:
            hi = mid
    near_d = (lo + hi) / 2.0
    v_near = _flat_wall_response(near_d)

    print(f"[same-value] near = {near_d * 1000:.3f}mm (value={v_near:.6f}), "
          f"far = {far_d * 1000:.1f}mm (value={v_far:.6f})")

    rel_diff = abs(v_near - v_far) / v_far
    assert rel_diff < 1e-6, f"二分法で探した2点の応答が一致しない（相対差 {rel_diff:.2e}）"
    # 二分法が探し当てたのが本当に別の距離であること（縮退して同じ点に収束していない）。
    assert far_d - near_d > 0.03


# ============================================================================
# 3. 壁に斜めに当てると応答が下がる
# ============================================================================
def test_incidence_angle_reduces_response():
    """壁までの距離を固定し（ピークより奥の 60mm）、入射角を振ると応答が単調に下がる。"""
    d = 0.060
    angles_deg = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    values = [_flat_wall_response(d, incidence_deg=a) for a in angles_deg]

    print(f"[incidence] distance = {d * 1000:.0f}mm")
    for a, v in zip(angles_deg, values):
        print(f"  {a:4.0f}deg  {v:.6f}")

    for v_prev, v_next in zip(values, values[1:]):
        assert v_next < v_prev, "入射角を増やしても応答が下がらない箇所がある"


# ============================================================================
# 4. 縦配置と横配置で応答が異なる条件
# ============================================================================
def test_vertical_and_horizontal_differ_somewhere():
    """縦配置・横配置で応答が異なる条件を実際に探す。

    教授セッションの比較（無限平面の壁に正対）は光軸まわりに回転対称になり、構造的に
    差を出せないものだった（note_034）。ここでは有限の壁＋床を使い、note_034 が挙げた
    「差が出るとしたら次の条件」のうち 1（近距離・応答のピークより内側）と
    5（床の反射。縦配置では PT が床に近い）を実際に試す。

    実測（本テストの `print` 出力）: d=10mm（ピーク44mmの内側の近距離）で、
    縦配置 3.787e-3・横配置 4.217e-3（相対差 約11.4%）。n_grid を 20〜120 まで振っても
    比はほぼ一定（0.898）なので、格子の粗さによる数値誤差ではなく構造的な差である。
    """
    sensor_v = IrSensorSpec(name="V", pos=(0.0, 0.0, 0.010), axis=(1.0, 0.0, 0.0),
                             layout="vertical")
    sensor_h = IrSensorSpec(name="H", pos=(0.0, 0.0, 0.010), axis=(1.0, 0.0, 0.0),
                             layout="horizontal")

    d_near = 0.010  # ピーク(44mm)より内側の近距離
    v_near = _flat_wall_response(d_near, sensor=sensor_v)
    h_near = _flat_wall_response(d_near, sensor=sensor_h)
    rel_diff_near = abs(h_near - v_near) / v_near
    print(f"[layout] d={d_near * 1000:.0f}mm (床あり)  vertical={v_near:.6e}  "
          f"horizontal={h_near:.6e}  相対差={rel_diff_near:.2%}")

    # 参考: 床を含めない場合・ピーク付近（壁の寄与が支配的）の場合は差がずっと小さいことも
    # 併せて印字する（「差が無い」のではなく「この条件でだけ測れる」ことの傍証）。
    v_near_nofloor = _flat_wall_response(d_near, sensor=sensor_v, include_floor=False)
    h_near_nofloor = _flat_wall_response(d_near, sensor=sensor_h, include_floor=False)
    v_peak = _flat_wall_response(0.044, sensor=sensor_v)
    h_peak = _flat_wall_response(0.044, sensor=sensor_h)
    print(f"[layout] d={d_near * 1000:.0f}mm (床なし)  vertical={v_near_nofloor:.6e}  "
          f"horizontal={h_near_nofloor:.6e}")
    print(f"[layout] d=44mm (ピーク付近・床あり)  vertical={v_peak:.6e}  "
          f"horizontal={h_peak:.6e}  相対差={abs(h_peak - v_peak) / v_peak:.2%}")

    assert rel_diff_near > 0.05, (
        "近距離・床ありの条件で縦配置と横配置に有意な差が見つからなかった"
    )


# ============================================================================
# 5. 表の作成・保存・読み込みの往復
# ============================================================================
def test_table_roundtrip(tmp_path: Path):
    """`build_table_from_model` → `save_table` → `load_table` が一致し、
    `lookup` の補間が端で破綻しない（例外もクランプ外れの異常値も出ない）ことを確認する。
    """
    distances_m = np.array([0.02, 0.03, 0.044, 0.06, 0.09])
    incidence_deg = np.array([0.0, 15.0, 30.0])
    lateral_m = np.array([0.0, 0.02])

    table = build_table_from_model(
        SENSOR, SURF,
        distances_m=distances_m, incidence_deg=incidence_deg, lateral_m=lateral_m,
        n_grid=16,
    )
    assert table.source == "model"
    assert table.values.shape == (len(distances_m), len(incidence_deg), len(lateral_m))

    path = tmp_path / "ir_table.npz"
    save_table(table, path)
    loaded = load_table(path)

    assert np.array_equal(table.distances_m, loaded.distances_m)
    assert np.array_equal(table.incidence_deg, loaded.incidence_deg)
    assert np.array_equal(table.lateral_m, loaded.lateral_m)
    assert np.array_equal(table.values, loaded.values)
    assert loaded.source == "model"
    assert loaded.meta["sensor_name"] == "T"
    assert loaded.meta["n_grid"] == 16

    # 格子点そのものを引くと、保存前の値と一致する。
    v_node = lookup(loaded, 0.044, 15.0, 0.02)
    v_exact = float(table.values[2, 1, 1])
    print(f"[table] lookup at grid node: {v_node:.6f} (exact {v_exact:.6f})")
    assert v_node == pytest.approx(v_exact, abs=1e-12)

    # 範囲外はクランプする（外挿で暴れた値やエラーにならない）。
    v_below = lookup(loaded, -0.05, 0.0, 0.0)
    v_edge_low = lookup(loaded, distances_m[0], incidence_deg[0], lateral_m[0])
    v_above = lookup(loaded, 5.0, 100.0, 5.0)
    v_edge_high = lookup(loaded, distances_m[-1], incidence_deg[-1], lateral_m[-1])
    print(f"[table] below-range clamp: {v_below:.6f} == edge {v_edge_low:.6f}")
    print(f"[table] above-range clamp: {v_above:.6f} == edge {v_edge_high:.6f}")
    assert v_below == pytest.approx(v_edge_low, abs=1e-12)
    assert v_above == pytest.approx(v_edge_high, abs=1e-12)
    assert np.isfinite(v_below) and np.isfinite(v_above)

    # 格子点の中間は両隣の値の間に入る（線形補間の健全性）。
    v_mid = lookup(loaded, 0.025, 0.0, 0.0)
    lo, hi = sorted([float(table.values[0, 0, 0]), float(table.values[1, 0, 0])])
    assert lo <= v_mid <= hi


# ============================================================================
# 6. 鏡面成分の否定対照（specular=0 で不変・specular>0 で変わる）
# ============================================================================
def test_specular_component_changes_response():
    """`specular` を 0→0.5 にすると応答が変わり、0 のままなら一致する（対で確認する）。

    実測: d=44mm 正対で specular=0 → 0.8337、specular=0.5 → 2.1700（+160%）。
    """
    surf_a = SurfaceSpec(specular=0.0)
    surf_b = SurfaceSpec(specular=0.0)  # 別インスタンスの「0のまま」（否定対照）
    surf_c = SurfaceSpec(specular=0.5)  # 「0.5にした」（陽性対照）

    d = 0.044
    v_a = _flat_wall_response(d, surf=surf_a)
    v_b = _flat_wall_response(d, surf=surf_b)
    v_c = _flat_wall_response(d, surf=surf_c)
    print(f"[specular] d={d * 1000:.0f}mm  specular=0: {v_a:.6f} / {v_b:.6f} "
          f"(一致={v_a == v_b})  specular=0.5: {v_c:.6f} "
          f"(変化 {100 * (v_c - v_a) / v_a:+.1f}%)")

    assert v_a == v_b, "specular=0 のままなのに値が変わった（再現性が壊れている）"
    assert v_c != v_a
    assert abs(v_c - v_a) / v_a > 0.10, "specular=0.5 にしても応答がほとんど変わらなかった"


# ============================================================================
# 付随: adc() の分解能・飽和・雑音（この項目は note_034 に明示のテスト仕様は無いが、
# response() の出力を実際に使う経路として簡単に確認しておく）
# ============================================================================
def test_adc_resolution_and_saturation():
    assert adc(0.0, bits=12, full_scale=1.0) == 0
    assert adc(1.0, bits=12, full_scale=1.0) == 4095
    assert adc(2.0, bits=12, full_scale=1.0) == 4095  # 飽和（クランプ）
    assert adc(-1.0, bits=12, full_scale=1.0) == 0    # 負側もクランプ
    assert adc(0.5, bits=12, full_scale=1.0) in (2047, 2048)  # 中間値付近

    rng = np.random.default_rng(0)
    codes = [adc(0.5, bits=12, full_scale=1.0, noise_sigma=0.2, rng=rng) for _ in range(200)]
    assert len(set(codes)) > 1, "雑音を与えても常に同じコードしか出ない"
    assert all(0 <= c <= 4095 for c in codes)
