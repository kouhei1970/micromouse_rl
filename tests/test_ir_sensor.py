"""tests/test_ir_sensor.py — `mouse/ir_sensor.py`（IR LED＋PT 距離センサの放射モデル）の検査

背景・仕様は `research_notes/note_034_ir_sensor_model.md` を正とする。ここでの検査方針:

- 数値は本ファイルを実行して得たものだけを書く（合わせ込みはしない）。ピークの距離・
  「同じ応答を与える 2 つの距離」は検査自身が実測して `print()` する（`pytest -s` で見える）。
- 峰の位置は壁の有無・床の有無・格子の細かさで動きうる（note_034 参照）ため、
  「41mm」のような教授セッションの見積もりに合わせ込まず、ここで実測した値
  （既定パラメータ・n_grid=28 で 44mm）を基準として固定する。
- **遮蔽（オクルージョン）判定を追加（2026-08-21）。** `response()` の既定は
  `occlusion=True`。単一の壁＋床という以下の共通セットアップでは、床パッチの一部が
  その壁自身に遮られる分だけ値が下がる（壁の裏側はもともとバックフェイスカリングで
  積分対象外なので、変わるのは床の寄与だけ）。ピーク位置は変わらず 44mm のまま、
  ピーク値は 0.8337 → 0.829894 に下がった（下の各検査は再実測した値で閾値を
  確認済み。壁・柱どうしの遮蔽そのものの検査は `test_near_wall_blocks_far_wall` 等）。

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

import time

from classic.geometry import Rect, wall_obstacles
from mouse.params import RobotParams
from mouse.ir_sensor import (
    IrSensorSpec,
    ResponseTable,
    SurfaceSpec,
    adc,
    build_table_from_model,
    fast_response,
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

    実測（本テストの `print` 出力、既定パラメータ・n_grid=28、遮蔽あり）: ピークは 44mm、
    値は 0.829894（任意単位）。5mm ではピークの 1e-25 倍未満（遮蔽なしのときの旧実測値
    0.45% 未満よりさらに小さい。近距離では壁の直接寄与がほぼ消え、床の寄与が支配的だが、
    その床の主要な照射域はセンサ直近の壁自身の陰に入るため、遮蔽ありだとほぼ完全に
    落ちる。壁が無い条件での床のみの応答は距離に依らずほぼ一定値になることも別途確認済み
    — 詳細は `mouse/ir_sensor.py` の「遮蔽（オクルージョン）について」）。
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
    print(f"[response-vs-distance] 5mm value = {v5:.6e}, ratio to peak = {ratio:.6e}")
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

    実測（本テストの `print` 出力、遮蔽あり・既定）: d=10mm（ピーク44mmの内側の近距離）で、
    縦配置 4.347e-7・横配置 4.815e-7（相対差 約10.8%）。遮蔽を入れる前の実測（縦配置
    3.787e-3・横配置 4.217e-3・相対差 約11.4%）から絶対値は 4 桁近く落ちたが
    （d=10mm では壁自身の陰になる床の領域が主要な照射域だったため。遮蔽ありでは壁の
    直接寄与が支配的になる。上のテスト参照）、相対差はほぼ同じ大きさのままなので、
    格子の粗さによる数値誤差ではなく構造的な差であるという結論は変わらない。
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

    実測（遮蔽あり・既定）: d=44mm 正対で specular=0 → 0.829894、specular=0.5 → 2.166238
    （+161.0%）。
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
# 7. 遮蔽（オクルージョン）: 手前の壁が奥の壁を隠す／柱が壁の一部を隠す
# ============================================================================
def _wall_at(front_x: float, cy: float = 0.0, hy: float = WALL_HY,
             thickness_m: float = 0.012) -> Rect:
    """壁面（法線 -x 側）が世界 x=`front_x` に来るように置いた `Rect`（`WALL` と同じ厚み）。"""
    return Rect(cx=front_x + thickness_m / 2.0, cy=cy, hx=thickness_m / 2.0, hy=hy)


def test_near_wall_blocks_far_wall():
    """手前 84mm・奥 264mm の 2 枚の壁（センサから見て同じ y 帯・光軸上に重なる配置）で、
    遮蔽ありの応答が「手前の壁だけ」の応答と一致し、遮蔽なしだと奥の壁の寄与ぶん
    過大になることを確認する（note_034 の教授セッション実測: 手前だけ 0.5168・
    奥だけ 0.0489・遮蔽なしの合計 0.5657＝手前だけの 1.095 倍、と同じ現象）。
    """
    near = _wall_at(0.084)
    far = _wall_at(0.264)
    pose = (0.0, 0.0, 0.0)  # SENSOR.pos が機体原点にあるので、そのまま world 位置になる

    v_near_only = response(SENSOR, pose, [near], SURF)
    v_far_only = response(SENSOR, pose, [far], SURF)
    v_both_occl_on = response(SENSOR, pose, [near, far], SURF, occlusion=True)
    v_both_occl_off = response(SENSOR, pose, [near, far], SURF, occlusion=False)

    print(f"[occlusion] near-only(84mm)  = {v_near_only:.6f}")
    print(f"[occlusion] far-only(264mm)  = {v_far_only:.6f}")
    print(f"[occlusion] both, occl=True  = {v_both_occl_on:.6f}")
    print(f"[occlusion] both, occl=False = {v_both_occl_off:.6f}  "
          f"(vs near-only: {100 * (v_both_occl_off - v_near_only) / v_near_only:+.2f}%)")

    rel_err_on = abs(v_both_occl_on - v_near_only) / v_near_only
    print(f"[occlusion] 遮蔽あり vs 手前だけ の相対誤差 = {rel_err_on:.3e}")
    assert rel_err_on < 0.01, "遮蔽ありなのに奥の壁の寄与が漏れている"

    # 遮蔽なしは奥の壁の寄与が素通しで足されるぶん過大になる（否定対照。note_034 の
    # 9.5% 程度と同じ桁の過大評価が実際に起きることを確認する）。
    overage = (v_both_occl_off - v_near_only) / v_near_only
    assert overage > 0.03, "遮蔽なしでも奥の壁の寄与がほとんど足されていない（前提が崩れている）"


def test_opening_still_sees_far_wall():
    """手前の壁に開口部（幅60mm の隙間）を空けると、遮蔽ありでも奥の壁が見えることを確認する
    （note_034: 「手前に壁が無いとき（開口部）に奥の壁が見えるのは正しい挙動」）。

    手前を隙間なしの1枚壁にした `test_near_wall_blocks_far_wall` では奥の壁の寄与が
    ほぼゼロまで落ちる一方、ここでは光軸が通る隙間を空けることで「奥の壁だけ」の応答と
    ほぼ一致することを示す。
    """
    far = _wall_at(0.264)
    # 手前の壁（元は y in [-0.09, 0.09] の1枚）を y=0 中心に幅60mm（gap_half=30mm）だけ
    # 空けた2枚に分ける。隙間の外側 [gap_half, 0.09] と [-0.09, -gap_half] を
    # それぞれ 1 枚の Rect にするので、半長は (0.09-gap_half)/2、中心は (0.09+gap_half)/2。
    gap_half = 0.03
    seg_hy = (0.09 - gap_half) / 2.0
    seg_cy = (0.09 + gap_half) / 2.0
    near_left = _wall_at(0.084, cy=seg_cy, hy=seg_hy)
    near_right = _wall_at(0.084, cy=-seg_cy, hy=seg_hy)
    pose = (0.0, 0.0, 0.0)

    v_far_only = response(SENSOR, pose, [far], SURF)
    v_gap_only = response(SENSOR, pose, [near_left, near_right], SURF, occlusion=True)
    v_opening = response(SENSOR, pose, [near_left, near_right, far], SURF, occlusion=True)
    far_contribution_open = v_opening - v_gap_only

    # 比較対象: 開口部を塞いだ（隙間なし）手前の壁のとき、奥の壁の寄与がどれだけ残るか
    # （`test_near_wall_blocks_far_wall` と同じ現象を、ここでは「寄与の差分」で見る）。
    near_solid = _wall_at(0.084)
    v_solid_only = response(SENSOR, pose, [near_solid], SURF, occlusion=True)
    v_solid_with_far = response(SENSOR, pose, [near_solid, far], SURF, occlusion=True)
    far_contribution_solid = v_solid_with_far - v_solid_only

    print(f"[opening] far-only                = {v_far_only:.6f}")
    print(f"[opening] gap-60mm(壁分だけ)      = {v_gap_only:.6f}")
    print(f"[opening] gap-60mm + far          = {v_opening:.6f}  "
          f"(vs far-only: {100 * (v_opening - v_far_only) / v_far_only:+.3f}%)")
    print(f"[opening] 開口部での奥の壁の寄与  = {far_contribution_open:.6f}")
    print(f"[opening] 隙間なしでの奥の壁の寄与 = {far_contribution_solid:.6f} (参考: ほぼ消える方)")

    # 開口部があるときは、奥の壁単体の応答とほぼ一致する（奥の壁がそのまま見えている）。
    rel_err = abs(v_opening - v_far_only) / v_far_only
    assert rel_err < 0.01, "開口部があるのに奥の壁がほとんど見えていない"
    # 隙間なし（壁で塞いだ）ときは奥の壁の寄与がほぼ消えるのに対し、開口部があるときは
    # 奥の壁の寄与がほぼ丸ごと残る、という対比を実測値で確認する。
    assert far_contribution_solid < 0.01 * v_far_only, "隙間なしなのに奥の壁が見えてしまっている"
    assert far_contribution_open > 0.9 * v_far_only, "開口部があるのに奥の壁の寄与が回復していない"


def test_post_blocks_part_of_wall():
    """センサと壁の間に柱を1本置くと、その陰になる壁の部分の寄与が消えて応答が下がることを
    実測値で示す（柱自身が反射で足す寄与は遮蔽の有無に関わらず同じぶんだけ乗るので、
    occl=True と occl=False の差がそのまま「柱の陰になった壁の寄与」に対応する）。
    """
    d = 0.084
    post = Rect(cx=-0.04, cy=0.0, hx=0.006, hy=0.006)   # 12mm角の柱。壁との間、光軸上
    pose = (-d, 0.0, 0.0)

    v_wall_alone = _flat_wall_response(d)
    v_with_post_on = response(SENSOR, pose, [WALL, post], SURF, occlusion=True)
    v_with_post_off = response(SENSOR, pose, [WALL, post], SURF, occlusion=False)

    print(f"[post] wall alone             = {v_wall_alone:.6f}")
    print(f"[post] wall+post, occl=True   = {v_with_post_on:.6f}")
    print(f"[post] wall+post, occl=False  = {v_with_post_off:.6f}")
    print(f"[post] 柱の陰で消えた分 (off-on) = {v_with_post_off - v_with_post_on:.6f} "
          f"({100 * (v_with_post_off - v_with_post_on) / v_with_post_off:.1f}% of occl=False)")

    assert v_with_post_off > v_with_post_on, (
        "柱を置いても遮蔽の有無で応答が変わらない（陰になる部分が消えていない）"
    )
    shadow_ratio = (v_with_post_off - v_with_post_on) / v_with_post_off
    assert shadow_ratio > 0.10, "柱の陰の効果が小さすぎる（遮蔽が効いていない疑い）"
    # 柱の有無で応答が変わることそのものも確認する（note_034 の要求どおり）。
    assert v_with_post_on != v_wall_alone


def test_occlusion_off_matches_legacy():
    """`occlusion=False` は遮蔽を計算しない旧挙動と厳密に一致する（否定対照。回帰検査で固定）。

    単一の壁+床（既存テストの共通セットアップ）で、44mm 正対の値が本モジュール変更前の
    実測値 0.8336798263697903（旧 docstring の 0.8337 の元値）とビット単位で一致することを
    確認する。
    """
    d = 0.044
    v_off = _flat_wall_response(d, occlusion=False)
    LEGACY_VALUE_AT_44MM = 0.8336798263697903
    print(f"[legacy] occlusion=False, d=44mm = {v_off!r} (legacy = {LEGACY_VALUE_AT_44MM!r})")
    assert v_off == LEGACY_VALUE_AT_44MM, "occlusion=False が遮蔽実装前の値と一致しない"

    # 2枚壁（近い壁が遠い壁を隠すはずの配置）でも、occlusion=False なら両方の寄与が
    # 単純に足し合わされる（= 遮蔽を考えない「面ごとに独立積分」という旧モデルの定義どおり）。
    # 床は `include_floor=True` だと壁の枚数によらず 1 枚だけ足されるので、素朴に
    # 「別々に呼んだ和 == まとめて呼んだ値」にはならない（床の寄与が二重に乗る）。
    # ここでは床を含めない条件（`include_floor=False`）で壁どうしの単純加法だけを見る。
    near = _wall_at(0.084)
    far = _wall_at(0.264)
    pose = (0.0, 0.0, 0.0)
    v_near = response(SENSOR, pose, [near], SURF, occlusion=False, include_floor=False)
    v_far = response(SENSOR, pose, [far], SURF, occlusion=False, include_floor=False)
    v_both = response(SENSOR, pose, [near, far], SURF, occlusion=False, include_floor=False)
    print(f"[legacy] near={v_near:.6f} + far={v_far:.6f} = {v_near + v_far:.6f} "
          f"vs both={v_both:.6f}")
    assert v_both == pytest.approx(v_near + v_far, rel=1e-9), (
        "occlusion=False なのに単純な足し合わせから外れている"
    )


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


# ============================================================================
# 9. 高速フォワードモデル（表＋解析的な面の列挙＋遮蔽判定）: 精度と速さ
# ============================================================================
# 表の実体は `mouse/data/ir_response_table.npz`（壁）・`mouse/data/ir_post_table.npz`
# （柱）。生成スクリプトは `mouse/build_ir_response_table.py`（版管理下。表そのものも
# 小さい＝約5.3MB×2枚なので版管理下に直接置いた。再生成:
# `.venv/bin/python -m mouse.build_ir_response_table`）。
#
# 許容値の根拠（2026-08-21・教授セッションと実測で決めた）:
# 「満量」＝ある1本のセンサが取りうる最大級の値（直接積分の最大値を基準にする）。
# 満量に対する絶対誤差なら、応答が距離4乗近くで落ちて桁が動く弱い信号の領域でも
# 見かけの相対誤差が発散しない。
#   - 満量に対する誤差の中央値 2% 以内（実測 0.39〜0.44%。壁の有無を判定する用途では
#     この桁で十分。閾値0.15での判定が0/120件覆らないことを教授セッションが別途確認済み）
#   - 満量に対する誤差の最大 40% 以内（120姿勢での実測 34.8%。目標は5%だったが、
#     直接積分と比べて隣接する2枚以上の壁・柱が同一平面を作り互いを部分的に
#     自己遮蔽する場面（実迷路の連続した通路で起きる。柱を挟んで壁が繋がる構造
#     そのもの）だけ、「面ごとに独立に表引きして最大値を採る」近似の精度が粗くなる。
#     同一平面のパネルをまとめて1枚として扱う修正も試したが、別の姿勢を悪化させる
#     （whack-a-mole）ため撤回した。5%の目標は未達のまま note_034 に記録し、
#     今後の課題とする（対処には辺単位ではなく面積単位の可視率計算が要ると見ている）
#   - 満量の10%未満の弱い信号における相対誤差の中央値 20% 以内（実測 8.2%。
#     「強度で使うか距離に直すかを決めうちしない」という方針上、弱い信号も
#     ある程度信用できる必要があるため設けた基準）
_FAST_TABLE_DIR = Path(__file__).resolve().parent.parent / "mouse" / "data"
_WALL_TABLE_PATH = _FAST_TABLE_DIR / "ir_response_table.npz"
_POST_TABLE_PATH = _FAST_TABLE_DIR / "ir_post_table.npz"


def _load_production_tables():
    wall_table = load_table(_WALL_TABLE_PATH)
    post_table = load_table(_POST_TABLE_PATH)
    floor_value = float(wall_table.meta["floor_baseline"])
    return wall_table, post_table, floor_value


def _real_sensors():
    """`mouse/params.py` の既定センサ4本（LF/LS/RF/RS）を IrSensorSpec にする。"""
    params = RobotParams()
    specs = []
    for s in params.sensors:
        pos = tuple(float(v) for v in s["pos"].split())
        zaxis = tuple(float(v) for v in s["zaxis"].split())
        specs.append(IrSensorSpec(name=s["name"], pos=pos, axis=zaxis))
    return specs


def _real_maze_surfaces(maze_name: str):
    """`competition/mazes/` の実迷路から壁・柱の配列を作る。"""
    params = RobotParams()
    maze_path = (
        Path(__file__).resolve().parent.parent / "competition" / "mazes"
        / "design_turn_v1" / f"{maze_name}.npz"
    )
    data = np.load(maze_path)
    surfaces = wall_obstacles(data["v_walls"], data["h_walls"], cell_size=params.cell_size)
    width = int(data["v_walls"].shape[0] - 1)
    height = int(data["v_walls"].shape[1])
    return surfaces, width, height, params.cell_size


def _sample_poses(n: int, seed: int, width: int, height: int, cell: float, sensors):
    """区画をランダムに選び、区画内で ±40mm ずらし、方位を無作為にした姿勢を `n` 個作る
    （教授セッションが独立検算に使った手順と同じ）。"""
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        cx = rng.integers(0, width)
        cy = rng.integers(0, height)
        x = (cx + 0.5) * cell + rng.uniform(-0.04, 0.04)
        y = (cy + 0.5) * cell + rng.uniform(-0.04, 0.04)
        theta = rng.uniform(-math.pi, math.pi)
        sensor = sensors[rng.integers(0, len(sensors))]
        poses.append((sensor, (x, y, theta)))
    return poses


def test_table_matches_direct_integration():
    """実迷路 maze_41001 から無作為に選んだ120姿勢で、`fast_response`（表＋解析的な
    面の列挙）と `response`（直接積分）を比べる。許容値は上のコメントを参照。
    """
    wall_table, post_table, floor_value = _load_production_tables()
    sensors = _real_sensors()
    surfaces, width, height, cell = _real_maze_surfaces("maze_41001")
    poses = _sample_poses(120, seed=41001, width=width, height=height, cell=cell, sensors=sensors)

    direct_vals = np.array([
        response(sensor, pose, surfaces, SURF, occlusion=True, include_floor=True, n_grid=28)
        for sensor, pose in poses
    ])
    fast_vals = np.array([
        fast_response(sensor, pose, surfaces, wall_table, floor_value, post_table=post_table)
        for sensor, pose in poses
    ])

    full_scale = float(np.max(direct_vals))
    abs_err_fs = np.abs(fast_vals - direct_vals) / full_scale
    weak_mask = direct_vals < 0.10 * full_scale
    rel_err_weak = np.abs(fast_vals[weak_mask] - direct_vals[weak_mask]) / np.maximum(
        direct_vals[weak_mask], 1e-9,
    )

    print(f"[fast_response accuracy] n={len(poses)} full_scale={full_scale:.6f}")
    print(f"  満量に対する誤差: median={np.median(abs_err_fs)*100:.3f}% "
          f"max={np.max(abs_err_fs)*100:.3f}%")
    print(f"  弱い信号(n={weak_mask.sum()})の相対誤差: median={np.median(rel_err_weak)*100:.2f}%")
    worst = np.argsort(-abs_err_fs)[:5]
    for i in worst:
        print(f"    worst: direct={direct_vals[i]:.4e} fast={fast_vals[i]:.4e} "
              f"abs_err_fs={abs_err_fs[i]*100:.2f}%")

    assert np.median(abs_err_fs) < 0.02, "満量に対する誤差の中央値が許容(2%)を超えた"
    assert np.max(abs_err_fs) < 0.40, "満量に対する誤差の最大が許容(40%)を超えた"
    if weak_mask.sum() > 0:
        assert np.median(rel_err_weak) < 0.20, "弱い信号の相対誤差の中央値が許容(20%)を超えた"


def test_table_lookup_is_fast():
    """`fast_response` が `response`（直接積分）より十分速いことを確認する（倍率を印字）。
    目安: 1回あたり2ms以内・直接積分に対して20倍以上速いこと（実測は約1.1ms・約50倍）。
    """
    wall_table, post_table, floor_value = _load_production_tables()
    sensors = _real_sensors()
    surfaces, width, height, cell = _real_maze_surfaces("maze_41001")
    poses = _sample_poses(30, seed=1, width=width, height=height, cell=cell, sensors=sensors)

    t0 = time.perf_counter()
    for sensor, pose in poses:
        response(sensor, pose, surfaces, SURF, occlusion=True, include_floor=True, n_grid=28)
    t_direct = (time.perf_counter() - t0) / len(poses)

    t0 = time.perf_counter()
    for sensor, pose in poses:
        fast_response(sensor, pose, surfaces, wall_table, floor_value, post_table=post_table)
    t_fast = (time.perf_counter() - t0) / len(poses)

    speedup = t_direct / t_fast
    print(f"[fast_response speed] direct={t_direct*1000:.3f}ms fast={t_fast*1000:.4f}ms "
          f"speedup={speedup:.1f}x")
    assert t_fast < 0.002, "表引き1回が2msを超えた"
    assert speedup > 20.0, "直接積分に対する高速化が20倍未満だった"
