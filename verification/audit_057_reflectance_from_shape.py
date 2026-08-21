"""
verification/audit_057_reflectance_from_shape.py
================
事前登録: `verification/AUDIT_057_PREREG_reflectance_from_shape.md`

実測の距離依存の「形」で、壁の拡散反射率 ρ を縛れるかを調べる。
方法・判定量・分割点・ρ の走査範囲（0.2〜1.0 を 0.05 刻み）は事前登録のものを一切変えない。

## 反射回数ごとの分解

反射 k 回の経路は必ず ρ の k 乗を持つので、応答は次のように分解できる（`audit_054` と同じ道具立て）:

    S(d, ρ) = specular_offset(d) + Σ_{k=1}^{4} ρ^k A_k(d)

- `A_1`（拡散のみ・反射 1 回）は `mouse/ir_sensor.py::response()`（`n_grid=56`）で面積分。
  `specular_offset`（鏡面成分）は同じ `response()` で拡散 0・鏡面 0.10・とがり 40（現行モデルの
  既定値）として別に 1 回だけ計算する。**鏡面成分は ρ（拡散反射率）でスケールしない**
  （note_034 追記14: 鏡面成分は角度の実測から別途較正済みの固定量であり、本監査が
  縛ろうとしている「拡散反射率」とは別の物理量であるため。事前登録は A_1 の計算法しか
  指定しておらず、鏡面の扱いは本スクリプトの実装判断。報告に明記する）。
- `A_2..A_4`（反射 2〜4 回の増分）は光線追跡（`audit_050_raycast.py::raycast_response`、
  反射率 1.0 で計算し、`max_bounces` の差を取る。鏡面はそもそもこの光線追跡に実装が無く、
  2 回目以降は拡散のみという `mouse/ir_sensor.py` の相互反射モデルの前提と同じ）。

`A_k`・`specular_offset` は「距離のずれ込みの実効距離」ごとに 1 回だけ計算し、
`outputs/audit_057/coeff_table_right_80_200.json` に保存する。利得・距離のずれ・ρ の
最適化はこのテーブルの参照だけで行い、光線追跡・面積分を再計算しない。

使い方:
  .venv/bin/python verification/audit_057_reflectance_from_shape.py
"""
from __future__ import annotations

import json
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
from verification.audit_050_raycast import Sensor, raycast_response
from verification.audit_051_measured_vs_model import FIT_RANGE_RIGHT_MM, load_measurement

OUT_DIR = REPO_ROOT / "outputs" / "audit_057"

# ----------------------------------------------------------------------------
# 事前登録の走査範囲・判定量・分割点（§2・§3。変更しない）
# ----------------------------------------------------------------------------
RHOS = np.round(np.arange(0.20, 1.001, 0.05), 2)
Q_LOW, Q_HIGH = 1.2, 2.0
NOISE_SIGMA_REL = 0.0062           # 事前登録 §3: 実測の残差と同じ大きさ（0.62%）
N_NOISE_REPEATS = 10                # 事前登録 §5: 1 回では判断できない
RHO_TRUE_CASES = (0.8, 0.3)
RHO_HAT_TOL = 0.1                   # 事前登録 §3: ρ̂ が ρ_true ± 0.1 に入るか

# audit_051 と同じ当てはめ手順（この手順を変えない）
OFFSETS_MM = np.arange(-30.0, 30.5, 0.5)

# 現行モデルの既定値（2026-08-21 更新分。IrSensorSpec/SurfaceSpec のデータクラス既定と同じ）
SPECULAR = 0.10
SHININESS = 40.0
N_GRID = 56                          # AUDIT_056 工程1' の指摘どおり、当てはめの雑音を下げる側に倒す

N_RAYS = 300_000                     # 光線追跡（A_2..A_4）の本数
RAY_SEED = 777001


# ============================================================================
# センサ仕様
# ============================================================================
def front_sensor():
    """`mouse/params.py` の前方センサ LF の pos/axis（機体座標）。"""
    p = RobotParams()
    s = next(x for x in p.sensors if x["name"] == "LF")
    pos = tuple(float(v) for v in s["pos"].split())
    axis = tuple(float(v) for v in s["zaxis"].split())
    return pos, axis


def wall_rect(pos_x_m: float, dist_mm: float) -> Rect:
    """センサから光軸方向に `dist_mm` 離れた正対する壁（`audit_051.model_response` と同じ配置）。"""
    x_face = pos_x_m + dist_mm / 1000.0
    return Rect(cx=x_face + 0.006, cy=0.0, hx=0.006, hy=0.30)


# ============================================================================
# 係数テーブル（A_1・specular_offset・A_2..A_4）を実効距離ごとに 1 回だけ計算
# ============================================================================
def compute_coeffs_at(dist_mm: float, ir_spec: IrSensorSpec, ray_sensor: Sensor):
    """1 つの実効距離について (A1, A2, A3, A4, specular_offset) を返す。"""
    rect = wall_rect(ir_spec.pos[0], dist_mm)

    # A_1: 拡散のみ・反射率1.0（あとで ρ^1 を掛ける）。鏡面はここに含めない。
    surf_diffuse = SurfaceSpec(diffuse=1.0, specular=0.0, shininess=SHININESS)
    a1 = response(ir_spec, (0.0, 0.0, 0.0), [rect], surf_diffuse,
                  include_floor=True, occlusion=True, n_grid=N_GRID, bounces=1)

    # 鏡面成分: 現行モデルの既定値（鏡面0.10・とがり40）で固定。ρ でスケールしない。
    surf_specular = SurfaceSpec(diffuse=0.0, specular=SPECULAR, shininess=SHININESS)
    spec_off = response(ir_spec, (0.0, 0.0, 0.0), [rect], surf_specular,
                         include_floor=True, occlusion=True, n_grid=N_GRID, bounces=1)

    # A_2..A_4: 光線追跡（反射率1.0）。max_bounces の差分。
    vals = [raycast_response(
        ray_sensor, (0.0, 0.0, 0.0), [rect], n_rays=N_RAYS, seed=RAY_SEED,
        max_bounces=k, include_floor=True,
        led_half_angle_deg=ir_spec.led_half_angle_deg, pt_half_angle_deg=ir_spec.pt_half_angle_deg,
        max_range_m=0.35, wall_height_m=0.05, separation_m=ir_spec.separation_m, diffuse=1.0,
    ) for k in (1, 2, 3, 4)]
    a2 = vals[1] - vals[0]
    a3 = vals[2] - vals[1]
    a4 = vals[3] - vals[2]
    return float(a1), float(a2), float(a3), float(a4), float(spec_off)


def build_table(d_mm: np.ndarray, offsets_mm: np.ndarray, ir_spec, ray_sensor):
    """`d_mm + offsets_mm` の全組み合わせに現れる実効距離を重複なく列挙し、1 回だけ計算する。

    `d_mm` は 5mm 刻み・`offsets_mm` は 0.5mm 刻みなので、和は必ず 0.5mm グリッド上に乗る
    （重複が大量にあるため、ユニーク値だけ計算すれば十分。事前登録の制約
    「光線追跡は距離ごとに1回だけ計算して保存する」を効率よく満たす）。
    """
    xs = sorted({round(float(d + o), 1) for d in d_mm for o in offsets_mm})
    table = {}
    t0 = time.time()
    for i, x in enumerate(xs):
        table[x] = compute_coeffs_at(x, ir_spec, ray_sensor)
        if (i + 1) % 60 == 0:
            print(f"    係数テーブル {i + 1}/{len(xs)}（{time.time() - t0:.1f}s 経過）")
    print(f"  係数テーブル構築完了: {len(xs)} 点、{time.time() - t0:.1f}s")
    return table


def save_table(table: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "columns": ["A1_diffuse", "A2", "A3", "A4", "specular_offset"],
        "n_grid": N_GRID, "n_rays": N_RAYS, "seed": RAY_SEED,
        "specular": SPECULAR, "shininess": SHININESS,
        "table": {str(k): list(v) for k, v in sorted(table.items())},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def coeff_grid(d_mm: np.ndarray, offsets_mm: np.ndarray, table: dict) -> np.ndarray:
    """`table` から (n_offsets, n_dist, 5) の配列を作る（[a1,a2,a3,a4,spec_off] の順）。"""
    n_d, n_o = len(d_mm), len(offsets_mm)
    grid = np.empty((n_o, n_d, 5), dtype=float)
    for j, off in enumerate(offsets_mm):
        for i, d in enumerate(d_mm):
            grid[j, i, :] = table[round(float(d + off), 1)]
    return grid


# ============================================================================
# 当てはめ（利得・距離のずれを ρ ごとに最適化し直す。audit_051 と同じ手順）
# ============================================================================
def fit_gain_and_offset_for_rho(ad: np.ndarray, grid: np.ndarray, rho: float, offsets_mm: np.ndarray):
    """`grid`（(n_offsets, n_dist, 5)）から S(d+off, ρ) を作り、利得・距離のずれを当てはめる。

    返り値: (best_offset, best_gain, best_rms)。`audit_051.fit_gain_and_offset` と同じ規約
    （残差 RMS は相対値）。オフセット走査をベクトル化しているだけで、探索の手順自体は同じ。
    """
    a1, a2, a3, a4, spec_off = grid[:, :, 0], grid[:, :, 1], grid[:, :, 2], grid[:, :, 3], grid[:, :, 4]
    mv = spec_off + rho * a1 + rho ** 2 * a2 + rho ** 3 * a3 + rho ** 4 * a4  # (n_offsets, n_dist)
    denom = np.sum(mv * mv, axis=1)
    gain = np.sum(mv * ad[None, :], axis=1) / denom
    rms = np.sqrt(np.mean(((gain[:, None] * mv - ad[None, :]) / ad[None, :]) ** 2, axis=1))
    j_best = int(np.argmin(rms))
    return float(offsets_mm[j_best]), float(gain[j_best]), float(rms[j_best])


def fit_gain_and_offset_a1_only(ad: np.ndarray, grid: np.ndarray, offsets_mm: np.ndarray):
    """ρ に依存しない「空振り」対照用: モデルを A_1（拡散のみ・反射1回）だけにして当てはめる。"""
    a1 = grid[:, :, 0]
    denom = np.sum(a1 * a1, axis=1)
    gain = np.sum(a1 * ad[None, :], axis=1) / denom
    rms = np.sqrt(np.mean(((gain[:, None] * a1 - ad[None, :]) / ad[None, :]) ** 2, axis=1))
    j_best = int(np.argmin(rms))
    return float(offsets_mm[j_best]), float(gain[j_best]), float(rms[j_best])


def scan_rho(ad: np.ndarray, grid: np.ndarray, offsets_mm: np.ndarray, rhos: np.ndarray = RHOS):
    rows = []
    for rho in rhos:
        off, gain, rms = fit_gain_and_offset_for_rho(ad, grid, float(rho), offsets_mm)
        rows.append((float(rho), off, gain, rms))
    return rows


def q_of(rows) -> float:
    rms = np.array([r[3] for r in rows])
    return float(rms.max() / rms.min())


def rho_hat_of(rows) -> float:
    rho = np.array([r[0] for r in rows])
    rms = np.array([r[3] for r in rows])
    return float(rho[int(np.argmin(rms))])


def within_11_range(rows):
    rho = np.array([r[0] for r in rows])
    rms = np.array([r[3] for r in rows])
    keep = rho[rms <= rms.min() * 1.1]
    return float(keep.min()), float(keep.max())


# ============================================================================
# 否定対照（事前登録 §3）
# ============================================================================
def negative_control_rho_true(ad_real: np.ndarray, grid: np.ndarray, offsets_mm: np.ndarray,
                               rho_true: float, n_repeats: int = N_NOISE_REPEATS, base_seed: int = 20260822):
    """ρ_true で合成した「実測」に、実測の残差と同じ大きさの正規雑音を乗せ、
    ρ̂ が ρ_true ± 0.1 に入るかを見る（乱数種を変えて n_repeats 回）。
    """
    off_fit, gain_fit, rms_fit = fit_gain_and_offset_for_rho(ad_real, grid, rho_true, offsets_mm)
    a1, a2, a3, a4, spec_off = grid[:, :, 0], grid[:, :, 1], grid[:, :, 2], grid[:, :, 3], grid[:, :, 4]
    mv_all = spec_off + rho_true * a1 + rho_true ** 2 * a2 + rho_true ** 3 * a3 + rho_true ** 4 * a4
    j_fit = int(np.argmin(np.abs(offsets_mm - off_fit)))
    clean = gain_fit * mv_all[j_fit]  # (n_dist,)

    rho_hats = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(base_seed + rep)
        noisy = clean * (1.0 + rng.normal(0.0, NOISE_SIGMA_REL, size=clean.shape))
        rows = scan_rho(noisy, grid, offsets_mm)
        rho_hats.append(rho_hat_of(rows))
    return off_fit, gain_fit, rms_fit, rho_hats


def negative_control_null(ad_real: np.ndarray, grid: np.ndarray, offsets_mm: np.ndarray,
                           n_repeats: int = N_NOISE_REPEATS, base_seed: int = 90000):
    """空振り側: ρ に依存しない合成データ（A_1 だけ）で Q ≤ 1.2 になるかを見る。"""
    off0, gain0, rms0 = fit_gain_and_offset_a1_only(ad_real, grid, offsets_mm)
    a1 = grid[:, :, 0]
    j_fit = int(np.argmin(np.abs(offsets_mm - off0)))
    clean = gain0 * a1[j_fit]  # (n_dist,)

    qs = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(base_seed + rep)
        noisy = clean * (1.0 + rng.normal(0.0, NOISE_SIGMA_REL, size=clean.shape))
        rows = scan_rho(noisy, grid, offsets_mm)
        qs.append(q_of(rows))
    return off0, gain0, rms0, qs


# ============================================================================
# メイン
# ============================================================================
def main():
    t_start = time.time()

    d_mm_all, left, right = load_measurement()
    lo, hi = FIT_RANGE_RIGHT_MM
    mask = (d_mm_all >= lo) & (d_mm_all <= hi)
    d_mm = d_mm_all[mask]
    ad = right[mask]

    pos, axis = front_sensor()
    ir_spec = IrSensorSpec(name="LF", pos=pos, axis=axis)  # 既定値（LED3.0°/PT6.0°/離隔6.0mm）
    ray_sensor = Sensor(name="LF", pos=pos, axis=axis)

    print("=" * 78)
    print("監査057: 実測の形で壁の拡散反射率 ρ を縛れるか")
    print("=" * 78)
    print(f"当てはめ帯 {lo:.0f}-{hi:.0f}mm（右センサ）: {len(d_mm)} 点")
    print(f"IrSensorSpec 既定値: LED半値角{ir_spec.led_half_angle_deg}° "
          f"PT半値角{ir_spec.pt_half_angle_deg}° 離隔{ir_spec.separation_m * 1000:.1f}mm")
    print(f"鏡面（固定・ρでスケールしない）: specular={SPECULAR} shininess={SHININESS}")
    print(f"ρ 走査範囲: {RHOS[0]:.2f}〜{RHOS[-1]:.2f}（{len(RHOS)} 点） / N_RAYS={N_RAYS} / N_GRID={N_GRID}")

    # --- 係数テーブル構築 ---
    t0 = time.time()
    table = build_table(d_mm, OFFSETS_MM, ir_spec, ray_sensor)
    t_table = time.time() - t0
    print(f"[所要時間] 係数テーブル構築: {t_table:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_table(table, OUT_DIR / "coeff_table_right_80_200.json")

    t0 = time.time()
    grid = coeff_grid(d_mm, OFFSETS_MM, table)
    print(f"[所要時間] グリッド化: {time.time() - t0:.1f}s")

    # --- 否定対照（事前登録 §3。実測を見る前に走らせる） ---
    print("\n" + "=" * 78)
    print("否定対照（事前登録 §3）")
    print("=" * 78)
    t0 = time.time()

    control_results = {}
    for rho_true in RHO_TRUE_CASES:
        off_fit, gain_fit, rms_fit, rho_hats = negative_control_rho_true(ad, grid, OFFSETS_MM, rho_true)
        arr = np.array(rho_hats)
        control_results[rho_true] = (off_fit, gain_fit, rms_fit, rho_hats)
        print(f"\nρ_true={rho_true}")
        print(f"  実測への当てはめ値: offset={off_fit:+.1f}mm gain={gain_fit:.1f} 残差RMS={rms_fit * 100:.2f}%")
        print(f"  ρ̂ 10回: {[round(v, 3) for v in rho_hats]}")
        print(f"  ρ̂ 中央値={np.median(arr):.3f}  範囲=[{arr.min():.3f}, {arr.max():.3f}]")
        print(f"  ρ_true±{RHO_HAT_TOL} = [{rho_true - RHO_HAT_TOL:.2f}, {rho_true + RHO_HAT_TOL:.2f}]")

    off0, gain0, rms0, qs = negative_control_null(ad, grid, OFFSETS_MM)
    qs_arr = np.array(qs)
    print(f"\n空振り（ρ非依存合成データ・A1のみ）")
    print(f"  実測への当てはめ値: offset={off0:+.1f}mm gain={gain0:.1f} 残差RMS={rms0 * 100:.2f}%")
    print(f"  Q 10回: {[round(v, 3) for v in qs]}")
    print(f"  Q 中央値={np.median(qs_arr):.3f}  範囲=[{qs_arr.min():.3f}, {qs_arr.max():.3f}]")

    t_negctrl = time.time() - t0
    print(f"\n[所要時間] 否定対照: {t_negctrl:.1f}s")

    # --- 事前登録どおりの合否判定（数値のみ。判断・推奨は書かない） ---
    passed = []
    for rho_true in RHO_TRUE_CASES:
        arr = np.array(control_results[rho_true][3])
        ok = bool((arr.min() >= rho_true - RHO_HAT_TOL) and (arr.max() <= rho_true + RHO_HAT_TOL))
        passed.append(ok)
        print(f"[否定対照] ρ_true={rho_true}: 10回すべてが ρ_true±{RHO_HAT_TOL} に入るか = {ok}")
    null_ok = bool(qs_arr.max() <= 1.2)
    print(f"[否定対照] 空振り: 10回すべて Q≤1.2 か = {null_ok}")

    if not (all(passed) and null_ok):
        print("\n" + "!" * 78)
        print("否定対照が通らなかった。事前登録 §3 により、実測の判定（§2）は行わずここで止める。")
        print("!" * 78)
        print(f"\n[総所要時間] {time.time() - t_start:.1f}s")
        return

    # --- 実測での走査（事前登録 §2。対照が通った場合のみ） ---
    print("\n" + "=" * 78)
    print("実測での走査（事前登録 §2）")
    print("=" * 78)
    t0 = time.time()
    rows = scan_rho(ad, grid, OFFSETS_MM)
    t_real = time.time() - t0

    print("   ρ    offset[mm]     gain     残差RMS[%]")
    for rho, off, gain, rms in rows:
        print(f"  {rho:4.2f}    {off:8.1f}   {gain:9.1f}    {rms * 100:7.3f}")

    q = q_of(rows)
    rho_hat = rho_hat_of(rows)
    lo11, hi11 = within_11_range(rows)
    print(f"\nQ = R_max/R_min = {q:.3f}")
    print(f"分割: Q≤{Q_LOW}→区別しない / {Q_LOW}<Q≤{Q_HIGH}→弱いが向きはある / Q>{Q_HIGH}→縛る")
    print(f"ρ̂ = argmin R(ρ) = {rho_hat:.2f}")
    print(f"R_min の 1.1 倍以内に入る ρ の範囲: [{lo11:.2f}, {hi11:.2f}]")
    print(f"[所要時間] 実測走査: {t_real:.1f}s")

    print(f"\n[総所要時間] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
