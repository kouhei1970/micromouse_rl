"""verification/audit_060_table_sensor.py

`verification/AUDIT_060_PREREG_table_sensor.md`（事前登録）§3 自己検査1〜3 を実行する
（本測定・否定対照・`response_table()` は次の作業段階。ここでは「表を作る仕組み」の
正しさだけを確かめる）。

使い方（前景で実行。1 回で数分以内に終わる）:

    .venv/bin/python verification/audit_060_table_sensor.py

`mouse/data/ir_cumulative_table.npz`（`mouse/ir_table.py::build_cumulative_table()` で
あらかじめ作成済み）を読み込んで検査する。表そのものを作り直すには
`mouse/ir_table.py` を直接呼ぶこと（本ファイル末尾のコメント参照）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import Rect
from mouse import ir_table as T
from mouse.ir_sensor import DEFAULT_MAX_RANGE_M, DEFAULT_WALL_HEIGHT_M, SurfaceSpec, response

I_FULL = 0.8298934   # 満量（AUDIT_050 §2-2 以来の固定値。既存の全監査と同じ規約を踏襲）
TABLE_PATH = REPO_ROOT / "mouse" / "data" / "ir_cumulative_table.npz"
N_GRID_REF = 48       # 自己検査1・2 の基準（response() の細かい格子）
THIN_M = 1e-6         # 比較用の「厚みゼロ」壁の厚み（十分薄ければ端面の寄与は無視できる）

SEED = 20260822
SURF = SurfaceSpec()


def _thin_rect_for_u_range(y_cross: float, u_lo: float, u_hi: float) -> Rect:
    """`u` が `[u_lo, u_hi]` の区間に広がる、厚み `THIN_M` の壁（`-x` 面が x=0）を作る。
    `mouse/ir_table.py` の座標系（面は世界 x=0、外向き法線 -x）に合わせる。"""
    cy = y_cross + (u_lo + u_hi) / 2.0
    hy = (u_hi - u_lo) / 2.0
    return Rect(cx=THIN_M / 2.0, cy=cy, hx=THIN_M / 2.0, hy=hy)


def _reference_response(sensor, d_m: float, theta_deg: float, u_lo: float, u_hi: float) -> float:
    """`response()`（n_grid=48・厚みゼロの1枚壁・遮蔽なし・床なし）で、面が `[u_lo,u_hi]` に
    広がっているときの応答を直接計算する（表を一切使わない独立な基準）。"""
    pose = T._pose_for_dtheta(sensor, d_m, theta_deg)
    led, _pt = T._sensor_world_geometry(sensor, pose)
    y_cross = T._axis_plane_crossing_y(led)
    rect = _thin_rect_for_u_range(y_cross, u_lo, u_hi)
    return response(
        sensor, pose, [rect], SURF,
        wall_height_m=DEFAULT_WALL_HEIGHT_M, n_grid=N_GRID_REF, include_floor=False,
        max_range_m=DEFAULT_MAX_RANGE_M, occlusion=False,
    )


# ============================================================================
# 自己検査1: 表の格子点そのものの値 vs response()（n_grid=48・1枚の平面）
# ============================================================================
def selfcheck1(table: T.CumulativeTable, n_points: int = 40) -> dict:
    """`u = -120mm`（表に保存された最も広い区間。生成時の余白込みで `+220mm` まで積んで
    あるので、比較対象の response() も同じ `[-120mm, +220mm]` の壁で呼ぶ）での表の値を、
    独立な `response()` 呼び出しと比較する。"""
    rng = np.random.default_rng(SEED + 1)
    i_d = rng.integers(0, len(T.D_AXIS_M), size=n_points)
    i_th = rng.integers(0, len(T.THETA_AXIS_DEG), size=n_points)
    u_lo = float(T.U_AXIS_M[0])                      # -120mm（表の最小 u）
    u_hi = float(T.U_AXIS_M[-1]) + T.U_MARGIN_M       # +120mm + 余白100mm = +220mm

    diffs = []
    rows = []
    for k in range(n_points):
        d = float(T.D_AXIS_M[i_d[k]])
        th = float(T.THETA_AXIS_DEG[i_th[k]])
        table_val = float(table.values[i_d[k], i_th[k], 0])   # u_axis[0] = -120mm
        ref_val = _reference_response(table.meta_sensor, d, th, u_lo, u_hi)
        diff = abs(table_val - ref_val)
        diffs.append(diff)
        rows.append({"d_mm": d * 1000, "theta_deg": th, "table": table_val, "ref": ref_val, "diff": diff})

    diffs = np.array(diffs)
    max_ratio = float(np.max(diffs) / I_FULL)
    return {"n": n_points, "max_diff": float(np.max(diffs)), "max_ratio": max_ratio, "rows": rows}


# ============================================================================
# 自己検査2: G(a)-G(b) vs response()（区間 [a,b] への直接積分）
# ============================================================================
def selfcheck2(table: T.CumulativeTable, n_points: int = 40) -> dict:
    rng = np.random.default_rng(SEED + 2)
    i_d = rng.integers(0, len(T.D_AXIS_M), size=n_points)
    i_th = rng.integers(0, len(T.THETA_AXIS_DEG), size=n_points)
    # a < b になるよう、u 軸上の2点をソートして選ぶ（同じ点は除外）。
    n_u = len(T.U_AXIS_M)
    idx_pairs = rng.integers(0, n_u, size=(n_points, 2))
    idx_pairs = np.sort(idx_pairs, axis=1)
    # 同一点になった場合は片方をずらす。
    same = idx_pairs[:, 0] == idx_pairs[:, 1]
    idx_pairs[same, 1] = np.minimum(idx_pairs[same, 1] + 1, n_u - 1)

    diffs = []
    rows = []
    for k in range(n_points):
        d = float(T.D_AXIS_M[i_d[k]])
        th = float(T.THETA_AXIS_DEG[i_th[k]])
        ia, ib = int(idx_pairs[k, 0]), int(idx_pairs[k, 1])
        a, b = float(T.U_AXIS_M[ia]), float(T.U_AXIS_M[ib])
        table_val = T.segment_from_indices(table, i_d[k], i_th[k], ia, ib)
        ref_val = _reference_response(table.meta_sensor, d, th, a, b)
        diff = abs(table_val - ref_val)
        diffs.append(diff)
        rows.append({
            "d_mm": d * 1000, "theta_deg": th, "a_mm": a * 1000, "b_mm": b * 1000,
            "table": table_val, "ref": ref_val, "diff": diff,
        })

    diffs = np.array(diffs)
    max_ratio = float(np.max(diffs) / I_FULL)
    return {"n": n_points, "max_diff": float(np.max(diffs)), "max_ratio": max_ratio, "rows": rows}


# ============================================================================
# 自己検査3: |u|=120mm での累積の残りが満量比 1e-4 以下か（全格子点で走査）
# ============================================================================
def selfcheck3(sensor, surf, margin_m: float = T.U_MARGIN_M) -> dict:
    """`build_cumulative_table()` と同じ生成過程で、全 (d,θ) 格子点について
    forward残差 `G(+120mm)` と backward残差 `G(-(120+margin)mm) - G(-120mm)` を測る
    （収束の実測: 余白100mmで220mm・400mmと機械精度まで一致するのを別途確認済み。
    本作業の報告を参照）。"""
    n_margin_nodes = int(round(margin_m / T.U_STEP_M))
    u_gen_axis = T._axis_points(
        -(T.U_MAX_M + margin_m), T.U_MAX_M + margin_m, T.U_STEP_M, T.U_N + 2 * n_margin_nodes,
    )
    lo, hi = n_margin_nodes, n_margin_nodes + T.U_N

    forward = np.empty((len(T.D_AXIS_M), len(T.THETA_AXIS_DEG)))
    backward = np.empty_like(forward)
    t0 = time.time()
    for i, d in enumerate(T.D_AXIS_M):
        for j, th in enumerate(T.THETA_AXIS_DEG):
            bins = T.generate_dtheta_bin_integrals(sensor, surf, float(d), float(th), u_gen_axis, n_v=T.N_V_QUAD)
            G = T.cumulate_from_bin_integrals(bins)
            forward[i, j] = G[hi - 1]
            backward[i, j] = G[0] - G[lo]
    elapsed = time.time() - t0

    thr = 1e-4 * I_FULL
    mask_f = np.abs(forward) > thr
    mask_b = np.abs(backward) > thr

    def _worst(arr):
        i, j = np.unravel_index(int(np.argmax(np.abs(arr))), arr.shape)
        return {"d_mm": float(T.D_AXIS_M[i] * 1000), "theta_deg": float(T.THETA_AXIS_DEG[j]),
                "value": float(arr[i, j]), "ratio": float(arr[i, j] / I_FULL)}

    return {
        "elapsed_s": elapsed, "margin_mm": margin_m * 1000,
        "forward_max_ratio": float(np.max(np.abs(forward)) / I_FULL),
        "backward_max_ratio": float(np.max(np.abs(backward)) / I_FULL),
        "forward_n_over": int(mask_f.sum()), "backward_n_over": int(mask_b.sum()),
        "n_total": int(forward.size),
        "forward_worst": _worst(forward), "backward_worst": _worst(backward),
    }


def main() -> None:
    if not TABLE_PATH.exists():
        raise SystemExit(f"表が見つかりません: {TABLE_PATH}（先に mouse/ir_table.py で生成すること）")
    table = T.load_cumulative_table(TABLE_PATH)
    sensor = T.lf_sensor_spec()
    table.meta_sensor = sensor   # 検査関数の便宜上、動的に生やす（保存はしない）

    print("=" * 70)
    print("自己検査1: 表の格子点 vs response()（u=-120mm, [-120,+220]mm の壁）")
    r1 = selfcheck1(table)
    print(f"  n={r1['n']}  最大差={r1['max_diff']:.3e}  最大差(満量比)={r1['max_ratio']:.3e}"
          f"  {'PASS' if r1['max_ratio'] <= 1e-4 else 'FAIL'} (判定 <= 1e-4)")

    print("=" * 70)
    print("自己検査2: G(a)-G(b) vs response()（区間 [a,b] への直接積分）")
    r2 = selfcheck2(table)
    print(f"  n={r2['n']}  最大差={r2['max_diff']:.3e}  最大差(満量比)={r2['max_ratio']:.3e}"
          f"  {'PASS' if r2['max_ratio'] <= 1e-4 else 'FAIL'} (判定 <= 1e-4)")

    print("=" * 70)
    print("自己検査3: |u|=120mm での累積の残り（全29583格子点を走査）")
    r3 = selfcheck3(sensor, SurfaceSpec())
    print(f"  所要時間={r3['elapsed_s']:.1f}s  余白={r3['margin_mm']:.0f}mm")
    print(f"  forward 最大(満量比)={r3['forward_max_ratio']:.3e}"
          f"  閾値超え={r3['forward_n_over']}/{r3['n_total']}  最悪={r3['forward_worst']}")
    print(f"  backward最大(満量比)={r3['backward_max_ratio']:.3e}"
          f"  閾値超え={r3['backward_n_over']}/{r3['n_total']}  最悪={r3['backward_worst']}")
    ok3 = r3['forward_max_ratio'] <= 1e-4 and r3['backward_max_ratio'] <= 1e-4
    print(f"  {'PASS' if ok3 else 'FAIL'} (判定: 両方とも <= 1e-4)")


if __name__ == "__main__":
    main()
