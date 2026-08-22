"""verification/audit_060_table_sensor.py

`verification/AUDIT_060_PREREG_table_sensor.md`（事前登録）§3〜§5 を実行する
（第2段階: `response_table()` の幾何・組み上げの正しさを確かめ、本測定まで行う）。

使い方（前景で実行。`--stage` で分割する。1 回の呼び出しは 10 分以内）:

    .venv/bin/python verification/audit_060_table_sensor.py --stage selfcheck123
    .venv/bin/python verification/audit_060_table_sensor.py --stage selfcheck4
    .venv/bin/python verification/audit_060_table_sensor.py --stage selfcheck5
    .venv/bin/python verification/audit_060_table_sensor.py --stage negctrl1
    .venv/bin/python verification/audit_060_table_sensor.py --stage negctrl2
    .venv/bin/python verification/audit_060_table_sensor.py --stage main
    .venv/bin/python verification/audit_060_table_sensor.py --stage breakdown
    .venv/bin/python verification/audit_060_table_sensor.py --stage summary

`mouse/data/ir_cumulative_table.npz`（`mouse/ir_table.py::build_cumulative_table()` で
あらかじめ作成済み）を読み込んで検査する。表そのものを作り直すには
`mouse/ir_table.py` を直接呼ぶこと（本ファイル末尾のコメント参照）。

本測定・否定対照は `verification/audit_059_fast_sensor.py` と**同じ標本・同じ基準**
（無作為400 + 行き止まりの奥364 = 764姿勢、`outputs/audit_059/baseline48.json` ＋
光線追跡の増分）を使う。姿勢生成・基準の組み立ては同ファイルの関数をそのまま import する
（事前登録「標本と基準はAUDIT_059と同じ」を字面どおり満たすため、独自に再実装しない）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import Rect, wall_obstacles
from mouse import ir_table as T
from mouse.ir_sensor import (
    DEFAULT_MAX_RANGE_M,
    DEFAULT_WALL_HEIGHT_M,
    IrSensorSpec,
    SurfaceSpec,
    build_maze_cell_index,
    response,
    response_table,
)
from mouse.params import RobotParams
from verification.audit_059_fast_sensor import (
    MAZE_PATH,
    N_RANDOM,
    OUT_DIR as AUDIT059_OUT_DIR,
    SEED as AUDIT059_SEED,
    _corner_baseline_values,
    _percentile95,
    all_poses,
    build_ir_specs,
    gen_random_poses,
    load_geometry,
)

I_FULL = 0.8298934   # 満量（AUDIT_050 §2-2 以来の固定値。既存の全監査と同じ規約を踏襲）
TABLE_PATH = REPO_ROOT / "mouse" / "data" / "ir_cumulative_table.npz"
N_GRID_REF = 48       # 自己検査1・2 の基準（response() の細かい格子）
THIN_M = 1e-6         # 比較用の「厚みゼロ」壁の厚み（十分薄ければ端面の寄与は無視できる）

SEED = 20260822
SURF = SurfaceSpec()

OUT_DIR = REPO_ROOT / "outputs" / "audit_060"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def stage_selfcheck123() -> None:
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


# ============================================================================
# 第2/3段階: response_table() の幾何・組み上げ・本測定
# ============================================================================
CELL_SIZE = RobotParams().cell_size


def _load_maze_and_specs():
    p, rects, W, H, cell = load_geometry()
    specs = build_ir_specs(p)
    cell_index = build_maze_cell_index(rects, cell)
    return p, rects, W, H, cell, specs, cell_index


# ---------------------------------------------------------------------------
# 自己検査4: 索引で集めた候補が、全走査で得られる近傍を取りこぼさない
# ---------------------------------------------------------------------------
def _brute_force_near_owner_ids(led, pt, rect_arr: np.ndarray, max_range_m: float) -> np.ndarray:
    """「全走査」で得る近傍（事前登録§3自己検査4）を独立に（索引を使わず）求める。

    🔴 **最初の定義（射程内 かつ バックフェイスカリングだけ）は広すぎた**（実測: 764姿勢中
    743姿勢で取りこぼしが出た）。射程内判定は方向を問わない350mm半径の円なので、光軸から
    大きく外れた——原理的にLEDからもPTからも見えようがない——矩形まで「近傍」に含めて
    しまい、区画索引（光軸に沿って歩く設計）が見つけられなくて当然だった。

    正しい基準は `response_fast()`/`_corner_interreflection_total()` が実際に候補として
    使う集合——「射程内・バックフェイスカリング済みの面のうち、LED光錐**または**PT視野に
    入り得るもの」（`_corner_interreflection_total()` の `cand_mask = led_mask | pt_mask`
    と同じ式。直接光の対象=`led_mask & pt_mask`のANDより広く、隅の相互反射の対象を含む
    ORの方が「寄与し得る全体」を過不足なく表す）。これを索引に一切頼らず、`rects` 全体を
    ブルートフォースで `_facets_maybe_in_cone_batch` に通して独立に求める。
    """
    cx, cy, hx, hy = rect_arr[:, 0], rect_arr[:, 1], rect_arr[:, 2], rect_arr[:, 3]
    diag = np.hypot(hx, hy)
    dist = np.hypot(cx - led.pos[0], cy - led.pos[1])
    near = (dist - diag) <= max_range_m

    n_rect = rect_arr.shape[0]
    zc = np.zeros_like(cx)
    centers4 = np.stack([
        np.stack([cx + hx, cy, zc], axis=-1),
        np.stack([cx - hx, cy, zc], axis=-1),
        np.stack([cx, cy + hy, zc], axis=-1),
        np.stack([cx, cy - hy, zc], axis=-1),
    ], axis=0)
    normals4 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    u_y = np.array([0.0, 1.0, 0.0]); u_x = np.array([1.0, 0.0, 0.0]); v_z = np.array([0.0, 0.0, 1.0])
    u_vecs4 = np.array([u_y, u_y, u_x, u_x])
    v_vecs4 = np.array([v_z, v_z, v_z, v_z])
    half_u4 = np.stack([hy, hy, hx, hx], axis=0)
    half_v4 = np.full((4, n_rect), DEFAULT_WALL_HEIGHT_M / 2.0)

    to_led4 = led.pos[None, None, :] - centers4
    vis_mask4 = np.einsum("fmk,fk->fm", to_led4, normals4) > 0.0
    f_sel, m_sel = np.nonzero(vis_mask4)
    if f_sel.size == 0:
        return np.nonzero(near & False)[0]

    from mouse.ir_sensor import (
        _facets_maybe_in_cone_batch, DEFAULT_LED_CONE_MARGIN_DEG, DEFAULT_PT_CONE_MARGIN_DEG,
        DEFAULT_CONE_FILTER_N_U, DEFAULT_CONE_FILTER_N_V,
    )
    centers_sel = centers4[f_sel, m_sel]
    normals_sel = normals4[f_sel]
    u_sel = u_vecs4[f_sel]
    v_sel = v_vecs4[f_sel]
    half_u_sel = half_u4[f_sel, m_sel]
    half_v_sel = half_v4[f_sel, m_sel]
    owner_sel = m_sel

    stacked = (centers_sel, normals_sel, u_sel, v_sel, half_u_sel, half_v_sel)
    led_mask = _facets_maybe_in_cone_batch(
        stacked, led.pos, led.axis, DEFAULT_LED_CONE_MARGIN_DEG, DEFAULT_CONE_FILTER_N_U, DEFAULT_CONE_FILTER_N_V,
    )
    pt_mask = _facets_maybe_in_cone_batch(
        stacked, pt.pos, pt.axis, DEFAULT_PT_CONE_MARGIN_DEG, DEFAULT_CONE_FILTER_N_U, DEFAULT_CONE_FILTER_N_V,
    )
    cand_mask = led_mask | pt_mask
    owners_in_cone = np.unique(owner_sel[cand_mask])

    owners_near = np.nonzero(near)[0]
    keep = np.intersect1d(owners_in_cone, owners_near, assume_unique=False)
    return keep


def stage_selfcheck4() -> None:
    table = T.load_cumulative_table(TABLE_PATH)
    p, rects, W, H, cell, specs, cell_index = _load_maze_and_specs()
    rect_arr = np.asarray(rects, dtype=float)
    poses = all_poses(specs, W, H, cell)

    n_miss_total = 0
    n_poses_with_miss = 0
    miss_examples = []
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        from mouse.ir_sensor import _sensor_world_geometry
        led, pt = _sensor_world_geometry(sensor, pose)
        near_ids = _brute_force_near_owner_ids(led, pt, rect_arr, DEFAULT_MAX_RANGE_M)

        cand_ids = response_table(
            sensor, pose, rects, SURF, table, cell_index, cell,
            return_candidate_ids=True,
        )
        missing = [int(i) for i in near_ids if int(i) not in cand_ids]
        if missing:
            n_poses_with_miss += 1
            n_miss_total += len(missing)
            if len(miss_examples) < 5:
                miss_examples.append({"idx": pose_d["idx"], "group": pose_d["group"], "missing": missing})

    out = {
        "n_poses": len(poses), "n_poses_with_miss": n_poses_with_miss,
        "n_miss_total": n_miss_total, "miss_examples": miss_examples,
    }
    (OUT_DIR / "selfcheck4.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"自己検査4: 取りこぼし姿勢 {n_poses_with_miss}/{len(poses)}  取りこぼし総数 {n_miss_total}"
          f"  {'PASS' if n_miss_total == 0 else 'FAIL'} (判定: 764姿勢で0件)")
    if miss_examples:
        print("  例:", miss_examples)


# ---------------------------------------------------------------------------
# 自己検査5: 遮蔽の無い単純な配置（壁1枚・正対と斜め）で response() と満量比6e-4以下
# ---------------------------------------------------------------------------
def _place_wall_for_dtheta(sensor: IrSensorSpec, d_m: float, theta_deg: float, hy: float = 0.09):
    """`sensor` から見て距離 `d_m`・入射角 `theta_deg` の位置に、厚み12mmの壁1枚（半長`hy`）を
    正対させて置く（機体姿勢 `(0,0,theta)` 固定・壁の位置を動かす）。"""
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    axis_world = np.array([c * sensor.axis[0] - s * sensor.axis[1], s * sensor.axis[0] + c * sensor.axis[1]])
    sensor_pos_world = np.array([c * sensor.pos[0] - s * sensor.pos[1], s * sensor.pos[0] + c * sensor.pos[1]])
    face_point = sensor_pos_world + d_m * axis_world / np.linalg.norm(axis_world)
    wall = Rect(cx=float(face_point[0]) + 0.006, cy=float(face_point[1]), hx=0.006, hy=hy)
    pose = (0.0, 0.0, theta)
    return wall, pose


def stage_selfcheck5() -> None:
    table = T.load_cumulative_table(TABLE_PATH)
    sensor = T.lf_sensor_spec()
    cell_index = build_maze_cell_index([Rect(0.0, 0.0, 0.006, 0.09)], CELL_SIZE)  # ダミー（都度作り直す）

    cases = [(d, th) for d in (0.020, 0.030, 0.044, 0.060, 0.090, 0.150, 0.250)
              for th in (0.0, 10.0, -20.0, 35.0, -50.0)]
    diffs = []
    rows = []
    for d, th in cases:
        wall, pose = _place_wall_for_dtheta(sensor, d, th)
        idx = build_maze_cell_index([wall], CELL_SIZE)
        v_ref = response(sensor, pose, [wall], SURF, n_grid=N_GRID_REF, include_floor=False)
        v_tbl = response_table(sensor, pose, [wall], SURF, table, idx, CELL_SIZE, interreflection=False)
        diff = abs(v_tbl - v_ref)
        diffs.append(diff)
        rows.append({"d_mm": d * 1000, "theta_deg": th, "ref": v_ref, "table": v_tbl, "diff": diff})

    diffs = np.array(diffs)
    max_ratio = float(np.max(diffs) / I_FULL)
    out = {"n": len(cases), "max_diff": float(np.max(diffs)), "max_ratio": max_ratio, "rows": rows}
    (OUT_DIR / "selfcheck5.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"自己検査5: n={len(cases)}  最大差(満量比)={max_ratio:.3e}"
          f"  {'PASS' if max_ratio <= 6e-4 else 'FAIL'} (判定 <= 6e-4。追記1で改めたしきい値)")
    worst = max(rows, key=lambda r: r["diff"])
    print(f"  最悪ケース: d={worst['d_mm']:.0f}mm theta={worst['theta_deg']:.0f}deg"
          f"  ref={worst['ref']:.6f} table={worst['table']:.6f}")


# ---------------------------------------------------------------------------
# 本測定・否定対照の共通部品
# ---------------------------------------------------------------------------
def _coarsen_table(table: T.CumulativeTable, stride_d: int, stride_th: int, stride_u: int) -> T.CumulativeTable:
    """否定対照1用: 表を間引く（距離2mm*4=8mm・角度1度*4=4度・壁沿い0.5mm*4=2mm）。
    既存の細かい表から strided slice を取るだけ（生成し直さない）。"""
    return T.CumulativeTable(
        d_axis_m=table.d_axis_m[::stride_d],
        theta_axis_deg=table.theta_axis_deg[::stride_th],
        u_axis_m=table.u_axis_m[::stride_u],
        values=table.values[::stride_d, ::stride_th, ::stride_u],
        meta=dict(table.meta, coarsened=[stride_d, stride_th, stride_u]),
    )


def _run_table_series(specs, rects, cell_index, cell, table, poses, **kw) -> Dict:
    values = []
    times = []
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        t0 = time.perf_counter()
        v = response_table(sensor, pose, rects, SURF, table, cell_index, cell, **kw)
        t1 = time.perf_counter()
        values.append(float(v))
        times.append(t1 - t0)
    return {"n_poses": len(poses), "values": values, "times": times}


def stage_negctrl1() -> None:
    """否定対照1: 表を粗くする（距離8mm・角度4度・壁沿い2mm）。M6 > 0.03 となるはず。"""
    table = T.load_cumulative_table(TABLE_PATH)
    coarse = _coarsen_table(table, 4, 4, 4)
    print(f"粗い表: d軸{len(coarse.d_axis_m)}点(刻み{(coarse.d_axis_m[1]-coarse.d_axis_m[0])*1000:.1f}mm) "
          f"theta軸{len(coarse.theta_axis_deg)}点(刻み{coarse.theta_axis_deg[1]-coarse.theta_axis_deg[0]:.1f}deg) "
          f"u軸{len(coarse.u_axis_m)}点(刻み{(coarse.u_axis_m[1]-coarse.u_axis_m[0])*1000:.1f}mm)")

    p, rects, W, H, cell, specs, cell_index = _load_maze_and_specs()
    poses = all_poses(specs, W, H, cell)
    out = _run_table_series(specs, rects, cell_index, cell, coarse, poses, interreflection=False)
    (OUT_DIR / "negctrl1.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"negctrl1: 完了 平均時間 {1000.0 * sum(out['times']) / len(out['times']):.3f} ms")


def stage_negctrl2() -> None:
    """否定対照2: 影の境目を無視する（面が見えていれば全長を使う）。M6 > 0.03 となるはず。"""
    table = T.load_cumulative_table(TABLE_PATH)
    p, rects, W, H, cell, specs, cell_index = _load_maze_and_specs()
    poses = all_poses(specs, W, H, cell)
    out = _run_table_series(specs, rects, cell_index, cell, table, poses,
                             interreflection=False, ignore_shadow_boundary=True)
    (OUT_DIR / "negctrl2.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"negctrl2: 完了 平均時間 {1000.0 * sum(out['times']) / len(out['times']):.3f} ms")


def stage_main() -> None:
    """本測定: response_table()（既定=隅の相互反射込み）の値・時間。"""
    table = T.load_cumulative_table(TABLE_PATH)
    p, rects, W, H, cell, specs, cell_index = _load_maze_and_specs()
    poses = all_poses(specs, W, H, cell)
    out = _run_table_series(specs, rects, cell_index, cell, table, poses, interreflection=True)
    (OUT_DIR / "main.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"main: 完了 平均時間 {1000.0 * sum(out['times']) / len(out['times']):.3f} ms")


def stage_breakdown() -> None:
    """時間の内訳（幾何／表引き／隅）を764姿勢平均で測る。"""
    table = T.load_cumulative_table(TABLE_PATH)
    p, rects, W, H, cell, specs, cell_index = _load_maze_and_specs()
    poses = all_poses(specs, W, H, cell)
    acc = {"geometry": 0.0, "lookup": 0.0, "corner": 0.0}
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        tb: Dict = {}
        response_table(sensor, pose, rects, SURF, table, cell_index, cell,
                        interreflection=True, time_breakdown=tb)
        for k in acc:
            acc[k] += tb.get(k, 0.0)
    n = len(poses)
    out = {"n_poses": n, "mean_ms": {k: 1000.0 * v / n for k, v in acc.items()}}
    (OUT_DIR / "breakdown.json").write_text(json.dumps(out), encoding="utf-8")
    print("breakdown:", json.dumps(out, indent=2, ensure_ascii=False))


def stage_summary() -> None:
    b48 = json.loads((AUDIT059_OUT_DIR / "baseline48.json").read_text(encoding="utf-8"))
    v48 = np.array(b48["values"])

    p, rects, W, H, cell, specs, cell_index = _load_maze_and_specs()
    poses = all_poses(specs, W, H, cell)
    assert len(poses) == len(v48)
    baseline = _corner_baseline_values(poses, v48)

    def m6_of(path: Path) -> float:
        d = json.loads(path.read_text(encoding="utf-8"))
        v = np.array(d["values"])
        return _percentile95(np.abs(v - baseline)) / I_FULL

    print("=" * 70)
    sc4 = json.loads((OUT_DIR / "selfcheck4.json").read_text(encoding="utf-8"))
    print(f"自己検査4: 取りこぼし総数 {sc4['n_miss_total']}/764姿勢"
          f"  {'PASS' if sc4['n_miss_total'] == 0 else 'FAIL'}")

    sc5 = json.loads((OUT_DIR / "selfcheck5.json").read_text(encoding="utf-8"))
    print(f"自己検査5: 最大差(満量比) = {sc5['max_ratio']:.3e}"
          f"  {'PASS' if sc5['max_ratio'] <= 6e-4 else 'FAIL'} (判定 <= 6e-4)")

    m6_neg1 = m6_of(OUT_DIR / "negctrl1.json")
    print(f"否定対照1（表を粗く: 8mm/4deg/2mm）: M6 = {m6_neg1:.6f}"
          f"  {'作動(>0.03)' if m6_neg1 > 0.03 else '★不作動(検査が鈍い)'}")

    m6_neg2 = m6_of(OUT_DIR / "negctrl2.json")
    print(f"否定対照2（影の境目を無視）: M6 = {m6_neg2:.6f}"
          f"  {'作動(>0.03)' if m6_neg2 > 0.03 else '★不作動(検査が鈍い)'}")

    negctrl_ok = (m6_neg1 > 0.03) and (m6_neg2 > 0.03)
    print()
    if not negctrl_ok:
        print("否定対照が両方とも作動しなかった。事前登録§5「どちらも入らなければ判定を行わない」"
              "に従い、以下のM6・Tの判定は行わない（値は参考として出す）。")

    main_d = json.loads((OUT_DIR / "main.json").read_text(encoding="utf-8"))
    v_main = np.array(main_d["values"])
    m6_main = _percentile95(np.abs(v_main - baseline)) / I_FULL
    T_ms = 1000.0 * float(np.mean(main_d["times"]))

    print(f"M6（本測定） = {m6_main:.6f}")
    print(f"T（本測定・平均） = {T_ms:.4f} ms")
    if negctrl_ok:
        m6_verdict = "合格(<=0.01)" if m6_main <= 0.01 else ("惜しい(0.01-0.03)" if m6_main <= 0.03 else "不合格(>0.03)")
        t_verdict = "合格(<=0.5ms)" if T_ms <= 0.5 else ("目標未達(0.5-1.0ms)" if T_ms <= 1.0 else "不合格(>1.0ms)")
        print(f"M6判定: {m6_verdict}")
        print(f"T判定: {t_verdict}")

    if (OUT_DIR / "breakdown.json").exists():
        br = json.loads((OUT_DIR / "breakdown.json").read_text(encoding="utf-8"))
        print(f"時間の内訳（764姿勢平均）: {br['mean_ms']}")

    table_path_size_mb = TABLE_PATH.stat().st_size / (1024 * 1024)
    print(f"表の大きさ: {table_path_size_mb:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                     choices=["selfcheck123", "selfcheck4", "selfcheck5",
                              "negctrl1", "negctrl2", "main", "breakdown", "summary"])
    args = ap.parse_args()

    stage_fn = {
        "selfcheck123": stage_selfcheck123,
        "selfcheck4": stage_selfcheck4,
        "selfcheck5": stage_selfcheck5,
        "negctrl1": stage_negctrl1,
        "negctrl2": stage_negctrl2,
        "main": stage_main,
        "breakdown": stage_breakdown,
        "summary": stage_summary,
    }[args.stage]
    stage_fn()


if __name__ == "__main__":
    main()
