"""verification/audit_059_fast_sensor.py

`verification/AUDIT_059_PREREG_fast_sensor.md`（段A: 直接光のみ）の測定を実行する。

使い方（1 回の呼び出しは 10 分以内。前景で実行すること。ステージごとに分けてある）:

    .venv/bin/python verification/audit_059_fast_sensor.py --stage baseline48
    .venv/bin/python verification/audit_059_fast_sensor.py --stage baseline64
    .venv/bin/python verification/audit_059_fast_sensor.py --stage selfcheck
    .venv/bin/python verification/audit_059_fast_sensor.py --stage negctrl1
    .venv/bin/python verification/audit_059_fast_sensor.py --stage negctrl2
    .venv/bin/python verification/audit_059_fast_sensor.py --stage main
    .venv/bin/python verification/audit_059_fast_sensor.py --stage breakdown
    .venv/bin/python verification/audit_059_fast_sensor.py --stage summary

各ステージは `outputs/audit_059/*.json` に結果を保存する（再実行しても前段の結果は
壊さない。`summary` はそれらを読み直して事前登録の判定表を出すだけ）。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import wall_obstacles
from mouse.ir_sensor import (
    IrSensorSpec,
    SurfaceSpec,
    response,
    response_fast,
    _sensor_world_geometry,
    _wall_facets,
    _obstacle_boxes,
    _facet_grid,
    _segment_occluded,
    _integrate_facet,
    _facet_maybe_in_cone,
    _stack_facets,
    _facets_maybe_in_cone_batch,
    DEFAULT_N_GRID_FAST,
    DEFAULT_LED_CONE_MARGIN_DEG,
    DEFAULT_PT_CONE_MARGIN_DEG,
    DEFAULT_CONE_FILTER_N_U,
    DEFAULT_CONE_FILTER_N_V,
    DEFAULT_MAX_RANGE_M,
    DEFAULT_WALL_HEIGHT_M,
)
from mouse.params import RobotParams

I_FULL = 0.8298934   # 満量（AUDIT_050 §2-2 と同じ固定値。事前登録がこの規約を踏襲）
MAZE_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "maze_41001.npz"
FRONT_GRID_PATH = REPO_ROOT / "outputs" / "audit_056" / "front_grid.json"
OUT_DIR = REPO_ROOT / "outputs" / "audit_059"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260822
N_RANDOM = 400

SURF = SurfaceSpec()   # 既定のまま（diffuse=0.8・specular=0.10・shininess=40。事前登録§0）


# ============================================================================
# 迷路・センサ・姿勢の標本
# ============================================================================
def load_geometry():
    p = RobotParams()
    d = np.load(MAZE_PATH)
    rects = wall_obstacles(d["v_walls"], d["h_walls"], cell_size=p.cell_size)
    W = int(d["v_walls"].shape[0] - 1)
    H = int(d["v_walls"].shape[1])
    return p, rects, W, H, p.cell_size


def build_ir_specs(p: RobotParams) -> List[IrSensorSpec]:
    specs = []
    for s in p.sensors:
        pos = tuple(float(v) for v in s["pos"].split())
        axis = tuple(float(v) for v in s["zaxis"].split())
        specs.append(IrSensorSpec(name=s["name"], pos=pos, axis=axis))
    return specs


def gen_random_poses(seed: int, n: int, W: int, H: int, cell: float) -> List[Dict]:
    """事前登録 §2: 無作為 400 姿勢（区画無作為・中心±40mm・方位一様・センサ無作為1本）。"""
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n):
        cx = rng.integers(0, W)
        cy = rng.integers(0, H)
        x = (cx + 0.5) * cell + rng.uniform(-0.04, 0.04)
        y = (cy + 0.5) * cell + rng.uniform(-0.04, 0.04)
        th = rng.uniform(-math.pi, math.pi)
        sensor_idx = int(rng.integers(0, 4))
        poses.append({
            "idx": i, "group": "random",
            "sensor_idx": sensor_idx,
            "x": float(x), "y": float(y), "theta": float(th),
        })
    return poses


def load_front_grid_poses(specs: List[IrSensorSpec]) -> List[Dict]:
    """事前登録 §2: 行き止まりの奥364点（outputs/audit_056/front_grid.json）。"""
    name_to_idx = {s.name: i for i, s in enumerate(specs)}
    data = json.loads(FRONT_GRID_PATH.read_text(encoding="utf-8"))
    pts = data["points"]
    assert len(pts) == 364, f"front_grid.json の点数が想定と違う: {len(pts)}"
    poses = []
    for i, pt in enumerate(pts):
        poses.append({
            "idx": i, "group": "front_grid",
            "sensor_idx": name_to_idx[pt["sensor_name"]],
            "x": float(pt["x"]), "y": float(pt["y"]), "theta": float(pt["theta"]),
        })
    return poses


def all_poses(specs: List[IrSensorSpec], W: int, H: int, cell: float) -> List[Dict]:
    """本測定・否定対照が使う標本（400無作為 + 364行き止まりの奥 = 764）。"""
    return gen_random_poses(SEED, N_RANDOM, W, H, cell) + load_front_grid_poses(specs)


# ============================================================================
# ステージ: baseline48 / baseline64（response() を n_grid=48/64 で回す。段Aの錨とM6の基準）
# ============================================================================
def _run_response_series(specs, rects, poses, n_grid: int) -> Dict:
    values = []
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        v = response(sensor, pose, rects, SURF, n_grid=n_grid, include_floor=False)
        values.append(float(v))
    return {"n_grid": n_grid, "n_poses": len(poses), "values": values}


def stage_baseline48(specs, rects, poses) -> None:
    out = _run_response_series(specs, rects, poses, n_grid=48)
    (OUT_DIR / "baseline48.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"baseline48: {len(poses)} 姿勢 完了")


def stage_baseline64(specs, rects, poses_random_only) -> None:
    out = _run_response_series(specs, rects, poses_random_only, n_grid=64)
    (OUT_DIR / "baseline64.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"baseline64: {len(poses_random_only)} 姿勢（無作為400のみ） 完了")


# ============================================================================
# ステージ: selfcheck（絞り込みあり vs なし。§自己検査）
# ============================================================================
def stage_selfcheck(specs, rects, poses) -> None:
    diffs = []
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        v_narrow = response_fast(sensor, pose, rects, SURF, n_grid=DEFAULT_N_GRID_FAST, narrow_facets=True)
        v_wide = response_fast(sensor, pose, rects, SURF, n_grid=DEFAULT_N_GRID_FAST, narrow_facets=False)
        diffs.append(abs(v_narrow - v_wide))
    out = {"n_poses": len(poses), "diffs": diffs}
    (OUT_DIR / "selfcheck.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"selfcheck: 最大差(満量比) = {max(diffs) / I_FULL:.3e}")


# ============================================================================
# ステージ: selfcheck2（速さの追加対策1: 遮蔽候補直方体のAABB削減の前後で答えが変わらないこと）
# ============================================================================
def stage_selfcheck2(specs, rects, poses) -> None:
    """`prune_occlusion_boxes=True/False` で `response_fast` の値が変わらないことを確かめる
    （教授セッションの指示: 「絞ったことで答えが変わっていないことを必ず実測で確かめる」）。"""
    diffs = []
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        v_pruned = response_fast(sensor, pose, rects, SURF, prune_occlusion_boxes=True)
        v_full = response_fast(sensor, pose, rects, SURF, prune_occlusion_boxes=False)
        diffs.append(abs(v_pruned - v_full))
    out = {"n_poses": len(poses), "diffs": diffs}
    (OUT_DIR / "selfcheck2.json").write_text(json.dumps(out), encoding="utf-8")
    max_ratio = max(diffs) / I_FULL
    print(f"selfcheck2: 最大差(満量比) = {max_ratio:.3e}"
          f"（判定: {'合格' if max_ratio <= 1e-9 else '不合格'} — 基準1e-9以下・{len(poses)}姿勢全数）")


# ============================================================================
# ステージ: selfcheck3（速さの追加対策2: 絞り込み判定のnumpy一括版が面ごとループ版と一致すること）
# ============================================================================
def stage_selfcheck3(specs, rects, poses) -> None:
    """`_facets_maybe_in_cone_batch`（numpy一括版）が `_facet_maybe_in_cone`（面ごとループ版）と
    同じ「残す面の集合」を返すことを、response_fast と同じ前処理（射程の足切り・
    バックフェイスカリング）を経た候補面すべてで確かめる。"""
    n_mismatch = 0
    n_facets_total = 0
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        led, pt = _sensor_world_geometry(sensor, pose)

        near_rects = []
        for r in rects:
            diag = math.hypot(r.hx, r.hy)
            dist_center = math.hypot(r.cx - led.pos[0], r.cy - led.pos[1])
            if dist_center - diag > DEFAULT_MAX_RANGE_M:
                continue
            near_rects.append(r)

        facets = []
        for r in near_rects:
            facets.extend(_wall_facets([r], DEFAULT_WALL_HEIGHT_M))

        vis_facets = [f for f in facets if float(np.dot(led.pos - f.center, f.normal)) > 0.0]
        if not vis_facets:
            continue
        n_facets_total += len(vis_facets)

        # 面ごとループ版（絞り込みの中身の正本）。
        loop_keep = np.array([
            _facet_maybe_in_cone(f, led.pos, led.axis, DEFAULT_LED_CONE_MARGIN_DEG,
                                  DEFAULT_CONE_FILTER_N_U, DEFAULT_CONE_FILTER_N_V)
            and _facet_maybe_in_cone(f, pt.pos, pt.axis, DEFAULT_PT_CONE_MARGIN_DEG,
                                      DEFAULT_CONE_FILTER_N_U, DEFAULT_CONE_FILTER_N_V)
            for f in vis_facets
        ])

        # numpy 一括版。
        stacked = _stack_facets(vis_facets)
        batch_led = _facets_maybe_in_cone_batch(
            stacked, led.pos, led.axis, DEFAULT_LED_CONE_MARGIN_DEG,
            DEFAULT_CONE_FILTER_N_U, DEFAULT_CONE_FILTER_N_V,
        )
        batch_pt = _facets_maybe_in_cone_batch(
            stacked, pt.pos, pt.axis, DEFAULT_PT_CONE_MARGIN_DEG,
            DEFAULT_CONE_FILTER_N_U, DEFAULT_CONE_FILTER_N_V,
        )
        batch_keep = batch_led & batch_pt

        if not np.array_equal(loop_keep, batch_keep):
            n_mismatch += 1

    out = {"n_poses": len(poses), "n_facets_total": n_facets_total, "n_mismatch_poses": n_mismatch}
    (OUT_DIR / "selfcheck3.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"selfcheck3: 不一致姿勢数 = {n_mismatch}/{len(poses)}（延べ候補面数 {n_facets_total}）"
          f"（判定: {'合格' if n_mismatch == 0 else '不合格'} — 全姿勢で面の集合が完全一致であること）")


# ============================================================================
# ステージ: negctrl1 / negctrl2（否定対照。事前登録§5。本測定の前に走らせる）
# ============================================================================
def stage_negctrl1(specs, rects, poses) -> None:
    """光錐の判定角を半分（LED±1.5°・PT±3°）にした版。"""
    values = []
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        v = response_fast(
            sensor, pose, rects, SURF, n_grid=DEFAULT_N_GRID_FAST,
            led_cone_margin_deg=1.5, pt_cone_margin_deg=3.0,
        )
        values.append(float(v))
    out = {"n_poses": len(poses), "values": values, "variant": "cone_margin_halved(1.5/3.0deg)"}
    (OUT_DIR / "negctrl1.json").write_text(json.dumps(out), encoding="utf-8")
    print("negctrl1: 完了")


def stage_negctrl2(specs, rects, poses) -> None:
    """n_grid=6 にした版。"""
    values = []
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        v = response_fast(sensor, pose, rects, SURF, n_grid=6)
        values.append(float(v))
    out = {"n_poses": len(poses), "values": values, "variant": "n_grid=6"}
    (OUT_DIR / "negctrl2.json").write_text(json.dumps(out), encoding="utf-8")
    print("negctrl2: 完了")


def _response_fast_single_facet(sensor, pose, rects, surf, n_grid=DEFAULT_N_GRID_FAST):
    """事前登録 追記1 の否定対照: 絞り込みを「光軸が最初に当たる1枚だけを積分対象にする」に
    差し替えた版（絞り込みを本当に壊す対照）。response_fast() 本体は変更しない、診断専用の複製。

    「当たる」は矩形内に実際に入る交点だけを候補にする真の当たり判定
    （`_facet_closest_direction_point` のようなクランプはしない）。候補のうち光軸パラメータ
    `t`（LED からの距離）が最小のものを「最初に当たる面」として採用する。"""
    led, pt = _sensor_world_geometry(sensor, pose)
    half_angle_max = max(sensor.led_half_angle_deg, sensor.pt_half_angle_deg)

    near_rects = []
    for r in rects:
        diag = math.hypot(r.hx, r.hy)
        dist_center = math.hypot(r.cx - led.pos[0], r.cy - led.pos[1])
        if dist_center - diag > DEFAULT_MAX_RANGE_M:
            continue
        near_rects.append(r)

    facets = []
    facet_owner = []
    for owner_idx, r in enumerate(near_rects):
        for f in _wall_facets([r], DEFAULT_WALL_HEIGHT_M):
            facets.append(f)
            facet_owner.append(owner_idx)

    vis_facets = []
    vis_owner = []
    for facet, owner_idx in zip(facets, facet_owner):
        to_led = led.pos - facet.center
        if float(np.dot(to_led, facet.normal)) <= 0.0:
            continue
        vis_facets.append(facet)
        vis_owner.append(owner_idx)

    if not vis_facets:
        return 0.0

    best_idx = -1
    best_t = math.inf
    for i, facet in enumerate(vis_facets):
        denom = float(np.dot(led.axis, facet.normal))
        if abs(denom) <= 1e-9:
            continue
        t = float(np.dot(facet.center - led.pos, facet.normal)) / denom
        if t <= 0.0:
            continue
        p_hit = led.pos + t * led.axis
        rel = p_hit - facet.center
        pu = float(np.dot(rel, facet.u))
        pv = float(np.dot(rel, facet.v))
        if abs(pu) > facet.half_u or abs(pv) > facet.half_v:
            continue
        if t < best_t:
            best_t = t
            best_idx = i

    if best_idx < 0:
        return 0.0

    sel_facets = [vis_facets[best_idx]]
    sel_owner = [vis_owner[best_idx]]

    boxes = _obstacle_boxes(near_rects, DEFAULT_WALL_HEIGHT_M)

    grids = [_facet_grid(f, led, n_grid, half_angle_max, sensor.separation_m) for f in sel_facets]
    points_list = [g[0] for g in grids]
    dA_list = [g[1] for g in grids]

    if boxes.shape[0] > 0:
        all_points = np.stack(points_list, axis=0)
        owner_arr = np.array(sel_owner, dtype=int)
        occ_led = _segment_occluded(all_points, led.pos, boxes, owner_arr)
        occ_pt = _segment_occluded(all_points, pt.pos, boxes, owner_arr)
        occluded_all = occ_led | occ_pt
        occluded_list = [occluded_all[i] for i in range(len(sel_facets))]
    else:
        occluded_list = [None] * len(sel_facets)

    total = 0.0
    for facet, points, dA, occluded in zip(sel_facets, points_list, dA_list, occluded_list):
        total += _integrate_facet(facet, led, pt, surf, points, dA, 1.0, 1.0, DEFAULT_MAX_RANGE_M, occluded)

    return total * sensor.gain


def stage_negctrl1b(specs, rects, poses) -> None:
    """事前登録 追記1 の否定対照（差し替え後）: 光軸が最初に当たる1枚だけを積分対象にする版。"""
    values = []
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        v = _response_fast_single_facet(sensor, pose, rects, SURF)
        values.append(float(v))
    out = {"n_poses": len(poses), "values": values, "variant": "closest_hit_facet_only"}
    (OUT_DIR / "negctrl1b.json").write_text(json.dumps(out), encoding="utf-8")
    print("negctrl1b: 完了")


# ============================================================================
# ステージ: main（本測定。response_fast 既定設定。値＋時間）
# ============================================================================
def stage_main(specs, rects, poses) -> None:
    values = []
    times = []
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        t0 = time.perf_counter()
        v = response_fast(sensor, pose, rects, SURF)
        t1 = time.perf_counter()
        values.append(float(v))
        times.append(t1 - t0)
    out = {"n_poses": len(poses), "values": values, "times": times}
    (OUT_DIR / "main.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"main: 完了 平均時間 {1000.0 * sum(times) / len(times):.3f} ms")


# ============================================================================
# ステージ: breakdown（1回あたりの時間の内訳: 絞り込み／積分（格子構築+積分）／遮蔽）
# ============================================================================
def _response_fast_timed(sensor, pose, rects, surf, n_grid=DEFAULT_N_GRID_FAST):
    """response_fast() と同じ処理を、区間ごとに計測しながら行う（診断専用の複製。
    本番コード mouse/ir_sensor.py には一切手を入れない）。

    2026-08-22: response_fast() 本体の変更（絞り込み判定のnumpy一括化・遮蔽候補直方体の
    AABB削減）に合わせて複製し直した。区間は cone（絞り込み・一括版）／grid（積分格子構築）／
    prune（遮蔽候補のAABB削減）／occ（遮蔽の一括判定）／integrate（面積分）。
    """
    t_cone = 0.0
    t_grid = 0.0
    t_prune = 0.0
    t_occ = 0.0
    t_integrate = 0.0

    led, pt = _sensor_world_geometry(sensor, pose)
    half_angle_max = max(sensor.led_half_angle_deg, sensor.pt_half_angle_deg)

    near_rects = []
    for r in rects:
        diag = math.hypot(r.hx, r.hy)
        dist_center = math.hypot(r.cx - led.pos[0], r.cy - led.pos[1])
        if dist_center - diag > DEFAULT_MAX_RANGE_M:
            continue
        near_rects.append(r)

    facets = []
    facet_owner = []
    for owner_idx, r in enumerate(near_rects):
        for f in _wall_facets([r], DEFAULT_WALL_HEIGHT_M):
            facets.append(f)
            facet_owner.append(owner_idx)

    all_boxes = _obstacle_boxes(near_rects, DEFAULT_WALL_HEIGHT_M)

    vis_facets = []
    vis_owner = []
    for facet, owner_idx in zip(facets, facet_owner):
        to_led = led.pos - facet.center
        if float(np.dot(to_led, facet.normal)) <= 0.0:
            continue
        vis_facets.append(facet)
        vis_owner.append(owner_idx)

    zero_breakdown = {"cone": 0.0, "grid": 0.0, "prune": 0.0, "occ": 0.0, "integrate": 0.0,
                       "n_before": 0, "n_after": 0,
                       "n_boxes_before": int(all_boxes.shape[0]), "n_boxes_after": 0}
    if not vis_facets:
        return 0.0, zero_breakdown

    n_before = len(vis_facets)

    t0 = time.perf_counter()
    stacked = _stack_facets(vis_facets)
    led_mask = _facets_maybe_in_cone_batch(stacked, led.pos, led.axis, DEFAULT_LED_CONE_MARGIN_DEG,
                                            DEFAULT_CONE_FILTER_N_U, DEFAULT_CONE_FILTER_N_V)
    pt_mask = _facets_maybe_in_cone_batch(stacked, pt.pos, pt.axis, DEFAULT_PT_CONE_MARGIN_DEG,
                                           DEFAULT_CONE_FILTER_N_U, DEFAULT_CONE_FILTER_N_V)
    keep_mask = led_mask & pt_mask
    vis_facets = [f for f, k in zip(vis_facets, keep_mask) if k]
    vis_owner = [o for o, k in zip(vis_owner, keep_mask) if k]
    t_cone += time.perf_counter() - t0
    n_after = len(vis_facets)

    if not vis_facets:
        return 0.0, {"cone": t_cone, "grid": 0.0, "prune": 0.0, "occ": 0.0, "integrate": 0.0,
                      "n_before": n_before, "n_after": 0,
                      "n_boxes_before": int(all_boxes.shape[0]), "n_boxes_after": 0}

    t0 = time.perf_counter()
    grids = [_facet_grid(f, led, n_grid, half_angle_max, sensor.separation_m) for f in vis_facets]
    points_list = [g[0] for g in grids]
    dA_list = [g[1] for g in grids]
    t_grid += time.perf_counter() - t0

    t0 = time.perf_counter()
    all_points = np.stack(points_list, axis=0)
    owner_arr = np.array(vis_owner, dtype=int)
    boxes = all_boxes
    if all_boxes.shape[0] > 0:
        pts_flat = all_points.reshape(-1, 3)
        combined = np.vstack([pts_flat, led.pos[None, :], pt.pos[None, :]])
        aabb_min = combined.min(axis=0)
        aabb_max = combined.max(axis=0)
        box_mins = all_boxes[:, (0, 2, 4)]
        box_maxs = all_boxes[:, (1, 3, 5)]
        keep_box = np.all((box_maxs >= aabb_min) & (box_mins <= aabb_max), axis=1)
        kept_idx = np.nonzero(keep_box)[0]
        boxes = all_boxes[kept_idx]
        remap = -np.ones(len(near_rects), dtype=int)
        remap[kept_idx] = np.arange(len(kept_idx))
        owner_arr = remap[owner_arr]
    t_prune += time.perf_counter() - t0

    t0 = time.perf_counter()
    if boxes.shape[0] > 0:
        occ_led = _segment_occluded(all_points, led.pos, boxes, owner_arr)
        occ_pt = _segment_occluded(all_points, pt.pos, boxes, owner_arr)
        occluded_all = occ_led | occ_pt
        occluded_list = [occluded_all[i] for i in range(len(vis_facets))]
    else:
        occluded_list = [None] * len(vis_facets)
    t_occ += time.perf_counter() - t0

    t0 = time.perf_counter()
    total = 0.0
    for facet, points, dA, occluded in zip(vis_facets, points_list, dA_list, occluded_list):
        total += _integrate_facet(facet, led, pt, surf, points, dA, 1.0, 1.0,
                                   DEFAULT_MAX_RANGE_M, occluded)
    t_integrate += time.perf_counter() - t0

    return total * sensor.gain, {
        "cone": t_cone, "grid": t_grid, "prune": t_prune, "occ": t_occ, "integrate": t_integrate,
        "n_before": n_before, "n_after": n_after,
        "n_boxes_before": int(all_boxes.shape[0]), "n_boxes_after": int(boxes.shape[0]),
    }


def stage_breakdown(specs, rects, poses) -> None:
    acc = {"cone": 0.0, "grid": 0.0, "prune": 0.0, "occ": 0.0, "integrate": 0.0}
    n_before_list = []
    n_after_list = []
    n_boxes_before_list = []
    n_boxes_after_list = []
    for pose_d in poses:
        sensor = specs[pose_d["sensor_idx"]]
        pose = (pose_d["x"], pose_d["y"], pose_d["theta"])
        _, br = _response_fast_timed(sensor, pose, rects, SURF)
        for k in acc:
            acc[k] += br[k]
        n_before_list.append(br["n_before"])
        n_after_list.append(br["n_after"])
        n_boxes_before_list.append(br["n_boxes_before"])
        n_boxes_after_list.append(br["n_boxes_after"])
    n = len(poses)
    out = {
        "n_poses": n,
        "mean_ms": {k: 1000.0 * v / n for k, v in acc.items()},
        "mean_n_facets_before": float(np.mean(n_before_list)),
        "mean_n_facets_after": float(np.mean(n_after_list)),
        "mean_n_boxes_before": float(np.mean(n_boxes_before_list)),
        "mean_n_boxes_after": float(np.mean(n_boxes_after_list)),
    }
    (OUT_DIR / "breakdown.json").write_text(json.dumps(out), encoding="utf-8")
    print("breakdown:", json.dumps(out, indent=2, ensure_ascii=False))


# ============================================================================
# ステージ: summary（事前登録の判定表を出す）
# ============================================================================
def _percentile95(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 95))


def stage_summary() -> None:
    b48 = json.loads((OUT_DIR / "baseline48.json").read_text(encoding="utf-8"))
    b64 = json.loads((OUT_DIR / "baseline64.json").read_text(encoding="utf-8"))
    v48 = np.array(b48["values"])
    v64 = np.array(b64["values"])
    n_random = len(v64)
    sigma_ref = _percentile95(np.abs(v48[:n_random] - v64)) / I_FULL
    print(f"sigma_ref（錨・無作為400姿勢） = {sigma_ref:.6f}")

    sc = json.loads((OUT_DIR / "selfcheck.json").read_text(encoding="utf-8"))
    sc_max = max(sc["diffs"]) / I_FULL
    print(f"自己検査: 絞り込みあり/なし の最大差(満量比) = {sc_max:.3e}"
          f"（判定: {'合格' if sc_max <= 1e-6 else '不合格'} — 基準1e-6以下）")

    sc2_path = OUT_DIR / "selfcheck2.json"
    if sc2_path.exists():
        sc2 = json.loads(sc2_path.read_text(encoding="utf-8"))
        sc2_max = max(sc2["diffs"]) / I_FULL
        print(f"自己検査2: 遮蔽候補AABB削減 前後の最大差(満量比) = {sc2_max:.3e}"
              f"（判定: {'合格' if sc2_max <= 1e-9 else '不合格'} — 基準1e-9以下）")

    sc3_path = OUT_DIR / "selfcheck3.json"
    if sc3_path.exists():
        sc3 = json.loads(sc3_path.read_text(encoding="utf-8"))
        print(f"自己検査3: 絞り込み判定 numpy一括版とループ版の不一致姿勢数 = "
              f"{sc3['n_mismatch_poses']}/{sc3['n_poses']}"
              f"（判定: {'合格' if sc3['n_mismatch_poses'] == 0 else '不合格'} — 0であること）")

    def m6_of(values_path: str, label: str) -> float:
        d = json.loads((OUT_DIR / values_path).read_text(encoding="utf-8"))
        v = np.array(d["values"])
        m6 = _percentile95(np.abs(v - v48)) / I_FULL
        print(f"{label}: M6 = {m6:.6f}")
        return m6

    m6_neg1 = m6_of("negctrl1.json", "否定対照1（光錐半分・追記1で不作動と判明。参考値）")
    neg1b_path = OUT_DIR / "negctrl1b.json"
    m6_neg1b = m6_of("negctrl1b.json", "否定対照1差し替え（追記1: 光軸が最初に当たる1枚だけ）") if neg1b_path.exists() else None
    m6_neg2 = m6_of("negctrl2.json", "否定対照2（n_grid=6）")
    m6_main = m6_of("main.json", "本測定")

    main = json.loads((OUT_DIR / "main.json").read_text(encoding="utf-8"))
    T = 1000.0 * float(np.mean(main["times"]))
    print(f"T（平均時間） = {T:.4f} ms")

    print()
    print("=== 事前登録 §4 の判定 ===")
    print(f"M6 = {m6_main:.6f}  ->  {'合格(<=0.01)' if m6_main <= 0.01 else ('惜しい(0.01-0.03)' if m6_main <= 0.03 else '不合格(>0.03)')}")
    print(f"T  = {T:.4f} ms -> {'合格(<=2ms)' if T <= 2.0 else ('目標未達(2-5ms)' if T <= 5.0 else '不合格(>5ms)')}")
    print()
    print("=== 事前登録 §5 の否定対照（M6>0.03 であるべき） ===")
    print(f"否定対照1（参考・追記1で不作動と判明）: M6={m6_neg1:.6f}")
    if m6_neg1b is not None:
        print(f"否定対照1差し替え（追記1・実際の検査）: M6={m6_neg1b:.6f} -> "
              f"{'想定どおり不合格帯' if m6_neg1b > 0.03 else '★不合格帯に入らず(検査が甘い)'}")
    print(f"否定対照2: M6={m6_neg2:.6f} -> {'想定どおり不合格帯' if m6_neg2 > 0.03 else '★不合格帯に入らず(検査が甘い)'}")


# ============================================================================
# main
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                     choices=["baseline48", "baseline64", "selfcheck", "selfcheck2", "selfcheck3",
                              "negctrl1", "negctrl1b", "negctrl2", "main", "breakdown", "summary"])
    args = ap.parse_args()

    if args.stage == "summary":
        stage_summary()
        return

    p, rects, W, H, cell = load_geometry()
    specs = build_ir_specs(p)
    poses_random = gen_random_poses(SEED, N_RANDOM, W, H, cell)
    poses_front = load_front_grid_poses(specs)
    poses_all = poses_random + poses_front

    if args.stage == "baseline48":
        stage_baseline48(specs, rects, poses_all)
    elif args.stage == "baseline64":
        stage_baseline64(specs, rects, poses_random)
    elif args.stage == "selfcheck":
        stage_selfcheck(specs, rects, poses_all)
    elif args.stage == "selfcheck2":
        stage_selfcheck2(specs, rects, poses_all)
    elif args.stage == "selfcheck3":
        stage_selfcheck3(specs, rects, poses_all)
    elif args.stage == "negctrl1":
        stage_negctrl1(specs, rects, poses_all)
    elif args.stage == "negctrl1b":
        stage_negctrl1b(specs, rects, poses_all)
    elif args.stage == "negctrl2":
        stage_negctrl2(specs, rects, poses_all)
    elif args.stage == "main":
        stage_main(specs, rects, poses_all)
    elif args.stage == "breakdown":
        stage_breakdown(specs, rects, poses_all)


if __name__ == "__main__":
    main()
