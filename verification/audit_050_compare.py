"""verification/audit_050_compare.py

`verification/AUDIT_050_PREREG_ir_raycast.md`（事前登録）§3〜§6 の測定を実行し、
一次記録（姿勢ごとの値）を `outputs/audit_050/*.json` に書き出す。

**このスクリプトは判定（合格・不合格の言い渡し）を行わない。** 事前登録の分割表に
照らして `M1/σ95` 等の数値を出すところまでを担う（`stage F`）。分割のどの帯に
入るかの記述は `verification/AUDIT_050_RESULT.md` 側で行う。

使い方（1 回の呼び出しは 10 分以内に収めること。範囲指定で分割して再開できる）:

    python verification/audit_050_compare.py --stage v0
    python verification/audit_050_compare.py --stage A --start 0 --end 200
    python verification/audit_050_compare.py --stage B --start 0 --end 200
    python verification/audit_050_compare.py --stage C --start 0 --end 200
    python verification/audit_050_compare.py --stage D --start 0 --end 200
    python verification/audit_050_compare.py --stage E --start 0 --end 70
    python verification/audit_050_compare.py --stage E --start 70 --end 140
    python verification/audit_050_compare.py --stage E --start 140 --end 200
    python verification/audit_050_compare.py --stage F

既に計算済みの姿勢（出力 JSON に載っている idx）は再計算せず飛ばす。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

from classic.geometry import Rect, wall_obstacles
from mouse.ir_sensor import IrSensorSpec, SurfaceSpec, response
from mouse.params import RobotParams
from verification.audit_050_raycast import Sensor, raycast_response, sensors_from_params

# ============================================================================
# 定数（事前登録 §2-2・追記2 で固定された値。変更禁止）
# ============================================================================
I_FULL = 0.8298934          # 満量（固定値。姿勢の標本に依存しない）
N_RAYS = 480_000            # 光線本数（追記2 で固定）
POSE_SEED = 20250821        # 姿勢の標本の乱数種（§2-1）
N_POSES = 200                # 姿勢の本数（§2-1）
SEED_A = 777001              # 光線追跡 1 回目の乱数種（σ95 の錨・M1 双方に使う）
SEED_B = 777002              # 光線追跡 2 回目の乱数種（σ95 専用）

WALL_HEIGHT_M = 0.05
FLOOR_HALFEXTENT_M = 0.20
MAX_RANGE_M = 0.35
DIFFUSE = 0.8

REPO_ROOT = Path(__file__).resolve().parent.parent
MAZE_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "maze_41001.npz"
OUT_DIR = REPO_ROOT / "outputs" / "audit_050"


# ============================================================================
# 迷路・姿勢の標本（PREREG §2-1。乱数の引き方の順番は変えない）
# ============================================================================
def load_geometry():
    """迷路の壁・柱の矩形列と `RobotParams` を返す。"""
    p = RobotParams()
    d = np.load(MAZE_PATH)
    rects = wall_obstacles(d["v_walls"], d["h_walls"], cell_size=p.cell_size)
    W = int(d["v_walls"].shape[0] - 1)
    H = int(d["v_walls"].shape[1])
    return p, rects, W, H, p.cell_size


def gen_poses() -> List[Tuple[int, Tuple[float, float, float]]]:
    """PREREG §2-1 の手順どおり、乱数の引き方の順番を変えずに 200 姿勢を作る。

    戻り値は `(sensor_idx, (x, y, theta))` の列（`sensor_idx` は
    `sensors_from_params()`/`RobotParams().sensors` の並び順 = LF,LS,RF,RS）。
    """
    _, _, W, H, cell = load_geometry()
    rng = np.random.default_rng(POSE_SEED)
    poses = []
    for _ in range(N_POSES):
        cx = rng.integers(0, W)
        cy = rng.integers(0, H)
        x = (cx + 0.5) * cell + rng.uniform(-0.04, 0.04)
        y = (cy + 0.5) * cell + rng.uniform(-0.04, 0.04)
        th = rng.uniform(-math.pi, math.pi)
        poses.append((int(rng.integers(0, 4)), (float(x), float(y), float(th))))
    return poses


def build_ir_specs(p: RobotParams) -> List[IrSensorSpec]:
    """面積分側のセンサ仕様（`mouse/params.py` から光線追跡側と同じ取付位置・光軸で作る）。

    既定値（離隔 0.0065・縦配置・半値角 5°・アライメント誤差 0・gain 1.0）を使う。
    """
    specs = []
    for s in p.sensors:
        pos = tuple(float(v) for v in s["pos"].split())
        axis = tuple(float(v) for v in s["zaxis"].split())
        specs.append(IrSensorSpec(name=s["name"], pos=pos, axis=axis))
    return specs


SURF = SurfaceSpec(diffuse=DIFFUSE, specular=0.0)


def _check_sensor_alignment(ray_sensors: Sequence[Sensor], ir_specs: Sequence[IrSensorSpec]) -> None:
    """光線追跡側 `Sensor` と面積分側 `IrSensorSpec` が同じ取付位置・光軸であることを確認する。"""
    assert len(ray_sensors) == len(ir_specs) == 4, "センサ本数が 4 本（LF/LS/RF/RS）でない"
    for rs, ir in zip(ray_sensors, ir_specs):
        assert rs.name == ir.name, f"センサの並び順が一致しない: {rs.name} vs {ir.name}"
        assert rs.pos == ir.pos, f"取付位置が一致しない（{rs.name}）: {rs.pos} vs {ir.pos}"
        assert rs.axis == ir.axis, f"光軸が一致しない（{rs.name}）: {rs.axis} vs {ir.axis}"


# ============================================================================
# JSON 入出力（再開可能な形。idx をキーに、既に計算済みの姿勢は飛ばす）
# ============================================================================
def _load_existing(path: Path) -> Dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"records": {}}


def _save(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp.replace(path)


# ============================================================================
# 段 A / B / D / E: 光線追跡
# ============================================================================
def run_raycast_stage(
    out_name: str,
    *,
    seed: int,
    max_bounces: int,
    led_half_angle_deg: float,
    pt_half_angle_deg: float,
    start: int,
    end: int,
    meta_extra: Optional[Dict] = None,
) -> None:
    p, rects, W, H, cell = load_geometry()
    ray_sensors = sensors_from_params(p)
    poses = gen_poses()
    end = min(end, len(poses))

    out_path = OUT_DIR / out_name
    data = _load_existing(out_path)
    data.setdefault("meta", {
        "stage": out_name,
        "n_rays": N_RAYS,
        "seed": seed,
        "max_bounces": max_bounces,
        "led_half_angle_deg": led_half_angle_deg,
        "pt_half_angle_deg": pt_half_angle_deg,
        "include_floor": True,
        "floor_halfextent_m": FLOOR_HALFEXTENT_M,
        "max_range_m": MAX_RANGE_M,
        "wall_height_m": WALL_HEIGHT_M,
        "diffuse": DIFFUSE,
        "pose_seed": POSE_SEED,
        "n_poses": N_POSES,
        "maze": str(MAZE_PATH.relative_to(REPO_ROOT)),
    })
    if meta_extra:
        data["meta"].update(meta_extra)
    records: Dict[str, Dict] = data["records"]

    print(f"[{out_name}] 範囲 {start}..{end}（既計算 {len(records)} 件）", flush=True)
    n_done_this_call = 0
    t_stage_start = time.time()
    for idx in range(start, end):
        key = str(idx)
        if key in records:
            continue
        sensor_idx, pose = poses[idx]
        sensor = ray_sensors[sensor_idx]
        t0 = time.time()
        value = raycast_response(
            sensor,
            pose,
            rects,
            n_rays=N_RAYS,
            seed=seed,
            max_bounces=max_bounces,
            include_floor=True,
            led_half_angle_deg=led_half_angle_deg,
            pt_half_angle_deg=pt_half_angle_deg,
            max_range_m=MAX_RANGE_M,
            wall_height_m=WALL_HEIGHT_M,
            floor_halfextent_m=FLOOR_HALFEXTENT_M,
            diffuse=DIFFUSE,
        )
        dt = time.time() - t0
        records[key] = {
            "idx": idx,
            "sensor_idx": sensor_idx,
            "sensor_name": sensor.name,
            "x": pose[0],
            "y": pose[1],
            "theta": pose[2],
            "value": value,
            "i_full_ratio": value / I_FULL,
            "elapsed_s": dt,
        }
        n_done_this_call += 1
        print(f"[{out_name}] 姿勢 {idx+1}/{N_POSES} (センサ={sensor.name}) {dt:.3f}s", flush=True)
        if n_done_this_call % 20 == 0:
            _save(out_path, data)
    _save(out_path, data)
    total_dt = time.time() - t_stage_start
    print(
        f"[{out_name}] この呼び出しで {n_done_this_call} 件計算・所要 {total_dt:.1f}s"
        f"（全体 {len(records)}/{N_POSES} 件完了）",
        flush=True,
    )


# ============================================================================
# 段 C: 面積分
# ============================================================================
def run_integration_stage(start: int, end: int) -> None:
    p, rects, W, H, cell = load_geometry()
    ir_specs = build_ir_specs(p)
    poses = gen_poses()
    end = min(end, len(poses))

    out_path = OUT_DIR / "integration.json"
    data = _load_existing(out_path)
    data.setdefault("meta", {
        "stage": "C",
        "include_floor": True,
        "floor_halfextent_m": FLOOR_HALFEXTENT_M,
        "max_range_m": MAX_RANGE_M,
        "wall_height_m": WALL_HEIGHT_M,
        "diffuse": DIFFUSE,
        "specular": 0.0,
        "occlusion": True,
        "pose_seed": POSE_SEED,
        "n_poses": N_POSES,
        "maze": str(MAZE_PATH.relative_to(REPO_ROOT)),
    })
    records: Dict[str, Dict] = data["records"]

    print(f"[C] 範囲 {start}..{end}（既計算 {len(records)} 件）", flush=True)
    n_done_this_call = 0
    t_stage_start = time.time()
    for idx in range(start, end):
        key = str(idx)
        if key in records:
            continue
        sensor_idx, pose = poses[idx]
        spec = ir_specs[sensor_idx]
        t0 = time.time()
        value = response(
            spec,
            pose,
            rects,
            SURF,
            wall_height_m=WALL_HEIGHT_M,
            include_floor=True,
            floor_halfextent_m=FLOOR_HALFEXTENT_M,
            max_range_m=MAX_RANGE_M,
            occlusion=True,
        )
        dt = time.time() - t0
        records[key] = {
            "idx": idx,
            "sensor_idx": sensor_idx,
            "sensor_name": spec.name,
            "x": pose[0],
            "y": pose[1],
            "theta": pose[2],
            "value": value,
            "i_full_ratio": value / I_FULL,
            "elapsed_s": dt,
        }
        n_done_this_call += 1
        print(f"[C] 姿勢 {idx+1}/{N_POSES} (センサ={spec.name}) {dt:.3f}s", flush=True)
        if n_done_this_call % 50 == 0:
            _save(out_path, data)
    _save(out_path, data)
    total_dt = time.time() - t_stage_start
    print(
        f"[C] この呼び出しで {n_done_this_call} 件計算・所要 {total_dt:.1f}s"
        f"（全体 {len(records)}/{N_POSES} 件完了）",
        flush=True,
    )


# ============================================================================
# 検証 0（PREREG §3）: 規格合わせ。単一パネル・床なし・遮蔽なし・正対。
# ============================================================================
def run_v0() -> None:
    """§2-2 と同じ単一パネルに対し、距離 20/44/84/150mm の 4 点で
    光線追跡と面積分が同じ満量比を返すことを確かめる（判定は報告側で行う。
    ここでは差だけを記録する）。"""
    half_len = 0.084   # 半長 84mm
    height = 0.05       # 高さ 50mm（迷路の壁高さと同じ）
    # 厚み12mmのパネルを x=0 に正対させる。中心は x=thickness/2, 半長は§2-2の記述どおり84mm。
    thickness = 0.012
    wall = Rect(cx=thickness / 2.0, cy=0.0, hx=thickness / 2.0, hy=half_len)

    p = RobotParams()
    ray_sensor = Sensor(name="V0", pos=(0.0, 0.0, 0.01), axis=(1.0, 0.0, 0.0))
    ir_spec = IrSensorSpec(name="V0", pos=(0.0, 0.0, 0.01), axis=(1.0, 0.0, 0.0))

    distances_mm = [20.0, 44.0, 84.0, 150.0]
    results = []
    for d_mm in distances_mm:
        d = d_mm / 1000.0
        # センサ基準点が壁面（x=0）から距離 d のところに正対して来るように pose を決める。
        pose = (-d, 0.0, 0.0)
        ray_val = raycast_response(
            ray_sensor, pose, [wall],
            n_rays=N_RAYS, seed=SEED_A, max_bounces=1,
            include_floor=False, led_half_angle_deg=5.0, pt_half_angle_deg=5.0,
            max_range_m=MAX_RANGE_M, wall_height_m=height, diffuse=DIFFUSE,
        )
        int_val = response(
            ir_spec, pose, [wall], SURF,
            wall_height_m=height, include_floor=False, max_range_m=MAX_RANGE_M,
            occlusion=False,
        )
        ray_ratio = ray_val / I_FULL
        int_ratio = int_val / I_FULL
        results.append({
            "distance_mm": d_mm,
            "ray_value": ray_val,
            "integration_value": int_val,
            "ray_i_full_ratio": ray_ratio,
            "integration_i_full_ratio": int_ratio,
            "diff_i_full_ratio": abs(ray_ratio - int_ratio),
        })
        print(
            f"[v0] d={d_mm}mm 光線追跡={ray_ratio:.6f} 面積分={int_ratio:.6f} "
            f"差={abs(ray_ratio-int_ratio):.6f}",
            flush=True,
        )

    out_path = OUT_DIR / "v0_calibration.json"
    data = {
        "meta": {
            "stage": "v0",
            "n_rays": N_RAYS,
            "seed": SEED_A,
            "half_length_m": half_len,
            "height_m": height,
            "thickness_m": thickness,
            "include_floor": False,
            "occlusion": False,
        },
        "results": results,
        "max_diff_i_full_ratio": max(r["diff_i_full_ratio"] for r in results),
    }
    _save(out_path, data)
    print(f"[v0] 最大差(満量比) = {data['max_diff_i_full_ratio']:.6f}（事前登録の判定基準は 0.01 以下）", flush=True)


# ============================================================================
# 段 F: 集計
# ============================================================================
def _load_records(name: str) -> Optional[Dict[str, Dict]]:
    path = OUT_DIR / name
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("records", {})


def _diffs_by_idx(a: Dict[str, Dict], b: Dict[str, Dict]) -> List[float]:
    """共通の idx について満量比の差の絶対値を返す（不足があれば None を混ぜない。
    欠けている idx は無視して、そのことを呼び出し側に別途報告させる）。"""
    common = sorted(set(a.keys()) & set(b.keys()), key=int)
    return [abs(a[k]["i_full_ratio"] - b[k]["i_full_ratio"]) for k in common], common


def run_summary() -> None:
    rec_a = _load_records("ray_s777001.json")
    rec_b = _load_records("ray_s777002.json")
    rec_c = _load_records("integration.json")
    rec_d = _load_records("ray_ha7.json")
    rec_e = _load_records("ray_b4.json")

    summary: Dict = {"n_poses_target": N_POSES}

    def pct95(xs: List[float]) -> Optional[float]:
        if not xs:
            return None
        return float(np.percentile(np.array(xs), 95))

    # σ95: A と B の差
    if rec_a is not None and rec_b is not None:
        diffs_ab, common_ab = _diffs_by_idx(rec_a, rec_b)
        summary["sigma95"] = pct95(diffs_ab)
        summary["sigma95_n"] = len(diffs_ab)
        summary["sigma95_complete"] = len(common_ab) == N_POSES and len(rec_a) == N_POSES and len(rec_b) == N_POSES
    else:
        summary["sigma95"] = None
        summary["sigma95_n"] = 0
        summary["sigma95_complete"] = False

    # M1: A と C の差
    if rec_a is not None and rec_c is not None:
        diffs_ac, common_ac = _diffs_by_idx(rec_a, rec_c)
        summary["M1"] = pct95(diffs_ac)
        summary["M1_n"] = len(diffs_ac)
        summary["M1_complete"] = len(common_ac) == N_POSES and len(rec_a) == N_POSES and len(rec_c) == N_POSES
        if summary["M1"] is not None and summary.get("sigma95"):
            summary["M1_over_sigma95"] = summary["M1"] / summary["sigma95"]
        # 差の大きい姿勢の上位5件
        rows = []
        for k in common_ac:
            rows.append({
                "idx": int(k),
                "sensor_name": rec_a[k]["sensor_name"],
                "x": rec_a[k]["x"], "y": rec_a[k]["y"], "theta": rec_a[k]["theta"],
                "ray_i_full_ratio": rec_a[k]["i_full_ratio"],
                "integration_i_full_ratio": rec_c[k]["i_full_ratio"],
                "diff_i_full_ratio": abs(rec_a[k]["i_full_ratio"] - rec_c[k]["i_full_ratio"]),
            })
        rows.sort(key=lambda r: -r["diff_i_full_ratio"])
        summary["M1_top5_diff_poses"] = rows[:5]
    else:
        summary["M1"] = None
        summary["M1_n"] = 0
        summary["M1_complete"] = False

    # 否定対照: D と C の差
    if rec_d is not None and rec_c is not None:
        diffs_dc, common_dc = _diffs_by_idx(rec_d, rec_c)
        summary["M1_negctrl"] = pct95(diffs_dc)
        summary["M1_negctrl_n"] = len(diffs_dc)
        summary["M1_negctrl_complete"] = len(common_dc) == N_POSES and len(rec_d) == N_POSES and len(rec_c) == N_POSES
        if summary["M1_negctrl"] is not None and summary.get("sigma95"):
            summary["M1_negctrl_over_sigma95"] = summary["M1_negctrl"] / summary["sigma95"]
    else:
        summary["M1_negctrl"] = None
        summary["M1_negctrl_n"] = 0
        summary["M1_negctrl_complete"] = False

    # M2: E と A の差
    if rec_e is not None and rec_a is not None:
        diffs_ea, common_ea = _diffs_by_idx(rec_e, rec_a)
        summary["M2"] = pct95(diffs_ea)
        summary["M2_n"] = len(diffs_ea)
        summary["M2_complete"] = len(common_ea) == N_POSES and len(rec_e) == N_POSES and len(rec_a) == N_POSES
        rows = []
        for k in common_ea:
            rows.append({
                "idx": int(k),
                "sensor_name": rec_a[k]["sensor_name"],
                "x": rec_a[k]["x"], "y": rec_a[k]["y"], "theta": rec_a[k]["theta"],
                "bounce1_i_full_ratio": rec_a[k]["i_full_ratio"],
                "bounce4_i_full_ratio": rec_e[k]["i_full_ratio"],
                "diff_i_full_ratio": abs(rec_e[k]["i_full_ratio"] - rec_a[k]["i_full_ratio"]),
            })
        rows.sort(key=lambda r: -r["diff_i_full_ratio"])
        summary["M2_top5_diff_poses"] = rows[:5]
    else:
        summary["M2"] = None
        summary["M2_n"] = 0
        summary["M2_complete"] = False

    out_path = OUT_DIR / "summary.json"
    _save(out_path, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


# ============================================================================
# エントリポイント
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["v0", "A", "B", "C", "D", "E", "F"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=N_POSES)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage != "F":
        p, rects, W, H, cell = load_geometry()
        _check_sensor_alignment(sensors_from_params(p), build_ir_specs(p))

    if args.stage == "v0":
        run_v0()
    elif args.stage == "A":
        run_raycast_stage(
            "ray_s777001.json", seed=SEED_A, max_bounces=1,
            led_half_angle_deg=5.0, pt_half_angle_deg=5.0,
            start=args.start, end=args.end,
        )
    elif args.stage == "B":
        run_raycast_stage(
            "ray_s777002.json", seed=SEED_B, max_bounces=1,
            led_half_angle_deg=5.0, pt_half_angle_deg=5.0,
            start=args.start, end=args.end,
        )
    elif args.stage == "C":
        run_integration_stage(args.start, args.end)
    elif args.stage == "D":
        run_raycast_stage(
            "ray_ha7.json", seed=SEED_A, max_bounces=1,
            led_half_angle_deg=7.0, pt_half_angle_deg=5.0,
            start=args.start, end=args.end,
        )
    elif args.stage == "E":
        run_raycast_stage(
            "ray_b4.json", seed=SEED_A, max_bounces=4,
            led_half_angle_deg=5.0, pt_half_angle_deg=5.0,
            start=args.start, end=args.end,
        )
    elif args.stage == "F":
        run_summary()


if __name__ == "__main__":
    main()
