"""verification/audit_056_interreflection.py

`verification/AUDIT_056_PREREG_interreflection.md` の測定を実行する。

使い方（1 回の呼び出しは 10 分以内に収める。前景で実行すること）:

    .venv/bin/python verification/audit_056_interreflection.py --stage verify0

現時点で実装されているのは検証0（事前登録 §3・0-a/0-b/0-c）と、参考測定として
1 姿勢あたりの計測時間（bounces=4）のみ。工程1（§4・光線追跡との突き合わせ）と
工程2（§5・格子上の比較。本題）は別作業（後続のセッションが実施）。`--stage` の
枝はここへ追加していく作りにしてある（`verify0` 以外は未実装で `NotImplementedError`）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import (
    Pose,
    Rect,
    _rect_clearance,
    _rect_polygon,
    robot_corners,
    wall_obstacles,
)
from mouse.ir_sensor import (
    DEFAULT_N_COARSE_FLOOR,
    DEFAULT_N_COARSE_WALL_U,
    DEFAULT_N_COARSE_WALL_V,
    DEFAULT_N_GRID_INTERREFLECTION_SOURCE,
    IrSensorSpec,
    SurfaceSpec,
    response,
)
from mouse.params import RobotParams
from verification.audit_050_raycast import Sensor, raycast_response, sensors_from_params

I_FULL = 0.8298934   # 満量（AUDIT_050 §2-2 と同じ固定値。事前登録 §2-1）
MAZE_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "maze_41001.npz"
OUT_DIR = REPO_ROOT / "outputs" / "audit_056"

# 光線追跡の光学パラメータ（更新後の既定値。raycast_response() 自身の既定引数は
# 旧仕様 5.0°/5.0°/0.0065m のままなので、呼ぶ側で毎回明示する。追記1・追記2）
RAY_LED_HALF_DEG = 3.0
RAY_PT_HALF_DEG = 6.0
RAY_SEPARATION_M = 0.0060


# ============================================================================
# 迷路・センサ・姿勢の標本（AUDIT_050 の作法を踏襲。検証0 の姿勢選びは事前登録が
# 指定していないので、AUDIT_050 §2-1 と同じ「区画中心±40mm・方位無作為・センサ無作為」
# の作法をそのまま使う）
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


def gen_poses(seed: int, n: int, W: int, H: int, cell: float) -> List[Tuple[int, Tuple[float, float, float]]]:
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        cx = rng.integers(0, W)
        cy = rng.integers(0, H)
        x = (cx + 0.5) * cell + rng.uniform(-0.04, 0.04)
        y = (cy + 0.5) * cell + rng.uniform(-0.04, 0.04)
        th = rng.uniform(-math.pi, math.pi)
        poses.append((int(rng.integers(0, 4)), (float(x), float(y), float(th))))
    return poses


SURF08 = SurfaceSpec(diffuse=0.8, specular=0.0)   # 0-b/0-c は鏡面なしで比べる（事前登録 §1・§4）


# ============================================================================
# 検証0-a: bounces=1 が「本作業前の response()」と厳密一致する
# ============================================================================
def _load_baseline_response():
    """git HEAD（本作業でのコミット前）の `mouse/ir_sensor.py::response` を
    独立にロードする（`bounces`/`return_breakdown` 引数を追加する前の実装そのもの）。
    """
    src = subprocess.run(
        ["git", "show", "HEAD:mouse/ir_sensor.py"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    tmp_dir = Path(tempfile.mkdtemp(prefix="audit056_baseline_"))
    tmp_path = tmp_dir / "ir_sensor_baseline.py"
    tmp_path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ir_sensor_baseline_056", tmp_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod   # dataclass の型解決（__future__ annotations）に必要
    spec.loader.exec_module(mod)
    return mod.response


def run_verify0_a(n_poses: int = 60, seed: int = 56001) -> Dict:
    p, rects, W, H, cell = load_geometry()
    specs = build_ir_specs(p)
    poses = gen_poses(seed, n_poses, W, H, cell)
    baseline_response = _load_baseline_response()

    max_reldiff = 0.0
    worst = None
    for sensor_idx, pose in poses:
        sensor = specs[sensor_idx]
        v_old = baseline_response(sensor, pose, rects, SURF08)
        v_new = response(sensor, pose, rects, SURF08, bounces=1)
        denom = abs(v_old) if abs(v_old) > 1e-300 else 1.0
        reldiff = abs(v_new - v_old) / denom
        if reldiff >= max_reldiff:
            max_reldiff = reldiff
            worst = (sensor.name, pose, v_old, v_new)
    return {
        "n_checked": len(poses), "max_reldiff": max_reldiff,
        "pass": max_reldiff <= 1e-12, "worst": worst,
    }


# ============================================================================
# 検証0-b: 面素の分割数を 1.5 倍にしても増分がほぼ変わらない（満量比 0.002 以下）
# ============================================================================
def run_verify0_b(n_poses: int = 16, seed: int = 56002) -> Dict:
    p, rects, W, H, cell = load_geometry()
    specs = build_ir_specs(p)
    poses = gen_poses(seed, n_poses, W, H, cell)

    base = (DEFAULT_N_COARSE_WALL_U, DEFAULT_N_COARSE_WALL_V, DEFAULT_N_COARSE_FLOOR)
    scaled = tuple(max(1, int(round(v * 1.5))) for v in base)

    max_delta = 0.0
    worst = None
    per_pose = []
    for sensor_idx, pose in poses:
        sensor = specs[sensor_idx]
        _total_base, bd_base = response(
            sensor, pose, rects, SURF08, bounces=4, return_breakdown=True,
            n_coarse_wall_u=base[0], n_coarse_wall_v=base[1], n_coarse_floor=base[2],
        )
        _total_scaled, bd_scaled = response(
            sensor, pose, rects, SURF08, bounces=4, return_breakdown=True,
            n_coarse_wall_u=scaled[0], n_coarse_wall_v=scaled[1], n_coarse_floor=scaled[2],
        )
        inc_base = sum(bd_base.values())
        inc_scaled = sum(bd_scaled.values())
        delta = abs(inc_scaled - inc_base) / I_FULL
        per_pose.append((sensor.name, pose, inc_base, inc_scaled, delta))
        if delta >= max_delta:
            max_delta = delta
            worst = (sensor.name, pose, inc_base, inc_scaled, delta)
    return {
        "n_checked": len(poses), "base_grid": base, "scaled_grid": scaled,
        "max_delta": max_delta, "pass": max_delta <= 0.002, "worst": worst,
        "per_pose": per_pose,
    }


# ============================================================================
# 検証0-c（是正版・追記2）: 「壁1枚＋床」と「向かい合う壁2枚＋床」で、
# モデル I の増分（response(...,bounces=4) のラジオシティ近似）が光線追跡の増分と
# 一致するかを見る。どちらの配置も反射2回目以降が0にならない（面の偶奇を壊す）。
#
# 半値角は更新後の LED 3.0°／PT 6.0°、離隔は更新後の 0.0060m を明示的に使う
# （`raycast_response()` 自身の既定値は旧仕様 5.0°/5.0°/0.0065m のままなので、
#  ここで明示しないと検証にならない）。鏡面成分は §1 の規約どおり 0（response 側
#  も SURF08=diffuse0.8/specular0.0 を使う。光線追跡に鏡面成分が無いため）。
# ============================================================================
PROBE_SENSOR_IR = IrSensorSpec(
    name="probe", pos=(0.0, 0.0, 0.010), axis=(1.0, 0.0, 0.0),
    separation_m=0.0060, led_half_angle_deg=3.0, pt_half_angle_deg=6.0,
)
PROBE_SENSOR_RAY = Sensor(name="probe", pos=(0.0, 0.0, 0.010), axis=(1.0, 0.0, 0.0))
N_RAYS_0C = 120_000    # audit_050_bounce_parity.py と同じ（既に妥当性確認済みの本数）
SEED_0C = 777001
DISTANCES_MM_0C = (20, 44, 84, 150)


def _wall_panel(d_m: float) -> Rect:
    """センサ（原点・+x 向き）から距離 `d_m` に正対する厚み12mm・半長84mmのパネル。"""
    return Rect(cx=d_m + 0.006, cy=0.0, hx=0.006, hy=0.084)


def _config_wall_floor(d_m: float) -> List[Rect]:
    """配置1「壁1枚＋床」: センサ前方 +x に壁1枚（床は response()/raycast_response() 側の
    include_floor=True で別途持つので、ここでは壁だけを返す）。"""
    return [_wall_panel(d_m)]


def _config_facing_walls_floor(d_m: float) -> List[Rect]:
    """配置2「向かい合う壁2枚＋床」: センサを挟んで +x と -x に壁（センサの光軸上に
    向かい合う2枚。互いの面が向き合う）。床は include_floor=True で別途持つ。"""
    wall_front = _wall_panel(d_m)
    wall_back = Rect(cx=-(d_m + 0.006), cy=0.0, hx=0.006, hy=0.084)
    return [wall_front, wall_back]


VERIFY0C_CONFIGS = (
    ("wall_floor", _config_wall_floor),
    ("facing_walls_floor", _config_facing_walls_floor),
)


def run_verify0_c() -> Dict:
    rows = []
    max_diff = 0.0
    worst = None
    for config_name, config_fn in VERIFY0C_CONFIGS:
        for d_mm in DISTANCES_MM_0C:
            d_m = d_mm / 1000.0
            walls = config_fn(d_m)

            _total, bd = response(
                PROBE_SENSOR_IR, (0.0, 0.0, 0.0), walls, SURF08, bounces=4, return_breakdown=True,
                include_floor=True,
            )
            inc_model = sum(bd.values())

            v1_ray = raycast_response(
                PROBE_SENSOR_RAY, (0.0, 0.0, 0.0), walls, n_rays=N_RAYS_0C, seed=SEED_0C,
                max_bounces=1, include_floor=True, diffuse=0.8,
                led_half_angle_deg=RAY_LED_HALF_DEG, pt_half_angle_deg=RAY_PT_HALF_DEG,
                separation_m=RAY_SEPARATION_M,
            )
            v4_ray = raycast_response(
                PROBE_SENSOR_RAY, (0.0, 0.0, 0.0), walls, n_rays=N_RAYS_0C, seed=SEED_0C,
                max_bounces=4, include_floor=True, diffuse=0.8,
                led_half_angle_deg=RAY_LED_HALF_DEG, pt_half_angle_deg=RAY_PT_HALF_DEG,
                separation_m=RAY_SEPARATION_M,
            )
            inc_ray = v4_ray - v1_ray

            diff = abs(inc_model - inc_ray) / I_FULL
            rows.append((config_name, d_mm, inc_model, inc_ray, diff))
            if diff >= max_diff:
                max_diff = diff
                worst = (config_name, d_mm, inc_model, inc_ray, diff)
    return {"rows": rows, "max_diff": max_diff, "pass": max_diff <= 0.01, "worst": worst}


# ============================================================================
# 参考測定: 1 姿勢 1 センサあたりの計測時間（bounces=4・実迷路の姿勢）
# ============================================================================
def run_timing(n_poses: int = 20, seed: int = 56003) -> Dict:
    p, rects, W, H, cell = load_geometry()
    specs = build_ir_specs(p)
    poses = gen_poses(seed, n_poses, W, H, cell)

    times = []
    for sensor_idx, pose in poses:
        sensor = specs[sensor_idx]
        t0 = time.perf_counter()
        response(sensor, pose, rects, SURF08, bounces=4)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    return {
        "n": len(times), "mean_s": float(times.mean()), "max_s": float(times.max()),
        "p95_s": float(np.percentile(times, 95)),
    }


# ============================================================================
# 共通: JSON 入出力（再開可能な形。idx をキーに、既に計算済みの姿勢は飛ばす）
# ============================================================================
def _load_existing(path: Path) -> Dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"records": {}}


def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _save(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp.replace(path)


def _pct95(xs) -> Optional[float]:
    if not xs:
        return None
    return float(np.percentile(np.array(xs, dtype=float), 95))


def _diffs_by_idx(a: Dict[str, Dict], b: Dict[str, Dict]):
    """共通の idx について満量比の差の絶対値を返す（`audit_050_compare.py` と同じ作法）。"""
    common = sorted(set(a.keys()) & set(b.keys()), key=int)
    return [abs(a[k]["i_full_ratio"] - b[k]["i_full_ratio"]) for k in common], common


# ============================================================================
# 工程1'（事前登録 追記2）: 面積分と反射1回の光線追跡を、AUDIT_050 §2-1 と同じ手順の
# 200姿勢で突き合わせる。半値角は更新後の LED3.0°/PT6.0°・離隔0.0060m、鏡面なし。
# ============================================================================
STAGE1_POSE_SEED = 20250821     # AUDIT_050 §2-1 と同じ乱数種
STAGE1_N_POSES = 200
STAGE1_N_RAYS = 480_000         # AUDIT_050 追記2 で固定された本数をそのまま使う（錨の性質上の要求）
STAGE1_SEED_A = 777001
STAGE1_SEED_B = 777002
STAGE1_NEGCTRL_LED_HALF_DEG = 7.0   # AUDIT_050 §6 と同じ否定対照


def stage1_poses():
    p, rects, W, H, cell = load_geometry()
    poses = gen_poses(STAGE1_POSE_SEED, STAGE1_N_POSES, W, H, cell)
    return p, rects, poses


def run_stage1_raycast(out_name: str, *, seed: int, led_half_angle_deg: float, start: int, end: int) -> None:
    p, rects, poses = stage1_poses()
    ray_sensors = sensors_from_params(p)
    end = min(end, len(poses))

    out_path = OUT_DIR / out_name
    data = _load_existing(out_path)
    data.setdefault("meta", {
        "stage": out_name, "n_rays": STAGE1_N_RAYS, "seed": seed,
        "max_bounces": 1, "led_half_angle_deg": led_half_angle_deg,
        "pt_half_angle_deg": RAY_PT_HALF_DEG, "separation_m": RAY_SEPARATION_M,
        "include_floor": True, "diffuse": 0.8,
        "pose_seed": STAGE1_POSE_SEED, "n_poses": STAGE1_N_POSES,
        "maze": str(MAZE_PATH.relative_to(REPO_ROOT)),
    })
    records: Dict[str, Dict] = data["records"]

    print(f"[{out_name}] 範囲 {start}..{end}（既計算 {len(records)} 件）", flush=True)
    n_done = 0
    t_start = time.time()
    for idx in range(start, end):
        key = str(idx)
        if key in records:
            continue
        sensor_idx, pose = poses[idx]
        sensor = ray_sensors[sensor_idx]
        t0 = time.time()
        value = raycast_response(
            sensor, pose, rects,
            n_rays=STAGE1_N_RAYS, seed=seed, max_bounces=1, include_floor=True,
            led_half_angle_deg=led_half_angle_deg, pt_half_angle_deg=RAY_PT_HALF_DEG,
            separation_m=RAY_SEPARATION_M, diffuse=0.8,
        )
        dt = time.time() - t0
        records[key] = {
            "idx": idx, "sensor_idx": sensor_idx, "sensor_name": sensor.name,
            "x": pose[0], "y": pose[1], "theta": pose[2],
            "value": value, "i_full_ratio": value / I_FULL, "elapsed_s": dt,
        }
        n_done += 1
        if n_done % 20 == 0:
            _save(out_path, data)
            print(f"[{out_name}] {idx+1}/{STAGE1_N_POSES} 完了 ({dt:.2f}s/件)", flush=True)
    _save(out_path, data)
    print(f"[{out_name}] この呼び出しで {n_done} 件・所要 {time.time()-t_start:.1f}s"
          f"（全体 {len(records)}/{STAGE1_N_POSES} 件）", flush=True)


def run_stage1_integration() -> None:
    """面積分（モデルS相当・bounces=1・鏡面なし SURF08）を200姿勢ぶん一括計算する。"""
    p, rects, poses = stage1_poses()
    ir_specs = build_ir_specs(p)   # IrSensorSpec 既定値 = 更新後の 3.0°/6.0°/0.0060m

    out_path = OUT_DIR / "stage1_integration.json"
    data = _load_existing(out_path)
    data.setdefault("meta", {
        "stage": "stage1_integration", "bounces": 1, "specular": 0.0, "diffuse": 0.8,
        "pose_seed": STAGE1_POSE_SEED, "n_poses": STAGE1_N_POSES,
    })
    records: Dict[str, Dict] = data["records"]
    t0 = time.time()
    for idx, (sensor_idx, pose) in enumerate(poses):
        key = str(idx)
        if key in records:
            continue
        spec = ir_specs[sensor_idx]
        value = response(spec, pose, rects, SURF08, bounces=1)
        records[key] = {
            "idx": idx, "sensor_idx": sensor_idx, "sensor_name": spec.name,
            "x": pose[0], "y": pose[1], "theta": pose[2],
            "value": value, "i_full_ratio": value / I_FULL,
        }
    _save(out_path, data)
    print(f"[stage1_integration] {len(records)}/{STAGE1_N_POSES} 件・所要 {time.time()-t0:.1f}s", flush=True)


def run_stage1_summary() -> Dict:
    def _rec(name):
        p = OUT_DIR / name
        return _load_json(p)["records"] if p.exists() else None

    rec_a = _rec("stage1_ray_s777001.json")
    rec_b = _rec("stage1_ray_s777002.json")
    rec_c = _rec("stage1_integration.json")
    rec_neg = _rec("stage1_negctrl.json")

    summary: Dict = {"n_poses_target": STAGE1_N_POSES}

    if rec_a is not None and rec_b is not None:
        diffs_ab, common_ab = _diffs_by_idx(rec_a, rec_b)
        summary["sigma95"] = _pct95(diffs_ab)
        summary["sigma95_n"] = len(diffs_ab)
        summary["sigma95_complete"] = (
            len(common_ab) == STAGE1_N_POSES and len(rec_a) == STAGE1_N_POSES and len(rec_b) == STAGE1_N_POSES
        )
    else:
        summary["sigma95"] = None
        summary["sigma95_complete"] = False

    if rec_a is not None and rec_c is not None:
        diffs_ac, common_ac = _diffs_by_idx(rec_a, rec_c)
        summary["M1"] = _pct95(diffs_ac)
        summary["M1_n"] = len(diffs_ac)
        summary["M1_complete"] = (
            len(common_ac) == STAGE1_N_POSES and len(rec_a) == STAGE1_N_POSES and len(rec_c) == STAGE1_N_POSES
        )
        if summary["M1"] is not None and summary.get("sigma95"):
            summary["M1_over_sigma95"] = summary["M1"] / summary["sigma95"]
        rows = []
        for k in common_ac:
            rows.append({
                "idx": int(k), "sensor_name": rec_a[k]["sensor_name"],
                "x": rec_a[k]["x"], "y": rec_a[k]["y"], "theta": rec_a[k]["theta"],
                "ray_i_full_ratio": rec_a[k]["i_full_ratio"],
                "integration_i_full_ratio": rec_c[k]["i_full_ratio"],
                "diff_i_full_ratio": abs(rec_a[k]["i_full_ratio"] - rec_c[k]["i_full_ratio"]),
            })
        rows.sort(key=lambda r: -r["diff_i_full_ratio"])
        summary["M1_top5_diff_poses"] = rows[:5]
    else:
        summary["M1"] = None
        summary["M1_complete"] = False

    if rec_neg is not None and rec_c is not None:
        diffs_nc, common_nc = _diffs_by_idx(rec_neg, rec_c)
        summary["M1_negctrl"] = _pct95(diffs_nc)
        summary["M1_negctrl_complete"] = (
            len(common_nc) == STAGE1_N_POSES and len(rec_neg) == STAGE1_N_POSES and len(rec_c) == STAGE1_N_POSES
        )
        if summary["M1_negctrl"] is not None and summary.get("sigma95"):
            summary["M1_negctrl_over_sigma95"] = summary["M1_negctrl"] / summary["sigma95"]
    else:
        summary["M1_negctrl"] = None
        summary["M1_negctrl_complete"] = False

    _save(OUT_DIR / "stage1_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


# ============================================================================
# 工程2（事前登録 §5・追記2）: 格子上でモデル S とモデル I(=S+Δ) の差 M5 を測る。
# ============================================================================
DX_MM = (-40, 0, 40)
DY_MM_LIST = tuple(range(-40, 41, 10))     # -40..40 の 10mm 刻み・9点
DTHETA_DEG = (-20, -10, 0, 10, 20)
STAGE2_N_RAYS = 15_000
STAGE2_SEED = 777001
STAGE2_NEGCTRL_LED_HALF_DEG = 7.0

CHASSIS_HALF_WIDTH_M = 0.030    # シャーシ全幅60mm（事前登録の指定そのもの）
CHASSIS_HALF_LENGTH_M = 0.050   # RobotParams().chassis_length/2（全長は事前登録が明示していないが、
                                 # 全幅だけでは前後端の重なりを見落とすため、既定の機体全長を併用する。
                                 # 除外判定を甘くする側ではなく厳しくする側の補完なので、判定を歪めない）


def _situation_cells(v_walls: np.ndarray, h_walls: np.ndarray, W: int, H: int) -> Dict[str, Tuple[int, int]]:
    """事前登録 §2-2 のとおり、左下から行優先（cy 昇順の外側ループ・cx 昇順の内側ループ）で
    走査し、状況 A/B/C それぞれ最初に当たった区画を返す（機体の向きを北＝+y に取ったときの相対）。
    """
    found: Dict[str, Tuple[int, int]] = {}
    for cy in range(H):
        for cx in range(W):
            left = bool(v_walls[cx, cy])
            right = bool(v_walls[cx + 1, cy])
            front = bool(h_walls[cx, cy + 1])
            if left and right and not front and "A" not in found:
                found["A"] = (cx, cy)
            if left and right and front and "B" not in found:
                found["B"] = (cx, cy)
            if left and not right and not front and "C" not in found:
                found["C"] = (cx, cy)
    return found


def _situation_pose(cx: int, cy: int, cell: float, dx_mm: float, dy_mm: float, dtheta_deg: float):
    """区画中心からの姿勢を作る。**符号の取り決め**（事前登録は符号を定義していないので
    ここで固定する）: 機体の向きが北（+y）のとき、dx は北を正（前後）、dy は東（+x）を正
    （横＝機体が北を向いたときの右手側）。dx・dy は「北基準の世界座標系」でのオフセットで
    あり、dθ による回転はかけない（実走の位置ずれをそのまま世界座標のずれとして表すため）。
    """
    ccx = (cx + 0.5) * cell
    ccy = (cy + 0.5) * cell
    x = ccx + dy_mm / 1000.0
    y = ccy + dx_mm / 1000.0
    theta = math.pi / 2.0 + math.radians(dtheta_deg)
    return x, y, theta


def _chassis_clearance(x: float, y: float, theta: float, obstacles) -> float:
    """機体（半幅30mm・半長50mm）と壁・柱群との最短符号付き距離 [m]（負なら重なる）。"""
    pose = Pose(x, y, theta)
    robot_poly = robot_corners(pose, half_width=CHASSIS_HALF_WIDTH_M, half_length=CHASSIS_HALF_LENGTH_M)
    c, s = math.cos(theta), math.sin(theta)
    axes = ((c, s), (-s, c))
    best = math.inf
    for obs in obstacles:
        d = _rect_clearance(robot_poly, axes, _rect_polygon(obs))
        if d < best:
            best = d
            if best < 0.0:
                # 一度でも負が見つかれば「除外」の判定には十分（最短値そのものは以後不要）。
                # ただし記録用にもう少し悪化しうるので、そのまま探索は続けて最小値を返す。
                pass
    return best


def build_stage2_grid() -> Dict:
    p, rects, W, H, cell = load_geometry()
    d = np.load(MAZE_PATH)
    v_walls, h_walls = d["v_walls"], d["h_walls"]
    situations = _situation_cells(v_walls, h_walls, W, H)
    missing = {"A", "B", "C"} - set(situations.keys())
    if missing:
        raise RuntimeError(f"状況 {missing} が maze_41001 内に見つからない（事前登録の前提が崩れる）")

    ir_specs = build_ir_specs(p)
    sensor_names = [s.name for s in ir_specs]

    points: List[Dict] = []
    n_pose_total = 0
    n_pose_excluded = 0
    idx = 0
    for sit in ("A", "B", "C"):
        cx, cy = situations[sit]
        for dx_mm in DX_MM:
            for dy_mm in DY_MM_LIST:
                for dtheta_deg in DTHETA_DEG:
                    x, y, theta = _situation_pose(cx, cy, cell, dx_mm, dy_mm, dtheta_deg)
                    n_pose_total += 1
                    clr = _chassis_clearance(x, y, theta, rects)
                    if clr < 0.0:
                        n_pose_excluded += 1
                        continue
                    for sensor_idx, name in enumerate(sensor_names):
                        points.append({
                            "idx": idx, "situation": sit, "cell_cx": cx, "cell_cy": cy,
                            "dx_mm": dx_mm, "dy_mm": dy_mm, "dtheta_deg": dtheta_deg,
                            "sensor_idx": sensor_idx, "sensor_name": name,
                            "x": x, "y": y, "theta": theta,
                        })
                        idx += 1

    meta = {
        "n_pose_total": n_pose_total,
        "n_pose_excluded": n_pose_excluded,
        "n_pose_included": n_pose_total - n_pose_excluded,
        "n_points": len(points),
        "situations": {k: list(v) for k, v in situations.items()},
        "chassis_half_width_m": CHASSIS_HALF_WIDTH_M,
        "chassis_half_length_m": CHASSIS_HALF_LENGTH_M,
        "dx_mm": list(DX_MM), "dy_mm_list": list(DY_MM_LIST), "dtheta_deg": list(DTHETA_DEG),
        "pose_convention": (
            "x=区画中心x + dy[mm->m]（東+）, y=区画中心y + dx[mm->m]（北+）, "
            "theta=pi/2 + dtheta[deg->rad]（dx/dyはdθで回転させない世界座標オフセット）"
        ),
        "maze": str(MAZE_PATH.relative_to(REPO_ROOT)),
    }
    out = {"meta": meta, "points": points}
    _save(OUT_DIR / "stage2_grid.json", out)
    print(f"[stage2_grid] 総姿勢 {n_pose_total}（うち除外 {n_pose_excluded}）"
          f"-> 採用 {meta['n_pose_included']} 姿勢 x 4センサ = {len(points)} 点", flush=True)
    return out


def _stage2_load_grid() -> Dict:
    return _load_json(OUT_DIR / "stage2_grid.json")


def run_stage2_raycast(bounces: int, start: int, end: int) -> None:
    grid = _stage2_load_grid()
    points = grid["points"]
    p, rects, W, H, cell = load_geometry()
    ray_sensors = sensors_from_params(p)
    end = min(end, len(points))

    out_name = f"stage2_ray_b{bounces}.json"
    out_path = OUT_DIR / out_name
    data = _load_existing(out_path)
    data.setdefault("meta", {
        "stage": out_name, "n_rays": STAGE2_N_RAYS, "seed": STAGE2_SEED, "max_bounces": bounces,
        "led_half_angle_deg": RAY_LED_HALF_DEG, "pt_half_angle_deg": RAY_PT_HALF_DEG,
        "separation_m": RAY_SEPARATION_M, "diffuse": 0.8, "include_floor": True,
        "n_points_target": len(points),
    })
    records: Dict[str, Dict] = data["records"]

    print(f"[{out_name}] 範囲 {start}..{end}（既計算 {len(records)} 件 / 全 {len(points)} 点）", flush=True)
    n_done = 0
    t_start = time.time()
    for i in range(start, end):
        pt = points[i]
        key = str(pt["idx"])
        if key in records:
            continue
        sensor = ray_sensors[pt["sensor_idx"]]
        pose = (pt["x"], pt["y"], pt["theta"])
        t0 = time.time()
        value = raycast_response(
            sensor, pose, rects,
            n_rays=STAGE2_N_RAYS, seed=STAGE2_SEED, max_bounces=bounces, include_floor=True,
            led_half_angle_deg=RAY_LED_HALF_DEG, pt_half_angle_deg=RAY_PT_HALF_DEG,
            separation_m=RAY_SEPARATION_M, diffuse=0.8,
        )
        dt = time.time() - t0
        records[key] = {"idx": pt["idx"], "value": value, "i_full_ratio": value / I_FULL, "elapsed_s": dt}
        n_done += 1
        if n_done % 100 == 0:
            _save(out_path, data)
            print(f"[{out_name}] {i+1}/{end} (この呼び出し内 {n_done} 件) {dt:.3f}s/件", flush=True)
    _save(out_path, data)
    print(f"[{out_name}] この呼び出しで {n_done} 件・所要 {time.time()-t_start:.1f}s"
          f"（全体 {len(records)}/{len(points)} 件）", flush=True)


def run_stage2_response_s(led_half_angle_deg: Optional[float] = None) -> None:
    """モデルS（面積分・bounces=1・既定 SurfaceSpec=diffuse0.8/specular0.10/shininess40）を
    格子の全点について一括計算する。`led_half_angle_deg` を指定すると §5-3 否定対照用
    （半値角だけ変えた版）になる。"""
    grid = _stage2_load_grid()
    points = grid["points"]
    p, rects, W, H, cell = load_geometry()
    ir_specs_base = build_ir_specs(p)

    if led_half_angle_deg is None:
        out_name = "stage2_response_s.json"
        ir_specs = ir_specs_base
    else:
        out_name = "stage2_response_s_negctrl.json"
        ir_specs = [
            IrSensorSpec(
                name=s.name, pos=s.pos, axis=s.axis,
                separation_m=s.separation_m, led_half_angle_deg=led_half_angle_deg,
                pt_half_angle_deg=s.pt_half_angle_deg,
            )
            for s in ir_specs_base
        ]

    out_path = OUT_DIR / out_name
    data = _load_existing(out_path)
    data.setdefault("meta", {
        "stage": out_name, "bounces": 1,
        "led_half_angle_deg": led_half_angle_deg if led_half_angle_deg is not None else IrSensorSpec.__dataclass_fields__["led_half_angle_deg"].default,
        "n_points_target": len(points),
    })
    records: Dict[str, Dict] = data["records"]
    t0 = time.time()
    for pt in points:
        key = str(pt["idx"])
        if key in records:
            continue
        spec = ir_specs[pt["sensor_idx"]]
        pose = (pt["x"], pt["y"], pt["theta"])
        value = response(spec, pose, rects, SURF, bounces=1)
        records[key] = {"idx": pt["idx"], "value": value, "i_full_ratio": value / I_FULL}
    _save(out_path, data)
    print(f"[{out_name}] {len(records)}/{len(points)} 件・所要 {time.time()-t0:.1f}s", flush=True)


SURF = SurfaceSpec()   # モデルS の既定反射面（diffuse=0.8, specular=0.10, shininess=40。ir_sensor.py既定値）


def run_stage2_summary() -> Dict:
    grid = _stage2_load_grid()
    points_by_idx = {p["idx"]: p for p in grid["points"]}

    b1 = _load_json(OUT_DIR / "stage2_ray_b1.json")["records"]
    b4 = _load_json(OUT_DIR / "stage2_ray_b4.json")["records"]
    s_recs = _load_json(OUT_DIR / "stage2_response_s.json")["records"] if (OUT_DIR / "stage2_response_s.json").exists() else {}

    common = sorted(set(b1.keys()) & set(b4.keys()), key=int)
    complete = len(common) == len(grid["points"]) == len(b1) == len(b4)

    delta_rows = []
    for k in common:
        pt = points_by_idx[int(k)]
        delta = b4[k]["i_full_ratio"] - b1[k]["i_full_ratio"]   # 符号を保持（相互反射の実測増分）
        row = dict(pt)
        row["delta_i_full_ratio"] = delta
        if k in s_recs:
            s_val = s_recs[k]["i_full_ratio"]
            row["s_i_full_ratio"] = s_val
            row["rel_diff"] = delta / s_val if abs(s_val) > 1e-12 else None
        delta_rows.append(row)

    abs_deltas = [abs(r["delta_i_full_ratio"]) for r in delta_rows]
    M5 = _pct95(abs_deltas)

    rel_vals = [r["rel_diff"] for r in delta_rows if r.get("rel_diff") is not None]
    rel_abs = [abs(v) for v in rel_vals]

    top20 = sorted(delta_rows, key=lambda r: -abs(r["delta_i_full_ratio"]))[:20]

    # 差の地図: 状況/センサ/dx ごとに (dy,dtheta) -> delta（符号つき）
    diff_map: Dict[str, Dict] = {}
    for r in delta_rows:
        sit = r["situation"]; sensor = r["sensor_name"]; dx = r["dx_mm"]
        key = f"{sit}|{sensor}|dx={dx}"
        diff_map.setdefault(key, []).append({
            "dy_mm": r["dy_mm"], "dtheta_deg": r["dtheta_deg"], "delta_i_full_ratio": r["delta_i_full_ratio"],
        })

    summary = {
        "n_points_total": len(grid["points"]),
        "n_points_compared": len(common),
        "complete": complete,
        "M5": M5,
        "M5_pass_band": (
            "帯1(<=0.01)" if (M5 is not None and M5 <= 0.01) else
            "帯2(0.01<..<=0.05)" if (M5 is not None and M5 <= 0.05) else
            "帯3(>0.05)" if M5 is not None else "不明"
        ),
        "rel_diff_median_abs": float(np.median(rel_abs)) if rel_abs else None,
        "rel_diff_p95_abs": _pct95(rel_abs),
        "rel_diff_n": len(rel_vals),
        "top20": [
            {k: v for k, v in r.items() if k not in ("x", "y", "theta")} for r in top20
        ],
    }
    _save(OUT_DIR / "stage2_summary.json", {"summary": summary, "diff_map": diff_map})
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return {"summary": summary, "diff_map": diff_map, "delta_rows": delta_rows}


# ----------------------------------------------------------------------------
# 工程2 副次A: ρ 走査（1620点から等間隔で200点を抜く。反射回数ごとの増分が ρ^k に
# 比例する性質を使い、bounces=2,3 だけ追加で測って ρ∈{0.5,0.65,0.8} の M5 を出す）
# ----------------------------------------------------------------------------
STAGE2_RHO_TARGET_N = 200


def stage2_rho_subsample_indices() -> Tuple[List[int], int]:
    """1620点（実際には除外後のn_points）の通し番号 idx を、等間隔（通し番号順）で
    間引いて約200点を選ぶ。再現できるよう、選び方は `range(0, n, step)` で固定する。"""
    grid = _stage2_load_grid()
    n = grid["meta"]["n_points"]
    step = max(1, round(n / STAGE2_RHO_TARGET_N))
    idxs = list(range(0, n, step))
    return idxs, step


def run_stage2_rho_raycast(bounces: int, start: int, end: int) -> None:
    """ρ走査用: 部分集合（間引いたidx）だけに bounces=2 or 3 の光線追跡を行う。"""
    idxs, step = stage2_rho_subsample_indices()
    grid = _stage2_load_grid()
    points_by_idx = {p["idx"]: p for p in grid["points"]}
    p, rects, W, H, cell = load_geometry()
    ray_sensors = sensors_from_params(p)
    end = min(end, len(idxs))

    out_name = f"stage2_rho_ray_b{bounces}.json"
    out_path = OUT_DIR / out_name
    data = _load_existing(out_path)
    data.setdefault("meta", {
        "stage": out_name, "n_rays": STAGE2_N_RAYS, "seed": STAGE2_SEED, "max_bounces": bounces,
        "subsample_step": step, "subsample_n": len(idxs),
    })
    records: Dict[str, Dict] = data["records"]

    print(f"[{out_name}] 範囲 {start}..{end}（既計算 {len(records)} 件 / 全 {len(idxs)} 点、間引き step={step}）",
          flush=True)
    n_done = 0
    t_start = time.time()
    for i in range(start, end):
        idx = idxs[i]
        key = str(idx)
        if key in records:
            continue
        pt = points_by_idx[idx]
        sensor = ray_sensors[pt["sensor_idx"]]
        pose = (pt["x"], pt["y"], pt["theta"])
        t0 = time.time()
        value = raycast_response(
            sensor, pose, rects,
            n_rays=STAGE2_N_RAYS, seed=STAGE2_SEED, max_bounces=bounces, include_floor=True,
            led_half_angle_deg=RAY_LED_HALF_DEG, pt_half_angle_deg=RAY_PT_HALF_DEG,
            separation_m=RAY_SEPARATION_M, diffuse=0.8,
        )
        dt = time.time() - t0
        records[key] = {"idx": idx, "value": value, "i_full_ratio": value / I_FULL, "elapsed_s": dt}
        n_done += 1
        if n_done % 50 == 0:
            _save(out_path, data)
    _save(out_path, data)
    print(f"[{out_name}] この呼び出しで {n_done} 件・所要 {time.time()-t_start:.1f}s"
          f"（全体 {len(records)}/{len(idxs)} 件）", flush=True)


def run_stage2_rho_summary() -> Dict:
    idxs, step = stage2_rho_subsample_indices()
    b1_full = _load_json(OUT_DIR / "stage2_ray_b1.json")["records"]
    b4_full = _load_json(OUT_DIR / "stage2_ray_b4.json")["records"]
    b2 = _load_json(OUT_DIR / "stage2_rho_ray_b2.json")["records"]
    b3 = _load_json(OUT_DIR / "stage2_rho_ray_b3.json")["records"]

    common = [i for i in idxs if all(str(i) in d for d in (b1_full, b4_full, b2, b3))]
    complete = len(common) == len(idxs)

    base_rho = 0.8
    rows = []
    for i in common:
        k = str(i)
        v1 = b1_full[k]["i_full_ratio"]
        v2 = b2[k]["i_full_ratio"]
        v3 = b3[k]["i_full_ratio"]
        v4 = b4_full[k]["i_full_ratio"]
        inc2 = v2 - v1   # 反射2回目ちょうどの増分（ρ=0.8 で測定）。ρ^2 に比例
        inc3 = v3 - v2   # 反射3回目ちょうどの増分。ρ^3 に比例
        inc4 = v4 - v3   # 反射4回目ちょうどの増分。ρ^4 に比例
        rows.append({"idx": i, "inc2": inc2, "inc3": inc3, "inc4": inc4})

    result: Dict = {
        "subsample_step": step, "subsample_n": len(idxs), "n_common": len(common), "complete": complete,
        "base_rho": base_rho,
    }
    for rho in (0.5, 0.65, 0.8):
        deltas = []
        for r in rows:
            scale2 = (rho / base_rho) ** 2
            scale3 = (rho / base_rho) ** 3
            scale4 = (rho / base_rho) ** 4
            delta_rho = r["inc2"] * scale2 + r["inc3"] * scale3 + r["inc4"] * scale4
            deltas.append(delta_rho)
        m5_rho = _pct95([abs(d) for d in deltas])
        result[f"M5_rho_{rho}"] = m5_rho

    _save(OUT_DIR / "stage2_rho_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return result


# ----------------------------------------------------------------------------
# 工程2 副次B（事前登録 §5-3）: モデルSどうし・LED半値角 3.0°→7.0° の否定対照。
# 面積分どうしなので速い（`run_stage2_response_s(led_half_angle_deg=7.0)` を先に呼ぶこと）。
# ----------------------------------------------------------------------------
def run_stage2_negctrl_summary() -> Dict:
    s = _load_json(OUT_DIR / "stage2_response_s.json")["records"]
    s7 = _load_json(OUT_DIR / "stage2_response_s_negctrl.json")["records"]
    diffs, common = _diffs_by_idx(s, s7)
    m5_neg = _pct95(diffs)
    result = {
        "n_common": len(common), "M5_negctrl": m5_neg,
        "pass": (m5_neg is not None and m5_neg > 0.05),
    }
    _save(OUT_DIR / "stage2_negctrl_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return result


# ============================================================================
# 実行・報告
# ============================================================================
STAGE_CHOICES = [
    "verify0",
    "stage1_anchor_a", "stage1_anchor_b", "stage1_integration", "stage1_negctrl", "stage1_summary",
    "stage2_grid",
    "stage2_raycast_b1", "stage2_raycast_b4",
    "stage2_response_s", "stage2_response_s_negctrl",
    "stage2_summary",
    "stage2_rho_raycast_b2", "stage2_rho_raycast_b3", "stage2_rho_summary",
    "stage2_negctrl_summary",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGE_CHOICES, default="verify0")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    args = ap.parse_args()

    if args.stage == "stage1_anchor_a":
        return run_stage1_raycast(
            "stage1_ray_s777001.json", seed=STAGE1_SEED_A, led_half_angle_deg=RAY_LED_HALF_DEG,
            start=args.start, end=min(args.end, STAGE1_N_POSES),
        ) or 0
    if args.stage == "stage1_anchor_b":
        return run_stage1_raycast(
            "stage1_ray_s777002.json", seed=STAGE1_SEED_B, led_half_angle_deg=RAY_LED_HALF_DEG,
            start=args.start, end=min(args.end, STAGE1_N_POSES),
        ) or 0
    if args.stage == "stage1_integration":
        return run_stage1_integration() or 0
    if args.stage == "stage1_negctrl":
        return run_stage1_raycast(
            "stage1_negctrl.json", seed=STAGE1_SEED_A, led_half_angle_deg=STAGE1_NEGCTRL_LED_HALF_DEG,
            start=args.start, end=min(args.end, STAGE1_N_POSES),
        ) or 0
    if args.stage == "stage1_summary":
        run_stage1_summary()
        return 0
    if args.stage == "stage2_grid":
        build_stage2_grid()
        return 0
    if args.stage == "stage2_raycast_b1":
        run_stage2_raycast(1, args.start, args.end)
        return 0
    if args.stage == "stage2_raycast_b4":
        run_stage2_raycast(4, args.start, args.end)
        return 0
    if args.stage == "stage2_response_s":
        run_stage2_response_s(None)
        return 0
    if args.stage == "stage2_response_s_negctrl":
        run_stage2_response_s(STAGE2_NEGCTRL_LED_HALF_DEG)
        return 0
    if args.stage == "stage2_summary":
        run_stage2_summary()
        return 0
    if args.stage == "stage2_rho_raycast_b2":
        run_stage2_rho_raycast(2, args.start, args.end)
        return 0
    if args.stage == "stage2_rho_raycast_b3":
        run_stage2_rho_raycast(3, args.start, args.end)
        return 0
    if args.stage == "stage2_rho_summary":
        run_stage2_rho_summary()
        return 0
    if args.stage == "stage2_negctrl_summary":
        run_stage2_negctrl_summary()
        return 0

    assert args.stage == "verify0"
    print("=" * 78)
    print("AUDIT_056 検証0: bounces=1 の厳密一致 / 面素分割の収束 / 光線追跡との突き合わせ")
    print("=" * 78)

    t0 = time.time()
    r_a = run_verify0_a()
    print(f"\n--- 0-a: bounces=1 が厳密一致するか（{r_a['n_checked']} 姿勢） ---")
    print(f"  相対差の最大: {r_a['max_reldiff']:.3e}（分割点 1e-12）"
          f" -> {'合格' if r_a['pass'] else '不合格'}")

    r_b = run_verify0_b()
    print(f"\n--- 0-b: 面素分割 1.5 倍での収束（{r_b['n_checked']} 姿勢・bounces=4） ---")
    print(f"  面素分割: 基準 {r_b['base_grid']} -> 1.5倍 {r_b['scaled_grid']}")
    print(f"  増分の変化（満量比）の最大: {r_b['max_delta']:.5f}（分割点 0.002）"
          f" -> {'合格' if r_b['pass'] else '不合格'}")
    if not r_b["pass"]:
        print(f"  最悪姿勢: {r_b['worst']}")

    r_c = run_verify0_c()
    print(f"\n--- 0-c（是正版）: 光線追跡との突き合わせ（壁1枚＋床／向かい合う壁2枚＋床） ---")
    print(f"  {'配置':>20s} {'距離[mm]':>10s} {'モデルI増分':>14s} {'光線追跡増分':>14s} {'差(満量比)':>12s}")
    for config_name, d_mm, inc_model, inc_ray, diff in r_c["rows"]:
        print(f"  {config_name:>20s} {d_mm:>10d} {inc_model:>14.6f} {inc_ray:>14.6f} {diff:>12.5f}")
    print(f"  差の最大（満量比）: {r_c['max_diff']:.5f}（分割点 0.01）"
          f" -> {'合格' if r_c['pass'] else '不合格'}")

    r_t = run_timing()
    print(f"\n--- 参考: 1 姿勢 1 センサあたりの計測時間（bounces=4・実迷路の姿勢 n={r_t['n']}） ---")
    print(f"  平均 {r_t['mean_s']*1000:.1f}ms / 95%点 {r_t['p95_s']*1000:.1f}ms"
          f" / 最大 {r_t['max_s']*1000:.1f}ms（要件: 2秒以内）"
          f" -> {'合格' if r_t['max_s'] <= 2.0 else '不合格'}")

    all_pass = r_a["pass"] and r_b["pass"] and r_c["pass"]
    print(f"\n所要時間: {time.time()-t0:.1f}秒")
    print(f"\n検証0 総合: {'すべて合格' if all_pass else '不合格あり'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
