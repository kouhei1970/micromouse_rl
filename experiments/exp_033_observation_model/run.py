"""
experiments/exp_033_observation_model/run.py
================
exp_033（`classic/obs_model.py`。任意の姿勢で成り立つ距離センサ観測モデル）の測定。
`PREREG.md` §2・§3・§4・§6・§7 のとおりに測る。

対象: 実大会迷路 3 面（`competition/mazes/contest_historical/`。決定的な規則で選ぶ
— 下記 `_select_mazes` 参照。無作為には選ばない）。

姿勢: PREREG §3 の直積（区画16×区画内x5×区画内y5×ヨー角24 = 9600通り/迷路）で
決定的に作る（走行の軌跡からは採らない）。機体が壁・柱に食い込む姿勢は
`classic/geometry.py` の干渉判定（`clearance() < 0`）で弾き、除いた数を報告する。

任意の姿勢をシミュレータに取らせる方法: `mouse.sim.MouseSim` に区画中心以外へ
置く手段が無いため（`reset_to_start`/`full_reset` は区画中心+ヨー角のみを受け取る）、
`tests/test_classic_localization.py::_set_pose` と同じ作法で `data.qpos` を直接
書いて `mujoco.mj_forward` を呼ぶ（物理は進めない。運動学だけを反映させて
距離センサの値を読む）。

一次記録（raw_records.json）に姿勢・予測・実測を全件残す。集計は metrics.json。
`anchor_check.py` が一次記録だけから q_95 を数え直して照合する。

【追記（ユーザ指示・PREREG §7 の副次の記録を追加。§2 の主判定量・分割・予測は変更しない）】
「壁のあるなしは離散的にサンプリングした個々の情報（ある/なし）を判断すべきではなく、
連続的にサンプリングしていき確率的に判断すべき」という指示に対応するため、姿勢・
センサごとに「その柱間（壁が立ちうる場所）に壁があると仮定したときの予測」
(`d_hat_wall`) と「無いと仮定したときの予測」(`d_hat_open`) を反実仮想で計算し、
実測 (`d_meas`) と併せて一次記録へ残す（`_gap_analysis_for_pose`）。
「その柱間」は、真の壁の有無を見ずに、光線が最初に交わる壁スロット位置
（`classic/obs_model.py` と同じ Rect の作り方。壁が無くても位置だけは決まる）を
機械的に選ぶ（`_find_nearest_gap_slot`）。真の壁配列は複製してから 1 マスだけ
反転させ、元の配列は変更しない。
`classic/sensing.py:23` 以降が報告している側方センサの壁あり/なし重なり帯
（[0.0488,0.0885] m と [0.0787,0.2655] m）は 245 姿勢を**周辺化**した分布であり、
姿勢を固定すれば分離できるか、という仮説をこの記録で確かめる（§7 副次。判定には使わない）。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import mujoco

from classic.geometry import Pose, Rect, clearance, wall_obstacles
from classic.obs_model import (
    POST_SIZE_M,
    WALL_THICKNESS_M,
    _cells_radius,
    _ray_rect_hit,
    _sensor_local_ray,
    predict_ranges,
    predict_ranges_with_diagnostics,
)
from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim
from common.output_manager import OutputManager

CONTEST_DIR = os.path.join(REPO_ROOT, "competition", "mazes", "contest_historical")

# PREREG §3 の直積。
WITHIN_CELL_OFFSETS_M: List[float] = [-0.06, -0.03, 0.0, 0.03, 0.06]
YAW_DEG: List[int] = list(range(0, 360, 15))  # 24通り。0,45,90,...,315（45°の倍数）を含む
N_CELLS_PER_SIDE = 4  # 4x4=16区画

# 柱を見ていると判定する閾値（含む/除くで予測が 0.1mm 超動けば「柱が効いている」とみなす。
# しきい値ではなく分類のための実務的な値であり、q_95 の判定には使わない）。
POST_SIGHT_THRESHOLD_M = 1e-4

# N1（否定対照）: 予測に渡す姿勢を 10mm ずらす量。x 方向に固定でずらす（PREREG は
# 方向を指定していないので、機体座標に依らない世界 x 方向へのずらしを採用する）。
N1_POSE_SHIFT_M = 0.010

# N2（否定対照）: 壁の配置を無作為に置き換えるときの RNG シード。評価用に予約された
# 迷路の seed 帯（[1000, 40999]）とは無関係の値（RESEARCH_PLAN §2・§9-7 はマイコン用
# 迷路データセットの seed 帯の話であり、本 RNG は単一マイコンの壁配列を作るだけの
# ローカルな使い捨て乱数なので、その帯を参照する必要も抵触もない）。
N2_RANDOM_WALLS_SEED = 20260823


# ==========================================================================
# 迷路の選択・読み込み
# ==========================================================================
def _select_mazes() -> List[str]:
    """実大会迷路 102 面から 3 面を決定的に選ぶ: manifest.json の `mazes` 列（固定順）
    の先頭・中央・末尾。無作為選択ではなく、大会年代・迷路の並びに渡って
    3 点をまんべんなく取るための単純な規則。"""
    manifest = json.loads(
        open(os.path.join(CONTEST_DIR, "manifest.json"), encoding="utf-8").read()
    )
    mazes = manifest["mazes"]
    n = len(mazes)
    idxs = [0, n // 2, n - 1]
    return [mazes[i] for i in idxs]


def _load_maze(maze_id: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    d = np.load(os.path.join(CONTEST_DIR, f"{maze_id}.npz"))
    v_walls, h_walls = d["v_walls"], d["h_walls"]
    width, height = int(d["width"]), int(d["height"])
    return v_walls, h_walls, width, height


def _select_cells(width: int, height: int, n_side: int = N_CELLS_PER_SIDE) -> List[Tuple[int, int]]:
    """迷路の大きさだけから決定的に `n_side x n_side` 区画を選ぶ（内容に依存しない
    ので迷路が変わっても同じ規則が使える）。4x4 の等間隔格子は 4 隅を自動的に含む
    （x,y とも 0 と width-1/height-1 が含まれるため）。"""
    xs = [round(i * (width - 1) / (n_side - 1)) for i in range(n_side)]
    ys = [round(j * (height - 1) / (n_side - 1)) for j in range(n_side)]
    return [(x, y) for x in xs for y in ys]


# ==========================================================================
# 任意姿勢をシミュレータに取らせる（qpos 直書き + mj_forward。物理は進めない）
# ==========================================================================
def _set_pose(sim: MouseSim, x: float, y: float, theta_rad: float) -> None:
    qpos_adr = sim.model.jnt_qposadr[sim._root_joint_id]
    sim.data.qpos[qpos_adr:qpos_adr + 3] = [x, y, 0.002]
    sim.data.qpos[qpos_adr + 3] = math.cos(theta_rad / 2.0)
    sim.data.qpos[qpos_adr + 4] = 0.0
    sim.data.qpos[qpos_adr + 5] = 0.0
    sim.data.qpos[qpos_adr + 6] = math.sin(theta_rad / 2.0)
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def _build_sim(v_walls: np.ndarray, h_walls: np.ndarray, xml_path: str, params: RobotParams) -> MouseSim:
    build_maze_robot_xml(v_walls, h_walls, xml_path, model_name="exp033", params=params)
    return MouseSim(xml_path, params=params)


# ==========================================================================
# 追加の副次記録: 柱間ごとの反実仮想（壁あり/なし）予測（ユーザ指示、PREREG §7）
# ==========================================================================
def _find_nearest_gap_slot(
    ox_w: float, oy_w: float, dx_w: float, dy_w: float,
    v_shape: Tuple[int, int], h_shape: Tuple[int, int],
    cell_size: float, cells_radius: int,
) -> Optional[Tuple[str, int, int]]:
    """センサの光線が最初に交わる「柱間」（壁が立ちうるスロット）を、真の壁の
    有無を一切見ずに求める。壁スロットの矩形位置は `classic/obs_model.py::
    _local_obstacles` と同じ式（壁の有無に関わらず、位置だけは決まる）。

    Returns: (kind, i, j) — kind は "v"（縦壁 v_walls[i,j]）または
    "h"（横壁 h_walls[i,j]）。近傍に候補が無ければ None。
    """
    width = v_shape[0] - 1
    height = v_shape[1]
    i_center = int(math.floor(ox_w / cell_size))
    j_center = int(math.floor(oy_w / cell_size))
    i_lo, i_hi = i_center - cells_radius, i_center + cells_radius
    j_lo, j_hi = j_center - cells_radius, j_center + cells_radius

    half_wall = WALL_THICKNESS_M / 2.0
    half_post = POST_SIZE_M / 2.0
    half_run = cell_size / 2.0 - half_post

    best: Optional[Tuple[float, str, int, int]] = None

    for i in range(max(i_lo, 0), min(i_hi, width) + 1):
        for j in range(max(j_lo, 0), min(j_hi, height - 1) + 1):
            rect = Rect(float(i) * cell_size, float(j) * cell_size + cell_size / 2.0, half_wall, half_run)
            t = _ray_rect_hit(ox_w, oy_w, dx_w, dy_w, rect)
            if t is not None and (best is None or t < best[0]):
                best = (t, "v", i, j)

    for i in range(max(i_lo, 0), min(i_hi, width - 1) + 1):
        for j in range(max(j_lo, 0), min(j_hi, height) + 1):
            rect = Rect(float(i) * cell_size + cell_size / 2.0, float(j) * cell_size, half_run, half_wall)
            t = _ray_rect_hit(ox_w, oy_w, dx_w, dy_w, rect)
            if t is not None and (best is None or t < best[0]):
                best = (t, "h", i, j)

    if best is None:
        return None
    _t, kind, i, j = best
    return kind, i, j


def _toggle_wall(
    v_walls: np.ndarray, h_walls: np.ndarray, kind: str, i: int, j: int, value: int
) -> Tuple[np.ndarray, np.ndarray]:
    """真の壁配列を複製し、指定した 1 マスだけ値を差し替えて返す（元の配列は変更しない）。"""
    v2, h2 = v_walls.copy(), h_walls.copy()
    if kind == "v":
        v2[i, j] = value
    else:
        h2[i, j] = value
    return v2, h2


def _gap_analysis_for_pose(
    x: float, y: float, theta: float,
    v_walls: np.ndarray, h_walls: np.ndarray,
    params: RobotParams, cells_radius: int,
) -> List[Dict]:
    """姿勢 1 つぶん、センサ 4 本それぞれについて `d_hat_wall`/`d_hat_open`/
    `d_meas` 用の反実仮想予測を作る（`d_meas` は呼び出し側が実測を追記する）。
    LS・RS を含む全 4 本を対象にする（LF・RF も同じ枠組みで計算できるため）。
    """
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out: List[Dict] = []
    for idx, s in enumerate(params.sensors):
        ox_l, oy_l, dx_l, dy_l, _h = _sensor_local_ray(s['pos'], s['zaxis'])
        ox_w = x + ox_l * cos_t - oy_l * sin_t
        oy_w = y + ox_l * sin_t + oy_l * cos_t
        dx_w = dx_l * cos_t - dy_l * sin_t
        dy_w = dx_l * sin_t + dy_l * cos_t

        gap = _find_nearest_gap_slot(
            ox_w, oy_w, dx_w, dy_w, v_walls.shape, h_walls.shape, params.cell_size, cells_radius
        )
        if gap is None:
            out.append({"sensor": s['name'], "gap": None, "d_hat_wall": None,
                        "d_hat_open": None, "d_hat_open_nopost": None})
            continue

        kind, gi, gj = gap
        true_val = int((v_walls if kind == "v" else h_walls)[gi, gj])

        v_wall_on, h_wall_on = _toggle_wall(v_walls, h_walls, kind, gi, gj, 1)
        d_hat_wall = float(predict_ranges((x, y, theta), (v_wall_on, h_wall_on), params)[idx])

        v_wall_off, h_wall_off = _toggle_wall(v_walls, h_walls, kind, gi, gj, 0)
        d_hat_open = float(predict_ranges((x, y, theta), (v_wall_off, h_wall_off), params)[idx])
        # 「壁を無くしても柱が近くにあって視界が伸びない」かどうかの判定用
        # （柱も無視した場合との差で判定。ユーザ指示の「柱を見ている可能性」の直接の検査）。
        d_hat_open_nopost = float(
            predict_ranges((x, y, theta), (v_wall_off, h_wall_off), params, include_posts=False)[idx]
        )

        out.append({
            "sensor": s['name'],
            "gap": {"kind": kind, "i": gi, "j": gj, "true_is_wall": bool(true_val)},
            "d_hat_wall": d_hat_wall,
            "d_hat_open": d_hat_open,
            "d_hat_open_nopost": d_hat_open_nopost,
        })
    return out


# ==========================================================================
# 1 迷路ぶんの測定
# ==========================================================================
def measure_maze(maze_id: str, params: RobotParams, xml_dir: str) -> Dict:
    v_walls, h_walls, width, height = _load_maze(maze_id)
    obstacles = wall_obstacles(
        v_walls, h_walls, cell_size=params.cell_size,
        wall_thickness=0.012, post_size=0.012, center_goal=True,
    )
    cells = _select_cells(width, height)

    xml_path = os.path.join(xml_dir, f"{maze_id}.xml")
    sim = _build_sim(v_walls, h_walls, xml_path, params)
    os.remove(xml_path)

    # N2 用: この迷路と同じ shape の無作為な壁配置（迷路ごとに固定シード+迷路名で
    # 派生させ、3 迷路が同一パターンにならないようにする）。
    rng = np.random.default_rng((N2_RANDOM_WALLS_SEED, hash(maze_id) & 0xFFFFFFFF))
    random_v = rng.integers(0, 2, size=v_walls.shape).astype(np.uint8)
    random_h = rng.integers(0, 2, size=h_walls.shape).astype(np.uint8)

    records: List[Dict] = []
    n_excluded = 0
    cell_size = params.cell_size
    cells_radius = _cells_radius(params.sensor_cutoff, cell_size)

    for (cx, cy) in cells:
        base_x = cx * cell_size + cell_size / 2.0
        base_y = cy * cell_size + cell_size / 2.0
        for dx in WITHIN_CELL_OFFSETS_M:
            for dy in WITHIN_CELL_OFFSETS_M:
                x = base_x + dx
                y = base_y + dy
                for yaw_deg in YAW_DEG:
                    theta = math.radians(yaw_deg)

                    d_clear = clearance(Pose(x, y, theta), obstacles)
                    if d_clear < 0.0:
                        n_excluded += 1
                        continue

                    _set_pose(sim, x, y, theta)
                    actual = sim.observation()[:4].tolist()

                    pred, counts = predict_ranges_with_diagnostics((x, y, theta), (v_walls, h_walls), params)
                    pred_nopost = predict_ranges((x, y, theta), (v_walls, h_walls), params, include_posts=False)
                    pred_shifted = predict_ranges((x + N1_POSE_SHIFT_M, y, theta), (v_walls, h_walls), params)
                    pred_random_walls = predict_ranges((x, y, theta), (random_v, random_h), params)

                    abs_diff = [abs(p - a) for p, a in zip(pred, actual)]
                    sees_post = any(
                        abs(p - pnp) > POST_SIGHT_THRESHOLD_M for p, pnp in zip(pred, pred_nopost)
                    )
                    saturated_actual = [a >= params.sensor_cutoff - 1e-9 for a in actual]

                    gap_analysis = _gap_analysis_for_pose(x, y, theta, v_walls, h_walls, params, cells_radius)
                    for g, a in zip(gap_analysis, actual):
                        g["d_meas"] = a

                    records.append({
                        "maze_id": maze_id,
                        "cell": [cx, cy],
                        "dx": dx, "dy": dy, "yaw_deg": yaw_deg,
                        "x": x, "y": y, "theta_rad": theta,
                        "clearance_m": d_clear,
                        "predicted": pred.tolist(),
                        "actual": actual,
                        "abs_diff": abs_diff,
                        "candidate_counts": counts.tolist(),
                        "sees_post": bool(sees_post),
                        "saturated_actual": saturated_actual,
                        "n1_shifted_pose_pred": pred_shifted.tolist(),
                        "n2_random_walls_pred": pred_random_walls.tolist(),
                        "n3_no_post_pred": pred_nopost.tolist(),
                        "gap_analysis": gap_analysis,
                    })

    return {
        "maze_id": maze_id,
        "width": width, "height": height,
        "n_cells": len(cells),
        "n_poses_total": len(cells) * len(WITHIN_CELL_OFFSETS_M) ** 2 * len(YAW_DEG),
        "n_excluded_interference": n_excluded,
        "n_valid": len(records),
        "records": records,
    }


# ==========================================================================
# 集計
# ==========================================================================
def _percentile95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def _flatten_diffs(records: List[Dict], key: str = "abs_diff") -> List[float]:
    out: List[float] = []
    for r in records:
        out.extend(r[key])
    return out


def _flatten_diffs_where(records: List[Dict], pred_key: str, predicate) -> List[float]:
    out: List[float] = []
    for r in records:
        if not predicate(r):
            continue
        for p, a in zip(r[pred_key], r["actual"]):
            out.append(abs(p - a))
    return out


def _summarize_gap_analysis(all_records: List[Dict]) -> Dict:
    """ユーザ指示の追加副次記録: 柱間ごとの反実仮想（壁あり/なし）予測の分離度
    （PREREG §7 相当の追加。判定には使わない）。"""
    entries: List[Dict] = []
    for r in all_records:
        for g in r["gap_analysis"]:
            if g["d_hat_wall"] is None or g["d_hat_open"] is None:
                continue
            sep = abs(g["d_hat_open"] - g["d_hat_wall"])
            open_sees_post = (
                g["d_hat_open_nopost"] is not None
                and abs(g["d_hat_open"] - g["d_hat_open_nopost"]) > POST_SIGHT_THRESHOLD_M
            )
            entries.append({
                "sensor": g["sensor"], "separation_m": sep,
                "dx": r["dx"], "dy": r["dy"], "yaw_deg": r["yaw_deg"],
                "d_hat_wall": g["d_hat_wall"], "d_hat_open": g["d_hat_open"],
                "d_meas": g["d_meas"], "open_sees_post": bool(open_sees_post),
                "monotone_ok": g["d_hat_wall"] <= g["d_hat_open"] + 1e-9,
            })

    def _stats(vals: List[float]) -> Dict:
        if not vals:
            return {"median": None, "p5": None, "min": None, "n": 0}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "median": float(np.median(arr)), "p5": float(np.percentile(arr, 5)),
            "min": float(np.min(arr)), "n": int(arr.size),
        }

    all_sep = [e["separation_m"] for e in entries]
    ls_rs_sep = [e["separation_m"] for e in entries if e["sensor"] in ("LS", "RS")]
    lf_rf_sep = [e["separation_m"] for e in entries if e["sensor"] in ("LF", "RF")]
    n_monotone_fail = sum(1 for e in entries if not e["monotone_ok"])

    # 分離度が小さい姿勢の特徴づけ（下位 5% の post 視認率と全体の post 視認率を比較）。
    sorted_entries = sorted(entries, key=lambda e: e["separation_m"])
    n_low = max(1, len(sorted_entries) // 20)
    low_sep_entries = sorted_entries[:n_low]
    low_rate = (sum(1 for e in low_sep_entries if e["open_sees_post"]) / len(low_sep_entries)
                if low_sep_entries else None)
    overall_rate = (sum(1 for e in entries if e["open_sees_post"]) / len(entries)
                    if entries else None)

    ls_rs_wall_vals = [e["d_hat_wall"] for e in entries if e["sensor"] in ("LS", "RS")]
    ls_rs_open_vals = [e["d_hat_open"] for e in entries if e["sensor"] in ("LS", "RS")]

    return {
        "n_pairs": len(entries),
        "n_monotone_violations": n_monotone_fail,
        "separation_all_m": _stats(all_sep),
        "separation_LS_RS_m": _stats(ls_rs_sep),
        "separation_LF_RF_m": _stats(lf_rf_sep),
        "low_separation_open_sees_post_rate": low_rate,
        "overall_open_sees_post_rate": overall_rate,
        "low_separation_examples": [
            {k: v for k, v in e.items() if k != "monotone_ok"} for e in low_sep_entries[:20]
        ],
        "marginalized_LS_RS_pooled_ranges_for_comparison": {
            "d_hat_wall_range_m": [min(ls_rs_wall_vals), max(ls_rs_wall_vals)] if ls_rs_wall_vals else None,
            "d_hat_open_range_m": [min(ls_rs_open_vals), max(ls_rs_open_vals)] if ls_rs_open_vals else None,
            "note": ("classic/sensing.py:23 以降が報告する重なり帯 "
                     "wall=[0.0488,0.0885] open=[0.0787,0.2655] と比較するための、"
                     "本実験の姿勢集合をプールした（周辺化した）参考値。"
                     "姿勢を固定した場合の分離度は separation_LS_RS_m を見ること。"),
        },
    }


def _mcu_budget(median_candidates: float, max_candidates: int) -> Dict:
    """マイコン実装の予算（PREREG §6・note_037 §13-4）。実測した候補数
    （`median_candidates_observed`/`max_candidates_observed`）から見積もる。

    演算の数え方（`classic/obs_model.py` の実装に対応）:
      - 姿勢の回転 (cosθ・sinθ) は 1 制御周期に 1 回だけ計算し、4 本のセンサで
        共有する（センサ 1 本あたりの三角関数呼び出しは 0 回）。
      - センサ 1 本あたり: 原点・方向の回転に乗算 8 回（除算・三角関数なし）。
      - 光線方向の逆数 (1/dx, 1/dy) をセンサ 1 本につき 1 回ずつ計算し
        （除算 2 回）、候補ごとの乗算に変換する（候補ごとの除算をゼロにする）。
      - 候補 1 個のスラブ法判定: 減算 4 回・乗算 4 回・比較 8 回程度。
      - 最後に水平距離を光線距離へ直す 1/cos(仰角) は姿勢によらない定数なので
        起動時に逆数として 1 回だけ計算しておき、実行時は乗算 1 回で済む。
    """
    mults_per_sensor_fixed = 8 + 1  # 回転8 + 最終スケール1（逆数を使うので乗算）
    subs_per_sensor_fixed = 6        # 回転の加減算
    divs_per_sensor_fixed = 2        # 1/dx, 1/dy（候補ごとの除算をゼロにするため）

    def _per_sensor(n_cand: float) -> Dict[str, float]:
        return {
            "mults": mults_per_sensor_fixed + 4 * n_cand,
            "subs": subs_per_sensor_fixed + 4 * n_cand,
            "divs": divs_per_sensor_fixed,
            "comparisons": 8 * n_cand,
        }

    n_sensors = 4
    control_hz = 1000  # 実機の制御周期（note_037 §13-1）
    mcu_hz_range = (168e6, 480e6)  # STM32F405級〜H743級（note_037 §13-1）

    def _totals(n_cand: float) -> Dict:
        per_sensor = _per_sensor(n_cand)
        per_tick = {
            "mults": per_sensor["mults"] * n_sensors,
            "subs": per_sensor["subs"] * n_sensors,
            "divs": per_sensor["divs"] * n_sensors,
            "comparisons": per_sensor["comparisons"] * n_sensors,
            "trig_calls": 2,  # cosθ・sinθ（4本で共有。姿勢1つにつき1回ずつ）
        }
        # 単精度浮動小数点演算器つき Cortex-M4/M7 級を想定した概算サイクル数
        # （乗算・加減算・比較=1サイクル、除算=15サイクル、三角関数=50サイクルで概算。
        # 厳密な命令サイクル数はコンパイラ・CPU依存だが、負荷率の桁を見るには十分）。
        cycles_per_tick = (
            per_tick["mults"] * 1 + per_tick["subs"] * 1 + per_tick["comparisons"] * 1
            + per_tick["divs"] * 15 + per_tick["trig_calls"] * 50
        )
        cycles_per_sec = cycles_per_tick * control_hz
        load_pct_range = [
            100.0 * cycles_per_sec / mcu_hz_range[1],  # 480MHz級（下限の負荷率）
            100.0 * cycles_per_sec / mcu_hz_range[0],  # 168MHz級（上限の負荷率）
        ]
        return {
            "n_candidates_per_sensor": n_cand,
            "per_sensor": per_sensor,
            "per_control_tick_4sensors": per_tick,
            "estimated_cycles_per_tick": cycles_per_tick,
            "estimated_load_pct_at_1khz": load_pct_range,
        }

    ram_bytes_per_sensor_constants = 4 * 5  # ox,oy,dx,dy,inv_h（単精度4byte×5）を起動時に1回計算し保持
    ram_bytes_total = ram_bytes_per_sensor_constants * n_sensors  # + 一時変数(スタック、無視できる)

    return {
        "assumptions": {
            "mcu_class": "STM32F405級(168MHz)〜H743級(480MHz)。単精度FPUあり（note_037 §13-1）",
            "control_hz": control_hz,
            "cycle_cost_model": "mult/add/cmp=1cycle, div=15cycle, trig=50cycle（概算）",
        },
        "using_median_candidates": _totals(median_candidates),
        "using_max_candidates_observed": _totals(float(max_candidates)),
        "ram_bytes_obs_model_only": ram_bytes_total,
        "ram_note": "壁の信念地図(544バイト、note_037 §13-2)は別モジュール(classic/wall_belief.py予定)の見積もりで、本表には含めない",
    }


def summarize(all_records: List[Dict]) -> Dict:
    main_diffs = _flatten_diffs(all_records, "abs_diff")
    q95 = _percentile95(main_diffs)

    # --- 否定対照 (PREREG §4) ---
    n1_diffs = _flatten_diffs_where(all_records, "n1_shifted_pose_pred", lambda r: True)
    n2_diffs = _flatten_diffs_where(all_records, "n2_random_walls_pred", lambda r: True)
    # N3 は「柱を見ている姿勢で」測る（sees_post=True の記録に限定）。
    n3_diffs_seeing_post = _flatten_diffs_where(all_records, "n3_no_post_pred", lambda r: r["sees_post"])
    main_diffs_seeing_post = [
        abs(p - a) for r in all_records if r["sees_post"]
        for p, a in zip(r["predicted"], r["actual"])
    ]

    # --- 副次の記録 (PREREG §7) ---
    by_yaw: Dict[int, List[float]] = {}
    for r in all_records:
        by_yaw.setdefault(r["yaw_deg"], []).extend(r["abs_diff"])
    yaw_q95 = {str(yaw): _percentile95(vs) for yaw, vs in sorted(by_yaw.items())}

    post_only_diffs = [
        abs(p - a) for r in all_records if r["sees_post"]
        for p, a in zip(r["predicted"], r["actual"])
    ]

    saturated_diffs = []
    for r in all_records:
        for p, a, sat in zip(r["predicted"], r["actual"], r["saturated_actual"]):
            if sat:
                saturated_diffs.append(abs(p - a))

    max_candidates_observed = max(
        (c for r in all_records for c in r["candidate_counts"]), default=0
    )
    median_candidates_observed = float(np.median(
        [c for r in all_records for c in r["candidate_counts"]]
    )) if all_records else None

    return {
        "q95_main_m": q95,
        "max_abs_diff_m": max(main_diffs) if main_diffs else None,
        "n_pairs": len(main_diffs),
        "negative_controls": {
            "N1_pose_shift_10mm": {
                "q95_m": _percentile95(n1_diffs),
                "worse_than_main": (_percentile95(n1_diffs) or 0) > (q95 or 0),
            },
            "N2_random_walls": {
                "q95_m": _percentile95(n2_diffs),
                "worse_than_main": (_percentile95(n2_diffs) or 0) > (q95 or 0),
            },
            "N3_ignore_posts_seeing_post_subset": {
                "q95_m": _percentile95(n3_diffs_seeing_post),
                "q95_main_same_subset_m": _percentile95(main_diffs_seeing_post),
                "n_pairs_in_subset": len(n3_diffs_seeing_post),
            },
        },
        "secondary": {
            "q95_by_yaw_deg": yaw_q95,
            "post_only_subset": {
                "n_pairs": len(post_only_diffs),
                "q95_m": _percentile95(post_only_diffs),
            },
            "saturated_subset": {
                "n_pairs": len(saturated_diffs),
                "q95_m": _percentile95(saturated_diffs),
                "max_abs_diff_m": max(saturated_diffs) if saturated_diffs else None,
            },
            "max_candidates_observed": int(max_candidates_observed),
            "median_candidates_observed": median_candidates_observed,
        },
        "gap_analysis_separability": _summarize_gap_analysis(all_records),
        "mcu_budget": _mcu_budget(median_candidates_observed or 0.0, int(max_candidates_observed)),
    }


def main() -> None:
    t_start = time.time()
    params = RobotParams()
    om = OutputManager("exp_033_observation_model")
    xml_dir = str(om.get_path("."))
    os.makedirs(xml_dir, exist_ok=True)

    maze_ids = _select_mazes()
    print(f"対象迷路（決定的選択）: {maze_ids}")

    per_maze_results = []
    all_records: List[Dict] = []
    n_excluded_total = 0
    n_total_total = 0
    for maze_id in maze_ids:
        t0 = time.time()
        result = measure_maze(maze_id, params, xml_dir)
        dt = time.time() - t0
        print(f"{maze_id}: valid={result['n_valid']} excluded={result['n_excluded_interference']} "
              f"time={dt:.1f}s")
        per_maze_results.append({k: v for k, v in result.items() if k != "records"})
        all_records.extend(result["records"])
        n_excluded_total += result["n_excluded_interference"]
        n_total_total += result["n_poses_total"]

    raw_path = om.get_path("raw_records.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"records": all_records}, f, ensure_ascii=False)

    summary = summarize(all_records)
    summary["mazes"] = maze_ids
    summary["n_poses_total_before_exclusion"] = n_total_total
    summary["n_excluded_interference_total"] = n_excluded_total
    summary["n_valid_total"] = len(all_records)
    summary["elapsed_s"] = time.time() - t_start
    summary["per_maze"] = per_maze_results

    om.save_metrics(
        {"summary": {
            "q95_main_m": summary["q95_main_m"],
            "n_excluded_interference_total": n_excluded_total,
        }},
        phase_specific=summary,
    )

    q95 = summary["q95_main_m"]
    bucket = "不明"
    if q95 is not None:
        if q95 < 0.002:
            bucket = "[0, 0.002) 幾何が合っている"
        elif q95 < 0.010:
            bucket = "[0.002, 0.010) ずれがある（原因特定要）"
        else:
            bucket = "[0.010, inf) 幾何モデルが足りない"

    om.finalize(
        summary=(
            f"exp_033: q95={q95:.6f}m ({bucket}) / n_valid={len(all_records)} / "
            f"除外(干渉)={n_excluded_total} / "
            f"N1(10mmずらし)={summary['negative_controls']['N1_pose_shift_10mm']['q95_m']:.6f}m / "
            f"N2(乱数壁)={summary['negative_controls']['N2_random_walls']['q95_m']:.6f}m / "
            f"N3(柱無視,柱視認姿勢のみ)={summary['negative_controls']['N3_ignore_posts_seeing_post_subset']['q95_m']}"
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
