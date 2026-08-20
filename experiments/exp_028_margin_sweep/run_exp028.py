"""
experiments/exp_028_margin_sweep/run_exp028.py
================
exp_028（`classic/fast_planner.py::plan_fast_run` に足した `clearance_margin_m`
の掃引と、余裕を実測の横ずれ（10〜20mm）に合わせたときの完走の有無を測る）の
一次記録を測る測定スクリプト。

背景（任務指示 2026-08-20「余裕を実測に合わせる」）: `design_turn_v1` の10迷路で
北向き開始（`reset_to_start` 後・競技の本番条件）が経路の28%で頭打ちになる。
衝突地点は半径0.178mの弧の途中で、設計上の余裕（`margin`、既定5mm）に対し
実測の横ずれが10〜20mmある疑い。本実験は `margin` を5〜30mmで掃引し、
①完走する迷路が出るか ②`margin` を上げるとηの上限（=`t_plan`）が下がる
という摩擦円の使用率 `u` と同じ構造が成り立つか、を確かめる。

🔴 exp_027（`friction_use` の掃引）と異なり、本実験は EXPLORE/RETURN を
一切走らせない。理由:
  1. `margin`/`friction_use`/`wall_correction` は Phase.FAST/RETURN2 の
     `plan_fast_run`/追従にしか効かない（EXPLORE/RETURN は常にコマンド方式）。
     したがって「学習した地図」は margin/u の値に関わらず同一になるはずで、
     10迷路につき1回だけ地図を作ればよい（EXPLORE を120回繰り返すのは無駄）。
  2. `tests/test_classic_fast_run_profile.py::
     test_north_start_is_not_catastrophically_more_fragile_than_south_start`
     が既に踏んでいる作法（迷路の**真の壁**をそのまま「既知の地図」として
     `MazeMap` へ直接書き込み、Phase.FAST を直接開始する）をそのまま使う。
     これにより探索の質のばらつきを切り離し、「FAST走行そのものの追従性能」
     だけを見る（本実験の問い＝弧の余裕の効き方、に直接対応する簡略化）。
  3. 北向き/南向きの開始は、`competition.evaluator` の状態機械が実際に
     生成する順序（大きい margin で1本目=南向きが成功し続けると北向きの
     サンプルが1つも取れない、という欠落が起こりうる）に頼らず、
     `MouseSim.full_reset(heading_deg=90.0 / -90.0)` + `ex.heading` で
     **両方を毎回直接・確実に**測る（`_begin_known_map_fast_run` と同じ作法）。

🔴 測定は前景で完走させること（バックグラウンドへ投げてターンを終えると
プロセスが落ちる。5回起きている）。
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "manifest.json"
OUT_ROOT = REPO_ROOT / "outputs" / "exp_028"

# 任務指示: margin 5/10/15/20/25/30mm の6水準・u 0.30/0.50 の2水準・壁センサ補正あり・
# 北向き/南向き開始の両方。
MARGIN_LEVELS_MM: List[float] = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
U_LEVELS: List[float] = [0.30, 0.50]
WALL_CORRECTION = True

ATTEMPT_TIME_BUDGET_S = 90.0  # 1回のFAST走行にかける上限（t_planは最大でも数十秒）
MAX_TICKS = 12000  # 安全弁（control_dt=0.01sなら120秒ぶん）

# 評価用に予約された seed 範囲（docs/RESEARCH_PLAN.md §2・§9-7 が正）。
RESERVED_SEED_RANGE = (1000, 40999)


def _margin_dirname(margin_mm: float) -> str:
    return f"margin_{margin_mm:04.1f}mm"


def _u_dirname(u: float) -> str:
    return f"u_{u:.2f}"


# ==========================================================================
# 対象迷路の選定（exp_027 と同じ作法。結果に依存しない・seedを直書きしない）
# ==========================================================================
def select_target_mazes(manifest_path: Path = MANIFEST_PATH) -> List[Dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mazes = manifest["mazes"]
    ordered = sorted(mazes, key=lambda m: int(m["seed"]))
    selected = [{"seed": int(m["seed"]), "d0": int(m["d0"])} for m in ordered]

    lo, hi = RESERVED_SEED_RANGE
    for m in selected:
        if lo <= m["seed"] <= hi:
            raise RuntimeError(
                f"評価用に予約された seed 範囲 [{lo}, {hi}] に含まれる seed {m['seed']} を"
                f"選んでしまった（測定禁止）。"
            )
    return selected


# ==========================================================================
# 迷路ごとの「既知の地図」（真の壁をそのまま MazeMap へ）と XML の準備
# ==========================================================================
def _known_map_and_xml(maze_dir: Path, seed: int, xml_cache_dir: Path):
    """`tests/test_classic_fast_run_profile.py::maze_41004_known_map_and_xml`
    と同じ作法: 真の壁配列をそのまま `MazeMap`（WallState）へ書き込む。"""
    import numpy as np

    from classic.maze_map import MazeMap, WallState
    from mouse.mjcf import build_maze_robot_xml
    from mouse.params import RobotParams

    npz_path = maze_dir / f"maze_{seed}.npz"
    data = np.load(npz_path)
    v_walls_bool = data["v_walls"]
    h_walls_bool = data["h_walls"]
    width = int(data["width"]) if "width" in data else int(v_walls_bool.shape[0] - 1)
    height = int(data["height"]) if "height" in data else int(h_walls_bool.shape[1])

    maze = MazeMap(width, height)
    maze.v_walls[:, :] = np.where(v_walls_bool == 1, int(WallState.WALL), int(WallState.OPEN))
    maze.h_walls[:, :] = np.where(h_walls_bool == 1, int(WallState.WALL), int(WallState.OPEN))

    xml_cache_dir.mkdir(parents=True, exist_ok=True)
    xml_path = xml_cache_dir / f"maze_{seed}.xml"
    if not xml_path.exists():
        build_maze_robot_xml(v_walls_bool, h_walls_bool, str(xml_path),
                              model_name=f"exp028_{seed}", params=RobotParams())
    return maze, str(xml_path), width, height


# ==========================================================================
# 1回のFAST走行を、既知の地図・指定した開始方位から直接駆動する
# ==========================================================================
def run_single_attempt(xml_path: str, maze, width: int, height: int,
                        start_heading, heading_deg: float,
                        friction_use: float, clearance_margin_m: float,
                        wall_correction: bool,
                        time_budget: float = ATTEMPT_TIME_BUDGET_S,
                        max_ticks: int = MAX_TICKS) -> Dict:
    """`tests/test_classic_fast_run_profile.py::_begin_known_map_fast_run` +
    `_drive_fast_run_to_completion` を、η・到達距離の計算に要る
    run_time（前端がゴール領域へ入った時刻）・path_length_m（真の位置の
    実軌跡長）まで測れるように拡張したもの。

    `outcome` は "goal"（先頭がゴール領域に入り、その後 `plan_id ==
    "fast:goal_stop"` まで衝突なく到達）/ "collision" / "plan_failed"（学習
    地図から到達不能。既知の地図＝真の壁なので通常は起こらない）/
    "max_ticks_exceeded"（安全弁）のいずれか。
    """
    import mujoco

    from classic.explorer import ClassicExplorer
    from competition.evaluator import front_offset, front_point, goal_region_bounds, in_goal_region
    from mouse.params import RobotParams
    from mouse.sim import MouseSim

    params = RobotParams()
    cell_size = params.cell_size

    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 0), heading_deg=heading_deg)

    ex = ClassicExplorer(width, height, params=params, fast_mode="profile",
                          friction_use=friction_use, clearance_margin_m=clearance_margin_m,
                          wall_correction=wall_correction)
    ex.maze = maze
    ex.heading = start_heading
    obs = sim.observation()
    ex._begin_fast_run(obs)
    ex._need_replan = False

    if ex._fast_command_fallback:
        return {"outcome": "plan_failed", "run_time": None, "path_length_m": 0.0,
                "n_ticks": 0, "t_plan": None}

    # `_begin_fast_run` が内部で呼んだ `plan_fast_run` の結果（t_plan）をそのまま
    # 使う（judge.py で同じ計算をやり直さない。margin=25/30mmのような重いターン数の
    # 迷路では1回の幾何探索が数十秒かかるため、後から呼び直すと極端に遅い）。
    t_plan = ex._fast_plan.t_plan if ex._fast_plan is not None else None

    mouse_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    f_off = front_offset(sim.model, sim.data, mouse_bid)
    goal_bounds = goal_region_bounds(width, height, cell_size)

    x, y, yaw = sim.privileged_pose()
    prev_xy = (x, y)
    path_len = 0.0
    t_goal_front: Optional[float] = None
    n_ticks = 0

    for _ in range(max_ticks):
        n_ticks += 1
        obs = sim.observation()
        vl, vr, plan_id = ex.tick(obs)
        result = sim.step_control(vl, vr)
        t = result["sim_time"]
        x, y, yaw = sim.privileged_pose()
        path_len += math.hypot(x - prev_xy[0], y - prev_xy[1])
        prev_xy = (x, y)

        if result["collision"]:
            return {"outcome": "collision", "run_time": None,
                    "path_length_m": path_len, "n_ticks": n_ticks, "t_plan": t_plan}
        if result["tipped"]:
            return {"outcome": "tipover", "run_time": None,
                    "path_length_m": path_len, "n_ticks": n_ticks, "t_plan": t_plan}

        if t_goal_front is None:
            fx, fy = front_point(x, y, yaw, f_off)
            if in_goal_region(fx, fy, width, height, cell_size):
                t_goal_front = t  # 前端がゴール領域へ入った瞬間(NTF基準の暫定タイム)

        if plan_id == "fast:goal_stop":
            # 計画（ゴール区画中心が終点）を衝突なく完遂し、停止ホールドに入った
            # ＝ゴール成立（`t_goal_front` は必ずこの手前で確定しているはず）。
            run_time = t_goal_front if t_goal_front is not None else t
            return {"outcome": "goal", "run_time": run_time,
                    "path_length_m": path_len, "n_ticks": n_ticks, "t_plan": t_plan}

        if t >= time_budget:
            return {"outcome": "timeout", "run_time": None,
                    "path_length_m": path_len, "n_ticks": n_ticks, "t_plan": t_plan}

    return {"outcome": "max_ticks_exceeded", "run_time": None,
            "path_length_m": path_len, "n_ticks": n_ticks, "t_plan": t_plan}


# ==========================================================================
# 1 (迷路, margin, u) の測定（北向き・南向きの両方）
# ==========================================================================
def measure_one(task: Dict) -> Dict:
    from classic.maze_map import Direction

    seed = task["seed"]
    margin_mm = task["margin_mm"]
    u = task["u"]
    maze_dir = Path(task["maze_dir"])
    xml_cache_dir = Path(task["xml_cache_dir"])
    out_path = Path(task["out_path"])

    maze, xml_path, width, height = _known_map_and_xml(maze_dir, seed, xml_cache_dir)

    north = run_single_attempt(
        xml_path, maze, width, height,
        start_heading=Direction.N, heading_deg=90.0,
        friction_use=u, clearance_margin_m=margin_mm / 1000.0, wall_correction=WALL_CORRECTION,
    )
    south = run_single_attempt(
        xml_path, maze, width, height,
        start_heading=Direction.S, heading_deg=-90.0,
        friction_use=u, clearance_margin_m=margin_mm / 1000.0, wall_correction=WALL_CORRECTION,
    )

    record = {
        "seed": seed, "margin_mm": margin_mm, "u": u,
        "wall_correction": WALL_CORRECTION,
        "north": north, "south": south,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def _build_tasks(target_mazes: List[Dict], maze_dir: Path, xml_cache_dir: Path,
                  margin_levels_mm: List[float], u_levels: List[float]) -> List[Dict]:
    tasks = []
    for margin_mm in margin_levels_mm:
        for u in u_levels:
            for m in target_mazes:
                out_path = (OUT_ROOT / _margin_dirname(margin_mm) / _u_dirname(u)
                            / f"maze_{m['seed']}.json")
                tasks.append({
                    "seed": m["seed"], "margin_mm": margin_mm, "u": u,
                    "maze_dir": str(maze_dir), "xml_cache_dir": str(xml_cache_dir),
                    "out_path": str(out_path),
                })
    return tasks


def _print_progress(i: int, total: int, task: Optional[Dict], r: Dict, t0: float) -> None:
    n_o, s_o = r["north"]["outcome"], r["south"]["outcome"]
    n_t = r["north"]["run_time"]
    s_t = r["south"]["run_time"]
    print(f"[{i}/{total}] seed={r['seed']:>5} margin={r['margin_mm']:>4.1f}mm "
          f"u={r['u']:.2f} north={n_o:>10}({f'{n_t:.3f}s' if n_t else '-':>8}) "
          f"south={s_o:>10}({f'{s_t:.3f}s' if s_t else '-':>8}) "
          f"elapsed={time.time()-t0:6.1f}s", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                         help="対象条件の一覧の印字だけで終わる（測定は1本も走らせない）")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--only-margin-mm", type=float, default=None,
                         help="指定した margin[mm] の水準だけを測定する（分割実行用）")
    parser.add_argument("--only-u", type=float, default=None,
                         help="指定した u の水準だけを測定する（分割実行用）")
    parser.add_argument("--serial", action="store_true",
                         help="逐次実行に切り替える（既定は multiprocessing による並列実行）")
    parser.add_argument("--workers", type=int, default=None,
                         help="並列数（既定: min(16, os.cpu_count()-2)）")
    args = parser.parse_args(argv)

    maze_dir = args.manifest.parent
    xml_cache_dir = OUT_ROOT / "_xml_cache"
    target_mazes = select_target_mazes(args.manifest)
    if args.only_seed is not None:
        target_mazes = [m for m in target_mazes if m["seed"] == args.only_seed]

    margin_levels = MARGIN_LEVELS_MM if args.only_margin_mm is None else [args.only_margin_mm]
    u_levels = U_LEVELS if args.only_u is None else [args.only_u]

    print(f"対象 {len(target_mazes)} 迷路（manifest: {args.manifest}）")
    print(f"margin水準[mm]: {margin_levels}")
    print(f"u水準: {u_levels}")
    print(f"wall_correction: {WALL_CORRECTION}（固定・北向き/南向きの両方を毎回直接測る）")

    if args.dry_run:
        print("\n--dry-run: 測定は実行しない。")
        return 0

    # XMLキャッシュを並列実行の前に逐次で作っておく（複数workerが同じパスへ
    # 同時に書き込むと壊れるため。measure_one 自体はキャッシュがあれば
    # 再生成しない）。
    print("\nXMLキャッシュを準備中...")
    for m in target_mazes:
        _known_map_and_xml(maze_dir, m["seed"], xml_cache_dir)

    tasks = _build_tasks(target_mazes, maze_dir, xml_cache_dir, margin_levels, u_levels)
    total = len(tasks)
    mode = "逐次" if args.serial else "並列"
    print(f"\n{total} 条件（{len(target_mazes)}迷路 × {len(margin_levels)}margin水準 × "
          f"{len(u_levels)}u水準）を{mode}・前景で実行する（各条件で北向き/南向きを1本ずつ計{total*2}走行）。")

    t0 = time.time()
    if args.serial:
        for i, task in enumerate(tasks, 1):
            r = measure_one(task)
            _print_progress(i, total, task, r, t0)
    else:
        n_workers = args.workers if args.workers is not None else max(1, min(16, (os.cpu_count() or 2) - 2))
        n_workers = min(n_workers, total)
        print(f"並列数: {n_workers}")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(measure_one, tasks), 1):
                _print_progress(i, total, None, r, t0)

    print(f"\n完了: {total} 条件 / 実時間 {time.time()-t0:.1f}s")
    print("🔴 合否・η の判定はここでは行わない。"
          "experiments/exp_028_margin_sweep/judge.py で行うこと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
