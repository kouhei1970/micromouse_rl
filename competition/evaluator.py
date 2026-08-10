"""
competition/evaluator.py
================
マイクロマウス競技評価器 (CompetitionEvaluator)。

公式評価プロトコル v2「自走帰還方式」（RESEARCH_PLAN §2、コミット e3c5409 で
確定）を実装する。仕様は指揮セッション作成の spec_evaluator.md を参照。

--------------------------------------------------------------------------
プロトコル概要（詳細はモジュール内の各関数 docstring も参照）
--------------------------------------------------------------------------
- 1 迷路あたり: 持ち時間 300 秒（シミュレーション時間 = data.time）、最大 5 走行。
  走行回数 = ロボットがスタート区画から出発した回数（ゴール到達しても
  再配置しない。自走で戻ってきて再出発すればそれが次の走行になる）。
- 評価器は 2 状態の状態機械を持つ:
    FREE:       スタート区画内、またはゴール到達後の自由移動（帰路・探索）中。
                走行タイム非計測。
    RUN_ACTIVE: ロボット中心がスタートセル領域を「出た瞬間」に開始。
                走行タイム計測中。
  RUN_ACTIVE は次のいずれかで終了する: ゴール到達 / 衝突 / 転倒 / スタック /
  持ち時間切れ。ゴール到達以外はすべて「係員回収」（reset_to_start、
  data.time は保持）を伴い、走行は失敗として記録される。
- FREE 状態中の衝突・転倒・スタック（主に帰路や無操作での放置）は
  「incidents」として記録されるのみで、走行を消費しない（係員回収のみ行う）。
  スタート区画内で 20 秒静止し続けるケースも「スタック」としてここに含まれる
  （＝実質無操作。無限ループにはならず、いずれ持ち時間切れで評価が終わる）。
- 成績は「完走した走行のうち最速タイム」(best_time)。1 本も完走しなければ
  DNF（best_time=None, success=False）。

方策インタフェースは competition/policy_interface.py の MousePolicy を参照。

--------------------------------------------------------------------------
実装上の判断（spec に明記のない詳細）— 詳細は最終報告書にもまとめる
--------------------------------------------------------------------------
1. RUN_ACTIVE 終了理由が同一制御ステップで複数同時に真になった場合の優先順位:
   goal > collision > tipover > stuck > timeout。物理的に確定した事象
   （ゴール到達・衝突・転倒）を、時計条件（timeout）より優先する。
2. スタック判定の「20 秒未満は判定しない」ゲートは、RUN_ACTIVE では走行
   開始時刻、FREE では直近のゴール到達/係員回収/評価開始時刻を基準とする
   （＝リセット直後に、リセット前後の位置を跨いだ変位で誤判定しないため）。
3. 「衝突位置 (x,y)」は outcome=="collision" のときのみ runs[].collision_pos
   に記録する（stuck/timeout/tipover では None。tipover は壁との接触を伴う
   とは限らないため）。
4. 走行数が max_runs に達し、かつ RUN_ACTIVE でなくなった時点で早期終了する
   （spec で明示的に許容されている最適化）。
"""
import argparse
import importlib
import json
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

# プロトコル版: 憲章の該当改訂コミットを指す。
#   e3c5409 = プロトコル v2（自走帰還方式）制定
#   850f283 = スタック判定の意味論明確化（閉じ込め判定。旧: 端点間変位 → 21 件の誤検出を是正）
PROTOCOL_VERSION = "850f283"
DEFAULT_CELL_SIZE = 0.18
DEFAULT_STUCK_WINDOW_S = 20.0
DEFAULT_STUCK_DISP_M = 0.05


# ==========================================================================
# セル判定（座標 → 領域判定）。評価器の計測ロジックであり、方策への入力には
# 使わない（方策は observation のみ、または特権方策なら bind_sim/bind_maze
# 経由で自前に同種の判定をしてよい）。
# ==========================================================================
def in_start_region(x: float, y: float, cell_size: float = DEFAULT_CELL_SIZE) -> bool:
    """ロボット中心 (x,y) がスタート区画 [0,cell_size)×[0,cell_size) 内かどうか。"""
    return (0.0 <= x < cell_size) and (0.0 <= y < cell_size)


def goal_region_bounds(width: int, height: int, cell_size: float = DEFAULT_CELL_SIZE):
    """中央 2x2 ゴール領域の座標範囲 (x_min, x_max, y_min, y_max) を返す。
    16x16 迷路なら [1.26,1.62)×[1.26,1.62)（spec_evaluator.md §1 記載の値と一致）。"""
    x_min = (width // 2 - 1) * cell_size
    x_max = (width // 2 + 1) * cell_size
    y_min = (height // 2 - 1) * cell_size
    y_max = (height // 2 + 1) * cell_size
    return x_min, x_max, y_min, y_max


def in_goal_region(x: float, y: float, width: int = 16, height: int = 16,
                    cell_size: float = DEFAULT_CELL_SIZE) -> bool:
    """ロボット中心 (x,y) が中央 2x2 ゴール領域内かどうか。"""
    x_min, x_max, y_min, y_max = goal_region_bounds(width, height, cell_size)
    return (x_min <= x < x_max) and (y_min <= y < y_max)


def pos_to_cell(x: float, y: float, width: int, height: int,
                 cell_size: float = DEFAULT_CELL_SIZE):
    """世界座標 (x,y) が属するセル (cx,cy) を返す（迷路範囲外は最寄りセルに丸める）。"""
    cx = int(x // cell_size)
    cy = int(y // cell_size)
    cx = min(max(cx, 0), width - 1)
    cy = min(max(cy, 0), height - 1)
    return cx, cy


def goal_cells(width: int, height: int):
    """中央 2x2 ゴール領域を構成するセル座標のリストを返す。"""
    gx0, gx1 = width // 2 - 1, width // 2
    gy0, gy1 = height // 2 - 1, height // 2
    return [(gx0, gy0), (gx0, gy1), (gx1, gy0), (gx1, gy1)]


# ==========================================================================
# セル隣接判定・BFS（評価器の中間指標計算に使う。評価迷路の真の壁情報
# v_walls/h_walls を使うのは評価器自身のみで、方策の observation には
# 出さない）。verify_eval_mazes.py の _connects と同じ座標規約:
#   v_walls[x,y]: セル(x-1,y)-(x,y) 間の縦壁（x=0,width は外周）
#   h_walls[x,y]: セル(x,y-1)-(x,y) 間の横壁（y=0,height は外周）
# ==========================================================================
def _cell_neighbors(x: int, y: int, width: int, height: int):
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if 0 <= nx < width and 0 <= ny < height:
            yield nx, ny


def _cell_connects(v_walls, h_walls, x: int, y: int, nx: int, ny: int) -> bool:
    if (nx, ny) == (x + 1, y):
        return bool(v_walls[x + 1, y] == 0)
    if (nx, ny) == (x - 1, y):
        return bool(v_walls[x, y] == 0)
    if (nx, ny) == (x, y + 1):
        return bool(h_walls[x, y + 1] == 0)
    if (nx, ny) == (x, y - 1):
        return bool(h_walls[x, y] == 0)
    raise ValueError(f"({x},{y}) と ({nx},{ny}) は隣接していません")


def bfs_distances_from_targets(v_walls, h_walls, width: int, height: int, targets):
    """targets（複数可）からの最短セル距離を、到達可能な全セルについて返す
    （multi-source BFS）。dict[(x,y)] = 距離（セル数）。"""
    targets = list(targets)
    dist = {c: 0 for c in targets}
    q = deque(targets)
    while q:
        x, y = q.popleft()
        for nx, ny in _cell_neighbors(x, y, width, height):
            if (nx, ny) not in dist and _cell_connects(v_walls, h_walls, x, y, nx, ny):
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return dist


def bfs_shortest_path(v_walls, h_walls, width: int, height: int, start, targets):
    """start から targets（集合/リスト）のいずれかへの最短セル経路
    （start を含むセル座標のリスト）を返す。到達不能なら None。"""
    targets = set(targets)
    if start in targets:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for nx, ny in _cell_neighbors(x, y, width, height):
            if (nx, ny) not in prev and _cell_connects(v_walls, h_walls, x, y, nx, ny):
                prev[(nx, ny)] = (x, y)
                if (nx, ny) in targets:
                    path = [(nx, ny)]
                    cur = (x, y)
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    path.reverse()
                    return path
                q.append((nx, ny))
    return None


# ==========================================================================
# スタック検出: 100Hz 位置履歴リングバッファ
# ==========================================================================
class PositionRingBuffer:
    """位置履歴リングバッファ。100Hz で (t,x,y) を保持し、スタックを**閉じ込め判定**で行う:
    「直近 window_s 秒間の全サンプルが、現在位置を中心とする半径 disp_threshold 圏内に
    留まり続けた」場合にスタック（憲章 §2、2026-08-10 意味論明確化 = コミット 850f283）。

    旧実装は「現在位置と window_s 秒前の位置の端点間距離」で判定しており、
    探索で同じ地点を約 20 秒後に再訪しただけの走行を誤検出していた
    （v1.0 公式評価で 21 件の誤検出。反証: 発火時に 0.22 m/s で走行中・
    窓内実走行距離 2.54 m。research_notes/ の note_004 素材参照）。

    現在の区間（走行開始 or FREE 区間開始）から window_s 秒未満しか
    経過していない間は判定しない（check_stuck の segment_start_t 引数で
    区間の基準時刻を渡す）。この基準時刻はリセット（係員回収・ゴール到達・
    評価開始）のたびに呼び出し側で更新すること。
    """

    def __init__(self, window_s: float = DEFAULT_STUCK_WINDOW_S, history_margin_s: float = 1.0):
        self.window_s = window_s
        self._history_margin_s = history_margin_s
        self._buf = deque()  # [(t, x, y), ...] 時刻昇順

    def reset(self):
        self._buf.clear()

    def push(self, t: float, x: float, y: float):
        self._buf.append((t, x, y))
        cutoff = t - self.window_s - self._history_margin_s
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def check_stuck(self, t: float, x: float, y: float, segment_start_t: float,
                     disp_threshold: float = DEFAULT_STUCK_DISP_M):
        """(is_stuck: bool, max_displacement: float|None) を返す。
        t - segment_start_t < window_s の間は (False, None)。

        閉じ込め判定: 窓 [t - window_s, t] 内の全サンプルについて現在位置との
        距離を取り、その最大値が disp_threshold 未満のときスタック
        （= 20 秒間、現在位置の 5cm 圏から一度も出ていない）。
        1 サンプルでも圏外があれば走行中とみなし早期打ち切りで返す。"""
        if (t - segment_start_t) < self.window_s:
            return False, None
        window_start_t = t - self.window_s
        max_disp = 0.0
        seen_any = False
        for (bt, bx, by) in self._buf:
            if bt < window_start_t - 1e-9:
                continue
            seen_any = True
            disp = math.hypot(x - bx, y - by)
            if disp > max_disp:
                max_disp = disp
                if max_disp >= disp_threshold:
                    return False, max_disp  # 圏外サンプルあり = 走行中（早期打ち切り）
        if not seen_any:
            return False, None
        return (max_disp < disp_threshold), max_disp


# ==========================================================================
# CompetitionEvaluator 本体
# ==========================================================================
class CompetitionEvaluator:
    """マイクロマウス競技評価器。1 迷路 / 20 迷路一括の評価を行う。

    迷路のロード: npz（v_walls, h_walls, seed, [width, height]）→ 対応する
    XML（同ディレクトリの maze_<seed>.xml）を探し、無ければ
    mouse.mjcf.build_maze_robot_xml で npz とロボット仕様から生成してキャッシュ
    する（XML は npz とロボット仕様からの再生成可能な派生物、という憲章の
    規約に従う）。regenerate_xml=True で既存 XML があっても強制再生成する。
    """

    def __init__(self, maze_dir="competition/mazes/eval", out_dir=None,
                 time_budget: float = 300.0, max_runs: int = 5,
                 regenerate_xml: bool = False,
                 stuck_window_s: float = DEFAULT_STUCK_WINDOW_S,
                 stuck_disp_m: float = DEFAULT_STUCK_DISP_M,
                 params: RobotParams = None):
        self.maze_dir = Path(maze_dir)
        self.out_dir = Path(out_dir) if out_dir else Path("competition/results")
        self.time_budget = float(time_budget)
        self.max_runs = int(max_runs)
        self.regenerate_xml = bool(regenerate_xml)
        self.stuck_window_s = float(stuck_window_s)
        self.stuck_disp_m = float(stuck_disp_m)
        self.params = params if params is not None else RobotParams()

    # ------------------------------------------------------------------
    def _load_or_build_xml(self, maze_npz_path: Path):
        """npz を読み込み、対応する XML パスを返す（無ければ生成してキャッシュ）。
        戻り値: (xml_path, seed, v_walls, h_walls, width, height)"""
        data = np.load(maze_npz_path)
        v_walls = data["v_walls"]
        h_walls = data["h_walls"]
        seed = int(data["seed"])
        width = int(data["width"]) if "width" in data else int(v_walls.shape[0] - 1)
        height = int(data["height"]) if "height" in data else int(v_walls.shape[1])

        xml_path = maze_npz_path.with_suffix(".xml")
        if self.regenerate_xml or not xml_path.exists():
            build_maze_robot_xml(
                v_walls, h_walls, str(xml_path),
                model_name=f"micromouse_eval_{maze_npz_path.stem}",
                params=self.params,
            )
        return xml_path, seed, v_walls, h_walls, width, height

    # ------------------------------------------------------------------
    def evaluate_maze(self, maze_npz_path, policy) -> dict:
        """1 迷路の評価を実行し、結果 dict を返す（JSON 保存はしない。
        保存は evaluate_all または呼び出し側の責務）。"""
        maze_npz_path = Path(maze_npz_path)
        maze_id = maze_npz_path.stem  # 例: "maze_1000"

        xml_path, seed, v_walls, h_walls, width, height = self._load_or_build_xml(maze_npz_path)

        sim = MouseSim(str(xml_path), params=self.params)
        sim.full_reset(cell=(0, 0), heading_deg=90.0)

        if getattr(policy, "requires_privileged", False):
            policy.bind_sim(sim)
            policy.bind_maze(v_walls, h_walls)

        maze_info = {
            "maze_id": maze_id, "seed": seed, "xml_path": str(xml_path),
            "width": width, "height": height,
        }
        policy.on_maze_start(maze_info)

        targets = goal_cells(width, height)
        dist_map = bfs_distances_from_targets(v_walls, h_walls, width, height, targets)
        start_to_goal_dist = dist_map.get((0, 0))

        cell_size = self.params.cell_size
        ring = PositionRingBuffer(window_s=self.stuck_window_s)

        runs = []
        incidents = []

        state = "FREE"
        run_count = 0
        cur = None  # 進行中の走行の作業変数（dict）

        x, y, _yaw = sim.privileged_pose()
        prev_in_start = in_start_region(x, y, cell_size)
        segment_start_t = sim.sim_time
        ring.push(sim.sim_time, x, y)

        finished = False
        max_steps = int(round(self.time_budget / self.params.control_dt)) + 10  # 安全弁

        for _ in range(max_steps):
            if finished:
                break

            obs = sim.observation()
            vl, vr = policy.act(obs)
            result = sim.step_control(float(vl), float(vr))
            t = result["sim_time"]
            x, y, _yaw = sim.privileged_pose()
            ring.push(t, x, y)

            in_start = in_start_region(x, y, cell_size)
            in_goal = in_goal_region(x, y, width, height, cell_size)
            is_stuck, _disp = ring.check_stuck(t, x, y, segment_start_t, self.stuck_disp_m)
            time_up = t >= self.time_budget

            if state == "RUN_ACTIVE":
                cur["path_len"] += math.hypot(x - cur["prev_xy"][0], y - cur["prev_xy"][1])
                cur["prev_xy"] = (x, y)
                cell = pos_to_cell(x, y, width, height, cell_size)
                cur["visited"].add(cell)
                cell_dist = dist_map.get(cell)
                if cell_dist is not None and (cur["min_goal_dist"] is None or cell_dist < cur["min_goal_dist"]):
                    cur["min_goal_dist"] = cell_dist

                outcome = None
                if in_goal:
                    outcome = "goal"
                elif result["collision"]:
                    outcome = "collision"
                elif result["tipped"]:
                    outcome = "tipover"
                elif is_stuck:
                    outcome = "stuck"
                elif time_up:
                    outcome = "timeout"

                if outcome is not None:
                    run_time = t - cur["t_start"]
                    max_progress = None
                    if start_to_goal_dist is not None and cur["min_goal_dist"] is not None:
                        max_progress = start_to_goal_dist - cur["min_goal_dist"]
                    path_eff = None
                    if outcome == "goal" and start_to_goal_dist:
                        shortest_m = start_to_goal_dist * cell_size
                        path_eff = (cur["path_len"] / shortest_m) if shortest_m > 0 else None

                    runs.append({
                        "index": cur["index"],
                        "t_start": cur["t_start"],
                        "t_end": t,
                        "run_time": run_time,
                        "outcome": outcome,
                        "max_progress_cells": max_progress,
                        "path_length_m": cur["path_len"],
                        "collision_pos": [x, y] if outcome == "collision" else None,
                        "visited_cells": len(cur["visited"]),
                        "path_efficiency": path_eff,
                    })
                    policy.on_run_end(outcome)
                    cur = None

                    if outcome == "goal":
                        state = "FREE"
                        segment_start_t = t
                        prev_in_start = in_start
                        if run_count >= self.max_runs:
                            finished = True
                    elif outcome == "timeout":
                        finished = True
                    else:  # collision / tipover / stuck: 係員回収
                        sim.reset_to_start(cell=(0, 0), heading_deg=90.0)
                        policy.on_retrieval()
                        x, y, _yaw = sim.privileged_pose()
                        state = "FREE"
                        prev_in_start = True
                        segment_start_t = t
                        if run_count >= self.max_runs:
                            finished = True

            else:  # state == "FREE"
                incident_kind = None
                if result["collision"]:
                    incident_kind = "collision"
                elif result["tipped"]:
                    incident_kind = "tipover"
                elif is_stuck:
                    incident_kind = "stuck"

                if incident_kind is not None:
                    incidents.append({"t": t, "kind": incident_kind, "pos": [x, y]})
                    sim.reset_to_start(cell=(0, 0), heading_deg=90.0)
                    policy.on_retrieval()
                    x, y, _yaw = sim.privileged_pose()
                    prev_in_start = True
                    segment_start_t = t
                else:
                    if prev_in_start and not in_start and run_count < self.max_runs:
                        run_count += 1
                        cur = {
                            "index": run_count, "t_start": t,
                            "visited": set(), "path_len": 0.0, "prev_xy": (x, y),
                            "min_goal_dist": dist_map.get(
                                pos_to_cell(x, y, width, height, cell_size), start_to_goal_dist),
                        }
                        state = "RUN_ACTIVE"
                        policy.on_run_start(run_count)
                    prev_in_start = in_start

            if time_up:
                finished = True

        # 安全弁: ループが RUN_ACTIVE を開いたまま終了した場合
        # （time_budget の境界ちょうどで走行が開始した極端な端数ケース等）は
        # timeout として強制的に閉じ、runs から走行が欠落しないようにする。
        if cur is not None:
            t_final = sim.sim_time
            run_time = t_final - cur["t_start"]
            max_progress = None
            if start_to_goal_dist is not None and cur["min_goal_dist"] is not None:
                max_progress = start_to_goal_dist - cur["min_goal_dist"]
            runs.append({
                "index": cur["index"], "t_start": cur["t_start"], "t_end": t_final,
                "run_time": run_time, "outcome": "timeout",
                "max_progress_cells": max_progress, "path_length_m": cur["path_len"],
                "collision_pos": None, "visited_cells": len(cur["visited"]),
                "path_efficiency": None,
            })
            policy.on_run_end("timeout")
            cur = None

        goal_runs = [r for r in runs if r["outcome"] == "goal"]
        best_time = min((r["run_time"] for r in goal_runs), default=None)
        success = best_time is not None

        return {
            "maze_id": maze_id,
            "seed": seed,
            "policy": {
                "name": getattr(policy, "name", "unnamed"),
                "requires_privileged": bool(getattr(policy, "requires_privileged", False)),
            },
            "protocol": {
                "time_budget": self.time_budget,
                "max_runs": self.max_runs,
                "control_hz": round(1.0 / self.params.control_dt),
                "collision_force_threshold": self.params.collision_force_threshold,
                "stuck_window_s": self.stuck_window_s,
                "stuck_disp_m": self.stuck_disp_m,
                "protocol_version": PROTOCOL_VERSION,
            },
            "runs": runs,
            "incidents": incidents,
            "best_time": best_time,
            "success": success,
        }

    # ------------------------------------------------------------------
    def evaluate_all(self, policy, maze_paths=None) -> dict:
        """maze_dir 内の全 npz（maze_paths 指定時はそれのみ）を評価し、
        迷路ごとの結果 JSON + summary.json を out_dir に保存して summary を返す。"""
        if maze_paths is None:
            maze_paths = sorted(self.maze_dir.glob("maze_*.npz"))
        else:
            maze_paths = [Path(p) for p in maze_paths]

        policy_name = getattr(policy, "name", "unnamed")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_out_dir = self.out_dir / f"{policy_name}_{timestamp}"
        run_out_dir.mkdir(parents=True, exist_ok=True)

        wall_clock_start = time.time()
        maze_results = []
        for p in maze_paths:
            result = self.evaluate_maze(p, policy)
            maze_results.append(result)
            out_path = run_out_dir / f"{result['maze_id']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        total_wall_clock = time.time() - wall_clock_start

        n_mazes = len(maze_results)
        successes = [r for r in maze_results if r["success"]]
        n_success = len(successes)
        best_times = {r["maze_id"]: r["best_time"] for r in maze_results}
        success_times = [r["best_time"] for r in successes]

        summary = {
            "policy": {
                "name": policy_name,
                "requires_privileged": bool(getattr(policy, "requires_privileged", False)),
            },
            "n_mazes": n_mazes,
            "n_success": n_success,
            "success_rate": (n_success / n_mazes) if n_mazes else None,
            "best_times": best_times,
            "median_best_time": float(np.median(success_times)) if success_times else None,
            "mean_best_time": float(np.mean(success_times)) if success_times else None,
            "total_wall_clock": total_wall_clock,
            "out_dir": str(run_out_dir),
        }
        with open(run_out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary


# ==========================================================================
# CLI
# ==========================================================================
def _load_policy_instance(spec_str: str):
    """"module.path:ClassName" 形式の文字列から方策クラスをロードし、
    引数無しでインスタンス化して返す。"""
    module_name, sep, class_name = spec_str.partition(":")
    if not sep:
        raise ValueError(f"--policy は 'module:ClassName' の形式で指定してください: {spec_str!r}")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="マイクロマウス競技評価器 CLI（評価迷路 20 面を既定で評価）")
    parser.add_argument("--policy", type=str, required=True,
                         help="評価する方策クラス。'module.path:ClassName' 形式")
    parser.add_argument("--maze-dir", type=str, default="competition/mazes/eval",
                         help="評価迷路 npz が入ったディレクトリ（既定: 評価用20面）")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="結果 JSON の出力先（既定: competition/results）")
    parser.add_argument("--time-budget", type=float, default=300.0, help="迷路あたりの持ち時間 [s]")
    parser.add_argument("--max-runs", type=int, default=5, help="迷路あたりの最大走行数")
    parser.add_argument("--regenerate-xml", action="store_true",
                         help="既存の maze_<seed>.xml があっても npz から強制再生成する")
    args = parser.parse_args(argv)

    policy = _load_policy_instance(args.policy)
    evaluator = CompetitionEvaluator(
        maze_dir=args.maze_dir, out_dir=args.out_dir,
        time_budget=args.time_budget, max_runs=args.max_runs,
        regenerate_xml=args.regenerate_xml,
    )
    summary = evaluator.evaluate_all(policy)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
