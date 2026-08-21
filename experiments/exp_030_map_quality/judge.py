"""
experiments/exp_030_map_quality/judge.py
================
exp_030 の判定スクリプト。`run_exp030.py` が保存した「方策が探索で作った地図」
（`requires_privileged=False` のまま作られたもの）と、迷路 npz が持つ「真の壁配列」
を突き合わせ、`research_notes/note_033_exploration_first.md` §「地図の正しさ」の
量を迷路ごとに算出する。

🔴 真の壁配列（npz）を読むのはこのファイルの中だけ。`run_exp030.py`（測定）や
`classic/`（方策本体）は一切読まない。ここで求めた量は診断専用であり、
ロボットにフィードバックしない。

判定量の定義（note_033 の表をそのまま実装する）:
  - 致命的な誤り: 真は壁 (true wall) なのに学習した地図が「開通(OPEN)」と
    判定した箇所の数。そこへ突っ込む。
  - 無害な誤り: 真は開通なのに学習した地図が「壁(WALL)」と判定した箇所の数。
    遠回りになるだけ。
  - 未知の割合: 探索後も UNKNOWN のままの壁の割合。
  - 経路の一致: 学習した地図から引いた最短経路（`classic/route.py::shortest_path`、
    悲観モード=最短走行が実際に使うのと同じ関数・同じタイブレーク規則）が、
    真の地図（＝全壁既知として同じ関数にかけたもの）の最短経路と、区画列として
    完全一致するかどうか。
  - 探索の所要時間: `run_exp030.py` が記録した explore_time_s。

外周の壁（迷路の境界）は真値・学習値のどちらも構造的に必ず「壁あり」で
一致する（`classic/maze_map.py` の `MazeMap.__init__` が外周を WALL で初期化する
のと同じ規約）ため、致命的／無害／未知の集計からは除外する（内側の壁だけを見る。
外周を含めると常に正解が積み増しされて誤り率が薄まってしまうため）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.flood import FloodMode  # noqa: E402
from classic.maze_map import Direction, MazeMap, WallState  # noqa: E402
from classic.route import NoRouteError, shortest_path  # noqa: E402

MAZE_DIR = REPO_ROOT / "competition" / "mazes" / "design_turn_v1"
OUT_ROOT = REPO_ROOT / "outputs" / "exp_030_map_quality"


def goal_cells(width: int, height: int) -> List[Tuple[int, int]]:
    """中央 2x2 ゴール領域（`classic/explorer.py::_goal_cells` /
    `competition/evaluator.py::goal_cells` と同じ規約の複製）。"""
    gx0, gx1 = width // 2 - 1, width // 2
    gy0, gy1 = height // 2 - 1, height // 2
    return [(gx0, gy0), (gx0, gy1), (gx1, gy0), (gx1, gy1)]


# ==========================================================================
# 地図の読み込み
# ==========================================================================
def load_true_map(seed: int) -> MazeMap:
    """npz から真の壁配列を読み、全壁既知の MazeMap を作る（診断専用）。"""
    npz = np.load(MAZE_DIR / f"maze_{seed}.npz")
    width = int(npz["width"])
    height = int(npz["height"])
    v_true = npz["v_walls"]  # 0=壁なし, 非0=壁あり（evaluator側の規約）
    h_true = npz["h_walls"]

    maze = MazeMap(width, height)
    maze.v_walls[:, :] = np.where(v_true != 0, int(WallState.WALL), int(WallState.OPEN))
    maze.h_walls[:, :] = np.where(h_true != 0, int(WallState.WALL), int(WallState.OPEN))
    return maze


def load_learned_map(record: Dict) -> MazeMap:
    """run_exp030.py が保存した学習済み地図（WallState 規約そのまま）を復元する。"""
    v = np.array(record["v_walls_known"], dtype=np.int8)
    h = np.array(record["h_walls_known"], dtype=np.int8)
    width, height = v.shape[0] - 1, v.shape[1]
    maze = MazeMap(width, height)
    maze.v_walls[:, :] = v
    maze.h_walls[:, :] = h
    return maze


# ==========================================================================
# 誤りの集計（外周を除いた内側の壁だけ）
# ==========================================================================
def _interior_mask_v(width: int, height: int) -> np.ndarray:
    m = np.ones((width + 1, height), dtype=bool)
    m[0, :] = False
    m[width, :] = False
    return m


def _interior_mask_h(width: int, height: int) -> np.ndarray:
    m = np.ones((width, height + 1), dtype=bool)
    m[:, 0] = False
    m[:, height] = False
    return m


def count_errors(true_maze: MazeMap, learned_maze: MazeMap) -> Dict:
    width, height = true_maze.width, true_maze.height
    mv = _interior_mask_v(width, height)
    mh = _interior_mask_h(width, height)

    tv, th = true_maze.v_walls, true_maze.h_walls
    lv, lh = learned_maze.v_walls, learned_maze.h_walls

    true_wall_v = (tv == int(WallState.WALL)) & mv
    true_wall_h = (th == int(WallState.WALL)) & mh
    true_open_v = (tv == int(WallState.OPEN)) & mv
    true_open_h = (th == int(WallState.OPEN)) & mh

    fatal_v = true_wall_v & (lv == int(WallState.OPEN))
    fatal_h = true_wall_h & (lh == int(WallState.OPEN))
    harmless_v = true_open_v & (lv == int(WallState.WALL))
    harmless_h = true_open_h & (lh == int(WallState.WALL))

    unknown_v = (lv == int(WallState.UNKNOWN)) & mv
    unknown_h = (lh == int(WallState.UNKNOWN)) & mh

    n_interior = int(mv.sum() + mh.sum())
    n_unknown = int(unknown_v.sum() + unknown_h.sum())

    fatal_locs = [("v", (int(idx[0]), int(idx[1]))) for idx in np.argwhere(fatal_v)] + \
                 [("h", (int(idx[0]), int(idx[1]))) for idx in np.argwhere(fatal_h)]
    harmless_locs = [("v", (int(idx[0]), int(idx[1]))) for idx in np.argwhere(harmless_v)] + \
                     [("h", (int(idx[0]), int(idx[1]))) for idx in np.argwhere(harmless_h)]

    return {
        "fatal": len(fatal_locs),
        "harmless": len(harmless_locs),
        "unknown_frac": float(n_unknown / n_interior) if n_interior else 0.0,
        "n_interior_walls": n_interior,
        "fatal_locs": fatal_locs,
        "harmless_locs": harmless_locs,
    }


# ==========================================================================
# 経路の一致
# ==========================================================================
def route_match(true_maze: MazeMap, learned_maze: MazeMap) -> Dict:
    width, height = true_maze.width, true_maze.height
    goals = goal_cells(width, height)
    start = (0, 0)

    out = {"true_path": None, "learned_path": None, "match": False,
           "true_len": None, "learned_len": None, "note": ""}

    try:
        true_path = shortest_path(true_maze, start, goals, FloodMode.PESSIMISTIC)
        out["true_path"] = true_path
        out["true_len"] = len(true_path) - 1
    except NoRouteError:
        out["note"] = "真の地図でも到達不能（迷路生成の異常。あり得ないはず）"
        return out

    try:
        learned_path = shortest_path(learned_maze, start, goals, FloodMode.PESSIMISTIC)
        out["learned_path"] = learned_path
        out["learned_len"] = len(learned_path) - 1
    except NoRouteError:
        out["note"] = "学習した地図（悲観）では経路なし＝探索が完了していない"
        return out

    out["match"] = (true_path == learned_path)
    if not out["match"] and out["true_len"] == out["learned_len"]:
        out["note"] = "歩数は真の最短と同じだが区画列が異なる（タイブレークの分岐）"
    elif not out["match"]:
        out["note"] = f"歩数も異なる（真={out['true_len']}, 学習={out['learned_len']}）"
    return out


# ==========================================================================
# 迷路ごとの判定 → 表
# ==========================================================================
def judge_one(seed: int) -> Dict:
    record = json.loads((OUT_ROOT / f"maze_{seed}.json").read_text(encoding="utf-8"))
    true_maze = load_true_map(seed)
    learned_maze = load_learned_map(record)

    err = count_errors(true_maze, learned_maze)
    rt = route_match(true_maze, learned_maze)

    return {
        "seed": seed,
        "fatal": err["fatal"],
        "harmless": err["harmless"],
        "unknown_frac": err["unknown_frac"],
        "n_interior_walls": err["n_interior_walls"],
        "fatal_locs": err["fatal_locs"],
        "harmless_locs": err["harmless_locs"],
        "route_match": rt["match"],
        "route_note": rt["note"],
        "true_len": rt["true_len"],
        "learned_len": rt["learned_len"],
        "explore_time_s": record["explore_time_s"],
        "n_runs": len(record["result"]["runs"]),
        "n_goal": sum(1 for r in record["result"]["runs"] if r["outcome"] == "goal"),
        "best_time": record["result"]["best_time"],
    }


def main(argv=None) -> int:
    manifest = json.loads((MAZE_DIR / "manifest.json").read_text(encoding="utf-8"))
    seeds = sorted(int(m["seed"]) for m in manifest["mazes"])

    rows = []
    for seed in seeds:
        out_path = OUT_ROOT / f"maze_{seed}.json"
        if not out_path.exists():
            print(f"警告: {out_path} が無い（run_exp030.py を先に実行すること）。スキップ。")
            continue
        rows.append(judge_one(seed))

    print(f"\n{'seed':>6} {'致命的':>6} {'無害':>6} {'未知%':>7} {'経路一致':>8} "
          f"{'探索時間[s]':>11} {'走行数':>6} {'ゴール数':>8} {'best[s]':>9}")
    for r in rows:
        et = f"{r['explore_time_s']:.2f}" if r["explore_time_s"] is not None else "未到達"
        bt = f"{r['best_time']:.3f}" if r["best_time"] else "-"
        print(f"{r['seed']:>6} {r['fatal']:>6} {r['harmless']:>6} "
              f"{r['unknown_frac']*100:>6.2f}% {str(r['route_match']):>8} "
              f"{et:>11} {r['n_runs']:>6} {r['n_goal']:>8} {bt:>9}")

    n_fatal_zero = sum(1 for r in rows if r["fatal"] == 0)
    print(f"\n致命的な誤り=0 の迷路: {n_fatal_zero} / {len(rows)}")

    for r in rows:
        if r["fatal"] > 0:
            print(f"\nseed={r['seed']} の致命的な誤り箇所: {r['fatal_locs']}")
        if not r["route_match"]:
            print(f"seed={r['seed']} 経路不一致: {r['route_note']} "
                  f"(真の歩数={r['true_len']}, 学習した歩数={r['learned_len']})")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_ROOT / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n判定結果を保存: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
