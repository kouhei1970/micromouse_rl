#!/usr/bin/env python3
"""exp_013（凍結帯 v4 での L0 再評価）の指標を、生データだけから独立に再計算する。

作成: 2026-08-13 准教授セッション（独立検証担当）
任務: 教授セッション（4 代目）からの指示 — exp_013 の結果の独立再計算（盲検）

盲検の範囲（准教授メモ §A の区分に従う）:
  - 読んでよい: outputs/exp_013_band_v4_reeval/ の生データ（runs_detail.json・traj/*.npz）、
                competition/mazes/eval/*.npz（壁配列）、experiments/exp_013_band_v4_reeval/card.md、
                docs/RESEARCH_PLAN.md §2
  - 読まない  : 学生A の集計スクリプトと報告数値（教授が突き合わせを終えるまで）

本スクリプトが「独立」である範囲:
  1. **迷路の事実**（スタート区画・ゴール区画・真の最短距離 D_0）は
     壁配列 npz から §2 の規定文だけを根拠に復元する。competition/ の実装を参照しない。
  2. **走行ごとの幾何量**（実走経路長・平均速度・区画列・区画数・旋回回数）は
     traj/*.npz の時系列 (t, x, y) から再計算する。runs_detail.json の
     同名フィールドは**照合にのみ使い、計算には使わない**。
  3. **指標 (a)〜(e')** は docs/RESEARCH_PLAN.md §2 の定義文だけから実装する。

使い方:
    .venv/bin/python verification/independent_exp013.py [腕1 腕2 ...]
    （腕を省略すると outputs/exp_013_band_v4_reeval/ 配下に存在する腕をすべて処理する）
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "outputs" / "exp_013_band_v4_reeval"
MAZE_DIR = REPO / "competition" / "mazes" / "eval"
RESULT_JSON = REPO / "verification" / "out" / "independent_exp013.json"

N = 16                      # 16x16 区画（§2 迷路の規格）
CELL = 0.18                 # 18 cm 角（§2 迷路の規格）
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]      # 北, 東, 南, 西
GOALS = [(7, 7), (8, 7), (7, 8), (8, 8)]        # 中央の 4 区画（§2）
CORNERS = [(0, 0), (N - 1, 0), (0, N - 1), (N - 1, N - 1)]


# ---------------------------------------------------------------- 迷路の事実
def load_maze(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {"v": d["v_walls"], "h": d["h_walls"]}


def true_wall(m: dict, x: int, y: int, d: int) -> int:
    """区画 (x,y) の方向 d 側に壁があるか。"""
    if d == 0:
        return int(m["h"][x, y + 1])
    if d == 1:
        return int(m["v"][x + 1, y])
    if d == 2:
        return int(m["h"][x, y])
    return int(m["v"][x, y])


def bfs_from_goals(m: dict) -> list[list[int]]:
    """真の迷路でゴール区画群からの区画数距離を求める。"""
    dist = [[10**6] * N for _ in range(N)]
    q: deque = deque()
    for gx, gy in GOALS:
        dist[gx][gy] = 0
        q.append((gx, gy))
    while q:
        x, y = q.popleft()
        for d, (dx, dy) in enumerate(DIRS):
            if true_wall(m, x, y, d):
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and dist[nx][ny] > dist[x][y] + 1:
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))
    return dist


def start_candidates(m: dict) -> list[tuple[int, int]]:
    """§2「スタート区画は四隅のいずれか・3 方向が壁」を満たす隅を列挙する。"""
    return [c for c in CORNERS
            if sum(1 for d in range(4) if not true_wall(m, *c, d)) == 1]


# ------------------------------------------------------ 走行ごとの幾何量（軌跡から）
def cell_of(xm: float, ym: float) -> tuple[int, int]:
    """位置 [m] → 区画インデックス。原点は迷路の隅、区画は [i*CELL, (i+1)*CELL)。"""
    return (int(math.floor(xm / CELL)), int(math.floor(ym / CELL)))


def compress(seq: list) -> list:
    """連続する重複を潰す。"""
    out = [seq[0]]
    for s in seq[1:]:
        if s != out[-1]:
            out.append(s)
    return out


def turn_count_r4(cells: list[tuple[int, int]]) -> int:
    """裁定 R4 の正本定義: 実走の区画列から進行方向の変化を数える（90°=1, 180°=2）。"""
    moves = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(cells, cells[1:])]
    n = 0
    for u, v in zip(moves, moves[1:]):
        if u == v:
            continue
        if (u[0] + v[0], u[1] + v[1]) == (0, 0):       # 逆向き = 180°
            n += 2
        else:                                            # 直交 = 90°
            n += 1
    return n


def run_geometry(t, x, y, t0: float, t1: float) -> dict:
    """1 走行ぶんの窓 [t0, t1] を切り出して幾何量を再計算する。"""
    sel = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
    xs, ys = x[sel], y[sel]
    path_len = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    cells = compress([cell_of(a, b) for a, b in zip(xs, ys)])
    n_moves = len(cells) - 1
    non_adjacent = sum(1 for a, b in zip(cells, cells[1:])
                       if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1)
    dur = float(t1 - t0)
    return {
        "n_samples": int(sel.sum()),
        "duration": dur,
        "path_length_m": path_len,
        "mean_speed": path_len / dur if dur > 0 else float("nan"),
        "cells": cells,
        "n_cells_nodes": len(cells),        # 通過した区画の個数
        "n_cells_moves": n_moves,           # 区画間の移動回数
        "distinct_cells": len(set(cells)),
        "n_turns_r4": turn_count_r4(cells),
        "non_adjacent_transitions": non_adjacent,
    }


# ----------------------------------------------------------------- 指標 (§2)
def maze_metrics(runs: list[dict], d_true: int) -> dict:
    """1 面ぶんの (a)〜(e') を §2 の定義文から計算する。

    runs は走行順のリスト。各要素は独立再計算済みの
    {outcome, run_time, path_length_m, mean_speed, n_cells_nodes} を持つ。
    """
    goal_idx = [i for i, r in enumerate(runs) if r["outcome"] == "goal"]
    res: dict = {
        "n_runs": len(runs),
        "n_goal_runs": len(goal_idx),
        "a_goal_reached": bool(goal_idx),
    }
    if not goal_idx:
        res.update({"b_fast_run": False, "c_effective": None, "d_best_time": None})
        return res

    i_exp = goal_idx[0]                       # 探索走行 = 初めてゴールへ到達した走行
    post = [i for i in goal_idx if i > i_exp]  # 初回ゴール「より後に開始」して到達した走行
    res["explore_run_index"] = i_exp + 1
    res["n_post_goal_runs"] = len(post)
    res["b_fast_run"] = bool(post)

    t_exp = runs[i_exp]["run_time"]
    res["t_explore"] = t_exp
    res["d_best_time"] = min(runs[i]["run_time"] for i in goal_idx)   # (d) 完走走行の最速値

    if not post:
        res.update({"c_shorten_rate": None, "c_effective": None,
                    "c1_path": None, "c2_speed": None,
                    "e_first_fast_eff": None, "e_status": "no_post_run",
                    "e2_path_eff": None})
        return res

    i_best = min(post, key=lambda i: runs[i]["run_time"])
    t_best = runs[i_best]["run_time"]
    res["t_fast_best"] = t_best
    res["fast_run_index"] = i_best + 1

    # (c) 有効最短走行率と短縮率
    s = 1.0 - t_best / t_exp
    res["c_shorten_rate"] = s
    res["c_effective"] = bool(s >= 0.10)

    # (c1) 経路短縮率 / (c2) 速度向上率
    c1 = 1.0 - runs[i_best]["path_length_m"] / runs[i_exp]["path_length_m"]
    c2 = runs[i_best]["mean_speed"] / runs[i_exp]["mean_speed"] - 1.0
    res["c1_path"] = c1
    res["c2_speed"] = c2
    res["identity_residual"] = s - (1.0 - (1.0 - c1) / (1.0 + c2))

    # (e) 初回最短走行効率 = 初回の最短走行タイム / その面の最良タイム
    #   分母は §2 を正とする（全ゴール走行にわたる最良タイム。裁定 R15）。
    #   凍結ハーネスの first_fast_efficiency は分母が「探索後の走行のうち最速」なので
    #   別量として (e-harness) の名で併記する。
    i_first_post = post[0]
    t_first_post = runs[i_first_post]["run_time"]
    t_best_all = res["d_best_time"]
    e = t_first_post / t_best_all
    # §2 の退化ガード: 探索後の走行が 1 回だけで、かつ最良タイムがその走行 → 未定義
    degenerate = (len(post) == 1) and (abs(t_first_post - t_best_all) < 1e-12)
    res["e_first_fast_eff"] = None if degenerate else e
    res["e_raw"] = e
    res["e_status"] = "degenerate" if degenerate else "defined"
    # (e-harness): 分母 = 探索後の走行のうち最速（凍結ハーネスの定義）
    e_h = t_first_post / t_best
    res["e_harness"] = e_h
    res["e_defs_differ"] = bool(abs(e - e_h) > 1e-12)

    # (e') 初回最短走行の経路効率 = 通過した区画数 / D_0
    n_nodes = runs[i_first_post]["n_cells_nodes"]
    res["e2_path_eff"] = n_nodes / d_true
    res["e2_nodes"] = n_nodes
    res["e2_alt_moves"] = (n_nodes - 1) / d_true      # 移動回数で数えた場合（感度）
    return res


# ------------------------------------------------------------------ 集計・出力
def q(vals: list[float]) -> dict:
    v = sorted(x for x in vals if x is not None and not math.isnan(x))
    if not v:
        return {"n": 0}
    return {
        "n": len(v), "min": v[0], "max": v[-1],
        "median": st.median(v),
        "q1": np.percentile(v, 25), "q3": np.percentile(v, 75),
        "mean": st.fmean(v),
    }


def process_arm(arm_dir: Path) -> dict:
    detail = json.loads((arm_dir / "runs_detail.json").read_text())
    recorded = detail["runs"]
    mazes = sorted({r["maze"] for r in recorded})

    per_maze: dict = {}
    mismatches: list[dict] = []
    checked = 0

    for mz in mazes:
        m = load_maze(MAZE_DIR / f"{mz}.npz")
        dist = bfs_from_goals(m)
        z = np.load(arm_dir / "traj" / f"{mz}.npz")
        t = z["t"]
        x = z["x"].astype(float)
        y = z["y"].astype(float)
        st_cell = cell_of(float(x[0]), float(y[0]))
        d0 = dist[st_cell[0]][st_cell[1]]

        rec_runs = [r for r in recorded if r["maze"] == mz]
        rec_runs.sort(key=lambda r: r["run"])
        runs = []
        for k, ridx in enumerate(z["run_index"].tolist()):
            g = run_geometry(t, x, y, float(z["run_t_start"][k]), float(z["run_t_end"][k]))
            g["outcome"] = str(z["run_outcome"][k])
            g["run_time"] = g["duration"]
            g["run"] = int(ridx)
            runs.append(g)

            # ---- 記録の忠実性の照合（計算には使わない）
            rr = next((r for r in rec_runs if r["run"] == int(ridx)), None)
            if rr is not None:
                for key, mine, theirs, tol in [
                    ("run_time", g["run_time"], rr["run_time"], 1e-9),
                    ("path_length_m", g["path_length_m"], rr["path_length_m"], 1e-6),
                    ("mean_speed", g["mean_speed"], rr["mean_speed"], 1e-6),
                    ("n_cells", g["n_cells_moves"], rr["n_cells"], 0),
                    ("n_turns", g["n_turns_r4"], rr["n_turns"], 0),
                    ("visited_cells", g["distinct_cells"], rr["visited_cells"], 0),
                    ("outcome", g["outcome"], rr["outcome"], None),
                    ("d_true", d0, rr["d_true"], 0),
                ]:
                    checked += 1
                    ok = (mine == theirs) if tol is None or tol == 0 else \
                         (abs(mine - theirs) <= tol * max(1.0, abs(theirs)))
                    if not ok:
                        mismatches.append({"maze": mz, "run": int(ridx), "field": key,
                                           "independent": mine, "recorded": theirs})

        mm = maze_metrics(runs, d0)
        mm["d_true"] = d0
        mm["start_cell"] = list(st_cell)
        mm["start_candidates"] = [list(c) for c in start_candidates(m)]
        mm["per_run"] = [{k: v for k, v in r.items() if k != "cells"} for r in runs]
        per_maze[mz] = mm

    n = len(mazes)
    a_n = sum(1 for v in per_maze.values() if v["a_goal_reached"])
    b_n = sum(1 for v in per_maze.values() if v["b_fast_run"])
    c_pool = [v for v in per_maze.values() if v["b_fast_run"]]
    c_n = sum(1 for v in c_pool if v["c_effective"])

    e_defined = [v["e_first_fast_eff"] for v in per_maze.values()
                 if v.get("e_status") == "defined"]
    e_degen = [k for k, v in per_maze.items() if v.get("e_status") == "degenerate"]
    e_none = [k for k, v in per_maze.items() if v.get("e_status") == "no_post_run"]

    summary = {
        "arm": detail.get("arm"),
        "policy": detail.get("policy"),
        "maze_dir": detail.get("maze_dir"),
        "n_mazes": n,
        "a_goal_rate": a_n / n,
        "a_count": [a_n, n],
        "b_fast_rate": b_n / n,
        "b_count": [b_n, n],
        "b_post_run_dist": q([v.get("n_post_goal_runs", 0) for v in per_maze.values()]),
        "c_effective_rate": (c_n / len(c_pool)) if c_pool else None,
        "c_count": [c_n, len(c_pool)],
        "c_shorten_rate_dist": q([v["c_shorten_rate"] for v in c_pool]),
        "c1_path_dist": q([v["c1_path"] for v in c_pool]),
        "c2_speed_dist": q([v["c2_speed"] for v in c_pool]),
        "identity_max_abs_residual": max(
            (abs(v["identity_residual"]) for v in c_pool), default=None),
        "d_best_time_dist": q([v["d_best_time"] for v in per_maze.values()]),
        "t_explore_dist": q([v.get("t_explore") for v in per_maze.values()]),
        "t_fast_dist": q([v.get("t_fast_best") for v in per_maze.values()]),
        "e_dist": q(e_defined),
        "e_defined_count": len(e_defined),
        "e_degenerate_mazes": e_degen,
        "e_no_post_run_mazes": e_none,
        "e_harness_dist": q([v.get("e_harness") for v in per_maze.values()]),
        "e_defs_differ_mazes": [k for k, v in per_maze.items() if v.get("e_defs_differ")],
        "e2_path_eff_dist": q([v.get("e2_path_eff") for v in per_maze.values()]),
        "e2_alt_moves_dist": q([v.get("e2_alt_moves") for v in per_maze.values()]),
        "d_true_dist": q([v["d_true"] for v in per_maze.values()]),
        "recon_checked": checked,
        "recon_mismatches": mismatches,
    }
    return {"summary": summary, "per_maze": per_maze}


def main() -> None:
    arms = sys.argv[1:] or sorted(
        p.name for p in OUT_ROOT.iterdir()
        if p.is_dir() and (p / "runs_detail.json").exists())
    out: dict = {"git_head": None, "arms": {}}
    try:
        import subprocess
        out["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        pass

    for arm in arms:
        d = OUT_ROOT / arm
        print(f"=== {arm} ===")
        r = process_arm(d)
        out["arms"][arm] = r
        s = r["summary"]
        print(f"  (a) {s['a_goal_rate']:.0%} {s['a_count']}   "
              f"(b) {s['b_fast_rate']:.0%} {s['b_count']}   "
              f"(c) {s['c_effective_rate']:.0%} {s['c_count']}")
        print(f"  (d) 最速タイム中央値 {s['d_best_time_dist']['median']:.2f} s "
              f"[{s['d_best_time_dist']['min']:.2f}, {s['d_best_time_dist']['max']:.2f}]")
        print(f"  探索 {s['t_explore_dist']['median']:.2f} s / 最短 {s['t_fast_dist']['median']:.2f} s")
        print(f"  (c) 短縮率中央値 {s['c_shorten_rate_dist']['median']:.4f}  "
              f"(c1) {s['c1_path_dist']['median']:.4f}  (c2) {s['c2_speed_dist']['median']:.4f}  "
              f"恒等式残差 max {s['identity_max_abs_residual']:.2e}")
        ed = s["e_dist"]
        print(f"  (e) n={ed.get('n')} 中央値 {ed.get('median')}  退化 {len(s['e_degenerate_mazes'])} 面")
        print(f"  (e') 中央値 {s['e2_path_eff_dist']['median']:.4f} "
              f"[{s['e2_path_eff_dist']['min']:.4f}, {s['e2_path_eff_dist']['max']:.4f}]")
        print(f"  照合 {s['recon_checked']} 項目 / 不一致 {len(s['recon_mismatches'])} 件")
        for mm in s["recon_mismatches"][:10]:
            print("    ", mm)

    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    print(f"\n書き出し: {RESULT_JSON}")


if __name__ == "__main__":
    main()
