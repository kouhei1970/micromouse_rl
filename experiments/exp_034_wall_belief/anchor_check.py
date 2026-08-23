"""
experiments/exp_034_wall_belief/anchor_check.py
================
exp_034（壁の信念）の錨の独立再計算（PREREG.md §5、`docs/RESEARCH_PLAN.md` §12-9）。

入力は `run.py` が保存した一次記録（`outputs/exp_034_wall_belief/raw/maze_<seed>_final_state.npz`:
柱間ごとの最終対数オッズ・真の壁配列・真の壁配列と、run.py 自身が計算した宣言配列）だけ。

🔴 本ファイルは `run.py` の集計出力（`outputs/exp_034_wall_belief/maze_*.json`・
`summary.json`）を一切読まない・importしない。「宣言」（WALL/OPEN/UNKNOWN）は
run.py が書き出した値をそのまま信用せず、ここで対数オッズと
`classic.wall_belief` の**公開しきい値定数**（T_WALL_DEFAULT・T_OPEN_DEFAULT。
これは対象モジュール自身の定数であり run.py の出力ではない）から自分で
導出し直す。run.py が書き出した declared_v/h は「run.py 自身の宣言計算が
正しかったか」を確かめるクロスチェックとしてのみ使う。

$N_{fatal}$（真は壁なのに開通と宣言した柱間の数、20迷路合計）と宣言数を
自分で数え直し、`run.py` が出した値と一致するかどうかを確かめる作業自体は、
本ファイルの外（両者の JSON を読み比べる別の手順）で行う。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

# classic/wall_belief.py の公開しきい値定数(対象モジュール自身のもの。run.pyの
# 出力ではない)。WallState も同様に classic/maze_map.py 自身の定義を使う。
from classic.maze_map import WallState  # noqa: E402
from classic.wall_belief import T_OPEN_DEFAULT, T_WALL_DEFAULT  # noqa: E402

RAW_DIR = REPO_ROOT / "outputs" / "exp_034_wall_belief" / "raw"
OUT_PATH = REPO_ROOT / "outputs" / "exp_034_wall_belief" / "anchor_check_results.json"


def _declare(log_odds: np.ndarray, t_wall: float, t_open: float) -> np.ndarray:
    """`classic.wall_belief.declare_state` と同じ非対称しきい値を、ここで
    ゼロから配列へ適用する（run.py の `declared_array` は import しない
    ── 同じ関数を再利用すると、その関数にバグがあった場合に集計側の誤りを
    集計側でもう一度確認しているだけになってしまうため）。"""
    out = np.zeros(log_odds.shape, dtype=np.int8)
    out[log_odds > t_wall] = int(WallState.WALL)
    out[log_odds < -t_open] = int(WallState.OPEN)
    return out


def recompute_for_file(npz_path: Path) -> dict:
    d = np.load(npz_path)
    seed = int(d["seed"])
    log_odds_v = d["log_odds_v"]
    log_odds_h = d["log_odds_h"]
    true_v = d["true_v"]
    true_h = d["true_h"]
    stored_declared_v = d["declared_v"]
    stored_declared_h = d["declared_h"]
    conv_v = d["conv_v"]
    conv_h = d["conv_h"]

    declared_v = _declare(log_odds_v, T_WALL_DEFAULT, T_OPEN_DEFAULT)
    declared_h = _declare(log_odds_h, T_WALL_DEFAULT, T_OPEN_DEFAULT)

    cross_check_ok = bool(np.array_equal(declared_v, stored_declared_v)
                           and np.array_equal(declared_h, stored_declared_h))

    true_wall_v = true_v != 0
    true_wall_h = true_h != 0

    n_fatal = (int(np.count_nonzero((declared_v == int(WallState.OPEN)) & true_wall_v))
               + int(np.count_nonzero((declared_h == int(WallState.OPEN)) & true_wall_h)))
    n_benign = (int(np.count_nonzero((declared_v == int(WallState.WALL)) & (~true_wall_v)))
                + int(np.count_nonzero((declared_h == int(WallState.WALL)) & (~true_wall_h))))
    n_declared = (int(np.count_nonzero(declared_v != int(WallState.UNKNOWN)))
                  + int(np.count_nonzero(declared_h != int(WallState.UNKNOWN))))
    n_declared_conventional = (int(np.count_nonzero(conv_v != int(WallState.UNKNOWN)))
                                + int(np.count_nonzero(conv_h != int(WallState.UNKNOWN))))
    n_fatal_conventional = (int(np.count_nonzero((conv_v == int(WallState.OPEN)) & true_wall_v))
                             + int(np.count_nonzero((conv_h == int(WallState.OPEN)) & true_wall_h)))

    return {
        "seed": seed,
        "n_fatal": n_fatal,
        "n_benign": n_benign,
        "n_declared": n_declared,
        "n_declared_conventional": n_declared_conventional,
        "n_fatal_conventional": n_fatal_conventional,
        "declared_below_conventional": bool(n_declared < n_declared_conventional),
        "declared_array_matches_run_py": cross_check_ok,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)

    files = sorted(args.raw_dir.glob("maze_*_final_state.npz"),
                    key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise RuntimeError(f"{args.raw_dir} に一次記録(maze_*_final_state.npz)が見つからない")

    results = [recompute_for_file(f) for f in files]

    for r in results:
        print(f"[anchor_check] seed={r['seed']:>5} n_fatal={r['n_fatal']:>2} "
              f"n_declared={r['n_declared']:>3}(従来={r['n_declared_conventional']:>3}) "
              f"cross_check={r['declared_array_matches_run_py']}")

    n_fatal_total = sum(r["n_fatal"] for r in results)
    n_declared_total = sum(r["n_declared"] for r in results)
    mazes_below_conventional = [r["seed"] for r in results if r["declared_below_conventional"]]
    all_cross_check_ok = all(r["declared_array_matches_run_py"] for r in results)

    print(f"\n[anchor_check] 迷路数 = {len(results)}")
    print(f"[anchor_check] N_fatal 合計(再計算) = {n_fatal_total}")
    print(f"[anchor_check] 宣言数合計(再計算) = {n_declared_total}")
    print(f"[anchor_check] 従来の3値地図を下回った迷路 = {mazes_below_conventional}")
    print(f"[anchor_check] 全迷路でrun.pyの宣言配列と一致 = {all_cross_check_ok}")

    out = {
        "n_files": len(results),
        "per_maze": results,
        "n_fatal_total": n_fatal_total,
        "n_declared_total": n_declared_total,
        "mazes_below_conventional": mazes_below_conventional,
        "all_cross_check_ok": all_cross_check_ok,
        "note": "run.py の集計出力(maze_*.json/summary.json)は一切読んでいない。"
                "宣言はここで log_odds + classic.wall_belief の公開しきい値定数から"
                "独立に導出し直した(run.py が書き出した declared_v/h はクロスチェックにのみ使用)。",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"[anchor_check] 保存: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
