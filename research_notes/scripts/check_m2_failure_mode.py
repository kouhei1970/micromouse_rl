"""
research_notes/scripts/check_m2_failure_mode.py
===============================================
**M2-0 が何をして失敗しているのか**を、失敗走行の中身から直接見る（学習は回さない）。

`maze6_eval.py` の集計は**ゴールした走行のみ**なので、ゴール率 0 のときは全部 nan になり
何も分からない。本スクリプトは**失敗走行そのもの**を調べる:

- **ゴールにどこまで近づいたか**（`dist_to_goal` の最小値）。到達していないのか、
  惜しいところまで行っているのか
- **何区画を訪問したか**（探索しているのか、同じところを回っているのか）
- **オドメトリ誤差**（観測に入るゴール相対位置がどれだけ嘘か）
- **軌跡が閉じているか**（同じ区画を何度も通っているか ＝ 周回している）

使い方:
    .venv/bin/python research_notes/scripts/check_m2_failure_mode.py \
        --model models/exp_010_m2_0_seed1.zip
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from mouse.maze6_env import Maze6Env  # noqa: E402
from common.seed_bands import describe_seeds  # noqa: E402
from mouse.maze6_eval import VALIDATION_MAZE_DIR, _trial_seed  # noqa: E402


def diagnose(model, maze_dir, maze_seed, tseed):
    env = Maze6Env(maze_dir=maze_dir, maze_seeds=[maze_seed], max_cache=2,
                   mode="fixed", gamma=0.995)
    obs, info = env.reset(seed=tseed)
    d0 = info["dist_to_goal"]
    dmin = d0
    cells = collections.Counter([tuple(env.maze["start"])])
    odom = []
    n = 0
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(a)
        n += 1
        d = info["dist_to_goal"]
        if d >= 0:
            dmin = min(dmin, d)
        cells[tuple(info["cell"])] += 1
        odom.append(info["odom_error_m"])
        if term or trunc:
            break
    outcome = ("goal" if info.get("goal") else
               "collision" if info.get("collision") else "timeout")
    n_uniq = len(cells)
    revisit = sum(v for v in cells.values()) / max(n_uniq, 1)
    return dict(maze_seed=maze_seed, outcome=outcome, n_steps=n,
                d_start=d0, d_min=dmin, n_visited=n_uniq,
                revisit_ratio=revisit,
                odom_final_mm=odom[-1] * 1000 if odom else float("nan"),
                odom_max_mm=max(odom) * 1000 if odom else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exp_010_m2_0_seed1.zip")
    ap.add_argument("--n-trials", type=int, default=1)
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    seeds = sorted(int(np.load(p)["seed"])
                   for p in Path(VALIDATION_MAZE_DIR).glob("maze6_*.npz"))
    # 帯の明示（R40 条件 4 (a)）。本スクリプトは帯を CLI で選べない（検証帯に固定）ので
    # 拒否の安全弁 (b) は要らないが、**どの帯で測ったかがログに残らない**のは別の問題である。
    # 「固定しているつもり」と「実際に固定されている」の差は、ログが無いと後から確認できない。
    print(describe_seeds(seeds, "maze6"))
    rows = [diagnose(model, VALIDATION_MAZE_DIR, ms, _trial_seed(0, ms, t))
            for ms in seeds for t in range(args.n_trials)]

    print(f"=== {args.model} / 検証帯 {len(seeds)} 面 ×{args.n_trials} 試行 ===")
    print(f"{'迷路':>7}{'結果':>10}{'歩数':>7}{'最短d':>7}{'最接近d':>8}"
          f"{'訪問':>6}{'再訪率':>8}{'odom最終[mm]':>14}{'odom最大[mm]':>14}")
    for r in rows:
        print(f"{r['maze_seed']:>7}{r['outcome']:>10}{r['n_steps']:>7}{r['d_start']:>7}"
              f"{r['d_min']:>8}{r['n_visited']:>6}{r['revisit_ratio']:>8.1f}"
              f"{r['odom_final_mm']:>14.0f}{r['odom_max_mm']:>14.0f}")

    n = len(rows)
    print("\n" + "=" * 78)
    print("要約")
    print("=" * 78)
    c = collections.Counter(r["outcome"] for r in rows)
    print(f"  結果の内訳: {dict(c)}  （n={n}）")
    reached = [r for r in rows if r["d_min"] == 0]
    print(f"  **ゴール区画に入った走行: {len(reached)} / {n}**"
          f"（環境は真の位置で終端するので、入れば必ず goal になる）")
    print(f"  最接近距離 d_min: 平均 {np.mean([r['d_min'] for r in rows]):.1f} 歩 / "
          f"最小 {min(r['d_min'] for r in rows)} / 最大 {max(r['d_min'] for r in rows)}")
    print(f"  出発時の距離 d_start: 平均 {np.mean([r['d_start'] for r in rows]):.1f} 歩")
    近づけた = [r for r in rows if r["d_min"] < r["d_start"]]
    print(f"  出発時より近づけた走行: {len(近づけた)} / {n}")
    print(f"  訪問区画数: 平均 {np.mean([r['n_visited'] for r in rows]):.1f} / 36")
    print(f"  再訪率（延べ/実数）: 平均 {np.mean([r['revisit_ratio'] for r in rows]):.1f} 倍")
    print(f"  オドメトリ誤差 最終: 平均 {np.mean([r['odom_final_mm'] for r in rows]):.0f} mm"
          f" ／ 最大 {max(r['odom_max_mm'] for r in rows):.0f} mm"
          f"（区画 180 mm）")
    to = [r for r in rows if r["outcome"] == "timeout"]
    if to:
        print(f"\n  時間切れ {len(to)} 本の内訳: 訪問 平均 {np.mean([r['n_visited'] for r in to]):.1f} 区画、"
              f"再訪率 平均 {np.mean([r['revisit_ratio'] for r in to]):.1f} 倍、"
              f"最接近 平均 {np.mean([r['d_min'] for r in to]):.1f} 歩")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
