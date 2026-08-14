#!/usr/bin/env python3
"""exp_019: v2 環境での「狭い周回か・遅い這いか」の探索的測定。

Exploratory measurement (NOT part of the pre-registered judgment).

🔴 **これは判定外・事前登録なしの探索的測定である。**判定文書（`judgment.md`）には
影響しない。結果は `revisit_v2_memo.md` に記録する。

## 何を測るか（**量を名指しする** — §9-17）

`check_m2_failure_mode.py` の `revisit_ratio` は **総歩数 ÷ 異なる区画数**である
（`cells[cell] += 1` を毎歩行い、その総和を異なる区画数で割っている）。
**この量は「狭い周回」と「遅い這い」を区別しない** — **どちらでも大きくなる**。

区別するには**区画に入った回数**が要る:

| 量 | 定義 | 周回のとき | 這いのとき |
|---|---|---|---|
| `steps_per_distinct_cell` | 総歩数 ÷ 異なる区画数 | 大 | **大**（区別できない） |
| `cell_entries` | **区画が変わった歩の数 ＋ 1** | **大** | 小 |
| `entries_per_distinct_cell` | `cell_entries` ÷ 異なる区画数 | **大** | **小** ← **判別できる** |
| `steps_per_entry` | 総歩数 ÷ `cell_entries` | 小（≈ 通過に要る歩数） | **大**（長く留まる） |

**基準値**: 公称速度 0.96 m/s・`control_dt` 0.01 s なら **1 区画の通過は 18.75 歩**。
**`steps_per_entry` がこれに近ければ「まともな速度で動いている」**、
**大きく上回れば「遅い」**と読める。

## 限界（**これで確定はしない**）

- **`cell_entries` は区画境界の往復も数える**（壁際で行き来すると増える）
- **`steps_per_entry` は 1 区画の平均滞在歩数であって、進行距離そのものではない。**
  **本判別の正本は R51-2（歩あたりの進行距離）である**
- **母集団は最終方策の rollout** であり、学習中の全走行ではない（P5 と同じ限界）

## 構成（**P5 の rollout と同一**にしてある）

環境・seed 規約とも `measure_p5.py` と同じなので、**同じ面では同じ軌跡になる**
（`_trial_seed(0, maze_seed, 0)`・面ごとに env を作り直す・`deterministic=True`）。
したがって **P5 の母集団（リスポーン経験あり）との対応がそのまま取れる**。

使い方:
    .venv/bin/python experiments/exp_019_env_v2_baseline/measure_revisit_v2.py \
        --models models/exp_019_v2_seed{1,2,3,4,5,6}.zip --out outputs/exp_019_revisit_v2.json
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import PPO  # noqa: E402

from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402
from mouse.maze6_env import Maze6Env  # noqa: E402
from mouse.maze6_eval import VALIDATION_MAZE_DIR, _trial_seed  # noqa: E402

GAMMA = 0.995
_SEED_BASE, _TRIAL_IDX = 0, 0          # measure_p5.py と同一
_ENV_KWARGS = dict(                    # measure_p5.py と同一（学習環境 v2 ＋ 条件 E）
    continuous_potential=True,
    goal_rule_containment=True,
    collision_respawn=True,
    episode_limit_steps=2000,
)
#: 公称速度 0.96 m/s・control_dt 0.01 s での 1 区画の通過歩数（読みの基準）
_STEPS_PER_CELL_NOMINAL = 0.18 / (0.96 * 0.01)


def _run(maze_dir: Path, maze_seed: int, policy_fn) -> dict:
    """1 面 1 エピソード。区画の**滞在歩数**と**入場回数**を分けて数える。"""
    env = Maze6Env(maze_dir=maze_dir, maze_seeds=[maze_seed], max_cache=2,
                   gamma=GAMMA, mode="fixed", maze_mode="loop", **_ENV_KWARGS)
    tseed = _trial_seed(_SEED_BASE, maze_seed, _TRIAL_IDX)
    obs, info = env.reset(seed=tseed)
    d0 = int(info["dist_to_goal"])
    d_min = d0
    prev_cell = tuple(info["cell"])
    distinct = {prev_cell}
    entries = 1                        # 開始区画への入場を 1 と数える
    n_steps, n_respawn = 0, 0
    while True:
        a = np.clip(np.asarray(policy_fn(obs), dtype=np.float64), -1.0, 1.0)
        obs, _r, terminated, truncated, info = env.step(a)
        n_steps += 1
        cell = tuple(info["cell"])
        if cell != prev_cell:
            entries += 1               # **区画が変わった歩 ＝ 入場**
            prev_cell = cell
        distinct.add(cell)
        d = int(info["dist_to_goal"])
        if d >= 0:
            d_min = min(d_min, d)
        n_respawn = int(info.get("n_respawn", 0))
        if terminated:
            outcome = "goal" if info.get("goal") else "collision"
            break
        if truncated:
            outcome = "timeout"
            break
    if hasattr(env, "close"):
        env.close()
    n_uniq = len(distinct)
    return dict(
        maze_seed=maze_seed, trial_seed=int(tseed), outcome=outcome,
        d0=d0, d_min=d_min, n_steps=n_steps, n_respawn=n_respawn,
        n_distinct_cells=n_uniq, cell_entries=entries,
        steps_per_distinct_cell=n_steps / max(n_uniq, 1),
        entries_per_distinct_cell=entries / max(n_uniq, 1),
        steps_per_entry=n_steps / max(entries, 1),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="v2 環境での周回/這いの探索的測定（判定外）")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--maze-dir", type=str, default=VALIDATION_MAZE_DIR)
    p.add_argument("--out", type=str, default="outputs/exp_019_revisit_v2.json")
    args = p.parse_args()

    maze_dir = Path(args.maze_dir)
    maze_seeds = sorted(int(np.load(q)["seed"]) for q in maze_dir.glob("maze6_*.npz"))
    print(describe_seeds(maze_seeds, "maze6"))
    assert_seeds_allowed(maze_seeds, "maze6", "validate")
    print(f"  基準: 公称速度での 1 区画の通過 = {_STEPS_PER_CELL_NOMINAL:.2f} 歩")

    per_seed, rows = {}, []
    for m in args.models:
        name = Path(m).stem
        model = PPO.load(str(m), device="cpu")
        pol = lambda o: model.predict(o, deterministic=True)[0]   # noqa: E731
        eps = [_run(maze_dir, ms, pol) for ms in maze_seeds]
        per_seed[name] = eps
        rows += eps
        med = lambda k: statistics.median([e[k] for e in eps])    # noqa: E731
        print(f"[{name}] 中央値: 滞在/区画 {med('steps_per_distinct_cell'):6.1f} 歩  "
              f"入場/区画 {med('entries_per_distinct_cell'):5.1f} 回  "
              f"1 入場あたり {med('steps_per_entry'):6.1f} 歩  "
              f"異なる区画 {med('n_distinct_cells'):.0f}")

    # 走行の種別ごと（P5 の母集団との対応が取れる形）
    def summarize(sel, label):
        s = [e for e in rows if sel(e)]
        if not s:
            return dict(label=label, n=0)
        return dict(label=label, n=len(s),
                    steps_per_entry_median=statistics.median(
                        [e["steps_per_entry"] for e in s]),
                    entries_per_cell_median=statistics.median(
                        [e["entries_per_distinct_cell"] for e in s]),
                    distinct_cells_median=statistics.median(
                        [e["n_distinct_cells"] for e in s]))

    groups = [summarize(lambda e: e["n_respawn"] == 0 and e["outcome"] != "goal",
                        "リスポーン 0 回・非ゴール"),
              summarize(lambda e: e["n_respawn"] >= 1, "リスポーン 1 回以上"),
              summarize(lambda e: e["outcome"] == "goal", "ゴール")]
    print("\n=== 走行の種別ごと（中央値）===")
    for g in groups:
        if g["n"] == 0:
            print(f"  {g['label']}: n=0")
            continue
        print(f"  {g['label']}: n={g['n']}  1 入場あたり {g['steps_per_entry_median']:.1f} 歩  "
              f"入場/区画 {g['entries_per_cell_median']:.1f} 回  "
              f"異なる区画 {g['distinct_cells_median']:.0f}")

    out = dict(
        note="判定外・事前登録なしの探索的測定。judgment.md には影響しない",
        env_kwargs=_ENV_KWARGS,
        seed_rule=dict(fn="mouse.maze6_eval._trial_seed", base=_SEED_BASE,
                       trial_idx=_TRIAL_IDX, note="measure_p5.py と同一 = 同じ軌跡"),
        steps_per_cell_nominal=_STEPS_PER_CELL_NOMINAL,
        groups=groups, per_seed=per_seed,
    )
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"→ {op}")


if __name__ == "__main__":
    main()
