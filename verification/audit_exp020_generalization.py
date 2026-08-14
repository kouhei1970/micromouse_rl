#!/usr/bin/env python3
"""監査 038: 汎化ギャップは実在するか — 学習に使った迷路での対照試験

**事前登録**: `verification/AUDIT_038_PREREG_generalization.md`（**実行前にコミット済み**）
**判定形は凍結してある。**

`AUDIT_037`（検証用の 20 迷路）と**変えるのは 1 点だけ** — **迷路を「学習に使ったもの」に替える。**
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_exp019_goal_vs_d0 import d0_of  # noqa: E402  自前 BFS

from mouse.maze6_env import Maze6Env  # noqa: E402
from mouse.maze6_eval import _trial_seed  # noqa: E402
from mouse.maze6_gen import generate_maze  # noqa: E402

SEEDS = [1, 2, 3, 4, 5, 6]
N_TRIALS = 20                      # 事前登録
CUT = 8                            # 事前登録: 8 件以上なら汎化ギャップ実在
MODEL = "models/exp_020_seed{n}.zip"
# 事前登録に記録した 20 迷路（実行前に確定）
MAZES = [8000, 8012, 8015, 9707, 9709, 9740, 10160, 10161, 10171, 10173,
         10177, 10399, 10404, 10415, 10419, 10421, 10445, 10532, 10589, 10639]


def run(model, ms: int, t: int) -> bool:
    # 🔴 AUDIT_037 と**まったく同じ構成**にする（mode="fixed" ＋ maze_seeds=[ms]）。
    # mode="generate" にすると `_next_maze_seed()` が np_random を消費せず、
    # 後続の初期擾乱（横 ±0.02 m・方位 ±10°）の引きがまるごとずれる。
    # 迷路は maze_dir に関係なく seed から生成されるので、これで学習用迷路を回せる。
    # ⚠️ `mode="fixed"` は `maze_dir` を必須にするが、**`self.maze_dir` は代入されるだけで
    # 一度も読まれない**（`mouse/maze6_env.py`:356 が唯一の出現）。迷路は `_load_maze()` が
    # **seed から生成する**ので、ここに渡すディレクトリは結果に影響しない。
    # **AUDIT_037 と同じ値を渡して、構成を完全に一致させる。**
    env = Maze6Env(maze_dir="assets/maze6/loop/validation", maze_seeds=[ms],
                   max_cache=2, mode="fixed", maze_mode="loop",
                   goal_rule_containment=True)
    obs, info = env.reset(seed=_trial_seed(0, ms, t))
    for _ in range(6000):
        a = model.predict(obs, deterministic=False)[0]
        obs, _r, term, trunc, info = env.step(a)
        if term or trunc:
            return bool(info["goal"])
    return False


def main() -> int:
    from stable_baselines3 import PPO
    d0 = {m: d0_of(generate_maze(m, mode="loop")) for m in MAZES}
    # 学習中の出題回数（全 seed 共通の迷路列なので seed1 で数える）
    exposure = collections.Counter(
        json.loads(l)["maze_seed"]
        for l in Path("logs/exp_020_seed1/episode_seeds.jsonl").open())

    rows, total = [], 0
    for n in SEEDS:
        model = PPO.load(MODEL.format(n=n), device="cpu")
        g, hits = 0, []
        for ms in MAZES:
            for t in range(N_TRIALS):
                if run(model, ms, t):
                    g += 1
                    hits.append(dict(maze=ms, d0=d0[ms], trial=t))
        total += g
        rows.append(dict(seed=n, n_ep=len(MAZES) * N_TRIALS, n_goal=g, hits=hits))
        print(f"  seed{n}: {len(MAZES)*N_TRIALS} エピソード中 ゴール {g} 件"
              f"{'  ' + str([(h['maze'], h['d0']) for h in hits]) if hits else ''}")

    print("=" * 74)
    print(f"合計 {total} 件 / {sum(r['n_ep'] for r in rows)} エピソード")
    print(f"  事前登録の期待: 方策が弱いだけ = 4.0 件 / 汎化ギャップ実在 = 12.4 件")
    verdict = ("汎化ギャップが実在する" if total >= CUT else "汎化ギャップは無い（方策が弱いだけ）")
    print(f"  → **{verdict}**（境目 {CUT} 件）")
    print(f"  対比: 検証用の 20 迷路（AUDIT_037）は 1,200 エピソード中 2 件"
          f"（成功率 0.00167）／本試験は {total/2400:.5f}")
    print("  学習中の出題回数（選んだ 20 迷路）: "
          f"中央値 {sorted(exposure[m] for m in MAZES)[len(MAZES)//2]} 回・"
          f"範囲 {min(exposure[m] for m in MAZES)}〜{max(exposure[m] for m in MAZES)} 回")
    print("=" * 74)

    out = Path(__file__).resolve().parent / "out" / "exp020_generalization.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(total=total, cut=CUT, verdict=verdict,
                                   n_trials=N_TRIALS, mazes=MAZES,
                                   d0={str(k): v for k, v in d0.items()},
                                   exposure={str(m): exposure[m] for m in MAZES},
                                   per_seed=rows), ensure_ascii=False, indent=2))
    print(f"出力: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
