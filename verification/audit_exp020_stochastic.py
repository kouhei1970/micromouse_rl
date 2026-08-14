#!/usr/bin/env python3
"""監査 037: 検証用迷路のゴールがゼロなのは「方策の確率性の違い」か

**事前登録**: `verification/AUDIT_037_PREREG_stochastic.md`（**実行前にコミット済み**）
**判定形は凍結してある。**

変えるのは 1 点だけ — **`deterministic=False`（確率的な方策）**。
環境・迷路・初期姿勢の固定はすべて通常の定期評価と同じ。

使い方: `.venv/bin/python verification/audit_exp020_stochastic.py`
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_exp019_p5 import d0_of  # noqa: E402  自前 BFS を流用

from mouse.maze6_env import Maze6Env  # noqa: E402
from mouse.maze6_eval import _trial_seed  # noqa: E402

SEEDS = [1, 2, 3, 4, 5, 6]
N_TRIALS = 10                      # 事前登録: 1 迷路あたり 10 回
VALID_DIR = "assets/maze6/loop/validation"
MODEL = "models/exp_020_seed{n}.zip"
EXPECTED = 6.2                     # 事前登録した期待件数


def run(model, ms: int, t: int) -> dict:
    env = Maze6Env(maze_dir=VALID_DIR, maze_seeds=[ms], max_cache=2,
                   mode="fixed", maze_mode="loop", goal_rule_containment=True)
    obs, info = env.reset(seed=_trial_seed(0, ms, t))
    for _ in range(6000):          # 評価環境の既定の上限
        a = model.predict(obs, deterministic=False)[0]   # ← ここだけが違う
        obs, _r, term, trunc, info = env.step(a)
        if term or trunc:
            return dict(goal=bool(info["goal"]), steps=env._step_count)
    return dict(goal=False, steps=6000)


def main() -> int:
    from stable_baselines3 import PPO
    faces = sorted(d0_of(p) for p in glob.glob(f"{VALID_DIR}/*.npz"))
    rows, total = [], 0
    for n in SEEDS:
        model = PPO.load(MODEL.format(n=n), device="cpu")
        g = 0
        hits = []
        for ms, d0 in faces:
            for t in range(N_TRIALS):
                r = run(model, ms, t)
                if r["goal"]:
                    g += 1
                    hits.append(dict(maze=ms, d0=d0, trial=t, steps=r["steps"]))
        total += g
        rows.append(dict(seed=n, n_ep=len(faces) * N_TRIALS, n_goal=g, hits=hits))
        print(f"  seed{n}: {len(faces)*N_TRIALS} エピソード中 ゴール {g} 件"
              f"{'  ' + str([(h['maze'], h['d0']) for h in hits]) if hits else ''}")

    print("=" * 74)
    print(f"合計 {total} 件 / {sum(r['n_ep'] for r in rows)} エピソード"
          f"（事前登録の期待値 {EXPECTED} 件）")
    band = ("(b) を棄却 — 確率性は原因ではない" if total == 0 else
            "(b) は部分的にしか効かない" if total <= 2 else
            "(b) が主因")
    print(f"→ **{band}**")
    print("⚠️ 決定的な方策での同じ評価は 0 件だった（exp_020 の 6 本とも）")
    print("=" * 74)

    out = Path(__file__).resolve().parent / "out" / "exp020_stochastic.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(total=total, expected=EXPECTED,
                                   n_trials=N_TRIALS, band=band, per_seed=rows),
                              ensure_ascii=False, indent=2))
    print(f"出力: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
