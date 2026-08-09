"""
評価迷路 (seed 1000-1019) の一括生成 CLI
Generates the 20 evaluation mazes (seed 1000-1019) and saves them as npz.

規約 (RESEARCH_PLAN §2): 評価迷路 = seed 1000〜1019、学習用は seed 2000 以降
(本スクリプトでは学習用は生成しない)。

使い方 (Usage):
    .venv/bin/python competition/generate_eval_mazes.py

生成後、恒久検証スクリプト (verify_eval_mazes.py) を自動実行する。
不合格の迷路が1件でもあれば exit code 1 で終了する。
After generation, automatically runs verify_eval_mazes.py; exits with code 1
if any maze fails verification.
"""
import datetime
import json
import os
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from competition.maze_gen import EvalMazeGenerator  # noqa: E402

# 評価迷路: seed 1000-1019 (20面)
EVAL_SEEDS = list(range(1000, 1020))
OUTPUT_DIR = os.path.join(_REPO_ROOT, "competition", "mazes", "eval")


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for seed in EVAL_SEEDS:
        v_walls, h_walls = EvalMazeGenerator.generate(seed)
        out_path = os.path.join(OUTPUT_DIR, f"maze_{seed}.npz")
        np.savez(
            out_path,
            v_walls=v_walls,
            h_walls=h_walls,
            seed=seed,
            width=16,
            height=16,
        )
        print(f"[generate_eval_mazes] saved {out_path}")

    manifest = {
        "seeds": EVAL_SEEDS,
        "count": len(EVAL_SEEDS),
        "generated_at": datetime.datetime.now().isoformat(),
        "generator": "competition/maze_gen.py: EvalMazeGenerator (Plan A / 案A)",
        "procedure_summary": [
            "1. random.seed(seed) -> phase3_maze.maze_generator.RandomMazeGenerator(16,16)"
            ".generate_maze() (DFSによる完全迷路生成)",
            "2. 中央2x2(ゴール区画)の内壁4枚を開放: v[8,7]=0, v[8,8]=0, h[7,8]=0, h[8,8]=0",
            "3. 孤立柱(壁0枚接続の柱)を修復。中央2x2の内壁4枚は候補から除外"
            "(案A: common/maze_generator.py の潜在バグの修正)",
            "4. スタート区画(0,0)規定: h[0,1]=0(北開放), v[1,0]=1(東壁),"
            " h[0,0]=1, v[0,0]=1(外周閉鎖)",
            "5. 中央2x2を再度開放(手順3の保険)",
        ],
        "wall_array_shapes": {"v_walls": [17, 16], "h_walls": [16, 17]},
    }
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[generate_eval_mazes] saved {manifest_path}")

    # 生成後に恒久検証スクリプトを自動実行する
    # Automatically run the permanent verification script after generation
    verify_script = os.path.join(_REPO_ROOT, "competition", "verify_eval_mazes.py")
    result = subprocess.run([sys.executable, verify_script])
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
