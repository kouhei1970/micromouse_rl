"""
評価迷路 XML の一括再生成スクリプト
Regenerate evaluation-maze MuJoCo XMLs from the wall-array npz files.

憲章 (RESEARCH_PLAN §2): 壁配列 npz が一次成果物（真実の源）であり、
MuJoCo XML は「npz + 現行ロボット仕様」から再生成可能な派生物である。
ロボット仕様 (mouse/params.py, mouse/robot_v2.py) を変更したら必ず本スクリプトで
全 XML を再生成すること。陳腐化した XML は不正な挙動（例: 即転倒）を引き起こす
（2026-08-10 の評価器実装時に実証済み）。

使い方 / Usage:
    .venv/bin/python -m competition.regenerate_maze_xml [--maze-dir competition/mazes/eval]
"""
import argparse
import glob
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mouse.mjcf import build_maze_robot_xml  # noqa: E402
from mouse.params import RobotParams  # noqa: E402


def regenerate(maze_dir: str) -> int:
    params = RobotParams()
    npz_paths = sorted(glob.glob(os.path.join(maze_dir, "maze_*.npz")))
    for npz_path in npz_paths:
        d = np.load(npz_path)
        xml_path = npz_path[:-4] + ".xml"
        build_maze_robot_xml(
            d["v_walls"], d["h_walls"], xml_path,
            model_name=os.path.basename(npz_path)[:-4],
            params=params,
        )
    return len(npz_paths)


def main(argv=None):
    ap = argparse.ArgumentParser(description="npz から評価迷路 XML を一括再生成")
    ap.add_argument("--maze-dir", default="competition/mazes/eval")
    args = ap.parse_args(argv)
    n = regenerate(args.maze_dir)
    print(f"[regenerate_maze_xml] {n} 面の XML を再生成しました ({args.maze_dir})")
    if n == 0:
        print("[regenerate_maze_xml] 警告: npz が見つかりません", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
