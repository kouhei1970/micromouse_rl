"""
EvalMazeGenerator: 評価迷路 (seed 1000-1019) 生成器
Evaluation maze generator: given a seed, deterministically builds a 16x16
micromouse-compliant maze (v_walls, h_walls) for competition-style evaluation.

規約 (RESEARCH_PLAN §2): 評価迷路 = seed 1000〜1019、学習用 = seed 2000 以降。
Convention: evaluation mazes use seeds 1000-1019; training mazes use seeds >= 2000.

生成手順 (教授承認済み「案A」。この順序を厳守する):
Generation procedure (professor-approved "Plan A". This order must be followed exactly):
1. random.seed(seed) の後、phase3_maze.maze_generator.RandomMazeGenerator(16, 16) の
   generate_maze() (DFS による完全迷路: 全256セル連結・ループなし) を実行する。
   Step 1: seed the RNG, then run the plain DFS perfect-maze generator
   (RandomMazeGenerator.generate_maze()) — connects all 256 cells with no loops.
2. 中央 2x2 (ゴール区画) の内壁 4 枚を開放する。
   Step 2: open the 4 inner walls of the central 2x2 goal area.
3. 孤立柱 (どの壁も接続していない柱) を修復する。中央柱 (8,8) は柱そのものが
   撤去されているため対象外。ただし中央 2x2 の内壁 4 枚 (手順2 で開放したもの)
   は修復候補から除外する。
   注: 親クラス common/maze_generator.py の同処理はこの除外を行っていないため、
   ここで追加した壁が手順5 で再び除去され、孤立柱が復活する潜在バグを持つ。
   本実装ではこれを修正している。
   Step 3: repair isolated posts (posts with zero attached walls). The center
   post (8,8) itself is skipped (physically removed). Candidates that fall on
   one of the 4 central-2x2 inner walls are excluded from the repair choices —
   this fixes a latent bug in common/maze_generator.py's equivalent step, where
   omitting this exclusion lets a newly-added wall get silently removed again
   in the "re-open center" step, leaving an isolated post behind.
4. スタート区画 (0,0) の壁配置を規定する: 北開放・東壁あり・南西は外周閉鎖。
   Step 4: enforce the start-cell (0,0) wall configuration: open north, walled
   east, boundary-closed south/west.
5. 中央 2x2 の内壁 4 枚を再度開放する (手順3 で万一追加されていた場合の保険)。
   Step 5: re-open the central 2x2 inner walls again, as a safety net in case
   step 3 happened to touch one of them.
"""
import os
import random
import sys

import numpy as np

# リポジトリルートを sys.path に追加する (phase3_maze を import するため)。
# Add the repository root to sys.path so that phase3_maze can be imported.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from phase3_maze.maze_generator import RandomMazeGenerator  # noqa: E402

# 中央 2x2 (ゴール区画) の内壁4枚。(壁種別, x, y) のタプルで表す。
# The four inner walls of the central 2x2 goal area, as (wall_kind, x, y) tuples.
_CENTER_WALLS = frozenset(
    {
        ("v", 8, 7),
        ("v", 8, 8),
        ("h", 7, 8),
        ("h", 8, 8),
    }
)


class EvalMazeGenerator:
    """seed -> 16x16 micromouse 準拠迷路 (v_walls (17,16), h_walls (16,17)) を
    決定的に生成するクラス。評価迷路 (seed 1000-1019) 専用。

    Deterministically builds a 16x16 micromouse-compliant maze
    (v_walls shape (17,16), h_walls shape (16,17)) from a seed.
    Intended for the evaluation mazes (seed 1000-1019) only.
    """

    @staticmethod
    def generate(seed: int) -> tuple[np.ndarray, np.ndarray]:
        # 手順1: 素の DFS 迷路生成
        # Step 1: raw DFS perfect-maze generation
        random.seed(seed)
        gen = RandomMazeGenerator(16, 16)
        gen.generate_maze()
        v_walls = gen.v_walls
        h_walls = gen.h_walls

        # 手順2: 中央 2x2 開放
        # Step 2: open the central 2x2 goal area
        EvalMazeGenerator._open_center(v_walls, h_walls)

        # 手順3: 孤立柱の修復 (中央2x2の内壁は候補から除外)
        # Step 3: repair isolated posts (excluding the central 2x2 inner walls)
        EvalMazeGenerator._fix_isolated_posts(v_walls, h_walls)

        # 手順4: スタート区画規定
        # Step 4: enforce the start-cell wall configuration
        h_walls[0, 1] = 0  # 北開放 (open north)
        v_walls[1, 0] = 1  # 東壁 (east wall present)
        h_walls[0, 0] = 1  # 南=外周閉鎖 (south = boundary, closed)
        v_walls[0, 0] = 1  # 西=外周閉鎖 (west = boundary, closed)

        # 手順5: 中央 2x2 再開放 (念のため)
        # Step 5: re-open the central 2x2 (safety net)
        EvalMazeGenerator._open_center(v_walls, h_walls)

        return v_walls, h_walls

    @staticmethod
    def _open_center(v_walls: np.ndarray, h_walls: np.ndarray) -> None:
        v_walls[8, 7] = 0  # (7,7)-(8,7) 間
        v_walls[8, 8] = 0  # (7,8)-(8,8) 間
        h_walls[7, 8] = 0  # (7,7)-(7,8) 間
        h_walls[8, 8] = 0  # (8,7)-(8,8) 間

    @staticmethod
    def _fix_isolated_posts(v_walls: np.ndarray, h_walls: np.ndarray) -> None:
        # 柱 (px,py) に接続する壁:
        # Walls attached to post (px, py):
        #   N: v_walls[px, py]     (py < 16)
        #   S: v_walls[px, py-1]   (py > 0)
        #   E: h_walls[px, py]     (px < 16)
        #   W: h_walls[px-1, py]   (px > 0)
        for px in range(17):
            for py in range(17):
                if px == 8 and py == 8:
                    continue  # 中央柱は物理的に撤去済みのため対象外

                candidates = []  # [(wall_kind, x, y), ...] N, S, E, W の順
                if py < 16:
                    candidates.append(("v", px, py))
                if py > 0:
                    candidates.append(("v", px, py - 1))
                if px < 16:
                    candidates.append(("h", px, py))
                if px > 0:
                    candidates.append(("h", px - 1, py))

                connected = any(
                    (v_walls[x, y] if kind == "v" else h_walls[x, y]) == 1
                    for kind, x, y in candidates
                )
                if connected:
                    continue

                # 中央2x2の内壁4枚は候補から除外 (案A の要点)
                usable = [c for c in candidates if c not in _CENTER_WALLS]
                if not usable:
                    # 理論上到達しない (中央柱そのものは対象外のため)。保険として何もしない。
                    continue
                kind, x, y = random.choice(usable)
                if kind == "v":
                    v_walls[x, y] = 1
                else:
                    h_walls[x, y] = 1
