"""
EvalMazeGenerator: 評価迷路 (seed 1000-1019) 生成器
Evaluation maze generator: given a seed, deterministically builds a 16x16
micromouse-compliant maze (v_walls, h_walls) for competition-style evaluation.

規約 (RESEARCH_PLAN §2): 評価迷路 = seed 1000〜1019、学習用 = seed 2000 以降。
Convention: evaluation mazes use seeds 1000-1019; training mazes use seeds >= 2000.

生成手順 (教授承認済み「案A」。この順序を厳守する):
Generation procedure (professor-approved "Plan A". This order must be followed exactly):
1. random.seed(seed) の後、_dfs_perfect_maze(16, 16) (DFS による完全迷路: 全256セル
   連結・ループなし) を実行する。この関数は凍結レガシー
   legacy/phase3_maze/maze_generator.py の RandomMazeGenerator._dfs() を移植したもので、
   random.shuffle の呼び出し順序・方向リストの順序・再帰の順序を一字一句保っている
   (2026-08-10 import 依存解消、seed 1000-1019 再生成で既存 npz と完全一致することを
   検証済み)。
   Step 1: seed the RNG, then run the plain DFS perfect-maze generator
   (_dfs_perfect_maze, ported verbatim from the frozen legacy
   RandomMazeGenerator._dfs() — same shuffle call order, same direction-list
   order, same recursion order) — connects all 256 cells with no loops.
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
import random

import numpy as np


def _dfs_perfect_maze(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """DFS (再帰バックトラッカー) による完全迷路生成 (全セル連結・ループなし)。
    DFS (recursive backtracker) perfect-maze generation (all cells connected, no loops).

    凍結レガシー legacy/phase3_maze/maze_generator.py の
    RandomMazeGenerator._dfs()/generate_maze() を一字一句移植したもの。
    random.shuffle の呼び方・方向リストの順序・再帰の順序を変えると
    seed 再現性 (既存の competition/mazes/eval/maze_*.npz との一致) が壊れるため、
    アルゴリズムの構造を変更しないこと。
    Ported verbatim from the frozen legacy RandomMazeGenerator._dfs()/generate_maze().
    Do not change the algorithm's structure (shuffle call, direction-list order,
    recursion order) — doing so breaks seed reproducibility against the frozen
    competition/mazes/eval/maze_*.npz files.

    戻り値 (Returns): v_walls (width+1, height), h_walls (width, height+1)。
    1 = 壁あり (wall present)、0 = 壁なし (開通、passable)。
    """
    v_walls = np.ones((width + 1, height), dtype=int)
    h_walls = np.ones((width, height + 1), dtype=int)
    visited = np.zeros((width, height), dtype=bool)

    def _dfs(x: int, y: int) -> None:
        visited[x, y] = True

        # Directions: North, East, South, West
        # (dx, dy, wall_type, wall_x, wall_y)
        directions = [
            (0, 1, 'h', x, y + 1),   # North
            (1, 0, 'v', x + 1, y),   # East
            (0, -1, 'h', x, y),      # South
            (-1, 0, 'v', x, y),      # West
        ]

        random.shuffle(directions)

        for dx, dy, w_type, wx, wy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < width and 0 <= ny < height and not visited[nx, ny]:
                # Remove wall between (x,y) and (nx,ny)
                if w_type == 'h':
                    h_walls[wx, wy] = 0
                else:
                    v_walls[wx, wy] = 0

                _dfs(nx, ny)

    _dfs(0, 0)
    return v_walls, h_walls


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
        v_walls, h_walls = _dfs_perfect_maze(16, 16)

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
