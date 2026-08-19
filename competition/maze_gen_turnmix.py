"""
competition/maze_gen_turnmix.py
================================
note_030 §4 の S3（最短走行）専用検証迷路 — 「ターン種別が満遍なく出る迷路」の生成器。

■ 位置づけ（教授裁定 2026-08-19）
既存の `competition/maze_gen_v2.py` は、確保済みの評価用・検証用迷路を生成した
記録であり、受理条件を書き換えると再現性が失われる。そのため**この生成器は
maze_gen_v2 の受理条件を一切変更しない**。maze_gen_v2.generate_maze() を
そのまま呼び出し、その出力（規定準拠・D0 窓を満たす迷路）を「ターン種別の
混ざり具合」でさらに選抜する、**外側に足したふるい**として実装する。

■ 選抜条件（教授が確定した条文。そのまま実装する）
迷路の真の壁をすべて既知として classic/maze_map.MazeMap を組み立て、
classic/route.py の plan_route(maze, start=(0,0), goals=中央2x2,
mode=FloodMode.PESSIMISTIC, start_heading=Direction.N) でスタートから
ゴールまでの最短経路のコマンド列を求める。そのコマンド列が次を**すべて**
満たすものだけを採る:
    (1) CommandType.TURN_RIGHT90 が 3 個以上
    (2) CommandType.TURN_LEFT90 が 3 個以上
    (3) 連続ターン（間に STRAIGHT を挟まない隣り合うターン）が 1 組以上
    (4) STRAIGHT の cells が 3 以上のものが 2 個以上（直線を伸ばす効果が出る条件）

■ 条件 (3)「連続ターン」の判定について（実装上の読み替え・要確認）
`classic/route.py` の `path_to_commands` は 1 区画の移動を必ず「STRAIGHT を
1 個以上」の形で記録する（1 回のターンの直後でも、その直後の 1 区画移動は
必ず STRAIGHT(cells=1) として積まれてから次のターンが来る。実装（run_len を
毎ステップ無条件に +1 する構造）上、STRAIGHT(cells=0) は原理的に発生しない）。
したがって「間に STRAIGHT を全く挟まない隣り合う 2 個のターン」を文字通りに
探すと、**歩数最短経路のコマンド列には構造上ただの一度も現れない**（実測:
seed 41000-41059 の 60 面全てで 0 件）。本ファイルはこれを「間に**加速の
余地がある長さの直進（cells>=2）**を挟まない隣り合うターン」と読み替え、
[TURN, STRAIGHT(cells=1), TURN] という「1 区画だけ挟んだ角continuoue（ジグザグ）」
も連続ターンとして数える（cells=1 の直進は note_030 §3 の「直進を伸ばして
加速する」効果を持たない、という同じ資料の理由づけに沿う）。文字通りの
[TURN, TURN]（STRAIGHT が皆無）も検出はするが、上記の理由で実際には出現しない。
**この読み替えは教授セッションの確認を得ていない実装判断であり、報告で
明示している。**

■ 180° 折返しを条件に入れない理由
note_030 §4 の表は「探索走行も含む一般のターン種別」を挙げたものだが、本条文が
対象にするのは **plan_route が返す歩数最短の経路**である。歩数最短の経路は
BFS の歩数マップ上で 1 手ごとに歩数が単調に 1 ずつ減る区画列でしか作れない
（`classic/route.py` の `shortest_path` 参照）。180° 折返しは「直前に来た区画へ
戻る」動きであり、それは歩数が 1 ずつ減る制約と両立しない（戻れば歩数が
増える側へ進むことになる）。したがって**歩数最短の経路には 180° 折返しは
そもそも現れず**、条件に入れても入れなくても判定結果は変わらない。それでも
条件から明示的に外しているのは、この生成器が「歩数最短の経路の性質」だけを
見ていることを条文の上でもはっきりさせるためである。

■ 使い方
    .venv/bin/python -m competition.maze_gen_turnmix --count 10

主なオプション:
    --seed-start   探索開始 seed（既定 41000。これ未満は起動前に例外で止める）
    --count        採用する迷路数（既定 10）
    --max-scan     1 回の実行で試す候補 seed 数の上限（安全弁。既定 20000）
    --out-dir      出力先（既定 competition/mazes/design_turn_v1）
    --exclude-manifest
                   除外する既存 manifest（既定 competition/mazes/design_v4/manifest.json。
                   ここに載っている seed は重複を避けるため採らない）
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from classic.flood import FloodMode  # noqa: E402
from classic.maze_map import Direction, MazeMap, WallState  # noqa: E402
from classic.route import Command, CommandType, plan_route  # noqa: E402
from common.seed_bands import assert_seeds_allowed  # noqa: E402
from competition.maze_gen_v2 import GOAL_CELLS, generate_maze  # noqa: E402

#: 評価用に予約された候補プール [1000, 40999]（研究計画書 §9-7）の直後。
#: これより小さい seed-start は起動前に例外で止める。
MIN_SEED = 41000

# 選抜条件のしきい値（教授裁定 2026-08-19。本ファイルで変更しない）。
MIN_RIGHT90 = 3
MIN_LEFT90 = 3
MIN_CONSECUTIVE_TURN_PAIRS = 1
LONG_STRAIGHT_CELLS = 3
MIN_LONG_STRAIGHT_COUNT = 2

# plan_route に渡す固定引数（S3 の最短走行と同じ設定）。
START_CELL = (0, 0)
START_HEADING = Direction.N


# ==========================================================================
# 壁配列 → MazeMap（全壁既知）
# ==========================================================================
def build_full_maze_map(v_walls, h_walls) -> MazeMap:
    """真の壁配列（0=壁なし・非0=壁あり。maze_gen_v2 / evaluator と同じ添字規約）
    から、全壁が既知の MazeMap を組み立てる。

    競技評価では機体は壁を少しずつ知るが、S3 の最短経路そのものの性質を
    調べる本条文では「真の壁をすべて既知として」判定する（教授裁定）。
    """
    v_walls = np.asarray(v_walls)
    h_walls = np.asarray(h_walls)
    width = v_walls.shape[0] - 1
    height = v_walls.shape[1]
    maze = MazeMap(width=width, height=height)
    # classic/maze_map.py の添字規約は maze_gen_v2 の v/h 配列とそのまま一致する
    # （両者とも「competition/evaluator.py の v_walls/h_walls と同じ添字規約」に
    # 揃えてあるため）。形状も一致するので要素ごとに変換するだけでよい。
    maze.v_walls[:, :] = np.where(v_walls != 0, WallState.WALL, WallState.OPEN)
    maze.h_walls[:, :] = np.where(h_walls != 0, WallState.WALL, WallState.OPEN)
    return maze


# ==========================================================================
# ターン種別の内訳と選抜条件
# ==========================================================================
def turn_mix_metrics(commands) -> dict:
    """コマンド列（`classic.route.path_to_commands` の返り値）から
    ターン種別の内訳を数える。

    Returns:
        right90: 90°右の個数
        left90: 90°左の個数
        consecutive_turns: 「間に加速の余地がある直進（cells>=2）を挟まない
            隣り合うターン」の組数。[TURN, TURN]（直進皆無。歩数最短経路には
            構造上現れない）と [TURN, STRAIGHT(cells=1), TURN]（1 区画だけの
            ジグザグ）の両方を数える。TURN_RIGHT90 / TURN_LEFT90 / TURN_180 の
            いずれも「ターン」として数える（歩数最短経路に TURN_180 は現れない
            — 本ファイル冒頭の docstring 参照）。読み替えの理由は本ファイル
            冒頭の docstring「条件 (3)」の節を参照。
        long_straights: cells が LONG_STRAIGHT_CELLS(=3) 以上の STRAIGHT の個数
    """
    turn_types = (CommandType.TURN_RIGHT90, CommandType.TURN_LEFT90, CommandType.TURN_180)
    right90 = sum(1 for c in commands if c.type == CommandType.TURN_RIGHT90)
    left90 = sum(1 for c in commands if c.type == CommandType.TURN_LEFT90)
    long_straights = sum(
        1 for c in commands
        if c.type == CommandType.STRAIGHT and c.cells >= LONG_STRAIGHT_CELLS
    )
    consecutive_turns = 0
    n = len(commands)
    for i, c in enumerate(commands):
        if c.type not in turn_types:
            continue
        # 直後のコマンドが直接ターン（STRAIGHT 皆無。実際には現れない想定）。
        if i + 1 < n and commands[i + 1].type in turn_types:
            consecutive_turns += 1
            continue
        # 直後が 1 区画だけの STRAIGHT で、その次がターン（1 区画ジグザグ）。
        if (i + 2 < n and commands[i + 1].type == CommandType.STRAIGHT
                and commands[i + 1].cells == 1 and commands[i + 2].type in turn_types):
            consecutive_turns += 1
    return dict(
        right90=right90,
        left90=left90,
        consecutive_turns=consecutive_turns,
        long_straights=long_straights,
    )


def passes_turn_mix(metrics: dict) -> bool:
    """教授裁定の 4 条件をすべて満たすか。"""
    return (
        metrics["right90"] >= MIN_RIGHT90
        and metrics["left90"] >= MIN_LEFT90
        and metrics["consecutive_turns"] >= MIN_CONSECUTIVE_TURN_PAIRS
        and metrics["long_straights"] >= MIN_LONG_STRAIGHT_COUNT
    )


def route_metrics_for_walls(v_walls, h_walls):
    """壁配列から MazeMap を組み立て、S3 の最短経路のターン内訳を返す。

    Returns: (commands, metrics)
    """
    maze = build_full_maze_map(v_walls, h_walls)
    _path, commands = plan_route(
        maze,
        start=START_CELL,
        goals=list(GOAL_CELLS),
        mode=FloodMode.PESSIMISTIC,
        start_heading=START_HEADING,
    )
    metrics = turn_mix_metrics(commands)
    return commands, metrics


# ==========================================================================
# 既存生成物との重複除外
# ==========================================================================
def load_excluded_seeds(manifest_path) -> set:
    """既存 manifest（例: design_v4/manifest.json）に載っている採用 seed の集合。

    ファイルが無ければ空集合を返す（除外対象が無いだけなので致命ではない）。
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return set()
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    return {int(m["seed"]) for m in data.get("mazes", [])}


# ==========================================================================
# 生成本体
# ==========================================================================
def generate_turnmix_set(seed_start: int, count: int, max_scan: int, excluded_seeds: set):
    """seed_start から昇順に候補 seed を試し、選抜条件を満たす迷路を count 面集める。

    Returns: (accepted, n_scanned)
        accepted: [(seed, v_walls, h_walls, gen_info, metrics, commands), ...]
        n_scanned: 実際に maze_gen_v2.generate_maze を呼んだ候補 seed の数
    """
    if seed_start < MIN_SEED:
        raise ValueError(
            f"seed_start={seed_start} は評価用に予約された候補プール "
            f"[1000, 40999] に踏み込みます（研究計画書 §9-7）。{MIN_SEED} 以上を指定してください。"
        )
    # common/seed_bands.py の安全弁で、走査する候補範囲全体が凍結帯に触れないことを
    # 実行前に確かめる（防御の二重化。上の明示チェックと合わせて実行前に必ず落とす）。
    assert_seeds_allowed(
        range(seed_start, seed_start + max_scan),
        namespace="competition",
        purpose="validate",
    )

    accepted = []
    n_scanned = 0
    seed = seed_start
    while len(accepted) < count and n_scanned < max_scan:
        if seed in excluded_seeds:
            seed += 1
            continue
        n_scanned += 1
        try:
            v, h, gen_info = generate_maze(seed)
        except RuntimeError:
            # maze_gen_v2 側の受理条件（規定準拠・D0 窓）を満たせなかった seed。
            # 本ふるいの対象外として次の seed へ進む。
            seed += 1
            continue
        commands, metrics = route_metrics_for_walls(v, h)
        if passes_turn_mix(metrics):
            accepted.append((seed, v, h, gen_info, metrics, commands))
        seed += 1

    if len(accepted) < count:
        raise RuntimeError(
            f"候補 seed を {n_scanned} 個走査しましたが、選抜条件を満たす迷路が "
            f"{len(accepted)}/{count} 面しか見つかりませんでした（--max-scan を増やしてください）。"
        )
    return accepted, n_scanned


def _commands_to_manifest(commands):
    """manifest.json に書ける形へコマンド列を変換する（人間が読める内訳）。"""
    return [
        dict(type=c.type.value, cells=c.cells) if c.type == CommandType.STRAIGHT
        else dict(type=c.type.value)
        for c in commands
    ]


def main():
    ap = argparse.ArgumentParser(
        description="S3（最短走行）検証用: ターン種別が満遍なく出る迷路の選抜生成")
    ap.add_argument("--seed-start", type=int, default=MIN_SEED)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--max-scan", type=int, default=20000)
    ap.add_argument("--out-dir", default="competition/mazes/design_turn_v1")
    ap.add_argument("--exclude-manifest", default="competition/mazes/design_v4/manifest.json")
    args = ap.parse_args()

    excluded_seeds = load_excluded_seeds(args.exclude_manifest)

    accepted, n_scanned = generate_turnmix_set(
        seed_start=args.seed_start,
        count=args.count,
        max_scan=args.max_scan,
        excluded_seeds=excluded_seeds,
    )

    from mouse.mjcf import build_maze_robot_xml
    from mouse.params import RobotParams
    params = RobotParams()
    os.makedirs(args.out_dir, exist_ok=True)

    mazes_manifest = []
    for seed, v, h, gen_info, metrics, commands in accepted:
        npz = os.path.join(args.out_dir, f"maze_{seed}.npz")
        np.savez(npz, v_walls=v, h_walls=h, seed=seed, width=v.shape[0] - 1, height=v.shape[1])
        build_maze_robot_xml(v, h, npz[:-4] + ".xml", model_name=f"maze_{seed}", params=params)
        mazes_manifest.append(dict(
            seed=seed,
            d0=gen_info["d0"],
            d_shortest=gen_info["d_shortest"],
            cycles=gen_info["cycles"],
            gateway=gen_info["gateway"],
            turn_right90=metrics["right90"],
            turn_left90=metrics["left90"],
            consecutive_turns=metrics["consecutive_turns"],
            long_straights=metrics["long_straights"],
            commands=_commands_to_manifest(commands),
        ))
        print(f"[maze_gen_turnmix] seed={seed} 最短{gen_info['d_shortest']}区画(D0={gen_info['d0']}) "
              f"右90={metrics['right90']} 左90={metrics['left90']} "
              f"連続ターン={metrics['consecutive_turns']} 長い直線={metrics['long_straights']}")

    manifest = dict(
        generator="competition/maze_gen_turnmix.py",
        source_generator="competition/maze_gen_v2.py",
        purpose="note_030 §4 S3（最短走行）検証用: ターン種別が満遍なく出る迷路の選抜",
        seed_start=args.seed_start,
        seed_scanned=n_scanned,
        excluded_manifest=args.exclude_manifest,
        excluded_seed_count=len(excluded_seeds),
        selection_conditions=dict(
            note="plan_route(start=(0,0), goals=中央2x2, mode=PESSIMISTIC, start_heading=N) の"
                 "コマンド列に対する条件。180°折返しは歩数最短経路に現れないため条件に含めない"
                 "（本ファイル冒頭の docstring 参照）。",
            min_turn_right90=MIN_RIGHT90,
            min_turn_left90=MIN_LEFT90,
            min_consecutive_turn_pairs=MIN_CONSECUTIVE_TURN_PAIRS,
            long_straight_cells_threshold=LONG_STRAIGHT_CELLS,
            min_long_straight_count=MIN_LONG_STRAIGHT_COUNT,
        ),
        mazes=mazes_manifest,
    )
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n採用 {len(accepted)} 面（候補 {n_scanned} 個を走査）。出力先: {args.out_dir}")


if __name__ == "__main__":
    main()
