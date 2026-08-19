"""
tests/test_maze_gen_turnmix.py
================
competition/maze_gen_turnmix.py（S3 最短走行の検証専用: ターン種別が満遍なく
出る迷路の選抜ふるい）の単体テスト。pytest で実行する:

    .venv/bin/python -m pytest tests/test_maze_gen_turnmix.py -q -s

🔴 検査は「壊れたときに鳴る」ことを実測してから使う（tests/test_classic_checks.py
の作法を踏襲）。選抜条件の判定関数は、条件を満たす人工のコマンド列で通り
（空振り側）、満たさない列で落ちる（発火側）ことを対で確かめる。

検証内容:
  1. turn_mix_metrics / passes_turn_mix: 4 条件それぞれについて、他 3 条件を
     満たしたまま当該条件だけを壊した人工コマンド列が正しく弾かれること
  2. 「連続ターン」の判定の読み替え（本ファイル冒頭の docstring 参照）が
     意図通り動くこと: 0 区画・1 区画の直進を挟んだターンは連続ターンとして
     数え、2 区画以上の直進を挟んだターンは数えないこと
  3. 生成した 10 迷路すべてが 4 条件を満たすことを、**保存済み npz を読み直し
     生成時の判定を経由せず独立に**再計算して確認する
  4. 生成した迷路の seed がすべて予約プール [1000, 40999] の外（41000 以上）
     であること・design_v4 と重複しないこと
  5. seed 探索範囲のガード（41000 未満を指定したら実行前に例外で止まること）
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from classic.flood import FloodMode  # noqa: E402
from classic.maze_map import Direction  # noqa: E402
from classic.route import Command, CommandType, plan_route  # noqa: E402
from competition.maze_gen_turnmix import (  # noqa: E402
    MIN_SEED,
    build_full_maze_map,
    generate_turnmix_set,
    passes_turn_mix,
    turn_mix_metrics,
)
from competition.maze_gen_v2 import GOAL_CELLS  # noqa: E402

DESIGN_TURN_V1_DIR = Path(REPO_ROOT) / "competition" / "mazes" / "design_turn_v1"
DESIGN_V4_MANIFEST = Path(REPO_ROOT) / "competition" / "mazes" / "design_v4" / "manifest.json"

R = CommandType.TURN_RIGHT90
L = CommandType.TURN_LEFT90
S = CommandType.STRAIGHT
GOAL = CommandType.GOAL_STOP


def _straight(cells):
    return Command(S, cells)


# ==========================================================================
# 1. 選抜条件（4 条件すべてを満たす基準列と、1 条件だけを壊した対照列）
# ==========================================================================
def _baseline_pass_commands():
    """4 条件をすべてぎりぎり満たす人工コマンド列（右90=3・左90=3・
    連続ターン=2・長い直線=2）。"""
    return [
        Command(R), _straight(4), Command(L), _straight(1),
        Command(R), _straight(5), Command(L), _straight(1),
        Command(R), _straight(2), Command(L), Command(GOAL),
    ]


def test_baseline_satisfies_all_four_conditions():
    """空振り側: 4 条件をすべて満たす列は正しく通ること。"""
    metrics = turn_mix_metrics(_baseline_pass_commands())
    assert metrics == dict(right90=3, left90=3, consecutive_turns=2, long_straights=2)
    assert passes_turn_mix(metrics)


def test_rejects_when_right90_short_of_three():
    """発火側: 条件(1)だけを壊す（右90 を 1 個 左90 に変える → 右90=2）。"""
    cmds = _baseline_pass_commands()
    cmds[8] = Command(L)  # 3 個目の右90 を左90に
    metrics = turn_mix_metrics(cmds)
    assert metrics["right90"] == 2
    assert metrics["left90"] == 4
    assert metrics["consecutive_turns"] >= 1  # 他の条件は生きたまま
    assert metrics["long_straights"] == 2
    assert not passes_turn_mix(metrics)


def test_rejects_when_left90_short_of_three():
    """発火側: 条件(2)だけを壊す（左90 を 1 個 右90 に変える → 左90=2）。"""
    cmds = _baseline_pass_commands()
    cmds[10] = Command(R)  # 3 個目の左90 を右90に
    metrics = turn_mix_metrics(cmds)
    assert metrics["left90"] == 2
    assert metrics["right90"] == 4
    assert metrics["consecutive_turns"] >= 1
    assert metrics["long_straights"] == 2
    assert not passes_turn_mix(metrics)


def test_rejects_when_no_consecutive_turn():
    """発火側: 条件(3)だけを壊す（ターン間の直進を 1→2 区画にして
    「加速の余地がある直進」に変える。読み替えの定義どおり連続ターンが消える）。"""
    cmds = _baseline_pass_commands()
    cmds[3] = _straight(2)
    cmds[7] = _straight(2)
    metrics = turn_mix_metrics(cmds)
    assert metrics["consecutive_turns"] == 0
    assert metrics["right90"] == 3
    assert metrics["left90"] == 3
    assert metrics["long_straights"] == 2  # 2区画は長い直線の閾値未満のまま
    assert not passes_turn_mix(metrics)


def test_rejects_when_long_straight_short_of_two():
    """発火側: 条件(4)だけを壊す（長い直線を 1 個に減らす）。"""
    cmds = _baseline_pass_commands()
    cmds[1] = _straight(2)  # 4区画 -> 2区画（長い直線の対象外に）
    metrics = turn_mix_metrics(cmds)
    assert metrics["long_straights"] == 1
    assert metrics["right90"] == 3
    assert metrics["left90"] == 3
    assert metrics["consecutive_turns"] >= 1
    assert not passes_turn_mix(metrics)


# ==========================================================================
# 2. 「連続ターン」判定の読み替え（本体 docstring の条件(3)節）
# ==========================================================================
def test_consecutive_turn_counts_direct_adjacency_with_no_straight():
    """STRAIGHT が皆無で隣接する2ターン（歩数最短経路には現れない想定だが、
    関数としては検出できること）。"""
    metrics = turn_mix_metrics([Command(R), Command(L), Command(GOAL)])
    assert metrics["consecutive_turns"] == 1


def test_consecutive_turn_counts_one_cell_zigzag():
    """1区画だけの直進を挟んだターンは連続ターンとして数える。"""
    metrics = turn_mix_metrics([Command(R), _straight(1), Command(L), Command(GOAL)])
    assert metrics["consecutive_turns"] == 1


def test_consecutive_turn_ignores_real_straight_of_two_or_more():
    """空振り側: 2区画以上の直進を挟んだターンは連続ターンとして数えない
    （加速の余地がある直進なので「間に STRAIGHT を挟む」に該当する）。"""
    metrics = turn_mix_metrics([Command(R), _straight(2), Command(L), Command(GOAL)])
    assert metrics["consecutive_turns"] == 0


# ==========================================================================
# 3. 生成物（npz）を独立に読み直しての再検査
# ==========================================================================
def _npz_files():
    return sorted(DESIGN_TURN_V1_DIR.glob("maze_*.npz"))


def test_generated_ten_mazes_exist():
    files = _npz_files()
    assert len(files) == 10, (
        f"design_turn_v1 の npz が 10 面ではない: {len(files)} 面（{DESIGN_TURN_V1_DIR}）。"
        "先に .venv/bin/python -m competition.maze_gen_turnmix --count 10 を実行すること。"
    )


def test_generated_mazes_all_satisfy_turn_mix_conditions_independently():
    """生成時の判定（manifest.json の記録値）は一切参照せず、保存された npz の
    生の壁配列だけから MazeMap を組み立て直し・経路を引き直し・4 条件を
    再計算して確かめる。"""
    files = _npz_files()
    assert files, "生成物が無い（先に生成コマンドを実行すること）"
    for npz_path in files:
        data = np.load(npz_path)
        maze = build_full_maze_map(data["v_walls"], data["h_walls"])
        _path, commands = plan_route(
            maze, start=(0, 0), goals=list(GOAL_CELLS),
            mode=FloodMode.PESSIMISTIC, start_heading=Direction.N,
        )
        metrics = turn_mix_metrics(commands)
        assert passes_turn_mix(metrics), (
            f"{npz_path.name}: npz からの独立再計算で 4 条件を満たさない（metrics={metrics}）"
        )


def test_generated_seeds_are_all_at_or_above_min_seed():
    """seed が評価用に予約された候補プール [1000, 40999] を踏んでいないこと
    （研究計画書 §9-7）。"""
    files = _npz_files()
    assert files
    for npz_path in files:
        data = np.load(npz_path)
        seed = int(data["seed"])
        assert seed >= MIN_SEED, f"{npz_path.name}: seed={seed} が予約プールに踏み込んでいる"


def test_generated_seeds_do_not_overlap_design_v4():
    """既存の design_v4 と同じ seed を採っていないこと。"""
    with open(DESIGN_V4_MANIFEST, encoding="utf-8") as f:
        v4 = json.load(f)
    v4_seeds = {int(m["seed"]) for m in v4["mazes"]}
    files = _npz_files()
    assert files
    for npz_path in files:
        data = np.load(npz_path)
        seed = int(data["seed"])
        assert seed not in v4_seeds, f"{npz_path.name}: design_v4 と seed が重複している"


def test_manifest_maze_count_matches_npz_count():
    manifest_path = DESIGN_TURN_V1_DIR / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    files = _npz_files()
    assert len(manifest["mazes"]) == len(files)
    manifest_seeds = {int(m["seed"]) for m in manifest["mazes"]}
    npz_seeds = {int(np.load(p)["seed"]) for p in files}
    assert manifest_seeds == npz_seeds


# ==========================================================================
# 4. seed 探索範囲のガード
# ==========================================================================
def test_seed_guard_rejects_seed_start_below_reserved_pool():
    """発火側: 予約プールに踏み込む seed_start は実行前に例外で止まること。"""
    with pytest.raises(ValueError):
        generate_turnmix_set(seed_start=40000, count=1, max_scan=10, excluded_seeds=set())


def test_seed_guard_allows_seed_start_at_min_seed():
    """空振り側: MIN_SEED ちょうどは通ること（実際に 1 面生成できる）。"""
    accepted, n_scanned = generate_turnmix_set(
        seed_start=MIN_SEED, count=1, max_scan=50, excluded_seeds=set()
    )
    assert len(accepted) == 1
    assert accepted[0][0] >= MIN_SEED
    assert n_scanned >= 1
