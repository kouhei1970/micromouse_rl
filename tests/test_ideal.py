"""tests/test_ideal.py — `classic/ideal.py`（経路から理想時間を出す層）の検査

    .venv/bin/python -m pytest tests/test_ideal.py -q

対象は `competition/mazes/design_turn_v1` の10迷路（真の壁・manifest の d0 付き）。
幾何探索（`classic.geometry.max_feasible_radius` 系）が重いので、同じ
`(seed, mode, margin)` の理想時間計算はテスト間でキャッシュして使い回す
（`_result` 参照）。

検査内容:
  1. `true_shortest_path` の経路長が manifest の d0 と一致する
  2. 同じ経路で `mode="slalom"` が `mode="spin"` より速い（全10迷路）
  3. `TurnPlan.limited_by` が "geometry"/"prev"/"next" のいずれかで、
     全ターンについて埋まっている（内訳を印字する）
  4. 組み立てた経路を `geometry.sweep_clearance` で掃引し、最小余裕が margin 以上
  5. 否定対照: margin を大きく壊すと、通れる半径が無くなる（その場旋回に降格する）
     ぶん時間が延びる。壊さなければ同じ結果になる（空振り側）
  6. `_fast_max_feasible_radius`（本モジュールの高速版）が
     `classic.geometry.max_feasible_radius`（元の実装）と同じ値を返す
     （速度のための最適化が判定結果を変えていないことの直接照合）
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from classic.geometry import Pose, max_feasible_radius, poses_along, sweep_clearance, wall_obstacles
from classic.ideal import (
    CELL_SIZE,
    _dir_angle,
    _fast_max_feasible_radius,
    _geometry_blocks,
    _turn_delta,
    _turns_and_runs,
    ideal_time_for_path,
    true_shortest_path,
)
from classic.maze_map import Direction

REPO_ROOT = Path(__file__).resolve().parent.parent
MAZE_DIR = REPO_ROOT / "competition" / "mazes" / "design_turn_v1"
START = (0, 0)
GOALS = [(7, 7), (7, 8), (8, 7), (8, 8)]  # manifest の生成条件と同じ中央2x2（note_031 §生成条件参照）


def _manifest():
    with open(MAZE_DIR / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


MANIFEST = _manifest()
SEEDS = [m["seed"] for m in MANIFEST["mazes"]]


def _load(seed):
    d = np.load(MAZE_DIR / f"maze_{seed}.npz")
    return d["v_walls"], d["h_walls"]


# ============================================================================
# キャッシュ: 同じ (seed, mode, margin) の理想時間計算はテスト間で使い回す
# （幾何探索が重く、テストごとに計算し直すと現実的な時間に収まらない）。
# ============================================================================
_CACHE = {}


def _result(seed, mode="slalom", margin=0.005):
    key = (seed, mode, margin)
    if key not in _CACHE:
        v, h = _load(seed)
        path = true_shortest_path(v, h, START, GOALS, Direction.N)
        res = ideal_time_for_path(path, v, h, Direction.N, mode=mode, margin=margin)
        _CACHE[key] = (path, v, h, res)
    return _CACHE[key]


# ============================================================================
# 1. true_shortest_path が manifest の d0 と一致する
# ============================================================================
def test_true_shortest_path_matches_the_flood_map():
    seed = SEEDS[0]
    m = next(x for x in MANIFEST["mazes"] if x["seed"] == seed)
    v, h = _load(seed)
    path = true_shortest_path(v, h, START, GOALS, Direction.N)

    print(f"\n[検査1] seed={seed}: 経路長(区画数-1)={len(path)-1}  manifest d0={m['d0']}")

    assert len(path) - 1 == m["d0"], f"経路長 {len(path)-1} が manifest の d0={m['d0']} と不一致"
    assert path[0] == tuple(START)
    assert tuple(path[-1]) in [tuple(g) for g in GOALS]
    # 隣接区画が1マス分の移動になっていること（経路が飛んでいない）
    for a, b in zip(path[:-1], path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, (a, b)


# ============================================================================
# 2. slalom が spin より速い（全10迷路）
# ============================================================================
@pytest.mark.parametrize("seed", SEEDS)
def test_slalom_is_faster_than_spin(seed):
    _, _, _, res_slalom = _result(seed, mode="slalom")
    _, _, _, res_spin = _result(seed, mode="spin")

    print(
        f"\n[検査2] seed={seed}: T_slalom={res_slalom.total:.3f}s  T_spin={res_spin.total:.3f}s  "
        f"n_turns={res_slalom.n_turns}"
    )

    assert res_slalom.n_turns == res_spin.n_turns  # 同じ経路のはず
    assert res_slalom.total < res_spin.total


# ============================================================================
# 3. limited_by の内訳（全10迷路合計。test2 のキャッシュを使い回すので追加計算は無い）
# ============================================================================
def test_radius_limits_are_recorded():
    counts = Counter()
    n_turns_total = 0
    for seed in SEEDS:
        _, _, _, res = _result(seed, mode="slalom")
        n_turns_total += res.n_turns
        for t in res.turns:
            assert t.limited_by in ("geometry", "prev", "next"), (seed, t)
            counts[t.limited_by] += 1

    print(f"\n[検査3] limited_by の内訳（design_turn_v1 全10迷路・ターン計{n_turns_total}）: "
          f"{dict(counts)}")

    assert n_turns_total > 0
    assert sum(counts.values()) == n_turns_total


# ============================================================================
# 4. 衝突無し（組み立てた経路を掃引して最小余裕を確認。全10迷路）
# ============================================================================
@pytest.mark.parametrize("seed", SEEDS)
def test_no_collision_along_the_ideal_path(seed):
    margin = 0.005
    path, v, h, res = _result(seed, mode="slalom", margin=margin)
    turns, runs = _turns_and_runs(path, Direction.N)
    blocks, starts = _geometry_blocks(path, Direction.N, res.turns, runs)
    obstacles = wall_obstacles(v, h)

    worst = math.inf
    for block, start_pose in zip(blocks, starts):
        if not block:
            continue
        poses = poses_along(block, start_pose)
        mc, _ = sweep_clearance(poses, obstacles)
        worst = min(worst, mc)

    print(f"\n[検査4] seed={seed}: 最小余裕={worst*1000:.4f}mm (margin={margin*1000:.1f}mm)")

    assert math.isfinite(worst)
    assert worst >= margin - 1e-6


# ============================================================================
# 5. 否定対照: margin を大きく壊すと通れる半径が無くなる（時間が延びる）
#    空振り側: 壊さなければ同じ結果になる
# ============================================================================
NEG_CONTROL_SEEDS = SEEDS[:3]  # 3迷路で十分（全数は幾何探索が重く時間がかかりすぎる）
BROKEN_MARGIN = 0.05  # 通路幅(0.18m)に対して大きすぎる余裕


@pytest.mark.parametrize("seed", NEG_CONTROL_SEEDS)
def test_margin_negative_control_slows_down_or_forces_spin(seed):
    """margin=0.05（壊した値）にすると、幾何的に通れる半径が無くなって
    その場旋回に降格するターンが増え、結果として時間が延びる（作動側）。"""
    _, _, _, baseline = _result(seed, mode="slalom", margin=0.005)
    v, h = _load(seed)
    path = true_shortest_path(v, h, START, GOALS, Direction.N)  # 経路自体は margin に依存しない
    broken = ideal_time_for_path(path, v, h, Direction.N, mode="slalom", margin=BROKEN_MARGIN)

    n_forced_base = sum(1 for t in baseline.turns if t.radius <= 0.0)
    n_forced_broken = sum(1 for t in broken.turns if t.radius <= 0.0)

    print(
        f"\n[検査5-作動側] seed={seed}: T(margin=0.005)={baseline.total:.3f}s  "
        f"T(margin={BROKEN_MARGIN})={broken.total:.3f}s  "
        f"forced_spin: {n_forced_base}->{n_forced_broken} / {baseline.n_turns}"
    )

    assert broken.total >= baseline.total - 1e-9, "margin を壊しても時間が延びなかった"
    assert n_forced_broken > n_forced_base, "margin を壊してもその場旋回への降格が増えなかった"


def test_margin_positive_control_unchanged():
    """空振り側: margin を壊さず同じ値で2回計算すれば、同じ結果になる。"""
    seed = SEEDS[0]
    v, h = _load(seed)
    path = true_shortest_path(v, h, START, GOALS, Direction.N)
    r1 = ideal_time_for_path(path, v, h, Direction.N, mode="slalom", margin=0.005)
    r2 = ideal_time_for_path(path, v, h, Direction.N, mode="slalom", margin=0.005)

    print(f"\n[検査5-空振り側] seed={seed}: T1={r1.total:.6f}s  T2={r2.total:.6f}s（同じはず）")

    assert r1.total == r2.total
    assert [t.radius for t in r1.turns] == [t.radius for t in r2.turns]
    assert [t.limited_by for t in r1.turns] == [t.limited_by for t in r2.turns]


# ============================================================================
# 6. 高速版 _fast_max_feasible_radius が元の max_feasible_radius と同じ値を返す
# ============================================================================
def test_fast_radius_matches_geometry_module():
    """`classic/ideal.py` の高速化（空間フィルタ・並べ替え・探索上限の絞り込み・
    粗い ds）が `classic.geometry.max_feasible_radius`（元の実装。絞り込み無し）
    と同じ答えを返すことを、実際の迷路のターンで直接照合する。

    `classic/ideal.py` docstring「幾何判定の高速化」節の根拠となる検査。
    元の実装は遅い（1回あたり数秒）ので、サンプルは少数に絞る。
    """
    seed = SEEDS[0]
    v, h = _load(seed)
    obstacles = wall_obstacles(v, h)
    path = true_shortest_path(v, h, START, GOALS, Direction.N)
    turns, runs = _turns_and_runs(path, Direction.N)

    # delta_theta の符号（左右）が両方入るよう、90°ターンを間引いて5個サンプルする。
    sample = [t for t in turns if abs(abs(_turn_delta(t[1], t[2])) - math.pi) > 1e-9][:5]
    assert len(sample) >= 3, "サンプルできる90°ターンが少なすぎる（迷路データを確認）"

    n_checked = 0
    for move_idx, from_dir, to_dir in sample:
        delta_theta = _turn_delta(from_dir, to_dir)
        cell = path[move_idx]
        corner = Pose((cell[0] + 0.5) * CELL_SIZE, (cell[1] + 0.5) * CELL_SIZE, _dir_angle(from_dir))

        r_slow = max_feasible_radius(delta_theta, obstacles, corner, margin=0.005, r_lo=0.02, r_hi=0.40)
        r_fast = _fast_max_feasible_radius(
            delta_theta, obstacles, corner, margin=0.005, r_lo=0.02, r_hi=0.40
        )
        print(f"\n[検査6] cell={cell}: r_slow={r_slow*1000:.4f}mm  r_fast={r_fast*1000:.4f}mm")
        assert math.isclose(r_slow, r_fast, abs_tol=1e-6), (cell, r_slow, r_fast)
        n_checked += 1

    assert n_checked >= 3
