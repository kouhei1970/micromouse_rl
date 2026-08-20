"""tests/test_ideal.py — `classic/ideal.py`（経路から理想時間を出す層）の検査

    .venv/bin/python -m pytest tests/test_ideal.py -q

対象は `competition/mazes/design_turn_v1` の10迷路（真の壁・manifest の d0 付き）。
幾何探索（`classic.geometry.max_feasible_radius` 系）が重いので、同じ
`(seed, mode, margin)` の理想時間計算はテスト間でキャッシュして使い回す
（`_result` 参照）。

検査内容:
  1. `true_shortest_path` の経路長が manifest の d0 と一致する
  2. 同じ経路で `mode="slalom"` が `mode="spin"` より速い（全10迷路）
  3. `TurnPlan.limited_by` が "geometry"/"shared"/"floor" のいずれかで、
     全ターンについて埋まっている（内訳を印字する）
  4. 組み立てた経路を `geometry.sweep_clearance` で掃引し、最小余裕が margin 以上
  5. 否定対照: margin を大きく壊すと、通れる半径が無くなる（その場旋回に降格する）
     ぶん時間が延びる。壊さなければ同じ結果になる（空振り側）
  6. `_fast_max_feasible_radius`（本モジュールの高速版）が
     `classic.geometry.max_feasible_radius`（元の実装）と同じ値を返す
     （速度のための最適化が判定結果を変えていないことの直接照合）
  7. 共有する直線を挟んで隣り合う2つのターンが、比例配分(`allocation="proportional"`)
     では両方とも正の半径を得る（先取り(`allocation="greedy"`)では片方が0に潰れていた）
  8. `_allocate_shared_straight`（共有する直線の比例配分・緩和法）が
     design_turn_v1 全10迷路で例外なく収束する（反復回数の最大値を印字する）
  9. 否定対照（対で）: 緩和法の反復予算(`max_iter`)を実際に必要な回数未満に
     壊すと収束せず例外になる。壊さなければ同じ結果になる（空振り側）。
     加えて、収束の閾値(`tol`)を 1e-1 まで緩めても最終値が変わらないことを確認する
     （🔴 単なる想定ではなく、本ファイル作成時に数理的な理由と乱数検査で
     確かめた。理由は検査9のコメント参照）
  10. `allocation="best"` の `total` が、`"greedy"`・`"proportional"` それぞれの
      `total` 以下であること（全10迷路）。🔴 当初「比例配分は先取り以下になる
      はず」という前提で検査していたが、10迷路中4迷路で実際には比例配分の方が
      わずかに遅いと判明した（半径と直線消費のトレードオフを、どちらの配分も
      モデル化していないため）。コーディネータの裁定により、理想時間は
      両方を計算して速い方（`allocation="best"`）を採る方式に改めた。
  11. `allocation="proportional"` は design_turn_v1 全10迷路で forced_spin
      （その場旋回への降格）が0件であること（構造上の改善として保持する）
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
    _allocate_shared_straight,
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
# キャッシュ: 同じ (seed, mode, margin, allocation) の理想時間計算はテスト間で
# 使い回す（幾何探索が重く、テストごとに計算し直すと現実的な時間に収まらない）。
# ============================================================================
_CACHE = {}


def _result(seed, mode="slalom", margin=0.005, allocation="best"):
    key = (seed, mode, margin, allocation)
    if key not in _CACHE:
        v, h = _load(seed)
        path = true_shortest_path(v, h, START, GOALS, Direction.N)
        res = ideal_time_for_path(path, v, h, Direction.N, mode=mode, margin=margin, allocation=allocation)
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
_LIMITED_BY_VOCAB = {
    "greedy": ("geometry", "prev", "next"),
    "proportional": ("geometry", "shared", "floor"),
}


def test_radius_limits_are_recorded():
    """`limited_by` の値は、そのターンが属する `IdealResult.allocation_used`
    （既定 `allocation="best"` なので迷路ごとに "greedy"/"proportional" のどちらか）
    に応じた語彙のいずれかで埋まっている（`_LIMITED_BY_VOCAB` 参照）。"""
    counts = Counter()
    allocation_counts = Counter()
    n_turns_total = 0
    for seed in SEEDS:
        _, _, _, res = _result(seed, mode="slalom")
        n_turns_total += res.n_turns
        allocation_counts[res.allocation_used] += 1
        vocab = _LIMITED_BY_VOCAB[res.allocation_used]
        for t in res.turns:
            assert t.limited_by in vocab, (seed, res.allocation_used, t)
            counts[t.limited_by] += 1

    print(f"\n[検査3] limited_by の内訳（design_turn_v1 全10迷路・ターン計{n_turns_total}）: "
          f"{dict(counts)}  採用配分: {dict(allocation_counts)}")

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


# ============================================================================
# 7. 共有する直線を挟む隣接2ターンが、両方とも正の半径を得る
# ============================================================================
# design_turn_v1 の seed=41004 から、隣り合う2つのターンが runs=1区画(0.18m)の
# 直線を共有する箇所をそのまま切り出した合成経路（真の壁データはそのまま使うので
# 幾何は現実のもの）。旧実装（このコミット直前の `classic/ideal.py`。経路の先頭から
# 順に処理し「既に前のターンが消費した分を引いた残り長」だけを次のターンに渡す
# 貪欲な先取り）をこの経路にそのまま適用すると、手前のターンが178.154mmを
# 先取りし、次のターンは残り0mmで その場旋回（radius=0.0）に降格していた
# （このコミット作成時に旧実装を直接実行して確認した実測値）。
SHARED_STRAIGHT_SEED = 41004
SHARED_STRAIGHT_CELLS = [(5, 3), (4, 3), (3, 3), (2, 3), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)]
SHARED_STRAIGHT_START_HEADING = Direction.W


def test_shared_straight_is_split_between_neighbours():
    """1区画ぶんの直線を挟んで向き合う2つの90°ターンが、比例配分では両方とも
    正の半径を得る。`allocation="proportional"` を明示する（既定の
    `allocation="best"` は greedy/proportional のうち速い方を選ぶだけなので、
    この合成経路でたまたま先取りが勝つ場合にも比例配分そのものの検査として
    成立させるため）。"""
    v, h = _load(SHARED_STRAIGHT_SEED)
    res = ideal_time_for_path(
        SHARED_STRAIGHT_CELLS, v, h, SHARED_STRAIGHT_START_HEADING, mode="slalom",
        allocation="proportional",
    )

    assert res.n_turns == 2, "この合成経路はターンちょうど2個のはず（切り出し方を確認）"
    radii_mm = [t.radius * 1000.0 for t in res.turns]
    limited = [t.limited_by for t in res.turns]

    print(
        f"\n[検査7] 共有する直線を挟む2ターンの半径: {radii_mm[0]:.3f}mm, {radii_mm[1]:.3f}mm  "
        f"limited_by={limited}（旧実装は 178.154mm / 0.000mm だった）"
    )

    for t in res.turns:
        assert t.radius > 0.0, f"隣接ターンの一方が0に潰れた（旧実装の非対称バグが再発）: {t}"
        assert t.limited_by == "shared", (
            f"両ターンとも需要が共有直線の長さを超えて縮められるはず: {t}"
        )


# ============================================================================
# 8. `_allocate_shared_straight`（比例配分の緩和法）が全10迷路で収束する
# ============================================================================
def test_allocation_converges():
    """design_turn_v1 全10迷路で、共有する直線の比例配分が例外を出さずに収束する。

    `allocation="proportional"` を明示する（既定の `allocation="best"` だと、
    先取りの方が速い迷路では `alloc_iterations` が比例配分を計算していない
    ことを示す 0 のままになる — best 自体は内部で両方を計算するので収束性の
    検査としては成立するが、ここでは比例配分そのものを対象にする）。
    `_result` の呼び出し自体が、収束しなければ `_allocate_shared_straight` の
    中で `RuntimeError` を送出する（`_ideal_slalom` が呼び出し元に伝播させる）
    ので、ここで10迷路すべてをエラー無く読み終えること自体が収束の証拠になる。
    反復回数の最大値を印字する。
    """
    max_iterations = 0
    for seed in SEEDS:
        _, _, _, res = _result(seed, mode="slalom", allocation="proportional")
        assert res.allocation_used == "proportional"
        assert res.alloc_iterations >= 1
        max_iterations = max(max_iterations, res.alloc_iterations)

    print(f"\n[検査8] design_turn_v1 全10迷路: 緩和法の反復回数の最大値={max_iterations}")
    assert max_iterations >= 1


# ============================================================================
# 9. 否定対照（対で）: 緩和法の反復予算(max_iter)を壊すと収束せず例外になる
# ============================================================================
# 🔴 当初の想定（収束の閾値 tol を 1e-1 まで緩めると最終値が変わる）は、実装後に
# 数理的に成り立たないと分かった: `_ideal_slalom` の按分では、ターン k は
# 直線 runs[k]・runs[k+1] の**高々2本**にしか接しない（ターンと直線が交互に
# 並ぶ「鎖」構造）。ある直線の需要は、一度そこを処理すると容量以下になり、
# それ以降どちらの端のターンがさらに縮められても需要は減る一方（増えることは
# 無い）ので、一度満たした直線が後から再び違反することは無い。したがって
# **全直線を1回走査するだけで、どの直線も需要<=容量を同時に満たす状態
# （厳密解）に到達する**。2回目の走査は「もう変化が無い」ことを確認する
# だけの空振りであり、実際 `_allocate_shared_straight` は design_turn_v1
# 全10迷路で反復回数=2に収まる（検査8で確認済み。1回目=実質的な配分、
# 2回目=空振りの確認）。乱数生成した2000件の入力で `tol=1e-9` と `tol=1e-1`
# の最終値を直接比較しても、一度も差が出なかった（本ファイル作成時に確認。
# 乱数検査自体は重いので常設のテストにはしていない）。
# そのため「壊すと結果が変わる」対照は `tol` ではなく `max_iter`（反復回数の
# 予算）で作る: 実際に必要な反復回数（>=2）未満に予算を削ると、収束する前に
# 上限へ達し、黙って打ち切らず例外を投げる（作動側）。予算を壊さなければ
# 何度呼んでも同じ結果になる（空振り側）。`tol` については、上の理由どおり
# 1e-1 まで緩めても結果が変わらないことを別途アサーションで確認する。
_ALLOC_C0_FOR_BUDGET_TEST = [0.19, 0.19]
_ALLOC_RUNS_FOR_BUDGET_TEST = [0.18, 0.18, 0.18]


def test_allocation_budget_negative_control():
    """作動側: 反復予算を実際に必要な回数未満に壊すと、収束せず例外になる。"""
    c_ok, n_ok = _allocate_shared_straight(_ALLOC_C0_FOR_BUDGET_TEST, _ALLOC_RUNS_FOR_BUDGET_TEST)
    assert n_ok >= 2, "この対照は2回以上の反復が要る入力のはず（前提が崩れている）"

    print(f"\n[検査9-作動側] 既定予算(max_iter=100): 反復回数={n_ok}  結果={c_ok}")

    with pytest.raises(RuntimeError):
        _allocate_shared_straight(
            _ALLOC_C0_FOR_BUDGET_TEST, _ALLOC_RUNS_FOR_BUDGET_TEST, max_iter=n_ok - 1
        )


def test_allocation_budget_positive_control_unchanged():
    """空振り側: 予算を壊さなければ、何度呼んでも同じ結果になる。"""
    c1, n1 = _allocate_shared_straight(_ALLOC_C0_FOR_BUDGET_TEST, _ALLOC_RUNS_FOR_BUDGET_TEST)
    c2, n2 = _allocate_shared_straight(_ALLOC_C0_FOR_BUDGET_TEST, _ALLOC_RUNS_FOR_BUDGET_TEST)

    print(f"\n[検査9-空振り側] c1={c1} n1={n1}  c2={c2} n2={n2}（同じはず）")

    assert c1 == c2
    assert n1 == n2


def test_allocation_tolerance_does_not_affect_the_result():
    """`tol` を極端に緩めても最終値が変わらないことを確認する（上のコメント参照）。"""
    tight, n_tight = _allocate_shared_straight(
        _ALLOC_C0_FOR_BUDGET_TEST, _ALLOC_RUNS_FOR_BUDGET_TEST, tol=1e-9
    )
    loose, n_loose = _allocate_shared_straight(
        _ALLOC_C0_FOR_BUDGET_TEST, _ALLOC_RUNS_FOR_BUDGET_TEST, tol=1e-1
    )

    print(f"\n[検査9-tol] tol=1e-9: c={tight} it={n_tight}   tol=1e-1: c={loose} it={n_loose}")

    assert tight == loose


# ============================================================================
# 10. allocation="best" の total が、"greedy"・"proportional" それぞれ以下になる
# ============================================================================
# 🔴 旧検査（`test_fair_allocation_is_not_slower_than_greedy`）は「比例配分は
# 先取り以下になるはず」という前提だったが、design_turn_v1 の10迷路中4迷路
# （41000/41002/41005/41012）で実際には比例配分の方がわずかに遅い
# （+0.44%〜+0.81%）と判明した。半径が大きいほど遠心加速度の制約下で速く
# 曲がれるが直線消費が増えて前後の加速・巡航に使える距離が減る、という
# トレードオフを、先取り・比例配分のどちらもモデル化していないためであり、
# 「共有する直線を公平に分け合う」ことと「常に旧実装以下の時間になる」ことは
# 別の主張だった（前提が誤っていた）。コーディネータの裁定により、理想時間は
# 両方を計算して速い方（`allocation="best"`）を採る方式に改めたので、この
# 検査も「best は両方以下」という、必ず成り立つはずの関係を確認する形にした。
@pytest.mark.parametrize("seed", SEEDS)
def test_best_allocation_is_at_most_each_heuristic(seed):
    """`allocation="best"` の `total` が、`"greedy"`・`"proportional"` 双方の
    `total` 以下であること（全10迷路）。"""
    _, _, _, res_greedy = _result(seed, mode="slalom", allocation="greedy")
    _, _, _, res_proportional = _result(seed, mode="slalom", allocation="proportional")
    _, _, _, res_best = _result(seed, mode="slalom", allocation="best")

    print(
        f"\n[検査10] seed={seed}: T_greedy={res_greedy.total:.3f}s  "
        f"T_proportional={res_proportional.total:.3f}s  T_best={res_best.total:.3f}s  "
        f"採用={res_best.allocation_used}"
    )

    assert res_best.total <= res_greedy.total + 1e-9
    assert res_best.total <= res_proportional.total + 1e-9
    assert res_best.allocation_used in ("greedy", "proportional")
    if res_best.allocation_used == "greedy":
        assert res_best.total == res_greedy.total
    else:
        assert res_best.total == res_proportional.total


# ============================================================================
# 11. 比例配分は design_turn_v1 全10迷路で forced_spin が0件
# ============================================================================
def test_proportional_allocation_has_no_forced_spin():
    """`allocation="proportional"` は design_turn_v1 全10迷路で、その場旋回への
    降格（`TurnPlan.radius<=0.0`）が1件も起きない。

    旧実装（先取り）は5迷路で forced_spin が発生していた（手前のターンが
    共有する直線を先取りし、次のターンの残り長が尽きて降格するため）。
    比例配分はこの非対称を解消したことの構造上の改善として、別検査で保持する
    （`allocation="best"` が先取りを選んだ迷路では forced_spin が残りうるので、
    これは `allocation="proportional"` を明示した検査でなければ意味を持たない）。
    """
    total_forced = 0
    for seed in SEEDS:
        _, _, _, res = _result(seed, mode="slalom", allocation="proportional")
        n_forced = sum(1 for t in res.turns if t.radius <= 0.0)
        total_forced += n_forced
        assert n_forced == 0, f"seed={seed}: 比例配分なのに forced_spin が{n_forced}件ある"

    print(f"\n[検査11] design_turn_v1 全10迷路: 比例配分の forced_spin 合計={total_forced}")
    assert total_forced == 0
