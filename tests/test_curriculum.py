#!/usr/bin/env python3
"""exp_020（距離カリキュラム）の単体テスト。

`experiments/exp_020_distance_curriculum/card.md` の §2-3（不活性の bit 一致）と
§6（実装）が要求する性質を検査する。**投入前の必須条件**である。

| # | 検査 | 条文 |
|---|---|---|
| T1 | $D_0$ フィルタが上限を守る（かつ予約帯を飛ばす） | §6-1 |
| T2 | **既定 off のとき seed 列が導入前と同一**（連番・予約帯スキップのみ） | §2-3「不活性」1 |
| T3 | **$D_0$ フィルタは env の乱数を消費しない** | §6-1 |
| T4 | **新しい記録列（4 列）は env の乱数を消費しない**＋記録列の整合 | §2-3「不活性」2 |

**T3・T4 が最も重要**である。**`reset()` は `np_random` から初期擾乱（横 ±0.02 m・
方位 ±10°）を引くので、消費が 1 回ずれるだけで初期姿勢がまるごと変わる**
（exp_019 の是正 2 の実例）。**乱数を消費しないことが、bit 一致の前提**である。

使い方:
    .venv/bin/python tests/test_curriculum.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from mouse.maze6_env import _RESERVED_MAZE_SEEDS, Maze6Env  # noqa: E402
from mouse.maze6_gen import generate_maze, shortest_distances  # noqa: E402

V2 = dict(goal_rule_containment=True, collision_respawn=True, episode_limit_steps=2000)


def _d0(seed: int) -> int:
    m = generate_maze(seed, mode="loop")
    return int(shortest_distances(m["v_walls"], m["h_walls"])[tuple(m["start"])])


def _rng_state(env):
    """env の乱数の内部状態（**これが変われば乱数を消費したということ**）。"""
    return repr(env.np_random.bit_generator.state)


def t1_filter_respects_cap() -> bool:
    print("\n[T1] D0 フィルタが上限を守る")
    ok = True
    for cap, n in ((4, 50), (6, 50), (9, 50)):
        env = Maze6Env(mode="generate", base_seed=8000, gamma=0.995)
        env.set_d0_max(cap)
        seeds = [env._next_maze_seed() for _ in range(n)]
        d0s = [_d0(s) for s in seeds]
        bad = [(s, d) for s, d in zip(seeds, d0s) if d > cap]
        reserved = [s for s in seeds if s in _RESERVED_MAZE_SEEDS]
        print(f"  上限 {cap}: {n} 面  D0 の最大 {max(d0s)}  上限超過 {len(bad)} 件  "
              f"予約帯の混入 {len(reserved)} 件")
        ok = ok and not bad and not reserved
    return ok


def t2_default_off_is_identical() -> bool:
    print("\n[T2] 既定 off のとき seed 列が導入前と同一（連番・予約帯スキップのみ）")
    env = Maze6Env(mode="generate", base_seed=8000, gamma=0.995)
    got = [env._next_maze_seed() for _ in range(200)]
    # 導入前の挙動（連番から予約帯だけを飛ばす）を独立に再現して照合する
    want, s = [], 8000
    while len(want) < 200:
        if s not in _RESERVED_MAZE_SEEDS:
            want.append(s)
        s += 1
    same = got == want
    print(f"  既定の _d0_max = {env._d0_max}（None なら off）")
    print(f"  200 面の一致: {same}（先頭 5 = {got[:5]}）")
    # 予約帯をまたぐ場合も確認する（6000 帯の直前から始める）
    env2 = Maze6Env(mode="generate", base_seed=5995, gamma=0.995)
    got2 = [env2._next_maze_seed() for _ in range(30)]
    skipped = [x for x in range(5995, 6030) if x in _RESERVED_MAZE_SEEDS]
    ok2 = all(x not in _RESERVED_MAZE_SEEDS for x in got2) and len(skipped) > 0
    print(f"  予約帯をまたぐ場合: 混入 0 件 = {ok2}（読み飛ばした帯 {len(skipped)} 個）")
    return same and ok2


def t3_filter_consumes_no_rng() -> bool:
    print("\n[T3] D0 フィルタは env の乱数を消費しない")
    ok = True
    for cap in (None, 4):
        env = Maze6Env(mode="generate", base_seed=8000, gamma=0.995)
        env.reset(seed=123)
        env.set_d0_max(cap)
        before = _rng_state(env)
        for _ in range(20):
            env._next_maze_seed()          # 棄却が何度起きても乱数は引かないはず
        after = _rng_state(env)
        print(f"  上限 {cap}: 20 回の seed 取得で乱数状態が不変 = {before == after}")
        ok = ok and (before == after)
    return ok


def t4_records_consume_no_rng() -> bool:
    print("\n[T4] 新しい記録列は env の乱数を消費しない ＋ 記録列の整合")
    env = Maze6Env(mode="fixed", maze_dir="assets/maze6/loop/validation",
                   maze_seeds=[7000], max_cache=2, gamma=0.995,
                   continuous_potential=True, **V2)
    obs, info = env.reset(seed=4242)
    for k in ("path_len_m", "visit_steps", "min_d_since_respawn", "cell_entries"):
        if k not in info:
            print(f"  🔴 reset の info に {k} が無い")
            return False
    before = _rng_state(env)
    # 🔴 乱数方策だと開始区画から出ないことがあり、入場のカウントが検査されない
    # （最初にこれで空振りさせた）。**前進させて区画をまたがせる。**
    rng = np.random.default_rng(7)

    def drive(_i):
        return np.array([0.9, 0.9]) + 0.05 * rng.uniform(-1, 1, size=2)
    path_prev, mono, respawn_seen, min_reset_ok = 0.0, True, False, True
    ent_prev, ent_mono, cells_seen = info["cell_entries"], True, {tuple(info["cell"])}
    ent_at_respawn = None
    for _i in range(1500):
        obs, r, term, trunc, info = env.step(drive(_i))
        if info["path_len_m"] < path_prev - 1e-12:
            mono = False
        path_prev = info["path_len_m"]
        if info["cell_entries"] < ent_prev:
            ent_mono = False
        if info.get("respawned"):
            respawn_seen = True
            # リスポーンした歩の D は開始区画の距離（= D0）に戻る
            if info["min_d_since_respawn"] != info["d_start"]:
                min_reset_ok = False
            # 🔴 リスポーンの瞬間移動は入場に数えない（定義。駆動による通過だけを数える）
            if ent_at_respawn is None and info["cell_entries"] != ent_prev:
                ent_at_respawn = (ent_prev, info["cell_entries"])
        ent_prev = info["cell_entries"]
        cells_seen.add(tuple(info["cell"]))
        if term or trunc:
            break
    after = _rng_state(env)
    n_uniq = info["n_visited"]
    visits_ok = len(info["visit_steps"]) == n_uniq
    print(f"  {info['cell_entries']} 回の入場を含む走行で乱数状態が不変 = {before == after}"
          f"（リスポーン {'あり' if respawn_seen else 'なし'}）")
    print(f"  path_len_m が単調非減少 = {mono}（最終 {path_prev:.3f} m）")
    print(f"  visit_steps の要素数 {len(info['visit_steps'])} == n_visited {n_uniq} = {visits_ok}")
    print(f"  リスポーン直後の min_d_since_respawn が D0 に戻る = {min_reset_ok}")
    ent_ok = ent_mono and (ent_at_respawn is None)
    print(f"  cell_entries が単調非減少 = {ent_mono}（最終 {ent_prev} 回）")
    print(f"  リスポーンの瞬間移動を入場に数えない = {ent_at_respawn is None}"
          f"{'' if ent_at_respawn is None else f'（{ent_at_respawn} で増えた）'}")
    print(f"  cell_entries {ent_prev} >= 異なる区画数 {len(cells_seen)} = "
          f"{ent_prev >= len(cells_seen)}（再入場を含むので下回らない）")
    ent_ok = ent_ok and ent_prev >= len(cells_seen)
    # 🔴 空振りの検出: 区画を一度も移動していないと、入場のカウントを検査したことにならない
    if ent_prev < 2:
        print("  🔴 区画をまたいでいない ＝ 入場のカウントが検査されていない（空振り）")
        ent_ok = False
    if not respawn_seen:
        print("  ⚠️ この走行ではリスポーンが起きなかった（min D の窓の検査は空振り）")
    return (before == after) and mono and visits_ok and min_reset_ok and ent_ok


def main() -> int:
    print("=" * 78)
    print("exp_020 距離カリキュラムの単体テスト")
    print("=" * 78)
    results = [("T1 フィルタが上限を守る", t1_filter_respects_cap()),
               ("T2 既定 off が導入前と同一", t2_default_off_is_identical()),
               ("T3 フィルタが乱数を消費しない", t3_filter_consumes_no_rng()),
               ("T4 記録列が乱数を消費しない", t4_records_consume_no_rng())]
    print("\n" + "=" * 78)
    n_ok = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✅ PASS' if ok else '🔴 FAIL'}  {name}")
    print(f"\n  {n_ok} / {len(results)} PASS")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
