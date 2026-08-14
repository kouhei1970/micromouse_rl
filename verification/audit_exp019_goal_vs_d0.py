#!/usr/bin/env python3
"""監査: exp_019 の学習側ゴール 43 件から **P(ゴール | D₀)** を出す

動機
----
`AUDIT_028` の T0 で **seed4 のゴール面が 7011（$D_0$=9）** と出た。
v1（exp_012）では `AUDIT_020`・`AUDIT_021` が
「**非ゼロのゴール率が出た面は {7010, 7017} = $D_0$ 最小(=4) の集合と完全一致**」
（$D_0$=5 の面すら 0 回）と確定しており、**その型から外れている**。

**そこで「$D_0$ が小さいほど易しい」という単調性が v2 でも成り立つかを、
$n$=1 の観測ではなく学習側の全エピソード（6,023 本）で検定する。**

独立性
------
- $D_0$ は `generate_maze()` が返す**壁配列から自前の BFS**（`shortest_distances()` は呼ばない）
- ゴールの有無は `episode_seeds.jsonl` の `outcome`（学習側の一次記録）

使い方: `.venv/bin/python verification/audit_exp019_goal_vs_d0.py`
"""
from __future__ import annotations

import collections
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mouse.maze6_gen import generate_maze  # noqa: E402

SEEDS = [1, 2, 3, 4, 5, 6]
MIN_N = 30            # この本数未満の帯は率を出さない（刻みが粗すぎるため）


def d0_of(m) -> int:
    """壁配列から自前の BFS で D₀ を出す。"""
    v, h = m["v_walls"], m["h_walls"]
    w, hh = v.shape[0] - 1, h.shape[1] - 1
    start = tuple(int(x) for x in m["start"])
    goals = {tuple(int(x) for x in g) for g in m["goal_cells"]}

    def conn(x, y, nx, ny):
        if nx == x + 1:
            return v[x + 1, y] == 0
        if nx == x - 1:
            return v[x, y] == 0
        if ny == y + 1:
            return h[x, y + 1] == 0
        return h[x, y] == 0

    dist, q = {start: 0}, deque([start])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < hh and (nx, ny) not in dist \
                    and conn(x, y, nx, ny):
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return min(dist[g] for g in goals if g in dist)


def main() -> int:
    ep, go, cache = collections.Counter(), collections.Counter(), {}
    for n in SEEDS:
        p = Path(f"logs/exp_019_v2_seed{n}/episode_seeds.jsonl")
        for line in p.open():
            r = json.loads(line)
            s = r["maze_seed"]
            if s not in cache:
                cache[s] = d0_of(generate_maze(s, mode="loop"))
            ep[cache[s]] += 1
            if r.get("outcome") == "goal":
                go[cache[s]] += 1

    print("=" * 62)
    print("監査: P(ゴール | D₀) — exp_019 学習側の全エピソード")
    print("=" * 62)
    print(f"  エピソード {sum(ep.values()):,} 本 / 迷路 {len(cache):,} 面 / "
          f"ゴール {sum(go.values())} 件")
    print(f"{'D₀':>4}{'エピソード':>11}{'ゴール':>8}{'率':>10}")
    for k in sorted(ep):
        if ep[k] >= MIN_N:
            print(f"{k:>4}{ep[k]:>11,}{go[k]:>8}{go[k]/ep[k]:>10.4f}")

    lo_e = sum(ep[k] for k in ep if k <= 5)
    lo_g = sum(go[k] for k in go if k <= 5)
    hi_e = sum(ep[k] for k in ep if k >= 6)
    hi_g = sum(go[k] for k in go if k >= 6)
    print("-" * 62)
    print(f"  D₀ ≤ 5 : {lo_g:>3} / {lo_e:,} = {lo_g/lo_e:.4f}")
    print(f"  D₀ ≥ 6 : {hi_g:>3} / {hi_e:,} = {hi_g/hi_e:.4f}"
          f"   → **{(lo_g/lo_e)/(hi_g/hi_e):.0f} 倍の差**")
    print("=" * 62)

    out = Path(__file__).resolve().parent / "out" / "exp019_goal_vs_d0.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"episodes": dict(sorted(ep.items())), "goals": dict(sorted(go.items())),
         "low_rate": lo_g / lo_e, "high_rate": hi_g / hi_e,
         "n_mazes": len(cache)}, ensure_ascii=False, indent=2))
    print(f"出力: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
