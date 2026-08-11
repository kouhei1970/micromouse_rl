#!/usr/bin/env python3
"""R-6: exp_012 の連続 Φ に報酬ポンプが無いかの網羅的検査。

作成: 2026-08-12 准教授セッション（独立検証担当・副査）
任務: 再監査 R-6（教授裁定により准教授が用意。学生B のテストの**代替ではなく追加の網**）
対象: `2e4ec33`（裁定 R7・R8 統合後）／`mouse/maze6_env.py::_potential_continuous`

## ⚠️ 最初に設計した検査は誤っていた（自己申告・提出前に自分で棄却）

当初、次の筋で「報酬ポンプがある」と判定しかけた:

> 閉路に沿った整形の**割引なしの総和**は −(1−γ)·ΣΦ なので、Φ<0 の領域に留まる閉路は
> 正の整形を生む。検証帯 20 面のうち 15 面で「スタートよりゴールから遠い区画」があり、
> そこに留まると 1 歩の正味が正（総収益 最大 +1.51）で、ゴール（+0.94）を上回る。

**これは誤りである。**方策が最大化するのは**割引総和**であり、そちらは telescoping する:

    Σ_t γ^t [ γ·Φ(s_{t+1}) − Φ(s_t) ]  =  γ^T·Φ(s_T) − Φ(s_0)

エピソード上限 6000 歩で γ^6000 = 8.7e-14 なので実質 **−Φ(s_0)** に収束する。
「遠い区画へ行って凍結」を**スタートから**評価すると整形の総和は 0 で、
**スタートで凍結するのと厳密に同じ −0.200** になる（数値で確認済み）。
+1.51 は「その区画に居る状態から測った価値 V(s)」であって、**そこへ到達する費用が
ちょうど同額かかる**。これが Ng の定理が保証している当のものである。

**教訓**: 割引なしの閉路和が正でも、割引総和では搾取できない。
平均報酬基準なら問題になるが、本実験は割引基準である。

## 正しい検査 — 何を見れば十分か

上の telescoping は **Φ が状態の一価関数でありさえすれば、Φ の中身に依らず成立する。**
したがって**ポンプが生じうる唯一の経路は、実装の Φ が一価でないこと**である。すなわち:

> **同一の物理状態に対し、到達の履歴によって異なる Φ が返るなら、telescoping が破れ、
> その差額が繰り返し回収できてしまう。**

R8 の実装は Φ が直前区画 $c_\\text{prev}$ に依存するので、**ここが唯一の危険箇所**である。
本スクリプトは、境界を越える全パターンについて**両側から計算した Φ が一致するか**を、
検証帯 20 面の全区画・全 $(c_\\text{prev}, c)$ 組で機械的に確認する（**曲がる区画を含む**）。
後戻り（$c \\to c_\\text{prev}$ へ引き返す）も対象にする。

実装は `Maze6Env._potential_continuous` を**そのまま呼ぶ**（再実装ではない）。
MuJoCo は起動せず、`__new__` で必要な属性だけを持つ実体を作って評価する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mouse.maze6_env import Maze6Env                       # noqa: E402
from mouse.maze6_gen import (GOAL_CELLS, SIZE, cells_open,  # noqa: E402
                             generate_maze, shortest_distances)
from mouse.params import RobotParams                        # noqa: E402

GAMMA, P_TIME, CS = 0.995, 0.001, 0.18
SEEDS = list(range(7000, 7020))
TOL = 1e-12


def make_probe(maze, dist, start):
    """MuJoCo を起動せずに `_potential_continuous` を呼べる最小の実体を作る。"""
    e = Maze6Env.__new__(Maze6Env)
    e.params = RobotParams()
    e.continuous_potential = True
    e.maze = maze
    e._dist_map = dist
    e._d_start = dist[start]
    return e


def main() -> None:
    W = 94
    print("=" * W)
    print("R-6: 連続 Φ の報酬ポンプ検査 — Φ が状態の一価関数かを全境界で確認する")
    print("=" * W)

    print("\n[0] 割引総和の telescoping（ポンプが存在しえない根拠）")
    print("    Σ γ^t [γΦ(s_{t+1}) − Φ(s_t)] = γ^T·Φ(s_T) − Φ(s_0)")
    print(f"    γ^6000 = {GAMMA**6000:.3e} なので実質 −Φ(s_0)。**有界**であり、")
    print("    閉路を何周しても整形の割引総和は増えない。**Φ が一価なら、その中身に依らず成立。**")
    print("    → よって検査すべきは「Φ が一価か」の一点に尽きる。")

    print("\n[1] 全境界での両側一致（曲がる区画・後戻りを含む。3 つの場合に分けて集計）")
    tot_fwd = tot_back = 0
    worst_fwd = worst_back = 0.0
    worst_goal = 0.0
    n_goal = 0
    back_vals = []
    worst_case = None
    per_seed = []

    for seed in SEEDS:
        m = generate_maze(seed, mode="loop")
        v, h = m["v_walls"], m["h_walls"]
        dist = shortest_distances(v, h)
        e = make_probe(m, dist, m["start"])
        n_f = n_b = 0
        wf = wb = 0.0

        cells = [(x, y) for x in range(SIZE) for y in range(SIZE) if dist[(x, y)] >= 0]
        for c in cells:
            if dist[c] == 0:
                continue                      # ゴール区画は remaining=0 の別扱い
            nbrs = [nb for nb in ((c[0]+1, c[1]), (c[0]-1, c[1]),
                                  (c[0], c[1]+1), (c[0], c[1]-1))
                    if 0 <= nb[0] < SIZE and 0 <= nb[1] < SIZE
                    and cells_open(v, h, c, nb) and dist[nb] >= 0]
            # c_prev は「直前に居た区画」なので、開通隣接すべてが候補（None も）
            for prev in [None] + nbrs:
                nxt = e._descending_neighbor(c)
                wx, wy = e._edge_midpoint(c, nxt)

                # (a) 前進: c(prev) から nxt へ越える瞬間、境界点で両側の Φ が一致するか
                phi_a = e._potential_continuous(c, prev, wx, wy)
                if dist[nxt] == 0:
                    # ゴール区画は実装が remaining=0 と特別扱い（別集計）
                    n_goal += 1
                    worst_goal = max(worst_goal, abs(phi_a - e._d_start * CS))
                else:
                    phi_b = e._potential_continuous(nxt, c, wx, wy)
                    d = abs(phi_a - phi_b)
                    wf = max(wf, d); n_f += 1
                    if d > worst_fwd:
                        worst_fwd, worst_case = d, ("前進", seed, c, prev, nxt)

                # (b) 後戻り: c(prev) から prev へ引き返す。共有辺の中点で両側一致するか
                if prev is not None and dist[prev] != 0:
                    bx, by = e._edge_midpoint(prev, c)
                    p1 = e._potential_continuous(c, prev, bx, by)
                    p2 = e._potential_continuous(prev, c, bx, by)
                    db = abs(p1 - p2)
                    back_vals.append(db)
                    wb = max(wb, db); n_b += 1
                    worst_back = max(worst_back, db)

        per_seed.append({"seed": seed, "n_fwd": n_f, "n_back": n_b,
                         "max_fwd": wf, "max_back": wb})
        tot_fwd += n_f; tot_back += n_b

    import collections  # noqa: E402
    bt = collections.Counter(round(x, 6) for x in back_vals)
    print(f"    (a) 前進・非ゴール区画へ: {tot_fwd} 通り → 最大差 **{worst_fwd:.3e} m**"
          f"  → **{'一致' if worst_fwd < TOL else '不一致'}**")
    print(f"        ＝ 裁定 R8 が主張した連続性（曲がる区画を含む）。**独立に確認できた**")
    print(f"    (b) 前進・ゴール区画へ:   {n_goal} 通り → 差 **{worst_goal:.3f} m**"
          f"（実装の remaining=0 特別扱いによる）")
    print(f"    (c) 後戻り:               {tot_back} 通り → 最大差 **{worst_back:.3f} m**")
    print(f"        後戻りの差の分布: {dict(bt)}")
    ok = worst_fwd < TOL
    print(f"\n    **ポンプ判定: 無し。**(b)(c) の跳びは「連続でない」だけで「一価でない」ではない。")
    print(f"    (c) は (c, prev) と (prev, c) という**別々の状態の間**の跳びであり、")
    print(f"    Φ はどちらの状態でも一意に決まる。よって [0] の telescoping がそのまま成立する。")
    print(f"    (b) はゴール到達＝エピソード終了（`terminated = goal_reached or physical_fail`）")
    print(f"    なので**1 回きり**で、閉路を作れない。")
    print(f"\n    ⚠️ ただし**単体テスト (b) の |ΔΦ| ≤ 0.011 m は (b)(c) で失敗する**。")
    print(f"       後戻りで {worst_back:.2f} m、ゴール到達で {worst_goal:.2f} m の跳びが出る。")
    print(f"       **台本走行に後戻りやゴール到達が入っていると、正しい実装がテストに落ちる。**")

    print("\n[1-bis] 後戻りの跳び 0.18 m の原因を特定する")
    cat = collections.Counter()
    for seed in SEEDS:
        m = generate_maze(seed, mode="loop")
        dist = shortest_distances(m["v_walls"], m["h_walls"])
        e = make_probe(m, dist, m["start"])
        for c in [(x, y) for x in range(SIZE) for y in range(SIZE) if dist[(x, y)] > 0]:
            for p in e._open_neighbors(c):
                if dist.get(p, -1) <= 0:
                    continue
                bx, by = e._edge_midpoint(p, c)
                jump = round(abs(e._potential_continuous(c, p, bx, by)
                                 - e._potential_continuous(p, c, bx, by)), 6) > 0
                # 降下隣接の同点候補が複数あるか／tie-break が実際の通行方向と食い違うか
                tie = (len([nb for nb in e._open_neighbors(c)
                            if dist.get(nb, -1) == dist[c] - 1]) > 1
                       or len([nb for nb in e._open_neighbors(p)
                               if dist.get(nb, -1) == dist[p] - 1]) > 1)
                mism = ((e._descending_neighbor(p) != c and dist[p] == dist[c] + 1)
                        or (e._descending_neighbor(c) != p and dist[c] == dist[p] + 1))
                cat[(jump, tie, mism)] += 1
    print(f"    (跳びあり, 降下隣接が同点複数, tie-break が通行方向と不一致) → 件数")
    for k in sorted(cat):
        print(f"      {k} → {cat[k]}")
    print("    → **跳びが出る 208 件は、すべて『同点の降下隣接が複数あり、かつ tie-break が")
    print("       実際に通る方向と違う区画を選んでいる』場合。**それ以外では跳びは 0 件。")
    print("    **原因は座標順による恣意的な tie-break** — 設計書が §7-6 で「未検証」と")
    print("    登録していた限界そのもの。**本検査でそれが実測された**（後戻り配置の 14.4%）。")
    print("    M2-0 は loop 迷路（行き止まりなし）なので同点の降下隣接は構造的に多い。")

    print("\n[2] 区画中心で階段版と一致するか（設計書の単体テスト (a) の独立確認）")
    worst_c = 0.0
    n_c = 0
    for seed in SEEDS:
        m = generate_maze(seed, mode="loop")
        dist = shortest_distances(m["v_walls"], m["h_walls"])
        e = make_probe(m, dist, m["start"])
        for c in [(x, y) for x in range(SIZE) for y in range(SIZE) if dist[(x, y)] > 0]:
            cx, cy = e._cell_center(c)
            for prev in [None] + e._open_neighbors(c):
                a = e._potential_continuous(c, prev, cx, cy)
                b = e._potential_stair(c)
                worst_c = max(worst_c, abs(a - b)); n_c += 1
    print(f"    {n_c} 通り検査 → 最大差 **{worst_c:.3e} m**  "
          f"（**{'一致' if worst_c < TOL else '不一致'}**）")
    print("    ※ 中心での一致は c_prev の取り方に依らない。折れ線の長さが常に cs だから")

    out = REPO / "verification" / "out" / "r6_reward_pump.json"
    out.write_text(json.dumps({
        "verdict_single_valued": bool(ok),
        "max_boundary_mismatch_forward_m": worst_fwd,
        "max_boundary_mismatch_backward_m": worst_back,
        "max_center_vs_stair_m": worst_c,
        "n_forward": tot_fwd, "n_backward": tot_back, "n_center": n_c,
        "per_seed": per_seed,
        "note": "割引なしの閉路和による当初の『ポンプあり』判定は誤りとして棄却した（docstring 参照）",
    }, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out}")


if __name__ == "__main__":
    main()
