#!/usr/bin/env python3
"""裁定 R4 関連: (c2) 監査は n_turns の定義差の影響を受けるか。

作成: 2026-08-12 准教授セッション（独立検証担当・副査）
発端: 学生A からの確認依頼（教授経由）。裁定 R4 で方式間比較の正本は
      「区画列の進行方向変化」に決まった。前任准教授の (c2) 監査
      （L0-a は 4 帯すべてで速度向上率が負）がこの定義差の影響を受けるか。

## 答えの骨子（コード読解で確定）

`audit_c2_negative.py` は**記録された n_turns を一切読んでいない**。
結果 JSON から読むのは `outcome` / `run_time` / `path_length_m` の 3 つだけで（226〜244 行）、
旋回回数は**すべて壁配列から自分で数えている**:

- `turns_on_shortest`（134 行 `cost = cur + (0 if d == h else 1)`）… 区画列の進行方向変化
- `simulate_explore`（198〜199 行 `if best_d != h: turns += 1`）… 同上

**どちらも裁定 R4 の正本（区画列定義）そのもの。**学生B の ±45° リセット定義は使われていない。
→ **(c2) 監査は定義差の影響を受けない。**

## ただし未実装の感度が 1 件あった（自己申告）

`turns_on_shortest` の docstring は「180° を 2 回と数える版も §感度 で出す」と書いているが、
**その版は実装されていない。**本スクリプトでそれを実施する。

- **最短走行には 180° は出ない**（ゴール距離が毎歩 1 ずつ減るので反転は起こりえない）
  → `turn_min` / `turn_max` は 180° の数え方に**不感**
- **探索走行には 180° が出る**（行き止まりからの引き返し）→ `dens_exp` は変わりうる
- L0-a は超信地旋回なので、180° の所要時間は 90° の約 2 倍。**時間モデルの立場では 2 と数えるのが妥当**

⚠️ 教授指示により (c2) 系の数値は確定扱いにしない（帯の再作成待ち）。本スクリプトは
**定義差が結論の向きを変えるかどうか**だけを見る。
"""

from __future__ import annotations

import json
import statistics as st
from collections import deque
from pathlib import Path

import numpy as np

import audit_c2_negative as c2

REPO = Path(__file__).resolve().parent.parent
N, INF = c2.N, c2.INF
DIRS = c2.DIRS


def simulate_explore_weighted(m, half_turn_cost: int):
    """`audit_c2_negative.simulate_explore` と同一だが、180° の重みを可変にする。

    half_turn_cost=1 … 現行（180° も 1 回）
    half_turn_cost=2 … 180° を 90° 2 回ぶんと数える
    """
    known = [[[False] * 4 for _ in range(N)] for _ in range(N)]
    is_w = [[[False] * 4 for _ in range(N)] for _ in range(N)]
    goals = m["goals"]

    def observe(x, y):
        for d in range(4):
            w = c2.wall(m, x, y, d)
            known[x][y][d] = True
            is_w[x][y][d] = w
            nx, ny = x + DIRS[d][0], y + DIRS[d][1]
            if 0 <= nx < N and 0 <= ny < N:
                known[nx][ny][(d + 2) % 4] = True
                is_w[nx][ny][(d + 2) % 4] = w

    def flood():
        dist = [[INF] * N for _ in range(N)]
        q = deque()
        for gx, gy in goals:
            dist[gx][gy] = 0
            q.append((gx, gy))
        while q:
            x, y = q.popleft()
            for d in range(4):
                if known[x][y][d] and is_w[x][y][d]:
                    continue
                nx, ny = x + DIRS[d][0], y + DIRS[d][1]
                if 0 <= nx < N and 0 <= ny < N and dist[nx][ny] > dist[x][y] + 1:
                    dist[nx][ny] = dist[x][y] + 1
                    q.append((nx, ny))
        return dist

    x, y = m["start"]
    h = c2.START_HEADING
    steps = turns = n180 = 0
    for _ in range(4 * N * N * 4):
        if (x, y) in goals:
            return steps, turns, n180
        observe(x, y)
        dist = flood()
        best_d, best_v = None, None
        for d in [h, (h + 1) % 4, (h + 3) % 4, (h + 2) % 4]:
            if known[x][y][d] and is_w[x][y][d]:
                continue
            nx, ny = x + DIRS[d][0], y + DIRS[d][1]
            if not (0 <= nx < N and 0 <= ny < N):
                continue
            if best_v is None or dist[nx][ny] < best_v:
                best_v, best_d = dist[nx][ny], d
        if best_d is None:
            return None, None, None
        if best_d != h:
            if (best_d - h) % 4 == 2:
                turns += half_turn_cost
                n180 += 1
            else:
                turns += 1
        h = best_d
        x, y = x + DIRS[h][0], y + DIRS[h][1]
        steps += 1
    return None, None, None


def main() -> None:
    W = 92
    print("=" * W)
    print("裁定 R4: (c2) 監査は n_turns の定義差の影響を受けるか")
    print("=" * W)

    print("\n[1] 記録された n_turns への依存（コード読解）")
    src = (REPO / "verification" / "audit_c2_negative.py").read_text()
    print(f"    'n_turns' の出現回数: **{src.count('n_turns')} 回**")
    print("    結果 JSON から読むキー: outcome / run_time / path_length_m のみ")
    print("    旋回回数は turns_on_shortest・simulate_explore が壁配列から自分で数える")
    print("    → **どちらも区画列の進行方向変化 ＝ 裁定 R4 の正本。定義差の影響を受けない。**")

    rows = json.loads((REPO / "verification" / "out" / "c2_negative_audit.json").read_text())["rows"]
    bands = {b: REPO / "competition" / "mazes" / b
             for b in ("eval", "validation", "contest_reference", "eval_v2_short")}
    A = np.array([[r["d0"], r["turn_min"]] for r in rows], dtype=float)
    y = np.array([r["t_fast"] for r in rows], dtype=float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])

    print("\n[2] 未実装だった感度 — 180° を 90° 2 回ぶんと数えたら結論は変わるか")
    print("    最短走行: ゴール距離が毎歩 1 減るので **180° は原理的に出ない**（数え方に不感）")

    recs, missing = [], []
    for r in rows:
        mp = bands[r["band"]] / (r["maze"] + ".npz")
        if not mp.exists():
            # eval / validation は 19:03 の v3 凍結で中身が入れ替わり、
            # 当時の面がもう存在しない。**黙って落とさず件数を出す。**
            missing.append(f"{r['band']}/{r['maze']}")
            continue
        m = c2.load_maze(mp)
        s1, t1, n180 = simulate_explore_weighted(m, 1)
        s2, t2, _ = simulate_explore_weighted(m, 2)
        if s1 is None:
            continue
        d_fast = r["turn_min"] / r["d0"]
        pred = lambda te, se: (a + b * (te / se)) / (a + b * d_fast) - 1.0
        recs.append({
            "band": r["band"], "n180": n180, "steps": s1,
            "pred_1": pred(t1, s1), "pred_2": pred(t2, s2),
            "obs": (r["L_fast"] / r["t_fast"]) / (r["L_exp"] / r["t_exp"]) - 1.0,
        })

    n180_tot = [x["n180"] for x in recs]
    print(f"    探索走行の 180° 回数: 中央値 {st.median(n180_tot):.1f} / "
          f"最大 {max(n180_tot)} （{len(recs)} 面）")
    print(f"    180° が 1 回も出ない面: {sum(1 for v in n180_tot if v == 0)} / {len(recs)} 面")

    print(f"\n    {'帯':<20}{'n':>4}{'予測(180°=1)':>15}{'予測(180°=2)':>15}{'実測':>12}")
    for band in bands:
        v = [x for x in recs if x["band"] == band]
        if not v:
            continue
        print(f"    {band:<20}{len(v):>4}"
              f"{st.median([x['pred_1'] for x in v])*100:>+14.2f}%"
              f"{st.median([x['pred_2'] for x in v])*100:>+14.2f}%"
              f"{st.median([x['obs'] for x in v])*100:>+11.2f}%")
    p1 = st.median([x["pred_1"] for x in recs])
    p2 = st.median([x["pred_2"] for x in recs])
    ob = st.median([x["obs"] for x in recs])
    neg1 = sum(1 for x in recs if x["pred_1"] < 0)
    neg2 = sum(1 for x in recs if x["pred_2"] < 0)
    print(f"\n    全 {len(recs)} 面 中央値: 予測(180°=1) {p1*100:+.2f}% / "
          f"予測(180°=2) {p2*100:+.2f}% / 実測 {ob*100:+.2f}%")
    print(f"    予測が負になる面数: 180°=1 で {neg1}/{len(recs)} 面、180°=2 で {neg2}/{len(recs)} 面")

    verdict = "変わらない" if (p1 < 0) == (p2 < 0) else "**変わる**"
    print(f"\n    → 「(c2) は負」という結論の向きは {verdict}。")

    out = REPO / "verification" / "out" / "r4_nturns_impact.json"
    out.write_text(json.dumps({"coef": [a, b], "recs": recs,
                               "median": {"pred_1": p1, "pred_2": p2, "obs": ob}},
                              ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out}")


if __name__ == "__main__":
    main()
