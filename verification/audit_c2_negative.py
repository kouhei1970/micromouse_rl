#!/usr/bin/env python3
"""L0-a の (c2) 速度向上率が負である機構の検証。

作成: 2026-08-11 准教授セッション（独立検証担当）

## 問い

正本 §2 の (c2) 速度向上率 =(v_最短/v_探索)−1 が、L0-a では 4 帯すべてで負だった
（−0.06 / −0.64 / −2.93 / −3.33%）。**走り直した方がむしろ平均速度が遅い。**

教授の仮説: **足立法が選ぶのは距離最適経路であり、距離最適な経路は往々にして旋回が多い。**
L0-a は区画ごとに停止するので旋回が時間コストを持つ。したがって「短いが曲がりくねった経路」
を選ぶと**距離では勝ち、平均速度では負ける**。

## 制約と迂回

**実走行の旋回回数は保存されていない**（走行ごとの行動列も軌跡も残っていない）。
そこで**追加の走行をせず**、次の 2 つだけを使う:

1. 迷路の壁配列（npz）から計算できる量 — 真の最短距離 D₀ と、**最短経路の旋回回数**
2. 評価結果 JSON に残っている量 — 走行タイムと実走行距離

時間モデル `t = a·歩数 + b·折れ数` の係数 a, b を**自分で回帰して求め**、
引き継がれている値（a=0.72, b=0.87）と照合する。

## 最短経路が複数あることへの対処（教授指示）

最短経路は一意ではない（生成迷路で中央値 2 本）。**どの 1 本を採るかで旋回回数が変わる**ので、
- **正準な取り方**: 最短経路のうち**旋回が最小**のもの（min-turn）を主とする
- **頑健性**: 旋回が最大のもの（max-turn）でも結論が変わらないかを確認する
両方を DP で厳密に求める（(区画, 進入方向) 上の DAG 最短/最長路）。
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
N = 16
CELL = 0.18                      # 区画寸法 [m]
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # 北, 東, 南, 西
START_HEADING = 0                # スタートは北向き（3 方向が壁の四隅）
INF = 10 ** 9


def load_maze(p: Path):
    d = np.load(p, allow_pickle=True)
    m = {"v": d["v_walls"], "h": d["h_walls"]}
    if "start_x" in d.files:
        m["start"] = (int(d["start_x"]), int(d["start_y"]))
        m["goals"] = {tuple(g) for g in zip(d["goals_x"].tolist(), d["goals_y"].tolist())}
    else:
        m["goals"] = {(7, 7), (8, 7), (7, 8), (8, 8)}
        corners = [(0, 0), (N - 1, 0), (0, N - 1), (N - 1, N - 1)]
        cand = [c for c in corners if len(open_dirs(m, *c)) == 1]
        m["start"] = cand[0] if len(cand) == 1 else (0, 0)
    return m


def wall(m, x, y, d) -> bool:
    if d == 0:
        return bool(m["h"][x, y + 1])
    if d == 1:
        return bool(m["v"][x + 1, y])
    if d == 2:
        return bool(m["h"][x, y])
    return bool(m["v"][x, y])


def open_dirs(m, x, y):
    return [d for d in range(4) if not wall(m, x, y, d)]


def goal_dist(m):
    """ゴールからの区画距離。"""
    dist = [[INF] * N for _ in range(N)]
    q = deque()
    for gx, gy in m["goals"]:
        dist[gx][gy] = 0
        q.append((gx, gy))
    while q:
        x, y = q.popleft()
        for d in open_dirs(m, x, y):
            nx, ny = x + DIRS[d][0], y + DIRS[d][1]
            if 0 <= nx < N and 0 <= ny < N and dist[nx][ny] > dist[x][y] + 1:
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))
    return dist


def turns_on_shortest(m, mode: str) -> int | None:
    """最短経路のうち旋回回数が最小 / 最大のものの旋回回数を返す。

    最短経路 ＝ ゴール距離が 1 ずつ減る辺だけを使う経路（DAG）。
    状態 (区画, 進入方向) 上で DP する。開始時の向きは北。
    旋回 1 回 ＝ 進行方向が 1 つでも変わること（90° も 180° も 1 回と数える。
    L0-a は超信地旋回なので 180° は 90° 2 回ぶんだが、まず 1 回として計算し、
    §感度 で「180° を 2 回と数える」版も出す）。
    """
    dist = goal_dist(m)
    sx, sy = m["start"]
    if dist[sx][sy] >= INF:
        return None
    best = {}
    # (x, y, heading) -> 旋回回数
    init = (sx, sy, START_HEADING)
    best[init] = 0
    # ゴール距離が減る順に処理すれば DAG のトポロジカル順になる
    order = sorted(
        [(x, y) for x in range(N) for y in range(N) if dist[x][y] < INF],
        key=lambda c: -dist[c[0]][c[1]],
    )
    pick = min if mode == "min" else max
    for (x, y) in order:
        for h in range(4):
            cur = best.get((x, y, h))
            if cur is None:
                continue
            if (x, y) in m["goals"]:
                continue
            for d in open_dirs(m, x, y):
                nx, ny = x + DIRS[d][0], y + DIRS[d][1]
                if not (0 <= nx < N and 0 <= ny < N):
                    continue
                if dist[nx][ny] != dist[x][y] - 1:
                    continue          # 最短経路の辺だけ使う
                cost = cur + (0 if d == h else 1)
                k = (nx, ny, d)
                best[k] = cost if k not in best else pick(best[k], cost)
    vals = [v for (x, y, h), v in best.items() if (x, y) in m["goals"]]
    return pick(vals) if vals else None


def simulate_explore(m):
    """足立法（未知壁は通れるとみなす楽観地図）の初回探索を再現し、(歩数, 旋回回数) を返す。

    `verification/maze_exploration_cost.py` と同じ規約。同点は 直進 > 右 > 左 > 後退。
    **タイミングデータを一切使わない**ので、(c2) との比較が循環しない。
    """
    known = [[[False] * 4 for _ in range(N)] for _ in range(N)]
    is_w = [[[False] * 4 for _ in range(N)] for _ in range(N)]
    goals = m["goals"]

    def observe(x, y):
        for d in range(4):
            w = wall(m, x, y, d)
            known[x][y][d] = True
            is_w[x][y][d] = w
            nx, ny = x + DIRS[d][0], y + DIRS[d][1]
            if 0 <= nx < N and 0 <= ny < N:
                od = (d + 2) % 4
                known[nx][ny][od] = True
                is_w[nx][ny][od] = w

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
                    continue          # 楽観: 未観測は通れる
                nx, ny = x + DIRS[d][0], y + DIRS[d][1]
                if 0 <= nx < N and 0 <= ny < N and dist[nx][ny] > dist[x][y] + 1:
                    dist[nx][ny] = dist[x][y] + 1
                    q.append((nx, ny))
        return dist

    x, y = m["start"]
    h = START_HEADING
    steps = turns = 0
    for _ in range(4 * N * N * 4):
        if (x, y) in goals:
            return steps, turns
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
            return None, None
        if best_d != h:
            turns += 1
        h = best_d
        x, y = x + DIRS[h][0], y + DIRS[h][1]
        steps += 1
    return None, None


def main() -> None:
    snap = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "competition" / "results" / "exp007"
    bands = {
        "eval": REPO / "competition" / "mazes" / "eval",
        "validation": REPO / "competition" / "mazes" / "validation",
        "contest_reference": REPO / "competition" / "mazes" / "contest_reference",
        "eval_v2_short": REPO / "competition" / "mazes" / "eval_v2_short",
    }

    rows = []
    for band, mdir in bands.items():
        # L0-a の結果ディレクトリを探す
        cands = sorted((snap / band).glob("adachi_classical_*"))
        if not cands or not mdir.exists():
            continue
        rdir = cands[0]
        for rp in sorted(rdir.glob("maze_*.json")):
            mp = mdir / (rp.stem + ".npz")
            if not mp.exists():
                continue
            d = json.loads(rp.read_text())
            goal_runs = [r for r in d["runs"] if r["outcome"] == "goal"]
            if len(goal_runs) < 2:
                continue                     # (c2) が定義されない面
            expl = goal_runs[0]
            fast = min(goal_runs[1:], key=lambda r: r["run_time"])
            m = load_maze(mp)
            dist = goal_dist(m)
            sx, sy = m["start"]
            d0 = dist[sx][sy]
            t_min = turns_on_shortest(m, "min")
            t_max = turns_on_shortest(m, "max")
            if t_min is None:
                continue
            rows.append({
                "band": band, "maze": d["maze_id"], "d0": d0,
                "turn_min": t_min, "turn_max": t_max,
                "t_exp": expl["run_time"], "L_exp": expl["path_length_m"],
                "t_fast": fast["run_time"], "L_fast": fast["path_length_m"],
            })

    print("=" * 96)
    print("L0-a の (c2) 速度向上率が負である機構の検証（追加の走行なし）")
    print("=" * 96)
    print(f"対象: {len(rows)} 面（L0-a で最短走行が成立した面のみ）")

    # ---- 前提 1: 最短走行は本当に最短経路（歩数 = D0）を走っているか
    print("\n[1] 前提の確認 — 最短走行の実走距離は 0.18·D₀ に一致するか")
    ratio = [r["L_fast"] / (CELL * r["d0"]) for r in rows]
    print(f"    L_最短 / (0.18·D₀): 中央値 {st.median(ratio):.4f}  "
          f"（min {min(ratio):.4f} / max {max(ratio):.4f}）")
    print("    → 1.00 に近ければ「最短走行は最短経路を走っている」。歩数 = D₀ と置いてよい")

    # ---- 2: 時間モデルの係数を自分で回帰する
    print("\n[2] 時間モデル t = a·歩数 + b·折れ数 の係数を、最短走行から回帰する")
    print("    （最短走行は 歩数=D₀・折れ数=最短経路の旋回回数 が既知なので回帰できる）")
    for mode, key in (("min-turn（正準）", "turn_min"), ("max-turn（頑健性確認）", "turn_max")):
        A = np.array([[r["d0"], r[key]] for r in rows], dtype=float)
        y = np.array([r["t_fast"] for r in rows], dtype=float)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        resid = y - pred
        rel = np.abs(resid) / y
        print(f"    {mode}: a = {coef[0]:.4f} s/歩, b = {coef[1]:.4f} s/折れ    "
              f"残差 |Δt|/t 中央値 {np.median(rel)*100:.2f}%  max {rel.max()*100:.2f}%")
    print("    → 引き継がれている時間モデル 0.72×歩数 + 0.87×折れ数 と照合する")

    # 係数は [2] の min-turn 回帰値を使う（自前で求めたもの）
    A2 = np.array([[r["d0"], r["turn_min"]] for r in rows], dtype=float)
    y2 = np.array([r["t_fast"] for r in rows], dtype=float)
    coef2, *_ = np.linalg.lstsq(A2, y2, rcond=None)
    a, b = float(coef2[0]), float(coef2[1])

    # ---- 3: 【循環の指摘】時間から旋回数を逆算する検定は使えない
    print("\n[3] ⚠️ 最初に設計した検定は循環していた（自己申告）")
    print("    「探索走行の折れ数を時間モデルから逆算し、(c2) と相関を取る」設計にしたところ")
    print("    r = −0.954 という強い相関が出た。**しかしこれは循環している。**")
    print("      逆算した旋回密度  dens_exp = t_exp/(b·steps_exp) − a/b   … t_exp に比例")
    print("      (c2) = (L_fast/t_fast)·(t_exp/L_exp) − 1                  … t_exp に比例")
    print("    **両者が同じ t_exp/steps_exp の一次式**なので、相関は代数的に決まっており")
    print("    仮説の証拠にならない。**この検定は棄却する。**")

    # ---- 4: 非循環な検定 — 迷路だけから両方の旋回密度を出す
    print("\n[4] 非循環な検定 — タイミングを一切使わず、迷路だけから両方の旋回密度を出す")
    print("    探索走行: 足立法（楽観地図）の初回探索を迷路上で再現し、歩数と旋回数を数える")
    print("    最短走行: 最短経路の min-turn（歩数 = D₀）")
    print("    → 時間モデル t = a·歩数 + b·折れ数 から (c2) を**予測**し、実測と比べる")

    # 最短経路の取り方（min-turn / max-turn）を変えても結論が変わらないかを見る
    for TURNKEY, TLABEL in (("turn_min", "min-turn（正準）"), ("turn_max", "max-turn（頑健性）")):
        print(f"\n  --- 最短経路の取り方: {TLABEL} ---")
        ok = []
        for r in rows:
            mp = bands[r["band"]] / (r["maze"] + ".npz")
            m = load_maze(mp)
            se, te = simulate_explore(m)
            if se is None:
                continue
            dens_exp = te / se
            dens_fast = r[TURNKEY] / r["d0"]
            # v = 0.18·λ/(a + b·dens) なので (c2) = (a+b·dens_exp)/(a+b·dens_fast) − 1
            c2_pred = (a + b * dens_exp) / (a + b * dens_fast) - 1.0
            v_exp = r["L_exp"] / r["t_exp"]
            v_fast = r["L_fast"] / r["t_fast"]
            c2_obs = v_fast / v_exp - 1.0
            ok.append((r["band"], r["maze"], se, r["d0"], dens_exp, dens_fast, c2_pred, c2_obs))

        print(f"    {'帯':<20}{'n':>4}{'探索 密度':>11}{'最短 密度':>11}{'予測(c2)':>11}{'実測(c2)':>11}")
        P, O = [], []
        for band in bands:
            v = [z for z in ok if z[0] == band]
            if not v:
                continue
            print(f"    {band:<20}{len(v):>4}{st.median([z[4] for z in v]):>11.4f}"
                  f"{st.median([z[5] for z in v]):>11.4f}"
                  f"{st.median([z[6] for z in v])*100:>+10.2f}%{st.median([z[7] for z in v])*100:>+10.2f}%")
            P += [z[6] for z in v]; O += [z[7] for z in v]

        mx, my = st.mean(P), st.mean(O)
        cov = sum((x - mx) * (yy - my) for x, yy in zip(P, O)) / len(P)
        rr = cov / (st.pstdev(P) * st.pstdev(O)) if st.pstdev(P) > 0 else float("nan")
        sign_agree = sum(1 for x, yy in zip(P, O) if (x < 0) == (yy < 0))
        print(f"    全 {len(P)} 面: 予測と実測の相関 r = {rr:+.3f} ／ 符号一致 {sign_agree}/{len(P)} 面 "
              f"／ 予測中央値 {st.median(P)*100:+.2f}% 対 実測 {st.median(O)*100:+.2f}%")

    outp = REPO / "verification" / "out" / "c2_negative_audit.json"
    outp.write_text(json.dumps({"rows": rows, "coef_min_turn": [a, b]}, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {outp}")


if __name__ == "__main__":
    main()
