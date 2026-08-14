#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
監査 043: exp_021 でゴールした 4 走行の軌跡の追跡（「解けた」のか「たまたま届いた」のか）

准教授セッション（8 代目）・2026-08-15・教授指示（`AUDIT_042` 未確認 1 の発注・最優先）

## 問い

**介入群は検証用 20 迷路で 4 回ゴールした。対照群（exp_019 の最終方策）は 120 走行で 0 回である。**
**この 4 回は「方策が解いた」のか「たまたま届いた」のか。**
次の一手（第 2 弾 B へ進むか）の一次入力になる。

## 反証の形（**数える前に決める** — レビュー原則 1）

**「たまたま届いた」が真なら、ゴールまでの $D(t)$ の動きは向きを持たないはず**である。
すなわち **$D$ が減った歩と増えた歩がほぼ同数**になる（$D$ 上の無向な歩き）。
**「解けた」が真なら、減る側に強く偏る。**

- **判別**: $D$ が変化した歩だけを取り、**減少が偏っているかを符号検定（両側）で見る**
  （帰無 = 増減が等確率）。**p < 0.05 なら「向きがある」＝ たまたまでは説明しにくい**
- **⚠️ 限界**: 実際の力学は $D$ 上の無向な歩きではない（物理があり、方策は決定的）。
  **これは「向きがあるか」を測る道具であって、機構の模型ではない。**
  **帰無を棄却できなければ「解けた」を否定できる、という向きには使えない**（検出力の問題）

## 併せて測るもの

- **規約成立までの遅れ**: $D=0$（機体中心がゴール区画）に最初に達した歩と、走行が終わった歩の差。
  **本測定は `goal_rule_containment=True`（機体全体の内包）**なので、
  **中心が入っただけでは成立しない**（`AUDIT_018` → 裁定 R42 の論点）
- **最大の後退**: 走行中の最小 $D$ からどれだけ戻ったか
- **対照群が同じ迷路で何をしたか**（同じ迷路・同じ seed 番号）
- **同じ方策が他の 19 迷路で何をしたか**（その迷路だけが特別なのか、方策全体が良いのか）
"""

import json
import math
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    return json.load(open(os.path.join(REPO, f"outputs/exp_021_driving_{name}.json"),
                          encoding="utf-8"))


def seed_of(name):
    return int(re.search(r"seed(\d+)", name).group(1))


def two_sided_sign_test(down, up):
    """増減が等確率という帰無のもとで、これ以上の偏りが出る確率（両側）。"""
    n = down + up
    if n == 0:
        return 1.0
    k = max(down, up)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def describe(dh, d0):
    ups = sum(1 for a, b in zip(dh, dh[1:]) if b > a)
    downs = sum(1 for a, b in zip(dh, dh[1:]) if b < a)
    first0 = next((i for i, x in enumerate(dh) if x == 0), None)
    # 走行中の最小 D からの最大の後退
    run_min, worst_back = dh[0], 0
    for x in dh:
        run_min = min(run_min, x)
        worst_back = max(worst_back, x - run_min)
    return dict(n_steps=len(dh), d0=d0, ups=ups, downs=downs,
                p=two_sided_sign_test(downs, ups), first0=first0,
                gap=(len(dh) - 1 - first0) if first0 is not None else None,
                worst_back=worst_back, max_d=max(dh), min_d=min(dh))


def main():
    treat, ctrl = load("treat_final"), load("control_final")

    # 介入群のゴール走行を拾う
    goals = []
    for name, b in treat["detail"].items():
        raw = {r["maze_seed"]: r for r in b["raw"]}
        for m in b["metrics"]:
            if m["outcome"] == "goal":
                goals.append((seed_of(name), m, raw[m["maze_seed"]]))
    goals.sort(key=lambda g: (g[0], g[1]["maze_seed"]))

    print("=" * 76)
    print("ゴールした 4 走行の中身")
    print("=" * 76)
    print(f"{'seed':>4}{'迷路':>7}{'D0':>4}{'歩数':>6}{'立直':>5}"
          f"{'D減':>5}{'D増':>5}{'符号検定 p':>11}{'最大後退':>9}{'規約成立の遅れ':>15}")
    for s, m, r in goals:
        d = describe(r["d_hist"], m["d0"])
        print(f"{s:>4}{m['maze_seed']:>7}{d['d0']:>4}{d['n_steps']:>6}{m['n_respawn']:>5}"
              f"{d['downs']:>5}{d['ups']:>5}{d['p']:>11.4f}{d['worst_back']:>9}"
              f"{d['gap']:>15}")

    print("\n【読み】符号検定の p < 0.05 なら「D の動きに向きがある」"
          "＝ 向きのない歩きでは説明しにくい。")
    print("      規約成立の遅れ = 機体中心がゴール区画に入ってから、"
          "機体全体が収まるまでの歩数（AUDIT_018・裁定 R42）。")

    # ---- 対照群は同じ迷路で何をしたか ----
    print("\n" + "=" * 76)
    print("対照群（exp_019 の最終方策）は同じ迷路で何をしたか")
    print("=" * 76)
    cmap = {}
    for name, b in ctrl["detail"].items():
        raw = {r["maze_seed"]: r for r in b["raw"]}
        for m in b["metrics"]:
            cmap[(seed_of(name), m["maze_seed"])] = (m, raw[m["maze_seed"]])
    print(f"{'seed':>4}{'迷路':>7}{'結末':>9}{'歩数':>6}{'最小D':>6}{'D0':>4}"
          f"{'到達した最深':>13}")
    for s, m, _ in goals:
        cm, cr = cmap[(s, m["maze_seed"])]
        print(f"{s:>4}{m['maze_seed']:>7}{cm['outcome']:>9}{cm['n_steps']:>6}"
              f"{cm['min_d']:>6}{cm['d0']:>4}{cm['d0'] - cm['min_d']:>13} 区画")

    # ---- 同じ方策の他の迷路での成績 ----
    print("\n" + "=" * 76)
    print("ゴールを出した方策は、他の 19 迷路でどうだったか"
          "（その迷路だけが特別か・方策全体が良いか）")
    print("=" * 76)
    for s in sorted({s for s, _, _ in goals}):
        name = next(n for n in treat["detail"] if seed_of(n) == s)
        b = treat["detail"][name]
        raw = {r["maze_seed"]: r for r in b["raw"]}
        gm = {m["maze_seed"] for m in b["metrics"] if m["outcome"] == "goal"}
        others = [m for m in b["metrics"] if m["maze_seed"] not in gm]
        reach = [(m["d0"] - m["min_d"]) for m in others]
        print(f"  seed{s}: ゴール {sorted(gm)}／他 {len(others)} 迷路の到達最深 "
              f"中央値 {statistics.median(reach)} 区画・最大 {max(reach)} 区画"
              f"（D0 の中央値 {statistics.median([m['d0'] for m in others])}）")

    # ---- 対照群の到達最深の分布（比較の基準） ----
    print("\n" + "=" * 76)
    print("比較の基準: 両群の「到達した最深（D0 − 最小 D）」の分布")
    print("=" * 76)
    for lab, doc in (("対照", ctrl), ("介入", treat)):
        vals = [m["d0"] - m["min_d"] for b in doc["detail"].values() for m in b["metrics"]]
        # ⚠️ v = D0 − min_d は「到達した最深」。v == 0 は**前進ゼロ**であってゴールではない
        #    （初版でここに「ゴール到達」と誤ったラベルを付けていた。2026-08-15 是正）
        print(f"  {lab}: 中央値 {statistics.median(vals)} / 最大 {max(vals)} / "
              f"5 区画以上 {sum(1 for v in vals if v >= 5)}/120 / "
              f"前進ゼロの走行 {sum(1 for v in vals if v == 0)}/120")
        # D0 に到達した = min_d 0
        n0 = sum(1 for b in doc["detail"].values() for m in b["metrics"] if m["min_d"] == 0)
        print(f"      機体中心がゴール区画に入った走行: {n0}/120")

    print("""
⚠️ 限界:
  1. 符号検定の帰無（D の増減が等確率）は力学の模型ではない。**向きの有無を測る道具**であり、
     棄却できなくても「解けていない」の証拠にはならない（検出力の問題）。
  2. 本監査は D(t) だけを見ている。**実際の経路（どの区画をどう通ったか）は見ていない。**
  3. 4 走行は n=4 である。**方策の能力についての一般化はできない。**
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
