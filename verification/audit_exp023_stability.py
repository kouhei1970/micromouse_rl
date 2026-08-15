#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
W-j: exp_023 の学習の安定性（`AUDIT_048_PREREG` §4・**判定には使わない記述**）

准教授セッション（9 代目）・2026-08-15・教授裁定で実施 GO

## 問い（悪化の機構の読みに効く）

**群 1・群 2 は対照より大幅に悪い**（深さ 12 → 0・9／衝突 2.125 → 7.25・11.625）。
**その悪化は「学習が不安定になった結果」なのか、「安定してこの方策に収束した結果」なのか。**

**この 2 つは次の一手を変える**:
- **不安定なら** → 学習率・系列長など `RecurrentPPO` の設定の問題であり、**再帰構造の否定にはならない**
- **安定して収束したなら** → **設定ではなく方策の表現の問題**であり、**限界 3（隠れ 32 では足りない）が効く**

## 見る量（`progress.csv`・追記型）

| 量 | 不安定の兆候 |
|---|---|
| `train/approx_kl` | 更新ごとに大きく振れる・後半で増える |
| `train/clip_fraction` | 飽和（PPO の信頼領域に当たり続ける） |
| `train/explained_variance` | 崩れる（価値関数が学習できていない） |
| `train/std` | 発散（行動の分散が広がり続ける ＝ 探索へ後退） |
| `rollout/ep_rew_mean` | 後半で下がる |

**3 群（対照 exp_021・群 1・群 2）を同じ量で並べる。**対照が基準線になる。
"""

import csv
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ("train/approx_kl", "train/clip_fraction", "train/explained_variance",
        "train/std", "rollout/ep_rew_mean", "train/value_loss")


def series(prefix, n):
    p = os.path.join(REPO, f"logs/{prefix}_seed{n}/progress.csv")
    if not os.path.exists(p):
        return None
    out = {c: [] for c in COLS}
    out["steps"] = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if not r.get("time/total_timesteps"):
            continue
        out["steps"].append(int(float(r["time/total_timesteps"])))
        for c in COLS:
            v = r.get(c)
            out[c].append(float(v) if v not in (None, "") else float("nan"))
    return out


def half(vals, steps, first):
    """前半（<= 100 万歩）／後半（> 100 万歩）の中央値。"""
    sel = [v for v, s in zip(vals, steps)
           if (s <= 1_000_000) == first and v == v]
    return statistics.median(sel) if sel else float("nan")


def main():
    groups = {"対照 exp_021": "exp_021", "群 1（再帰）": "exp_023a", "群 2（+リセット）": "exp_023b"}
    print("=" * 78)
    print("W-j: 学習の安定性（判定には使わない記述）")
    print("=" * 78)
    print("各群 6 seed の中央値。前半 = 100 万歩まで／後半 = それ以降\n")

    data = {}
    for tag, pre in groups.items():
        ss = [series(pre, n) for n in range(1, 7)]
        ss = [s for s in ss if s]
        if not ss:
            print(f"  {tag}: progress.csv が無い")
            continue
        data[tag] = ss

    print(f"{'量':<26}" + "".join(f"{t:>22}" for t in data))
    for c in COLS:
        for lbl, first in (("前半", True), ("後半", False)):
            row = f"  {c.split('/')[-1]:<20} {lbl:<4}"
            for tag, ss in data.items():
                vals = [half(s[c], s["steps"], first) for s in ss]
                vals = [v for v in vals if v == v]
                row += f"{statistics.median(vals):>22.4f}" if vals else f"{'—':>22}"
            print(row)
        print()

    # 発散していないか（後半 / 前半 の比）
    print("--- 後半 / 前半 の比（1 に近いほど定常。大きく動けば不安定の兆候）---")
    print(f"{'量':<26}" + "".join(f"{t:>22}" for t in data))
    for c in COLS:
        row = f"  {c.split('/')[-1]:<24}"
        for tag, ss in data.items():
            rs = []
            for s in ss:
                a, b = half(s[c], s["steps"], True), half(s[c], s["steps"], False)
                if a == a and b == b and a != 0:
                    rs.append(b / a)
            row += f"{statistics.median(rs):>22.3f}" if rs else f"{'—':>22}"
        print(row)

    # 個々の seed が壊れていないか
    #
    # 🔴 閾値は**対照で校正する**。固定値（例: approx_kl > 0.5）を使うと
    #    **対照 6 本を含む 18 本すべてで作動して判別力がゼロになる**（実測でそうなった）。
    #    「必ず立つ旗は旗ではない」（中核原則 3）を、自分の検出器にも当てる。
    print("\n--- 崩れた seed の検出（**閾値は対照で校正する**）---")
    ctrl = data.get("対照 exp_021")
    kl_max_ctrl = max(max(v for v in s["train/approx_kl"] if v == v) for s in ctrl)
    ev_min_ctrl = min(min(v for v in s["train/explained_variance"] if v == v) for s in ctrl)
    print(f"  対照の範囲: approx_kl の最大 = {kl_max_ctrl:.2f}／"
          f"explained_variance の最小 = {ev_min_ctrl:.3f}")
    print(f"  → **対照の範囲を超えたものだけ**を「崩れた」と呼ぶ（超えなければ対照と同じ挙動）")
    for tag, ss in data.items():
        bad = []
        for i, s in enumerate(ss, 1):
            kl = max(v for v in s["train/approx_kl"] if v == v)
            ev = min(v for v in s["train/explained_variance"] if v == v)
            if kl > kl_max_ctrl:
                bad.append(f"seed{i}(approx_kl {kl:.1f} > 対照の最大 {kl_max_ctrl:.1f})")
            if ev < ev_min_ctrl:
                bad.append(f"seed{i}(explained_variance {ev:.2f} < 対照の最小 {ev_min_ctrl:.2f})")
        print(f"  {tag}: {'・'.join(bad) if bad else '🟢 対照の範囲を超えた seed なし'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
