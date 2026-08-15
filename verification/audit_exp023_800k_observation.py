#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
exp_023 群 1 の 80 万歩時点の記述（**判定には使わない**）

准教授セッション（9 代目）・2026-08-15

**目的 2 つ**:
1. **学生B の報告トリガーの測定を、一次記録から独立に再計算して照合する**
2. **対照 exp_021 の「80 万歩 → 最終」の動きを使って、完走後に何が起きそうかを先に見る**

**🔴 これは記述であり判定ではない。**判定は `AUDIT_048_PREREG`（`8ac0994`）の条文で、
**完走後の最終方策に対してのみ**行う。**本スクリプトの出力を判定に使ってはならない。**
"""

import json
import statistics


def load(name):
    return json.load(open(f"outputs/{name}", encoding="utf-8"))


def seed_medians(doc, field):
    """seed ごとに 20 迷路の中央値（`measure_driving.py` の集約と同じ形）。"""
    return [statistics.median([m[field] for m in v["metrics"]]) for v in doc["detail"].values()]


def med(doc, field):
    return statistics.median(seed_medians(doc, field))


def main():
    g1 = load("exp_023a_driving_800k.json")
    c8 = load("exp_021_driving_treat_800k.json")
    cf = load("exp_021_driving_treat_final.json")

    print("=" * 74)
    print("exp_023 群 1・80 万歩時点の記述（判定には使わない）")
    print("=" * 74)

    for f in ("net_progress_per_1000", "respawn_per_1000"):
        print(f"\n--- {f} ---")
        for tag, d in (("群 1 80 万歩", g1), ("対照 80 万歩", c8), ("対照 最終", cf)):
            print(f"  {tag:<12} 中央値 {med(d, f):>6}   seed ごと {sorted(seed_medians(d, f))}")

    # 1. 報告トリガー（カード §4-2）の照合
    npg = med(g1, "net_progress_per_1000")
    trig = 0.5 * med(c8, "net_progress_per_1000")
    below = sum(1 for v in seed_medians(g1, "net_progress_per_1000") if v <= trig)
    print(f"\n--- 報告トリガー ---")
    print(f"  作動線 {trig}（対照 80 万歩の中央値の 0.5 倍）／群 1 は {npg} → "
          f"{'🔴 作動' if npg <= trig else '不作動'}（下回る seed {below} 本）")

    # 2. 衝突: 完全分離しているか（n=6 対 6 で判別できるのはここまで）
    a, b = seed_medians(g1, "respawn_per_1000"), seed_medians(c8, "respawn_per_1000")
    print(f"\n--- 衝突（記述）---")
    print(f"  群 1 / 対照 の中央値の比 = {med(g1,'respawn_per_1000') / med(c8,'respawn_per_1000'):.2f} 倍")
    print(f"  群 1 の最小 {min(a)} 対 対照の最大 {max(b)} → "
          f"{'🔴 完全分離（重なりゼロ）' if min(a) > max(b) else '重なりあり'}")

    # 3. 対照の 80 万歩 → 最終 の動きで外挿する（**記述。判定ではない**）
    r = med(cf, "respawn_per_1000") / med(c8, "respawn_per_1000")
    proj = med(g1, "respawn_per_1000") * r
    print(f"\n--- 完走後に何が起きそうか（外挿・記述）---")
    print(f"  対照は 80 万歩 → 最終で衝突が {r:.2f} 倍になった")
    print(f"  群 1 が同じ比で動くなら最終は {proj:.2f}")
    print(f"  R4 の閾値は 2.125 以下（同値は当たり）→ 外挿値は閾値の {proj / 2.125:.1f} 倍")
    print(f"\n  ⚠️ 外挿は「対照と同じ比で動く」という仮定に依存する。判定は完走後の実測で行う。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
