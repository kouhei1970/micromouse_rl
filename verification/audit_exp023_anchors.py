#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
監査 048（事前登録の付属）: exp_023 の判定に使う「対照 exp_021 の錨」の独立再計算

准教授セッション（9 代目）・2026-08-15

## なぜ結果が出る前に走らせるか

**exp_023 の閾値は、すべて対照 exp_021 の実測値から導かれている**（カード §3-1）:

| 予測 | 閾値 | 導出 |
|---|---|---|
| R1 | `n_reach_ge7` ≥ 24 件 | **対照の 12 件の 2 倍** |
| R3 | `net_progress_per_1000` < 1.563 | **対照の 1.250 の 1.25 倍** |
| R4 | `respawn_per_1000` ≤ 2.125 | **対照の値そのもの** |
| R5 | ゴール率 < 0.05 | 対照 0.000（絶対値） |
| 報告トリガー | 80 万歩の `net_progress_per_1000` ≤ 0.5625 | **対照の 80 万歩 1.125 の 0.5 倍** |

**錨が違えば閾値が違う。**したがって**錨は exp_023 の結果を 1 件も見ていない時点で
自分で再計算し、事前登録に固定しておく**（`AUDIT_046_PREREG` §2 の L4 と同じ考え方）。

**本スクリプトは exp_023 のデータを一切読まない。**読むのは exp_021 の測定出力だけである。

## 用語

警報・トリガーが働くこと = **作動**／条文・判定条件 = **成立**。
"""

import json
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")


def load(name):
    for cand in (name, name.replace(".json", ".slim.json")):
        p = os.path.join(OUT, cand)
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8")), cand
    return None, None


def seed_medians(doc, key):
    """seed ごとに 20 迷路の中央値を取る（カード §3-1 の集約）。"""
    return [statistics.median([m[key] for m in b["metrics"]])
            for b in doc["detail"].values()]


def agg_median(doc, key):
    return statistics.median(seed_medians(doc, key))


def depths(doc):
    """120 走行それぞれの「到達した最深」= D0 − min_d。"""
    return [m["d0"] - m["min_d"] for b in doc["detail"].values() for m in b["metrics"]]


def count_goal(doc):
    return sum(1 for b in doc["detail"].values() for m in b["metrics"]
               if m["outcome"] == "goal")


def main():
    print("=" * 78)
    print("exp_023 の判定に使う錨（対照 exp_021）の独立再計算 — exp_023 は一切読まない")
    print("=" * 78)

    fin, fin_src = load("exp_021_driving_treat_final.json")
    k800, k800_src = load("exp_021_driving_treat_800k.json")
    if fin is None:
        print("🔴 exp_021 の最終方策の測定が見つからない")
        return 2

    print(f"\n出所: 最終方策 = {fin_src} / 80 万歩 = {k800_src}")

    n_run = sum(len(b["metrics"]) for b in fin["detail"].values())
    print(f"走行数: {n_run}（6 seed × 検証用の 20 迷路〔seed 7000〜7019〕）")

    d = depths(fin)
    print("\n--- 到達した最深の分布（対照 exp_021・最終方策）---")
    print(f"  {'深さ以上':>8}{'件数':>8}")
    for k in (5, 6, 7, 8, 10):
        print(f"  {k:>8}{sum(1 for v in d if v >= k):>8}")

    ge7 = sum(1 for v in d if v >= 7)

    # R1・R6 は「件数」の比較なので、seed 間のばらつきが判別力を決める。
    # 群 2 対 群 1（R6）の検出力を事前に見積もるため、対照の seed ごとの内訳を出す。
    per_seed_ge7 = sorted(sum(1 for m in b["metrics"] if (m["d0"] - m["min_d"]) >= 7)
                          for b in fin["detail"].values())
    print(f"\n  seed ごとの内訳（7 区画以上・各 20 走行中）: {per_seed_ge7}"
          f"／合計 {sum(per_seed_ge7)}")
    print(f"  → 12 件は 6 seed に均等ではなく、{sum(1 for c in per_seed_ge7 if c == 0)} 本が 0 件。"
          f"**件数の比較は少数の seed に支配される**（R6 の判別力に効く）")
    npg = agg_median(fin, "net_progress_per_1000")
    rsp = agg_median(fin, "respawn_per_1000")
    goal = count_goal(fin)

    print("\n--- 錨と、そこから導かれる exp_023 の閾値 ---")
    rows = [
        ("R1  n_reach_ge7（件/120）", ge7, "≥ 24 件（2 倍）", 2 * ge7, 24),
        ("R3  net_progress_per_1000", npg, "< 1.563（1.25 倍）", 1.25 * npg, 1.563),
        ("R4  respawn_per_1000", rsp, "≤ 2.125（同値）", rsp, 2.125),
    ]
    for name, anchor, card_txt, derived, card_val in rows:
        ok = abs(derived - card_val) <= 5e-4
        print(f"  {name:<28} 錨 {anchor:>8}  カード {card_txt:<20} "
              f"導出 {derived:.4f}  {'一致' if ok else '🔴 不一致'}")

    # R5 の判定量は「最終評価のゴール率」= 定期評価の記録の最終点であって、
    # 走行測定の outcome ではない（exp_021 の判定 Q3 がこの経路を使っている）。
    # 出所を取り違えると値が変わるので、両方を出して区別を明示する。
    print(f"\n--- R5 の判定量の出所（2 経路あるので区別する）---")
    print(f"  (i) 走行測定でゴールした走行 = {goal} 件 /120（これは R5 の判定量ではない）")
    gr = []
    for n in range(1, 7):
        p = os.path.join(REPO, f"logs/exp_021_seed{n}/validation_history.json")
        if os.path.exists(p):
            h = json.load(open(p, encoding="utf-8"))
            gr.append((h[-1]["total_timesteps"], h[-1]["goal_rate"]))
    if len(gr) == 6:
        rates = sorted(r for _, r in gr)
        print(f"  (ii) 定期評価の最終点の goal_rate（= R5 の判定量）: {rates}")
        print(f"       6 seed 中央値 = {statistics.median(rates)}（カード記載 0.000）"
              f"／学習量 {sorted({s for s, _ in gr})}")
        print(f"       🔴 seed 単位では最大 {max(rates)} が出ている"
              f"（境界 0.05 に乗りうる — 「厳密に 0.05 なら外れ」）")
    else:
        print(f"  (ii) 🔴 定期評価の記録が 6 本そろわない（{len(gr)} 本）→ 未確認")

    if k800 is not None:
        npg8 = agg_median(k800, "net_progress_per_1000")
        per_seed = seed_medians(k800, "net_progress_per_1000")
        trig = 0.5 * npg8
        print("\n--- 報告トリガー（80 万歩・判定には使わない）---")
        print(f"  対照の 80 万歩の中央値 = {npg8}（カード記載 1.125）")
        print(f"  作動線 = その 0.5 倍 = {trig:.4f}（カード記載 0.5625）")
        print(f"  対照 6 seed のばらつき = {sorted(per_seed)}")
        print(f"  最小 {min(per_seed)} は作動線の "
              f"{'外側 ✅（健全な走行では作動しない）' if min(per_seed) > trig else '🔴 内側'}"
              f"（余裕 {min(per_seed) / trig:.2f} 倍）")
    else:
        print("\n🔴 exp_021 の 80 万歩の測定が見つからない → トリガーの錨は未確認")

    print("\n" + "=" * 78)
    print("この値を AUDIT_048_PREREG §1 に固定する。exp_023 の判定はこの錨で行う。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
