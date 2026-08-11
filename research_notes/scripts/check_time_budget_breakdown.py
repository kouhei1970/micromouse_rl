#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""持ち時間 420 秒の内訳を分解する — 何が走行回数を縛っているのか。

L0-a（足立法 + E1）は 1 面の持ち時間を次の 4 区間に使う:

  1. 探索走行 …… スタート → ゴール（`target_mode == "to_goal"`）
  2. **E1 追加探索** …… ゴール到達後、最短経路が確定するまで（`"verify"`）
  3. 帰路 …… 確定後、スタートへ戻る（`"to_start"`）
  4. 最速走行 …… 既知の最短経路を走る

**2 つの落とし穴**（どちらも実際に踏んだので明記する）:

- **落とし穴 A**: 評価器は「スタート区画を出る」ことで走行の開始を判定する。
  E1 の追加探索中は迷路の中にいるので**走行として記録されない**。したがって
  `420 − max(走行の t_end)` を「残り時間」と読むと、**追加探索の時間が丸ごと
  「残り」に化ける**。実際には持ち時間を使い切っている。
  （2026-08-11 に准教授と学生A が独立に同じ誤りに落ちた）
- **落とし穴 B**: 確定に到達しなかった面の追加探索時間を「未計上」にすると、
  同じく「残り時間」に化ける。**未確定 = ゴール到達後の残り全部を追加探索に
  費やした**、が正しい扱い。

本スクリプトは確定時刻の記録（`check_e1_confirmation_timing.py` の出力）と
評価結果 JSON を突き合わせ、両方の落とし穴を避けて内訳を出す。

使い方:
    .venv/bin/python research_notes/scripts/check_e1_confirmation_timing.py \
        --maze-dir competition/mazes/contest_reference     # 先にこれ
    .venv/bin/python research_notes/scripts/check_time_budget_breakdown.py \
        --results competition/results/exp007/contest_reference/adachi_classical_20260811_074936
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIRM_JSON = REPO_ROOT / "research_notes" / "data" / "e1_confirmation_timing.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--confirm-json", default=str(CONFIRM_JSON))
    ap.add_argument("--budget", type=float, default=420.0)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="層別のために除外する面（maze_id）")
    args = ap.parse_args()

    E = json.load(open(args.confirm_json, encoding="utf-8"))
    B = args.budget
    rows = []
    for r in E["rows"]:
        p = os.path.join(args.results, r["maze"] + ".json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p, encoding="utf-8"))
        gr = [x for x in j["runs"] if x["outcome"] == "goal"]
        if not gr:
            continue
        t_goal = gr[0]["t_end"]
        explore = gr[0]["run_time"]
        tc = r["t_confirm"]
        # 落とし穴 B の回避: 未確定なら残り全部が追加探索
        verify = (tc - t_goal) if tc is not None else (B - t_goal)
        ret = r["t_return"]
        rows.append(dict(maze=r["maze"], explore=explore, verify=verify, ret=ret,
                         rest=B - (explore + verify + ret), confirmed=tc is not None,
                         confirm_pct=(explore + verify) / B * 100 if tc is not None else None,
                         fast_done=r["fast_done"], fast_time=j["kpi"]["fast_time"]))

    def report(sel, label):
        s = [x for x in rows if sel(x)]
        if not s:
            return
        med = lambda k: float(np.median([x[k] for x in s]))  # noqa: E731
        nc = sum(1 for x in s if x["confirmed"])
        nf = [x for x in s if not x["fast_done"]]
        pct = [x["confirm_pct"] for x in s if x["confirmed"]]
        print(f"\n【{label}】n={len(s)}")
        print(f"  探索走行 {med('explore'):.1f}s ／ **E1追加探索 {med('verify'):.1f}s** ／ "
              f"帰路 {med('ret'):.1f}s ／ 残り {med('rest'):.1f}s")
        print(f"  確定到達 {nc}/{len(s)}／最速走行 不成立 {len(nf)}/{len(s)}"
              f"（うち確定済み {sum(1 for x in nf if x['confirmed'])} 面）")
        if pct:
            print(f"  確定時刻の持ち時間比: 中央値 {np.median(pct):.0f}%"
                  f"（{min(pct):.0f}〜{max(pct):.0f}%）")
        ft = [x["fast_time"] for x in s if x["fast_time"]]
        if ft:
            print(f"  最速走行に必要な時間（成立面）: 中央値 {np.median(ft):.1f}s"
                  f"  ← 残り時間 {med('rest'):.1f}s と比べる")

    print(f"{'面':<26}{'探索':>9}{'E1追加探索':>12}{'帰路':>8}{'残り':>8}{'確定':>6}{'最速':>6}")
    for x in rows:
        print(f"{x['maze']:<26}{x['explore']:>8.1f}s{x['verify']:>11.1f}s{x['ret']:>7.1f}s"
              f"{x['rest']:>7.1f}s{('○' if x['confirmed'] else '×'):>6}"
              f"{('○' if x['fast_done'] else '×'):>6}")
    report(lambda x: True, "全面")
    if args.exclude:
        report(lambda x: x["maze"] not in args.exclude, f"除外 {len(args.exclude)} 面を引いた残り")
        report(lambda x: x["maze"] in args.exclude, "除外した面のみ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
