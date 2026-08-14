#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
exp_023 群 1 の費用の実測（カード §5 の「投入 30 分後に実測で更新する」の履行検査）

准教授セッション（9 代目）・2026-08-15

## なぜ測るか

**カード §5 は見込みを「単一プロセスでの比（2.8 倍）を exp_021 の 6 本並列の実測時間に掛けた」
値として出し、自ら但し書きを付けている**:

> **並列時の比が単一プロセスと同じである保証は無い**（**投入 30 分後に実測で更新する**）。

**この但し書きが当たっているかを、走行中に確かめる。**
**外れていれば見込みが動き、群 2 の起動時刻・私の L6 の着手時刻・学生A の CPU 調整に効く。**

## 測り方と、その限界

**`logs/exp_023a_seed*/validation_history.json` の mtime を「最終評価点が書かれた時刻」とみなし、
投入時刻（`verification/evidence/exp_023a_launch_argv.txt` に保全した `lstart`）との差で割る。**

**🔴 限界（作法 20 / `AUDIT_026` の実測）**: **mtime は上書きされる。**
**この値は「いま最後に書かれた評価点」のものであり、後から遡って別の評価点の時刻は取れない。**
**費用の見積もりには十分だが、判定に使う量ではない**（本スクリプトの出力は判定に使わない）。

**評価の所要時間は分離していない** — 得られるのは
**「学習 ＋ 定期評価を込みにした実効の速さ」**であり、**完走時刻の見積もりにはこちらが正しい。**
"""

import datetime as dt
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 投入時刻（verification/evidence/exp_023a_launch_argv.txt に保全した ps の lstart）
LAUNCH = dt.datetime(2026, 8, 15, 7, 53, 36)
TOTAL_STEPS = 2_000_896          # カード §3-1・exp_021/022 と同じ学習量

# カード §5 の見込み（照合の対象）
CARD_HOURS_PER_ARM = 3.2
CARD_SINGLE_PROC_RATIO = 2.8     # RecurrentPPO 624 歩/秒 対 PPO 1,754 歩/秒
CARD_EXP021_MINUTES = 68         # exp_021 の 6 本並列の実測（カード記載。私は未確認）


def progress_series(n):
    """`progress.csv`（**追記型**）から (経過秒, 歩数, エピソード長) の時系列を返す。

    🔴 **これが正しい計器である**（2026-08-15 追記）。
    **`mtime` は上書きされるので最後の 1 点しか取れず、累積の速さしか出せない。**
    **累積は「冷えていた最初の速い区間」を含むので、残り時間の見積もりでは楽観側に出る。**
    **`progress.csv` は書かれたら変わらない記録なので、区間ごとの速さが取れる。**
    """
    import csv
    p = os.path.join(REPO, f"logs/exp_023a_seed{n}/progress.csv")
    if not os.path.exists(p):
        return []
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r.get("time/total_timesteps"):
            out.append((float(r["time/time_elapsed"]),
                        int(float(r["time/total_timesteps"])),
                        float(r.get("rollout/ep_len_mean") or 0)))
    return out


def by_progress(window_s=300.0):
    """追記型の記録から、累積の速さと直近 `window_s` 秒の速さを出す。"""
    rows = []
    for n in range(1, 7):
        pts = progress_series(n)
        if len(pts) < 3:
            continue
        t, s, _ = pts[-1]
        w = [p for p in pts if p[0] >= t - window_s]
        if len(w) < 2 or w[-1][0] <= w[0][0]:
            continue
        recent = (w[-1][1] - w[0][1]) / (w[-1][0] - w[0][0])
        rows.append((n, s, t, s / t, recent, (TOTAL_STEPS - s) / recent / 3600.0))
    return rows


def main():
    rows = []
    for n in range(1, 7):
        p = os.path.join(REPO, f"logs/exp_023a_seed{n}/validation_history.json")
        if not os.path.exists(p):
            continue
        m = dt.datetime.fromtimestamp(os.path.getmtime(p))
        steps = json.load(open(p, encoding="utf-8"))[-1]["total_timesteps"]
        el = (m - LAUNCH).total_seconds()
        if el <= 0:
            continue
        rows.append((n, steps, el, steps / el))
    if not rows:
        print("🔴 評価点がまだ 1 つも書かれていない")
        return 2

    print("=" * 78)
    print("exp_023 群 1 の費用の実測（カード §5 の但し書きの履行検査）")
    print("=" * 78)
    print(f"投入 {LAUNCH:%Y-%m-%d %H:%M:%S} JST（保全した argv の lstart）")
    print(f"観測 {dt.datetime.now():%Y-%m-%d %H:%M:%S} JST\n")
    print(f"{'seed':>5}{'最終評価点':>12}{'経過[s]':>10}{'歩/秒':>9}"
          f"{'200 万歩の見込み[時間]':>24}")
    for n, s, el, r in rows:
        print(f"{n:>5}{s:>12,}{el:>10.0f}{r:>9.1f}{TOTAL_STEPS / r / 3600:>24.2f}")

    avg = sum(r for *_, r in rows) / len(rows)
    h1 = TOTAL_STEPS / avg / 3600.0
    print(f"\n平均 {avg:.1f} 歩/秒 → **1 群 {h1:.2f} 時間・2 群 {2 * h1:.2f} 時間**")
    print(f"カードの見込み: 1 群 {CARD_HOURS_PER_ARM} 時間・2 群 {2 * CARD_HOURS_PER_ARM} 時間"
          f"（含意する速さ {TOTAL_STEPS / (CARD_HOURS_PER_ARM * 3600):.1f} 歩/秒）")

    # 並列時の比を逆算する（カードは単一プロセスの 2.8 倍を使った）
    exp021_rate = TOTAL_STEPS / (CARD_EXP021_MINUTES * 60.0)
    print(f"\n--- 但し書きの検証: 並列時の比は単一プロセスと同じか ---")
    print(f"  exp_021（前向き・6 本並列）: {exp021_rate:.1f} 歩/秒"
          f"（カード記載の {CARD_EXP021_MINUTES} 分から逆算。**私は未確認**）")
    print(f"  exp_023 群 1（再帰・6 本並列）: {avg:.1f} 歩/秒（本実測）")
    print(f"  **並列時の比 = {exp021_rate / avg:.2f} 倍**"
          f"（カードが使った単一プロセスの比は {CARD_SINGLE_PROC_RATIO} 倍）")
    if abs(exp021_rate / avg - CARD_SINGLE_PROC_RATIO) > 0.3:
        print(f"  🔴 **但し書きが当たった** — 並列時の比は単一プロセスの比と一致しない。")
        print(f"     見込みは {'過大' if exp021_rate / avg < CARD_SINGLE_PROC_RATIO else '過小'}"
              f"だった（実測のほうが"
              f"{'速い' if h1 < CARD_HOURS_PER_ARM else '遅い'}）。")
    else:
        print(f"  🟢 但し書きの心配は外れた（並列でも比はほぼ同じ）。")

    # ------------------------------------------------------------------
    # 🔴 2026-08-15 追記: 上の mtime による累積は**楽観側に出る**。
    #    追記型の progress.csv で区間の速さを取り直す（これが正しい計器）。
    # ------------------------------------------------------------------
    pr = by_progress()
    if not pr:
        print("\n🔴 progress.csv がまだ読めない")
        return 1

    import statistics
    print("\n" + "=" * 78)
    print("追記型の記録（progress.csv）による測り直し ← **こちらを正とする**")
    print("=" * 78)
    print(f"{'seed':>5}{'現在歩数':>11}{'経過[s]':>9}{'累積歩/秒':>11}"
          f"{'直近5分歩/秒':>14}{'残り[時間]':>12}")
    for n, s, t, c, r5, left in pr:
        print(f"{n:>5}{s:>11,}{t:>9.0f}{c:>11.1f}{r5:>14.1f}{left:>12.2f}")

    cum = statistics.mean(r[3] for r in pr)
    rec = statistics.mean(r[4] for r in pr)
    left_med = statistics.median(r[5] for r in pr)
    left_max = max(r[5] for r in pr)
    now = dt.datetime.now()

    print(f"\n  累積 平均 **{cum:.1f} 歩/秒** ← mtime での測り方と同じ作り（**楽観側**）")
    print(f"  直近 5 分 平均 **{rec:.1f} 歩/秒** ← **残り時間の見積もりはこちらで行う**")
    print(f"  速さは単調に低下して頭打ちになる形"
          f"（冷えた最初が速い。ファンの無い機体の熱制限が第一候補 — **推測**）。")
    print(f"  100 万歩ごとではなく **10 万歩ごとの定期評価で区間の速さが落ち込む**"
          f"（評価の所要時間が乗るため）。")

    print(f"\n--- 完了の見込み（**直近の速さで外挿**）---")
    print(f"  群 1 完走: **{now + dt.timedelta(hours=left_med):%H:%M} 〜 "
          f"{now + dt.timedelta(hours=left_max):%H:%M}**")
    g2 = TOTAL_STEPS / rec / 3600.0
    print(f"  群 2 単独: **{g2:.2f} 時間**"
          f"（**群 1 の実測 {TOTAL_STEPS / cum / 3600.0:.2f} 時間より長い** — "
          f"**群 2 は最初から熱い機体で始まるので、冷えた速い区間を持たない**）")
    print(f"  全完走: **{now + dt.timedelta(hours=left_max + g2):%H:%M} 頃**")
    print("=" * 78)
    print("⚠️ mtime は上書きされる値であり（作法 20・AUDIT_026）、費用の見積もり専用。")
    print("   本スクリプトの出力は判定に使わない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
