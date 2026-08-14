#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
監査 042: exp_021 の完走後判定の独立再計算

准教授セッション（8 代目）・2026-08-15
**判定形は `verification/AUDIT_042_PREREG_exp021_verdict.md` に、結果を 1 件も見ていない
時点でコミット済み（`59ebca1`）。本スクリプトはその条文をそのまま実装したものである。**

## 層（`AUDIT_042_PREREG` §2）

  L1  出所      … 学習量の揃い・予約帯への接触・重みの保全
  L2  式        … 各判定量を保存されている入力から自分で計算し直す
  L3  集約      … 迷路 20 本の中央値 → 6 seed 中央値（プール集計をしない）
  L4  判定      … 閾値との比較・不等号の向き・境界の扱い
  L5  軌跡      … 毎歩の D と立て直しの旗から min_d と Q4 の窓を自分で再構成する

**L5 が核心である。**L2〜L4 は「相手の集計が正しいか」で、
**L5 だけが「判定量そのものが正しいか」を見る。**

## 用語（2026-08-14 ユーザ規範）

**警報・トリガー・安全弁が働くこと = 「作動」／条文・判定条件 = 「成立」。**
（事前登録文書は「発火」で書かれているが、字句はそのままにし、本スクリプトの出力から新語で書く。）
"""

import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")

# 事前登録した閾値（カード §5・AUDIT_042_PREREG §1）
Q1_FACTOR, Q2_FACTOR, Q2_GUARD = 1.25, 0.80, 0.90
Q3_LINE, Q4_LINE, Q5_NEED = 0.05, 0.50, 6
TRIGGER = 0.50
ABORT_LINE, ABORT_PTS, ABORT_STEPS = 0.05, 10, 1_000_000

results = []


def rec(tag, item, ok, detail):
    results.append((tag, item, ok, detail))
    mark = "PASS" if ok is True else ("FAIL" if ok is False else "INFO")
    print(f"  [{mark}] {item}: {detail}")


def load(name):
    p = os.path.join(OUT, f"exp_021_driving_{name}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def seed_medians(doc, key):
    """seed ごとに「迷路 20 本の中央値」を出す（カード §4-1。プール集計はしない）。"""
    return [statistics.median([m[key] for m in b["metrics"]])
            for b in doc["detail"].values()]


def l5_reconstruct(doc, label):
    """L5: 毎歩の記録から判定量を自分で再構成して照合する（条文だけから実装した）。"""
    if not all("raw" in b for b in doc["detail"].values()):
        rec("L5", f"{label} の軌跡", False,
            "毎歩の記録が出力に無い → L5 は実行不能（AUDIT_039 §3-1 と同じ限界）")
        return
    n = q1 = q2 = 0
    p5_ok = p5_den = bnd = 0
    for b in doc["detail"].values():
        raw = {r["maze_seed"]: r for r in b["raw"]}
        for m in b["metrics"]:
            r = raw[m["maze_seed"]]
            dh, rh = r["d_hist"], r["resp_hist"]
            n += 1
            q1 += abs((r["d0"] - min(dh)) / len(dh) * 1000.0
                      - m["net_progress_per_1000"]) <= 1e-9
            q2 += abs(sum(1 for x in rh if x) / len(dh) * 1000.0
                      - m["respawn_per_1000"]) <= 1e-9
        p5 = {j["maze_seed"]: j for j in b["p5"]}
        for ms, r in raw.items():
            dh, rh = r["d_hist"], r["resp_hist"]
            idx = [i for i, x in enumerate(rh) if x]
            if not idx:
                continue                      # 母集団 = 立て直しを 1 回以上経験
            p5_den += 1
            last = idx[-1]
            inc = min(dh[last:])
            exc = min(dh[last + 1:]) if last + 1 < len(dh) else inc
            bnd += (inc <= r["d0"] - 1) != (exc <= r["d0"] - 1)
            j = p5.get(ms)
            if j is not None:
                p5_ok += (inc == j["min_d_after_last_respawn"]
                          and (inc <= r["d0"] - 1) == j["advanced"])
    ok = (q1 == n and q2 == n and p5_ok == p5_den)
    rec("L5", f"{label} を軌跡から再構成", ok,
        f"min_d {q1}/{n}・立て直し回数 {q2}/{n}・Q4 の窓 {p5_ok}/{p5_den}"
        f"（窓の境界で判定が変わる {bnd} 件）")


def p5_rates(doc):
    out = []
    for b in doc["detail"].values():
        js = b["p5"]
        out.append(sum(1 for j in js if j["advanced"]) / len(js) if js else None)
    return out


def main():
    print("=" * 72)
    print("監査 042: exp_021 完走後判定の独立再計算（判定形は 59ebca1 で確定済み）")
    print("=" * 72)

    # ---------------- L1 出所 ----------------
    print("\n=== L1. 出所 ===")
    steps, seeds_ok = [], True
    for n in range(1, 7):
        h = json.load(open(os.path.join(REPO, f"logs/exp_021_seed{n}/validation_history.json")))
        steps.append(h[-1]["total_timesteps"])
        ms = [json.loads(l)["maze_seed"]
              for l in open(os.path.join(REPO, f"logs/exp_021_seed{n}/episode_seeds.jsonl"))
              if l.strip()]
        if any(6000 <= s < 6020 or 7000 <= s < 7020 or 7100 <= s < 7300 for s in ms):
            seeds_ok = False
    rec("L1", "学習量が 6 本とも揃っているか", len(set(steps)) == 1,
        f"{sorted(set(steps))} 歩")
    rec("L1", "予約帯への接触（W-c）", seeds_ok, "6 本とも 0 件" if seeds_ok else "🔴 接触あり")

    # ---------------- Q5 / Q3（定期評価から） ----------------
    print("\n=== Q5: 打ち切り条文の成立（100 万歩までの 10 点すべてでゴール率 < 0.05） ===")
    n_hold, detail5, finals = 0, [], []
    for n in range(1, 7):
        h = json.load(open(os.path.join(REPO, f"logs/exp_021_seed{n}/validation_history.json")))
        pts = [r for r in h if r["total_timesteps"] <= ABORT_STEPS][:ABORT_PTS]
        hold = len(pts) == ABORT_PTS and all(r["goal_rate"] < ABORT_LINE for r in pts)
        n_hold += hold
        worst = max(r["goal_rate"] for r in pts)
        detail5.append((n, hold, worst))
        finals.append(h[-1]["goal_rate"])
        print(f"  seed{n}: 10 点の最大ゴール率 {worst:.3f} → 条文 {'成立' if hold else '**不成立**'}")
    rec("Q5", "成立した seed 数", n_hold == Q5_NEED,
        f"{n_hold}/6（予測は 6 で当たり・5 以下で外れ）→ **{'当たり' if n_hold == Q5_NEED else '外れ'}**")
    # 不等号の感度（AUDIT_013 指摘 1 の論点）
    loose = sum(1 for n in range(1, 7)
                if all(r["goal_rate"] <= ABORT_LINE
                       for r in json.load(open(os.path.join(
                           REPO, f"logs/exp_021_seed{n}/validation_history.json")))
                       if r["total_timesteps"] <= ABORT_STEPS))
    rec("Q5", "不等号への感度", None,
        f"条文どおり（< 0.05）で {n_hold}/6 ／ もし ≤ 0.05 と読むと {loose}/6"
        + ("　🔴 **判定が逆転する**" if (n_hold == Q5_NEED) != (loose == Q5_NEED) else ""))

    print("\n=== Q3: 最終評価のゴール率の 6 seed 中央値 < 0.05 ===")
    med3 = statistics.median(finals)
    rec("Q3", "6 seed 中央値", med3 < Q3_LINE,
        f"seed ごと {finals} → 中央値 {med3} → **{'当たり' if med3 < Q3_LINE else '外れ'}**"
        f"（0.05 ちょうどは外れ側）")

    # ---------------- Q1 / Q2 / Q4（rollout から） ----------------
    ctrl_f, treat_f = load("control_final"), load("treat_final")
    ctrl_8, treat_8 = load("control_800k"), load("treat_800k")

    print("\n=== 報告トリガー（80 万歩どうし。判定には使わない） ===")
    if ctrl_8 and treat_8:
        mc = statistics.median(seed_medians(ctrl_8, "net_progress_per_1000"))
        mt = statistics.median(seed_medians(treat_8, "net_progress_per_1000"))
        line = TRIGGER * mc
        rec("TRIG", "作動したか", None,
            f"介入 {mt} 対 作動線 {line}（対照 {mc} の {TRIGGER} 倍）→ "
            f"**{'🔴 作動' if mt <= line else '作動せず'}**（対照比 {mt/mc:.3f} 倍）")
        l5_reconstruct(treat_8, "介入群 80 万歩")
    else:
        rec("TRIG", "80 万歩の測定", None, "未着手")

    print("\n=== Q1 / Q2 / Q4（最終方策の rollout） ===")
    if not (ctrl_f and treat_f):
        rec("Q1/Q2/Q4", "最終方策の測定", None,
            "🔶 まだ出ていない（測定の完了後に再実行すること）")
    else:
        l5_reconstruct(ctrl_f, "対照群 最終")
        l5_reconstruct(treat_f, "介入群 最終")

        cn = statistics.median(seed_medians(ctrl_f, "net_progress_per_1000"))
        tn_s = seed_medians(treat_f, "net_progress_per_1000")
        tn = statistics.median(tn_s)
        cr = statistics.median(seed_medians(ctrl_f, "respawn_per_1000"))
        tr_s = seed_medians(treat_f, "respawn_per_1000")
        tr = statistics.median(tr_s)

        q1 = tn >= Q1_FACTOR * cn
        rec("Q1", "正味の前進", None,
            f"介入 {tn}（seed ごと {sorted(tn_s)}）／合格線 {Q1_FACTOR} × {cn} = {Q1_FACTOR*cn}"
            f" → **{'当たり' if q1 else '外れ'}**（対照比 {tn/cn:.3f} 倍）")
        q2a, q2b = tr <= Q2_FACTOR * cr, tn >= Q2_GUARD * cn
        rec("Q2", "衝突の頻度（複合条件）", None,
            f"立て直し 介入 {tr}（seed ごと {sorted(tr_s)}）／合格線 {Q2_FACTOR} × {cr} = {Q2_FACTOR*cr}"
            f" → {'満たす' if q2a else '満たさない'}（対照比 {tr/cr:.3f} 倍）"
            f"／前進の下限 {Q2_GUARD} × {cn} = {Q2_GUARD*cn} → {'満たす' if q2b else '満たさない'}"
            f" → **{'当たり' if (q2a and q2b) else '外れ'}**")

        cp, tp = p5_rates(ctrl_f), p5_rates(treat_f)
        tp_ok = [r for r in tp if r is not None]
        m4 = statistics.median(tp_ok) if tp_ok else None
        rec("Q4", "立て直しの成立割合", None,
            f"介入 seed ごと {[round(r,3) if r is not None else None for r in tp]}"
            f" → 中央値 {m4}／合格線 {Q4_LINE}"
            f" → **{'当たり' if (m4 is not None and m4 >= Q4_LINE) else '外れ'}**"
            f"（対照の中央値 {statistics.median([r for r in cp if r is not None]):.3f}）")

        # W-a: 歩数の分布（刻みが変わっていないか）
        for lab, doc in (("対照", ctrl_f), ("介入", treat_f)):
            st = sorted({m["n_steps"] for b in doc["detail"].values() for m in b["metrics"]})
            oc = sorted({m["outcome"] for b in doc["detail"].values() for m in b["metrics"]})
            rec("W-a", f"{lab}群の歩数と結末", None,
                f"歩数 {st[:5]}{'…' if len(st) > 5 else ''} / 結末 {oc}")

    print("\n" + "=" * 72)
    nf = sum(1 for _, _, ok, _ in results if ok is False)
    print(f"総括: 不合格 {nf} 件")
    if nf:
        for tag, item, ok, d in results:
            if ok is False:
                print(f"  🔴 [{tag}] {item}: {d}")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
