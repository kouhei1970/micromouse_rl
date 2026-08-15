#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
監査 048: exp_023（再帰型方策）の完走後判定の独立再計算

准教授セッション（9 代目）・2026-08-15

**判定形は `verification/AUDIT_048_PREREG_exp023_verdict.md` に、
exp_023 の結果を 1 件も見ていない時点でコミット済み（`8ac0994`）。
本スクリプトはその条文をそのまま実装したものであり、条文は変えない。**

## 独立性の作り（`AUDIT_048_PREREG` §2・作法「独立なのはどの層か」）

- **学生B の判定出力（`outputs/exp_023_judgment.json`）は照合にのみ使い、計算には使わない。**
- **錨（対照 exp_021）も自分で再計算する**（`AUDIT_048_PREREG` §1 で確定済みの値と照合）。
- **L5**: 毎歩の `d_hist` / `resp_hist` から `min_d` と立て直し回数を再構成する。
- **L6**: 評価時の隠れ状態の持ち越しは `audit_exp023_l6_hidden_state.py` で別途実施済み
  （80 万歩・迷路 5 本・合格。`verification/evidence/exp_023a_l6_800k.json`）。

使い方:
    .venv/bin/python verification/audit_exp023_verdict.py
"""

import hashlib
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_exp023_pvalues import mw_exact, mw_normal  # 私自身の p 値の実装

OUT = os.path.join(REPO, "outputs")

# ---- 事前登録した条文（AUDIT_048_PREREG §1・§3。変えない）----
ANCHOR = dict(n_reach_ge7=12, net_progress=1.250, respawn=2.125, goal_rate=0.000)
R1_TH, R3_TH, R3_DERIVED, R4_TH, R5_TH = 24, 1.563, 1.5625, 2.125, 0.05
DEPTH = 7
DESCRIPTIVE_DEPTHS = (5, 6, 7, 8, 10)
TOTAL_STEPS = 2_000_896
RESERVED = [(6000, 6020), (7000, 7020), (7100, 7300)]
# 代案 B（教授裁定）: 良い向きは判定量ごとに事前登録し、予測の向きと混同しない
IMPROVEMENT_IS_HIGH = {"n_reach_ge7": True, "net_progress": True,
                       "respawn": False, "goal_rate": True}

rows = []


def rec(layer, item, ok, detail):
    rows.append((layer, item, ok, detail))
    mark = "  " if ok is None else ("✅" if ok else "🔴")
    print(f"  {mark} [{layer}] {item}: {detail}")


def load(name):
    for c in (name, name.replace(".json", ".slim.json")):
        p = os.path.join(OUT, c)
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8")), c
    return None, None


def depths(doc):
    return {k: [m["d0"] - m["min_d"] for m in v["metrics"]] for k, v in doc["detail"].items()}


def count_ge(doc, k):
    d = depths(doc)
    per = {s: sum(1 for x in v if x >= k) for s, v in d.items()}
    return sum(per.values()), per


def seed_meds(doc, f):
    return [statistics.median([m[f] for m in v["metrics"]]) for v in doc["detail"].values()]


def med(doc, f):
    return statistics.median(seed_meds(doc, f))


def final_goal_rates(prefix):
    out = {}
    for n in range(1, 7):
        p = os.path.join(REPO, f"logs/{prefix}_seed{n}/validation_history.json")
        if not os.path.exists(p):
            return None
        h = json.load(open(p, encoding="utf-8"))
        out[f"seed{n}"] = (h[-1]["total_timesteps"], h[-1]["goal_rate"])
    return out


# ---------------------------------------------------------------- L1 出所
def layer1(prefix, tag):
    steps, band_ok, n_ep = set(), True, 0
    for n in range(1, 7):
        p = os.path.join(REPO, f"logs/{prefix}_seed{n}/validation_history.json")
        if os.path.exists(p):
            steps.add(json.load(open(p, encoding="utf-8"))[-1]["total_timesteps"])
        q = os.path.join(REPO, f"logs/{prefix}_seed{n}/episode_seeds.jsonl")
        if os.path.exists(q):
            for line in open(q, encoding="utf-8"):
                if not line.strip():
                    continue
                s = json.loads(line)["maze_seed"]
                n_ep += 1
                if any(lo <= s < hi for lo, hi in RESERVED):
                    band_ok = False
    rec("L1", f"{tag} の学習量（W-a）", len(steps) == 1 and TOTAL_STEPS in steps,
        f"{sorted(steps)} 歩")
    rec("L1", f"{tag} の予約 seed への接触（W-b）", band_ok,
        f"0 件 / {n_ep:,} エピソード" if band_ok else "🔴 接触あり")
    # 重みの保全（W-c）
    n_w = sum(1 for n in range(1, 7)
              if os.path.exists(os.path.join(REPO, f"logs/{prefix}_seed{n}/rl_model_2000000_steps.zip")))
    rec("L1", f"{tag} の最終重みの保全（W-c）", n_w == 6, f"{n_w}/6 本")
    # 群の識別（PREREG §7）— argv の保全と train.log の字句
    ev = os.path.join(REPO, f"verification/evidence/{prefix.replace('exp_023','exp_023')}_launch_argv.txt")
    has_reset = None
    lp = os.path.join(REPO, f"logs/{prefix}_seed1/train.log")
    if os.path.exists(lp):
        for line in open(lp, encoding="utf-8", errors="replace"):
            if "リスポーンで隠れ状態をリセット" in line:
                has_reset = "有効" in line
                break
    rec("L1", f"{tag} の群の識別（PREREG §7）", os.path.exists(ev),
        f"argv 保全 {'有' if os.path.exists(ev) else '無'}／train.log のリセット = {has_reset}")
    return has_reset


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- L5 軌跡
def layer5(doc, tag):
    if not all("raw" in b for b in doc["detail"].values()):
        rec("L5", f"{tag} を軌跡から再構成", None, "毎歩の記録が無い → 実行不能（未確認）")
        return
    n = a = b_ = 0
    for blk in doc["detail"].values():
        raw = {r["maze_seed"]: r for r in blk["raw"]}
        for m in blk["metrics"]:
            r = raw[m["maze_seed"]]
            n += 1
            a += (min(r["d_hist"]) == m["min_d"])
            b_ += abs(sum(1 for x in r["resp_hist"] if x) / len(r["d_hist"]) * 1000.0
                      - m["respawn_per_1000"]) <= 1e-9
    rec("L5", f"{tag} を軌跡から再構成", a == n and b_ == n,
        f"min_d {a}/{n}・立て直し {b_}/{n} 一致")


def main():
    print("=" * 78)
    print("監査 048: exp_023 完走後判定の独立再計算（条文は 8ac0994 で確定済み）")
    print("=" * 78)

    # L1（出所）は測定出力に依存しないので先に回す — 測定を待つ間に確かめられる
    print("\n=== L1. 出所 ===")
    r1_reset = layer1("exp_023a", "群 1")
    r2_reset = layer1("exp_023b", "群 2")
    rec("L1", "群の取り違え（W-e）", r1_reset is False and r2_reset is True,
        f"群 1 のリセット = {r1_reset}／群 2 = {r2_reset}")

    control, cs = load("exp_021_driving_treat_final.json")
    g1, g1s = load("exp_023a_driving_final.json")
    g2, g2s = load("exp_023b_driving_final.json")
    if control is None or g1 is None or g2 is None:
        print("\n⏳ 測定出力が揃っていない（L1 まで実施）:",
              {"対照": cs, "群 1": g1s, "群 2": g2s})
        return 2

    print("\n=== L4. 錨を自分で再計算して事前登録と照合 ===")
    a_ge7, a_per = count_ge(control, DEPTH)
    rec("L4", "錨 n_reach_ge7", a_ge7 == ANCHOR["n_reach_ge7"],
        f"{a_ge7}（登録 {ANCHOR['n_reach_ge7']}）／内訳 {sorted(a_per.values())}")
    for f, key in (("net_progress_per_1000", "net_progress"), ("respawn_per_1000", "respawn")):
        v = med(control, f)
        rec("L4", f"錨 {f}", abs(v - ANCHOR[key]) <= 1e-9, f"{v}（登録 {ANCHOR[key]}）")
    gc = final_goal_rates("exp_021")
    if gc:
        v = statistics.median([r for _, r in gc.values()])
        rec("L4", "錨 goal_rate", abs(v - ANCHOR["goal_rate"]) <= 1e-9,
            f"{v}（登録 {ANCHOR['goal_rate']}）")

    print("\n=== L2/L3. 判定量を自分で再計算 ===")
    v = {}
    v["R1"], p1 = count_ge(g1, DEPTH)
    v["R3"] = med(g1, "net_progress_per_1000")
    v["R4"] = med(g1, "respawn_per_1000")
    g1g = final_goal_rates("exp_023a")
    g2g = final_goal_rates("exp_023b")
    ts1 = {t for t, _ in g1g.values()} if g1g else set()
    rec("L2", "R5 の測定時点の揃い", len(ts1) == 1 and TOTAL_STEPS in ts1, f"{sorted(ts1)} 歩")
    v["R5"] = statistics.median([r for _, r in g1g.values()]) if g1g else None
    v["R6"], p6 = count_ge(g2, DEPTH)
    v["R7"] = med(g2, "respawn_per_1000")

    print("\n=== 判定（事前登録した同値の扱い・AUDIT_048_PREREG §3-2）===")
    hits = {}
    hits["R1"] = v["R1"] >= R1_TH                      # 同値は当たり
    hits["R3"] = v["R3"] < R3_TH                       # 同値は外れ
    hits["R4"] = v["R4"] <= R4_TH                      # 同値は当たり
    hits["R5"] = v["R5"] < R5_TH                       # 厳密に 0.05 は外れ
    hits["R6"] = v["R6"] > v["R1"]                     # 同数は外れ
    hits["R7"] = v["R7"] <= v["R4"]                    # 同値は当たり
    clause = {"R1": f"≥ {R1_TH} 件（同値は当たり）", "R3": f"< {R3_TH}（同値は外れ）",
              "R4": f"≤ {R4_TH}（同値は当たり）", "R5": f"< {R5_TH}（厳密に 0.05 は外れ）",
              "R6": f"群 1 の {v['R1']} 件より多い（同数は外れ）",
              "R7": f"群 1 の {v['R4']} 以下（同値は当たり）"}
    for k in ("R1", "R3", "R4", "R5", "R6", "R7"):
        print(f"  {'✅ 当たり' if hits[k] else '❌ 外れ'}  {k} = {v[k]}   条文: {clause[k]}")
    print(f"  群 1 の seed 内訳（R1）: {sorted(p1.values())}／群 2（R6）: {sorted(p6.values())}")
    print(f"  R5 の seed ごと: {sorted(r for _, r in g1g.values())}")

    if R3_DERIVED <= v["R3"] < R3_TH:
        rec("判定", "R3 の曖昧な帯", None,
            f"🔴 {v['R3']} は [{R3_DERIVED}, {R3_TH}) に入った。字句では当たり・導出値では外れ")

    print("\n=== L5. 軌跡から判定量の入力を再構成 ===")
    layer5(g1, "群 1")
    layer5(g2, "群 2")

    print("\n=== 記述（旧 R2・判定には使わない）===")
    dd = {"対照": depths(control), "群 1": depths(g1), "群 2": depths(g2)}
    print(f"  {'深さ以上':>8}{'対照':>8}{'群 1':>8}{'群 2':>8}")
    for k in DESCRIPTIVE_DEPTHS:
        print(f"  {k:>8}" + "".join(f"{sum(1 for v_ in x.values() for y in v_ if y >= k):>8}"
                                    for x in dd.values()))
    names = list(dd)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sa = [statistics.median(x) for x in dd[a].values()]
            sb = [statistics.median(x) for x in dd[b].values()]
            ra = [y for x in dd[a].values() for y in x]
            rb = [y for x in dd[b].values() for y in x]
            print(f"  {a} 対 {b}: seed 水準 p={mw_exact(sa, sb)[0]:.3f}／"
                  f"走行水準 p={mw_normal(ra, rb)[0]:.3f}")

    print("\n=== 第 3 の読み（代案 B・教授裁定）===")
    q = {"n_reach_ge7": (a_ge7, v["R1"], v["R6"]),
         "net_progress": (ANCHOR["net_progress"], v["R3"], med(g2, "net_progress_per_1000")),
         "respawn": (ANCHOR["respawn"], v["R4"], v["R7"]),
         "goal_rate": (ANCHOR["goal_rate"], v["R5"],
                       statistics.median([r for _, r in g2g.values()]) if g2g else None)}
    flags = {}
    for name, (c, a, b) in q.items():
        if b is None:
            continue
        high = IMPROVEMENT_IS_HIGH[name]
        worse = [t for t, x in (("群 1", a), ("群 2", b)) if (x < c if high else x > c)]
        flags[name] = worse
        print(f"  {name}: 対照 {c} / 群 1 {a} / 群 2 {b} → "
              f"{'🔴 旗（' + '・'.join(worse) + ' が対照より悪い）' if worse else '旗なし'}")
    suppress = bool(flags.get("n_reach_ge7"))
    print(f"\n=== §3-4 全域被覆 ===")
    if suppress:
        print("  🔴 表の構成量 n_reach_ge7 で旗が立った → カード :277 により表は引かない")
    else:
        table = {(True, True): "再帰構造が効き、汚染を除くと更に効く",
                 (True, False): "再帰構造は効くが、汚染の除去は効かない",
                 (False, True): "汚染されたままでは効かないが、除けば効く",
                 (False, False): "再帰構造でも届かない"}
        print(f"  R1 {'当' if hits['R1'] else '外'} × R6 {'当' if hits['R6'] else '外'} → "
              f"{table[(hits['R1'], hits['R6'])]}")

    print("\n" + "=" * 78)
    bad = [r for r in rows if r[2] is False]
    print(f"L1〜L5 の検査: {len(rows)} 件中 不合格 {len(bad)} 件"
          f"{'（' + '／'.join(r[1] for r in bad) + '）' if bad else ''}")
    print(f"判定: " + "・".join(f"{k}{'当' if hits[k] else '外'}" for k in
                                ("R1", "R3", "R4", "R5", "R6", "R7")))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
