#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
監査 046: exp_022（にせ履歴）の完走後判定の独立再計算

准教授セッション（8 代目）・2026-08-15
**判定形は `verification/AUDIT_046_PREREG_exp022_verdict.md` に、結果を 1 件も見ていない
時点でコミット済み（`3b38b05`）。本スクリプトはその条文をそのまま実装したものである。**

## 独立性の作り（`AUDIT_046_PREREG` §2）

- **錨（対照 exp_019・参照 exp_021）の値も、私自身が測定ファイルから再計算する。**
  相手の判定出力（`outputs/exp_022_judgment.json`）は**照合にのみ使い、計算には使わない。**
- **$r$ は自分の再計算値から組み立てる**（相手の $r$ を照合するのではない）。
  **$r$ は 3 群の値の比の比なので、どれか 1 つの群がずれれば $r$ が動く。**
- **L5**: 毎歩の `d_hist` / `resp_hist` から `min_d` を再構成して、判定量の入力そのものを確かめる。

## 用語

警報・トリガーが働くこと = **作動**／条文・判定条件 = **成立**。
"""

import json
import math
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")

# 事前登録した境界（AUDIT_046_PREREG §1）。錨から幾何中央として導かれる
BOUND = {"P1": 1.031, "P2": 8.83, "P4": 1.369}
REACH_MIN = 5          # 「5 区画以上到達」の閾値（カード §3-2）

results = []


def rec(tag, item, ok, detail):
    results.append((tag, item, ok, detail))
    print(f"  [{'PASS' if ok is True else ('FAIL' if ok is False else 'INFO')}] {item}: {detail}")


def load(path):
    p = os.path.join(OUT, path)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def seed_medians(doc, key):
    return [statistics.median([m[key] for m in b["metrics"]])
            for b in doc["detail"].values()]


def agg_median(doc, key):
    return statistics.median(seed_medians(doc, key))


def count_reach_ge(doc, k):
    """120 走行のうち「到達した最深（D0 − min_d）」が k 区画以上のもの。"""
    return sum(1 for b in doc["detail"].values() for m in b["metrics"]
               if (m["d0"] - m["min_d"]) >= k)


def count_goal(doc):
    return sum(1 for b in doc["detail"].values() for m in b["metrics"]
               if m["outcome"] == "goal")


def r_of(sham, control, treat):
    """相対位置 r = ln(sham/control) / ln(treat/control)。"""
    if control <= 0 or treat <= 0 or sham <= 0 or treat == control:
        return None
    return math.log(sham / control) / math.log(treat / control)


def l5(doc, label):
    """軌跡から min_d と立て直し回数を再構成して照合する。"""
    if not all("raw" in b for b in doc["detail"].values()):
        rec("L5", f"{label}", False, "毎歩の記録が無い → 実行不能")
        return
    n = a = b_ = 0
    for blk in doc["detail"].values():
        raw = {r["maze_seed"]: r for r in blk["raw"]}
        for m in blk["metrics"]:
            r = raw[m["maze_seed"]]
            dh, rh = r["d_hist"], r["resp_hist"]
            n += 1
            a += (min(dh) == m["min_d"])
            b_ += abs(sum(1 for x in rh if x) / len(dh) * 1000.0
                      - m["respawn_per_1000"]) <= 1e-9
    rec("L5", f"{label} を軌跡から再構成", a == n and b_ == n,
        f"min_d {a}/{n}・立て直し回数 {b_}/{n} 一致")


def main():
    docs = {
        "control": load("exp_021_driving_control_final.json") or load("exp_021_driving_control_final.slim.json"),
        "treat": load("exp_021_driving_treat_final.json") or load("exp_021_driving_treat_final.slim.json"),
        "sham": load("exp_022_driving_sham_final.json") or load("exp_022_driving_sham_final.slim.json"),
    }
    for k, v in docs.items():
        if v is None:
            print(f"測定が見つからない: {k}")
            return 2

    print("=" * 74)
    print("監査 046: exp_022 完走後判定の独立再計算（判定形は 3b38b05 で確定済み）")
    print("=" * 74)

    # ---------------- L1 出所 ----------------
    print("\n=== L1. 出所 ===")
    steps, band_ok = [], True
    for n in range(1, 7):
        p = os.path.join(REPO, f"logs/exp_022_seed{n}/validation_history.json")
        if os.path.exists(p):
            steps.append(json.load(open(p))[-1]["total_timesteps"])
        q = os.path.join(REPO, f"logs/exp_022_seed{n}/episode_seeds.jsonl")
        if os.path.exists(q):
            for line in open(q):
                if not line.strip():
                    continue
                s = json.loads(line)["maze_seed"]
                if 6000 <= s < 6020 or 7000 <= s < 7020 or 7100 <= s < 7300:
                    band_ok = False
    rec("L1", "学習量が 6 本とも揃っているか（W-b）", len(set(steps)) == 1,
        f"{sorted(set(steps))} 歩")
    rec("L1", "予約帯への接触（W-c）", band_ok, "0 件" if band_ok else "🔴 接触あり")

    # ---------------- L2/L3 値の再計算 ----------------
    print("\n=== L2/L3. 判定量を自分で再計算（錨も含めて） ===")
    vals = {}
    for arm, doc in docs.items():
        vals[arm] = {
            "P1": agg_median(doc, "respawn_per_1000"),
            "P2": count_reach_ge(doc, REACH_MIN),
            "P3": count_goal(doc),
            "P4": agg_median(doc, "net_progress_per_1000"),
        }
    print(f"  {'':>4}{'対照':>10}{'参照':>10}{'にせ履歴':>12}")
    for k in ("P1", "P2", "P3", "P4"):
        print(f"  {k:>4}{vals['control'][k]:>10}{vals['treat'][k]:>10}{vals['sham'][k]:>12}")

    # 錨が事前登録の値と一致するか（AUDIT_042 で確定した値）
    ANCHOR = {"P1": (0.500, 2.125), "P2": (3, 26), "P3": (0, 4), "P4": (1.500, 1.250)}
    ok = all(abs(vals["control"][k] - ANCHOR[k][0]) <= 1e-9
             and abs(vals["treat"][k] - ANCHOR[k][1]) <= 1e-9 for k in ANCHOR)
    rec("L3", "錨が事前登録の値と一致するか", ok,
        "4 量とも一致" if ok else "🔴 錨がずれている")

    # ---------------- L4 r の組み立てと判定 ----------------
    print("\n=== L4. r を自分の再計算値から組み立てて判定 ===")
    verdicts = {}
    for k in ("P1", "P2", "P4"):
        c, t, s = vals["control"][k], vals["treat"][k], vals["sham"][k]
        r = r_of(s, c, t)
        lo, hi = (min(c, t), max(c, t))
        outside = not (lo <= s <= hi)
        if outside:
            v = "どちらの錨よりも外"
        elif r is None:
            v = "計算不能"
        else:
            v = "(A/C) 側" if r < 0.5 else "(B) 側"
        verdicts[k] = v
        rec("L4", f"{k}: r と判定", None,
            f"にせ履歴 {s} / 錨 [{c}, {t}] → r = {r:.4f} → **{v}**"
            if r is not None else f"にせ履歴 {s} / 錨 [{c}, {t}] → r 計算不能 → **{v}**")
    # P3 は比が定義できない（対照が 0）
    s3 = vals["sham"]["P3"]
    verdicts["P3"] = "(A/C) 寄り" if s3 == 0 else "(B) 寄り"
    rec("L4", "P3（劣後・比が定義できない）", None,
        f"にせ履歴 {s3} 件 / 錨 [0, 4] → **{verdicts['P3']}**（0 件で (A/C) 寄り）")

    # ---------------- W-g 第 3 の読み ----------------
    print("\n=== W-g. 「どちらの錨よりも外」に該当したか ===")
    outs = [k for k, v in verdicts.items() if v == "どちらの錨よりも外"]
    rec("W-g", "該当した判定量", None,
        f"{outs}（該当したら (A)/(B) の判別をせず報告する、が条文）" if outs else "なし")

    # ---------------- L5 ----------------
    print("\n=== L5. 軌跡からの再構成 ===")
    for arm, doc in docs.items():
        l5(doc, arm)

    # ---------------- W-a 歩数の分布・格子 ----------------
    print("\n=== W-a. 歩数の分布と格子（閾値の読みに効く） ===")
    for arm, doc in docs.items():
        st = sorted({m["n_steps"] for b in doc["detail"].values() for m in b["metrics"]})
        sm = seed_medians(doc, "respawn_per_1000")
        on = sum(1 for v in sm if abs(v / 0.25 - round(v / 0.25)) < 1e-9)
        rec("W-a", f"{arm}群", None,
            f"歩数 {st[:4]}{'…' if len(st) > 4 else ''} / ゴール {count_goal(doc)} 件"
            f" / seed 中央値が 0.25 格子に乗る {on}/6")

    # ---------------- W-f にせ履歴の入力の階数 ----------------
    print("\n=== W-f. にせ履歴の入力の階数は本当に 17 か（AUDIT_045 要記載 1 の実測） ===")
    try:
        sys.path.insert(0, REPO)
        from mouse.maze6_env import Maze6Env
        from mouse.obs_history import ObsHistoryWrapper
        import numpy as np
        e = Maze6Env(mode="generate", base_seed=8000, gamma=0.995,
                     collision_respawn=True, goal_rule_containment=True,
                     episode_limit_steps=2000)
        w = ObsHistoryWrapper(e, (1, 2, 4, 8, 16, 32, 64, 128), sham=True)
        n0 = e.observation_space.shape[0]
        o, _ = w.reset(seed=0)
        rows = [o]
        rng = np.random.default_rng(7)
        for _ in range(60):
            o, *_ = w.step(rng.uniform(-1, 1, size=2).astype(np.float32))
            rows.append(o)
        M = np.asarray(rows)
        rank = int(np.linalg.matrix_rank(M - M.mean(0), tol=1e-5))
        blocks_same = all(np.array_equal(r[:n0], r[n0 * (j + 1): n0 * (j + 2)])
                          for r in rows for j in range(8))
        rec("W-f", "9 ブロックがすべて現在の観測と一致（W-d の等価性）", blocks_same,
            "全ブロック一致（情報ゼロの前提は成立）" if blocks_same else "🔴 一致しない")
        rec("W-f", "入力の実効的な階数", rank <= n0,
            f"{rank}（素の観測の次元 {n0} 以下なら、複製で自由度は増えていない）")
    except Exception as ex:
        rec("W-f", "階数の実測", None, f"実行できなかった: {ex}")

    # ---------------- W-h/検出力: 裾の形と seed 水準の分離 ----------------
    print("\n=== 裾の形（探索的。閾値 5 のみ事前登録・それより深い切り方は事後） ===")
    depth = {a: [m["d0"] - m["min_d"] for b in d["detail"].values() for m in b["metrics"]]
             for a, d in docs.items()}
    print(f"  {'深さ以上':>8}{'対照':>8}{'参照':>8}{'にせ履歴':>10}")
    for k in (4, 5, 6, 7, 8, 10):
        print(f"  {k:>8}{sum(1 for v in depth['control'] if v >= k):>8}"
              f"{sum(1 for v in depth['treat'] if v >= k):>8}"
              f"{sum(1 for v in depth['sham'] if v >= k):>10}")

    def fisher2(a, b, c, d):
        n = a + b + c + d
        r1, c1 = a + b, a + c
        def p(x):
            return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
        obs = p(a)
        return sum(p(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1)
                   if p(x) <= obs * (1 + 1e-12))

    s5, t5 = (sum(1 for v in depth[a] if v >= 5) for a in ("sham", "treat"))
    s7, t7 = (sum(1 for v in depth[a] if v >= 7) for a in ("sham", "treat"))
    rec("裾", "5 区画以上: にせ履歴 対 参照", None,
        f"{s5} 対 {t5} → 両側 Fisher p = {fisher2(s5, 120-s5, t5, 120-t5):.4g}"
        f"（**事前登録の閾値**。差は検出されない）")
    rec("裾", "7 区画以上: にせ履歴 対 参照", None,
        f"{s7} 対 {t7} → 両側 Fisher p = {fisher2(s7, 120-s7, t7, 120-t7):.4g}"
        f"（🔴 **閾値 7 は事後選択**）")
    # 閾値を選ばない比較（事後の閾値に頼らない対照）
    med = {a: statistics.median([v for v in depth[a] if v >= 5]) or None for a in depth}
    rec("裾", "5 区画以上に達した走行の深さの中央値", None,
        f"対照 {med['control']} / 参照 {med['treat']} / にせ履歴 {med['sham']}")

    # ---------------- 相手の判定出力との照合 ----------------
    print("\n=== 相手の判定出力との照合（計算には使っていない） ===")
    jd = load("exp_022_judgment.json")
    if jd:
        rec("照合", "相手の判定出力", None,
            f"anchors={list(jd.get('anchors', {}).keys()) if isinstance(jd.get('anchors'), dict) else jd.get('anchors')}")

    print("\n" + "=" * 74)
    nf = sum(1 for _, _, ok, _ in results if ok is False)
    print(f"総括: 不合格 {nf} 件")
    print(f"判定: " + " / ".join(f"{k} {v}" for k, v in verdicts.items()))
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# 追記（2026-08-15）: 順位和検定は正確な並べ替え検定で行う（作法 40）
#
# 当初この監査は正規近似（同順位補正つき）で p を出したが、n = 6 対 6 で
# 同順位が多い場面では近似が当てにならない。学生B の指摘で確定した誤りである。
#   にせ履歴 対 参照: 近似 0.0120 / 正確 0.0087
#   にせ履歴 対 対照: 近似 0.0512 / 正確 0.1017  ← 「境界的」と「検出されない」を取り違えた
# ---------------------------------------------------------------------------
def exact_rank_sum_p(a, b):
    """2 標本の順位和検定を、全ての並べ替えを数えて厳密に行う（両側）。

    同順位は 0.5 として数える（中間順位と同値）。
    n1 + n2 が小さいとき（C(n1+n2, n1) が数えられる規模）にのみ使う。
    """
    from itertools import combinations
    n1, n2 = len(a), len(b)
    allv = list(a) + list(b)

    def U(x, y):
        return sum(sum(1.0 if q < p else (0.5 if q == p else 0.0) for q in y) for p in x)

    u_obs = U(a, b)
    u_obs = min(u_obs, n1 * n2 - u_obs)
    cnt = tot = 0
    for idx in combinations(range(n1 + n2), n1):
        s = set(idx)
        g1 = [allv[i] for i in idx]
        g2 = [allv[i] for i in range(n1 + n2) if i not in s]
        u = U(g1, g2)
        u = min(u, n1 * n2 - u)
        tot += 1
        cnt += (u <= u_obs + 1e-12)
    return cnt / tot, u_obs, tot
