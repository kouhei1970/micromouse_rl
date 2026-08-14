#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
`AUDIT_047` §2 の p 値の表の出所を回復する（**作法 12 の自己是正**）

准教授セッション（9 代目）・2026-08-15

## なぜこのスクリプトを書くか

**`AUDIT_047` §2 に載せた 6 つの p 値（3 比較 × 走行水準・seed 水準）を出したコードが
`verification/` に無かった。**（8 代目が対話の中で計算し、コミットしていなかった。）
**作法 12（報告書に載せた数値を出したコードは必ず commit する）の、私自身の違反である。**

**しかも重い**: **その表は学生B が実験カード `exp_023` の §3-1-bis に逐語で転記しており、
事前登録された文書の中に「再現できない数値」が入った状態になっていた。**

本スクリプトは、**採った規約を明示したうえで**同じ量を計算し直す。

## 規約（**値が割れたら、まずここを比べる**）

| 軸 | 本スクリプトの選択 |
|---|---|
| 判定量 | **到達した最深** = `d0 − min_d`（走行ごと） |
| 走行水準 | 120 対 120 の Mann-Whitney（**同順位補正つき正規近似**。$C(240,120)$ は数え切れないので正確法は使えない） |
| seed 水準 | 6 対 6 の**並べ替えによる正確法**（$C(12,6)=924$ 通りを全数）。**seed ごとの代表値は「その seed の 20 走行の最深の中央値」** |
| 同順位 | $U$ の計算で **0.5 として数える**（中間順位と同値） |
| 両側 | $\min(U, n_1n_2-U)$ **以下の並べ替えの割合**（正確法）／$|z|$ の両側（近似） |

**走行水準に正規近似を使うのは作法 40 に反しない** — 作法 40 は
**「小さい $n$ で同順位が多いとき」**に近似を使うなという規約である。
**$n=120$ 対 120 は小さくない。**seed 水準（6 対 6）では正確法を使っている。
"""

import json
import math
import os
import statistics
from itertools import combinations

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")


def load(name):
    for cand in (name, name.replace(".json", ".slim.json")):
        p = os.path.join(OUT, cand)
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    return None


def depths_per_seed(doc):
    """seed ごとに、その 20 走行の「到達した最深」の列を返す。"""
    return [[m["d0"] - m["min_d"] for m in b["metrics"]] for b in doc["detail"].values()]


def u_stat(x, y):
    """同順位を 0.5 として数える U 統計量。"""
    return sum(sum(1.0 if q < p else (0.5 if q == p else 0.0) for q in y) for p in x)


def mw_normal(a, b):
    """Mann-Whitney の両側 p（同順位補正つき正規近似）。n が大きいとき用。"""
    n1, n2 = len(a), len(b)
    u = u_stat(a, b)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    # 同順位補正: sum(t^3 − t) を全体の値の重複度から作る
    allv = list(a) + list(b)
    tie = sum(t ** 3 - t for t in
              (allv.count(v) for v in set(allv)))
    var = n1 * n2 / 12.0 * ((n + 1) - tie / float(n * (n - 1)))
    if var <= 0:
        return 1.0, u
    z = (abs(u - mu) - 0.5) / math.sqrt(var)      # 連続性の補正つき
    p = math.erfc(z / math.sqrt(2.0))             # 両側
    return min(1.0, p), u


def mw_exact(a, b):
    """Mann-Whitney の両側 p（並べ替えを全数。n が小さいとき用）。"""
    n1, n2 = len(a), len(b)
    allv = list(a) + list(b)
    u_obs = u_stat(a, b)
    u_obs = min(u_obs, n1 * n2 - u_obs)
    cnt = tot = 0
    for idx in combinations(range(n1 + n2), n1):
        s = set(idx)
        g1 = [allv[i] for i in idx]
        g2 = [allv[i] for i in range(n1 + n2) if i not in s]
        u = u_stat(g1, g2)
        u = min(u, n1 * n2 - u)
        tot += 1
        cnt += (u <= u_obs + 1e-12)
    return cnt / float(tot), tot


def main():
    docs = {
        "exp_019": load("exp_021_driving_control_final.json"),
        "exp_021": load("exp_021_driving_treat_final.json"),
        "exp_022": load("exp_022_driving_sham_final.json"),
    }
    for k, v in docs.items():
        if v is None:
            print(f"🔴 測定が見つからない: {k}")
            return 2

    per_seed = {k: depths_per_seed(v) for k, v in docs.items()}
    runs = {k: [d for seed in v for d in seed] for k, v in per_seed.items()}
    med = {k: [statistics.median(seed) for seed in v] for k, v in per_seed.items()}

    print("=" * 78)
    print("AUDIT_047 §2 の p 値の表の出所の回復（作法 12 の自己是正）")
    print("=" * 78)
    print("\n判定量 = 到達した最深（d0 − min_d）")
    for k in ("exp_019", "exp_021", "exp_022"):
        print(f"  {k}: 走行 {len(runs[k])} 本／seed ごとの中央値 {med[k]}")

    # AUDIT_047 §2 に載せた値（照合の対象）
    reported = {
        ("exp_021", "exp_019"): (0.397, 0.805),
        ("exp_022", "exp_019"): (0.082, 0.636),
        ("exp_022", "exp_021"): (0.534, 0.602),
    }

    print(f"\n{'比較':<22}{'走行水準':>12}{'（報告値）':>12}"
          f"{'seed 水準':>12}{'（報告値）':>12}")
    ok_all = True
    for (a, b), (rep_run, rep_seed) in reported.items():
        p_run, _ = mw_normal(runs[a], runs[b])
        p_seed, tot = mw_exact(med[a], med[b])
        ok = (abs(p_run - rep_run) <= 5e-4) and (abs(p_seed - rep_seed) <= 5e-4)
        ok_all &= ok
        print(f"{a + ' 対 ' + b:<22}{p_run:>12.3f}{rep_run:>12.3f}"
              f"{p_seed:>12.3f}{rep_seed:>12.3f}   {'一致' if ok else '🔴 不一致'}")
    print(f"\n  （seed 水準の正確法は {tot} 通りの全数。p は 1/{tot} = "
          f"{1.0 / tot:.6f} の倍数になる）")

    print("\n" + "=" * 78)
    if ok_all:
        print("🟢 報告値をすべて再現した。出所が版管理下に入った。")
    else:
        print("🔴 再現しない。AUDIT_047 §2 とカード §3-1-bis の表の是正が要る。")
        print("   まず自分を疑う（作法 35）: 規約の軸（seed ごとの代表値・同順位・両側の定義）を")
        print("   1 つずつ変えて、どれで一致するかを特定すること。")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
