#!/usr/bin/env python3
"""監査 033: exp_020 投入時の 2 件

1. **bit 一致検証の独立再現**（カード §2-3 手順 5 の准教授分）
   — 学生B の `verify_bit_identity.py` は**使わない**。**自分で書いた照合**で、
     **全共通列＋片側のみの列**を突き合わせる（**列の取りこぼしを検出するため**）。
2. **Q2 の検出力の前提の検査**（カード限界 8 の帰結）
   — カードは各 seed が独立に Poisson に従うとしているが、
     **exp_019 の実測は過分散**である。同じ分散比の負の二項で引き直す。

⚠️ **走行の再生成はしていない**（CPU 競合を避けたため）。
**照合したのは学生B が生成した `logs/exp_020_bitcheck/` の出力**である。
**独立なのは照合の側**であって、**新側の走行そのものではない**
（`AUDIT_029` §3 と同じ区分）。

使い方: `.venv/bin/python verification/audit_exp020_launch.py`
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from math import comb, exp, factorial, lgamma
from pathlib import Path

NEW = "logs/exp_020_bitcheck/episode_seeds.jsonl"
REF = "logs/exp_019_v2_seed1/episode_seeds.jsonl"
BITCHECK_REV = "f1bb379"      # 検証を回した版（run_summary.json より）
LAUNCH_REV = "f149a69"        # 投入版（学生B の版通知）
CODE_PATHS = ["mouse/", "common/", "experiments/exp_012_continuous_potential/train.py"]

# exp_019 の seed 別の学習ゴール件数（AUDIT_027 §1・episode_seeds.jsonl から）
EXP019_GOALS = [11, 5, 1, 4, 1, 1]
LAMBDA_Q2 = 8.24              # カード §4-2 の段 1 の期待件数
THRESHOLD_Q2 = 12             # Q2 の閾値（事前登録・動かさない）


def bit_identity() -> dict:
    new = [json.loads(l) for l in Path(NEW).open()]
    ref = [json.loads(l) for l in Path(REF).open()][:len(new)]
    only_new = set(new[0]) - set(ref[0])
    only_ref = set(ref[0]) - set(new[0])

    mismatch, cols = [], set()
    for i, (a, b) in enumerate(zip(new, ref)):
        ks = (set(a) | set(b)) - only_new
        cols |= ks
        for k in sorted(ks):
            if a.get(k) != b.get(k):
                mismatch.append(dict(idx=i, col=k, new=a.get(k), ref=b.get(k)))

    # 空振りでないことの確認（照合列が実際に値を持っているか）
    n_goal = sum(1 for r in ref if r.get("outcome") == "goal")
    n_resp = sum(1 for r in ref if r.get("n_respawn", 0) >= 1)
    n_gcr = sum(1 for r in ref if "goal_contained_rule" in r)

    print("=" * 74)
    print("1. bit 一致検証の独立再現（自前の照合・全共通列）")
    print("=" * 74)
    print(f"  照合本数 = {len(new)}  照合列 = {sorted(cols)}")
    print(f"  新側だけの列 = {sorted(only_new)}")
    print(f"  参照側だけの列 = {sorted(only_ref)}  ← **空であること**（列の取りこぼしが無い）")
    print(f"  🔎 不一致 = {len(mismatch)} 件 → {'✅ 合格' if not mismatch else '🔴 不合格'}")
    print(f"  空振りでないこと: 参照 {len(ref)} 本中 ゴール {n_goal} 件・"
          f"リスポーン経験 {n_resp} 本・ゴール限定列を持つ行 {n_gcr}")
    return dict(n=len(new), mismatch=len(mismatch), only_new=sorted(only_new),
                only_ref=sorted(only_ref), n_goal=n_goal, n_respawn=n_resp)


def version_drift() -> dict:
    """検証を回した版と投入版の間に、実行コードの差があるか。"""
    out = subprocess.run(["git", "diff", "--stat", BITCHECK_REV, LAUNCH_REV, "--",
                          *CODE_PATHS], capture_output=True, text=True).stdout.strip()
    print("\n" + "=" * 74)
    print(f"2. 版ずれ: {BITCHECK_REV}（検証）→ {LAUNCH_REV}（投入）の実行コード差分")
    print("=" * 74)
    print(f"  {out if out else '(空) → **実行コードに差なし。検証は投入版へ転用できる** ✅'}")
    return dict(diff=out, transferable=not out)


def q2_power() -> dict:
    m = statistics.mean(EXP019_GOALS)
    v = statistics.variance(EXP019_GOALS)
    disp = v / m
    lam = LAMBDA_Q2

    p_pois = 1 - sum(exp(-lam) * lam ** k / factorial(k) for k in range(THRESHOLD_Q2))
    # 負の二項（平均 lam・分散 disp*lam）
    r = lam / (disp - 1)
    p = r / (r + lam)

    def nb(k):
        return exp(lgamma(k + r) - lgamma(r) - lgamma(k + 1)) * p ** r * (1 - p) ** k

    p_nb = 1 - sum(nb(k) for k in range(THRESHOLD_Q2))
    four = lambda q: sum(comb(6, k) * q ** k * (1 - q) ** (6 - k) for k in range(4, 7))

    print("\n" + "=" * 74)
    print("3. Q2 の検出力 — 独立 Poisson の仮定 対 実測の過分散")
    print("=" * 74)
    print(f"  exp_019 の seed 別ゴール件数 {EXP019_GOALS}")
    print(f"    平均 {m:.2f} / 分散 {v:.2f} → **分散÷平均 = {disp:.2f}**（Poisson なら 1.0）")
    print(f"  Poisson({lam})       : P(X≥{THRESHOLD_Q2}) = {p_pois:.4f} → "
          f"6 本中 4 本以上 = **{four(p_pois):.5f}**（カードの値）")
    print(f"  負の二項（分散 {disp:.1f} 倍）: P(X≥{THRESHOLD_Q2}) = {p_nb:.4f} → "
          f"6 本中 4 本以上 = **{four(p_nb):.5f}**")
    print(f"  → **偽陽性率は約 {four(p_nb)/four(p_pois):.0f} 倍**")
    print("  ⚠️ 限定: exp_019 の件数は 200 万歩・全 D₀ の量。段 1（40 万歩・D₀=4）の")
    print("     分散比の推定としては近似である。**向きは確か・倍率は目安**")
    return dict(dispersion=disp, p_poisson=four(p_pois), p_nb=four(p_nb))


def main() -> int:
    res = dict(bit_identity=bit_identity(), version=version_drift(), q2=q2_power())
    out = Path(__file__).resolve().parent / "out" / "exp020_launch.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n出力: {out}")
    return 0 if (res["bit_identity"]["mismatch"] == 0
                 and not res["bit_identity"]["only_ref"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
