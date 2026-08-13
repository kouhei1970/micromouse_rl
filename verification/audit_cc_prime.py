"""条件 C・C' の独立再計算（exp_012 便 3）。

⚠️ **本スクリプトは結果が出る前に書いた**（条件 C は投入直後・評価点 2〜3 点の時点）。
`design.md` §5-3 の解釈表と §6 の打ち切り条文を、**実行可能な形にしただけ**である。
**結果を見てから判定形を作っていない**ことを、コミットの時系列で担保する。

やること（すべて生ログから。報告値は照合にしか使わない）:
  1. **打ち切り条文の逐語比較**（N-11。判定基準が動いていないか）
  2. **打ち切りの発火**を `validation_history.json` から独立に再現
  3. **R7 基準**を条件ごとに適用（中央値 ≥ 0.30 で機能／3 seed すべて < 0.10 で不成立／
     打ち切りは不成立側に算入）。**H の宣言に使ってよいのは条件 E だけ**（裁定 R24-2）
  4. **2×2 解釈表 第 2 段**（C 対 C'・C' 対 E）。「大きく」は**中央値の差 0.15 以上**
  5. **第 3 段の機構の横断確認**のうち、評価点から取れる量（衝突率／時間切れ率の比）
  6. **1/ρ_field の記録**（裁定 R24-1）が実在するかと、その分布

**使い方**: `.venv/bin/python verification/audit_cc_prime.py`
（存在する条件だけを自動で拾う。未完走の条件は「未完走」と表示して判定しない）
"""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/Users/kouhei/tmp/github/micromouse_rl")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# 条件 → ログディレクトリの雛形（E は完走済み・C/C' はこれから）
CONDITIONS = {"E": "logs/exp_012_condE_seed{}", "C": "logs/exp_012_condC_seed{}",
              "Cp": "logs/exp_012_condCp_seed{}"}
SEEDS = (1, 2, 3)
ABORT_LIMIT = 1_000_000        # 打ち切りを見る歩数の上限（§6 の条文）
ABORT_THRESH = 0.05            # 「ゴール率 < 0.05」（**厳密不等号**。AUDIT_013 指摘 1）
ABORT_POINTS = 10              # 100 万歩までの評価点の数
SUPPORT = 0.30                 # R7: 中央値 ≥ 0.30 で機能
REJECT = 0.10                  # R7: 3 seed すべて < 0.10 で不成立
BIG = 0.15                     # 解釈表の「大きく」（中央値の差）
CLAUSE_KEY = "打ち切りの基準"
CLAUSE_REGISTERED = "84facf4"  # 条文が登録されたコミット（AUDIT_013 §1）
DESIGN = "experiments/exp_012_continuous_potential/design.md"


def clause_line(rev):
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{rev}:{DESIGN}"],
                         capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if CLAUSE_KEY in l]


def load(cond):
    """条件ごとに seed 別の評価履歴を読む。無ければ None。"""
    out = {}
    for s in SEEDS:
        p = REPO_ROOT / CONDITIONS[cond].format(s) / "validation_history.json"
        out[s] = json.load(open(p)) if p.exists() else None
    return out


def abort_fired(rows):
    """§6 の条文をそのまま適用する（**厳密不等号**。0.05 ちょうどは発火を阻む）。"""
    pts = [(r["total_timesteps"], r["goal_rate"]) for r in rows
           if r["total_timesteps"] <= ABORT_LIMIT]
    return (len(pts) >= ABORT_POINTS and all(g < ABORT_THRESH for _, g in pts)), pts


def verdict(medians, aborted, cond):
    """R7 基準。**H の宣言に使ってよいのは条件 E だけ**（裁定 R24-2）。"""
    finals = medians
    med = float(np.median(finals))
    if all(v < REJECT for v in finals):
        v = "不成立（3 seed すべて < 0.10）"
    elif med >= SUPPORT:
        v = "機能（中央値 ≥ 0.30）"
    else:
        v = "中間（機能したが不十分）"
    note = ("**H の支持／棄却を宣言してよい**（検証的条件）" if cond == "E"
            else "**探索的条件。H の宣言には使わない**（裁定 R24-2）")
    return med, v, note


def main():
    print("=" * 78)
    print("1. 打ち切り条文の逐語比較（N-11: 判定基準を動かしていないか）")
    print("=" * 78)
    a, b = clause_line(CLAUSE_REGISTERED), clause_line("HEAD")
    print(f"  登録版 {CLAUSE_REGISTERED} 対 HEAD: "
          f"{'一致（動いていない）' if a == b and a else '🔴 不一致'}")

    results = {}
    for cond in CONDITIONS:
        rows_by_seed = load(cond)
        if all(v is None for v in rows_by_seed.values()):
            continue
        print()
        print("=" * 78)
        print(f"2. 条件 {cond}")
        print("=" * 78)
        finals, aborted, done = [], [], True
        for s in SEEDS:
            rows = rows_by_seed[s]
            if rows is None:
                print(f"  seed{s}: ログなし")
                done = False
                continue
            fired, pts = abort_fired(rows)
            last = rows[-1]
            nz = [(r["total_timesteps"], r["goal_rate"],
                   [x for x in range(7000, 7020) if x not in r["failed_maze_seeds"]])
                  for r in rows if r["goal_rate"] > 0]
            if last["total_timesteps"] < 2_000_000:
                done = False
            print(f"  seed{s}: 評価点 {len(rows)}（100 万歩まで {len(pts)}）"
                  f"  最新 {last['total_timesteps']} 歩"
                  f"  最終ゴール率 {last['goal_rate']}"
                  f"  打ち切り {'🔴 発火' if fired else '発火せず'}")
            if nz:
                print(f"          非ゼロ点 {len(nz)}: "
                      + " / ".join(f"{t//1000}k→{g}（面 {m}）" for t, g, m in nz))
            print(f"          衝突率 {last['collision_rate']}  "
                  f"時間切れ率 {last['timeout_rate']}"
                  + (f"  → 衝突/時間切れ = {last['collision_rate']/last['timeout_rate']:.2f}"
                     if last["timeout_rate"] else "  → 時間切れ 0"))
            finals.append(last["goal_rate"])
            aborted.append(fired)
        if not done:
            print("  → **未完走。判定しない**（上記は途中経過であり、"
                  "判定は 3 seed が 200 万歩に達してから行う）")
            continue
        med, v, note = verdict(finals, aborted, cond)
        results[cond] = med
        print(f"  → 3 seed の最終ゴール率 {finals}  中央値 **{med:.4f}**")
        print(f"  → R7 基準: **{v}**   {note}")
        if any(aborted):
            print(f"  → 打ち切りが発火した seed は**不成立側に算入**（裁定 R7-2）")

    if len(results) >= 2:
        print()
        print("=" * 78)
        print("3. 解釈表 第 2 段（C 対 C'・C' 対 E）。「大きく」= 中央値の差 0.15 以上")
        print("=" * 78)
        for x, y in (("C", "Cp"), ("Cp", "E")):
            if x in results and y in results:
                d = results[y] - results[x]
                print(f"  {y} − {x} = {d:+.4f} → "
                      + ("**大きく差がある**" if abs(d) >= BIG else "差は見られない"))
        print("  ⚠️ 差が 0.15 未満なら**順位だけを根拠に機構を語らない**（design.md §5-3）")

    print()
    print("=" * 78)
    print("4. 1/ρ_field の記録（裁定 R24-1）")
    print("=" * 78)
    for cond in ("C", "Cp"):
        vals = []
        for s in SEEDS:
            p = REPO_ROOT / CONDITIONS[cond].format(s) / "episode_seeds.jsonl"
            if not p.exists():
                continue
            for line in open(p):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "geo_inv_rho_field" in r:
                    vals.append(r["geo_inv_rho_field"])
        if vals:
            a = np.array(vals)
            print(f"  条件 {cond}: n={len(a)}  中央値 {np.median(a):.4f}  "
                  f"[{a.min():.4f}, {a.max():.4f}]")
        else:
            print(f"  条件 {cond}: 記録なし（未投入か、記録が入っていない）")


if __name__ == "__main__":
    main()
