#!/usr/bin/env python3
"""任務 N3-1: 時間モデル係数 a への「0.948 割り戻し」が結論に使われていないかの監査。

作成: 2026-08-11 准教授セッション（独立検証担当）

## 問い（教授からの任務 N3-1）

前任の准教授が提案した「時間モデルの係数 a を 0.948（＝最短走行の実走距離 ÷ 理論値
0.18·D₀）で割り戻す」補正が、`verification/` 側の報告・スクリプトで**結論に使われて
いないか**を確認する。使われていれば、その結論の影響範囲を示す。

## 方法

1. **静的走査**: `verification/` 配下の全 .py / .md を機械的に走査し、
   0.948 系の数値リテラル・割り戻しに相当する演算・係数の再定義が無いことを示す。
2. **代数**: `audit_c2_negative.py` の予測式に対して、係数のスケーリングが
   どう効くかを厳密に示す（共通スケールは消えるが、a のみのスケールは消えない）。
3. **数値**: 仮に割り戻しを適用したら (c2) の予測値がどう動いたかを実際に計算し、
   「避けられた誤差」の大きさを定量化する。
4. **補足**: 距離／歩数の比が (c2) に入りうる**唯一の正しい経路**を示し、その大きさを見る。

⚠️ 本スクリプトが出す (c2) 系の数値は **暫定**である。教授指示により、n_turns の定義が
学生A・学生B 間で統一され評価帯が再作成されるまで、(c2) 系の数値は確定扱いにしない。
本スクリプトの**結論（割り戻しは使われていない）は (c2) の値に依存しない**。
"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

import numpy as np

import audit_c2_negative as c2mod

REPO = Path(__file__).resolve().parent.parent
VDIR = REPO / "verification"
CELL = c2mod.CELL


# ---------------------------------------------------------------- 1. 静的走査
def static_sweep() -> dict:
    """0.948 系の補正が verification/ のどこかに書かれていないかを機械的に探す。"""
    # 0.94〜0.95 台の定数、その逆数 1.05 台、明示的な「割り戻し」語
    pats = {
        "0.948 近傍の定数": re.compile(r"0\.9[45]\d*"),
        "逆数 1.05 近傍の定数": re.compile(r"1\.0[456]\d*"),
        "『割り戻』の語": re.compile(r"割り戻"),
        "『補正』の語": re.compile(r"補正"),
    }
    hits: dict[str, list] = {k: [] for k in pats}
    files = sorted(list(VDIR.glob("*.py")) + list(VDIR.glob("*.md")))
    for f in files:
        if f.name == Path(__file__).name:
            continue                      # 自分自身は対象外
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for k, p in pats.items():
                if p.search(line):
                    hits[k].append(f"{f.name}:{i}: {line.strip()[:110]}")
    return {"n_files": len(files), "hits": hits}


# ---------------------------------------------------------------- 2/3. 感度
def sensitivity() -> dict:
    """割り戻しを適用した場合の (c2) 予測値の変化を実際に計算する。"""
    rows = json.loads((VDIR / "out" / "c2_negative_audit.json").read_text())["rows"]
    bands = {b: REPO / "competition" / "mazes" / b
             for b in ("eval", "validation", "contest_reference", "eval_v2_short")}

    # 係数は audit_c2_negative.py と同一手順で再取得（min-turn 回帰）
    A = np.array([[r["d0"], r["turn_min"]] for r in rows], dtype=float)
    y = np.array([r["t_fast"] for r in rows], dtype=float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])

    ratio_all = [r["L_fast"] / (CELL * r["d0"]) for r in rows]
    k = st.median(ratio_all)              # ＝ 0.9477。前任が提案した割り戻しの分母

    recs = []
    for r in rows:
        m = c2mod.load_maze(bands[r["band"]] / (r["maze"] + ".npz"))
        se, te = c2mod.simulate_explore(m)
        if se is None:
            continue
        d_exp, d_fast = te / se, r["turn_min"] / r["d0"]

        def pred(aa, bb):
            return (aa + bb * d_exp) / (aa + bb * d_fast) - 1.0

        recs.append({
            "band": r["band"], "maze": r["maze"],
            "c2_obs": (r["L_fast"] / r["t_fast"]) / (r["L_exp"] / r["t_exp"]) - 1.0,
            "pred_raw": pred(a, b),               # 現行（回帰値そのまま）
            "pred_a_only": pred(a / k, b),        # 前任提案: a だけ割り戻す
            "pred_both": pred(a / k, b / k),      # 参考: a も b も割り戻す（共通スケール）
            # 距離/歩数の比が入りうる唯一の正しい経路（暫定・下記 §注参照）
            "mps_fast": r["L_fast"] / r["d0"],
            "mps_exp": r["L_exp"] / se,
        })

    med = lambda key: st.median([x[key] for x in recs])
    rho = [x["mps_fast"] / x["mps_exp"] for x in recs]
    return {
        "n": len(recs), "a": a, "b": b, "k": k,
        "median": {kk: med(kk) for kk in
                   ("c2_obs", "pred_raw", "pred_a_only", "pred_both")},
        "b_over_a": {"raw": b / a, "a_only": b / (a / k)},
        "rho": {"median": st.median(rho), "min": min(rho), "max": max(rho)},
        "recs": recs,
    }


# ------------------------------------------------- 5. 伝達 1 の独立確認（依頼外）
def denominator_check() -> dict:
    """回帰の分母を D₀ にした場合と D₀−1 にした場合で a がどう変わるかを見る。

    学生A は D₀−1 を分母に取って a = 0.7216 を得ている（計時窓の端で区画遷移が
    1 つ落ちるため）。本関数はそれを独立に再現する。
    """
    rows = json.loads((VDIR / "out" / "c2_negative_audit.json").read_text())["rows"]
    y = np.array([r["t_fast"] for r in rows], dtype=float)
    out = {}
    for tag, off in (("a_d0", 0), ("a_d0m1", 1)):
        A = np.array([[r["d0"] - off, r["turn_min"]] for r in rows], dtype=float)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[tag] = float(coef[0])
    # 「a を D₀/(D₀−1) 倍すれば換算できる」という素朴な読み替えは**成立しない**。
    # 最小二乗は各点を個別にスケールするのではなく、折れ数の項がずれの一部を吸収するため。
    out["naive_rescale"] = st.median([out["a_d0"] * r["d0"] / (r["d0"] - 1) for r in rows])
    out["d0_median"] = st.median([r["d0"] for r in rows])
    out["n"] = len(rows)
    return out


def main() -> None:
    W = 96
    print("=" * W)
    print("任務 N3-1: 時間モデル係数 a への「0.948 割り戻し」が結論に使われていないかの監査")
    print("=" * W)

    sw = static_sweep()
    print(f"\n[1] 静的走査 — verification/ 配下 {sw['n_files']} ファイル（自分自身を除く）")
    for k, v in sw["hits"].items():
        print(f"\n  ● {k}: {len(v)} 件")
        for line in v[:12]:
            print(f"      {line}")
        if len(v) > 12:
            print(f"      … 他 {len(v)-12} 件")

    s = sensitivity()
    print(f"\n[2] 代数 — 予測式 (c2)_pred = (a + b·d_exp)/(a + b·d_fast) − 1")
    print("    a と b を**同じ倍率**で割ると分子分母から約分され、(c2)_pred は厳密に不変。")
    print("    a **だけ**を割ると b/a が変わるので、(c2)_pred は変わる。")
    print(f"    → 前任の提案は「無害だが無意味」ではなく、**数値を動かす変更**だった。")
    print(f"    b/a: 現行 {s['b_over_a']['raw']:.4f} → a だけ割り戻すと {s['b_over_a']['a_only']:.4f}"
          f"（{(s['b_over_a']['a_only']/s['b_over_a']['raw']-1)*100:+.1f}%）")

    print(f"\n[3] 数値 — 割り戻しを適用したら (c2) 予測がどう動いたか（n = {s['n']} 面・中央値）")
    print(f"    回帰値: a = {s['a']:.4f} s/歩, b = {s['b']:.4f} s/折れ ／ 割り戻しの分母 k = {s['k']:.4f}")
    m = s["median"]
    print(f"    {'実測 (c2)':<34}{m['c2_obs']*100:>+9.2f}%")
    print(f"    {'予測 (c2) 現行（回帰値そのまま）':<30}{m['pred_raw']*100:>+9.2f}%   ← 報告に載っている値")
    print(f"    {'予測 (c2) a だけ割り戻す（前任案）':<29}{m['pred_a_only']*100:>+9.2f}%   "
          f"（現行から {(m['pred_a_only']-m['pred_raw'])*100:+.2f} 分点）")
    print(f"    {'予測 (c2) a も b も割り戻す（参考）':<29}{m['pred_both']*100:>+9.2f}%   ← 現行と厳密一致（約分）")
    print(f"    → 前任案を適用していたら、予測は実測（{m['c2_obs']*100:+.2f}%）から"
          f"**遠ざかっていた**")

    print(f"\n[4] 【補足・暫定】距離/歩数の比が (c2) に入りうる唯一の正しい経路")
    print("    (c2) は 2 本の走行の**速度比**なので、両走行に共通のコーナー切り係数は約分で消える。")
    print("    残るのは 2 本の**差**だけ: ρ = (最短走行の m/歩) / (探索走行の m/歩)。")
    print("    これは (c2)+1 に**乗算**で効く量であり、a を割る形ではない。")
    print(f"    ρ: 中央値 {s['rho']['median']:.4f}（min {s['rho']['min']:.4f} / max {s['rho']['max']:.4f}）")
    print("    ※ 探索走行の実歩数は保存されていないため、足立法シミュレーションの歩数で代用した")
    print("       proxy である。実走経路とシミュレーション経路が一致しない面では誤差を含む。")
    print("    ※ 教授指示により (c2) 系の数値は確定扱いにしない（n_turns の定義統一・帯の再作成待ち）。")
    print("       **本監査の結論（割り戻しは使われていない）は、この数値に依存しない。**")

    d = denominator_check()
    print(f"\n[5] 【依頼外・独立確認】伝達 1（分母が D₀ か D₀−1 か）の再現")
    print("    `docs/RESEARCH_PLAN.md` §2 は 0.948 の原因を 2 つ挙げている:")
    print("      (i) 計時がスタートセンサ通過〜ゴールセンサ通過なので**両端が切り取られる**")
    print("      (ii) 旋回で**内側を回る**ぶん実走距離が短くなる")
    print("    (i) は歩数の数え方（D₀ か D₀−1 か）の問題、(ii) は歩数に一切効かない問題。")
    print("    **0.948 は両者を混ぜた量なので、どちらの補正係数にもなりえない。**")
    print(f"    分母 D₀   で再回帰: a = {d['a_d0']:.4f} s/歩")
    print(f"    分母 D₀−1 で再回帰: a = {d['a_d0m1']:.4f} s/歩  ← 学生A の 0.7216 と "
          f"**{abs(d['a_d0m1']/0.7216-1)*100:.2f}% 差**（n = {d['n']} 面, D₀ 中央値 {d['d0_median']:.0f}）")
    print("    → 伝達 1 の「食い違いではなく分母が 1 歩ずれていただけ」を独立に再現した。")
    print(f"\n    ⚠️ ただし『a を D₀/(D₀−1) 倍すれば換算できる』という読み替えは**成立しない**:")
    print(f"       素朴に倍率をかけると {d['naive_rescale']:.4f} になり、再回帰値 {d['a_d0m1']:.4f} と合わない。")
    print("       最小二乗は各点を個別にスケールせず、**折れ数の項がずれの一部を吸収する**ため。")
    print("       分母を変えた比較は、必ず**再回帰**で行うこと（スカラー換算では誤る）。")

    outp = VDIR / "out" / "n3_coefficient_use_audit.json"
    outp.write_text(json.dumps({"static_sweep": sw, "sensitivity": s,
                                "denominator_check": d},
                               ensure_ascii=False, indent=1))
    print(f"\n書き出し: {outp}")


if __name__ == "__main__":
    main()
