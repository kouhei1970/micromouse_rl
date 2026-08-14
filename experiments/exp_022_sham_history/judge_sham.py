#!/usr/bin/env python3
"""exp_022 の P1〜P4 を機械的に判定する（カード §3 の条文と 1 対 1 対応）。

Mechanically evaluate the pre-registered predictions P1-P4 for exp_022.

🔴 **判定は手で計算しない。**判定の型は「にせ履歴群が 2 つの錨
（対照群 exp_019・参照群 exp_021）のどちらに近いか」であり、**錨の値は投入前に確定した実測**である。
**本スクリプトは錨を定数として持ち、測定出力から再計算した値と照合して食い違えば落ちる**
（**後から錨を差し替えられない形にする**）。

## 条文（`card.md` §3-1・§3-2）との対応

| 条文 | 実装 |
|---|---|
| 相対位置 **r = ln(sham ÷ control) ÷ ln(treat ÷ control)** | `_relative_position()` |
| **r < 0.5 → (A/C) 情報／汚染の説** ／ **r ≥ 0.5 → (B) 次元の説** | 同 |
| **どちらの錨よりも外**（r < 0 または r > 1）は **3 つ目の読み**として分ける | 同（`verdict="outside"`） |
| **P1** = `respawn_per_1000`（対照 0.500・参照 2.125・境界 1.031） | `ANCHORS["P1"]` |
| **P2** = 5 区画以上到達した走行数（対照 3・参照 26・境界 8.8） | `ANCHORS["P2"]` |
| **P3** = rollout のゴール件数（対照 0・参照 4）— **比が定義できない** | `judge_p3()`（別扱い） |
| **P4** = `net_progress_per_1000`（対照 1.500・参照 1.250・境界 1.369） | `ANCHORS["P4"]` |
| **P3・P4 は分離能が弱く、P1・P2 と食い違ったら P1・P2 を採る** | 出力の `primary` / `secondary` |
| 集約は **迷路 20 本の中央値 → 6 seed の中央値**（P1・P4）／**120 走行の件数**（P2・P3） | `measure_driving.py` の `summarize()`（本 script は読むだけ） |

## 使い方

```bash
.venv/bin/python experiments/exp_022_sham_history/judge_sham.py \
    --control outputs/exp_021_driving_control_final.json \
    --treat   outputs/exp_021_driving_treat_final.json \
    --sham    outputs/exp_022_driving_sham_final.json \
    --out outputs/exp_022_judgment.json
```
"""
import argparse
import json
import math
from pathlib import Path

#: 🔴 事前登録した錨（`card.md` §3-2。**投入前に確定した実測**。**変更しない**）。
#: `where` = 判定量の場所（"median" は summary.across_seeds_median、"count" は summary 直下）。
#: 🔴 **向きを表すキーは置かない** — **対数比の式が向きを自動で吸収する**ので
#: （P4 は treat < control だが `_relative_position` がそのまま扱う）、
#: 向きのキーがあると「これが効いている」と誤解される死んだ情報になる。
ANCHORS = {
    "P1": dict(name="P1 衝突の頻度", field="respawn_per_1000", where="median",
               control=0.500, treat=2.125),
    "P2": dict(name="P2 5 区画以上到達の走行数", field="n_reach_ge5", where="count",
               control=3, treat=26),
    "P4": dict(name="P4 正味の前進", field="net_progress_per_1000", where="median",
               control=1.500, treat=1.250),
}
#: P3 は対照が 0 件で比が定義できないので別扱い（カード §3-4）。
P3 = dict(name="P3 rollout のゴール件数", field="n_goal_rollout",
          control=0, treat=4)

R_BOUNDARY = 0.5          # 2 つの錨の幾何中央
#: 錨の照合の許容差（浮動小数の丸めのみを吸収する幅。**設計上の余裕ではない**）
ANCHOR_TOL = 1e-9


def _value(summary: dict, spec: dict):
    if spec["where"] == "median":
        return summary["across_seeds_median"][spec["field"]]
    return summary[spec["field"]]


def _relative_position(sham, control, treat) -> float:
    """r = ln(sham/control) / ln(treat/control)。

    **比で測る量なので対数の中点を採る**（カード §3-1）。
    **向きは式が自動で扱う**（treat < control の量では ln(treat/control) が負になるため）。
    """
    if control <= 0 or treat <= 0 or sham <= 0:
        raise ValueError(f"r は正の値でしか定義できない（sham={sham} control={control} treat={treat}）")
    denom = math.log(treat / control)
    if denom == 0:
        raise ValueError("2 つの錨が同じ値なので r が定義できない")
    # 🔴 `+ 0.0` は符号つきゼロの正規化（准教授 AUDIT の指摘）。
    # sham == control のとき分子は +0.0・分母は負になりうるので r が -0.0 になり、
    # 「-0.000」と表示されて「わずかに対照側へ寄っている」向きがあるように読める。
    # 実際には向きはなく錨と厳密に同値なので、+0.0 を足して 0.0 に正規化する。
    return math.log(sham / control) / denom + 0.0


def judge_one(key: str, spec: dict, sham_summary: dict) -> dict:
    v = _value(sham_summary, spec)
    r = _relative_position(v, spec["control"], spec["treat"])
    boundary = spec["control"] * (spec["treat"] / spec["control"]) ** R_BOUNDARY
    if r < 0.0 or r > 1.0:
        # 🔴 3 つ目の読み（カード §3-3）: 2 つの錨で挟む前提が崩れている
        verdict = "outside"
        side = ("対照群より外側（対照群よりも良い）" if r < 0.0
                else "参照群より外側（exp_021 よりも悪い）")
    elif r < R_BOUNDARY:
        verdict, side = "A_or_C", "対照群の側 →（A/C）情報／汚染の説"
    else:
        verdict, side = "B", "参照群の側 →（B）次元の説"
    return dict(key=key, name=spec["name"], field=spec["field"],
                sham=v, control=spec["control"], treat=spec["treat"],
                boundary=boundary, r=r, verdict=verdict, reading=side)


def judge_p3(sham_summary: dict) -> dict:
    v = sham_summary[P3["field"]]
    # 対照が 0 件なので r は計算できない。粗い読みにとどめる（カード §3-4）。
    verdict = "A_or_C" if v == 0 else "B"
    return dict(key="P3", name=P3["name"], field=P3["field"], sham=v,
                control=P3["control"], treat=P3["treat"], boundary=None, r=None,
                verdict=verdict,
                reading=("0 件 →（A/C）寄り" if v == 0 else f"{v} 件 →（B）寄り"),
                note=("対照が 0 件なので比が定義できず r は計算しない。"
                      "分離能が弱く（錨の差自体が両側 Fisher で p = 0.122）、"
                      "P1・P2 と食い違ったら P1・P2 を採る（投入前に固定）"))


def _check_anchors(control_summary: dict, treat_summary: dict) -> list:
    """🔴 錨が事前登録の値と一致することを、測定出力から再計算して確かめる。

    **一致しなければ落とす**（**錨を後から差し替えられない形にする**）。
    """
    msgs = []
    for key, spec in ANCHORS.items():
        for tag, summ in (("control", control_summary), ("treat", treat_summary)):
            got, want = _value(summ, spec), spec[tag]
            if abs(got - want) > ANCHOR_TOL:
                msgs.append(f"{key} の錨 {tag} が事前登録と違う（登録 {want} / 実測 {got}）")
    for tag, summ in (("control", control_summary), ("treat", treat_summary)):
        got, want = summ[P3["field"]], P3[tag]
        if got != want:
            msgs.append(f"P3 の錨 {tag} が事前登録と違う（登録 {want} / 実測 {got}）")
    return msgs


def main() -> None:
    p = argparse.ArgumentParser(description="exp_022 の P1〜P4 の機械的判定")
    p.add_argument("--control", required=True, help="対照群 exp_019 の最終方策の測定出力")
    p.add_argument("--treat", required=True, help="参照群 exp_021 の最終方策の測定出力")
    p.add_argument("--sham", required=True, help="にせ履歴群 exp_022 の最終方策の測定出力")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    c = json.loads(Path(args.control).read_text(encoding="utf-8"))
    t = json.loads(Path(args.treat).read_text(encoding="utf-8"))
    s = json.loads(Path(args.sham).read_text(encoding="utf-8"))

    # 🔴 判定の前に落とす安全弁
    for tag, d in (("control", c), ("treat", t), ("sham", s)):
        n = len(d["summary"]["per_seed_median"])
        if n != 6:
            raise SystemExit(f"{tag} の seed 数が 6 でない（{n}）。集計が壊れている")
        steps = sorted({m["num_timesteps"] for m in d["models"]})
        if len(steps) != 1:
            raise SystemExit(f"{tag} の学習量が seed 間で揃っていない（{steps}）")
    steps = [sorted({m["num_timesteps"] for m in d["models"]})[0] for d in (c, t, s)]
    if len(set(steps)) != 1:
        raise SystemExit(f"3 群の学習量が違う（control/treat/sham = {steps}）")
    if not s.get("history_sham"):
        raise SystemExit("にせ履歴群の測定に history_sham が記録されていない（--history-sham 忘れ）")
    if t.get("history_sham") or c.get("history_sham"):
        raise SystemExit("対照群または参照群の測定に history_sham が入っている")
    if not t["history_lags"] or c["history_lags"]:
        raise SystemExit("参照群に遅れが無い、または対照群に遅れが入っている")
    if s["history_lags"] != t["history_lags"]:
        raise SystemExit(f"にせ履歴群と参照群の遅れが違う（{s['history_lags']} 対 {t['history_lags']}）")
    bad = _check_anchors(c["summary"], t["summary"])
    if bad:
        raise SystemExit("錨が事前登録と食い違う:\n  " + "\n  ".join(bad))

    primary = [judge_one(k, ANCHORS[k], s["summary"]) for k in ("P1", "P2")]
    secondary = [judge_p3(s["summary"]), judge_one("P4", ANCHORS["P4"], s["summary"])]

    out = dict(clause="experiments/exp_022_sham_history/card.md §3",
               num_timesteps=steps[0], history_lags=s["history_lags"],
               anchors={k: {kk: v[kk] for kk in ("control", "treat")}
                        for k, v in list(ANCHORS.items()) + [("P3", P3)]},
               inputs=dict(control=str(args.control), treat=str(args.treat), sham=str(args.sham)),
               primary=primary, secondary=secondary)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"学習量: 3 群とも {steps[0]:,} 歩 / 遅れ {s['history_lags']} / にせ履歴 = 有効")
    print("=" * 78)
    for tag, rows in (("【主】判定に使う", primary), ("【副】分離能が弱い・単独では判別しない", secondary)):
        print(f"\n{tag}")
        for r in rows:
            rs = "—" if r["r"] is None else f"{r['r']:.3f}"
            bs = "—" if r["boundary"] is None else f"{r['boundary']:.4f}"
            print(f"  {r['name']}: にせ履歴 {r['sham']} "
                  f"（対照 {r['control']} / 参照 {r['treat']} / 境界 {bs}）")
            print(f"      r = {rs} → {r['verdict']}（{r['reading']}）")
    print("=" * 78)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
