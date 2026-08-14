#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp_021 の対照群測定の独立検算（投入前）

准教授セッション（8 代目）・2026-08-14・`AUDIT_041` §8

## 何をするか

学生B が投入前に測った対照群（`outputs/exp_021_driving_control_{final,800k}.json`）について:

  A. 判定量の**式**を、保存されている入力（`d0`・`min_d`・`n_steps`・`n_respawn`）から再計算する
  B. **集約**（迷路 20 本の中央値 → 6 seed 中央値）を再計算する
  C. **報告トリガーの閾値 0.5 倍が、80 万歩の対照の seed 間のばらつきの外側にあるか**を確かめる
     （教授が投入条件にした項目。内側なら投入停止）
  D. **判定量の刻み**を実測から求め、Q1・Q2 の閾値が刻みに対してどこに落ちるかを示す
     （研究計画書 §9-16。カードは Q3 には当てているが Q1・Q2 には当てていない）

## 検算の限界（**報告に必ず併記する**）

`measure_driving.py` は毎歩の `d_hist` / `resp_hist` を作るが**出力に残さない**ので、
**`min_d` と `min_d_after_last_respawn` そのものは検証できない**。
本スクリプトが確かめられるのは**式と集約**までであり、
**軌跡から判定量を再構成する層は現存物では実行できない**（`AUDIT_039` §3-1 と同じ構造）。
とくに **Q4 の窓（最後のリスポーン以降）は裁定 R50 で実際に誤りが見つかった箇所**である。
"""

import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(REPO, "outputs/exp_021_driving_control_final.json")
K800 = os.path.join(REPO, "outputs/exp_021_driving_control_800k.json")

# カード §5 の閾値（結果を見る前に条文で決まっているもの）
Q1_FACTOR = 1.25      # net_progress は対照の 1.25 倍以上で当たり
Q2_FACTOR = 0.80      # respawn は対照の 0.80 倍以下で当たり
Q2_GUARD = 0.90       # かつ net_progress が対照の 0.90 倍を下回らない
Q4_LINE = 0.50        # 立て直しの成立割合 50% 以上
TRIGGER = 0.50        # 報告トリガー: 80 万歩の対照の 0.5 倍以下で即報告


def seed_medians(doc, key):
    """seed ごとに「迷路 20 本の中央値」を出す（カード §4-1 の集約の前半）。"""
    return [statistics.median([m[key] for m in blk["metrics"]])
            for blk in doc["detail"].values()]


def main():
    for p in (FINAL, K800):
        if not os.path.exists(p):
            print(f"対照群の測定が見つからない: {p}")
            return 2

    docs = {"final": json.load(open(FINAL)), "800k": json.load(open(K800))}
    fails = []

    # ---------------------------------------------------------------
    print("=== A. 判定量の式の再現（保存されている入力から） ===")
    for tag, d in docs.items():
        n = ok_np = ok_rs = 0
        for blk in d["detail"].values():
            for m in blk["metrics"]:
                n += 1
                # カード §4-1 の定義そのまま
                np_mine = (m["d0"] - m["min_d"]) / m["n_steps"] * 1000.0
                rs_mine = m["n_respawn"] / m["n_steps"] * 1000.0
                ok_np += abs(np_mine - m["net_progress_per_1000"]) <= 1e-9
                ok_rs += abs(rs_mine - m["respawn_per_1000"]) <= 1e-9
        print(f"  [{tag}] net_progress {ok_np}/{n} ・ respawn {ok_rs}/{n}")
        if ok_np != n or ok_rs != n:
            fails.append(f"{tag}: 式の再現に不一致")

    # ---------------------------------------------------------------
    print("\n=== B. 集約の再現（迷路 20 本の中央値 → 6 seed 中央値） ===")
    agg = {}
    for tag, d in docs.items():
        rec = d["summary"]["across_seeds_median"]
        for key in ("net_progress_per_1000", "respawn_per_1000"):
            sm = seed_medians(d, key)
            mine = statistics.median(sm)
            agg[(tag, key)] = (sm, mine)
            ok = abs(mine - rec[key]) <= 1e-9
            print(f"  [{tag}] {key}: 自前 {mine} / 記載 {rec[key]}  {'一致' if ok else '🔴 不一致'}")
            if not ok:
                fails.append(f"{tag}/{key}: 集約が不一致")

    # ---------------------------------------------------------------
    print("\n=== C. 報告トリガーの閾値がばらつきの外側にあるか（投入条件） ===")
    sm8, med8 = agg[("800k", "net_progress_per_1000")]
    line = TRIGGER * med8
    print(f"  80 万歩の対照: seed ごとの中央値 {sorted(sm8)} → 6 seed 中央値 {med8}")
    print(f"  ばらつき: {min(sm8)} 〜 {max(sm8)}"
          f"（中央値の {min(sm8)/med8:.2f}〜{max(sm8)/med8:.2f} 倍）")
    print(f"  発火線: {TRIGGER} × {med8} = {line}")
    outside = min(sm8) > line
    print(f"  最小の seed は発火線の {min(sm8)/line:.2f} 倍 → "
          f"{'🟢 ばらつきの外側（投入条件を満たす）' if outside else '🔴 内側（投入停止・教授へ報告）'}")
    if not outside:
        fails.append("報告トリガーの閾値がばらつきの内側にある")

    # 参考: 是正前の条文（200 万歩基準）ならどうだったか
    _, medf = agg[("final", "net_progress_per_1000")]
    old_line = TRIGGER * medf
    print(f"  （参考）是正前の条文なら発火線 {TRIGGER} × {medf} = {old_line}"
          f" → 最小の seed で {min(sm8)/old_line:.2f} 倍。**空振りの発火は起きなかった**")

    # ---------------------------------------------------------------
    print("\n=== D. 判定量の刻みと閾値の関係（§9-16。カードは Q3 にしか当てていない） ===")
    d = docs["final"]
    allm = [m for blk in d["detail"].values() for m in blk["metrics"]]
    steps = sorted({m["n_steps"] for m in allm})
    outs = sorted({m["outcome"] for m in allm})
    print(f"  対照群の歩数: {steps} ／ 結末: {outs}（{len(allm)} エピソード）")
    for key, factor, name in (("net_progress_per_1000", Q1_FACTOR, "Q1"),
                              ("respawn_per_1000", Q2_FACTOR, "Q2")):
        vals = sorted({m[key] for m in allm})
        grid = min((b - a for a, b in zip(vals, vals[1:])), default=float("nan"))
        _, med = agg[("final", key)]
        th = factor * med
        # 6 seed 中央値の刻み = 迷路ごとの刻み / 4（20 本の中央値で 1/2・6 本の中央値で更に 1/2）
        g6 = grid / 4.0
        on_grid = abs(th / g6 - round(th / g6)) < 1e-9
        print(f"  {name}: 迷路ごとの相異なる値 {vals[:8]}{'…' if len(vals) > 8 else ''}"
              f" → 刻み {grid}")
        print(f"      対照の 6 seed 中央値 {med} → 合格線 {factor} × {med} = {th}")
        print(f"      6 seed 中央値の刻み {g6} → 合格線は格子上に"
              f"{'ある' if on_grid else '**無い**'}")
        if not on_grid:
            import math
            best = math.floor(th / g6) * g6 if factor < 1 else math.ceil(th / g6) * g6
            print(f"      → 実際に合格しうる最も近い値は {best}"
                  f"（条文の {factor} 倍ではなく実質 {best/med:.3f} 倍）")

    # ---------------------------------------------------------------
    print("\n=== E. Q4（立て直し）の対照 ===")
    rates, dens = [], []
    for blk in docs["final"]["detail"].values():
        js = blk["p5"]
        dens.append(len(js))
        rates.append(sum(1 for j in js if j["advanced"]) / len(js) if js else None)
    ok = [r for r in rates if r is not None]
    print(f"  seed ごとの成立割合: {[round(r, 3) for r in ok]}")
    print(f"  母集団（リスポーン 1 回以上のエピソード数）: {dens}")
    print(f"  6 seed 中央値 {statistics.median(ok):.3f} / 合格線 {Q4_LINE}"
          f" → {'余裕あり' if statistics.median(ok) > Q4_LINE else '境界近傍'}")

    # ---------------------------------------------------------------
    # L5: 軌跡から判定量を自分で再構成する（生データが残っている場合のみ）
    # ---------------------------------------------------------------
    print("\n=== L5. 軌跡から判定量を自分で再構成（AUDIT_042_PREREG §4 の核心） ===")
    has_raw = all("raw" in blk for d in docs.values() for blk in d["detail"].values())
    if not has_raw:
        print("  🔴 毎歩の記録（raw）が出力に無い → L5 は実行不能。")
        print("     L2〜L4 の一致をもって「判定量が正しい」と書いてはならない（AUDIT_039 §3-1 と同じ限界）")
    else:
        for tag, d in docs.items():
            n = q1 = q2 = 0
            p5_ok = p5_den = bnd = 0
            for blk in d["detail"].values():
                raw = {r["maze_seed"]: r for r in blk["raw"]}
                for m in blk["metrics"]:
                    r = raw[m["maze_seed"]]
                    dh, rh = r["d_hist"], r["resp_hist"]
                    n += 1
                    # カード §4-1 の条文から自分で書く（相手のコードは見ていない）
                    q1 += abs((r["d0"] - min(dh)) / len(dh) * 1000.0
                              - m["net_progress_per_1000"]) <= 1e-9
                    q2 += abs(sum(1 for x in rh if x) / len(dh) * 1000.0
                              - m["respawn_per_1000"]) <= 1e-9
                # Q4: 窓（最後のリスポーン以降）を自分で切る
                p5 = {j["maze_seed"]: j for j in blk["p5"]}
                for ms, r in raw.items():
                    dh, rh = r["d_hist"], r["resp_hist"]
                    idx = [i for i, x in enumerate(rh) if x]
                    if not idx:
                        continue                     # 母集団 = リスポーン 1 回以上
                    p5_den += 1
                    last = idx[-1]
                    inc = min(dh[last:])
                    exc = min(dh[last + 1:]) if last + 1 < len(dh) else inc
                    # 境界を含む/含まないで判定が変わるか（R51 の「含む側で実装」の検査）
                    bnd += (inc <= r["d0"] - 1) != (exc <= r["d0"] - 1)
                    j = p5.get(ms)
                    if j is not None:
                        p5_ok += (inc == j["min_d_after_last_respawn"]
                                  and (inc <= r["d0"] - 1) == j["advanced"])
            print(f"  [{tag}] min_d の再構成 {q1}/{n} ・ 立て直し回数 {q2}/{n} ・ "
                  f"Q4 の窓 {p5_ok}/{p5_den}（境界で判定が変わる {bnd} 件）")
            if q1 != n or q2 != n or p5_ok != p5_den:
                fails.append(f"{tag}: L5 の再構成に不一致")

    # ---------------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"総括: 不合格 {len(fails)} 件" + ("" if not fails else f" → {fails}"))
    print("""
⚠️ 限界: `measure_driving.py` は毎歩の d_hist / resp_hist を出力に残さないので、
   `min_d` と `min_d_after_last_respawn` そのものは検証できない。
   確かめられるのは式と集約までである（AUDIT_039 §3-1 と同じ構造）。
""")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
