#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-G の対照づくり — **速度引き上げ試験（速度水準を下から順に上げて走らせる
一連の試験）の成績を、旋回安全率 0.70 と 0.75 で突き合わせる**。

`card_016g.md` §1-1:「016-cal（旋回安全率の校正実験）で既定が 0.75 になったので、
016-G の対照は速度引き上げ試験を 0.75 で測り直したものである」。

**主判定は「20 迷路すべてが A-成立する最大の速度水準」＝ 迷路ごとの最小値**なので、
**作法 8 に従い迷路ごとの内訳を必ず出す**（1 迷路の悪化で主判定が下がり、
他の迷路の改善が見えなくなるため）。

**本スクリプトは読むだけで、走らせない。**入力の JSON は
`run_016f0_ladder.py` が書いたもの。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/compare_016g_ladder.py \
        --base outputs/exp_016_diagonal/016g/repro_k1_ri_default_sf.json \
        --new  outputs/exp_016_diagonal/016g/ladder_k1_ri_sf0.75.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def per_face_pass(rows, ladder):
    """迷路ごとの合格速度（A-成立した最大の速度水準。1 つも無ければ 0.0）。"""
    out = []
    for r in rows:
        ok = [v for v in ladder if r["per_speed"][f"{v:g}"]["verdict"] == "A-成立"]
        out.append(max(ok) if ok else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="対照の JSON（旋回安全率 0.70）")
    ap.add_argument("--new", required=True, help="処理の JSON（旋回安全率 0.75）")
    ap.add_argument("--base-label", default="安全率 0.70")
    ap.add_argument("--new-label", default="安全率 0.75")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016g" / "ladder_sf_compare.json"))
    args = ap.parse_args()

    a = json.load(open(args.base, encoding="utf-8"))
    b = json.load(open(args.new, encoding="utf-8"))
    ladder = a["ladder"]
    assert ladder == b["ladder"], "速度水準の一覧が違う。突き合わせられない"
    assert [r["maze"] for r in a["rows"]] == [r["maze"] for r in b["rows"]], "迷路の並びが違う"

    pa = per_face_pass(a["rows"], ladder)
    pb = per_face_pass(b["rows"], ladder)

    print(f"対照 = {args.base_label}（{Path(args.base).name}・git_rev {a['git_rev'][:8]}）")
    print(f"処理 = {args.new_label}（{Path(args.new).name}・git_rev {b['git_rev'][:8]}）\n")

    # ---- 迷路ごとの内訳（作法 8）----------------------------------------
    print("【迷路ごとの合格速度】と、速度水準 0.45 での余裕・横偏差の最大")
    print(f"{'迷路':<12}{'対照':>6}{'処理':>6}  {'向き':<4}"
          f"{'余裕min [mm]':>22}{'e_y max [mm]':>22}")
    faces = []
    for ra, rb, va, vb in zip(a["rows"], b["rows"], pa, pb):
        d = "上" if vb > va else ("下" if vb < va else "不変")
        ma, mb = ra["per_speed"]["0.45"], rb["per_speed"]["0.45"]
        m0, m1 = ma["margin_min_m"] * 1000, mb["margin_min_m"] * 1000
        e0, e1 = ma["e_y_max_m"] * 1000, mb["e_y_max_m"] * 1000
        print(f"{ra['maze']:<12}{va:>6.2f}{vb:>6.2f}  {d:<4}"
              f"{m0:>9.3f} →{m1:>9.3f}{e0:>10.3f} →{e1:>9.3f}")
        faces.append(dict(maze=ra["maze"], pass_base=va, pass_new=vb, direction=d,
                          margin_min_mm_base=m0, margin_min_mm_new=m1,
                          e_y_max_mm_base=e0, e_y_max_mm_new=e1))

    n_up = sum(1 for q in faces if q["direction"] == "上")
    n_dn = sum(1 for q in faces if q["direction"] == "下")
    n_eq = len(faces) - n_up - n_dn
    print(f"\n上がった {n_up} 迷路 / 下がった {n_dn} 迷路 / 不変 {n_eq} 迷路")
    print(f"迷路ごとの合格速度: 対照 最小 {min(pa):.2f}／中央値 {np.median(pa):.2f}／最大 {max(pa):.2f}"
          f"  →  処理 最小 {min(pb):.2f}／中央値 {np.median(pb):.2f}／最大 {max(pb):.2f}")
    print(f"**主判定 v_斜め^max: {a['v_diag_max']} → {b['v_diag_max']} m/s**")

    # ---- 速度水準ごとの A-不成立の数 ------------------------------------
    print("\n【速度水準ごとの A-不成立の迷路数】")
    levels = []
    for v in ladder:
        k = f"{v:g}"
        na = sum(1 for r in a["rows"] if r["per_speed"][k]["verdict"] != "A-成立")
        nb = sum(1 for r in b["rows"] if r["per_speed"][k]["verdict"] != "A-成立")
        print(f"  {v:.2f} m/s : 対照 {na:2d} 迷路 → 処理 {nb:2d} 迷路")
        levels.append(dict(v=v, n_fail_base=na, n_fail_new=nb))

    # ---- 同じ迷路どうしの対応差（§9-15・裁定 R17）-----------------------
    print("\n【同じ迷路どうしの対応差】（要約統計量どうしの比較ではない）")
    paired = {}
    for v in ladder:
        k = f"{v:g}"
        dm = [(rb["per_speed"][k]["margin_min_m"] - ra["per_speed"][k]["margin_min_m"]) * 1000
              for ra, rb in zip(a["rows"], b["rows"])]
        de = [(rb["per_speed"][k]["e_y_max_m"] - ra["per_speed"][k]["e_y_max_m"]) * 1000
              for ra, rb in zip(a["rows"], b["rows"])]
        dp = [(rb["per_speed"][k]["psi_max_deg"] - ra["per_speed"][k]["psi_max_deg"])
              for ra, rb in zip(a["rows"], b["rows"])]
        print(f"  速度水準 {v:.2f}: 余裕 {np.median(dm):+.3f} mm"
              f"（{min(dm):+.3f}〜{max(dm):+.3f}）／"
              f"e_y最大 {np.median(de):+.3f} mm（{min(de):+.3f}〜{max(de):+.3f}）／"
              f"ψ最大 {np.median(dp):+.3f}°（{min(dp):+.3f}〜{max(dp):+.3f}）")
        paired[k] = dict(d_margin_mm_med=float(np.median(dm)),
                         d_margin_mm_min=float(min(dm)), d_margin_mm_max=float(max(dm)),
                         d_e_y_mm_med=float(np.median(de)),
                         d_e_y_mm_min=float(min(de)), d_e_y_mm_max=float(max(de)),
                         d_psi_deg_med=float(np.median(dp)),
                         d_psi_deg_min=float(min(dp)), d_psi_deg_max=float(max(dp)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(base=str(args.base), new=str(args.new),
                   base_label=args.base_label, new_label=args.new_label,
                   git_rev_base=a["git_rev"], git_rev_new=b["git_rev"],
                   v_diag_max_base=a["v_diag_max"], v_diag_max_new=b["v_diag_max"],
                   n_up=n_up, n_down=n_dn, n_same=n_eq,
                   per_face=faces, per_level=levels, paired_diff=paired),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
