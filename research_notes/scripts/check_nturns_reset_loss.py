#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学生B の `n_turns` は、なぜ「区画列の 2 倍」にならないのか — 積算のリセットを直接数える。

**先に事実**（`check_nturns_definitions.py` の実測）: 2 定義の比 B/A は
**L0-a（超信地旋回）で約 1.1、L0-c（スラローム）で約 2.0** と、走り方で違った。

**理想的にはどちらも 2.0 のはず**である。区画列の定義で 1 と数える 90° 転回は、
ヨー角が 90° 動くので ±45° の閾値を 2 回跨ぐ。**では L0-a で何が起きているのか。**

**仮説**: 学生B の実装は「積算の向きと逆の増分が来たら積算を 0 に戻す」

    if yaw_acc * dyaw < 0.0:   # mouse/corridor_eval.py L116
        yaw_acc = 0.0

という規則を持つ。**超信地旋回は各区画で停止するので、停止・整定のたびに制御の
リップルでヨー角の増分の符号が反転しうる。**反転が旋回の途中で起きると、
**そこまで溜めた本物の回転がまるごと捨てられる**。

**反証の形で確かめる**: 仮説が偽なら、リセットで捨てられる累積は総ヨー角に対して
無視できるはずである（数％以下）。真なら、**L0-a では大きく、L0-c では小さい**という
方式ごとの差が出るはずである。**捨てられた量を直接測って判別する。**

入力は `check_nturns_definitions.py` が保存した軌跡（走行境界つき）。
**走らせ直さない**（同じ軌跡に別の数え方を当てるだけ）。

使い方:
    .venv/bin/python research_notes/scripts/check_nturns_reset_loss.py \
        --traj-dir outputs/nturns_defs/l0a/traj outputs/nturns_defs/l0c/traj
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def analyse_run(yaw):
    """1 走行ぶんのヨー角列から、学生B の計数とリセットの内訳を出す。"""
    if len(yaw) < 2:
        return None
    acc, n_turns, n_reset, lost, mags = 0.0, 0, 0, 0.0, []
    total = 0.0
    prev = yaw[0]
    for y in yaw[1:]:
        d = math.atan2(math.sin(y - prev), math.cos(y - prev))
        prev = y
        total += abs(d)
        if acc * d < 0.0:
            if abs(acc) > 1e-9:
                n_reset += 1
                lost += abs(acc)
                mags.append(math.degrees(abs(acc)))
            acc = 0.0
        acc += d
        if abs(acc) >= math.pi / 4:
            n_turns += 1
            acc = 0.0
    return dict(n_turns=n_turns, total_deg=math.degrees(total),
                n_reset=n_reset, lost_deg=math.degrees(lost), reset_mags=mags,
                # リセットが無ければ何回数えたか（総ヨー角 ÷ 45°）
                n_turns_noreset=math.degrees(total) / 45.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj-dir", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = {}
    for d in args.traj_dir:
        p = Path(d) if Path(d).is_absolute() else REPO_ROOT / d
        files = sorted(p.glob("maze_*.npz"))
        label = p.parent.name if p.name == "traj" else p.name
        rows = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            t, yaw = z["t"], z["yaw"]
            for i, (t0, t1) in enumerate(zip(z["run_t_start"], z["run_t_end"])):
                seg = yaw[(t >= t0 - 1e-9) & (t <= t1 + 1e-9)]
                r = analyse_run(seg)
                if r is None:
                    continue
                r.update(maze=f.stem, run=int(z["run_index"][i]),
                         outcome=str(z["run_outcome"][i]))
                rows.append(r)
        if not rows:
            print(f"【{label}】軌跡が見つからない: {p}")
            continue

        tot = np.array([r["total_deg"] for r in rows])
        lost = np.array([r["lost_deg"] for r in rows])
        nt = np.array([r["n_turns"] for r in rows], float)
        nn = np.array([r["n_turns_noreset"] for r in rows])
        frac = lost / np.maximum(tot, 1e-9)
        mags = np.concatenate([np.array(r["reset_mags"]) for r in rows if r["reset_mags"]])
        print(f"\n【{label}】n={len(rows)} 走行")
        print(f"  総ヨー角: 中央値 {np.median(tot):.0f}°／走行")
        print(f"  **リセットで捨てられた累積の割合**: 中央値 {np.median(frac) * 100:.1f}%"
              f"（四分位 {np.percentile(frac, 25) * 100:.1f}〜{np.percentile(frac, 75) * 100:.1f}%"
              f"、範囲 {frac.min() * 100:.1f}〜{frac.max() * 100:.1f}%）")
        print(f"  リセット回数: 中央値 {np.median([r['n_reset'] for r in rows]):.0f}／走行")
        print(f"  リセット時の |累積| [deg]: 中央値 {np.median(mags):.2f}"
              f"、四分位 {np.percentile(mags, 25):.2f}〜{np.percentile(mags, 75):.2f}"
              f"、最大 {mags.max():.1f}"
              f"（**45° 直前で捨てた**ものが {(mags >= 40).sum()} 件 / {mags.size}）")
        print(f"  学生B の計数: 実測 中央値 {np.median(nt):.0f} 対 "
              f"リセットが無ければ {np.median(nn):.0f}"
              f"（実測 / リセット無し = {np.median(nt / nn):.3f}）")
        out[label] = dict(
            n_runs=len(rows), lost_frac_med=float(np.median(frac)),
            lost_frac_p25=float(np.percentile(frac, 25)),
            lost_frac_p75=float(np.percentile(frac, 75)),
            n_reset_med=float(np.median([r["n_reset"] for r in rows])),
            reset_mag_med=float(np.median(mags)), reset_mag_max=float(mags.max()),
            n_reset_near45=int((mags >= 40).sum()), n_reset_total=int(mags.size),
            n_turns_med=float(np.median(nt)), n_turns_noreset_med=float(np.median(nn)))

    if len(out) >= 2:
        ks = list(out)
        print(f"\n【判別】捨てられた割合の中央値: "
              + "／".join(f"{k} {out[k]['lost_frac_med'] * 100:.1f}%" for k in ks))
        print("  方式ごとに大きく違えば、**リセット規則が走り方に依存して効いている**"
              "（＝定義が走り方をまたいだ比較に使えない）。")

    p = Path(args.out) if args.out else REPO_ROOT / "research_notes" / "data" / "nturns_reset_loss.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n数値 JSON: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
