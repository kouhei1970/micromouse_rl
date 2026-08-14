#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**経路効率 (e′) の照合** — `card_016cal_switch.md` §7-2 の申し送りの実行。

> **(e′)（経路効率の指標 ＝ 計時窓で通過した区画の数 ÷ 真の最短歩数）は
> 照合できない。**本カードのランナー `run_016cal.py` が `n_cells` を
> 記録していないためである。……**(e′) の照合を `exp_013/run_arm.py` 系で行う。
> 確保済みの評価用 20 迷路 × 素の L0-c・安全率 0.75 を 1 回。**

**問いは 1 つだけ**: **旋回安全率を 0.70 → 0.75 に上げたとき、経路効率は動いたか。**
（**速度計画は経路を変えないので動かないはず**だが、**それは推論であって実測ではない**。
016-cal の切り替えではここが「判定不能」のまま残っていた。）

--------------------------------------------------------------------------
定義は書き直さない（裁定 R23）
--------------------------------------------------------------------------
**`exp_013/aggregate.py` を `importlib` で読み込み、`load_arm` と
`per_maze_metrics` をそのまま呼ぶ。**指標の定義をここへ写すと乖離する。
**`exp_013` 側のファイルは 1 行も変更しない**（完了済み実験なので再現性を守る）。

**(e′) の定義は裁定 R14**: 分子 = **節点数 = 移動回数 + 1**
（`runs_detail.json` の `n_cells` は移動回数なので +1 する）。
`per_maze_metrics` がその補正込みで計算する。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/compare_eprime.py
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AGG_PATH = REPO_ROOT / "experiments" / "exp_013_band_v4_reeval" / "aggregate.py"


def load_agg():
    """`exp_013/aggregate.py` を**そのまま**読み込む（定義を写さない・R23）。"""
    spec = importlib.util.spec_from_file_location("exp013_aggregate", AGG_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp013_aggregate"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="l0c", help="対照（素の L0-c・安全率 0.70）")
    ap.add_argument("--new", default="l0c_sf0.75_eprime",
                    help="処理（素の L0-c・安全率 0.75）")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016g" / "eprime_check.json"))
    args = ap.parse_args()

    agg = load_agg()
    rows = {}
    for tag, arm in (("base", args.base), ("new", args.new)):
        data, err = agg.load_arm(arm)
        if err:
            raise SystemExit(err)
        rows[tag] = dict(policy=data["policy"], maze_dir=data["maze_dir"],
                         m={mz: agg.per_maze_metrics(mz, rec)
                            for mz, rec in data["mazes"].items()})
        print(f"{tag:5s} = {arm:22s} {data['policy']}  （{data['maze_dir']}）")

    a, b = rows["base"]["m"], rows["new"]["m"]
    common = sorted(set(a) & set(b))
    print(f"\n共通の迷路 {len(common)} 件\n")

    print(f"{'迷路':<12}{'D_true':>7}{'(e′) 0.70':>11}{'(e′) 0.75':>11}{'差':>9}"
          f"{'超過区画 0.70':>14}{'超過区画 0.75':>14}  (a)(b)(c)")
    diffs, per = [], []
    n_abc_same = n_ep_same = 0
    for mz in common:
        ra, rb = a[mz], b[mz]
        ea, eb = ra["e_prime"], rb["e_prime"]
        d = None if (ea is None or eb is None) else (eb - ea)
        if d is not None:
            diffs.append(d)
            n_ep_same += (d == 0.0)
        abc_same = all(ra[k] == rb[k] for k in ("a", "b", "c"))
        n_abc_same += abc_same
        print(f"{mz:<12}{ra['d_true']:>7}"
              f"{('%.4f' % ea) if ea is not None else '  —   ':>11}"
              f"{('%.4f' % eb) if eb is not None else '  —   ':>11}"
              f"{('%+.4f' % d) if d is not None else '  —   ':>9}"
              f"{str(ra['excess_cells']):>14}{str(rb['excess_cells']):>14}"
              f"  {'一致' if abc_same else '**不一致**'}")
        per.append(dict(maze=mz, d_true=ra["d_true"], e_prime_base=ea, e_prime_new=eb,
                        diff=d, excess_base=ra["excess_cells"], excess_new=rb["excess_cells"],
                        abc_same=abc_same,
                        a=[ra["a"], rb["a"]], b=[ra["b"], rb["b"]], c=[ra["c"], rb["c"]]))

    print(f"\n【判定】")
    print(f"  (a) ゴール到達・(b) 最短走行の成立・(c) 最短走行が有効: "
          f"**{n_abc_same} / {len(common)} 迷路で一致**")
    if diffs:
        print(f"  **(e′) の同じ迷路どうしの差: 中央値 {np.median(diffs):+.4f}"
              f"（{min(diffs):+.4f}〜{max(diffs):+.4f}）**")
        print(f"  **(e′) が 1 ビットも動かなかった迷路: {n_ep_same} / {len(diffs)}**")
        ep_a = [a[m]["e_prime"] for m in common if a[m]["e_prime"] is not None]
        ep_b = [b[m]["e_prime"] for m in common if b[m]["e_prime"] is not None]
        print(f"  (e′) 中央値: 0.70 で {np.median(ep_a):.4f} → 0.75 で {np.median(ep_b):.4f}")
        if n_ep_same == len(diffs):
            print("\n  ⇒ **経路効率は動いていない。**"
                  "「速度計画は経路を変えない」が実測で確かめられた")
        else:
            print("\n  ⇒ 🔴 **経路効率が動いた。**"
                  "「速度計画が経路を変えた」ことの発見であり、別の問題として扱う")
    else:
        print("  **(e′) が定義された迷路が無い**（判定不能）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(base_arm=args.base, new_arm=args.new,
                   base_policy=rows["base"]["policy"], new_policy=rows["new"]["policy"],
                   maze_dir=rows["new"]["maze_dir"], n_common=len(common),
                   n_abc_same=n_abc_same, n_e_prime_same=n_ep_same,
                   e_prime_diff_median=float(np.median(diffs)) if diffs else None,
                   per_maze=per),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
