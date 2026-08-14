#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-G の判定 — **速度引き上げ試験（速度水準を下から順に上げて走らせる一連の
試験）を、45° の遷移にクロソイドを挟んだ経路で再実行する**。

カード `card_016g.md` §3:「判定の harness は 016-C の速度引き上げ試験
（`run_016f0_ladder.py`）。**迷路の選定・計装・主判定の定義を変えない**」。

そこで**本スクリプトは harness を書き直さない**。`run_016f0_ladder` と同じ手口で、
**`run_016c` の中の「参照経路を作る関数」だけを差し替えて `run_016c.main()` を呼ぶ**:

    run_016c.build_diagonal_path = <クロソイド版>

**速度水準の一覧・迷路の選定・余裕の計算・A-成立の判定は 016-C のまま**である。
**方策（速度ループ・操舵）にも一切触らない** — 変えるのは**参照経路の形だけ**。

--------------------------------------------------------------------------
無害性（016-F0 と同じ型）
--------------------------------------------------------------------------
**`--L-c 0` を渡すと `build_clothoid_path` は現行とビット単位で同じ経路を返す**
（`tests/test_clothoid_path.py` が調整用迷路 20 件で検査済み）。
したがって **`--L-c 0` の結果は対照と差分 0 になるはず**であり、
**それを実測で確かめてから本番の L_c を通す**。

使い方:
    # 無害性の確認（対照と差分 0 になるはず）
    .venv/bin/python -u experiments/exp_016_diagonal/run_016g_ladder.py \
        --safety 0.75 --L-c 0
    # 本番
    .venv/bin/python -u experiments/exp_016_diagonal/run_016g_ladder.py \
        --safety 0.75 --L-c 0.04712
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal",
          REPO_ROOT / "experiments" / "exp_015_time_optimal_route"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_016c  # noqa: E402
import run_016f0_ladder  # noqa: E402
from clothoid_path import CLOTHOID_TURNS, build_clothoid_path  # noqa: E402


def make_builder(L_c: float, turns=CLOTHOID_TURNS):
    """`run_016c.run_one` が呼ぶ形（位置引数 4 つ）に合わせた差し替え用の関数。"""

    def builder(nodes, dirs, cell_size, R, **kw):
        kw.setdefault("L_c", L_c)
        kw.setdefault("turns", turns)
        return build_clothoid_path(nodes, dirs, cell_size, R, **kw)

    return builder


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L-c", dest="L_c", type=float, required=True,
                    help="クロソイド 1 本の長さ [m]（**0 なら現行と同じ経路**）")
    ap.add_argument("--safety", type=float, default=0.75,
                    help="旋回安全率（016-cal の校正値 0.75 が既定）")
    ap.add_argument("--k-acc-ff", type=float, default=1.0)
    ap.add_argument("--ref-interp", action="store_true", default=True)
    ap.add_argument("--turns", type=int, nargs="*", default=list(CLOTHOID_TURNS),
                    help="クロソイドを入れる旋回角 [deg]（裁定 (a) により既定は 45 のみ）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    args = ap.parse_args()

    tag = (f"ladder_g_lc{args.L_c*1000:g}mm_sf{args.safety:g}"
           + ("" if tuple(args.turns) == CLOTHOID_TURNS
              else "_t" + "-".join(str(t) for t in args.turns)))
    out = args.out or str(REPO_ROOT / "outputs" / "exp_016_diagonal" / "016g" / f"{tag}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print(f"⚠️ 016-G: クロソイド長 L_c = {args.L_c*1000:g} mm"
          f"／対象の旋回角 {args.turns}°／旋回安全率 {args.safety:g}"
          f"／F0 k_acc_ff = {args.k_acc_ff:g}・F0-b ref_interp = {args.ref_interp}")
    if args.L_c == 0.0:
        print("   （**L_c = 0 は無害性の確認**。対照と差分 0 になるはず）")
    print()

    # **harness も方策も書き換えない。参照経路を作る関数だけを差し替える**
    run_016c.build_diagonal_path = make_builder(args.L_c, tuple(args.turns))
    run_016c.SegSpeedPolicy = run_016f0_ladder.make_policy_class(
        args.k_acc_ff, args.ref_interp, 0.0, 0.0, args.safety)
    sys.argv = ["run_016c.py", "--maze-dir", args.maze_dir, "--out", out]
    run_016c.main()


if __name__ == "__main__":
    main()
