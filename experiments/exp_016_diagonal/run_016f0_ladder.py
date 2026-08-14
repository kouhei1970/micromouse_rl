#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-F0 の判定 — **016-C の梯子を、速度ループの是正つきで再実行する**。

カード `card_016f0.md` §5:「**判定の harness は 016-C の `run_016c.py`
（梯子・面の選定・計装を変えない）**」。

そこで**本スクリプトは harness を書き直さない**。`run_016c` を import し、
**方策クラスの名前だけを差し替えて `run_016c.main()` をそのまま呼ぶ**
（裁定 R23「集計器は `importlib` で先行実験の定義を再利用する。コピー禁止」と同じ思想）。

    run_016c.SegSpeedPolicy = <VelocityLoopMixin を混ぜたもの>

**梯子 `LADDER`・面の選定・余裕 $m$ の計算・A-成立の判定は 016-C のまま**である。

使い方:
    # 是正後（物理から導いた全量）
    .venv/bin/python -u experiments/exp_016_diagonal/run_016f0_ladder.py --k-acc-ff 1.0
    # 是正前（基準スナップショットの再現。差分 0 になるはず）
    .venv/bin/python -u experiments/exp_016_diagonal/run_016f0_ladder.py --k-acc-ff 0.0
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
from competition.velocity_loop import VelocityLoopMixin  # noqa: E402


def make_policy_class(k_acc_ff: float):
    """016-C の方策に**速度ループの是正だけ**を混ぜ込んだクラスを作る。"""

    class SegSpeedPolicyF0(VelocityLoopMixin, run_016c.SegSpeedPolicy):
        """`_wheel_targets_to_voltage` だけが違う。**経路・速度計画・操舵は 016-C のまま。**"""

        def __init__(self, *a, **kw):
            kw.setdefault("k_acc_ff", k_acc_ff)
            super().__init__(*a, **kw)

    return SegSpeedPolicyF0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k-acc-ff", type=float, required=True,
                    help="加速度前置補償の係数（0 = 是正前・1.0 = 物理から導いた全量）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    args = ap.parse_args()

    out = args.out or str(REPO_ROOT / "outputs" / "exp_016_diagonal" / "016f0"
                          / f"ladder_k{args.k_acc_ff:g}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print(f"⚠️ 速度ループの是正: k_acc_ff = {args.k_acc_ff:g}"
          f"（0 = 是正前・1.0 = 物理から導いた全量）\n")
    # **harness は書き換えず、方策の名前だけ差し替える**
    run_016c.SegSpeedPolicy = make_policy_class(args.k_acc_ff)
    sys.argv = ["run_016c.py", "--maze-dir", args.maze_dir, "--out", out]
    run_016c.main()


if __name__ == "__main__":
    main()
