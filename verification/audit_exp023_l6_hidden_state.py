#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
監査 048 の L6: **評価時に隠れ状態が走行をまたいで持ち越されていないか**（exp_023）

准教授セッション（9 代目）・2026-08-15
**判定形は `verification/AUDIT_048_PREREG_exp023_verdict.md` §2 に、
exp_023 の結果を 1 件も見ていない時点でコミット済み（`8ac0994`）。**

## なぜこの検査が要るか（`AUDIT_048_PREREG` §2）

**再帰型方策は内部状態を持つ。**評価の配管が試行ごとに隠れ状態を捨てていなければ、
**120 走行すべてが「前の迷路の記憶」で汚染される。**
**例外も出ず、観測の次元も合い、値だけが誤る。**
**これは判定量の計算ではなく判定量の入力そのものを壊す欠陥であり、
数値の照合をいくら重ねても検出できない。**

## 独立なのはどの層か（**作法 36** — 「独立再計算をした」と書くときは層を名指しする）

- **本スクリプトは `measure_driving.py` の `_run_episode` をそのまま呼ぶ。**
  **実装を書き直さない** — 検査したいのは**その配管そのもの**だからである。
- **独立なのは実装ではなく実験の design である**:
  **(A) 迷路の提示順を変える**・**(B) リセットを外した否定対照を置く**。
  **どちらも学生B の T-R3（同じ迷路を 2 回続けて走らせる bit 一致）では検出できない**
  — T-R3 は「同じ順序で同じ結果」＝ **再現性**の検査であって、**順序依存の検出には弱い**。

## 検査（**空振りしない形にする**）

| # | 内容 | 合格 |
|---|---|---|
| **L6-a** | **提示順を変えて測り直す**（リセットあり ＝ 本番の配管） | **迷路ごとの判定量が全件 bit 一致** |
| **L6-b** | **否定対照: リセットを外して同じことをする** | **🔴 少なくとも 1 件は食い違う**（食い違わなければ L6-a は空振りで、何も証明していない） |

**L6-b が要る理由**: **隠れ状態が判定量に効かないなら、リセットの有無に関わらず L6-a は通る。**
**そのとき L6-a の合格は「配管が正しい」ではなく「この検査に判別力が無い」を意味する。**

## 使い方

    .venv/bin/python verification/audit_exp023_l6_hidden_state.py \
        --model models/exp_023a_seed1_800k.zip --n-maze 5

**CPU の注意**: 学習 6 本が走っている間は迷路数を絞ること（既定 5）。
"""

import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp_021_observation_history"))

import measure_driving as MD  # noqa: E402

# exp_023 群 1 の観測は exp_021 と同じ 8 つの遅れ（カード §2-1）
LAGS = [1, 2, 4, 8, 16, 32, 64, 128]

# 判定量に効く項目だけを比べる（乱数の内部状態などは比べない）
KEYS = ("d0", "min_d", "outcome", "n_steps", "n_respawn", "cell_entries")


def episode_key(ep):
    """1 走行の結果を、比較できる形へ（毎歩の記録も畳んで入れる）。"""
    m = MD._episode_metrics(ep)
    return {
        **{k: ep[k] for k in ("d0", "outcome", "n_steps", "n_respawn", "cell_entries")},
        "min_d": m["min_d"],
        "net_progress_per_1000": m["net_progress_per_1000"],
        "respawn_per_1000": m["respawn_per_1000"],
        # 毎歩の列そのもの（畳まずに比べたいので tuple 化）
        "d_hist": tuple(ep["d_hist"]),
        "resp_hist": tuple(ep["resp_hist"]),
    }


def run_sequence(policy_fn, maze_dir, seeds, reset_between):
    """迷路を `seeds` の順に走らせ、迷路 seed → 結果 の辞書を返す。

    reset_between=False は**否定対照**（本番の配管ではない）。

    🔴 **どちらの条件でも、列の先頭で必ず 1 回リセットする。**
    そうしないと、否定対照の 2 本（順 A・順 B）が
    **「順序が違う」ことに加えて「列に入る前の隠れ状態が違う」ことでも変わってしまい、
    食い違いの原因を順序に帰属できない。**
    先頭でだけ揃えれば、**2 本の唯一の違いが提示順になる。**
    """
    out = {}
    policy_fn.reset()                      # 列の先頭で揃える（両条件で共通）
    saved = policy_fn.reset
    if not reset_between:
        policy_fn.reset = lambda: None     # 迷路の切り替わりでのリセットだけを殺す
    try:
        for s in seeds:
            out[s] = episode_key(MD._run_episode(maze_dir, s, policy_fn, LAGS))
    finally:
        policy_fn.reset = saved
    return out


def diff_count(a, b):
    """迷路ごとに食い違った件数と、最初の食い違いの中身を返す。"""
    n_diff, first = 0, None
    for s in a:
        if a[s] != b[s]:
            n_diff += 1
            if first is None:
                first = (s, {k: (a[s][k], b[s][k]) for k in a[s]
                             if a[s][k] != b[s][k] and k not in ("d_hist", "resp_hist")})
    return n_diff, first


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="再帰型方策の .zip")
    ap.add_argument("--n-maze", type=int, default=5,
                    help="使う迷路の数（学習中は絞る。既定 5）")
    ap.add_argument("--maze-dir", type=str, default=str(MD.VALIDATION_MAZE_DIR))
    ap.add_argument("--shuffle-seed", type=int, default=20260815,
                    help="提示順の並べ替えの seed（迷路の中身には影響しない）")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    from pathlib import Path
    maze_dir = Path(args.maze_dir)

    # 検証用の 20 迷路（seed 7000〜7019）から先頭 n 本
    seeds = list(range(7000, 7000 + args.n_maze))
    order_b = seeds[:]
    random.Random(args.shuffle_seed).shuffle(order_b)
    if order_b == seeds:
        print("🔴 並べ替えが恒等になった。--shuffle-seed を変えること")
        return 2

    print("=" * 78)
    print("AUDIT_048 L6: 評価時の隠れ状態が走行をまたいでいないか")
    print("=" * 78)
    print(f"方策: {args.model}")
    print(f"迷路: {seeds}（検証用の 20 迷路〔seed 7000〜7019〕の先頭 {args.n_maze} 本）")
    print(f"順 A: {seeds}")
    print(f"順 B: {order_b}")

    policy_fn, nsteps = MD._load_policy(Path(args.model), recurrent=True)
    print(f"方策の実歩数: {nsteps:,}")

    print("\n--- L6-a: 本番の配管（リセットあり）で順 A と順 B ---")
    a1 = run_sequence(policy_fn, maze_dir, seeds, reset_between=True)
    a2 = run_sequence(policy_fn, maze_dir, order_b, reset_between=True)
    n_diff_a, first_a = diff_count(a1, a2)
    ok_a = (n_diff_a == 0)
    print(f"  食い違い {n_diff_a} / {len(seeds)} 件 → "
          f"{'🟢 合格（順序に依存しない）' if ok_a else '🔴 不合格'}")
    if first_a:
        print(f"  最初の食い違い: {first_a}")

    print("\n--- L6-b: 否定対照（迷路の切り替わりでのリセットを外す）で順 A と順 B ---")
    b1 = run_sequence(policy_fn, maze_dir, seeds, reset_between=False)
    b2 = run_sequence(policy_fn, maze_dir, order_b, reset_between=False)
    n_diff_b, first_b = diff_count(b1, b2)

    # 🔴 「前の迷路」が両方の順で同じ迷路は、リセットを外しても入る隠れ状態が同じなので
    #    構成上一致する。判別力を測る母集団は「前の迷路が変わった迷路」だけである。
    def prev_of(order):
        return {s: (order[i - 1] if i > 0 else None) for i, s in enumerate(order)}
    pa, pb = prev_of(seeds), prev_of(order_b)
    changed = [s for s in seeds if pa[s] != pb[s]]
    n_diff_changed = sum(1 for s in changed if b1[s] != b2[s])
    ok_b = (n_diff_changed > 0)
    print(f"  食い違い {n_diff_b} / {len(seeds)} 件（全体）")
    print(f"  **前の迷路が変わった {len(changed)} 件のうち {n_diff_changed} 件が食い違い** → "
          f"{'🟢 合格（検査に判別力がある）' if ok_b else '🔴 空振り'}")
    print(f"  （前の迷路が両方の順で同じ迷路は、構成上一致するので母集団から外す）")
    if first_b:
        print(f"  最初の食い違い: {first_b}")

    print("\n" + "=" * 78)
    if ok_a and ok_b:
        print("🟢 L6 合格: 配管は試行ごとに隠れ状態を捨てており、")
        print("   かつ捨てなければ結果が変わる（＝ この検査は空振りしていない）。")
    elif ok_a and not ok_b:
        print("🟠 L6 判定保留: 順序に依存しないが、リセットを外しても変わらない。")
        print("   **この方策ではこの検査に判別力が無い**（隠れ状態が判定量に効いていない）。")
        print("   「配管が正しい」とは書けない。未確認として報告すること。")
    else:
        print("🔴 L6 不合格: 判定量が迷路の提示順に依存している。")
        print("   **120 走行すべてが前の迷路の記憶で汚染されている疑い。**")
        print("   判定へ進む前に教授へ即報告すること。")
    print("=" * 78)

    if args.out:
        json.dump({"model": args.model, "num_timesteps": nsteps,
                   "order_a": seeds, "order_b": order_b,
                   "L6a_diff": n_diff_a, "L6b_diff": n_diff_b,
                   "L6b_prev_changed": len(changed),
                   "L6b_diff_among_changed": n_diff_changed,
                   "verdict": "pass" if (ok_a and ok_b) else
                              ("no_power" if ok_a else "fail")},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n保存: {args.out}")

    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
