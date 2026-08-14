#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**是正 (A) の単独検証と無害性**（`card_016diag_fixA.md` §3 の手順 2〜4）。

--------------------------------------------------------------------------
⚠️ 使う迷路 — **判定に使う 20 迷路の【外】**
--------------------------------------------------------------------------
**校正用の 60 迷路（seed 51000〜）を使う。**
**空振り検査の一般規則**（`card_016diag_switch.md` §4-0）に従い、
**判定量の値そのものは印字しない**（印字は「一致したか」「何件違ったか」だけ）。

--------------------------------------------------------------------------
確かめること
--------------------------------------------------------------------------
| # | 検査 | 期待 |
|---|---|---|
| **A1** | **人工的に係員回収を起こした直後、印が持続するか** | **(A) 版は真のまま／(B) 版は偽になる**（**同じ検査で両方**＝ 検査の検出力の実証） |
| **A2** | **回収が起きない走行では (B) 版とビット単位で一致** | 全項目・全ティック一致 |
| **空振り** | `use_maze_flag=True` が (B) 版と違う | 相違が 1 件以上 |

使い方:
    .venv/bin/python -u experiments/exp_016_diagonal/check_diag_fixa.py --n-mazes 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal",
          REPO_ROOT / "experiments" / "exp_015_time_optimal_route"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402
from competition.baseline_slalom_diag_cal_fixab import (  # noqa: E402
    SlalomDiagCalFixABPolicy)
from competition.baseline_slalom_diag_cal_fixb import (  # noqa: E402
    SlalomDiagCalFixBPolicy)
from competition.evaluator import CompetitionEvaluator  # noqa: E402
from geometry import git_rev  # noqa: E402

RUN_FIELDS = ("index", "outcome", "run_time", "t_start", "t_end",
              "path_length_m", "visited_cells")


class Probe:
    """方策を包んで電圧と回収の回数を記録する（**読むだけ**）。"""

    def __init__(self, inner):
        self._inner = inner
        self.volts = []
        self.n_retrieval = 0

    name = property(lambda self: getattr(self._inner, "name", "unnamed"))
    requires_privileged = property(
        lambda self: getattr(self._inner, "requires_privileged", False))

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def on_retrieval(self):
        self.n_retrieval += 1
        return self._inner.on_retrieval()

    def act(self, obs):
        out = self._inner.act(obs)
        self.volts.append((float(out[0]), float(out[1])))
        return out


# ==========================================================================
# A1 — **人工的に係員回収を起こす**（単独の検証）
# ==========================================================================
def a1_probe_flag(cls, kw, maze, maze_dir, out_dir):
    """ゴール到達後に `on_retrieval()` を 1 回呼び、`_use_diag()` の成立を見る。

    **走行を最後まで回さない**（ゴール到達を検出した時点で観測する）。
    観測するのは**方策の内部状態だけ**で、走行タイム等の判定量は読まない。
    """
    pol = cls(**kw)
    ev = CompetitionEvaluator(maze_dir=maze_dir, out_dir=str(out_dir))
    seen = {}

    orig_flip = pol._flip_target_mode

    def flip_spy():
        was_goal = (pol.target_mode == "to_goal")
        orig_flip()
        if was_goal and "before" not in seen:
            # 初回ゴール到達の直後 = 印が立ったはずの時点
            seen["before"] = dict(
                maze_explored=bool(getattr(pol, "_maze_explored", None) or False),
                explored_once=bool(pol._explored_once))
            # ★ **人工的に係員回収を起こす**
            pol.on_retrieval()
            pol.target_mode = "to_goal"     # 回収後はスタートから再びゴールを目指す
            seen["after"] = dict(
                maze_explored=bool(getattr(pol, "_maze_explored", None) or False),
                explored_once=bool(pol._explored_once),
                use_diag=bool(pol._use_diag()))
    pol._flip_target_mode = flip_spy

    try:
        ev.evaluate_maze(maze, pol)
    except Exception:
        pass
    return seen


# ==========================================================================
def run(cls, kw, maze, maze_dir, out_dir):
    pol = Probe(cls(**kw))
    ev = CompetitionEvaluator(maze_dir=maze_dir, out_dir=str(out_dir))
    r = ev.evaluate_maze(maze, pol)
    return dict(runs=[{k: q.get(k) for k in RUN_FIELDS} for q in r["runs"]],
                best=r.get("best_time"), volts=np.asarray(pol.volts, dtype=float),
                n_retrieval=pol.n_retrieval)


def n_run_diffs(a, b):
    n = 0
    if a["best"] != b["best"]:
        n += 1
    if len(a["runs"]) != len(b["runs"]):
        return n + 1
    for qa, qb in zip(a["runs"], b["runs"]):
        n += sum(1 for k in RUN_FIELDS if qa.get(k) != qb.get(k))
    return n


def volts_identical(a, b):
    va, vb = a["volts"], b["volts"]
    return len(va) == len(vb) and bool(np.array_equal(va, vb))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-mazes", type=int, default=3)
    ap.add_argument("--vacuous-maze", default="maze_51032",
                    help="**空振り検査に使う迷路**。**係員回収が起きる迷路でないと、"
                         "(A) と (B) が同じで当然になり検査が無情報になる**。"
                         "既定は校正用の 60 迷路を走査して見つけたもの（回収 2 回）")
    ap.add_argument("--maze-dir", default="competition/mazes/cal_v4")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016diag_fixa" / "check.json"))
    args = ap.parse_args()

    assert "design_v4" not in args.maze_dir, \
        "判定に使う迷路集合は使えない（空振り検査の一般規則 ①）"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))[:args.n_mazes]
    seeds = [int(q.stem.split("_")[1]) for q in mazes]
    print(describe_seeds(seeds, "competition"))
    assert_seeds_allowed(seeds, namespace="competition", purpose="validate")
    print(f"迷路 {len(mazes)} 件 / {args.maze_dir}（**判定に使う 20 迷路の外**）\n")

    # ---------------- A1 -------------------------------------------------
    print("=" * 68)
    print("【A1】人工的に係員回収を起こした直後、印は持続するか")
    print("  （(A) 版は持続・**(B) 版は消える**ことを同じ検査で確かめる）")
    print("=" * 68)
    a1_rows, a1_ok = [], True
    for m in mazes:
        sa = a1_probe_flag(SlalomDiagCalFixABPolicy, {}, m, args.maze_dir,
                           out.parent / "a1_A")
        sb = a1_probe_flag(SlalomDiagCalFixBPolicy, {}, m, args.maze_dir,
                           out.parent / "a1_B")
        # (A) 版: 回収後も印が真・_use_diag が成立
        ok_a = bool(sa.get("after", {}).get("maze_explored")) and \
            bool(sa.get("after", {}).get("use_diag"))
        # (B) 版: 同じ操作で _use_diag が成立しない（検査の検出力の実証）
        ok_b = not bool(sb.get("after", {}).get("use_diag", False))
        # どちらも回収で走行水準の状態は落ちているはず
        fell = (sa.get("after", {}).get("explored_once") is False)
        a1_rows.append(dict(maze=m.stem, A_flag_after=sa.get("after"),
                            B_use_diag_after=sb.get("after", {}).get("use_diag"),
                            ok_a=ok_a, ok_b=ok_b, explored_once_fell=fell))
        a1_ok &= (ok_a and ok_b and fell)
        print(f"  {m.stem}: (A) 印が持続 {ok_a}／(A) 走行水準の状態は落ちた {fell}"
              f"／(B) 同じ操作で不成立 {ok_b}")
    print(f"  → {'✅ 印は走行をまたいで持続する（検査の検出力も実証）' if a1_ok else '🔴 **満たさない**'}")

    # ---------------- A2 ＋ 空振り検査 ------------------------------------
    print("\n" + "=" * 68)
    print("【A2】回収が起きない走行では (B) 版とビット単位で一致するか")
    print("【空振り】use_maze_flag=True が (B) 版と違うか（値は印字しない）")
    print("=" * 68)
    a2_ok, rows, n_diff_total = True, [], 0
    for m in mazes:
        b = run(SlalomDiagCalFixBPolicy, {}, m, args.maze_dir, out.parent / "t_b")
        off = run(SlalomDiagCalFixABPolicy, dict(use_maze_flag=False), m,
                  args.maze_dir, out.parent / "t_off")
        on = run(SlalomDiagCalFixABPolicy, dict(use_maze_flag=True), m,
                 args.maze_dir, out.parent / "t_on")
        nd_off = n_run_diffs(b, off)
        id_off = volts_identical(b, off)
        nd_on = n_run_diffs(b, on)
        no_retr = (b["n_retrieval"] == 0)
        # A2 の対象は「回収が起きない迷路」（カード §2 の測定条件）
        a2_target = no_retr
        a2_hit = (nd_on == 0 and volts_identical(b, on)) if a2_target else None
        if a2_target:
            a2_ok &= bool(a2_hit)
        n_diff_total += nd_on
        rows.append(dict(maze=m.stem, n_retrieval_b=b["n_retrieval"],
                         off_run_diffs=nd_off, off_volts_identical=id_off,
                         on_run_diffs=nd_on, a2_target=a2_target, a2_hit=a2_hit))
        print(f"  {m.stem}: (B) 版での回収 {b['n_retrieval']} 回"
              f"／機能オフの相違 {nd_off} 件・電圧一致 {id_off}"
              f"／機能オンの相違 {nd_on} 件"
              + ("  ← A2 の対象" if a2_target else "  （回収あり → A2 の対象外）"))
    off_ok = all(q["off_run_diffs"] == 0 and q["off_volts_identical"] for q in rows)
    print(f"\n  **機能オフ（use_maze_flag=False）が (B) 版と一致**: "
          f"{'✅' if off_ok else '🔴 **一致しない**'}")
    n_t = sum(1 for q in rows if q["a2_target"])
    print(f"  **A2（回収が起きない {n_t} 迷路）**: "
          f"{'✅ ビット単位で一致' if (a2_ok and n_t) else ('🔴 **一致しない**' if n_t else '⚠️ 対象の迷路が無い')}")
    print(f"  （上の {len(rows)} 迷路は**回収が起きない**ので、"
          f"(A) と (B) が同じなのは当然。相違 合計 {n_diff_total} 件）")

    # ---------------- 空振り検査（**回収が起きる迷路で行う**）--------------
    print("\n" + "=" * 68)
    print(f"【空振り検査】**係員回収が起きる迷路**（{args.vacuous_maze}・判定集合の外）で比べる")
    print("  （回収が起きない迷路で比べても無情報になる — 上の表がその実例）")
    print("=" * 68)
    vm = REPO_ROOT / args.maze_dir / f"{args.vacuous_maze}.npz"
    vac_ok, vac = None, {}
    if not vm.exists():
        print(f"  ⚠️ {vm} が無い。**空振り検査を実施できていない**")
    else:
        assert_seeds_allowed([int(vm.stem.split("_")[1])], namespace="competition",
                             purpose="validate")
        vb = run(SlalomDiagCalFixBPolicy, {}, vm, args.maze_dir, out.parent / "v_b")
        va = run(SlalomDiagCalFixABPolicy, {}, vm, args.maze_dir, out.parent / "v_a")
        nd = n_run_diffs(vb, va)
        same = volts_identical(vb, va)
        vac = dict(maze=args.vacuous_maze, n_retrieval_b=vb["n_retrieval"],
                   run_diffs=nd, volts_identical=same)
        vac_ok = (nd > 0) or (not same)
        print(f"  (B) 版での回収 {vb['n_retrieval']} 回"
              f"／走行ごとの記録の相違 **{nd} 件**"
              f"／電圧の列 {'一致' if same else '**食い違う**'}")
        print(f"  → {'✅ 是正が効いている（空振りではない）' if vac_ok else '🔴 **同じ。空振り**'}")

    json.dump(dict(git_rev=git_rev(), maze_dir=args.maze_dir, n_mazes=len(mazes),
                   a1_ok=a1_ok, a1=a1_rows, off_identical=off_ok,
                   a2_ok=a2_ok, a2_n_target=n_t, not_vacuous=vac_ok,
                   vacuous_check=vac, rows=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(f"\n書き出し: {out}")
    return 0 if (a1_ok and off_ok and a2_ok and vac_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
