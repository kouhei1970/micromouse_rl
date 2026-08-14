#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**是正 (B) の無害性と作用の確認**（`card_016diag_fixB.md` §3 の手順 2・3）。

--------------------------------------------------------------------------
⚠️ 使う迷路 — **判定に使う 20 迷路の【外】**
--------------------------------------------------------------------------
**校正用の 60 迷路（seed 51000〜）の先頭数件を使う。**

**空振り検査の一般規則**（教授裁定 2026-08-14・`card_016diag_switch.md` §4-0）:

> **① 判定に使う迷路集合の【外】の迷路で行い、
> ② 出力は「相違の有無と件数」に限る（判定量の値そのものは印字しない）。**

**本スクリプトは走行タイムなどの判定量を 1 つも印字しない。**
**印字するのは「一致したか」「何件違ったか」だけである。**
**調整用迷路（seed 41000〜。判定に使う集合）には触れない。**

--------------------------------------------------------------------------
確かめること
--------------------------------------------------------------------------
| # | 検査 | 期待 |
|---|---|---|
| **無害性 (a)** | `align_check=False` が**是正前と全走行でビット単位で一致** | 相違 0 件 |
| **無害性 (b)** | **向きが一致している間は、是正前と挙動が変わらない** | **最初の相違が、最初の「委ね」より後**（教授指示） |
| **空振り検査** | `align_check=True` が**是正前と違う** | 相違が 1 件以上（同じなら検査が空振り） |

使い方:
    .venv/bin/python -u experiments/exp_016_diagonal/check_diag_fixb.py --n-mazes 3
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
from competition.baseline_slalom_diag_cal import SlalomDiagCalPolicy  # noqa: E402
from competition.baseline_slalom_diag_cal_fixb import (  # noqa: E402
    SlalomDiagCalFixBPolicy)
from competition.evaluator import CompetitionEvaluator  # noqa: E402
from geometry import git_rev  # noqa: E402

RUN_FIELDS = ("index", "outcome", "run_time", "t_start", "t_end",
              "path_length_m", "visited_cells")


class Probe:
    """方策を包んで、制御周期ごとの左右のモータ電圧を記録する（**読むだけ**）。"""

    def __init__(self, inner):
        self._inner = inner
        self.volts = []
        self.defer_tick = None      # 最初に「親へ委ねた」ティック

    name = property(lambda self: getattr(self._inner, "name", "unnamed"))
    requires_privileged = property(
        lambda self: getattr(self._inner, "requires_privileged", False))

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def act(self, obs):
        n0 = getattr(self._inner, "n_align_defer", 0)
        out = self._inner.act(obs)
        self.volts.append((float(out[0]), float(out[1])))
        if (self.defer_tick is None
                and getattr(self._inner, "n_align_defer", 0) > n0):
            self.defer_tick = len(self.volts)
        return out


def run(cls, kw, maze, maze_dir, out_dir):
    pol = Probe(cls(**kw))
    ev = CompetitionEvaluator(maze_dir=maze_dir, out_dir=str(out_dir))
    r = ev.evaluate_maze(maze, pol)
    return dict(runs=[{k: q.get(k) for k in RUN_FIELDS} for q in r["runs"]],
                best=r.get("best_time"),
                volts=np.asarray(pol.volts, dtype=float),
                defer_tick=pol.defer_tick,
                n_defer=getattr(pol._inner, "n_align_defer", None),
                n_ok=getattr(pol._inner, "n_align_ok", None))


def n_run_diffs(a, b):
    """走行ごとの記録の不一致の**件数だけ**を返す（値は返さない）。"""
    n = 0
    if a["best"] != b["best"]:
        n += 1
    if len(a["runs"]) != len(b["runs"]):
        return n + 1
    for qa, qb in zip(a["runs"], b["runs"]):
        n += sum(1 for k in RUN_FIELDS if qa.get(k) != qb.get(k))
    return n


def first_volt_divergence(a, b):
    """左右の電圧の列が最初に食い違うティック（1 始まり）。一致なら None。"""
    va, vb = a["volts"], b["volts"]
    m = min(len(va), len(vb))
    d = np.any(va[:m] != vb[:m], axis=1)
    idx = np.flatnonzero(d)
    if idx.size:
        return int(idx[0]) + 1
    return None if len(va) == len(vb) else m + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-mazes", type=int, default=3)
    ap.add_argument("--maze-dir", default="competition/mazes/cal_v4",
                    help="**判定に使う 20 迷路の外**であること")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016diag_fixb" / "check.json"))
    args = ap.parse_args()

    assert "design_v4" not in args.maze_dir, \
        "判定に使う迷路集合（調整用迷路 seed 41000〜）は使えない（空振り検査の一般規則 ①）"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))[:args.n_mazes]
    seeds = [int(q.stem.split("_")[1]) for q in mazes]
    print(describe_seeds(seeds, "competition"))
    assert_seeds_allowed(seeds, namespace="competition", purpose="validate")
    print(f"迷路 {len(mazes)} 件 / {args.maze_dir}"
          f"（**判定に使う 20 迷路の外**）\n")

    rows = []
    for m in mazes:
        base = run(SlalomDiagCalPolicy, {}, m, args.maze_dir, out.parent / "t_base")
        off = run(SlalomDiagCalFixBPolicy, dict(align_check=False), m,
                  args.maze_dir, out.parent / "t_off")
        on = run(SlalomDiagCalFixBPolicy, dict(align_check=True), m,
                 args.maze_dir, out.parent / "t_on")
        rows.append(dict(
            maze=m.stem,
            a_run_diffs=n_run_diffs(base, off),
            a_volt_diverge=first_volt_divergence(base, off),
            b_first_diverge=first_volt_divergence(base, on),
            b_first_defer=on["defer_tick"],
            c_run_diffs=n_run_diffs(base, on),
            n_defer=on["n_defer"], n_ok=on["n_ok"]))
        q = rows[-1]
        print(f"  {q['maze']}: 無害性(a) 相違 {q['a_run_diffs']} 件"
              f"／向きが揃って斜めへ入った {q['n_ok']} 回・親へ委ねた {q['n_defer']} 回", flush=True)

    print("\n" + "=" * 68)
    print("【無害性 (a)】align_check=False は是正前と一致するはず")
    print("=" * 68)
    na = sum(q["a_run_diffs"] for q in rows)
    nv = sum(1 for q in rows if q["a_volt_diverge"] is not None)
    print(f"  走行ごとの記録の相違 合計 {na} 件／電圧の列が食い違った迷路 {nv} 件")
    ok_a = (na == 0 and nv == 0)
    print(f"  → {'✅ 一致' if ok_a else '🔴 **一致しない**'}")

    print("\n" + "=" * 68)
    print("【無害性 (b)】向きが揃っている間は是正前と同じはず")
    print("  （**最初の相違が、最初の「委ね」以降**であること）")
    print("=" * 68)
    ok_b = True
    for q in rows:
        d, f = q["b_first_diverge"], q["b_first_defer"]
        if f is None:
            verdict = "委ねが 1 度も起きていない（この迷路では判定できない）"
            good = (d is None)
        else:
            good = (d is None) or (d >= f)
            verdict = ("最初の相違は委ね以降" if good
                       else "🔴 **委ねより前に食い違っている**")
        ok_b &= good
        print(f"  {q['maze']}: 最初の相違 {d}／最初の委ね {f} … {verdict}")
    print(f"  → {'✅ 一致ケースの挙動は変わっていない' if ok_b else '🔴 **変わっている**'}")

    print("\n" + "=" * 68)
    print("【空振り検査】align_check=True は是正前と違うはず（値は印字しない）")
    print("=" * 68)
    nc = sum(q["c_run_diffs"] for q in rows)
    print(f"  走行ごとの記録の相違 合計 {nc} 件"
          f"／親へ委ねた回数の合計 {sum(q['n_defer'] or 0 for q in rows)} 回")
    ok_c = nc > 0
    print(f"  → {'✅ 是正が効いている' if ok_c else '🔴 **同じ。検査が空振り**'}")

    json.dump(dict(git_rev=git_rev(), maze_dir=args.maze_dir, n_mazes=len(mazes),
                   harmless_a=ok_a, harmless_b=ok_b, not_vacuous=ok_c, rows=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(f"\n書き出し: {out}")
    return 0 if (ok_a and ok_b and ok_c) else 1


if __name__ == "__main__":
    sys.exit(main())
