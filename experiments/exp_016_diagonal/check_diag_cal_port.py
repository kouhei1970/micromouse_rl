#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**斜め方策の載せ替えの無害性を、走らせて確かめる**（`card_016diag_switch.md` §2-2）。

`tests/test_diag_cal_port.py` が構造（積み上げ・差し替えの後始末・経路のビット一致）を
押さえているのに対し、**こちらは実際に走らせて走行の記録が一致することを見る**。

--------------------------------------------------------------------------
何を比べるのか
--------------------------------------------------------------------------
| 条件 | 中身 | 期待 |
|---|---|---|
| **(A) 対照** | `SlalomE1TTRF0bCalPolicy`（斜めなしの新既定） | — |
| **(B) 無害性** | `SlalomDiagCalPolicy(L_c=0)` ＋ **斜めを無効**にしたもの | **(A) と完全一致** |
| **(C) 空振り検査** | `SlalomDiagCalPolicy`（既定。斜めあり・クロソイドあり） | **(A) と違うこと** |

**(B) が (A) と一致すれば、「混ぜ込みを足したこと自体」は何も変えていない**と言える。
**(C) が (A) と違うことまで見ないと、検査が空振りしていても気づけない**
（**016-G の D1 で「構成上必ず成立する判定量」を登録した反省**。
`card_016g.md` §7-6 の教訓 1）。

⚠️ **本スクリプトは調整用迷路（seed 41000〜）だけを使う。**
**確保済みの評価用 20 迷路には触れない。**

使い方:
    .venv/bin/python -u experiments/exp_016_diagonal/check_diag_cal_port.py --n-mazes 3
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal",
          REPO_ROOT / "experiments" / "exp_015_time_optimal_route"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402
from competition.baseline_slalom_diag_cal import SlalomDiagCalPolicy  # noqa: E402
from competition.baseline_slalom_e1t_tr_f0b_cal import (  # noqa: E402
    SlalomE1TTRF0bCalPolicy)
from competition.evaluator import CompetitionEvaluator  # noqa: E402
from geometry import git_rev  # noqa: E402

# 突き合わせる走行ごとの項目（**評価器が返すものを全部**）
RUN_FIELDS = ("index", "outcome", "run_time", "t_start", "t_end",
              "path_length_m", "visited_cells", "path_efficiency")


class DiagDisabled(SlalomDiagCalPolicy):
    """**斜めを無効にした載せ替え版**（無害性の確認用）。

    `_use_diag()` が偽なら親の `_replan` へそのまま委ねるので、
    **斜めなしの新既定とまったく同じ経路・同じ制御になるはず**である。
    """

    name = "diag-disabled port (harmlessness check)"

    def __init__(self, *a, **kw):
        kw.setdefault("L_c", 0.0)
        super().__init__(*a, **kw)

    def _use_diag(self) -> bool:
        return False


def run_one(policy_factory, maze, maze_dir, out_dir):
    ev = CompetitionEvaluator(maze_dir=maze_dir, out_dir=str(out_dir))
    r = ev.evaluate_maze(maze, policy_factory())
    return dict(maze=r["maze_id"], best_time=r.get("best_time"),
                success=r.get("success"),
                runs=[{k: q.get(k) for k in RUN_FIELDS} for q in r["runs"]])


def compare(a, b):
    """2 条件の記録を突き合わせ、不一致の件数と中身を返す。"""
    diffs = []
    if a["best_time"] != b["best_time"]:
        diffs.append(("best_time", a["best_time"], b["best_time"]))
    if len(a["runs"]) != len(b["runs"]):
        diffs.append(("n_runs", len(a["runs"]), len(b["runs"])))
        return diffs
    for qa, qb in zip(a["runs"], b["runs"]):
        for k in RUN_FIELDS:
            if qa.get(k) != qb.get(k):
                diffs.append((f"run{qa.get('index')}.{k}", qa.get(k), qb.get(k)))
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-mazes", type=int, default=3, help="使う迷路の数（先頭から）")
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016diag_switch" / "harmlessness.json"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))[:args.n_mazes]
    seeds = [int(q.stem.split("_")[1]) for q in mazes]
    print(describe_seeds(seeds, "competition"))
    assert_seeds_allowed(seeds, namespace="competition", purpose="validate")
    print(f"迷路 {len(mazes)} 件 / {args.maze_dir}\n")

    conds = (("A_control", SlalomE1TTRF0bCalPolicy, "対照（斜めなしの新既定）"),
             ("B_disabled", DiagDisabled, "載せ替え版・斜め無効・L_c=0"),
             ("C_enabled", SlalomDiagCalPolicy, "載せ替え版・既定（斜めあり・クロソイドあり）"))
    rec = {}
    for tag, cls, label in conds:
        rec[tag] = []
        print(f"--- {tag}: {label} ---")
        for m in mazes:
            r = run_one(cls, m, args.maze_dir, out.parent / f"traj_{tag}")
            rec[tag].append(r)
            print(f"  {r['maze']}: 走行 {len(r['runs'])} 本"
                  f"／(d) 最速 {r['best_time']}", flush=True)

    print("\n" + "=" * 70)
    print("【無害性】(B) 斜め無効・L_c=0  対  (A) 対照 — **完全一致するはず**")
    print("=" * 70)
    n_bad = 0
    for ra, rb in zip(rec["A_control"], rec["B_disabled"]):
        d = compare(ra, rb)
        n_bad += len(d)
        print(f"  {ra['maze']}: 不一致 {len(d)} 件"
              + ("" if not d else "  " + "／".join(f"{k}: {x} → {y}" for k, x, y in d[:4])))
    verdict_b = (n_bad == 0)
    print(f"\n  **不一致の合計 {n_bad} 件 → "
          f"{'✅ 無害性を確認' if verdict_b else '🔴 **不一致がある。載せ替えが挙動を変えている**'}**")

    print("\n" + "=" * 70)
    print("【空振り検査】(C) 既定  対  (A) 対照 — **違うはず**（同じなら検査が空振り）")
    print("=" * 70)
    n_diff = 0
    for ra, rc in zip(rec["A_control"], rec["C_enabled"]):
        d = compare(ra, rc)
        n_diff += len(d)
        print(f"  {ra['maze']}: 相違 {len(d)} 件"
              f"／(d) 最速 {ra['best_time']} → {rc['best_time']}")
    verdict_c = (n_diff > 0)
    print(f"\n  **相違の合計 {n_diff} 件 → "
          f"{'✅ 斜めとクロソイドが効いている' if verdict_c else '🔴 **同じ。検査が空振りしている**'}**")

    json.dump(dict(git_rev=git_rev(), maze_dir=args.maze_dir, n_mazes=len(mazes),
                   harmless=verdict_b, n_mismatch=n_bad,
                   not_vacuous=verdict_c, n_diff=n_diff, records=rec),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n書き出し: {out}")
    return 0 if (verdict_b and verdict_c) else 1


if __name__ == "__main__":
    sys.exit(main())
