#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1 の確定判定はいつ真になるか — 「帰還が遅い」のか「探索が終わっていない」のか。

**背景**: exp_007 の腕 3（大会実迷路 18 面）で L0-a の帰還時間の中央値が 199.7 s と、
探索走行 141.3 s より長かった。**帰り道が行き道より長い**のは、帰路でも探索を
続けているから（E1 の設計そのもの）である可能性が高い。だとすると問題の本質は
次のどちらかで、**意味がまったく違う**（教授指摘 2026-08-11）:

  (i) **確定に達しているのに、その後も遠回りして帰っている**
      → 「確定後は既知の最短経路を最速で戻る」改良が効く
  (ii) **持ち時間内に確定に達していない**
      → 帰還を速くしても解決しない。**L0-a は大会実迷路で時間が根本的に足りない**、
        が正直な結論になる

**判別に必要なのは 1 つの事実だけ**: `is_shortest_confirmed` が真になる時刻。

測るもの（面ごと）:
- 確定に到達したか／到達時刻 [s] と持ち時間に対する割合 [%]
- 確定時点で何走行目だったか
- 確定後に消費した時間 [s]（＝改良の余地の上限）
- 帰路（`target_mode == "to_start"`）に費やした時間と、そのうち確定後の割合

方策は改造せず、包んで内部状態を読むだけ（凍結ハーネスも触らない）。

使い方:
    .venv/bin/python research_notes/scripts/check_e1_confirmation_timing.py \
        --maze-dir competition/mazes/contest_reference
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.baseline_classical import AdachiPolicy  # noqa: E402
from competition.evaluator import CompetitionEvaluator  # noqa: E402

CHECK_EVERY = 100     # 制御周期 10 ms → 1 s ごとに確定判定を見る（BFS 2 回/回）


class ConfirmProbe:
    """L0-a を包み、確定判定・目的モード・時刻を記録する。"""

    def __init__(self, inner):
        self._inner = inner
        self._sim = None
        self._i = 0
        self.t_confirm = None
        self.rec = []          # (時刻, 確定か, target_mode)

    name = property(lambda self: getattr(self._inner, "name", "unnamed"))
    requires_privileged = property(lambda self: getattr(self._inner, "requires_privileged", False))

    def bind_sim(self, sim):
        self._sim = sim
        return self._inner.bind_sim(sim)

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def act(self, obs):
        out = self._inner.act(obs)
        if self._sim is not None:
            t = self._sim.sim_time
            mode = getattr(self._inner, "target_mode", None)
            # 2026-08-11 修正: 以前は `or self.t_confirm is None` を付けていたため、
            # 確定するまで**毎制御周期**（100 Hz）確定判定を呼んでいた。
            # is_shortest_confirmed は BFS を 2 回回すので、確定が遅い面ほど
            # 二乗的に重くなり、腕1 の測定が実用にならなかった（20 面で 7 時間超）。
            # 1 s 刻みで十分（確定時刻の分解能 1 s、持ち時間 420 s に対し 0.24%）。
            if self._i % CHECK_EVERY == 0:
                try:
                    conf = bool(self._inner._shortest_confirmed())
                except Exception:      # 地図が未初期化の期間など
                    conf = False
                if conf and self.t_confirm is None:
                    self.t_confirm = t
                self.rec.append((t, conf, mode))
            self._i += 1
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maze-dir", default="competition/mazes/contest_reference")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"))[:args.n]
    budget = 420.0
    rows = []
    print(f"対象 {len(mazes)} 面（{args.maze_dir}）／持ち時間 {budget:.0f} s")
    print(f"{'面':<24}{'確定':>6}{'確定時刻':>10}{'持ち時間比':>11}{'確定時の走行':>13}"
          f"{'確定後の時間':>13}{'帰路時間':>10}{'うち確定後':>11}")
    for m in mazes:
        probe = ConfirmProbe(AdachiPolicy())
        ev = CompetitionEvaluator(maze_dir=args.maze_dir,
                                  out_dir=str(REPO_ROOT / "outputs" / "e1_confirm"))
        r = ev.evaluate_maze(m, probe)
        tc = probe.t_confirm
        t_end = probe.rec[-1][0] if probe.rec else 0.0
        # 帰路に費やした時間（記録は 1 s 間隔なので区間長を足し合わせる）
        ret_t = ret_after = 0.0
        for (t1, _c1, m1), (t2, _c2, _m2) in zip(probe.rec, probe.rec[1:]):
            if m1 == "to_start":
                ret_t += t2 - t1
                if tc is not None and t1 >= tc:
                    ret_after += t2 - t1
        # 確定時点で何走行目か
        run_idx = None
        if tc is not None:
            for run in r["runs"]:
                if run["t_start"] <= tc <= run["t_end"]:
                    run_idx = run["index"]
                    break
        after = (t_end - tc) if tc is not None else None
        rows.append(dict(maze=r["maze_id"], confirmed=tc is not None, t_confirm=tc,
                         frac=(tc / budget) if tc is not None else None,
                         run_index=run_idx, t_after=after,
                         t_return=ret_t, t_return_after=ret_after,
                         n_runs=len(r["runs"]), best_time=r.get("best_time"),
                         fast_done=r["kpi"]["fast_run_done"]))
        print(f"{r['maze_id']:<24}{'○' if tc is not None else '×':>6}"
              f"{(f'{tc:.1f} s' if tc is not None else '—'):>10}"
              f"{(f'{tc/budget*100:.0f}%' if tc is not None else '—'):>11}"
              f"{(f'run{run_idx}' if run_idx else ('帰路/待機' if tc is not None else '—')):>13}"
              f"{(f'{after:.1f} s' if after is not None else '—'):>13}"
              f"{ret_t:>9.1f}s{(f'{ret_after:.1f} s' if tc is not None else '—'):>11}", flush=True)

    n_conf = sum(1 for r in rows if r["confirmed"])
    print(f"\n確定に到達: {n_conf}/{len(rows)} 面")
    if n_conf:
        fr = [r["frac"] for r in rows if r["confirmed"]]
        af = [r["t_after"] for r in rows if r["confirmed"]]
        print(f"  確定時刻の持ち時間比: 中央値 {np.median(fr)*100:.0f}%"
              f"（{min(fr)*100:.0f}〜{max(fr)*100:.0f}%）")
        print(f"  確定後に消費した時間: 中央値 {np.median(af):.1f} s"
              f"（改良の余地の上限）")
    nf = [r for r in rows if not r["fast_done"]]
    print(f"  最速走行が成立しなかった面: {len(nf)}/{len(rows)}"
          f"（うち確定に到達していたのは {sum(1 for r in nf if r['confirmed'])} 面）")
    print("\n判定:")
    if n_conf == 0:
        print("  **(ii) 持ち時間内に確定に達していない** → 帰還の改良では解決しない")
    elif len(nf) and sum(1 for r in nf if r["confirmed"]) == 0:
        print("  最速走行が不成立の面はいずれも確定に未到達 → **(ii) 寄り**")
    else:
        print("  確定後にも時間を消費している面がある → **(i) 寄り（帰還の改良に余地）**")

    out = REPO_ROOT / "research_notes" / "data" / "e1_confirmation_timing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(maze_dir=args.maze_dir, budget=budget, rows=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n数値 JSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
