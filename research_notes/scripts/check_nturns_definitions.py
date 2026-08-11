#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`n_turns` の 2 つの定義はどれだけ乖離するか — L0-a と L0-c で実測する。

**なぜ測るのか（教授指示 2026-08-11）**: 同じ「旋回回数」という名前の量が、
学生A と 学生B で別々に実装されている。

| 実装 | 定義 |
|---|---|
| 学生B（`mouse/corridor_eval.py` L86-121・`mouse/maze6_eval.py` L66-91） | **ヨー角の累積が ±45° に達するたびに 1**。達したら積算を 0 に戻し、途中で符号が反転しても 0 に戻す |
| 学生A（`research_notes/scripts/check_time_model_residual.py` の `path_stats`） | **区画列から進行方向の変化**を数える。90° 転回 = 1、180° 反転 = 2 |

**揃える必要がある理由**: 准教授の (c2) 監査（L0-a は 4 帯すべてで速度向上率が負）が
旋回回数に依存している。**定義がずれていると、あの結論の検算ができない。**

**予想される乖離の向き**（測る前に書いておく）:

  - **超信地旋回（L0-a）**: 90° の転回でヨー角は 90° 動くので、±45° の閾値を
    **2 回**跨ぐ。したがって **学生B の定義は学生A の約 2 倍**になるはず
  - **スラローム（L0-c）**: 円弧で連続的に曲がるので、上と同じ 2 倍になるか、
    追従の揺れで余分に数えるか、事前には分からない
  - **直進中のふらつき**: 符号が反転するたびに積算が 0 に戻るので、
    単調に 45° 溜まらない限り数えない。ただし**片側に偏った揺れは数える**

**予想が当たっても外れても、そのまま報告する**（実験規律 §9）。

## 走行記録について（重要）

**exp007 の結果 JSON には軌跡が入っていない**（`runs` は t_start / t_end /
run_time / path_length_m 等のみ）。したがって「既存の記録に定義を当てる」ことは
できず、**同一条件で走らせ直して軌跡を取る**。再現できたかを確認するため、
**exp007 に保存された run_time との照合**を併せて出す。

軌跡は評価器に記録項目を足さず、方策を包んで取る（凍結ハーネスは触らない）。

使い方:
    .venv/bin/python -u research_notes/scripts/check_nturns_definitions.py \
        --policy l0a --maze-dir competition/mazes/eval
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.evaluator import CompetitionEvaluator  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

# exp007 の結果（run_time の照合先）
EXP007 = {
    "l0a": REPO_ROOT / "competition/results/exp007/eval/adachi_classical_20260811_070512",
    "l0c": REPO_ROOT / "competition/results/exp007/eval/l0c_slalom_20260811_070512",
}


def make_policy(kind):
    if kind == "l0a":
        from competition.baseline_classical import AdachiPolicy
        return AdachiPolicy()
    if kind == "l0c":
        from competition.baseline_slalom import SlalomPolicy
        return SlalomPolicy()
    raise SystemExit(f"未知の方式: {kind}")


class PoseProbe:
    """方策を包み、制御周期ごとに (時刻, x, y, ヨー角) を記録する。"""

    def __init__(self, inner):
        self._inner = inner
        self._sim = None
        self.rec = []

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
            x, y, yaw = self._sim.privileged_pose()
            self.rec.append((self._sim.sim_time, x, y, yaw))
        return out


def slice_run(rec, t0, t1):
    """時刻 [t0, t1] のサンプルだけを取り出す。"""
    return [s for s in rec if t0 - 1e-9 <= s[0] <= t1 + 1e-9]


def turns_cellseq(seg, cell):
    """【学生A の定義】区画列から進行方向の変化を数える。180° 反転は 2。

    `check_time_model_residual.py` の `path_stats` と同一のロジック。
    """
    seq = []
    for _, x, y, _ in seg:
        c = (int(x // cell), int(y // cell))
        if not seq or seq[-1] != c:
            seq.append(c)
    if len(seq) < 2:
        return max(len(seq) - 1, 0), 0
    order = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
    dirs = [(int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))) for a, b in zip(seq, seq[1:])]
    turns = 0
    for d1, d2 in zip(dirs, dirs[1:]):
        if d1 not in order or d2 not in order:
            continue
        k1, k2 = order[d1], order[d2]
        turns += min((k2 - k1) % 4, (k1 - k2) % 4)
    return len(seq) - 1, turns


def turns_yaw45(seg):
    """【学生B の定義】ヨー角の累積が ±45° に達するたびに 1。

    `mouse/corridor_eval.py` L113-121 の漸化式をそのまま写す（読むだけ。触らない）。
    """
    if len(seg) < 2:
        return 0
    yaw_prev = seg[0][3]
    yaw_acc, n_turns = 0.0, 0
    for _, _, _, yaw in seg[1:]:
        dyaw = math.atan2(math.sin(yaw - yaw_prev), math.cos(yaw - yaw_prev))
        yaw_prev = yaw
        if yaw_acc * dyaw < 0.0:
            yaw_acc = 0.0
        yaw_acc += dyaw
        if abs(yaw_acc) >= math.pi / 4:
            n_turns += 1
            yaw_acc = 0.0
    return n_turns


def yaw_total(seg):
    """参考: 走行中に回った総ヨー角 [deg]（絶対値の積算）。"""
    if len(seg) < 2:
        return 0.0
    tot, prev = 0.0, seg[0][3]
    for _, _, _, yaw in seg[1:]:
        tot += abs(math.atan2(math.sin(yaw - prev), math.cos(yaw - prev)))
        prev = yaw
    return math.degrees(tot)


def load_exp007(kind, maze_id):
    """exp007 に保存された run_time を読む（再現の照合用）。"""
    p = EXP007.get(kind)
    if p is None:
        return None
    f = p / f"{maze_id}.json"
    if not f.exists():
        return None
    d = json.load(open(f, encoding="utf-8"))
    return [r["run_time"] for r in d["runs"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", choices=["l0a", "l0c"], required=True)
    ap.add_argument("--maze-dir", default="competition/mazes/eval")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--traj-dir", default=None,
                    help="軌跡の保存先（研究計画書 §9-3。既定は outputs/nturns_defs/<方式>/traj）")
    args = ap.parse_args()

    cell = RobotParams().cell_size
    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"))[:args.n]
    print(f"方式 {args.policy}／迷路 {len(mazes)} 面／{args.maze_dir}", flush=True)
    print(f"  区画寸法 {cell * 1000:.0f} mm（ハードコードせず RobotParams から読む）", flush=True)

    traj_dir = Path(args.traj_dir) if args.traj_dir else (
        REPO_ROOT / "outputs" / "nturns_defs" / args.policy / "traj")
    traj_dir.mkdir(parents=True, exist_ok=True)
    print(f"  軌跡の保存先（§9-3）: {traj_dir}", flush=True)

    rows = []
    for m in mazes:
        probe = PoseProbe(make_policy(args.policy))
        ev = CompetitionEvaluator(maze_dir=args.maze_dir,
                                  out_dir=str(REPO_ROOT / "outputs" / "nturns_defs"))
        r = ev.evaluate_maze(m, probe)
        ref = load_exp007(args.policy, r["maze_id"])
        # 走行境界つきで軌跡そのものを保存する（研究計画書 §9-3）。
        # 制御周期 100 Hz で 1 面あたり数万点になるので float32 + 圧縮で置く。
        rec = np.asarray(probe.rec, dtype=np.float64)
        np.savez_compressed(
            traj_dir / f"{r['maze_id']}.npz",
            t=rec[:, 0].astype(np.float64), x=rec[:, 1].astype(np.float32),
            y=rec[:, 2].astype(np.float32), yaw=rec[:, 3].astype(np.float64),
            run_index=np.array([q["index"] for q in r["runs"]], dtype=np.int32),
            run_t_start=np.array([q["t_start"] for q in r["runs"]], dtype=np.float64),
            run_t_end=np.array([q["t_end"] for q in r["runs"]], dtype=np.float64),
            run_outcome=np.array([q["outcome"] for q in r["runs"]]))
        for i, run in enumerate(r["runs"]):
            seg = slice_run(probe.rec, run["t_start"], run["t_end"])
            cells, t_a = turns_cellseq(seg, cell)
            t_b = turns_yaw45(seg)
            rows.append(dict(maze=r["maze_id"], run=run["index"], outcome=run["outcome"],
                             run_time=run["run_time"], cells=cells,
                             turns_cellseq=t_a, turns_yaw45=t_b,
                             yaw_total_deg=yaw_total(seg), n_samples=len(seg),
                             ref_run_time=(ref[i] if ref and i < len(ref) else None)))
            print(f"  {r['maze_id']} run{run['index']} {run['outcome']:>8} "
                  f"t={run['run_time'] if run['run_time'] else float('nan'):7.2f}s "
                  f"区画{cells:>4}  学生A {t_a:>4} / 学生B {t_b:>4}  "
                  f"比 {(t_b / t_a) if t_a else float('nan'):5.2f}  "
                  f"総ヨー {yaw_total(seg):7.0f}°", flush=True)

    # --- 集計
    ok = [r for r in rows if r["turns_cellseq"] > 0]
    a = np.array([r["turns_cellseq"] for r in ok], float)
    b = np.array([r["turns_yaw45"] for r in ok], float)
    ratio = b / a
    print(f"\n【集計】n={len(ok)} 走行（学生A の定義が 0 の走行は比を作れないので除外: "
          f"{len(rows) - len(ok)} 本）")
    print(f"  学生A（区画列）  : 中央値 {np.median(a):.0f}（四分位 {np.percentile(a, 25):.0f}"
          f"〜{np.percentile(a, 75):.0f}、範囲 {a.min():.0f}〜{a.max():.0f}）")
    print(f"  学生B（ヨー±45°）: 中央値 {np.median(b):.0f}（四分位 {np.percentile(b, 25):.0f}"
          f"〜{np.percentile(b, 75):.0f}、範囲 {b.min():.0f}〜{b.max():.0f}）")
    print(f"  **比 B/A**       : 中央値 {np.median(ratio):.3f}"
          f"（四分位 {np.percentile(ratio, 25):.3f}〜{np.percentile(ratio, 75):.3f}"
          f"、範囲 {ratio.min():.3f}〜{ratio.max():.3f}）")
    if a.size >= 3:
        r_pear = float(np.corrcoef(a, b)[0, 1])
        # 原点を通る最小二乗（B = k·A）
        k = float((a @ b) / (a @ a))
        resid = b - k * a
        print(f"  相関 {r_pear:.4f}／原点を通る当てはめ B = {k:.3f}×A "
              f"（残差 RMS {np.sqrt(np.mean(resid ** 2)):.2f} 回、最大 {np.max(np.abs(resid)):.0f} 回）")

    # --- 再現の照合（exp007 との run_time 一致）
    cmp = [r for r in rows if r["ref_run_time"] is not None and r["run_time"] is not None]
    if cmp:
        d = np.array([abs(r["run_time"] - r["ref_run_time"]) for r in cmp])
        print(f"\n【再現の照合】exp007 の run_time と比較（n={len(cmp)} 走行）")
        print(f"  絶対差: 中央値 {np.median(d):.4f} s、最大 {d.max():.4f} s、"
              f"一致（<1e-6 s）{int((d < 1e-6).sum())}/{len(cmp)} 本")
        if d.max() >= 1e-6:
            print("  → 完全一致しない走行がある。**再現できなかった走行は個別に報告する**")
            for r in cmp:
                if abs(r["run_time"] - r["ref_run_time"]) >= 1e-6:
                    print(f"     {r['maze']} run{r['run']}: 今回 {r['run_time']:.3f} s / "
                          f"exp007 {r['ref_run_time']:.3f} s")
    else:
        print("\n【再現の照合】比較先の exp007 記録が見つからず、**未確認**")

    out = Path(args.out) if args.out else (REPO_ROOT / "research_notes" / "data" /
                                           f"nturns_defs_{args.policy}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(policy=args.policy, maze_dir=args.maze_dir, cell_size=cell, rows=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n数値 JSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
