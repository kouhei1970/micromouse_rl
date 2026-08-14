#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""測定「M-1」（安全率 0.75・素の L0-c 方策・確保済みの評価用 20 迷路）を
**軌跡を記録しながら再実行**する（rerun with trajectory recording）。

目的
--------------------------------------------------------------------------
記録した軌跡（各制御周期ごとの時刻・位置・姿勢）と走行境界（各走行の
開始/終了時刻・結果）を素材に、最速タイムなどの KPI（重要業績評価指標。
ここでは完走可否・最速タイム等の集計値）を**独立に**再計算するため。
**このスクリプトは走らせて記録するだけで、集計・判定は一切行わない**
（判定は別スクリプトで行う）。

手本にしたコード（必ずここに明記すること — 黙って変えないため）
--------------------------------------------------------------------------
1. `experiments/exp_016_diagonal/run_016cal.py`
   - 引数の扱い（`--safety` / `--maze-dir` / `--policy`）: run_016cal.py:94-98
   - seed 帯の安全弁（`common/seed_bands.py` の `assert_seeds_allowed`）を
     `purpose='gate'` と理由文字列つきで通す箇所: run_016cal.py:109-115
   - 方策を `module:Class` 形式で読み込み、`safety_factor` を渡す箇所
     （`Probed(safety_factor=args.safety)`）: run_016cal.py:122, 126
   - `CompetitionEvaluator` の構築と `evaluate_maze` の呼び出し:
     run_016cal.py:127-129

2. `experiments/exp_013_band_v4_reeval/run_arm.py`
   - 軌跡を拾う仕組み（`PoseProbe` クラス。方策を**合成（コンポジション）で
     包む**——継承ではなく、内側の方策インスタンスを保持して `act()` の
     前後に処理を挟む——ことで、方策の出力に一切手を加えずに毎ステップの
     姿勢だけを読み取る）: run_arm.py:49-72
   - `np.savez_compressed` で軌跡 npz を保存する箇所（キー名・dtype は
     ここと**完全に同一**にする）: run_arm.py:122-130

--------------------------------------------------------------------------
⚠️ 独立性の限界（必ず読むこと）
--------------------------------------------------------------------------
**評価器（`competition/evaluator.py` の `CompetitionEvaluator`）そのものは
共有しているので、独立なのは KPI の計算部分だけである。**
評価器が「走行の区切り方」「完走・失敗の判定」「ゴール判定の 2 段階検査」
などを誤って実装していた場合、その誤りは本スクリプトにもそのまま伝播する。
本スクリプトが再計算に対して独立性を持つのは、あくまで
「evaluate_maze が返す走行ごとの生記録（t_start / t_end / outcome /
run_time）と、記録した軌跡から、最速タイム等の集計値を求める計算式」の
部分に限られる。

使い方:
    .venv/bin/python verification/rerun_m1_with_traj.py --safety 0.75
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402
from competition.evaluator import CompetitionEvaluator  # noqa: E402

# 課題 J の合言葉（裁定 R40 条件 4）。確保済みの評価用迷路（凍結帯）を
# 使うので purpose='gate' で通す。理由文字列は教授指示のとおり固定する。
GATE_REASON = ("課題 J: M5 確定基準表の独立再計算（准教授・検証目的の再実行）。"
               "教授指示 2026-08-14")

# 表示専用（stdout の進捗行にのみ使う。rerun_detail.json には保存しない）。
# 「完走」は集計・判定の一種だが、ここでは**画面表示のためだけ**に評価器の
# outcome を素通しで見ているだけで、独自の判定ロジックを追加してはいない
# （run_016cal.py の OK_OUTCOMES と同じ考え方: goal/timeout 以外が混ざって
# いなければ「壊れていない」）。
OK_OUTCOMES = {"goal", "timeout"}


class PoseProbe:
    """方策を包み、制御周期ごとに (時刻, x, y, ヨー角) を記録する（読むだけ）。

    run_arm.py の PoseProbe をそのまま写した（コンポジションで方策を包み、
    `act()` の返り値はそのまま返す。電圧も軌跡も 1 ビットも変えない）。
    """

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


def load_policy(spec: str, safety_factor: float):
    """`module:Class` 形式から方策クラスを読み込み、`safety_factor` を渡して
    インスタンス化する（run_016cal.py の `Probed(safety_factor=args.safety)`
    と同じ渡し方。方策ファイルは一切変更しない）。"""
    mod, _, cls = spec.partition(":")
    cls_obj = getattr(importlib.import_module(mod), cls)
    return cls_obj(safety_factor=safety_factor)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safety", type=float, default=0.75,
                    help="安全率（M-1 既定値 0.75）")
    ap.add_argument("--maze-dir", default="competition/mazes/eval",
                    help="確保済みの評価用迷路ディレクトリ（既定: 評価用20面）")
    ap.add_argument("--policy", default="competition.baseline_slalom:SlalomPolicy",
                    help="方策（module:Class）。既定は素の L0-c")
    ap.add_argument("--out-dir", default="outputs/verification/m1_rerun_traj",
                    help="軌跡 npz と集計 json の出力先")
    args = ap.parse_args()

    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    seeds = [int(p.stem.split("_")[1]) for p in mazes]
    print(describe_seeds(seeds, "competition"))
    print(f"⚠️ 確保済みの評価用迷路を使う（purpose='gate'）／理由: {GATE_REASON}")
    assert_seeds_allowed(seeds, namespace="competition", purpose="gate",
                         reason=GATE_REASON)

    out_dir = REPO_ROOT / args.out_dir
    traj_dir = out_dir / "traj"
    traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"安全率 {args.safety:g}／{len(mazes)} 迷路／{args.maze_dir}"
          f"／方策 {args.policy}\n", flush=True)

    mazes_detail = []
    t_all_start = time.time()
    for m in mazes:
        t_maze_start = time.time()
        probe = PoseProbe(load_policy(args.policy, args.safety))
        ev = CompetitionEvaluator(maze_dir=args.maze_dir, out_dir=str(out_dir))
        r = ev.evaluate_maze(m, probe)

        # --- 軌跡 npz（run_arm.py:122-130 とキー名・dtype を完全に揃える） ---
        rec = np.asarray(probe.rec, dtype=np.float64)
        np.savez_compressed(
            traj_dir / f"{r['maze_id']}.npz",
            t=rec[:, 0], x=rec[:, 1].astype(np.float32), y=rec[:, 2].astype(np.float32),
            yaw=rec[:, 3],
            run_index=np.array([q["index"] for q in r["runs"]], dtype=np.int32),
            run_t_start=np.array([q["t_start"] for q in r["runs"]], dtype=np.float64),
            run_t_end=np.array([q["t_end"] for q in r["runs"]], dtype=np.float64),
            run_outcome=np.array([q["outcome"] for q in r["runs"]]))

        # --- 走行ごとの生の記録（集計・判定はしない。評価器の値をそのまま並べる） ---
        runs_raw = [dict(run=q["index"], outcome=q["outcome"],
                         t_start=q["t_start"], t_end=q["t_end"],
                         run_time=q["run_time"]) for q in r["runs"]]

        kpi = r["kpi"]
        mazes_detail.append(dict(
            maze=r["maze_id"],
            runs=runs_raw,
            # 評価器（凍結ハーネス）が計算した集約値。取り違え防止のため
            # harness_ 接頭辞を付けて格納する（独自の再計算値と混同しないため）。
            harness_kpi=kpi,
            harness_best_time=r["best_time"],
            harness_success=bool(r["success"]),
        ))

        outs = [q["outcome"] for q in r["runs"]]
        broken = [o for o in outs if o not in OK_OUTCOMES]
        completed_for_display = bool(kpi["goal_reached"] and not broken)
        elapsed = time.time() - t_maze_start
        print(f"  {r['maze_id']}: {'完走' if completed_for_display else '**未完走**'}"
              f" 走行 {len(outs)} 本 {outs}"
              f" 評価器の最速 {kpi['fast_time'] if kpi['fast_time'] else '-'}"
              f" 経過 {elapsed:.1f}s", flush=True)

    out_path = out_dir / "rerun_detail.json"
    json.dump(dict(measurement="M-1", safety_factor=args.safety, maze_dir=args.maze_dir,
                   policy=args.policy, gate_reason=GATE_REASON, mazes=mazes_detail),
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    total_elapsed = time.time() - t_all_start
    print(f"\n軌跡: {traj_dir}（{len(mazes)} 迷路）")
    print(f"走行ごと記録: {out_path}")
    print(f"総経過 {total_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
