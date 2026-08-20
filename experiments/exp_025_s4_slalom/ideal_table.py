#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""experiments/exp_025_s4_slalom/ideal_table.py

`competition/mazes/design_turn_v1` の10迷路すべてについて、`classic/ideal.py`
（経路から理想時間を出す層）で理想時間の表を作る。**走行は1本も回さない**
（`classic/ideal.py` は `classic/` パッケージの規約どおり MuJoCo も `mouse/` の
シミュレータも import しない — 表の数値はすべて真の壁からの解析計算である）。

出す表（1迷路1行）:
    seed | D0 | 経路区画数 | ターン数 | T_ideal(spin) |
    T_ideal(slalom, greedy) | T_ideal(slalom, proportional) |
    T_ideal(slalom, best) | 採用された配分 | T_lower_bound | T_ideal/T_lower_bound |
    経路長 | 最高速 | 直線/円弧/旋回の内訳

`T_ideal(slalom, *)` は `classic/ideal.py` の配分方式ごとの理想時間
（`greedy`=経路の先頭から見た先取り、`proportional`=共有する直線の比例配分、
`best`=両方を計算して速い方。既定の理想時間として使うのは `best`）。
`T_lower_bound` は、`best` が採用した幾何経路の長さ（区画中心を結んだ折れ線
ではなく、弧で角を切った後の実際の走行距離）を、曲率を一切無視した1本の
直線として `profile.min_time` に通した時間である（物理的に到達不可能な
厳密下界。`T_ideal >= T_lower_bound` が常に成り立つはずで、比が1に近いほど
理想時間が真の最小に近いことを示す）。

加えて、全迷路まとめて:
  - 半径が何で決まったか（配分方式ごとに語彙が違う。`classic/ideal.py`
    `TurnPlan.limited_by` docstring参照）の内訳
  - `best` でどちらの配分が何回採用されたか
  - 参考として `v_cap=0.12`（現行実装の巡航速度上限。`classic/motion.py`
    `DEFAULT_V_CRUISE` と同じ値）を課した slalom(best) の理想時間
    （`IdealResult.segments` を `profile.min_time` に通し直すだけなので、
    半径探索をやり直さない）

結果は `experiments/exp_025_s4_slalom/ideal_table.json`（判定の分母として
版管理下に置く）と、従来どおり `outputs/exp_025_s4/ideal_table.json`
（`.gitignore` 対象・再生成できる控え）の両方に保存する。

使い方:
    .venv/bin/python experiments/exp_025_s4_slalom/ideal_table.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.ideal import ideal_time_for_path, true_shortest_path  # noqa: E402
from classic.maze_map import Direction  # noqa: E402
from classic.profile import Segment, min_time, vehicle_limits  # noqa: E402

MAZE_DIR = REPO_ROOT / "competition" / "mazes" / "design_turn_v1"
OUT_PATH_TRACKED = REPO_ROOT / "experiments" / "exp_025_s4_slalom" / "ideal_table.json"
OUT_PATH_SCRATCH = REPO_ROOT / "outputs" / "exp_025_s4" / "ideal_table.json"

START = (0, 0)
GOALS = [(7, 7), (7, 8), (8, 7), (8, 8)]  # manifest の生成条件と同じ中央2x2
V_CAP_REFERENCE = 0.12  # classic/motion.py DEFAULT_V_CRUISE と同じ値（現行実装の巡航速度上限）


def _by_kind_str(by_kind: Dict[str, float]) -> str:
    return (
        f"straight={by_kind.get('straight', 0.0):.3f}s "
        f"arc={by_kind.get('arc', 0.0):.3f}s "
        f"spin={by_kind.get('spin', 0.0):.3f}s"
    )


def main() -> int:
    manifest = json.loads((MAZE_DIR / "manifest.json").read_text(encoding="utf-8"))
    limits = vehicle_limits()

    rows: List[Dict] = []
    radius_reason_counts: Dict[str, int] = {}
    allocation_used_counts: Dict[str, int] = {"greedy": 0, "proportional": 0}
    vcap_totals = {"slalom": 0.0, "slalom_vcap": 0.0}

    header = (
        f"{'seed':>7}{'D0':>5}{'ターン数':>8}"
        f"{'T_greedy':>10}{'T_prop':>10}{'T_best':>10}{'採用':>8}"
        f"{'T_lb':>9}{'比':>7}"
    )
    print(header)
    print("-" * len(header))

    t0 = time.time()
    for m in manifest["mazes"]:
        seed = m["seed"]
        d0 = m["d0"]
        data = np.load(MAZE_DIR / f"maze_{seed}.npz")
        v_walls, h_walls = data["v_walls"], data["h_walls"]

        path = true_shortest_path(v_walls, h_walls, START, GOALS, Direction.N)
        assert len(path) - 1 == d0, f"seed={seed}: 経路長がmanifestのd0と不一致（{len(path)-1} vs {d0}）"

        res_spin = ideal_time_for_path(path, v_walls, h_walls, Direction.N, mode="spin")
        res_greedy = ideal_time_for_path(
            path, v_walls, h_walls, Direction.N, mode="slalom", allocation="greedy"
        )
        res_proportional = ideal_time_for_path(
            path, v_walls, h_walls, Direction.N, mode="slalom", allocation="proportional"
        )
        # "best" は greedy/proportional のうち速い方を選ぶだけなので、幾何探索を
        # やり直さず、上ですでに計算した2つから直接選ぶ（`classic/ideal.py` の
        # `_ideal_slalom` ディスパッチャと同じ選び方 = total が小さい方）。
        res_best = res_greedy if res_greedy.total <= res_proportional.total else res_proportional
        assert res_best.allocation_used in ("greedy", "proportional")
        allocation_used_counts[res_best.allocation_used] += 1

        for t in res_best.turns:
            radius_reason_counts[t.limited_by] = radius_reason_counts.get(t.limited_by, 0) + 1

        # 曲率を一切無視した厳密下界: best が採用した幾何経路の長さ（区画中心を
        # 結んだ折れ線ではなく、弧で角を切った後の実際の走行距離）を、曲率0の
        # 1本の直線として min_time に通す。物理的に到達不可能だが、T_ideal を
        # 割ることのできない下からの限界にする（比が1に近いほど理想時間が
        # 真の最小に近い）。
        lb_segment = Segment(length=res_best.path_length, curvature=0.0, kind="straight")
        t_lower_bound = min_time([lb_segment], limits, v_start=0.0, v_end=0.0).total
        ratio = res_best.total / t_lower_bound if t_lower_bound > 0.0 else float("nan")

        # 参考: 現行実装の巡航速度上限(v_cap=0.12)を課したら理想時間はどうなるか。
        # best の segments（半径は既に確定済み）を min_time に通し直すだけで、
        # 半径探索（重い幾何計算）はやり直さない。
        vcap_total = 0.0
        # segments は「ブロックごとに独立」という分割情報を持たないため、v_cap 版は
        # 円弧を含む1本のリストとして再度 min_time するだけの単純化ができるのは
        # 「ブロックが1つ（forced-spinターンが無い）」場合に限られる。ここでは
        # 参考値として、forced-spin を無視して全 segments を1本として解く
        # （forced-spin が無い迷路では厳密に一致する。ある迷路では区間の
        # 継ぎ目の速度連続性がわずかに甘くなる — 参考値としての性格上、この
        # 単純化を許容する。詳細は下の集計後コメント参照）。
        if res_best.segments:
            it_vcap = min_time(res_best.segments, limits, v_start=0.0, v_end=0.0, v_cap=V_CAP_REFERENCE)
            vcap_total = it_vcap.total
        vcap_totals["slalom"] += res_best.total
        vcap_totals["slalom_vcap"] += vcap_total

        row = {
            "seed": seed,
            "d0": d0,
            "path_cells": len(path),
            "n_turns": res_best.n_turns,
            "t_ideal_spin": res_spin.total,
            "t_ideal_slalom_greedy": res_greedy.total,
            "t_ideal_slalom_proportional": res_proportional.total,
            "t_ideal_slalom_best": res_best.total,
            "allocation_used": res_best.allocation_used,
            "t_lower_bound": t_lower_bound,
            "ratio_best_over_lower_bound": ratio,
            # 後方互換: "t_ideal_slalom" は従来どおり採用された（best の）値を指す。
            "t_ideal_slalom": res_best.total,
            "path_length_slalom": res_best.path_length,
            "path_length_spin": res_spin.path_length,
            "v_max_slalom": res_best.v_max,
            "v_max_spin": res_spin.v_max,
            "by_kind_slalom": dict(res_best.by_kind),
            "by_kind_spin": dict(res_spin.by_kind),
            "t_ideal_slalom_vcap0.12": vcap_total,
            "n_forced_spin_in_slalom": sum(1 for t in res_best.turns if t.radius <= 0.0),
        }
        rows.append(row)

        print(
            f"{seed:>7}{d0:>5}{res_best.n_turns:>8}"
            f"{res_greedy.total:>10.3f}{res_proportional.total:>10.3f}{res_best.total:>10.3f}"
            f"{res_best.allocation_used:>8}"
            f"{t_lower_bound:>9.3f}{ratio:>7.3f}"
        )
        print(
            f"         slalom(best)内訳: {_by_kind_str(res_best.by_kind)}   "
            f"spin内訳: {_by_kind_str(res_spin.by_kind)}"
        )

    elapsed = time.time() - t0
    print("-" * len(header))
    print(f"計算時間: {elapsed:.1f}s（10迷路・spin + slalom greedy/proportional の3通り）")

    print("\nbest でどちらの配分が採用されたか（design_turn_v1 全10迷路）:")
    for k in ("greedy", "proportional"):
        n = allocation_used_counts.get(k, 0)
        print(f"  {k:<13}: {n:>2} 迷路")

    print("\n半径が何で決まったか（design_turn_v1 全10迷路・best採用時の全ターン合計。"
          "配分方式ごとに語彙が違う。classic/ideal.py の TurnPlan.limited_by docstring参照）:")
    n_turns_total = sum(radius_reason_counts.values())
    for k, n in sorted(radius_reason_counts.items()):
        pct = 100.0 * n / n_turns_total if n_turns_total else 0.0
        print(f"  {k:<10}: {n:>4} 回 ({pct:5.1f}%)")
    print(f"  合計       : {n_turns_total:>4} 回")

    print(
        f"\n参考: v_cap={V_CAP_REFERENCE} m/s（現行実装の巡航速度上限）を課した場合の"
        f" slalom(best) 理想時間（10迷路合計）:"
    )
    print(f"  上限なし    : {vcap_totals['slalom']:.3f}s")
    print(f"  v_cap={V_CAP_REFERENCE}: {vcap_totals['slalom_vcap']:.3f}s")
    ratio_vcap = (
        vcap_totals["slalom_vcap"] / vcap_totals["slalom"] if vcap_totals["slalom"] else float("nan")
    )
    print(f"  比(v_cap有/無): {ratio_vcap:.3f}")
    print(
        "  🔴 この v_cap 参考値は、forced-spin（ターンでその場旋回に降格した箇所）が"
        "ある迷路では、区間の継ぎ目をまたいだ速度連続性を無視した単純化である"
        "（本来はブロックごとに v_start=v_end=0 で個別に解くべきところを、"
        "全 segments を1本として解いている）。forced-spin が無い迷路では厳密に正しい。"
    )

    out = {
        "generator": __file__,
        "maze_dir": str(MAZE_DIR.relative_to(REPO_ROOT)),
        "start": list(START),
        "goals": [list(g) for g in GOALS],
        "v_cap_reference": V_CAP_REFERENCE,
        "rows": rows,
        "radius_reason_counts": radius_reason_counts,
        "allocation_used_counts": allocation_used_counts,
        "vcap_reference_totals": vcap_totals,
        "elapsed_seconds": elapsed,
    }
    payload = json.dumps(out, ensure_ascii=False, indent=2)

    # 判定の分母となる表なので版管理下（experiments/）に置く。outputs/ 側にも
    # 従来どおり書く（再生成できる控え。.gitignore 対象）。
    OUT_PATH_TRACKED.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH_TRACKED.write_text(payload, encoding="utf-8")
    OUT_PATH_SCRATCH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH_SCRATCH.write_text(payload, encoding="utf-8")
    print(f"\n数値 JSON（版管理下）: {OUT_PATH_TRACKED.relative_to(REPO_ROOT)}")
    print(f"数値 JSON（控え）    : {OUT_PATH_SCRATCH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
