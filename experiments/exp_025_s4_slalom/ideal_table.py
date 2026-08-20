#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""experiments/exp_025_s4_slalom/ideal_table.py

`competition/mazes/design_turn_v1` の10迷路すべてについて、`classic/ideal.py`
（経路から理想時間を出す層）で理想時間の表を作る。**走行は1本も回さない**
（`classic/ideal.py` は `classic/` パッケージの規約どおり MuJoCo も `mouse/` の
シミュレータも import しない — 表の数値はすべて真の壁からの解析計算である）。

出す表（1迷路1行）:
    seed | D0 | 経路区画数 | ターン数 | T_ideal(spin) | T_ideal(slalom) |
    経路長 | 最高速 | 直線/円弧/旋回の内訳

加えて、全迷路まとめて:
  - 半径が何で決まったか（geometry/prev/next）の内訳
  - 参考として `v_cap=0.12`（現行実装の巡航速度上限。`classic/motion.py`
    `DEFAULT_V_CRUISE` と同じ値）を課した slalom の理想時間
    （`IdealResult.segments` を `profile.min_time` に通し直すだけなので、
    半径探索をやり直さない）

結果は `outputs/exp_025_s4/ideal_table.json` に保存する。

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
from classic.profile import min_time, vehicle_limits  # noqa: E402

MAZE_DIR = REPO_ROOT / "competition" / "mazes" / "design_turn_v1"
OUT_PATH = REPO_ROOT / "outputs" / "exp_025_s4" / "ideal_table.json"

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
    radius_reason_counts: Dict[str, int] = {"geometry": 0, "prev": 0, "next": 0}
    vcap_totals = {"slalom": 0.0, "slalom_vcap": 0.0}

    header = (
        f"{'seed':>7}{'D0':>5}{'経路区画数':>9}{'ターン数':>8}"
        f"{'T_spin[s]':>11}{'T_slalom[s]':>13}{'経路長[m]':>10}{'最高速[m/s]':>12}"
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
        res_slalom = ideal_time_for_path(path, v_walls, h_walls, Direction.N, mode="slalom")

        for t in res_slalom.turns:
            radius_reason_counts[t.limited_by] = radius_reason_counts.get(t.limited_by, 0) + 1

        # 参考: 現行実装の巡航速度上限(v_cap=0.12)を課したら理想時間はどうなるか。
        # slalom の segments（半径は既に確定済み）を min_time に通し直すだけで、
        # 半径探索（重い幾何計算）はやり直さない。
        vcap_total = 0.0
        # segments は「ブロックごとに独立」という分割情報を持たないため、v_cap 版は
        # 円弧を含む1本のリストとして再度 min_time するだけの単純化ができるのは
        # 「ブロックが1つ（forced-spinターンが無い）」場合に限られる。ここでは
        # 参考値として、forced-spin を無視して全 segments を1本として解く
        # （forced-spin が無い迷路では厳密に一致する。ある迷路では区間の
        # 継ぎ目の速度連続性がわずかに甘くなる — 参考値としての性格上、この
        # 単純化を許容する。詳細は下の集計後コメント参照）。
        if res_slalom.segments:
            it_vcap = min_time(res_slalom.segments, limits, v_start=0.0, v_end=0.0, v_cap=V_CAP_REFERENCE)
            vcap_total = it_vcap.total
        vcap_totals["slalom"] += res_slalom.total
        vcap_totals["slalom_vcap"] += vcap_total

        row = {
            "seed": seed,
            "d0": d0,
            "path_cells": len(path),
            "n_turns": res_slalom.n_turns,
            "t_ideal_spin": res_spin.total,
            "t_ideal_slalom": res_slalom.total,
            "path_length_slalom": res_slalom.path_length,
            "path_length_spin": res_spin.path_length,
            "v_max_slalom": res_slalom.v_max,
            "v_max_spin": res_spin.v_max,
            "by_kind_slalom": dict(res_slalom.by_kind),
            "by_kind_spin": dict(res_spin.by_kind),
            "t_ideal_slalom_vcap0.12": vcap_total,
            "n_forced_spin_in_slalom": sum(1 for t in res_slalom.turns if t.radius <= 0.0),
        }
        rows.append(row)

        print(
            f"{seed:>7}{d0:>5}{len(path):>9}{res_slalom.n_turns:>8}"
            f"{res_spin.total:>11.3f}{res_slalom.total:>13.3f}"
            f"{res_slalom.path_length:>10.3f}{res_slalom.v_max:>12.3f}"
        )
        print(
            f"         slalom内訳: {_by_kind_str(res_slalom.by_kind)}   "
            f"spin内訳: {_by_kind_str(res_spin.by_kind)}"
        )

    elapsed = time.time() - t0
    print("-" * len(header))
    print(f"計算時間: {elapsed:.1f}s（10迷路・spin+slalom）")

    print("\n半径が何で決まったか（design_turn_v1 全10迷路・slalomの全ターン合計）:")
    n_turns_total = sum(radius_reason_counts.values())
    for k in ("geometry", "prev", "next"):
        n = radius_reason_counts.get(k, 0)
        pct = 100.0 * n / n_turns_total if n_turns_total else 0.0
        print(f"  {k:<10}: {n:>4} 回 ({pct:5.1f}%)")
    print(f"  合計       : {n_turns_total:>4} 回")

    print(
        f"\n参考: v_cap={V_CAP_REFERENCE} m/s（現行実装の巡航速度上限）を課した場合の"
        f" slalom 理想時間（10迷路合計）:"
    )
    print(f"  上限なし    : {vcap_totals['slalom']:.3f}s")
    print(f"  v_cap={V_CAP_REFERENCE}: {vcap_totals['slalom_vcap']:.3f}s")
    ratio = vcap_totals["slalom_vcap"] / vcap_totals["slalom"] if vcap_totals["slalom"] else float("nan")
    print(f"  比(v_cap有/無): {ratio:.3f}")
    print(
        "  🔴 この v_cap 参考値は、forced-spin（ターンでその場旋回に降格した箇所）が"
        "ある迷路では、区間の継ぎ目をまたいだ速度連続性を無視した単純化である"
        "（本来はブロックごとに v_start=v_end=0 で個別に解くべきところを、"
        "全 segments を1本として解いている）。forced-spin が無い迷路では厳密に正しい。"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "generator": __file__,
        "maze_dir": str(MAZE_DIR.relative_to(REPO_ROOT)),
        "start": list(START),
        "goals": [list(g) for g in GOALS],
        "v_cap_reference": V_CAP_REFERENCE,
        "rows": rows,
        "radius_reason_counts": radius_reason_counts,
        "vcap_reference_totals": vcap_totals,
        "elapsed_seconds": elapsed,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n数値 JSON: {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
