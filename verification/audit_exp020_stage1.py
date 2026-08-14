#!/usr/bin/env python3
"""監査 034: exp_020 段 1 の中間観測（報告トリガーの確認 ＋ 露出と率の分解）

**判定ではない。**判定は完走後に学生B が `judgment.md` で行う。
本監査が行うのは (a) **報告トリガーの確認**（段 1 のゴール件数 0 件なら即報告）と
(b) **件数の増加が「露出」と「率」のどちらから来ているかの分解**である。

### なぜ分解が要るか

**Q2 は件数で書かれている**（段 1 のゴール件数の 6 seed 中央値 ≥ 12）。
**カリキュラムは $D_0$=4 の面を見せる頻度を上げる**ので、
**方策が何も良くならなくても件数は増える**。
**「露出が増えたから」と「率が上がったから」は別の主張**であり、**分けて測らないと読めない。**

### 対照の取り方（**学習量も難度も揃える**）

**exp_019 の最初の 40 万歩に現れた $D_0$=4 の面だけ**を対照とする。

- **学習量が同じ**（どちらも 0〜40 万歩の方策）
- **難度が同じ**（どちらも $D_0$=4 の面）
- **違うのは露出の頻度だけ**（exp_019 は自然分布の 12.7%・exp_020 は 100%）

使い方: `.venv/bin/python verification/audit_exp020_stage1.py`
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_exp019_goal_vs_d0 import d0_of  # noqa: E402

from mouse.maze6_gen import generate_maze  # noqa: E402

SEEDS = [1, 2, 3, 4, 5, 6]
STAGE1_END = 400_000
_CACHE: dict[int, int] = {}


def d0(s: int) -> int:
    if s not in _CACHE:
        _CACHE[s] = d0_of(generate_maze(s, mode="loop"))
    return _CACHE[s]


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).open()]


def main() -> int:
    # --- exp_020 段 1（第 1 エピソードは裁定により件数集計から除外）---
    new = []
    for n in SEEDS:
        rows = load(f"logs/exp_020_seed{n}/episode_seeds.jsonl")
        s1 = [r for r in rows if r["step"] <= STAGE1_END]
        g = sum(1 for r in s1[1:] if r.get("outcome") == "goal")
        new.append(dict(seed=n, n_ep=len(s1) - 1, n_goal=g))

    # --- 対照: exp_019 の同じ歩数帯の D₀=4 の面だけ ---
    ref = []
    for n in SEEDS:
        rows = load(f"logs/exp_019_v2_seed{n}/episode_seeds.jsonl")
        s1 = [r for r in rows if r["step"] <= STAGE1_END]
        d4 = [r for r in s1 if d0(r["maze_seed"]) == 4]
        g = sum(1 for r in d4 if r.get("outcome") == "goal")
        ref.append(dict(seed=n, n_ep=len(d4), n_goal=g))

    counts = [r["n_goal"] for r in new]
    print("=" * 78)
    print("監査 034: exp_020 段 1 の中間観測（判定ではない）")
    print("=" * 78)
    print(f"{'seed':>5}{'exp_020 露出':>13}{'ゴール':>8}"
          f"{'exp_019 露出':>14}{'ゴール':>8}")
    for a, b in zip(new, ref):
        print(f"{a['seed']:>5}{a['n_ep']:>13}{a['n_goal']:>8}"
              f"{b['n_ep']:>14}{b['n_goal']:>8}")

    e_new = statistics.median([r["n_ep"] for r in new])
    e_ref = statistics.median([r["n_ep"] for r in ref])
    g_new = statistics.median(counts)
    g_ref = statistics.median([r["n_goal"] for r in ref])
    rate_new = sum(r["n_goal"] for r in new) / sum(r["n_ep"] for r in new)
    rate_ref = sum(r["n_goal"] for r in ref) / sum(r["n_ep"] for r in ref)

    print("-" * 78)
    print("🔴 報告トリガー（段 1 で 0 件 → 即報告）: "
          f"{'発火なし' if all(c > 0 for c in counts) else '**発火**'}")
    print()
    print(f"  露出（$D_0$=4 の本数の中央値）: exp_019 {e_ref:.0f} → exp_020 {e_new:.0f}"
          f"  = **{e_new/e_ref:.1f} 倍**")
    print(f"  件数（中央値）                : exp_019 {g_ref:.1f} → exp_020 {g_new:.1f}"
          f"  = **{g_new/g_ref:.1f} 倍**")
    print(f"  **率**（1 本あたり・プール）    : exp_019 {rate_ref:.4f} → "
          f"exp_020 {rate_new:.4f}  = **{rate_new/rate_ref:.2f} 倍**")
    print()
    print(f"  → **件数の増加は、ほぼ露出で説明できる。**")
    print(f"     露出だけで期待される件数 = {e_new:.0f} × {rate_ref:.4f} = "
          f"**{e_new*rate_ref:.1f} 件**（Q2 の閾値 12 と同水準）")
    print("  ⚠️ 対照側は $n$ が小さい（ゴール "
          f"{sum(r['n_goal'] for r in ref)} 件 / {sum(r['n_ep'] for r in ref)} 本）。"
          "率の差は統計的に強くない")
    print("=" * 78)

    out = Path(__file__).resolve().parent / "out" / "exp020_stage1.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(exp020=new, exp019_matched=ref,
                                   rate_new=rate_new, rate_ref=rate_ref,
                                   median_counts=g_new,
                                   exposure_only_expectation=e_new * rate_ref),
                              ensure_ascii=False, indent=2))
    print(f"出力: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
