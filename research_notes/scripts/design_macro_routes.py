#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(b) 現行の生成方式で「候補経路の本数」に届くか — と、届かせると何が壊れるか。

**背景**: ユーザ（実務家）の「大会は最短になりうる経路が 2 本あるのが通常」を
測れる形に定義したところ（`competition/macro_routes.py`）、大会実迷路は
**中央値 2.0・82% が 2 本以上・最大 5 本**、現行の v3 帯は
**中央値 1.0・45%・最大 2 本**で届いていない。

**判定したいこと**:
  (b-1) 現行の生成方式は 3 本以上の候補経路を持つ迷路を**作れるのか**。
        作れないなら受理条件を置いても受理率がゼロになるだけで、
        **生成手順そのものを変えるしかない**
  (b-2) 候補経路の本数を目標に入れると、**既に合っている軸（経路比 R）が壊れないか**
        （教授指示: 軸を足すたびに既存の軸を確認する。今日 3 回同じ失敗をしている）

**生成器は改造しない。**`design_multiaxis_windows.gen_once` で手順のパラメータを
振るだけ。設計専用 seed 帯 61000〜 を使う（評価帯・検証帯は使わない）。

使い方:
    .venv/bin/python -u research_notes/scripts/design_macro_routes.py [--n 150]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "competition" / "reference_mazes",
          REPO_ROOT / "research_notes" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from competition.explore_cost import detour_ratio, true_shortest  # noqa: E402
from competition.macro_routes import contest_files, load, n_macro_routes  # noqa: E402
from design_multiaxis_windows import gen_once  # noqa: E402

SEED_BASE = 61000
DELTA, THETA = 4, 3          # 受理条件に使う値（凍結）


def measure(v, h):
    d = true_shortest(v, h)
    return dict(D=int(d), R=float(detour_ratio(v, h)),
                K=int(n_macro_routes(v, h, delta=DELTA, theta=THETA)),
                beta=int((v[1:16, :] == 0).sum() + (h[:, 1:16] == 0).sum()) - 256 + 1)


def summarize(label, rows, used, elapsed):
    if not rows:
        print(f"{label:<34} 面数 0（消費 {used} seed）")
        return None
    K = np.array([r["K"] for r in rows])
    R = np.array([r["R"] for r in rows])
    D = np.array([r["D"] for r in rows])
    print(f"{label:<34}{len(rows):>5}{np.median(K):>7.1f}"
          f"{np.mean(K >= 2) * 100:>8.0f}%{np.mean(K >= 3) * 100:>8.0f}%{K.max():>6}"
          f"{np.median(R):>9.3f}{np.median(D):>7.0f}{used:>8}{elapsed:>7.0f}s")
    return dict(label=label, n=len(rows), K_med=float(np.median(K)),
                K_ge2=float(np.mean(K >= 2)), K_ge3=float(np.mean(K >= 3)),
                K_max=int(K.max()), R_med=float(np.median(R)),
                D_med=float(np.median(D)), seeds=used,
                rows=[{k: v for k, v in r.items()} for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150, help="各条件で作る候補数")
    ap.add_argument("--max-seeds", type=int, default=3000)
    args = ap.parse_args()

    print(f"候補経路の本数 K（Δ={DELTA}, θ={THETA}）と経路比 R を同時に見る")
    hdr = (f"{'条件':<34}{'n':>5}{'K中央':>7}{'K>=2':>9}{'K>=3':>8}{'K最大':>6}"
           f"{'R中央':>9}{'D中央':>7}{'消費':>8}{'所要':>8}")
    print("\n=== 目標: 大会実迷路 ===")
    print(hdr)
    for lab, win in (("大会実迷路 窓[45,110]内", True), ("大会実迷路 42 面全体", False)):
        rows = []
        for f in contest_files(window=win):
            v, h, s, g = load(f)
            d = true_shortest(v, h, s, g)
            rows.append(dict(D=int(d), R=float(detour_ratio(v, h, s, g)),
                             K=int(n_macro_routes(v, h, s, g, delta=DELTA, theta=THETA)),
                             beta=0))
        summarize(lab, rows, 0, 0)

    print("\n=== 現行の帯（保存済み npz） ===")
    print(hdr)
    for d, lab in (("eval", "現行 eval（v3）"),
                   ("eval_v2_low_detour", "1 回目の是正（v2）"),
                   ("eval_v2_short", "是正前")):
        rows = []
        for f in sorted((REPO_ROOT / "competition" / "mazes" / d).glob("maze_*.npz")):
            v, h, s, g = load(f)
            rows.append(measure(v, h))
        summarize(lab, rows, 0, 0)

    print("\n=== 生成方式の到達範囲（設計専用 seed 61000〜。生成器は無改造） ===")
    print(hdr)
    CONFIGS = [
        ("保護あり・除去15・最終D窓（v3 相当）", True, 15, (45, 110), "final", 1.0),
        ("保護あり・除去30・最終D窓", True, 30, (45, 110), "final", 1.0),
        ("保護あり・除去60・最終D窓", True, 60, (45, 110), "final", 1.0),
        ("床0.85・除去30・最終D窓", True, 30, (45, 110), "final", 0.85),
        ("床0.50・除去30・最終D窓", True, 30, (45, 110), "final", 0.50),
        ("保護なし・除去30・最終D窓", False, 30, (45, 110), "final", 1.0),
    ]
    out = []
    for lab, protect, target, window, won, fr in CONFIGS:
        t0 = time.time()
        rows, seed, used = [], SEED_BASE, 0
        while len(rows) < args.n and used < args.max_seeds:
            r = gen_once(seed, protect, target, window, won, fr)
            used += 1
            seed += 1
            if r is not None:
                rows.append(measure(r[0], r[1]))
        out.append(summarize(lab, rows, used, time.time() - t0))

    p = REPO_ROOT / "research_notes" / "data" / "macro_route_design.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(delta=DELTA, theta=THETA, configs=[o for o in out if o]),
              open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n数値 JSON: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
