#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 帯の事後受け入れ検査 — 凍結してよいかの判断材料を全部出す。

教授の指示（2026-08-11）による検査項目:

  A. **選抜した軸**（経路比 R）が大会実迷路の分布に一致しているか
     — 中央値だけでなく**四分位**も見る。層別サンプリングの目的がそこにあるため
  B. **選抜していない軸**が大会実迷路の範囲に入るか（事後検査）
     — $N_\Delta$・β・行き止まり数・最短経路本数・橋の割合
     **選抜していない軸は、合っていなくても直ちに不合格ではない**が、
     大きく外れていれば報告する（note_008 §2 の教訓: 最適化していない軸は測らないと分からない）
  C. **同点処理への依存**（追加条件 B）
     — R の定義に使う探索器の同点規則を変えた 2 つ目の R と整合するか。
     大きくずれるなら「同点処理の癖に迷路を合わせた」ことになる
  D. 規定 6 項目（`audit_maze_rules.py` が本体。ここでは再掲のみ）

**比較先**: 大会実迷路のうち $D$ が窓 [45,110] 内の 33 面（層の境界を決めたのと同じ集合）。

使い方:
    .venv/bin/python -m competition.audit_band_v3 \\
        --dirs competition/mazes/eval_v3 competition/mazes/validation_v3
"""
import argparse
import glob
import json
import os
import sys
from collections import deque

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from competition.explore_cost import (  # noqa: E402
    DIRS, GOAL_CELLS, detour_ratio, n_delta, true_shortest, true_wall)

W = H = 16
D_WINDOW = (45, 110)
CONTEST_DIR = os.path.join(_REPO_ROOT, "competition", "reference_mazes", "contest")


def load(f):
    z = np.load(f)
    v, h = np.array(z["v_walls"], int), np.array(z["h_walls"], int)
    if "start_x" in z.files:
        start = (int(z["start_x"]), int(z["start_y"]))
        goals = tuple((int(a), int(b)) for a, b in zip(z["goals_x"], z["goals_y"]))
    else:
        start, goals = (0, 0), GOAL_CELLS
    return v, h, start, goals


def dead_ends(v, h):
    return sum(1 for x in range(W) for y in range(H)
               if sum(1 for d in range(4)
                      if not true_wall(v, h, x, y, d)
                      and 0 <= x + DIRS[d][0] < W and 0 <= y + DIRS[d][1] < H) == 1)


def n_shortest_paths(v, h, start, goals):
    """最短経路の本数（DP）。"""
    d0 = true_shortest(v, h, start, goals)
    dist = {tuple(start): 0}
    dq = deque([tuple(start)])
    while dq:
        c = dq.popleft()
        for d, (dx, dy) in enumerate(DIRS):
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < W and 0 <= n[1] < H) or n in dist:
                continue
            if true_wall(v, h, c[0], c[1], d):
                continue
            dist[n] = dist[c] + 1
            dq.append(n)
    cnt = {tuple(start): 1.0}
    for c in sorted(dist, key=lambda k: dist[k]):
        if dist[c] == 0:
            continue
        s = 0.0
        for d, (dx, dy) in enumerate(DIRS):
            p = (c[0] + dx, c[1] + dy)
            if p in dist and dist[p] == dist[c] - 1 and not true_wall(v, h, c[0], c[1], d):
                s += cnt.get(p, 0.0)
        cnt[c] = s
    return int(sum(cnt.get(tuple(g), 0.0) for g in goals if dist.get(tuple(g)) == d0))


def bridge_fraction(v, h, start, goals):
    """最短経路上の壁のうち、塞ぐとゴールへ到達できなくなる壁（橋）の割合。"""
    from competition.explore_cost import true_shortest_path
    path = true_shortest_path(v, h, start, goals)
    if not path or len(path) < 2:
        return float("nan")
    d0 = true_shortest(v, h, start, goals)
    n_inf = 0
    for a, b in zip(path, path[1:]):
        if a[0] == b[0]:
            key, idx = "h", (a[0], max(a[1], b[1]))
        else:
            key, idx = "v", (max(a[0], b[0]), a[1])
        arr = h if key == "h" else v
        arr[idx] = 1
        if true_shortest(v, h, start, goals) < 0:
            n_inf += 1
        arr[idx] = 0
    return n_inf / (len(path) - 1)


def metrics(f):
    v, h, start, goals = load(f)
    d0 = true_shortest(v, h, start, goals)
    nd = n_delta(v, h, (2, 8), start, goals)
    open_edges = int((v[1:W, :] == 0).sum() + (h[:, 1:H] == 0).sum())
    return dict(
        name=os.path.basename(f)[:-4], d_true=int(d0),
        R=detour_ratio(v, h, start, goals),
        R_compass=detour_ratio(v, h, start, goals, tiebreak="compass"),
        N2=nd["N2"] / d0, N8=nd["N8"] / d0,
        beta=open_edges - W * H + 1,
        dead_ends=dead_ends(v, h),
        n_paths=n_shortest_paths(v, h, start, goals),
        bridge=bridge_fraction(v, h, start, goals))


def summ(rows, key):
    a = np.array([r[key] for r in rows], dtype=float)
    a = a[~np.isnan(a)]
    return dict(n=int(a.size), med=float(np.median(a)),
                p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
                lo=float(a.min()), hi=float(a.max()))


def line(label, s, fmt="{:.2f}"):
    return (f"{label:<26}{s['n']:>4}  " + fmt.format(s["med"]) + "  "
            + fmt.format(s["p25"]) + "〜" + fmt.format(s["p75"]) + "  "
            + fmt.format(s["lo"]) + "〜" + fmt.format(s["hi"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", required=True)
    args = ap.parse_args()

    con = []
    for f in sorted(glob.glob(os.path.join(CONTEST_DIR, "contest_*.npz"))):
        m = metrics(f)
        if D_WINDOW[0] <= m["d_true"] <= D_WINDOW[1]:
            con.append(m)

    groups = [("大会実迷路 窓内（目標）", con)]
    for d in args.dirs:
        groups.append((os.path.basename(d),
                       [metrics(f) for f in sorted(glob.glob(os.path.join(d, "maze_*.npz")))]))
    # 参考として現行帯も
    cur = os.path.join(_REPO_ROOT, "competition", "mazes", "eval")
    if os.path.isdir(cur):
        groups.append(("（参考）現行 eval",
                       [metrics(f) for f in sorted(glob.glob(os.path.join(cur, "maze_*.npz")))]))

    print("=" * 92)
    print("v3 帯の事後受け入れ検査（中央値 / 四分位 / 範囲）")
    print("=" * 92)

    print("\n【A】選抜した軸: 経路比 R —— 中央値だけでなく四分位も一致しているか")
    print(f"{'群':<26}{'n':>4}  {'中央値':<6}{'四分位':<14}{'範囲'}")
    for lab, rows in groups:
        print(line(lab, summ(rows, "R"), "{:.3f}"))

    print("\n【B】選抜していない軸（事後検査。外れていれば報告する）")
    for key, lab, fmt in (("d_true", "真の最短距離 D_true", "{:.0f}"),
                          ("N2", "N_2 / D_0", "{:.2f}"),
                          ("N8", "N_8 / D_0", "{:.2f}"),
                          ("beta", "独立閉路数 β", "{:.0f}"),
                          ("dead_ends", "行き止まり数", "{:.0f}"),
                          ("n_paths", "最短経路の本数", "{:.0f}"),
                          ("bridge", "橋の割合", "{:.3f}")):
        print(f"\n  ● {lab}")
        for g_lab, rows in groups:
            print("  " + line(g_lab, summ(rows, key), fmt))

    print("\n【C】同点処理への依存（追加条件 B）")
    print(f"{'群':<26}{'n':>4}  {'R(straight)':>12}{'R(compass)':>12}{'差の中央値':>12}"
          f"{'順位相関':>10}")
    for lab, rows in groups:
        a = np.array([r["R"] for r in rows])
        b = np.array([r["R_compass"] for r in rows])
        rk = lambda z: np.argsort(np.argsort(z))  # noqa: E731
        rho = float(np.corrcoef(rk(a), rk(b))[0, 1]) if len(a) > 2 else float("nan")
        print(f"{lab:<26}{len(rows):>4}  {np.median(a):>12.3f}{np.median(b):>12.3f}"
              f"{np.median(b - a):>12.3f}{rho:>10.3f}")

    out = os.path.join(_REPO_ROOT, "research_notes", "data", "band_v3_audit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({lab: rows for lab, rows in groups}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n数値 JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
