#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大会実迷路 42 面を**競技クラス別に分け**、ゴール条件と難度指標を点検する。

背景: ファイル名の `H` が何を指すか未確定だったため、実戦帯（完成判定に使う集合）の
定義が固まらなかった。上流 `kerikun11/micromouse-maze-data` の README（§迷路サイズの推定）と
ファイル名の内訳から、**先頭の数字が迷路の一辺の区画数**であることが確定した。

  - `16MM<年>CX` / `16MM<年>C_*` … **クラシック規格**（区画 180 mm）。本プロジェクトの対象
  - `16MM<年>H_*` / `HX` … **ハーフサイズ規格**（区画 90 mm）を 16×16 の格子で行ったもの
  - `32MM<年>HX` … 全日本ハーフ決勝（32×32）。本プロジェクトには取り込んでいない

    .venv/bin/python research_notes/scripts/check_contest_maze_classes.py
"""
import json
import re
from collections import Counter, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CONTEST = ROOT / "competition" / "reference_mazes" / "contest"
DELTA = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def d_true(z):
    """壁を考慮した幅優先探索での、スタート → ゴールの最短距離（移動回数）。"""
    v, h = z["v_walls"], z["h_walls"]
    M = v.shape[1]
    start = (int(z["start_x"]), int(z["start_y"]))
    goals = set(zip(z["goals_x"].tolist(), z["goals_y"].tolist()))
    dist = {start: 0}
    q = deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) in goals:
            return dist[(x, y)]
        for dx, dy in DELTA:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < M and 0 <= ny < M) or (nx, ny) in dist:
                continue
            blocked = (v[x + 1, y] if dx == 1 else v[x, y] if dx == -1 else
                       h[x, y + 1] if dy == 1 else h[x, y])
            if int(blocked) == 1:
                continue
            dist[(nx, ny)] = dist[(x, y)] + 1
            q.append((nx, ny))
    return None


def klass(name):
    """ファイル名から競技クラスの記号を取り出す（CX / C / H / HX）。"""
    return re.sub(r"^16MM\d{4}", "", name).split("_")[0] or "C"


def main():
    rows = []
    for p in sorted(CONTEST.glob("contest_*.npz")):
        name = p.stem.replace("contest_", "")
        z = np.load(p)
        M = z["v_walls"].shape[1]
        c = M // 2
        goals = set(zip(z["goals_x"].tolist(), z["goals_y"].tolist()))
        start = (int(z["start_x"]), int(z["start_y"]))
        dt = d_true(z)
        manh = abs(start[0] - min(g[0] for g in goals)) + abs(start[1] - min(g[1] for g in goals))
        k = klass(name)
        rows.append(dict(
            name=name, klass=k,
            spec="classic" if k in ("CX", "C") else "half",
            M=M, start=list(start), n_goal=len(goals),
            center2x2=goals == {(c - 1, c - 1), (c - 1, c), (c, c - 1), (c, c)},
            d_true=dt, detour=dt / max(manh, 1)))

    print("=== クラス別（M = 格子の一辺）===")
    for k in ("CX", "C", "H", "HX"):
        sub = [r for r in rows if r["klass"] == k]
        if not sub:
            continue
        print(f"  {k:3s} {len(sub):2d} 面  M={sorted({r['M'] for r in sub})}  "
              f"ゴール中央 2×2 {sum(r['center2x2'] for r in sub)}/{len(sub)}  "
              f"ゴール区画数 {sorted(Counter(r['n_goal'] for r in sub).items())}")
    print(f"  スタート区画: {sorted(Counter(str(r['start']) for r in rows).items())}")

    print("\n=== 規格別の難度指標 ===")
    for spec in ("classic", "half"):
        d = np.array([r["d_true"] for r in rows if r["spec"] == spec])
        de = np.array([r["detour"] for r in rows if r["spec"] == spec])
        print(f"  {spec:8s} n={len(d):2d}  最短距離 中央値 {np.median(d):6.1f} "
              f"（{d.min()}〜{d.max()}）  迂回率 中央値 {np.median(de):5.2f} "
              f"（{de.min():.2f}〜{de.max():.2f}）")
    allr = np.array([r["d_true"] for r in rows])
    print(f"  {'混合 42':8s} n={len(allr)}  最短距離 中央値 {np.median(allr):6.1f} "
          f"（{allr.min()}〜{allr.max()}）  ← `docs/MAZE_DIFFICULTY_REPORT.md` が使った母集団")

    cl = [r for r in rows if r["spec"] == "classic"]
    print(f"\n=== 実戦帯の候補（クラシック規格のみ）= {len(cl)} 面 ===")
    print(f"  全 {len(cl)} 面が 16×16・ゴール中央 2×2・スタート (0,0) "
          f"⇒ 評価器の前提を満たす: {all(r['center2x2'] and r['M'] == 16 for r in cl)}")
    print(f"  設計窓 [45,110] に入るもの: "
          f"{sum(1 for r in cl if 45 <= r['d_true'] <= 110)}/{len(cl)}")

    out = ROOT / "research_notes" / "scripts" / "contest_class_audit.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n書き出し: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
