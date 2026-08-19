"""巨視的に異なる候補経路の本数を数える
Count macroscopically distinct candidate routes in a maze.

本ファイルは 2026-08-19 の破棄（dc01c51）で巻き添えになり、同日に復元した。
迷路の性質を測る道具であり、破棄された古典走行の実装には依存しない（教授裁定）。

2026-08-11 新設。**ユーザ（マイクロマウスの実務家）の指摘**が起点:

  > 大会は最短になりうる経路が 2 本あるのが通常なので、そこも再現できたら最高です

これは我々が持っていなかった領域知識であり、**評価迷路の設計に直接効く**。
まず「2 本」が何を数えているのかを、測れる形に定義する必要がある。

## 既存の「最短経路の本数」ではだめな理由

`compare_generated_vs_contest.n_shortest_paths` は BFS の距離ラベルから作る
有向非巡回グラフ上の経路数を数える。これは**階段状の並び替えを別経路として数える**:

    右→上→右→上   と   上→右→上→右

は同じ廊下を通っているのに別の 2 本になる。大会実迷路の中央値 5、最大 265 という
値はこの数え方によるもので、**競技者の言う「別ルート」ではない**。

## 本モジュールの定義

競技者が言う「別ルート」は、**巨視的に離れた 2 本**——「左回りで行くか右回りで行くか」
のような、**探索して初めてどちらが速いか分かる分岐**のはずである。
そうでなければ探索する意味がない。そこで次のように定義する。

  **手順**:
    1. 真の最短経路 $P_1$ を 1 本取る（正準な 1 本）
    2. $P_1$ 上の壁を**1 枚ずつ**塞ぎ、そのつど最短経路を求め直す
    3. 得られた経路のうち、長さが $D_0 + \Delta$ 以下で、かつ **$P_1$ から空間的に
       $\theta$ 区画より離れている**ものを「別ルートの候補」とする
    4. 候補どうしも $\theta$ で clustering し、**互いに離れているものだけ**を数える
    5. **候補経路の本数 = 1（$P_1$）+ 別ルートの個数**

  **「経路の辺をすべて塞いで完全に辺素な別経路を探す」ことはできない。**
  規定でゴールの入口は 1 箇所・スタートの開口も 1 箇所なので、**どの 2 本の経路も
  必ずその 2 辺を共有する**。辺素性を要求すると必ず 0 本になる（実際に一度そう
  実装して全 42 面が 1 本になり、定義の誤りだと分かった）。

  **2 つのパラメータ**:
    - $\Delta$ … 「最短になりうる」の許容差 [区画]。$D_0 + \Delta$ 以下の経路を候補とする
    - $\theta$ … 巨視的に異なるとみなす空間的な離れ具合 [区画]。
      2 本の経路の**ハウスドルフ距離**（一方の各区画から他方の最寄り区画までの
      距離の最大値。区画座標のマンハッタン距離で測る）が $\theta$ を超えることを要求

  **$\theta$ が階段状の並び替えを除外する仕掛けである。**並び替えは辺を共有しないが
  空間的には隣接したままなので、ハウスドルフ距離が小さく、別ルートとして数えない。

**パラメータは恣意的に決めない。**大会実迷路 42 面で $\Delta$・$\theta$ を掃引し、
**ユーザの言う「2 本が通常」と整合する組み合わせがあるか**を確認する
（`--sweep`）。整合しなければ我々の解釈が誤りなので、その旨を報告する。

使い方:
    .venv/bin/python -m competition.macro_routes --sweep
    .venv/bin/python -m competition.macro_routes --dirs competition/mazes/eval
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
    DIRS, GOAL_CELLS, true_shortest, true_shortest_path, true_wall)

W = H = 16
D_WINDOW = (45, 110)
CONTEST_DIR = os.path.join(_REPO_ROOT, "competition", "reference_mazes", "contest")


def _edges_of(path):
    """セル列を、壁の識別子（('v'|'h', x, y)）の集合にする。"""
    out = set()
    for a, b in zip(path, path[1:]):
        if a[0] == b[0]:
            out.add(("h", a[0], max(a[1], b[1])))
        else:
            out.add(("v", max(a[0], b[0]), a[1]))
    return out


def _set_wall(v, h, e, val):
    k, x, y = e
    if k == "v":
        v[x, y] = val
    else:
        h[x, y] = val


def hausdorff(a, b):
    """2 つの区画集合のハウスドルフ距離（区画座標のマンハッタン距離）。"""
    def one_way(p, q):
        return max(min(abs(px - qx) + abs(py - qy) for qx, qy in q) for px, py in p)
    return max(one_way(a, b), one_way(b, a))


def macro_routes(v_walls, h_walls, start=(0, 0), goals=GOAL_CELLS,
                 delta=6, theta=3, max_routes=8):
    """巨視的に異なる候補経路を貪欲に列挙する。

    返り値: 経路（セル列）のリスト。先頭は真の最短経路。
    """
    v = np.array(v_walls, dtype=int)
    h = np.array(h_walls, dtype=int)
    d0 = true_shortest(v, h, start, goals)
    if d0 <= 0:
        return []
    first = true_shortest_path(v, h, start, goals)
    routes = [first]
    # 最短経路上の壁を 1 枚ずつ塞ぎ、そのつど代替経路を集める
    for e in sorted(_edges_of(first)):
        _set_wall(v, h, e, 1)
        d = true_shortest(v, h, start, goals)
        p = true_shortest_path(v, h, start, goals) if (0 <= d <= d0 + delta) else None
        _set_wall(v, h, e, 0)
        if p is None:
            continue
        # 既に数えた経路（P1 を含む）のどれかと空間的に近ければ、並び替えの類とみなす
        if any(hausdorff(set(p), set(q)) <= theta for q in routes):
            continue
        routes.append(p)
        if len(routes) >= max_routes:
            break
    return routes


def n_macro_routes(v_walls, h_walls, start=(0, 0), goals=GOAL_CELLS, delta=6, theta=3):
    return len(macro_routes(v_walls, h_walls, start, goals, delta, theta))


# ==========================================================================
def load(f):
    z = np.load(f)
    v, h = np.array(z["v_walls"], int), np.array(z["h_walls"], int)
    if "start_x" in z.files:
        start = (int(z["start_x"]), int(z["start_y"]))
        goals = tuple((int(a), int(b)) for a, b in zip(z["goals_x"], z["goals_y"]))
    else:
        start, goals = (0, 0), GOAL_CELLS
    return v, h, start, goals


def contest_files(window=True):
    out = []
    for f in sorted(glob.glob(os.path.join(CONTEST_DIR, "contest_*.npz"))):
        v, h, s, g = load(f)
        d = true_shortest(v, h, s, g)
        if window and not (D_WINDOW[0] <= d <= D_WINDOW[1]):
            continue
        out.append(f)
    return out


def sweep():
    """大会実迷路で Δ・θ を掃引し、「2 本が通常」と整合する組み合わせを探す。"""
    files_win = contest_files(window=True)
    files_all = contest_files(window=False)
    print(f"大会実迷路: 窓 [45,110] 内 {len(files_win)} 面／全 {len(files_all)} 面")
    print("\nΔ = 「最短になりうる」の許容差 [区画]、θ = 巨視的に異なるとみなす離れ具合 [区画]")
    print(f"\n{'θ':>3}" + "".join(f"{'Δ=' + str(d):>14}" for d in (0, 2, 4, 6, 8, 12)))
    best = []
    for theta in (2, 3, 4, 5):
        row = f"{theta:>3}"
        for delta in (0, 2, 4, 6, 8, 12):
            ns = [n_macro_routes(*load(f), delta=delta, theta=theta) for f in files_win]
            med = float(np.median(ns))
            frac2 = np.mean(np.array(ns) >= 2)
            row += f"{med:>7.1f}({frac2 * 100:>3.0f}%)"
            best.append((abs(med - 2.0), -frac2, theta, delta, med, frac2))
        print(row)
    print("  表の見方: 中央値（2 本以上ある面の割合）")
    best.sort()
    print("\n『2 本が通常』に最も整合する組み合わせ（中央値が 2 に最も近い順）:")
    seen = set()
    for _, _, theta, delta, med, frac2 in best:
        if (theta, delta) in seen:
            continue
        seen.add((theta, delta))
        print(f"  θ={theta}, Δ={delta}: 中央値 {med:.1f}、2 本以上の面 {frac2 * 100:.0f}%")
        if len(seen) >= 5:
            break
    return best


def measure(dirs, delta, theta):
    print(f"\n定義: Δ={delta}（許容差 [区画]）、θ={theta}（巨視的に異なる離れ具合 [区画]）")
    print(f"{'群':<28}{'n':>4}{'中央値':>8}{'四分位':>12}{'範囲':>10}{'2本以上':>9}")
    groups = [("大会実迷路 窓内", contest_files(True)),
              ("大会実迷路 42 面全体", contest_files(False))]
    for d in dirs:
        groups.append((os.path.basename(d), sorted(glob.glob(os.path.join(d, "maze_*.npz")))))
    out = {}
    for lab, files in groups:
        ns = np.array([n_macro_routes(*load(f), delta=delta, theta=theta) for f in files])
        out[lab] = ns.tolist()
        print(f"{lab:<28}{len(ns):>4}{np.median(ns):>8.1f}"
              f"{np.percentile(ns, 25):>6.1f}〜{np.percentile(ns, 75):<5.1f}"
              f"{ns.min():>5}〜{ns.max():<4}{np.mean(ns >= 2) * 100:>8.0f}%")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--delta", type=int, default=6)
    ap.add_argument("--theta", type=int, default=3)
    args = ap.parse_args()
    res = {}
    if args.sweep:
        sweep()
    if args.dirs or not args.sweep:
        res = measure([os.path.join(_REPO_ROOT, d) for d in args.dirs], args.delta, args.theta)
    if res:
        p = os.path.join(_REPO_ROOT, "research_notes", "data", "macro_routes.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(dict(delta=args.delta, theta=args.theta, counts=res),
                  open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n数値 JSON: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
