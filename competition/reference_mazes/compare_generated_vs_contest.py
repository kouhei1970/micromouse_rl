#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成迷路と大会実迷路 42 面の難度比較（教授指示の追加検査 A / B）
Compare generated evaluation mazes against the 42 real contest mazes.

2026-08-11、`docs/MAZE_DIFFICULTY_REPORT.md` §5 案 3 による評価迷路の是正を受けて、
教授から追加で 2 件の受け入れ検査を指示された。本スクリプトはその 2 件を実施する。

■ 追加検査 A: D_true 分布が大会迷路と重なっているか
  受理窓 [45, 110] に入れるだけでは不十分で、**窓の内側でどう分布するか**が問題。
  リジェクトサンプリングは往々にして分布を窓の端に張りつかせる。中央値・四分位・
  最小・最大を並べ、ヒストグラムを重ね描きして、
    - 中央値が大会の 63 から大きく外れていないか
    - 分布が窓の両端に二峰化していないか
  を確認する。棄却域は `docs/MAZE_DIFFICULTY_REPORT.md` §6 の V1〜V3。

■ 追加検査 B: 「ほぼ同着の別経路」があるか
  案 3 の手順 2 は「最短距離を 1 区画も縮めない壁だけを開ける」ので、**設計上、
  閉路はすべて純粋な遠回りになる**。大会迷路には「最短とほぼ同じ長さの別経路」が
  存在し、これが探索の難しさの本体である可能性がある。是正後の迷路が
  「一本道＋無意味な枝」ばかりになっていないかを次の 2 指標で確認する。
    B-1 最短経路の本数（既測。本番 40 面で分布まで測り直す）
    B-2 **最短経路上の壁を 1 枚ずつ塞いだときの最短距離の増分**（新規指標）
        小さいほど「すぐ隣に代替経路がある」＝大会迷路的。
        大きいほど「一本道で、外れると大きく損をする」構造。

  B-2 の定義上の注意（重要）:
    - 「最短経路上の壁」の取り方は 2 通りあり、両方を計算して併記する。
      (i) **正準最短経路** — BFS の親を決定的な近傍順（北→東→南→西）で選んだ
          1 本の経路。長さは常に D 辺で迷路間で揃うが、複数の最短経路がある場合に
          どれを選ぶかは恣意的。
      (ii) **全最短経路辺** — 「少なくとも 1 本の最短経路に含まれる辺」全体。
          d_start(u) + 1 + d_goal(v) == D で判定でき、経路の選び方に依存しない。
          ただし最短経路が複数あるほど辺数が増え、増分 0 が出やすくなる。
    - **構造上、必ず ∞ になる辺が 2 本ある**: スタート区画は開口 1 箇所（IEEE 4.3）、
      ゴールは入口 1 箇所（IEEE 4.3）なので、この 2 辺はいずれも橋（cut edge）であり、
      塞ぐとゴールへ到達できなくなる。これは規定が要求する構造であって迷路の
      難しさではないため、**∞ の辺は本数を別途数え、増分の分布からは除外する**。

使い方:
    cd competition/reference_mazes
    ../../.venv/bin/python compare_generated_vs_contest.py
出力:
    difficulty_comparison.json      … 全数値（面ごと・集計）
    figs/d_true_overlay.png         … 追加検査 A のヒストグラム重ね描き
    figs/detour_increment.png       … 追加検査 B-2 の分布
"""
import glob
import json
import math
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
CELL_M = 0.18   # 区画幅 [m]（物理距離への換算用）

GEN_DIRS = [
    ("eval", os.path.join(REPO_ROOT, "competition", "mazes", "eval")),
    ("validation", os.path.join(REPO_ROOT, "competition", "mazes", "validation")),
]
CONTEST_DIR = os.path.join(HERE, "contest")


# ==========================================================================
# 迷路の基本操作（16x16 固定。v_walls (17,16) / h_walls (16,17)）
# ==========================================================================
N = 16


def cells_open(v, h, a, b):
    """隣接セル a, b の間が開通しているか（1=壁、0=開通）。"""
    (ax, ay), (bx, by) = a, b
    if ax == bx:
        return h[ax, max(ay, by)] == 0
    return v[max(ax, bx), ay] == 0


def edge_between(a, b):
    """隣接セル a, b を隔てる壁の識別子 ('v'|'h', x, y)。"""
    (ax, ay), (bx, by) = a, b
    if ax == bx:
        return ("h", ax, max(ay, by))
    return ("v", max(ax, bx), ay)


def set_wall(v, h, e, val):
    k, x, y = e
    if k == "v":
        v[x, y] = val
    else:
        h[x, y] = val


def bfs_from(v, h, sources):
    """sources（セルの集合）からの 4 近傍 BFS 距離。到達不能は -1。"""
    d = -np.ones((N, N), dtype=int)
    dq = deque()
    for c in sources:
        d[c] = 0
        dq.append(c)
    while dq:
        c = dq.popleft()
        cx, cy = c
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            n = (cx + dx, cy + dy)
            if 0 <= n[0] < N and 0 <= n[1] < N and d[n] < 0 and cells_open(v, h, c, n):
                d[n] = d[c] + 1
                dq.append(n)
    return d


def d_true(v, h, start, goals):
    """スタートからゴール集合までの真の最短距離（区画数）。到達不能は -1。"""
    d = bfs_from(v, h, [start])
    vals = [int(d[g]) for g in goals if d[g] >= 0]
    return min(vals) if vals else -1


def n_shortest_paths(v, h, start, goals):
    """最短経路の本数（DP で数える。距離 0 の層から順に足し上げる）。"""
    d = bfs_from(v, h, [start])
    D = d_true(v, h, start, goals)
    if D < 0:
        return 0
    cnt = np.zeros((N, N), dtype=float)
    cnt[start] = 1.0
    order = sorted(((int(d[x, y]), x, y) for x in range(N) for y in range(N) if d[x, y] >= 0))
    for dist, x, y in order:
        if dist == 0:
            continue
        s = 0.0
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if (0 <= nx < N and 0 <= ny < N and d[nx, ny] == dist - 1
                    and cells_open(v, h, (x, y), (nx, ny))):
                s += cnt[nx, ny]
        cnt[x, y] = s
    return int(sum(cnt[g] for g in goals if d[g] == D))


def canonical_shortest_path(v, h, start, goals):
    """正準な最短経路 1 本（セル列）。近傍順 北→東→南→西 で決定的に選ぶ。"""
    d = bfs_from(v, h, [start])
    D = d_true(v, h, start, goals)
    if D < 0:
        return []
    # ゴール候補のうち距離最小・座標最小のものを終点にする（決定的）
    end = min((g for g in goals if d[g] == D))
    path = [end]
    cur = end
    while cur != start:
        cx, cy = cur
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            n = (cx + dx, cy + dy)
            if (0 <= n[0] < N and 0 <= n[1] < N and d[n] == d[cur] - 1
                    and cells_open(v, h, cur, n)):
                cur = n
                path.append(cur)
                break
        else:
            raise RuntimeError("正準最短経路の復元に失敗しました")
    return path[::-1]


def all_shortest_path_edges(v, h, start, goals):
    """少なくとも 1 本の最短経路に含まれる辺の一覧（経路の選び方に依存しない）。

    判定式: d_start(u) + 1 + d_goal(w) == D なら辺 (u,w) は或る最短経路上にある。
    ゴール側の距離は「ゴール集合からの BFS」で一度に求める。
    """
    ds = bfs_from(v, h, [start])
    dg = bfs_from(v, h, list(goals))
    D = d_true(v, h, start, goals)
    if D < 0:
        return []
    out = set()
    for x in range(N):
        for y in range(N):
            if ds[x, y] < 0:
                continue
            for dx, dy in ((0, 1), (1, 0)):     # 各辺を 1 回だけ見る
                nx, ny = x + dx, y + dy
                if not (0 <= nx < N and 0 <= ny < N):
                    continue
                if not cells_open(v, h, (x, y), (nx, ny)):
                    continue
                a, b = (x, y), (nx, ny)
                for u, w in ((a, b), (b, a)):
                    if ds[u] >= 0 and dg[w] >= 0 and ds[u] + 1 + dg[w] == D:
                        out.add(edge_between(a, b))
                        break
    return sorted(out)


def block_increments(v, h, start, goals, edges):
    """edges の各壁を 1 枚ずつ塞いだときの最短距離の増分。

    返り値: (有限の増分のリスト, 到達不能になった辺の本数)
    """
    base = d_true(v, h, start, goals)
    inc, n_inf = [], 0
    for e in edges:
        set_wall(v, h, e, 1)
        d2 = d_true(v, h, start, goals)
        set_wall(v, h, e, 0)
        if d2 < 0:
            n_inf += 1
        else:
            inc.append(int(d2 - base))
    return inc, n_inf


# ==========================================================================
# 統計ユーティリティ
# ==========================================================================
def summ(xs):
    """中央値・四分位・最小・最大・平均・n をまとめる（NaN は除外して n_nan で報告）。"""
    a = np.asarray(xs, dtype=float)
    n_nan = int(np.isnan(a).sum())
    a = a[~np.isnan(a)]
    if a.size == 0:
        return dict(n=0, n_nan=n_nan)
    return dict(n=int(a.size), n_nan=n_nan, min=float(a.min()),
                p25=float(np.percentile(a, 25)), median=float(np.median(a)),
                p75=float(np.percentile(a, 75)), max=float(a.max()), mean=float(a.mean()))


def median_with_inf(finite, n_inf):
    """有限値のリストと ∞ の個数から中央値を求める（∞ は任意の有限値より大きいとする）。

    橋（塞ぐとゴールへ到達できなくなる壁）の増分は ∞ である。∞ を単に捨てると
    「一本道ほど中央値が小さく見える」という逆向きの歪みが出るので、順序統計量
    として正しく扱う。∞ が過半を占めれば中央値は ∞ になる。
    """
    total = len(finite) + n_inf
    if total == 0:
        return float("nan")
    s = sorted(finite)

    def at(i):
        return float(s[i]) if i < len(s) else float("inf")

    if total % 2 == 1:
        return at(total // 2)
    lo, hi = at(total // 2 - 1), at(total // 2)
    return float("inf") if math.isinf(hi) else (lo + hi) / 2.0


def mannwhitney_u(x, y):
    """Mann-Whitney U 検定（両側・正規近似・同順位補正あり）。

    scipy が環境に無いため自前実装。n1=40, n2=42 なら正規近似で十分。
    返り値: (U, z, p)。**本検査では「有意差が消える」ことが望ましい**（p>0.05 が成功）
    という逆向きの使い方をするので、p が大きいほど 2 標本の分布が近い。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = x.size, y.size
    allv = np.concatenate([x, y])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=float)
    sv = allv[order]
    i = 0
    while i < sv.size:                      # 同順位は平均順位を与える
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2.0
    # 同順位補正つき分散
    _, counts = np.unique(allv, return_counts=True)
    tie = float(np.sum(counts ** 3 - counts))
    nn = n1 + n2
    var = n1 * n2 / 12.0 * ((nn + 1) - tie / (nn * (nn - 1)))
    if var <= 0:
        return float(u), float("nan"), float("nan")
    z = (u - mu + 0.5) / math.sqrt(var)     # 連続性補正
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(u), float(z), float(min(1.0, p))


# ==========================================================================
# 迷路の読み込み
# ==========================================================================
def load_generated(path):
    z = np.load(path)
    v = np.array(z["v_walls"], dtype=int)
    h = np.array(z["h_walls"], dtype=int)
    goals = [(7, 7), (7, 8), (8, 7), (8, 8)]
    return v, h, (0, 0), goals, f"maze_{int(z['seed'])}"


def load_contest(path):
    z = np.load(path)
    v = np.array(z["v_walls"], dtype=int)
    h = np.array(z["h_walls"], dtype=int)
    start = (int(z["start_x"]), int(z["start_y"]))
    goals = [(int(a), int(b)) for a, b in zip(z["goals_x"], z["goals_y"])]
    return v, h, start, goals, str(z["source_file"])


def manhattan_to_goal(start, goals):
    return min(abs(start[0] - g[0]) + abs(start[1] - g[1]) for g in goals)


def _inc_stats(prefix, inc, n_inf):
    """増分リスト（有限）と ∞ の本数から、1 面ぶんの指標をまとめる。

    橋（∞）を捨てずに扱うのが要点。橋の割合が高いほど「最短経路が一本道で、
    壁が 1 枚増えるとゴールへ行けなくなる」構造である。
    """
    total = len(inc) + n_inf
    a = np.asarray(inc, dtype=float)
    return {
        f"{prefix}_n_edges": total,
        f"{prefix}_n_inf": n_inf,
        f"{prefix}_frac_inf": (n_inf / total) if total else float("nan"),
        f"{prefix}_inc": inc,
        f"{prefix}_median_with_inf": median_with_inf(inc, n_inf),
        f"{prefix}_median_finite": float(np.median(a)) if a.size else float("nan"),
        f"{prefix}_max_finite": float(a.max()) if a.size else float("nan"),
        f"{prefix}_frac_zero": float(np.mean(a == 0)) if a.size else float("nan"),
        # 全辺（∞ を含む）に対する「増分 0」の割合。橋が多い面ほど小さくなる
        f"{prefix}_frac_zero_of_all": (float(np.sum(a == 0)) / total) if total else float("nan"),
    }


def analyse(v, h, start, goals, name):
    D = d_true(v, h, start, goals)
    man = manhattan_to_goal(start, goals)
    path = canonical_shortest_path(v, h, start, goals)
    canon_edges = [edge_between(path[i], path[i + 1]) for i in range(len(path) - 1)]
    inc_c, inf_c = block_increments(v, h, start, goals, canon_edges)
    all_edges = all_shortest_path_edges(v, h, start, goals)
    inc_a, inf_a = block_increments(v, h, start, goals, all_edges)
    # 独立閉路数 β = 開通辺数 − セル数 + 連結成分数。大会迷路には到達不能セルを
    # 含む面があるので、連結成分数を数えて補正する（そうしないと β を過小評価する）
    open_edges = int((v[1:N, :] == 0).sum() + (h[:, 1:N] == 0).sum())
    n_comp, seen = 0, set()
    for x in range(N):
        for y in range(N):
            if (x, y) in seen:
                continue
            n_comp += 1
            dq = deque([(x, y)])
            seen.add((x, y))
            while dq:
                c = dq.popleft()
                for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    nb = (c[0] + dx, c[1] + dy)
                    if (0 <= nb[0] < N and 0 <= nb[1] < N and nb not in seen
                            and cells_open(v, h, c, nb)):
                        seen.add(nb)
                        dq.append(nb)
    r = dict(
        name=name, d_true=int(D), manhattan=int(man),
        detour=float(D) / man if man else float("nan"),
        n_shortest_paths=n_shortest_paths(v, h, start, goals),
        open_edges=open_edges, n_components=n_comp,
        beta=open_edges - N * N + n_comp,
    )
    r.update(_inc_stats("canon", inc_c, inf_c))
    r.update(_inc_stats("all", inc_a, inf_a))
    return r


# ==========================================================================
# 図
# ==========================================================================
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for cand in ["/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
                 "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"]:
        if os.path.exists(cand):
            font_manager.fontManager.addfont(cand)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
            break
    return plt


def _style(ax, plt):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def fig_d_true(gen, con, window, out_path):
    """追加検査 A: D_true のヒストグラム重ね描き（割合で正規化）。"""
    plt = _setup_mpl()
    g = np.array([r["d_true"] for r in gen], dtype=float)
    c = np.array([r["d_true"] for r in con], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # 左: 全域（大会の長距離面まで含む）
    ax = axes[0]
    bins = np.arange(0, 270, 10)
    ax.hist(c, bins=bins, weights=np.ones_like(c) / c.size, color=C_ORANGE,
            alpha=0.55, label=f"大会実迷路 n={c.size}")
    ax.hist(g, bins=bins, weights=np.ones_like(g) / g.size, histtype="step", lw=2.0,
            color=C_BLUE, label=f"生成迷路 n={g.size}")
    ax.set_title("真の最短距離 $D_{true}$ の分布（全域）", color=TEXT_PRIMARY, fontsize=11)
    ax.set_xlabel("$D_{true}$ [区画]", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("割合", color=TEXT_SECONDARY, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, plt)

    # 右: 受理窓の近傍を拡大し、窓の端への張りつきを見る
    ax = axes[1]
    bins = np.arange(30, 135, 5)
    ax.hist(c, bins=bins, weights=np.ones_like(c) / c.size, color=C_ORANGE,
            alpha=0.55, label=f"大会実迷路 n={c.size}")
    ax.hist(g, bins=bins, weights=np.ones_like(g) / g.size, histtype="step", lw=2.0,
            color=C_BLUE, label=f"生成迷路 n={g.size}")
    for xv, lab in ((window[0], f"窓 下限 {window[0]}"), (window[1], f"窓 上限 {window[1]}")):
        ax.axvline(xv, color=TEXT_SECONDARY, ls="--", lw=1.0)
        ax.text(xv, ax.get_ylim()[1] * 0.97, lab, rotation=90, va="top", ha="right",
                fontsize=8, color=TEXT_SECONDARY)
    ax.set_title("受理窓 [45, 110] の近傍（端への張りつきの確認）",
                 color=TEXT_PRIMARY, fontsize=11)
    ax.set_xlabel("$D_{true}$ [区画]", color=TEXT_SECONDARY, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, plt)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def fig_increment(groups, out_path):
    """追加検査 B-2: 最短経路上の壁を 1 枚塞いだときの増分の分布。

    groups: [(ラベル, 面のリスト, 色), ...]
    """
    plt = _setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # 左: 面ごとの「橋の割合」（塞ぐとゴール到達不能になる壁の割合）
    ax = axes[0]
    bins = np.arange(0, 1.05, 0.05)
    for lab, rs, col, style in groups:
        a = np.array([r["canon_frac_inf"] for r in rs], dtype=float)
        kw = dict(color=col, label=f"{lab} n={a.size}")
        if style == "fill":
            ax.hist(a, bins=bins, weights=np.ones_like(a) / a.size, alpha=0.55, **kw)
        else:
            ax.hist(a, bins=bins, weights=np.ones_like(a) / a.size, histtype="step",
                    lw=2.0, ls=style, **kw)
    ax.set_title("最短経路上の壁のうち「橋」の割合", color=TEXT_PRIMARY, fontsize=11)
    ax.set_xlabel("橋の割合（塞ぐとゴールへ到達できなくなる壁）",
                  color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("面の割合", color=TEXT_SECONDARY, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, plt)

    # 右: 壁 1 枚を単位にプールした累積分布。曲線の頭打ち = 1 − 橋の割合
    ax = axes[1]
    for lab, rs, col, style in groups:
        finite = np.array([x for r in rs for x in r["canon_inc"]], dtype=float)
        total = sum(r["canon_n_edges"] for r in rs)
        xs = np.sort(finite)
        ys = np.arange(1, xs.size + 1) / total   # 分母は ∞ を含む全辺
        ax.step(np.concatenate([[0], xs]), np.concatenate([[0], ys]), where="post",
                color=col, lw=2.0, ls=("-" if style == "fill" else style),
                label=f"{lab} n={total} 枚（橋 {(1 - ys[-1]) * 100:.0f}%）")
    ax.set_ylim(0, 1.0)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_title("壁 1 枚あたりの増分の累積分布（頭打ち = 橋以外の割合）",
                 color=TEXT_PRIMARY, fontsize=11)
    ax.set_xlabel("最短距離の増分 [区画]（10 まで線形・以降は対数）",
                  color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("累積割合（全辺に対する）", color=TEXT_SECONDARY, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, plt)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


# ==========================================================================
def row(label, s, w=34):
    if s.get("n", 0) == 0:
        return f"{label:<{w}} n=0"
    nan = f"  (NaN {s['n_nan']} 面)" if s.get("n_nan") else ""
    return (f"{label:<{w}} n={s['n']:>3}  中央値 {s['median']:>7.2f}  "
            f"四分位 {s['p25']:>6.2f}〜{s['p75']:>6.2f}  "
            f"範囲 {s['min']:>6.2f}〜{s['max']:>7.2f}  IQR {s['p75'] - s['p25']:>5.2f}{nan}")


def load_group(dirs, loader, band_names=None):
    out = []
    for i, d in enumerate(dirs):
        band = band_names[i] if band_names else os.path.basename(d)
        pat = "maze_*.npz" if loader is load_generated else "contest_*.npz"
        for f in sorted(glob.glob(os.path.join(d, pat))):
            v, h, s, g, name = loader(f)
            r = analyse(v, h, s, g, f"{band}/{name}")
            r["band"] = band
            out.append(r)
    return out


def inc_block(label, rs, lims=(0, 2, 4, 10)):
    """B-2 の 1 グループ分の要約行を作る。"""
    finite = np.array([x for r in rs for x in r["canon_inc"]], dtype=float)
    total = sum(r["canon_n_edges"] for r in rs)
    n_inf = sum(r["canon_n_inf"] for r in rs)
    lines = [f"  {label}（{len(rs)} 面・最短経路上の壁 計 {total} 枚）"]
    lines.append(f"    橋（塞ぐとゴール到達不能）: {n_inf}/{total} 枚 = {n_inf / total * 100:.1f}%"
                 f"  ／ 面ごとの橋の割合 中央値 "
                 f"{np.median([r['canon_frac_inf'] for r in rs]) * 100:.1f}%")
    med_all = median_with_inf(list(finite), n_inf)
    lines.append(f"    全 {total} 枚に対する増分の中央値（∞ を順序統計量として含む）: "
                 f"{'∞' if math.isinf(med_all) else f'{med_all:.1f}'} 区画")
    for lim in lims:
        lines.append(f"      増分 <= {lim:>2} 区画: {np.sum(finite <= lim) / total * 100:5.1f}%"
                     f"（橋を除いた {finite.size} 枚の中では "
                     f"{np.mean(finite <= lim) * 100:5.1f}%）")
    lines.append(f"    橋を除く {finite.size} 枚の増分: 中央値 {np.median(finite):.1f}  "
                 f"四分位 {np.percentile(finite, 25):.1f}〜{np.percentile(finite, 75):.1f}  "
                 f"最大 {finite.max():.0f}")
    return "\n".join(lines)


def tarjan_bridges(v, h):
    """無向グラフの橋を Tarjan の低リンク法で列挙する（B-2 の反証テスト用の独立実装）。"""
    adj = {}
    for x in range(N):
        for y in range(N):
            adj[(x, y)] = [(x + dx, y + dy) for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0))
                           if 0 <= x + dx < N and 0 <= y + dy < N
                           and cells_open(v, h, (x, y), (x + dx, y + dy))]
    disc, low, bridges, timer = {}, {}, set(), [0]
    for root in adj:
        if root in disc:
            continue
        disc[root] = low[root] = timer[0]
        timer[0] += 1
        stack = [(root, None, iter(adj[root]))]
        while stack:
            u, pu, it = stack[-1]
            advanced = False
            for w in it:
                if w == pu:
                    continue
                if w in disc:
                    low[u] = min(low[u], disc[w])
                else:
                    disc[w] = low[w] = timer[0]
                    timer[0] += 1
                    stack.append((w, u, iter(adj[w])))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[u])
                    if low[u] > disc[p]:
                        bridges.add(edge_between(p, u))
    return bridges


def selftest_bridges(n_each=6):
    """反証テスト: 「塞ぐと到達不能」で数えた橋の本数が、独立実装（Tarjan）と一致するか。

    B-2 の中心的な主張（生成迷路は最短経路の 7 割が橋）は、この判定が正しいことに
    全面的に依存している。判定手順そのものを別アルゴリズムで検算する。
    一致しなければ B-2 の数値は測定手順の誤りであり、結論は使えない。
    """
    rows = []
    files = ([(f, load_generated) for f in
              sorted(glob.glob(os.path.join(GEN_DIRS[0][1], "maze_*.npz")))[:n_each]]
             + [(f, load_contest) for f in
                sorted(glob.glob(os.path.join(CONTEST_DIR, "contest_*.npz")))[:n_each]])
    for f, loader in files:
        v, h, s, g, name = loader(f)
        path = canonical_shortest_path(v, h, s, g)
        edges = [edge_between(path[i], path[i + 1]) for i in range(len(path) - 1)]
        _, n_direct = block_increments(v, h, s, g, edges)
        n_tarjan = sum(1 for e in edges if e in tarjan_bridges(v, h))
        rows.append((name, len(edges), n_direct, n_tarjan, n_direct == n_tarjan))
    print(f"{'面':<28}{'経路辺数':>9}{'直接法の橋':>11}{'Tarjanの橋':>12}{'一致':>6}")
    for n, e, a, b, ok in rows:
        print(f"{n:<28}{e:>9}{a:>11}{b:>12}{'OK' if ok else 'NG':>6}")
    n_ok = sum(1 for r in rows if r[4])
    print(f"\n橋判定の一致: {n_ok}/{len(rows)} 面 → "
          f"{'B-2 の測定手順は健全' if n_ok == len(rows) else '**不一致あり: B-2 の数値は使えない**'}")
    return n_ok == len(rows)


def main():
    if "--selftest" in sys.argv:
        return 0 if selftest_bridges() else 1
    lo, hi = 45, 110
    gen = load_group([d for _, d in GEN_DIRS], load_generated, [b for b, _ in GEN_DIRS])
    con = load_group([CONTEST_DIR], load_contest, ["contest"])
    # 是正前の旧セット（対照群。是正で何が変わったかを見るため）
    old_dirs = [os.path.join(REPO_ROOT, "competition", "mazes", d)
                for d in ("eval_v2_short", "validation_v2_short")]
    old = load_group([d for d in old_dirs if os.path.isdir(d)], load_generated,
                     ["eval_pre", "validation_pre"])
    con_win = [r for r in con if lo <= r["d_true"] <= hi]

    print("=" * 104)
    print(f"生成迷路 {len(gen)} 面（eval 20 + validation 20） vs 大会実迷路 {len(con)} 面"
          f"（うち窓 [{lo},{hi}] 内 {len(con_win)} 面）／対照: 是正前 {len(old)} 面")
    print("=" * 104)

    # ================= 追加検査 A =================
    gD = [r["d_true"] for r in gen]
    cD = [r["d_true"] for r in con]
    wD = [r["d_true"] for r in con_win]
    sG, sC, sW = summ(gD), summ(cD), summ(wD)
    print("\n【追加検査 A】真の最短距離 D_true の分布 [区画]")
    print(row("  生成 40 面", sG))
    print(row("  うち eval 20 面", summ([r["d_true"] for r in gen if r["band"] == "eval"])))
    print(row("  うち validation 20 面",
              summ([r["d_true"] for r in gen if r["band"] == "validation"])))
    print(row(f"  大会 窓[{lo},{hi}]内 {len(con_win)} 面 ★目標", sW))
    print(row("  大会 42 面 全体", sC))
    print(row("  （対照）是正前 40 面", summ([r["d_true"] for r in old])))

    print("\n  比較 1: 生成 40 面 vs 大会・窓内 33 面 —「生成器は狙った分布を再現できているか」")
    u1, z1, p1 = mannwhitney_u(gD, wD)
    print(f"    Mann-Whitney U（両側・正規近似・同順位補正）: U={u1:.1f} z={z1:.3f} p={p1:.3f}"
          f"  → {'p>0.05: 差は検出されない（成功）' if p1 > 0.05 else 'p<=0.05: 有意差あり'}")
    print(f"    中央値 {sG['median']:.1f} vs {sW['median']:.1f}（差 {sG['median'] - sW['median']:+.1f}）")
    print(f"    IQR   {sG['p75'] - sG['p25']:.1f} vs {sW['p75'] - sW['p25']:.1f}"
          f"（生成/目標 = {(sG['p75'] - sG['p25']) / (sW['p75'] - sW['p25']):.2f}）"
          f"  ← 著しく狭ければ評価帯の多様性不足")
    n_in_w_iqr = sum(1 for x in gD if sW["p25"] <= x <= sW["p75"])
    print(f"    生成 40 面のうち大会・窓内の四分位範囲 [{sW['p25']:.0f}, {sW['p75']:.0f}] に入る: "
          f"{n_in_w_iqr}/40 面（{n_in_w_iqr / 40 * 100:.0f}%）")

    print("\n  比較 2: 生成 40 面 vs 大会 42 面 全体 —「評価帯は競技をどれだけ代表しているか」")
    u2, z2, p2 = mannwhitney_u(gD, cD)
    print(f"    Mann-Whitney U: U={u2:.1f} z={z2:.3f} p={p2:.3f}")
    print(f"    中央値 {sG['median']:.1f} vs {sC['median']:.1f}／IQR "
          f"{sG['p75'] - sG['p25']:.1f} vs {sC['p75'] - sC['p25']:.1f}")
    n_lo = sum(1 for x in cD if x < lo)
    n_hi = sum(1 for x in cD if x > hi)
    print(f"    窓による除外: 下限 {lo} 未満 {n_lo} 面（{n_lo / len(cD) * 100:.1f}%）"
          f"／上限 {hi} 超 {n_hi} 面（{n_hi / len(cD) * 100:.1f}%）"
          f"／合計 {n_lo + n_hi} 面（{(n_lo + n_hi) / len(cD) * 100:.1f}%）を評価帯から除外")
    print(f"      除外された値: 下端 {sorted(x for x in cD if x < lo)}"
          f"／上端 {sorted(x for x in cD if x > hi)}")
    print(f"    重なり: 生成の範囲 {min(gD)}〜{max(gD)} vs 大会の範囲 {min(cD)}〜{max(cD)} "
          f"→ 分布は{'重なっている' if max(gD) >= min(cD) else '重なっていない'}"
          f"（是正前は 15〜26 で完全非重複）")

    edge_lo = sum(1 for x in gD if x < lo + 10)
    edge_hi = sum(1 for x in gD if x > hi - 10)
    print(f"\n  窓の端への張りつき（二峰化の検査）: 下端 {lo}〜{lo + 10} に "
          f"{edge_lo}/40 面（{edge_lo / 40 * 100:.0f}%）、上端 {hi - 10}〜{hi} に "
          f"{edge_hi}/40 面（{edge_hi / 40 * 100:.0f}%）、中間 {40 - edge_lo - edge_hi}/40 面")
    hist, edges = np.histogram(gD, bins=np.arange(45, 120, 10))
    print("    生成 40 面の 10 区画刻みヒストグラム: "
          + " ".join(f"[{int(edges[i])}-{int(edges[i + 1])}):{hist[i]}" for i in range(len(hist))))
    print(f"  物理距離 [m]（1 区画 = {CELL_M} m）: 生成 中央値 {sG['median'] * CELL_M:.2f} m、"
          f"大会 中央値 {sC['median'] * CELL_M:.2f} m、是正前 "
          f"{summ([r['d_true'] for r in old])['median'] * CELL_M:.2f} m")

    print("\n  迂回率（D_true / マンハッタン距離）")
    print(row("  生成 40 面", summ([r["detour"] for r in gen])))
    print(row(f"  大会 窓内 {len(con_win)} 面", summ([r["detour"] for r in con_win])))
    print(row("  大会 42 面 全体", summ([r["detour"] for r in con])))
    print(row("  （対照）是正前 40 面", summ([r["detour"] for r in old])))

    # ================= 追加検査 B-1 =================
    print("\n【追加検査 B-1】最短経路の本数")
    print(row("  生成 40 面", summ([r["n_shortest_paths"] for r in gen])))
    print(row(f"  大会 窓内 {len(con_win)} 面", summ([r["n_shortest_paths"] for r in con_win])))
    print(row("  大会 42 面 全体", summ([r["n_shortest_paths"] for r in con])))
    print(row("  （対照）是正前 40 面", summ([r["n_shortest_paths"] for r in old])))
    for lab, rs in (("生成", gen), ("大会・窓内", con_win), ("大会 全体", con), ("是正前", old)):
        n1 = sum(1 for r in rs if r["n_shortest_paths"] == 1)
        print(f"    最短経路が一意（1 本）の面: {lab} {n1}/{len(rs)}"
              f"（{n1 / len(rs) * 100:.0f}%）")

    # ================= 追加検査 B-2 =================
    print("\n【追加検査 B-2】最短経路上の壁を 1 枚ずつ塞いだときの最短距離の増分 [区画]")
    print("  定義: 正準最短経路（BFS 親を北→東→南→西の順で決定的に選んだ 1 本）の各辺を")
    print("        1 枚ずつ壁に戻し、最短距離を測り直した差。到達不能なら ∞（= その辺は橋）。")
    for lab, rs in (("生成 40 面", gen), (f"大会・窓内 {len(con_win)} 面", con_win),
                    ("大会 42 面 全体", con), ("（対照）是正前 40 面", old)):
        print(inc_block(lab, rs))
    print("    ※ スタート開口 1 箇所・ゴール入口 1 箇所は規定上の橋なので、最低 2 枚は構造的に発生する")

    # 交絡の統制: 橋の割合の差が「閉路の数」で説明できてしまわないかを確かめる。
    # β が同水準なら、差は「閉路をどこに作ったか（位置）」に帰属できる。
    print("\n  交絡因子の統制: 独立閉路数 β = 開通辺数 − 256 + 連結成分数")
    print(row("  β: 生成 40 面", summ([r["beta"] for r in gen])))
    print(row(f"  β: 大会・窓内 {len(con_win)} 面", summ([r["beta"] for r in con_win])))
    print(row("  β: 大会 42 面 全体", summ([r["beta"] for r in con])))
    print(row("  β: （対照）是正前 40 面", summ([r["beta"] for r in old])))
    ub, zb, pb = mannwhitney_u([r["beta"] for r in gen], [r["beta"] for r in con_win])
    print(f"    Mann-Whitney U（β・生成 vs 大会窓内）: U={ub:.1f} z={zb:.3f} p={pb:.3f}")
    # β をそろえた層別比較（大会の β が生成の四分位範囲に入る面だけを取る）
    b25, b75 = summ([r["beta"] for r in gen])["p25"], summ([r["beta"] for r in gen])["p75"]
    strat = [r for r in con_win if b25 <= r["beta"] <= b75]
    if strat:
        print(f"    層別: 大会・窓内のうち β が生成の四分位範囲 [{b25:.0f}, {b75:.0f}] に入る "
              f"{len(strat)} 面 の橋の割合 → "
              f"中央値 {np.median([r['canon_frac_inf'] for r in strat]) * 100:.1f}%"
              f"（生成 40 面は {np.median([r['canon_frac_inf'] for r in gen]) * 100:.1f}%）")

    print("\n  面ごとの代表値の比較")
    print(row("  橋の割合: 生成 40 面", summ([r["canon_frac_inf"] for r in gen])))
    print(row(f"  橋の割合: 大会・窓内 {len(con_win)} 面",
              summ([r["canon_frac_inf"] for r in con_win])))
    print(row("  橋の割合: 大会 42 面 全体", summ([r["canon_frac_inf"] for r in con])))
    print(row("  橋の割合: （対照）是正前 40 面", summ([r["canon_frac_inf"] for r in old])))
    u3, z3, p3 = mannwhitney_u([r["canon_frac_inf"] for r in gen],
                               [r["canon_frac_inf"] for r in con_win])
    print(f"    Mann-Whitney U（橋の割合・生成 vs 大会窓内）: U={u3:.1f} z={z3:.3f} p={p3:.2e}")

    print("\n  (ii) 全最短経路辺（少なくとも 1 本の最短経路に含まれる辺。経路の選び方に非依存）")
    print(row("    橋の割合: 生成 40 面", summ([r["all_frac_inf"] for r in gen])))
    print(row(f"    橋の割合: 大会・窓内 {len(con_win)} 面",
              summ([r["all_frac_inf"] for r in con_win])))
    print(row("    辺数: 生成 40 面", summ([r["all_n_edges"] for r in gen])))
    print(row(f"    辺数: 大会・窓内 {len(con_win)} 面", summ([r["all_n_edges"] for r in con_win])))

    # ================= 出力 =================
    figs = os.path.join(HERE, "figs")
    os.makedirs(figs, exist_ok=True)
    fig_d_true(gen, con, (lo, hi), os.path.join(figs, "d_true_overlay.png"))
    fig_increment([("生成 40 面", gen, C_BLUE, "-"),
                   (f"大会・窓内 {len(con_win)} 面", con_win, C_ORANGE, "fill"),
                   ("是正前 40 面", old, TEXT_SECONDARY, ":")],
                  os.path.join(figs, "detour_increment.png"))

    def grp(rs, key):
        return summ([r[key] for r in rs])

    out = dict(
        window=[lo, hi],
        generated=gen, contest=con, pre_fix=old,
        summary=dict(
            d_true=dict(
                generated=sG, contest_all=sC, contest_in_window=sW,
                eval=summ([r["d_true"] for r in gen if r["band"] == "eval"]),
                validation=summ([r["d_true"] for r in gen if r["band"] == "validation"]),
                pre_fix=summ([r["d_true"] for r in old]),
                mw_vs_window=dict(U=u1, z=z1, p=p1),
                mw_vs_all=dict(U=u2, z=z2, p=p2),
                n_excluded_low=n_lo, n_excluded_high=n_hi,
                n_in_window_iqr=n_in_w_iqr, edge_lo=edge_lo, edge_hi=edge_hi),
            detour=dict(generated=grp(gen, "detour"), contest_in_window=grp(con_win, "detour"),
                        contest_all=grp(con, "detour"), pre_fix=grp(old, "detour")),
            n_shortest_paths=dict(
                generated=grp(gen, "n_shortest_paths"),
                contest_in_window=grp(con_win, "n_shortest_paths"),
                contest_all=grp(con, "n_shortest_paths"), pre_fix=grp(old, "n_shortest_paths")),
            bridge_fraction=dict(
                generated=grp(gen, "canon_frac_inf"),
                contest_in_window=grp(con_win, "canon_frac_inf"),
                contest_all=grp(con, "canon_frac_inf"), pre_fix=grp(old, "canon_frac_inf"),
                mw_gen_vs_window=dict(U=u3, z=z3, p=p3)),
        ),
    )
    out_path = os.path.join(HERE, "difficulty_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n数値 JSON: {out_path}")
    print(f"図: {figs}/d_true_overlay.png, {figs}/detour_increment.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
