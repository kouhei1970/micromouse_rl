"""
research_notes/scripts/check_wall_follower_reachability.py
============================================================
「壁伝い法（記憶を持たない解法。片手を壁につけて進む）は、`mouse/maze6_gen.py` が
作る 6x6 迷路で必ず中央ゴールへ届くのか」を、**迷路の位相（壁のつながり方）だけ**を
見て決定的に判定する。物理シミュレーションは使わない。

壁判定・最短距離・開始方位は mouse/maze6_gen.py の既存ヘルパをそのまま流用する
（本スクリプト側で壁判定ロジックを書き直さない。書き直すと解釈の食い違いが入る）。

使い方:
    .venv/bin/python research_notes/scripts/check_wall_follower_reachability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mouse.maze6_gen import (  # noqa: E402
    GOAL_CELLS,
    GOAL_RING_EDGES,
    SIZE,
    _neighbors,  # 到達可否の自己検査で使う（既存ヘルパを流用）
    cells_open,
    generate_maze,
    initial_heading_deg,
    shortest_distances,
)
from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402

MODE = "loop"  # mouse/maze6_env.py の既定 maze_mode（grep で確認済み。報告に記載）
MAX_STEPS = 100_000

# 使用する seed 帯（厳守: 評価帯 6000-6019・検証帯 7000-7019 は不使用）
SEEDS_20 = list(range(8000, 8020))     # 学習で実際に最初に引かれる 20 面
SEEDS_1000 = list(range(8000, 9000))   # 率を出すための 1000 面

OUT_PATH = REPO_ROOT / "outputs" / "wall_follower_reachability.json"

# 方位角 [deg] → 単位ベクトル (dx, dy)。initial_heading_deg の戻り値の規約
# （mouse/maze6_gen.py 297〜298 行目: 0=+x, 90=+y, 180=-x, 270=-y）と一致させる。
# _neighbors が使う (dx, dy) ∈ {(1,0),(-1,0),(0,1),(0,-1)} = {E, W, N, S} とも一致。
_HEADING_VEC = {0.0: (1, 0), 90.0: (0, 1), 180.0: (-1, 0), 270.0: (0, -1)}
_VEC_NAME = {(1, 0): "E", (0, 1): "N", (-1, 0): "W", (0, -1): "S"}


# ==========================================================================
# 壁伝い法（左手法／右手法）のシミュレータ
# ==========================================================================
def _left_of(d):
    """反時計回りに 90°（数学座標系: (x,y) -> (-y,x)）。E(1,0)->N(0,1) で検算済み。"""
    return (-d[1], d[0])


def _right_of(d):
    """時計回りに 90°（(x,y) -> (y,-x)）。E(1,0)->S(0,-1) で検算済み。"""
    return (d[1], -d[0])


def _back_of(d):
    return (-d[0], -d[1])


def _is_open(v, h, cell, d):
    """cell から方向 d へ 1 歩進めるか。

    d が盤外（迷路の外）を指す場合でも、mouse/maze6_gen.py の _edge_between は
    隣接 2 セルの相対位置だけから辺を決めるため、盤外方向は必ず外周壁の配列要素
    （v[0,:]・v[SIZE,:]・h[:,0]・h[:,SIZE]。いずれも生成時に 1 で固定）へ写像され、
    負インデックスや範囲外アクセスにはならない（手計算で確認済み・下記自己検査でも再確認）。
    """
    nb = (cell[0] + d[0], cell[1] + d[1])
    return cells_open(v, h, cell, nb)


def _step_order(heading, rule):
    """左手法: 左→前→右→後ろ。右手法: 右→前→左→後ろ。"""
    if rule == "left":
        return [_left_of(heading), heading, _right_of(heading), _back_of(heading)]
    return [_right_of(heading), heading, _left_of(heading), _back_of(heading)]


def simulate_wall_follower(v, h, start, start_heading, goal_cells, rule, max_steps=MAX_STEPS):
    """壁伝い法を格子上・決定的に走らせる。

    停止条件:
      (a) セルが goal_cells に入った -> "reached"
      (b) (セル, 向き) の状態が再訪された -> "cycle"（決して到達しない。決定的な
          状態遷移なので、状態が一巡すれば以後は同じ軌道を無限に繰り返す）
      (c) 上限 max_steps 歩 -> "step_limit"（打ち切り。(b) とは区別して報告する）
    """
    cell = start
    heading = start_heading
    path = [cell]
    visited_states = {(cell, heading)}

    if cell in goal_cells:
        return dict(result="reached", steps=0, path=path)

    for step in range(1, max_steps + 1):
        moved = False
        for d in _step_order(heading, rule):
            if _is_open(v, h, cell, d):
                heading = d
                cell = (cell[0] + d[0], cell[1] + d[1])
                moved = True
                break
        # 検証済み迷路は全区画が到達可能（開通数 >= 1）で、直前に通って来た
        # back 方向は必ず開いている。moved=False は本来起こらない設計上の不変条件。
        assert moved, f"壁伝い法が身動きできない状態に陥った: cell={cell} heading={heading}"
        path.append(cell)
        if cell in goal_cells:
            return dict(result="reached", steps=step, path=path)
        state = (cell, heading)
        if state in visited_states:
            return dict(result="cycle", steps=step, path=path)
        visited_states.add(state)
    return dict(result="step_limit", steps=max_steps, path=path)


# ==========================================================================
# 補助診断: ゴール外周壁が迷路の外周壁と連結しているか
# ==========================================================================
def _wall_vertices(edge):
    kind, x, y = edge
    if kind == "v":
        return (x, y), (x, y + 1)
    return (x, y), (x + 1, y)


def _goal_wall_connected_to_perimeter(v, h, gateway):
    """ゴール外周（gateway を除く 7 辺）の壁が、迷路の外周壁と頂点共有で連結しているか。

    壁を「格子点（頂点）を結ぶ線分」とみなし、線分どうしが頂点を共有していれば連結と
    する（Union-Find）。mode="loop" は行き止まりを潰す＝通路網にループ（閉路）を作る
    ため、迷路の壁は必ずしも 1 本の木にはならない。ゴール外周壁が外周壁と非連結なら、
    「外周から手を離さずに壁を辿る」操作ではゴール外周へ到達できない、という構造的説明になる。
    """
    parent = {(x, y): (x, y) for x in range(SIZE + 1) for y in range(SIZE + 1)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for x in range(SIZE + 1):
        for y in range(SIZE):
            if v[x, y] == 1:
                union((x, y), (x, y + 1))
    for x in range(SIZE):
        for y in range(SIZE + 1):
            if h[x, y] == 1:
                union((x, y), (x + 1, y))

    ring_walls = [e for e in GOAL_RING_EDGES if tuple(e) != tuple(gateway)]
    assert len(ring_walls) == 7, f"gateway を除いたゴール外周壁が 7 辺でない: {len(ring_walls)}"
    goal_vertex = _wall_vertices(ring_walls[0])[0]
    perimeter_vertex = (0, 0)  # 外周壁は必ず 1 つの連結成分（生成時に全周固定）
    return find(goal_vertex) == find(perimeter_vertex)


# ==========================================================================
# 1 迷路の評価
# ==========================================================================
def evaluate_maze(seed, mode=MODE):
    m = generate_maze(seed, mode=mode)
    v, h, start, gateway = m["v_walls"], m["h_walls"], m["start"], m["gateway"]
    goal_cells = set(tuple(c) for c in m["goal_cells"])
    heading_deg = initial_heading_deg(v, h, start)
    start_dir = _HEADING_VEC[heading_deg]

    left = simulate_wall_follower(v, h, start, start_dir, goal_cells, "left")
    right = simulate_wall_follower(v, h, start, start_dir, goal_cells, "right")

    # 到達した経路は全ての 1 歩が実際に壁を抜けていないか cells_open で再検査
    for res in (left, right):
        if res["result"] == "reached":
            for a, b in zip(res["path"], res["path"][1:]):
                assert cells_open(v, h, a, b), (
                    f"seed={seed}: 壁を抜けている疑いのある移動 {a}->{b}")

    dist = shortest_distances(v, h, targets=GOAL_CELLS)
    connected = _goal_wall_connected_to_perimeter(v, h, gateway)

    return dict(
        seed=seed, mode=mode, start=list(start),
        heading_deg=heading_deg, heading_name=_VEC_NAME[start_dir],
        gateway=list(gateway), d0=dist[start],
        left_result=left["result"], left_steps=left["steps"],
        left_visited_cells=len(set(left["path"])),
        right_result=right["result"], right_steps=right["steps"],
        right_visited_cells=len(set(right["path"])),
        goal_wall_connected_to_perimeter=connected,
    )


# ==========================================================================
# 自己検査: 1 面を手で追って検算する（seed=8000 の最初の数歩）
# ==========================================================================
def manual_trace(seed, n_steps=6):
    m = generate_maze(seed, mode=MODE)
    v, h, start = m["v_walls"], m["h_walls"], m["start"]
    heading_deg = initial_heading_deg(v, h, start)
    heading = _HEADING_VEC[heading_deg]
    lines = [f"[検算] seed={seed} start={start} heading={_VEC_NAME[heading]}({heading_deg} deg)"]
    cell = start
    for i in range(n_steps):
        order = _step_order(heading, "left")
        checks = []
        chosen = None
        for d in order:
            nb = (cell[0] + d[0], cell[1] + d[1])
            open_ = _is_open(v, h, cell, d)
            checks.append(f"{_VEC_NAME[d]}->{'開' if open_ else '壁'}")
            if open_ and chosen is None:
                chosen = d
        lines.append(f"  step{i+1}: cell={cell} 試行[{', '.join(checks)}] 選択={_VEC_NAME[chosen]}")
        heading = chosen
        cell = (cell[0] + chosen[0], cell[1] + chosen[1])
    lines.append(f"  -> 到達セル列: start={start} 以降 {n_steps} 歩で {cell} 付近")
    return "\n".join(lines)


# ==========================================================================
# main
# ==========================================================================
def main():
    print(describe_seeds(SEEDS_20 + SEEDS_1000, namespace="maze6"))
    print("[bands] 使用 seed = 8000-8999（学習帯）／予約帯 6000-6019・7000-7019 は不使用")
    # common/seed_bands.py の正本（mouse/maze6_env._RESERVED_MAZE_SEEDS）と突き合わせて
    # 予約帯を 1 つも含んでいないことを機械的に確認する（purpose='train' は 'free' のみ許可）。
    assert_seeds_allowed(SEEDS_20, namespace="maze6", purpose="train")
    assert_seeds_allowed(SEEDS_1000, namespace="maze6", purpose="train")
    print(f"[mode] 使用する maze_mode = {MODE!r}（mouse/maze6_env.py の既定値。grep で確認）")

    print("\n=== (a) 20 面の一覧 ===")
    header = (f"{'seed':>6} {'D0':>4} {'左手法':>10} {'左歩数':>8} "
              f"{'右手法':>10} {'右歩数':>8} {'ゴール壁-外周連結':>16}")
    print(header)
    results_20 = []
    for seed in SEEDS_20:
        r = evaluate_maze(seed)
        results_20.append(r)
        print(f"{r['seed']:>6} {r['d0']:>4} {r['left_result']:>10} {r['left_steps']:>8} "
              f"{r['right_result']:>10} {r['right_steps']:>8} "
              f"{str(r['goal_wall_connected_to_perimeter']):>16}")

    print("\n=== (b) 1000 面の集計 ===")
    results_1000 = [evaluate_maze(seed) for seed in SEEDS_1000]
    n = len(results_1000)
    left_reach = sum(1 for r in results_1000 if r["left_result"] == "reached")
    right_reach = sum(1 for r in results_1000 if r["right_result"] == "reached")
    either_reach = sum(1 for r in results_1000
                       if r["left_result"] == "reached" or r["right_result"] == "reached")
    both_fail = sum(1 for r in results_1000
                    if r["left_result"] != "reached" and r["right_result"] != "reached")
    disconnected = sum(1 for r in results_1000 if not r["goal_wall_connected_to_perimeter"])
    left_cycle = sum(1 for r in results_1000 if r["left_result"] == "cycle")
    right_cycle = sum(1 for r in results_1000 if r["right_result"] == "cycle")
    left_limit = sum(1 for r in results_1000 if r["left_result"] == "step_limit")
    right_limit = sum(1 for r in results_1000 if r["right_result"] == "step_limit")
    print(f"  n = {n}")
    print(f"  左手法 到達率     = {left_reach}/{n} = {left_reach/n:.3f}"
          f"（巡回未到達 {left_cycle}／上限打ち切り {left_limit}）")
    print(f"  右手法 到達率     = {right_reach}/{n} = {right_reach/n:.3f}"
          f"（巡回未到達 {right_cycle}／上限打ち切り {right_limit}）")
    print(f"  左右いずれかで到達 = {either_reach}/{n} = {either_reach/n:.3f}")
    print(f"  左右とも未到達     = {both_fail}/{n} = {both_fail/n:.3f}")
    print(f"  ゴール壁が外周と非連結の率 = {disconnected}/{n} = {disconnected/n:.3f}")

    # ---------------- 自己検査 ----------------
    print("\n=== 自己検査 ===")
    reached_all = [r for r in results_1000 if r["left_result"] == "reached"] + \
                  [r for r in results_1000 if r["right_result"] == "reached"]
    if reached_all:
        steps_list = [r["left_steps"] for r in results_1000 if r["left_result"] == "reached"] + \
                     [r["right_steps"] for r in results_1000 if r["right_result"] == "reached"]
        visited_list = [r["left_visited_cells"] for r in results_1000 if r["left_result"] == "reached"] + \
                       [r["right_visited_cells"] for r in results_1000 if r["right_result"] == "reached"]
        print(f"  空振りでない証拠: 到達件数(左右合算)={len(steps_list)}, "
              f"歩数レンジ=[{min(steps_list)}, {max(steps_list)}], "
              f"訪問セル数レンジ=[{min(visited_list)}, {max(visited_list)}]（いずれも 1 セル/1 歩超あり）")
    else:
        print("  到達した迷路が 1 件もない（要確認）")

    diff = [r for r in results_1000
           if r["left_result"] != r["right_result"] or r["left_steps"] != r["right_steps"]]
    print(f"  左右で結果が異なる迷路: {len(diff)}/{n} 件"
          f"（例: seed={diff[0]['seed']}）" if diff else "  左右で結果が異なる迷路: 0 件（要注意）")

    if len(diff) == 0:
        print("  [警告] 左右が全て同一 -> 実装の取り違えを疑い、1 面を手で検算する:")
        print(manual_trace(SEEDS_20[0]))
    else:
        print("  検算用トレース（seed=8000 左手法 最初の 6 歩）:")
        print(manual_trace(SEEDS_20[0]))

    # ---------------- 保存 ----------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        mode=MODE, max_steps=MAX_STEPS,
        seed_bands=dict(used="8000-8999", reserved_eval="6000-6019", reserved_validation="7000-7019"),
        results_20=results_20,
        summary_1000=dict(
            n=n, left_reach_rate=left_reach / n, right_reach_rate=right_reach / n,
            either_reach_rate=either_reach / n, both_fail_rate=both_fail / n,
            goal_wall_disconnected_rate=disconnected / n,
            left_cycle=left_cycle, right_cycle=right_cycle,
            left_step_limit=left_limit, right_step_limit=right_limit,
        ),
        results_1000=results_1000,
    )
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
