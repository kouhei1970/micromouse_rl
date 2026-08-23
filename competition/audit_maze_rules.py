"""
評価迷路の競技規定適合性 検査（検査のみ・是正はしない）
Audit the frozen evaluation mazes against classic micromouse rules.

2026-08-10、ユーザ指摘「迷路がクラシック競技の本当のルールに準拠していない。
ゴールの規格が少なくとも違う」を受けて実施する全数検査。

検査する規定と出典:
- IEEE R2 SAC 2020 規定 4.3: "The destination square has only one gateway."
  （ゴール区画の入口は 1 箇所のみ）／スタート区画は四隅のいずれかで 3 方向が壁
  https://attend.ieee.org/r2sac-2020/wp-content/uploads/sites/175/2020/01/MicroMouse_Rules_2020.pdf
- IEEE R2 SAC 2020 規定 4.5: 複数経路は許容され想定されるべき。ゴールは
  壁づたい走行では発見できない位置に置かれる
- NTF クラシック規定 注意 9: 迷路の終点となる 4 区画内には壁や柱は存在しない
- NTF クラシック規定 2-4: 終点の中央を除いた全格子点に少なくとも 1 つの壁が接する／
  外周の壁は全て存在する

使い方:
    .venv/bin/python -m competition.audit_maze_rules
"""
import glob
import json
import os
import sys
from collections import deque

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

GOAL_CELLS = {(7, 7), (7, 8), (8, 7), (8, 8)}
CENTER_POST = (8, 8)  # ゴール 2x2 の中央にあたる格子点


def _open(v, h, a, b):
    """セル a,b 間が開通しているか（隣接前提）。"""
    (ax, ay), (bx, by) = a, b
    if ax == bx:
        return h[ax, max(ay, by)] == 0
    return v[max(ax, bx), ay] == 0


def goal_gateway_count(v, h):
    """ゴール 2x2 の外周 8 辺のうち開いている数（＝入口の数）。"""
    n = 0
    edges = []
    for (cx, cy) in sorted(GOAL_CELLS):
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in GOAL_CELLS:
                continue  # 内部の辺は外周ではない
            if not (0 <= nx < 16 and 0 <= ny < 16):
                continue  # 外周壁（ゴールは中央なので発生しない）
            if _open(v, h, (cx, cy), (nx, ny)):
                n += 1
                edges.append(((cx, cy), (nx, ny)))
    return n, edges


def goal_interior_walls(v, h):
    """ゴール 4 区画の内部（2x2 の内壁 4 枚）に残っている壁の数。"""
    return int(v[8, 7] == 1) + int(v[8, 8] == 1) + int(h[7, 8] == 1) + int(h[8, 8] == 1)


def isolated_posts(v, h, exclude_center=True):
    """どの壁とも接していない格子点の数（ゴール中央は除外可）。"""
    n = 0
    for px in range(17):
        for py in range(17):
            if exclude_center and (px, py) == CENTER_POST:
                continue
            attached = False
            if py < 16 and v[px, py] == 1: attached = True
            if py > 0 and v[px, py - 1] == 1: attached = True
            if px < 16 and h[px, py] == 1: attached = True
            if px > 0 and h[px - 1, py] == 1: attached = True
            if not attached:
                n += 1
    return n


def center_post_attached_walls(v, h):
    """ゴール中央の格子点に接している壁の数（NTF 規定では 0 = 柱も壁もない）。"""
    px, py = CENTER_POST
    return (int(v[px, py] == 1) + int(v[px, py - 1] == 1)
            + int(h[px, py] == 1) + int(h[px - 1, py] == 1))


def independent_cycles(v, h):
    """独立閉路数 = 開通辺数 − セル数 + 1（完全迷路なら 0）。"""
    open_edges = int((v[1:16, :] == 0).sum() + (h[:, 1:16] == 0).sum())
    return open_edges - 256 + 1, open_edges


def outer_walls_complete(v, h):
    return bool(v[0, :].all() and v[16, :].all() and h[:, 0].all() and h[:, 16].all())


def start_cell_walls(v, h):
    """スタート区画 (0,0) の壁枚数（規定: 3 方向が壁 = 開口 1）。"""
    walls = int(v[0, 0] == 1) + int(v[1, 0] == 1) + int(h[0, 0] == 1) + int(h[0, 1] == 1)
    return walls


def wall_follow_reaches_goal(v, h, hand="left"):
    """左手法/右手法（壁づたい走行）でゴールに到達できるか。

    スタート (0,0)・北向きから、(セル, 向き) の状態遷移を追う。同じ状態に
    戻ったら閉路に入って永久にゴールへ到達しないと判定する。
    """
    # 向き: 0=東, 1=北, 2=西, 3=南（classic.maze_map.Direction と同じ
    # E=0,N=1,W=2,S=3 の反時計回り順。2026-08-23 付け替え。本関数は
    # classic.maze_map には依存しないが、方位インデックスの意味を全体で
    # 揃えるためここも合わせる。反時計回り順では +1 が左90°になる）。
    d_vec = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    # 左手法: 左→前→右→後 の順に試す。右手法はその鏡像
    order = [1, 0, -1, 2] if hand == "left" else [-1, 0, 1, 2]

    def can_go(cell, d):
        cx, cy = cell
        dx, dy = d_vec[d]
        nx, ny = cx + dx, cy + dy
        if not (0 <= nx < 16 and 0 <= ny < 16):
            return None
        return (nx, ny) if _open(v, h, cell, (nx, ny)) else None

    cell, head = (0, 0), 1  # 北向きで出発（d_vec[1]=北）
    seen = set()
    for _ in range(100000):
        if cell in GOAL_CELLS:
            return True
        state = (cell, head)
        if state in seen:
            return False  # 閉路に入った
        seen.add(state)
        for turn in order:
            nd = (head + turn) % 4
            nxt = can_go(cell, nd)
            if nxt is not None:
                cell, head = nxt, nd
                break
        else:
            return False
    return False


def bfs_goal_reachable(v, h):
    seen = {(0, 0)}
    dq = deque([(0, 0)])
    while dq:
        c = dq.popleft()
        if c in GOAL_CELLS:
            return True
        cx, cy = c
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            n = (cx + dx, cy + dy)
            if 0 <= n[0] < 16 and 0 <= n[1] < 16 and n not in seen and _open(v, h, c, n):
                seen.add(n)
                dq.append(n)
    return False


def xml_has_center_post(xml_path):
    """生成 XML にゴール中央の柱 geom が描画されているか。"""
    if not os.path.exists(xml_path):
        return None
    return f'name="post_{CENTER_POST[0]}_{CENTER_POST[1]}"' in open(xml_path).read()


def xml_ring_posts_present(xml_path):
    """ゴール外周 8 格子点に柱 geom が描画されているか（ユーザ確認 2026-08-10:
    「中央のゴール区画の中心一本だけ柱はありません」= 外周 8 点には柱がある）。"""
    if not os.path.exists(xml_path):
        return None
    src = open(xml_path).read()
    ring_posts = [(7, 7), (8, 7), (9, 7), (7, 8), (9, 8), (7, 9), (8, 9), (9, 9)]
    return sum(1 for (px, py) in ring_posts if f'name="post_{px}_{py}"' in src)


def audit_one(npz_path):
    d = np.load(npz_path)
    v, h = d["v_walls"], d["h_walls"]
    seed = int(d["seed"])
    gw, _edges = goal_gateway_count(v, h)
    cyc, open_edges = independent_cycles(v, h)
    return dict(
        seed=seed,
        goal_gateways=gw,
        goal_interior_walls=goal_interior_walls(v, h),
        center_post_walls=center_post_attached_walls(v, h),
        xml_center_post=xml_has_center_post(str(npz_path).replace(".npz", ".xml")),
        xml_ring_posts=xml_ring_posts_present(str(npz_path).replace(".npz", ".xml")),
        start_walls=start_cell_walls(v, h),
        isolated_posts=isolated_posts(v, h),
        cycles=cyc,
        open_edges=open_edges,
        outer_ok=outer_walls_complete(v, h),
        goal_reachable=bfs_goal_reachable(v, h),
        left_hand_reaches=wall_follow_reaches_goal(v, h, "left"),
        right_hand_reaches=wall_follow_reaches_goal(v, h, "right"),
    )


def main():
    rows = [audit_one(f) for f in sorted(glob.glob(os.environ.get("AUDIT_MAZE_DIR", "competition/mazes/eval") + "/maze_*.npz"))]
    print("=" * 100)
    print("評価迷路 20 面 競技規定 適合性検査（検査のみ・是正なし）")
    print("=" * 100)
    print(f"{'seed':>5} {'ゴール入口':>9} {'ゴール内壁':>9} {'中央柱壁':>8} {'XML柱':>6} "
          f"{'ｽﾀｰﾄ壁':>7} {'孤立柱':>6} {'閉路':>5} {'外周':>5} {'左手法':>7} {'右手法':>7}")
    for r in rows:
        print(f"{r['seed']:>5} {r['goal_gateways']:>9} {r['goal_interior_walls']:>9} "
              f"{r['center_post_walls']:>8} {str(r['xml_center_post']):>6} {r['start_walls']:>7} "
              f"{r['isolated_posts']:>6} {r['cycles']:>5} {str(r['outer_ok']):>5} "
              f"{str(r['left_hand_reaches']):>7} {str(r['right_hand_reaches']):>7}")

    n = len(rows)
    gw = [r["goal_gateways"] for r in rows]
    print("\n" + "=" * 100)
    print("集計")
    print("=" * 100)
    print(f"1. ゴール入口数: 中央値 {np.median(gw):.0f}、範囲 {min(gw)}〜{max(gw)}。"
          f"**規定(1 箇所)に適合 {sum(1 for x in gw if x == 1)}/{n} 面**")
    print(f"   内訳: " + ", ".join(f"{k}箇所={sum(1 for x in gw if x == k)}面"
                                    for k in sorted(set(gw))))
    print(f"2. XML にゴール中央の柱: {sum(1 for r in rows if r['xml_center_post'])}/{n} 面"
          f"（規定では中央のみ柱なしが正 → 0/{n} が適合）")
    print(f"2b. XML のゴール外周 8 格子点の柱: 全 8 本ある面 "
          f"{sum(1 for r in rows if r['xml_ring_posts'] == 8)}/{n}"
          f"（実測本数: {sorted(set(r['xml_ring_posts'] for r in rows))}）")
    print(f"3. ゴール 4 区画内部の壁: 全 0 枚 {sum(1 for r in rows if r['goal_interior_walls'] == 0)}/{n} 面")
    print(f"4. スタート区画の壁 3 枚（開口 1）: {sum(1 for r in rows if r['start_walls'] == 3)}/{n} 面"
          f"（実測の壁枚数: {sorted(set(r['start_walls'] for r in rows))}）")
    print(f"5. 孤立柱ゼロ（中央除く）: {sum(1 for r in rows if r['isolated_posts'] == 0)}/{n} 面")
    cyc = [r["cycles"] for r in rows]
    print(f"6. 独立閉路数: 中央値 {np.median(cyc):.0f}、範囲 {min(cyc)}〜{max(cyc)}"
          f"（完全迷路なら 0。ゴール開放で増えた分）")
    print(f"7. 壁づたい走行でゴール到達: 左手法 {sum(1 for r in rows if r['left_hand_reaches'])}/{n} 面、"
          f"右手法 {sum(1 for r in rows if r['right_hand_reaches'])}/{n} 面"
          f"（IEEE 4.5 は到達**できない**配置を要求 → 0/{n} が適合）")
    print(f"8. 外周壁が全て存在: {sum(1 for r in rows if r['outer_ok'])}/{n} 面")
    print(f"（参考）ゴール到達可能: {sum(1 for r in rows if r['goal_reachable'])}/{n} 面")

    out = os.path.join(os.environ.get("AUDIT_MAZE_DIR", "competition/mazes/eval"), "rule_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n結果 JSON: {out}")


if __name__ == "__main__":
    main()
