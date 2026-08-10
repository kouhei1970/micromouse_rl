"""
探索戦略 E1（追加探索ループ）の中核
Exploration strategy E1: additional exploration until the shortest path is confirmed.

研究計画書 §7（コミット 8f84e48）で L0 系の標準探索戦略として確定。
ユーザ指摘「探索アルゴリズムが単純すぎる・未探索区域が多く最短経路を見出せていない」
への対応。

■ 理論的根拠（最短経路の確定判定）
未知壁を「通れる」とみなした**楽観地図**の辺集合は、真の迷路の辺集合の上位集合である。
したがって楽観地図での最短距離 D_opt は真の最短距離 D_true の**下界**:

    D_opt ≤ D_true

一方、楽観地図上の最短経路が**確定済み（既知で開いている）の辺だけ**で構成されて
いれば、その経路は真の迷路にも実在するので

    D_true ≤ D_opt

両者を合わせて **D_opt = D_true**。すなわち

    「現在の楽観最短経路上に未知壁が 1 つも残っていない」瞬間に、
    それが真の最短経路であると確定できる。

迷路の他の場所がどれだけ未探索でも、この結論は変わらない（全面探索は理論上不要）。

■ 判定の実装
多始点 BFS を 2 回（始点側 Df、ゴール側 Db）走らせ、D = Db[START] とする。
区画 (x,y) が「ある楽観最短経路上にある」⇔ Df[x,y] + Db[x,y] == D。
そのような区画から**未知壁を隔てて**隣接する区画へ渡る辺のうち、
Df[x,y] + 1 + Db[nx,ny] == D（またはその逆向き）を満たすものが
「**もし開いていたら最短経路に使われうる未知壁**」= relevant である。
relevant が空になった時点で最短経路が確定する。

■ 楽観・悲観の非対称性（要点）
- 往路探索・追加探索は**楽観的**（未知壁 = 通行可）に進む — 探索の効率のため
- **帰路だけ悲観的**（未知壁 = 壁）に既知経路のみで戻る — 未確認の壁を信じて
  帰路に突っ込むと行き止まりに追い込まれるため

本モジュールは壁地図（-1=未知 / 0=壁なし / 1=壁あり）に対する純粋関数として実装し、
L0-a / L0-b / L0-c の 3 方式から共通で使えるようにする。
"""
from collections import deque

UNKNOWN = -1
OPEN = 0
WALL = 1

_DELTA = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}


def wall_state(v_known, h_known, x: int, y: int, nx: int, ny: int) -> int:
    """区画 (x,y)-(nx,ny) 間の壁の状態（UNKNOWN / OPEN / WALL）を返す。"""
    if (nx, ny) == (x + 1, y):
        return int(v_known[x + 1, y])
    if (nx, ny) == (x - 1, y):
        return int(v_known[x, y])
    if (nx, ny) == (x, y + 1):
        return int(h_known[x, y + 1])
    if (nx, ny) == (x, y - 1):
        return int(h_known[x, y])
    raise ValueError(f"({x},{y}) と ({nx},{ny}) は隣接していません")


def passable(v_known, h_known, x, y, nx, ny, pessimistic: bool = False) -> bool:
    """通行可能か。pessimistic=False（楽観）なら未知壁は通行可、True（悲観）なら不可。"""
    s = wall_state(v_known, h_known, x, y, nx, ny)
    if s == WALL:
        return False
    if s == UNKNOWN:
        return not pessimistic
    return True


def bfs_distances(v_known, h_known, width, height, sources, pessimistic: bool = False):
    """sources からの多始点 BFS 距離場（到達不能な区画は含めない）。"""
    dist = {tuple(c): 0 for c in sources}
    q = deque(dist.keys())
    while q:
        x, y = q.popleft()
        for dx, dy in _DELTA.values():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in dist:
                continue
            if passable(v_known, h_known, x, y, nx, ny, pessimistic):
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return dist


def relevant_unknown_walls(v_known, h_known, width, height, start, goals):
    """「もし開いていたら最短経路に使われうる未知壁」の一覧を返す。

    返り値: [((x, y), (nx, ny)), ...]（未知壁を隔てて隣接する区画対）。
    空リストなら**現在の楽観最短経路が真の最短経路として確定**している。
    """
    df = bfs_distances(v_known, h_known, width, height, [start])
    db = bfs_distances(v_known, h_known, width, height, goals)
    if start not in db:
        return []          # ゴールへ到達不能（想定外）。判定不能なので確定扱いにしない
    d_opt = db[start]

    out = []
    for (x, y), fwd in df.items():
        back = db.get((x, y))
        if back is None or fwd + back != d_opt:
            continue        # この区画はどの楽観最短経路上にもない
        for dx, dy in _DELTA.values():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if wall_state(v_known, h_known, x, y, nx, ny) != UNKNOWN:
                continue    # 既知の辺は確認不要
            nb_back = db.get((nx, ny))
            nb_fwd = df.get((nx, ny))
            # この未知壁を「開いている」として使う最短経路が存在するか（両向きを見る）
            if (nb_back is not None and fwd + 1 + nb_back == d_opt) or \
               (nb_fwd is not None and nb_fwd + 1 + back == d_opt):
                out.append(((x, y), (nx, ny)))
    return out


def is_shortest_confirmed(v_known, h_known, width, height, start, goals) -> bool:
    """現在の楽観最短経路が真の最短経路として確定しているか。"""
    return not relevant_unknown_walls(v_known, h_known, width, height, start, goals)


def explore_targets(v_known, h_known, width, height, start, goals):
    """追加探索で向かうべき区画の集合を返す。

    relevant な未知壁に隣接する区画のうち、**現在地から楽観的に到達できるもの**。
    ここへ行って壁を観測すれば、その未知壁が確定して relevant が減る。
    """
    rel = relevant_unknown_walls(v_known, h_known, width, height, start, goals)
    targets = set()
    for a, b in rel:
        targets.add(a)
        targets.add(b)
    return targets


def shortest_path_cells(v_known, h_known, width, height, start, goals,
                         pessimistic: bool = False):
    """start から goals への最短経路（区画列）を 1 本返す。到達不能なら None。
    帰路では pessimistic=True で呼び、**既知の通れる辺だけ**を使う。"""
    goals = set(map(tuple, goals))
    if tuple(start) in goals:
        return [tuple(start)]
    db = bfs_distances(v_known, h_known, width, height, goals, pessimistic)
    if tuple(start) not in db:
        return None
    path = [tuple(start)]
    cur = tuple(start)
    while cur not in goals:
        best = None
        for dx, dy in _DELTA.values():
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if not passable(v_known, h_known, cur[0], cur[1], nxt[0], nxt[1], pessimistic):
                continue
            if nxt in db and db[nxt] == db[cur] - 1:
                best = nxt
                break
        if best is None:
            return None
        path.append(best)
        cur = best
    return path
