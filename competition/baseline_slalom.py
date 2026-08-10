"""
competition/baseline_slalom.py
================
古典ベースライン方策 SlalomPolicy（L0-c: スラローム走行）。

docs/L0PLUS_DESIGN.md §2・§3・§9・§10 の設計。足立法（flood-fill）による経路決定
（どのセルへ進むか）は `competition/baseline_classical.py`（L0-a）・
`competition/baseline_straightrun.py`（L0-b）と完全に同一のロジックを流用する。
**L0-a/L0-b との違いは走行部のみ**: L0-a/L0-b は「超信地旋回走行」（曲がる
たびに停止して機体中心まわりに回頭する）のに対し、L0-c は**曲がるところでも
停止しない**スラローム走行（区画中心の折れ線を「直線＋90°ターンの円弧
（接線連続）」に置き換えた軌道を、速度計画に沿って一括で追従する）。

--------------------------------------------------------------------------
全体構成（設計文書の節番号に対応）
--------------------------------------------------------------------------
1. §2.1 軌道生成: 区画中心の折れ線 → 直線＋90°円弧（接線連続）。円弧半径 R は
   候補 {45,60,75,90} mm から、壁への余裕 15mm 以上を満たす最大値を
   幾何計算＋実際の壁配列を模した数値検査の両方で選ぶ（`select_arc_radius`）。
   S字（連続する逆向きターン）は中間直線が短くて干渉する場合に半径を下げる
   （`fit_corner_radii`）。
2. §2.2 速度計画: 旋回速度 v_turn=sqrt(a_lat・R)、直線は台形速度（次のターンの
   v_turn を終端速度に）。前後進台形加減速の限界速度包絡線を作ってから、
   前方・後方2パスで加速度制約を反映する（`build_speed_profile`。CNC等で
   広く使われる「時間最適速度包絡線」の標準的な3パス法）。
3. §2.3 探索走行と最短走行の区別: 探索走行は未知壁のため次の1〜2区画ぶんだけ
   軌道生成し v<=0.8m/s に抑える。最短走行（初回ゴール到達後）は既知地図で
   目標セルまでの全経路を一括生成する。
4. §3・§10 制御則: 経路追従は Stanley 標準形（速度依存ゲインで低速時の
   横偏差項を飽和させ、L0-a/L0-b で確認されたチャタリングの根因＝低速域で
   横偏差・方位偏差の補正が拮抗する現象を回避する）。速度指令には下限
   （クリープ）を明示し、目標到達（ゴール）以外では停止しない。車輪速度制御は
   逆モデル前向き補償＋PI（アンチワインドアップ付き）。

--------------------------------------------------------------------------
足立法（flood-fill）— L0-a/L0-b と同一ロジック
--------------------------------------------------------------------------
壁地図 v_walls_known (W+1,H) / h_walls_known (W,H+1) を内部状態として持つ
（値は {0: 壁なし, 1: 壁あり, -1: 未知}）。未知壁は経路決定では楽観的に
「壁なし」とみなす（multi-source BFS）。直進連続の先読み判定（§7 由来、
L0-b で新設）専用に、未知壁を保守的に「壁あり」とみなす `_known_open` も
流用する（スラローム軌道生成でも「既知区間だけを高速走行してよい」という
同じ考え方を使う）。

--------------------------------------------------------------------------
走行部の状態機械
--------------------------------------------------------------------------
act() は 100Hz で呼ばれ、内部状態 "DRIVE"（経路追従）と "TURN_INPLACE"
（超信地旋回。ダブりを避けるため以下の2つの場合のみ使う）を遷移する:
  (a) 現在セルに到達した瞬間、足立法が選ぶ次方向が現在の進行方向と異なり、
      かつそのセルが軌道の先頭（先読みの直前の脚で滑らかに曲がり始める
      余地が無い）場合 — 90°または180°の場その場旋回で向きを合わせてから
      改めてスラローム軌道を張り直す。
  (b) 90°ターンとして円弧で処理できない180°（行き止まり折り返し）。
どちらも「稀な例外処理」であり、通常の経路追従は円弧つき軌道を直線的に
たどるだけで曲がるところでも停止しない（L0-c の主旨）。

DRIVE 中は、現在セルが新しいセルに変わるたび（＝区画中心付近を通過する
たび。停止しないため「到達して停止」ではなく「通過」で検出する）に
`_do_sense` で壁を観測し、探索走行なら毎回、最短走行（初回ゴール到達後）
なら目標セル到達時のみ軌道を張り直す（§2.3）。

ゴールでは 2 段階判定（competition/evaluator.py の body_fully_inside 由来）
に合わせ、ゴール区画の中心を目標点とする軌道を生成し終端速度 0 で停止する
（他のセル通過ではいっさい停止しない）。

位置推定は L0-a/L0-b と同じく特権情報（`privileged_pose`）を使用し、
requires_privileged=True。
"""
import math
from collections import deque

import numpy as np

from competition.evaluator import goal_cells, pos_to_cell
from competition.policy_interface import MousePolicy
from mouse.mjcf import POST_SIZE, WALL_THICKNESS

# ==========================================================================
# 4 方位・座標規約（L0-a/L0-b と完全に同一）
# ==========================================================================
DIRS = ("N", "E", "S", "W")
_DELTA = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
_HEADING_RAD = {"N": math.pi / 2, "E": 0.0, "S": -math.pi / 2, "W": math.pi}
_CW_NEXT = {"N": "E", "E": "S", "S": "W", "W": "N"}


def _wrap_pi(angle: float) -> float:
    """角度を [-pi, pi) にラップする。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def _dir_vec(d: str):
    dx, dy = _DELTA[d]
    return float(dx), float(dy)


def _left_normal(vx: float, vy: float):
    """ベクトル(vx,vy)を反時計回りに90°回した左法線。"""
    return -vy, vx


def _turn_kind(dir_in: str, dir_out: str):
    """方位文字2つからターン種別を返す。

    (kind, turn_sign) を返す。kind は 'straight'（直進）/'left'（左＝反時計回り
    90°）/'right'（右＝時計回り90°）/'reverse'（180°、円弧では表現できない）。
    turn_sign は円弧の向き（+1=反時計回り、-1=時計回り、直進/180°ではNone）。

    DIRS=("N","E","S","W") の並びは時計回り（_CW_NEXT参照）なので、
    インデックス差分 diff=(idx(out)-idx(in))%4 から直接判定できる
    （diff=1→右折、diff=3→左折、diff=2→180°、diff=0→直進）。
    """
    i0, i1 = DIRS.index(dir_in), DIRS.index(dir_out)
    diff = (i1 - i0) % 4
    if diff == 0:
        return "straight", 0
    if diff == 1:
        return "right", -1
    if diff == 3:
        return "left", 1
    return "reverse", None


# ==========================================================================
# §2.1 軌道生成: コーナー円弧の幾何
# ==========================================================================
def corner_arc_params(corner_xy, dir_in: str, dir_out: str, R: float):
    """90°コーナー（区画中心 corner_xy で dir_in→dir_out に曲がる）の円弧
    パラメータを返す。円弧は dir_in 方向の直線・dir_out 方向の直線の両方に
    接線連続（G1連続）でつながる（設計文書§2.1）。

    導出（本ファイル実装時の幾何導出。区画180mm・R<=90mmで常に区画内に収まる）:
      turn_sign=s（+1=左折/CCW, -1=右折/CW）として、
        接線開始点 tangent_in  = corner - R*dir_in
        接線終了点 tangent_out = corner + R*dir_out
        中心 O = tangent_in + R * s * left_normal(dir_in)
      円弧上の点は P(l) = O + R*(cos(phi0 + s*l/R), sin(phi0 + s*l/R))
      （l: 円弧に沿った弧長、0<=l<=R*pi/2）、方位は phi0 + s*l/R + s*pi/2。
      （両方の接線方向から独立に導出しても同じ O・同じ関係式になることを
      本実装時に手計算で確認済み: O - tangent_out も同じ形 R*s*left_normal(dir_out)
      になる対称性がある）

    Returns:
        dict(center, R, phi_start, turn_sign, tangent_in, tangent_out, arclen)
    """
    kind, s = _turn_kind(dir_in, dir_out)
    if kind not in ("left", "right"):
        raise ValueError(f"corner_arc_params は90°ターン専用です: dir_in={dir_in}, dir_out={dir_out} ({kind})")

    dvi = _dir_vec(dir_in)
    dvo = _dir_vec(dir_out)
    ln_in = _left_normal(*dvi)
    normal_toward_center = (s * ln_in[0], s * ln_in[1])

    tangent_in = (corner_xy[0] - R * dvi[0], corner_xy[1] - R * dvi[1])
    tangent_out = (corner_xy[0] + R * dvo[0], corner_xy[1] + R * dvo[1])
    center = (tangent_in[0] + R * normal_toward_center[0],
              tangent_in[1] + R * normal_toward_center[1])
    phi_start = math.atan2(tangent_in[1] - center[1], tangent_in[0] - center[0])

    return dict(center=center, R=R, phi_start=phi_start, turn_sign=s,
                tangent_in=tangent_in, tangent_out=tangent_out,
                arclen=R * math.pi / 2.0)


def arc_pose_at(params: dict, l: float):
    """corner_arc_params() の結果と弧長 l（0<=l<=arclen）から (x,y,heading,curvature) を返す。"""
    R = params["R"]
    s = params["turn_sign"]
    cx, cy = params["center"]
    phi = params["phi_start"] + s * l / R
    x = cx + R * math.cos(phi)
    y = cy + R * math.sin(phi)
    heading = _wrap_pi(phi + s * math.pi / 2.0)
    curvature = s / R
    return x, y, heading, curvature


# ==========================================================================
# §2.1 S字干渉時の半径整定（中間直線が短い連続ターンの半径を下げる）
# ==========================================================================
def fit_corner_radii(waypoints_xy, preferred_R: float, margin_m: float = 0.01, n_pass: int = 8):
    """区画中心の折れ線 waypoints_xy（内部要素 1..len-2 がコーナー）に対し、
    各コーナーの円弧半径を preferred_R から出発し、隣接コーナー間の直線区間
    長 L から接線トリム距離の和が (L - margin_m) を超えないよう反復的に
    縮小する（設計文書§2.1「S字は円弧2つの連結、中間直線が短くて干渉する
    場合はRを下げる」）。

    90°ターンの接線トリム距離は R そのもの（tan(90°/2)=1）なので、
    コーナー i, i+1 が同じ直線区間（長さ L）を挟むとき、必要条件は
    r_i + r_{i+1} <= L - margin_m。超える場合は両者を比例縮小する。
    区間長 L は最小でも1区画(=cell_size)あるため、隣接ターン同士の
    干渉はあっても、縮小後の半径が候補最小値を大きく下回ることは
    通常ない（本ファイルの検証テストで実測して確認する）。

    Returns:
        corner_radii: list[float]、長さ = len(waypoints_xy)-2（内部要素の数）
    """
    n_corners = len(waypoints_xy) - 2
    if n_corners <= 0:
        return []
    r = [float(preferred_R)] * n_corners
    # 区間長 L[k]: waypoints_xy[k] と waypoints_xy[k+1] の間（k=0..len-2）。
    # 区間 L[k] の両端にあるコーナーは (k-1番目のコーナー: 出射側) と
    # (k番目のコーナー: 入射側)。端の区間（最初/最後）は片側のみコーナー。
    L = [math.hypot(waypoints_xy[k + 1][0] - waypoints_xy[k][0],
                     waypoints_xy[k + 1][1] - waypoints_xy[k][1])
         for k in range(len(waypoints_xy) - 1)]

    for _ in range(n_pass):
        changed = False
        for seg_idx, length in enumerate(L):
            left_r = r[seg_idx - 1] if seg_idx - 1 >= 0 else 0.0
            right_r = r[seg_idx] if seg_idx < n_corners else 0.0
            total = left_r + right_r
            budget = max(length - margin_m, 0.0)
            if total > budget and total > 1e-12:
                scale = budget / total
                if seg_idx - 1 >= 0:
                    r[seg_idx - 1] *= scale
                if seg_idx < n_corners:
                    r[seg_idx] *= scale
                changed = True
        if not changed:
            break
    return r


# ==========================================================================
# §2.1 円弧半径の候補選定: 壁への余裕15mm以上を満たす最大のRを選ぶ
# ==========================================================================
def _point_box_dist(px, py, cx, cy, hx, hy):
    """点(px,py)から軸並行の矩形（中心cx,cy・半幅hx,hy）までの最短距離。"""
    dx = max(abs(px - cx) - hx, 0.0)
    dy = max(abs(py - cy) - hy, 0.0)
    return math.hypot(dx, dy)


def _isolated_turn_wall_boxes(cell_size: float, wall_thickness: float, post_size: float):
    """孤立した単一90°ターン（両側とも十分長い直線回廊）の最悪ケース壁配置を
    mouse/mjcf.py の壁生成規則と同じ式で構築する（設計文書§2.1「実際の壁配列
    での数値検査」）。

    ローカル座標: ターン区画が [0,W]x[0,W]、進入回廊が南（区画(0,-1)）・
    退出回廊が東（区画(1,0)）の北→東ターン（右折）を想定する。左折・他の
    3方位の組み合わせは合同変換（回転・鏡映）で同じ形状になるため、この
    1パターンの数値検査で全ターン種別を代表できる。
    turn cell: 南開放(進入)・東開放(退出)・北閉鎖・西閉鎖。
    進入回廊・退出回廊は単一幅の直線回廊（両側壁あり）と仮定する
    （区画は必ず1区画幅なので、これが実現しうる最も狭い配置＝最悪ケース）。
    """
    W = cell_size
    half_wt = wall_thickness / 2.0
    half_ps = post_size / 2.0
    boxes = []

    def add_vwall(gx, gy):
        boxes.append((gx * W, gy * W + W / 2.0, half_wt, W / 2.0 - half_ps))

    def add_hwall(gx, gy):
        boxes.append((gx * W + W / 2.0, gy * W, W / 2.0 - half_ps, half_wt))

    def add_post(gx, gy):
        boxes.append((gx * W, gy * W, half_ps, half_ps))

    # turn cell (0,0): 西壁・北壁
    add_vwall(0, 0)
    add_hwall(0, 1)
    # entry corridor (0,-1): 西壁・東壁（直進回廊）
    add_vwall(0, -1)
    add_vwall(1, -1)
    # exit corridor (1,0): 南壁・北壁（直進回廊）
    add_hwall(1, 0)
    add_hwall(1, 1)
    # 周辺の格子点の柱（壁の有無に関わらず常に存在。JA_ENGINEERING_TERMS.md「柱」）
    for gx in range(0, 3):
        for gy in range(-1, 2):
            add_post(gx, gy)
    return boxes


def _clearance_for_radius(R: float, footprint, cell_size: float,
                           wall_thickness: float, post_size: float,
                           lead: float = 0.15, n_samples: int = 600) -> float:
    """半径Rの単一90°ターン軌道（進入直線lead[m] + 円弧 + 退出直線lead[m]）を
    走行したときの、車体外周（footprintの4隅を姿勢に応じて回転した点で近似）
    から周辺の壁・柱（_isolated_turn_wall_boxes）までの最小クリアランス[m]を返す。
    """
    x_min, x_max, y_min, y_max = footprint
    walls = _isolated_turn_wall_boxes(cell_size, wall_thickness, post_size)
    xc = yc = cell_size / 2.0

    corner = corner_arc_params((xc, yc), "N", "E", R)
    s_arc = corner["arclen"]
    s_total = lead + s_arc + lead

    min_clear = math.inf
    for i in range(n_samples):
        s = s_total * i / (n_samples - 1)
        if s < lead:
            d = lead - s
            x, y = xc, (yc - R) - d
            heading = math.pi / 2.0
        elif s < lead + s_arc:
            x, y, heading, _ = arc_pose_at(corner, s - lead)
        else:
            d = s - (lead + s_arc)
            x, y = (xc + R) + d, yc
            heading = 0.0
        c, si = math.cos(heading), math.sin(heading)
        for lx in (x_min, x_max):
            for ly in (y_min, y_max):
                wx = x + lx * c - ly * si
                wy = y + lx * si + ly * c
                for (bx, by, hx, hy) in walls:
                    d2 = _point_box_dist(wx, wy, bx, by, hx, hy)
                    if d2 < min_clear:
                        min_clear = d2
    return min_clear


def select_arc_radius(footprint, cell_size: float, wall_thickness: float = WALL_THICKNESS,
                       post_size: float = POST_SIZE, candidates_mm=(45, 60, 75, 90),
                       margin_m: float = 0.015):
    """候補円弧半径から、孤立した単一90°ターンで壁余裕 margin_m 以上を
    満たす最大のRを選ぶ（設計文書§2.1）。

    Returns:
        (selected_R[m], table) where table = [(R_mm, clearance_m), ...]
    """
    table = []
    for r_mm in candidates_mm:
        R = r_mm / 1000.0
        clr = _clearance_for_radius(R, footprint, cell_size, wall_thickness, post_size)
        table.append((r_mm, clr))
    ok = [(r_mm, clr) for r_mm, clr in table if clr >= margin_m]
    if ok:
        chosen_mm = max(ok, key=lambda t: t[0])[0]
    else:
        # 候補がすべて余裕不足の場合の安全弁: 最もクリアランスが大きい候補を選ぶ
        chosen_mm = max(table, key=lambda t: t[1])[0]
    return chosen_mm / 1000.0, table


# ==========================================================================
# §2.1 経路生成本体: セル列 → 直線＋円弧の密なサンプル列（ReferencePath）
# ==========================================================================
class ReferencePath:
    """密にサンプルした参照経路。s（弧長）・x・y・heading・curvature の
    numpy配列と、終端で停止すべきか（stop_at_end）を保持する。

    speed_profile は build_speed_profile() で後付けする（経路の形状と
    速度計画を分離し、単体テストしやすくするため）。
    """

    __slots__ = ("s", "x", "y", "heading", "curvature", "speed", "stop_at_end", "length")

    def __init__(self, s, x, y, heading, curvature, stop_at_end):
        self.s = s
        self.x = x
        self.y = y
        self.heading = heading
        self.curvature = curvature
        self.speed = None
        self.stop_at_end = stop_at_end
        self.length = float(s[-1]) if len(s) else 0.0


def build_line_path(p0, p1, heading: float, stop_at_end: bool, ds: float = 0.005,
                     min_length: float = 0.01) -> ReferencePath:
    """目標点p1を通り、方位heading（自身の走行方位。ラジアン）に沿った
    「理想直線」を、p0のこの直線への射影点からp1まで密サンプルする
    （_replan()の「先頭セル自体での方位転換」フォールバック。曲がる前に
    セル中心まで減速して停止する短い経路を作るのに使う）。

    経路の位置(xs,ys)と参照方位headsを常に幾何学的に整合させる
    （heads=heading一定、xs,ysはheading方向の直線上）。p0（車体の実位置）
    がheading軸から横方向にずれていても、経路そのものはheading軸上の
    理想直線のままにし、その横方向のずれはStanley制御のe_y項
    （arctanで飽和する穏やかな補正）に委ねる。これはメインのスラローム
    経路（直線区間はすべて格子方位に沿う）と同じ考え方であり、
    「現在位置p0から目標点p1への直線」をそのまま経路にすると、p0の
    横ずれ次第で経路の向き（p1-p0方向）が車体の実際の走行方位と大きく
    食い違うことがあり（例: 南向きに走行中、わずかに東へずれた位置から
    目標点までの直線が西寄りになる）、位置配列と参照方位が矛盾した経路に
    なって制御が暴れる不具合を実装時に確認した。

    p0のheading軸への射影長がmin_length未満（既にp1付近まで来ている等）
    の場合はmin_lengthを使う（長さ0の経路を避けるための安全弁）。
    """
    ux, uy = math.cos(heading), math.sin(heading)
    rel = (p0[0] - p1[0], p0[1] - p1[1])
    s0 = rel[0] * ux + rel[1] * uy  # heading軸上、p1から見たp0の符号付き位置（手前なら負）
    length = max(-s0, min_length)
    start = (p1[0] - length * ux, p1[1] - length * uy)
    n = max(2, int(round(length / ds)) + 1)
    t = np.linspace(0.0, length, n)
    xs = start[0] + ux * t
    ys = start[1] + uy * t
    heads = np.full(n, heading)
    curvs = np.zeros(n)
    return ReferencePath(t, xs, ys, heads, curvs, stop_at_end)


def build_reference_path(cells, directions, heading_in: str, corner_radii,
                          cell_size: float, stop_at_end: bool, ds: float = 0.005) -> ReferencePath:
    """セル列 cells（区画座標、長さm>=1）と directions（cells[i]→cells[i+1]の
    方位、長さm-1）から、直線＋円弧（接線連続）の密サンプル経路を作る。

    heading_in: cells[0] に進入した方位（cells[0]自体がコーナーになる場合は
        呼び出し側で既にチェーンを打ち切っている前提＝directions[0]と一致する
        ことを呼び出し側が保証する。本関数はassertで確認する）。
    corner_radii: 内部セル（cells[1..m-2]）に対応する円弧半径のリスト
        （fit_corner_radii()の出力。長さ=len(cells)-2）。180°（'reverse'）は
        呼び出し側で既にチェーンを打ち切っている前提。
    stop_at_end: True なら終端（cells[-1]の中心）で速度0になる区間として
        マークする（速度計画側で使用。経路形状そのものには影響しない）。
    """
    m = len(cells)
    assert m >= 1
    assert len(directions) == m - 1
    n_corners = m - 2
    assert len(corner_radii) == max(n_corners, 0)
    if m >= 2:
        assert directions[0] == heading_in or _turn_kind(heading_in, directions[0])[0] in ("left", "right"), \
            "cells[0]自体のコーナーはbuild_reference_pathでは扱えません（呼び出し側で打ち切ること）"
        # cells[0]自体のコーナー（heading_in != directions[0]）はサポートしない
        assert heading_in == directions[0], \
            "cells[0]自体で曲がる場合は呼び出し側でチェーンを打ち切ってください"

    waypoints = [((c[0] + 0.5) * cell_size, (c[1] + 0.5) * cell_size) for c in cells]

    # セグメント列（'line' or 'arc'）を組み立てる
    segments = []
    cur_pos = waypoints[0]
    cur_dir = heading_in
    for i in range(1, m - 1):
        kind, _s = _turn_kind(directions[i - 1], directions[i])
        if kind == "straight":
            continue  # 直進継続（呼び出し側で既にマージ済みの想定だが念のため許容）
        if kind == "reverse":
            raise ValueError("180°ターンはbuild_reference_pathでは扱えません（呼び出し側で打ち切ること）")
        R = corner_radii[i - 1]
        corner = corner_arc_params(waypoints[i], directions[i - 1], directions[i], R)
        segments.append({"type": "line", "p0": cur_pos, "p1": corner["tangent_in"], "heading": _HEADING_RAD[cur_dir]})
        segments.append({"type": "arc", **corner})
        cur_pos = corner["tangent_out"]
        cur_dir = directions[i]
    segments.append({"type": "line", "p0": cur_pos, "p1": waypoints[-1], "heading": _HEADING_RAD[cur_dir]})

    xs, ys, heads, curvs, ss = [], [], [], [], []
    s_acc = 0.0
    for seg_idx, seg in enumerate(segments):
        if seg["type"] == "line":
            p0, p1 = seg["p0"], seg["p1"]
            length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if length < 1e-9:
                continue
            n = max(2, int(round(length / ds)) + 1)
            heading = seg["heading"]
            for k in range(n):
                t = length * k / (n - 1)
                if k == 0 and seg_idx > 0:
                    continue  # 直前セグメント終端と重複するので飛ばす
                xs.append(p0[0] + math.cos(heading) * t)
                ys.append(p0[1] + math.sin(heading) * t)
                heads.append(heading)
                curvs.append(0.0)
                ss.append(s_acc + t)
            s_acc += length
        else:
            arclen = seg["arclen"]
            n = max(2, int(round(arclen / ds)) + 1)
            for k in range(n):
                if k == 0 and seg_idx > 0:
                    continue
                l = arclen * k / (n - 1)
                x, y, heading, curv = arc_pose_at(seg, l)
                xs.append(x)
                ys.append(y)
                heads.append(heading)
                curvs.append(curv)
                ss.append(s_acc + l)
            s_acc += arclen

    return ReferencePath(np.asarray(ss), np.asarray(xs), np.asarray(ys),
                          np.asarray(heads), np.asarray(curvs), stop_at_end)


# ==========================================================================
# §2.2 速度計画: 曲率依存の速度上限包絡線 → 前後進加速度制約の2パス
# ==========================================================================
def build_speed_profile(s_arr, curv_arr, v_ceil_straight, a_lat: float,
                         a_max: float, v_creep: float, stop_at_end: bool) -> np.ndarray:
    """曲率依存の速度上限包絡線（v_turn=sqrt(a_lat/|curvature|)、設計文書§2.2）
    を作り、前後2パス（CNC等で標準的な時間最適速度包絡線の手法）で
    前後進加速度 a_max の制約を反映する。これは経路上の位置 s だけに依存する
    「その地点で許される速度の上限」であり、車体が実際に今どの速度で走って
    いるかは考慮しない（張り替え直後の実速度との整合は、この関数ではなく
    制御側のレートリミット済み速度設定値で取る。_do_drive_control参照）。

    v_ceil_straight: スカラ、または経路点ごとの上限を並べた長さ n の配列。
        配列を受け付けるのは、**経路の区間ごとに既知/未知で速度上限を
        切り替える**ため（2026-08-10 の設計是正、教授裁定）。既に壁を観測
        済みの区画は摩擦円の限界まで出してよいが、これから進入する未観測の
        区画は「壁が現れても曲がれる/止まれる速度」に抑える必要がある。
        従来はチェーン末尾で一律に停止（stop_at_end）させていたため、
        既知区間を走るときまで不必要に減速していた。

    速度指令の下限（クリープ、設計文書§10-2）を v_ceil に明示的に折り込む
    （stop_at_end の場合のみ終端1点を0にしてクリープ下限を上書きする＝
    「目標到達のときのみ停止する」を速度計画そのもので保証する）。
    """
    n = len(s_arr)
    base = np.asarray(v_ceil_straight, dtype=float)
    if base.ndim == 0:
        base = np.full(n, float(base))
    elif base.shape != (n,):
        raise ValueError(f"v_ceil_straight の長さ {base.shape} が経路点数 {n} と一致しません")
    v_ceil = np.empty(n)
    with np.errstate(divide="ignore"):
        for i in range(n):
            k = abs(curv_arr[i])
            if k > 1e-6:
                v_ceil[i] = min(base[i], math.sqrt(a_lat / k))
            else:
                v_ceil[i] = base[i]
    v_ceil = np.maximum(v_ceil, v_creep)
    if stop_at_end and n > 0:
        v_ceil[-1] = 0.0

    v = v_ceil.copy()
    # 後方パス（この先の減速に間に合うよう手前を制限）
    for i in range(n - 2, -1, -1):
        ds = s_arr[i + 1] - s_arr[i]
        v_allow = math.sqrt(v[i + 1] ** 2 + 2.0 * a_max * ds)
        if v[i] > v_allow:
            v[i] = v_allow
    # 前方パス（手前からの加速で間に合う速度に制限）
    for i in range(1, n):
        ds = s_arr[i] - s_arr[i - 1]
        v_allow = math.sqrt(v[i - 1] ** 2 + 2.0 * a_max * ds)
        if v[i] > v_allow:
            v[i] = v_allow
    return v


# ==========================================================================
# §10-3 車輪速度制御: 逆モデル前向き補償 + PI（アンチワインドアップ）
# ==========================================================================
class WheelPI:
    """1輪ぶんの逆モデル前向き補償 + PI アンチワインドアップ制御器。

    V = Ke_eff・ω_ref + (1/gainprm0)・(b・ω_ref + τc・sgn(ω_ref))   … 逆モデル前向き補償
        + Kp・e_ω + Ki・∫e_ω dt                                      … PI

    逆モデル項の導出（設計文書§3-3、mouse/params.pyの確定値から）:
    車輪の定常負荷トルク τ_wheel = b・ω_wheel + τc・sgn(ω_wheel)（粘性+乾性摩擦）
    に対し、ギヤ比Nの減速機を介したモータ電流 I=τ_wheel/(N・Kt)、
    定常電圧 V=I・R + Ke・(N・ω_wheel) = R/(N・Kt)・τ_wheel + Ke・N・ω_wheel。
    R/(N・Kt) = 1/(N・Kt/R) = 1/gainprm0（RobotParams.gainprm0 プロパティ）。
    Ke・N = Ke_eff（motor_Ke * gear_ratio、L0-a/L0-bと同じ導出）。

    アンチワインドアップ: 出力が電圧上限で飽和し、かつ誤差がさらに同方向へ
    積分を進める向きのときは積分を止める（clamping/conditional積分の
    標準的な組み合わせ）。積分状態そのものも上限でクランプする。
    """

    def __init__(self, Ke_eff: float, inv_gain: float, damping: float, friction_torque: float,
                 kp: float, ki: float, voltage_limit: float, int_clamp: float):
        self.Ke_eff = Ke_eff
        self.inv_gain = inv_gain
        self.damping = damping
        self.friction_torque = friction_torque
        self.kp = kp
        self.ki = ki
        self.voltage_limit = voltage_limit
        self.int_clamp = int_clamp
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def step(self, omega_ref: float, omega_act: float, dt: float) -> float:
        sgn = 1.0 if omega_ref > 1e-9 else (-1.0 if omega_ref < -1e-9 else 0.0)
        ff = self.Ke_eff * omega_ref + self.inv_gain * (self.damping * omega_ref + self.friction_torque * sgn)
        e = omega_ref - omega_act
        unclamped = ff + self.kp * e + self.ki * self.integral
        v = max(-self.voltage_limit, min(self.voltage_limit, unclamped))
        pushing_further = (v >= self.voltage_limit and e > 0.0) or (v <= -self.voltage_limit and e < 0.0)
        if not pushing_further:
            self.integral = max(-self.int_clamp, min(self.int_clamp, self.integral + e * dt))
        return v


# ==========================================================================
# 物理限界の実測値（設計文書§2.2「検証スイートの実測値」。研究計画書§4.3・
# ROBOT_SPEC§4 由来。実装定数としてはここに一度だけ書き、以後は
# safety_factor（可変パラメータ、コンストラクタ引数）でスケールする）
# ==========================================================================
V_MAX_MEASURED = 3.86    # 最高速度 実測 [m/s]
A_MAX_MEASURED = 5.6     # 前後加速度 実測（トラクション律速）[m/s^2]
A_LAT_MEASURED = 5.87    # 横加速度 実測（Test5プラトー実測）[m/s^2]
# 【2026-08-10 設計是正】旧 V_EXPLORE = 0.8 m/s（決め打ち）は廃止した。
# 方策自身が bind_sim で v_turn = sqrt(a_lat*R) = 0.61 m/s を計算しているのに
# 探索速度が 0.8 m/s であり、「曲がると分かった瞬間に曲がれる速度」を超えて
# いた（当時は探索走行が超信地旋回だったため露見していなかった）。
# 現在は _derive_uncertain_speed() でモデル量から実行時導出する。


class SlalomPolicy(MousePolicy):
    """足立法（flood-fill）+ スラローム走行（曲がるところでも停止しない）に
    よる古典ベースライン（L0-c）。docs/L0PLUS_DESIGN.md §2・§3・§10。

    経路決定（どのセルへ進むか）は L0-a/L0-b と完全に同一ロジック。走行部の
    みが異なる: 区画中心の折れ線を「直線＋90°ターンの円弧（接線連続）」に
    置き換えた軌道を、Stanley標準形の経路追従＋逆モデル前向き補償＋PIの
    車輪速度制御で、曲がるところでも停止せずに追従する。

    特権情報（bind_sim/bind_maze）を使用するため requires_privileged=True。
    """

    name = "l0c_slalom"
    requires_privileged = True

    def __init__(self,
                 safety_factor: float = 0.7,
                 v_max_measured: float = V_MAX_MEASURED, a_max_measured: float = A_MAX_MEASURED,
                 a_lat_measured: float = A_LAT_MEASURED, v_explore: float = None,
                 arc_radius_candidates_mm=(45, 60, 75, 90), wall_margin_m: float = 0.015,
                 corner_fit_margin_m: float = 0.01, explore_horizon_cells: int = 2,
                 replan_trigger_dist: float = 0.15,
                 path_ds: float = 0.005, cursor_search_window: int = 80,
                 # 【2026-08-10 是正】k_psi 3.0 -> 12.0, k_y 2.5 -> 10.0。
                 # 旧値では方位偏差の復元力が弱すぎ、円弧上の指令ヨー角速度
                 # (平均 7 rad/s) に対して補正項が 0.1 rad/s 程度しか働かず、
                 # 横偏差が最大 48mm（円弧の壁余裕は約 32mm）まで拡大していた。
                 # 旧実装は「チェーン末尾ごとに必ず停止する」構造がこの弱さを
                 # 隠していた（停止のたびに偏差がリセットされる）。停止をやめて
                 # 流して走るようにした結果として露呈した。
                 # 検証帯 seed 4000-4004 での実測（失敗率 / |e_y|平均・最大）:
                 #   k_psi= 3: 60.0% / 14.1mm・48.0mm
                 #   k_psi= 6: 20.0% / 10.3mm・45.8mm
                 #   k_psi= 9:  0.0% /  7.6mm・27.3mm
                 #   k_psi=12:  0.0% /  6.1mm・22.4mm  <- 採用
                 # k_y は 6〜24 のいずれでも失敗率 0% で感度が低い（支配的なのは
                 # k_psi）。偏差が最小になる 10.0 を採る。
                 k_psi: float = 12.0, k_y: float = 10.0, v_eps: float = 0.15, v_creep: float = 0.15,
                 kp_wheel: float = 0.05, ki_wheel: float = 0.3, int_clamp_frac: float = 0.5,
                 kp_turn: float = 8.0, kd_turn: float = 0.6, turn_omega_limit: float = 12.0,
                 turn_done_deg: float = 2.0, turn_done_gyro: float = 0.2,
                 goal_done_dist: float = 0.02, goal_done_speed: float = 0.05):
        # --- 安全率・物理限界（設計文書§2.2。安全率は可変パラメータとし、
        #     「完走率100%を維持できる最大値」を探索して確定する） ---
        self.safety_factor = safety_factor
        self.v_max_measured = v_max_measured
        self.a_max_measured = a_max_measured
        self.a_lat_measured = a_lat_measured
        self.v_cap = v_max_measured * safety_factor
        self.a_max = a_max_measured * safety_factor
        self.a_lat = a_lat_measured * safety_factor
        # 未知区画へ進入するときの速度上限。既定 None で bind_sim() が
        # モデル量から導出する（_derive_uncertain_speed）。明示的に数値を
        # 渡せるのはアブレーション実験のためだけで、通常は None のまま使う。
        self.v_explore_override = v_explore
        self.v_uncertain = None       # bind_sim() で確定
        self.min_arc_radius = min(arc_radius_candidates_mm) / 1000.0

        # --- 軌道生成パラメータ（§2.1） ---
        self.arc_radius_candidates_mm = tuple(arc_radius_candidates_mm)
        self.wall_margin_m = wall_margin_m
        self.corner_fit_margin_m = corner_fit_margin_m
        self.explore_horizon_cells = explore_horizon_cells
        # 再計画（探索走行時、次のホライズンぶんへ経路を継ぎ足す）のトリガー
        # 距離[m]。曲率がほぼ0（直線区間）かつ残り経路長がこの値未満のときに
        # 再計画する。区画境界の通過そのものでは再計画しない（旧チェーンが
        # 既に滑らかな円弧で処理中のコーナーを、区画境界通過のたびに「先頭
        # セルでの急な方位転換」と誤認して停止+その場旋回に落ちる不具合が
        # 実装時のスモークテストで見つかったため。曲率ゼロの直線区間まで
        # 進んでから再計画することで、進行中の円弧を割り込ませない）。
        self.replan_trigger_dist = replan_trigger_dist
        self.path_ds = path_ds
        self.cursor_search_window = cursor_search_window
        self.R = None                # bind_sim() で選定（実車体フットプリントから導出）
        self.arc_radius_table = None
        self.v_turn = None

        # --- 経路追従（Stanley標準形、§10） ---
        self.k_psi = k_psi
        self.k_y = k_y
        self.v_eps = v_eps
        self.v_creep = v_creep

        # --- 車輪速度制御（逆モデル前向き補償+PI、§10-3） ---
        self.kp_wheel = kp_wheel
        self.ki_wheel = ki_wheel
        self.int_clamp_frac = int_clamp_frac

        # --- その場旋回（180°折返し・先頭セル方位転換のフォールバック用） ---
        self.kp_turn = kp_turn
        self.kd_turn = kd_turn
        self.turn_omega_limit = turn_omega_limit
        self.turn_done_rad = math.radians(turn_done_deg)
        self.turn_done_gyro = turn_done_gyro

        # --- ゴール停止判定（2段階ゴール判定に合わせ区画中心付近で停止） ---
        self.goal_done_dist = goal_done_dist
        self.goal_done_speed = goal_done_speed

        # --- bind_sim/bind_maze で埋まる特権情報 ---
        self._sim = None
        self._true_v = None
        self._true_h = None

        # --- params から導出する物理定数（bind_sim で上書きされるまでの既定値） ---
        self.cell_size = 0.18
        self.wheel_radius = 0.0135
        self.tread = 0.072
        self.Ke_eff = 9.9e-3
        self.voltage_limit = 3.0
        self.control_dt = 0.01
        self.wheel_damping = 2.0e-6
        self.wheel_frictionloss = 9.1e-4
        self.inv_gain = 1.0 / 9.252336e-3

        self.width = 16
        self.height = 16
        self.v_walls_known = None
        self.h_walls_known = None

        self._wheel_pi_l = None
        self._wheel_pi_r = None

        self._reset_run_state()

    # ------------------------------------------------------------------
    # 内部状態リセット
    # ------------------------------------------------------------------
    def _reset_run_state(self):
        self._state = "INIT"
        self.target_mode = "to_goal"
        self._heading_dir = "N"
        self._explored_once = False
        # --- 走り方の内訳カウンタ（1迷路ぶん。教授指示 2026-08-10: 先頭セルの
        #     方位転換のうち、円弧を張れずその場旋回へフォールバックした割合を
        #     毎回報告すること。常時発動していれば実質L0-bでありL0-cを
        #     名乗れない） ---
        self._n_lead_turns = 0                 # 先頭セルでの90度転換の総数
        self._n_lead_turn_pivot_fallback = 0   # うち円弧を張れずその場旋回に落ちた数
        self._n_pivot_entries = 0              # その場旋回に入った総回数（180度折返し含む）
        self._path = None
        self._cursor = 0
        self._path_ticks = 0
        self._path_end_reason = None
        self._pending_dir_after_turn = None
        self._turn_target_yaw = None
        self._last_sensed_cell = None
        # 速度指令のレートリミット済み設定値（L0-a/L0-bの_fwd_v_setpointと
        # 同じ考え方）。build_speed_profileが作る速度上限は経路上の位置
        # だけに依存する空間的な値であり、張り替え直後に車体が実際どの
        # 速度で走っているかは知らない。カーソル参照点の速度をそのまま
        # 指令すると、張り替え直後に指令が不連続に跳ぶ（スモークテストで
        # 確認: ヨー発散→壁衝突）。ここでティックごとにa_max/a_latで
        # レートリミットしてから使うことで滑らかにする。
        self._v_setpoint = 0.0
        if self._wheel_pi_l is not None:
            self._wheel_pi_l.reset()
            self._wheel_pi_r.reset()

    # ------------------------------------------------------------------
    # 特権情報バインド（policy_interface.py の契約）
    # ------------------------------------------------------------------
    def bind_sim(self, sim) -> None:
        self._sim = sim
        p = sim.params
        # 観測ベクトルの並びは [距離 ×n, accel(3), gyro(3), 車輪角速度(2)] であり、
        # 距離センサ本数 n は構成で変わる。固定インデックスにしない（L0-a/L0-bと同方針）。
        n_dist = len(p.sensors)
        self._i_gyro_z = n_dist + 5
        self._i_wheel = n_dist + 6

        self.cell_size = p.cell_size
        self.wheel_radius = p.wheel_radius
        self.tread = p.tread
        self.Ke_eff = p.motor_Ke * p.gear_ratio
        self.voltage_limit = p.voltage_limit
        self.control_dt = p.control_dt
        self.wheel_damping = p.wheel_damping
        self.wheel_frictionloss = p.wheel_frictionloss
        self.inv_gain = 1.0 / p.gainprm0  # = motor_R/(gear_ratio*motor_Kt)、§10-3の逆モデル項

        # --- 円弧半径の選定（§2.1）: 実車体フットプリントをモデルから導出
        #     （ハードコード禁止の原則。competition/evaluator.py の
        #     body_footprint と同じ手法をそのまま再利用する） ---
        import mujoco
        from competition.evaluator import body_footprint, front_offset
        mouse_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
        footprint = body_footprint(sim.model, sim.data, mouse_bid)
        self.R, self.arc_radius_table = select_arc_radius(
            footprint, self.cell_size, candidates_mm=self.arc_radius_candidates_mm,
            margin_m=self.wall_margin_m)
        self.v_turn = math.sqrt(self.a_lat * self.R)

        # --- 未知区画へ進入するときの速度上限（2026-08-10 設計是正、§2.3 改） ---
        self._front_offset = front_offset(sim.model, sim.data, mouse_bid)
        self.v_uncertain = (self.v_explore_override if self.v_explore_override is not None
                            else self._derive_uncertain_speed())

        # --- 車輪速度PI（アンチワインドアップ付き）。積分クランプは
        #     Ki・積分値が電圧上限の一定割合(int_clamp_frac)を超えないように。 ---
        int_clamp = self.int_clamp_frac * self.voltage_limit / max(self.ki_wheel, 1e-9)
        self._wheel_pi_l = WheelPI(self.Ke_eff, self.inv_gain, self.wheel_damping,
                                    self.wheel_frictionloss, self.kp_wheel, self.ki_wheel,
                                    self.voltage_limit, int_clamp)
        self._wheel_pi_r = WheelPI(self.Ke_eff, self.inv_gain, self.wheel_damping,
                                    self.wheel_frictionloss, self.kp_wheel, self.ki_wheel,
                                    self.voltage_limit, int_clamp)

    def _derive_uncertain_speed(self) -> float:
        """**未知の区画へ進入するときの速度上限**をモデル量から導出する。

        区画へ進入して壁を観測した瞬間に、次の2つを両方満たせる速度:

          (i)  曲がる必要が判明しても曲がれる:  v <= sqrt(a_lat * R_worst)
          (ii) 行き止まりでも止まれる:          v <= sqrt(2 * a_max * d_stop)

        (ii) はL0-bの直進速度上限 v <= sqrt(2*a*d) と同じ考え方（教授指示）。

        R_worst は「発見が最も遅れた場合でも張れる円弧半径」。区画の中心で
        90度曲がる円弧の入口は中心の R 手前にあるので、車体が区画中心まで
        あと (cell_size/2 - 遅れぶんの前進) しか無いとき、張れる半径は
        その残距離までである:

          R_worst = cell_size/2 - v * t_delay - corner_fit_margin

        遅れ時間 t_delay は制御周期 control_dt の2周期ぶんとする。内訳は
        (1) 区画境界の通過を検知するまでの標本化遅れ（境界通過は制御周期の
        どのタイミングでも起こりうるので最悪1周期）、(2) 経路を張り替えた
        指令が実際に効き始めるまでの遅れ（1周期）。**制御周期というモデル
        側の量から決めており、決め打ちの数値ではない**。

        R_worst が v に依存するので不動点反復で解く（単調減少写像なので
        数回で収束する。上の初期値から下方へ単調に収束する）。
        """
        d_stop = self.cell_size - self._front_offset - self.wall_margin_m
        v_stop_limit = math.sqrt(2.0 * self.a_max * max(d_stop, 1e-4))
        t_delay = 2.0 * self.control_dt
        v = math.sqrt(self.a_lat * (self.cell_size / 2.0))
        for _ in range(20):
            r_worst = self.cell_size / 2.0 - v * t_delay - self.corner_fit_margin_m
            v_new = min(math.sqrt(self.a_lat * max(r_worst, 1e-4)), v_stop_limit, self.v_cap)
            if abs(v_new - v) < 1e-6:
                v = v_new
                break
            v = v_new
        return max(v, self.v_creep)

    def _dist_to_cell_center_along_heading(self, x: float, y: float, cell) -> float:
        """車体位置 (x,y) から区画 cell の中心までの、進行方位に沿った符号付き
        残距離[m]（まだ手前なら正、行き過ぎていれば負）。先頭セルのコーナーを
        円弧にできるかの幾何判定に使う。"""
        cx = (cell[0] + 0.5) * self.cell_size
        cy = (cell[1] + 0.5) * self.cell_size
        ux = math.cos(_HEADING_RAD[self._heading_dir])
        uy = math.sin(_HEADING_RAD[self._heading_dir])
        return (cx - x) * ux + (cy - y) * uy

    def bind_maze(self, v_walls, h_walls) -> None:
        self._true_v = v_walls
        self._true_h = h_walls

    # ------------------------------------------------------------------
    # ライフサイクル通知フック
    # ------------------------------------------------------------------
    def on_maze_start(self, maze_info: dict) -> None:
        self.width = int(maze_info["width"])
        self.height = int(maze_info["height"])

        self.v_walls_known = np.full((self.width + 1, self.height), -1, dtype=int)
        self.h_walls_known = np.full((self.width, self.height + 1), -1, dtype=int)
        self.v_walls_known[0, :] = 1
        self.v_walls_known[self.width, :] = 1
        self.h_walls_known[:, 0] = 1
        self.h_walls_known[:, self.height] = 1
        # スタート区画規定（competition/maze_gen.py 手順4）: 北開放・東壁あり
        if self.height > 1:
            self.h_walls_known[0, 1] = 0
        if self.width > 1:
            self.v_walls_known[1, 0] = 1

        self._reset_run_state()

    def on_run_start(self, run_index: int) -> None:
        pass

    def on_run_end(self, outcome: str) -> None:
        pass

    def on_retrieval(self) -> None:
        self._reset_run_state()

    # ------------------------------------------------------------------
    # 足立法（flood-fill）— L0-a/L0-b（baseline_straightrun.py）と同一ロジック
    # ------------------------------------------------------------------
    def _target_cells(self):
        if self.target_mode == "to_goal":
            return goal_cells(self.width, self.height)
        return [(0, 0)]

    def _wall_value(self, x: int, y: int, nx: int, ny: int) -> int:
        """既知壁配列で (x,y)-(nx,ny) 間の壁値を返す（0=壁なし, 1=壁あり, -1=未知）。"""
        if (nx, ny) == (x + 1, y):
            return int(self.v_walls_known[x + 1, y])
        if (nx, ny) == (x - 1, y):
            return int(self.v_walls_known[x, y])
        if (nx, ny) == (x, y + 1):
            return int(self.h_walls_known[x, y + 1])
        if (nx, ny) == (x, y - 1):
            return int(self.h_walls_known[x, y])
        raise ValueError(f"({x},{y}) と ({nx},{ny}) は隣接していません")

    def _connects_known(self, x: int, y: int, nx: int, ny: int) -> bool:
        """既知壁配列で (x,y)-(nx,ny) が通行可能か。未知(-1)は楽観的に通行可能扱い。"""
        return self._wall_value(x, y, nx, ny) != 1

    def _flood_fill(self, targets) -> dict:
        """targets からの multi-source BFS 距離場（未知壁=壁なし扱い）。L0-a/L0-bと同一。"""
        dist = {c: 0 for c in targets}
        q = deque(targets)
        while q:
            x, y = q.popleft()
            for d in DIRS:
                dx, dy = _DELTA[d]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in dist:
                    continue
                if self._connects_known(x, y, nx, ny):
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
        return dist

    @staticmethod
    def _tiebreak_order(prev_dir: str):
        """決定的タイブレーク順: 直進優先 → 前回進行方向から時計回り優先。L0-a/L0-bと同一。"""
        order = [prev_dir]
        d = prev_dir
        for _ in range(3):
            d = _CW_NEXT[d]
            order.append(d)
        return order

    def _select_direction(self, cell, prev_dir: str, dist_field: dict):
        """cell における flood-fill 候補選定。L0-a/L0-bの_do_plan/_select_directionと同一ロジック。"""
        cx, cy = cell
        candidates = []
        for d in DIRS:
            dx, dy = _DELTA[d]
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            if not self._connects_known(cx, cy, nx, ny):
                continue
            nd = dist_field.get((nx, ny))
            if nd is None:
                continue
            candidates.append((d, nd))
        if not candidates:
            return None
        min_dist = min(nd for _, nd in candidates)
        best_set = {d for d, nd in candidates if nd == min_dist}
        return next(d for d in self._tiebreak_order(prev_dir) if d in best_set)

    def _do_sense(self, cx: int, cy: int) -> None:
        """現在セル (cx,cy) の4方向の真の壁を読み、既知地図に書き込む。L0-a/L0-bと同一。"""
        self.v_walls_known[cx + 1, cy] = self._true_v[cx + 1, cy]
        self.v_walls_known[cx, cy] = self._true_v[cx, cy]
        self.h_walls_known[cx, cy + 1] = self._true_h[cx, cy + 1]
        self.h_walls_known[cx, cy] = self._true_h[cx, cy]

    def _flip_target_mode(self) -> None:
        """目標集合をゴール⇔スタートで反転する（L0-a/L0-bと同じ「到達→反転」機構）。
        初回ゴール到達の検出もここで行う（§2.3「最短走行は初回ゴール到達後」）。"""
        if self.target_mode == "to_goal":
            self._explored_once = True
        self.target_mode = "to_start" if self.target_mode == "to_goal" else "to_goal"

    # ------------------------------------------------------------------
    # チェーン構築: 現在セルから足立法で進む方向を horizon 区画ぶん先読みする
    # ------------------------------------------------------------------
    def _build_chain(self, start_cell, heading_in: str, max_cells: int,
                      require_known_exit: bool = False,
                      allow_lead_turn: bool = False) -> dict:
        """start_cell から heading_in の続きとして最大 max_cells 区画ぶんの
        セル列・方位列を作る。

        require_known_exit=True（探索走行、§2.3）のときは、2区画目以降で
        「まだ壁を観測していないセルの出口方向」まで軌道生成しない
        （既知地図が楽観的=未知壁を壁なし扱いするflood-fillの経路選択自体は
        変えないが、その選択がまだ検証されていない場合は円弧を張らずに
        そこでチェーンを打ち切る。start_cellは常に既知＝直前にセンス済み
        なので1区画目はこの制限を受けない）。これによりL0-b由来の
        _known_open と同じ「未検知の壁ぎわで先読みを保守的に打ち切る」
        考え方を、直進だけでなく円弧の先読みにも適用する。実装当初の
        スモークテストで、未検知セルの出口壁を楽観的に壁なしと仮定して
        曲がった結果、実際には壁があり衝突する不具合を確認して追加した。

        Returns:
            dict(cells, directions, reached_target, need_turn_to)
            - reached_target: 現在の目標集合（ゴール/スタート）に到達したら True
            - need_turn_to: 180°折返し、または先頭セル自体での方位転換
              （滑らかな進入余地が無い）が必要な場合、その方向。それ以外は None。
              cells/directions にはこの手前までの安全に走行できる区間が入る。
        """
        targets = self._target_cells()
        dist_field = self._flood_fill(targets)
        if dist_field.get(start_cell) == 0:
            return dict(cells=[start_cell], directions=[], reached_target=True, need_turn_to=None)

        cells = [start_cell]
        directions = []
        cur = start_cell
        prev_dir = heading_in
        reached_target = False
        for step_i in range(max_cells):
            chosen = self._select_direction(cur, prev_dir, dist_field)
            if chosen is None:
                break  # 安全弁（到達可能な隣接セルなし）
            if step_i >= 1 and require_known_exit:
                nx, ny = cur[0] + _DELTA[chosen][0], cur[1] + _DELTA[chosen][1]
                if self._wall_value(cur[0], cur[1], nx, ny) == -1:
                    break  # 未検知セルの出口: これ以上は伸ばさずここで打ち切る
            kind, _s = _turn_kind(prev_dir, chosen)
            # 180度折返しは円弧にできないので常にその場旋回へ回す。
            # 先頭セル(step_i==0)での90度転換は、allow_lead_turn=True なら
            # ここで打ち切らず、呼び出し側(_replan)が「1区画手前のセルを
            # 前置する」手法で円弧として張る（2026-08-10 設計是正: 探索走行
            # でもスラロームを維持するため。従来は無条件にその場旋回へ
            # 落としており、探索走行が実質L0-b相当になっていた）。
            # 幾何が許さない場合（旋回の発見が遅れた場合）は_replanが
            # その場旋回へフォールバックする。
            if kind == "reverse" or (step_i == 0 and kind in ("left", "right")
                                      and not allow_lead_turn):
                return dict(cells=cells, directions=directions, reached_target=False, need_turn_to=chosen)
            directions.append(chosen)
            cur = (cur[0] + _DELTA[chosen][0], cur[1] + _DELTA[chosen][1])
            cells.append(cur)
            prev_dir = chosen
            if dist_field.get(cur) == 0:
                reached_target = True
                break
        return dict(cells=cells, directions=directions, reached_target=reached_target, need_turn_to=None)

    # ------------------------------------------------------------------
    # 軌道の張り直し（§2.1〜§2.3）
    # ------------------------------------------------------------------
    def _enter_turn_inplace(self, direction: str) -> None:
        self._n_pivot_entries += 1
        self._turn_target_yaw = _HEADING_RAD[direction]
        self._pending_dir_after_turn = direction
        self._state = "TURN_INPLACE"
        self._path = None
        self._v_setpoint = 0.0
        if self._wheel_pi_l is not None:
            self._wheel_pi_l.reset()
            self._wheel_pi_r.reset()

    def _begin_pre_turn(self, x: float, y: float, cell, direction: str) -> None:
        """円弧では曲がれない方位転換（180度折返し、または旋回の発見が遅れて
        円弧の入口を通り過ぎた場合）のフォールバック。

        実際の車体現在位置(x,y)からそのセル中心までの短い直線減速停止経路を
        挟んでから、その場旋回へ移る（動いたままいきなり回頭を始めると位置が
        ずれる不具合がスモークテストで見つかったため）。既にセル中心にいる
        場合は直ちにその場旋回へ入る。
        """
        target_xy = ((cell[0] + 0.5) * self.cell_size, (cell[1] + 0.5) * self.cell_size)
        if math.hypot(target_xy[0] - x, target_xy[1] - y) > 1e-3:
            path = build_line_path((x, y), target_xy, _HEADING_RAD[self._heading_dir],
                                    stop_at_end=True, ds=self.path_ds)
            # ここは常に停止経路（この先で方位転換が必要と判明した区間）
            # なので、常に保守的な速度上限 v_uncertain を使う。
            path.speed = build_speed_profile(path.s, path.curvature, self.v_uncertain,
                                              self.a_lat, self.a_max, self.v_creep, True)
            self._path = path
            self._cursor = 0
            self._path_ticks = 0
            self._path_end_reason = "pre_turn"
            self._pending_dir_after_turn = direction
            self._state = "DRIVE"
        else:
            self._enter_turn_inplace(direction)

    def _replan(self, x: float, y: float, yaw: float) -> None:
        """現在位置から軌道を張り直す。呼び出しタイミング: (1) 起動直後、
        (2) 区画境界通過ごと（探索走行=初回ゴール到達前のみ、§2.3）、
        (3) 目標セル到達時（ゴール/スタートの反転直後）、(4) その場旋回完了直後。

        探索走行では次の1〜2区画ぶんだけ軌道生成する。最短走行（初回ゴール
        到達後）は既知地図で目標セルまでの全経路を一括生成する（§2.3）。

        速度上限は「区画が既知か未知か」で切り替える（2026-08-10 設計是正）。
        観測済みの区画を通る区間は摩擦円の限界 v_cap まで、チェーン末尾の
        「出口の壁が未観測」な区画へ進入する時点では v_uncertain まで落とす。
        先頭セルでの90度転換も、幾何が許すかぎり円弧（スラローム）で曲がる。

        require_known_exit=True は両モード共通で常に適用する。理由:
        実装時のスモークテストで、最短走行(一括生成)が「一度も訪れていない
        セル」を経由するflood-fill最短経路（未知壁=壁なしの楽観的既定値を
        使用）を一括で軌道化し、実際には壁があり衝突する不具合を確認した
        （経路決定自体は楽観的規則のままでよいが、円弧つき軌道の生成に
        コミットするのは既知区間に限るべき、というのはL0-bの_known_openと
        同じ考え方）。既知地図が最短経路の全区間をカバーしていれば
        「一括生成」は変わらず成立し、未知区間に達したときだけ1区画ずつの
        チェーンに自動的に縮退する（次にその区画へ到達しセンスした時点で
        続きが伸びる）。"""
        cur_cell = pos_to_cell(x, y, self.width, self.height, self.cell_size)
        for _attempt in range(3):
            max_cells = (self.explore_horizon_cells if not self._explored_once
                         else self.width * self.height)
            chain = self._build_chain(cur_cell, self._heading_dir, max_cells,
                                       require_known_exit=True, allow_lead_turn=True)
            cells = chain["cells"]

            if len(cells) >= 2:
                directions = chain["directions"]

                # --- 先頭セルでの90度転換を円弧にする（2026-08-10 設計是正） ---
                # cells[0]自体で曲がる場合、build_reference_path は
                # heading_in == directions[0] を要求するので直接は張れない。
                # そこで**1区画手前のセルを前置**してそのコーナーを「内部
                # コーナー」に変え、既存の円弧生成機構をそのまま使う。前置した
                # 直線区間が車体の実位置を含むのでカーソル探索も破綻しない
                # （幾何は事前検証済み: 円弧の入口がちょうど区画の入口境界に
                # 一致し、経路長も理論値と一致する）。
                path_cells, path_dirs = list(cells), list(directions)
                lead_R = None
                if directions[0] != self._heading_dir:
                    self._n_lead_turns += 1
                    prev_cell = (cur_cell[0] - _DELTA[self._heading_dir][0],
                                 cur_cell[1] - _DELTA[self._heading_dir][1])
                    in_bounds = (0 <= prev_cell[0] < self.width
                                 and 0 <= prev_cell[1] < self.height)
                    # 区画中心までの残距離が、そのまま今から張れる円弧半径の
                    # 上限になる（中心で90度曲がる円弧の入口は中心のR手前）。
                    d_center = self._dist_to_cell_center_along_heading(x, y, cur_cell)
                    lead_R = min(self.R, d_center - self.corner_fit_margin_m)
                    if (not in_bounds) or lead_R < self.min_arc_radius:
                        # 旋回の発見が遅れた（既に円弧の入口を通り過ぎた）か
                        # 盤外。円弧では曲がれないのでその場旋回へ落とす
                        # （教授裁定 2026-08-10: 安全弁として残すが、発生率を
                        # 毎回報告すること。常時発動していれば実質L0-b）。
                        self._n_lead_turn_pivot_fallback += 1
                        self._begin_pre_turn(x, y, cur_cell, directions[0])
                        return
                    path_cells = [prev_cell] + path_cells
                    path_dirs = [self._heading_dir] + path_dirs

                waypoints_xy = [((c[0] + 0.5) * self.cell_size, (c[1] + 0.5) * self.cell_size)
                                 for c in path_cells]
                radii = fit_corner_radii(waypoints_xy, self.R, margin_m=self.corner_fit_margin_m)
                if lead_R is not None and radii:
                    # 先頭コーナーは「今から張れる半径」までしか使えない
                    radii[0] = min(radii[0], lead_R)

                if chain["reached_target"] and self.target_mode == "to_goal":
                    stop_at_end, end_reason = True, "goal"
                elif chain["need_turn_to"] is not None:
                    stop_at_end, end_reason = True, "pre_turn"
                    self._pending_dir_after_turn = chain["need_turn_to"]
                elif chain["reached_target"]:  # to_start
                    # ゴールと同じく実際に停止してから反転する。to_startは
                    # 評価器のKPI（憲章§2の4指標）で計時されないため速度上の
                    # デメリットが無い一方、スタート区画(0,0)は西・南が外壁の
                    # 狭いポケットであり、減速せず一定速度で目標点へ進み
                    # 続けると、わずかな横ずれでも外壁に接触してスタックする
                    # 不具合をスモークテストで確認した。
                    stop_at_end, end_reason = True, "start"
                else:
                    stop_at_end, end_reason = False, "continue"

                path = build_reference_path(path_cells, path_dirs, path_dirs[0], radii,
                                             self.cell_size, stop_at_end, ds=self.path_ds)

                # --- 区間ごとの速度上限（教授裁定 2026-08-10「既知は速く、
                #     未知は慎重に」。実機のマウサーも既に見た区画は速く、
                #     未知の区画は慎重に走る） ---
                # require_known_exit により、チェーンに含まれる区画は末尾を
                # 除きすべて出口の壁が観測済みである。したがって末尾以外は
                # 摩擦円の限界(v_cap)まで出してよい。末尾の区画だけは「出口の
                # 壁が未観測」なので、そこへ**進入する時点**で v_uncertain
                # （壁が現れても曲がれる/止まれる速度。_derive_uncertain_speed
                # がモデル量から導出）まで落としておく。後方パスがこの上限に
                # 間に合うよう手前を自動的に制限する。
                #
                # 旧実装はこの区間を stop_at_end 相当で**ほぼ停止**まで減速
                # させ、かつ全体の上限も探索中は一律 v_explore に固定して
                # いたため、既に観測済みの区画を走るときまで不必要に遅かった
                # （E1の追加探索は既知経路を通って未知の壁へ向かうので、この
                # 切り替えの有無で探索時間が大きく変わる）。
                if end_reason == "continue":
                    # 末尾セルの出口の壁が未観測＝この先が未知。ここは
                    # v_uncertain に固定する。実測（検証帯 seed 4000-4004）で、
                    # 「既知セルを含むので v_cap まで加速してよい」とすると
                    # 失敗率が 12% → 64% に悪化した。最短走行では経路の
                    # 張り替え（_replan）が起こりうる区間でもあり、高速で
                    # 未知の境界へ突っ込むリスクに見合わない。
                    # （E1 の追加探索を統合したら、目標セルまで既知地図で
                    #  一括到達するチェーンになるので end_reason は
                    #  "continue" ではなくなり、自動的に v_cap 側になる）
                    v_ceil_base = self.v_uncertain
                elif self.target_mode == "to_start":
                    # 帰路は評価器のKPI（研究計画書§2）で計時されず高速化の
                    # 得点上の意味がない一方、スタート区画(0,0)は西・南が外壁の
                    # 狭いポケットで接触・スタックの実績がある。得点にならない
                    # 区間でリスクだけ取らないよう保守側に倒す（設計判断）。
                    v_ceil_base = min(self.v_cap, self.v_uncertain)
                else:
                    v_ceil_base = self.v_cap
                # 終端で満たすべき速度と、そこまでの距離。
                #   stop_at_end（ゴール/スタート/その場旋回の手前）: 終端で 0
                #   continue（末尾セルの出口が未観測）: 末尾セルへ**進入する
                #     時点**で v_uncertain
                v_end = 0.0 if stop_at_end else min(v_ceil_base, self.v_uncertain)
                s_term = path.length if stop_at_end else (path.length - self.cell_size / 2.0)
                # 【重要】v_ceil_base まで加速してよいのは、そこから v_end へ
                # 落としきるだけの距離が実際にあるときだけ。これを怠ると、
                # 「区画1つぶんの既知区間で全開加速 → 即全開制動」を区画ごとに
                # 繰り返す鋸歯状の速度指令になり、経路追従が破綻する
                # （実測: 横偏差が45mmまで拡大し探索走行で壁に衝突。
                #  速度計画としては各区間で時間最適だが、車体が追従できない）。
                # 距離が足りないときは全区間を保守的な上限に固定して等速で走る。
                # 逆に E1 の追加探索のように**既知の通路を長く走る**場合は
                # 距離が足りるので v_cap まで加速する（教授裁定「既知は速く、
                # 未知は慎重に」の実効的な実装）。
                d_brake = max((v_ceil_base ** 2 - v_end ** 2) / (2.0 * self.a_max), 0.0)
                v_ceil_arr = np.full(len(path.s), v_ceil_base, dtype=float)
                if s_term < d_brake:
                    v_ceil_arr[:] = max(v_end, min(v_ceil_base, self.v_uncertain))
                elif not stop_at_end:
                    v_ceil_arr[path.s >= s_term] = v_end
                path.speed = build_speed_profile(path.s, path.curvature, v_ceil_arr,
                                                  self.a_lat, self.a_max, self.v_creep,
                                                  stop_at_end)
                self._path = path
                self._cursor = 0
                self._path_ticks = 0
                self._path_end_reason = end_reason
                self._state = "DRIVE"
                return

            if chain["need_turn_to"] is not None:
                self._begin_pre_turn(x, y, cells[0], chain["need_turn_to"])
                return

            if chain["reached_target"]:
                # 現在セルが既に目標集合（境界ケース: 反転直後に現在セルが
                # 早々に新しい目標集合にも属していた等）。反転して再試行する。
                self._flip_target_mode()
                continue

            break

        # 到達可能な隣接セルが無い（安全弁）。IDLEで0V待機し、評価器の
        # スタック検出に委ねる（L0-a/L0-bと同じ思想）。
        self._state = "IDLE"
        self._path = None

    def _on_path_complete(self, x: float, y: float, yaw: float) -> None:
        reason = self._path_end_reason
        if reason == "pre_turn":
            self._enter_turn_inplace(self._pending_dir_after_turn)
            return
        if reason in ("goal", "start"):
            # ゴール停止(v_ref=0まで減速)直後は、車輪PIの積分項が減速中の
            # 負の誤差を蓄積したまま残っている。反転直後は v_ref が
            # 0→巡航速度へ不連続に飛ぶため、この古い積分値がそのまま
            # 上乗せされて大きな過渡応答（ヨー発散）を起こす不具合を
            # スモークテストで確認した。TURN_INPLACE遷移時と同様にここでも
            # リセットする。
            self._wheel_pi_l.reset()
            self._wheel_pi_r.reset()
            self._flip_target_mode()
        self._replan(x, y, yaw)

    # ------------------------------------------------------------------
    # ヘルパー: 方位推定、経路完了判定、カーソル探索、再計画トリガー判定
    # ------------------------------------------------------------------
    @staticmethod
    def _snap_heading(heading_rad: float) -> str:
        """方位角[rad]を最も近い4方位文字に丸める（直線区間はcurvature=0で
        厳密に軸に平行なので、ここでの丸め誤差は数値誤差レベルのみ）。"""
        best_d, best_diff = "N", None
        for d, hr in _HEADING_RAD.items():
            diff = abs(_wrap_pi(hr - heading_rad))
            if best_diff is None or diff < best_diff:
                best_diff, best_d = diff, d
        return best_d

    def _check_drive_completion(self, x: float, y: float, v_fwd: float) -> bool:
        path = self._path
        at_end_idx = self._cursor >= len(path.s) - 3
        if not at_end_idx:
            return False
        dist_to_end = math.hypot(x - path.x[-1], y - path.y[-1])
        if path.stop_at_end:
            # 通常は位置・速度の両方の条件で判定する（速度条件を外すと、
            # まだ高速走行中に終端付近を通過しただけで「完了」と誤判定し、
            # 減速しないままその場旋回へ突入して衝突する不具合があった）。
            # ただし位置がgoal_done_distへわずかに届かない場合に限り、
            # 速度指令(v_setpoint)が既に0まで減速し実際の速度も十分小さければ
            # 完了とみなすフォールバックを追加する。速度指令0からはこれ以上
            # 有意に前進できないため、位置しきい値だけで厳密判定すると
            # 数mmの差で永久に完了しないデッドロックに陥る不具合を
            # スモークテストで確認した（cur=len-1のまま停止し続けるケース）。
            if dist_to_end < self.goal_done_dist:
                return abs(v_fwd) < self.goal_done_speed
            return (abs(v_fwd) < self.goal_done_speed) and (self._v_setpoint < self.goal_done_speed)
        # 停止しない経路（"continue"）: 位置しきい値だけで判定すると、1ティック
        # の前進量が大きいときに 2cm の窓を跨いで永久に完了せず、カーソルが
        # 終端に張り付いて盲目的に直進する。終端点を**通り過ぎた**ことも
        # 完了とみなす（終端の参照方位への符号付き射影で判定）。
        if dist_to_end < self.goal_done_dist:
            return True
        h_end = float(path.heading[-1])
        passed = (x - float(path.x[-1])) * math.cos(h_end) \
            + (y - float(path.y[-1])) * math.sin(h_end)
        return passed >= 0.0

    def _advance_cursor(self, x: float, y: float) -> int:
        path = self._path
        n = len(path.s)
        lo = self._cursor
        hi = min(n, lo + self.cursor_search_window)
        best_i, best_d2 = lo, None
        for i in range(lo, hi):
            d2 = (path.x[i] - x) ** 2 + (path.y[i] - y) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    # ------------------------------------------------------------------
    # 制御則本体（§10）
    # ------------------------------------------------------------------
    def _wheel_targets_to_voltage(self, v_cmd: float, omega_cmd: float, obs: np.ndarray):
        """(前進速度指令, ロボット角速度指令[正=反時計回り]) → 左右目標車輪角速度
        → 逆モデル前向き補償+PI電圧（WheelPI）。運動学はL0-a/L0-bと同一符号規約。"""
        r = self.wheel_radius
        tread = self.tread
        omega_l_des = v_cmd / r - omega_cmd * tread / (2.0 * r)
        omega_r_des = v_cmd / r + omega_cmd * tread / (2.0 * r)
        omega_l_act = float(obs[self._i_wheel])
        omega_r_act = float(obs[self._i_wheel + 1])
        vl = self._wheel_pi_l.step(omega_l_des, omega_l_act, self.control_dt)
        vr = self._wheel_pi_r.step(omega_r_des, omega_r_act, self.control_dt)
        return vl, vr

    def _do_drive_control(self, obs: np.ndarray, x: float, y: float, yaw: float):
        """経路追従（Stanley標準形、§10-1）。

        omega_ref = curvature_ff・v_ref + K_psi・e_psi - atan(K_y・e_y/(|v|+v_eps))

        第1項は経路曲率によるフィードフォワード、第2項は方位偏差の比例
        フィードバック。第3項が刷新の核心（Stanley標準形）: 実速度|v|を
        分母に持つことで低速では横偏差の寄与が飽和し、L0-a/L0-bで確認された
        「低速域で横偏差PDと方位偏差PDが拮抗して準停留する」チャチャリングの
        根因を回避する（設計文書§10）。速度指令 v_ref は速度計画
        （build_speed_profile）が既にクリープ下限を折り込み済みなので、
        目標到達（ゴール）以外では0に張り付かない。
        """
        path = self._path
        idx = self._cursor
        px, py = float(path.x[idx]), float(path.y[idx])
        phead = float(path.heading[idx])
        pcurv = float(path.curvature[idx])
        v_target = float(path.speed[idx])

        # 速度指令のレートリミット（_reset_run_stateのdocstring参照）。
        # 上げ方向（加速）のみ a_max でレートリミットする。張り替え直後に
        # 指令が不連続に跳び上がると、車輪PIが大きなフィードフォワード
        # 電圧を要求して操舵に使う電圧の余裕がなくなりヨーが発散する
        # 不具合をスモークテストで確認したため。下げ方向（減速）は
        # レートリミットせず即座に追従する: 曲率0点でのv_ceilはbuild_
        # speed_profileの後方パスで既に「その先の停止/コーナーに間に合う
        # よう」位置基準でa_max制約済みであり、張り替え直後に「本来もっと
        # 減速していたはずの区間だった」と判明するケース（例: 直前まで
        # 気づかなかった至近距離のコーナー）では、むしろ即座に追従しないと
        # 制動距離が足りず壁に衝突する不具合を確認した。
        if self._v_setpoint < v_target:
            self._v_setpoint = min(v_target, self._v_setpoint + self.a_max * self.control_dt)
        else:
            self._v_setpoint = v_target
        v_ref = self._v_setpoint

        ux, uy = math.cos(phead), math.sin(phead)
        rel_x, rel_y = x - px, y - py
        e_y = rel_x * (-uy) + rel_y * ux  # 正=経路の左側（L0-a/L0-bのlateral_errと同じ符号規約）
        e_psi = _wrap_pi(phead - yaw)

        v_fwd, _omega_z = self._sim.privileged_velocity()
        v_for_gain = max(abs(v_fwd), 1e-6)
        lateral_term = math.atan2(self.k_y * e_y, v_for_gain + self.v_eps)

        omega_ref = pcurv * v_ref + self.k_psi * e_psi - lateral_term
        omega_ref = max(-self.turn_omega_limit, min(self.turn_omega_limit, omega_ref))

        return self._wheel_targets_to_voltage(v_ref, omega_ref, obs)

    def _turn_inplace_voltage(self, obs: np.ndarray, yaw_err: float, gyro_z: float):
        omega_cmd = self.kp_turn * yaw_err - self.kd_turn * gyro_z
        omega_cmd = max(-self.turn_omega_limit, min(self.turn_omega_limit, omega_cmd))
        return self._wheel_targets_to_voltage(0.0, omega_cmd, obs)

    def act(self, obs: np.ndarray):
        if self._sim is None:
            return 0.0, 0.0

        x, y, yaw = self._sim.privileged_pose()
        v_fwd, _omega_z = self._sim.privileged_velocity()

        if self._state == "INIT":
            cur_cell = pos_to_cell(x, y, self.width, self.height, self.cell_size)
            self._do_sense(*cur_cell)
            self._last_sensed_cell = cur_cell
            self._replan(x, y, yaw)

        if self._state == "DRIVE":
            cur_cell = pos_to_cell(x, y, self.width, self.height, self.cell_size)
            if cur_cell != self._last_sensed_cell:
                self._do_sense(*cur_cell)
                self._last_sensed_cell = cur_cell

        if self._state == "DRIVE" and self._path is not None:
            self._cursor = self._advance_cursor(x, y)
            idx = self._cursor
            self._path_ticks += 1
            # heading_dirの更新は "pre_turn" 以外の全チェーン（continue・
            # start・goal）の直線区間で行う。continue・start・goalは
            # 区画中心を通る格子沿いの直線＋円弧で構成されるため、曲率0点の
            # 方位を最寄りの4方位へ丸めるのは常に正しい。"pre_turn"（現在
            # 位置→セル中心への短い直線減速経路）だけは、始点が区画中心
            # からずれているため線分の幾何学的な向きが格子方位と一致すると
            # 限らず、ここでheading_dirを上書きすると直後のその場旋回・
            # 再計画が誤った方位を基準に行われる不具合をスモークテストで
            # 確認したため除外する。以前は"continue"以外を全て除外して
            # いたが、それでは"start"（帰路の一括経路）が複数の区画・
            # コーナーを経る間ずっとheading_dirが更新されず、その先の
            # "pre_turn"遷移で古い方位を使ってしまう不具合があったため
            # 範囲を広げた。
            if self._path_end_reason != "pre_turn" and abs(float(self._path.curvature[idx])) < 1e-6:
                self._heading_dir = self._snap_heading(float(self._path.heading[idx]))
            if self._path_end_reason == "continue":
                remaining = self._path.length - float(self._path.s[idx])
                # 【2026-08-10 是正】再計画は**曲率ゼロの直線区間にいるときだけ**
                # 行う。この条件は replan_trigger_dist の設計コメント（本クラスの
                # __init__）に「曲率ゼロの直線区間まで進んでから再計画する」と
                # 書かれていたが、**実装に入っていなかった**（曲率チェックは
                # すぐ上の方位更新にしか掛かっていない）。従来は探索走行の
                # チェーンに円弧が含まれなかったため露見しなかったが、探索でも
                # 円弧を張るようになると、円弧の途中で経路を張り替えてしまい、
                # かつ _heading_dir が円弧中は更新されない（古い方位のまま）ので
                # 誤った方位を基準に再計画して破綻する。
                # 【2026-08-10 是正2】最短走行(_explored_once)でも継ぎ足しを
                # 行う。旧実装は探索走行のみで継ぎ足し、最短走行の"continue"
                # チェーンは終端で**停止**する速度計画(decel_for_uncertainty)
                # に頼っていた。停止をやめて流して走るようにした結果、
                # 経路の終端に到達しても完了判定(_check_drive_completion)の
                # 位置しきい値2cmを跨いでしまうとカーソルが終端に張り付き、
                # 最後の参照点の方位・速度を出し続けて未知領域へ**盲目的に
                # 直進**して壁に衝突する不具合が実測で出た（検証帯 seed 4001
                # の最短走行が毎回 2.24s で衝突）。終端に達する前に継ぎ足す。
                on_straight = abs(float(self._path.curvature[idx])) < 1e-6
                if on_straight and remaining < self.replan_trigger_dist:
                    self._replan(x, y, yaw)
            if self._state == "DRIVE" and self._path is not None and self._check_drive_completion(x, y, v_fwd):
                self._on_path_complete(x, y, yaw)

        if self._state == "TURN_INPLACE":
            gyro_z = float(obs[self._i_gyro_z])
            yaw_err = _wrap_pi(self._turn_target_yaw - yaw)
            if (abs(yaw_err) < self.turn_done_rad) and (abs(gyro_z) < self.turn_done_gyro):
                self._heading_dir = self._pending_dir_after_turn
                self._wheel_pi_l.reset()
                self._wheel_pi_r.reset()
                self._last_sensed_cell = pos_to_cell(x, y, self.width, self.height, self.cell_size)
                self._replan(x, y, yaw)
            else:
                return self._turn_inplace_voltage(obs, yaw_err, gyro_z)

        if self._state == "DRIVE" and self._path is not None:
            # このティック内で_replan()が呼ばれ経路が張り替わっている場合に
            # 備え、制御直前に必ずカーソル探索をやり直す（_replan()自身は
            # カーソルを0にリセットするのみで、新しい経路上での実際の
            # 車体位置の探索まではしないため。これを怠ると、張り替え直後の
            # ティックで「新しい経路の先頭点（セル中心）」を実際の車体位置と
            # 誤認して大きなe_y/e_psiを計算し、暴走・壁衝突する不具合が
            # スモークテストで見つかった）。経路が張り替わっていなければ
            # 冪等（同じ結果）なので毎ティック呼んでも無害。
            self._cursor = self._advance_cursor(x, y)
            return self._do_drive_control(obs, x, y, yaw)

        return 0.0, 0.0
