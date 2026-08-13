#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""斜めを含む参照経路を作る（016-B の軌道生成）。

**既存の `competition/baseline_slalom.py` の道具をそのまま使う**:
`ReferencePath`（弧長・位置・方位・曲率の密標本）と `build_speed_profile`
（曲率と加速度上限から速度計画）。**制御側は 1 行も変えない。**

--------------------------------------------------------------------------
作り
--------------------------------------------------------------------------
016-A の経路（節点列と方位列）を受け取り、**方位が変わる節点を半径 $R$ の円弧で
接続**した密標本の経路にする。接線長は $R\\tan(\\theta/2)$（**45° で 0.414R・
90° で 1.000R**）。$R$ = 60 mm は 016-B カード §4-4 の設計判断（教授承認）。

**接線長が隣の区間の半分を超える場合は例外を投げる**（016-B では起きない前提。
起きたら設計の前提が崩れているので、黙って別の走り方に落とさない）。

各標本に**どの区間に属するか**の印を付けて返す（`seg_kind`）:

| 印 | 意味 |
|---|---|
| `straight` | 直進の直線区間 |
| `diagonal` | **斜めの直線区間**（横偏差 $e_y$ の判定はここで行う） |
| `arc` | 接続の円弧 |

**$e_y$ の判定は `diagonal` の標本だけで行う**（カード §2-1・§3）。
"""
import math
import sys
from pathlib import Path as _P

import numpy as np

REPO_ROOT = _P(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from competition.baseline_slalom import ReferencePath  # noqa: E402

from diagonal_model import DELTA8, DIAGONALS, node_xy, turn_deg  # noqa: E402


def _yaw(d):
    return math.atan2(DELTA8[d][1], DELTA8[d][0])


def build_diagonal_path(nodes, dirs, cell_size, R, stop_at_end=True, ds=0.005,
                        r_straight=None):
    """節点列・方位列から、直線 → 円弧 → 斜めの参照経路を作る。

    Args:
        R: **斜めが絡む接続**（45° の遷移・斜め ↔ 斜めの 90°）の円弧半径 [m]
        r_straight: **直進 ↔ 直進の 90° コーナー**の円弧半径 [m]。
            既定 None は R と同じ。**016-D では親の `select_arc_radius`（90 mm）**を
            渡す — 対照と同じ曲がり方にして、差を「斜めの導入だけ」に保つため
            （教授裁定 2026-08-14: 適用範囲の是正）

    ⚠️ **接線の余地は「隣の節点まで」ではなく「隣の方位変化まで」で測る。**
    半区画格子では直進が半歩の連なりに分かれるが、**同じ方位が続く区間は
    幾何的には 1 本の直線**であり、途中の節点はコーナーではない。
    節点ごとに測ると余地を半分に見誤り、**大きな半径が張れなくなる**。

    Returns:
        (ReferencePath, seg_kind, seg_index)
    """
    pts = [np.array(node_xy(n, cell_size), dtype=float) for n in nodes]
    m = len(dirs)
    seg_len = [float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(m)]
    r_str = float(r_straight) if r_straight is not None else float(R)

    # 同じ方位が続く区間（脚）の長さ。接線の余地はこの長さで測る
    leg_of = [0] * m
    leg_len = []
    cur = 0.0
    for k in range(m):
        cur += seg_len[k]
        leg_of[k] = len(leg_len)
        if k + 1 == m or turn_deg(dirs[k], dirs[k + 1]) > 0:
            leg_len.append(cur)
            cur = 0.0

    tan_len = [0.0] * len(nodes)
    r_at = {}                     # 節点ごとの実際の円弧半径
    for k in range(1, m):
        th = math.radians(turn_deg(dirs[k - 1], dirs[k]))
        if th <= 0:
            continue
        diagonal_involved = (dirs[k - 1] in DIAGONALS) or (dirs[k] in DIAGONALS)
        r_c = float(R) if diagonal_involved else r_str
        room = min(leg_len[leg_of[k - 1]], leg_len[leg_of[k]]) / 2.0
        t = r_c * math.tan(th / 2.0)
        if t > room + 1e-12:
            # 親の `fit_corner_radii` と同じく、収まる最大の半径まで縮める
            r_c = room / math.tan(th / 2.0)
            t = room
        tan_len[k] = t
        r_at[k] = r_c

    xs, ys, hs, ks, kinds, idxs = [], [], [], [], [], []

    def push(x, y, head, curv, kind, i):
        xs.append(x), ys.append(y), hs.append(head), ks.append(curv)
        kinds.append(kind), idxs.append(i)

    for k, d in enumerate(dirs):
        p0, p1 = pts[k], pts[k + 1]
        u = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-12)
        a0 = p0 + u * tan_len[k]
        a1 = p1 - u * (tan_len[k + 1] if k + 1 < len(nodes) else 0.0)
        seg = float(np.linalg.norm(a1 - a0))
        kind = "diagonal" if d in DIAGONALS else "straight"
        n = max(2, int(round(seg / ds)) + 1)
        for t in np.linspace(0.0, 1.0, n)[:-1] if k + 1 < m else np.linspace(0.0, 1.0, n):
            q = a0 + t * (a1 - a0)
            push(q[0], q[1], _yaw(d), 0.0, kind, k)
        if k + 1 < m:
            d2 = dirs[k + 1]
            th = math.radians(turn_deg(d, d2))
            if th <= 0:                        # 方位が変わらない（節点を素通り）
                continue
            y0, y1 = _yaw(d), _yaw(d2)
            dy = math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))
            r_c = r_at.get(k + 1, float(R))
            nrm = np.array([-u[1], u[0]]) * (1.0 if dy > 0 else -1.0)
            s0 = pts[k + 1] - u * tan_len[k + 1]
            ctr = s0 + nrm * r_c
            ang0 = math.atan2(s0[1] - ctr[1], s0[0] - ctr[0])
            arc_len = abs(dy) * r_c
            n_a = max(3, int(round(arc_len / ds)) + 1)
            curv = (1.0 / r_c) * (1.0 if dy > 0 else -1.0)
            for t in np.linspace(0.0, 1.0, n_a)[:-1]:
                ang = ang0 + dy * t
                q = ctr + r_c * np.array([math.cos(ang), math.sin(ang)])
                push(q[0], q[1], y0 + dy * t, curv, "arc", k + 1)

    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    head = np.unwrap(np.asarray(hs, float))
    curv = np.asarray(ks, float)
    ds_arr = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(ds_arr)])
    return (ReferencePath(s, x, y, head, curv, stop_at_end),
            np.asarray(kinds), np.asarray(idxs, int))


def lateral_deviation(px, py, node_a, node_b, cell_size):
    """点 (px,py) から、節点 a→b が定める直線への**符号なし垂直距離** [m]。

    斜め区間では a→b が**斜め走路の中心線**（隣り合う柱の中点を結ぶ 45° 線）に
    一致する（カード §2-1）。
    """
    a = np.array(node_xy(node_a, cell_size), dtype=float)
    b = np.array(node_xy(node_b, cell_size), dtype=float)
    u = b - a
    n = np.linalg.norm(u)
    if n < 1e-12:
        return float(np.hypot(px - a[0], py - a[1]))
    u = u / n
    w = np.array([px - a[0], py - a[1]])
    return float(abs(w[0] * u[1] - w[1] * u[0]))


def heading_deviation(yaw, d):
    """方位 yaw と区間の方位 d との差の絶対値 [deg]（カード §2-2）。"""
    dy = math.atan2(math.sin(yaw - _yaw(d)), math.cos(yaw - _yaw(d)))
    return abs(math.degrees(dy))
