#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-G — **45° の遷移にだけクロソイド（緩和曲線）を挟んだ参照経路**。

教授裁定 2026-08-14（`card_016g.md` §6-3 の案 (a)）:

> **45° の遷移（直進 ↔ 斜め）にだけクロソイドを入れる。90° は現行の円弧のまま。**

--------------------------------------------------------------------------
なぜ新しいファイルを作るのか（作法 1）
--------------------------------------------------------------------------
**`diag_path.py` は変更しない。**016-B・016-C・016-F0・016-F0-b の全記録が
`build_diagonal_path` の出力に依存しており、**書き換えると過去の対照を
ビット単位で再現できなくなる**。

**本モジュールの `build_clothoid_path` は `build_diagonal_path` の写しに
「曲がり角の作り方」だけを差し替えたものである。**
`competition/reference_interp.py` の `ReferenceInterpMixin` と同じ作りで、
**写しが元からずれると気づけない**ので、`tests/test_clothoid_path.py` が

    **L_c = 0 なら `build_diagonal_path` と全配列がビット単位で一致すること**

を調整用迷路（seed 41000〜）の 20 迷路で検査する（元を書き換えたら落ちる）。

--------------------------------------------------------------------------
クロソイドの幾何
--------------------------------------------------------------------------
曲率が弧長に比例して 0 → 1/R まで増える緩和曲線を、円弧の**入口と出口**に挟む。

    τ   = L_c / (2R)                     … クロソイド 1 本ぶんの方位変化 [rad]
    X_s = ∫₀^{L_c} cos( σ² / (2·R·L_c) ) dσ
    Y_s = ∫₀^{L_c} sin( σ² / (2·R·L_c) ) dσ
    ΔR  = Y_s − R·(1 − cos τ)            … 円弧が内側へずれる量（シフト）[m]
    k   = X_s − R·sin τ
    T_s = k + (R + ΔR)·tan(θ/2)          … 交点から緩和曲線の始点までの接線長 [m]

制約は 2 つ（`run_016g_diag.py` の D3 と同じ）:
  **(1) T_s ≤ 余地**（現行の円弧接続と同じ余地に収まること）
  **(2) 2τ ≤ θ**（曲率が 1/R に達する前に曲がり終わってはいけない ＝ 円弧が残ること）

**どちらかを破る L_c が渡されたら、その曲がり角だけ収まる最大まで縮める**
（黙って別の走り方に落とさない。縮めた事実は `report` に残す）。

--------------------------------------------------------------------------
区間の印（`seg_kind`）について
--------------------------------------------------------------------------
**クロソイドの標本にも `arc` の印を付ける。**
016-C の判定は「`diagonal` の標本だけで横偏差を測る」「`arc` は 45° 旋回の費用として
別に数える」という形なので、**印を変えると判定の定義が動いてしまう。**
**判定の定義を変えないために、印は `arc` のままにする。**
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

# クロソイドを入れる旋回角 [deg]（裁定 (a): 45° だけ）
CLOTHOID_TURNS = (45,)


def _yaw(d):
    return math.atan2(DELTA8[d][1], DELTA8[d][0])


# ==========================================================================
# クロソイドの幾何
# ==========================================================================
def clothoid_tangent(L_c, R, theta, n=4001):
    """接線長 T_s・シフト ΔR・クロソイド 1 本の方位変化 τ を返す。"""
    if L_c <= 0.0:
        return R * math.tan(theta / 2.0), 0.0, 0.0
    tau = L_c / (2.0 * R)
    u = np.linspace(0.0, L_c, n)
    ang = u * u / (2.0 * R * L_c)
    X_s = float(np.trapezoid(np.cos(ang), u))
    Y_s = float(np.trapezoid(np.sin(ang), u))
    dR = Y_s - R * (1.0 - math.cos(tau))
    k = X_s - R * math.sin(tau)
    return k + (R + dR) * math.tan(theta / 2.0), dR, tau


def max_clothoid_len(R, theta, room, hi=0.20, tol=1e-9):
    """制約 (1) T_s ≤ room かつ (2) 2τ ≤ θ を満たす最大の L_c [m]。"""
    cap = R * theta                                   # 制約 (2)
    if clothoid_tangent(0.0, R, theta)[0] > room + 1e-12:
        return 0.0
    lo, high = 0.0, min(hi, cap)
    if clothoid_tangent(high, R, theta)[0] <= room:
        return high
    for _ in range(200):
        mid = 0.5 * (lo + high)
        if clothoid_tangent(mid, R, theta)[0] <= room:
            lo = mid
        else:
            high = mid
        if high - lo < tol:
            break
    return lo


def _integrate_curvature(kappa_of_s, length, ds_out, x0, y0, h0, h_fine=2e-4):
    """曲率の関数から標本列を作る（中点則で細かく積分し、ds_out ごとに拾う）。

    Returns: (xs, ys, heads, kaps) — **終点を含む**。
    """
    n_fine = max(4, int(math.ceil(length / h_fine)))
    h = length / n_fine
    n_out = max(2, int(round(length / ds_out)) + 1)
    want = np.linspace(0.0, length, n_out)

    xs, ys, hs, ks = [x0], [y0], [h0], [float(kappa_of_s(0.0))]
    s, x, y, head = 0.0, x0, y0, h0
    wi = 1
    for i in range(n_fine):
        s_mid = s + 0.5 * h
        h_mid = head + kappa_of_s(s_mid) * 0.5 * h     # 中点の方位
        x += h * math.cos(h_mid)
        y += h * math.sin(h_mid)
        head += kappa_of_s(s_mid) * h
        s += h
        # 出力の刻みを跨いだら拾う（細かい刻みの端点で近似する）
        while wi < n_out and s + 0.5 * h >= want[wi]:
            xs.append(x), ys.append(y), hs.append(head)
            ks.append(float(kappa_of_s(min(s, length))))
            wi += 1
    while len(xs) < n_out:
        xs.append(x), ys.append(y), hs.append(head), ks.append(float(kappa_of_s(length)))
    return (np.asarray(xs), np.asarray(ys), np.asarray(hs), np.asarray(ks))


def corner_samples(P, u_in, yaw_in, dy, R, L_c, ds, closure_tol=2e-6):
    """クロソイド → 円弧 → クロソイド の標本（**終点は含めない**）を返す。

    Args:
        P: 曲がり角（交点）の位置
        u_in: 進入方向の単位ベクトル
        yaw_in: 進入方向の方位 [rad]
        dy: 符号つきの旋回角 [rad]（左が正）
        R: 円弧部の半径 [m]／L_c: クロソイド 1 本の長さ [m]

    **閉合の検査つき**: 積分して出た終点が、幾何から決まる終点
    `P + u_out·T_s` と `closure_tol` 以内で一致することを確かめる
    （符号や積分の誤りをここで捕まえる）。
    """
    theta = abs(dy)
    sgn = 1.0 if dy > 0 else -1.0
    T_s, _dR, tau = clothoid_tangent(L_c, R, theta)
    arc_sweep = theta - 2.0 * tau
    if arc_sweep < -1e-12:
        raise ValueError(f"円弧が残らない: θ={math.degrees(theta):.2f}° "
                         f"に対し 2τ={math.degrees(2*tau):.2f}°")
    arc_sweep = max(arc_sweep, 0.0)

    start = P - u_in * T_s
    xs, ys, hs, ks = [], [], [], []

    def add(seg, drop_last=True):
        a, b, c, d = seg
        n = len(a) - 1 if drop_last else len(a)
        xs.extend(a[:n]), ys.extend(b[:n]), hs.extend(c[:n]), ks.extend(d[:n])
        return float(a[-1]), float(b[-1]), float(c[-1])

    x, y, head = float(start[0]), float(start[1]), float(yaw_in)

    # (1) 入口のクロソイド: κ(s) = sgn · s / (R·L_c)
    if L_c > 0.0:
        x, y, head = add(_integrate_curvature(
            lambda s: sgn * s / (R * L_c), L_c, ds, x, y, head))
    # (2) 円弧: κ = sgn / R
    if arc_sweep > 1e-12:
        x, y, head = add(_integrate_curvature(
            lambda s: sgn / R, R * arc_sweep, ds, x, y, head))
    # (3) 出口のクロソイド: κ(s) = sgn · (1 − s/L_c) / R
    if L_c > 0.0:
        x, y, head = add(_integrate_curvature(
            lambda s: sgn * (1.0 - s / L_c) / R, L_c, ds, x, y, head))

    # ---- 閉合の検査（符号・積分の誤りをここで捕まえる）----
    u_out = np.array([math.cos(yaw_in + dy), math.sin(yaw_in + dy)])
    want = P + u_out * T_s
    err = math.hypot(x - want[0], y - want[1])
    if err > closure_tol:
        raise AssertionError(
            f"クロソイド接続の閉合が合わない: 誤差 {err*1e6:.3f} µm "
            f"（θ={math.degrees(theta):.1f}°・R={R*1000:.1f} mm・L_c={L_c*1000:.2f} mm）")
    head_err = abs(math.atan2(math.sin(head - (yaw_in + dy)),
                              math.cos(head - (yaw_in + dy))))
    if head_err > 1e-6:
        raise AssertionError(f"クロソイド接続の方位が合わない: 誤差 {math.degrees(head_err):.2e}°")
    return np.asarray(xs), np.asarray(ys), np.asarray(hs), np.asarray(ks), T_s


# ==========================================================================
# 参照経路（**`diag_path.build_diagonal_path` の写し ＋ 曲がり角の差し替え**）
# ==========================================================================
def build_clothoid_path(nodes, dirs, cell_size, R, stop_at_end=True, ds=0.005,
                        r_straight=None, L_c=0.0, turns=CLOTHOID_TURNS, report=None):
    """直線 → クロソイド → 円弧 → クロソイド → 斜め の参照経路を作る。

    Args:
        L_c: **クロソイド 1 本の長さ [m]**。**0 なら現行の円弧接続と
            ビット単位で同じ出力**（`tests/test_clothoid_path.py` が検査する）
        turns: クロソイドを入れる旋回角 [deg] の組（裁定 (a) により既定は 45° のみ）
        report: dict を渡すと、曲がり角ごとの採用値を書き込む（縮めた事実も残る）

    Returns: (ReferencePath, seg_kind, seg_index) — `build_diagonal_path` と同じ
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
    r_at, lc_at = {}, {}
    corners = []
    for k in range(1, m):
        deg = turn_deg(dirs[k - 1], dirs[k])
        if deg <= 0:
            continue
        th = math.radians(deg)
        diagonal_involved = (dirs[k - 1] in DIAGONALS) or (dirs[k] in DIAGONALS)
        r_c = float(R) if diagonal_involved else r_str
        room = min(leg_len[leg_of[k - 1]], leg_len[leg_of[k]]) / 2.0
        t = r_c * math.tan(th / 2.0)
        if t > room + 1e-12:
            # **元と同じ**: 収まる最大の半径まで縮める（クロソイドは入れない）
            r_c = room / math.tan(th / 2.0)
            t = room
            lc = 0.0
        else:
            # ---- ここだけが元との差分: 対象の旋回角ならクロソイドを挟む ----
            lc = 0.0
            if L_c > 0.0 and deg in turns:
                lc = min(float(L_c), max_clothoid_len(r_c, th, room))
                if lc > 1e-9:
                    t = clothoid_tangent(lc, r_c, th)[0]
                else:
                    lc = 0.0
        tan_len[k] = t
        r_at[k] = r_c
        lc_at[k] = lc
        corners.append(dict(k=k, turn_deg=deg, room_m=room, R_m=r_c, L_c_m=lc,
                            tangent_m=t, shrunk=bool(L_c > 0.0 and deg in turns
                                                     and lc < L_c - 1e-9)))
    if report is not None:
        report["corners"] = corners
        report["L_c_requested_m"] = float(L_c)
        report["turns"] = tuple(turns)

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
            lc = lc_at.get(k + 1, 0.0)

            if lc > 0.0:
                # ---- 016-G: クロソイド → 円弧 → クロソイド ----
                cx, cy, ch, ck, _T = corner_samples(
                    pts[k + 1], u, y0, dy, r_c, lc, ds)
                for qx, qy, qh, qk in zip(cx, cy, ch, ck):
                    push(qx, qy, qh, qk, "arc", k + 1)
            else:
                # ---- 現行のまま（`build_diagonal_path` と 1 文字も違わない）----
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
