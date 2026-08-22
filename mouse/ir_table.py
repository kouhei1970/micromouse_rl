"""mouse/ir_table.py — IR センサの反射応答を「累積」表にする（1 枚の平面用）。

背景・事前登録: `verification/AUDIT_060_PREREG_table_sensor.md`。`mouse/ir_sensor.py::response()`
の面積分（`_integrate_facet`）を、1 枚の平面（厚みゼロ）に対して「壁に沿った位置 `u` について
1 回だけ積分し、`u` の大きい側から小さい側へ累積和を取る」ことで表にする。

これにより、任意の区間 `[a, b]`（`a < b`）への寄与を厳密に `G(a) - G(b)` として取り出せる
（区間ごとに `response()` を面ごとに呼び直すと求積格子が面ごとに変わって恒等式が崩れる。
事前登録 §0-2・§1「表の作り方」参照）。**面から表引きした値を組み立てて実際のセンサ応答に
する処理（`response_table()`、遮蔽・柱・複数壁の合成を含む）は次の作業段階で実装する。
本モジュールは「1 枚の平面の表を作る・保存する・読み込む・引き算する」までを担う。**

🔴 **`mouse/ir_sensor.py` は変更しない**（本モジュールは読み取りに import するだけ）。

## 表の軸の定義（座標系。ここが唯一の正本 — 次の作業段階もこの定義に従うこと）

面は世界座標 `x = 0` に固定し、外向き法線は `-x`（センサへ向く）。センサは常に `x < 0` 側。

- **`d`** [m]: センサ基準点（`IrSensorSpec.pos`、機体座標での取付基準点）から面までの
  垂直距離（＝ `-x_sensor`）。
- **`θ`** [deg]: LED の光軸（機体座標 `sensor.axis` を機体姿勢で回したもの）の**世界方位角**
  （`atan2(axis_y, axis_x)`、水平面内）そのもの。**`θ = 0` は LED 軸が面に正対（垂直入射）**。
  **正の `θ` は LED 軸を +y 側へ振る**（反時計回り）ことに対応する。
- **`u`** [m]: 面内で壁に沿った方向（世界 `y` 座標）の位置。**原点は LED の光軸が面（`x=0`）と
  交わる点**（`_axis_plane_crossing_y()` で解析的に厳密計算。`d`・`θ` から一意に定まる）。
  `u` はこの交点からのオフセット。**正の `u` は +y 側**。
  縦配置（`IrSensorSpec.layout="vertical"`、既定）では LED と PT の水平位置が一致するため、
  この原点は PT 側の光軸ともほぼ一致する（アライメント誤差 `tilt=0` の既定仕様のとき厳密に一致。
  `tilt≠0` を使う場合は本表の前提が崩れるので使わないこと）。

高さ方向（面の `v` 軸、`z = 0 〜 wall_height_m`）は表の軸に**含めない**。機体・壁の仕様上
高さは常に固定（`DEFAULT_WALL_HEIGHT_M`）なので、生成のたびに高さ方向は必ず全域を数値積分
してしまい、`u` 方向だけを表の軸として残す。

`G(d, θ, u)` ＝ この面が `u` から `+y` 方向（`+∞`）に広がっているときの、センサ 1 本の応答。
`gain` は表に焼き込まない（`IrSensorSpec.gain` は使わない＝ 1.0 相当のまま）。反射面は既定の
`SurfaceSpec()`（`diffuse=0.8`・鏡面込み）。**床は含めない**（`research_notes/note_034...` 追記16、
ユーザの決定）。

## 表の作り方（事前登録 §1 の指示どおり、`u` 方向に沿って 1 回の積分で累積する）

`(d, θ)` の組ごとに:

1. `u` 軸（表の刻み 0.5mm。保存する `±120mm` の範囲より `U_MARGIN_M` だけ広く取った、
   生成用の余白込みの節点列）が作る **`M-1` 個のビン**（隣り合う節点の間）それぞれについて、
   **ビンの中点**で被積分値を評価する（`response()` の warped 格子とは違い、`u` 方向は
   意図的に一様。理由は本モジュール docstring 末尾「なぜ `u` 方向は一様格子か」を参照）。
   **高さ（`v`）方向は `response()` と同じ tan ワープ格子**（`_warped_axis`。狭い指向性を
   踏み外さないため）で求積点を取る。
2. 被積分値は `_integrate_facet`（`mouse/ir_sensor.py`）と同じ式（LED 側 × 反射 × PT 側）。
   高さ方向だけ先に足し込み、ビンごとの積分値（面積要素込み）を作る（`_facet_u_density`・
   `generate_dtheta_bin_integrals`）。
3. ビン積分を `u` の大きい側から累積和を取ると、各節点での `G(d,θ,u)` が 1 回の積分で
   全部そろう（`cumulate_from_bin_integrals`）。

🔴 **ビンは中点で評価すること（左端＝節点そのもので評価してはならない）**: 節点そのもの
（＝ビンの左端）で評価して刻み幅を掛けると、左端則（積分誤差 `O(du)`）になり、0.5mm という
刻みに対して不釣り合いに大きい誤差が出ることが実測で分かった（`d=113mm, θ=7°` で
`[2,85]mm` の区間を左端則で積分すると 0.059835、十分細かい格子で収束させた真値は
0.056380、満量比で 4.2e-3 の差＝事前登録の判定閾値 1e-4 の 42 倍）。**ビンの中点で評価する
中点則（誤差 `O(du²)`）に変えると、同じ 0.5mm 刻み・同じ計算コストのまま、上の例で
満量比 1.9e-5 まで縮む**（Gauss-Legendre 2〜8点による収束値との比較で確認済み。中点則
1点で既に収束値と一致する＝被積分関数は各 0.5mm ビンの内側では滑らかで、必要なのは
評価点の位置の取り方であって、格子の細かさそのものではなかった）。

## なぜ `u` 方向は一様格子か（`response()` の warped 格子と違う理由）

`response()` の `_facet_grid` は面 1 枚の**合計**を精度よく求めるために、LED の明部へ求積点を
集中させる非一様（tan ワープ）格子を使う。本モジュールは合計ではなく**「どこまで足したか」を
表す累積量**が欲しいので、ビンの境界が必ず表の刻み（0.5mm）と一致していないと「区間 `[a,b]`
の寄与は `G(a)-G(b)` である」という恒等式が保証されない（ビンの境界が区間の境界と一致しないと、
その境界をまたぐビンの重みを裂けない）。0.5mm 刻みのビン分割そのものは、上の「中点で評価」の
とおりビンの中点で評価する限り十分な精度が出ることを実測で確認済み。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from mouse.ir_sensor import (
    DEFAULT_MAX_RANGE_M,
    DEFAULT_WALL_HEIGHT_M,
    IrSensorSpec,
    SurfaceSpec,
    _Facet,
    _facet_anchor_and_scale,
    _sensor_world_geometry,
    _solve_pose_for_sensor_target,
    _warped_axis,
)

__all__ = [
    "D_AXIS_M",
    "THETA_AXIS_DEG",
    "U_AXIS_M",
    "U_MARGIN_M",
    "U_STEP_M",
    "N_V_QUAD",
    "CumulativeTable",
    "lf_sensor_spec",
    "generate_dtheta_bin_integrals",
    "cumulate_from_bin_integrals",
    "build_cumulative_table",
    "save_cumulative_table",
    "load_cumulative_table",
    "segment_from_indices",
    "nearest_index",
    "interp_G",
]


# ============================================================================
# 表の軸（事前登録 §1。刻み・点数は変えない）
# ============================================================================
def _axis_points(lo: float, hi: float, step: float, n: int) -> np.ndarray:
    """`lo` から `step` 刻みで `n` 点を作る（`hi` は仕様の記述どおりの上限の目安として
    引数に残すが、実際に使うのは `lo`・`step`・`n` の 3 つだけ。下の「d 軸の端点について」
    参照）。"""
    pts = lo + step * np.arange(n)
    assert abs(float(pts[-1]) - hi) < step + 1e-9, (
        f"軸の端点が事前登録の記述と大きくずれている: 計算値={pts[-1]}, 記述={hi}"
    )
    return pts


# --- d 軸の端点について（🔴 事前登録の記述に内部矛盾があった。報告参照） ---
# 事前登録 §1 の表は「距離 d: 範囲 5〜350mm・刻み2mm・点数173」と書いてあるが、
# 5mmから2mm刻みで173点取ると最後の点は 5+172*2=349mm であり、350mm ちょうどには
# ならない（350mmちょうどにするには開始5mmではなく6mmが要る）。刻み2mmと点数173は
# 🔴（太字）で強調されている2つの値なので、この2つを厳守し、開始点5mmも明記通りに使う
# （＝終点は349mm。350mmという記述は近似的な説明と解釈した）。詳しくは本作業の報告を参照。
D_MIN_M = 0.005
D_STEP_M = 0.002
D_N = 173
D_AXIS_M = _axis_points(D_MIN_M, 0.350, D_STEP_M, D_N)   # 5, 7, ..., 349 [mm]

THETA_MIN_DEG = -85.0
THETA_STEP_DEG = 1.0
THETA_N = 171
THETA_AXIS_DEG = _axis_points(THETA_MIN_DEG, 85.0, THETA_STEP_DEG, THETA_N)   # -85..+85 [deg]

U_MAX_M = 0.120
U_STEP_M = 0.0005
U_N = 481
U_AXIS_M = _axis_points(-U_MAX_M, U_MAX_M, U_STEP_M, U_N)   # -120..+120 [mm]

# --- 生成時だけ使う余白（自己検査3: ±120mm で十分かを確かめるための余白）。
# 保存する表には入らない（保存直前に中央 481 点へ切り出す）。
#
# 🔴 実測で 30mm では収束しないことが分かった（片側30mmで打ち切ると、極端な斜入射
# （|θ|≈76〜85°）では G(+120mm) の値が真値（さらに広い余白での収束値）より約2〜3割
# 小さく出る＝表の値そのものが不正確になる）。100mm まで広げると、220mm・400mm と
# 比べて機械精度の桁まで一致する（実測。詳細は本作業の報告参照）。そこで生成時の余白は
# 100mm を既定にする（**登録された表の軸 `U_AXIS_M`（±120mm・0.5mm刻み・481点）は
# 変えていない。ここは値を正しく積むための内部の計算範囲であり、保存される表の形は
# 従来どおり**）。
U_MARGIN_M = 0.100   # 片側100mm（0.5mm刻みで200ノード）。収束の実測に基づく既定値

# 高さ（v）方向の求積点数（response() の n_grid=48 の基準に合わせる。自己検査1の対象）。
N_V_QUAD = 48


# ============================================================================
# センサ仕様（LF。事前登録 §0 の指示どおり params.py を正本とする）
# ============================================================================
def lf_sensor_spec() -> IrSensorSpec:
    """`mouse/params.py::RobotParams().sensors` の "LF" から `IrSensorSpec` を作る
    （`gain` は既定の 1.0 のまま＝表に焼き込まない）。"""
    from mouse.params import RobotParams

    p = RobotParams()
    for s in p.sensors:
        if s["name"] == "LF":
            pos = tuple(float(v) for v in s["pos"].split())
            axis = tuple(float(v) for v in s["zaxis"].split())
            return IrSensorSpec(name="LF", pos=pos, axis=axis)
    raise ValueError('RobotParams().sensors に "LF" が見つからない')


# ============================================================================
# 幾何: (d, θ) から姿勢・LED光軸と面の交点を作る
# ============================================================================
def _pose_for_dtheta(sensor: IrSensorSpec, d_m: float, theta_deg: float) -> Tuple[float, float, float]:
    """`d`・`θ` の定義どおりの機体姿勢を作る。センサ基準点は世界 `(-d, 0)`、
    LED 光軸の世界方位角がちょうど `theta_deg` になるように機体の向きを逆算する。"""
    az_body = math.atan2(sensor.axis[1], sensor.axis[0])
    theta_pose = math.radians(theta_deg) - az_body
    return _solve_pose_for_sensor_target(sensor, (-d_m, 0.0), theta_pose)


def _axis_plane_crossing_y(led) -> float:
    """LED の光軸（`led.pos`・`led.axis`、ワールド座標）が世界座標 `x=0` の平面と交わる点の
    `y` 座標を解析的に求める（`u` の原点）。"""
    if abs(float(led.axis[0])) < 1e-12:
        raise ValueError("LED 光軸が面とほぼ平行（axis_x ~ 0）で交点が定義できません")
    t = -float(led.pos[0]) / float(led.axis[0])
    return float(led.pos[1]) + t * float(led.axis[1])


# ============================================================================
# 求積（u は一様格子・v は response() と同じ tan ワープ格子）
# ============================================================================
def _facet_u_density(
    facet: _Facet, led, pt, surf: SurfaceSpec,
    points: np.ndarray, dA: np.ndarray, max_range_m: float,
) -> np.ndarray:
    """`mouse/ir_sensor.py::_integrate_facet` と同じ被積分式（LED側×反射×PT側）で、
    高さ（v・points/dA の第2軸）だけ先に足し込んだ `u` ごとの密度を返す（shape `(n_u,)`）。
    `led_intensity=1.0`・`pt_responsivity=1.0`・`gain` は掛けない（事前登録の指示どおり）。
    """
    d_e = points - led.pos
    r_e = np.linalg.norm(d_e, axis=-1)
    r_e_safe = np.maximum(r_e, 1e-6)
    dir_e = d_e / r_e_safe[..., None]

    d_v = pt.pos - points
    r_v = np.linalg.norm(d_v, axis=-1)
    r_v_safe = np.maximum(r_v, 1e-6)
    dir_v = d_v / r_v_safe[..., None]

    cos_e = np.clip(np.einsum("ijk,k->ij", dir_e, led.axis), 0.0, 1.0)
    irradiance = cos_e ** led.m / (r_e_safe ** 2)

    cos_i = np.clip(np.einsum("ijk,k->ij", -dir_e, facet.normal), 0.0, 1.0)
    radiance = irradiance * cos_i * (surf.diffuse / math.pi)

    if surf.specular > 0.0:
        n = facet.normal
        dot_dn = np.einsum("ijk,k->ij", dir_e, n)
        reflect_dir = dir_e - 2.0 * dot_dn[..., None] * n[None, None, :]
        cos_phi = np.clip(np.einsum("ijk,ijk->ij", reflect_dir, dir_v), 0.0, 1.0)
        radiance = radiance + irradiance * cos_i * surf.specular * cos_phi ** surf.shininess

    cos_v = np.clip(np.einsum("ijk,k->ij", dir_v, facet.normal), 0.0, 1.0)
    cos_r = np.clip(np.einsum("ijk,k->ij", -dir_v, pt.axis), 0.0, 1.0)
    pt_sensitivity = cos_r ** pt.m

    contribution = radiance * cos_v * pt_sensitivity / (r_v_safe ** 2)
    valid = (r_e < max_range_m) & (r_e > 1e-9) & (r_v > 1e-9)
    contribution = np.where(valid, contribution, 0.0)

    return np.sum(contribution * dA, axis=1)   # v 軸（第2軸）を足し込む → shape (n_u,)


def generate_dtheta_bin_integrals(
    sensor: IrSensorSpec, surf: SurfaceSpec, d_m: float, theta_deg: float,
    u_axis_m: np.ndarray, *, n_v: int = N_V_QUAD,
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M, max_range_m: float = DEFAULT_MAX_RANGE_M,
) -> np.ndarray:
    """`(d, θ)` 1 組について、`u_axis_m`（一様格子。表の刻みと揃えること。`M` 点）が作る
    `M-1` 個のビン `[u_axis_m[i], u_axis_m[i+1]]` それぞれの積分値を返す（shape `(M-1,)`）。

    🔴 **ビンの中点で評価すること（実装中に発覚した誤り）**: 最初の実装は `u_axis_m` の
    各点そのもの（＝ビンの左端）で被積分値を評価し、それに刻み幅を掛けてビン積分としていた。
    これは左端則（誤差 `O(du)`）に相当し、自己検査1・2 が 0.5mm 刻みのまま大きく不合格に
    なった（実測: `d=113mm, θ=7°, [2,85]mm` で左端則 0.059835・真値（細かい格子で収束）
    0.056380・満量比で 4.2e-3 の差。0.5mm という刻みの細かさに対して不釣り合いに大きい
    誤差だった）。**ビンの中点で評価する中点則（誤差 `O(du²)`）に変えると、同じ 0.5mm 刻み・
    同じ計算コストのまま、上の例で満量比 1.9e-5 まで縮む**（Gauss-Legendre 2〜8点による
    収束値との比較で確認済み。中点則1点で既に収束値と一致）。詳細は本作業の報告を参照。
    """
    pose = _pose_for_dtheta(sensor, d_m, theta_deg)
    led, pt = _sensor_world_geometry(sensor, pose)
    y_cross = _axis_plane_crossing_y(led)

    u_mid = (u_axis_m[:-1] + u_axis_m[1:]) / 2.0   # 各ビンの中点（M-1 点）
    du = u_axis_m[1:] - u_axis_m[:-1]               # 通常はどれも U_STEP_M に等しい

    facet = _Facet(
        center=np.array([0.0, y_cross, wall_height_m / 2.0]),
        u=np.array([0.0, 1.0, 0.0]), v=np.array([0.0, 0.0, 1.0]),
        normal=np.array([-1.0, 0.0, 0.0]),
        half_u=float(np.max(np.abs(u_axis_m))) + 1e-6, half_v=wall_height_m / 2.0,
    )
    half_angle_max = max(sensor.led_half_angle_deg, sensor.pt_half_angle_deg)
    _anchor_u, anchor_v, w = _facet_anchor_and_scale(facet, led, half_angle_max, sensor.separation_m)
    sv, wv = _warped_axis(facet.half_v, anchor_v, w, n_v)

    SU, SV = np.meshgrid(u_mid, sv, indexing="ij")
    WV = np.tile(wv[None, :], (len(u_mid), 1))
    dA = WV   # v 方向の重みのみ（u 方向は後で du を掛ける。中点則は「1点×ビン幅」）
    points = (
        facet.center[None, None, :]
        + SU[:, :, None] * facet.u[None, None, :]
        + SV[:, :, None] * facet.v[None, None, :]
    )
    density = _facet_u_density(facet, led, pt, surf, points, dA, max_range_m)   # shape (M-1,)
    return density * du


def cumulate_from_bin_integrals(bin_integrals: np.ndarray) -> np.ndarray:
    """ビン積分（`M-1` 個）から、各節点での累積 `G`（`M` 個）を作る。

    `G[k] = Σ_{i=k}^{M-2} bin_integrals[i]`（節点 `k` から先のビンをすべて足す）。
    最後の節点（`u_axis_m` の末尾＝生成時の余白の外縁）は `G=0` とする
    （その先には面が無い、という打ち切りの近似。余白の取り方は自己検査3 で確かめる）。
    """
    n_bins = len(bin_integrals)
    G = np.empty(n_bins + 1, dtype=bin_integrals.dtype)
    G[-1] = 0.0
    G[:-1] = np.cumsum(bin_integrals[::-1])[::-1]
    return G


# ============================================================================
# 表: 生成・保存・読み込み・引き算
# ============================================================================
@dataclass
class CumulativeTable:
    """`G(d, θ, u)` の表（1 枚の平面。厚みゼロ。床なし・ρ=0.8・鏡面込み・gain=1.0）。"""

    d_axis_m: np.ndarray
    theta_axis_deg: np.ndarray
    u_axis_m: np.ndarray
    values: np.ndarray          # shape (len(d_axis_m), len(theta_axis_deg), len(u_axis_m))  float32
    meta: Dict


def build_cumulative_table(
    sensor: Optional[IrSensorSpec] = None, surf: Optional[SurfaceSpec] = None,
    *, d_start: int = 0, d_end: Optional[int] = None,
    n_v: int = N_V_QUAD, u_margin_m: float = U_MARGIN_M,
    progress_every: int = 2000,
) -> Tuple[np.ndarray, Dict]:
    """`D_AXIS_M[d_start:d_end] × THETA_AXIS_DEG` の累積表を作る（`d_start`/`d_end` で
    距離軸を分割できる。10 分以内に収まらない見込みのときはこれで分割して呼び出すこと）。

    戻り値: `(values, info)`。`values` は shape `(d_end-d_start, len(THETA_AXIS_DEG), U_N)`
    の float32 配列。`info` には所要時間・自己検査3 用の残差サマリを入れる。
    """
    sensor = sensor or lf_sensor_spec()
    surf = surf or SurfaceSpec()
    d_end = len(D_AXIS_M) if d_end is None else d_end

    n_margin_nodes = int(round(u_margin_m / U_STEP_M))
    u_gen_axis = _axis_points(
        -(U_MAX_M + u_margin_m), U_MAX_M + u_margin_m, U_STEP_M, U_N + 2 * n_margin_nodes,
    )
    lo, hi = n_margin_nodes, n_margin_nodes + U_N

    n_d = d_end - d_start
    n_theta = len(THETA_AXIS_DEG)
    values = np.empty((n_d, n_theta, U_N), dtype=np.float32)

    # 自己検査3 用: forward残差 = G(+120mm)（満量スケール前の生値。呼び出し側で満量比にする）
    #             backward残差 = G(-150mm) - G(-120mm)（margin ぶん広げて余分に取れる量）
    forward_residual = np.empty((n_d, n_theta), dtype=np.float64)
    backward_residual = np.empty((n_d, n_theta), dtype=np.float64)

    t0 = time.time()
    for ii, i in enumerate(range(d_start, d_end)):
        d = float(D_AXIS_M[i])
        for j, th in enumerate(THETA_AXIS_DEG):
            bins = generate_dtheta_bin_integrals(sensor, surf, d, float(th), u_gen_axis, n_v=n_v)
            G_ext = cumulate_from_bin_integrals(bins)
            values[ii, j, :] = G_ext[lo:hi].astype(np.float32)
            forward_residual[ii, j] = float(G_ext[hi - 1])          # G(+120mm)
            backward_residual[ii, j] = float(G_ext[0] - G_ext[lo])  # G(-150mm) - G(-120mm)
        if progress_every and (ii + 1) % progress_every == 0:
            elapsed = time.time() - t0
            done = (ii + 1) * n_theta
            total = n_d * n_theta
            print(f"  d[{d_start}:{d_end}] 進捗 {ii+1}/{n_d} 行 "
                  f"({done}/{total} 組, {elapsed:.1f}s, {elapsed/done*1000:.3f}ms/組)")
    elapsed = time.time() - t0

    info = {
        "d_start": d_start, "d_end": d_end, "elapsed_s": elapsed,
        "n_v": n_v, "u_margin_m": u_margin_m,
        "forward_residual_max": float(np.max(np.abs(forward_residual))),
        "backward_residual_max": float(np.max(np.abs(backward_residual))),
        "sensor_name": sensor.name, "sensor_pos": list(sensor.pos), "sensor_axis": list(sensor.axis),
        "surf_diffuse": surf.diffuse, "surf_specular": surf.specular, "surf_shininess": surf.shininess,
    }
    return values, info


def save_cumulative_table(table: CumulativeTable, path) -> None:
    """表を npz 形式（単精度）で保存する。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        d_axis_m=table.d_axis_m.astype(np.float64),
        theta_axis_deg=table.theta_axis_deg.astype(np.float64),
        u_axis_m=table.u_axis_m.astype(np.float64),
        values=table.values.astype(np.float32),
        meta_json=np.array(json.dumps(table.meta)),
    )


def load_cumulative_table(path) -> CumulativeTable:
    data = np.load(str(path), allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    return CumulativeTable(
        d_axis_m=data["d_axis_m"], theta_axis_deg=data["theta_axis_deg"], u_axis_m=data["u_axis_m"],
        values=data["values"], meta=meta,
    )


# ============================================================================
# 引き算（面の寄与）: 格子点そのもの（線形補間なし）
# ============================================================================
def nearest_index(axis: np.ndarray, value: float) -> int:
    """`axis`（昇順）上で `value` に最も近い格子点の添字を返す（丸め込み。補間はしない）。"""
    idx = int(np.searchsorted(axis, value))
    if idx <= 0:
        return 0
    if idx >= len(axis):
        return len(axis) - 1
    lo, hi = axis[idx - 1], axis[idx]
    return idx - 1 if (value - lo) <= (hi - value) else idx


def segment_from_indices(table: CumulativeTable, i_d: int, i_theta: int, i_u_a: int, i_u_b: int) -> float:
    """区間 `[u_a, u_b]`（`u_a <= u_b` を仮定）への面の寄与を、表の格子点そのもの
    （線形補間なし）で `G(u_a) - G(u_b)` として返す。"""
    return float(table.values[i_d, i_theta, i_u_a] - table.values[i_d, i_theta, i_u_b])


# ============================================================================
# 3次元線形補間（第2段階: response_table() から使う。事前登録の「組み上げ」節
# 「G(d,θ,u_start)−G(d,θ,u_end)（表の3次元線形補間）」に対応する）
# ============================================================================
def interp_G(table: CumulativeTable, d_m, theta_deg, u_m) -> np.ndarray:
    """`G(d,θ,u)` を表の格子上で3次元線形補間する（ベクトル化。`d_m`/`theta_deg`/`u_m` は
    同じ形の `np.ndarray` かスカラー。戻り値は入力と同じ形）。

    3軸とも等間隔格子（`d_axis_m`・`theta_axis_deg`・`u_axis_m` はいずれも `_axis_points`
    で作った一様刻み）なので、各軸で `(値-始点)/刻み` から浮動小数の格子位置を出し、
    整数部で下側ノードを、小数部で補間比を取る。範囲外は軸の端にクランプする
    （`nearest_index`/`segment_from_indices` と同じ「格子の外は端の値」という規約を、
    ここでは最近傍ではなく線形補間の対象範囲のクランプとして踏襲する）。
    """
    d_arr = np.asarray(d_m, dtype=np.float64)
    th_arr = np.asarray(theta_deg, dtype=np.float64)
    u_arr = np.asarray(u_m, dtype=np.float64)

    def _frac(axis: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo = float(axis[0])
        step = float(axis[1] - axis[0])
        n = len(axis)
        pos = (val - lo) / step
        pos = np.clip(pos, 0.0, n - 1)
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, n - 2)
        frac = pos - i0
        return i0, frac

    id0, fd = _frac(table.d_axis_m, d_arr)
    it0, ft = _frac(table.theta_axis_deg, th_arr)
    iu0, fu = _frac(table.u_axis_m, u_arr)

    V = table.values

    def _g(di: int, ti: int, ui: int) -> np.ndarray:
        return V[id0 + di, it0 + ti, iu0 + ui].astype(np.float64)

    c000, c001 = _g(0, 0, 0), _g(0, 0, 1)
    c010, c011 = _g(0, 1, 0), _g(0, 1, 1)
    c100, c101 = _g(1, 0, 0), _g(1, 0, 1)
    c110, c111 = _g(1, 1, 0), _g(1, 1, 1)

    c00 = c000 * (1.0 - fd) + c100 * fd
    c01 = c001 * (1.0 - fd) + c101 * fd
    c10 = c010 * (1.0 - fd) + c110 * fd
    c11 = c011 * (1.0 - fd) + c111 * fd
    c0 = c00 * (1.0 - ft) + c10 * ft
    c1 = c01 * (1.0 - ft) + c11 * ft
    return c0 * (1.0 - fu) + c1 * fu


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="累積表（1枚の平面, G(d,θ,u)）を作る")
    parser.add_argument("--d-start", type=int, default=0)
    parser.add_argument("--d-end", type=int, default=len(D_AXIS_M))
    parser.add_argument("--out", type=str, required=True, help="出力 npz（部分結果。距離軸の一部）")
    parser.add_argument("--n-v", type=int, default=N_V_QUAD)
    args = parser.parse_args()

    sensor = lf_sensor_spec()
    surf = SurfaceSpec()
    values, info = build_cumulative_table(
        sensor, surf, d_start=args.d_start, d_end=args.d_end, n_v=args.n_v,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        values=values.astype(np.float32),
        d_start=args.d_start, d_end=args.d_end,
        info_json=np.array(json.dumps(info)),
    )
    print(f"保存: {out_path} shape={values.shape} elapsed={info['elapsed_s']:.1f}s "
          f"forward残差max={info['forward_residual_max']:.3e} "
          f"backward残差max={info['backward_residual_max']:.3e}")
