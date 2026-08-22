"""mouse/ir_sensor.py — IR LED ＋ フォトトランジスタ（PT）距離センサの放射モデル

背景・確定した物理パラメータ・設計方針は `research_notes/note_034_ir_sensor_model.md` を正とする
（本ファイルはそこに書かれた仕様の実装であり、数値の出典はすべて同ノート）。

## モデルの骨子

1 個のセンサは LED と PT の対（`IrSensorSpec`）。両者は機体座標での取り付け位置・光軸が
別々にある（並行だが一致しない。離隔 `separation_m`、並べ方 `layout`＝縦/横、
アライメント誤差 `led_tilt_deg`/`pt_tilt_deg`）。

放射計算は note_034「モデルに入れるもの」の式のとおり:

    LED の放射強度      I(θ) = I0 · cos^m(θ)         m は半値角から m = ln(0.5)/ln(cos θ_half)
    面素への放射照度    E = I(θ_e) / r_e²
    拡散で出る放射輝度  L = E · cos(θ_i) · ρ/π         θ_i は面での入射角（LED 側）
    鏡面成分            正反射方向からのずれに cos^s を掛けた項を足す（specular で重み付け）
    PT が受ける         L · cos(θ_v) · S(θ_r) / r_v²    θ_v は面での出射角（PT 側）・
                                                          θ_r は PT 光軸からのずれ

壁・柱は `classic/geometry.py::wall_obstacles` が返す軸平行の `Rect`（top-view の足跡）を
そのまま渡せる（本モジュールは内部でこれを壁の高さぶんの側面 4 枚の矩形パッチに展開する）。
床（z=0 の水平面）も面として常に含める（縦配置では PT が床に近く、床からの反射が効くため。
note_034 参照）。`classic/` からは真の壁を読まない、という規約は本モジュールには適用されない
（`classic.geometry.Rect` 型を読み取りに import するだけで、真の壁データそのものは
呼び出し側が渡す）。

## 数値積分について

LED・PT とも半値角が数度と非常に狭い（cos^m の m は LED の 3° で ~505 という鋭さ）。面全体を
一様格子で積分すると、格子が粗い距離でピーク位置の細い明部を踏み外して過小評価する。
そこで面ごとに「LED の光軸が面と交わる点（無ければ LED に最も近い面上の点）」を中心に、
tan ワープした非一様格子（`_warped_axis`）で標本点を集中させる。集中幅はその点までの距離と
半値角から見積もった円錐の広がりと、離隔 `separation_m` の大きい方を使う
（近距離では LED と PT の円錐が面上でずれてまったく重ならないことがあり、これが
「近づきすぎると値が下がる」山なりの応答の本体になる。両者の明部を格子が両方とも
捉えられるよう、窓の幅は離隔ぶんの余裕も持たせてある）。

## 遮蔽（オクルージョン）について

面素ごとの寄与を足す前に、「LED からその点まで」と「その点から PT まで」の 2 本の線分が
他の壁・柱で遮られていないかを判定する（`response(..., occlusion=True)` が既定。
`occlusion=False` で遮蔽なしの旧挙動と厳密に一致する＝否定対照）。迷路の壁・柱は軸平行の
直方体（`Rect` に高さ `wall_height_m` を与えたもの）なので、線分と直方体の交差は
slab 法（3軸それぞれで交差する媒介変数区間を出し積を取る）で厳密に解ける。詳しい手順は
`_segment_occluded()` のコメントを見ること。

自分自身（いま寄与を計算している面が属する直方体）との交差は、始点をずらす近似ではなく
**その直方体を番号で除外する**方法で正確に取り除く（面上の点は定義よりその直方体の表面上に
あるので、除外しないとほぼ確実に自己遮蔽の誤検出になる。番号除外なら「ずらす量」の
調整が要らない）。隣接する別の直方体との境界でのかすめ当たりは、交差区間を
`[eps_t, 1-eps_t]` に収めることで弾く。

床は「壁の直方体を遮る側」には回らない（壁の面素の z 座標は常に `wall_height_m/2 > 0` で、
LED・PT の取付高さも常に 0 より大きいので、両端の z が正である線分は z=0 を横切れない
＝床は壁向けの光線を遮り得ない。この非対称性は意図的で、床を遮蔽候補の直方体に
含めていない理由でもある）。逆に壁は床パッチの一部を遮る（縦配置では PT が床に近く
床の寄与が効くため、これは無視できない）。

## 既知の限界

- 床パッチは無限平面ではなく `floor_halfextent_m` の有限矩形（センサ周辺のみ）。
  遠方の床からの寄与は無視できるという前提（近似の妥当性はテストの数値で確認する）。
- PT の受光面積・LED の絶対光度は較正していない（`response()` の出力は任意単位。
  個体差は `IrSensorSpec.gain` でまとめて表す）。

## 出力の形（note_034 の設計方針）

強度で使うか距離に直すかは、本モジュールでは決めない。`response()` は AD 変換器が見るであろう
生の値（任意単位）を返すだけで、分解能・飽和・雑音を加えた整数値にするのは別関数 `adc()`。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

# classic/geometry.py の Rect（軸平行の長方形。壁・柱の足跡）を読み取りに import するだけ
# （note_034 の指示どおり。真の壁データそのものは呼び出し側から渡してもらう）。
from classic.geometry import Rect

__all__ = [
    "IrSensorSpec",
    "SurfaceSpec",
    "response",
    "adc",
    "ResponseTable",
    "build_table_from_model",
    "save_table",
    "load_table",
    "lookup",
    "DEFAULT_WALL_HEIGHT_M",
    "DEFAULT_FLOOR_HALFEXTENT_M",
    "DEFAULT_MAX_RANGE_M",
    "DEFAULT_N_GRID",
    # 相互反射（AUDIT_056・モデル I）用の粗い面素分割の既定値
    "DEFAULT_N_COARSE_WALL_U",
    "DEFAULT_N_COARSE_WALL_V",
    "DEFAULT_N_COARSE_FLOOR",
    "DEFAULT_N_GRID_INTERREFLECTION_SOURCE",
    "DEFAULT_N_HOTSPOT_SCAN",
    # 高速フォワードモデル（光線による可視面判定＋表引き。note_034 追記分）
    "build_wall_table",
    "build_post_table",
    "floor_baseline",
    "fast_response",
    "DEFAULT_WALL_HALF_LENGTH_M",
    "DEFAULT_FLOOR_SAMPLE_RADIUS_M",
    "DEFAULT_FLOOR_SAMPLE_RING",
    "DEFAULT_N_OCCLUSION_SAMPLES",
    "DEFAULT_POST_HALF_EXTENT_THRESHOLD_M",
    "DEFAULT_POST_HALF_LENGTH_M",
    "fast_response_or_direct",
    "DEFAULT_ADJACENCY_GAP_M",
    "DEFAULT_ADJACENCY_SIGNIFICANCE_FRAC",
    "DEFAULT_ADJACENCY_DOMINANT_MAX_D_M",
    # 高速版直接光モデル（AUDIT_059・段A: 反射1回・床なし）
    "response_fast",
    "DEFAULT_N_GRID_FAST",
    "DEFAULT_LED_CONE_MARGIN_DEG",
    "DEFAULT_PT_CONE_MARGIN_DEG",
    "DEFAULT_CONE_FILTER_N_U",
    "DEFAULT_CONE_FILTER_N_V",
    "DEFAULT_OCCLUSION_POINT_WEIGHT_FRAC",
    # 隅の相互反射（AUDIT_059・段B: 直角に交わる2壁の2〜4回反射）
    "DEFAULT_CORNER_ADJACENCY_TOL_M",
    "DEFAULT_CORNER_GATE_N_GRID",
    "DEFAULT_CORNER_N_GRID",
    "DEFAULT_CORNER_GATE_THRESHOLD",
]


PoseLike = Union[Tuple[float, float, float], object]

DEFAULT_WALL_HEIGHT_M: float = 0.05    # 壁の高さ [m]（迷路規格。note_034: 壁上端 50mm）
DEFAULT_FLOOR_HALFEXTENT_M: float = 0.20   # 床パッチの半幅 [m]（センサ周辺のみを面として持つ）
DEFAULT_MAX_RANGE_M: float = 0.35      # これより遠い壁は最初から積分対象に入れない
DEFAULT_N_GRID: int = 28               # 面 1 枚あたりの求積点数（1 軸あたり）

# 相互反射（AUDIT_056・モデル I）: 反射2回目以降を解くラジオシティ的な粗い面素の分割数。
# 間接照明はなめらかなので、1回目の細かい求積格子（DEFAULT_N_GRID）とは別に粗い格子を使う
# （面素数の2乗で効く計算量を抑えるため。`verification/AUDIT_056_PREREG_interreflection.md` §1）。
DEFAULT_N_COARSE_WALL_U: int = 3       # 壁側面の粗い面素（長さ方向の分割数）
DEFAULT_N_COARSE_WALL_V: int = 2       # 壁側面の粗い面素（高さ方向の分割数）
DEFAULT_N_COARSE_FLOOR: int = 4        # 床の粗い面素（1 辺あたりの分割数）
# 相互反射の起点 L^(1) と PT 集光係数を面素へ畳み込むときに使う warped 格子の求積点数
# （1回目の経路 DEFAULT_N_GRID=28 と同じ仕組み。narrow beam を粗い面素の中心1点で
# 評価すると踏み外すため、1回目と同じ warped 格子で先に積分してから面素へ畳み込む。
# `verification/AUDIT_056_PREREG_interreflection.md` §1・本モジュール docstring 参照）。
DEFAULT_N_GRID_INTERREFLECTION_SOURCE: int = 16

# 高速フォワードモデル（下の「表（距離×入射角×横ずれ）」節のさらに下、
# 「高速フォワードモデル」節を参照）で使う既定値。
DEFAULT_WALL_HALF_LENGTH_M: float = 0.084   # 実際の迷路の壁半長 [m]（cell_size/2 - post_size/2 = 0.09-0.006）
# 床パッチの可視判定に使う代表点（中心1点＋リング。note_034 追記分「解析的な列挙に
# 作り直した」節を参照。床は壁・柱と違って「面」の単位が無いので、点で代表させる）。
DEFAULT_FLOOR_SAMPLE_RADIUS_M: float = 0.03
DEFAULT_FLOOR_SAMPLE_RING: int = 8
DEFAULT_N_OCCLUSION_SAMPLES: int = 5    # 壁・柱の辺ごとの遮蔽判定の標本点数
# 同一平面上で連続する矩形（壁・柱を区別しない）をまとめる隙間の閾値 [m]。
# 教授セッションの実測（2026-08-21）: 連続している場合の隙間は厳密に0.000mm、
# 本物の開口部（区画1つぶん欠けている）は168mm級。どちらでもない中間の隙間は
# 迷路の規格上存在しないので、閾値はこの2つの間ならどこでもよい。
_COPLANAR_GAP_EPS_M: float = 0.005
DEFAULT_POST_HALF_EXTENT_THRESHOLD_M: float = 0.01  # これ未満の半長×半長の矩形は柱とみなす
DEFAULT_POST_HALF_LENGTH_M: float = 0.006           # 柱の半幅 [m]（post_size/2）。柱表の反射面の半長


# ============================================================================
# 仕様の dataclass
# ============================================================================
@dataclass
class IrSensorSpec:
    """1 個のセンサ（LED ＋ PT の対）の仕様。"""

    name: str
    pos: Tuple[float, float, float]    # 機体座標での取り付け位置（LED と PT の中点）[m]
    axis: Tuple[float, float, float]   # 光軸（機体座標の単位ベクトル。正規化前でもよい）
    separation_m: float = 0.0060       # LED と PT の離隔 [m]（実測への当てはめ。note_034 追記14）
    layout: str = "vertical"           # 既定は縦配置（ユーザ判断）。"horizontal" も選べる
    led_half_angle_deg: float = 3.0    # OSRAM SFH 4550 のデータシート（note_034 追記9）
    pt_half_angle_deg: float = 6.0     # KODENSHI ST-1KL3A のデータシート（同上）
    led_tilt_deg: float = 0.0          # アライメント誤差（±1° を想定）
    pt_tilt_deg: float = 0.0
    gain: float = 1.0                  # 個体差（応答全体に掛かる係数）


@dataclass
class SurfaceSpec:
    """反射面の性質。"""

    diffuse: float = 0.8       # 拡散反射率（白い射出成形プラスチック）
    specular: float = 0.10     # 鏡面成分（角度の実測への当てはめ。note_034 追記7・追記14）
    shininess: float = 40.0    # 鏡面のとがり（Phong 的な冪指数。同上）


# ============================================================================
# 内部ユーティリティ（座標変換・指向性）
# ============================================================================
def _pose_xytheta(pose: PoseLike) -> Tuple[float, float, float]:
    """`classic.geometry.Pose` でも `(x, y, theta)` のタプルでも受け付ける。"""
    if hasattr(pose, "x") and hasattr(pose, "y") and hasattr(pose, "theta"):
        return float(pose.x), float(pose.y), float(pose.theta)
    x, y, theta = pose
    return float(x), float(y), float(theta)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("ゼロベクトルは正規化できません")
    return v / n


def _rotate_z(v: np.ndarray, theta: float) -> np.ndarray:
    """世界 z 軸まわりに角度 theta だけ回す（機体は水平面内のみで姿勢を持つ前提）。"""
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = v
    return np.array([x * c - y * s, x * s + y * c, z])


def _half_angle_to_m(half_angle_deg: float) -> float:
    """半値角 [deg] から cos^m モデルの指数 m を出す（m = ln(0.5)/ln(cos θ_half))。"""
    cos_half = math.cos(math.radians(half_angle_deg))
    if not (0.0 < cos_half < 1.0):
        raise ValueError(f"led/pt_half_angle_deg が不正です: {half_angle_deg}")
    return math.log(0.5) / math.log(cos_half)


def _tilt_axis(axis: np.ndarray, sep_dir: np.ndarray, tilt_deg: float) -> np.ndarray:
    """アライメント誤差を、光軸と離隔の向きが張る面内で axis に加える。

    離隔の向き（縦配置なら上下、横配置なら左右）と同じ面内で傾けることで、
    「アライメント誤差と離隔の向きの組み合わせ」で差が出るかを試せるようにする
    （note_034「差が出るとしたら次の条件」の 2）。
    """
    if abs(tilt_deg) < 1e-12:
        return axis
    perp = sep_dir - np.dot(sep_dir, axis) * axis
    norm_perp = float(np.linalg.norm(perp))
    if norm_perp < 1e-9:
        return axis  # 離隔方向と光軸がほぼ平行（縮退）。傾けようがない。
    perp = perp / norm_perp
    t = math.radians(tilt_deg)
    return _normalize(axis * math.cos(t) + perp * math.sin(t))


@dataclass(frozen=True)
class _Emitter:
    """LED か PT、片方ぶんのワールド座標系での位置・光軸・半値角の指数。"""

    pos: np.ndarray    # world [m] shape (3,)
    axis: np.ndarray   # world 単位ベクトル shape (3,)
    m: float           # cos^m の指数
    half_angle_deg: float


def _sensor_world_geometry(sensor: IrSensorSpec, pose: PoseLike) -> Tuple[_Emitter, _Emitter]:
    """センサ仕様と機体姿勢から、LED・PT それぞれのワールド座標での位置・光軸を作る。"""
    x, y, theta = _pose_xytheta(pose)
    center_body = np.array(sensor.pos, dtype=float)
    axis_body = _normalize(np.array(sensor.axis, dtype=float))

    if sensor.layout == "vertical":
        # 縦配置: LED が上・PT が下（note_034: 取付高さ10mm・離隔6.5mmで LED13.25mm/PT6.75mm）。
        sep_dir_body = np.array([0.0, 0.0, 1.0])
    elif sensor.layout == "horizontal":
        # 横配置: 光軸の水平面内成分に直交する向き（左右）に離隔を取る。
        horiz = np.array([axis_body[0], axis_body[1], 0.0])
        norm_h = float(np.linalg.norm(horiz))
        if norm_h < 1e-9:
            sep_dir_body = np.array([0.0, 1.0, 0.0])
        else:
            horiz = horiz / norm_h
            sep_dir_body = np.array([-horiz[1], horiz[0], 0.0])
    else:
        raise ValueError(f"layout が不正です: {sensor.layout!r}（'vertical' か 'horizontal'）")

    half = sensor.separation_m / 2.0
    led_pos_body = center_body + sep_dir_body * half
    pt_pos_body = center_body - sep_dir_body * half

    led_axis_body = _tilt_axis(axis_body, sep_dir_body, sensor.led_tilt_deg)
    pt_axis_body = _tilt_axis(axis_body, sep_dir_body, sensor.pt_tilt_deg)

    def to_world_point(p_body: np.ndarray) -> np.ndarray:
        return _rotate_z(p_body, theta) + np.array([x, y, 0.0])

    def to_world_dir(d_body: np.ndarray) -> np.ndarray:
        return _rotate_z(d_body, theta)

    led = _Emitter(
        pos=to_world_point(led_pos_body), axis=to_world_dir(led_axis_body),
        m=_half_angle_to_m(sensor.led_half_angle_deg), half_angle_deg=sensor.led_half_angle_deg,
    )
    pt = _Emitter(
        pos=to_world_point(pt_pos_body), axis=to_world_dir(pt_axis_body),
        m=_half_angle_to_m(sensor.pt_half_angle_deg), half_angle_deg=sensor.pt_half_angle_deg,
    )
    return led, pt


# ============================================================================
# 面（積分対象のパッチ）
# ============================================================================
@dataclass(frozen=True)
class _Facet:
    """積分の対象になる 1 枚の平面矩形パッチ。"""

    center: np.ndarray     # (3,)
    u: np.ndarray           # 面内の単位ベクトル（1 軸目）
    v: np.ndarray           # 面内の単位ベクトル（2 軸目）
    normal: np.ndarray      # 外向き法線
    half_u: float
    half_v: float


def _wall_facets(rects: Sequence[Rect], wall_height_m: float) -> list:
    """壁・柱の top-view 足跡（`Rect`）を、高さぶんの側面 4 枚の矩形パッチへ展開する。"""
    facets = []
    z_center = wall_height_m / 2.0
    u_y = np.array([0.0, 1.0, 0.0])
    u_x = np.array([1.0, 0.0, 0.0])
    v_z = np.array([0.0, 0.0, 1.0])
    for r in rects:
        facets.append(_Facet(
            center=np.array([r.cx + r.hx, r.cy, z_center]), u=u_y, v=v_z,
            normal=np.array([1.0, 0.0, 0.0]), half_u=r.hy, half_v=wall_height_m / 2.0,
        ))
        facets.append(_Facet(
            center=np.array([r.cx - r.hx, r.cy, z_center]), u=u_y, v=v_z,
            normal=np.array([-1.0, 0.0, 0.0]), half_u=r.hy, half_v=wall_height_m / 2.0,
        ))
        facets.append(_Facet(
            center=np.array([r.cx, r.cy + r.hy, z_center]), u=u_x, v=v_z,
            normal=np.array([0.0, 1.0, 0.0]), half_u=r.hx, half_v=wall_height_m / 2.0,
        ))
        facets.append(_Facet(
            center=np.array([r.cx, r.cy - r.hy, z_center]), u=u_x, v=v_z,
            normal=np.array([0.0, -1.0, 0.0]), half_u=r.hx, half_v=wall_height_m / 2.0,
        ))
    return facets


def _floor_facet(center_xy: np.ndarray, halfextent_m: float) -> _Facet:
    return _Facet(
        center=np.array([center_xy[0], center_xy[1], 0.0]),
        u=np.array([1.0, 0.0, 0.0]), v=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]), half_u=halfextent_m, half_v=halfextent_m,
    )


# ============================================================================
# 数値積分（tan ワープの非一様格子）
# ============================================================================
def _warped_axis(H: float, anchor: float, w: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """`[-H, H]` を `anchor` 付近に集中させた `n` 点の求積点と重みを作る（tan ワープ）。

    `w` は集中させる幅の目安。`w` が `H` に対して十分小さければ `anchor` 付近の分解能が
    上がり、`w` が `H` 程度以上なら実質一様格子に近づく。中点則で `∫f ds ≈ Σ f(s_i)*weight_i`
    になるよう、重みは `|ds/dt| * Δt`（解析的なヤコビアン）で計算する。
    """
    anchor = min(max(anchor, -H), H)
    w = max(w, H * 1e-3, 1e-6)
    phi_plus = math.atan2(H - anchor, w)
    phi_minus = math.atan2(anchor + H, w)
    t = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    phi = np.where(t >= 0.0, phi_plus, phi_minus)
    s = anchor + w * np.tan(t * phi)
    dt = 2.0 / n
    ds_dt = w * phi / np.cos(t * phi) ** 2
    weight = np.abs(ds_dt) * dt
    return s, weight


def _facet_anchor_and_scale(
    facet: _Facet, led: _Emitter, half_angle_max_deg: float, separation_m: float
) -> Tuple[float, float, float]:
    """LED の光軸が面と交わる点（無ければ LED に最も近い面上の点）を中心に、
    円錐の広がりと離隔から積分窓の集中幅を見積もる。"""
    denom = float(np.dot(led.axis, facet.normal))
    t = float(np.dot(facet.center - led.pos, facet.normal)) / denom if abs(denom) > 1e-9 else -1.0
    if t > 0.0:
        p_hit = led.pos + t * led.axis
        r_est = t
    else:
        rel = led.pos - facet.center
        pu = float(np.clip(np.dot(rel, facet.u), -facet.half_u, facet.half_u))
        pv = float(np.clip(np.dot(rel, facet.v), -facet.half_v, facet.half_v))
        p_hit = facet.center + pu * facet.u + pv * facet.v
        r_est = max(float(np.linalg.norm(led.pos - p_hit)), 1e-4)

    rel = p_hit - facet.center
    anchor_u = float(np.dot(rel, facet.u))
    anchor_v = float(np.dot(rel, facet.v))
    spread = r_est * math.tan(math.radians(4.0 * half_angle_max_deg))
    w = max(spread, separation_m * 2.0, 1e-4)
    return anchor_u, anchor_v, w


# ============================================================================
# 遮蔽（オクルージョン）: 軸平行直方体との線分交差（slab 法）
# ============================================================================
def _obstacle_boxes(rects: Sequence[Rect], wall_height_m: float) -> np.ndarray:
    """壁・柱の top-view 足跡 `Rect` を、遮蔽判定用の 3D 直方体（xmin,xmax,ymin,ymax,zmin,zmax）
    の配列 shape (N, 6) にする。`rects` と同じ並び順（`facet` 側の owner インデックスと対応させる）。
    """
    if len(rects) == 0:
        return np.zeros((0, 6), dtype=float)
    return np.array(
        [[r.cx - r.hx, r.cx + r.hx, r.cy - r.hy, r.cy + r.hy, 0.0, wall_height_m] for r in rects],
        dtype=float,
    )


def _segment_occluded(
    A: np.ndarray, B: np.ndarray, boxes: np.ndarray,
    skip_idx: Union[None, int, np.ndarray],
    eps_t: float = 1e-6,
) -> np.ndarray:
    """線分 A→B が `boxes`（行 = xmin,xmax,ymin,ymax,zmin,zmax）のいずれかに遮られているかを
    slab 法で判定する。迷路の壁・柱はすべて軸平行の直方体なので、この判定は近似ではなく厳密解。

    `A`・`B` は shape `(..., 3)` で、互いにブロードキャスト可能なら形が違ってもよい
    （例: 片方が LED の 1 点、もう片方が面素の求積点グリッド全体。あるいは複数の面を
    先頭軸にまとめて一括判定することもできる。下記 `skip_idx` 参照）。戻り値は
    ブロードキャスト後の形の bool 配列（True = 遮られている）。

    やり方（slab 法）: 線分を `P(t) = A + t*(B-A)`, `t ∈ [0,1]` とパラメータ化し、
    各軸ごとに「直方体の範囲に入っている t の区間」を求めて 3 軸分の積（共通部分）を取る。
    共通部分が空でなければ交差している。軸に平行な光線（分母 ≈ 0）は 0 除算になるので、
    「始点がその軸の範囲内にあるか」だけで区間を作り直す（範囲内なら常に交差 = 区間は
    `(-inf, inf)`、範囲外なら絶対に交差しない = 区間は空）。

    自分自身との交差の除外: `skip_idx` に「いま寄与を計算している面が属する直方体」の
    番号を渡すと、その箱だけ判定から外す。面上の求積点は定義よりその箱の表面上にあるので、
    除外しないと「自分の壁に遮られた」という誤検出になる。始点をわずかにずらす近似ではなく
    箱を番号で正確に除外する理由: ずらす量の大小に判定結果が左右されない（ずらしすぎると
    本当の遮蔽を見逃し、少なすぎると数値誤差で自己交差を拾う、という調整が要らない）。
    一方、別の（除外対象ではない）箱との境界でのかすめ当たり（例: 壁と柱が角で接する）は
    `eps_t` で弾く。交差区間を `[eps_t, 1-eps_t]` にクランプしてから空かどうかを見ることで、
    「線分の両端ぎりぎり」での測度ゼロの接触を遮蔽とみなさないようにしている。

    `skip_idx` は `None`（除外なし）・`int`（`A` の先頭軸すべてに同じ箱番号を適用）・
    `A` の先頭軸（面の枚数ぶん）と同じ長さの整数配列（面ごとに別の箱番号。負の値＝
    その面は除外なし＝床のような「属する箱が無い」面用）のいずれかを受け付ける。
    後者は「面ごとに 1 回ずつ `_segment_occluded` を呼ぶ」代わりに「複数の面をまとめて
    1 回で判定する」ための足場（速さのための numpy 一括化。`response()` 側で
    可視な面すべての求積点を 1 本の配列に積んでから 1 回だけ呼ぶのに使う）。
    """
    out_shape = np.broadcast_shapes(A.shape[:-1], B.shape[:-1])
    if boxes.shape[0] == 0:
        return np.zeros(out_shape, dtype=bool)

    d = B - A                      # (...,3)  ブロードキャストされる
    A3 = A[..., None, :]           # (...,1,3)
    d3 = d[..., None, :]           # (...,1,3)
    mins = boxes[:, (0, 2, 4)]     # (Nbox,3)
    maxs = boxes[:, (1, 3, 5)]     # (Nbox,3)

    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (mins - A3) / d3      # (...,Nbox,3)
        t2 = (maxs - A3) / d3
    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)

    parallel = np.abs(d3) < 1e-15
    inside = (A3 >= mins) & (A3 <= maxs)
    t_near = np.where(parallel, np.where(inside, -np.inf, np.inf), t_near)
    t_far = np.where(parallel, np.where(inside, np.inf, -np.inf), t_far)

    tmin = np.max(t_near, axis=-1)     # (...,Nbox)  3 軸の共通部分の下端
    tmax = np.min(t_far, axis=-1)      # (...,Nbox)  3 軸の共通部分の上端

    t_enter = np.maximum(tmin, eps_t)
    t_exit = np.minimum(tmax, 1.0 - eps_t)
    hit = t_enter < t_exit             # (...,Nbox)

    if isinstance(skip_idx, np.ndarray):
        # 面ごとに別の箱番号を除外する（先頭軸 = 面の軸という前提。`response()` 側で
        # そう積んでいる）。負の値の面（床など、属する箱が無い）は何も除外しない。
        n_facets = skip_idx.shape[0]
        skip_mask = np.zeros((n_facets, boxes.shape[0]), dtype=bool)
        valid = skip_idx >= 0
        skip_mask[np.nonzero(valid)[0], skip_idx[valid]] = True
        # hit の形は (n_facets, グリッドの軸..., Nbox)。skip_mask ((n_facets, Nbox)) を
        # 中間のグリッド軸ぶんだけ 1 埋めして reshape し、そのままブロードキャストで引く。
        broadcast_shape = (n_facets,) + (1,) * (hit.ndim - 2) + (boxes.shape[0],)
        hit = hit & ~skip_mask.reshape(broadcast_shape)
    elif skip_idx is not None:
        hit = hit.copy()
        hit[..., skip_idx] = False

    return np.any(hit, axis=-1)


def _facet_grid(
    facet: _Facet, led: _Emitter, n_grid: int, half_angle_max_deg: float, separation_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """面素の求積点（ワールド座標、shape (n_grid, n_grid, 3)）と面積要素 `dA` を作る
    （`_warped_axis` による tan ワープ格子。詳しくはモジュール docstring「数値積分について」）。
    """
    anchor_u, anchor_v, w = _facet_anchor_and_scale(facet, led, half_angle_max_deg, separation_m)
    su, wu = _warped_axis(facet.half_u, anchor_u, w, n_grid)
    sv, wv = _warped_axis(facet.half_v, anchor_v, w, n_grid)

    SU, SV = np.meshgrid(su, sv, indexing="ij")
    WU, WV = np.meshgrid(wu, wv, indexing="ij")
    dA = WU * WV

    points = (
        facet.center[None, None, :]
        + SU[:, :, None] * facet.u[None, None, :]
        + SV[:, :, None] * facet.v[None, None, :]
    )
    return points, dA


def _integrate_facet(
    facet: _Facet, led: _Emitter, pt: _Emitter, surf: SurfaceSpec,
    points: np.ndarray, dA: np.ndarray,
    led_intensity: float, pt_responsivity: float, max_range_m: float,
    occluded: Optional[np.ndarray] = None,
) -> float:
    """`_facet_grid()` で作った求積点 `points`・`dA` を使って、面素ごとの寄与を積分する。

    `occluded`（`None` か、`points.shape[:-1]` と同じ形の bool 配列。True の点は寄与ゼロ）は
    呼び出し側（`response()`）が渡す。複数の面をまとめて `_segment_occluded` に 1 回だけ
    通した結果のうち、この面に対応する部分を切り出して渡す形にしてある
    （速さのための一括化。詳しくは `response()` 側のコメント参照）。
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
    irradiance = led_intensity * cos_e ** led.m / (r_e_safe ** 2)

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

    contribution = radiance * cos_v * pt_responsivity * pt_sensitivity / (r_v_safe ** 2)
    valid = (r_e < max_range_m) & (r_e > 1e-9) & (r_v > 1e-9)

    if occluded is not None:
        valid = valid & ~occluded

    contribution = np.where(valid, contribution, 0.0)

    return float(np.sum(contribution * dA))


# ============================================================================
# 相互反射（AUDIT_056・モデル I）: 反射2回目以降をラジオシティ的な粗い面素で解く
# ============================================================================
"""
🔴 **この経路（`bounces >= 2`）は検証を通っていない。評価・測定に使ってはならない。**
2026-08-21 の実測で、壁1枚＋床の配置において**反射3回目の増分が独立な光線追跡の
0.1〜4%（最大 500 分の 1）しか出ない**という深刻な過小評価が見つかった。面素分割を上げても
値は単調に収束せず（壁9×6・床12×12 で 1 姿勢 79 秒＝速度要件の 40 倍）、原因は
「センサ近傍の幾何（PT は床から 10mm）に対して粗い面素（床 100mm 角）が大きすぎ、
点対点の形態係数近似が破綻すること」と見ている。**距離で段階的に細かくする面素分割
（適応分割）が要る。**それまでこの経路は未完成である（`AUDIT_056` 追記2）。

**`bounces=1`（既定）は元の計算式をそのまま通る**（実測で相対差 0.0 の厳密一致を確認済み）。
既存の呼び出し側の挙動は一切変わっていない。

背景・仕様は `verification/AUDIT_056_PREREG_interreflection.md` §1 を正とする。

## 骨子

1回目の経路（fine-grid の面積分。`_integrate_facet` 等、上のコードは一切変更しない）に、
反射2回目以降の寄与を**加算する項**として足す。2回目以降の輸送は拡散のみ（鏡面は1回目にだけ
効く。事前登録の根拠: 鏡面のエネルギー比は約10%なので、2回目以降に鏡面を入れても効くのは
「数%の項のさらに10%」＝満量比で10⁻³台であり、判定量の分割点0.01の1/10以下）。

各面（壁の側面4枚×矩形＋床）を粗い面素に割り、面素ごとに中心・法線・面積を持たせる:

    E_j          = LED から面素 j への直接放射照度（cosθ_i 込み・遮蔽込み）
    L_j^(1)      = ρ・E_j/π                                    （反射1回目の放射輝度）
    L_i^(k+1)    = ρ・Σ_j [cosθ_i・cosθ_j/(π r_ij²)]・V_ij・A_j・L_j^(k)
                                                                （反射を1回進める）
    増分_k       = Σ_i L_i^(k)・（PT での集光係数。1回目と同じ規格）

**π の置き場所は 1 回目の経路（`_integrate_facet`）と揃えてある**（放射照度→放射輝度の
`ρ/π`、PT 集光の式に π を追加で挟まない）。事前登録の文中の式 `F_ij = cosθ_i cosθ_j/(π r²)`
を文字どおり「B_i^(k+1) = ρ/π・Σ F_ij…」と重ねると π が二重になる（標準的な放射伝達の
導出と食い違う）。本実装は放射輝度の物理定義から導出し直した上記の式を採用し、
その正しさは検証0-c（独立な光線追跡との突き合わせ）で確認する。

## 🔴 粗い面素の「中心1点で評価」では narrow beam を踏み外す（実装中に発覚）

LED・PT の半値角は数度と鋭く（`m` は LED の 3° で ~505）、粗い面素（例: 壁1枚を3×2に分割）の
中心点だけで `E_j`（LED側）や集光係数（PT側）を評価すると、ビームの明部が面素の中心から
外れた途端に値がほぼゼロになり、逆にたまたま中心が明部を捉えた面素だけが桁違いに大きい値を
持つ（実測で確認: 壁を3×2＝6面素に割ったところ、1枚だけが `E=605`・残り5枚が `1e-40` 以下）。
モジュール冒頭の「数値積分について」に書かれている、1回目の経路がまさに warped 格子で
避けている落とし穴そのものを、粗い面素で再現してしまう。

**対策**: 面素ごとに `E_j`（LED起点）と PT 集光係数の**どちらも**、narrow beam を正しく
捉える warped 格子で積分し、その結果を面素へビニングして畳み込む（`_facet_radiosity_cells`）。
**粗い面素はラジオシティの伝播（面素→面素の形態係数 `F_ij`。間接照明なのでなめらか）にだけ
使い**、LED・PT が絡む narrow beam の部分は fine grid の求積精度をそのまま引き継ぐ。

面素間の可視判定は法線方向にわずか（1e-6m）ずらした点どうしで行う（`_segment_occluded` の
番号除外は「面素→LED/PT」の対では使えるが、「面素→面素」では両端が別々の直方体の上に
あり得るため、既存の光線追跡 `verification/audit_050_raycast.py` と同じ「ずらす」方式を使う）。

## 🔴 1回目の経路と同じ `_facet_grid`（`_facet_anchor_and_scale`）は床では使えない（実装中に発覚）

1回目の経路の anchor（`_facet_anchor_and_scale`）は「LED 光軸が面と前方で交わる点」を
中心に据えるが、**LED 光軸が面とほぼ平行（前方交点が無い＝`t<=0`）の場合は「LED に
最も近い点」へフォールバックする**。これは壁（LED がほぼ正対する面）では妥当だが、
**床では成り立たない**: 実機センサの光軸は水平に近い（`mouse/params.py` の
`zaxis` は z 成分 0.026 のみ）ので、床に対しては「前方交点なし」がほぼ常に起こり、
「LED に最も近い点」（センサ直下）へ anchor してしまう。しかし cos^m(θ_e)/r_e² は
grazing 角度では **センサ直下ではなく、面内の有限距離だけ離れた点で最大になる**
（LED 高さ `h`・指数 `m` に対して解析的に `x_peak = h・√(m/2)`。本機体の実測値では
約126mm — センサ直下から10cm以上離れている）。「壁→床→壁→PT」（床が主経路。
`verification/audit_050_bounce_parity.py` docstring 参照）で床の `E_j` をこの anchor で
求めると、実際のホットスポットを外し、**壁+床での3回反射の増分を光線追跡比で
最大 500 分の1 まで過小評価する**ことが検証0-c を補強する自主検算で見つかった
（`_facet_anchor_and_scale`/`_facet_grid` は 1 回目の経路と共有しており、bounces=1 の
厳密一致要件のため変更できない。本モジュール専用の別実装で対処する）。

**対策**: `_facet_grid` の代わりに、面素ごとに `_facet_led_hotspot_anchor()` で
`cos^m(θ_e)/r_e²` を面内の密な一様格子（既定 64×64。定式だけの評価で遮蔽計算を伴わず
軽い）で直接探索し、その最大点を anchor にした warped 格子（`_warped_axis`）を
自前で作る（`_facet_grid_for_radiosity()`）。前方交点があるケースでも「実際の最大点」を
直接探すほうが `_facet_anchor_and_scale` の解析的な見積もりより頑健なので、
場合分けせず常にこちらを使う。

## 🔴 既知の限界（未解決）: 近距離の壁+床では粗い面素の分割数が根本的に足りない

上のホットスポット anchor 修正のあと、検証0-c（単一パネル・床なし）は光線追跡と
厳密に一致するようになったが、**これは単一パネル・床なしでは両モデルとも増分が
厳密にゼロになるため（面の偶奇。`verification/audit_050_bounce_parity.py` 参照）、
π の規格や式が正しいかを実質検査できていない**。そこで検証0-c の対象外である
「壁1枚＋床」（`audit_050_bounce_parity.py` と同じ配置）で独自に光線追跡と突き合わせたところ、
既定の粗い面素分割（壁 3×2・床 4×4）では **3 回反射の増分が光線追跡比で
0.1%〜4%（距離 20〜150mm）しか出ない**ことが分かった。

原因を段階的に調べた結果:

1. 面素間の可視判定・遮蔽つきホットスポット探索・r 下限クランプ・L^(1) のフラックス
   加重重心（送り手位置）は、いずれも実測で効果があった（特に r 下限クランプは、
   隣接する柱・壁の面素どうしが偶然近づいたときの発散を抑え、検証0-b の最悪ケースを
   満量比 0.13 から 0.002 付近まで改善した）
2. **PT の集光カーネルの受け手位置に加重重心を使う対策も試したが、効果はほぼ無かった**
   （40mm で 0.02879→0.02881、誤差はほぼ変化なし）。計算時間はほぼ倍増し、検証0-b の
   余裕も失われたため不採用にした（実装は `_facet_radiosity_cells` の `collect_centroid`
   に残っているが、`_interreflection_increments` では使っていない）
3. 面素の分割数を大きく上げても（壁 9×6・床 12×12）増分は単調に真値へ収束せず、
   1姿勢の計算時間が 79 秒（既定の 2 秒要件の 40 倍）まで伸びた

**根本原因はセンサの視野角（半値角 3〜6°）に対して面素 1 枚が近距離では大きすぎること**
だと判断している。PT がおよそ 20〜150mm という近さで壁を見るとき、面素（既定で壁の
長さ方向 56mm・高さ方向 25mm）が張る角度はセンサの半値角よりずっと広く、面素を
1点（幾何中心や重心）で代表させる近似がそもそも成り立たない。解決には面素分割を
均一格子ではなく、fine grid と同じ warped（anchor 集中）格子で作る、あるいは
可視角度に応じた適応的な細分割が要ると考えられるが、**本作業の時間内では実装・検証
できなかった**。

**この結果、検証0-b は姿勢によって合否が変わる状態にある**（後述の報告参照）。
判断（面素分割の作り方を変える・速度要件を緩める・別の近似に切り替える等）は
教授セッションへ上げる。
"""


DEFAULT_N_HOTSPOT_SCAN: int = 64   # LED ホットスポット探索の一様格子（1軸あたり）


def _facet_led_hotspot_anchor(
    facet: _Facet, led: _Emitter, max_range_m: float,
    boxes: np.ndarray, owner_idx: int,
    n_scan: int = DEFAULT_N_HOTSPOT_SCAN,
) -> Tuple[float, float, float]:
    """面 `facet` の中で `cos^m(θ_e)/r_e²`（LED の放射照度に比例する量。cos_i は含めない
    近似だが、**遮蔽と `max_range_m` は含める**）が最大になる点を、密な一様格子で直接探す。
    戻り値は `(anchor_u, anchor_v, r_est)`（面内座標・その点までの距離）。
    モジュール docstring「1回目の経路と同じ `_facet_grid` は床では使えない」節を参照
    （LED 光軸が面とほぼ平行なとき、解析的な anchor 推定 `_facet_anchor_and_scale` は
    実際のホットスポットを大きく外すため、代わりに使う）。

    🔴 遮蔽を含めない版では、遮られた領域（例: 手前の壁の向こうの床）にある無遮蔽時の
    最大点へ anchor してしまい、実際に効く手前側の領域が warped 格子から外れて
    大幅な過小評価になることが実測で見つかった（壁+床の3回反射が光線追跡比で
    最大500分の1）。遮蔽込みで探索することで、この見落としを防ぐ。
    """
    us = np.linspace(-facet.half_u, facet.half_u, n_scan)
    vs = np.linspace(-facet.half_v, facet.half_v, n_scan)
    UU, VV = np.meshgrid(us, vs, indexing="ij")
    pts = (
        facet.center[None, None, :]
        + UU[:, :, None] * facet.u[None, None, :]
        + VV[:, :, None] * facet.v[None, None, :]
    )
    d_e = pts - led.pos
    r_e = np.linalg.norm(d_e, axis=-1)
    r_e_safe = np.maximum(r_e, 1e-6)
    dir_e = d_e / r_e_safe[..., None]
    cos_e = np.clip(np.einsum("ijk,k->ij", dir_e, led.axis), 0.0, 1.0)
    field = cos_e ** led.m / (r_e_safe ** 2)
    valid = r_e < max_range_m
    if boxes.shape[0] > 0:
        # 🔴 owner_idx=-1（床。属する箱が無い）を裸のスカラーのまま渡すと、
        # `_segment_occluded` は Python の負インデックス規約で「配列末尾の箱」を除外して
        # しまい、遮蔽が効かなくなる（`skip_idx` の -1=除外なし 規約は配列形のときだけ
        # 有効）。ここで None に変換して防ぐ。
        occ = _segment_occluded(pts, led.pos, boxes, owner_idx if owner_idx >= 0 else None)
        valid = valid & ~occ
    field = np.where(valid, field, 0.0)
    if not np.any(valid):
        # 面全体が遮蔽/範囲外なら、どこを anchor にしても寄与ゼロ。中心を返しておく。
        return 0.0, 0.0, max(float(np.min(r_e_safe)), 1e-4)
    idx = np.unravel_index(int(np.argmax(field)), field.shape)
    return float(UU[idx]), float(VV[idx]), float(r_e_safe[idx])


def _facet_grid_for_radiosity(
    facet: _Facet, led: _Emitter, n_grid: int, half_angle_max_deg: float, separation_m: float,
    max_range_m: float, boxes: np.ndarray, owner_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """相互反射の起点 `E_j`/`L^(1)` を積分するための warped 格子（`_facet_grid` と同じ
    tan ワープ・同じ戻り値の形だが、anchor は `_facet_led_hotspot_anchor()` の直接探索を使う。
    1回目の経路が使う `_facet_grid`/`_facet_anchor_and_scale` とは別実装（`bounces=1` の
    厳密一致要件のため、共有関数の側は変更できない。モジュール docstring 参照）。
    """
    anchor_u, anchor_v, r_est = _facet_led_hotspot_anchor(facet, led, max_range_m, boxes, owner_idx)
    spread = r_est * math.tan(math.radians(4.0 * half_angle_max_deg))
    w = max(spread, separation_m * 2.0, 1e-4)
    su, wu = _warped_axis(facet.half_u, anchor_u, w, n_grid)
    sv, wv = _warped_axis(facet.half_v, anchor_v, w, n_grid)

    SU, SV = np.meshgrid(su, sv, indexing="ij")
    WU, WV = np.meshgrid(wu, wv, indexing="ij")
    dA = WU * WV
    points = (
        facet.center[None, None, :]
        + SU[:, :, None] * facet.u[None, None, :]
        + SV[:, :, None] * facet.v[None, None, :]
    )
    return points, dA


def _facet_radiosity_cells(
    facet: _Facet, owner_idx: int, led: _Emitter, pt: _Emitter, surf: SurfaceSpec,
    led_intensity: float, pt_responsivity: float, max_range_m: float, boxes: np.ndarray,
    n_grid_source: int, half_angle_max_deg: float, separation_m: float,
    n_u: int, n_v: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1 枚の面 `facet` を `n_u × n_v` の粗い面素へ割り、面素ごとに次を返す:

    - `centers`・`normals`・`areas`: 伝播（形態係数 `F_ij`）に使う幾何量（面素の中心・面積）
    - `L1_cell`: LED の直接照明による反射1回目の放射輝度の面素内平均。narrow beam を
      1回目の経路と同じ warped 格子（`_facet_grid`）で正しく積分してから面素へ畳み込む
      （モジュール docstring「粗い面素の『中心1点で評価』では narrow beam を踏み外す」参照）
    - `collect_cell`: PT での集光係数（`cosθ_v・cos^m_pt(θ_r)/r_v²`）を同じ warped 格子で
      積分した値（dA 込み）。`L^(k)` が面素内で一定という区分定数近似のもとで、
      `Σ_cell L^(k)・collect_cell` として fine grid 抜きで集光できるようにする
    - `l1_centroid`: 面素内での **L^(1) フラックス加重重心**（フラックスが無い面素は
      幾何中心にフォールバック）。narrow beam のせいで L^(1) が面素内で極端に偏るとき
      （検証0-bで実測: 面素分割を1.5倍にしただけで増分が最大17分の1まで動いた）、
      形態係数 `F_ij` の送り手位置に幾何中心をそのまま使うと分割数に結果が強く依存して
      しまう。重心を送り手位置として使うことで、この依存を落とす
      （`_interreflection_increments` の F1 を参照）。
    - `collect_centroid`: 面素内での **PT 集光カーネル加重重心**（重みが無い面素は
      幾何中心にフォールバック）。PT も半値角が数度と狭いため、集光もまた面素内で
      極端に偏る（実測: 壁の面素6枚中1枚だけが集光の97%超を占める）。`L^(k)` を
      幾何中心で評価して集めると、壁+床の3回反射が光線追跡比で最大500分の1まで
      過小評価された（L^(k) 自体は間接照明でなめらかでも、それを PT へ集める重みが
      面素内でなめらかでないため、幾何中心の値がその面素の「PT が実際に見ている点」の
      値からずれる）。集光の受け手位置にこの重心を使うことで、この見落としを防ぐ
      （`_interreflection_increments` の F1_collect/F_collect を参照）。
    """
    n_cells = n_u * n_v
    du = 2.0 * facet.half_u / n_u
    dv = 2.0 * facet.half_v / n_v
    cell_area = du * dv

    centers = np.empty((n_cells, 3))
    for iu in range(n_u):
        cu = -facet.half_u + (iu + 0.5) * du
        for iv in range(n_v):
            cv = -facet.half_v + (iv + 0.5) * dv
            centers[iu * n_v + iv] = facet.center + cu * facet.u + cv * facet.v
    normals = np.tile(facet.normal, (n_cells, 1))
    areas = np.full(n_cells, cell_area)

    points, dA = _facet_grid_for_radiosity(
        facet, led, n_grid_source, half_angle_max_deg, separation_m, max_range_m, boxes, owner_idx,
    )

    # 🔴 owner_idx=-1（床）を裸のスカラーのまま `_segment_occluded` に渡すと、Python の
    # 負インデックス規約で「配列末尾の箱」を誤って除外してしまう（-1=除外なし の規約は
    # 配列形の skip_idx のときだけ有効）。None に変換してから渡す。
    skip_box = owner_idx if owner_idx >= 0 else None

    d_e = points - led.pos
    r_e = np.linalg.norm(d_e, axis=-1)
    r_e_safe = np.maximum(r_e, 1e-6)
    dir_e = d_e / r_e_safe[..., None]
    cos_e = np.clip(np.einsum("ijk,k->ij", dir_e, led.axis), 0.0, 1.0)
    cos_i = np.clip(np.einsum("ijk,k->ij", -dir_e, facet.normal), 0.0, 1.0)
    valid_e = (r_e < max_range_m) & (r_e > 1e-9)
    if boxes.shape[0] > 0:
        occ_e = _segment_occluded(points, led.pos, boxes, skip_box)
        valid_e = valid_e & ~occ_e
    irradiance = np.where(valid_e, led_intensity * cos_e ** led.m / (r_e_safe ** 2) * cos_i, 0.0)
    radiance_1 = irradiance * (surf.diffuse / math.pi)   # L^(1)（点ごと）

    d_v = pt.pos - points
    r_v = np.linalg.norm(d_v, axis=-1)
    r_v_safe = np.maximum(r_v, 1e-6)
    dir_v = d_v / r_v_safe[..., None]
    cos_v = np.clip(np.einsum("ijk,k->ij", dir_v, facet.normal), 0.0, 1.0)
    cos_r = np.clip(np.einsum("ijk,k->ij", -dir_v, pt.axis), 0.0, 1.0)
    valid_v = r_v > 1e-9
    if boxes.shape[0] > 0:
        occ_v = _segment_occluded(points, pt.pos, boxes, skip_box)
        valid_v = valid_v & ~occ_v
    collect_kernel = np.where(valid_v, cos_v * cos_r ** pt.m / (r_v_safe ** 2), 0.0) * pt_responsivity

    rel = points - facet.center
    uu = np.einsum("ijk,k->ij", rel, facet.u)
    vv = np.einsum("ijk,k->ij", rel, facet.v)
    ui = np.clip(((uu + facet.half_u) / (2.0 * facet.half_u) * n_u).astype(int), 0, n_u - 1)
    vi = np.clip(((vv + facet.half_v) / (2.0 * facet.half_v) * n_v).astype(int), 0, n_v - 1)
    cell_idx = (ui * n_v + vi).ravel()

    flux = (radiance_1 * dA).ravel()
    weight_collect = (collect_kernel * dA).ravel()
    L1_flux = np.bincount(cell_idx, weights=flux, minlength=n_cells)
    collect_cell = np.bincount(cell_idx, weights=weight_collect, minlength=n_cells)
    L1_cell = L1_flux / cell_area

    flat_points = points.reshape(-1, 3)

    def _weighted_centroid(weight_flat: np.ndarray, weight_sum: np.ndarray) -> np.ndarray:
        """`weight_flat`（点ごとの重み）で面素内の点を加重平均した重心を返す
        （重みが無い面素は幾何中心にフォールバック）。"""
        acc = np.zeros((n_cells, 3))
        for dim in range(3):
            acc[:, dim] = np.bincount(cell_idx, weights=weight_flat * flat_points[:, dim], minlength=n_cells)
        has_w = weight_sum > 0.0
        return np.where(has_w[:, None], acc / np.where(has_w, weight_sum, 1.0)[:, None], centers)

    # L^(1) のフラックス加重重心（送り手位置。上のコメント「narrow beam を踏み外す」参照）。
    l1_centroid = _weighted_centroid(flux, L1_flux)
    # PT 集光カーネルの加重重心（受け手位置）。PT も半値角が数度と狭いため、collect_cell の
    # 中身も面素内で極端に偏る（実測: 壁面素6枚中1枚だけが集光の97%超を占める）。この重心を
    # 使わず面素の幾何中心で L^(k) を評価すると、壁+床の3回反射が光線追跡比で最大500分の1
    # まで過小評価された（`_interreflection_increments` の F1_collect/F_collect を参照）。
    collect_centroid = _weighted_centroid(weight_collect, collect_cell)

    return centers, normals, areas, L1_cell, collect_cell, l1_centroid, collect_centroid


def _interreflection_increments(
    led: _Emitter, pt: _Emitter, surf: SurfaceSpec,
    near_rects: Sequence[Rect], boxes: np.ndarray,
    wall_height_m: float, include_floor: bool, floor_halfextent_m: float,
    led_intensity: float, pt_responsivity: float, max_range_m: float, bounces: int,
    half_angle_max_deg: float, separation_m: float,
    n_coarse_wall_u: int, n_coarse_wall_v: int, n_coarse_floor: int, n_grid_source: int,
) -> Dict[int, float]:
    """反射 2 回目以降の増分をラジオシティ近似で計算する（`{反射回数: 増分}`。k=2..bounces）。

    モジュール docstring「相互反射（AUDIT_056・モデル I）」節を参照。呼び出し側（`response()`）
    が既に計算済みの `led`/`pt`/`near_rects`/`boxes` をそのまま受け取る（1 回目の経路と
    同じ面・同じ遮蔽候補を使うことを保証するため、ここで作り直さない）。
    """
    if bounces <= 1:
        return {}

    wall_facets = _wall_facets(near_rects, wall_height_m)
    facet_specs = [(f, idx // 4, n_coarse_wall_u, n_coarse_wall_v) for idx, f in enumerate(wall_facets)]
    if include_floor:
        mid_xy = (led.pos[:2] + pt.pos[:2]) / 2.0
        facet_specs.append((_floor_facet(mid_xy, floor_halfextent_m), -1, n_coarse_floor, n_coarse_floor))

    if not facet_specs:
        return {k: 0.0 for k in range(2, bounces + 1)}

    centers_l, normals_l, areas_l, L1_l, collect_l, l1cen_l, colcen_l = [], [], [], [], [], [], []
    for facet, owner_idx, n_u, n_v in facet_specs:
        c, n, a, l1, col, l1cen, colcen = _facet_radiosity_cells(
            facet, owner_idx, led, pt, surf, led_intensity, pt_responsivity, max_range_m, boxes,
            n_grid_source, half_angle_max_deg, separation_m, n_u, n_v,
        )
        centers_l.append(c); normals_l.append(n); areas_l.append(a)
        L1_l.append(l1); collect_l.append(col); l1cen_l.append(l1cen); colcen_l.append(colcen)

    centers = np.concatenate(centers_l, axis=0)
    normals = np.concatenate(normals_l, axis=0)
    areas = np.concatenate(areas_l, axis=0)
    L = np.concatenate(L1_l, axis=0)               # L^(1)（面素ごとの平均）
    collect_weight = np.concatenate(collect_l, axis=0)   # dA 込みの PT 集光係数（面素ごと）
    l1_centroid = np.concatenate(l1cen_l, axis=0)        # L^(1) のフラックス加重重心（送り手位置）
    collect_centroid = np.concatenate(colcen_l, axis=0)  # PT 集光カーネルの加重重心（集光の受け手位置）

    n_p = centers.shape[0]

    # 迷路は柱・壁が隙間ゼロで隣接する構造（本モジュール冒頭・`_COPLANAR_GAP_EPS_M` の
    # コメント参照）なので、別々の直方体に属す面素どうしが「ほぼ接している」姿勢が実迷路で
    # 普通に起こる。点対点の形態係数カーネル `cosθcosθ/(πr²)` は r→0 で発散するが、
    # 面積積分した本当の形態係数（例: 共有辺で接する2枚の垂直面）は有限であり、発散は
    # 「面を1点で代表させる」近似の産物である。r の下限にパッチの実効半径（`sqrt(area)/2`
    # の和）を使う（標準的なラジオシティの工学的対処）。
    # 🔴 検証0-b で実測: この下限が無いと、面素分割数を1.5倍にしただけで増分が
    # 最大約100倍（0.112 -> 0.0011 桁で変動）動く姿勢があった（隣接する柱・壁の面素どうしの
    # r が分割の切り方で偶然ゼロに近づくため）。
    r_floor_recv = 0.5 * np.sqrt(areas)

    def _form_factor_matrix(
        recv_centers: np.ndarray, src_centers: np.ndarray, src_normals: np.ndarray, src_areas: np.ndarray,
    ) -> np.ndarray:
        """受け手 `recv_centers`（法線 `normals`）・送り手 `src_centers`（法線 `src_normals`）
        の形態係数 `F[i,j] = cosθ_i・cosθ_j/(π r²)`（可視判定込み・近接パッチの r 下限込み）を作る。
        """
        rel = src_centers[None, :, :] - recv_centers[:, None, :]     # rel[i,j] = src_j - recv_i
        r = np.linalg.norm(rel, axis=-1)
        r_floor = r_floor_recv[:, None] + 0.5 * np.sqrt(src_areas)[None, :]
        r_safe = np.maximum(np.maximum(r, r_floor), 1e-6)
        dir_ij = rel / r_safe[..., None]
        cos_i_ij = np.clip(np.einsum("ijk,ik->ij", dir_ij, normals), 0.0, None)
        cos_j_ij = np.clip(np.einsum("ijk,jk->ij", -dir_ij, src_normals), 0.0, None)
        mat = cos_i_ij * cos_j_ij / (math.pi * r_safe ** 2)
        if r_safe.shape[0] == r_safe.shape[1]:
            np.fill_diagonal(mat, 0.0)   # 同一面素どうし（i==j）は伝播させない
        if boxes.shape[0] > 0 and n_p > 0:
            nudged_recv = recv_centers + normals * 1e-6
            nudged_src = src_centers + src_normals * 1e-6
            V = _segment_occluded(nudged_recv[:, None, :], nudged_src[None, :, :], boxes, None)
            mat = np.where(V, 0.0, mat)
        return mat

    # --- 形態係数 F1（反射1回目→2回目の伝播だけに使う。送り手位置は面素の幾何中心ではなく
    #     L^(1) のフラックス加重重心を使う） ---
    #
    # 🔴 検証0-b で実測した問題（面素分割を1.5倍にしただけで増分が最大17分の1まで動いた）
    # への対処。narrow beam のせいで L^(1) は面素内で極端に偏っている（実測: 3×2分割の
    # 1面素だけが E=605、残り5面素は1e-40以下）。送り手位置に面素の幾何中心をそのまま
    # 使うと、分割の切り方が変わるたびに「実際に光っている場所」と「幾何中心」のずれが
    # 変わり、cosθ・r² が敏感に動いて収束しない。フラックス加重重心を使うことで、
    # 送り手位置が実際の明部に一致し、分割数への依存を落とす。
    F1 = _form_factor_matrix(centers, l1_centroid, normals, areas)

    # --- 形態係数 F（反射2回目以降どうしの伝播。間接照明はなめらかなので幾何中心のままでよい） ---
    F = _form_factor_matrix(centers, centers, normals, areas)

    # 🔴 集光の受け手位置に PT 集光カーネルの加重重心（`collect_centroid`）を使う版も試した
    # （壁+床で3回反射が光線追跡比最大500分の1という自主検算の結果への対処案）。
    # 実測: 改善はほぼ無く（40mm: 満量比差 0.02879→0.02881、ほぼ不変）、計算時間はほぼ倍増
    # （実迷路姿勢の平均590ms→1340ms）、検証0-bはぎりぎり不合格側へ動いた（0.00185→0.00208、
    # 分割点0.002）。**費用に見合う効果が無いため不採用**（`collect_centroid` 自体は
    # `_facet_radiosity_cells` の戻り値として残すが、ここでは使わない）。根本原因は
    # 「面素1枚が近距離センサの視野角（数度）に対して大きすぎる」という解像度の問題であり、
    # 重心を使う程度の局所補正では直らない。詳細は本ファイル docstring の限界の節、
    # および教授セッションへの報告を参照。
    del collect_centroid

    breakdown: Dict[int, float] = {}
    L_k = L
    for k in range(2, bounces + 1):
        Fmat = F1 if k == 2 else F   # 1回目→2回目だけ重心を使う（上のコメント参照）
        L_k = surf.diffuse * (Fmat @ (areas * L_k))
        breakdown[k] = float(np.sum(L_k * collect_weight))
    return breakdown


# ============================================================================
# 公開 API: response / adc
# ============================================================================
def response(
    sensor: IrSensorSpec,
    pose: PoseLike,
    surfaces: Sequence[Rect],
    surf: SurfaceSpec,
    *,
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M,
    include_floor: bool = True,
    floor_halfextent_m: float = DEFAULT_FLOOR_HALFEXTENT_M,
    n_grid: int = DEFAULT_N_GRID,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    led_intensity: float = 1.0,
    pt_responsivity: float = 1.0,
    occlusion: bool = True,
    bounces: int = 1,
    return_breakdown: bool = False,
    n_coarse_wall_u: int = DEFAULT_N_COARSE_WALL_U,
    n_coarse_wall_v: int = DEFAULT_N_COARSE_WALL_V,
    n_coarse_floor: int = DEFAULT_N_COARSE_FLOOR,
    n_grid_interreflection_source: int = DEFAULT_N_GRID_INTERREFLECTION_SOURCE,
) -> Union[float, Tuple[float, Dict[int, float]]]:
    """機体姿勢 `pose` のとき、そのセンサが受ける光量（任意単位）を返す。

    Args:
        sensor: センサ仕様。
        pose: 機体の世界座標系での姿勢。`classic.geometry.Pose`（`.x/.y/.theta`）か
            `(x, y, theta[rad])` のタプル。
        surfaces: 壁・柱の足跡（`classic.geometry.wall_obstacles` の出力をそのまま渡せる）。
        surf: 反射面の性質。
        wall_height_m: 壁の高さ [m]。
        include_floor: 床（z=0）も面として積分に含めるか。
        floor_halfextent_m: 床パッチの半幅 [m]（センサ周辺のみの有限矩形）。
        n_grid: 面 1 枚あたりの求積点数（1 軸あたり。多いほど精度が上がるが遅くなる）。
        max_range_m: これより遠い壁・遠い面素は積分から除く。
        led_intensity: LED の基準強度 I0（任意単位）。
        pt_responsivity: PT の基準感度（任意単位。面積などを丸めて含む）。
        occlusion: 面素ごとの寄与を足す前に、LED→点・点→PT の 2 本の線分が他の壁・柱で
            遮られていないかを判定し、遮られていれば寄与をゼロにするか（既定 True）。
            `False` にすると遮蔽を計算しない旧挙動と厳密に一致する（否定対照。
            `research_notes/note_034_ir_sensor_model.md` 追記分・本モジュール冒頭
            「遮蔽（オクルージョン）について」参照）。
        bounces: 反射回数（既定 1）。🔴 `bounces=1` は本引数を追加する前の `response()` と
            数値まで同一になる（相対差 1e-12 以下。`verification/AUDIT_056_PREREG_interreflection.md`
            検証0-a）。`bounces>=2` では、反射2回目以降の寄与をラジオシティ近似
            （粗い面素・拡散のみ。モジュール docstring「相互反射（AUDIT_056・モデル I）」節）で
            加算する。4 まで扱える。
        return_breakdown: True のとき `(total, breakdown)` を返す。`breakdown` は
            `{反射回数: その回数ちょうどの増分（gain 込み）}`（`bounces=1` のときは空 dict）。
            既定 False のときの戻り値の型は変わらない（float のまま）。
        n_coarse_wall_u/n_coarse_wall_v/n_coarse_floor: `bounces>=2` のときだけ使う、
            相互反射を解くラジオシティ用の粗い面素の分割数（壁側面: 長さ方向×高さ方向、床: 1辺）。
        n_grid_interreflection_source: `bounces>=2` のときだけ使う、相互反射の起点 `L^(1)` と
            PT 集光係数を粗い面素へ畳み込むときの warped 格子の求積点数（1軸あたり。
            narrow beam を正しく積分するために使う。`n_grid` とは独立）。

    Returns:
        センサが受ける光量（任意単位）。強度のまま使うか距離に直すかはここでは決めない
        （`adc()` で AD 変換器の生の値に変換してから、使い方はアルゴリズム側で選ぶ）。
        `return_breakdown=True` のときは `(total, breakdown)` のタプル。
    """
    led, pt = _sensor_world_geometry(sensor, pose)
    half_angle_max = max(sensor.led_half_angle_deg, sensor.pt_half_angle_deg)

    # 速さのための足切り 1: そもそもセンサの最大到達距離に入らない壁は最初から除く
    # （面積分の対象からも、遮蔽の候補直方体からも除ける。迷路全体の壁数によらず、
    # センサ周辺だけに計算量を抑える）。
    near_rects: list = []
    for r in surfaces:
        diag = math.hypot(r.hx, r.hy)
        dist_center = math.hypot(r.cx - led.pos[0], r.cy - led.pos[1])
        if dist_center - diag > max_range_m:
            continue
        near_rects.append(r)

    facets: list = []
    facet_owner: list = []   # 各 facet が属する遮蔽用直方体の番号（床は -1 = 属する箱が無い）
    for owner_idx, r in enumerate(near_rects):
        for f in _wall_facets([r], wall_height_m):
            facets.append(f)
            facet_owner.append(owner_idx)

    if include_floor:
        mid_xy = (led.pos[:2] + pt.pos[:2]) / 2.0
        facets.append(_floor_facet(mid_xy, floor_halfextent_m))
        facet_owner.append(-1)

    # 遮蔽の候補直方体（`near_rects` と同じ並び順 = facet_owner の番号と対応）。
    # occlusion=False のときは空配列にし、下の occluded_list がすべて None になることで
    # 従来コードと完全に同じ計算経路（同じ演算・同じ浮動小数）を通す（否定対照用）。
    boxes = _obstacle_boxes(near_rects, wall_height_m) if occlusion else np.zeros((0, 6))

    # バックフェイスカリング（LED から見て向こう向きの面は最初から除く）を先に済ませ、
    # 残った可視な面だけを以降の求積・遮蔽判定にかける。
    vis_facets: list = []
    vis_owner: list = []
    for facet, owner_idx in zip(facets, facet_owner):
        to_led = led.pos - facet.center
        if float(np.dot(to_led, facet.normal)) <= 0.0:
            continue
        vis_facets.append(facet)
        vis_owner.append(owner_idx)

    if not vis_facets:
        # LED を向いている面が1枚も無い＝1回目の直接光がゼロ。相互反射の起点
        # （L^(1) = ρ・E/π）も全面素でゼロになるので、bounces>1 でも増分は必ずゼロ。
        return (0.0, {}) if return_breakdown else 0.0

    grids = [_facet_grid(f, led, n_grid, half_angle_max, sensor.separation_m) for f in vis_facets]
    points_list = [g[0] for g in grids]
    dA_list = [g[1] for g in grids]

    # 速さのための足切り 2: 交差判定を numpy でまとめて行う。
    # 面ごとに `_segment_occluded` を逐次呼ぶと（可視な面の枚数）× 2 回の小さな numpy 呼び出しに
    # なる。可視な面すべての求積点を先頭軸にまとめて 1 本の配列に積み、LED 側・PT 側それぞれ
    # 1 回だけ `_segment_occluded` を呼ぶことで、小さな呼び出しの繰り返しを大きな配列演算
    # 1 回にまとめる（結果は数学的に「面ごとに呼んだ場合」と同一。詰め方が変わるだけ。
    # 計測では Python 呼び出し回数はこれで 30〜40 分の 1 に減ったが、実測時間の大半は
    # 呼び出しオーバーヘッドではなく（面の枚数）×（格子点数）×（候補直方体数）に比例する
    # 素の演算量だった。現実の迷路のように候補直方体数が多い場面で本当に効くのは、
    # `max_range_m` によるセンサ周辺への足切り＝上の「足切り 1」の方である）。
    if occlusion and boxes.shape[0] > 0:
        all_points = np.stack(points_list, axis=0)      # (n_vis, n_grid, n_grid, 3)
        owner_arr = np.array(vis_owner, dtype=int)       # 床は -1（除外なしの目印）
        occ_led = _segment_occluded(all_points, led.pos, boxes, owner_arr)
        occ_pt = _segment_occluded(all_points, pt.pos, boxes, owner_arr)
        occluded_all = occ_led | occ_pt                  # (n_vis, n_grid, n_grid)
        occluded_list = [occluded_all[i] for i in range(len(vis_facets))]
    else:
        occluded_list = [None] * len(vis_facets)

    total = 0.0
    for facet, points, dA, occluded in zip(vis_facets, points_list, dA_list, occluded_list):
        total += _integrate_facet(
            facet, led, pt, surf, points, dA, led_intensity, pt_responsivity,
            max_range_m, occluded,
        )

    if bounces <= 1 and not return_breakdown:
        # 🔴 検証0-a: bounces=1・breakdown不要のときは、ここまでの式・演算を一切変えない
        # （本引数を追加する前の `response()` と数値まで同一という要件そのもの）。
        return total * sensor.gain

    breakdown: Dict[int, float] = {}
    if bounces > 1:
        # 反射2回目以降は、1回目と同じ near_rects/boxes/led/pt を使うラジオシティ近似で加算する
        # （反射1回目の経路である上のループには一切触れない。モジュール docstring
        # 「相互反射（AUDIT_056・モデル I）」節参照）。
        increments = _interreflection_increments(
            led, pt, surf, near_rects, boxes, wall_height_m, include_floor, floor_halfextent_m,
            led_intensity, pt_responsivity, max_range_m, bounces,
            half_angle_max, sensor.separation_m,
            n_coarse_wall_u, n_coarse_wall_v, n_coarse_floor, n_grid_interreflection_source,
        )
        for k, delta in increments.items():
            total += delta
            breakdown[k] = delta * sensor.gain

    final = total * sensor.gain
    if return_breakdown:
        return final, breakdown
    return final


def adc(
    value: float,
    bits: int = 12,
    full_scale: float = 1.0,
    noise_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """`response()` の生の値を、AD 変換器が返すであろう整数値にする（分解能・飽和・雑音）。

    Args:
        value: `response()` の出力（任意単位）。
        bits: AD 変換器の分解能 [bit]。
        full_scale: この値のときフルスケール（`2**bits - 1`）になるとして正規化する。
        noise_sigma: 雑音の標準偏差（`value` と同じ任意単位）。量子化の前に加える。
        rng: 乱数生成器。省略時は毎回新規に作る（再現性が要る場合は渡すこと）。

    Returns:
        `0` 〜 `2**bits - 1` にクランプした整数コード。
    """
    v = float(value)
    if noise_sigma > 0.0:
        gen = rng if rng is not None else np.random.default_rng()
        v = v + float(gen.normal(0.0, noise_sigma))
    max_code = (1 << bits) - 1
    normalized = v / full_scale if full_scale != 0.0 else 0.0
    code = int(round(normalized * max_code))
    return max(0, min(max_code, code))


# ============================================================================
# 高速フォワードモデル（AUDIT_059・段A: 直接光のみ）
# ============================================================================
"""
背景・事前登録: `verification/AUDIT_059_PREREG_fast_sensor.md`。`response()` は実迷路で
1 回あたり 59ms（`n_grid=28`・床なし）かかる。射程 350mm には壁・柱が約 42 枚（面素で
168 枚）入るが、読みに効くのは中央値 2 枚・最大 5 枚（寄与が全体の 0.1% 超のもの）で、
残りにも同じだけ格子を敷いて積分・遮蔽判定をしているのが遅さの理由（教授セッションの実測）。

`response_fast()` は反射 1 回（直接光）だけを、次の 2 段構えで高速化する:

1. **積分する面を絞る。** LED の光錐 **かつ** PT の視野の**両方**に入り得る面だけを積分対象
   にする（`_facet_maybe_in_cone`）。判定は保守的（寄与する面を絶対に落とさない）に作る:
   面の四隅・中心だけの抜き取りでは、長い壁では光が壁の途中（端でも中心でもない点）に
   当たる場合を見落とす（教授セッションの試作で実際に起きた失敗。読みが 0.61→0.07 まで
   落ちた）。そこで、光軸（射線）が面の平面と交わる点を面の矩形範囲へクランプした
   **解析点**（`_facet_closest_direction_point`。壁の途中の当たり点を厳密に当てる）を
   必ず候補に加え、さらに面内の密な格子（既定 9×5）を加えた候補点全体の中で最小角度
   （＝最大 `cos`）を取る。カットオフ角は既定で LED ±15°・PT ±25°（`cos^m` がそれぞれ
   1e-8・7e-6 まで落ちる角度。note_034 の実測に基づく、余裕を大きく取った値）。
2. **遮る面は絞らない。** 射程内の矩形（`near_rects`。`response()` と同じ足切り）は
   すべて遮蔽の候補直方体として残す（積分対象と遮蔽候補は別々に選ぶ。積分対象を
   絞り込むのと同じ基準で遮蔽候補も絞ると、手前の壁を誤って落として奥の壁が
   見えてしまう事故が教授セッションの試作で 60 姿勢中 2 件起きた）。

積分の格子（`_facet_grid`。tan ワープした非一様格子）と遮蔽の一括判定
（`_segment_occluded`）は `response()` と同じ実装をそのまま使う（作り直さない）。
反射は 1 回のみ（`bounces=1` 相当）・床は含めない（ユーザの決定。`note_034` 追記16:
「床の反射は無視する」）。**`response()` 自体は一切変更しない。**

## 速さの追加対策（AUDIT_059 §4 実測: `M6` 合格・`T`=6.01ms で目標未達。2026-08-22）

実測で支配的だった 2 箇所を、**判定の中身を変えずに**速くする。

1. **遮蔽候補直方体を、「絞る」のではなく数学的に安全な範囲で「削る」。**
   上の「2. 遮る面は絞らない」は**積分対象の絞り込みと同じ近似的な基準**で遮蔽候補も
   絞ることを禁じたものであり、幾何学的に厳密な削減まで禁じてはいない。遮る可能性の
   ある直方体は、必ず「センサ（LED・PT）と、積分対象の面の求積点」を結ぶ線分の
   どれかと交わる。線分の両端は凸な軸平行直方体（AABB）
   `{LED,PT} ∪ {積分対象の面の求積点すべて}` の内部にあるので、線分全体がその AABB の
   内部に収まる。**この AABB と足跡がまったく交わらない直方体は、原理的にどの線分とも
   交わり得ない**（`prune_occlusion_boxes`。既定 True）。積分対象を絞り込んだ**あと**の
   求積点で AABB を作るので、この削減は「2. 遮る面は絞らない」の意図（積分対象の
   絞り込み基準をそのまま遮蔽候補に流用しない）と両立する。`prune_occlusion_boxes=False`
   にすると削減前（`near_rects` 全体を遮蔽候補にする旧挙動）に戻る（自己検査2用）。
2. **絞り込み判定（`_facet_maybe_in_cone`）を numpy で面をまたいで一括化する。**
   判定の中身・カットオフ角・候補点の作り方（解析点＋面内格子）は変えない。
   `_facets_maybe_in_cone_batch()` が面ごとの Python ループを、面を軸に積んだ配列への
   一括演算に置き換える（`_facet_maybe_in_cone` は自己検査3の比較対象として残す）。

## 速さの追加対策・その2（実測: `T`=2.872ms・遮蔽判定が52%を占めていた。2026-08-22）

対策1・2のあとも遮蔽判定（`_segment_occluded`）が支配的だった。積分対象は平均6.5面まで
絞れているのに、面あたり `n_grid`×`n_grid`（既定20×20=400 点）の求積点すべてを、遮蔽候補の
直方体（平均7枚）に対して判定しており、これが時間の大半を占めていた。**判定の中身は
変えず**、次の2つを追加した。

3. **遮蔽を判定する前に、寄与しない求積点を落とす。** LED・PT とも半値角が数度と鋭い
   （`m` は LED の 3° で約505）ため、面 1 枚の求積点の大半は寄与が桁違いに小さい。
   遮蔽判定より先に、遮蔽なしで安く計算できる「寄与見積り」
   `weight_led・weight_pt・dA`（`weight_led=cos^m(θ_e)/r_e²`・`weight_pt=cos^m(θ_r)/r_v²`。
   ともに `_integrate_facet` の `irradiance`/`pt_sensitivity・1/r_v²` と同じ量。`cos_i`・
   `cos_v` は 1 以下、`diffuse/π+specular` は定数上限があるので、この見積りは実際の寄与の
   定数倍の上界になる）を面ごとの求積点すべてで作り、**小さい方から順に足していって
   面の総和の `occlusion_point_weight_frac` 倍に達するまでの点だけ**を、遮蔽を判定せず
   最初から寄与ゼロ（＝遮蔽されたのと同じ扱い）とする（`prune_occlusion_points`。
   既定 True。実測は自己検査4・`verification/audit_059_fast_sensor.py`）。

   🔴 **点ごとの絶対しきい値（面の最大の frac 倍未満は落とす）では不十分だった**
   （実測で発覚。修正の経緯）:
   (a) 最初 LED 側の重みだけで判定したところ、LED と PT は離隔ぶんだけ光軸がずれており、
   近距離では両者の明部が面上でずれて重ならないことがあるため（モジュール docstring
   「数値積分について」節）、「LED からは暗いが PT からは非常に近い（`r_v` が小さく
   `1/r_v²` で増幅される）」点を誤って落とし、自己検査4で満量比 1.5e-6 の差が出た
   （しきい値 1e-8 の150倍）。→ PT側の重みも掛けて判定するよう直した。
   (b) それでも LED×PT の積を面の最大値と比べる絶対しきい値では、満量比 7.5e-7 の
   差が残った。tan ワープ格子は遠方の点ほど1点あたりの面積要素 `dA` が大きく
   （密度の粗さを重みで補っている）、「個々にはしきい値未満でも `dA` が大きい点」が
   大量にあるケースで、落とした点の総量が積み上がっていた。→ 点ごとの絶対しきい値では
   なく、**面ごとの総和に対する累積の取りこぼし量**で予算管理する方式（小さい点から
   ソートして累積和を取り、面の総和の `frac` 倍を超えない範囲でだけ落とす）に直した。
   (c) この累積予算方式でも `frac=1e-6`（事前の見立て）はまだ甘く、自己検査4で
   満量比 6.4e-8 の差が出た。`frac` を下げて実測したところ
   `1e-8→6.8e-9`・`1e-9→6.1e-10`・`1e-10→5.9e-11` と単調に下がったため、
   自己検査4の基準（1e-8）に対して1桁の余裕を持たせ、既定を
   `DEFAULT_OCCLUSION_POINT_WEIGHT_FRAC = 1e-9` にした（764姿勢の実測値。
   事前登録の判定量・分割点・標本・乱数種はいずれも変えていない — これは
   `response_fast()` 内部の実装パラメータであり、事前登録の対象ではない）。
   面の最大値そのものを持つ点は必ず残るので、面が丸ごと空になることはない。
4. **射程の足切り・面の展開・バックフェイスカリングを numpy でまとめる。** `_wall_facets`・
   `_obstacle_boxes`（`response()` と共有）は変更せず、`response_fast()` の中だけで
   矩形配列（`np.asarray(surfaces)`。`Rect` は `NamedTuple` なのでそのまま配列化できる）
   から同じ式を一括計算する形に置き換える。判定の中身・使う定数・面の並べ方の値は
   `_wall_facets`/backface culling と同一（順序が変わることはあるが、寄与の総和は
   加算の順序に依存しないので判定量には影響しない）。
"""

DEFAULT_N_GRID_FAST: int = 20              # 積分格子（既定。誤差と速さの表から選定。事前登録§4）
DEFAULT_LED_CONE_MARGIN_DEG: float = 15.0  # LED光錐の絞り込みカットオフ角（cos^mが1e-8まで落ちる角）
DEFAULT_PT_CONE_MARGIN_DEG: float = 25.0   # PT視野の絞り込みカットオフ角（cos^mが7e-6まで落ちる角）
DEFAULT_CONE_FILTER_N_U: int = 9           # 絞り込み判定の標本数（面の長さ方向。解析点に加える格子）
DEFAULT_CONE_FILTER_N_V: int = 5           # 絞り込み判定の標本数（面の高さ方向）
# 求積点の足切り（速さの追加対策3）: 面ごとの寄与見積り（LED側×PT側の重み×dA）の総和のうち
# 小さい方からこの割合までを遮蔽判定なしで落とす（既定 1e-9。実測は自己検査4。
# モジュール docstring「速さの追加対策・その2」節参照）。
DEFAULT_OCCLUSION_POINT_WEIGHT_FRAC: float = 1e-9


def _facet_closest_direction_point(facet: _Facet, apex: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """`facet` の矩形範囲内で、頂点 `apex`・軸 `axis` の光軸に最も近い点を返す（解析点）。

    光軸（射線）が面の平面と前方で交わるならその交点を、面の矩形範囲
    （`half_u`/`half_v`）へクランプして返す。前方交点が無ければ `apex` を面の平面へ
    投影した点（同じくクランプ）を返す（`_facet_anchor_and_scale` と発想は同じだが、
    あちらは積分格子の集中点を決めるためのもので矩形範囲へクランプしない。こちらは
    常に「面の上の点」を返す必要があるので必ずクランプする）。

    抜き取り（四隅＋中心の 5 点）だけで絞り込み判定をすると、長い壁では光が壁の途中
    （端でも中心でもない点）に当たる場合を見落とす — 本関数はその「途中」の当たり点を
    解析的に当てることでこの見落としを防ぐ（`_facet_maybe_in_cone` が使う）。
    """
    denom = float(np.dot(axis, facet.normal))
    t = float(np.dot(facet.center - apex, facet.normal)) / denom if abs(denom) > 1e-9 else -1.0
    if t > 0.0:
        p_hit = apex + t * axis
        rel = p_hit - facet.center
    else:
        rel = apex - facet.center
    pu = float(np.clip(np.dot(rel, facet.u), -facet.half_u, facet.half_u))
    pv = float(np.clip(np.dot(rel, facet.v), -facet.half_v, facet.half_v))
    return facet.center + pu * facet.u + pv * facet.v


def _facet_maybe_in_cone(
    facet: _Facet, apex: np.ndarray, axis: np.ndarray, margin_deg: float, n_u: int, n_v: int,
) -> bool:
    """`facet` が、頂点 `apex`・軸 `axis` の円錐（半頂角 `margin_deg`）に入り得るかを
    保守的に判定する（モジュール docstring「高速フォワードモデル」節を参照）。

    解析点（`_facet_closest_direction_point`）＋面内の密な格子（`n_u × n_v`）を合わせた
    候補点の中で最小角度（＝最大 `cos`）を取り、カットオフ角と比べる。候補点のどれか 1 つ
    でもカットオフ角以内なら「入り得る」と判定して残す（抜き取りではなく、壁の途中の
    当たり点を解析点で必ず拾う設計。詳細はモジュール docstring 参照）。
    """
    p_analytic = _facet_closest_direction_point(facet, apex, axis)

    us = np.linspace(-facet.half_u, facet.half_u, n_u)
    vs = np.linspace(-facet.half_v, facet.half_v, n_v)
    UU, VV = np.meshgrid(us, vs, indexing="ij")
    grid_pts = (
        facet.center[None, None, :]
        + UU[:, :, None] * facet.u[None, None, :]
        + VV[:, :, None] * facet.v[None, None, :]
    ).reshape(-1, 3)

    pts = np.vstack([p_analytic[None, :], grid_pts])
    d = pts - apex[None, :]
    r = np.linalg.norm(d, axis=-1)
    r_safe = np.maximum(r, 1e-9)
    cos_ang = (d @ axis) / r_safe
    max_cos = float(np.max(cos_ang))
    return max_cos >= math.cos(math.radians(margin_deg))


def _stack_facets(facets: Sequence[_Facet]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`facets` の位置・向き・寸法を numpy 配列へまとめてスタックする
    （`_facets_maybe_in_cone_batch` で LED・PT 両方の判定に使い回す）。
    戻り値: `(centers, normals, u_vecs, v_vecs, half_u, half_v)`（先頭軸は面の枚数）。
    """
    centers = np.stack([f.center for f in facets])
    normals = np.stack([f.normal for f in facets])
    u_vecs = np.stack([f.u for f in facets])
    v_vecs = np.stack([f.v for f in facets])
    half_u = np.array([f.half_u for f in facets])
    half_v = np.array([f.half_v for f in facets])
    return centers, normals, u_vecs, v_vecs, half_u, half_v


def _facets_maybe_in_cone_batch(
    stacked: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    apex: np.ndarray, axis: np.ndarray, margin_deg: float, n_u: int, n_v: int,
) -> np.ndarray:
    """`_facet_maybe_in_cone()` の numpy 一括版（面をまたいでループしない）。

    判定の中身・カットオフ角・候補点の作り方（解析点＋面内の密な格子）は一切変えない。
    `_facet_maybe_in_cone` を面ごとに呼んだ場合と同じ結果になることは自己検査3
    （`verification/audit_059_fast_sensor.py`）で全姿勢・全面について確認している。
    `stacked` は `_stack_facets()` の戻り値。戻り値は shape `(n_facets,)` の bool 配列。
    """
    centers, normals, u_vecs, v_vecs, half_u, half_v = stacked
    n_facets = centers.shape[0]
    if n_facets == 0:
        return np.zeros(0, dtype=bool)

    # 解析点（`_facet_closest_direction_point` の一括版。同じ場合分けをベクトル化しただけ）。
    denom = normals @ axis
    safe_denom = np.where(np.abs(denom) > 1e-9, denom, 1.0)
    t = np.where(
        np.abs(denom) > 1e-9,
        np.einsum("nk,nk->n", centers - apex[None, :], normals) / safe_denom,
        -1.0,
    )
    p_hit = apex[None, :] + t[:, None] * axis[None, :]
    rel_hit = p_hit - centers
    rel_apex = apex[None, :] - centers
    rel = np.where((t > 0.0)[:, None], rel_hit, rel_apex)
    pu = np.clip(np.einsum("nk,nk->n", rel, u_vecs), -half_u, half_u)
    pv = np.clip(np.einsum("nk,nk->n", rel, v_vecs), -half_v, half_v)
    analytic_pts = centers + pu[:, None] * u_vecs + pv[:, None] * v_vecs   # (n_facets,3)

    # 面内の密な格子（n_u×n_v点。面ごとに half_u/half_v でスケールした linspace(-h,h,n) と等価）。
    t_u = np.linspace(-1.0, 1.0, n_u)
    t_v = np.linspace(-1.0, 1.0, n_v)
    us = t_u[None, :] * half_u[:, None]        # (n_facets, n_u)
    vs = t_v[None, :] * half_v[:, None]        # (n_facets, n_v)
    grid_pts = (
        centers[:, None, None, :]
        + us[:, :, None, None] * u_vecs[:, None, None, :]
        + vs[:, None, :, None] * v_vecs[:, None, None, :]
    )   # (n_facets, n_u, n_v, 3)
    grid_pts = grid_pts.reshape(n_facets, n_u * n_v, 3)

    all_pts = np.concatenate([analytic_pts[:, None, :], grid_pts], axis=1)  # (n_facets, 1+n_u*n_v, 3)

    d = all_pts - apex[None, None, :]
    r = np.linalg.norm(d, axis=-1)
    r_safe = np.maximum(r, 1e-9)
    cos_ang = np.einsum("fck,k->fc", d, axis) / r_safe
    max_cos = np.max(cos_ang, axis=1)
    return max_cos >= math.cos(math.radians(margin_deg))


"""
# ============================================================================
# 隅の相互反射（AUDIT_059・段B: 直角に交わる2壁の2〜4回反射）
# ============================================================================
背景・事前登録: `verification/AUDIT_059_PREREG_fast_sensor.md`（段B）。ユーザの決定
（`note_034` 追記16: ρ=0.8・**床は無視する**）により、間接光は「直角に交わる2枚の壁が
作る隅」だけに絞られる（平らな壁1枚では面の偶奇により厳密にゼロ。実測で確認済み）。

## 隅の見つけ方（安い判定）

迷路の壁・柱の矩形足跡（`Rect`）はすべて軸平行なので、面（`_wall_facets` が壁1枚を
展開した側面4枚）は法線が必ず ±x か ±y のどちらかを向く。**直角に交わる面のペア**は
「法線が x 向きの面」と「法線が y 向きの面」の組み合わせだけである。

**柱（`hx`・`hy` がともに `DEFAULT_POST_HALF_EXTENT_THRESHOLD_M` 未満の矩形）に属する
面はペア候補から除く。**理由: 壁は柱ぶんの半長（`post_size/2`）を引いた位置で止まるため、
壁の端と柱の面は隙間ゼロで接するが、**壁どうし（柱を挟んだ向かい）は柱1個ぶん
（`post_size/2`＝6mm）の隙間がある**。柱自身の2面（同じ柱の隣り合う面どうし）は
常に凸（外向きの corner）で、両方の面が互いを見る向き（`cosθ` が正）にならず光が
通らない（`_corner_pairs()` の判定式そのものが凹の隅だけを正の寄与にする。凸の隅は
`cosθ` の clip でゼロになるため、含めても無害だが計算が無駄になる。柱を候補から
除くことで、この無駄と柱どうしの偽陽性を両方減らす）。壁どうしを直接ペアにし、
柱ぶんの隙間は許容差 `DEFAULT_CORNER_ADJACENCY_TOL_M`（=post_size 相当＋余裕）で
橋渡しする（柱自体を経由する2ホップより、柱を無視して壁どうしを直接結ぶほうが、
柱の面積が壁よりずっと小さい＝支配的な壁-壁結合を過小評価しない）。

判定式（`_corner_pairs()`）: 法線 ±x の面 A（位置 `x0`・y方向の範囲 `[y_lo,y_hi]`）と
法線 ±y の面 B（位置 `y0`・x方向の範囲 `[x_lo,x_hi]`）が隅を作るのは、
`x_lo - tol <= x0 <= x_hi + tol` かつ `y_lo - tol <= y0 <= y_hi + tol` のとき
（＝互いの「延長線」が相手の矩形範囲内で交わる）。同じ矩形（柱・壁）に属する面どうし
はそもそも候補から除いてあるので、`owner` が同じペアは出てこない。

## 隅の寄与（教授セッションの試作。式の由来はモジュール docstring 冒頭を参照）

    寄与 = (ρ/π)² × Σ_x Σ_y E(x)・[cosθ_x cosθ_y / r_xy²]・G(y)・dA_x dA_y

`E(x)` は面 A 上の点 `x` での LED 放射照度（cos^m(θ_e)/r_e²・cosθ_i。`response()`/
`response_fast()` の `_integrate_facet` の `irradiance*cos_i` と同じ量）、`G(y)` は
面 B 上の点 `y` での PT 集光係数（`cosθ_v・cos^m(θ_r)/r_v²`）。**A・B とも積分点は
`_facet_grid()`（LED 光軸のホットスポットに集中させた tan ワープ格子。直接光と同じ
実装をそのまま流用）で取る**——A・B の少なくとも一方は LED の光軸から外れて grazing
にしか当たらないが、鋭い指向性でもまだ無視できない重みが残ることを教授セッションが
実測で確認済み（モジュール docstring 冒頭）。窓の幅は離隔ぶんの余裕（`separation_m`）
を含むので、PT のホットスポットが LED と少しずれていてもおおむね拾える。

**向きは2通りあり、両方足す**（教授セッションの実測: 片方向だけだと基準の0.64倍にしか
ならない）。A が LED に照らされ B を PT が見る「A→B」と、その逆「B→A」を両方計算する
（`corner_both_directions=False` で片方向だけにできる。否定対照用）。

**3回目・4回目の反射も、同じ核行列 `K`（cosθ_x cosθ_y/r_xy²、A・B の点の対ごと）を
再利用して安く足す。**2回目で「A→Bに届く放射照度」`irr_b2`（=A の送信ベクトルに `K` を
掛けたもの）が副産物として手に入るので、これを B の新しい送信ベクトルにして `K` に
もう一度通せば3回目（B→A）が、さらにもう一度通せば4回目（A→B）が出る（行列 `K` を
毎回作り直さない。追加コストは行列ベクトル積1回ぶんだけ）。分布はすでに2回散っていて
なめらかという教授セッションの見立て（モジュール docstring 冒頭）とも整合する
（`K` そのものは変えず、送信ベクトルだけを更新する反復なので、暗に「送り手位置は
そのつどのなめらかな分布」を使っている）。

**A・B間（隅を挟んだ点どうし）の遮蔽は判定しない**（凹の隅なので原理的に自分自身
どうしには遮られない。他の壁による遮蔽は近似的に無視——隅のごく近傍が主経路なので
影響は小さいと考えている。既知の限界）。LED→A・LED→B・PT→A・PT→B の4方向は
`response_fast()` 本体と同じ `_segment_occluded()`（`near_arr` 由来の全遮蔽候補直方体、
自分自身は owner 番号で除外）で判定する。

🔴 **実装時に見つかった誤り（重要）**: `E(x)`・`G(y)` を1回だけ `ρ/π` で結んだ式
（Σ_x Σ_y E(x)K(x,y)G(y)dAxdAy に `(ρ/π)¹` だけ掛ける）を最初に実装したところ、
行き止まりの奥の実測値（光線追跡の増分）に対して3〜4倍の過大評価になった。
原因は「面に届く放射照度」と「その面が送り出す放射輝度」の取り違え——
PT の集光係数 `G` は放射輝度に掛ける量なので、受け取った放射照度に**もう一段** `ρ/π`
（その面自身の反射）を掛けて放射輝度に変換してから `G` と結ぶ必要がある
（`(ρ/π)²` になるのはこのため。式の由来どおり）。この段になって、当初「面素の
大きさでクランプする」としていた `r_xy²` の下限クランプも誤りの産物だったと判明した
——過大評価をこの過大なクランプ値が偶然打ち消し、つじつまが合っているように
見えていただけだった。式を正しく直すと、warped 格子（`_facet_grid`。直接光と同じ
実装）は `r_xy²` に格子依存のクランプを掛けなくても素直に収束する（`n_grid` を
上げるほど光線追跡の増分に近づくことを実測で確認済み）。`r_xy²` の下限は
0除算を避けるためだけの極小値（`r_clamp_min`）にとどめてある。

## 安い判定で「隅が効く姿勢」だけ計算する

`_corner_pairs()` で見つかる幾何学的なペアは、行き止まりの奥では姿勢によらず**常に
存在する**（隅そのものは姿勢に関係なく物理的にそこにある）。効くかどうかは
向き（LED/PT がどれだけ隅を向いているか）に強く依存するため、幾何学的な存在だけでは
「安い判定」にならない。そこで2段構えにする:

1. **粗い格子（`corner_gate_n_grid`。既定 1 面あたり数点角）** で `_corner_pair_contribution`
   を計算し、`contribution2+3+4` が `corner_gate_threshold` を超えたペアだけを残す
   （2〜4回すべてを含めて判定する。既定の粗さでも安価——面あたりの点数が小さいので
   核行列 `K` も小さい）。
2. 残ったペアだけ、**細かい格子（`corner_n_grid`）** で本計算をやり直す。

`corner_stats`（呼び出し側が渡す `dict`）に、姿勢ごとの作動状況
（見つかったペア数・作動したペア数・粗い/細かい計算に使った時間）を記録できる
（`verification/audit_059_fast_sensor.py` の測定で「作動した割合」を出すのに使う）。
"""


DEFAULT_CORNER_ADJACENCY_TOL_M: float = 0.010   # 柱1個ぶん(6mm)の隙間を橋渡しする許容差
DEFAULT_CORNER_GATE_N_GRID: int = 6             # 作動判定（安い判定）用の格子（1面あたり6×6）
DEFAULT_CORNER_N_GRID: int = 16                 # 本計算用の格子（1面あたり16×16。速さ優先の実測で選定）
DEFAULT_CORNER_GATE_THRESHOLD: float = 1.0e-4   # 作動判定のしきい値（生値。led_intensity=1基準）


def _corner_pairs(
    centers: np.ndarray, normals: np.ndarray, half_u: np.ndarray, owner: np.ndarray,
    near_arr: np.ndarray, tol: float, cand_mask: Optional[np.ndarray] = None,
) -> list:
    """`centers`/`normals`/`half_u`/`owner`（`response_fast()` の射程内・バックフェイス
    カリング済みの候補面。`narrow_facets` の絞り込み前の集合）から、直角に交わり
    柱1個ぶんの隙間以内で接する壁の面ペアを列挙する（モジュール docstring「隅の
    相互反射」節参照）。戻り値は `(i, j)` のリスト（`i` は法線 ±x の面、`j` は法線 ±y
    の面。どちらも `centers` 等と同じ通し番号）。

    `cand_mask`（`centers` と同じ長さの bool 配列。省略時は絞り込みなし）を渡すと、
    `cand_mask` が `False` の面は候補から除く（`_corner_interreflection_total()` が
    LED光錐・PT視野のどちらにも入り得ない面をあらかじめ落とすのに使う。安い判定・
    本計算の対象そのものを減らす、計算量削減のための絞り込み）。
    """
    is_post = (near_arr[:, 2] < DEFAULT_POST_HALF_EXTENT_THRESHOLD_M) & \
              (near_arr[:, 3] < DEFAULT_POST_HALF_EXTENT_THRESHOLD_M)
    if centers.shape[0] == 0:
        return []
    owner_is_post = is_post[owner]
    x_facing = (np.abs(normals[:, 0]) > 0.5) & ~owner_is_post
    y_facing = (np.abs(normals[:, 1]) > 0.5) & ~owner_is_post
    if cand_mask is not None:
        x_facing = x_facing & cand_mask
        y_facing = y_facing & cand_mask
    idx_x = np.nonzero(x_facing)[0]
    idx_y = np.nonzero(y_facing)[0]
    if idx_x.size == 0 or idx_y.size == 0:
        return []

    x0 = centers[idx_x, 0]
    y_lo = centers[idx_x, 1] - half_u[idx_x]
    y_hi = centers[idx_x, 1] + half_u[idx_x]
    y0 = centers[idx_y, 1]
    x_lo = centers[idx_y, 0] - half_u[idx_y]
    x_hi = centers[idx_y, 0] + half_u[idx_y]
    owner_x = owner[idx_x]
    owner_y = owner[idx_y]

    cond_x = (x_lo[None, :] - tol <= x0[:, None]) & (x0[:, None] <= x_hi[None, :] + tol)
    cond_y = (y_lo[:, None] - tol <= y0[None, :]) & (y0[None, :] <= y_hi[:, None] + tol)
    same_owner = owner_x[:, None] == owner_y[None, :]
    touch = cond_x & cond_y & ~same_owner
    ii, jj = np.nonzero(touch)
    return [(int(idx_x[a]), int(idx_y[b])) for a, b in zip(ii, jj)]


def _corner_face_led_irradiance(
    points: np.ndarray, normal: np.ndarray, led: _Emitter, led_intensity: float,
    max_range_m: float, boxes: np.ndarray, owner: int, occlusion: bool,
) -> np.ndarray:
    """面上の点群 `points`（shape (N,3)）での LED 放射照度 `E(x)`
    （`cos^m(θ_e)/r_e²・cosθ_i`。`_integrate_facet` の `irradiance*cos_i` と同じ式）。"""
    d = points - led.pos
    r = np.maximum(np.linalg.norm(d, axis=-1), 1e-6)
    dirv = d / r[:, None]
    cos_e = np.clip(dirv @ led.axis, 0.0, 1.0)
    cos_i = np.clip(np.einsum("ik,k->i", -dirv, normal), 0.0, 1.0)
    val = led_intensity * cos_e ** led.m / (r ** 2) * cos_i
    valid = (r < max_range_m) & (r > 1e-9)
    if occlusion and boxes.shape[0] > 0:
        occ = _segment_occluded(points, led.pos, boxes, owner if owner >= 0 else None)
        valid = valid & ~occ
    return np.where(valid, val, 0.0)


def _corner_face_pt_collection(
    points: np.ndarray, normal: np.ndarray, pt: _Emitter, pt_responsivity: float,
    boxes: np.ndarray, owner: int, occlusion: bool,
) -> np.ndarray:
    """面上の点群 `points` での PT 集光係数 `G(y)`（`cosθ_v・cos^m(θ_r)/r_v²`）。"""
    d = pt.pos - points
    r = np.maximum(np.linalg.norm(d, axis=-1), 1e-6)
    dirv = d / r[:, None]
    cos_v = np.clip(np.einsum("ik,k->i", dirv, normal), 0.0, 1.0)
    cos_r = np.clip(np.einsum("ik,k->i", -dirv, pt.axis), 0.0, 1.0)
    val = cos_v * cos_r ** pt.m / (r ** 2) * pt_responsivity
    valid = r > 1e-9
    if occlusion and boxes.shape[0] > 0:
        occ = _segment_occluded(points, pt.pos, boxes, owner if owner >= 0 else None)
        valid = valid & ~occ
    return np.where(valid, val, 0.0)


def _corner_pair_contribution(
    facet_a: _Facet, facet_b: _Facet, owner_a: int, owner_b: int,
    led: _Emitter, pt: _Emitter, surf: SurfaceSpec,
    n_grid: int, half_angle_max_deg: float, separation_m: float,
    led_intensity: float, pt_responsivity: float, max_range_m: float,
    boxes: np.ndarray, occlusion: bool, both_directions: bool,
) -> float:
    """面 A・B の隅で2〜4回目の反射を計算する（モジュール docstring「隅の相互反射」節）。

    `both_directions=False` にすると「A→B」だけ（LED が A を照らし PT が B を見る経路）
    にする（否定対照用。教授セッションの実測では基準の0.64倍にしかならなかった）。
    """
    pts_a, dA_a = _facet_grid(facet_a, led, n_grid, half_angle_max_deg, separation_m)
    pts_b, dA_b = _facet_grid(facet_b, led, n_grid, half_angle_max_deg, separation_m)
    pts_a = pts_a.reshape(-1, 3)
    dA_a = dA_a.reshape(-1)
    pts_b = pts_b.reshape(-1, 3)
    dA_b = dA_b.reshape(-1)

    E_a = _corner_face_led_irradiance(pts_a, facet_a.normal, led, led_intensity, max_range_m, boxes, owner_a, occlusion)
    G_a = _corner_face_pt_collection(pts_a, facet_a.normal, pt, pt_responsivity, boxes, owner_a, occlusion)
    vec_ga = G_a * dA_a

    if both_directions:
        E_b = _corner_face_led_irradiance(pts_b, facet_b.normal, led, led_intensity, max_range_m, boxes, owner_b, occlusion)
    else:
        E_b = np.zeros(pts_b.shape[0])
    G_b = _corner_face_pt_collection(pts_b, facet_b.normal, pt, pt_responsivity, boxes, owner_b, occlusion)
    vec_gb = G_b * dA_b

    # r_xy² の下限クランプ（0除算・数値的な特異点の回避だけが目的の極小値）。
    # 🔴 当初は「面素の大きさ」（格子が細かいほど縮む量）でクランプしていたが、
    # これは実測で誤り（下の「見つかった実装の誤り」参照）と判明したバグと
    # 見かけ上つじつまが合ってしまっていた（過大評価をこの過大なクランプが打ち消して
    # いた）ため取り除いた。正しい式（下記）は warped 格子のまま r_clamp 無しで
    # 収束することを実測で確認済み（自己検査5とは別に、光線追跡の増分との突き合わせ
    # で確認。`verification/audit_059_fast_sensor.py` の段Bステージ参照）。
    r_clamp_min = 1e-6

    diff = pts_a[:, None, :] - pts_b[None, :, :]        # (Na, Nb, 3)  a側 - b側
    r2 = np.maximum(np.sum(diff * diff, axis=-1), r_clamp_min ** 2)
    r = np.sqrt(r2)
    dir_ba = diff / r[..., None]                        # B→A 方向
    cos_a = np.clip(np.einsum("ijk,k->ij", -dir_ba, facet_a.normal), 0.0, 1.0)   # A→B 方向で評価
    cos_b = np.clip(np.einsum("ijk,k->ij", dir_ba, facet_b.normal), 0.0, 1.0)    # B→A 方向で評価
    K = cos_a * cos_b / r2                               # (Na, Nb)

    coef = surf.diffuse / math.pi

    send_a1 = coef * E_a * dA_a      # A(直接光)が送る量＝A から出る放射輝度×面積要素
    send_b1 = coef * E_b * dA_b      # B(直接光)が送る量（both_directions=False なら0）

    # 🔴 見つかった実装の誤り（実測で発覚。行き止まりの奥の実測値との突き合わせで
    # 3〜4倍の過大評価が出て発覚した）: `irr_bN`/`irr_aN` は「面に届く放射照度」であって
    # 「その面が送り出す放射輝度」ではない。PT の集光係数 `G` は放射輝度に掛ける量
    # （`_integrate_facet` の `radiance * cos_v * pt_sensitivity / r_v²` と同じ規格）なので、
    # 収集する直前に必ずもう一段 `coef`（=ρ/π。その面自身の反射）を掛けて放射輝度に
    # 変換してから `G` と内積を取る（`send_*` へ変換するのと同じ変換をコレクトの直前にも
    # 適用する。3回目・4回目も同様——`irr_a3`/`irr_b3`/`irr_a4`/`irr_b4` はいずれも
    # 「届いた放射照度」であり、コレクトの直前に `coef` を掛けるまでは放射輝度になっていない）。
    irr_b2 = send_a1 @ K             # A→B: 2回目の反射でBに届く放射照度
    irr_a2 = K @ send_b1             # B→A: 2回目の反射でAに届く放射照度
    contrib2 = coef * float(irr_b2 @ vec_gb + irr_a2 @ vec_ga)

    send_b2 = coef * irr_b2 * dA_b
    send_a2 = coef * irr_a2 * dA_a
    irr_a3 = K @ send_b2             # 3回目: B→A
    irr_b3 = send_a2 @ K             # 3回目: A→B
    contrib3 = coef * float(irr_a3 @ vec_ga + irr_b3 @ vec_gb)

    send_a3 = coef * irr_a3 * dA_a
    send_b3 = coef * irr_b3 * dA_b
    irr_b4 = send_a3 @ K             # 4回目: A→B
    irr_a4 = K @ send_b3             # 4回目: B→A
    contrib4 = coef * float(irr_b4 @ vec_gb + irr_a4 @ vec_ga)

    return contrib2 + contrib3 + contrib4


def _corner_interreflection_total(
    centers_sel: np.ndarray, normals_sel: np.ndarray, u_sel: np.ndarray, v_sel: np.ndarray,
    half_u_sel: np.ndarray, half_v_sel: np.ndarray, owner_sel: np.ndarray, near_arr: np.ndarray,
    led: _Emitter, pt: _Emitter, surf: SurfaceSpec,
    led_intensity: float, pt_responsivity: float, max_range_m: float,
    boxes: np.ndarray, occlusion: bool, separation_m: float, half_angle_max_deg: float,
    corner_adjacency_tol_m: float, corner_gate_n_grid: int, corner_n_grid: int,
    corner_gate_threshold: float, both_directions: bool,
    led_cone_margin_deg: float, pt_cone_margin_deg: float,
    cone_filter_n_u: int, cone_filter_n_v: int,
    corner_stats: Optional[Dict] = None,
) -> float:
    """`response_fast()` の射程内・バックフェイスカリング済みの候補面すべてから隅の
    ペアを見つけ、安い判定（粗い格子）→本計算（細かい格子）の2段で合計する
    （モジュール docstring「隅の相互反射」節参照）。

    候補を絞る前に、直接光の絞り込みと同じ光錐判定（`_facets_maybe_in_cone_batch`。
    既定の余裕角 LED±15°・PT±25°は直接光向けの実測に基づく大きな余裕）を再利用して
    候補面自体を間引く: 隅のペア (A,B) が寄与し得るのは「A が LED 光錐に入り得て
    B が PT 視野に入り得る」（A→B）か「B が LED 光錐・A が PT 視野」（B→A）のどちらか
    だけなので、LED 光錐にもPT視野にも入り得ない面はどちらの役にも立てず除外できる
    （教授セッションの実測: 隅の近くの面でも LED 光軸から 4.4° と、既定の余裕角
    15°の中に収まっていた。この絞り込みで安全に候補を減らせる）。
    """
    t0 = time.perf_counter()

    if centers_sel.shape[0] > 0:
        stacked = (centers_sel, normals_sel, u_sel, v_sel, half_u_sel, half_v_sel)
        led_mask = _facets_maybe_in_cone_batch(
            stacked, led.pos, led.axis, led_cone_margin_deg, cone_filter_n_u, cone_filter_n_v,
        )
        pt_mask = _facets_maybe_in_cone_batch(
            stacked, pt.pos, pt.axis, pt_cone_margin_deg, cone_filter_n_u, cone_filter_n_v,
        )
        cand_mask = led_mask | pt_mask
    else:
        cand_mask = np.zeros(0, dtype=bool)

    pairs = _corner_pairs(centers_sel, normals_sel, half_u_sel, owner_sel, near_arr,
                           corner_adjacency_tol_m, cand_mask)
    n_activated = 0
    total = 0.0
    t_gate = 0.0
    t_full = 0.0

    for i, j in pairs:
        facet_a = _Facet(center=centers_sel[i], u=u_sel[i], v=v_sel[i], normal=normals_sel[i],
                          half_u=float(half_u_sel[i]), half_v=float(half_v_sel[i]))
        facet_b = _Facet(center=centers_sel[j], u=u_sel[j], v=v_sel[j], normal=normals_sel[j],
                          half_u=float(half_u_sel[j]), half_v=float(half_v_sel[j]))
        owner_a = int(owner_sel[i])
        owner_b = int(owner_sel[j])

        tg0 = time.perf_counter()
        gate_val = _corner_pair_contribution(
            facet_a, facet_b, owner_a, owner_b, led, pt, surf,
            corner_gate_n_grid, half_angle_max_deg, separation_m,
            led_intensity, pt_responsivity, max_range_m, boxes, occlusion, both_directions,
        )
        t_gate += time.perf_counter() - tg0

        if abs(gate_val) <= corner_gate_threshold:
            continue

        n_activated += 1
        tf0 = time.perf_counter()
        total += _corner_pair_contribution(
            facet_a, facet_b, owner_a, owner_b, led, pt, surf,
            corner_n_grid, half_angle_max_deg, separation_m,
            led_intensity, pt_responsivity, max_range_m, boxes, occlusion, both_directions,
        )
        t_full += time.perf_counter() - tf0

    if corner_stats is not None:
        corner_stats["n_pairs_found"] = len(pairs)
        corner_stats["n_pairs_activated"] = n_activated
        corner_stats["activated"] = n_activated > 0
        corner_stats["time_gate_s"] = t_gate
        corner_stats["time_full_s"] = t_full
        corner_stats["time_total_s"] = time.perf_counter() - t0

    return total


def response_fast(
    sensor: IrSensorSpec,
    pose: PoseLike,
    surfaces: Sequence[Rect],
    surf: SurfaceSpec,
    *,
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M,
    n_grid: int = DEFAULT_N_GRID_FAST,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    led_intensity: float = 1.0,
    pt_responsivity: float = 1.0,
    occlusion: bool = True,
    narrow_facets: bool = True,
    led_cone_margin_deg: float = DEFAULT_LED_CONE_MARGIN_DEG,
    pt_cone_margin_deg: float = DEFAULT_PT_CONE_MARGIN_DEG,
    cone_filter_n_u: int = DEFAULT_CONE_FILTER_N_U,
    cone_filter_n_v: int = DEFAULT_CONE_FILTER_N_V,
    prune_occlusion_boxes: bool = True,
    prune_occlusion_points: bool = True,
    occlusion_point_weight_frac: float = DEFAULT_OCCLUSION_POINT_WEIGHT_FRAC,
    interreflection: bool = True,
    corner_adjacency_tol_m: float = DEFAULT_CORNER_ADJACENCY_TOL_M,
    corner_gate_n_grid: int = DEFAULT_CORNER_GATE_N_GRID,
    corner_n_grid: int = DEFAULT_CORNER_N_GRID,
    corner_gate_threshold: float = DEFAULT_CORNER_GATE_THRESHOLD,
    corner_both_directions: bool = True,
    corner_stats: Optional[Dict] = None,
) -> float:
    """`response()` の高速版（段A: 直接光のみ・反射1回・床なし）。

    設計の骨子はモジュール docstring「高速フォワードモデル（AUDIT_059・段A: 直接光のみ）」
    節・「速さの追加対策」節・「速さの追加対策・その2」節を参照。`response()` の面積分
    （`_facet_grid`/`_integrate_facet`）・遮蔽の一括判定（`_segment_occluded`）はそのまま
    使い回す。積分対象の面を LED 光錐＋PT 視野の両方に入り得るものだけへ絞り込む点、
    射程の足切り・面の展開・バックフェイスカリングを numpy で一括処理する点、遮蔽を
    判定する前に寄与しない求積点を落とす点が `response()` と違う（遮蔽の候補直方体
    そのものは近似的には絞らない＝`near_rects` 全体が母集団。ただし
    `prune_occlusion_boxes`/`prune_occlusion_points` で数学的に安全な削減はする。下記参照）。

    Args:
        sensor/pose/surfaces/surf/wall_height_m/n_grid/max_range_m/led_intensity/
        pt_responsivity/occlusion: `response()` と同じ意味（`include_floor`・`bounces`・
            `return_breakdown`・相互反射関連の引数は無い。床は含めず反射は 1 回だけを
            扱うため）。`n_grid` の既定は `response()`（28）と異なり `DEFAULT_N_GRID_FAST`
            （20）。
        narrow_facets: `False` にすると積分対象の絞り込みを行わず、射程内の可視な面
            すべてを積分する（自己検査用。同じ `n_grid` なら絞り込みありと一致するはず、
            という検査に使う。既定 `True`）。
        led_cone_margin_deg/pt_cone_margin_deg/cone_filter_n_u/cone_filter_n_v:
            絞り込み判定のカットオフ角・標本密度（既定は事前登録の設計値。否定対照で
            意図的に緩めた/崩した値を渡す）。
        prune_occlusion_boxes: `False` にすると遮蔽候補直方体の AABB 削減
            （モジュール docstring「速さの追加対策」節参照）を行わず、`near_rects` 全体を
            遮蔽候補にする（削減前の旧挙動。自己検査2用。既定 `True`）。
        prune_occlusion_points: `False` にすると求積点の足切り（モジュール docstring
            「速さの追加対策・その2」節参照）を行わず、積分対象の全求積点を遮蔽判定に
            かける（削減前の旧挙動。自己検査4用。既定 `True`）。
        occlusion_point_weight_frac: 求積点の足切りの予算（面ごとの寄与見積りの総和に対する
            比率。小さい点から順にこの割合に達するまでを落とす。既定
            `DEFAULT_OCCLUSION_POINT_WEIGHT_FRAC` = 1e-9）。
        interreflection: `True`（既定）で隅の相互反射（段B。モジュール docstring「隅の
            相互反射」節）を足す。`False` にすると段Aだけ（直接光のみ）になり、この
            引数を追加する前の `response_fast()` と厳密に一致する（自己検査5）。
        corner_adjacency_tol_m/corner_gate_n_grid/corner_n_grid/corner_gate_threshold:
            隅の検出・安い判定・本計算の設定（既定は事前登録の設計値。否定対照で
            意図的に崩した値を渡す）。
        corner_both_directions: `False` にすると隅の相互反射を「A→B」の片方向だけにする
            （否定対照用。教授セッションの実測では基準の0.64倍にしかならなかった）。
        corner_stats: 渡すと、隅の判定の作動状況（見つかったペア数・作動したペア数・
            粗い/細かい計算に使った時間）をこの `dict` に書き込む（測定用。既定 `None`
            で何もしない）。

    Returns:
        センサが受ける光量（任意単位・`response()` と同じ規格。`gain` を含む）。
    """
    led, pt = _sensor_world_geometry(sensor, pose)
    half_angle_max = max(sensor.led_half_angle_deg, sensor.pt_half_angle_deg)

    if len(surfaces) == 0:
        return 0.0

    # 速さの追加対策4（モジュール docstring 参照）: 射程の足切り1（response()と同じ判定）・
    # 壁1枚→側面4枚への展開・バックフェイスカリングを、矩形ごとの Python ループではなく
    # numpy でまとめて行う。判定の中身・使う値は `_wall_facets`/response() の元ループと
    # 完全に同じ（`Rect` は `NamedTuple` なので `np.asarray` でそのまま (N,4) 配列にできる）。
    rect_arr = np.asarray(surfaces, dtype=float)   # (N,4) = cx,cy,hx,hy
    diag_all = np.hypot(rect_arr[:, 2], rect_arr[:, 3])
    dist_center_all = np.hypot(rect_arr[:, 0] - led.pos[0], rect_arr[:, 1] - led.pos[1])
    near_mask = (dist_center_all - diag_all) <= max_range_m
    near_arr = rect_arr[near_mask]                  # (M,4)
    M = near_arr.shape[0]
    if M == 0:
        return 0.0

    # 遮蔽の候補直方体（`_obstacle_boxes(near_rects, wall_height_m)` と同じ式。床は含めない）。
    # 🔴 近似的には絞らない: near_arr 全体（response() の near_rects と同じ集合）が母集団
    # （モジュール docstring「遮る面は絞らない」節参照）。AABB による安全な削減は
    # 積分対象を絞り込んだあとに行う（下記）。
    if occlusion:
        cx, cy, hx, hy = near_arr[:, 0], near_arr[:, 1], near_arr[:, 2], near_arr[:, 3]
        all_boxes = np.column_stack([
            cx - hx, cx + hx, cy - hy, cy + hy,
            np.zeros(M), np.full(M, wall_height_m),
        ])
    else:
        all_boxes = np.zeros((0, 6))

    cx, cy, hx, hy = near_arr[:, 0], near_arr[:, 1], near_arr[:, 2], near_arr[:, 3]
    zc = np.full(M, wall_height_m / 2.0)
    # 壁1枚→側面4枚への展開（`_wall_facets` と面の並び順・center/u/v/normal/half_u/half_v の
    # 値がすべて同じ。床は含めない＝ユーザの決定。note_034 追記16）。
    centers4 = np.stack([
        np.stack([cx + hx, cy, zc], axis=-1),
        np.stack([cx - hx, cy, zc], axis=-1),
        np.stack([cx, cy + hy, zc], axis=-1),
        np.stack([cx, cy - hy, zc], axis=-1),
    ], axis=0)                                                    # (4,M,3)
    normals4 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])       # (4,3)
    u_y = np.array([0.0, 1.0, 0.0])
    u_x = np.array([1.0, 0.0, 0.0])
    v_z = np.array([0.0, 0.0, 1.0])
    u_vecs4 = np.array([u_y, u_y, u_x, u_x])                       # (4,3)
    v_vecs4 = np.array([v_z, v_z, v_z, v_z])                       # (4,3)
    half_u4 = np.stack([hy, hy, hx, hx], axis=0)                   # (4,M)
    half_v4 = np.full((4, M), wall_height_m / 2.0)
    owner4 = np.tile(np.arange(M), (4, 1))    # (4,M) 遮蔽候補直方体の番号（near_arr内の通し番号）

    # バックフェイスカリング（LED から見て向こう向きの面を除く。response() と同じ判定）。
    to_led4 = led.pos[None, None, :] - centers4                    # (4,M,3)
    vis_mask4 = np.einsum("fmk,fk->fm", to_led4, normals4) > 0.0    # (4,M)

    f_sel, m_sel = np.nonzero(vis_mask4)
    if f_sel.size == 0:
        return 0.0

    centers_sel = centers4[f_sel, m_sel]      # (K,3)
    normals_sel = normals4[f_sel]              # (K,3)
    u_sel = u_vecs4[f_sel]                     # (K,3)
    v_sel = v_vecs4[f_sel]                     # (K,3)
    half_u_sel = half_u4[f_sel, m_sel]         # (K,)
    half_v_sel = half_v4[f_sel, m_sel]         # (K,)
    owner_sel = owner4[f_sel, m_sel]           # (K,)

    if narrow_facets:
        # 積分対象の絞り込み: LED光錐 かつ PT視野 の両方に入り得る面だけを残す
        # （保守的な判定。モジュール docstring 参照）。`_Facet` に包まず、抽出済みの配列を
        # そのまま `_facets_maybe_in_cone_batch` に渡す（判定の中身は変えない）。
        stacked = (centers_sel, normals_sel, u_sel, v_sel, half_u_sel, half_v_sel)
        led_mask = _facets_maybe_in_cone_batch(
            stacked, led.pos, led.axis, led_cone_margin_deg, cone_filter_n_u, cone_filter_n_v,
        )
        pt_mask = _facets_maybe_in_cone_batch(
            stacked, pt.pos, pt.axis, pt_cone_margin_deg, cone_filter_n_u, cone_filter_n_v,
        )
        keep = np.nonzero(led_mask & pt_mask)[0]
    else:
        keep = np.arange(centers_sel.shape[0])

    # 🔴 keep.size==0（絞り込みで直接光の積分対象がゼロ）でもここで打ち切らない。
    # `interreflection=True` のとき、隅のペア探索は `narrow_facets` の絞り込み前の
    # `centers_sel` 等（このすぐ下、直接光の絞り込みには影響されない）を使うため、
    # 直接光がゼロでも隅の間接光だけが残る姿勢がありうる（モジュール docstring
    # 「隅の相互反射」節）。そのため直接光パートは `keep.size>0` のときだけ実行し、
    # 段Bは常に（`interreflection=True` なら）評価する。
    vis_facets: list = []
    vis_owner: list = []
    points_list: list = []
    dA_list: list = []
    occluded_list: list = []

    if keep.size > 0:
        vis_facets = [
            _Facet(center=centers_sel[i], u=u_sel[i], v=v_sel[i], normal=normals_sel[i],
                   half_u=float(half_u_sel[i]), half_v=float(half_v_sel[i]))
            for i in keep
        ]
        vis_owner = [int(owner_sel[i]) for i in keep]

        grids = [_facet_grid(f, led, n_grid, half_angle_max, sensor.separation_m) for f in vis_facets]
        points_list = [g[0] for g in grids]
        dA_list = [g[1] for g in grids]

    if keep.size > 0 and occlusion and all_boxes.shape[0] > 0:
        all_points = np.stack(points_list, axis=0)
        owner_arr = np.array(vis_owner, dtype=int)

        if prune_occlusion_boxes:
            # 🔴 安全な削減（絞り込みではない）: {LED,PT} ∪ {積分対象の求積点すべて} を
            # 包む AABB と足跡が交わらない直方体は、どの遮蔽判定用の線分とも交わり得ない
            # （線分の両端が凸な AABB の内部にあるので線分全体が AABB に収まるため）。
            # モジュール docstring「速さの追加対策」節参照。自己検査2で答えが変わらない
            # ことを確認済み。
            pts_flat = all_points.reshape(-1, 3)
            combined = np.vstack([pts_flat, led.pos[None, :], pt.pos[None, :]])
            aabb_min = combined.min(axis=0)
            aabb_max = combined.max(axis=0)
            box_mins = all_boxes[:, (0, 2, 4)]
            box_maxs = all_boxes[:, (1, 3, 5)]
            keep_box = np.all((box_maxs >= aabb_min) & (box_mins <= aabb_max), axis=1)
            kept_idx = np.nonzero(keep_box)[0]
            boxes = all_boxes[kept_idx]
            # 面の owner 番号（near_arr の通し番号）を、削減後の boxes 配列の番号へ張り替える。
            # 削減されなかった箱の番号は必ず見つかる（積分対象の面はその owner 箱の表面上に
            # 求積点を持つので、その箱は AABB と必ず交わる。上のコメント参照）。
            remap = -np.ones(M, dtype=int)
            remap[kept_idx] = np.arange(len(kept_idx))
            owner_arr = remap[owner_arr]
        else:
            boxes = all_boxes

        if boxes.shape[0] > 0:
            if prune_occlusion_points:
                # 速さの追加対策3（モジュール docstring「速さの追加対策・その2」節参照）:
                # 遮蔽を判定する前に、遮蔽なしで安く計算できる量（LED側の重み
                # cos^m(θ_e)/r_e² と PT側の重み cos^m(θ_r)/r_v²。ともに `_integrate_facet`
                # の `irradiance`/`pt_sensitivity・1/r_v²` と同じ量）の積に面積要素 `dA` を
                # 掛けた「遮蔽なし・cos_i/cos_v なしの寄与見積り」（cos_i・cos_v は 1 以下・
                # `diffuse/π+specular` は定数上限があるので、この見積りは実際の寄与の
                # 定数倍の上界になる）を使い、面ごとに**その総和のうち小さい方から
                # `occlusion_point_weight_frac` 倍までを占める点だけ**を、遮蔽を判定せず
                # 寄与ゼロ（＝遮蔽扱い）にする。
                #
                # 🔴 単純な「面の最大値の frac 倍未満は落とす」という点ごとの絶対しきい値では
                # 不十分だった（実測で発覚）: (a) LED側の重みだけで判定すると、LED と PT の
                # 光軸が離隔ぶんだけずれているため、近距離では両者の明部が面上でずれて
                # 重ならないことがあり（モジュール docstring「数値積分について」節）、
                # 「LED からは暗いが PT からは非常に近い」点を誤って落として自己検査4で
                # 満量比 1.5e-6 の差が出た。(b) LED×PT の積で判定しても、tan ワープ格子は
                # 遠方の点ほど1点あたりの面積要素 `dA` が大きく（密度で重みを補っている）、
                # 「個々には僅かでも `dA` が大きい点が大量にある」ケースで足切りの総量が
                # 積み上がり、満量比 7.5e-7 の差が残った。**個々の点でなく、面ごとの
                # 総和に対する累積の取りこぼし量**で予算管理することでこれを防ぐ
                # （小さい点から順に足していき、面の総和の `occlusion_point_weight_frac`
                # 倍を超えない範囲でだけ落とす）。
                d_e_all = all_points - led.pos
                r_e_all = np.linalg.norm(d_e_all, axis=-1)
                r_e_all_safe = np.maximum(r_e_all, 1e-6)
                dir_e_all = d_e_all / r_e_all_safe[..., None]
                cos_e_all = np.clip(np.einsum("aijk,k->aij", dir_e_all, led.axis), 0.0, 1.0)
                weight_led_all = cos_e_all ** led.m / (r_e_all_safe ** 2)

                d_v_all = pt.pos - all_points
                r_v_all = np.linalg.norm(d_v_all, axis=-1)
                r_v_all_safe = np.maximum(r_v_all, 1e-6)
                dir_v_all = d_v_all / r_v_all_safe[..., None]
                cos_r_all = np.clip(np.einsum("aijk,k->aij", -dir_v_all, pt.axis), 0.0, 1.0)
                weight_pt_all = cos_r_all ** pt.m / (r_v_all_safe ** 2)

                dA_all = np.stack(dA_list, axis=0)   # (n_vis, n_grid, n_grid)
                flux_all = weight_led_all * weight_pt_all * dA_all

                n_vis_local = flux_all.shape[0]
                flat_flux = flux_all.reshape(n_vis_local, -1)
                order = np.argsort(flat_flux, axis=1)               # 小さい順
                sorted_flux = np.take_along_axis(flat_flux, order, axis=1)
                cumsum = np.cumsum(sorted_flux, axis=1)
                facet_budget = cumsum[:, -1:] * occlusion_point_weight_frac
                drop_sorted = cumsum <= facet_budget                 # 累積が予算内の間だけ落とす
                keep_sorted = ~drop_sorted
                keep_flat = np.empty_like(keep_sorted)
                np.put_along_axis(keep_flat, order, keep_sorted, axis=1)
                keep_pts = keep_flat.reshape(all_points.shape[:-1])

                fi, ui, vi = np.nonzero(keep_pts)
                pts_flat = all_points[fi, ui, vi]
                owner_flat = owner_arr[fi]

                occluded_all = np.ones(all_points.shape[:-1], dtype=bool)   # 既定=足切り(寄与ゼロ)
                if pts_flat.shape[0] > 0:
                    occ_led_flat = _segment_occluded(pts_flat, led.pos, boxes, owner_flat)
                    occ_pt_flat = _segment_occluded(pts_flat, pt.pos, boxes, owner_flat)
                    occluded_all[fi, ui, vi] = occ_led_flat | occ_pt_flat
            else:
                occ_led = _segment_occluded(all_points, led.pos, boxes, owner_arr)
                occ_pt = _segment_occluded(all_points, pt.pos, boxes, owner_arr)
                occluded_all = occ_led | occ_pt
            occluded_list = [occluded_all[i] for i in range(len(vis_facets))]
        else:
            occluded_list = [None] * len(vis_facets)
    else:
        occluded_list = [None] * len(vis_facets)

    total = 0.0
    for facet, points, dA, occluded in zip(vis_facets, points_list, dA_list, occluded_list):
        total += _integrate_facet(
            facet, led, pt, surf, points, dA, led_intensity, pt_responsivity,
            max_range_m, occluded,
        )

    if interreflection:
        # 段B（隅の相互反射。モジュール docstring「隅の相互反射」節）: `narrow_facets`
        # の絞り込み前の候補面全体（`centers_sel` 等）から隅を探す。直接光の積分対象
        # （`vis_facets`）とは独立な処理なので、`narrow_facets`/`prune_occlusion_*` の
        # 設定には左右されない。遮蔽候補は `all_boxes`（削減前。母集団は近傍矩形すべて）
        # をそのまま使う（隅は作動する姿勢が少数なので、削減の手間を省いても速さへの
        # 影響は小さい）。
        total += _corner_interreflection_total(
            centers_sel, normals_sel, u_sel, v_sel, half_u_sel, half_v_sel, owner_sel, near_arr,
            led, pt, surf, led_intensity, pt_responsivity, max_range_m,
            all_boxes, occlusion, sensor.separation_m, half_angle_max,
            corner_adjacency_tol_m, corner_gate_n_grid, corner_n_grid,
            corner_gate_threshold, corner_both_directions,
            led_cone_margin_deg, pt_cone_margin_deg, cone_filter_n_u, cone_filter_n_v,
            corner_stats,
        )

    return total * sensor.gain


# ============================================================================
# 表（距離 × 入射角 × 横ずれ）
# ============================================================================
@dataclass
class ResponseTable:
    """距離・入射角・横ずれ に対する応答の表。モデル由来でも実測由来でも同じ形式。"""

    distances_m: np.ndarray
    incidence_deg: np.ndarray
    lateral_m: np.ndarray
    values: np.ndarray          # shape (len(distances_m), len(incidence_deg), len(lateral_m))
    source: str                 # "model" / "measured"
    meta: Dict                  # 生成に使ったパラメータ・日付など


def _solve_pose_for_sensor_target(
    sensor: IrSensorSpec, target_xy: Tuple[float, float], theta: float
) -> Tuple[float, float, float]:
    """センサの取付基準点（`sensor.pos`）が世界座標 `target_xy` に来て、機体の向きが
    `theta` になるような機体姿勢 `(x, y, theta)` を逆算する。"""
    bx, by, _ = sensor.pos
    c, s = math.cos(theta), math.sin(theta)
    rx = bx * c - by * s
    ry = bx * s + by * c
    return target_xy[0] - rx, target_xy[1] - ry, theta


def build_table_from_model(
    sensor: IrSensorSpec,
    surf: SurfaceSpec,
    *,
    distances_m: Optional[np.ndarray] = None,
    incidence_deg: Optional[np.ndarray] = None,
    lateral_m: Optional[np.ndarray] = None,
    wall_half_length_m: float = 0.09,
    wall_thickness_m: float = 0.012,
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M,
    n_grid: int = DEFAULT_N_GRID,
    **response_kwargs,
) -> ResponseTable:
    """平らな壁 1 枚に対する応答を (距離, 入射角, 横ずれ) の格子で計算し、表にする。

    壁は世界座標 x=0 の面（法線 -x 側、センサは x<0 側から見る）に固定し、`distances_m` は
    センサ基準点（`sensor.pos`）から壁面までの垂直距離、`incidence_deg` は機体の向き
    （＝壁法線からの光軸のずれ）、`lateral_m` はセンサ基準点の壁沿い方向（y）の位置。
    `**response_kwargs` は `response()` にそのまま渡す（`include_floor` など）。
    """
    if distances_m is None:
        distances_m = np.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035,
                                 0.040, 0.050, 0.060, 0.084, 0.120, 0.180, 0.250])
    if incidence_deg is None:
        incidence_deg = np.array([0.0, 15.0, 30.0, 45.0])
    if lateral_m is None:
        lateral_m = np.array([0.0])

    distances_m = np.asarray(distances_m, dtype=float)
    incidence_deg = np.asarray(incidence_deg, dtype=float)
    lateral_m = np.asarray(lateral_m, dtype=float)

    values = np.zeros((len(distances_m), len(incidence_deg), len(lateral_m)))

    x_face = 0.0
    wall_cx = x_face + wall_thickness_m / 2.0
    wall = Rect(cx=wall_cx, cy=0.0, hx=wall_thickness_m / 2.0, hy=wall_half_length_m)
    az_body = math.atan2(sensor.axis[1], sensor.axis[0])

    for i, d in enumerate(distances_m):
        for j, inc in enumerate(incidence_deg):
            theta = math.radians(inc) - az_body
            for k, lat in enumerate(lateral_m):
                target = (x_face - float(d), float(lat))
                pose = _solve_pose_for_sensor_target(sensor, target, theta)
                values[i, j, k] = response(
                    sensor, pose, [wall], surf,
                    wall_height_m=wall_height_m, n_grid=n_grid, **response_kwargs,
                )

    meta = {
        "wall_half_length_m": wall_half_length_m,
        "wall_thickness_m": wall_thickness_m,
        "wall_height_m": wall_height_m,
        "n_grid": n_grid,
        "sensor_name": sensor.name,
        "sensor_layout": sensor.layout,
        "surf_diffuse": surf.diffuse,
        "surf_specular": surf.specular,
        "surf_shininess": surf.shininess,
        "response_kwargs": {k: v for k, v in response_kwargs.items()},
    }
    return ResponseTable(
        distances_m=distances_m, incidence_deg=incidence_deg, lateral_m=lateral_m,
        values=values, source="model", meta=meta,
    )


def save_table(t: ResponseTable, path) -> None:
    """表を npz 形式で保存する（`source`/`meta` を必ず含める）。"""
    np.savez(
        str(path),
        distances_m=t.distances_m, incidence_deg=t.incidence_deg, lateral_m=t.lateral_m,
        values=t.values, source=np.array(t.source), meta_json=np.array(json.dumps(t.meta)),
    )


def load_table(path) -> ResponseTable:
    """`save_table()` で保存した表を読み直す。実測から作った表もこの形式に合わせれば読める。"""
    data = np.load(str(path), allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    return ResponseTable(
        distances_m=data["distances_m"], incidence_deg=data["incidence_deg"],
        lateral_m=data["lateral_m"], values=data["values"],
        source=str(data["source"]), meta=meta,
    )


def _interp_axis(value: float, axis: np.ndarray) -> Tuple[int, int, float]:
    """`value` を `axis` 上の 2 点で挟むインデックスと重みを返す（範囲外はクランプ）。"""
    n = len(axis)
    if n == 1:
        return 0, 0, 0.0
    if value <= axis[0]:
        return 0, 1, 0.0
    if value >= axis[-1]:
        return n - 2, n - 1, 1.0
    idx = int(np.searchsorted(axis, value)) - 1
    idx = max(0, min(n - 2, idx))
    lo, hi = axis[idx], axis[idx + 1]
    w = 0.0 if hi == lo else (value - lo) / (hi - lo)
    return idx, idx + 1, float(w)


def lookup(t: ResponseTable, distance: float, incidence_deg: float, lateral: float) -> float:
    """表を (距離, 入射角, 横ずれ) で線形補間して引く。範囲外は端の値にクランプする
    （外挿はしない。境界を超えても壊れた値を返さないことを検査で確認している）。"""
    di0, di1, wd = _interp_axis(distance, t.distances_m)
    ii0, ii1, wi = _interp_axis(incidence_deg, t.incidence_deg)
    li0, li1, wl = _interp_axis(lateral, t.lateral_m)

    def v(i: int, j: int, k: int) -> float:
        return float(t.values[i, j, k])

    c00 = v(di0, ii0, li0) * (1 - wd) + v(di1, ii0, li0) * wd
    c10 = v(di0, ii1, li0) * (1 - wd) + v(di1, ii1, li0) * wd
    c01 = v(di0, ii0, li1) * (1 - wd) + v(di1, ii0, li1) * wd
    c11 = v(di0, ii1, li1) * (1 - wd) + v(di1, ii1, li1) * wd

    c0 = c00 * (1 - wi) + c10 * wi
    c1 = c01 * (1 - wi) + c11 * wi

    return c0 * (1 - wl) + c1 * wl


# ============================================================================
# 高速フォワードモデル（光線による可視面判定 ＋ 表引き）
# ============================================================================
"""
背景: `response()` の数値積分は実迷路で 1 本あたり 29〜35ms かかり、探索・学習の実行時間に
そのまま乗る（`research_notes/note_034_ir_sensor_model.md` 追記分「表を作る」節）。

**壁ごとに独立な表引きを機械的に足すのは誤り**（教授セッションの指摘、2026-08-21）。
表の値 `lookup(wall_table, d, θ, L)` は「その壁だけが単独で存在する」前提の値であり、
手前の壁が奥の壁を隠しているかどうかを知らない。足すと隠れているはずの壁の寄与が
戻ってきてしまう（実測: 手前84mm・奥264mmの2枚壁で、正しい値0.4635に対し単純な和は
0.5134、+10.8%の過大評価）。

## 光線による可視面判定（最初の実装）とその限界

最初の実装は、狭いビーム（半値角5°）であることを使い、センサから光線を数本（中心1本＋
リング状に数本）飛ばして最初に当たる面だけ表を引く、という方式だった。手前の壁が奥の壁を
隠す場合は正しく直ったが、**別の系統的な過小評価が残った**（教授セッションが実測・診断、
2026-08-21）: 応答が真値の1/10未満になる「弱い信号」の姿勢だけ相対誤差が中央値95%に達し、
光線を9本→97本に増やしても直らなかった。原因は、弱い信号はビームの中心から外れた面が
裾野（`cos^m` の裾）だけで作る応答であり、**どの光線も、その面をピンポイントで貫くことは
ほぼ無い**ため（本数を増やしても「当たるか外れるか」の二値判定である限り、当たる確率が
上がるだけで期待値は改善しない）。

## 対策: 光線を飛ばすのをやめ、近傍の面をすべて解析的に列挙する

**表の値そのものは元から裾野を正しく持っている**（`build_table_from_model` は実際の放射
計算をそのまま実行するので、入射角がどれだけ大きくても、その面から来る光を正しく積分
している。表を細かく持つ限り、`lookup()` は裾野も含めて正確に返す）。問題は「光線が
当たった面だけ表を引く」という**面の発見の仕方**の方にあった。

そこで、光線を飛ばす代わりに、`max_range_m` 以内にある壁・柱の矩形（`near_rects`）
**すべて**について、その4辺のうちセンサ側を向いている辺（バックフェイスカリング。
`_wall_facets` と同じ判定）を候補にし、辺ごとに (distance, incidence_deg, lateral) を
解析的に計算する（三角関数だけ。光線も求積もいらない）。**遮蔽は、辺ごとに代表点
（センサの垂線の足を辺の実際の長さにクランプした点）を取り、その点が LED から
（`dual_origin=True` なら PT からも）見えるかを `_segment_occluded`（スラブ法。
既存の遮蔽判定と同じ関数）で判定する**ことで両立させる。自分自身が属する矩形は
`skip_idx` で除外する（`response()` の自己遮蔽除外と同じ考え方）。

  - 見えている壁の広い面 → 壁専用の表 `wall_table`（`build_wall_table()`。床を含まない）
  - 見えている柱・壁の妻面（軸平行2辺のうち短い方） → 柱専用の表 `post_table`
    （`build_post_table()`。反射面の半長を柱の実寸6mmにした表。当初は柱自身の反射を
    無視する近似だったが、柱がピーク距離付近にあり他に光る面が無い姿勢で2桁以上
    過小評価する例が見つかったため追加した。`post_table` を渡さない場合は寄与ゼロの
    まま＝後方互換の既定）
  - 遮られている面 → 0（重み付けはしない。見えているか否かの二値。「裾の重みを掛けて
    足す」という案も検討したが、表の値自体が入射角・横ずれを通じて既に裾を正しく
    含んでいるため、追加の重みを掛けると二重に減衰させてしまう。実測でも重み無しの
    ほうが精度が良かった）
  - 床 → 代表点（センサ直下を中心に半径 `floor_sample_radius_m` のリング `floor_sample_ring`
    点＋中心1点）ごとに LED・PT からの可視判定をし、**見えている点の割合**を
    `floor_value`（`floor_baseline()`。壁が無いときの床だけの応答。ほぼ姿勢に依らない
    定数として扱う）に掛けて足す（床は「面」の単位を持たないので、辺の列挙ではなく
    点の標本化で近似する）

複数の壁が同時に見える場合の合成は「見えている面の表の値をそのまま足す」（重み無し）。
床の二重計上は「床は専用の点標本で別扱い」なので構造的に起きない（`fast_response()`）。
"""


def _default_table_distances_m() -> np.ndarray:
    """距離軸: ピーク付近（40mm前後）を密に、近傍・遠方を粗くする（約90点）。"""
    near = np.arange(0.003, 0.020, 0.002)     # 3〜19mm、2mm刻み
    peak = np.arange(0.020, 0.081, 0.001)     # 20〜80mm、1mm刻み（ピークを密に）
    mid = np.arange(0.085, 0.151, 0.005)      # 85〜150mm、5mm刻み
    far = np.arange(0.160, 0.301, 0.020)      # 160〜300mm、20mm刻み
    return np.unique(np.concatenate([near, peak, mid, far]))


def _default_table_incidence_deg() -> np.ndarray:
    """入射角軸: ±69°までは3°刻み（主要な範囲）、それを超えるすれすれの範囲は4°刻みで粗く。

    実測して分かったこと（2026-08-21、実迷路200姿勢での精度検査で発覚）: 入射角70〜90°の
    応答は cos^m のような急な打ち切りではなく、緩やかに（80°で69°の約1/5、89°で約1/300）
    減衰するだけで、無視できるほど小さくはならない。**当初 ±69° までしか表に持たず、
    それを超える分は端の値へクランプしていたところ、実迷路の側方センサ（LS/RS）や
    通路の折れ角で入射角80°前後の姿勢が現実に多数出現し、真値の5倍前後を返す誤りが
    生じた**（`fast_response` の精度検査 `test_table_matches_direct_integration` の
    デバッグで発見。中央値27%・最大2700倍という誤差の主因だった）。刻みは主要な範囲より
    粗いままでよい（この範囲の値は元々小さく、多少粗くても表全体の誤差に効きにくい）。
    """
    main = np.arange(-69.0, 69.1, 3.0)            # -69〜69°、3°刻み（主要な範囲）
    grazing = np.arange(72.0, 89.1, 4.0)          # 72,76,80,84,88°（すれすれの範囲。粗い）
    return np.unique(np.concatenate([main, -grazing, grazing]))


def _default_table_lateral_m() -> np.ndarray:
    """横ずれ軸: パネル本体の範囲（±84mm）は2mm刻みで密に、そこを外れた「パネルの
    実際の縁より外」の範囲は10mm刻みで粗く、±300mm（`DEFAULT_MAX_RANGE_M`と同じ
    桁）まで延ばす。

    **経緯（2026-08-21・精度検査で発覚）**: 横ずれがパネルの実際の縁（±84mm）を
    外れても、応答は急には0にならない。しかも「どこで0に近づくか」は距離・入射角の
    組み合わせで大きく動く（実測: ある組では120mmで已に無視できるほど小さいのに、
    別の組では260mm・424mmまで無視できない値が残る — LED・PTそれぞれの狭い光錐が、
    入射角の効果でたまたま噛み合う「山」が横ずれ方向にも立つため。`build_post_table`
    の docstring 参照）。表を±84mmで打ち切ってクランプすると、パネルの縁のすぐ外に
    ある山を安全側（過大評価）に倒すのではなく無視できない過大評価（実測+10万%超）を
    生むことが分かったので、範囲を広げた。±300mmを超える分は `_lookup_or_zero` で
    寄与ゼロにする（`max_range_m` の既定 0.35m と同程度の距離ではもう対象外という
    判断。それでも一部の組み合わせでは範囲外に山が残り得るが、実測できた誤差の
    範囲では許容内に収まった）。
    """
    near = np.linspace(-0.084, 0.084, 85)              # ±84mm、2mm刻み（パネル本体）
    far_pos = np.arange(0.094, 0.301, 0.010)            # 94〜300mm、10mm刻み
    far_neg = -far_pos
    return np.unique(np.concatenate([near, far_neg, far_pos]))


def build_wall_table(
    sensor: IrSensorSpec,
    surf: SurfaceSpec,
    *,
    distances_m: Optional[np.ndarray] = None,
    incidence_deg: Optional[np.ndarray] = None,
    lateral_m: Optional[np.ndarray] = None,
    wall_half_length_m: float = DEFAULT_WALL_HALF_LENGTH_M,
    **kwargs,
) -> ResponseTable:
    """高速フォワードモデル用の壁表を作る（床を含まない・入射角は符号付き）。

    `response(d,θ,L) = response(d,-θ,-L)` という対称性は距離・入射角・横ずれを
    **同時に**反転させたときだけ成り立つため（`incidence_deg` だけ非負にして
    `lateral_m` の符号で代用することはできない）、`incidence_deg` は符号付きで持つ。

    既定の `sensor.gain` は表には焼き込まない設計にすること（`fast_response()` 側で
    実機ごとの `gain` を最後に 1 回だけ掛ける。表を作る側の `sensor.gain` は
    1.0 にしておくこと。二重に掛かるのを避けるため）。
    """
    if distances_m is None:
        distances_m = _default_table_distances_m()
    if incidence_deg is None:
        incidence_deg = _default_table_incidence_deg()
    if lateral_m is None:
        lateral_m = _default_table_lateral_m()
    kwargs.setdefault("include_floor", False)
    return build_table_from_model(
        sensor, surf,
        distances_m=distances_m, incidence_deg=incidence_deg, lateral_m=lateral_m,
        wall_half_length_m=wall_half_length_m, **kwargs,
    )


def build_post_table(
    sensor: IrSensorSpec,
    surf: SurfaceSpec,
    *,
    distances_m: Optional[np.ndarray] = None,
    incidence_deg: Optional[np.ndarray] = None,
    lateral_m: Optional[np.ndarray] = None,
    post_half_length_m: float = DEFAULT_POST_HALF_LENGTH_M,
    **kwargs,
) -> ResponseTable:
    """柱（12mm角）自身の反射のための表（`build_wall_table` と同じ形式・同じ関数を使うが、
    反射面の半長を柱の実寸 `post_half_length_m`=6mm にする）。

    **経緯（2026-08-21・精度検査で発覚）**: 当初「柱自身の反射は無視する」近似（占有判定
    にだけ使う）を採ったが、実迷路200姿勢での精度検査で、柱がピーク距離（40mm前後）の
    近くにあるとき、柱1本の反射だけで応答の大半を占める姿勢が現実に多数あり（柱に隠れて
    奥の壁が見えず、他に光る面が無い場面）、無視すると真値を2桁以上下回る誤りが生じた。
    壁表をそのまま使う（`wall_half_length_m`を変えない）と逆に過大評価になる（柱は壁の
    1/14の幅しかない反射面なので、壁ぶんの広い面を仮定した表を引くと集める光の量を
    過大に見積もる）。柱専用の小さな表を別に持つのが正しい。

    横ずれ軸の範囲について（同じく精度検査で発覚）: 柱は小さいので「正対（横ずれ0）が
    山で、外れると単調に減る」と当初見積もっていたが誤りだった。**入射角が大きいとき、
    横ずれ0付近の値はほぼ0で、横ずれ30〜50mm付近（柱の実寸12mmの3〜4倍も外れた位置）
    に別の山が立つ**（LED・PTそれぞれ半値角5°の狭い光錐が、入射角の効果で横ずれのある
    位置でたまたま柱の上で重なるため）。既定の横ずれ軸を壁表と同じ ±84mm に広げて
    この山を含めてある（±30mmでは山を丸ごと落として実質1桁過小評価していた）。
    """
    if distances_m is None:
        distances_m = _default_table_distances_m()
    if incidence_deg is None:
        incidence_deg = _default_table_incidence_deg()
    if lateral_m is None:
        lateral_m = _default_table_lateral_m()
    kwargs.setdefault("include_floor", False)
    return build_table_from_model(
        sensor, surf,
        distances_m=distances_m, incidence_deg=incidence_deg, lateral_m=lateral_m,
        wall_half_length_m=post_half_length_m, **kwargs,
    )


def floor_baseline(sensor: IrSensorSpec, surf: SurfaceSpec, **kwargs) -> float:
    """壁が無いときの床だけの応答（`note_034`: 距離に依らずほぼ一定。定数として扱う）。"""
    kwargs.setdefault("include_floor", True)
    return response(sensor, (0.0, 0.0, 0.0), [], surf, **kwargs)


def _rect_candidate_faces(rect: Rect) -> Tuple[Tuple[np.ndarray, np.ndarray, float], ...]:
    """矩形 `rect` の4辺すべてを (外向き法線, 面中心, 面内方向の半長) として返す
    （バックフェイスカリング前。呼び出し側で「センサ側を向いている辺」だけに絞る。
    `_wall_facets` と同じ向き付け規約）。"""
    return (
        (np.array([1.0, 0.0]), np.array([rect.cx + rect.hx, rect.cy]), rect.hy),
        (np.array([-1.0, 0.0]), np.array([rect.cx - rect.hx, rect.cy]), rect.hy),
        (np.array([0.0, 1.0]), np.array([rect.cx, rect.cy + rect.hy]), rect.hx),
        (np.array([0.0, -1.0]), np.array([rect.cx, rect.cy - rect.hy]), rect.hx),
    )


def _face_dtl(
    sensor_xy: np.ndarray, axis_xy_hat: np.ndarray, face_center: np.ndarray, n_hat: np.ndarray,
) -> Tuple[float, float, float]:
    """面（外向き法線 `n_hat`・中心 `face_center`）について、表の規約
    （`build_table_from_model` と同じ座標系）に合わせた (distance, incidence_deg, lateral)
    を解析的に計算する（三角関数だけ。光線も求積もいらない）。

    `u_hat` は `n_hat` を -90° 回した向き（横ずれの正方向。`build_table_from_model` の
    正準配置（壁法線 -x・横ずれ=センサの y 座標）に一致するよう選んである）。
    """
    u_hat = np.array([n_hat[1], -n_hat[0]])
    rel = sensor_xy - face_center
    distance = float(np.dot(rel, n_hat))
    lateral = float(np.dot(rel, u_hat))

    neg_n = -n_hat
    cross_z = neg_n[0] * axis_xy_hat[1] - neg_n[1] * axis_xy_hat[0]
    dot_z = neg_n[0] * axis_xy_hat[0] + neg_n[1] * axis_xy_hat[1]
    incidence_deg = math.degrees(math.atan2(cross_z, dot_z))
    return distance, incidence_deg, lateral


def _lookup_or_zero(t: ResponseTable, distance: float, incidence_deg: float, lateral: float) -> float:
    """`lookup()` は範囲外をクランプする（表の縁の値をそのまま返す）が、横ずれについては
    それが大きく間違うことが分かった（2026-08-21・精度検査で発覚）。壁パネルは有限長
    （半長 `DEFAULT_WALL_HALF_LENGTH_M`=84mm）なので、横ずれがパネルの実際の縁から
    大きく外れた配置では、真の応答はパネルの縁付近の値ではなく急激にゼロへ落ちる
    （実測: 84mmでは0.678、120mmでは0.119、140mmでは2.7e-7 — 縁の値をクランプで
    流用すると桁違いの過大評価になる）。横ずれが表の格子の外に出た場合はクランプせず
    寄与ゼロを返す（縁のすぐ外側のなだらかな減衰を多少過小評価する近似だが、
    クランプによる桁違いの過大評価より遥かに小さい誤差で済む。距離・入射角は
    このような鋭い崖が無い＝クランプのままでよいことを確認済み）。
    """
    lat_max = float(t.lateral_m[-1])
    lat_min = float(t.lateral_m[0])
    if lateral > lat_max or lateral < lat_min:
        return 0.0
    return lookup(t, distance, incidence_deg, lateral)


def _floor_sample_points(center_xy: np.ndarray, radius_m: float, n_ring: int) -> np.ndarray:
    """床パッチの可視判定に使う代表点（中心1点＋半径 `radius_m` のリング `n_ring` 点、
    いずれも z=0）。床は「面」の単位を持たないので、辺の列挙ではなく点の標本化で
    可視・不可視を近似する（モジュール docstring 参照）。"""
    pts = [[center_xy[0], center_xy[1], 0.0]]
    for k in range(n_ring):
        phi = 2.0 * math.pi * k / n_ring
        pts.append([center_xy[0] + radius_m * math.cos(phi),
                    center_xy[1] + radius_m * math.sin(phi), 0.0])
    return np.array(pts)


def _fast_response_breakdown(
    sensor: IrSensorSpec,
    pose: PoseLike,
    surfaces: Sequence[Rect],
    wall_table: ResponseTable,
    floor_value: float,
    *,
    post_table: Optional[ResponseTable] = None,
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M,
    include_floor: bool = True,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    dual_origin: bool = True,
    post_half_extent_threshold_m: float = DEFAULT_POST_HALF_EXTENT_THRESHOLD_M,
    floor_sample_radius_m: float = DEFAULT_FLOOR_SAMPLE_RADIUS_M,
    floor_sample_ring: int = DEFAULT_FLOOR_SAMPLE_RING,
    n_occlusion_samples: int = DEFAULT_N_OCCLUSION_SAMPLES,
) -> Tuple[float, Dict[int, float], list]:
    """`response()` の高速版。数値積分をせず、`wall_table`/`post_table`/`floor_value`
    （`build_wall_table()`/`build_post_table()`/`floor_baseline()` で事前に作ったもの）を
    解析的な幾何計算と表引きだけで姿勢から応答を推定する。
    モジュール docstring の「高速フォワードモデル」節を参照。

    Args:
        sensor/pose/surfaces: `response()` と同じ（`surfaces` は迷路全体の壁・柱でよい。
            `max_range_m` より遠いものは内部で足切りする）。
        wall_table: `build_wall_table()` で作った表（床を含まない・入射角は符号付き）。
        floor_value: `floor_baseline()` で作った床の基準値。
        post_table: `build_post_table()` で作った柱専用の表（省略時は柱・壁の妻面の
            寄与をゼロにする近似。精度検査で分かったとおり、柱がピーク距離付近に
            あるとき無視すると2桁以上過小評価しうるので、本番では渡すこと）。
        dual_origin: True（既定）なら、各面の代表点が LED からだけでなく PT からも
            見えるかを追加検査し、どちらかから遮られていれば寄与させない
            （LED と PT は離隔 `separation_m` ぶん別の点にあり、光軸を厳密には
            一致させられないという物理的制約 — `note_034` の出発点）。
        floor_sample_radius_m/floor_sample_ring: 床の可視判定に使う代表点（中心＋リング）
            の半径・本数。
        n_occlusion_samples: 壁・柱の各辺の可視判定に使う標本点の数（辺の実際の長さに
            沿って等間隔。隣接する面どうしが互いを部分的に隠す場面の近似精度に効く）。

    Returns:
        `(total, best_per_owner, near_rects)`。`total` は `response()` と同じ任意単位の
        推定値（床を含む・`sensor.gain` を掛けた最終値）。`best_per_owner` は壁・柱の
        矩形番号（`near_rects` の添字）ごとの `(寄与, 距離)`（床を含まない・
        `sensor.gain` を掛ける前）。`fast_response()` は `total` だけを返す薄い
        ラッパーで、`fast_response_or_direct()` は `best_per_owner` を使って
        「精度が粗くなる姿勢」を判定する。
    """
    led, pt = _sensor_world_geometry(sensor, pose)

    near_rects: list = []
    for r in surfaces:
        diag = math.hypot(r.hx, r.hy)
        dist_center = math.hypot(r.cx - led.pos[0], r.cy - led.pos[1])
        if dist_center - diag > max_range_m:
            continue
        near_rects.append(r)
    boxes = _obstacle_boxes(near_rects, wall_height_m)

    axis_xy = led.axis[:2]
    norm_axis_xy = float(np.linalg.norm(axis_xy))
    axis_xy_hat = axis_xy / norm_axis_xy if norm_axis_xy > 1e-9 else np.array([1.0, 0.0])

    sensor_xy = (led.pos[:2] + pt.pos[:2]) / 2.0  # sensor.pos の world 変換先（表の基準点）
    query_z = (led.pos[2] + pt.pos[2]) / 2.0      # 可視判定の代表点の高さ（LED・PTの中間）

    # --- 壁・柱: 同一平面上で隙間ゼロで連続する矩形（壁・柱を区別しない）をまとめる ---
    # 教授セッションが実測で確認した事実（2026-08-21）: 迷路の壁（厚み12mm）と柱
    # （12mm角）はどちらも格子線上に中心を持つので、同じ壁面の上では中心x・半長xが
    # 完全に一致し、y方向の隙間が厳密に0.000mmになる（規格による。偶然ではない）。
    # 本物の切れ目（開口部）だけが1区画ぶん（168mm）の隙間になる。したがって
    # 「同じ法線・同じ平面位置の矩形を、隙間が本物の開口部よりずっと小さい
    # `_COPLANAR_GAP_EPS_M` 以下なら連続とみなしてまとめる」ことで、壁と柱を区別せず
    # 物理的に1枚の連続した面を1枚として扱える（以前の失敗: 壁だけをまとめて柱を
    # 除外していたため隙間が残り、隣接する矩形どうしの干渉が消えなかった）。
    plane_groups: Dict[Tuple[float, float, float], list] = {}
    for owner_idx, r in enumerate(near_rects):
        for n_hat, face_center, half_u in _rect_candidate_faces(r):
            plane_coord = round(float(np.dot(face_center, n_hat)), 6)
            key = (round(float(n_hat[0]), 3), round(float(n_hat[1]), 3), plane_coord)
            u_hat = np.array([n_hat[1], -n_hat[0]])
            center_u = float(np.dot(face_center, u_hat))
            plane_groups.setdefault(key, []).append((owner_idx, center_u, half_u))

    # 平面ごとに、隙間が `_COPLANAR_GAP_EPS_M` 以下の矩形を連続クラスタへまとめる
    # （壁の妻面どうしの隙間は実測0.000mm、本物の開口部は168mm級なので、
    # 間の値ならどこで閾値を切ってもよい）。
    clusters: list = []   # 各要素: (n_hat, plane_coord, members, u_min, u_max)
    for (nx, ny, plane_coord), members in plane_groups.items():
        members.sort(key=lambda m: m[1])
        n_hat = np.array([nx, ny])
        cur = [members[0]]
        cur_min = members[0][1] - members[0][2]
        cur_max = members[0][1] + members[0][2]
        for m in members[1:]:
            m_min = m[1] - m[2]
            m_max = m[1] + m[2]
            if m_min - cur_max <= _COPLANAR_GAP_EPS_M:
                cur.append(m)
                cur_max = max(cur_max, m_max)
            else:
                clusters.append((n_hat, plane_coord, cur, cur_min, cur_max))
                cur = [m]
                cur_min, cur_max = m_min, m_max
        clusters.append((n_hat, plane_coord, cur, cur_min, cur_max))

    # 遮蔽は代表点1点ではなく、面の実際の長さに沿った `n_occlusion_samples` 点で判定し、
    # 見えている点の割合を表引きの値に掛ける（縁をまたぐ場合の階段状の誤差を抑える）。
    cluster_owners: list = []   # 各クラスタが束ねる元の矩形番号の集合（角判定に使う）
    cluster_dtl: list = []
    cluster_is_post: list = []
    cluster_samples_xy: list = []
    cluster_sample_owner: list = []
    for n_hat, plane_coord, members, u_min, u_max in clusters:
        u_hat = np.array([n_hat[1], -n_hat[0]])
        center_u = (u_min + u_max) / 2.0
        half_u = (u_max - u_min) / 2.0
        # クラスタ（合成パネル）の実際の中心点。平面位置は掃引時に得た `plane_coord` を
        # そのまま使う（元の矩形の面を検索し直して復元する回り道はしない。以前の
        # 実装ミスの元だった。u 成分を含めないと「横ずれ」が中心からのずれではなく
        # 原点からのずれになるので、必ず center_u もここで足すこと）。
        face_center = plane_coord * n_hat + center_u * u_hat

        rel = sensor_xy - face_center
        if float(np.dot(rel, n_hat)) <= 0.0:
            continue  # バックフェイスカリング
        d, inc, lat = _face_dtl(sensor_xy, axis_xy_hat, face_center, n_hat)
        if d > max_range_m:
            continue

        # 表引き専用の「横ずれ」の補正（2026-08-21・統合後に発覚した2つ目のバグ）。
        # 上の `lat` はクラスタ（合成パネル）の中心からのずれだが、表は
        # `DEFAULT_WALL_HALF_LENGTH_M`（84mm半長）の1枚パネルを前提に作ってある。
        # 統合で合成パネルが84mm半長よりずっと長くなると、センサが合成パネルの
        # 端の近くにいても「中心からのずれ」は大きな値のままになり、表の届く範囲を
        # 外れて `_lookup_or_zero` がゼロを返してしまう（実例: 直接0.82に対し
        # 表引きが3e-12まで潰れた）。合成パネルが十分長い場合は、「センサの垂線の
        # 足に最も近い、表がカバーできる範囲（半長ぶん）の仮想中心」を使う
        # （足がパネル奥深くにあれば横ずれ0＝内部の値に収束し、端の近くにあれば
        # 端からのずれとして正しく評価できる）。
        half_ref = DEFAULT_POST_HALF_LENGTH_M if half_u < post_half_extent_threshold_m \
            else DEFAULT_WALL_HALF_LENGTH_M
        sample_u_min, sample_u_max = u_min, u_max
        if half_u > half_ref:
            foot_u = float(np.dot(sensor_xy, u_hat))
            virtual_center = min(max(foot_u, u_min + half_ref), u_max - half_ref)
            lat = foot_u - virtual_center
            # 遮蔽判定の標本点も、合成パネル全体ではなく「表が実際にカバーしている
            # 局所窓」（仮想中心±半長）に絞る（2026-08-21・上の横ずれ補正だけでは
            # 直らなかった3つ目のバグ: 合成パネル全体に一様配置した標本点だと、
            # 足の近くが完全に見えていても、遠く離れた無関係な区間がたまたま
            # 遮られているだけで見えている割合が薄まってしまい、過小評価が残った
            # 実例が見つかった）。
            sample_u_min = max(u_min, virtual_center - half_ref)
            sample_u_max = min(u_max, virtual_center + half_ref)

        offsets_abs = np.linspace(sample_u_min, sample_u_max, n_occlusion_samples)
        samples_xy = face_center[None, :] + (offsets_abs - center_u)[:, None] * u_hat[None, :]
        sample_owner = np.empty(n_occlusion_samples, dtype=int)
        for j, off_abs in enumerate(offsets_abs):
            best_owner, best_gap = members[0][0], math.inf
            for owner_idx, c, h in members:
                gap = max(c - h - off_abs, off_abs - (c + h), 0.0)
                if gap < best_gap:
                    best_gap = gap
                    best_owner = owner_idx
            sample_owner[j] = best_owner

        cluster_owners.append({m[0] for m in members})
        cluster_dtl.append((d, inc, lat))
        cluster_is_post.append(half_u < post_half_extent_threshold_m)
        cluster_samples_xy.append(samples_xy)
        cluster_sample_owner.append(sample_owner)

    total = 0.0
    best_per_group: Dict[int, Tuple[float, float]] = {}
    if cluster_owners:
        n_cand = len(cluster_owners)
        all_samples_xy = np.concatenate(cluster_samples_xy, axis=0)
        query_points = np.concatenate(
            [all_samples_xy, np.full((all_samples_xy.shape[0], 1), query_z)], axis=1,
        )
        skip_idx = np.concatenate(cluster_sample_owner)
        occluded = _segment_occluded(query_points, led.pos, boxes, skip_idx)
        if dual_origin:
            occluded = occluded | _segment_occluded(query_points, pt.pos, boxes, skip_idx)
        occluded = occluded.reshape(n_cand, n_occlusion_samples)
        visible_frac_per_cluster = np.mean(~occluded, axis=1)

        cluster_values = []
        for i in range(n_cand):
            visible_frac = float(visible_frac_per_cluster[i])
            if visible_frac <= 0.0:
                cluster_values.append(0.0)
                continue
            d, inc, lat = cluster_dtl[i]
            if cluster_is_post[i]:
                v = _lookup_or_zero(post_table, d, inc, lat) if post_table is not None else 0.0
            else:
                v = _lookup_or_zero(wall_table, d, inc, lat)
            cluster_values.append(v * visible_frac)

        # 同じ矩形（角の柱など）が複数のクラスタ（別の法線方向）に同時に属す場合は、
        # 独立に足すと過大評価になる（角を見ている配置。直接積分は同じ箱の複数面を
        # まとめて1回の放射計算で扱うため、隣り合う面どうしが互いを一部自己遮蔽する）。
        # クラスタどうしが元の矩形を1つでも共有していれば同じグループとみなし
        # （Union-Find）、グループ内は最大値だけを、グループをまたいでは合計する。
        parent = list(range(n_cand))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        owner_to_cluster: Dict[int, int] = {}
        for i, owners in enumerate(cluster_owners):
            for o in owners:
                if o in owner_to_cluster:
                    _union(i, owner_to_cluster[o])
                else:
                    owner_to_cluster[o] = i

        best_by_root: Dict[int, Tuple[float, int]] = {}
        for i in range(n_cand):
            root = _find(i)
            v = cluster_values[i]
            if root not in best_by_root or v > best_by_root[root][0]:
                best_by_root[root] = (v, i)
        total = sum(v for v, _i in best_by_root.values())

        # best_per_owner（フォールバック判定用）: クラスタが束ねるすべての元の矩形に、
        # そのクラスタの (寄与, 距離) を割り当てる（矩形番号ベースの判定と互換にする）。
        for v, i in best_by_root.values():
            d_i = cluster_dtl[i][0]
            for owner_idx in cluster_owners[i]:
                if owner_idx not in best_per_group or v > best_per_group[owner_idx][0]:
                    best_per_group[owner_idx] = (v, d_i)

    # (寄与, 距離) の組。距離は「精度が粗くなる姿勢」の判定（近距離だけに絞る）に使う。
    best_per_owner_out: Dict[int, Tuple[float, float]] = best_per_group

    # --- 床: 代表点（中心＋リング）ごとに可視判定し、見えている割合ぶんだけ足す ---
    if include_floor:
        floor_pts = _floor_sample_points(sensor_xy, floor_sample_radius_m, floor_sample_ring)
        occluded_f = _segment_occluded(floor_pts, led.pos, boxes, None)
        if dual_origin:
            occluded_f = occluded_f | _segment_occluded(floor_pts, pt.pos, boxes, None)
        visible_frac = float(np.mean(~occluded_f))
        total += visible_frac * floor_value

    return total * sensor.gain, best_per_owner_out, near_rects


def fast_response(
    sensor: IrSensorSpec,
    pose: PoseLike,
    surfaces: Sequence[Rect],
    wall_table: ResponseTable,
    floor_value: float,
    *,
    post_table: Optional[ResponseTable] = None,
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M,
    include_floor: bool = True,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    dual_origin: bool = True,
    post_half_extent_threshold_m: float = DEFAULT_POST_HALF_EXTENT_THRESHOLD_M,
    floor_sample_radius_m: float = DEFAULT_FLOOR_SAMPLE_RADIUS_M,
    floor_sample_ring: int = DEFAULT_FLOOR_SAMPLE_RING,
    n_occlusion_samples: int = DEFAULT_N_OCCLUSION_SAMPLES,
) -> float:
    """`response()` の高速版。数値積分をせず、`wall_table`/`post_table`/`floor_value`
    （`build_wall_table()`/`build_post_table()`/`floor_baseline()` で事前に作ったもの）を
    解析的な幾何計算と表引きだけで姿勢から応答を推定する。
    モジュール docstring の「高速フォワードモデル」節を参照。中身は
    `_fast_response_breakdown()`（内訳つき）の `total` だけを返す薄いラッパー。

    Args:
        sensor/pose/surfaces: `response()` と同じ（`surfaces` は迷路全体の壁・柱でよい。
            `max_range_m` より遠いものは内部で足切りする）。
        wall_table: `build_wall_table()` で作った表（床を含まない・入射角は符号付き）。
        floor_value: `floor_baseline()` で作った床の基準値。
        post_table: `build_post_table()` で作った柱専用の表（省略時は柱・壁の妻面の
            寄与をゼロにする近似。精度検査で分かったとおり、柱がピーク距離付近に
            あるとき無視すると2桁以上過小評価しうるので、本番では渡すこと）。
        dual_origin: True（既定）なら、各面の代表点が LED からだけでなく PT からも
            見えるかを追加検査し、どちらかから遮られていれば寄与させない
            （LED と PT は離隔 `separation_m` ぶん別の点にあり、光軸を厳密には
            一致させられないという物理的制約 — `note_034` の出発点）。
        floor_sample_radius_m/floor_sample_ring: 床の可視判定に使う代表点（中心＋リング）
            の半径・本数。
        n_occlusion_samples: 壁・柱の各辺の可視判定に使う標本点の数（辺の実際の長さに
            沿って等間隔。隣接する面どうしが互いを部分的に隠す場面の近似精度に効く）。

    Returns:
        `response()` と同じ任意単位の推定値。
    """
    total, _breakdown, _near_rects = _fast_response_breakdown(
        sensor, pose, surfaces, wall_table, floor_value,
        post_table=post_table, wall_height_m=wall_height_m, include_floor=include_floor,
        max_range_m=max_range_m, dual_origin=dual_origin,
        post_half_extent_threshold_m=post_half_extent_threshold_m,
        floor_sample_radius_m=floor_sample_radius_m, floor_sample_ring=floor_sample_ring,
        n_occlusion_samples=n_occlusion_samples,
    )
    return total


# ============================================================================
# fast_response の精度が粗くなる姿勢を検出し、そこだけ直接積分に落とす
# ============================================================================
DEFAULT_ADJACENCY_GAP_M: float = 0.02   # これ以下の隙間の矩形どうしは「隣接」とみなす
DEFAULT_ADJACENCY_SIGNIFICANCE_FRAC: float = 0.05   # 最大寄与のこの割合以上を「支配的」とみなす
DEFAULT_ADJACENCY_DOMINANT_MAX_D_M: float = 0.30    # 支配的な矩形の距離がこれ以内のときだけ調べる


def _has_ambiguous_adjacency(
    best_per_owner: Dict[int, Tuple[float, float]], near_rects: Sequence[Rect],
    gap_threshold_m: float, significance_frac: float, dominant_max_d_m: float,
) -> bool:
    """`_fast_response_breakdown()` が返す矩形ごとの `(寄与, 距離)` のうち**実際に
    結果を左右するもの**（寄与が最大寄与の `significance_frac` 倍以上、かつ距離が
    `dominant_max_d_m` 以内）を「支配的な矩形」とし、それが**近傍のどれか
    （相手側の寄与の大小は問わない）に隣接**（隙間 <= `gap_threshold_m`。柱1本ぶんの
    隙間を想定）していないかを調べる。

    `fast_response` は「面ごとに独立に表引きして最大値を採る」近似なので、同一平面
    （または角）を作る隣接した面どうしが互いを部分的に自己遮蔽する効果を再現できない
    （`fast_response` のコメント・`note_034` 追記分を参照）。2026-08-21・教授セッションの
    独立検算で、この場面が壁の有無の判定を覆すほど大きく誤ることが分かった。

    **相手側の寄与の大小は問わない**（重要な設計判断）: 実例で、支配的な寄与
    （柱、寄与0.42）のすぐ隣にある壁の「その壁自身の独立な寄与」は 7.6e-4 と小さかった
    にもかかわらず、直接積分では2つが強く干渉して合計が 1.7e-6 まで潰れた。相手側も
    「寄与が大きいこと」を要求すると、この組を取り逃す。

    **`dominant_max_d_m` で絞る理由**: 迷路の壁は継ぎ目ごとに柱が立つ構造なので
    （`classic.geometry.wall_obstacles`: 「柱は格子点すべてに立つ」）、**どんな壁も
    必ずその両端で柱に隣接している**。相手側の寄与を問わずに「近傍の何かに隣接して
    いるか」だけで判定すると、ほぼすべての姿勢が該当してしまい、直接積分に落ちる
    割合がほぼ100%になって速さが出ない（実測で確認済み）。実際に問題が起きたのは
    支配的な矩形までの距離が近い場合（実測 d=1.7〜55mm）に限られていたので、
    支配的な矩形自身の距離が近いときだけに絞る。
    """
    if not best_per_owner:
        return False
    max_v = max(v for v, _d in best_per_owner.values())
    dominant = [
        owner for owner, (v, d) in best_per_owner.items()
        if v >= significance_frac * max_v and d <= dominant_max_d_m
    ]
    for owner in dominant:
        ri = near_rects[owner]
        for j, rj in enumerate(near_rects):
            if j == owner:
                continue
            dx = max(ri.cx - ri.hx - (rj.cx + rj.hx), rj.cx - rj.hx - (ri.cx + ri.hx), 0.0)
            dy = max(ri.cy - ri.hy - (rj.cy + rj.hy), rj.cy - rj.hy - (ri.cy + ri.hy), 0.0)
            if math.hypot(dx, dy) <= gap_threshold_m:
                return True
    return False


def fast_response_or_direct(
    sensor: IrSensorSpec,
    pose: PoseLike,
    surfaces: Sequence[Rect],
    wall_table: ResponseTable,
    floor_value: float,
    *,
    surf: Optional[SurfaceSpec] = None,
    post_table: Optional[ResponseTable] = None,
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M,
    include_floor: bool = True,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    adjacency_gap_m: float = DEFAULT_ADJACENCY_GAP_M,
    adjacency_significance_frac: float = DEFAULT_ADJACENCY_SIGNIFICANCE_FRAC,
    adjacency_dominant_max_d_m: float = DEFAULT_ADJACENCY_DOMINANT_MAX_D_M,
    n_grid: int = DEFAULT_N_GRID,
    **fast_kwargs,
) -> float:
    """`fast_response()` を使いつつ、精度が粗くなりうる姿勢（`_has_ambiguous_adjacency`
    が True を返す姿勢）だけ `response()`（直接積分）に落とす。

    背景（2026-08-21）: 隣接・同一平面の壁・柱が互いを部分的に自己遮蔽する場面で、
    `fast_response` 単体では壁の有無の判定が覆るほどの誤差が残った（実測: 500姿勢中
    数件〜十数件、閾値0.05〜0.30で）。「面ごとに独立に表引きして最大値を採る」近似の
    限界であり、表の形を変えずに直そうとすると別の姿勢を悪化させる
    （もぐら叩き。`fast_response` のコメント参照）。**正しさを速さより優先し**、
    この場面だけ直接積分に切り替える。

    🔴 **既定の閾値では、実測（実迷路500姿勢×3乱数種）で速さがほぼ出ない**
    （直接積分1.0倍・ほぼ全姿勢が直接積分に落ちる）。迷路は柱が格子点すべてに立つ
    構造なので（`classic.geometry.wall_obstacles`）、どの壁も必ず両端で柱に隣接して
    おり、「近傍の何かに隣接している」という条件を弱めに（`adjacency_significance_frac`
    を下げる・`adjacency_dominant_max_d_m` を広げる）すると常に真になってしまう。
    絞り込みを強めると（近距離・高い有意性しきい値だけに限定）速さは戻るが、
    壁の有無の判定が閾値0.05で500件中1〜3件覆る組み合わせが残った（実測。
    `research_notes/note_034_ir_sensor_model.md` 追記分に数値を記録）。
    **正しさを優先し、既定は「取りこぼしが出ない」側（低い有意性しきい値・
    広い距離範囲）にしてある。速さが要る用途では `adjacency_significance_frac`・
    `adjacency_dominant_max_d_m` を絞ることを検討すること（ただし壁の有無の
    判定の食い違いが再発しないか必ず確かめること）**。

    Args:
        sensor/pose/surfaces/wall_table/floor_value/post_table: `fast_response()` と同じ。
        surf: 直接積分に落ちたときに使う反射面の性質。表を作ったときと同じ
            `SurfaceSpec` を渡すこと（省略時は既定値 `SurfaceSpec()`。表を既定値以外の
            `surf` で作った場合は必ず渡す。渡し忘れると表と食い違う物理で直接積分する
            ことになる）。
        adjacency_gap_m: 「隣接」とみなす隙間のしきい値 [m]。既定は柱1本ぶん。
        adjacency_significance_frac: 支配的とみなす寄与の下限（最大寄与に対する比）。
        n_grid: 直接積分に落ちたときの `response()` の求積点数。
        **fast_kwargs: `_fast_response_breakdown()` へそのまま渡す追加引数。

    Returns:
        `response()`/`fast_response()` と同じ任意単位の推定値。
    """
    total, best_per_owner, near_rects = _fast_response_breakdown(
        sensor, pose, surfaces, wall_table, floor_value,
        post_table=post_table, wall_height_m=wall_height_m,
        include_floor=include_floor, max_range_m=max_range_m, **fast_kwargs,
    )

    if _has_ambiguous_adjacency(best_per_owner, near_rects, adjacency_gap_m,
                                 adjacency_significance_frac, adjacency_dominant_max_d_m):
        return response(
            sensor, pose, surfaces, surf if surf is not None else SurfaceSpec(),
            wall_height_m=wall_height_m, include_floor=include_floor,
            n_grid=n_grid, max_range_m=max_range_m, occlusion=True,
        )
    return total
