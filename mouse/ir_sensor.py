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

LED・PT とも半値角が数度と非常に狭い（cos^m の m は 5° で ~180 という鋭さ）。面全体を
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
]


PoseLike = Union[Tuple[float, float, float], object]

DEFAULT_WALL_HEIGHT_M: float = 0.05    # 壁の高さ [m]（迷路規格。note_034: 壁上端 50mm）
DEFAULT_FLOOR_HALFEXTENT_M: float = 0.20   # 床パッチの半幅 [m]（センサ周辺のみを面として持つ）
DEFAULT_MAX_RANGE_M: float = 0.35      # これより遠い壁は最初から積分対象に入れない
DEFAULT_N_GRID: int = 28               # 面 1 枚あたりの求積点数（1 軸あたり）

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
    separation_m: float = 0.0065       # LED と PT の離隔 [m]（note_034: 確定 6〜7mm の代表値）
    layout: str = "vertical"           # 既定は縦配置（ユーザ判断）。"horizontal" も選べる
    led_half_angle_deg: float = 5.0
    pt_half_angle_deg: float = 5.0
    led_tilt_deg: float = 0.0          # アライメント誤差（±1° を想定）
    pt_tilt_deg: float = 0.0
    gain: float = 1.0                  # 個体差（応答全体に掛かる係数）


@dataclass
class SurfaceSpec:
    """反射面の性質。"""

    diffuse: float = 0.8       # 拡散反射率（白い射出成形プラスチック）
    specular: float = 0.0      # 鏡面成分。ゼロにできるが、値を変えられる
    shininess: float = 20.0    # 鏡面のとがり（Phong 的な冪指数）


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
) -> float:
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

    Returns:
        センサが受ける光量（任意単位）。強度のまま使うか距離に直すかはここでは決めない
        （`adc()` で AD 変換器の生の値に変換してから、使い方はアルゴリズム側で選ぶ）。
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
        return 0.0

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

    return total * sensor.gain


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
