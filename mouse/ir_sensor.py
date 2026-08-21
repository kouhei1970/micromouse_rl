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
]


PoseLike = Union[Tuple[float, float, float], object]

DEFAULT_WALL_HEIGHT_M: float = 0.05    # 壁の高さ [m]（迷路規格。note_034: 壁上端 50mm）
DEFAULT_FLOOR_HALFEXTENT_M: float = 0.20   # 床パッチの半幅 [m]（センサ周辺のみを面として持つ）
DEFAULT_MAX_RANGE_M: float = 0.35      # これより遠い壁は最初から積分対象に入れない
DEFAULT_N_GRID: int = 28               # 面 1 枚あたりの求積点数（1 軸あたり）


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
