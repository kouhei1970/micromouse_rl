"""verification/audit_050_raycast.py

IR センサ（LED + フォトトランジスタ）1 本の応答を、モンテカルロ光線追跡で計算する。

**独立実装であることが本ファイルの存在理由そのものである**:
`mouse/ir_sensor.py`（同じ物理を面積分で解いた既存実装）は import はもとより、
参照・grep も一切行っていない。仕様の根拠は
`verification/AUDIT_050_PREREG_ir_raycast.md` の §1・§2 と、そこから引用された
本ファイル冒頭のコメント（後述）だけである。

読んでよいとされたファイルだけを使っている:
  - `mouse/params.py`（`RobotParams().sensors` の pos/zaxis 文字列）
  - `classic/geometry.py`（`Rect` 型と `wall_obstacles()` の返す矩形の意味）

------------------------------------------------------------------------
## 1. センサ 1 本の構成

- 離隔 0.0065 m・縦配置。機体座標で
  `led_pos = pos + (0,0,1)*0.00325`、`pt_pos = pos - (0,0,1)*0.00325`。
- LED・PT の光軸はどちらも `axis`（正規化）に平行。アライメント誤差 0。
- 指向性 `cos^m(θ)`。半値角 `half_angle_deg` から `m = ln(0.5)/ln(cos(half_angle_deg))`。
- 機体姿勢 `(x,y,theta)` → ワールド座標は z 軸まわり `theta` 回転 + `(x,y,0)` 平行移動。

## 2. 反射面

- 壁・柱: `classic.geometry.wall_obstacles()` が返す `Rect(cx,cy,hx,hy)` を、
  z∈[0, wall_height_m] に押し出した直方体とみなす。**側面 4 枚だけが反射面**
  （上面は反射面に含めない。上端は機体より高く、上から見下ろす経路が無いため）。
- 床: z=0 の水平面。センサ取り付け点（機体座標 `pos`）を中心に
  半幅 `floor_halfextent_m` の正方形パッチだけを持つ（これより外側に床は無い扱い）。
- ランバート面。拡散反射率 `diffuse`、鏡面成分なし。
  反射放射輝度 = 放射照度 × diffuse / π。

## 3. 光線追跡のアルゴリズム（next event estimation）

1. LED から `cos^m(θ)` に比例する確率密度で方向を重要度標本抽出する（下記§4）。
2. 光線が最初に当たる面（壁側面 or 床）を、軸平行直方体との交差＋床平面との交差で求める。
   直方体の上面・底面に当たった場合は反射面が無いので、そこで光線は終端する
   （寄与 0。以降の反射も追わない）。
3. 当たった点から PT へ直接結ぶ経路（NEE）で寄与を積算する。
   このとき面素→PT の線分が他の直方体に遮られていれば寄与 0（遮蔽）。
4. `max_bounces` 回まで、上記を繰り返す。2 回目以降の反射方向はランバート面の
   余弦分布から標本抽出し、重みに拡散反射率 `diffuse` を掛けて追跡を続ける
   （導出は下記§5）。反射回数は「LED を出てから PT に入るまでに面で反射した回数」。
5. `r_e`（LED から面素までの距離。**1 反射目の面素についてのみ**判定する）が
   `max_range_m` を超える面素の NEE 寄与は 0 にする。

## 4. 重要度標本抽出の重みの導出（LED の 1 次発光方向）

放射強度 `I(θ) = I_led * cos^m(θ)`（θ は LED 光軸からの角）を持つ放射源から、
半球上に確率密度 `p(ω) ∝ cos^m(θ)` で方向を標本抽出する。

正規化: `∫∫ p(θ,φ) sinθ dθ dφ = 1` を課すと
`p(θ,φ) = (m+1)/(2π) * cos^m(θ)`（`θ∈[0,π/2], φ∈[0,2π)` で積分すると 1 になることを
`∫_0^{π/2} cos^mθ sinθ dθ = 1/(m+1)` から確認できる）。

θ の周辺分布の累積分布関数は `F(θ) = 1 - cos^{m+1}(θ)`。逆関数法で
`u ~ U(0,1)` に対し `1 - cos^{m+1}θ = u` を解くと、`1-u` も一様分布なので
そのまま `u` を使ってよく、**`cosθ = u^{1/(m+1)}`**（仕様に明記された式そのもの）。
方位角 φ は一様 `U(0,2π)`。

この抽出密度で 1 本の光線に与える重みを導く。実際に計算したいのは
「反射面上の面素 dA への放射照度 `E=I_led cos^mθ_e/r_e^2` を、面積分ではなく
方向の積分として書き直したもの」である。面積分から立体角積分への変数変換は
標準の関係 `dω = cos(θ_i) dA / r_e^2`（θ_i は面法線と (面素→LED) のなす角）で行える
ので、

    E(x) dA = I_led cos^m(θ_e) / r_e^2 * dA = I_led cos^m(θ_e) * [cos(θ_i) dA / r_e^2] / cos(θ_i)

ではなく、より直接に「面積分（1 反射）の全体」を立体角積分に書き換える:

    ∬ E(x) cos(θ_i) diffuse/π * (PT 側の項) dA
      = ∫ I_led cos^m(θ_e) * diffuse/π * (PT 側の項) dω(θ_e,φ_e)

（`cos(θ_i) dA = r_e^2 dω` の関係を使って `dA` を `dω` に置き換えた。すなわち、
LED から出た方向 ω を光線追跡でそのまま面素に対応づければ、その面素の面積要素は
自動的に "その方向の立体角要素" として計算に取り込まれる。これが光線追跡で
面積分を解ける理由そのものである）。

この立体角積分をモンテカルロ推定するとき、標本方向 ω_i の重みは
`I_led cos^m(θ_e,i) / p(ω_i)` になる。ここで抽出則の θ_e,i はまさに標本方向の
LED 光軸からの角なので `cos^m(θ_e,i)` が p(ω_i) の `cos^m` 因子と厳密に相殺し、

    I_led cos^m(θ_e,i) / p(ω_i)
      = I_led cos^m(θ_e,i) / [ (m+1)/(2π) * cos^m(θ_e,i) ]
      = I_led * 2π / (m+1)

**光線ごとに完全に定数**（サンプルした角度に依らない）になる。これを
`Φ_LED := I_led * 2π/(m+1)` と書く（半球へ放射される全放射束に等しい）。
つまり、N 本の光線それぞれに固定の重み `Φ_LED` を持たせ、その光線が実際に
当たった面素での寄与（`diffuse/π * cos(θ_v) * S_pt * cos^m(θ_r) / r_v^2`。
1/r_e^2 も cos^mθ_e も既に重みへ吸収済みなのでここには現れない）を掛けて
N 本で平均する、というのが本実装の推定量である:

    response ≈ (1/N) Σ_i Φ_LED * [面素 i での NEE 寄与]

（面素に届かなかった／遮蔽された／r_e>max_range の光線は寄与 0）。

## 5. 2 反射目以降の重み伝播（ランバート面の余弦重点抽出）

面素 x で反射した光線が次にどの方向へ飛ぶかを、BRDF に比例する重点抽出
（余弦重点抽出）で決める。ランバート面の BRDF は `f_r = diffuse/π`（定数）。
余弦重点抽出の確率密度は `p(ω') = cos(θ')/π`（正規化: `∫cos(θ')/π dω' = 1`）。
新しい光線が運ぶ重み（「光子パワー」）の更新則は、標準的なフォトン追跡と同じく

    P_new = P_old * f_r * cos(θ') / p(ω') = P_old * (diffuse/π) * cos(θ') / (cos(θ')/π)
          = P_old * diffuse

角度 θ' が完全にキャンセルして、**重みに `diffuse` を掛けるだけ**になる
（仕様の「反射率0.8を重みに掛けて追跡を続け」はこの意味）。各反射点でも
同じ NEE（面素→PT）を行い、寄与を積算する。

------------------------------------------------------------------------
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from classic.geometry import Rect

__all__ = [
    "Sensor",
    "sensors_from_params",
    "raycast_response",
]


# ============================================================================
# 0. センサの軽量表現（mouse/ir_sensor.py の型は使わない・import しない）
# ============================================================================
@dataclass(frozen=True)
class Sensor:
    """LED+PT 対 1 本ぶんの機体座標での取り付け情報。"""

    name: str
    pos: Tuple[float, float, float]   # 機体座標での取り付け位置 [m]
    axis: Tuple[float, float, float]  # 機体座標での光軸（正規化前でもよい）


def sensors_from_params(params=None) -> List[Sensor]:
    """`mouse/params.py::RobotParams().sensors`（LF/LS/RF/RS）を `Sensor` 列に変換する。

    `params` を省略すると `mouse.params.RobotParams()`（既定値）を使う。
    """
    if params is None:
        from mouse.params import RobotParams  # 許可されたファイルのみ import

        params = RobotParams()
    out: List[Sensor] = []
    for s in params.sensors:
        pos = tuple(float(v) for v in s["pos"].split())
        axis = tuple(float(v) for v in s["zaxis"].split())
        out.append(Sensor(name=s["name"], pos=pos, axis=axis))
    return out


# ============================================================================
# 1. 幾何ユーティリティ（直方体との交差・床平面との交差・遮蔽判定）
# ============================================================================
def _prepare_boxes(rects: Sequence[Rect], wall_height_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """`Rect` 列を AABB の (min, max) 配列 (B,3) に変換する（z は [0, wall_height_m]）。"""
    if len(rects) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    mins = np.array([[r.cx - r.hx, r.cy - r.hy, 0.0] for r in rects])
    maxs = np.array([[r.cx + r.hx, r.cy + r.hy, wall_height_m] for r in rects])
    return mins, maxs


def _prune_boxes(
    mins: np.ndarray, maxs: np.ndarray, center_xy: Tuple[float, float], radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    """`center_xy` から見て、どう頑張っても交差し得ない箱を間引く（結果を変えない高速化）。

    箱の外接円（xy 平面上）が `center_xy` から `radius` 以内にあるものだけを残す。
    `radius` を「使う光線の到達しうる最大距離＋箱の最大半径」以上に取れば安全。
    """
    if mins.shape[0] == 0:
        return mins, maxs
    cx = (mins[:, 0] + maxs[:, 0]) / 2.0
    cy = (mins[:, 1] + maxs[:, 1]) / 2.0
    hx = (maxs[:, 0] - mins[:, 0]) / 2.0
    hy = (maxs[:, 1] - mins[:, 1]) / 2.0
    box_r = np.hypot(hx, hy)
    d = np.hypot(cx - center_xy[0], cy - center_xy[1])
    keep = (d - box_r) <= radius
    return mins[keep], maxs[keep]


def _nearest_box_hit(
    o: np.ndarray, d: np.ndarray, mins: np.ndarray, maxs: np.ndarray, active: np.ndarray, eps: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    """光線ごとに、直方体群のうち最も近い交差の t とその面の軸（0=x,1=y,2=z）を返す。

    スラブ法（AABB とレイの交差の標準解法）。`entry_axis` が 2（z 軸）のときは
    上面・底面に当たったことを意味し、呼び出し側で「反射面ではない」として扱う。
    """
    n = o.shape[0]
    best_t = np.full(n, np.inf)
    best_axis = np.full(n, -1, dtype=np.int8)
    if mins.shape[0] == 0:
        return best_t, best_axis
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_d = 1.0 / d
    for b in range(mins.shape[0]):
        t1 = (mins[b] - o) * inv_d
        t2 = (maxs[b] - o) * inv_d
        tsmall = np.minimum(t1, t2)
        tbig = np.maximum(t1, t2)
        t_enter = np.max(tsmall, axis=1)
        t_exit = np.min(tbig, axis=1)
        axis = np.argmax(tsmall, axis=1)
        valid = active & (t_enter <= t_exit) & (t_exit > eps) & (t_enter > eps) & (t_enter < best_t)
        best_t = np.where(valid, t_enter, best_t)
        best_axis = np.where(valid, axis, best_axis)
    return best_t, best_axis


def _floor_hit(
    o: np.ndarray,
    d: np.ndarray,
    floor_center: Tuple[float, float],
    floor_halfextent: float,
    active: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """z=0 の床平面との交差 t と有効フラグ（有限パッチ内かつ前方向）を返す。"""
    n = o.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -o[:, 2] / d[:, 2]
    valid = active & (d[:, 2] < -1e-12) & (t > eps)
    x = o[:, 0] + t * d[:, 0]
    y = o[:, 1] + t * d[:, 1]
    inside = (np.abs(x - floor_center[0]) <= floor_halfextent) & (np.abs(y - floor_center[1]) <= floor_halfextent)
    valid = valid & inside
    t = np.where(valid, t, np.inf)
    return t, valid


def _first_hit_all(
    o: np.ndarray,
    d: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    active: np.ndarray,
    include_floor: bool,
    floor_center: Tuple[float, float],
    floor_halfextent: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """光線ごとに最初に当たる「反射面」（壁側面 or 床）を求める。

    戻り値: `hit_t` (N,)・`hit_normal` (N,3)・`hit_kind` (N,)
    （0=当たらなかった/上面・底面に当たり終端した, 1=壁側面, 2=床）。
    """
    n = o.shape[0]
    box_t, box_axis = _nearest_box_hit(o, d, mins, maxs, active)
    if include_floor:
        floor_t, floor_valid = _floor_hit(o, d, floor_center, floor_halfextent, active)
    else:
        floor_t = np.full(n, np.inf)
        floor_valid = np.zeros(n, dtype=bool)

    box_has_hit = np.isfinite(box_t)
    use_box = active & box_has_hit & (box_t <= floor_t)
    use_floor = active & floor_valid & (~use_box)

    hit_kind = np.zeros(n, dtype=np.int8)
    hit_t = np.zeros(n)
    hit_normal = np.zeros((n, 3))

    side_mask = use_box & (box_axis == 0)
    hit_kind[side_mask] = 1
    hit_t[side_mask] = box_t[side_mask]
    hit_normal[side_mask, 0] = -np.sign(d[side_mask, 0])

    side_mask = use_box & (box_axis == 1)
    hit_kind[side_mask] = 1
    hit_t[side_mask] = box_t[side_mask]
    hit_normal[side_mask, 1] = -np.sign(d[side_mask, 1])

    # box_axis == 2（上面・底面）は反射面ではないので hit_kind=0 のまま（終端）。

    hit_t[use_floor] = floor_t[use_floor]
    hit_normal[use_floor] = np.array([0.0, 0.0, 1.0])
    hit_kind[use_floor] = 2

    return hit_t, hit_normal, hit_kind


def _segment_occluded(
    a: np.ndarray, b: np.ndarray, mins: np.ndarray, maxs: np.ndarray, active: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """線分 a→b（各 (N,3)）が直方体群のどれかに遮られているかを返す (N,) bool。

    `a` は交点そのもの（あるいはそこから法線方向へ僅かに逃がした点）を渡す想定。
    直方体は「反射面かどうか」に関わらず**不透明な固体**として扱う（上面・底面
    からでも内部を貫通する経路は塞がれる、という自然な扱い。ただし縦配置の
    センサでは通常このケースは起きない）。
    """
    n = a.shape[0]
    blocked = np.zeros(n, dtype=bool)
    if mins.shape[0] == 0:
        return blocked
    seg = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_d = 1.0 / seg
    for i in range(mins.shape[0]):
        t1 = (mins[i] - a) * inv_d
        t2 = (maxs[i] - a) * inv_d
        tsmall = np.minimum(t1, t2)
        tbig = np.maximum(t1, t2)
        t_enter = np.max(tsmall, axis=1)
        t_exit = np.min(tbig, axis=1)
        hit = active & (t_enter <= t_exit) & (t_enter < 1.0 - eps) & (t_exit > eps)
        blocked |= hit
    return blocked


# ============================================================================
# 2. 重要度標本抽出（LED の cos^m 発光・ランバート面の余弦重点抽出）
# ============================================================================
def _orthonormal_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """単位ベクトル `n` に直交する 2 本の単位ベクトルを 1 組作る。"""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


def _sample_cos_pow(rng: np.random.Generator, axis: np.ndarray, m: float, n: int) -> np.ndarray:
    """`axis` を中心に `p(ω) ∝ cos^m(θ)` で N 本の方向を標本抽出する（逆関数法）。"""
    u1 = rng.random(n)
    u2 = rng.random(n)
    cos_t = u1 ** (1.0 / (m + 1.0))
    sin_t = np.sqrt(np.clip(1.0 - cos_t * cos_t, 0.0, None))
    phi = 2.0 * np.pi * u2
    t1, t2 = _orthonormal_basis(axis)
    dirs = (
        (sin_t * np.cos(phi))[:, None] * t1
        + (sin_t * np.sin(phi))[:, None] * t2
        + cos_t[:, None] * axis
    )
    return dirs


def _sample_cosine_hemisphere(rng: np.random.Generator, normals: np.ndarray, n: int) -> np.ndarray:
    """各行の法線 `normals[i]` を中心に、余弦重点抽出（`p(ω')=cos(θ')/π`）で方向を抽出する。"""
    u1 = rng.random(n)
    u2 = rng.random(n)
    cos_t = np.sqrt(np.clip(1.0 - u2, 0.0, None))
    sin_t = np.sqrt(np.clip(u2, 0.0, None))
    phi = 2.0 * np.pi * u1

    cond = np.abs(normals[:, 0]) < 0.9
    a = np.zeros_like(normals)
    a[cond] = np.array([1.0, 0.0, 0.0])
    a[~cond] = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normals, a)
    t1 = t1 / np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normals, t1)

    dirs = (
        (sin_t * np.cos(phi))[:, None] * t1
        + (sin_t * np.sin(phi))[:, None] * t2
        + cos_t[:, None] * normals
    )
    return dirs


# ============================================================================
# 3. 光線追跡のコア（LED/PT の絶対位置を受け取る。分離量はここでは仮定しない）
# ============================================================================
def _raycast_core(
    led_pos: np.ndarray,
    led_axis: np.ndarray,
    pt_pos: np.ndarray,
    pt_axis: np.ndarray,
    rects: Sequence[Rect],
    *,
    n_rays: int,
    seed: int,
    max_bounces: int,
    include_floor: bool,
    m_led: float,
    m_pt: float,
    max_range_m: float,
    wall_height_m: float,
    floor_halfextent_m: float,
    diffuse: float,
    floor_center: Tuple[float, float],
) -> float:
    rng = np.random.default_rng(seed)

    led_pos = np.asarray(led_pos, dtype=float)
    pt_pos = np.asarray(pt_pos, dtype=float)
    led_axis = np.asarray(led_axis, dtype=float)
    led_axis = led_axis / np.linalg.norm(led_axis)
    pt_axis = np.asarray(pt_axis, dtype=float)
    pt_axis = pt_axis / np.linalg.norm(pt_axis)

    mins, maxs = _prepare_boxes(rects, wall_height_m)
    if max_bounces == 1:
        # 1 反射のみなら、r_e<=max_range_m を満たさない面素の寄与はどのみち 0 になる。
        # そのため LED から見て「箱のどの点も max_range_m + 余裕 を超える」箱は
        # 間引いても結果が変わらない（安全な高速化）。余裕は壁パネルの最大半長
        # （cell_size/2 程度 ≈0.084m）を十分覆うように取る。
        prune_r = max_range_m + 0.25
        mins, maxs = _prune_boxes(mins, maxs, (float(led_pos[0]), float(led_pos[1])), prune_r)
    # max_bounces>1 のときは、2 反射目以降の到達距離に上限が無いので間引かない。

    n = int(n_rays)
    o = np.tile(led_pos, (n, 1))
    d = _sample_cos_pow(rng, led_axis, m_led, n)
    phi_led = 2.0 * np.pi / (m_led + 1.0)  # I_led=1.0 のときの LED 全放射束（重要度標本抽出の重み。§4 参照）
    power = np.full(n, phi_led)
    alive = np.ones(n, dtype=bool)

    total = 0.0
    for bounce in range(1, max_bounces + 1):
        hit_t, hit_normal, hit_kind = _first_hit_all(
            o, d, mins, maxs, alive, include_floor, floor_center, floor_halfextent_m
        )
        hit_valid = alive & (hit_kind > 0)
        hit_pts = o + np.where(hit_valid, hit_t, 0.0)[:, None] * d

        if bounce == 1:
            r_e = np.linalg.norm(hit_pts - led_pos, axis=1)
            range_ok = r_e <= max_range_m
        else:
            range_ok = np.ones(n, dtype=bool)

        to_pt = pt_pos - hit_pts
        r_v = np.linalg.norm(to_pt, axis=1)
        r_v_safe = np.where(r_v > 1e-9, r_v, 1.0)
        dir_x_to_pt = to_pt / r_v_safe[:, None]
        cos_v = np.clip(np.sum(hit_normal * dir_x_to_pt, axis=1), 0.0, None)
        dir_pt_to_x = -dir_x_to_pt
        cos_r = np.clip(dir_pt_to_x @ pt_axis, 0.0, None)

        nudged = hit_pts + hit_normal * 1e-6  # 自己遮蔽（自分が乗っている箱に遮られる誤検出）を防ぐ
        occluded = _segment_occluded(nudged, np.tile(pt_pos, (n, 1)), mins, maxs, hit_valid)

        nee_ok = hit_valid & range_ok & (cos_v > 0.0) & (cos_r > 0.0) & (~occluded)
        g = np.where(
            nee_ok,
            diffuse / np.pi * cos_v * np.power(cos_r, m_pt) / (r_v_safe ** 2),
            0.0,
        )
        total += float(np.sum(power * g))

        if bounce < max_bounces:
            cont = hit_valid
            # hit_valid=False の行は hit_normal がゼロベクトルのまま（初期値）なので、
            # そのまま渡すと正規化で 0/0 が発生する（結果は cont マスクで捨てるので
            # 数値には影響しないが、無用な RuntimeWarning を避けるためダミー法線に差し替える）。
            safe_normal = np.where(hit_valid[:, None], hit_normal, np.array([0.0, 0.0, 1.0]))
            new_dir = _sample_cosine_hemisphere(rng, safe_normal, n)
            o = np.where(cont[:, None], nudged, o)
            d = np.where(cont[:, None], new_dir, d)
            power = np.where(cont, power * diffuse, power)
            alive = cont

    return total / n


# ============================================================================
# 4. 公開 API
# ============================================================================
def raycast_response(
    sensor: Sensor,
    pose: Tuple[float, float, float],
    rects: Sequence[Rect],
    *,
    n_rays: int,
    seed: int,
    max_bounces: int = 1,
    include_floor: bool = True,
    led_half_angle_deg: float = 5.0,
    pt_half_angle_deg: float = 5.0,
    max_range_m: float = 0.35,
    wall_height_m: float = 0.05,
    floor_halfextent_m: float = 0.20,
    diffuse: float = 0.8,
    separation_m: float = 0.0065,
) -> float:
    """センサ 1 本の応答を、`n_rays` 本の光線でモンテカルロ推定する。

    `sensor.pos`/`sensor.axis` は機体座標。`pose=(x,y,theta)` で
    ワールド座標に変換してから追跡する（z 軸まわり `theta` 回転 + 平行移動）。
    戻り値は §1-3（PREREG）の面積分と同じ規格の無次元応答。
    """
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    trans = np.array([x, y, 0.0])

    pos_b = np.asarray(sensor.pos, dtype=float)
    axis_b = np.asarray(sensor.axis, dtype=float)
    axis_b = axis_b / np.linalg.norm(axis_b)

    # 離隔の半分。縦配置（LED が上、PT が下）。既定 0.0065m は PREREG §1-1 の値。
    sep_half = np.array([0.0, 0.0, separation_m / 2.0])
    led_pos_b = pos_b + sep_half
    pt_pos_b = pos_b - sep_half

    led_pos_w = rot @ led_pos_b + trans
    pt_pos_w = rot @ pt_pos_b + trans
    axis_w = rot @ axis_b  # LED・PT とも同じ光軸方向（アライメント誤差 0）

    mount_pos_w = rot @ pos_b + trans
    floor_center = (float(mount_pos_w[0]), float(mount_pos_w[1]))

    m_led = math.log(0.5) / math.log(math.cos(math.radians(led_half_angle_deg)))
    m_pt = math.log(0.5) / math.log(math.cos(math.radians(pt_half_angle_deg)))

    return _raycast_core(
        led_pos_w,
        axis_w,
        pt_pos_w,
        axis_w,
        rects,
        n_rays=n_rays,
        seed=seed,
        max_bounces=max_bounces,
        include_floor=include_floor,
        m_led=m_led,
        m_pt=m_pt,
        max_range_m=max_range_m,
        wall_height_m=wall_height_m,
        floor_halfextent_m=floor_halfextent_m,
        diffuse=diffuse,
        floor_center=floor_center,
    )
