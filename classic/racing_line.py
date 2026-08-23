"""classic/racing_line.py — 柱間グラフの折れ線を丸めた走行ライン

`classic/gap_graph.py` が返す `GapPath`（頂点=柱間の中点、線分=0.09/0.12728/0.180m、
向き=0°/±45°/90°/±135°/180°の6通りだけの折れ線）は、そのままでは角が尖っている。
角では曲率が瞬間的に無限大になり、有限の加速度しか出せない機体は角の手前で
速度を0まで落とさざるを得ない（`experiments/exp_036_racing_line/PREREG.md` の
否定対照N1がこれを直接確かめる）。本モジュールは、この角を**丸めて連続な曲率
κ(s)を持つ1本の走行ラインにする**（`research_notes/note_037_probabilistic_localization.md`
§16 の指示）。

## 丸め方として何を選んだか、なぜか

選んだのは**対称な複クロソイド（両振りクロソイド。biclothoid）**である。
1つの角（旋回角 δ）を、次の形の曲率プロファイルで置き換える:

    κ(u) =  a・u          (0 <= u <= L)   … 立ち上がり半区間
    κ(u) =  a・(2L-u)     (L <= u <= 2L)  … 立ち下がり半区間

ここで a・L = 1/R（丸めの深さを表す仮のピーク半径R）、a・L^2 = |δ|（旋回角の
つじつま）。この2式から a = 1/(|δ|・R^2)、L = |δ|・R が一意に決まる
（`_corner_geometry` 参照）。

**候補には円弧・単純なクロソイド1本・スプラインもあった。選ばなかった理由:**

- **円弧（`classic/geometry.py::turn_path`が使う方式。従来のターン種別と同じ）**:
  直線(κ=0)から円弧(κ=1/R)へ**接点で瞬間的に**切り替わる。この切り替わり幅は
  丸めをどれだけ細かく離散化しても縮まらない**真の不連続**であり、
  PREREG §3-2「曲率は連続にすること」に反する。これが従来のターン種別
  （スラローム）の構造そのものであり、本実験はその代替を作るのが目的なので
  比較対象と同じ丸め方は選べない。
- **単純なクロソイド1本（0→κmaxの片道だけ）**: 角の前後で直線と滑らかに
  つながる保証がない（片道だと接続条件が旋回角・R・接線長の3つに対して
  自由度が不足する）。対称な往復（両振り）にすると、後述の二等分線対称性
  から接線長Tが解析的に一意に決まり、実装がむしろ単純になる。
- **スプライン（3次ベジエ等）**: 直線との接続点で曲率が一般に0にならない
  （不連続が残る）。クロソイドは定義上 κ(0)=0 から始まるので、直線との
  接続点で自動的に曲率0＝連続になる。これが決め手である。
- **クロソイド-円弧-クロソイド（道路・鉄道の緩和曲線の標準形）**: 立ち上がり
  と立ち下がりの間に定曲率の円弧区間を挟む3ピース構成。自由度がもう1つ増える
  （円弧区間の長さ）ため、丸めの深さを表す引数が2つに増えてしまう。
  「丸めの深さを引数にし」（作業指示）という1変数の探索に素直に収まる
  往復クロソイド（円弧区間の長さ=0の特別な場合）を採用した。

**トレードオフとして分かっていること**: 往復クロソイドは、同じピーク曲率
1/Rの円弧1本に比べて**全長が2倍**になる（円弧はR・|δ|、往復クロソイドは
2・L = 2・R・|δ|）。曲率が0から立ち上がる助走区間の分だけ、直線から
消費する長さ（接線長T）も円弧方式より長くなる。曲率連続性と引き換えの
コストとして許容する（`docs/JA_ENGINEERING_TERMS.md` 用語チェック済み）。

## 接線長Tの求め方（並進不変性を使う。実際に数値実験して確かめた）

角の頂点V、進入方向単位ベクトル d_in、退出方向単位ベクトル d_out
（旋回角 δ = d_inからd_outへの符号付き角）に対し、丸めの開始点
S1 = V − T・d_in から往復クロソイド全体（弧長2L）を積分すると終点 S2 になる。
Tを求めたい未知数だが、**積分は並進不変**（開始点をどこに置いても、得られる
軌跡は開始点だけ平行移動される）なので、**任意の開始点**（ここでは頂点Vそのもの、
T=0の仮の開始点）から一度だけ積分し、その変位 ΔS = S2(T=0) − V を求めれば、
実際の変位は開始点によらず常に同じ ΔS になる。

したがって恒等式 S2 = S1 + ΔS = (V − T・d_in) + ΔS が成り立つ。これが
S2 = V + T・d_out（丸め終了点は退出直線上でVからTだけ進んだ点）にも
一致してほしいので、

    ΔS = T・(d_in + d_out) = T・b

**（実際に数値積分してΔSとbの外積を計算すると、機械精度でゼロになることを
確認済み — ΔSは常にbに平行になる。これは往復クロソイドが「中点の接線を
軸とした鏡映対称」を持つことの帰結である）**。よって

    T = (ΔS・b) / (b・b)

**設計時の失敗と修正の記録**: 当初は「往復クロソイドは中点で点対称であり、
その中点が二等分線 b の上に乗る」という仮定から T = Xc − Yc・cot(|δ|/2)
という式を立てたが、`tests/test_racing_line.py` 作成前のざっと動作確認で
実際に角を組み立てて閉じるか検算したところ、位置の誤差が丸めの深さに
比例する大きさ（無視できない）で残った。原因は上記の仮定が誤っていたこと
（往復クロソイドは曲率の符号を反転させないので、中点に関して点対称ではなく
中点の接線を軸とした鏡映対称になる）。**「まず実際に計算して値を見てから
固定する」（作業指示）を地で行く形で見つけた誤りであり、上記の並進不変性を
使う導出に置き換えた。置き換え後は同じ確認で位置誤差が1e-15m
オーダー（浮動小数点誤差の範囲）に収まることを確認している。**

実装上は、`_corner_geometry` が仮開始点(0,0)・向きheading_inから一度だけ
2L分の積分を行い、その結果（ローカル点列とΔS）を保持する。`build_racing_line`
は実際の開始点 S1=V−T・d_in が分かった時点で、保持しておいた点列を
(S1.x, S1.y) だけ平行移動して使う（積分をやり直さない。並進不変性を使った
計算量の節約でもある）。

|δ|=180°（b=d_in+d_out=0でTが定義できない）は退化ケースであり
`RacingLineError` を送出する（`classic/ideal.py` が180°折返しをその場旋回に
落とすのと同じ扱いの退化点。柱間グラフの最短経路でこの角が実際に現れるかは
未確認だが、現れた場合は本モジュールでは丸めず例外にする — 呼び出し側が
それを検知して除外・報告できるようにする設計判断）。

## 丸めの深さの探索（PREREG §3-1 の合格条件）

丸めの深さは**ピーク半径R（1つのスカラー。経路全体で共通）**として表す。
`find_max_feasible_racing_line` が `classic.geometry.max_feasible_radius` と
同じ形の二分探索で、次の**両方**を満たす最大のRを探す:

1. **消費長の予算**: 1本の直線（区間長L_k）の両端が別々の角に食われるとき、
   両側の接線長の和がL_kを超えない（超えると2つの角の丸め区間が重なり、
   経路が幾何的に破綻する。`RacingLineOverlapError`）。
2. **機体の余裕**: `classic/geometry.py::clearance`（分離軸定理による厳密な
   干渉判定。本モジュールは変更しない・再利用するだけ）で経路全体を弧長に
   沿って刻んで掃引し、最小余裕が `margin_m` 以上（PREREG §3-1）。

Rが小さいほど丸めは角に忠実（＝もとの折れ線に近い。安全）、大きいほど
角を大きく短絡する（＝機体の外形が壁に近づく代わり最高速度は上がる）。
したがって`max_feasible_radius`と同じ「小さいRから安全側に始め、大きいR
に向かって二分探索する」方向で実装する。

## 丸めの深さは経路全体で1つ（設計の単純化）

角ごとに独立でRを最適化すれば理論上はもっと速い走行ラインになりうるが、
作業指示「丸めの深さを**引数**にし」は単数形であり、1スカラー引数の探索を
求めている。本モジュールもこれに従い、経路全体で共通の1つのRを探索する
（最も厳しい角・最も短い直線がボトルネックになり、そこがRの上限を決める）。
角ごとの独立最適化は将来の拡張として残す。

## マイコン実装の予算（PREREG §6・note_037 §13-4）

**重い計算（Rの二分探索。本モジュールの `find_max_feasible_racing_line`）は
壁地図が更新されたとき（探索完了後・最短走行の直前など）に1回だけ行う**
（`classic/gap_graph.py`のダイクストラ法と同じ運用。1kHzの制御周期には
収まらない — 下記の見積り参照）。走行中の毎ティックは、確定済みの
(s_grid, kappa_grid) 格子を`bisect`で引くだけ（`classic.profile.IdealTime.kappa_at`
と同じO(log n)の二分探索）であり、既存の`ProfileTracker`の実行モデルと
同じ負荷に収まる。

- **RAM**: 経路全長を目安3m、格子間隔`DEFAULT_DS_M=0.002m`とすると格子点数
  は概ね1500点。`s_grid`(float32)+`kappa_grid`(float32)で1点あたり8バイト、
  合計 約12KB（`classic/gap_graph.py`の柱間グラフ本体28KBと同程度の桁。
  1迷路ぶんの走行ラインとして許容範囲）。
- **演算量（探索1回）**: 二分探索`search_iters`回×(角ごとの半クロソイド
  数値積分`O(n_steps)`＋経路全体の掃引余裕`O(格子点数 × 近傍障害物数)`)。
  16x16迷路・角数十個・格子点数千数百・探索30回では数秒〜十数秒
  （実測は`experiments/exp_036_racing_line/run.py`のログに残す）。
  `classic/gap_graph.py`のダイクストラ法（5ms、1kHzに収まらない）と同様、
  **走行を止めている区間で1回計算する運用が前提**であり、むしろこちらの
  方が重い。毎ティックの実行（格子の線形補間）だけがリアルタイム制約の
  対象であり、そちらはO(log n)で十分に軽い。
- 数値積分・掃引ループは事前にサイズが分かっている固定長配列で書ける
  （`_integrate_curve`はPythonのlistを使っているが、組み込みでは
  `n_steps`が角ごとに求まった時点で固定長配列を1回だけ確保すればよく、
  ループ内で動的確保は発生しない設計にしてある）。

## 依存関係

`classic/gap_graph.py`・`classic/profile.py`・`classic/geometry.py`・
`classic/maze_map.py` を読むだけで、1行も変更しない。numpy以外に依存しない
（`classic/__init__.py`の層規約）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from classic.gap_graph import GapPath
from classic.geometry import Pose, clearance, wall_obstacles
from classic.maze_map import MazeMap, WallState
from classic.profile import Segment

__all__ = [
    "DEFAULT_DS_M",
    "DEFAULT_MARGIN_M",
    "DEFAULT_R_LO_M",
    "DEFAULT_R_HI_M",
    "DEFAULT_SEARCH_ITERS",
    "RacingLineError",
    "RacingLineOverlapError",
    "CornerInfo",
    "RacingLine",
    "build_racing_line",
    "find_max_feasible_racing_line",
    "to_segments",
    "evaluate_clearance",
    "max_kappa_jump",
    "diagonal_length_m",
]


# ============================================================================
# 例外
# ============================================================================
class RacingLineError(Exception):
    """走行ラインの構築に失敗した（角の幾何が閉じない・180°付近など）。"""


class RacingLineOverlapError(RacingLineError):
    """丸めの深さRが大きすぎて、隣り合う2つの角の丸め区間が同じ直線上で
    重なった（消費長の予算超過）。"""


# ============================================================================
# 既定値（`classic/geometry.py`の同種の既定値と揃えてある）
# ============================================================================
DEFAULT_DS_M = 0.002       # `classic.geometry.poses_along`の既定ds=0.002と同じ刻み
DEFAULT_MARGIN_M = 0.005   # `classic.geometry.max_feasible_radius`の既定marginと同じ
DEFAULT_R_LO_M = 0.02      # `classic.ideal._R_LO`と同じ（安全側の初期値）
DEFAULT_R_HI_M = 0.30      # 迷路のセル寸法(0.18m)より一回り大きい上限（探索の打ち切り）
DEFAULT_SEARCH_ITERS = 30  # `classic.geometry.max_feasible_radius`は40だが、
                            # 本探索は角ごとの数値積分＋全経路掃引を毎回行うため
                            # 重く、2^-30の相対精度で十分（1cmの探索幅なら1e-8m以下の分解能）
_MIN_CORNER_STEPS = 4       # 数値積分の最小分割数（Rが極端に小さい場合の安全弁）
_CORNER_EPS_RAD = 1e-6      # これ未満の旋回角は「直進継続」とみなし丸めない
_CLOSURE_TOL_M = 1e-4       # 角を閉じたときの位置の整合性検査の許容誤差
_CLOSURE_TOL_RAD = 1e-4     # 同・向きの許容誤差


# ============================================================================
# 0. 折れ線の向きの丸め（浮動小数点誤差の除去）
# ============================================================================
def _snap_heading(raw: float) -> float:
    """`GapPath.xy_m`から計算した向き[rad]を最寄りの45°へ丸める。

    柱間グラフの折れ線の向きは0°/±45°/90°/±135°/180°の6通りしかない
    （`classic/gap_graph.py`モジュールdocstring・`note_037`§15-1）。float32の
    丸め誤差で真の45°倍数からわずかにずれるので、ここで丸めておかないと
    「直進のはずの連続2線分」がごくわずかな旋回角を持つ偽の角として検出され
    てしまう。"""
    step = math.pi / 4.0
    return round(raw / step) * step


def _wrap_pi(angle: float) -> float:
    """角度を(-pi, pi]へ正規化する。"""
    a = math.fmod(angle + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


# ============================================================================
# 1. 角の検出
# ============================================================================
@dataclass(frozen=True)
class CornerInfo:
    """折れ線の1つの角（旋回角が0でない頂点）。"""

    vertex_index: int   # GapPath.xy_m内の頂点番号（0=出発, len-1=ゴール）
    delta_rad: float     # 符号付き旋回角[rad]（正=左/反時計回り。classic.geometry.arcと同じ符号）
    heading_in: float     # 進入区間の向き[rad]（45°の倍数に丸め済み）
    heading_out: float    # 退出区間の向き[rad]（同上）
    vertex_xy: Tuple[float, float]


def _find_corners(xy: np.ndarray) -> List[CornerInfo]:
    """折れ線の内部頂点から、旋回角が0でない角だけを抜き出す。"""
    xy64 = xy.astype(np.float64)
    n = xy64.shape[0]
    corners: List[CornerInfo] = []
    for i in range(1, n - 1):
        v_in = xy64[i] - xy64[i - 1]
        v_out = xy64[i + 1] - xy64[i]
        h_in = _snap_heading(math.atan2(v_in[1], v_in[0]))
        h_out = _snap_heading(math.atan2(v_out[1], v_out[0]))
        delta = _wrap_pi(h_out - h_in)
        if abs(delta) > _CORNER_EPS_RAD:
            corners.append(CornerInfo(
                vertex_index=i, delta_rad=delta, heading_in=h_in, heading_out=h_out,
                vertex_xy=(float(xy64[i, 0]), float(xy64[i, 1])),
            ))
    return corners


# ============================================================================
# 2. 定曲率セルの中点則による数値積分（`classic.geometry._pose_at`と同じ
#    閉じた式を、区分定数の曲率に対してセルごとに厳密に適用する）
# ============================================================================
def _integrate_curve(
    x0: float, y0: float, theta0: float,
    kappa_at: Callable[[float], float], length: float, n_steps: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """曲率 `kappa_at(u)`（u: このピース内の弧長, 0<=u<=length）に沿って
    `n_steps`個のセルへ分割し、各セルはその中点の曲率を使った定曲率の解析式
    （直線 or 円弧の厳密解）で厳密に進める中点則の数値積分。

    線形に変化する曲率に対しては誤差O(ds^2)（オイラー法・台形則のO(ds)より高精度）。

    戻り値: (xs, ys, thetas)（境界点。長さn_steps+1）と kappas（セル代表値。
    長さn_steps）のタプル。
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps は正整数である必要があります: {n_steps}")
    ds = length / n_steps
    xs = [0.0] * (n_steps + 1)
    ys = [0.0] * (n_steps + 1)
    ths = [0.0] * (n_steps + 1)
    kaps = [0.0] * n_steps
    xs[0], ys[0], ths[0] = x0, y0, theta0
    x, y, th = x0, y0, theta0
    for i in range(n_steps):
        u_mid = (i + 0.5) * ds
        k = kappa_at(u_mid)
        kaps[i] = k
        if abs(k) < 1e-12:
            x2 = x + ds * math.cos(th)
            y2 = y + ds * math.sin(th)
            th2 = th
        else:
            th2 = th + k * ds
            x2 = x + (math.sin(th2) - math.sin(th)) / k
            y2 = y - (math.cos(th2) - math.cos(th)) / k
        x, y, th = x2, y2, th2
        xs[i + 1], ys[i + 1], ths[i + 1] = x, y, th
    return xs, ys, ths, kaps


# ============================================================================
# 3. 1つの角の幾何（R依存の量。往復クロソイドのa・L・接線長T）
# ============================================================================
@dataclass(frozen=True)
class _CornerGeom:
    mag: float               # |delta| [rad]
    sign: float                # +1(左) or -1(右)
    L: float                   # 半区間(片道)の弧長 [m]
    a: float                   # 曲率の傾き。kappa(u) = sign*a*u
    T: float                   # 頂点から丸め開始点までの接線長 [m]（進入・退出とも同じ値）
    n_steps_half: int           # 片道あたりの積分分割数（往復で2倍）
    local_xs: List[float]         # 仮開始点(0,0)・向きheading_inから積分した点列のx（長さ2*n_steps_half+1）
    local_ys: List[float]         # 同y
    local_ths: List[float]        # 同向き[rad]（heading_inを織り込み済みなのでワールド座標でそのまま使える）
    local_kaps: List[float]       # 各セルの曲率（長さ2*n_steps_half）
    step_len: float                 # 1セルあたりの弧長 [m]（往復とも同じ長さ）


def _corner_geometry(delta_rad: float, heading_in: float, R_m: float, ds: float) -> _CornerGeom:
    """旋回角delta・進入向きheading_in・ピーク半径Rから、往復クロソイドの
    幾何を求める（1回の積分で完結する。理由はモジュールdocstring
    「接線長Tの求め方」参照）。

    a・L = 1/R、a・L^2 = |delta| （立ち上がり半区間で旋回角|delta|/2を
    使い切る）の2式から a = 1/(|delta|*R^2)、L = |delta|*R。
    """
    mag = abs(delta_rad)
    if mag <= _CORNER_EPS_RAD:
        raise ValueError("旋回角が0の頂点は角ではありません（呼び出し側の誤り）")
    sign = math.copysign(1.0, delta_rad)
    heading_out = heading_in + delta_rad
    d_in = (math.cos(heading_in), math.sin(heading_in))
    d_out = (math.cos(heading_out), math.sin(heading_out))
    b = (d_in[0] + d_out[0], d_in[1] + d_out[1])
    bb = b[0] * b[0] + b[1] * b[1]
    if bb < 1e-9:
        raise RacingLineError(
            f"旋回角が180°に近く({math.degrees(mag):.3f}°)、b=d_in+d_outが0に近くTが定義できません"
        )

    L = mag * R_m
    a = 1.0 / (mag * R_m * R_m)
    n_steps_half = max(int(math.ceil(L / ds)), _MIN_CORNER_STEPS)
    n_steps = 2 * n_steps_half
    total_len = 2.0 * L

    def _kappa_at(u: float) -> float:
        if u <= L:
            return sign * a * u
        return sign * a * (total_len - u)

    xs, ys, ths, kaps = _integrate_curve(0.0, 0.0, heading_in, _kappa_at, total_len, n_steps)

    # ΔS = 仮開始点(0,0)から積分した終点の変位。並進不変性からT = (ΔS・b)/(b・b)
    # （導出・実測での検証はモジュールdocstring参照）。
    dS = (xs[-1], ys[-1])
    T = (dS[0] * b[0] + dS[1] * b[1]) / bb

    return _CornerGeom(
        mag=mag, sign=sign, L=L, a=a, T=T, n_steps_half=n_steps_half,
        local_xs=xs, local_ys=ys, local_ths=ths, local_kaps=kaps,
        step_len=total_len / n_steps,
    )


# ============================================================================
# 4. 走行ライン本体
# ============================================================================
@dataclass(frozen=True)
class RacingLine:
    """丸めた走行ライン。弧長で表した(s, kappa(s))の格子
    （`classic.profile.IdealTime.s_grid`/`kappa_grid`と同じ規約: s_gridは
    境界点でn+1個、kappa_grid/kind_gridは各セルの値でn個）。"""

    R_m: float                 # 探索/指定で使った丸めの深さ [m]
    s_grid: List[float]         # 弧長格子境界 [m]（長さn+1）
    kappa_grid: List[float]      # 各セルの曲率 [1/m]（長さn）
    kind_grid: List[str]         # 各セルの種別（"straight"/"diagonal"/"corner"）
    xy_m: np.ndarray              # (n+1, 2) 各境界点の位置 [m]（float64）
    theta_grid: List[float]        # 各境界点の向き [rad]（長さn+1）
    corner_count: int
    original_length_m: float        # 丸める前のGapPath.distance_m
    total_length_m: float             # 丸めた後の経路長（= s_grid[-1]）

    def kappa_max_abs(self) -> float:
        return max((abs(k) for k in self.kappa_grid), default=0.0)


def _straight_only_racing_line(path: GapPath, R_m: float = 0.0) -> RacingLine:
    """折れ線に角が1つも無い（旋回角がすべて0）場合の走行ライン。
    曲率0の1セルで表す（PREREG検査1「直線だけの折れ線を通すと曲率が0のまま」）。"""
    xy64 = path.xy_m.astype(np.float64)
    length = float(path.distance_m)
    heading = _snap_heading(math.atan2(
        xy64[-1, 1] - xy64[0, 1], xy64[-1, 0] - xy64[0, 0]
    )) if len(xy64) >= 2 else 0.0
    kind = "diagonal" if (abs(math.cos(heading)) > 1e-6 and abs(math.sin(heading)) > 1e-6) else "straight"
    return RacingLine(
        R_m=R_m, s_grid=[0.0, length], kappa_grid=[0.0], kind_grid=[kind],
        xy_m=xy64[[0, -1]] if len(xy64) >= 2 else xy64,
        theta_grid=[heading, heading],
        corner_count=0, original_length_m=length, total_length_m=length,
    )


def build_racing_line(
    path: GapPath, R_m: float, ds: float = DEFAULT_DS_M,
) -> RacingLine:
    """折れ線`path`の角を、ピーク半径`R_m`（丸めの深さ。1つの引数）の往復
    クロソイドで丸め、走行ラインを組み立てる。**壁との干渉は見ない**
    （`evaluate_clearance`で別途確かめる。本関数は消費長の予算だけを検査する
    純粋な幾何構築である）。

    隣り合う2つの角が同じ直線を食い合って長さが負になったら
    `RacingLineOverlapError`。角の旋回角が180°付近なら`RacingLineError`
    （`_corner_geometry`参照）。
    """
    xy64 = path.xy_m.astype(np.float64)
    n_pts = xy64.shape[0]
    corners = _find_corners(xy64)
    if not corners:
        return _straight_only_racing_line(path, R_m)

    geoms: Dict[int, _CornerGeom] = {
        c.vertex_index: _corner_geometry(c.delta_rad, c.heading_in, R_m, ds) for c in corners
    }
    corner_by_vertex = {c.vertex_index: c for c in corners}

    seg_len = np.linalg.norm(np.diff(xy64, axis=0), axis=1)  # 長さ n_pts-1
    n_segs = n_pts - 1
    trims_head = [0.0] * n_segs  # セグメントkの先頭（頂点k側）で角に食われる長さ
    trims_tail = [0.0] * n_segs  # セグメントkの末尾（頂点k+1側）で角に食われる長さ
    for idx, g in geoms.items():
        if idx - 1 >= 0:
            trims_tail[idx - 1] = g.T
        if idx <= n_segs - 1:
            trims_head[idx] = g.T

    remaining = [seg_len[k] - trims_head[k] - trims_tail[k] for k in range(n_segs)]
    for k, rem in enumerate(remaining):
        if rem < -1e-6:
            raise RacingLineOverlapError(
                f"R={R_m:.5f}m はセグメント{k}（頂点{k}->{k+1}, 長さ{seg_len[k]:.5f}m）で"
                f"消費長の予算を{-rem:.6f}m超過した（隣り合う2つの角の丸め区間が重なった）"
            )
    remaining = [max(r, 0.0) for r in remaining]

    first_heading = _snap_heading(math.atan2(xy64[1, 1] - xy64[0, 1], xy64[1, 0] - xy64[0, 0]))
    xs: List[float] = [xy64[0, 0]]
    ys: List[float] = [xy64[0, 1]]
    ths: List[float] = [first_heading]
    kaps: List[float] = []
    kinds: List[str] = []
    lens: List[float] = []

    cur_x, cur_y = xy64[0, 0], xy64[0, 1]
    for k in range(n_segs):
        p0, p1 = xy64[k], xy64[k + 1]
        heading = _snap_heading(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        dxu, dyu = math.cos(heading), math.sin(heading)
        kind = "diagonal" if (abs(dxu) > 1e-6 and abs(dyu) > 1e-6) else "straight"

        length_k = remaining[k]
        if length_k > 1e-9:
            n_cells = max(int(math.ceil(length_k / ds)), 1)
            cell_len = length_k / n_cells
            for _ in range(n_cells):
                cur_x += cell_len * dxu
                cur_y += cell_len * dyu
                xs.append(cur_x); ys.append(cur_y); ths.append(heading)
                kaps.append(0.0); kinds.append(kind); lens.append(cell_len)

        nxt = k + 1
        if nxt in geoms:
            g = geoms[nxt]
            c = corner_by_vertex[nxt]
            # `g.local_xs/ys/ths`は仮開始点(0,0)・向きheading_inから積分済み
            # （並進不変性を使い、実開始点(cur_x,cur_y)へ平行移動するだけで済む
            # — モジュールdocstring「接線長Tの求め方」参照。積分をやり直さない）。
            base_x, base_y = cur_x, cur_y
            for xx, yy, tt, kk in zip(g.local_xs[1:], g.local_ys[1:], g.local_ths[1:], g.local_kaps):
                xs.append(base_x + xx); ys.append(base_y + yy); ths.append(tt)
                kaps.append(kk); kinds.append("corner"); lens.append(g.step_len)
            cur_x, cur_y = base_x + g.local_xs[-1], base_y + g.local_ys[-1]
            end_theta = g.local_ths[-1]

            # 整合性検査（実際に計算して閉じているか確かめる。docstring参照）:
            # 退出方向の直線に、向きと位置の両方でぴったり接続しているはず。
            h_out_expected = c.heading_out
            theta_err = _wrap_pi(end_theta - h_out_expected)
            if abs(theta_err) > _CLOSURE_TOL_RAD:
                raise RacingLineError(
                    f"角(頂点{nxt})の退出向きが一致しない: 期待{h_out_expected:.6f} "
                    f"実際{end_theta:.6f} 差{theta_err:.2e}rad（実装の前提が崩れている）"
                )
            vx, vy = xy64[nxt]
            dxo, dyo = math.cos(h_out_expected), math.sin(h_out_expected)
            exp_x, exp_y = vx + g.T * dxo, vy + g.T * dyo
            pos_err = math.hypot(cur_x - exp_x, cur_y - exp_y)
            if pos_err > _CLOSURE_TOL_M:
                raise RacingLineError(
                    f"角(頂点{nxt})の退出位置が一致しない: 誤差{pos_err:.2e}m"
                    "（実装の前提が崩れている）"
                )

    s_grid = [0.0] * (len(lens) + 1)
    for i, l in enumerate(lens):
        s_grid[i + 1] = s_grid[i] + l

    return RacingLine(
        R_m=R_m, s_grid=s_grid, kappa_grid=kaps, kind_grid=kinds,
        xy_m=np.array(list(zip(xs, ys)), dtype=np.float64), theta_grid=ths,
        corner_count=len(corners), original_length_m=float(path.distance_m),
        total_length_m=s_grid[-1],
    )


# ============================================================================
# 5. 機体の余裕（`classic/geometry.py`を再利用するだけ。変更しない）
# ============================================================================
# 迷路全体の障害物（16x16で500件超）を格子点1つずつに毎回全部渡すと、
# `clearance()`がSAT判定のために障害物ごとに多角形（4頂点のタプルの新規リスト）
# を作るコストが支配的になり、探索1回（30回の二分探索×20迷路）が現実的な
# 時間に収まらない（実測: 絞り込み無しだと1迷路の探索だけで2分超）。
# `classic/ideal.py`4.5節と同じ考え方（安価な矩形の重なり判定で候補を絞ってから
# `classic.geometry.clearance()`の厳密なSAT判定を呼ぶ）を、格子点1つずつではなく
# `_CLEARANCE_BLOCK_POINTS`点をまとめた区画（ブロック）ごとに1回だけ行う
# （区画内の点はどうせ数mmしか動かないので、区画の外接矩形+余裕で絞り込んでも
# 結果は変わらない）。**判定ロジック自体（分離軸定理）は一切変えていない**
# （`tests/test_racing_line.py::test_local_filter_matches_unfiltered_clearance`
# で絞り込み無しと同じ値になることを直接照合する）。
_CLEARANCE_BLOCK_POINTS = 60  # 1ブロックあたりの格子点数（ds=0.002なら約0.12m）
_CLEARANCE_REACH_M = 0.15     # 絞り込みの余裕（機体半対角+壁厚+ブロック内の移動分の安全側の値）


def _filter_nearby_obstacles(obstacles, xs: np.ndarray, ys: np.ndarray, reach: float):
    """`xs,ys`の外接矩形から`reach`以内にある障害物だけを返す（安価な矩形の
    重なり判定。厳密なSAT判定はしない — 候補を絞るためだけの下界チェック）。"""
    xmin, xmax = float(np.min(xs)) - reach, float(np.max(xs)) + reach
    ymin, ymax = float(np.min(ys)) - reach, float(np.max(ys)) + reach
    return [
        o for o in obstacles
        if (o.cx + o.hx) >= xmin and (o.cx - o.hx) <= xmax
        and (o.cy + o.hy) >= ymin and (o.cy - o.hy) <= ymax
    ]


def evaluate_clearance(
    line: RacingLine, maze: MazeMap,
    block_points: int = _CLEARANCE_BLOCK_POINTS, reach_m: float = _CLEARANCE_REACH_M,
) -> Tuple[float, int]:
    """走行ラインの全境界点を掃引したときの最小余裕[m]と、そのときの
    格子点番号を返す。`classic.geometry.clearance`（分離軸定理の厳密判定）を
    そのまま使う。境界点は`build_racing_line`が既に`ds`間隔で刻んでいるので、
    追加のサンプリングはしない（速度のための近傍絞り込みは上のコメント参照）。"""
    v_walls = np.asarray(maze.v_walls) == int(WallState.WALL)
    h_walls = np.asarray(maze.h_walls) == int(WallState.WALL)
    obstacles = wall_obstacles(v_walls, h_walls)

    xs_all = line.xy_m[:, 0]
    ys_all = line.xy_m[:, 1]
    n = line.xy_m.shape[0]

    best = math.inf
    best_i = 0
    for start in range(0, n, block_points):
        end = min(start + block_points, n)
        near = _filter_nearby_obstacles(obstacles, xs_all[start:end], ys_all[start:end], reach_m)
        if not near:
            continue  # このブロックの近傍に障害物が無い＝この区間は余裕が効かない
        for i in range(start, end):
            pose = Pose(float(xs_all[i]), float(ys_all[i]), line.theta_grid[i])
            d = clearance(pose, near)
            if d < best:
                best = d
                best_i = i
    return best, best_i


# ============================================================================
# 6. 丸めの深さRの探索（合格条件1・2を両方満たす最大のR）
# ============================================================================
def find_max_feasible_racing_line(
    path: GapPath, maze: MazeMap,
    margin_m: float = DEFAULT_MARGIN_M,
    r_lo: float = DEFAULT_R_LO_M,
    r_hi: float = DEFAULT_R_HI_M,
    ds: float = DEFAULT_DS_M,
    search_iters: int = DEFAULT_SEARCH_ITERS,
) -> RacingLine:
    """余裕(`margin_m`以上)と消費長の予算の両方を満たす最大のR（丸めの深さ）を
    二分探索し、そのRでの走行ラインを返す。`classic.geometry.max_feasible_radius`
    と同じ「小さいR=安全側から始め、大きいRへ向けて二分探索する」構造。

    折れ線に角が1つも無ければ探索せずそのまま返す（R_m=0として記録）。
    `r_lo`ですら実行不可能なら`RacingLineError`を送出する。
    """
    if not _find_corners(path.xy_m):
        return _straight_only_racing_line(path)

    def _try(R: float) -> Optional[RacingLine]:
        try:
            line = build_racing_line(path, R, ds=ds)
        except RacingLineOverlapError:
            return None
        min_clear, _ = evaluate_clearance(line, maze)
        if min_clear < margin_m:
            return None
        return line

    lo_line = _try(r_lo)
    if lo_line is None:
        raise RacingLineError(
            f"r_lo={r_lo}m でも余裕{margin_m}m または消費長の予算を満たさない"
            "（探索範囲の下限ですら丸められない）"
        )
    hi_line = _try(r_hi)
    if hi_line is not None:
        return hi_line

    lo, hi = r_lo, r_hi
    best = lo_line
    for _ in range(search_iters):
        mid = 0.5 * (lo + hi)
        mid_line = _try(mid)
        if mid_line is not None:
            lo = mid
            best = mid_line
        else:
            hi = mid
    return best


# ============================================================================
# 7. classic.profile.min_time へ渡す変換・副次記録の補助
# ============================================================================
def to_segments(line: RacingLine) -> List[Segment]:
    """走行ラインを`classic.profile.min_time`にそのまま渡せる`Segment`列にする。"""
    return [
        Segment(length=line.s_grid[i + 1] - line.s_grid[i], curvature=line.kappa_grid[i], kind=line.kind_grid[i])
        for i in range(len(line.kappa_grid))
    ]


def max_kappa_jump(line: RacingLine) -> float:
    """隣り合うセルどうしの曲率の跳びの最大値[1/m]（PREREG §3-2の実測値）。"""
    if len(line.kappa_grid) < 2:
        return 0.0
    diffs = np.abs(np.diff(np.asarray(line.kappa_grid, dtype=np.float64)))
    return float(np.max(diffs))


def diagonal_length_m(line: RacingLine) -> float:
    """直進部分のうち斜め（`kind=="diagonal"`）だった弧長の合計[m]
    （副次の記録。コーナー区間は斜め/直進の区別を持たないので含めない）。"""
    total = 0.0
    for i, kind in enumerate(line.kind_grid):
        if kind == "diagonal":
            total += line.s_grid[i + 1] - line.s_grid[i]
    return total
