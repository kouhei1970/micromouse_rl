"""
classic/wall_belief.py
================
柱間 544 か所（縦 16×17・横 17×16）それぞれについて、壁の有無を対数オッズで持つ
信念地図（`experiments/exp_034_wall_belief/PREREG.md`。設計の根拠は
`research_notes/note_037_probabilistic_localization.md` §7・§19・§20）。

【何を変えるか（note_037 §19）】
これまでの `classic/sensing.py` は区画の中心で 1 回だけ壁センサを読み、しきい値で
3 値に分けて地図へ書いていた。制御周期は 100 Hz なので、1 区画を通るあいだの
百数十回の読みのうち 4 つしか使っていなかった。本モジュールは**制御周期ごとに**
姿勢の信念（平均・共分散）と距離センサ 4 本の実測値を受け取り、見ている可能性のある
柱間それぞれへ対数尤度比を足し込む。決めるのは経路を引くために読み出すときだけである。

    l_w ← clamp( l_w + clip( log[p(z|w=1,bel(x)) / p(z|w=0,bel(x))], ±R_max ), ±L_max )

p(z|w, bel(x)) = N(z; d_hat(x_hat, w), sigma^2)。d_hat は `classic/obs_model.py` が出す
予測距離。sigma^2 = sigma_sensor^2 + J^T P J（J = ∂d_hat/∂x、P は姿勢の共分散）で
姿勢の不確かさを繰り込む。これにより、すれすれの入射で J が大きくなる姿勢の観測が
自動的に軽くなる（`exp_033` の外れ値 40 件が示した悪条件の場面）。

【真値を絶対に読まない】
本ファイルは `mouse.sim.MouseSim` を一切 import しない。`privileged_pose` /
`privileged_velocity` への属性アクセスも無い。壊れていないことは
`tests/test_wall_belief.py` がソース走査（`tests/test_classic_pose.py:56` と同じ手法）
で検査する。

【任意の姿勢で成り立つ（note_037 §14-2）】
区画の中心・軸並行に特殊化しない。柱間を「見ている」かどうかは姿勢から幾何で決める
（`classic/obs_model.py` と同じスラブ法の光線交差を使う）。45° を向いた姿勢でも
更新できる（`tests/test_wall_belief.py` が固定する）。

【設計判断（結果を見る前に決め、コメントに残す。すべて exp_034 で対照により校正する）】

1. **どの柱間を見ているかの決め方・候補が複数ある場合の扱い**:
   センサの光線が近傍の柱間（`classic/obs_model.py::_local_obstacles` と同じ探索窓の
   縦壁・横壁セル。柱は不確かさを持たない構造物なので候補にしない。外周の柱間も
   構造上確定なので候補にしない）の面と幾何的に交わるものはすべて候補にする
   （＝「候補が絞れない」を無理に一意化しない）。候補ごとに独立した仮説検定を行う:
     - `w=1`（この柱間だけが壁）と `w=0`（この柱間だけが開通）の 2 つの背景地図を、
       それ以外は**現在の点推定**（対数オッズ > 0 を「壁」とみなす。これは
       `declare()` の非対称しきい値とは別物 — 読み出し用の宣言には使わない、
       純粋に「他の柱間がどちらの状態に近いか」を仮定するための内部利用の閾値である）
       で埋めて `classic/obs_model.py::predict_ranges` 相当の予測を行う。
     - 実際にはもっと手前に「確定的に壁がある」背景物体があって、この柱間を
       forced True/False にしても予測が変わらない（＝この柱間は今回の観測では
       見えていない）場合、2 つの予測はほぼ一致し、対数尤度比は自然に 0 に近づく。
       これにより、明示的な「occlusion 判定」を別途書かなくても、遮蔽されている
       候補は自動的に無視に近い扱いになる（仮説検定そのものが遮蔽を織り込む）。
   候補どうしの相関（1 つの柱間の真の状態が別の候補の尤度にも影響しうること）は
   無視する — 単峰の点推定近似（note_037 §20-2 が明記する近似）の範囲内。

2. **飽和した読み（cutoff = 0.3 m）は捨てない**（note_037 §20-3 規則 5）。
   測定値が cutoff にちょうど張り付いているとき、これは「距離がちょうど 0.3 m だった」
   という点情報ではなく「真の距離は 0.3 m 以上だった」という**打ち切り**情報である。
   点の尤度 N(z=cutoff; mean, sigma^2) ではなく、上側裾確率
   P(D >= cutoff | mean, sigma) = 0.5*erfc((cutoff-mean)/(sigma*sqrt2)) の対数を使う
   （`_log_survival`）。これを w=1 と w=0 それぞれで計算し、比を取る。

3. **信念は弱まれる（規則 2）。**対数オッズは加算も減算もできる。矛盾する証拠が
   来れば戻る。「一度 WALL と書いたら変えない」という作りにしていない。

4. **対数オッズの上限 L_max（規則 3）。**1 バイト量子化（int8, ±127）にちょうど
   収まるよう、1 LSB = 0.1 nat とすると 127*0.1 = 12.7 になるよう選んだ
   （`quantize_log_odds`/`dequantize_log_odds`。量子化しても `declare()` の結果が
   変わらないことは `tests/test_wall_belief.py` が固定する）。

5. **決めるのは読み出すときだけ（規則 4）。**`declare()`/`WallBelief.to_maze_map()` が
   宣言のしきい値 `T_wall`/`T_open` を引数に取る。**非対称**にする: 「壁」は低い確信度
   （尤度比 e^2 ≈ 7.4 倍）で認め、「開通」は高い確信度（尤度比 e^8 ≈ 2981 倍）を
   要求する。`note_033` の出口条件「致命的な誤り（真は壁なのに開通と判定）= 0」を
   推定器の設計へ写したもの。**既定値は暫定であり、`exp_034` の主判定量
   （致命的な誤りの数）で校正する。**

6. **1 回の観測の寄与に上限 R_max（規則 1）。**外れ値 1 つで信念が飛ばないための
   上限。既定 0.5 nat は、`L_max`（12.7 nat）に対して「最大確信度の観測が連続しても
   26 回弱かかる」水準に選んだ ── note_037 §19-2 が言う「薄い証拠でも数十回積めば
   効く」という設計思想を、飽和までの回数として具体化したもの。**暫定値であり、
   対照（`exp_034` PREREG §4 の N4: L_max を外して飽和の実証を見る）で校正する。**

【マイコン実装の予算（note_037 §13-4・exp_034 PREREG §6。手計算であり実行時計測ではない）】

  本 Python 実装は「まず正しく動くこと」を優先し、d_hat(w=1)/d_hat(w=0) をそれぞれ
  `classic/obs_model.py` の予測（近傍探索つきの完全なスラブ法）で計算し、J も
  数値微分（中心差分、刻みは下記）で求めている。1 候補・1 観測あたり:

    - d_hat(w=1)・d_hat(w=0) の計算: 2 回のセンサ予測
      （1 回あたり `exp_033` 実測値: 乗算 約109・除算 2 ── `_predict_one_sensor` は
      `obs_model` のセンサ 1 本ぶんの内部ループをそのまま再利用しているため、
      コスト構造は同一）
    - J の数値微分（x, y, theta の中心差分・刻み `FD_STEP_XY_DEFAULT`=1e-4m・
      `FD_STEP_THETA_DEFAULT`=1e-4rad）: 6 回のセンサ予測
    - 合計: センサ予測 8 回 ≈ 乗算 872・除算 16
    - sigma^2 = sigma_sensor^2 + J^T P J（3×3 の二次形式）: 乗算 約 10
    - 対数尤度比（非飽和）: 乗算 2（二乗×2）・除算 1。**指数・対数の呼び出しは無い**
      （w=1 と w=0 で同じ sigma^2 を使うため、正規化定数 log(2*pi*sigma^2) が比を取ると
      打ち消え、二次形式の差だけで済む ── `_log_likelihood_ratio` 参照）
    - 対数尤度比（飽和時のみ）: `erfc` 2 回・`log` 2 回が追加で要る

    → 1 候補・1 観測あたり **乗算 約 884・除算 約 17**（非飽和時。飽和時はこれに
    erfc×2・log×2 が加わる）。

  🔴 **候補数は実測して桁を確かめた（推測で済ませない）。**当初「探索窓に入る
  柱間の個数」（`_wall_candidates` の生の戻り値）をそのまま候補数とみなし、開けた
  16x16 迷路（外周のみ壁）でロボットが境界から離れた位置にいる場合に**1 センサ
  あたり最大 98 個**（4 センサで 392 個/周期）に達することを確認した。だが実際に
  「対数尤度比の計算まで進む」候補は、この窓の中で**光線が実際に交わり、かつ
  cutoff+マージン以内**のものだけである。同じ開けた 16x16 迷路・境界から離れた
  位置・軸並行(北向き)と斜め(37°)の両方の姿勢で実測すると、**1 センサあたり
  実際に交わる候補は 2〜3 個**（4 センサで 8〜12 個/周期）だった。「窓の大きさ」と
  「実際に光線と交わる候補数」は別物であり、後者の方が対数尤度比の計算コストを
  決める。この実測に基づき `MAX_CANDIDATES_PER_SENSOR`（近い順に採用する候補数の
  上限。既定 8）を設け、収まりきらない稀な悪条件（多数の柱間がほぼ同一直線上に
  並ぶ姿勢）から予算を守る安全弁にした（同定数のコメント参照。実測では上限に
  達しないことがほとんどだった）。

  典型（実測 2〜3 候補/センサ、4 センサで 10 個/周期）: 乗算 約 8,840・除算 約 170。
  168 MHz 級では制御周期 168,000 サイクルの **1 割前後**（480 MHz 級なら数%）。
  上限（`MAX_CANDIDATES_PER_SENSOR`=8 に達した場合、4 センサで 32 個/周期）:
  乗算 約 28,300・除算 約 540。168 MHz 級では **2〜4 割程度**まで増えうる
  （480 MHz 級なら 1 割未満）。

  **実機実装では、d_hat(w=1)/d_hat(w=0) の再計算 6 回を、`obs_model` が候補を距離順に
  並べ替えた 1 回の走査（最も近い候補が w=1 の予測、その次が w=0 の予測）に、
  J の数値微分をスラブ法の解析的微分（本モジュールの `_predict_one_sensor` が使う
  交点の式を x, y, theta で直接微分する。除算 1 回の式なので微分も閉形式で書ける）に
  置き換えることで、1 候補あたりの乗算を 1 桁減らせる見込みである。**
  この置き換えは今回のタスク範囲外（正しさが先。`docs/RESEARCH_PLAN.md` §12-9）とし、
  ここに設計メモとして残す。

  地図の常駐 RAM: 本 Python 実装は取り回しのため `log_odds_v`/`log_odds_h` を
  float32 で保持している（544 個 × 4 バイト = **2176 バイト**）。マイコン実装では
  1 バイト量子化（int8, ±127）で **544 バイト**に収まる（`quantize_log_odds`/
  `dequantize_log_odds` がこの往復を実装し、`tests/test_wall_belief.py` が量子化しても
  `declare()` の結果が変わらないことを固定する）。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from classic.geometry import Rect
from classic.maze_map import Direction, MazeMap, WallState
from classic.obs_model import (
    POST_SIZE_M,
    SEARCH_MARGIN_M,
    WALL_THICKNESS_M,
    _cells_radius,
    _local_obstacles,
    _ray_rect_hit,
    _rotate,
    _sensor_local_ray,
)
from mouse.params import RobotParams

__all__ = [
    "SIGMA_SENSOR_DEFAULT",
    "R_MAX_DEFAULT",
    "L_MAX_DEFAULT",
    "T_WALL_DEFAULT",
    "T_OPEN_DEFAULT",
    "FD_STEP_XY_DEFAULT",
    "FD_STEP_THETA_DEFAULT",
    "MAX_CANDIDATES_PER_SENSOR",
    "Q_SCALE",
    "declare_state",
    "quantize_log_odds",
    "dequantize_log_odds",
    "WallBelief",
]


# ============================================================================
# 既定値（モジュール docstring の設計判断 5・6 参照。すべて exp_034 で対照により校正する）
# ============================================================================
# ==========================================================================
# 🔴 検収で分かった、姿勢の不確かさが効く範囲（2026-08-23 教授セッションが実測）
#
# sigma^2 = sigma_sensor^2 + J^T P J は「姿勢が怪しいほど観測を軽くする」仕組みだが、
# **分離が保たれている限り、実際には R_max の頭打ちが先に効いて実効値を変えない。**
#
#   exp_033 が測った側方センサの分離（壁ありと壁なしの予測距離の差）中央値 0.187 m で、
#   1 観測の対数尤度比を姿勢の標準偏差ごとに計算すると:
#     1 mm -> 8742 nat / 10 mm -> 173 nat / 30 mm -> 19.4 nat / 100 mm -> 1.75 nat
#   いずれも R_max=0.5 を大きく超えるので、実効値は 0.5 で頭打ちになる。
#   実効値が動き始めるのは姿勢の標準偏差が 200 mm を超えたあたりからで、
#   実走行（exp_031 の実測で位置誤差 最大 12 mm）ではまず起こらない。
#
# **これは欠陥ではなく、狙いどおりである。**分離が大きいとき観測は実際に決定的であり、
# 強く効いてよい。この仕組みが本当に効くのは**分離そのものが縮む場面** ──
# すれすれの入射で J が大きくなるときと、柱が読みを支配する場面（exp_033 で
# 分離の下位 5% の 99.8% がこれ）── である。そこで sigma が膨らみ、観測が軽くなる。
#
# したがって「姿勢の不確かさで一律に割り引く仕組み」と説明してはいけない。
# **退化した場面を狙って軽くする仕組み**である。
# ==========================================================================
SIGMA_SENSOR_DEFAULT: float = 0.001
# センサ・モデルの基礎雑音の標準偏差 [m]。`exp_033` は obs_model の幾何予測が
# MuJoCo の rangefinder と機械精度で一致することを示した（q95 ≈ 7.5e-16 m）ので、
# 幾何そのものには誤差が無い。しかし note_037 §12 決定2によりセンサ劣化（実機の
# 赤外線の量子化・光学ノイズ）は当面モデル化しない方針であり、それでも σ=0 では
# 姿勢が完全に既知（P=0）のとき尤度比が発散してしまう。実機の安価な赤外線測距
# センサの分解能・量子化幅のオーダー（1mm 程度）を下限雑音の当たりとして暫定的に
# 採用する。🔴 実測（センサ劣化フックを有効化した測定）で校正するまでの暫定値。

R_MAX_DEFAULT: float = 0.5
# 1 回の観測が寄与する対数尤度比の上限 [nat]（規則1）。L_MAX_DEFAULT=12.7 に対し
# 「最大確信度の観測が連続しても 26 回弱かかる」水準に選んだ。外れ値 1 つで
# 信念が一気に飽和しないことと、note_037 §19-2「薄い証拠でも数十回積めば効く」を
# 両立させる値として暫定的に採用する。

L_MAX_DEFAULT: float = 12.7
# 対数オッズの上限 [nat]（規則3）。1 バイト量子化(int8, ±127)にちょうど収まるよう、
# 1 LSB = 0.1 nat とすると 127*0.1 = 12.7 になるよう選んだ（Q_SCALE 参照）。
# exp(12.7) ≈ 3.3e5 であり、実務上十分な確信度を表現できる。

T_WALL_DEFAULT: float = 2.0
# 「壁」を宣言する対数オッズのしきい値 [nat]。尤度比 e^2 ≈ 7.4 倍で認める
# （低い確信度で良い）。note_037 の非対称方針（規則4）の「壁」側。

T_OPEN_DEFAULT: float = 8.0
# 「開通」を宣言する対数オッズのしきい値 [nat]（絶対値。l_w < -T_OPEN_DEFAULT で宣言）。
# 尤度比 e^8 ≈ 2981 倍という高い確信度を要求する。note_033 の出口条件
# 「真は壁なのに開通と判定した数 = 0」を推定器の設計へ写したもの（規則4）。
# 🔴 T_WALL_DEFAULT・T_OPEN_DEFAULT はいずれも暫定値。exp_034 の主判定量
# （致命的な誤りの数）で校正する。

MAX_CANDIDATES_PER_SENSOR: int = 8
# 1 センサ・1 制御周期あたりに実際に仮説検定する柱間候補の上限（近い順）。
# 🔴 実測で判明: 何も既知でない開けた迷路（16x16、外周のみ壁）でロボットが
# 境界から離れた位置にいると、探索窓に入る候補が 1 センサあたり最大 98 個
# （4 センサで 392 個/周期）に達する。1 候補あたり約 884 乗算（本ファイル冒頭の
# マイコン予算参照）かかる仮説検定を 392 回行うと、1 周期で 30 万乗算を超え、
# 168MHz 級 MCU の 1 制御周期(168,000 サイクル)を乗算だけで数倍超えてしまう。
# 候補を「光線に沿って近い順」に上限 8 個へ絞ることで、1 周期の候補数を
# 高々 4*8=32 個（乗算 約28,000）に抑える。遠い候補を切り捨てても情報は失われない
# ── ロボットは動き続けるので、切り捨てた遠い候補は距離が縮まった次の周期以降に
# 改めて候補として現れる（note_037 §19-2「制御周期ごとに連続して観測を積む」の
# 前提どおり、機会は繰り返し来る）。近い候補を優先するのは、実測との一致
# （尤度比）が最も大きく動くのが「実際に見ている可能性が高い」近傍の柱間で
# あるため（遠い候補は cutoff 付近でどのみち証拠が薄い）。

FD_STEP_XY_DEFAULT: float = 1e-4   # J の数値微分（x, y）の刻み [m]（中心差分）
FD_STEP_THETA_DEFAULT: float = 1e-4  # J の数値微分（theta）の刻み [rad]（中心差分）
# 刻みの選び方: obs_model の予測は概ね 0.03〜0.3m のオーダーで、倍精度演算のもとで
# ほぼ機械精度に近い決定論的関数（exp_033: q95 ≈ 7.5e-16m）なので、1e-4 という
# 「小さいが浮動小数点の丸め誤差には埋もれない」刻みで中心差分を取れば、局所的な
# 線形近似として十分な精度が出る（刻みをさらに小さくしても丸め誤差が増えるだけで
# 精度は上がらない領域）。

Q_SCALE: float = 127.0 / L_MAX_DEFAULT  # = 10.0（1 LSB = 0.1 nat）

_SQRT2 = math.sqrt(2.0)
_MIN_PROB = 1e-300  # log(0) を避けるための下限確率
_SATURATION_TOL = 1e-9  # z がこの値だけ cutoff に近ければ「打ち切られた観測」とみなす
_BACKGROUND_THRESHOLD = 0.0  # 内部の点推定に使う閾値（declare() の非対称しきい値とは別物）


# ============================================================================
# 宣言（読み出し。決めるのはここだけ）
# ============================================================================
def declare_state(log_odds: float, t_wall: float = T_WALL_DEFAULT,
                   t_open: float = T_OPEN_DEFAULT) -> WallState:
    """対数オッズ 1 個から WallState を宣言する（非対称しきい値。規則4）。"""
    if log_odds > t_wall:
        return WallState.WALL
    if log_odds < -t_open:
        return WallState.OPEN
    return WallState.UNKNOWN


def _declare_array(log_odds: np.ndarray, t_wall: float, t_open: float) -> np.ndarray:
    """`declare_state` の配列版（`MazeMap.v_walls`/`h_walls` と同じ int8 の 3 値）。"""
    out = np.full(log_odds.shape, int(WallState.UNKNOWN), dtype=np.int8)
    out[log_odds > t_wall] = int(WallState.WALL)
    out[log_odds < -t_open] = int(WallState.OPEN)
    return out


# ============================================================================
# 1 バイト量子化（マイコン実装の常駐表現。往復しても declare() の結果を変えない）
# ============================================================================
def quantize_log_odds(log_odds: np.ndarray, l_max: float = L_MAX_DEFAULT) -> np.ndarray:
    """対数オッズ配列を int8（±127）へ量子化する。1 LSB = l_max/127 nat。"""
    scale = 127.0 / l_max
    q = np.clip(np.round(log_odds.astype(np.float64) * scale), -127, 127)
    return q.astype(np.int8)


def dequantize_log_odds(q: np.ndarray, l_max: float = L_MAX_DEFAULT) -> np.ndarray:
    """`quantize_log_odds` の逆変換。float32 の対数オッズへ戻す。"""
    scale = 127.0 / l_max
    return (q.astype(np.float32)) / np.float32(scale)


# ============================================================================
# 柱間候補の探索（obs_model と同じ探索窓・同じ Rect の作り方だが、添字を保持する）
# ============================================================================
def _wall_candidates(ox: float, oy: float, cell_size: float, cells_radius: int,
                      width: int, height: int):
    """センサ原点 (ox,oy) の近傍にある柱間候補を (kind, i, j, Rect) の列で返す。

    `classic.obs_model._local_obstacles` と同じ探索窓（走査範囲。同モジュールの
    `_cells_radius` docstring参照）を使うが、Rect だけでなく元の柱間添字
    （`classic.maze_map.MazeMap` の v_walls/h_walls と同じ規約）も保持する点が
    異なる（obs_model 側は距離予測にしか使わないため添字を捨てているが、壁信念の
    更新にはどの柱間かの情報が要る）。

    外周（構造上確定済みの柱間）は候補に含めない（モジュール docstring の設計判断1）。
    柱（post）も不確かさを持たない構造物なので候補にしない。
    """
    i_center = int(math.floor(ox / cell_size))
    j_center = int(math.floor(oy / cell_size))
    i_lo, i_hi = i_center - cells_radius, i_center + cells_radius
    j_lo, j_hi = j_center - cells_radius, j_center + cells_radius

    half_wall = WALL_THICKNESS_M / 2.0
    half_post = POST_SIZE_M / 2.0
    half_run = cell_size / 2.0 - half_post

    out = []
    # 縦壁: i(格子線) in [1, width-1]（外周 0,width は除く）, j(区画行) in [0, height-1]
    for i in range(max(i_lo, 1), min(i_hi, width - 1) + 1):
        for j in range(max(j_lo, 0), min(j_hi, height - 1) + 1):
            px = float(i) * cell_size
            py = float(j) * cell_size + cell_size / 2.0
            out.append(("v", i, j, Rect(px, py, half_wall, half_run)))

    # 横壁: i(区画列) in [0, width-1], j(格子線) in [1, height-1]（外周 0,height は除く）
    for i in range(max(i_lo, 0), min(i_hi, width - 1) + 1):
        for j in range(max(j_lo, 1), min(j_hi, height - 1) + 1):
            px = float(i) * cell_size + cell_size / 2.0
            py = float(j) * cell_size
            out.append(("h", i, j, Rect(px, py, half_run, half_wall)))

    return out


def _predict_one_sensor(x: float, y: float, theta: float, v_walls: np.ndarray,
                         h_walls: np.ndarray, sensor_spec, cutoff: float,
                         cell_size: float, cells_radius: int) -> float:
    """1 本のセンサだけの予測距離を返す。

    `classic.obs_model.predict_ranges_with_diagnostics` のセンサ 1 本ぶんの内部
    ループをそのまま再利用したもの（`obs_model` の私的関数 `_local_obstacles`・
    `_ray_rect_hit`・`_sensor_local_ray`・`_rotate` を呼ぶだけで、光線・矩形交差の
    実装を 2 か所に増やさない — 挙動が obs_model と食い違うリスクを構造的に無くす）。
    4 本まとめて予測する `predict_ranges` と違い、壁信念の更新では「仮説ごとに
    1 本のセンサだけ」を何度も再評価する（`WallBelief.update` 参照）ので、
    無駄な残り 3 本の計算を省くために切り出してある。
    """
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    ox_local, oy_local, dx_local, dy_local, h = _sensor_local_ray(
        sensor_spec['pos'], sensor_spec['zaxis'])
    ox_world, oy_world = _rotate(ox_local, oy_local, cos_t, sin_t)
    ox_world += x
    oy_world += y
    dx_world, dy_world = _rotate(dx_local, dy_local, cos_t, sin_t)

    obstacles = _local_obstacles(v_walls, h_walls, ox_world, oy_world, cell_size,
                                  cells_radius, center_goal=True, include_posts=True)

    radius = cutoff + SEARCH_MARGIN_M
    best_t: Optional[float] = None
    for rect in obstacles:
        if abs(rect.cx - ox_world) > radius + rect.hx:
            continue
        if abs(rect.cy - oy_world) > radius + rect.hy:
            continue
        t = _ray_rect_hit(ox_world, oy_world, dx_world, dy_world, rect)
        if t is not None and (best_t is None or t < best_t):
            best_t = t

    if best_t is None:
        return cutoff
    d = best_t / h
    return d if d < cutoff else cutoff


def _jacobian_fd(x: float, y: float, theta: float, v_walls: np.ndarray,
                  h_walls: np.ndarray, sensor_spec, cutoff: float, cell_size: float,
                  cells_radius: int, step_xy: float, step_theta: float) -> np.ndarray:
    """d_hat の姿勢 (x,y,theta) に関する勾配 J を中心差分で求める（PREREG §1-1 が
    許容する「J は数値微分でよい」を採る。刻みは `FD_STEP_XY_DEFAULT`/
    `FD_STEP_THETA_DEFAULT` のモジュール docstring 参照）。"""
    def f(xx, yy, tt):
        return _predict_one_sensor(xx, yy, tt, v_walls, h_walls, sensor_spec, cutoff,
                                    cell_size, cells_radius)

    dx = (f(x + step_xy, y, theta) - f(x - step_xy, y, theta)) / (2.0 * step_xy)
    dy = (f(x, y + step_xy, theta) - f(x, y - step_xy, theta)) / (2.0 * step_xy)
    dt = (f(x, y, theta + step_theta) - f(x, y, theta - step_theta)) / (2.0 * step_theta)
    return np.array([dx, dy, dt], dtype=np.float64)


# ============================================================================
# 対数尤度比（飽和した読みは打ち切り＝上側裾確率として扱う。設計判断2）
# ============================================================================
def _log_survival(cutoff: float, mean: float, sigma: float) -> float:
    """P(D >= cutoff | mean, sigma) の対数（正規分布の上側裾）。
    math.erfc(u) = 2*Phi(-u) より P(D>=cutoff) = 0.5*erfc((cutoff-mean)/(sigma*sqrt2))。"""
    u = (cutoff - mean) / (sigma * _SQRT2)
    s = 0.5 * math.erfc(u)
    return math.log(max(s, _MIN_PROB))


def _log_likelihood_ratio(z: float, d_hat1: float, d_hat0: float, sigma2: float,
                           cutoff: float) -> float:
    """log[p(z|w=1)/p(z|w=0)] を返す。w=1/w=0 で同じ sigma^2 を使う前提
    （モジュール docstring の設計判断参照）なので、非飽和の通常観測では
    正規化定数 log(2*pi*sigma^2) が比を取ると打ち消え、二次形式の差だけで済む
    （指数・対数の呼び出しが要らない ── マイコン予算参照）。"""
    if z >= cutoff - _SATURATION_TOL:
        sigma = math.sqrt(sigma2)
        return _log_survival(cutoff, d_hat1, sigma) - _log_survival(cutoff, d_hat0, sigma)
    return ((z - d_hat0) ** 2 - (z - d_hat1) ** 2) / (2.0 * sigma2)


def _wall_slot(x: int, y: int, direction: Direction) -> Tuple[str, int, int]:
    """(区画, 方位) から (kind, i, j) を返す。`classic.maze_map.MazeMap._wall_index`
    と同じ添字の付け方（同モジュールの規約に合わせるための写し。読み取り専用の
    小さな添字算術であり、`classic/maze_map.py` は 1 行も変更していない）。"""
    if direction is Direction.E:
        return "v", x + 1, y
    if direction is Direction.W:
        return "v", x, y
    if direction is Direction.N:
        return "h", x, y + 1
    if direction is Direction.S:
        return "h", x, y
    raise ValueError(f"未知の方位: {direction!r}")


# ============================================================================
# 壁の信念（本体）
# ============================================================================
class WallBelief:
    """柱間 544 か所の対数オッズを持つ信念地図。

    使い方:
        wb = WallBelief(width, height, params)
        while True:
            pose = est.pose            # PoseEstimator の点推定
            cov = est.covariance       # 3x3 共分散
            ranges = obs[:n_sensors]   # 距離センサ4本の実測値
            wb.update(pose, cov, ranges)
        maze_map = wb.to_maze_map()    # 既存の経路計算へそのまま渡せる
    """

    def __init__(
        self,
        width: int,
        height: int,
        params: Optional[RobotParams] = None,
        *,
        sigma_sensor: float = SIGMA_SENSOR_DEFAULT,
        r_max: float = R_MAX_DEFAULT,
        l_max: float = L_MAX_DEFAULT,
        fd_step_xy: float = FD_STEP_XY_DEFAULT,
        fd_step_theta: float = FD_STEP_THETA_DEFAULT,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width, height は正の整数である必要があります")
        self.width = int(width)
        self.height = int(height)
        self.params = params if params is not None else RobotParams()
        self.sigma_sensor = float(sigma_sensor)
        self.r_max = float(r_max)
        self.l_max = float(l_max)
        self.fd_step_xy = float(fd_step_xy)
        self.fd_step_theta = float(fd_step_theta)

        # 対数オッズ本体（`classic.maze_map.MazeMap` と同じ形状規約: v_walls は
        # (width+1, height)、h_walls は (width, height+1)）。float32 で持つ
        # （マイコン実装の常駐表現は int8。モジュール docstring の RAM 見積もり参照）。
        self.log_odds_v = np.zeros((self.width + 1, self.height), dtype=np.float32)
        self.log_odds_h = np.zeros((self.width, self.height + 1), dtype=np.float32)

        # 外周は確定として初期化する（`MazeMap.__init__` と同じ規約）。
        self.log_odds_v[0, :] = self.l_max
        self.log_odds_v[self.width, :] = self.l_max
        self.log_odds_h[:, 0] = self.l_max
        self.log_odds_h[:, self.height] = self.l_max

        # 内部の点推定用バッファ（規則1の仮説検定で「他の柱間」を埋めるのに使う。
        # ループ内で動的確保しないよう、update() の中で毎回作り直さず使い回す）。
        self._bg_v = np.zeros_like(self.log_odds_v, dtype=bool)
        self._bg_h = np.zeros_like(self.log_odds_h, dtype=bool)

        self._n_sensors = len(self.params.sensors)

    # ------------------------------------------------------------------
    # 更新（制御周期ごとに 1 回）
    # ------------------------------------------------------------------
    def update(self, pose: Tuple[float, float, float], cov: np.ndarray,
               ranges) -> None:
        """姿勢の信念（点推定 `pose`・共分散 `cov`）と距離センサ 4 本の実測値
        `ranges`（`params.sensors` と同じ並び。`classic.pose.PoseEstimator.pose`/
        `covariance` および `mouse.sim.MouseSim.observation()` の距離部分と
        そのまま組み合わせられる）を受け取り、見ている可能性のある柱間それぞれの
        対数オッズへ 1 回ぶんの証拠を足し込む。

        任意の姿勢で成り立つ（区画の中心・軸並行に特殊化しない。斜め探索の前提）。
        """
        x, y, theta = pose
        cutoff = self.params.sensor_cutoff
        cell_size = self.params.cell_size
        cells_radius = _cells_radius(cutoff, cell_size)

        # 内部の点推定（宣言のしきい値 T_wall/T_open とは別物。モジュール docstring
        # 設計判断1）: 対数オッズ > 0 を「壁」とみなして、他の柱間の背景を作る。
        np.greater(self.log_odds_v, _BACKGROUND_THRESHOLD, out=self._bg_v)
        np.greater(self.log_odds_h, _BACKGROUND_THRESHOLD, out=self._bg_h)

        for s_idx, spec in enumerate(self.params.sensors):
            z = float(ranges[s_idx])
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            ox_local, oy_local, dx_local, dy_local, h_elev = _sensor_local_ray(
                spec['pos'], spec['zaxis'])
            ox_w, oy_w = _rotate(ox_local, oy_local, cos_t, sin_t)
            ox_w += x
            oy_w += y
            dx_w, dy_w = _rotate(dx_local, dy_local, cos_t, sin_t)

            candidates = _wall_candidates(ox_w, oy_w, cell_size, cells_radius,
                                           self.width, self.height)

            # 幾何的に光線と交わり、かつ cutoff+マージン以内の候補だけを残し、
            # 近い順に MAX_CANDIDATES_PER_SENSOR 個へ絞る（モジュール docstring
            # の設計判断1・MAX_CANDIDATES_PER_SENSOR のコメント参照）。
            reachable = []
            for kind, i, j, rect in candidates:
                t = _ray_rect_hit(ox_w, oy_w, dx_w, dy_w, rect)
                if t is None:
                    continue  # この姿勢からはこの柱間の面が視線と幾何的に交わらない
                d_w = t / h_elev
                if d_w > cutoff + SEARCH_MARGIN_M:
                    continue  # 探索窓の外縁で幾何的に交わっただけ（cutoff より遠い）
                reachable.append((d_w, kind, i, j, rect))
            reachable.sort(key=lambda item: item[0])

            for d_w, kind, i, j, rect in reachable[:MAX_CANDIDATES_PER_SENSOR]:
                arr = self._bg_v if kind == "v" else self._bg_h
                idx = (i, j)
                original = bool(arr[idx])

                # w=1（この柱間だけ壁）の仮説
                arr[idx] = True
                d_hat1 = _predict_one_sensor(x, y, theta, self._bg_v, self._bg_h, spec,
                                              cutoff, cell_size, cells_radius)
                jac = _jacobian_fd(x, y, theta, self._bg_v, self._bg_h, spec, cutoff,
                                    cell_size, cells_radius, self.fd_step_xy,
                                    self.fd_step_theta)

                # w=0（この柱間だけ開通）の仮説
                arr[idx] = False
                d_hat0 = _predict_one_sensor(x, y, theta, self._bg_v, self._bg_h, spec,
                                              cutoff, cell_size, cells_radius)

                arr[idx] = original  # この柱間の判断はまだ決めない。背景を元に戻す

                sigma2 = self.sigma_sensor ** 2 + float(jac @ cov @ jac)
                if sigma2 <= 0.0:
                    sigma2 = self.sigma_sensor ** 2

                llr = _log_likelihood_ratio(z, d_hat1, d_hat0, sigma2, cutoff)
                llr = max(-self.r_max, min(self.r_max, llr))

                target = self.log_odds_v if kind == "v" else self.log_odds_h
                new_val = float(target[idx]) + llr
                new_val = max(-self.l_max, min(self.l_max, new_val))
                target[idx] = np.float32(new_val)

    # ------------------------------------------------------------------
    # 読み出し（宣言。決めるのはここだけ）
    # ------------------------------------------------------------------
    def to_maze_map(self, t_wall: float = T_WALL_DEFAULT,
                     t_open: float = T_OPEN_DEFAULT) -> MazeMap:
        """現在の信念を 3 値（`classic.maze_map.MazeMap`）へ変換する。
        既存の経路計算（歩数マップ・最短経路）がそのまま使える。"""
        mm = MazeMap(self.width, self.height)
        mm.v_walls[:, :] = _declare_array(self.log_odds_v, t_wall, t_open)
        mm.h_walls[:, :] = _declare_array(self.log_odds_h, t_wall, t_open)
        return mm

    def log_odds_at(self, x: int, y: int, direction: Direction) -> float:
        """区画 (x,y) の direction 側の柱間の対数オッズを返す。"""
        kind, i, j = _wall_slot(x, y, direction)
        arr = self.log_odds_v if kind == "v" else self.log_odds_h
        return float(arr[i, j])

    def declare_at(self, x: int, y: int, direction: Direction,
                    t_wall: float = T_WALL_DEFAULT,
                    t_open: float = T_OPEN_DEFAULT) -> WallState:
        """区画 (x,y) の direction 側の柱間を宣言する（非対称しきい値。規則4）。"""
        return declare_state(self.log_odds_at(x, y, direction), t_wall, t_open)
