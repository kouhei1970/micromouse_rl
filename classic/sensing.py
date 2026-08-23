"""
classic/sensing.py
================
距離センサの生値（`mouse/sim.py` の `MouseSim.observation()` が返す観測ベクトルの
先頭 n 要素）から、機体の前方・左方・右方に壁があるかどうかを判定する。

【センサ配置の確定（`mouse/params.py` の既定 4 本構成、`docs/ROBOT_SPEC.md` §5.1）】

  名称  方位角(+x=機体正面基準)  役割
  LF    0.00°                    正面をほぼ真っ直ぐ見る → 前方判定に使う
  RF    0.00°                    同上（右寄りに取り付け）→ 前方判定に使う
  LS    +75.96°                  ほぼ左横（やや前方寄り）を見る → 左方判定に使う
  RS    −75.96°                  ほぼ右横（やや前方寄り）を見る → 右方判定に使う

`observation()` のセンサ並びは `params.sensors` の登録順そのまま
（[LF, LS, RF, RS, ...] が既定。r6 で 6 本→4 本になった経緯があるため、
並び順は observation() 側のコメントどおり固定値でなく `params.sensors` から
名前で引く。インデックスのハードコードはしない）。

【しきい値の決め方（実測に基づく。決め打ちではない）】

しきい値は、区画中心（迷路座標変換式 `mouse/mjcf.py` 準拠）を中心に
**位置ずれ ±15mm・方位ずれ ±4°の格子（7×7×5=245 通り）**で機体姿勢を振り、
「壁あり」「壁なし（隣の区画１つ分だけ壁が開いている、最も紛らわしいケース）」の
双方でシミュレータを実際に動かして得た距離分布から選んだ
（測定に使ったスクリプトの再現手順は `tests/test_classic_sensing.py` を参照。
同テストが実行時に同じ数値を実測し直して報告する）。

実測結果（5×5 迷路、区画 (2,2) 中心、245 姿勢サンプル）:

  前方 (LF・RF、値は左右対称なので一致):
    壁あり     range = [0.0337, 0.0688] m   (max 0.0688)
    壁なし     range = [0.2142, 0.2493] m   (min 0.2142)
    → 隙間 0.0688〜0.2142 m（0.15 m 近い幅）で **重なりゼロ**。

  側方 (LS/RS、片側ずつ壁を開けて測定):
    壁あり     range = [0.0488, 0.0885] m   (max 0.08851、99%点でも 0.08851)
    壁なし     range = [0.0787, 0.2655] m   (min 0.07873)
    → **壁あり最大 0.08851 > 壁なし最小 0.07873 で重なりが実在する**
      （245 サンプル中、壁ありの 47 個・壁なしの 7 個がこの重なり帯
      [0.07873, 0.08851] に入る。原因はセンサが横に約76°傾いており、
      区画内の位置・方位のずれが柱への視線角に直結するため）。
      この重なり帯を握りつぶさず、AMBIGUOUS として明示的に返す。

しきい値は上記の「重ならない」領域の内側に、実測の最悪値へ寄せて設定する
（＝安全側マージンを最小化しすぎない。実測の生の分離点をそのまま使う）。

🔴 **±15mm・±4° は「較正のときに姿勢を振った範囲」であって、判定が成り立つ
限界を測ったものではない**（2026-08-23 ユーザ指摘により明記）。この外側は
測っていないだけで、壊れることが分かっているわけではない。
**この数字を設計の拘束条件として使わないこと。**

実際に測れている事実は別にある ── 側方センサには壁あり／なしの距離が重なる帯
[0.07873, 0.08851] m が実在し、245 姿勢のうち 54 個がこの帯に入る。
**重なりがある以上、どこにしきい値を置いても原理的に切り分けられない。**
本モジュールの「ある瞬間に、区画中心にいる前提で、しきい値で判定する」やり方は
そこで頭打ちになる。姿勢を使って応答を予測し照合する観測モデル
（`research_notes/note_037_probabilistic_localization.md` §6）へ移す理由はここにある。
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from mouse.params import RobotParams


class WallState(Enum):
    """壁判定の結果。3 値。"""
    WALL = "wall"            # 壁がある
    CLEAR = "clear"          # 壁がない
    AMBIGUOUS = "ambiguous"  # どちらとも判定できない（握りつぶさず明示する）


# ------------------------------------------------------------------
# しきい値 [m]（モジュール先頭の docstring に実測根拠を記載）
# ------------------------------------------------------------------
# 前方 (LF/RF): 壁あり実測 max=0.0688、壁なし実測 min=0.2142。重なり無し。
#   実測値ぴったりに置くと浮動小数点の丸めやノイズで壁あり/壁なしの端の
#   サンプルが AMBIGUOUS に落ちてしまうため、実測の分離点そのものを
#   しきい値として採用しつつ、AMBIGUOUS 帯を明示的に持たせる
#   （実測範囲の外側かつ隙間の中に 2 本のしきい値を置く）。
FRONT_WALL_MAX: float = 0.09    # これ未満なら WALL（壁あり実測 max 0.0688 に対し+31%の余裕）
FRONT_CLEAR_MIN: float = 0.19   # これ以上なら CLEAR（壁なし実測 min 0.2142 の内側直前）

# 側方 (LS/RS): 壁あり実測 max=0.08851、壁なし実測 min=0.07873。**重なりあり**。
#   重なり帯 [0.07873, 0.08851] を挟んで、実測で片方にしか出なかった値の
#   すぐ外側にしきい値を置く（＝実測で確認できた分離点をそのまま使う）。
SIDE_WALL_MAX: float = 0.077    # これ未満なら WALL（壁なし実測 min 0.07873 の直前）
SIDE_CLEAR_MIN: float = 0.090   # これ以上なら CLEAR（壁あり実測 max 0.08851 の直後）


def _sensor_index(params: RobotParams, name: str) -> int:
    """params.sensors から名前でセンサのインデックスを引く（並び順のハードコード禁止）。"""
    for i, s in enumerate(params.sensors):
        if s.get('name') == name:
            return i
    raise ValueError(
        f"センサ '{name}' が params.sensors に見つかりません "
        f"(登録済み: {[s.get('name') for s in params.sensors]})"
    )


def _classify(dist: float, wall_max: float, clear_min: float) -> WallState:
    """1 方向分の距離 [m] を 3 値判定する。wall_max < clear_min が前提。"""
    if dist < wall_max:
        return WallState.WALL
    if dist >= clear_min:
        return WallState.CLEAR
    return WallState.AMBIGUOUS


@dataclass(frozen=True)
class WallSensing:
    """前方・左方・右方の壁判定結果と、判定に使った生の距離 [m]（診断用）。"""
    front: WallState
    left: WallState
    right: WallState
    front_dist: float
    left_dist: float
    right_dist: float

    @property
    def any_ambiguous(self) -> bool:
        """3 方向のいずれかが AMBIGUOUS ならば True（呼び出し側が握りつぶさず扱うためのフラグ）。"""
        return WallState.AMBIGUOUS in (self.front, self.left, self.right)


def sense_walls(obs: np.ndarray, params: Optional[RobotParams] = None) -> WallSensing:
    """observation() の観測ベクトルから前方・左方・右方の壁の有無を判定する。

    区画の中心付近（位置ずれ ±15mm・方位ずれ ±4° 程度）で呼ぶことを前提に
    しきい値を決めている（モジュール docstring 参照）。この範囲の外で呼んだ
    ときの精度は**測っていない**（壊れると分かっているわけではない）。
    この数字を設計の拘束条件として使わないこと。

    Args:
        obs: MouseSim.observation() が返す観測ベクトル（先頭が距離センサ）。
        params: RobotParams。None なら既定値（センサ名の対応付けに使うのみ、
            物理定数は参照しない）。

    Returns:
        WallSensing（front/left/right の 3 値判定 + 生の距離）。
    """
    p = params if params is not None else RobotParams()

    i_lf = _sensor_index(p, 'LF')
    i_rf = _sensor_index(p, 'RF')
    i_ls = _sensor_index(p, 'LS')
    i_rs = _sensor_index(p, 'RS')

    d_lf = float(obs[i_lf])
    d_rf = float(obs[i_rf])
    d_ls = float(obs[i_ls])
    d_rs = float(obs[i_rs])

    # 前方は LF・RF の悪い方（小さい方＝壁に近い方）を採る。
    # 「安全側に倒す」原則: 片方だけが壁を検知していれば WALL 寄りに判定する
    # （どちらか一方でも壁ありと言っていれば衝突回避を優先する）。
    front_dist = min(d_lf, d_rf)
    left_dist = d_ls
    right_dist = d_rs

    front = _classify(front_dist, FRONT_WALL_MAX, FRONT_CLEAR_MIN)
    left = _classify(left_dist, SIDE_WALL_MAX, SIDE_CLEAR_MIN)
    right = _classify(right_dist, SIDE_WALL_MAX, SIDE_CLEAR_MIN)

    return WallSensing(
        front=front, left=left, right=right,
        front_dist=front_dist, left_dist=left_dist, right_dist=right_dist,
    )
