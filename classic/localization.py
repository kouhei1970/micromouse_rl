"""
classic/localization.py
================
壁センサによる区画ごとの位置補正（S2）。
`research_notes/note_030_classical_rebuild_plan.md` §1・§2 L1「推測航法（車輪＋
ジャイロ）＋壁センサによる区画ごとの補正」の「壁センサによる補正」を実装する。

【なぜ要るか（実測で分かっている問題、note_030 §S2 任務指示）】
`classic/motion.py` の推測航法だけで L4 を通すと、次の 2 つの実測不具合が出る:

  1. maze_41038: 10 区画の連続直進で横方向ドリフトが 26〜28mm 蓄積し、
     直後のその場旋回で機体外形が壁をかすめて衝突する。
  2. maze_41049: 側方センサが較正済み AMBIGUOUS 帯のすぐ外側で誤判定し、
     前方確定の WALL と側方経由の CLEAR が同一区画で交互に上書きされ、
     旋回を繰り返す（`classic/sensing.py` の較正前提 位置ずれ±15mm・
     方位ずれ±4° を推測航法の誤差が超えたときに起こる）。

実機は推測航法だけで走っていない。区画の中心で壁センサを読み、その区画内での
横位置・前後位置のずれを推定して推測航法の状態を打ち消す方向へ補正している。
本モジュールはそれを再現する。

【幾何モデル（実測ではなく設計値からの導出。根拠は docs/ROBOT_SPEC.md §5.1・§8.2）】

区画は正方形（1 辺 `params.cell_size`）で、壁は区画境界のグリッド線を中心に
厚さ `WALL_THICKNESS` で立つ。したがって壁の「区画側の面」は区画中心から
`cell_size/2 - WALL_THICKNESS/2` だけ離れている（`wall_standoff()`）。これは
前後左右どちらの壁でも同じ値になる（区画が正方形のため）。

側方センサ (LS/RS) は機体中心から見て前方寄りに角度 θ（`docs/ROBOT_SPEC.md`
§5.1、実測ではなく取り付け設計値の zaxis から算出）で外を向いている。機体が
区画の中心線から横ずれ `e`（正=左）だけずれているとき、センサが壁に当たる
までの距離 `d` は次の関係を満たす（壁面は進行方向に平行な平面という近似。
壁の長手方向の途中を見ている限り厳密に成り立つ）:

    (センサの機体内 横位置) + d * (センサ方向ベクトルの横成分) = 壁の設計横位置 - e

これを `e` について解いたのが `estimate_lateral_offset()`。両側の壁が見えて
いるときは、この 2 本の式を足し合わせて `壁の設計位置`（cell_size 由来の
定数）への依存を消した式を使う（差分ベースの推定。**壁の設計位置の較正誤差に
対して頑健**という意味で「最も信頼できる」）。片側だけのときは壁の設計位置
そのものに依存する式を使う（`任務指示` の (a) が要求する 2 通りをそのまま
実装したもの）。

前方センサ (LF/RF) についても同じ考え方を「前後」方向に適用したのが
`estimate_forward_offset()`（任務指示 (b)）。

【補正の入れ方（2 通り。詳細は各関数のコメント）】

  (a) 横位置補正: 区画の中心（`_on_stationary`）で計算した横ずれ推定を、
      **次の 1 区画の直進コマンドの目標方位バイアス**として渡す
      (`Localizer.lateral_bias_for_forward` → `CellMotionController.
      bias_target_heading`)。1 区画かけて横ずれを解消する方向へ舵を切る。
      毎区画で読み直すので、蓄積したヨー推定誤差がその都度打ち消される。

  (b) 前後位置補正: **直進コマンドの実行中、毎ティック**前方センサを読み、
      壁が確定 (WALL) したら残距離 (`remaining`) を前方センサの実測値で
      上書きする（`Localizer.correct_forward_remaining`、
      `CellMotionController.update()` の FORWARD 分岐から呼ばれる）。
      壁に近づくほど推測航法よりセンサの実測が正確になる、という
      実機の「壁に頼って止まる」動作の再現。

【補正の可視性・無効化（任務指示 🔴 の要求）】
  - 実際に適用した補正は `Localizer.events`（`CorrectionEvent` のリスト）に
    記録する。何も補正しなかったティック・区画は記録しない
    （「補正が効いたかどうか」が外から分かるようにする）。
  - `Localizer(enabled=False)` で完全に無効化できる。無効化すると
    `lateral_bias_for_forward()`/`correct_forward_remaining()` は
    何も計算せず（`sense_walls` すら呼ばず）常に無補正の値を返す。
    `CellMotionController(localizer=None)`（=既定）と `CellMotionController
    (localizer=Localizer(enabled=False))` は完全に同じ経路を通る。
  - AMBIGUOUS・CLEAR な読みからは推定しない（`WallState.WALL` の判定が
    出た方向のみ使う。距離の意味が保証されるのは「壁がある」と確定した
    ときだけであり、握りつぶさず「補正しない」を選ぶ設計は `classic/
    sensing.py`・`classic/explorer.py` と同じ方針）。

🔴 真値（`mouse.sim.MouseSim.privileged_pose()`/`privileged_velocity()`）は
一切使わない。本モジュールが受け取るのは `classic.sensing.WallSensing`
（壁センサの生値から出した判定・距離）と `RobotParams`（設計値）だけである。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from classic.sensing import WallSensing, WallState, sense_walls
from mouse.params import RobotParams

# ------------------------------------------------------------------
# 迷路の幾何定数（docs/ROBOT_SPEC.md §8.2。mouse/mjcf.py の WALL_THICKNESS と
# 同じ値。classic/ は mouse.params 以外の mouse.* に依存させない既存の作法
# （classic/motion.py・sensing.py・explorer.py 参照）に合わせ、mjcf.py からの
# import ではなく同じ数値をここに明記する）。
# ------------------------------------------------------------------
WALL_THICKNESS: float = 0.012  # 壁厚 [m]

# 素朴な幾何計算 (cell_size/2 - WALL_THICKNESS/2) と、実際にシミュレータで
# 測った壁面までの距離との差 [m]（実測に基づく較正値。しきい値を実測で決める
# classic/sensing.py と同じ作法）。
#
# 実測手順（tests/test_classic_localization.py の
# test_wall_standoff_matches_measured_geometry が同じ手順で実行時に測り直して
# 報告する）: 1 区画幅のコリドー（左右または前方に壁のみ）の区画中心
# （`MouseSim.full_reset()` 直後、位置ずれ 0 が保証される）で側方センサ
# (LS)・前方センサ(LF) の距離を読み、d*dy_ls + y_ls（側方）/ d*dx_lf + x_lf
# （前方）から壁面までの距離を逆算すると、側方 0.082060m・前方 0.082001m
# （2026-08-19 実測、5x1〜1x12 コリドー、複数区画で再現）が得られ、
# 素朴な計算値 0.084m よりおよそ 1.94mm 小さい。前後・左右で独立に測った
# 値がほぼ一致する（差 0.06mm）ことから、位置ずれ等のノイズではなく
# モデル形状に起因する系統的な定数だと判断できる（原因そのものの追跡は
# 本タスクの範囲外）。両側センサ差分の式（`estimate_lateral_offset` の
# "both" 分岐）はこの定数に依存しないため、この較正は片側だけの推定
# （"left"/"right"/前方）の精度にのみ影響する。
MEASURED_STANDOFF_CORRECTION_M: float = 0.00194


def wall_standoff(params: RobotParams) -> float:
    """区画中心から、壁の「区画側の面」までの距離 [m]（前後左右共通。正方形区画）。
    幾何計算に実測較正（`MEASURED_STANDOFF_CORRECTION_M`）を適用した値。"""
    nominal = params.cell_size / 2.0 - WALL_THICKNESS / 2.0
    return nominal - MEASURED_STANDOFF_CORRECTION_M


# ------------------------------------------------------------------
# センサ幾何（params.sensors から名前で引く。インデックスのハードコード禁止
# ＝ classic/sensing.py の `_sensor_index` と同じ理由・同じ作法）。
# ------------------------------------------------------------------
def _parse_vec3(s: str) -> Tuple[float, float, float]:
    parts = [float(v) for v in s.split()]
    if len(parts) != 3:
        raise ValueError(f"3 要素のベクトル文字列ではありません: {s!r}")
    return parts[0], parts[1], parts[2]


def _sensor_geom(params: RobotParams, name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """センサ名から (機体座標系での位置, 光軸の単位ベクトル) を返す。"""
    for s in params.sensors:
        if s.get('name') == name:
            pos = _parse_vec3(s['pos'])
            zaxis = _parse_vec3(s['zaxis'])
            norm = math.sqrt(sum(c * c for c in zaxis))
            unit = (zaxis[0] / norm, zaxis[1] / norm, zaxis[2] / norm)
            return pos, unit
    raise ValueError(
        f"センサ '{name}' が params.sensors に見つかりません "
        f"(登録済み: {[s.get('name') for s in params.sensors]})"
    )


# ==========================================================================
# (a) 横位置補正: 側方センサ (LS/RS) から中心線からの横ずれを推定する
# ==========================================================================
@dataclass(frozen=True)
class LateralEstimate:
    """横ずれの推定結果。"""

    offset_m: float   # 中心線からの横ずれ推定 [m]。正 = 機体が左（+y機体座標）側へ寄っている
    source: str        # "both"（両側壁、最も信頼できる） / "left" / "right"


def estimate_lateral_offset(sensing: WallSensing, params: RobotParams) -> Optional[LateralEstimate]:
    """側方センサの生値から中心線からの横ずれを推定する。

    任務指示 (a) の 3 通りをそのまま実装する:
      - 両側に壁があるとき: 左右の距離差から横ずれが直接出る（壁の設計位置の
        値そのものへの依存が消える式。最も信頼できる）
      - 片側だけのとき: その壁までの距離と、壁の設計位置(`wall_standoff`)から出す
      - どちらも無いとき: None（呼び出し側は補正しない）

    AMBIGUOUS・CLEAR な側は使わない（距離の意味が保証されるのは
    `classic.sensing.WallState.WALL` と確定したときだけ）。
    """
    use_left = sensing.left is WallState.WALL
    use_right = sensing.right is WallState.WALL
    if not use_left and not use_right:
        return None

    standoff = wall_standoff(params)

    if use_left and use_right:
        (_, y_ls, _), (_, dy_ls, _) = _sensor_geom(params, 'LS')
        (_, y_rs, _), (_, dy_rs, _) = _sensor_geom(params, 'RS')
        # 2 本の式 (下記 left-only/right-only の導出と同じ式) を足し合わせ、
        # standoff（壁の設計位置）への依存を消去した形:
        #   d_LS*dy_ls = standoff - y_ls - e
        #   d_RS*dy_rs = -standoff - y_rs - e
        #   辺々足す -> d_LS*dy_ls + d_RS*dy_rs = -y_ls - y_rs - 2e
        e = -(sensing.left_dist * dy_ls + sensing.right_dist * dy_rs + y_ls + y_rs) / 2.0
        return LateralEstimate(offset_m=e, source="both")

    if use_left:
        (_, y_ls, _), (_, dy_ls, _) = _sensor_geom(params, 'LS')
        e = standoff - y_ls - sensing.left_dist * dy_ls
        return LateralEstimate(offset_m=e, source="left")

    # use_right のみ
    (_, y_rs, _), (_, dy_rs, _) = _sensor_geom(params, 'RS')
    e = -standoff - y_rs - sensing.right_dist * dy_rs
    return LateralEstimate(offset_m=e, source="right")


# ==========================================================================
# (b) 前後位置補正: 前方センサ (LF/RF) から区画内の前後位置ずれを推定する
# ==========================================================================
@dataclass(frozen=True)
class ForwardEstimate:
    """前後位置ずれの推定結果。"""

    offset_m: float  # 区画中心からの前後ずれ推定 [m]。正 = 進行方向前方(壁側)へ寄っている


def estimate_forward_offset(sensing: WallSensing, params: RobotParams) -> Optional[ForwardEstimate]:
    """前方センサの生値から区画中心からの前後ずれを推定する。前方が WALL と
    確定しているときのみ値を返す（`classic/sensing.py` の `sense_walls()` は
    LF・RF の悪い方（近い方）を `front_dist` として返すので、ここでも同じ値を
    そのまま使う。LF・RF は機体座標系の前後位置(x)が等しいので、どちらの
    センサ幾何を使っても式は同じになる）。"""
    if sensing.front is not WallState.WALL:
        return None
    (x_lf, _, _), (dx_lf, _, _) = _sensor_geom(params, 'LF')
    standoff = wall_standoff(params)
    f = standoff - x_lf - sensing.front_dist * dx_lf
    return ForwardEstimate(offset_m=f)


# ==========================================================================
# 補正イベントの記録（「補正が効いたかどうか」を外から見えるようにする）
# ==========================================================================
@dataclass(frozen=True)
class CorrectionEvent:
    """実際に適用した 1 回分の補正の記録。**適用しなかった場合は記録しない**
    （AMBIGUOUS で見送った、無効化されている、等は「イベントが無い」で表す）。"""

    axis: str                          # "lateral" | "forward"
    measured_offset_m: float           # 補正の元になった推定値 [m]
    applied_value: float               # 実際に適用した量（lateral=ヨー目標バイアス[rad]、forward=残距離補正量[m]）
    source: str                        # LateralEstimate.source ("both"/"left"/"right") または "front"
    cell: Optional[Tuple[int, int]] = None    # 分かる場合のみ（explorer 側から渡される）
    heading: Optional[int] = None             # 分かる場合のみ（classic.maze_map.Direction の int 値）


# ------------------------------------------------------------------
# 既定ゲイン（決め打ちではなく、tests/test_classic_localization.py の実測
# ドリフト比較で検証した値。根拠は同ファイルのコメント参照）
# ------------------------------------------------------------------
# 横ずれ e のうち、次の 1 区画で解消しようとする割合。1.0（全量）にすると
# 1 区画で完全に狙い直す分だけ大きく舵を切ることになり、区画到着時の方位が
# 較正前提（sensing.py の位置ずれ±15mm・方位ずれ±4°）を超えて次の壁判定が
# かえって不安定になりうる。緩やかに複数区画かけて収束させる。
DEFAULT_LATERAL_CORRECTION_FRACTION: float = 0.4
# 1 回の横位置補正で与える目標方位バイアスの上限（暴走防止の保険）。
DEFAULT_MAX_LATERAL_BIAS_RAD: float = math.radians(6.0)
# 前後位置補正 1 ティックあたりの残距離補正量の上限（幾何的には
# ±standoff 程度に収まるはずだが、保険として明示的に絶対値クランプする）。
DEFAULT_MAX_FORWARD_CORRECTION_M: float = 0.05


class Localizer:
    """区画ごとの壁センサ位置補正の本体。`classic/explorer.py`・
    `classic/motion.py` から呼ばれる（結線は両ファイルへの最小限の接続を参照）。

    `enabled=False` で完全に無効化できる（否定対照 N1/N2 用。無効化時は
    どちらのメソッドも `sense_walls`/`estimate_*` を一切呼ばず、無補正の値
    だけを返す。事前分岐で `CellMotionController.localizer=None` の場合と
    同じ実行経路になるようにしてある）。
    """

    def __init__(
        self,
        params: Optional[RobotParams] = None,
        enabled: bool = True,
        lateral_correction_fraction: float = DEFAULT_LATERAL_CORRECTION_FRACTION,
        max_lateral_bias_rad: float = DEFAULT_MAX_LATERAL_BIAS_RAD,
        max_forward_correction_m: float = DEFAULT_MAX_FORWARD_CORRECTION_M,
    ) -> None:
        self.params = params if params is not None else RobotParams()
        self.enabled = enabled
        self.lateral_gain = lateral_correction_fraction / self.params.cell_size  # [rad/m]
        self.max_lateral_bias_rad = max_lateral_bias_rad
        self.max_forward_correction_m = max_forward_correction_m
        self.events: List[CorrectionEvent] = []

    # ------------------------------------------------------------------
    # (a) 横位置補正: 次の直進コマンドの目標方位バイアスを返す
    # ------------------------------------------------------------------
    def lateral_bias_for_forward(
        self,
        sensing: WallSensing,
        cell: Optional[Tuple[int, int]] = None,
        heading: Optional[int] = None,
    ) -> float:
        """区画の中心で計算した横ずれ推定から、次の 1 区画の直進で与えるべき
        目標方位バイアス [rad] を返す（推定できない・無効化されているときは 0.0）。

        符号: `CellMotionController.bias_target_heading()` にそのまま渡す前提。
        目標方位（世界ヨー角、CCW 正）を正方向へ振ると機体は左（+y機体座標）
        側を向くようになる。横ずれ e は正=左側へ寄っている、なので e を
        打ち消すには右へ操舵する必要があり、バイアスは `-gain * e`（e>0 なら
        負のバイアス=目標方位を右回りへ）にする。
        """
        if not self.enabled:
            return 0.0
        est = estimate_lateral_offset(sensing, self.params)
        if est is None:
            return 0.0
        bias = float(np.clip(-self.lateral_gain * est.offset_m,
                              -self.max_lateral_bias_rad, self.max_lateral_bias_rad))
        if bias == 0.0:
            return 0.0
        self.events.append(CorrectionEvent(
            axis="lateral", measured_offset_m=est.offset_m, applied_value=bias,
            source=est.source, cell=cell, heading=heading,
        ))
        return bias

    # ------------------------------------------------------------------
    # (b) 前後位置補正: 直進コマンド実行中の残距離を前方センサで補正する
    # ------------------------------------------------------------------
    def correct_forward_remaining(self, remaining: float, obs: np.ndarray) -> float:
        """`CellMotionController.update()` の FORWARD 分岐から毎ティック呼ばれる。
        前方が WALL と確定できたときだけ、残距離 (`remaining`) を前方センサの
        実測値で上書きする（確定できなければ推測航法のままの `remaining` を
        そのまま返す）。無効化されているときは `sense_walls` すら呼ばない
        （`CellMotionController(localizer=None)` と同じ計算経路にするため）。
        """
        if not self.enabled:
            return remaining
        sensing = sense_walls(obs, self.params)
        est = estimate_forward_offset(sensing, self.params)
        if est is None:
            return remaining
        correction = float(np.clip(est.offset_m, -self.max_forward_correction_m,
                                    self.max_forward_correction_m))
        corrected = remaining - correction
        self.events.append(CorrectionEvent(
            axis="forward", measured_offset_m=est.offset_m, applied_value=correction, source="front",
        ))
        return corrected

    # ------------------------------------------------------------------
    # 診断用ユーティリティ
    # ------------------------------------------------------------------
    def clear_events(self) -> None:
        """記録した補正イベントをクリアする（走行ごとに区切って集計したいテスト用）。"""
        self.events.clear()

    def summary(self) -> str:
        """記録した補正イベントの簡単な集計（実測報告用）。"""
        lateral = [e for e in self.events if e.axis == "lateral"]
        forward = [e for e in self.events if e.axis == "forward"]
        lines = [
            f"横位置補正(lateral): {len(lateral)} 回",
            f"前後位置補正(forward): {len(forward)} 回",
        ]
        if lateral:
            mags = [abs(e.measured_offset_m) for e in lateral]
            lines.append(f"  横ずれ推定値の絶対値: 最大={max(mags)*1000:.2f}mm 平均={sum(mags)/len(mags)*1000:.2f}mm")
        if forward:
            mags = [abs(e.measured_offset_m) for e in forward]
            lines.append(f"  前後ずれ推定値の絶対値: 最大={max(mags)*1000:.2f}mm 平均={sum(mags)/len(mags)*1000:.2f}mm")
        return "\n".join(lines)
