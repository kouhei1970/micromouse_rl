"""
classic/tracker.py
================
速度プロファイル `v_ref(s)`・曲率プロファイル `kappa_ref(s)`（`classic/profile.py`
の `min_time` が返す `IdealTime`。`s_grid`/`v_grid`/`kappa_grid` の格子表現）に
沿って走る追従制御器。`research_notes/note_031_profile_planner_and_eta.md`
§「段 3 の結果」が指摘した、現行のコマンド方式（1 コマンド発行 → 完了待ち →
次）の欠陥（区間の切れ目で毎回停止し、特にターンが「角度は合っているのに
角速度がゼロになるのを待つ」整定待ちで壊滅的に遅くなる）を直しに行くもの。

方針（ユーザ裁定・2026-08-20）: これは「事故らないロボット」ではなく
**究極の速度を競うもの**である。整定待ちは受け入れる対象ではなく工学的な
工夫で消す対象。本ファイルは (1) 経路（弧長 s でパラメータ化）追従、
(2) その場旋回の時間最適軌道追従、(3) 車輪速度ループの遅れ（τ_v）を打ち消す
逆モデル前置き、の 3 つで整定待ちの概念そのものを消しに行く。

【この段では classic/motion.py・classic/explorer.py は 1 行も変更しない】
本ファイルと `tests/test_tracker.py` の 2 つだけが新規。差し替え（実際に
探索走行へ組み込む）は次の段の仕事。`classic/profile.py`（`min_time`・
`spin_turn_time`・`vehicle_limits`）は計画器として import して使う
（触ってはいけないのは motion.py/explorer.py のみ）。

【状態推定（真値は使わない）】
`classic/motion.py` の `CellMotionController` と同じ作法（`_split_obs` で
observation() から車輪角速度・ジャイロ z を切り出し、車輪角速度平均×車輪半径
を弧長へ積分、ジャイロ z を方位へ積分）を踏襲する:
  - `s`（走った弧長）: 車輪角速度の平均 × 車輪半径 を積分
  - `psi`（方位）: ジャイロ z を積分
  - `e_lat`（横ずれ）: 既定 0。壁センサからの補正を差し込める口
    （`apply_lateral_correction()`）を用意するだけで、本段では中身を実装しない
    （`classic/localization.py` の「差し込み口を用意するだけ」という作法に合わせる）。

【経路追従の制御則（前置き＋フィードバック）】
    v_ff   = v_ref(s)
    w_ff   = v_ref(s) * kappa_ref(s)
    psi_ref(s) = psi_0 + ∫_0^s kappa_ref(σ) dσ   （計画の時点で s の格子上に
                                                    積分済み・毎ティック積分しない）
    e_psi  = wrap(psi_ref(s) - psi)
    w_cmd  = w_ff + kp_psi * e_psi + kp_lat * e_lat   （kp_lat==0 のときは
                                                          この項自体を計算しない。
                                                          下記 update() 参照）
    v_cmd  = v_ff
    omega_l_target = (v_cmd - w_cmd * tread / 2) / r
    omega_r_target = (v_cmd + w_cmd * tread / 2) / r

【境界フォーシング（実測で追加した、指示書に無い最小限の補正。理由は下記）】
静止発進（`v_ref` の最初の格子値が 0）・停止着地（最後の格子値が 0）の近傍では、
`v_ff` がそのまま 0 に張り付く。この付近は速度そのものへのフィードバックが
無い（`w_cmd` は角速度のみを補正する）ため、実測すると **PI が発生させる
電圧が「目標角速度と実測角速度の差」から生まれる構造上、両方が 0 に近づくと
差もほぼ 0 になり、指数関数的に目標へ漸近するだけで有限時間では
到達しない**（3000 ティック回しても残り 0.01mm 未満まで詰まるが一度も
`s>=s_end` を満たさなかったことを実測で確認済み）。これは「速度がしきい値
未満に収束するのを待つ」整定待ちの逆（動き続けさせる側）の話であり、
本追加が消したい整定待ちの概念には抵触しない。`s` が計画の始点・終点から
`_BOUNDARY_FORCE_DISTANCE` 以内のとき、`v_ff` に最小接近速度
`_BOUNDARY_FORCE_SPEED` の下駄を履かせて確実に有限時間で通過させる
（値の根拠は `_BOUNDARY_FORCE_SPEED` 直上のコメント）。

【逆モデル前置き（ユーザ指示・追加 1。実装は「微分」ではなく「先読み」）】
車輪速度 PI（`classic/motion.py` の `_wheel_pi` と同じ形。時定数
`tau_v`（`docs/ROBOT_SPEC.md` §2.5 実測値）の一次遅れを持つ）に対し、
角速度目標の**フィードフォワード成分のみ**を `tau_v` 秒先読みしてかさ上げする
（逆モデル: `tau_v*domega/dt + omega ≈ K*V` の一次遅れ系を、目標を
`omega_target(t+tau_v)` 相当にかさ上げして与えると定常追従遅れがほぼ消える、
という標準的なリード補償）。

🔴 指示書の式は `omega_target + tau_v*d(omega_target)/dt`（1 次のテイラー近似）
だったが、**そのまま実装すると新しい不動点を作ることを実測で確認したため、
数学的に同値だが破綻しない形へ実装を変えた**（理由は下記）。バンバン型の
時間最適プロファイル（`min_time` の加減速・`spin_turn_time` の旋回）は
区間の境界で加速度が不連続に切り替わる。その境界の直前で 1 次のテイラー近似
`omega + tau_v*domega/dt` を使うと、`tau_v` が「目標がまさに 0 に落ちる
までの残り時間」と同程度の場面（実測: 直線減速区間で A_TR≈5.57m/s²・
tau_v≈0.124s が桁で一致、その場旋回でも同様）で前置き項が目標とほぼ同じ
大きさの負の値になり、加算後の目標がほぼ0または符号反転して
「目標がほぼ0→PIの誤差もほぼ0→電圧が生まれない」という不動点を作ってしまう
ことを実測で確認した（クランプで抑える対処も試したが、経路追従とその場旋回で
必要なクランプの強さが大きく異なり、その場旋回側は角速度の収束が悪いまま
残った）。

代わりに、`tau_v*domega/dt` を**同じ値の近似**である「`tau_v` 秒先の計画上の
目標をそのまま読む」形で計算する:
    omega_ff_now(s)      = (v_ref(s) - v_ref(s)*kappa_ref(s)*tread/2) / r  （現在点）
    s_preview            = s + tau_v * v_ff(s)   （tau_v 秒後の弧長の見積り。
                                                    ds/dt を計画上の v_ff で
                                                    近似するのは元の連鎖律と同じ考え）
    omega_ff_preview(s)  = (v_ref(s_preview) - v_ref(s_preview)*kappa_ref(s_preview)*tread/2) / r
    lead = omega_ff_preview - omega_ff_now
その場旋回も同じ考え方で `omega_ref(t)` を `t+tau_v` で直接引く
（`_spin_kinematics()` は境界（t<=0 や t>=time_total）で自動的に頭打ちになる
ので、先読みが計画の外へ出ても不連続な値を返さない）。この形は `v_ref`・
`kappa_ref`・`omega_ref` が区間内で滑らかな場面ではテイラー近似と一致しつつ、
**区間の境界をまたいでも実際の計画値（0 でクランプされる等）をそのまま返す
ので線形外挿のように暴れない**。**微分の対象・先読みの対象はいずれも
フィードフォワード成分だけに限定し、フィードバック成分（kp_psi*e_psi 等、
測定ヨーを含む）は先読みしない**（測定ノイズを含む項を先読みに使うと
雑音を増幅するため）。
`TrackerGains.use_lag_feedforward=False` にすると、この項の計算そのものを
一切行わない（否定対照の土台。下記 update() 参照）。

【その場旋回の時間最適軌道追従（ユーザ指示・追加 2）】
`load_spin_plan(delta_theta)` で `classic/profile.py` の `spin_turn_time()`
（バンバン制御。三角形/台形を自動判定）が返す `alpha`（角加速度）・
`omega_peak`・`time` から、時刻 `t` の閉形式の角速度・方位の時間最適軌道
`omega_ref(t)`・`psi_ref(t)` を作る。制御則:
    w_cmd  = omega_ref(t) + kp_psi * wrap(psi_ref(t) - psi)
    v_cmd  = 0
    done   = t >= time_total
`t` は計画ロード直後（`reset()`/直近の `load_*`）からの経過時間で、実際の
角速度に関わらず一定速度で進む（＝時間そのものが基準点なので、経路追従と
同種の「速度が 0 に近づくほど収束が遅くなる」不動点がそもそも起きない）。

**区間の切れ目・その場旋回の終端で止まらない。** 経路追従の `done` は
`s >= s_end` のときだけ、その場旋回の `done` は `t >= time_total` のときだけ
真になる。「速度・角速度がしきい値未満に収束するのを待つ」ような completion
判定は持たない（`classic/motion.py` の FORWARD/TURN の
`speed_settle`/`omega_settle` 待ちに相当する概念がそもそも無い）。

【kp_psi の決め方】
下記 `DEFAULT_KP_PSI` 直上のコメントを参照（実測スイープの結果と根拠）。
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from classic.profile import spin_turn_time, vehicle_limits
from mouse.params import RobotParams

__all__ = ["TrackerGains", "ProfileTracker"]


# ------------------------------------------------------------------
# 車輪速度 PI（classic/motion.py の DEFAULT_KP_WHEEL/DEFAULT_KI_WHEEL と同じ値。
# 二重管理を避けたいところだが、classic/motion.py は本段で 1 行も変更しない
# 対象なので、値そのものをここへ複写する（根拠は classic/motion.py モジュール
# docstring の「PI ゲインの決め方」節を参照。同じ実測較正に基づく）。
# ------------------------------------------------------------------
DEFAULT_KP_WHEEL: float = 0.025
DEFAULT_KI_WHEEL: float = 0.2
# 🔴 その場旋回専用の車輪 PI 積分ゲイン（実測スイープで追加。2026-08-20）。
# その場旋回は経路追従よりずっと高い車輪角速度（数十〜100rad/s級）・急な
# 加減速（560 rad/s²級）を要求する。`ki_wheel`（経路追従用に較正済み。
# `classic/motion.py` と同じ 0.2）のままだと車輪 PI の応答が追いつかず、
# 実測で旋回の到達角度誤差が大きいまま残る。`kp_wheel` を上げる案は
# `classic/motion.py` の「PI ゲインの決め方」節と同じ理由（左右非対称に
# 暴れて位置ずれが増える）で実測により却下し、`ki_wheel` だけを旋回時に
# 引き上げる（値の根拠は `DEFAULT_KP_PSI` 直上のコメント参照。同じ実測ログで
# 一緒にスイープした）。
DEFAULT_KI_WHEEL_SPIN: float = 0.8

# 方位フィードバックゲイン [1/s]（e_psi -> 角速度指令への比例ゲイン）。
# 🔴 実測スイープで決定（2026-08-20、tests/test_tracker.py 実行時の
# [実測] 行・本タスクの完了報告に詳細ログを残す）。直線4区画の停止→停止計画
# （開始直後に+10degのヨー推定誤差を注入）と、その場旋回90°/180°
# （時間最適バンバン軌道、DEFAULT_KI_WHEEL_SPIN と同時にスイープ）の両方を
# 判定に使った。直線側は kp_psi にほぼ非依存（無外乱の対称なシミュレータでは
# 直線中に方位誤差がそもそも生じないため。classic/motion.py の
# kp_heading と同じ事情）なので、その場旋回側の実測が実質的な決め手になった:
#   kp_psi=16: 90°誤差 30.3° / 180°誤差 29.6°（不足）
#   kp_psi=24: 90°誤差 13.9° / 180°誤差  7.0°
#   kp_psi=28（DEFAULT_KI_WHEEL_SPIN=0.8 併用）: 90°誤差 0.54° / 180°誤差 1.18°
#   kp_psi=32: 90°誤差  2.7° / 180°誤差  4.9°（180°側が再び悪化）
# 28 前後が両角度とも 3° 未満に収まる唯一の帯だったため、これを採用する。
DEFAULT_KP_PSI: float = 28.0

# 横ずれ -> 方位 の変換ゲイン [1/m]。既定 0（無効）。`apply_lateral_correction()`
# の差し込み口自体は用意するが、壁センサからの実際の推定・結線は本段の範囲外
# （`classic/localization.py` 側の仕事。ここでは「否定対照の土台」として
# kp_lat=0 のとき経路を一切通らないことだけを保証する）。
DEFAULT_KP_LAT: float = 0.0

# 逆モデル前置き（モジュール docstring 参照）。既定 ON。
DEFAULT_USE_LAG_FEEDFORWARD: bool = True
# 車輪速度ループの時定数 [s]（docs/ROBOT_SPEC.md §2.5 の実測較正値。
# classic/motion.py モジュール docstring の「PI ゲインの決め方」節が引用する
# のと同じ値: τ_v=0.124s）。
DEFAULT_TAU_V: float = 0.124

# 車輪 PI 積分器の絶対値クランプ（classic/motion.py の DEFAULT_INTEG_CLAMP と
# 同じ値・同じ理由: 暴走防止の保険）。
_INTEG_CLAMP: float = 200.0

# ------------------------------------------------------------------
# 境界フォーシング（モジュール docstring「境界フォーシング」節参照）。
#
# 値の根拠（2026-08-20 実測）: 車輪 PI の比例ゲイン kp_wheel=0.025・車輪の
# 乾性摩擦 wheel_frictionloss=9.1e-4 N·m（mouse/params.py）から、比例項だけで
# 静止摩擦を破る（角速度誤差 × kp_wheel が破断電圧
# frictionloss*motor_R/(N*Kt)≈0.098V を超える）のに必要な角速度誤差は
# 0.098/0.025≈3.9 rad/s、並進速度に換算すると 3.9*r≈0.053 m/s。これを
# 下回る指令では車輪が動き出さない（実測でも v_ref が 0.03〜0.05 m/s 程度の
# 領域では並進速度がほぼ 0 のまま張り付くことを確認した）。十分な余裕を見て
# 0.10 m/s を最小接近速度に採用する。
_BOUNDARY_FORCE_SPEED: float = 0.10   # m/s
# 境界フォーシングを発動する、始点・終点からの弧長距離のしきい値。
# 0.10 m/s で通過するとすれば 20mm は 1 ティック(10ms)あたり 1mm 程度の刻みで
# 通過できる距離であり、tests/test_tracker.py の終端位置誤差判定(20mm)と
# 同じ桁で保守的（フォーシングが効く区間自体を判定の許容誤差程度に抑える）。
_BOUNDARY_FORCE_DISTANCE: float = 0.020  # m

# 逆モデル前置きの下駄（フロア）比率（update() の `_apply_lead_floor` 参照）。
# 🔴 実測スイープ（2026-08-20、直線4区画・その場旋回90°/180°・
# 直線助走付き四分円を横断して比較）: 0.3 では直線が実測/計画比 1.68（既定
# ゲイン kp_psi=28 のもとで）とやや遅く、0.7 まで上げると比は 0.99 まで
# 改善するが、その場旋回の到達角度誤差が 7〜12° まで悪化する。0.4〜0.6 の
# 帯は直線が比 1.22（1.3 以内）・誤差 17mm（20mm 以内）、その場旋回が
# 90°/180° とも誤差 1° 未満、四分円がヨー変化 92.7°（90°±5° 以内）と、
# 4 種の計画すべてで判定基準を満たす唯一の帯だったため、その中央値 0.5 を採用。
_LEAD_MIN_FRACTION: float = 0.5


@dataclass
class TrackerGains:
    """ProfileTracker のゲイン一式。"""

    kp_wheel: float = DEFAULT_KP_WHEEL
    ki_wheel: float = DEFAULT_KI_WHEEL
    ki_wheel_spin: float = DEFAULT_KI_WHEEL_SPIN
    kp_psi: float = DEFAULT_KP_PSI
    kp_lat: float = DEFAULT_KP_LAT
    use_lag_feedforward: bool = DEFAULT_USE_LAG_FEEDFORWARD
    tau_v: float = DEFAULT_TAU_V


def _interp_linear(s: float, s_grid: Sequence[float], values: Sequence[float]) -> float:
    """`classic/profile.py` の `IdealTime.v_at()` と同じ作法（線形補間・範囲外は
    端の値でクランプ）。`values` は `s_grid` と同じ長さ（格子点の値）。"""
    if not s_grid:
        return 0.0
    if s <= s_grid[0]:
        return values[0]
    if s >= s_grid[-1]:
        return values[-1]
    i = bisect.bisect_right(s_grid, s) - 1
    i = min(max(i, 0), len(s_grid) - 2)
    s0, s1 = s_grid[i], s_grid[i + 1]
    v0, v1 = values[i], values[i + 1]
    if s1 <= s0:
        return v0
    frac = (s - s0) / (s1 - s0)
    return v0 + frac * (v1 - v0)


def _interp_step(s: float, s_grid: Sequence[float], cell_values: Sequence[float]) -> float:
    """`classic/profile.py` の `IdealTime.kappa_at()` と同じ作法（区分定数・
    範囲外は端のセルの値）。`cell_values` は `len(s_grid)-1` 個（セルの値）。"""
    if not cell_values:
        return 0.0
    if s <= s_grid[0]:
        return cell_values[0]
    if s >= s_grid[-1]:
        return cell_values[-1]
    i = bisect.bisect_right(s_grid, s) - 1
    i = min(max(i, 0), len(cell_values) - 1)
    return cell_values[i]


class ProfileTracker:
    """速度プロファイル `v_ref(s)`・曲率プロファイル `kappa_ref(s)` に沿って
    走る追従制御器、およびその場旋回の時間最適軌道追従（モジュール
    docstring 参照）。

    経路追従:
        tracker = ProfileTracker(params, gains)
        tracker.load_plan(ideal.s_grid, ideal.v_grid, ideal.kappa_grid,
                           psi_start=math.radians(90.0))
        tracker.reset(heading_deg=90.0)
        while True:
            obs = sim.observation()
            vl, vr, done = tracker.update(obs)
            sim.step_control(vl, vr)
            if done:
                break

    その場旋回:
        tracker.reset(heading_deg=90.0)
        tracker.load_spin_plan(math.radians(90.0))
        ...(同じ update() ループ)...

    **真値位置（privileged_pose）は一切使わない。**
    """

    def __init__(self, params: Optional[RobotParams] = None,
                 gains: Optional[TrackerGains] = None):
        self.params = params if params is not None else RobotParams()
        self.gains = gains if gains is not None else TrackerGains()
        self._n_sensors = len(self.params.sensors)

        # 経路追従の計画（load_plan() が設定するまでは未ロード）
        self._s_grid: List[float] = []
        self._v_ref: List[float] = []
        self._kappa_ref: List[float] = []
        self._psi_grid: List[float] = []
        self._s_end: float = 0.0

        # その場旋回の計画（load_spin_plan() が設定するまでは未ロード）
        self._spin_sign: float = 1.0
        self._spin_alpha: float = 0.0
        self._spin_omega_peak: float = 0.0
        self._spin_t_acc: float = 0.0
        self._spin_t_cruise: float = 0.0
        self._spin_time_total: float = 0.0
        self._spin_theta_total: float = 0.0
        self._spin_psi0: float = 0.0

        self._mode: Optional[str] = None  # "path" | "spin" | None(未ロード)

        self.reset()

    # ------------------------------------------------------------------
    # 計画のロード（経路追従）
    # ------------------------------------------------------------------
    def load_plan(self, s_grid: Sequence[float], v_ref: Sequence[float],
                   kappa_ref: Sequence[float], psi_start: float) -> None:
        """速度・曲率プロファイルをロードする。

        `s_grid`・`v_ref` は格子点の値（長さ n+1。`classic/profile.py` の
        `IdealTime.s_grid`/`v_grid` と同じ規約）、`kappa_ref` はセルの値
        （長さ n。同じく `IdealTime.kappa_grid` と同じ規約）。そのまま
        `tracker.load_plan(ideal.s_grid, ideal.v_grid, ideal.kappa_grid,
        psi_start=...)` の形で `min_time()` の戻り値を渡せる。

        `psi_ref(s)` はここで `s_grid` と同じ格子上に前もって積分しておく
        （毎ティック積分しない。曲率がセル内で区分定数なので、セル内の
        `psi_ref` は厳密に線形になり、格子点だけ積分すれば線形補間で
        セル内の値が厳密に再現できる）。
        """
        n = len(s_grid)
        if n < 2:
            raise ValueError(f"s_grid は 2 点以上必要です: {n}")
        if len(v_ref) != n:
            raise ValueError(f"v_ref は s_grid と同じ長さ({n})が必要です: {len(v_ref)}")
        if len(kappa_ref) != n - 1:
            raise ValueError(f"kappa_ref は len(s_grid)-1({n - 1})が必要です: {len(kappa_ref)}")

        self._s_grid = list(s_grid)
        self._v_ref = list(v_ref)
        self._kappa_ref = list(kappa_ref)
        self._s_end = self._s_grid[-1]

        psi_grid = [0.0] * n
        psi_grid[0] = psi_start
        for i in range(n - 1):
            ds = self._s_grid[i + 1] - self._s_grid[i]
            psi_grid[i + 1] = psi_grid[i] + self._kappa_ref[i] * ds
        self._psi_grid = psi_grid
        self._mode = "path"

    def _v_at(self, s: float) -> float:
        return _interp_linear(s, self._s_grid, self._v_ref)

    def _kappa_at(self, s: float) -> float:
        return _interp_step(s, self._s_grid, self._kappa_ref)

    def _psi_at(self, s: float) -> float:
        return _interp_linear(s, self._s_grid, self._psi_grid)

    # ------------------------------------------------------------------
    # 計画のロード（その場旋回）
    # ------------------------------------------------------------------
    def load_spin_plan(self, delta_theta: float, psi_start: Optional[float] = None) -> None:
        """その場旋回の時間最適軌道（バンバン制御）をロードする。

        `delta_theta` [rad] は符号付き旋回角（正=左/CCW、負=右/CW）。
        `classic/profile.py` の `spin_turn_time()` が返す `alpha`（角加速度）・
        `omega_peak`（頭打ち角速度。三角形なら三角ピークと一致）・`time`
        （総所要時間）から、加速→（頭打ちなら定常）→減速の閉形式の時間軌道
        `omega_ref(t)`・`psi_ref(t)` を組み立てる。

        `psi_start` を省略すると、呼び出し時点の `yaw_estimate`（推測航法の
        現在のヨー推定）を基準に使う（`classic/motion.py` の `_start_turn` が
        `self._target_yaw = self._yaw_est + delta_yaw` とするのと同じ考え方）。
        """
        limits = vehicle_limits(self.params)
        spin = spin_turn_time(delta_theta, limits)
        sign = 1.0 if delta_theta >= 0.0 else -1.0

        self._spin_sign = sign
        self._spin_alpha = spin.alpha
        self._spin_omega_peak = spin.omega_peak
        self._spin_theta_total = spin.theta
        self._spin_time_total = spin.time
        # t_acc = omega_peak/alpha は三角形・台形のどちらでも成り立つ
        # （三角形は omega_peak=omega_tri=sqrt(alpha*theta), time=2*sqrt(theta/alpha)
        # のとき t_acc=time/2 に一致し、t_cruise=0 が自動的に出る。台形は
        # spin_turn_time() 自身の式 `t=omega_cap/alpha+theta/omega_cap` の
        # 前半がそのまま t_acc に一致する）。
        self._spin_t_acc = spin.omega_peak / spin.alpha if spin.alpha > 0.0 else 0.0
        self._spin_t_cruise = max(spin.time - 2.0 * self._spin_t_acc, 0.0)
        self._spin_psi0 = psi_start if psi_start is not None else self._yaw_est
        self._t = 0.0
        self._mode = "spin"

    def _spin_kinematics(self, t: float):
        """時刻 `t`（旋回開始からの経過時間）における `(psi_ref, omega_ref,
        alpha_ref)` を返す（閉形式・区分二次/一次関数）。`alpha_ref` は
        `omega_ref` の時間微分（符号付き角加速度。逆モデル前置きで使う）。"""
        alpha = self._spin_alpha
        omega_peak = self._spin_omega_peak
        t_acc = self._spin_t_acc
        t_cruise = self._spin_t_cruise
        time_total = self._spin_time_total
        theta_total = self._spin_theta_total
        sign = self._spin_sign

        if t <= 0.0:
            theta, omega, alpha_signed = 0.0, 0.0, alpha
        elif t < t_acc:
            theta = 0.5 * alpha * t * t
            omega = alpha * t
            alpha_signed = alpha
        elif t < t_acc + t_cruise:
            theta = 0.5 * alpha * t_acc * t_acc + omega_peak * (t - t_acc)
            omega = omega_peak
            alpha_signed = 0.0
        elif t < time_total:
            td = t - (t_acc + t_cruise)
            theta = (0.5 * alpha * t_acc * t_acc + omega_peak * t_cruise
                     + omega_peak * td - 0.5 * alpha * td * td)
            omega = omega_peak - alpha * td
            alpha_signed = -alpha
        else:
            theta, omega, alpha_signed = theta_total, 0.0, 0.0

        psi_ref = self._spin_psi0 + sign * theta
        omega_ref = sign * omega
        alpha_ref = sign * alpha_signed
        return psi_ref, omega_ref, alpha_ref

    # ------------------------------------------------------------------
    # 状態リセット・観測の切り出し（classic/motion.py の CellMotionController
    # と同じ作法）
    # ------------------------------------------------------------------
    def reset(self, heading_deg: float = 90.0) -> None:
        """走行開始時に一度だけ呼ぶ。推測航法の内部状態を初期化する。

        `heading_deg` は評価器が走行開始時に機体を置く既知の初期方位
        （プロトコル上の定数）であり、走行中の真値取得ではない
        （`classic/motion.py` の `CellMotionController.reset()` と同じ扱い）。
        ロード済みの計画（`load_plan()`/`load_spin_plan()`）はクリアしない。
        """
        self._yaw_est = math.radians(heading_deg)
        self._s = 0.0
        self._t = 0.0
        self._e_lat = 0.0
        self._integ_l = 0.0
        self._integ_r = 0.0

    def _split_obs(self, obs: np.ndarray):
        """observation() から車輪角速度・ジャイロ z 成分を取り出す
        （`classic/motion.py` の `_split_obs` と同じ作法・同じ並び）。"""
        n = self._n_sensors
        gyro_z = float(obs[n + 5])
        omega_l = float(obs[n + 6])
        omega_r = float(obs[n + 7])
        return gyro_z, omega_l, omega_r

    @property
    def s(self) -> float:
        """推測航法による走行弧長推定 [m]（経路追従モードでのみ意味を持つ）。"""
        return self._s

    @property
    def yaw_estimate(self) -> float:
        """推測航法によるヨー角推定 [rad]（真値ではなく積算値）。"""
        return self._yaw_est

    # ------------------------------------------------------------------
    # S2 相当の差し込み口（横ずれ補正。本段では中身を実装しない）
    # ------------------------------------------------------------------
    def apply_lateral_correction(self, e_lat: float) -> None:
        """横ずれ推定値 `e_lat` [m]（正=左）をフィードバックへ差し込む。

        `gains.kp_lat == 0.0`（既定）のときは `update()` 側がこの値を一切
        参照しない（下記 `update()` の分岐参照。否定対照の土台）。実際の
        壁センサからの横ずれ推定・呼び出し配線は本段の範囲外（次の段で
        `classic/localization.py` 側と結線する）。経路追従モードでのみ使う
        （その場旋回では参照しない）。
        """
        self._e_lat = float(e_lat)

    # ------------------------------------------------------------------
    # 制御ステップ
    # ------------------------------------------------------------------
    def update(self, obs: np.ndarray):
        """1 制御ステップ分の (v_left, v_right, done) を返す。

        経路追従: `done` は `s >= s_end` のときだけ真になる。
        その場旋回: `done` は `t >= 旋回総所要時間` のときだけ真になる。
        いずれも区間の切れ目・完了直前の速度収束は一切待たない。呼び出し側は
        返された電圧を `sim.step_control()` へそのまま渡すこと。
        """
        if self._mode is None:
            raise RuntimeError("load_plan() か load_spin_plan() を先に呼んでください")

        dt = self.params.control_dt
        r = self.params.wheel_radius
        tread = self.params.tread
        gyro_z, omega_l, omega_r = self._split_obs(obs)

        # 推測航法の積算（車輪角速度 -> 弧長、ジャイロ -> 方位）。
        # classic/motion.py の _dist_est 積算と同じ形。
        v_meas = (omega_l + omega_r) / 2.0 * r
        self._s += v_meas * dt
        self._yaw_est += gyro_z * dt
        self._t += dt

        if self._mode == "path":
            v_ff, w_ff, psi_ref, v_ff_prev, w_ff_prev, done = self._path_step()
        else:
            v_ff, w_ff, psi_ref, v_ff_prev, w_ff_prev, done = self._spin_step()

        e_psi = self._wrap(psi_ref - self._yaw_est)
        w_cmd = w_ff + self.gains.kp_psi * e_psi
        # kp_lat==0（既定）のとき、または旋回モードのときは、この分岐そのものを
        # 通らない（横ずれ補正の経路が一切実行されないことの保証。否定対照の土台）。
        if self.gains.kp_lat != 0.0 and self._mode == "path":
            w_cmd += self.gains.kp_lat * self._e_lat
        v_cmd = v_ff

        omega_l_target = (v_cmd - w_cmd * tread / 2.0) / r
        omega_r_target = (v_cmd + w_cmd * tread / 2.0) / r

        # 逆モデル前置き（モジュール docstring「逆モデル前置き」節参照。
        # tau_v 秒先のフィードフォワードをそのまま読んで差分を乗せる「先読み」
        # 方式で、テイラー近似の破綻を避ける）。use_lag_feedforward=False の
        # ときはこの分岐そのものを通らない（否定対照の土台）。
        #
        # 🔴 実測で追加した下駄（フロア）: 先読み点が計画の終端（強制的に
        # 境界値へクランプされる領域）を越えると、先読み値と現在値の差が
        # 現在の目標とほぼ同じ大きさの逆符号になり、加算後の目標がほぼ0（＝
        # 「目標がほぼ0→PIの誤差もほぼ0→電圧が生まれない」という新しい
        # 不動点）になることを実測で確認した（例: 直線減速中に
        # tau_v*v_ff が残り距離を超えると、先読み値が「もう止まっている」
        # 前提の0を返してしまう）。前置きが目標を大きく減らす／符号を反転
        # させる方向に働く場合だけ、基準目標（前置き無しの目標）の
        # `_LEAD_MIN_FRACTION` 未満には落とさない下駄で止める（増やす方向
        # ＝オーバーシュートを追認する方向には制限をかけない。これは
        # 「前置きが効きすぎて行き過ぎる」ことより「不動点を作る」ことの方が
        # 実害が大きいという実測に基づく判断）。
        if self.gains.use_lag_feedforward:
            omega_l_ff_now = (v_ff - w_ff * tread / 2.0) / r
            omega_r_ff_now = (v_ff + w_ff * tread / 2.0) / r
            omega_l_ff_prev = (v_ff_prev - w_ff_prev * tread / 2.0) / r
            omega_r_ff_prev = (v_ff_prev + w_ff_prev * tread / 2.0) / r
            omega_l_target = self._apply_lead_floor(
                omega_l_target, omega_l_target + (omega_l_ff_prev - omega_l_ff_now))
            omega_r_target = self._apply_lead_floor(
                omega_r_target, omega_r_target + (omega_r_ff_prev - omega_r_ff_now))

        ki_wheel = self.gains.ki_wheel if self._mode == "path" else self.gains.ki_wheel_spin
        vl, vr = self._wheel_pi(omega_l_target, omega_r_target, omega_l, omega_r, dt, ki_wheel)
        return vl, vr, done

    @staticmethod
    def _apply_lead_floor(base: float, combined: float) -> float:
        """逆モデル前置き適用後の目標 `combined` が、基準目標 `base` の
        `_LEAD_MIN_FRACTION` 未満（符号反転を含む）まで削られていたら、
        その下駄まで戻す（update() 内のコメント参照）。"""
        floor = _LEAD_MIN_FRACTION * abs(base)
        if base >= 0.0:
            return combined if combined >= floor else floor
        return combined if combined <= -floor else -floor

    def _ff_at(self, s: float):
        """弧長 `s` におけるフィードフォワード `(v_ff, w_ff)`（境界フォーシング
        適用後）。逆モデル前置きの「現在点」「tau_v 秒先の点」の両方で使う
        共通ロジック（モジュール docstring「境界フォーシング」節参照）。"""
        v_ref = self._v_at(s)
        kappa_ref = self._kappa_at(s)
        remaining = self._s_end - s
        near_start = self._v_ref[0] <= 1e-9 and s <= _BOUNDARY_FORCE_DISTANCE
        near_end = self._v_ref[-1] <= 1e-9 and 0.0 < remaining < _BOUNDARY_FORCE_DISTANCE
        if near_start or near_end:
            v_ref = max(v_ref, _BOUNDARY_FORCE_SPEED)
        return v_ref, v_ref * kappa_ref

    def _path_step(self):
        """経路追従モードの 1 ティック分の
        (v_ff, w_ff, psi_ref, v_ff_preview, w_ff_preview, done)。"""
        s = self._s
        v_ff, w_ff = self._ff_at(s)
        psi_ref = self._psi_at(s)

        # tau_v 秒先の弧長の見積り（ds/dt を計画上の v_ff で近似する。
        # モジュール docstring 参照）。
        s_preview = s + self.gains.tau_v * v_ff
        v_ff_preview, w_ff_preview = self._ff_at(s_preview)

        done = self._s >= self._s_end
        return v_ff, w_ff, psi_ref, v_ff_preview, w_ff_preview, done

    def _spin_step(self):
        """その場旋回モードの 1 ティック分の
        (v_ff=0, w_ff=omega_ref, psi_ref, v_ff_preview=0, w_ff_preview, done)。"""
        psi_ref, omega_ref, _alpha_ref = self._spin_kinematics(self._t)
        _psi_preview, omega_ref_preview, _ = self._spin_kinematics(self._t + self.gains.tau_v)
        done = self._t >= self._spin_time_total
        return 0.0, omega_ref, psi_ref, 0.0, omega_ref_preview, done

    def _wheel_pi(self, omega_l_target, omega_r_target, omega_l, omega_r, dt, ki_wheel):
        """車輪角速度目標値と実測値の偏差から PI で左右電圧を作る（アンチ
        ワインドアップ付き）。`classic/motion.py` の `_wheel_pi` と同じ形
        （`kp_wheel` は経路追従・その場旋回で共通。`ki_wheel` は呼び出し側
        `update()` がモードに応じて `gains.ki_wheel`/`gains.ki_wheel_spin` を
        選んで渡す。理由は `DEFAULT_KI_WHEEL_SPIN` 直上のコメント参照）。"""
        vlim = self.params.voltage_limit
        kp_wheel = self.gains.kp_wheel

        err_l = omega_l_target - omega_l
        err_r = omega_r_target - omega_r

        vl_try = kp_wheel * err_l + ki_wheel * self._integ_l
        vr_try = kp_wheel * err_r + ki_wheel * self._integ_r
        if abs(vl_try) < vlim:
            self._integ_l = float(np.clip(self._integ_l + err_l * dt, -_INTEG_CLAMP, _INTEG_CLAMP))
        if abs(vr_try) < vlim:
            self._integ_r = float(np.clip(self._integ_r + err_r * dt, -_INTEG_CLAMP, _INTEG_CLAMP))

        vl = float(np.clip(kp_wheel * err_l + ki_wheel * self._integ_l, -vlim, vlim))
        vr = float(np.clip(kp_wheel * err_r + ki_wheel * self._integ_r, -vlim, vlim))
        return vl, vr

    @staticmethod
    def _wrap(angle: float) -> float:
        """角度を [-pi, pi) に正規化する。"""
        return math.atan2(math.sin(angle), math.cos(angle))
