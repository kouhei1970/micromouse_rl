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
(2) その場旋回の時間最適軌道追従、(3) 逆動力学（電圧）前置き、の 3 つで
整定待ちの概念そのものを消しに行く。

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

【その場旋回の時間最適軌道追従（ユーザ指示・追加 2）】
`load_spin_plan(delta_theta)` で `classic/profile.py` の `spin_turn_time()`
（バンバン制御。三角形/台形を自動判定）が返す `alpha`（角加速度）・
`omega_peak`・`time` から、時刻 `t` の閉形式の角速度・方位の時間最適軌道
`omega_ref(t)`・`psi_ref(t)` を作る。制御則:
    w_cmd  = omega_ref(t) + kp_psi * wrap(psi_ref(t) - psi)
    v_cmd  = 0
    done   = t >= time_total
`t` は計画ロード直後（`reset()`/直近の `load_*`）からの経過時間で、実際の
角速度に関わらず一定速度で進む。

**区間の切れ目・その場旋回の終端で止まらない。** 経路追従の `done` は
`s >= s_end` のときだけ、その場旋回の `done` は `t >= time_total` のときだけ
真になる。「速度・角速度がしきい値未満に収束するのを待つ」ような completion
判定は持たない（`classic/motion.py` の FORWARD/TURN の
`speed_settle`/`omega_settle` 待ちに相当する概念がそもそも無い）。

【逆動力学（電圧）前置き（ユーザ指示・2026-08-20 是正）】
🔴 経緯: 当初は「車輪速度 PI の目標角速度を tau_v 秒先読みしてかさ上げする」
という設計（角速度目標のリード補償）だったが、その場旋回の終端角速度が
0.2rad/s 以下に収まらなかった（実測 90°=6.9rad/s、180°=5.7rad/s）。これを
「車輪速度ループの時定数 tau_v がその場旋回の総所要時間と同程度だから構造的に
無理」と判断したのは**誤りだった**（教授セッションの検算: その場 90° 旋回で
1 輪に要る力は 0.291N、停動時に 1 輪が出せる力は 2.056N で 7.1 倍の余裕があり、
最大角速度時の逆起電力 0.480V に対し必要電圧は 0.904V で電源 3.0V に対し 2.1V
の余りがある。力にも電圧にも大きな余裕があるのに追従できていなかったのは、
**速度 PI が必要な電圧を出していなかっただけ**である）。

そこで角速度目標のかさ上げをやめ、**計画から必要な電圧を直接計算して足す**
方式（逆動力学前置き）に切り替えた。DC モータの逆モデル
（`V = I·Rw + N·Ke·ω_wheel`、`τ_wheel = N·Kt·I`、`F = τ_wheel/r` より
`V_ff = F·r·Rw/(N·Kt) + N·Ke·ω_wheel_ref`）を使い、各輪に要る力を計画の
並進加速度 `a_ref`・角加速度 `alpha_ref` から求める:
    F_common = M_eff·a_ref/2 + F_fric/2 + c_eff·v_ref/2   （走行抵抗・粘性を含む）
    F_diff   = I_eff·alpha_ref / tread                     （純ヨーぶん）
    F_L = F_common - F_diff,  F_R = F_common + F_diff
    V_ff_L = F_L·r·Rw/(N·Kt) + N·Ke·omega_l_target
    V_ff_R = F_R·r·Rw/(N·Kt) + N·Ke·omega_r_target
最終指令は `V = clip(V_ff + kp_wheel*err + ki_wheel*integ(err), ±Vmax)`
（`_wheel_pi` 参照。PI は残差だけを担う 2 自由度制御）。`a_ref`・`alpha_ref`
の求め方:
  - 経路追従: `a_ref(s)` はセル内の運動学 `v[i+1]²=v[i]²+2·a·Δs` から求める
    （`dv/ds × v_ref(s)` の連鎖律ではない。連鎖律は `v_ref(s)=0` の点で
    厳密に 0 になってしまい、静止発進直後の a_ref が消えてしまう。
    運動学の式なら `v[i]=0` でも `v[i+1]>0` である限り 0 にならない）。
    `alpha_ref(s) = a_ref(s) * kappa_ref(s)`（kappa はセル内区分定数の近似）。
  - その場旋回: `a_ref=0`（v_cmd=0 のため）、`alpha_ref` は加速/減速フェーズで
    `±alpha`、定常フェーズで `0`（`_spin_kinematics()` が返す符号付き値）。
`TrackerGains.use_voltage_feedforward=False` にすると、この項の計算そのものを
一切行わない（否定対照の土台。下記 update() 参照）。

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
# 車輪速度 PI・方位フィードバックゲイン。
#
# 🔴 逆動力学（電圧）前置きの追加（2026-08-20 是正）に伴い、経路追従と
# その場旋回で別々に実測スイープし直した。逆動力学前置きの導入前は
# `classic/motion.py` の DEFAULT_KP_WHEEL/DEFAULT_KI_WHEEL（0.025/0.2）を
# そのまま複写していたが、前置きが必要な電圧の大半を計画から直接計算する
# ようになった結果、フィードバック（PI）が担うのは残差の補正だけでよくなり、
# 前置き導入前の PI ゲインのままだと二重に補正して振動・オーバーシュートを
# 起こすことを実測で確認した（`kp_wheel` を上げるほど直線の到達が乱れ、
# `kp_wheel=0` 付近＋`ki_wheel` だけのほうが素直に収束した）。経路追従と
# その場旋回は要求される車輪角速度の桁（前者は数 rad/s〜数十 rad/s、後者は
# 数十〜100rad/s級・560rad/s²級の急加減速）が大きく異なるため、`kp_wheel`・
# `ki_wheel`・`kp_psi` のいずれも別々に持つ（`_spin` 接尾辞側がその場旋回用）。
#
# 経路追従側（直線4区画の停止→停止計画、+4deg注入・四分円・直線→弧→直線を
# 横断して比較）: kp_wheel は 0 付近が最も素直（0.0～0.01 の間で実測/計画比
# 1.03～1.15）、ki_wheel=0.4～0.5 が良好。kp_psi は四分円のヨー到達（目標
# 90°±5°）で決めた（4→87.0°・6→88.8°・8→92.5°・…・18→99.2° と単調に
# 過回頭が増える。6.0 が最も 90° に近い）。
DEFAULT_KP_WHEEL: float = 0.008
DEFAULT_KI_WHEEL: float = 0.5
DEFAULT_KP_PSI: float = 6.0

# その場旋回側（90°/180°の到達角度誤差・終端角速度を横断してスイープ）:
# `kp_wheel_spin`・`ki_wheel_spin`・`kp_psi_spin` を同時にスイープし、両角度
# とも到達角度誤差が3°未満に収まり、終端角速度が実測で最も小さくなる帯
# （kp_wheel_spin=0.05・ki_wheel_spin=0.04・kp_psi_spin=18 付近）を採用した。
# 終端角速度は 0.2rad/s 以下という目標には届かなかった（実測 90°=0.77rad/s、
# 180°=0.36rad/s。逆動力学前置き導入前の 6.9/5.7rad/s からは 9〜16 倍改善）。
# 詳細・スイープの生ログは本タスクの完了報告を参照。
DEFAULT_KP_WHEEL_SPIN: float = 0.05
DEFAULT_KI_WHEEL_SPIN: float = 0.04
DEFAULT_KP_PSI_SPIN: float = 18.0

# 横ずれ -> 方位 の変換ゲイン [1/m]。既定 0（無効）。`apply_lateral_correction()`
# の差し込み口自体は用意するが、壁センサからの実際の推定・結線は本段の範囲外
# （`classic/localization.py` 側の仕事。ここでは「否定対照の土台」として
# kp_lat=0 のとき経路を一切通らないことだけを保証する）。
DEFAULT_KP_LAT: float = 0.0

# 逆動力学（電圧）前置き（モジュール docstring 参照）。既定 ON。
DEFAULT_USE_VOLTAGE_FEEDFORWARD: bool = True

# 車輪 PI 積分器の絶対値クランプ（classic/motion.py の DEFAULT_INTEG_CLAMP と
# 同じ値・同じ理由: 暴走防止の保険）。
_INTEG_CLAMP: float = 200.0


@dataclass
class TrackerGains:
    """ProfileTracker のゲイン一式。"""

    kp_wheel: float = DEFAULT_KP_WHEEL
    ki_wheel: float = DEFAULT_KI_WHEEL
    kp_wheel_spin: float = DEFAULT_KP_WHEEL_SPIN
    ki_wheel_spin: float = DEFAULT_KI_WHEEL_SPIN
    kp_psi: float = DEFAULT_KP_PSI
    kp_psi_spin: float = DEFAULT_KP_PSI_SPIN
    kp_lat: float = DEFAULT_KP_LAT
    use_voltage_feedforward: bool = DEFAULT_USE_VOLTAGE_FEEDFORWARD


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


def _cell_index(s: float, s_grid: Sequence[float]) -> int:
    """`s` を含むセルの添字（範囲外はクランプ）を返す（`_interp_linear`・
    `_interp_step` の bisect と同じ規約。セル内の運動学から `a_ref` を
    取り出すのに使う）。"""
    if s <= s_grid[0]:
        return 0
    if s >= s_grid[-1]:
        return len(s_grid) - 2
    i = bisect.bisect_right(s_grid, s) - 1
    return min(max(i, 0), len(s_grid) - 2)


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
        # 逆動力学前置きで使う車両物理定数（モジュール docstring 参照）。
        # MuJoCo は読まない（vehicle_limits() は RobotParams の算術のみ）。
        self._limits = vehicle_limits(self.params)

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

    def _a_ref_at(self, s: float) -> float:
        """`s` を含むセル内での計画上の並進加速度 [m/s²]。

        運動学 `v[i+1]²=v[i]²+2·a·Δs` をセル内で解いたもの（`dv/ds × v_ref(s)`
        の連鎖律ではない）。連鎖律は `v_ref(s)=0` の点（静止発進の起点・
        停止着地の終点）で厳密に 0 になってしまうが、運動学の式なら
        `v[i]=0` でも隣の格子点 `v[i+1]>0` である限り 0 にならない
        （モジュール docstring「逆動力学（電圧）前置き」節参照）。
        """
        i = _cell_index(s, self._s_grid)
        s0, s1 = self._s_grid[i], self._s_grid[i + 1]
        if s1 <= s0:
            return 0.0
        v0, v1 = self._v_ref[i], self._v_ref[i + 1]
        return (v1 * v1 - v0 * v0) / (2.0 * (s1 - s0))

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
        spin = spin_turn_time(delta_theta, self._limits)
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
        `omega_ref` の時間微分（符号付き角加速度。逆動力学前置きで使う）。"""
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
        self._last_voltage_saturated = False

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

    @property
    def voltage_saturated(self) -> bool:
        """直近の `update()` で左右いずれかの電圧が `±voltage_limit` に
        張り付いていたか（診断用。「飽和していないのに追従できていない」の
        切り分けに使う）。"""
        return self._last_voltage_saturated

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
            v_ff, w_ff, psi_ref, a_ref, alpha_ref, done = self._path_step()
        else:
            v_ff, w_ff, psi_ref, a_ref, alpha_ref, done = self._spin_step()

        e_psi = self._wrap(psi_ref - self._yaw_est)
        kp_psi = self.gains.kp_psi if self._mode == "path" else self.gains.kp_psi_spin
        w_cmd = w_ff + kp_psi * e_psi
        # kp_lat==0（既定）のとき、または旋回モードのときは、この分岐そのものを
        # 通らない（横ずれ補正の経路が一切実行されないことの保証。否定対照の土台）。
        if self.gains.kp_lat != 0.0 and self._mode == "path":
            w_cmd += self.gains.kp_lat * self._e_lat
        v_cmd = v_ff

        omega_l_target = (v_cmd - w_cmd * tread / 2.0) / r
        omega_r_target = (v_cmd + w_cmd * tread / 2.0) / r

        # 逆動力学（電圧）前置き（モジュール docstring 参照）。
        # use_voltage_feedforward=False のときはこの分岐そのものを通らない
        # （否定対照の土台）。
        if self.gains.use_voltage_feedforward:
            v_ff_l, v_ff_r = self._voltage_feedforward(
                v_ff, a_ref, alpha_ref, omega_l_target, omega_r_target)
        else:
            v_ff_l = v_ff_r = 0.0

        if self._mode == "path":
            kp_wheel, ki_wheel = self.gains.kp_wheel, self.gains.ki_wheel
        else:
            kp_wheel, ki_wheel = self.gains.kp_wheel_spin, self.gains.ki_wheel_spin
        vl, vr = self._wheel_pi(omega_l_target, omega_r_target, omega_l, omega_r, dt,
                                 kp_wheel, ki_wheel, v_ff_l, v_ff_r)
        return vl, vr, done

    def _voltage_feedforward(self, v_ref: float, a_ref: float, alpha_ref: float,
                              omega_l_target: float, omega_r_target: float):
        """計画上の並進加速度 `a_ref`・角加速度 `alpha_ref` から、各輪に必要な
        電圧 `(V_ff_L, V_ff_R)` を DC モータの逆モデルで直接計算する
        （モジュール docstring「逆動力学（電圧）前置き」節参照。フィード
        バック成分（kp_psi*e_psi 等）は含めない。純フィードフォワード）。"""
        lim = self._limits
        p = self.params
        tread = p.tread
        r = p.wheel_radius
        N = p.gear_ratio
        Rw = p.motor_R
        Kt = p.motor_Kt
        Ke = p.motor_Ke

        F_common = lim.M_eff * a_ref / 2.0 + lim.F_fric / 2.0 + lim.c_eff * v_ref / 2.0
        F_diff = lim.I_eff * alpha_ref / tread
        F_l = F_common - F_diff
        F_r = F_common + F_diff

        k_torque = r * Rw / (N * Kt)  # F -> V(トルク分)の変換係数
        v_ff_l = F_l * k_torque + N * Ke * omega_l_target
        v_ff_r = F_r * k_torque + N * Ke * omega_r_target
        return v_ff_l, v_ff_r

    def _path_step(self):
        """経路追従モードの 1 ティック分の
        (v_ff, w_ff, psi_ref, a_ref, alpha_ref, done)。"""
        s = self._s
        v_ff = self._v_at(s)
        kappa_ref = self._kappa_at(s)
        w_ff = v_ff * kappa_ref
        psi_ref = self._psi_at(s)

        a_ref = self._a_ref_at(s)
        alpha_ref = a_ref * kappa_ref  # kappa はセル内区分定数の近似（モジュール docstring 参照）

        done = self._s >= self._s_end
        return v_ff, w_ff, psi_ref, a_ref, alpha_ref, done

    def _spin_step(self):
        """その場旋回モードの 1 ティック分の
        (v_ff=0, w_ff=omega_ref, psi_ref, a_ref=0, alpha_ref, done)。"""
        psi_ref, omega_ref, alpha_ref = self._spin_kinematics(self._t)
        done = self._t >= self._spin_time_total
        return 0.0, omega_ref, psi_ref, 0.0, alpha_ref, done

    def _wheel_pi(self, omega_l_target, omega_r_target, omega_l, omega_r, dt,
                  kp_wheel, ki_wheel, v_ff_l=0.0, v_ff_r=0.0):
        """車輪角速度目標値と実測値の偏差から PI で左右電圧を作る（アンチ
        ワインドアップ付き）。`classic/motion.py` の `_wheel_pi` に、計画から
        直接計算した電圧前置き `v_ff_l`/`v_ff_r` を足す形（2 自由度制御。
        モジュール docstring「逆動力学（電圧）前置き」節参照）を加えたもの。
        `v_ff_l=v_ff_r=0.0`（既定引数）のときは `classic/motion.py` の
        `_wheel_pi` と一字一句同じ式になる。`kp_wheel`・`ki_wheel` は呼び出し側
        `update()` がモードに応じて `gains.kp_wheel`/`gains.ki_wheel`（経路追従）
        または `gains.kp_wheel_spin`/`gains.ki_wheel_spin`（その場旋回）を選んで
        渡す（理由は `DEFAULT_KP_WHEEL_SPIN`/`DEFAULT_KI_WHEEL_SPIN` 直上の
        コメント参照。経路追従とその場旋回で要求される車輪 PI の応答速度が
        大きく異なるため分離した）。"""
        vlim = self.params.voltage_limit

        err_l = omega_l_target - omega_l
        err_r = omega_r_target - omega_r

        # アンチワインドアップ: 前置き込みの電圧が上限に張り付いていない間だけ積分する
        vl_try = v_ff_l + kp_wheel * err_l + ki_wheel * self._integ_l
        vr_try = v_ff_r + kp_wheel * err_r + ki_wheel * self._integ_r
        if abs(vl_try) < vlim:
            self._integ_l = float(np.clip(self._integ_l + err_l * dt, -_INTEG_CLAMP, _INTEG_CLAMP))
        if abs(vr_try) < vlim:
            self._integ_r = float(np.clip(self._integ_r + err_r * dt, -_INTEG_CLAMP, _INTEG_CLAMP))

        vl_raw = v_ff_l + kp_wheel * err_l + ki_wheel * self._integ_l
        vr_raw = v_ff_r + kp_wheel * err_r + ki_wheel * self._integ_r
        vl = float(np.clip(vl_raw, -vlim, vlim))
        vr = float(np.clip(vr_raw, -vlim, vlim))
        self._last_voltage_saturated = abs(vl_raw) >= vlim or abs(vr_raw) >= vlim
        return vl, vr

    @staticmethod
    def _wrap(angle: float) -> float:
        """角度を [-pi, pi) に正規化する。"""
        return math.atan2(math.sin(angle), math.cos(angle))
