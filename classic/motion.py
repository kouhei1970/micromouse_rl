"""
classic/motion.py
================
区画単位の動作（直進 N 区画／その場 90° 旋回（左右）／その場 180° 旋回／停止）を
生成する簡素な速度制御器。実機のマウサー用語でいう
**「超信地旋回走行（区画ごと停止）」**（`docs/JA_ENGINEERING_TERMS.md`）を再現する。

【推測航法のみを使う（真値位置は使わない）】
`mouse.sim.MouseSim.privileged_pose()` は一切参照しない。走行開始時の姿勢
（`heading_deg`）は競技規約上あらかじめ分かっている既知情報（評価器が
`reset_to_start(cell=(0,0), heading_deg=90.0)` で置く、というプロトコル上の
定数であり、走行中の真値ではない）としてのみ使い、以後は
**車輪角速度の積算（並進距離）とジャイロ角速度の積算（方位）のみ**で
自機の移動量を推定する（note_030 §3 の S1 出口条件: 壁補正は S2 の担当、
ここでは推測航法のみでよい）。

【制御構成（2 段構成）】
  上位（区画単位の司令）: 残距離・残ヨー角への比例制御で
    「左右それぞれの車輪角速度の目標値」を作る。直進中は方位を保持する
    差動補正（Kp_heading）も同じ比例則で重ね合わせる。
  下位（車輪速度ループ）: 車輪角速度目標値と実測車輪角速度
    （observation() の車輪角速度 2 チャンネル）の偏差を PI 制御で
    左右モータ電圧へ変換する（アンチワインドアップ付き: 出力が飽和して
    いる間は積分しない）。

【PI ゲインの決め方（実測に基づく。決め打ちではない）】
`docs/ROBOT_SPEC.md` §2.5 の実測較正値（速度時定数 τ_v=0.124 s、
最高速度 v_max=3.84 m/s @ V_max=3.0V → 並進 DC ゲイン K≈v_max/V_max×(1/r)
≈ 94.8 rad/s/V）を出発点に、IMC（内部モデル制御）流の PI 設計
Kp=τ/(K·λ), Ki=Kp/τ（λ: 閉ループ時定数、ここでは λ=0.05s 相当を狙う）で
概算した値を初期値とし、実際にシミュレータで単一車輪の速度ステップ応答を
とって発振しないことを確認して微調整した
（`tests/test_classic_sensing.py` ではなく、本ファイル作成時に手元で
実行したチューニングスクリプトでの確認。最終値は下記 DEFAULT_* 定数）。

実測した動作精度（開通路 5x5 迷路、区画 (2,2) 開始、詳細は
`tests/test_classic_motion.py` 参照）:
  - 直進 2 区画（目標 0.36 m）: 実移動 0.3562 m（誤差 -3.8 mm）、
    方位ずれは実質ゼロ
    【訂正 2026-08-19・note_029 検分: 「heading hold が効いている」という
    旧記述は測定が支持しない。外乱の無い対称なシミュレータでは、
    kp_heading をゼロに潰しても直進中に方位はそもそもずれない
    （左右車輪が対称に動くので曲がりようがない）。方位ずれが実質ゼロ
    なのはそのためであり、heading hold の効果の証拠ではない。
    heading hold（差動補正）自体が実際に効いていることは、走行開始
    直後に推測航法のヨー推定へ人為的な初期誤差を注入する検査
    （tests/test_classic_motion.py の
    test_default_heading_hold_corrects_an_injected_yaw_estimate_error、
    対照は test_zero_kp_heading_does_not_correct_the_injected_yaw_error）
    で別途、実測により確認している。】
  - その場 90° 左旋回: 実ヨー変化 +89.06°（誤差 0.94°）、
    旋回中の位置ずれ 0.092 mm、所要 200 ティック(2.00s)
  - その場 90° 右旋回: 実ヨー変化 -89.06°（誤差 0.94°）、所要 200 ティック(2.00s)
  - その場 180° 旋回: 実ヨー変化 +178.93°（誤差 1.07°）、位置ずれ 0.262 mm、
    所要 332 ティック(3.32s)
    【2026-08-19 是正・DEFAULT_OMEGA_SETTLE 0.05→0.20（探索走行の旋回過多の
    是正、詳細は下記 DEFAULT_OMEGA_SETTLE 直上のコメントと
    `research_notes/note_030_classical_rebuild_plan.md` 任務指示を参照）。
    是正前は 90° 旋回 476 ティック(4.76s)・180° 旋回 608 ティック(6.08s)
    だった。値の変化は「制御が悪化した」のではなく「完了判定を緩めて
    早く止めるようにした」ことによる（誤差は 0.44→0.94° 等に増えたが、
    いずれも下記の S1 出口条件には十分小さい）。】

いずれも S1 の出口条件（低速・確実。速さは目的ではない）に対して十分な精度。

【S2: 壁センサによる区画ごとの位置補正（note_030 §3 S2）】
本ファイル自体は壁は一切見ない（推測航法のみ、という設計は変わらない）。
補正は `classic/localization.py` の `Localizer` が持ち、本ファイルは補正量の
「差し込み口」を 2 つ提供するだけである:

  (a) 横位置補正 → `bias_target_heading()`。直進コマンド発行直後に呼ぶと
      目標方位にバイアスが乗る（呼び出し元は `classic/explorer.py`）。
  (b) 前後位置補正 → コンストラクタに `localizer` を渡すと、FORWARD 実行中
      毎ティック `Localizer.correct_forward_remaining()` を通して残距離が
      補正される。

`localizer=None`（既定）のときは a も b も一切発動せず、**本ファイルの
コード経路は S2 導入前と完全に同一**である（否定対照・空振り防止検査の
土台。詳細は `classic/localization.py` docstring）。
"""
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

import numpy as np

from mouse.params import RobotParams

if TYPE_CHECKING:  # 循環 import 回避（localization.py は motion.py に依存しない）
    from classic.localization import Localizer


class MotionKind(Enum):
    """区画単位の動作の種別（実機のターン種別に相当する語彙）。"""
    STOP = auto()          # 停止（速度指令ゼロを保持）
    FORWARD = auto()       # 直進 N 区画
    TURN_LEFT_90 = auto()  # その場 90° 左旋回（超信地旋回）
    TURN_RIGHT_90 = auto()  # その場 90° 右旋回（超信地旋回）
    TURN_180 = auto()      # その場 180° 旋回（超信地旋回）


# ------------------------------------------------------------------
# 既定ゲイン・速度上限（モジュール docstring に決定根拠を記載）
# ------------------------------------------------------------------
DEFAULT_KP_WHEEL: float = 0.025   # 車輪速度 PI の比例ゲイン [V/(rad/s)]
DEFAULT_KI_WHEEL: float = 0.2     # 車輪速度 PI の積分ゲイン [V/(rad/s)/s]（積分器へは誤差を dt 積算）
DEFAULT_V_CRUISE: float = 0.12    # 直進の巡航速度上限 [m/s]（v_max=3.84m/s の約 3%。低速優先）
DEFAULT_OMEGA_CRUISE: float = 1.2  # 旋回の角速度上限 [rad/s]（≈69°/s）
DEFAULT_KP_DIST: float = 3.0      # 残距離 → 目標並進速度 の比例ゲイン [1/s]
DEFAULT_KP_YAW: float = 3.0       # 残ヨー角 → 目標角速度 の比例ゲイン [1/s]
DEFAULT_KP_HEADING: float = 2.0   # 直進中の方位保持（差動補正）の比例ゲイン [1/s]
DEFAULT_DISTANCE_TOL: float = 0.003   # 直進の完了判定距離許容 [m]
DEFAULT_YAW_TOL: float = math.radians(1.0)  # 旋回の完了判定角度許容 [rad]
DEFAULT_SPEED_SETTLE: float = 0.01    # 直進完了判定の速度しきい値 [m/s]
# 旋回完了判定の角速度しきい値 [rad/s]。
#
# 🔴 2026-08-19 是正（探索走行が旋回だけで持ち時間の半分以上を消費していた
# 問題の是正。任務指示・`research_notes/note_030_classical_rebuild_plan.md`）。
#
# 【計装で確定した原因】design_v4 maze_42134（壁補正なし）を実走させ、
# plan_id ごとのティック列を「連続する同一 plan_id の区間」に分けて集計した
# ところ、旋回 1 回（1 コマンド）あたりのティック数が
#   90° 旋回: 平均 474.5〜458.9 ティック（4.7〜4.6 秒）
#   180° 旋回: 平均 606.7 ティック（6.07 秒）
# と、ほぼ一定値に高止まりしていた（同一区画内で旋回方向が何度も入れ替わる
# 「振動」ではない。57 回の旋回コマンドのうち、同一区画内で 2 回以上旋回した
# のは 5 区画のみ、前の方位へ戻る「往復」は 6 回のみ。歩数マップの候補方位が
# 複数(タイ)で現在方位も候補に入っていたのに別方位を選んでいたケースも
# 57 回中 2 回だけ）。→ 原因は「振動」でも「旋回の回数自体が多い」でもなく、
# **旋回 1 回そのものが遅い**ことだった。
#
# 単体の90°旋回を時系列で追ったところ（`_wheel_pi` の車輪速度追従の遅れが
# 原因）、旋回角度そのものは実測 2.0 秒付近で既に許容誤差(1°)内に収まって
# いたが、そこから機体が回頭しすぎて最大 +4.4° オーバーシュートし、それを
# 車輪PIの遅い応答で打ち消すのにさらに約2.7秒を要していた。つまり旋回時間の
# 半分以上は「角度はほぼ合っているのに、角速度がほぼゼロになるのを待つ」
# 完了判定(旧値 0.05 rad/s ≈ 2.9°/s)のために空費されていた。
#
# 車輪速度PI(DEFAULT_KP_WHEEL/DEFAULT_KI_WHEEL)を強めて追従を速くする案は
# 実測で却下した: ゲインを上げると（直進とは共有ゲインであるにも関わらず）
# 旋回中に左右車輪が非対称に暴れて超信地旋回が崩れ、位置ずれが最大190mm超に
# 発散した（既定ゲインは実際にスイープして確認した安定限界に近い）。
# 外側のヨー制御(DEFAULT_KP_YAW/DEFAULT_OMEGA_CRUISE)を緩める案も試したが、
# オーバーシュート自体は 3.3〜4.8° 程度で高止まりしたまま所要時間だけが
# 悪化した（例: kp_yaw=0.5 でも旋回に 12.4 秒かかった）。
#
# 残ったレバーが本しきい値である。0.05→0.20 rad/s（≈11.5°/s、
# DEFAULT_OMEGA_CRUISE=1.2 rad/s の約 17%）へ緩めると、実測で
#   90° 旋回: 476 ティック → 200 ティック（-58%）、角度誤差 0.44°→0.94°
#   180° 旋回: 608 ティック → 332 ティック（-45%）、角度誤差 0.48°→1.07°
# となり、旋回中の位置ずれ(<0.3mm)・完了角度誤差(<1.1°)とも
# `tests/test_classic_motion.py` の判定基準（位置ずれ<10mm・角度誤差<3°）に
# 十分な余裕を残したまま所要時間を大きく削れることを確認した
# （数値は `tests/test_classic_motion.py::test_turn_reaches_target_angle_without_translating`
# が実行のたびに再実測して検査する）。0.05 に戻すと
# `tests/test_classic_motion.py::test_turn_completes_within_a_tick_budget_at_default_gains`
# が落ちる（旋回1回が長すぎることの直接検査）。
DEFAULT_OMEGA_SETTLE: float = 0.20
DEFAULT_INTEG_CLAMP: float = 200.0    # 積分器の絶対値クランプ（暴走防止の保険）


@dataclass
class MotionCommand:
    """発行済みの区画単位コマンド（診断・テスト用に読み出せるように保持する）。"""
    kind: MotionKind
    n_cells: int = 1  # FORWARD のみ意味を持つ


class CellMotionController:
    """区画単位の動作を生成する速度制御器。

    使い方:
        ctrl = CellMotionController(params)
        ctrl.reset(heading_deg=90.0)     # 走行開始時に一度だけ
        ctrl.start_forward(1)            # 1 区画前進を発行
        while True:
            obs = sim.observation()
            vl, vr, done = ctrl.update(obs)
            sim.step_control(vl, vr)
            if done:
                break

    **真値位置（privileged_pose）は一切使わない。**内部状態は
    「車輪角速度・ジャイロの積算による推測航法」のみで更新する
    （壁センサによる補正は S2 の担当。ここでは持たない）。
    """

    def __init__(self, params: Optional[RobotParams] = None,
                 kp_wheel: float = DEFAULT_KP_WHEEL, ki_wheel: float = DEFAULT_KI_WHEEL,
                 v_cruise: float = DEFAULT_V_CRUISE, omega_cruise: float = DEFAULT_OMEGA_CRUISE,
                 kp_dist: float = DEFAULT_KP_DIST, kp_yaw: float = DEFAULT_KP_YAW,
                 kp_heading: float = DEFAULT_KP_HEADING,
                 distance_tol: float = DEFAULT_DISTANCE_TOL, yaw_tol: float = DEFAULT_YAW_TOL,
                 speed_settle: float = DEFAULT_SPEED_SETTLE, omega_settle: float = DEFAULT_OMEGA_SETTLE,
                 localizer: Optional["Localizer"] = None):
        self.params = params if params is not None else RobotParams()
        self.kp_wheel = kp_wheel
        self.ki_wheel = ki_wheel
        self.v_cruise = v_cruise
        self.omega_cruise = omega_cruise
        self.kp_dist = kp_dist
        self.kp_yaw = kp_yaw
        self.kp_heading = kp_heading
        self.distance_tol = distance_tol
        self.yaw_tol = yaw_tol
        self.speed_settle = speed_settle
        self.omega_settle = omega_settle
        # S2: 壁センサによる区画ごとの位置補正（None=既定は完全に無効。
        # `classic/localization.py` docstring 参照）。
        self.localizer = localizer

        self._n_sensors = len(self.params.sensors)
        self.reset()

    # ------------------------------------------------------------------
    # 状態リセット・観測の切り出し
    # ------------------------------------------------------------------
    def reset(self, heading_deg: float = 90.0) -> None:
        """走行開始時に一度だけ呼ぶ。推測航法の内部状態を初期化する。

        heading_deg は評価器が走行開始時に機体を置く既知の初期方位
        （競技プロトコル上の定数）であり、走行中の真値取得ではない。
        """
        self._yaw_est = math.radians(heading_deg)
        self._dist_est = 0.0
        self._target_dist = 0.0
        self._target_heading = self._yaw_est
        self._target_yaw = self._yaw_est
        self._integ_l = 0.0
        self._integ_r = 0.0
        self._cmd: Optional[MotionCommand] = None

    def _split_obs(self, obs: np.ndarray):
        """observation() から車輪角速度・ジャイロ z 成分を取り出す
        （並びは mouse/sim.py の observation() docstring どおり:
        [距離×n, 加速度×3, ジャイロ×3, 車輪角速度L, 車輪角速度R]）。"""
        n = self._n_sensors
        gyro_z = float(obs[n + 5])
        omega_l = float(obs[n + 6])
        omega_r = float(obs[n + 7])
        return gyro_z, omega_l, omega_r

    @property
    def yaw_estimate(self) -> float:
        """推測航法によるヨー角推定 [rad]（真値ではなく積算値）。"""
        return self._yaw_est

    @property
    def is_idle(self) -> bool:
        return self._cmd is None

    # ------------------------------------------------------------------
    # コマンド発行
    # ------------------------------------------------------------------
    def start_forward(self, n_cells: int) -> None:
        """直進 n_cells 区画を開始する（方位は発行時の推定方位を保持する）。"""
        if n_cells <= 0:
            raise ValueError(f"n_cells は正の整数で指定してください: {n_cells}")
        self._cmd = MotionCommand(MotionKind.FORWARD, n_cells)
        self._dist_est = 0.0
        self._target_dist = n_cells * self.params.cell_size
        self._target_heading = self._yaw_est
        self._integ_l = 0.0
        self._integ_r = 0.0

    def bias_target_heading(self, delta_rad: float) -> None:
        """直進の目標方位に補正角を足し込む（S2 (a) 横位置補正の差し込み口）。

        `start_forward()` の直後に呼ぶこと（`start_forward` が
        `_target_heading` を現在のヨー推定へリセットするため、その後で
        バイアスを足さないと上書きされて消える）。呼ばなければ（=補正なし）
        従来どおり `_target_heading == 発行時の _yaw_est` のまま変わらない。
        """
        self._target_heading += delta_rad

    @property
    def cells_completed(self) -> int:
        """現在の FORWARD コマンドで既に完了した区画数（note_030 §3 S3）。

        `int(self._dist_est / self.params.cell_size)` で求める。多区画直進
        （`start_forward(n)`, n>1）の途中で「何番目の区画中心を通過したか」を
        呼び出し側（`classic/explorer.py` の `tick`）が検出するために使う。
        FORWARD 以外のコマンド実行中（旋回・停止・アイドル）は常に 0
        （区画を積算する概念がそもそも無い）。"""
        if self._cmd is None or self._cmd.kind is not MotionKind.FORWARD:
            return 0
        return int(self._dist_est / self.params.cell_size)

    def reanchor_heading(self, bias_rad: float) -> None:
        """多区画直進の途中、区画中心を通過するたびに目標方位を「現在の
        ヨー推定を基準に置き直す」（note_030 §3 S3 (a)）。

        🔴 既存の `bias_target_heading()`（現在の `_target_heading` への
        **加算**）とは別物として用意してある。区画中心での掛け直しは
        「現在のヨー推定を基準に置き直す」処理であり、
        `_target_heading = self._yaw_est + bias_rad` と**上書き**する
        （加算ではない）。

        これは `start_forward()` が `_target_heading = self._yaw_est` へ
        リセットしてから `bias_target_heading()` で加算する、という
        1 区画コマンドの手順と**同じ意味**になる（`start_forward` 直後の
        `_yaw_est` を基準に置くのと、直進途中の「今」の `_yaw_est` を基準に
        置くのとで、やっていることは同一）。これにより、S2 で実測済みの
        「1 区画ごとに読み直して次の 1 区画へバイアスをかける」挙動が、
        多区画直進の中でもそのまま再現される（n 区画を 1 コマンドにしても、
        出発点で 1 回測った横ずれが n 区画ぶん一定のバイアスとして効き
        続けることはない）。"""
        self._target_heading = self._yaw_est + bias_rad

    def start_turn_left(self) -> None:
        """その場 90° 左旋回（超信地旋回）を開始する。左旋回はヨー角を正方向へ増やす
        （mouse.sim.MouseSim.privileged_pose() と同じ CCW 正の規約）。"""
        self._start_turn(math.radians(90.0))
        self._cmd = MotionCommand(MotionKind.TURN_LEFT_90)

    def start_turn_right(self) -> None:
        """その場 90° 右旋回を開始する。"""
        self._start_turn(math.radians(-90.0))
        self._cmd = MotionCommand(MotionKind.TURN_RIGHT_90)

    def start_turn_180(self) -> None:
        """その場 180° 旋回を開始する。"""
        self._start_turn(math.radians(180.0))
        self._cmd = MotionCommand(MotionKind.TURN_180)

    def start_stop(self) -> None:
        """速度指令ゼロを保持するコマンドを発行する（update() は常に done=True を返す）。"""
        self._cmd = MotionCommand(MotionKind.STOP)
        self._integ_l = 0.0
        self._integ_r = 0.0

    def _start_turn(self, delta_yaw: float) -> None:
        self._target_yaw = self._yaw_est + delta_yaw
        self._integ_l = 0.0
        self._integ_r = 0.0

    # ------------------------------------------------------------------
    # 制御ステップ
    # ------------------------------------------------------------------
    def update(self, obs: np.ndarray):
        """1 制御ステップ分の (v_left, v_right, done) を返す。

        呼び出し側は返された電圧を sim.step_control() へそのまま渡し、
        done=True になったら次のコマンドを発行すること（このメソッド自体は
        コマンドを進めない。呼び出し側が明示的に start_* を呼ぶ）。
        """
        if self._cmd is None:
            raise RuntimeError("start_forward()/start_turn_*()/start_stop() のいずれかを先に呼んでください")

        dt = self.params.control_dt
        r = self.params.wheel_radius
        tread = self.params.tread
        gyro_z, omega_l, omega_r = self._split_obs(obs)

        # 推測航法の積算（車輪角速度 → 並進距離、ジャイロ → 方位）
        v_meas = (omega_l + omega_r) / 2.0 * r
        self._dist_est += v_meas * dt
        self._yaw_est += gyro_z * dt

        kind = self._cmd.kind

        if kind == MotionKind.STOP:
            return 0.0, 0.0, True

        if kind == MotionKind.FORWARD:
            remaining = self._target_dist - self._dist_est
            # S2 (b) 前後位置補正の差し込み口: localizer が無ければ従来どおり
            # 推測航法だけの remaining（この if 文自体を通らないので計算経路も
            # 完全に元のまま）。
            if self.localizer is not None:
                remaining = self.localizer.correct_forward_remaining(remaining, obs)
            v_cmd = float(np.clip(self.kp_dist * remaining, -self.v_cruise, self.v_cruise))

            heading_err = self._wrap(self._target_heading - self._yaw_est)
            domega = self.kp_heading * heading_err

            omega_l_target = (v_cmd - domega * tread / 2.0) / r
            omega_r_target = (v_cmd + domega * tread / 2.0) / r
            vl, vr = self._wheel_pi(omega_l_target, omega_r_target, omega_l, omega_r, dt)

            done = abs(remaining) < self.distance_tol and abs(v_meas) < self.speed_settle
            if done:
                return 0.0, 0.0, True
            return vl, vr, False

        if kind in (MotionKind.TURN_LEFT_90, MotionKind.TURN_RIGHT_90, MotionKind.TURN_180):
            remaining = self._wrap(self._target_yaw - self._yaw_est)
            omega_cmd = float(np.clip(self.kp_yaw * remaining, -self.omega_cruise, self.omega_cruise))

            v_r = omega_cmd * tread / 2.0
            v_l = -omega_cmd * tread / 2.0
            omega_l_target = v_l / r
            omega_r_target = v_r / r
            vl, vr = self._wheel_pi(omega_l_target, omega_r_target, omega_l, omega_r, dt)

            done = abs(remaining) < self.yaw_tol and abs(gyro_z) < self.omega_settle
            if done:
                return 0.0, 0.0, True
            return vl, vr, False

        raise AssertionError(f"未対応の MotionKind: {kind}")

    def _wheel_pi(self, omega_l_target, omega_r_target, omega_l, omega_r, dt):
        """車輪角速度目標値と実測値の偏差から PI で左右電圧を作る（アンチワインドアップ付き）。"""
        vlim = self.params.voltage_limit

        err_l = omega_l_target - omega_l
        err_r = omega_r_target - omega_r

        vl_try = self.kp_wheel * err_l + self.ki_wheel * self._integ_l
        vr_try = self.kp_wheel * err_r + self.ki_wheel * self._integ_r
        # アンチワインドアップ: 出力が電圧上限に張り付いていない間だけ積分する
        if abs(vl_try) < vlim:
            self._integ_l = float(np.clip(self._integ_l + err_l * dt, -DEFAULT_INTEG_CLAMP, DEFAULT_INTEG_CLAMP))
        if abs(vr_try) < vlim:
            self._integ_r = float(np.clip(self._integ_r + err_r * dt, -DEFAULT_INTEG_CLAMP, DEFAULT_INTEG_CLAMP))

        vl = float(np.clip(self.kp_wheel * err_l + self.ki_wheel * self._integ_l, -vlim, vlim))
        vr = float(np.clip(self.kp_wheel * err_r + self.ki_wheel * self._integ_r, -vlim, vlim))
        return vl, vr

    @staticmethod
    def _wrap(angle: float) -> float:
        """角度を [-pi, pi) に正規化する。"""
        return math.atan2(math.sin(angle), math.cos(angle))
