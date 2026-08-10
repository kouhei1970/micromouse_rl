"""
mouse/params.py
================
マイクロマウス ロボット v2 の物理パラメータ定義（rev.B）。

数値の出典はすべて `docs/MODEL_VERIFICATION_PLAN.md` §4.2（目標仕様パラメータ表、
mouse_v2 確定値）であり、本ファイルでは同表の値を一字一句そのまま使用する
（独自の丸め・変更はしない）。棄却済み所見 R1/R2（同計画書 §2.2）の数値は使用しない。

rev.A（旧草稿）からの主な変更点:
  - DC モータ電気定数を FA-130RA-18100(3V定格) 相当・減速比 5:1 に更新
  - 車輪ジョイントに frictionloss（乾性摩擦）を新設、damping を縮小
  - 電池ボディを新設し mein_body1 の質量から分離
  - 接触摩擦（床・機体外殻/壁・キャスター<pair>）と solref を明示値化
  - 物理タイムステップを 0.001 → 0.0005s に短縮（C2 是正: timeconst ≥ 2Δt を満たす比 4 を確保）
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RobotParams:
    """マイクロマウス v2 の全物理定数をまとめた dataclass。"""

    # ------------------------------------------------------------------
    # 寸法（v1 踏襲。計画書 §4.2: 「現行踏襲」）
    # ------------------------------------------------------------------
    wheel_radius: float = 0.0135   # 車輪半径 r [m]。直径27mm（Decimus2実装27mm・Harrison指針「最小26mm」）
    tread: float = 0.072           # トレッド L（車輪間隔）[m]。車輪 pos は y = ±0.036 = ±tread/2
    cell_size: float = 0.18        # 迷路セルサイズ [m]（180mm）

    # ------------------------------------------------------------------
    # 質量配分（合計 0.106 kg。計画書 §4.2 質量配分表）
    # ------------------------------------------------------------------
    # mein_body1（シャーシ+PCB）[kg]。電池分離とモータ増量（r3: 15→18g×2）補償後の残余。
    # 総質量 0.106 kg 維持
    mass_body_center: float = 0.029
    # mein_body2〜5（コーナー部）各 [kg]（4個。rev.A 踏襲）
    mass_body_corner: float = 0.0015
    # caster ×2（前後）各 [kg]（rev.A 踏襲）
    mass_caster: float = 0.001
    # motorbox ×2 各 [kg]（r3: FAULHABER 1717 実測質量 18g（公式ページ）。ギヤ・マウント込みの丸め）
    mass_motorbox: float = 0.018
    # 車輪 ×2 各 [kg]。密度 5g/4.008cm^3 = 1.25 g/cm^3（ゴム+樹脂ハブ相当。C1 是正）
    mass_wheel: float = 0.005
    # センサポッド geom 各 [kg]（6ポッド×2geom（円柱+球）=12個。rev.A 踏襲）
    mass_sensor_geom: float = 0.00025
    # 電池（新規。1S LiPo 相当、低位置搭載で転倒余裕確保。minor所見(1)の是正）[kg]
    mass_battery: float = 0.020
    # 合計 = 0.035 + 4*0.0015 + 2*0.001 + 2*0.015 + 2*0.005 + 12*0.00025 + 0.020 = 0.106 kg
    # 慣性は XML compiler の inertiafromgeom="true" により各 geom 質量から自動計算される。

    # 電池ボディ geom（mouse body 直下に追加。計画書 §4.2/§5 手順1-3）
    battery_size: str = "0.0125 0.015 0.004"   # box half-size [m]
    battery_pos: str = "-0.015 0 0.007"        # 機体座標系での位置 [m]（低位置搭載）

    # ------------------------------------------------------------------
    # DC モータ電気モデル（FAULHABER 1717T003SR、コアレス、3V定格。計画書 §4.1/§4.2 改訂 r3
    # = コミット 410000b。FA-130RA は鉄心ロータで精密制御に不適とのユーザー判断により変更）
    # カタログ値出典: https://eshop.faulhaber.com/en/1717T003SR/1717T003SR
    # （データシート PDF: EN_1717_SR_DFF.pdf。無負荷 14100 rpm / 0.0918 A、k_M=1.98 mN・m/A、
    #  k_E=0.207 mV/min^-1(SI 換算 1.977e-3 V・s/rad)、J_rotor=0.59 g・cm^2、質量 18 g）
    # ------------------------------------------------------------------
    motor_R: float = 1.07          # 巻線抵抗 [Ω]（カタログ記載値）
    # 逆起電力定数 [V・s/rad]（カタログ k_E の SI 換算。SI では Kt=Ke が厳密に成立、丸め差のみ）
    motor_Ke: float = 1.98e-3
    # トルク定数 [N・m/A]（カタログ k_M）
    motor_Kt: float = 1.98e-3
    # 無負荷電流 I0 [A]（tau_c = N*Kt*I0 の導出用。カタログ記載値）
    motor_I0: float = 0.0918
    # 減速比 N（計画書 §3.2 ベンチマーク 3.5〜5:1、micromouseonline 計算例 G=5 を採用。r3 でも据置き）
    gear_ratio: float = 5.0

    # 車輪ジョイントの粘性減衰 b [N・m・s/rad]
    # （軸受相当の仮置き。r3 で縮小: Test0 のデータシート無負荷回転数との整合を優先。Test2 で較正）
    wheel_damping: float = 2.0e-6
    # 車輪ジョイントの乾性摩擦トルク tau_c（frictionloss）[N・m] = N*Kt*I0 = 5*1.98e-3*0.0918
    # （データシート無負荷電流由来。ギヤ損は Test2 較正で吸収。クーロン近似は上限側）
    wheel_frictionloss: float = 9.1e-4

    # ロータ慣性（モータ軸換算）J_rotor [kg・m^2] = 0.59 g・cm^2（データシート実測値。
    # r3: コアレス化で旧仮定値 3.2e-7 の約 1/5.4）
    rotor_inertia: float = 5.9e-8
    # 車輪ジョイントの armature（車輪軸換算ロータ慣性）[kg・m^2] = N^2 * J_rotor（下記 property 参照）

    # 印加電圧クランプ [V]（モータ定格。ctrlrange ±3、ctrl の単位は電圧）
    voltage_limit: float = 3.0

    # ------------------------------------------------------------------
    # 制御・物理周期（計画書 §4.2）
    # ------------------------------------------------------------------
    control_dt: float = 0.01       # 制御周期 100Hz [s]
    # 物理タイムステップ [s]。C2 是正: solref timeconst(0.002s) ≥ 2*physics_dt を要求する
    # 数値安定条件に対し比4を確保（1制御ステップ = control_dt/physics_dt = 20 サブステップ）
    physics_dt: float = 0.0005

    # ------------------------------------------------------------------
    # 判定しきい値
    # ------------------------------------------------------------------
    collision_force_threshold: float = 1.0    # 衝突判定: 接触法線力のしきい値 [N]
    tipover_zaxis_threshold: float = 0.5       # 転倒判定: 車体 z 軸の世界 z 成分のしきい値

    # ------------------------------------------------------------------
    # 接触パラメータ（計画書 §4.2、C2/C3/U4 是正）
    # ------------------------------------------------------------------
    # 床 geom の摩擦 [slide, torsion, roll]（μ=1.0 は Decimus 2B 実測グリップ限界 0.8G と同オーダー。やや楽観側）
    floor_friction: str = "1.0 0.005 1e-4"
    # 機体外殻(mein_body1〜5)・壁・柱 geom の摩擦（μ_wall=0.4。U4 是正、実測根拠なしの設計値と明記）
    wall_body_friction: str = "0.4 0.005 1e-4"
    # キャスター-床 <pair> の明示摩擦（C3 是正 + 計画書改訂 r2、コミット f41c073）。
    # r2: 理想無摩擦(1e-4)から現実的な滑りキャスター μ_c=0.08 へ変更（ユーザー指示）。
    # PTFE 等の滑り接地の動摩擦の一般的目安 0.04〜0.15 の中央値・直接の実測根拠なし・
    # M1 前に感度分析推奨。摩擦力は μ_c×法線力として接触ソルバが毎ステップ計算するため、
    # 加減速の荷重移動による摩擦力の増減が自動で再現される（Test8 で検証）。
    # 結合則(要素ごとmax)より優先される<pair>で指定する。
    caster_pair_friction: str = "0.08 0.08 0.005 1e-4 1e-4"
    # 接触ソルバ softness（C2 是正。timeconst=0.002s。既定[0.02,1]は機体スケールに対し過大に柔らかい）
    solref: str = "0.002 1"

    # ------------------------------------------------------------------
    # センサ（6本、全て cutoff 0.3m。計画書 §4.2、U2/minor(3) 対応）
    # 既存4本（LF, LS, RF, RS）は v1 と同一位置・同一向き。FL/FR を新設。
    # sensordata の並びは登録順（このリストの順）どおり: LF, LS, RF, RS, FL, FR
    # ------------------------------------------------------------------
    sensor_cutoff: float = 0.3
    sensors: List[Dict[str, str]] = field(default_factory=lambda: [
        {'name': 'LF', 'pos': '0.031 0.036 0.01', 'zaxis': '1 0 0.026'},
        {'name': 'LS', 'pos': '0.043 0.016 0.01', 'zaxis': '0.25 1 0.026'},
        {'name': 'RF', 'pos': '0.031 -0.036 0.01', 'zaxis': '1 0 0.026'},
        {'name': 'RS', 'pos': '0.043 -0.016 0.01', 'zaxis': '0.25 -1 0.026'},
        {'name': 'FL', 'pos': '0.048 0.009 0.01', 'zaxis': '1 0 0.026'},
        {'name': 'FR', 'pos': '0.048 -0.009 0.01', 'zaxis': '1 0 0.026'},
    ])

    # ------------------------------------------------------------------
    # 導出パラメータ（R, Kt, Ke, N, rotor_inertia から式で導出。ハードコード禁止 — 計画書 §5 手順4 と同じ思想）
    # ------------------------------------------------------------------
    @property
    def armature(self) -> float:
        """車輪ジョイントの armature = N^2 * J_rotor [kg・m^2]（計画書 §4.2: 公称 8.0e-6）。"""
        return (self.gear_ratio ** 2) * self.rotor_inertia

    @property
    def gainprm0(self) -> float:
        """<general> アクチュエータ gainprm[0] = N*Kt/R [N・m/V]（計画書 §4.1/§4.2: 公称 7.5704e-3）。"""
        return self.gear_ratio * self.motor_Kt / self.motor_R

    @property
    def biasprm2(self) -> float:
        """<general> アクチュエータ biasprm[2] = -N^2*Kt*Ke/R [N・m・s/rad]（計画書 §4.1/§4.2: 公称 -8.187e-5）。

        biasprm は ctrl=0 でも作用する（V=0 は端子短絡ブレーキを表す。計画書 §4.1 注意）。
        """
        return -(self.gear_ratio ** 2) * self.motor_Kt * self.motor_Ke / self.motor_R
