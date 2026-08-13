"""
mouse/robot_v2.py
================
マイクロマウス ロボット v2 のジオメトリ・アクチュエータ・センサを
xml.etree.ElementTree (ET) で構築する。

common/robot_builder.py（v1）と同じ構造・同じ寸法を踏襲するが、
v1 資産（凍結参照）は import せずコピー改変している。
rev.B（本改稿）での変更点は docs/MODEL_VERIFICATION_PLAN.md §5 手順1 の通り:
  - アクチュエータを <general>（電気項: gainprm/biasprm、単位は電圧[V]）に置換
  - 車輪ジョイントに frictionloss（乾性摩擦）を追加、damping を縮小
  - 電池ボディを新設（mein_body1 の質量から分離）
  - 機体外殻(mein_body1〜5)・キャスター<pair>の接触摩擦を明示値化（C3/U4 是正）
"""
import xml.etree.ElementTree as ET

from mouse.params import RobotParams


def add_micromouse_v2(worldbody, actuator, sensor, contact,
                       pos="0.09 0.09 0.002", euler="0 0 90", params=None):
    """マイクロマウス v2 を worldbody に追加する。

    Args:
        worldbody: MuJoCo worldbody 要素（ET.Element）
        actuator: MuJoCo actuator 要素
        sensor: MuJoCo sensor 要素
        contact: MuJoCo contact 要素
        pos: 初期位置 "x y z"
        euler: 初期姿勢 "roll pitch yaw" [deg]
        params: RobotParams（None なら既定値を使用）
    """
    if params is None:
        params = RobotParams()

    mouse_body = ET.SubElement(worldbody, 'body', {'name': 'mouse', 'pos': pos, 'euler': euler})
    ET.SubElement(mouse_body, 'freejoint', {'name': 'root'})

    # トラッキング用カメラ（真上視点で位置合わせを確認するため。v1 同様）
    ET.SubElement(mouse_body, 'camera', {
        'name': 'track',
        'mode': 'trackcom',
        'pos': '0 0 0.8',
        'xyaxes': '0 -1 0 1 0 0',
        'fovy': '45'
    })

    # --- 車体本体（v1 と同一ジオメトリ、質量・摩擦のみ v2/rev.B の値に変更） ---
    # mein_body1: シャーシ + PCB の集約質量（電池は rev.B で下記 battery ボディに分離）
    # friction は機体外殻-壁の μ_wall=0.4 を明示（U4 是正。暗黙の MuJoCo 既定値 [1, 0.005, 1e-4]
    # 依存を排除しないと、壁の摩擦と結合則(要素ごとmax)で意図せず μ=1.0 になる）
    # シャーシ外形は `RobotParams` を正とする（裁定 R16 ②・R11 バッチ項目 3。
    # 仕様の正の定義元は `docs/ROBOT_SPEC.md` §2.1）。**MJCF の box の size は半長**なので
    # 外形（全長）を 1/2 にして書く。**値は従来のリテラル '0.05 0.03 0.0008' と一致する**
    # （改修の合格条件は生成 XML のバイト一致）。
    _chassis_half = ' '.join(f'{v / 2:g}' for v in (params.chassis_length,
                                                    params.chassis_width,
                                                    params.chassis_thickness))
    ET.SubElement(mouse_body, 'geom', {
        'name': 'mein_body1',
        'type': 'box',
        'size': _chassis_half,
        'rgba': '0.05 0.4 0.15 1',
        'pos': '0 0 0.002',
        'friction': params.wall_body_friction,
        'mass': str(params.mass_body_center)
    })

    # 電池（新規。1S LiPo 相当・低位置搭載で転倒余裕確保。計画書 §4.2/§5 手順1-3、minor所見(1)是正）
    ET.SubElement(mouse_body, 'geom', {
        'name': 'battery',
        'type': 'box',
        'size': params.battery_size,
        'pos': params.battery_pos,
        'material': 'mat_battery',
        'mass': str(params.mass_battery),
    })

    # mein_body2〜5: コーナー部（メッシュ 'coner' を 4 隅に配置）
    # friction は mein_body1 と同じく μ_wall=0.4 を明示（U4 是正。旧 friction="0 0 0" は
    # 結合則により床/壁の既定摩擦に負けて無効化されていた — note_003 と同型の罠）
    corner_defs = [
        ('mein_body2', '0.014 0.03 0.002', None),
        ('mein_body3', '0.014 -0.03 0.002', '180 0 0'),
        ('mein_body4', '-0.014 0.03 0.002', '0 180 0'),
        ('mein_body5', '-0.014 -0.03 0.002', '180 180 0'),
    ]
    for name, cpos, ceuler in corner_defs:
        attrib = {
            'name': name,
            'type': 'mesh',
            'mesh': 'coner',
            'friction': params.wall_body_friction,
            'rgba': '0.05 0.4 0.15 1',
            'pos': cpos,
            'mass': str(params.mass_body_corner),
        }
        if ceuler is not None:
            attrib['euler'] = ceuler
        ET.SubElement(mouse_body, 'geom', attrib)

    # キャスター（前後、無摩擦球。v1 から質量変更なし）
    # Casters (front/rear frictionless spheres; mass unchanged from v1).
    #
    # 注意（C3 是正の機序）: MuJoCo は接触摩擦を両 geom の要素ごと最大値で合成するため
    # （priority 指定がない場合の既定動作）、friction="0 0 0" だけでは床のデフォルト摩擦
    # μ=1 が勝ってしまい無摩擦にならない（v1 から潜在していたバグ。キャスター抗力で
    # 駆動力が相殺され、車輪スリップ・接地の間欠化を引き起こしていた。research_notes/
    # note_003_caster_friction.md 参照）。
    # rev.B では研究ノート note_003 が暫定採用した priority="1"+condim="1" 方式
    # （機能的に等価であることは実測確認済み）を撤去し、下記 <contact><pair> による
    # 明示指定に一本化する（<pair> は摩擦結合則より優先され、かつ静的検査（計画書 §7 S2）
    # が <pair> 前提のため）。geom 自体は無摩擦の設計意図を表す friction="0 0 0" のまま。
    # r4（計画書コミット 360ab5b）: 3点接地化。前キャスターは接地、後キャスターは
    # 静止時クリアランス 0.4mm のバンプストップ（後傾時のみ接地する安全ストッパ）。
    # 4点接地では平面支持に必要な3点を超える冗長支持となり、接触ソルバの荷重配分が
    # 走行中に振動する（静力学的不静定。検証スイート初回実行 8/14 の共通根因）。
    ET.SubElement(mouse_body, 'geom', {
        'name': 'caster_front', 'type': 'sphere', 'size': '0.002',
        'pos': '0.045 0 0.002', 'rgba': '0.5 0.5 0.5 1', 'friction': '0 0 0',
        'mass': str(params.mass_caster)
    })
    ET.SubElement(mouse_body, 'geom', {
        'name': 'caster_back', 'type': 'sphere', 'size': '0.002',
        'pos': f'-0.045 0 {0.002 + params.caster_rear_clearance}',
        'rgba': '0.5 0.5 0.5 1', 'friction': '0 0 0',
        'mass': str(params.mass_caster)
    })

    # モータボックス（FA-130 級モータ相当）
    mbox1 = ET.SubElement(mouse_body, 'body', {'name': 'motorbox1', 'pos': '0 -0.018 0.0135'})
    ET.SubElement(mbox1, 'geom', {
        'type': 'box', 'size': '0.0117 0.0135 0.0117', 'rgba': '0.1 0.1 0.1 1',
        'mass': str(params.mass_motorbox)
    })
    mbox2 = ET.SubElement(mouse_body, 'body', {'name': 'motorbox2', 'pos': '0 0.018 0.0135'})
    ET.SubElement(mbox2, 'geom', {
        'type': 'box', 'size': '0.0117 0.0135 0.0117', 'rgba': '0.1 0.1 0.1 1',
        'mass': str(params.mass_motorbox)
    })

    # 左車輪（joint に damping/frictionloss/armature を設定。計画書 §4.2: b, tau_c は
    # 暫定値・Test2 で較正、armature = N^2*J_rotor は §4.1 のモータモデルに整合）
    l_wheel = ET.SubElement(mouse_body, 'body', {
        'name': 'left_wheel', 'pos': '0 0.036 0.0135', 'zaxis': '0 1 0'
    })
    ET.SubElement(l_wheel, 'joint', {
        'name': 'left_wheel_joint', 'type': 'hinge', 'axis': '0 0 1',
        'damping': str(params.wheel_damping),
        'frictionloss': str(params.wheel_frictionloss),
        'armature': str(params.armature),
    })
    ET.SubElement(l_wheel, 'geom', {
        'type': 'cylinder', 'size': '0.0135 0.0035', 'rgba': '0.3 0.3 0.3 1',
        'friction': '1.0 0 0', 'mass': str(params.mass_wheel)
    })
    ET.SubElement(l_wheel, 'site', {'type': 'box', 'size': '0.0012 0.008 0.004', 'rgba': '0.5 1 0.5 1'})

    # 右車輪（左と同一の joint 設定）
    r_wheel = ET.SubElement(mouse_body, 'body', {
        'name': 'right_wheel', 'pos': '0 -0.036 0.0135', 'zaxis': '0 1 0'
    })
    ET.SubElement(r_wheel, 'joint', {
        'name': 'right_wheel_joint', 'type': 'hinge', 'axis': '0 0 1',
        'damping': str(params.wheel_damping),
        'frictionloss': str(params.wheel_frictionloss),
        'armature': str(params.armature),
    })
    ET.SubElement(r_wheel, 'geom', {
        'type': 'cylinder', 'size': '0.0135 0.0035', 'rgba': '0.3 0.3 0.3 1',
        'friction': '1.0 0 0', 'mass': str(params.mass_wheel)
    })
    ET.SubElement(r_wheel, 'site', {'type': 'box', 'size': '0.0012 0.008 0.004', 'rgba': '0.5 1 0.5 1'})

    # --- センサポッド（6 本: LF, LS, RF, RS, FL, FR。FL/FR は v2 新設） ---
    # ポッドの視覚 geom は v1 と同じ作り（円柱 + 球、site は先端 +0.002）
    for s in params.sensors:
        s_body = ET.SubElement(mouse_body, 'body', {
            'name': f"visual_{s['name']}", 'pos': s['pos'], 'zaxis': s['zaxis']
        })
        # 円柱部分（基部）
        ET.SubElement(s_body, 'geom', {
            'type': 'cylinder', 'size': '0.002 0.003', 'pos': '0 0 -0.003',
            'rgba': '1 0 0 0.5', 'mass': str(params.mass_sensor_geom)
        })
        # 半球（先端）部分
        ET.SubElement(s_body, 'geom', {
            'type': 'sphere', 'size': '0.002', 'pos': '0 0 0',
            'rgba': '1 0 0 0.5', 'mass': str(params.mass_sensor_geom)
        })
        # 実際のセンサ site（先端 + 0.002）
        ET.SubElement(s_body, 'site', {
            'name': f"site_{s['name']}", 'type': 'box', 'size': '0.001 0.001 0.001',
            'pos': '0 0 0.002', 'rgba': '0 0 0 0'
        })

    # IMU（加速度計 + ジャイロ。v1 と同じ site）
    ET.SubElement(mouse_body, 'site', {'name': 'accerometer', 'type': 'box', 'size': '0.001 0.001 0.001', 'pos': '0 0 0.0033'})
    ET.SubElement(mouse_body, 'site', {'name': 'gyro', 'type': 'box', 'size': '0.001 0.001 0.001', 'pos': '0 0 0.0033'})

    # --- センサ登録 ---
    # sensordata の並び: LF, LS, RF, RS, FL, FR, Accel(3), Gyro(3) の順（spec §1.4）
    for s in params.sensors:
        ET.SubElement(sensor, 'rangefinder', {
            'name': s['name'], 'site': f"site_{s['name']}", 'cutoff': str(params.sensor_cutoff)
        })
    ET.SubElement(sensor, 'accelerometer', {'name': 'Accel', 'site': 'accerometer'})
    ET.SubElement(sensor, 'gyro', {'name': 'Gyro', 'site': 'gyro'})

    # 接触除外（シャーシ - 車輪 間の誤衝突防止。v1 同様）
    ET.SubElement(contact, 'exclude', {'body1': 'mouse', 'body2': 'left_wheel'})
    ET.SubElement(contact, 'exclude', {'body1': 'mouse', 'body2': 'right_wheel'})

    # キャスター-床 <pair>（C3 是正の本体。摩擦結合則(要素ごとmax)より優先される）
    # geom2="floor" は呼び出し側（mouse/mjcf.py）の床 geom 名と整合させている。
    # 迷路シーンでもこの add_micromouse_v2() 経由で必ず出力されるため、C2/C3 は
    # 迷路 XML でも成立する（計画書 §5 手順2）。
    ET.SubElement(contact, 'pair', {
        'geom1': 'caster_front', 'geom2': 'floor', 'friction': params.caster_pair_friction
    })
    ET.SubElement(contact, 'pair', {
        'geom1': 'caster_back', 'geom2': 'floor', 'friction': params.caster_pair_friction
    })

    # --- アクチュエータ ---
    # <general> による電気項の直接表現（計画書 §4.1）。ctrl の単位は電圧 [V]（±voltage_limit）。
    #   force = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*qpos + biasprm[2]*qvel
    #         = (N*Kt/R)*V         - (N^2*Kt*Ke/R)*omega_wheel
    # すなわち電気項 tau_w = (N*Kt/R)(V - Ke*N*omega) を gain/bias の和として表現する。
    # 機械損失項（粘性 b・乾性摩擦 tau_c）は上記 joint の damping/frictionloss が担う。
    # 全物理サブステップで MuJoCo が自動再計算するため、Python 側の stale トルク問題を回避する
    # （mouse/sim.py は data.ctrl に電圧を ZOH で書き込むのみ）。
    # 注意（モデル化の選択）: biasprm は ctrl=0 でも作用するため、V=0 は「端子短絡ブレーキ」を
    # 表す（H ブリッジのブレーキモードに相当。開放惰行ではない。計画書 §4.1 注意）。
    gainprm0_str = f"{params.gainprm0:.8g} 0 0"
    biasprm_str = f"0 0 {params.biasprm2:.8g}"
    ctrl_limit_str = f"-{params.voltage_limit:g} {params.voltage_limit:g}"
    ET.SubElement(actuator, 'general', {
        'name': 'act_left', 'joint': 'left_wheel_joint',
        'gainprm': gainprm0_str, 'biastype': 'affine', 'biasprm': biasprm_str,
        'ctrllimited': 'true', 'ctrlrange': ctrl_limit_str,
    })
    ET.SubElement(actuator, 'general', {
        'name': 'act_right', 'joint': 'right_wheel_joint',
        'gainprm': gainprm0_str, 'biastype': 'affine', 'biasprm': biasprm_str,
        'ctrllimited': 'true', 'ctrlrange': ctrl_limit_str,
    })
