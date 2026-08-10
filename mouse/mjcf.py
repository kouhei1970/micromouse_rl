"""
mouse/mjcf.py
================
MuJoCo XML (MJCF) ビルダー。ロボット v2 と迷路（またはただの平原）を
組み合わせて XML ファイルを生成する。

common/mjcf_builder.py, common/maze_assets.py, common/maze_generator.py
（v1、凍結参照）と同じ構造・同じ寸法を踏襲するが、import はせずコピー改変している。
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom

from mouse.params import RobotParams
from mouse.robot_v2 import add_micromouse_v2

# 迷路寸法（既存 common/mjcf_builder.py, common/maze_generator.py と同一値）
WALL_THICKNESS = 0.012  # 壁厚 [m]
WALL_HEIGHT = 0.05      # 壁高 [m]
POST_SIZE = 0.012       # 柱の一辺 [m]


def _add_assets(mujoco_node, floor_texrepeat="50 50"):
    """材質・テクスチャ・メッシュを asset ノードに追加する。
    common/maze_assets.py の add_assets と同内容（v1 凍結のためコピー）。"""
    asset = ET.SubElement(mujoco_node, 'asset')

    ET.SubElement(asset, 'texture', {
        'name': 'tex_floor', 'type': '2d', 'builtin': 'checker',
        'rgb1': '.2 .3 .4', 'rgb2': '.1 .2 .3', 'width': '512', 'height': '512'
    })
    ET.SubElement(asset, 'material', {
        'name': 'mat_floor', 'texture': 'tex_floor', 'reflectance': '0.0',
        'texrepeat': floor_texrepeat
    })
    ET.SubElement(asset, 'material', {'name': 'mat_wall', 'rgba': '1 1 1 1'})
    ET.SubElement(asset, 'material', {'name': 'mat_wall_top', 'rgba': '0.8 0 0 1'})
    ET.SubElement(asset, 'material', {'name': 'mat_post', 'rgba': '0.9 0.9 0.9 1'})

    ET.SubElement(asset, 'material', {'name': 'mat_pcb', 'rgba': '0.1 0.1 0.1 1', 'specular': '0.5', 'shininess': '0.5'})
    ET.SubElement(asset, 'material', {'name': 'mat_tire', 'rgba': '0.1 0.1 0.1 1', 'reflectance': '0.1'})
    ET.SubElement(asset, 'material', {'name': 'mat_wheel', 'rgba': '0.9 0.9 0.9 1'})
    ET.SubElement(asset, 'material', {'name': 'mat_motor', 'rgba': '0.6 0.6 0.6 1', 'specular': '1', 'shininess': '1'})
    ET.SubElement(asset, 'material', {'name': 'mat_battery', 'rgba': '0.1 0.1 0.8 1'})
    ET.SubElement(asset, 'material', {'name': 'mat_sensor_body', 'rgba': '0.2 0.2 0.2 1'})
    ET.SubElement(asset, 'material', {'name': 'mat_led', 'rgba': '0.9 0.9 0.9 0.5'})

    ET.SubElement(asset, 'mesh', {
        'name': 'coner',
        'vertex': '0 0 -0.0008  0.036 0 -0.0008  0.026 0.01 -0.0008  0 0.01 -0.0008  '
                  '0 0 0.0008  0.036 0 0.0008  0.026 0.01 0.0008  0 0.01 0.0008'
    })
    return asset


def _new_mujoco_root(model_name, params):
    """compiler / option / default / visual / asset を備えた <mujoco> ルート要素を作る。

    timestep・solref は params（docs/MODEL_VERIFICATION_PLAN.md §4.2）から取得する。
    平原・迷路の両シーンで必ず同じ値を出力すること（C2/C3 が迷路 XML でも成立する必要がある。
    計画書 §5 手順2）。
    """
    mujoco = ET.Element('mujoco', {'model': model_name})
    ET.SubElement(mujoco, 'compiler', {'angle': 'degree', 'coordinate': 'local', 'inertiafromgeom': 'true'})
    # 物理 timestep（計画書 §4.2: C2 是正。solref timeconst(0.002s) に対し比4を確保）
    ET.SubElement(mujoco, 'option', {'timestep': str(params.physics_dt), 'gravity': '0 0 -9.81'})
    # 全 geom の既定 solref（計画書 §4.2: C2 是正。MuJoCo 既定 [0.02, 1] は機体スケールに対し
    # 過大に柔らかく、壁貫入・トンネリングの原因だった）
    default = ET.SubElement(mujoco, 'default')
    ET.SubElement(default, 'geom', {'solref': params.solref})
    visual = ET.SubElement(mujoco, 'visual')
    ET.SubElement(visual, 'global', {'offwidth': '800', 'offheight': '800'})
    _add_assets(mujoco)
    return mujoco


def _write_xml(mujoco, out_path):
    xml_str = minidom.parseString(ET.tostring(mujoco)).toprettyxml(indent='  ')
    with open(out_path, 'w') as f:
        f.write(xml_str)
    return out_path


def build_maze_robot_xml(v_walls, h_walls, out_path, model_name="micromouse_v2_maze",
                          mouse_pos="0.09 0.09 0.002", mouse_euler="0 0 90",
                          center_goal=True, params=None):
    """v_walls/h_walls で表された迷路 + ロボット v2 の MJCF を生成する。

    Args:
        v_walls: 縦壁配列。shape (W+1, H)。v_walls[x, y] == 1 で
                 x 番目のグリッド線上、y 番目のセルの縦壁が存在。
        h_walls: 横壁配列。shape (W, H+1)。h_walls[x, y] == 1 で
                 x 番目のセル、y 番目のグリッド線上の横壁が存在。
        out_path: 出力 XML パス
        model_name: <mujoco model="..."> の値
        mouse_pos, mouse_euler: ロボットの初期位置・姿勢
        center_goal: True かつ 16x16 迷路の場合、中央柱 (8,8) を生成しない
        params: RobotParams（None なら既定値）
    """
    params = params or RobotParams()
    cell_size = params.cell_size  # 0.18

    width = v_walls.shape[0] - 1
    height = v_walls.shape[1]
    expected_h_shape = (width, height + 1)
    if h_walls.shape != expected_h_shape:
        raise ValueError(
            f"h_walls の shape が v_walls と整合しません: h_walls.shape={h_walls.shape}, "
            f"期待値={expected_h_shape} (v_walls.shape={v_walls.shape} から算出)"
        )

    mujoco = _new_mujoco_root(model_name, params)
    worldbody = ET.SubElement(mujoco, 'worldbody')

    # 床: plane、中心 (W×0.18/2, H×0.18/2)。friction は μ=1.0 を明示（計画書 §4.2、
    # 暗黙の MuJoCo 既定値依存を排除）。
    # half-size は迷路より一回り大きく（16x16 なら 2.0）。
    # 16x16 迷路の中心 (1.44, 1.44) ・ half-size 2.0 という既存の基準点
    # （common/maze_generator.py の generate_mjcf 実装）を再現しつつ一般の W,H に
    # 比例スケールする式として cells * 0.125 を採用した（他サイズでの厳密な仕様指定なし）。
    center_x = width * cell_size / 2.0
    center_y = height * cell_size / 2.0
    half_x = width * 0.125
    half_y = height * 0.125
    ET.SubElement(worldbody, 'geom', {
        'name': 'floor', 'type': 'plane', 'material': 'mat_floor',
        'pos': f'{center_x} {center_y} 0', 'size': f'{half_x} {half_y} 0.1',
        'friction': params.floor_friction,
    })
    ET.SubElement(worldbody, 'light', {
        'diffuse': '.5 .5 .5', 'pos': f'{center_x} {center_y} 3', 'dir': '0 0 -1'
    })

    # 柱（post）。friction は壁と同じ μ_wall=0.4 を明示（U4 是正）。
    for x in range(width + 1):
        for y in range(height + 1):
            if center_goal and width == 16 and height == 16 and x == 8 and y == 8:
                continue  # 16x16 迷路の中央柱は生成しない（common/maze_generator.py と同じ流儀）
            px = x * cell_size
            py = y * cell_size
            ET.SubElement(worldbody, 'geom', {
                'name': f'post_{x}_{y}', 'type': 'box',
                'size': f'{POST_SIZE / 2} {POST_SIZE / 2} {WALL_HEIGHT / 2}',
                'pos': f'{px} {py} {WALL_HEIGHT / 2}', 'material': 'mat_post',
                'friction': params.wall_body_friction,
            })

    # 縦壁（vertical walls）。friction は μ_wall=0.4 を明示（壁上端の赤 geom も同様。U4 是正）。
    for x in range(width + 1):
        for y in range(height):
            if v_walls[x, y] == 1:
                px = x * cell_size
                py = y * cell_size + cell_size / 2
                ET.SubElement(worldbody, 'geom', {
                    'name': f'v_wall_{x}_{y}', 'type': 'box',
                    'size': f'{WALL_THICKNESS / 2} {cell_size / 2 - POST_SIZE / 2} {WALL_HEIGHT / 2}',
                    'pos': f'{px} {py} {WALL_HEIGHT / 2}', 'material': 'mat_wall',
                    'friction': params.wall_body_friction,
                })
                # 赤い壁上端
                ET.SubElement(worldbody, 'geom', {
                    'name': f'v_wall_top_{x}_{y}', 'type': 'box',
                    'size': f'{WALL_THICKNESS / 2} {cell_size / 2 - POST_SIZE / 2} 0.001',
                    'pos': f'{px} {py} {WALL_HEIGHT}', 'material': 'mat_wall_top',
                    'friction': params.wall_body_friction,
                })

    # 横壁（horizontal walls）
    for x in range(width):
        for y in range(height + 1):
            if h_walls[x, y] == 1:
                px = x * cell_size + cell_size / 2
                py = y * cell_size
                ET.SubElement(worldbody, 'geom', {
                    'name': f'h_wall_{x}_{y}', 'type': 'box',
                    'size': f'{cell_size / 2 - POST_SIZE / 2} {WALL_THICKNESS / 2} {WALL_HEIGHT / 2}',
                    'pos': f'{px} {py} {WALL_HEIGHT / 2}', 'material': 'mat_wall',
                    'friction': params.wall_body_friction,
                })
                ET.SubElement(worldbody, 'geom', {
                    'name': f'h_wall_top_{x}_{y}', 'type': 'box',
                    'size': f'{cell_size / 2 - POST_SIZE / 2} {WALL_THICKNESS / 2} 0.001',
                    'pos': f'{px} {py} {WALL_HEIGHT}', 'material': 'mat_wall_top',
                    'friction': params.wall_body_friction,
                })

    actuator = ET.SubElement(mujoco, 'actuator')
    sensor = ET.SubElement(mujoco, 'sensor')
    contact = ET.SubElement(mujoco, 'contact')
    add_micromouse_v2(worldbody, actuator, sensor, contact,
                       pos=mouse_pos, euler=mouse_euler, params=params)

    return _write_xml(mujoco, out_path)


def build_open_floor_xml(out_path, params=None):
    """迷路なし平原 + ロボット v2 の MJCF を生成する（assets/mouse_v2.xml 用）。"""
    params = params or RobotParams()
    mujoco = _new_mujoco_root('micromouse_v2_open', params)
    worldbody = ET.SubElement(mujoco, 'worldbody')

    # 床（assets/micromouse_open.xml と同じ広さの平原）。friction は μ=1.0 を明示（計画書 §4.2）。
    ET.SubElement(worldbody, 'geom', {
        'name': 'floor', 'type': 'plane', 'material': 'mat_floor',
        'pos': '0 0 0', 'size': '9 9 0.1',
        'friction': params.floor_friction,
    })
    ET.SubElement(worldbody, 'light', {'diffuse': '.5 .5 .5', 'pos': '0 0 3', 'dir': '0 0 -1'})

    actuator = ET.SubElement(mujoco, 'actuator')
    sensor = ET.SubElement(mujoco, 'sensor')
    contact = ET.SubElement(mujoco, 'contact')
    add_micromouse_v2(worldbody, actuator, sensor, contact,
                       pos='0 0 0.002', euler='0 0 90', params=params)

    return _write_xml(mujoco, out_path)
