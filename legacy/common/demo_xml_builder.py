"""
デモ動画用のマーカー付きMuJoCo XMLを生成するユーティリティ
学習・評価には使用せず、視覚的なデモ動画作成専用

マーカーは contype="0" conaffinity="0" で設定され、
物理シミュレーションには影響を与えない視覚のみのオブジェクト
"""
import xml.etree.ElementTree as ET
import os
import tempfile


def generate_demo_xml(
    source_xml_path,
    goal_cell,
    intermediate_cell=None,
    cell_size=0.18,
    output_xml_path=None
):
    """
    既存のMuJoCo XMLにマーカーを追加してデモ用XMLを生成

    Args:
        source_xml_path: 元のXMLファイルパス
        goal_cell: ゴールセル座標 (x, y)
        intermediate_cell: 中間セル座標 (x, y) または None
        cell_size: セルサイズ（デフォルト0.18m）
        output_xml_path: 出力XMLファイルパス（Noneの場合は一時ファイル）

    Returns:
        output_xml_path: 生成されたXMLファイルのパス
    """
    tree = ET.parse(source_xml_path)
    root = tree.getroot()

    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("worldbody not found in XML")

    # ゴール位置計算（セル中央）
    goal_x = goal_cell[0] * cell_size + cell_size / 2
    goal_y = goal_cell[1] * cell_size + cell_size / 2

    # ゴールマーカー（外側リング）
    ET.SubElement(worldbody, 'geom', {
        'name': 'goal_marker_ring',
        'type': 'cylinder',
        'size': '0.04 0.002',  # 半径4cm、高さ2mm
        'pos': f'{goal_x} {goal_y} 0.001',
        'rgba': '1 0 0 0.8',  # 赤
        'contype': '0',
        'conaffinity': '0'
    })

    # ゴールマーカー（中心ドット）
    ET.SubElement(worldbody, 'geom', {
        'name': 'goal_marker_dot',
        'type': 'cylinder',
        'size': '0.015 0.003',  # 半径1.5cm、高さ3mm
        'pos': f'{goal_x} {goal_y} 0.002',
        'rgba': '1 0 0 1.0',  # 赤（不透明）
        'contype': '0',
        'conaffinity': '0'
    })

    # 中間マーカー（存在する場合）
    if intermediate_cell is not None:
        int_x = intermediate_cell[0] * cell_size + cell_size / 2
        int_y = intermediate_cell[1] * cell_size + cell_size / 2

        ET.SubElement(worldbody, 'geom', {
            'name': 'intermediate_marker_ring',
            'type': 'cylinder',
            'size': '0.03 0.002',  # 半径3cm、高さ2mm
            'pos': f'{int_x} {int_y} 0.001',
            'rgba': '1 0.5 0 0.7',  # オレンジ
            'contype': '0',
            'conaffinity': '0'
        })

    # 出力先を決定
    if output_xml_path is None:
        fd, output_xml_path = tempfile.mkstemp(suffix='.xml', prefix='demo_maze_')
        os.close(fd)

    # XMLを保存
    tree.write(output_xml_path, encoding='unicode')

    return output_xml_path


def update_marker_colors(model, goal_reached=False, intermediate_reached=False):
    """
    MuJoCoモデルのマーカー色を動的に変更（ランタイム）

    Args:
        model: mujoco.MjModel
        goal_reached: ゴール到達フラグ（Trueで緑色に変更）
        intermediate_reached: 中間到達フラグ（Trueで緑色に変更）
    """
    import mujoco

    # ゴールマーカーの色を更新
    try:
        goal_ring_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, 'goal_marker_ring'
        )
        goal_dot_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, 'goal_marker_dot'
        )

        if goal_reached:
            model.geom_rgba[goal_ring_id] = [0, 1, 0, 0.8]  # 緑
            model.geom_rgba[goal_dot_id] = [0, 1, 0, 1.0]   # 緑（不透明）
    except Exception:
        pass  # マーカーが存在しない場合はスキップ

    # 中間マーカーの色を更新
    try:
        int_ring_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, 'intermediate_marker_ring'
        )

        if intermediate_reached:
            model.geom_rgba[int_ring_id] = [0, 1, 0, 0.7]  # 緑
    except Exception:
        pass  # マーカーが存在しない場合はスキップ


def cleanup_demo_xml(xml_path):
    """
    一時的に生成したデモ用XMLファイルを削除

    Args:
        xml_path: 削除するXMLファイルのパス
    """
    if xml_path and os.path.exists(xml_path):
        try:
            os.remove(xml_path)
        except Exception:
            pass
