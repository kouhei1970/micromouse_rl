import xml.etree.ElementTree as ET

def add_assets(mujoco_node, floor_texrepeat="50 50"):
    """
    Adds common assets (materials, textures) to the MuJoCo node.
    """
    asset = ET.SubElement(mujoco_node, 'asset')
    
    # Floor Texture
    # User requirement: 180mm x 180mm checkerboard, vertices aligned with posts.
    # Posts are at 0.18m intervals.
    # A checker texture (builtin) usually has 2x2 squares.
    # If we want one square to be 0.18m, the full texture cycle is 0.36m.
    ET.SubElement(asset, 'texture', {
        'name': 'tex_floor', 
        'type': '2d', 
        'builtin': 'checker', 
        'rgb1': '.2 .3 .4', 
        'rgb2': '.1 .2 .3', 
        'width': '512', 
        'height': '512'
    })
    
    # Materials
    # Floor: Reflectance 0.0 (User requirement)
    ET.SubElement(asset, 'material', {
        'name': 'mat_floor', 
        'texture': 'tex_floor', 
        'reflectance': '0.0',
        'texrepeat': floor_texrepeat
    })
    
    # Walls & Posts
    ET.SubElement(asset, 'material', {'name': 'mat_wall', 'rgba': '1 1 1 1'})
    ET.SubElement(asset, 'material', {'name': 'mat_wall_top', 'rgba': '0.8 0 0 1'})
    ET.SubElement(asset, 'material', {'name': 'mat_post', 'rgba': '0.9 0.9 0.9 1'})
    
    # Mouse Materials
    ET.SubElement(asset, 'material', {'name': 'mat_pcb', 'rgba': '0.1 0.1 0.1 1', 'specular': '0.5', 'shininess': '0.5'})
    ET.SubElement(asset, 'material', {'name': 'mat_tire', 'rgba': '0.1 0.1 0.1 1', 'reflectance': '0.1'})
    ET.SubElement(asset, 'material', {'name': 'mat_wheel', 'rgba': '0.9 0.9 0.9 1'})
    ET.SubElement(asset, 'material', {'name': 'mat_motor', 'rgba': '0.6 0.6 0.6 1', 'specular': '1', 'shininess': '1'})
    ET.SubElement(asset, 'material', {'name': 'mat_battery', 'rgba': '0.1 0.1 0.8 1'})
    ET.SubElement(asset, 'material', {'name': 'mat_sensor_body', 'rgba': '0.2 0.2 0.2 1'})
    ET.SubElement(asset, 'material', {'name': 'mat_led', 'rgba': '0.9 0.9 0.9 0.5'})
    
    # Mouse Meshes
    ET.SubElement(asset, 'mesh', {'name': 'coner', 'vertex': '0 0 -0.0008  0.036 0 -0.0008  0.026 0.01 -0.0008  0 0.01 -0.0008  0 0 0.0008  0.036 0 0.0008  0.026 0.01 0.0008  0 0.01 0.0008'})

    return asset
