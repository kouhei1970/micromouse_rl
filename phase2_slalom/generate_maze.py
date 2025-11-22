import sys
import os
sys.path.append(os.getcwd())
from common.maze_generator import MicromouseMazeGenerator
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

class SlalomMazeGenerator(MicromouseMazeGenerator):
    def generate_maze(self):
        """
        Generates a simple L-shape maze for slalom turn training.
        Path: Start(0,0) -> (0,1) -> (0,2)[Turn Right] -> (1,2) -> (2,2)
        """
        # Initialize with NO walls (0)
        self.v_walls.fill(0)
        self.h_walls.fill(0)
        
        # Grid size is 16x16 in base class, but we only use a small corner.
        # Let's build walls around the path (0,0)->(0,1)->(0,2)->(1,2)->(2,2)
        
        # Path Cells: (0,0), (0,1), (0,2), (1,2), (2,2)
        
        # --- Vertical Walls (West/East of cell) ---
        # v_walls[x, y] is wall to the WEST of cell(x, y)
        # v_walls[x+1, y] is wall to the EAST of cell(x, y)
        
        # Column 0 (x=0): (0,0) to (0,2)
        self.v_walls[0, 0] = 1 # West of (0,0)
        self.v_walls[1, 0] = 1 # East of (0,0)
        
        self.v_walls[0, 1] = 1 # West of (0,1)
        self.v_walls[1, 1] = 1 # East of (0,1)
        
        self.v_walls[0, 2] = 1 # West of (0,2)
        # East of (0,2) is OPEN (connection to (1,2)) -> v_walls[1, 2] = 0
        
        # Column 1 (x=1): (1,2)
        # West of (1,2) is OPEN (connection to (0,2)) -> v_walls[1, 2] = 0
        # East of (1,2) is OPEN (connection to (2,2)) -> v_walls[2, 2] = 0
        
        # Column 2 (x=2): (2,2)
        # West of (2,2) is OPEN
        self.v_walls[3, 2] = 1 # East of (2,2) (End of path)
        
        # --- Horizontal Walls (South/North of cell) ---
        # h_walls[x, y] is wall to the SOUTH of cell(x, y)
        # h_walls[x, y+1] is wall to the NORTH of cell(x, y)
        
        # Row 0 (y=0): (0,0)
        self.h_walls[0, 0] = 1 # South of (0,0)
        # North of (0,0) is OPEN (connection to (0,1)) -> h_walls[0, 1] = 0
        
        # Row 1 (y=1): (0,1)
        # South of (0,1) is OPEN
        # North of (0,1) is OPEN (connection to (0,2)) -> h_walls[0, 2] = 0
        
        # Row 2 (y=2): (0,2), (1,2), (2,2)
        # South of (0,2) is OPEN (connection to (0,1))
        self.h_walls[0, 3] = 1 # North of (0,2)
        
        self.h_walls[1, 2] = 1 # South of (1,2)
        self.h_walls[1, 3] = 1 # North of (1,2)
        
        self.h_walls[2, 2] = 1 # South of (2,2)
        self.h_walls[2, 3] = 1 # North of (2,2)

        print("Slalom maze structure generated.")

    def generate_mjcf(self, filename="micromouse_slalom.xml"):
        # Reuse the generation logic but with specific walls
        # We can copy most of the logic from generate_open_maze but we need to actually place walls this time.
        
        mujoco = ET.Element('mujoco', {'model': 'micromouse_slalom'})
        
        compiler = ET.SubElement(mujoco, 'compiler', {'angle': 'degree', 'coordinate': 'local', 'inertiafromgeom': 'true'})
        ET.SubElement(mujoco, 'option', {'timestep': '0.001', 'gravity': '0 0 -9.81'})
        
        visual = ET.SubElement(mujoco, 'visual')
        ET.SubElement(visual, 'global', {'offwidth': '800', 'offheight': '800'})
        
        asset = ET.SubElement(mujoco, 'asset')
        ET.SubElement(asset, 'texture', {'name': 'tex_floor', 'type': '2d', 'builtin': 'checker', 'rgb1': '.2 .3 .4', 'rgb2': '.1 .2 .3', 'width': '512', 'height': '512'})
        ET.SubElement(asset, 'material', {'name': 'mat_floor', 'texture': 'tex_floor', 'texrepeat': '44.44 44.44', 'reflectance': '0.3'})
        ET.SubElement(asset, 'material', {'name': 'mat_wall', 'rgba': '0.9 0.9 0.9 1'})
        ET.SubElement(asset, 'material', {'name': 'mat_wall_top', 'rgba': '1.0 0.0 0.0 1'})
        
        # Mouse Materials (Same as before)
        ET.SubElement(asset, 'material', {'name': 'mat_pcb', 'rgba': '0.1 0.1 0.1 1', 'specular': '0.5', 'shininess': '0.5'})
        ET.SubElement(asset, 'material', {'name': 'mat_tire', 'rgba': '0.1 0.1 0.1 1', 'reflectance': '0.1'})
        ET.SubElement(asset, 'material', {'name': 'mat_wheel', 'rgba': '0.9 0.9 0.9 1'})
        ET.SubElement(asset, 'material', {'name': 'mat_motor', 'rgba': '0.6 0.6 0.6 1', 'specular': '1', 'shininess': '1'})
        ET.SubElement(asset, 'material', {'name': 'mat_battery', 'rgba': '0.1 0.1 0.8 1'})
        ET.SubElement(asset, 'material', {'name': 'mat_sensor_body', 'rgba': '0.2 0.2 0.2 1'})
        ET.SubElement(asset, 'material', {'name': 'mat_led', 'rgba': '0.9 0.9 0.9 0.5'})
        ET.SubElement(asset, 'mesh', {'name': 'coner', 'vertex': '0 0 -0.0008  0.036 0 -0.0008  0.026 0.01 -0.0008  0 0.01 -0.0008  0 0 0.0008  0.036 0 0.0008  0.026 0.01 0.0008  0 0.01 0.0008'})

        worldbody = ET.SubElement(mujoco, 'worldbody')
        
        # Floor
        ET.SubElement(worldbody, 'geom', {'name': 'floor', 'pos': '0.9 0.9 0', 'size': '2 2 0.1', 'type': 'plane', 'material': 'mat_floor'})
        ET.SubElement(worldbody, 'light', {'diffuse': '.5 .5 .5', 'pos': '0.9 0.9 4', 'dir': '0 0 -1'})

        # --- Generate Walls based on v_walls and h_walls ---
        # Cell size = 0.18m. Wall thickness = 0.012m. Wall height = 0.05m.
        cell_size = 0.18
        wall_thickness = 0.012
        wall_height = 0.05
        half_cell = cell_size / 2
        
        # We only iterate over the relevant area (0..3, 0..3)
        for x in range(4):
            for y in range(4):
                # Center of the cell
                cx = x * cell_size
                cy = y * cell_size
                
                # Vertical Wall (West)
                if self.v_walls[x, y] == 1:
                    # Position: Left edge of cell
                    wx = cx - half_cell
                    wy = cy
                    ET.SubElement(worldbody, 'geom', {
                        'name': f'vwall_{x}_{y}',
                        'type': 'box',
                        'pos': f'{wx} {wy} {wall_height/2}',
                        'size': f'{wall_thickness/2} {half_cell} {wall_height/2}',
                        'material': 'mat_wall'
                    })
                
                # Vertical Wall (East) - Only for the last column, or we can just check x+1
                # Actually, v_walls[x+1, y] handles the east wall of cell x.
                # So we just iterate x up to max.
                
                # Horizontal Wall (South)
                if self.h_walls[x, y] == 1:
                    # Position: Bottom edge of cell
                    wx = cx
                    wy = cy - half_cell
                    ET.SubElement(worldbody, 'geom', {
                        'name': f'hwall_{x}_{y}',
                        'type': 'box',
                        'pos': f'{wx} {wy} {wall_height/2}',
                        'size': f'{half_cell} {wall_thickness/2} {wall_height/2}',
                        'material': 'mat_wall'
                    })

        # Add Mouse
        actuator = ET.SubElement(mujoco, 'actuator')
        sensor = ET.SubElement(mujoco, 'sensor')
        contact = ET.SubElement(mujoco, 'contact')
        self.add_mouse(worldbody, actuator, sensor, contact)

        xml_str = minidom.parseString(ET.tostring(mujoco)).toprettyxml(indent="  ")
        with open(filename, "w") as f:
            f.write(xml_str)
        print(f"Maze generated and saved to {filename}")

    def add_mouse(self, worldbody, actuator, sensor, contact):
        # Start at (0,0) which is at (0,0) in world coordinates?
        # In our loop: cx = x * 0.18. For x=0, cx=0.
        # So (0,0) cell center is at (0,0).
        start_x = 0
        start_y = 0
        start_z = 0.002
        
        mouse_body = ET.SubElement(worldbody, 'body', {'name': 'mouse', 'pos': f'{start_x} {start_y} {start_z}', 'euler': '0 0 90'})
        ET.SubElement(mouse_body, 'freejoint', {'name': 'root'})
        
        # Camera (Top-Down View to verify alignment)
        ET.SubElement(mouse_body, 'camera', {
            'name': 'track',
            'mode': 'fixed',
            'pos': '0 0 0.6',
            'xyaxes': '0 -1 0 1 0 0',
            'fovy': '45'
        })

        # --- Mouse Geoms (Copied from generate_open_maze.py with latest updates) ---
        ET.SubElement(mouse_body, 'geom', {'name': 'mein_body1', 'type': 'box', 'size': '0.05 0.03 0.0008', 'rgba': '0.05 0.4 0.15 1', 'pos': '0 0 0.002', 'mass': '4.65e-3'})
        ET.SubElement(mouse_body, 'geom', {'name': 'mein_body2', 'type': 'mesh', 'mesh': 'coner', 'friction': '0.0 0.0 0.0', 'rgba': '0.05 0.4 0.15 1', 'pos': '0.014 0.03 0.002', 'mass': '4.65e-3'})
        ET.SubElement(mouse_body, 'geom', {'name': 'mein_body3', 'type': 'mesh', 'mesh': 'coner', 'friction': '0.0 0.0 0.0', 'rgba': '0.05 0.4 0.15 1', 'pos': '0.014 -0.03 0.002', 'euler': '180 0 0', 'mass': '4.65e-3'})
        ET.SubElement(mouse_body, 'geom', {'name': 'mein_body4', 'type': 'mesh', 'mesh': 'coner', 'friction': '0.0 0.0 0.0', 'rgba': '0.05 0.4 0.15 1', 'pos': '-0.014 0.03 0.002', 'euler': '0 180 0', 'mass': '4.65e-3'})
        ET.SubElement(mouse_body, 'geom', {'name': 'mein_body5', 'type': 'mesh', 'mesh': 'coner', 'friction': '0.0 0.0 0.0', 'rgba': '0.05 0.4 0.15 1', 'pos': '-0.014 -0.03 0.002', 'euler': '180 180 0', 'mass': '4.65e-3'})

        # Casters
        ET.SubElement(mouse_body, 'geom', {'name': 'caster_front', 'type': 'sphere', 'size': '0.002', 'pos': '0.045 0 0.002', 'rgba': '0.5 0.5 0.5 1', 'friction': '0 0 0', 'mass': '0.001'})
        ET.SubElement(mouse_body, 'geom', {'name': 'caster_back', 'type': 'sphere', 'size': '0.002', 'pos': '-0.045 0 0.002', 'rgba': '0.5 0.5 0.5 1', 'friction': '0 0 0', 'mass': '0.001'})

        # Motors
        mbox1 = ET.SubElement(mouse_body, 'body', {'name': 'motorbox1', 'pos': '0 -0.018 0.0135'})
        ET.SubElement(mbox1, 'geom', {'type': 'box', 'size': '0.0117 0.0135 0.0117', 'rgba': '0.1 0.1 0.1 1', 'mass': '0.05'})
        mbox2 = ET.SubElement(mouse_body, 'body', {'name': 'motorbox2', 'pos': '0 0.018 0.0135'})
        ET.SubElement(mbox2, 'geom', {'type': 'box', 'size': '0.0117 0.0135 0.0117', 'rgba': '0.1 0.1 0.1 1', 'mass': '0.05'})

        # Wheels
        l_wheel = ET.SubElement(mouse_body, 'body', {'name': 'left_wheel', 'pos': '0 0.036 0.0135', 'zaxis': '0 1 0'})
        ET.SubElement(l_wheel, 'joint', {'name': 'left_wheel_joint', 'type': 'hinge', 'axis': '0 0 1', 'damping': '0.000054'})
        ET.SubElement(l_wheel, 'geom', {'type': 'cylinder', 'size': '0.0135 0.0035', 'rgba': '0.3 0.3 0.3 1', 'friction': '1.0 0 0', 'mass': '0.05'})
        ET.SubElement(l_wheel, 'site', {'type': 'box', 'size': '0.0012 0.008 0.004', 'rgba': '0.5 1 0.5 1'})

        r_wheel = ET.SubElement(mouse_body, 'body', {'name': 'right_wheel', 'pos': '0 -0.036 0.0135', 'zaxis': '0 1 0'})
        ET.SubElement(r_wheel, 'joint', {'name': 'right_wheel_joint', 'type': 'hinge', 'axis': '0 0 1', 'damping': '0.000054'})
        ET.SubElement(r_wheel, 'geom', {'type': 'cylinder', 'size': '0.0135 0.0035', 'rgba': '0.3 0.3 0.3 1', 'friction': '1.0 0 0', 'mass': '0.05'})
        ET.SubElement(r_wheel, 'site', {'type': 'box', 'size': '0.0012 0.008 0.004', 'rgba': '0.5 1 0.5 1'})

        # Sensors (Updated Angle: 1.5 degrees -> zaxis z=0.026)
        sensors = [
            {'name': 'sensor_left_front', 'pos': '0.031 0.036 0.01', 'zaxis': '1 0 0.026'},
            {'name': 'sensor_left_side', 'pos': '0.043 0.016 0.01', 'zaxis': '0.25 1 0.026'},
            {'name': 'sensor_right_front', 'pos': '0.031 -0.036 0.01', 'zaxis': '1 0 0.026'},
            {'name': 'sensor_right_side', 'pos': '0.043 -0.016 0.01', 'zaxis': '0.25 -1 0.026'}
        ]
        
        for s in sensors:
            s_body = ET.SubElement(mouse_body, 'body', {'name': f"visual_{s['name']}", 'pos': s['pos'], 'zaxis': s['zaxis']})
            ET.SubElement(s_body, 'geom', {'type': 'cylinder', 'size': '0.002 0.003', 'pos': '0 0 -0.003', 'rgba': '1 0 0 0.5', 'mass': '0.001'}) 
            ET.SubElement(s_body, 'geom', {'type': 'sphere', 'size': '0.002', 'pos': '0 0 0', 'rgba': '1 0 0 0.5', 'mass': '0.001'})
            ET.SubElement(s_body, 'site', {'name': s['name'], 'type': 'box', 'size': '0.001 0.001 0.001', 'pos': '0 0 0.002', 'rgba': '0 0 0 0'})
        
        ET.SubElement(mouse_body, 'site', {'name': 'accerometer', 'type': 'box', 'size': '0.001 0.001 0.001', 'pos': '0 0 0.0033'})
        ET.SubElement(mouse_body, 'site', {'name': 'gyro', 'type': 'box', 'size': '0.001 0.001 0.001', 'pos': '0 0 0.0033'})

        ET.SubElement(sensor, 'rangefinder', {'name': 'LF', 'site': 'sensor_left_front', 'cutoff': '0.15'})
        ET.SubElement(sensor, 'rangefinder', {'name': 'LS', 'site': 'sensor_left_side', 'cutoff': '0.15'})
        ET.SubElement(sensor, 'rangefinder', {'name': 'RF', 'site': 'sensor_right_front', 'cutoff': '0.15'})
        ET.SubElement(sensor, 'rangefinder', {'name': 'RS', 'site': 'sensor_right_side', 'cutoff': '0.15'})
        ET.SubElement(sensor, 'accelerometer', {'name': 'Accel', 'site': 'accerometer'})
        ET.SubElement(sensor, 'gyro', {'name': 'Gyro', 'site': 'gyro'})

        ET.SubElement(contact, 'exclude', {'body1': 'mouse', 'body2': 'left_wheel'})
        ET.SubElement(contact, 'exclude', {'body1': 'mouse', 'body2': 'right_wheel'})

        ET.SubElement(actuator, 'motor', {'name': 'act_left', 'joint': 'left_wheel_joint', 'gear': '0.0072', 'ctrllimited': 'true', 'ctrlrange': '-3 3'})
        ET.SubElement(actuator, 'motor', {'name': 'act_right', 'joint': 'right_wheel_joint', 'gear': '0.0072', 'ctrllimited': 'true', 'ctrlrange': '-3 3'})

if __name__ == "__main__":
    generator = SlalomMazeGenerator()
    generator.generate_maze()
    generator.generate_mjcf("assets/micromouse_slalom.xml")
