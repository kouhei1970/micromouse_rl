import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import random

class MicromouseMazeGenerator:
    def __init__(self):
        self.width = 16
        self.height = 16
        self.cell_size = 0.180  # 180mm
        self.wall_thickness = 0.012  # 12mm
        self.wall_height = 0.050  # 50mm
        self.post_width = 0.012  # 12mm
        
        # Maze data: 0 for empty, 1 for wall
        # Vertical walls: (17, 16) - walls to the west of cell (x, y) and east of (15, y)
        # x index: 0 to 16 (0 is left of cell 0, 16 is right of cell 15)
        # y index: 0 to 15
        self.v_walls = np.zeros((17, 16), dtype=int)
        
        # Horizontal walls: (16, 17) - walls to the south of cell (x, y) and north of (x, 15)
        # x index: 0 to 15
        # y index: 0 to 17 (0 is bottom of cell 0, 16 is top of cell 15)
        self.h_walls = np.zeros((16, 17), dtype=int)

    def generate_maze(self):
        """
        Generates a random maze using Depth First Search (Recursive Backtracker).
        This ensures a perfect maze (no loops, fully connected).
        """
        # Initialize with all walls present
        self.v_walls.fill(1)
        self.h_walls.fill(1)
        
        visited = np.zeros((16, 16), dtype=bool)
        
        def get_neighbors(x, y):
            neighbors = []
            if x > 0: neighbors.append(('W', x-1, y))
            if x < 15: neighbors.append(('E', x+1, y))
            if y > 0: neighbors.append(('S', x, y-1))
            if y < 15: neighbors.append(('N', x, y+1))
            return neighbors

        def remove_wall(x, y, direction):
            if direction == 'W':
                self.v_walls[x, y] = 0
            elif direction == 'E':
                self.v_walls[x+1, y] = 0
            elif direction == 'S':
                self.h_walls[x, y] = 0
            elif direction == 'N':
                self.h_walls[x, y+1] = 0

        # Start DFS from (0,0)
        stack = [(0, 0)]
        visited[0, 0] = True
        
        while stack:
            cx, cy = stack[-1]
            neighbors = get_neighbors(cx, cy)
            unvisited_neighbors = [n for n in neighbors if not visited[n[1], n[2]]]
            
            if unvisited_neighbors:
                direction, nx, ny = random.choice(unvisited_neighbors)
                remove_wall(cx, cy, direction)
                visited[nx, ny] = True
                stack.append((nx, ny))
            else:
                stack.pop()

        # Open start cell (usually 0,0 has an opening, but in micromouse it's enclosed usually, 
        # except the starting orientation. Let's keep it closed for now or open a specific wall if needed)
        # Standard micromouse starts at a corner enclosed by 3 walls.
        
        # Clear the center area for the goal (2x2 in the middle)
        # Center cells are (7,7), (7,8), (8,7), (8,8)
        # Remove internal walls in the goal area
        self.v_walls[8, 7] = 0 # Between (7,7) and (8,7)
        self.v_walls[8, 8] = 0 # Between (7,8) and (8,8)
        self.h_walls[7, 8] = 0 # Between (7,7) and (7,8)
        self.h_walls[8, 8] = 0 # Between (8,7) and (8,8)
        
        # Ensure goal entrance (usually there is one entrance, but let's just ensure connectivity)
        # The DFS guarantees connectivity, so we are good.

        # Rule: "柱には必ず一つ以上の壁が隣接する" (Every post must have at least one wall attached)
        # Check all posts and add a random wall if isolated.
        # Posts are at intersections of grid lines.
        # Post at (px, py) corresponds to:
        # v_walls[px, py] (North of post), v_walls[px, py-1] (South of post) - wait, indexing needs care.
        
        # Let's define post coordinates:
        # px in 0..16, py in 0..16
        # Walls connected to post (px, py):
        # North: v_walls[px, py] (if py < 16)
        # South: v_walls[px, py-1] (if py > 0)
        # East: h_walls[px, py] (if px < 16)
        # West: h_walls[px-1, py] (if px > 0)
        
        for px in range(17):
            for py in range(17):
                # Skip the center post (no post there)
                if px == 8 and py == 8:
                    continue
                
                connected = False
                # Check North (Vertical wall above post)
                if py < 16 and self.v_walls[px, py] == 1: connected = True
                # Check South (Vertical wall below post)
                if py > 0 and self.v_walls[px, py-1] == 1: connected = True
                # Check East (Horizontal wall right of post)
                if px < 16 and self.h_walls[px, py] == 1: connected = True
                # Check West (Horizontal wall left of post)
                if px > 0 and self.h_walls[px-1, py] == 1: connected = True
                
                if not connected:
                    # Add a random wall attached to this post
                    options = []
                    if py < 16: options.append(('N'))
                    if py > 0: options.append(('S'))
                    if px < 16: options.append(('E'))
                    if px > 0: options.append(('W'))
                    
                    if options:
                        choice = random.choice(options)
                        if choice == 'N': self.v_walls[px, py] = 1
                        elif choice == 'S': self.v_walls[px, py-1] = 1
                        elif choice == 'E': self.h_walls[px, py] = 1
                        elif choice == 'W': self.h_walls[px-1, py] = 1

        # Enforce specific wall configuration for start cell (0,0)
        # North side: No wall
        self.h_walls[0, 1] = 0
        # East side: Has wall
        self.v_walls[1, 0] = 1
        # Ensure boundaries are closed (South and West of 0,0)
        self.h_walls[0, 0] = 1
        self.v_walls[0, 0] = 1

        # Re-apply removal of center walls to ensure no random walls were added to the center post
        # Center cells are (7,7), (7,8), (8,7), (8,8)
        # Remove internal walls in the goal area
        self.v_walls[8, 7] = 0 # Between (7,7) and (8,7)
        self.v_walls[8, 8] = 0 # Between (7,8) and (8,8)
        self.h_walls[7, 8] = 0 # Between (7,7) and (7,8)
        self.h_walls[8, 8] = 0 # Between (8,7) and (8,8)
        
        print(f"DEBUG: Center walls removed. v_walls[8,7]={self.v_walls[8,7]}, v_walls[8,8]={self.v_walls[8,8]}")
        print(f"DEBUG: h_walls[7,8]={self.h_walls[7,8]}, h_walls[8,8]={self.h_walls[8,8]}")

    def generate_mjcf(self, filename="maze.xml"):
        mujoco = ET.Element('mujoco', {'model': 'micromouse_maze'})
        
        # Compiler settings
        compiler = ET.SubElement(mujoco, 'compiler', {'angle': 'degree', 'coordinate': 'local', 'inertiafromgeom': 'true'})
        
        # Option settings
        ET.SubElement(mujoco, 'option', {'timestep': '0.001', 'gravity': '0 0 -9.81'})
        
        # Assets (Materials and Textures)
        asset = ET.SubElement(mujoco, 'asset')
        ET.SubElement(asset, 'texture', {'name': 'tex_floor', 'type': '2d', 'builtin': 'checker', 'rgb1': '.2 .3 .4', 'rgb2': '.1 .2 .3', 'width': '512', 'height': '512'})
        ET.SubElement(asset, 'material', {'name': 'mat_floor', 'texture': 'tex_floor', 'texrepeat': '5 5', 'reflectance': '0.3'})
        ET.SubElement(asset, 'material', {'name': 'mat_wall_white', 'rgba': '1 1 1 1'})
        ET.SubElement(asset, 'material', {'name': 'mat_wall_red', 'rgba': '1 0 0 1'})
        ET.SubElement(asset, 'material', {'name': 'mat_post', 'rgba': '1 1 1 1'})
        
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

        # Worldbody
        worldbody = ET.SubElement(mujoco, 'worldbody')
        
        # Floor
        # Maze size is 16 * 0.18 = 2.88m. Let's make the floor a bit larger.
        ET.SubElement(worldbody, 'geom', {'name': 'floor', 'pos': '1.44 1.44 0', 'size': '2 2 0.1', 'type': 'plane', 'material': 'mat_floor'})
        
        # Light
        ET.SubElement(worldbody, 'light', {'diffuse': '.5 .5 .5', 'pos': '1.44 1.44 4', 'dir': '0 0 -1'})

        # Posts
        post_half_width = self.post_width / 2
        post_half_height = self.wall_height / 2
        
        for px in range(17):
            for py in range(17):
                # Skip center post (8, 8)
                if px == 8 and py == 8:
                    continue
                
                pos_x = px * self.cell_size
                pos_y = py * self.cell_size
                pos_z = post_half_height
                
                ET.SubElement(worldbody, 'geom', {
                    'name': f'post_{px}_{py}',
                    'type': 'box',
                    'pos': f'{pos_x} {pos_y} {pos_z}',
                    'size': f'{post_half_width} {post_half_width} {post_half_height}',
                    'material': 'mat_post'
                })

        # Walls
        # Wall dimensions
        # Length of wall segment (between posts)
        wall_len = self.cell_size - self.post_width
        wall_half_len = wall_len / 2
        wall_half_thick = self.wall_thickness / 2
        wall_half_height = self.wall_height / 2
        
        # Red top layer dimensions
        red_layer_height = 0.002 # 2mm thick red top
        white_layer_height = self.wall_height - red_layer_height
        
        white_half_height = white_layer_height / 2
        red_half_height = red_layer_height / 2
        
        # Vertical Walls (along Y axis, separating X cells)
        for x in range(17):
            for y in range(16):
                if self.v_walls[x, y] == 1:
                    # Position: x corresponds to the grid line x.
                    # y corresponds to the cell y, so the wall is between y and y+1 grid lines.
                    # Center of wall:
                    pos_x = x * self.cell_size
                    pos_y = y * self.cell_size + self.cell_size / 2
                    
                    # White part
                    ET.SubElement(worldbody, 'geom', {
                        'name': f'v_wall_white_{x}_{y}',
                        'type': 'box',
                        'pos': f'{pos_x} {pos_y} {white_half_height}',
                        'size': f'{wall_half_thick} {wall_half_len} {white_half_height}',
                        'material': 'mat_wall_white'
                    })
                    # Red top part
                    ET.SubElement(worldbody, 'geom', {
                        'name': f'v_wall_red_{x}_{y}',
                        'type': 'box',
                        'pos': f'{pos_x} {pos_y} {white_layer_height + red_half_height}',
                        'size': f'{wall_half_thick} {wall_half_len} {red_half_height}',
                        'material': 'mat_wall_red'
                    })

        # Horizontal Walls (along X axis, separating Y cells)
        for x in range(16):
            for y in range(17):
                if self.h_walls[x, y] == 1:
                    # Position: y corresponds to the grid line y.
                    # x corresponds to the cell x.
                    pos_x = x * self.cell_size + self.cell_size / 2
                    pos_y = y * self.cell_size
                    
                    # White part
                    ET.SubElement(worldbody, 'geom', {
                        'name': f'h_wall_white_{x}_{y}',
                        'type': 'box',
                        'pos': f'{pos_x} {pos_y} {white_half_height}',
                        'size': f'{wall_half_len} {wall_half_thick} {white_half_height}',
                        'material': 'mat_wall_white'
                    })
                    # Red top part
                    ET.SubElement(worldbody, 'geom', {
                        'name': f'h_wall_red_{x}_{y}',
                        'type': 'box',
                        'pos': f'{pos_x} {pos_y} {white_layer_height + red_half_height}',
                        'size': f'{wall_half_len} {wall_half_thick} {red_half_height}',
                        'material': 'mat_wall_red'
                    })

        # Add Mouse
        actuator = ET.SubElement(mujoco, 'actuator')
        sensor = ET.SubElement(mujoco, 'sensor')
        contact = ET.SubElement(mujoco, 'contact')
        self.add_mouse(worldbody, actuator, sensor, contact)

        # Pretty print and save
        xml_str = minidom.parseString(ET.tostring(mujoco)).toprettyxml(indent="  ")
        with open(filename, "w") as f:
            f.write(xml_str)
        print(f"Maze generated and saved to {filename}")

    def add_mouse(self, worldbody, actuator, sensor, contact):
        print("DEBUG: add_mouse called")
        
        # Start position (Center of cell 0,0)
        start_x = 0 * self.cell_size + self.cell_size / 2
        start_y = 0 * self.cell_size + self.cell_size / 2
        # Goal center (8,8) - intersection
        # start_x = 8 * self.cell_size
        # start_y = 8 * self.cell_size
        start_z = 0.012 # Based on sample pos="0.09 0.09 0.012"
        
        # Mouse body
        mouse_body = ET.SubElement(worldbody, 'body', {'name': 'mouse', 'pos': f'{start_x} {start_y} {start_z}'})
        ET.SubElement(mouse_body, 'freejoint', {'name': 'root'})

        # --- Visuals & Geoms (Based on sample_micromouse.xml) ---

        # Main Body (Green Chassis)
        # geom name="mein_body1" type="box" size="0.05 0.03 0.0008" rgba="0.05 0.4 0.15 1"
        ET.SubElement(mouse_body, 'geom', {
            'name': 'mein_body1', 
            'type': 'box', 
            'size': '0.05 0.03 0.0008', 
            'rgba': '0.05 0.4 0.15 1', 
            'pos': '0 0 0.002',
            'mass': '4.65e-3'
        })

        # mein_body2 (coner)
        ET.SubElement(mouse_body, 'geom', {
            'name': 'mein_body2',
            'type': 'mesh',
            'mesh': 'coner',
            'friction': '0.0 0.0 0.0',
            'rgba': '0.05 0.4 0.15 1',
            'pos': '0.014 0.03 0.002',
            'mass': '4.65e-3'
        })
        # mein_body3 (coner)
        ET.SubElement(mouse_body, 'geom', {
            'name': 'mein_body3',
            'type': 'mesh',
            'mesh': 'coner',
            'friction': '0.0 0.0 0.0',
            'rgba': '0.05 0.4 0.15 1',
            'pos': '0.014 -0.03 0.002',
            'euler': '180 0 0',
            'mass': '4.65e-3'
        })
        # mein_body4 (coner)
        ET.SubElement(mouse_body, 'geom', {
            'name': 'mein_body4',
            'type': 'mesh',
            'mesh': 'coner',
            'friction': '0.0 0.0 0.0',
            'rgba': '0.05 0.4 0.15 1',
            'pos': '-0.014 0.03 0.002',
            'euler': '0 180 0',
            'mass': '4.65e-3'
        })
        # mein_body5 (coner)
        ET.SubElement(mouse_body, 'geom', {
            'name': 'mein_body5',
            'type': 'mesh',
            'mesh': 'coner',
            'friction': '0.0 0.0 0.0',
            'rgba': '0.05 0.4 0.15 1',
            'pos': '-0.014 -0.03 0.002',
            'euler': '180 180 0',
            'mass': '4.65e-3'
        })

        # Motor Boxes (Dark Grey)
        # motorbox1
        mbox1 = ET.SubElement(mouse_body, 'body', {'name': 'motorbox1', 'pos': '0 -0.018 0.0135'})
        ET.SubElement(mbox1, 'geom', {
            'type': 'box',
            'size': '0.0117 0.0135 0.0117',
            'rgba': '0.1 0.1 0.1 1',
            'mass': '0.05'
        })
        
        # motorbox2
        mbox2 = ET.SubElement(mouse_body, 'body', {'name': 'motorbox2', 'pos': '0 0.018 0.0135'})
        ET.SubElement(mbox2, 'geom', {
            'type': 'box',
            'size': '0.0117 0.0135 0.0117',
            'rgba': '0.1 0.1 0.1 1',
            'mass': '0.05'
        })

        # Left Wheel
        # body name="left wheel" pos="-0.0 0.036 0.0135" zaxis="0 1 0"
        l_wheel = ET.SubElement(mouse_body, 'body', {
            'name': 'left_wheel', 
            'pos': '0 0.036 0.0135', 
            'zaxis': '0 1 0'
        })
        ET.SubElement(l_wheel, 'joint', {
            'name': 'left_wheel_joint', 
            'type': 'hinge', 
            'axis': '0 0 1', 
            'damping': '0.000054'
        })
        ET.SubElement(l_wheel, 'geom', {
            'type': 'cylinder', 
            'size': '0.0135 0.0035', 
            'rgba': '0.3 0.3 0.3 1', 
            'friction': '1.0 0 0', 
            'mass': '0.05'
        })
        # Wheel site (Green marker)
        ET.SubElement(l_wheel, 'site', {
            'type': 'box', 
            'size': '0.0012 0.008 0.004', 
            'rgba': '0.5 1 0.5 1'
        })

        # Right Wheel
        # body name="right wheel" pos="-0.0 -0.036 0.0135" zaxis="0 1 0"
        r_wheel = ET.SubElement(mouse_body, 'body', {
            'name': 'right_wheel', 
            'pos': '0 -0.036 0.0135', 
            'zaxis': '0 1 0'
        })
        ET.SubElement(r_wheel, 'joint', {
            'name': 'right_wheel_joint', 
            'type': 'hinge', 
            'axis': '0 0 1', 
            'damping': '0.000054'
        })
        ET.SubElement(r_wheel, 'geom', {
            'type': 'cylinder', 
            'size': '0.0135 0.0035', 
            'rgba': '0.3 0.3 0.3 1', # Assuming same color
            'friction': '1.0 0 0', 
            'mass': '0.05'
        })
        ET.SubElement(r_wheel, 'site', {
            'type': 'box', 
            'size': '0.0012 0.008 0.004', 
            'rgba': '0.5 1 0.5 1'
        })

        # Sensors (Visual Bodies + Sites)
        sensors = [
            {'name': 'sensor_left_front', 'pos': '0.031 0.036 0.01', 'zaxis': '1 0 -0.01'},
            {'name': 'sensor_left_side', 'pos': '0.043 0.016 0.01', 'zaxis': '0.25 1 0'},
            {'name': 'sensor_right_front', 'pos': '0.031 -0.036 0.01', 'zaxis': '1 0 -0.01'},
            {'name': 'sensor_right_side', 'pos': '0.043 -0.016 0.01', 'zaxis': '0.25 -1 0'}
        ]
        
        for s in sensors:
            # Create a body for the sensor visual aligned with the measurement direction
            s_body = ET.SubElement(mouse_body, 'body', {'name': f"visual_{s['name']}", 'pos': s['pos'], 'zaxis': s['zaxis']})
            
            # Cylinder part (base) - extends backwards from the tip
            # Diameter is 0.004 (radius 0.002), same as sphere
            ET.SubElement(s_body, 'geom', {
                'type': 'cylinder', 
                'size': '0.002 0.003', 
                'pos': '0 0 -0.003', 
                'rgba': '1 0 0 0.5',
                'mass': '0.001'
            }) 
            
            # Hemisphere part (tip) - using sphere
            # Diameter is 0.004 (radius 0.002)
            ET.SubElement(s_body, 'geom', {
                'type': 'sphere', 
                'size': '0.002', 
                'pos': '0 0 0', 
                'rgba': '1 0 0 0.5',
                'mass': '0.001'
            })
            
            # The actual sensor site (at the tip, pointing +Z relative to body)
            # Made invisible so only the geoms are seen as the "sensor"
            ET.SubElement(s_body, 'site', {
                'name': s['name'], 
                'type': 'box', 
                'size': '0.001 0.001 0.001', 
                'pos': '0 0 0.002', 
                'rgba': '0 0 0 0'
            })
        
        # Accelerometer & Gyro
        ET.SubElement(mouse_body, 'site', {'name': 'accerometer', 'type': 'box', 'size': '0.001 0.001 0.001', 'pos': '0 0 0.0033'})
        ET.SubElement(mouse_body, 'site', {'name': 'gyro', 'type': 'box', 'size': '0.001 0.001 0.001', 'pos': '0 0 0.0033'})

        # Add sensors to sensor node
        ET.SubElement(sensor, 'rangefinder', {'name': 'LF', 'site': 'sensor_left_front', 'cutoff': '0.15'})
        ET.SubElement(sensor, 'rangefinder', {'name': 'LS', 'site': 'sensor_left_side', 'cutoff': '0.15'})
        ET.SubElement(sensor, 'rangefinder', {'name': 'RF', 'site': 'sensor_right_front', 'cutoff': '0.15'})
        ET.SubElement(sensor, 'rangefinder', {'name': 'RS', 'site': 'sensor_right_side', 'cutoff': '0.15'})
        ET.SubElement(sensor, 'accelerometer', {'name': 'Accel', 'site': 'accerometer'})
        ET.SubElement(sensor, 'gyro', {'name': 'Gyro', 'site': 'gyro'})

        # Exclude collisions
        ET.SubElement(contact, 'exclude', {'body1': 'mouse', 'body2': 'left_wheel'})
        ET.SubElement(contact, 'exclude', {'body1': 'mouse', 'body2': 'right_wheel'})

        # Actuators
        print("DEBUG: Adding actuators")
        # Using velocity actuators to maintain control script compatibility, but tuning for new mass.
        # Total mass ~ 0.11kg (50g*2 + 4.65g + etc) -> ~105g.
        # Previous stable kv was 0.005 for similar mass.
        ET.SubElement(actuator, 'velocity', {'name': 'act_left', 'joint': 'left_wheel_joint', 'kv': '0.005', 'ctrlrange': '-30 30'})
        ET.SubElement(actuator, 'velocity', {'name': 'act_right', 'joint': 'right_wheel_joint', 'kv': '0.005', 'ctrlrange': '-30 30'})
        print(f"DEBUG: Actuator content: {ET.tostring(actuator)}")
        print("DEBUG: Actuators added")


if __name__ == "__main__":
    generator = MicromouseMazeGenerator()
    generator.generate_maze()
    generator.generate_mjcf("micromouse_maze.xml")
