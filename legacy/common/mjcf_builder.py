import xml.etree.ElementTree as ET
from xml.dom import minidom
from common.maze_assets import add_assets
from common.robot_builder import add_micromouse

class MJCFBuilder:
    def __init__(self, maze_generator):
        self.gen = maze_generator
        self.width = maze_generator.width
        self.height = maze_generator.height
        self.cell_size = 0.18
        self.wall_thickness = 0.012
        self.wall_height = 0.05
        self.post_size = 0.012

    def build(self, filename, include_walls=True, include_posts=True, mouse_pos="0.09 0.09 0.002", mouse_euler="0 0 90"):
        mujoco = ET.Element('mujoco', {'model': f'micromouse_{self.width}x{self.height}'})
        
        # Compiler & Option
        ET.SubElement(mujoco, 'compiler', {'angle': 'degree', 'coordinate': 'local', 'inertiafromgeom': 'true'})
        ET.SubElement(mujoco, 'option', {'timestep': '0.001', 'gravity': '0 0 -9.81'})
        
        # Visual
        visual = ET.SubElement(mujoco, 'visual')
        ET.SubElement(visual, 'global', {'offwidth': '800', 'offheight': '800'})
        
        # Assets
        add_assets(mujoco, floor_texrepeat="50 50")
        
        # Worldbody
        worldbody = ET.SubElement(mujoco, 'worldbody')
        
        # Floor
        # Size: 18m x 18m (Half-size 9m)
        # Texture repeat: 36m / 0.36m = 100 repeats
        # This ensures 0.18m squares align with origin (0,0)
        ET.SubElement(worldbody, 'geom', {
            'name': 'floor', 
            'size': '9 9 0.1', 
            'type': 'plane', 
            'material': 'mat_floor'
        })
        
        # Light
        ET.SubElement(worldbody, 'light', {'diffuse': '.5 .5 .5', 'pos': '0 0 3', 'dir': '0 0 -1'})

        # Walls and Posts
        if include_posts:
            for x in range(self.width + 1):
                for y in range(self.height + 1):
                    px = x * self.cell_size
                    py = y * self.cell_size
                    ET.SubElement(worldbody, 'geom', {
                        'name': f'post_{x}_{y}',
                        'type': 'box',
                        'size': f'{self.post_size/2} {self.post_size/2} {self.wall_height/2}',
                        'pos': f'{px} {py} {self.wall_height/2}',
                        'material': 'mat_post'
                    })

        if include_walls:
            # Vertical Walls
            for x in range(self.width + 1):
                for y in range(self.height):
                    if self.gen.v_walls[x, y] == 1:
                        px = x * self.cell_size
                        py = y * self.cell_size + self.cell_size / 2
                        ET.SubElement(worldbody, 'geom', {
                            'name': f'v_wall_{x}_{y}',
                            'type': 'box',
                            'size': f'{self.wall_thickness/2} {self.cell_size/2 - self.post_size/2} {self.wall_height/2}',
                            'pos': f'{px} {py} {self.wall_height/2}',
                            'material': 'mat_wall'
                        })
                        # Red Top
                        ET.SubElement(worldbody, 'geom', {
                            'name': f'v_wall_top_{x}_{y}',
                            'type': 'box',
                            'size': f'{self.wall_thickness/2} {self.cell_size/2 - self.post_size/2} 0.001',
                            'pos': f'{px} {py} {self.wall_height}',
                            'material': 'mat_wall_top'
                        })

            # Horizontal Walls
            for x in range(self.width):
                for y in range(self.height + 1):
                    if self.gen.h_walls[x, y] == 1:
                        px = x * self.cell_size + self.cell_size / 2
                        py = y * self.cell_size
                        ET.SubElement(worldbody, 'geom', {
                            'name': f'h_wall_{x}_{y}',
                            'type': 'box',
                            'size': f'{self.cell_size/2 - self.post_size/2} {self.wall_thickness/2} {self.wall_height/2}',
                            'pos': f'{px} {py} {self.wall_height/2}',
                            'material': 'mat_wall'
                        })
                        # Red Top
                        ET.SubElement(worldbody, 'geom', {
                            'name': f'h_wall_top_{x}_{y}',
                            'type': 'box',
                            'size': f'{self.cell_size/2 - self.post_size/2} {self.wall_thickness/2} 0.001',
                            'pos': f'{px} {py} {self.wall_height}',
                            'material': 'mat_wall_top'
                        })

        # Add Mouse
        actuator = ET.SubElement(mujoco, 'actuator')
        sensor = ET.SubElement(mujoco, 'sensor')
        contact = ET.SubElement(mujoco, 'contact')
        
        add_micromouse(worldbody, actuator, sensor, contact, pos=mouse_pos, euler=mouse_euler)

        # Save to file
        xml_str = minidom.parseString(ET.tostring(mujoco)).toprettyxml(indent="  ")
        with open(filename, "w") as f:
            f.write(xml_str)
        print(f"Generated {filename}")
