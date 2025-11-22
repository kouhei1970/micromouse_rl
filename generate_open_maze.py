from maze_generator import MicromouseMazeGenerator
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

class OpenMazeGenerator(MicromouseMazeGenerator):
    def generate_maze(self):
        """
        Generates an open maze (only outer walls).
        Overrides the DFS maze generation.
        """
        # Initialize with NO walls (0)
        self.v_walls.fill(0)
        self.h_walls.fill(0)
        
        # Add outer walls
        # West and East borders
        # v_walls shape is (17, 16). Index 0 is West of cell 0. Index 16 is East of cell 15.
        self.v_walls[0, :] = 1   # Leftmost wall
        self.v_walls[16, :] = 1  # Rightmost wall
        
        # South and North borders
        # h_walls shape is (16, 17). Index 0 is South of row 0. Index 16 is North of row 15.
        self.h_walls[:, 0] = 1   # Bottom wall
        self.h_walls[:, 16] = 1  # Top wall
        
        print("Open maze structure generated.")

    def generate_mjcf(self, filename="maze.xml"):
        mujoco = ET.Element('mujoco', {'model': 'micromouse_maze'})
        
        # Compiler settings
        compiler = ET.SubElement(mujoco, 'compiler', {'angle': 'degree', 'coordinate': 'local', 'inertiafromgeom': 'true'})
        
        # Option settings
        ET.SubElement(mujoco, 'option', {'timestep': '0.001', 'gravity': '0 0 -9.81'})

        # Visual settings (Framebuffer size)
        visual = ET.SubElement(mujoco, 'visual')
        ET.SubElement(visual, 'global', {'offwidth': '1024', 'offheight': '1024'})
        
        # Assets (Materials and Textures)
        asset = ET.SubElement(mujoco, 'asset')
        ET.SubElement(asset, 'texture', {'name': 'tex_floor', 'type': '2d', 'builtin': 'checker', 'rgb1': '.2 .3 .4', 'rgb2': '.1 .2 .3', 'width': '512', 'height': '512'})
        # 90mm squares. Floor is 200m wide. 200 / (0.09 * 2) = 1111.11
        ET.SubElement(asset, 'material', {'name': 'mat_floor', 'texture': 'tex_floor', 'texrepeat': '1111 1111', 'reflectance': '0.3'})
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
        
        # Floor - HUGE OPEN SPACE
        ET.SubElement(worldbody, 'geom', {'name': 'floor', 'pos': '0 0 0', 'size': '100 100 0.1', 'type': 'plane', 'material': 'mat_floor'})
        
        # Light
        ET.SubElement(worldbody, 'light', {'diffuse': '.5 .5 .5', 'pos': '0 0 4', 'dir': '0 0 -1'})

        # NO POSTS OR WALLS for the infinite open field experiment
        
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
        # Start at center of the world (0, 0)
        start_x = 0
        start_y = 0
        start_z = 0.012 # Based on sample pos="0.09 0.09 0.012"
        
        # Mouse body
        mouse_body = ET.SubElement(worldbody, 'body', {'name': 'mouse', 'pos': f'{start_x} {start_y} {start_z}'})
        ET.SubElement(mouse_body, 'freejoint', {'name': 'root'})

        # Attached Camera (Fixed relative to body)
        ET.SubElement(mouse_body, 'camera', {
            'name': 'track',
            'mode': 'fixed',
            'pos': '-0.4 0 0.5',
            'xyaxes': '0 -1 0 0.74 0 0.67', # Look at front of mouse from rear-top (Height 50cm)
            'fovy': '60'
        })

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

        # Casters (Frictionless spheres to prevent pitching)
        ET.SubElement(mouse_body, 'geom', {
            'name': 'caster_front',
            'type': 'sphere',
            'size': '0.002',
            'pos': '0.045 0 0.002',
            'rgba': '0.5 0.5 0.5 1',
            'friction': '0 0 0',
            'mass': '0.001'
        })
        ET.SubElement(mouse_body, 'geom', {
            'name': 'caster_back',
            'type': 'sphere',
            'size': '0.002',
            'pos': '-0.045 0 0.002',
            'rgba': '0.5 0.5 0.5 1',
            'friction': '0 0 0',
            'mass': '0.001'
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
        # Adjusted zaxis to point ~1.5 degrees up (tan(1.5) approx 0.026) to avoid floor detection
        sensors = [
            {'name': 'sensor_left_front', 'pos': '0.031 0.036 0.01', 'zaxis': '1 0 0.026'},
            {'name': 'sensor_left_side', 'pos': '0.043 0.016 0.01', 'zaxis': '0.25 1 0.026'},
            {'name': 'sensor_right_front', 'pos': '0.031 -0.036 0.01', 'zaxis': '1 0 0.026'},
            {'name': 'sensor_right_side', 'pos': '0.043 -0.016 0.01', 'zaxis': '0.25 -1 0.026'}
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
        # Change to motor actuator for voltage control simulation
        # gear="0.0072" implies 1V input generates 0.0072 Nm torque (Adjusted for ~5m/s)
        # ctrlrange="-3 3" limits input voltage to +/- 3V
        ET.SubElement(actuator, 'motor', {'name': 'act_left', 'joint': 'left_wheel_joint', 'gear': '0.0072', 'ctrllimited': 'true', 'ctrlrange': '-3 3'})
        ET.SubElement(actuator, 'motor', {'name': 'act_right', 'joint': 'right_wheel_joint', 'gear': '0.0072', 'ctrllimited': 'true', 'ctrlrange': '-3 3'})
        print(f"DEBUG: Actuator content: {ET.tostring(actuator)}")
        print("DEBUG: Actuators added")

if __name__ == "__main__":
    generator = OpenMazeGenerator()
    generator.generate_maze()
    # Use the parent class's method to save XML, ensuring all assets/mouse are included
    generator.generate_mjcf("micromouse_open.xml")

