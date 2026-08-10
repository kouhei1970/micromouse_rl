import xml.etree.ElementTree as ET

def add_micromouse(worldbody, actuator, sensor, contact, pos="0.09 0.09 0.002", euler="0 0 90"):
    """
    Adds the Micromouse robot to the worldbody.
    
    Args:
        worldbody: The MuJoCo worldbody element.
        actuator: The MuJoCo actuator element.
        sensor: The MuJoCo sensor element.
        contact: The MuJoCo contact element.
        pos: Initial position string "x y z".
        euler: Initial orientation string "r p y".
    """
    
    mouse_body = ET.SubElement(worldbody, 'body', {'name': 'mouse', 'pos': pos, 'euler': euler})
    ET.SubElement(mouse_body, 'freejoint', {'name': 'root'})
    
    # Camera (Top-Down View to verify alignment)
    ET.SubElement(mouse_body, 'camera', {
        'name': 'track',
        'mode': 'trackcom',
        'pos': '0 0 0.8',
        'xyaxes': '0 -1 0 1 0 0',
        'fovy': '45'
    })

    # --- Visuals & Geoms ---

    # Main Body (Green Chassis)
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
    ET.SubElement(l_wheel, 'site', {
        'type': 'box', 
        'size': '0.0012 0.008 0.004', 
        'rgba': '0.5 1 0.5 1'
    })

    # Right Wheel
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
        'rgba': '0.3 0.3 0.3 1', 
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
        
        # Cylinder part (base)
        ET.SubElement(s_body, 'geom', {
            'type': 'cylinder', 
            'size': '0.002 0.003', 
            'pos': '0 0 -0.003', 
            'rgba': '1 0 0 0.5',
            'mass': '0.001'
        }) 
        
        # Hemisphere part (tip)
        ET.SubElement(s_body, 'geom', {
            'type': 'sphere', 
            'size': '0.002', 
            'pos': '0 0 0', 
            'rgba': '1 0 0 0.5',
            'mass': '0.001'
        })
        
        # The actual sensor site
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
    ET.SubElement(actuator, 'motor', {'name': 'act_left', 'joint': 'left_wheel_joint', 'gear': '0.0072', 'ctrllimited': 'true', 'ctrlrange': '-3 3'})
    ET.SubElement(actuator, 'motor', {'name': 'act_right', 'joint': 'right_wheel_joint', 'gear': '0.0072', 'ctrllimited': 'true', 'ctrlrange': '-3 3'})
