import mujoco
import numpy as np

# Load the model
model = mujoco.MjModel.from_xml_path("assets/micromouse_slalom.xml")
data = mujoco.MjData(model)

# Step the simulation forward a bit to let things settle (or explode)
print("Checking for collisions...")
for i in range(100):
    mujoco.mj_step(model, data)
    
    # Check contacts
    ncon = data.ncon
    if ncon > 0:
        print(f"Step {i}: {ncon} contacts detected.")
        for c in range(ncon):
            contact = data.contact[c]
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            dist = contact.dist
            print(f"  Contact between {geom1_name} and {geom2_name}, dist={dist}")
            
            # If distance is negative, it's penetration (stuck)
            if dist < 0:
                print(f"  !!! PENETRATION DETECTED !!! Amount: {dist}")

    # Print mouse position
    mouse_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    mouse_pos = data.xpos[mouse_body_id]
    print(f"Step {i}: Mouse Pos: {mouse_pos}")
    
    if i > 5: break # Just check the first few steps

# Check Camera Position
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
# Force update of kinematics to get correct camera pos
mujoco.mj_forward(model, data)
print(f"Camera World Pos: {data.cam_xpos[cam_id]}")

print("\n--- Detailed Geometry Check ---")
mouse_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
mouse_pos = data.xpos[mouse_body_id]
mouse_mat = data.xmat[mouse_body_id].reshape(3, 3)
print(f"Mouse Body Pos: {mouse_pos}")
print(f"Mouse Body Rot:\n{mouse_mat}")

# Iterate over all geoms in the mouse body
geom_start = model.body_geomadr[mouse_body_id]
geom_num = model.body_geomnum[mouse_body_id]

min_x, min_y, min_z = np.inf, np.inf, np.inf
max_x, max_y, max_z = -np.inf, -np.inf, -np.inf

for i in range(geom_num):
    geom_id = geom_start + i
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    geom_pos = data.geom_xpos[geom_id]
    # Get geom size (approximation for box/sphere)
    # This is hard because rotation matters.
    # Let's just look at the center positions first.
    print(f"Geom {geom_name}: World Pos {geom_pos}")
    
    # Update bounding box of CENTERS (not perfect but helpful)
    min_x = min(min_x, geom_pos[0])
    max_x = max(max_x, geom_pos[0])
    min_y = min(min_y, geom_pos[1])
    max_y = max(max_y, geom_pos[1])

print(f"Mouse Centers BBox: X[{min_x:.4f}, {max_x:.4f}], Y[{min_y:.4f}, {max_y:.4f}]")

# Check Wall Positions
print("\n--- Wall Check ---")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name and "wall" in name:
        pos = data.geom_xpos[i]
        size = model.geom_size[i]
        print(f"Wall {name}: Pos {pos}, Size {size}")
