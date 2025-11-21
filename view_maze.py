import mujoco
import mujoco.viewer

def main():
    try:
        print("Loading model...")
        model = mujoco.MjModel.from_xml_path("micromouse_maze.xml")
        data = mujoco.MjData(model)
        print("Model loaded. Launching viewer...")
        mujoco.viewer.launch(model, data)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
