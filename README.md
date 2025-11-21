# Micromouse MuJoCo Simulation

This project is a simulation environment for Micromouse competitions using the [MuJoCo](https://mujoco.org/) physics engine. It provides a procedural maze generator and a realistic robot model with sensors, suitable for developing and testing reinforcement learning algorithms or classic control strategies.

## Features

*   **Procedural Maze Generation**: Generates a random 16x16 Micromouse maze using a depth-first search algorithm.
    *   Compliant with standard Micromouse maze dimensions (180mm cell size).
    *   Includes a dedicated 2x2 goal area in the center (coordinates 7,7 to 8,8) with no internal walls or pillars.
    *   Configurable start position (default is 0,0).
*   **Realistic Robot Model**:
    *   Based on a reference design with specific dimensions and mass properties.
    *   Differential drive kinematics.
    *   **Sensors**: 4 distance sensors (LF, LS, RF, RS) visualized as semi-transparent red indicators (cylinder + sphere tip).
    *   **Physics**: Includes friction, damping, and actuator dynamics tuned for the mouse's mass (~100g).
*   **Simulation & Control**:
    *   `control_mouse.py`: A sample control script that demonstrates random movement (Forward, Backward, Turn Left/Right, Stop) to verify kinematics and physics stability.
    *   Real-time visualization using the MuJoCo viewer.

## Requirements

*   Python 3.x
*   [MuJoCo](https://pypi.org/project/mujoco/)
*   [NumPy](https://numpy.org/)

## Installation

1.  Clone this repository:
    ```bash
    git clone <repository-url>
    cd micromouse_mujoco
    ```

2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Generate the Maze and Robot Model
Run the generator script to create the `micromouse_maze.xml` file. This file contains the full definition of the maze geometry and the robot.

```bash
python maze_generator.py
```

### 2. Run the Simulation
Execute the control script to launch the MuJoCo viewer and see the mouse in action.

```bash
# Standard Python execution
python control_mouse.py

# On macOS, if you encounter issues with the viewer, try using mjpython
mjpython control_mouse.py
```

The mouse will perform random movements. You can observe the sensor visualizations and the physics interactions with the walls.

## File Structure

*   `maze_generator.py`: Main script to generate the MJCF (XML) file. Handles maze logic and robot XML construction.
*   `control_mouse.py`: Script to load the generated XML and run the simulation loop with a simple controller.
*   `micromouse_maze.xml`: The generated output file (do not edit manually if you plan to regenerate).
*   `sample_micromouse.xml`: Reference XML file used for robot specifications.
*   `requirements.txt`: Python dependencies.

## License

[MIT License](LICENSE) (or specify your license)
