# Micromouse Reinforcement Learning Project

This project aims to train a Micromouse agent using Reinforcement Learning (RL) with MuJoCo physics engine.
The project is divided into phases to incrementally build the agent's capabilities.

## Project Structure

```text
micromouse_rl/
├── models/                     # Trained models (.zip)
│   ├── phase1_open.zip         # Phase 1: Open field navigation
│   └── phase2_slalom.zip       # Phase 2: Slalom turn
├── common/                     # Common utilities and base classes
├── phase1_open/                # Phase 1: Open Field Task
├── phase2_slalom/              # Phase 2: Slalom Turn Task
├── assets/                     # MuJoCo XML assets and textures
├── docs/                       # Documentation
└── outputs/                    # Generated outputs (videos, logs)
```

## Phases

### Phase 1: Open Field
Focuses on basic stability and velocity control in an open space.
- **Goal**: Move forward and turn to target coordinates.
- **Status**: Completed.

### Phase 2: Slalom Turn
Focuses on navigating a specific L-shaped turn (Slalom).
- **Goal**: Navigate from start to goal through a turn without hitting walls.
- **Status**: In Progress (Refining simulation accuracy).

## Usage

### Prerequisites
- Python 3.10+
- MuJoCo
- Gymnasium
- Stable Baselines3

### Running Scripts
Run scripts from the project root directory.

**Example: Generate Slalom Maze**
```bash
python phase2_slalom/generate_maze.py
```

**Example: Train Slalom Agent**
```bash
python phase2_slalom/train.py
```

**Example: Create Slalom Video**
```bash
python phase2_slalom/create_video.py
```
