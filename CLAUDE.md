# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 言語設定

- 応答は日本語で行うこと
- コードのコメントも日本語で記述すること

## Project Overview

Micromouse reinforcement learning project using MuJoCo physics simulation and Stable-Baselines3 (PPO) to train a robot to navigate mazes autonomously.

**Key Technologies:** MuJoCo, Gymnasium, Stable-Baselines3, Python 3.10+

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
# Also needed: matplotlib opencv-python imageio

# Phase 1: Low-level velocity control (1M steps)
python phase1_open/train.py

# Phase 2: Slalom navigation
python phase2_slalom/train.py

# Phase 3: Maze navigation (requires Phase 1 model)
python phase3_maze/train.py

# Evaluate/analyze Phase 3
python phase3_maze/evaluate.py
python phase3_maze/analyze_final_results.py
python phase3_maze/diagnose_behavior.py

# Generate maze visualization
python phase3_maze/visualize_maze.py --size 7

# Extract learning curves from checkpoints
python common/extract_learning_curve.py \
  --checkpoint-pattern 'logs/ppo_phase3_maze_*_steps.zip' \
  --output 'outputs/phase3_maze/learning_curve.png'
```

## Architecture

### Hierarchical RL Structure
```
Phase 3 Policy (High-Level) → target velocities (linear, angular)
    ↓
Phase 1 Controller (Low-Level) → motor voltages (left, right)
    ↓
MuJoCo Physics Engine
```

- **Phase 1:** Low-level motor control. Learns to track target velocities via motor voltage commands.
- **Phase 2:** Slalom navigation (intermediate step).
- **Phase 3:** Maze navigation using Phase 1 as a frozen low-level controller.

### Key Directories
- `assets/` - MuJoCo XML model files (robot and environment definitions)
- `models/` - Trained model checkpoints (phase1_open.zip, phase2_slalom.zip, phase3_maze.zip)
- `phase{1,2,3}_*/` - Phase-specific environment (env.py) and training (train.py) code
- `common/` - Shared utilities (OutputManager, visualization, maze generation)
- `outputs/` - Experiment results with timestamped archives and `latest/` symlinks
- `logs/` - Training checkpoints saved during training

### Environment Pattern
All environments follow Gymnasium API in `env.py`:
- Observation/action spaces defined in `__init__`
- `step()` returns (obs, reward, terminated, truncated, info)
- `reset()` reinitializes environment with optional seed
- Rendering supports "human" and "rgb_array" modes

### Output Management
Use `OutputManager` from `common/output_manager.py`:
```python
from common.output_manager import OutputManager
output_mgr = OutputManager("phase3_maze")
# Saves to outputs/phase3_maze/archive/{timestamp}/ with latest/ symlink
output_mgr.save_metrics({"success_rate": 0.85})
output_mgr.finalize()
```

### Maze Generation
DFS-based perfect maze generation in `common/maze_generator.py`:
- Walls stored as `v_walls[x,y]` (vertical) and `h_walls[x,y]` (horizontal) numpy arrays
- Cell size: 0.18m (180mm standard)
- Generates MuJoCo XML via `common/maze_assets.py`

## Training Pattern

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

env = PhaseEnv(...)
model = PPO("MlpPolicy", env, ...)
checkpoint_cb = CheckpointCallback(save_freq=50000, save_path="logs/")
model.learn(total_timesteps=1000000, callback=checkpoint_cb)
model.save("models/phase_name.zip")
```

## Documentation Notes

Primary documentation is in Japanese. Key references:
- `README.md` - Project overview
- `docs/MAZE_SPECIFICATION.md` - Maze design specs
- `OUTPUT_STRUCTURE.md` - Output directory specification
- Phase-specific READMEs in each phase directory
