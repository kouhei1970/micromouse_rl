"""
Diagnose Phase 3 Navigation Behavior
Analyze why the robot is meandering and taking inefficient paths.
"""
import sys
import os
sys.path.append(os.getcwd())

from phase3_maze.env import MicromouseMazeEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def diagnose_navigation():
    """Diagnose navigation behavior issues"""

    print("=" * 60)
    print("Phase 3 Navigation Behavior Diagnosis")
    print("=" * 60)

    # Load trained model
    model_path = "models/phase3_maze.zip"
    env = MicromouseMazeEnv(
        render_mode=None,
        maze_size=(7, 7),
        spawn_area_size=5,
        max_steps=2000
    )
    model = PPO.load(model_path)

    # Run diagnostic episodes
    num_episodes = 5

    for episode in range(num_episodes):
        obs, info = env.reset()

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}")
        print(f"{'='*60}")
        print(f"Start: {env.start_cell}")
        print(f"Intermediate: {env.intermediate_cell}")
        print(f"Goal: {env.goal_cell}")

        # Calculate optimal Manhattan distance
        start = np.array(env.start_cell)
        intermediate = np.array(env.intermediate_cell) if env.intermediate_cell else None
        goal = np.array(env.goal_cell)

        optimal_dist = np.sum(np.abs(goal - start))
        print(f"Optimal Manhattan distance: {optimal_dist} cells")

        # Track behavior
        trajectory = []
        velocities = []
        angular_velocities = []
        actions = []

        for step in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Record data
            pos = env.data.qpos[0:2].copy()
            trajectory.append(pos)
            actions.append(action)

            # Get actual velocities from info
            if 'actual_linear' in info:
                velocities.append(info['actual_linear'])
                angular_velocities.append(info['actual_angular'])

            if terminated or truncated:
                break

        # Analyze trajectory
        trajectory = np.array(trajectory)
        actual_distance = np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)))

        print(f"\nTrajectory Analysis:")
        print(f"  Actual path length: {actual_distance:.2f}m")
        print(f"  Optimal path length (approx): {optimal_dist * 0.18:.2f}m")
        print(f"  Efficiency: {(optimal_dist * 0.18 / actual_distance * 100):.1f}%")
        print(f"  Steps taken: {len(trajectory)}")

        # Analyze control actions
        actions = np.array(actions)
        print(f"\nControl Action Statistics:")
        print(f"  Mean linear velocity command: {np.mean(actions[:, 0]):.3f} m/s")
        print(f"  Std linear velocity command: {np.std(actions[:, 0]):.3f}")
        print(f"  Mean angular velocity command: {np.mean(actions[:, 1]):.3f} rad/s")
        print(f"  Std angular velocity command: {np.std(actions[:, 1]):.3f}")
        print(f"  Angular velocity changes: {np.sum(np.abs(np.diff(actions[:, 1]))):.1f}")

        # Check for oscillation
        if len(actions) > 10:
            recent_angular = actions[-100:, 1] if len(actions) >= 100 else actions[:, 1]
            sign_changes = np.sum(np.diff(np.sign(recent_angular)) != 0)
            print(f"  Sign changes in angular vel (last 100 steps): {sign_changes}")
            if sign_changes > 50:
                print("  ⚠️  HIGH OSCILLATION DETECTED!")

    env.close()

    print("\n" + "=" * 60)
    print("Diagnosis Summary")
    print("=" * 60)
    print("\nLikely Issues:")
    print("1. Observation space may lack directional information")
    print("2. Reward structure doesn't penalize path length")
    print("3. High-level policy may need better goal-directed features")
    print("\nRecommended Solutions:")
    print("1. Add goal direction vector to observations")
    print("2. Add path efficiency reward component")
    print("3. Increase time penalty to discourage meandering")
    print("4. Add smoothness penalty for oscillating commands")

def visualize_observation_space():
    """Analyze what the high-level policy can observe"""

    print("\n" + "=" * 60)
    print("Observation Space Analysis")
    print("=" * 60)

    env = MicromouseMazeEnv(render_mode=None, maze_size=(7, 7), spawn_area_size=5, max_steps=2000)
    obs, _ = env.reset()

    print(f"\nObservation vector shape: {obs.shape}")
    print(f"Observation vector: {obs}")

    # Decode observation
    print("\nObservation components:")
    print(f"  [0:2] Robot position (x, y): {obs[0:2]}")
    print(f"  [2] Robot z-position: {obs[2]}")
    print(f"  [3:7] Robot orientation (quaternion): {obs[3:7]}")
    print(f"  [7:9] Robot linear velocity: {obs[7:9]}")
    print(f"  [9] Robot angular velocity (z): {obs[9]}")
    print(f"  [10:12] Goal position (x, y): {obs[10:12]}")

    # Calculate what's missing
    robot_pos = obs[0:2]
    goal_pos = obs[10:12]

    # Direction to goal
    direction = goal_pos - robot_pos
    distance = np.linalg.norm(direction)
    direction_normalized = direction / (distance + 1e-8)

    # Robot heading from quaternion
    q = obs[3:7]
    w, x, y, z = q
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    heading_vec = np.array([np.cos(yaw), np.sin(yaw)])

    # Angle to goal
    angle_to_goal = np.arctan2(direction[1], direction[0]) - yaw
    angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))  # Normalize to [-pi, pi]

    print("\n⚠️  MISSING FROM OBSERVATION:")
    print(f"  ❌ Direction to goal (normalized): {direction_normalized}")
    print(f"  ❌ Angle to goal: {np.degrees(angle_to_goal):.1f}°")
    print(f"  ❌ Distance to goal: {distance:.3f}m")
    print(f"  ❌ Robot heading vector: {heading_vec}")

    print("\n💡 The policy must learn to compute these from position/orientation!")
    print("   This makes learning harder and less sample-efficient.")

    env.close()

if __name__ == "__main__":
    diagnose_navigation()
    visualize_observation_space()
