"""
Extract learning curve from checkpoint models.
Since we don't have monitor logs, we can load checkpoints and evaluate them.
"""
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from phase3_maze.env import MicromouseMazeEnv
import glob
import re

def extract_from_checkpoints(checkpoint_pattern, output_path):
    """
    Extract learning curve by evaluating checkpoint models.

    Args:
        checkpoint_pattern: glob pattern for checkpoint files
        output_path: where to save the learning curve plot
    """

    print("=" * 60)
    print("Extracting Learning Curve from Checkpoints")
    print("=" * 60)

    # Find all checkpoints
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))

    if len(checkpoint_files) == 0:
        print(f"No checkpoint files found matching: {checkpoint_pattern}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint files")

    # Extract timestep numbers from filenames
    timesteps = []
    for f in checkpoint_files:
        match = re.search(r'(\d+)_steps\.zip$', f)
        if match:
            timesteps.append(int(match.group(1)))

    # Sort by timestep
    sorted_indices = np.argsort(timesteps)
    checkpoint_files = [checkpoint_files[i] for i in sorted_indices]
    timesteps = [timesteps[i] for i in sorted_indices]

    # Evaluate each checkpoint
    avg_rewards = []
    avg_lengths = []

    env = MicromouseMazeEnv(
        render_mode=None,
        maze_size=(7, 7),
        spawn_area_size=5,
        max_steps=2000
    )

    num_eval_episodes = 5  # Evaluate each checkpoint on 5 episodes

    for i, (checkpoint_file, step) in enumerate(zip(checkpoint_files, timesteps)):
        print(f"Evaluating checkpoint {i+1}/{len(checkpoint_files)}: {step} steps...", end=" ")

        try:
            model = PPO.load(checkpoint_file, env=env)

            episode_rewards = []
            episode_lengths = []

            for _ in range(num_eval_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0

                for step_count in range(2000):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1

                    if terminated or truncated:
                        break

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)

            avg_rewards.append(avg_reward)
            avg_lengths.append(avg_length)

            print(f"Avg Reward: {avg_reward:.1f}, Avg Length: {avg_length:.0f}")

        except Exception as e:
            print(f"Error: {e}")
            avg_rewards.append(np.nan)
            avg_lengths.append(np.nan)

    env.close()

    # Plot learning curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Reward plot
    ax1.plot(timesteps, avg_rewards, 'b-', linewidth=2, marker='o', markersize=4, label='Average Reward')
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Learning Curve: Reward', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Episode length plot
    ax2.plot(timesteps, avg_lengths, 'r-', linewidth=2, marker='s', markersize=4, label='Average Episode Length')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Average Episode Length (steps)', fontsize=12)
    ax2.set_title('Learning Curve: Episode Length', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nLearning curve saved to: {output_path}")

    # Also save data to CSV
    csv_path = output_path.replace('.png', '.csv')
    with open(csv_path, 'w') as f:
        f.write("timesteps,avg_reward,avg_length\n")
        for t, r, l in zip(timesteps, avg_rewards, avg_lengths):
            f.write(f"{t},{r},{l}\n")
    print(f"Data saved to: {csv_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract learning curve from checkpoint models')
    parser.add_argument('--checkpoint-pattern', type=str,
                       default='logs/ppo_phase3_maze_*_steps.zip',
                       help='Glob pattern for checkpoint files')
    parser.add_argument('--output', type=str,
                       default='outputs/phase3_maze/learning_curve_from_checkpoints.png',
                       help='Output path for learning curve plot')

    args = parser.parse_args()

    extract_from_checkpoints(args.checkpoint_pattern, args.output)
