"""
Phase 3: Maze Navigation Training
Train high-level policy to navigate 2 cells in random mazes.
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
import os
import datetime

sys.path.append(os.getcwd())
from phase3_maze.env import MicromouseMazeEnv
from common.visualization import plot_learning_curve, record_video
from common.output_manager import OutputManager

def main():
    print("=" * 60)
    print("Phase 3: Maze Navigation Training")
    print("High-Level Policy: 2-Cell Navigation in Random Mazes")
    print("=" * 60)

    # Initialize output manager
    output_mgr = OutputManager("phase3_maze")

    # Create environment with 7x7 maze
    env = MicromouseMazeEnv(
        render_mode=None,
        maze_size=(7, 7),
        spawn_area_size=5,
        max_steps=2000,
        maze_regen_interval=20  # Regenerate every 20 episodes
    )

    model_path = "models/phase3_maze.zip"

    # Check if model exists for continued training
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} to continue training...")
        model = PPO.load(model_path, env=env, verbose=1)
    else:
        print("Creating NEW model for Phase 3 Maze Navigation")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )

    print("\nTraining Configuration:")
    print("  Environment: 7x7 maze with 5x5 spawn area")
    print("  Max steps per episode: 2,000")
    print("  Maze regeneration: every 20 episodes")
    print("\nReward Structure:")
    print("  - Goal reached: +1000.0")
    print("  - Goal orientation (±15°): +200.0")
    print("  - Intermediate cell: +100.0")
    print("  - Time penalty: -1.0 per step")
    print("  - Collision penalty: -10.0")
    print("  - Tip-over: -100.0")
    print("\nPPO Hyperparameters:")
    print("  - Learning rate: 3e-4")
    print("  - Batch size: 64")
    print("  - n_steps: 2048")
    print("  - n_epochs: 10")
    print("  - Entropy coefficient: 0.01")

    print("\n" + "=" * 60)
    print("Starting training for 1,000,000 timesteps...")
    print("=" * 60)

    # Training with checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./logs/',
        name_prefix='ppo_phase3_maze'
    )

    model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    print("\nTraining complete.")

    # Save the final model
    model.save(model_path)
    print(f"Final model saved to {model_path}")

    # Generate outputs using OutputManager
    print(f"\nGenerating outputs...")

    # Plot learning curve
    learning_curve_path = output_mgr.get_path("learning_curve.png")
    plot_learning_curve("logs/", str(learning_curve_path), window=50)

    # Record evaluation video
    print("\nRecording evaluation videos...")
    eval_env = MicromouseMazeEnv(
        render_mode="rgb_array",
        maze_size=(7, 7),
        spawn_area_size=5,
        max_steps=2000
    )

    video_path = output_mgr.get_path("evaluation.mp4")
    record_video(eval_env, model, str(video_path), num_episodes=5, deterministic=True, fps=60)

    # Evaluate performance
    print("\n" + "=" * 60)
    print("Performance Evaluation (10 episodes)")
    print("=" * 60)

    success_count = 0
    intermediate_count = 0
    total_steps = 0
    total_reward = 0.0

    for episode in range(10):
        obs, info = eval_env.reset()
        episode_reward = 0.0
        episode_steps = 0

        for step in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                break

        # Check success
        if eval_env.intermediate_reached:
            intermediate_count += 1

        if episode_reward > 500:  # Goal reached
            success_count += 1
            print(f"Episode {episode + 1}: ✓ SUCCESS (steps={episode_steps}, reward={episode_reward:.1f})")
        elif eval_env.intermediate_reached:
            print(f"Episode {episode + 1}: △ INTERMEDIATE (steps={episode_steps}, reward={episode_reward:.1f})")
        else:
            print(f"Episode {episode + 1}: ✗ FAILED (steps={episode_steps}, reward={episode_reward:.1f})")

        total_steps += episode_steps
        total_reward += episode_reward

    eval_env.close()

    # Calculate success metrics
    success_rate = success_count / 10.0
    intermediate_rate = intermediate_count / 10.0
    avg_steps = total_steps / 10.0
    avg_reward = total_reward / 10.0

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Success Rate: {success_count}/10 ({success_count * 10}%)")
    print(f"Intermediate Rate: {intermediate_count}/10 ({intermediate_count * 10}%)")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Reward: {avg_reward:.1f}")

    # Save model info
    output_mgr.save_model_info({
        "Algorithm": "PPO",
        "Training Steps": 1000000,
        "Learning Rate": 3e-4,
        "Batch Size": 64,
        "n_steps": 2048,
        "n_epochs": 10,
        "Entropy Coefficient": 0.01,
        "Maze Size": "7x7",
        "Spawn Area": "5x5",
        "Max Episode Steps": 2000,
    })

    # Save metrics
    output_mgr.save_metrics(
        metrics={
            "training_steps": 1000000,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
        },
        phase_specific={
            "intermediate_rate": intermediate_rate,
            "maze_size": "7x7",
            "spawn_area": "5x5",
        }
    )

    # Finalize outputs
    output_mgr.finalize(
        summary=f"Training complete!\n"
                f"Success: {success_count}/10 ({success_rate*100:.0f}%)\n"
                f"Avg Steps: {avg_steps:.1f}\n"
                f"Model: {model_path}"
    )

if __name__ == "__main__":
    main()
