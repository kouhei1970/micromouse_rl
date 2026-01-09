"""
Phase 4: Speed Learning Training
センサーパターンから最適な速度を学習する訓練スクリプト
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from phase4_speed.env import MicromouseSpeedEnv
from common.visualization import plot_learning_curve, record_video
from common.output_manager import OutputManager


def main():
    print("=" * 60)
    print("Phase 4: Speed Learning Training")
    print("Goal: Learn optimal speed for each sensor pattern")
    print("=" * 60)

    # 出力マネージャを初期化
    output_mgr = OutputManager("phase4_speed")

    # 環境を作成
    env = MicromouseSpeedEnv(
        render_mode=None,
        maze_size=(7, 7),
        spawn_area_size=5,
        max_steps=4000,
        maze_regen_interval=20
    )

    model_path = "models/phase4_speed.zip"

    # TensorBoardログディレクトリ
    tensorboard_log = "logs/phase4_tensorboard"

    # 既存モデルがあれば継続訓練
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} to continue training...")
        model = PPO.load(model_path, env=env, verbose=1, tensorboard_log=tensorboard_log)
    else:
        print("Creating NEW model for Phase 4 Speed Learning")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_log,
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
    print("  Max steps per episode: 4,000")
    print("  Maze regeneration: every 20 episodes")
    print("\nReward Structure (Speed-Focused):")
    print("  - Goal reached: +500.0")
    print("  - Intermediate cell: +100.0")
    print("  - Speed reward: +3.0 * velocity")
    print("  - Time penalty: -0.5 per step")
    print("  - Collision penalty: -100.0")
    print("  - Smoothness penalty: -0.3*|Δv| - 0.2*|Δω|")
    print("\nObservation Space (10 dimensions):")
    print("  - Distance sensors: 4")
    print("  - Velocities: 2")
    print("  - Goal info: 2")
    print("  - Front openness: 1")
    print("  - Side balance: 1")

    print("\n" + "=" * 60)
    print("Starting training for 1,000,000 timesteps...")
    print("=" * 60)

    # チェックポイントコールバック
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./logs/',
        name_prefix='ppo_phase4_speed'
    )

    model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    print("\nTraining complete.")

    # モデルを保存
    model.save(model_path)
    print(f"Final model saved to {model_path}")

    # 出力を生成
    print(f"\nGenerating outputs...")

    # 学習曲線をプロット
    learning_curve_path = output_mgr.get_path("learning_curve.png")
    plot_learning_curve("logs/", str(learning_curve_path), window=50)

    # 評価動画を記録
    print("\nRecording evaluation videos...")
    eval_env = MicromouseSpeedEnv(
        render_mode="rgb_array",
        maze_size=(7, 7),
        spawn_area_size=5,
        max_steps=4000
    )

    video_path = output_mgr.get_path("evaluation.mp4")
    record_video(eval_env, model, str(video_path), num_episodes=5, deterministic=True, fps=60)

    # 性能評価
    print("\n" + "=" * 60)
    print("Performance Evaluation (10 episodes)")
    print("=" * 60)

    success_count = 0
    intermediate_count = 0
    total_steps = 0
    total_reward = 0.0
    total_avg_speed = 0.0
    total_max_speed = 0.0

    for episode in range(10):
        obs, info = eval_env.reset()
        episode_reward = 0.0
        episode_steps = 0
        speeds = []

        for step in range(4000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

            episode_reward += reward
            episode_steps += 1
            speeds.append(info.get("actual_lin", 0.0))

            if terminated or truncated:
                break

        # 速度統計
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0
        total_avg_speed += avg_speed
        total_max_speed += max_speed

        # 成功チェック
        if eval_env.intermediate_reached:
            intermediate_count += 1

        if episode_reward > 400:  # ゴール到達
            success_count += 1
            print(f"Episode {episode + 1}: ✓ SUCCESS "
                  f"(steps={episode_steps}, reward={episode_reward:.1f}, "
                  f"avg_speed={avg_speed:.3f}m/s, max_speed={max_speed:.3f}m/s)")
        elif eval_env.intermediate_reached:
            print(f"Episode {episode + 1}: △ INTERMEDIATE "
                  f"(steps={episode_steps}, reward={episode_reward:.1f}, "
                  f"avg_speed={avg_speed:.3f}m/s)")
        else:
            print(f"Episode {episode + 1}: ✗ FAILED "
                  f"(steps={episode_steps}, reward={episode_reward:.1f}, "
                  f"avg_speed={avg_speed:.3f}m/s)")

        total_steps += episode_steps
        total_reward += episode_reward

    eval_env.close()

    # 最終結果
    success_rate = success_count / 10.0
    intermediate_rate = intermediate_count / 10.0
    avg_steps = total_steps / 10.0
    avg_reward = total_reward / 10.0
    avg_avg_speed = total_avg_speed / 10.0
    avg_max_speed = total_max_speed / 10.0

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Success Rate: {success_count}/10 ({success_count * 10}%)")
    print(f"Intermediate Rate: {intermediate_count}/10 ({intermediate_count * 10}%)")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"Average Speed: {avg_avg_speed:.3f} m/s")
    print(f"Average Max Speed: {avg_max_speed:.3f} m/s")

    # モデル情報を保存
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
        "Max Episode Steps": 4000,
        "Observation Dimensions": 10,
        "Action Dimensions": 2,
    })

    # メトリクスを保存
    output_mgr.save_metrics(
        metrics={
            "training_steps": 1000000,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
        },
        phase_specific={
            "intermediate_rate": intermediate_rate,
            "avg_speed": avg_avg_speed,
            "max_speed": avg_max_speed,
            "maze_size": "7x7",
            "spawn_area": "5x5",
        }
    )

    # 出力を完了
    output_mgr.finalize(
        summary=f"Training complete!\n"
                f"Success: {success_count}/10 ({success_rate*100:.0f}%)\n"
                f"Avg Steps: {avg_steps:.1f}\n"
                f"Avg Speed: {avg_avg_speed:.3f} m/s\n"
                f"Model: {model_path}"
    )


if __name__ == "__main__":
    main()
