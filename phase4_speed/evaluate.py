"""
Phase 4: Speed Learning Evaluation
訓練済みモデルの評価と速度プロファイル分析
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from phase4_speed.env import MicromouseSpeedEnv
from common.visualization import record_video
from common.output_manager import OutputManager


def evaluate_model(model, env, num_episodes=10, verbose=True):
    """モデルを評価して統計を返す"""
    results = {
        "success_count": 0,
        "intermediate_count": 0,
        "collision_count": 0,
        "total_steps": 0,
        "total_reward": 0.0,
        "speeds": [],
        "episode_speeds": [],
        "episode_steps": [],
        "episode_rewards": []
    }

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        speeds = []

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_steps += 1
            speeds.append(info.get("actual_lin", 0.0))

            if terminated or truncated:
                break

        # 統計を収集
        results["speeds"].extend(speeds)
        results["episode_speeds"].append(speeds)
        results["episode_steps"].append(episode_steps)
        results["episode_rewards"].append(episode_reward)
        results["total_steps"] += episode_steps
        results["total_reward"] += episode_reward

        # 成功判定
        if episode_reward > 400:
            results["success_count"] += 1
            status = "✓ SUCCESS"
        elif env.intermediate_reached:
            results["intermediate_count"] += 1
            status = "△ INTERMEDIATE"
        else:
            status = "✗ FAILED"

        if episode_reward < 0:
            results["collision_count"] += 1

        if verbose:
            avg_speed = np.mean(speeds) if speeds else 0.0
            max_speed = np.max(speeds) if speeds else 0.0
            print(f"Episode {episode + 1}: {status} "
                  f"(steps={episode_steps}, reward={episode_reward:.1f}, "
                  f"avg_speed={avg_speed:.3f}m/s, max_speed={max_speed:.3f}m/s)")

    return results


def plot_speed_profile(episode_speeds, output_path):
    """速度プロファイルをプロット"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 全エピソードの速度時系列
    ax1 = axes[0, 0]
    for i, speeds in enumerate(episode_speeds[:5]):  # 最初の5エピソード
        ax1.plot(speeds, alpha=0.7, label=f"Episode {i+1}")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Linear Velocity (m/s)")
    ax1.set_title("Speed Profile (First 5 Episodes)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 速度分布ヒストグラム
    ax2 = axes[0, 1]
    all_speeds = [s for ep in episode_speeds for s in ep]
    ax2.hist(all_speeds, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(all_speeds), color='r', linestyle='--',
                label=f"Mean: {np.mean(all_speeds):.3f}")
    ax2.axvline(np.median(all_speeds), color='g', linestyle='--',
                label=f"Median: {np.median(all_speeds):.3f}")
    ax2.set_xlabel("Linear Velocity (m/s)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Speed Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. エピソード別の平均速度
    ax3 = axes[1, 0]
    avg_speeds = [np.mean(ep) for ep in episode_speeds]
    max_speeds = [np.max(ep) for ep in episode_speeds]
    x = range(1, len(episode_speeds) + 1)
    ax3.bar(x, avg_speeds, alpha=0.7, label="Average")
    ax3.scatter(x, max_speeds, color='r', marker='*', s=100, label="Max", zorder=5)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Speed (m/s)")
    ax3.set_title("Average and Max Speed per Episode")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 速度変化の滑らかさ
    ax4 = axes[1, 1]
    for i, speeds in enumerate(episode_speeds[:5]):
        if len(speeds) > 1:
            speed_changes = np.abs(np.diff(speeds))
            ax4.plot(speed_changes, alpha=0.7, label=f"Episode {i+1}")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("|Δv| (m/s)")
    ax4.set_title("Speed Change (Smoothness)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Speed profile saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 Speed Learning Model")
    parser.add_argument("--model", type=str, default="models/phase4_speed.zip",
                        help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--video", action="store_true",
                        help="Record evaluation video")
    parser.add_argument("--render", action="store_true",
                        help="Render in human mode")
    args = parser.parse_args()

    # モデルをロード
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        print("Please train the model first with: python phase4_speed/train.py")
        return

    print("=" * 60)
    print("Phase 4: Speed Learning Evaluation")
    print("=" * 60)

    model = PPO.load(args.model)
    print(f"Model loaded from {args.model}")

    # 環境を作成
    render_mode = "human" if args.render else None
    env = MicromouseSpeedEnv(
        render_mode=render_mode,
        maze_size=(7, 7),
        spawn_area_size=5,
        max_steps=4000
    )

    # 評価を実行
    print(f"\nEvaluating {args.episodes} episodes...")
    print("-" * 60)

    results = evaluate_model(model, env, num_episodes=args.episodes)

    # 結果を表示
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    success_rate = results["success_count"] / args.episodes
    intermediate_rate = results["intermediate_count"] / args.episodes
    collision_rate = results["collision_count"] / args.episodes
    avg_steps = results["total_steps"] / args.episodes
    avg_reward = results["total_reward"] / args.episodes
    all_speeds = results["speeds"]

    print(f"Success Rate: {results['success_count']}/{args.episodes} "
          f"({success_rate*100:.1f}%)")
    print(f"Intermediate Rate: {results['intermediate_count']}/{args.episodes} "
          f"({intermediate_rate*100:.1f}%)")
    print(f"Collision Rate: {results['collision_count']}/{args.episodes} "
          f"({collision_rate*100:.1f}%)")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"\nSpeed Statistics:")
    print(f"  Mean Speed: {np.mean(all_speeds):.3f} m/s")
    print(f"  Max Speed: {np.max(all_speeds):.3f} m/s")
    print(f"  Min Speed: {np.min(all_speeds):.3f} m/s")
    print(f"  Std Dev: {np.std(all_speeds):.3f} m/s")

    # 出力ディレクトリを作成
    output_dir = "outputs/phase4_speed/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    # 速度プロファイルをプロット
    plot_speed_profile(
        results["episode_speeds"],
        os.path.join(output_dir, "speed_profile.png")
    )

    # 動画を記録
    if args.video:
        print("\nRecording evaluation video...")
        video_env = MicromouseSpeedEnv(
            render_mode="rgb_array",
            maze_size=(7, 7),
            spawn_area_size=5,
            max_steps=4000
        )
        video_path = os.path.join(output_dir, "evaluation.mp4")
        record_video(video_env, model, video_path, num_episodes=3,
                     deterministic=True, fps=60)
        video_env.close()

    env.close()
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
