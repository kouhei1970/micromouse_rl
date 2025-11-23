"""
Analyze Phase 3 Final Training Results
Create comprehensive visualizations of the trained model's performance.
"""
import sys
import os
sys.path.append(os.getcwd())

from phase3_maze.env import MicromouseMazeEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

def analyze_performance():
    """Analyze trained model performance with detailed visualizations"""

    print("=" * 60)
    print("Phase 3 Final Results Analysis")
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

    # Run detailed analysis episodes
    num_episodes = 20
    results = []

    print(f"\nRunning {num_episodes} evaluation episodes...")

    for episode in range(num_episodes):
        obs, info = env.reset()

        episode_data = {
            'start': env.start_cell,
            'intermediate': env.intermediate_cell,
            'goal': env.goal_cell,
            'steps': 0,
            'total_reward': 0.0,
            'intermediate_reached': False,
            'goal_reached': False,
            'proper_orientation': False,
            'trajectory': []
        }

        for step in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Store position
            pos = env.data.qpos[0:2].copy()
            episode_data['trajectory'].append(pos)

            episode_data['steps'] += 1
            episode_data['total_reward'] += reward

            if env.intermediate_reached:
                episode_data['intermediate_reached'] = True

            if terminated:
                if reward > 1000:  # Goal with orientation
                    episode_data['goal_reached'] = True
                    episode_data['proper_orientation'] = True
                elif reward > 500:  # Goal without orientation
                    episode_data['goal_reached'] = True
                break

            if truncated:
                break

        results.append(episode_data)

        status = "✓ SUCCESS" if episode_data['goal_reached'] else "✗ FAILED"
        orientation = "(proper orientation)" if episode_data['proper_orientation'] else ""
        print(f"Episode {episode + 1}: {status} {orientation}")
        print(f"  Start={episode_data['start']}, Goal={episode_data['goal']}")
        print(f"  Steps={episode_data['steps']}, Reward={episode_data['total_reward']:.1f}")

    env.close()

    # Statistics
    success_rate = sum(1 for r in results if r['goal_reached']) / len(results) * 100
    orientation_rate = sum(1 for r in results if r['proper_orientation']) / len(results) * 100
    intermediate_rate = sum(1 for r in results if r['intermediate_reached']) / len(results) * 100

    successful_episodes = [r for r in results if r['goal_reached']]
    avg_steps = np.mean([r['steps'] for r in successful_episodes]) if successful_episodes else 0
    avg_reward = np.mean([r['total_reward'] for r in results])

    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"Success Rate: {success_rate:.1f}% ({sum(1 for r in results if r['goal_reached'])}/{len(results)})")
    print(f"Proper Orientation Rate: {orientation_rate:.1f}%")
    print(f"Intermediate Rate: {intermediate_rate:.1f}%")
    print(f"Avg Steps (successful): {avg_steps:.1f}")
    print(f"Avg Total Reward: {avg_reward:.1f}")

    # Create comprehensive visualization
    create_comprehensive_visualization(results, success_rate, orientation_rate, intermediate_rate)

    return results

def create_comprehensive_visualization(results, success_rate, orientation_rate, intermediate_rate):
    """Create comprehensive performance visualization"""

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)

    # 1. Success/Failure/Orientation breakdown
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Success\n(proper orient)', 'Success\n(poor orient)', 'Intermediate\nOnly', 'Failed']
    counts = [
        sum(1 for r in results if r['proper_orientation']),
        sum(1 for r in results if r['goal_reached'] and not r['proper_orientation']),
        sum(1 for r in results if r['intermediate_reached'] and not r['goal_reached']),
        sum(1 for r in results if not r['intermediate_reached'])
    ]
    colors = ['green', 'yellowgreen', 'orange', 'red']
    bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=10)
    ax1.set_title('Episode Outcomes', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    # 2. Reward distribution
    ax2 = fig.add_subplot(gs[0, 1])
    rewards = [r['total_reward'] for r in results]
    ax2.hist(rewards, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
    ax2.set_xlabel('Total Reward', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Reward Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Steps distribution (successful episodes only)
    ax3 = fig.add_subplot(gs[0, 2])
    successful_steps = [r['steps'] for r in results if r['goal_reached']]
    if successful_steps:
        ax3.hist(successful_steps, bins=15, color='coral', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(successful_steps), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(successful_steps):.0f}')
        ax3.set_xlabel('Episode Steps', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('Episode Length (Successful)', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

    # 4. Success rate pie chart
    ax4 = fig.add_subplot(gs[0, 3])
    pie_data = [success_rate, 100 - success_rate]
    pie_colors = ['green', 'lightgray']
    wedges, texts, autotexts = ax4.pie(pie_data, labels=['Success', 'Failed'],
                                        colors=pie_colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax4.set_title('Success Rate', fontsize=11, fontweight='bold')

    # 5-12. Sample trajectory visualizations (4 success, 4 failure/intermediate)
    success_results = [r for r in results if r['goal_reached']][:4]
    failure_results = [r for r in results if not r['goal_reached']][:4]

    for idx, result in enumerate(success_results):
        ax = fig.add_subplot(gs[1 + idx // 2, idx % 2])
        orient_text = " (Proper Orient)" if result['proper_orientation'] else ""
        visualize_trajectory(ax, result, f"Success {idx + 1}{orient_text}")

    for idx, result in enumerate(failure_results):
        ax = fig.add_subplot(gs[2 + idx // 2, idx % 2])
        status = "Intermediate Only" if result['intermediate_reached'] else "Failed"
        visualize_trajectory(ax, result, f"{status} {idx + 1}")

    # Overall title
    fig.suptitle(f'Phase 3 Final Training Results (1M timesteps)\n' +
                f'Success: {success_rate:.1f}% | Proper Orientation: {orientation_rate:.1f}% | Intermediate: {intermediate_rate:.1f}%',
                fontsize=16, fontweight='bold')

    # Save figure
    output_dir = "outputs/phase3_maze"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "final_performance_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFinal performance analysis saved to {output_path}")
    plt.close()

def visualize_trajectory(ax, result, title):
    """Visualize single episode trajectory"""

    # Draw 7x7 grid
    cell_size = 0.18
    for x in range(7):
        for y in range(7):
            rect = patches.Rectangle(
                (x * cell_size, y * cell_size),
                cell_size, cell_size,
                linewidth=1, edgecolor='gray', facecolor='white', alpha=0.3
            )
            ax.add_patch(rect)

    # Highlight cells
    start_x, start_y = result['start']
    start_rect = patches.Rectangle(
        (start_x * cell_size, start_y * cell_size),
        cell_size, cell_size,
        linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.5
    )
    ax.add_patch(start_rect)

    if result['intermediate']:
        int_x, int_y = result['intermediate']
        int_rect = patches.Rectangle(
            (int_x * cell_size, int_y * cell_size),
            cell_size, cell_size,
            linewidth=2, edgecolor='orange', facecolor='yellow', alpha=0.3
        )
        ax.add_patch(int_rect)

    goal_x, goal_y = result['goal']
    goal_rect = patches.Rectangle(
        (goal_x * cell_size, goal_y * cell_size),
        cell_size, cell_size,
        linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.5
    )
    ax.add_patch(goal_rect)

    # Plot trajectory
    if len(result['trajectory']) > 0:
        trajectory = np.array(result['trajectory'])
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1, alpha=0.6)

        # Start and end markers
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'bo', markersize=6)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'rx', markersize=8, markeredgewidth=2)

    ax.set_xlim(-0.02, 7 * cell_size + 0.02)
    ax.set_ylim(-0.02, 7 * cell_size + 0.02)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=7)
    ax.set_ylabel('Y (m)', fontsize=7)
    ax.set_title(f"{title}\nSteps: {result['steps']}, Reward: {result['total_reward']:.0f}",
                fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.2)

if __name__ == "__main__":
    analyze_performance()
