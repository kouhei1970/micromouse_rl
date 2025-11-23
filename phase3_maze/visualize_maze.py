"""
Visualize maze structure with start, goal, and initial orientation.
"""
import sys
import os
sys.path.append(os.getcwd())

from phase3_maze.env import MicromouseMazeEnv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_maze_episode(env, ax, episode_num):
    """
    Visualize a single maze episode with walls, start, goal, and orientation.
    """
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-0.02, env.maze_width * 0.18 + 0.02)
    ax.set_ylim(-0.02, env.maze_height * 0.18 + 0.02)
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)

    cell_size = 0.18
    wall_thickness = 0.012

    # Draw grid cells (light gray for spawn area, very light gray for outer area)
    for x in range(env.maze_width):
        for y in range(env.maze_height):
            # Check if in spawn area
            in_spawn_area = (env.spawn_offset <= x < env.spawn_offset + env.spawn_area_size and
                           env.spawn_offset <= y < env.spawn_offset + env.spawn_area_size)

            facecolor = 'white' if in_spawn_area else 'whitesmoke'
            rect = patches.Rectangle(
                (x * cell_size, y * cell_size),
                cell_size, cell_size,
                linewidth=0.5, edgecolor='lightgray', facecolor=facecolor
            )
            ax.add_patch(rect)

            # Add cell coordinates (smaller font for 7x7)
            if env.maze_width <= 5:
                fontsize = 7
            else:
                fontsize = 5
            ax.text(x * cell_size + cell_size/2, y * cell_size + cell_size/2,
                   f'({x},{y})', ha='center', va='center', fontsize=fontsize, color='lightgray')

    # Draw vertical walls (red)
    for x in range(env.maze_width + 1):
        for y in range(env.maze_height):
            if env.generator.v_walls[x, y] == 1:
                wall_x = x * cell_size - wall_thickness / 2
                wall_y = y * cell_size
                wall = patches.Rectangle(
                    (wall_x, wall_y),
                    wall_thickness, cell_size,
                    linewidth=0, edgecolor='none', facecolor='red'
                )
                ax.add_patch(wall)

    # Draw horizontal walls (red)
    for x in range(env.maze_width):
        for y in range(env.maze_height + 1):
            if env.generator.h_walls[x, y] == 1:
                wall_x = x * cell_size
                wall_y = y * cell_size - wall_thickness / 2
                wall = patches.Rectangle(
                    (wall_x, wall_y),
                    cell_size, wall_thickness,
                    linewidth=0, edgecolor='none', facecolor='red'
                )
                ax.add_patch(wall)

    # Highlight Start Cell (blue)
    start_x, start_y = env.start_cell
    start_rect = patches.Rectangle(
        (start_x * cell_size, start_y * cell_size),
        cell_size, cell_size,
        linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.5
    )
    ax.add_patch(start_rect)

    # Highlight Intermediate Cell (yellow) if exists
    if env.intermediate_cell is not None:
        int_x, int_y = env.intermediate_cell
        int_rect = patches.Rectangle(
            (int_x * cell_size, int_y * cell_size),
            cell_size, cell_size,
            linewidth=2, edgecolor='orange', facecolor='yellow', alpha=0.3
        )
        ax.add_patch(int_rect)

    # Highlight Goal Cell (green)
    goal_x, goal_y = env.goal_cell
    goal_rect = patches.Rectangle(
        (goal_x * cell_size, goal_y * cell_size),
        cell_size, cell_size,
        linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.5
    )
    ax.add_patch(goal_rect)

    # Draw robot position and orientation
    robot_pos = env.data.qpos[0:2]
    q = env.data.qpos[3:7]
    w, x, y, z = q
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    # Robot position (circle)
    robot_circle = patches.Circle(
        (robot_pos[0], robot_pos[1]),
        0.02,  # 2cm radius
        color='darkblue', zorder=10
    )
    ax.add_patch(robot_circle)

    # Robot orientation (arrow)
    arrow_length = 0.05
    dx = arrow_length * np.cos(yaw)
    dy = arrow_length * np.sin(yaw)
    ax.arrow(robot_pos[0], robot_pos[1], dx, dy,
            head_width=0.02, head_length=0.015, fc='darkblue', ec='darkblue', zorder=11)

    # Direction name
    yaw_deg = np.degrees(yaw)
    direction_names = {0: "East", 90: "North", 180: "West", -90: "South"}
    # Find closest direction
    closest_dir = min(direction_names.keys(), key=lambda k: abs(k - yaw_deg))
    direction_name = direction_names[closest_dir]

    # Title
    title = f'Episode {episode_num}: Start={env.start_cell} ({direction_name})\n'
    if env.intermediate_cell:
        title += f'Intermediate={env.intermediate_cell}, '
    title += f'Goal={env.goal_cell}'
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', label='Start Cell'),
        Patch(facecolor='yellow', edgecolor='orange', label='Intermediate'),
        Patch(facecolor='lightgreen', edgecolor='green', label='Goal Cell'),
        Patch(facecolor='white', edgecolor='lightgray', label='Spawn Area'),
        Patch(facecolor='red', edgecolor='red', label='Wall'),
        patches.Circle((0, 0), 0.02, color='darkblue', label='Robot')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)

def main():
    print("=" * 60)
    print("Maze Visualization for Phase 3 - 7x7 Maze")
    print("Spawn area: Inner 5x5 cells")
    print("=" * 60)

    # Create environment with 7x7 maze
    env = MicromouseMazeEnv(render_mode=None, maze_size=(7, 7), spawn_area_size=5, maze_regen_interval=5)

    # Create figure with subplots
    num_episodes = 6
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes = axes.flatten()

    for i in range(num_episodes):
        print(f"\nGenerating Episode {i+1}/{num_episodes}...")
        obs, info = env.reset()

        visualize_maze_episode(env, axes[i], env.episode_count)

    plt.tight_layout()

    # Save figure
    output_dir = "outputs/phase3_maze"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "maze_visualization_7x7.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")

    # Also show
    plt.show()

    env.close()

    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
