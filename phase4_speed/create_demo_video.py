"""
Phase 4: ゴールマーカー付きデモ動画の作成
MuJoCoのトップダウンカメラで撮影し、ゴール位置を赤いマーカーで表示
"""
import os
import sys
import cv2
import numpy as np
import imageio
import mujoco

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from phase4_speed.env import MicromouseSpeedEnv


class TopDownRenderer:
    """真上からのカメラでレンダリング"""

    def __init__(self, model, data, width=800, height=800):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)

        # カメラ設定（真上から）
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        # 7x7迷路の中心 (0.63, 0.63)
        maze_center = 7 * 0.18 / 2
        self.camera.lookat[:] = [maze_center, maze_center, 0]
        self.camera.distance = 2.0  # カメラ距離
        self.camera.azimuth = 90    # 方位角
        self.camera.elevation = -90  # 真上から

    def render(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def world_to_pixel_topdown(world_pos, maze_size, img_size):
    """ワールド座標をピクセル座標に変換（真上視点）"""
    maze_extent = maze_size * 0.18
    # ワールド座標 [0, maze_extent] → ピクセル座標 [0, img_size]
    # カメラが中心を見ているので、適切にオフセット
    margin = 0.15  # カメラの余白
    scale = img_size[0] / (maze_extent + 2 * margin)

    px = int((world_pos[0] + margin) * scale)
    py = int(img_size[1] - (world_pos[1] + margin) * scale)  # Y軸反転

    return (px, py)


def draw_markers(frame, goal_pos, intermediate_pos, robot_pos, maze_size=7,
                 intermediate_reached=False, goal_reached=False):
    """フレームにマーカーを描画"""
    h, w = frame.shape[:2]

    # ゴールマーカー（赤）
    goal_px = world_to_pixel_topdown(goal_pos, maze_size, (w, h))
    color_goal = (0, 255, 0) if goal_reached else (0, 0, 255)  # 到達したら緑
    cv2.circle(frame, goal_px, 20, color_goal, 3)
    cv2.circle(frame, goal_px, 6, color_goal, -1)
    label = "GOAL!" if goal_reached else "GOAL"
    cv2.putText(frame, label, (goal_px[0] - 25, goal_px[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_goal, 2)

    # 中間マーカー（オレンジ）
    if intermediate_pos is not None:
        int_px = world_to_pixel_topdown(intermediate_pos, maze_size, (w, h))
        color_int = (0, 255, 0) if intermediate_reached else (0, 165, 255)
        cv2.circle(frame, int_px, 12, color_int, 2)

    # スタート位置への矢印（任意で追加可能）

    return frame


def draw_info_overlay(frame, step, speed, goal_dist, episode, total_reward,
                      start_cell, goal_cell, intermediate_reached, goal_reached):
    """情報オーバーレイを描画"""
    h, w = frame.shape[:2]

    # 半透明の背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # ステータス
    if goal_reached:
        status = "GOAL REACHED!"
        status_color = (0, 255, 0)
    elif intermediate_reached:
        status = "Intermediate OK"
        status_color = (0, 165, 255)
    else:
        status = "Navigating..."
        status_color = (255, 255, 255)

    # テキスト情報
    texts = [
        (f"Episode: {episode}", (255, 255, 255)),
        (f"Start: {start_cell} -> Goal: {goal_cell}", (255, 255, 255)),
        (f"Step: {step}", (255, 255, 255)),
        (f"Speed: {speed:.3f} m/s", (255, 255, 255)),
        (f"Goal Dist: {goal_dist:.3f} m", (255, 255, 255)),
        (f"Reward: {total_reward:.1f}", (255, 255, 255)),
        (status, status_color)
    ]

    for i, (text, color) in enumerate(texts):
        cv2.putText(frame, text, (20, 32 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


def set_robot_orientation_toward_target(env, target_pos):
    """ロボットの向きをターゲット方向に設定"""
    robot_pos = env.data.qpos[0:2]
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    angle = np.arctan2(dy, dx)  # ターゲットへの角度

    # Quaternion設定 (Z軸回転)
    env.data.qpos[3] = np.cos(angle / 2)
    env.data.qpos[4] = 0
    env.data.qpos[5] = 0
    env.data.qpos[6] = np.sin(angle / 2)

    # 向きの名前を返す
    angle_deg = np.degrees(angle) % 360
    if 315 <= angle_deg or angle_deg < 45:
        return "East"
    elif 45 <= angle_deg < 135:
        return "North"
    elif 135 <= angle_deg < 225:
        return "West"
    else:
        return "South"


def create_demo_video(model_path, output_path, num_episodes=3, max_steps=1000,
                      face_target=True):
    """ゴールマーカー付きデモ動画を作成

    Args:
        face_target: Trueの場合、スポーン時に中間目標を向く
    """

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # 環境を作成（レンダリングなしで初期化）
    env = MicromouseSpeedEnv(
        render_mode=None,  # 自前でレンダリング
        maze_size=(7, 7),
        spawn_area_size=5,
        max_steps=max_steps
    )

    frames = []
    renderer = None

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        goal_reached = False

        # デモ時は中間目標を向くように設定
        if face_target and env.intermediate_cell is not None:
            target_pos = np.array([
                env.intermediate_cell[0] * 0.18 + 0.09,
                env.intermediate_cell[1] * 0.18 + 0.09
            ])
            direction = set_robot_orientation_toward_target(env, target_pos)
            # 観測を更新
            obs = env._get_obs()
        else:
            direction = "Random"

        # レンダラーを作成/更新
        if renderer is not None:
            renderer.close()
        renderer = TopDownRenderer(env.model, env.data)

        # ゴール位置と中間位置
        goal_pos = env.goal_pos
        intermediate_pos = None
        if env.intermediate_cell is not None:
            intermediate_pos = np.array([
                env.intermediate_cell[0] * 0.18 + 0.09,
                env.intermediate_cell[1] * 0.18 + 0.09
            ])

        print(f"\nEpisode {episode + 1}:")
        print(f"  Start: {env.start_cell} ({direction})")
        print(f"  Goal: {env.goal_cell}")
        print(f"  Intermediate: {env.intermediate_cell}")

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # 正しいゴール判定: 距離が閾値以下
            if info['dist_to_goal'] < 0.05:
                goal_reached = True

            # フレームをレンダリング
            frame = renderer.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # ロボット位置
            robot_pos = env.data.qpos[0:2]

            # マーカーを描画
            frame = draw_markers(
                frame, goal_pos, intermediate_pos, robot_pos,
                maze_size=7,
                intermediate_reached=env.intermediate_reached,
                goal_reached=goal_reached
            )

            # 情報オーバーレイ
            frame = draw_info_overlay(
                frame, step, info['actual_lin'], info['dist_to_goal'],
                episode + 1, total_reward,
                env.start_cell, env.goal_cell,
                env.intermediate_reached, goal_reached
            )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if terminated or truncated:
                # 終了フレームを追加
                for _ in range(45):  # 0.75秒
                    frames.append(frame)
                break

        result = "SUCCESS" if goal_reached else "FAILED"
        print(f"  Result: {result} (reward={total_reward:.1f}, steps={step+1})")

    if renderer is not None:
        renderer.close()
    env.close()

    # 動画を保存
    print(f"\nSaving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=60)
    print(f"Video saved! ({len(frames)} frames, {len(frames)/60:.1f}s)")

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/phase4_speed.zip")
    parser.add_argument("--output", default="outputs/phase4_speed/demo_with_goal.mp4")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=800)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_demo_video(args.model, args.output, args.episodes, args.max_steps)


if __name__ == "__main__":
    main()
