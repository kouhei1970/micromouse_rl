"""
スライド用のデモ動画・画像を生成するスクリプト
Phase3はMuJoCoネイティブマーカーを使用してゴール位置を表示
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import imageio
import cv2
import mujoco

from common.demo_xml_builder import (
    generate_demo_xml,
    update_marker_colors,
    cleanup_demo_xml
)


class TopDownRenderer:
    """真上からのカメラでレンダリング"""

    def __init__(self, model, data, width=800, height=800, maze_size=7):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)

        # カメラ設定（真上から）
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        # 迷路の中心
        maze_center = maze_size * 0.18 / 2
        self.camera.lookat[:] = [maze_center, maze_center, 0]
        self.camera.distance = 2.0  # カメラ距離
        self.camera.azimuth = 90    # 方位角
        self.camera.elevation = -90  # 真上から

    def render(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def draw_info_overlay(frame, step, episode, start_cell, goal_cell,
                      intermediate_reached, goal_reached):
    """情報オーバーレイを描画"""
    h, w = frame.shape[:2]

    # 半透明の背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 120), (0, 0, 0), -1)
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
        (status, status_color)
    ]

    for i, (text, color) in enumerate(texts):
        cv2.putText(frame, text, (20, 32 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


def generate_phase1_demo():
    """Phase 1（速度制御）のデモを生成 - 複数シードの動きのある部分を連結"""
    from stable_baselines3 import PPO
    from phase1_open.env import MicromouseEnv

    print("Phase 1 デモ生成中...")

    env = MicromouseEnv(render_mode="rgb_array", xml_file="assets/micromouse_open.xml")
    model = PPO.load("models/phase1_open.zip")

    all_dynamic_segments = []  # 各シードの動きのあるセグメント
    best_frame = None
    best_score = 0

    seeds = [42, 123, 456, 789, 1000, 2000, 3000]

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        frames = []
        scores = []

        for i in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            frame = env.render()
            if frame is not None:
                frames.append(frame)

                # 動きのスコア計算
                angular_vel = abs(obs[5])
                linear_vel = abs(obs[4])
                score = angular_vel * 0.5 + linear_vel * 0.5
                scores.append(score)

                # ベストフレーム更新
                if score > best_score and angular_vel > 0.3:
                    best_score = score
                    best_frame = frame.copy()

            if terminated or truncated:
                obs, _ = env.reset()

        # このシードで最も動きのある60フレーム（1秒）を抽出
        if len(scores) >= 60:
            scores_arr = np.array(scores)
            # 移動平均でスムージング
            window = 30
            smoothed = np.convolve(scores_arr, np.ones(window)/window, mode='valid')
            best_start = np.argmax(smoothed)
            segment = frames[best_start:best_start+60]
            if len(segment) == 60:
                avg_score = np.mean(scores[best_start:best_start+60])
                all_dynamic_segments.append((avg_score, segment))
                print(f"  seed={seed}: 動きのあるセグメント抽出（平均スコア={avg_score:.2f}）")

    env.close()

    # スコアの高い順に並べて上位4つを連結
    all_dynamic_segments.sort(key=lambda x: x[0], reverse=True)
    combined_frames = []
    for i, (score, segment) in enumerate(all_dynamic_segments[:4]):
        combined_frames.extend(segment)
        print(f"  セグメント{i+1}: スコア={score:.2f}")

    # スクリーンショット保存
    if best_frame is not None:
        imageio.imwrite("docs/images/phase1_screenshot.png", best_frame)
        print(f"  スクリーンショット保存（スコア={best_score:.2f}）")

    # 動画保存
    if combined_frames:
        imageio.mimsave("docs/images/phase1_demo.mp4", combined_frames, fps=60)
        print(f"  動画保存: {len(combined_frames)}フレーム（{len(combined_frames)/60:.1f}秒）")

    return len(combined_frames)


def generate_phase2_demo():
    """Phase 2（スラローム）のデモを生成 - ゴール到達まで撮影"""
    from stable_baselines3 import PPO
    from phase2_slalom.env import MicromouseSlalomEnv

    print("\nPhase 2 デモ生成中...")

    env = MicromouseSlalomEnv(render_mode="rgb_array")
    model = PPO.load("models/phase2_slalom.zip")

    frames = []
    goal_frame = None
    turning_frame = None
    best_turn_score = 0

    obs, _ = env.reset(seed=42)

    # ゴール到達まで撮影（最大1000フレーム）
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

            # 曲がっている瞬間を探す
            angular_vel = abs(obs[5])
            linear_vel = abs(obs[4])
            turn_score = angular_vel + linear_vel * 0.3

            if turn_score > best_turn_score and i > 100:
                best_turn_score = turn_score
                turning_frame = frame.copy()

        if terminated:
            # ゴール到達時のフレームを保存
            if info.get('dist_to_goal', 1) < 0.1:
                goal_frame = frame.copy()
                print(f"  ゴール到達！ step={i}")
            break

    env.close()

    # スクリーンショット：ゴール到達時 > 旋回中
    if goal_frame is not None:
        imageio.imwrite("docs/images/phase2_screenshot.png", goal_frame)
        print("  スクリーンショット保存（ゴール到達時）")
    elif turning_frame is not None:
        imageio.imwrite("docs/images/phase2_screenshot.png", turning_frame)
        print(f"  スクリーンショット保存（旋回中、スコア={best_turn_score:.2f}）")

    if frames:
        imageio.mimsave("docs/images/phase2_demo.mp4", frames, fps=60)
        print(f"  動画保存: {len(frames)}フレーム（{len(frames)/60:.1f}秒）")

    return len(frames)


def generate_phase3_demo():
    """Phase 3（迷路ナビゲーション）のデモを生成 - MuJoCoネイティブマーカー版"""
    from stable_baselines3 import PPO
    from phase3_maze.env import MicromouseMazeEnv

    print("\nPhase 3 デモ生成中（MuJoCoマーカー版）...")

    env = MicromouseMazeEnv(render_mode=None, maze_size=(7, 7), spawn_area_size=5)
    model = PPO.load("models/phase3_maze.zip")

    demo_xml_path = None
    renderer = None

    # 成功するエピソードを探す
    for seed in range(20):
        frames = []
        goal_frame = None
        goal_reached = False

        obs, _ = env.reset(seed=seed)

        # 以前のデモXMLをクリーンアップ
        if demo_xml_path:
            cleanup_demo_xml(demo_xml_path)
        if renderer is not None:
            renderer.close()
            renderer = None

        # マーカー付きデモ用XMLを生成
        demo_xml_path = generate_demo_xml(
            source_xml_path=env.xml_file,
            goal_cell=env.goal_cell,
            intermediate_cell=env.intermediate_cell,
            cell_size=0.18
        )

        # デモ用モデルをロード
        demo_model = mujoco.MjModel.from_xml_path(demo_xml_path)
        demo_data = mujoco.MjData(demo_model)

        # 環境からデモモデルに状態をコピー
        demo_data.qpos[:] = env.data.qpos[:]
        demo_data.qvel[:] = env.data.qvel[:]
        mujoco.mj_forward(demo_model, demo_data)

        # レンダラーを作成
        renderer = TopDownRenderer(demo_model, demo_data, maze_size=7)

        for i in range(1500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # 環境の状態をデモデータに同期
            demo_data.qpos[:] = env.data.qpos[:]
            demo_data.qvel[:] = env.data.qvel[:]
            mujoco.mj_forward(demo_model, demo_data)

            # ゴール判定
            if info.get('dist_to_goal', 1) < 0.1:
                goal_reached = True

            # マーカー色を動的に更新
            intermediate_reached = getattr(env, 'intermediate_reached', False)
            update_marker_colors(
                demo_model,
                goal_reached=goal_reached,
                intermediate_reached=intermediate_reached
            )

            # フレームをレンダリング
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 情報オーバーレイ
            frame_bgr = draw_info_overlay(
                frame_bgr, i, seed + 1,
                env.start_cell, env.goal_cell,
                intermediate_reached, goal_reached
            )

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            if terminated:
                if goal_reached:
                    goal_frame = frame_rgb.copy()
                    print(f"  seed={seed}: ゴール到達！ step={i}")

                    # 終了フレームを追加
                    for _ in range(45):
                        frames.append(frame_rgb)

                    # スクリーンショット保存
                    imageio.imwrite("docs/images/phase3_screenshot.png", goal_frame)
                    print("  スクリーンショット保存（ゴール到達時）")

                    # 動画保存
                    imageio.mimsave("docs/images/phase3_demo.mp4", frames, fps=60)
                    print(f"  動画保存: {len(frames)}フレーム（{len(frames)/60:.1f}秒）")

                    # クリーンアップ
                    if renderer is not None:
                        renderer.close()
                    if demo_xml_path:
                        cleanup_demo_xml(demo_xml_path)
                    env.close()
                    return len(frames)
                break

    # クリーンアップ
    if renderer is not None:
        renderer.close()
    if demo_xml_path:
        cleanup_demo_xml(demo_xml_path)
    env.close()
    print("  警告: ゴール到達エピソードが見つかりませんでした")
    return 0


def generate_phase4_demo():
    """Phase 4（高速走行）のデモを生成"""
    from phase4_speed.create_demo_video import create_demo_video

    print("\nPhase 4 デモ生成中（MuJoCoマーカー版）...")

    output_path = create_demo_video(
        model_path="models/phase4_speed.zip",
        output_path="docs/images/phase4_demo.mp4",
        num_episodes=3,
        max_steps=800
    )

    # スクリーンショットを動画から抽出
    import subprocess
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", output_path,
            "-vframes", "1", "-q:v", "2",
            "docs/images/phase4_screenshot.png"
        ], capture_output=True, check=True)
        print("  スクリーンショット保存")
    except Exception as e:
        print(f"  スクリーンショット生成エラー: {e}")

    return 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                        help="生成するフェーズを指定（省略時は全フェーズ）")
    args = parser.parse_args()

    os.makedirs("docs/images", exist_ok=True)

    print("=" * 50)
    print("スライド用デモコンテンツ生成")
    print("=" * 50)

    phases_to_generate = [args.phase] if args.phase else [1, 2, 3, 4]

    for phase in phases_to_generate:
        try:
            if phase == 1:
                n_frames = generate_phase1_demo()
                print(f"  → Phase 1: {n_frames}フレーム生成完了")
            elif phase == 2:
                n_frames = generate_phase2_demo()
                print(f"  → Phase 2: {n_frames}フレーム生成完了")
            elif phase == 3:
                n_frames = generate_phase3_demo()
                print(f"  → Phase 3: {n_frames}フレーム生成完了")
            elif phase == 4:
                generate_phase4_demo()
                print("  → Phase 4: 生成完了")
        except Exception as e:
            print(f"  Phase {phase} エラー: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("生成完了！")
    for f in sorted(os.listdir("docs/images")):
        size = os.path.getsize(f"docs/images/{f}") / 1024
        print(f"  - {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
