"""
スライド用のデモ動画・画像を生成するスクリプト
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import imageio

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
    """Phase 3（迷路ナビゲーション）のデモを生成 - ゴール到達まで撮影"""
    from stable_baselines3 import PPO
    from phase3_maze.env import MicromouseMazeEnv

    print("\nPhase 3 デモ生成中...")

    env = MicromouseMazeEnv(render_mode="rgb_array", maze_size=(7, 7), spawn_area_size=5)
    model = PPO.load("models/phase3_maze.zip")

    # 成功するエピソードを探す
    for seed in range(20):
        frames = []
        goal_frame = None

        obs, _ = env.reset(seed=seed)

        for i in range(1500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if terminated:
                if info.get('dist_to_goal', 1) < 0.1:
                    goal_frame = frame.copy()
                    print(f"  seed={seed}: ゴール到達！ step={i}")

                    # スクリーンショット保存
                    imageio.imwrite("docs/images/phase3_screenshot.png", goal_frame)
                    print("  スクリーンショット保存（ゴール到達時）")

                    # 動画保存
                    imageio.mimsave("docs/images/phase3_demo.mp4", frames, fps=60)
                    print(f"  動画保存: {len(frames)}フレーム（{len(frames)/60:.1f}秒）")

                    env.close()
                    return len(frames)
                break

    env.close()
    print("  警告: ゴール到達エピソードが見つかりませんでした")
    return 0


def main():
    os.makedirs("docs/images", exist_ok=True)

    print("=" * 50)
    print("スライド用デモコンテンツ生成")
    print("=" * 50)

    try:
        n_frames = generate_phase1_demo()
        print(f"  → Phase 1: {n_frames}フレーム生成完了")
    except Exception as e:
        print(f"  Phase 1 エラー: {e}")

    try:
        n_frames = generate_phase2_demo()
        print(f"  → Phase 2: {n_frames}フレーム生成完了")
    except Exception as e:
        print(f"  Phase 2 エラー: {e}")

    try:
        n_frames = generate_phase3_demo()
        print(f"  → Phase 3: {n_frames}フレーム生成完了")
    except Exception as e:
        print(f"  Phase 3 エラー: {e}")

    print("\n" + "=" * 50)
    print("生成完了！")
    for f in sorted(os.listdir("docs/images")):
        size = os.path.getsize(f"docs/images/{f}") / 1024
        print(f"  - {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
