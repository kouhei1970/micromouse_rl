# L0-a（古典ベースライン AdachiPolicy・超信地旋回走行「区画ごと停止」方式）の
# 新16x16評価迷路（規定準拠版）フル録画
#
# Full-length, real-time (1x), no-skip recording of the classical L0-a baseline
# (AdachiPolicy, flood-fill + 超信地旋回走行「区画ごと停止」) navigating a new
# NTF-compliant 16x16 evaluation maze through all 5 runs, no skipping.
#
# レンダリング本体（1920x1080レイアウト・evaluator完全一致タイミング取得・
# 失敗走行の記録等）は _video_l0_common.py に共通化されている
# （L0-b・L0-c もこれを使う）。本ファイルは方策・ラベル・出力先の指定のみ。
#
# 用語統一: 走行方式は「超信地旋回走行（区画ごと停止）」と呼ぶ（英語の
# "stop-and-go" は使わない）。
#
# 使い方 / Usage:
#   .venv/bin/python research_notes/scripts/video_l0_new_maze_full.py
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

from pathlib import Path  # noqa: E402

from competition.baseline_classical import AdachiPolicy  # noqa: E402
from research_notes.scripts._video_l0_common import (  # noqa: E402
    record_run_video, FPS, MAX_RUNS, TIME_BUDGET_S,
)

# 面の選定根拠: competition/mazes/eval/ の20面（規定準拠・maze_gen_v2生成）を
# L0-a + 420秒プロトコルで全面走査した結果、maze_1015（seed=1015）が
#   探索走行(第1走行) 82.73s -> 最短走行(第4走行) 25.30s（短縮率 65.6〜69.4%、
#   実行毎の物理エンジンの数値非決定性で±0.数秒変動。本ファイル末尾の
#   run_and_record docstring 相当は _video_l0_common.record_run_video 側で
#   evaluate_maze() の実測値をそのまま表示するため常に自己無矛盾）
# と20面中で最大の短縮率、かつ全5走行が衝突・スタック・転倒による係員回収
# なしで成立（incidents=[]）だったため選定。次点は maze_1017（55.3%）・
# maze_1004（55.0%）・maze_1000（45.3%）。L0-b・L0-c でも同一の maze_1015 を
# 使う（3方式比較のため面を統一）。
SELECTED_MAZE_ID = "maze_1015"
OUT_VIDEO_PATH = Path("outputs/videos/l0a_full.mp4")
METHOD_LABEL = "L0-a 超信地旋回走行・区画ごと停止"


def main():
    parser = argparse.ArgumentParser(description="L0-a 新16x16評価迷路 フル録画（5走行・省略なし・等倍速）")
    parser.add_argument("--maze-id", type=str, default=SELECTED_MAZE_ID,
                         help=f"録画対象の迷路ID（既定: {SELECTED_MAZE_ID}）")
    args = parser.parse_args()

    print("=" * 70)
    print(f"{METHOD_LABEL}（AdachiPolicy） 新16x16評価迷路 フル録画（全5走行）")
    print("=" * 70)
    result = record_run_video(AdachiPolicy(), METHOD_LABEL, OUT_VIDEO_PATH, args.maze_id,
                               max_runs=MAX_RUNS, time_budget=TIME_BUDGET_S, v_max_for_graph=0.3)

    print("\n--- 公式 evaluate_maze() の走行記録（本動画のタイム表示の正） ---")
    for r in result["official_runs"]:
        print(f"  走行{r['index']}: {r['outcome']:9s} t_start={r['t_start']:7.2f}s "
              f"t_end={r['t_end']:7.2f}s run_time={r['run_time']:6.2f}s")
    print(f"  best_time = {result['official_best_time']:.2f} s")

    print("\n--- コールバック計測値との整合確認 ---")
    for label, rt, is_fail in result["confirmed_times"]:
        print(f"  {label}: {rt:.2f} s" + ("  [失敗]" if is_fail else ""))
    if result["best"]:
        print(f"  最速: {result['best'][0]:.2f} s（第{result['best'][1]}走行）")

    print(f"\n書き出しフレーム数: {result['n_frames_written']} "
          f"（{result['n_frames_written'] / FPS:.1f} s @ {FPS}fps）")
    print(f"出力: {result['out_path'].resolve()}")
    print(f"実処理時間（wall clock）: {result['wall_clock_s']:.1f} s")


if __name__ == "__main__":
    main()
