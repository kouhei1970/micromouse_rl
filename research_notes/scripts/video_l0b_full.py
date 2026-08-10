# L0-b（古典ベースライン StraightRunPolicy・超信地旋回走行「直進連続」方式）の
# 新16x16評価迷路（規定準拠版）フル録画
#
# Full-length, real-time (1x), no-skip recording of the classical L0-b baseline
# (StraightRunPolicy, flood-fill + 超信地旋回走行「直進連続」) navigating the
# same NTF-compliant 16x16 evaluation maze (maze_1015) as L0-a, through all
# 5 runs, no skipping — including failed runs (stuck 等) if they occur.
#
# レンダリング本体は _video_l0_common.py（L0-a と共通）を使う。本ファイルは
# 方策・ラベル・出力先・補足キャプションの指定のみ。
#
# 用語統一: 走行方式は「超信地旋回走行（直進連続）」と呼ぶ（区画境界では
# 減速せず走り抜け、曲がる手前でのみ停止して超信地旋回する。詳細は
# competition/baseline_straightrun.py docstring 参照）。
#
# v_max について: StraightRunPolicy の既定 v_max=0.6 m/s は、壁検出→停止の
# 物理制約による上限 0.857 m/s（同ファイル docstring 参照）よりもさらに
# 保守的な整定値（PD制御則のゲインがL0-aの低速向けのまま流用されており、
# 物理上限近くまで詰めると位置決めが収束しない stuck が頻発するため）。
# 「速度は物理上限まで詰めていない暫定版」である旨を画面に明記する
# （2026-08-10 教授指示）。
#
# 既知の注意事項: L0-b は maze_1015 の探索走行（第1走行）で PD 制御則の
# チャタリングにより stuck することが分かっている（competition/results/ の
# 事前調査で確認）。本スクリプトは stuck が起きてもそれを省略せず記録する
# （_video_l0_common.py の失敗走行処理: 確定タイム欄に赤字で明記、既知壁
# 地図の軌跡も赤で描画）。
#
# 使い方 / Usage:
#   .venv/bin/python research_notes/scripts/video_l0b_full.py
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

from pathlib import Path  # noqa: E402

from competition.baseline_straightrun import StraightRunPolicy  # noqa: E402
from research_notes.scripts._video_l0_common import (  # noqa: E402
    record_run_video, FPS, MAX_RUNS, TIME_BUDGET_S,
)

# L0-a と同一の maze_1015 を使う（3方式比較のため面を統一。2026-08-10 教授指示）。
SELECTED_MAZE_ID = "maze_1015"
OUT_VIDEO_PATH = Path("outputs/videos/l0b_full.mp4")
METHOD_LABEL = "L0-b 超信地旋回走行・直進連続"
EXTRA_CAPTION = ["速度は物理上限まで詰めていない暫定版（v_max 0.6 m/s、物理上限 0.857 m/s）"]


def main():
    parser = argparse.ArgumentParser(description="L0-b 新16x16評価迷路 フル録画（5走行・省略なし・等倍速）")
    parser.add_argument("--maze-id", type=str, default=SELECTED_MAZE_ID,
                         help=f"録画対象の迷路ID（既定: {SELECTED_MAZE_ID}）")
    args = parser.parse_args()

    policy = StraightRunPolicy()

    print("=" * 70)
    print(f"{METHOD_LABEL}（StraightRunPolicy） 新16x16評価迷路 フル録画（全5走行）")
    print("=" * 70)
    result = record_run_video(policy, METHOD_LABEL, OUT_VIDEO_PATH, args.maze_id,
                               extra_caption=EXTRA_CAPTION,
                               max_runs=MAX_RUNS, time_budget=TIME_BUDGET_S,
                               v_max_for_graph=0.6)

    print("\n--- 公式 evaluate_maze() の走行記録（本動画のタイム表示の正） ---")
    for r in result["official_runs"]:
        print(f"  走行{r['index']}: {r['outcome']:9s} t_start={r['t_start']:7.2f}s "
              f"t_end={r['t_end']:7.2f}s run_time={r['run_time']:6.2f}s")
    print(f"  best_time = {result['official_best_time']}")

    print("\n--- コールバック計測値との整合確認 ---")
    for label, rt, is_fail in result["confirmed_times"]:
        print(f"  {label}: {rt:.2f} s" + ("  [失敗]" if is_fail else ""))
    if result["best"]:
        print(f"  最速: {result['best'][0]:.2f} s（第{result['best'][1]}走行）")
    print(f"  探索走行（最初にゴールした走行）: 第{result['explore_run_index']}走行")

    print(f"\n書き出しフレーム数: {result['n_frames_written']} "
          f"（{result['n_frames_written'] / FPS:.1f} s @ {FPS}fps）")
    print(f"出力: {result['out_path'].resolve()}")
    print(f"実処理時間（wall clock）: {result['wall_clock_s']:.1f} s")


if __name__ == "__main__":
    main()
