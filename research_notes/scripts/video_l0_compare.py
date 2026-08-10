# 3方式（L0-a / L0-b / L0-c）の同時比較動画
#
# 同一迷路・同一持ち時間（420秒）で3方式を**横並びに同時再生**し、走り方の違いを
# 一目で比較できるようにする。持ち時間の時計は3方式で共通（同じ秒数の場面が
# 横に並ぶ）。先に全走行を終えた方式は、最後の姿勢のまま自分のタイムを表示して待つ。
#
# 実装方針: MuJoCo の再レンダリングはしない。既に撮ってある 1920x1080 の各方式の
# フル動画から**迷路の描画部分（左 1080x1080）だけを切り出して 640x640 に縮小**し、
# 3枚を横に並べる。下段の情報パネルだけを毎フレーム新しく描く。
# 各フル動画は等倍速 30fps で「フレーム番号 i ↔ シミュレーション時刻 i/30 秒」の
# 対応になっているため、同じフレーム番号を取り出せば時刻が自動的に揃う。
#
# 使い方 / Usage:
#   .venv/bin/python research_notes/scripts/video_l0_compare.py
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from competition.evaluator import CompetitionEvaluator  # noqa: E402
from research_notes.scripts._video_l0_common import (  # noqa: E402
    EVAL_MAZE_DIR, FPS, MAX_RUNS, TIME_BUDGET_S, _load_font,
    PANEL_BG, SEC_BG_ALT, TEXT_WHITE, TEXT_DIM, TEXT_ACCENT,
)

ACCENT = TEXT_ACCENT
GOOD = (52, 199, 89, 255)     # 最速タイム（緑）
BAD = (255, 69, 58, 255)      # 失敗走行（赤）

OUT_W, OUT_H = 1920, 1080
CELL_W = OUT_W // 3          # 640: 1方式ぶんの横幅
HEADER_H = 56                # 最上段: 3方式共通の持ち時間の時計
VIEW_H = CELL_W              # 640: 迷路ビュー（正方形なので横幅と同じ）
VIEW_Y = HEADER_H            # 迷路ビューの上端
INFO_Y = HEADER_H + VIEW_H   # 696: 下段の情報パネルの上端
INFO_H = OUT_H - INFO_Y      # 384
SRC_MAZE_SIZE = 1080         # 元動画の迷路描画領域（左端の正方形）
# 元動画の迷路描画は 1080x1080 の中央に余白つきで写っている。迷路の外周だけを
# 残して切り出すことで、3面に縮小しても機体と壁が見える大きさを確保する
# （縮小率が 1080->640 なので、余白まで含めて縮めると機体が潰れる）。
SRC_CROP0, SRC_CROP1 = 100, 985
HOLD_TAIL_S = 3.0            # 全方式終了後の余韻

MAZE_ID = "maze_1015"
OUT_PATH = Path("outputs/videos/l0_abc_compare.mp4")

METHODS = [
    ("l0a", "L0-a 超信地旋回走行", "区画ごと停止", "outputs/videos/l0a_full.mp4"),
    ("l0b", "L0-b 超信地旋回走行", "直進連続", "outputs/videos/l0b_full.mp4"),
    ("l0c", "L0-c スラローム走行", "曲がるところでも停止しない", "outputs/videos/l0c_full.mp4"),
]

_OUTCOME_JA = {"goal": "ゴール", "collision": "衝突", "stuck": "スタック",
                "timeout": "時間切れ", "fell": "転倒"}


def _policy_for(key: str):
    if key == "l0a":
        from competition.baseline_classical import AdachiPolicy
        return AdachiPolicy()
    if key == "l0b":
        from competition.baseline_straightrun import StraightRunPolicy
        return StraightRunPolicy()
    from competition.baseline_slalom import SlalomPolicy
    return SlalomPolicy()


def collect_runs(key: str, maze_id: str):
    """描画せずに評価だけ回し、走行の開始/終了時刻と結果を取る。

    下段パネルは「いま何走行目か・その走行の経過秒・それまでの最速」しか使わない
    ので、各走行の (t_start, t_end, outcome, run_time) さえあれば毎フレームの表示を
    復元できる。動画そのものと同じ凍結ハーネスを同じ設定で回すので、表示される
    タイムは公式記録と一致する。"""
    npz = EVAL_MAZE_DIR / f"{maze_id}.npz"
    ev = CompetitionEvaluator(maze_dir=str(EVAL_MAZE_DIR),
                              time_budget=TIME_BUDGET_S, max_runs=MAX_RUNS)
    res = ev.evaluate_maze(npz, _policy_for(key))
    return res["runs"], res["best_time"]


def status_at(runs, t):
    """時刻 t における (状態文字列, 進行中の走行番号 or None, 経過秒, それまでの最速, 確定表示行)"""
    lines = []
    best = None
    active = None
    elapsed = 0.0
    seen_explore = False   # 最初に**ゴールできた**走行が探索走行（走行番号ではない）
    for r in runs:
        if r["t_end"] <= t:
            oc = r["outcome"]
            if oc == "goal":
                lines.append((f"第{r['index']}走行  {r['run_time']:.2f} s", False))
                # 探索走行（最初のゴール到達）は最速の対象外。第1走行が衝突・
                # スタックで終わった場合は第2走行が探索走行になるので、走行番号で
                # 判定してはいけない（凍結ハーネス evaluator.py と同じ規約）。
                if not seen_explore:
                    seen_explore = True
                elif best is None or r["run_time"] < best:
                    best = r["run_time"]
            else:
                lines.append((f"第{r['index']}走行  {_OUTCOME_JA.get(oc, oc)}・係員回収", True))
        elif r["t_start"] <= t < r["t_end"]:
            active = r["index"]
            elapsed = t - r["t_start"]
    if active is not None:
        state = f"第{active}走行 走行中  {elapsed:5.2f} s" if active > 1 \
            else f"第1走行（探索中）  {elapsed:5.2f} s"
    elif not runs or t < runs[0]["t_start"]:
        state = "スタート待機中"
    elif t >= runs[-1]["t_end"]:
        state = "全走行終了"
    else:
        state = "帰還中"
    return state, best, lines


def recent_event(runs, t, window=1.6):
    """直近 window 秒以内に終わった走行があれば ("goal"|"fail", run_time, 残り強調時間)。

    ゴールした瞬間が視覚的に分かるよう、迷路ビューの枠を光らせるために使う
    （教授助言 2026-08-10: 比較動画としての訴求力を上げる）。"""
    for r in runs:
        dt = t - r["t_end"]
        if 0.0 <= dt < window:
            kind = "goal" if r["outcome"] == "goal" else "fail"
            return kind, r["run_time"], 1.0 - dt / window
    return None, 0.0, 0.0


def render_info(draw, x0, name, subtitle, state, best, lines, fonts, finished):
    """1方式ぶんの下段情報を描く（左上を (x0, INFO_Y) とする 640x384 の領域）。"""
    f_title, f_body, f_small = fonts
    y = INFO_Y + 12
    draw.text((x0 + 20, y), name, font=f_title, fill=ACCENT)
    y += 34
    draw.text((x0 + 20, y), subtitle, font=f_small, fill=TEXT_DIM)
    y += 30
    draw.text((x0 + 20, y), state, font=f_body,
              fill=TEXT_DIM if finished else TEXT_WHITE)
    y += 36
    for txt, is_fail in lines[-MAX_RUNS:]:
        draw.text((x0 + 28, y), txt, font=f_small, fill=BAD if is_fail else TEXT_WHITE)
        y += 26
    if best is not None:
        draw.text((x0 + 20, OUT_H - 50), f"最速 {best:.2f} s", font=f_title, fill=GOOD)


def load_or_collect_runs(cache_path: Path, refresh: bool):
    """走行記録を取り直す（refresh）か、キャッシュから読む。

    走行記録の取得は3方式ぶんの評価を実際に回すので数分かかる。動画の合成だけを
    試したいとき（レイアウトの確認・スモークテスト）に毎回回さずに済むよう
    キャッシュする。**方策を変更したら必ず --refresh で取り直すこと**
    （古い記録のまま合成すると、映像と表示タイムが食い違う）。"""
    import json
    if (not refresh) and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  走行記録をキャッシュから読み込み: {cache_path}")
        return {k: v["runs"] for k, v in data.items()}
    out, runs_by_method = {}, {}
    for key, name, _sub, _path in METHODS:
        runs, best = collect_runs(key, MAZE_ID)
        runs_by_method[key] = runs
        out[key] = {"runs": runs, "best_time": best}
        print(f"  {name}: 走行数={len(runs)} 最速={best if best is None else round(best, 2)} s "
              f"outcomes={[r['outcome'] for r in runs]}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return runs_by_method


def main():
    import argparse
    ap = argparse.ArgumentParser(description="3方式 同時比較動画")
    ap.add_argument("--max-frames", type=int, default=0,
                     help="書き出すフレーム数の上限（0=全部）。スモークテスト用")
    ap.add_argument("--refresh", action="store_true",
                     help="走行記録をキャッシュせず取り直す（方策を変更したら必須）")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()
    t_wall = time.time()
    print("=" * 70)
    print("3方式 同時比較動画（同一迷路・同一持ち時間）")
    print("=" * 70)

    for _k, _n, _s, path in METHODS:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} が見つかりません。先に各方式のフル動画を撮ってください: "
                f".venv/bin/python research_notes/scripts/video_l0_new_maze_full.py --method <l0a|l0b|l0c>")

    # --- 各方式の走行記録（描画なしの評価で取得。動画のタイムと同じ凍結ハーネス） ---
    runs_by_method = load_or_collect_runs(
        Path(f"outputs/analysis/compare_runs_{MAZE_ID}.json"), args.refresh)

    readers = {key: imageio.get_reader(path) for key, _n, _s, path in METHODS}
    n_frames = {key: readers[key].count_frames() for key in readers}
    print(f"  各動画のフレーム数: {n_frames}")
    total = max(n_frames.values()) + int(HOLD_TAIL_S * FPS)
    if args.max_frames:
        total = min(total, args.max_frames)
        print(f"  [スモークテスト] {total} フレームで打ち切ります")
    out_path = Path(args.out)

    fonts = (_load_font(28), _load_font(24), _load_font(20))
    f_clock = _load_font(34)
    last_view = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=FPS, macro_block_size=1)
    print(f"  書き出し: {out_path.resolve()}  総フレーム {total}")

    # 元動画は先頭から順に読むので、フレーム番号指定のランダムアクセス
    # （get_data(i)）ではなく**逐次読み出し**にする。ランダムアクセスは
    # 1フレームごとに mp4 のシークが入り、後半になるほど遅くなる。
    iters = {key: readers[key].iter_data() for key in readers}

    for i in range(total):
        t = i / FPS
        canvas = Image.new("RGB", (OUT_W, OUT_H), PANEL_BG[:3])
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, INFO_Y, OUT_W, OUT_H], fill=SEC_BG_ALT[:3])

        for col, (key, name, sub, _path) in enumerate(METHODS):
            x0 = col * CELL_W
            # --- 迷路ビュー: 元動画の迷路部分を切り出して 640x640 に縮小 ---
            if i < n_frames[key]:
                frame = next(iters[key])
                view = Image.fromarray(
                    frame[SRC_CROP0:SRC_CROP1, SRC_CROP0:SRC_CROP1]
                ).resize((CELL_W, VIEW_H), Image.LANCZOS)
                last_view[key] = view
            else:
                view = last_view[key]      # 終わった方式は最後の姿勢のまま静止
            canvas.paste(view, (x0, VIEW_Y))

            # --- ゴール/失敗した瞬間の強調（枠を光らせ、タイムを大きく出す） ---
            kind, rt, strength = recent_event(runs_by_method[key], t)
            if kind is not None:
                ecol = GOOD[:3] if kind == "goal" else BAD[:3]
                w = max(2, int(10 * strength))
                draw.rectangle([x0 + 1, VIEW_Y + 1, x0 + CELL_W - 2, VIEW_Y + VIEW_H - 2],
                               outline=ecol, width=w)
                tag = f"ゴール {rt:.2f} s" if kind == "goal" else "係員回収"
                bb = draw.textbbox((0, 0), tag, font=f_clock)
                tx = x0 + (CELL_W - (bb[2] - bb[0])) // 2
                draw.rectangle([tx - 14, VIEW_Y + VIEW_H - 76, tx + (bb[2] - bb[0]) + 14,
                                VIEW_Y + VIEW_H - 22], fill=(0, 0, 0))
                draw.text((tx, VIEW_Y + VIEW_H - 72), tag, font=f_clock, fill=ecol)

            state, best, lines = status_at(runs_by_method[key], t)
            render_info(draw, x0, name, sub, state, best, lines, fonts,
                        finished=(i >= n_frames[key]))
            if col > 0:
                draw.line([(x0, VIEW_Y), (x0, OUT_H)], fill=(70, 74, 82), width=2)

        draw.line([(0, INFO_Y), (OUT_W, INFO_Y)], fill=(70, 74, 82), width=2)
        clock = f"持ち時間  {min(t, TIME_BUDGET_S):6.1f} / {TIME_BUDGET_S:.0f} s"
        bbox = draw.textbbox((0, 0), clock, font=f_clock)
        draw.text(((OUT_W - (bbox[2] - bbox[0])) // 2, 8), clock, font=f_clock, fill=TEXT_WHITE)

        writer.append_data(np.array(canvas))
        if i % 900 == 0:
            print(f"    ... {i}/{total} フレーム  (t={t:6.1f}s)  "
                  f"経過(実時間)={time.time() - t_wall:6.1f}s", flush=True)

    writer.close()
    for r in readers.values():
        r.close()
    print(f"\n完了: {out_path.resolve()}  {total} フレーム "
          f"({total / FPS:.1f} s @ {FPS}fps)  実処理時間 {time.time() - t_wall:.1f} s")


if __name__ == "__main__":
    main()
