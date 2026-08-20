# research_notes/scripts/video_kinematics/_common.py
# 発表用解説動画クリップ第1弾の共通描画設定（配色・フォント・保持フレーム・字幕）。
#
# 対象クリップ: clip_01_opening / clip_02_friction_circle / clip_06_sweep_clearance /
# clip_07_diagonal / clip_08_velocity_profile / clip_09_eta（すべて本ディレクトリ配下）。
#
# 🔴 数値は必ず classic/profile.py（vehicle_limits/min_time）・classic/geometry.py
# から計算した値を描く。本ファイル自身には物理定数・幾何定数は一切置かない
# （置いてよいのは配色・レイアウト・フレームレートなどの純粋な描画上の値のみ）。
#
# 出力: 1920x1080・30fps・末尾1.5秒静止保持の mp4（H.264, yuv420p）。
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import font_manager

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "outputs" / "video_kinematics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 配色（図解 Artifact と揃える。暗い背景。ユーザ指定の 5 色のみを使う）
# ----------------------------------------------------------------------------
BG = "#0F1113"          # 背景
FG = "#E7E8E3"          # 文字
GRID = "#2C3033"        # 補助線
C_LIMIT = "#4BCBD1"     # 物理限界（青緑）
C_MEASURED = "#FF7A6A"  # 実測（赤）
C_CAP_COST = "#E2AC42"  # 速度上限の代償（黄）

# ----------------------------------------------------------------------------
# 動画仕様
# ----------------------------------------------------------------------------
WIDTH_PX = 1920
HEIGHT_PX = 1080
DPI = 120
FIG_W = WIDTH_PX / DPI
FIG_H = HEIGHT_PX / DPI
FPS = 30
HOLD_SECONDS = 1.5
HOLD_FRAMES = round(FPS * HOLD_SECONDS)


def setup_style() -> None:
    """日本語フォント（Hiragino Sans。実在確認済み）と暗い背景の既定スタイルを設定する。"""
    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["text.color"] = FG
    plt.rcParams["axes.edgecolor"] = GRID
    plt.rcParams["axes.labelcolor"] = FG
    plt.rcParams["xtick.color"] = FG
    plt.rcParams["ytick.color"] = FG
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = BG
    plt.rcParams["savefig.facecolor"] = BG


def new_figure():
    """暗い背景の 1920x1080 図を 1 枚返す（軸は各クリップ側で作る）。"""
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
    return fig


def style_axes(ax) -> None:
    """暗い背景に合わせて軸の見た目を整える共通処理。"""
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.6)


def add_title(fig, text: str, y: float = 0.95, fontsize: int = 30):
    return fig.text(0.5, y, text, ha="center", va="top", color=FG,
                     fontsize=fontsize, fontweight="bold")


def add_caption(fig, text: str, y: float = 0.055, fontsize: int = 22):
    """字幕（画面下端の常設キャプション）。戻り値の Text は `set_alpha` 等で
    段階的な出現に使える（返り値を無視すれば従来どおり常設として使える）。"""
    return fig.text(0.5, y, text, ha="center", va="center", color=FG, fontsize=fontsize,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="#1A1D1F",
                               edgecolor=GRID, linewidth=1.0))


def add_stat_panel(fig, lines: Sequence[str], x: float = 0.985, y: float = 0.90,
                    fontsize: int = 18, ha: str = "right"):
    """右上（既定）などに置く数値パネル。等幅フォントで揃える。
    戻り値の Text は `set_alpha` 等で段階的な出現に使える。"""
    # 等幅英数字（Menlo）と日本語（Hiragino Sans）が混在するため、Menlo 指定はしない
    # （Menlo は CJK グリフを持たず欠落警告が出る）。既定の Hiragino Sans に揃える。
    text = "\n".join(lines)
    return fig.text(x, y, text, ha=ha, va="top", color=FG, fontsize=fontsize,
                     linespacing=1.6,
                     bbox=dict(boxstyle="round,pad=0.6", facecolor="#1A1D1F",
                               edgecolor=GRID, linewidth=1.0))


def make_writer(fps: int = FPS) -> animation.FFMpegWriter:
    return animation.FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"],
    )


def render_clip(fig, draw_frame: Callable[[int], Iterable], n_active_frames: int,
                 out_path: Path, hold_frames: int = HOLD_FRAMES) -> Path:
    """`draw_frame(i)` を `i=0..n_active_frames-1` で呼びながら mp4 を書き出す。

    末尾は最後の有効フレーム（`n_active_frames-1`）の状態を `hold_frames` 枚
    複製して静止保持する（`HOLD_SECONDS` 秒ぶん）。
    """
    total = n_active_frames + hold_frames

    def _update(i: int):
        j = min(i, n_active_frames - 1)
        return draw_frame(j)

    writer = make_writer()
    with writer.saving(fig, str(out_path), dpi=DPI):
        for i in range(total):
            _update(i)
            writer.grab_frame(facecolor=BG)
    return out_path


def seconds_to_active_frames(total_seconds: float, hold_frames: int = HOLD_FRAMES,
                              fps: int = FPS) -> int:
    """クリップ全体の目標秒数から、保持を除いた可動フレーム数を求める。"""
    return max(round(total_seconds * fps) - hold_frames, 1)


def stage_bounds(weights: Sequence[float], n_active: int) -> list[int]:
    """重み（台本の文の文字数など）に比例して `n_active` を区切った境界フレーム番号。

    台本の文の切れ目で画面の要素が現れるようにするための土台。戻り値は
    `len(weights)+1` 個（先頭 0・末尾 `n_active`）で、`weights[k]` に対応する
    区間は `[bounds[k], bounds[k+1])`。文字数はナレーションの読み上げ時間に
    ほぼ比例するという単純な近似であり、音声のタイムスタンプそのものではない
    （台本の文単位で「だいたい合わせる」ための土台。フレーム単位の同期はしない）。
    """
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights の合計が 0 以下")
    bounds = [0]
    cum = 0.0
    for w in weights[:-1]:
        cum += w
        bounds.append(min(round(n_active * cum / total), n_active))
    bounds.append(n_active)
    return bounds
