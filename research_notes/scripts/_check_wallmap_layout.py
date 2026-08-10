# research_notes/scripts/_check_wallmap_layout.py
# _video_l0_common.py の WallMapPanel（SEC1: 既知壁地図パネル）レイアウトを、
# 物理シミュレーションを回さずに単体で確認するスクリプト。
#
# 背景: SEC1 の凡例は従来「地図の下」に置かれていたが、幅840pxのSEC1に対し
# 地図は330x330しか使っておらず左右に大きな余白を持て余していた
# （ユーザ指摘）。加えて凡例の項目数が5に増えたときにSEC1の枠（高さ400px）
# をはみ出し、下のSEC2の方式名テキストと重なる不具合があった。
# これを是正し、凡例を地図横の余白（右優先・不足時は左）へ縦積みで移し、
# 地図を正方形のままSEC1の高さいっぱいまで拡大した
# （WallMapPanel.render / _plan_legend_columns / _force_fit_legend、
# 2026-08-10）。
#
# 本スクリプトは、凡例の項目数を 3/4/5/6/8 個に変えた5通りでSEC1パネル
# （840x400）を描画し、それぞれについて
#   (1) パネル画像の最上行・最下行・最左列・最右列に背景色以外の画素が
#       乗っていないか（= SEC1 の矩形からのはみ出し）
#   (2) render() が内部で実際に使う _plan_legend_columns()/_force_fit_legend()
#       の計算結果を使って各凡例項目のスウォッチ中心画素を実際にサンプリング
#       し、期待した色で描かれているか（= 項目が欠落・SEC1外へクリップされて
#       いないか）
# を自動判定し、PASS/FAIL を標準出力する（目視だけに頼らない）。
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from research_notes.scripts import _video_l0_common as vc  # noqa: E402

OUT_DIR = _REPO / "outputs" / "analysis"
MAZE_NPZ = _REPO / "competition" / "mazes" / "eval" / "maze_1015.npz"

# 凡例に使うダミー項目のプール（実プロジェクトの配色をそのまま使い、
# 全て不透明(alpha=255)にして画素比較を単純にする。実装で使う実際の
# 5項目 [探索走行/帰還/探索より後/その時点の最速/失敗] に、項目数を
# 6・8まで増やすための追加ダミー2項目を足した8項目プール）。
LEGEND_POOL = [
    (vc.COLOR_EXPLORE, "探索走行"),
    (vc.COLOR_RETURN, "帰還"),
    (vc.COLOR_LATER_RUN, "探索より後"),
    (vc.COLOR_BEST_RUN, "その時点の最速"),
    (vc.COLOR_FAILED_RUN, "失敗（係員回収）"),
    (vc.COLOR_ROBOT_DOT, "現在位置マーカー（ダミー項目6）"),
    (vc.COLOR_WALL_KNOWN, "既知の壁（ダミー項目7）"),
    (vc.TEXT_ACCENT, "テスト項目8（ダミー）"),
]


def _load_dummy_maze():
    data = np.load(MAZE_NPZ)
    return int(data["width"]), int(data["height"]), data["v_walls"], data["h_walls"]


def _build_dummy_wallmap(width, height, cell_size, v_walls, h_walls):
    """maze_1015 の壁配置を下敷きに、既知/未知が入り混じったダミーの
    壁知識状態と、いくつかのダミー走行軌跡（探索走行・帰還・最速走行・
    失敗走行）を持つ WallMapPanel を作る。"""
    wall_map = vc.WallMapPanel(width, height, cell_size)

    rng = np.random.default_rng(0)
    v_known = np.where(v_walls > 0, 1, np.where(rng.random(v_walls.shape) < 0.5, -1, 0)).astype(np.int8)
    h_known = np.where(h_walls > 0, 1, np.where(rng.random(h_walls.shape) < 0.5, -1, 0)).astype(np.int8)

    # 第1走行（探索）: 往路
    for gx, gy in [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2)]:
        wall_map.add_point("active", 1, (gx + 0.5) * cell_size, (gy + 0.5) * cell_size)
    # 帰還区間
    for gx, gy in [(3, 2), (2, 2), (1, 1), (0, 0)]:
        wall_map.add_point("return", 1, (gx + 0.5) * cell_size, (gy + 0.5) * cell_size)
    # 第2走行（その時点の最速）
    for gx, gy in [(0, 0), (1, 1), (2, 2), (width - 2, height - 2)]:
        wall_map.add_point("active", 2, (gx + 0.5) * cell_size, (gy + 0.5) * cell_size)
    # 第3走行（失敗・係員回収）
    wall_map.add_point("active", 3, 0.5 * cell_size, 0.5 * cell_size)
    wall_map.add_point("active", 3, 1.5 * cell_size, 0.5 * cell_size)
    wall_map.mark_failed(3)

    return wall_map, v_known, h_known


def _check_borders(img: Image.Image, bg) -> list:
    """パネル画像の最上行・最下行・最左列・最右列に背景色以外の画素が
    乗っていないかを調べる。違反があれば [(x, y, pixel), ...] を返す
    （空リスト = 合格）。"""
    w, h = img.size
    px = img.load()
    bad = []
    for x in range(w):
        for y in (0, h - 1):
            if px[x, y] != bg:
                bad.append((x, y, px[x, y]))
    for y in range(h):
        for x in (0, w - 1):
            if px[x, y] != bg:
                bad.append((x, y, px[x, y]))
    return bad


def _recompute_legend_plan(wall_map, legend_items):
    """render() 内部で実際に使われる _plan_legend_columns() /
    _force_fit_legend() をそのまま呼び出し、side・plan を再現する
    （レイアウト計算ロジックを重複実装しない = render() の実装と
    常に一致することを保証する）。"""
    canvas_w, canvas_h = vc.RIGHT_WIDTH, vc.SEC1_H
    map_x0 = (canvas_w - wall_map.size_px) // 2
    map_y0 = vc.WALLMAP_MARGIN_V
    right_x0 = map_x0 + wall_map.size_px + vc.WALLMAP_GAP
    right_avail_w = canvas_w - vc.WALLMAP_OUTER_MARGIN - right_x0
    left_avail_w = map_x0 - vc.WALLMAP_GAP - vc.WALLMAP_OUTER_MARGIN
    avail_h = wall_map.size_px

    tmp_draw = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    plan = vc._plan_legend_columns(tmp_draw, legend_items, avail_h, right_avail_w)
    side = "right"
    if plan is None:
        plan = vc._plan_legend_columns(tmp_draw, legend_items, avail_h, left_avail_w)
        side = "left"
    if plan is None:
        plan = vc._force_fit_legend(tmp_draw, legend_items, avail_h, right_avail_w)
        side = "right"

    lx0 = right_x0 if side == "right" else map_x0 - vc.WALLMAP_GAP - plan["total_w"]
    ly0 = map_y0 + (wall_map.size_px - plan["total_h"]) // 2
    return plan, side, lx0, ly0


def _check_legend_items_visible(img: Image.Image, wall_map, legend_items) -> list:
    """各凡例項目のスウォッチ中心画素を実際にサンプリングし、期待した色で
    描かれているかを確認する。不一致があれば
    [(index, expected_rgb, actual_or_error), ...] を返す（空 = 合格）。"""
    plan, side, lx0, ly0 = _recompute_legend_plan(wall_map, legend_items)
    row_h, sw, max_rows = plan["row_h"], plan["sw"], plan["max_rows"]

    px = img.load()
    w, h = img.size
    mismatches = []
    cx = lx0
    idx = 0
    for c in range(plan["n_cols"]):
        col_items = legend_items[c * max_rows:(c + 1) * max_rows]
        for r, (color, _label) in enumerate(col_items):
            ry = ly0 + r * row_h
            sy0 = ry + (row_h - sw) // 2
            scx, scy = cx + sw // 2, sy0 + sw // 2
            if not (0 <= scx < w and 0 <= scy < h):
                mismatches.append((idx, color[:3], "OUT_OF_BOUNDS"))
            else:
                actual = px[scx, scy]
                if actual[:3] != color[:3]:
                    mismatches.append((idx, color[:3], actual))
            idx += 1
        cx += plan["col_widths"][c] + plan["col_gap"]
    return mismatches


def run_case(n_items, wall_map, v_known, h_known, cell_size) -> bool:
    legend_items = LEGEND_POOL[:n_items]
    img = wall_map.render(v_known, h_known, robot_xy=(1.5 * cell_size, 1.5 * cell_size),
                           explore_run_index=1, best_run_index=2, legend_items=legend_items)
    assert img.size == (vc.RIGHT_WIDTH, vc.SEC1_H), f"SEC1画像サイズが想定外: {img.size}"

    out_path = OUT_DIR / f"wallmap_layout_check_{n_items}items.png"
    img.convert("RGB").save(out_path)

    border_bad = _check_borders(img, vc.PANEL_BG)
    legend_bad = _check_legend_items_visible(img, wall_map, legend_items)

    ok = (not border_bad) and (not legend_bad)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] n_items={n_items}  out={out_path}")
    if border_bad:
        print(f"    -> 境界はみ出し: {len(border_bad)} px 違反  例: {border_bad[:5]}")
    if legend_bad:
        print(f"    -> 凡例スウォッチ不一致（欠落/誤配置の疑い）: {legend_bad}")
    return ok


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height, v_walls, h_walls = _load_dummy_maze()
    cell_size = vc.DEFAULT_CELL_SIZE
    wall_map, v_known, h_known = _build_dummy_wallmap(width, height, cell_size, v_walls, h_walls)

    print(f"迷路: {MAZE_NPZ.name} (width={width}, height={height})")
    print(f"地図辺長: {wall_map.size_px}px / SEC1: {vc.RIGHT_WIDTH}x{vc.SEC1_H}px")
    print()

    results = {}
    for n in (3, 4, 5, 6, 8):
        results[n] = run_case(n, wall_map, v_known, h_known, cell_size)

    print()
    if all(results.values()):
        print("ALL PASS")
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"SOME FAILED: n_items={failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
