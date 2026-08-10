#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""16x16 の大会迷路(.maze)を npz 形式(v_walls(17,16), h_walls(16,17))に変換して保存する。"""
import sys, glob, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_maze import parse_maze_file, to_npz_format, MazeFormatError

SRC_DIR = Path(__file__).resolve().parent / "micromouse-maze-data" / "data"
OUT_DIR = Path(__file__).resolve().parent / "contest"
OUT_DIR.mkdir(exist_ok=True)

files = sorted(SRC_DIR.glob("*.maze"))
converted = []
skipped_unknown = []
skipped_other_size = []
errors = []

for idx, f in enumerate(files):
    try:
        pm = parse_maze_file(str(f))
    except MazeFormatError as e:
        errors.append((f.name, str(e)))
        continue
    if pm.size != 16:
        skipped_other_size.append((f.name, pm.size))
        continue
    try:
        v_walls, h_walls = to_npz_format(pm)
    except ValueError as e:
        skipped_unknown.append((f.name, str(e)))
        continue

    out_name = f"contest_{f.stem}.npz"
    out_path = OUT_DIR / out_name
    np.savez(
        out_path,
        v_walls=v_walls.astype(np.int64),
        h_walls=h_walls.astype(np.int64),
        seed=np.int64(-1),          # 大会迷路のためシードは無意味。-1 でダミー
        width=np.int64(16),
        height=np.int64(16),
        source_file=str(f.name),    # 由来ファイル名（追跡用の追加メタデータ）
        start_x=np.int64(pm.start[0]) if pm.start else np.int64(-1),
        start_y=np.int64(pm.start[1]) if pm.start else np.int64(-1),
        goals_x=np.array([g[0] for g in pm.goals], dtype=np.int64),
        goals_y=np.array([g[1] for g in pm.goals], dtype=np.int64),
    )
    converted.append((f.name, out_name, pm.start, sorted(pm.goals)))

print(f"入力ファイル総数: {len(files)}")
print(f"16x16 以外(除外): {len(skipped_other_size)}")
print(f"16x16 だが未知壁を含む(除外): {len(skipped_unknown)}")
for name, reason in skipped_unknown:
    print(f"  - {name}: {reason}")
print(f"パースエラー: {len(errors)}")
for name, e in errors:
    print(f"  - {name}: {e}")
print(f"変換成功(16x16, 壁完全既知): {len(converted)}")

manifest = {
    "n_total_files": len(files),
    "n_16x16_total": len(converted) + len(skipped_unknown),
    "n_16x16_unknown_wall_excluded": len(skipped_unknown),
    "n_converted": len(converted),
    "converted": [
        {"source": name, "npz": out, "start": list(start), "goals": [list(g) for g in goals]}
        for name, out, start, goals in converted
    ],
    "excluded_unknown_wall": [name for name, _ in skipped_unknown],
}
with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as fp:
    json.dump(manifest, fp, ensure_ascii=False, indent=2)

print(f"\n出力先: {OUT_DIR}")
