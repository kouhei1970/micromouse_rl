"""mouse/build_ir_response_table.py — 高速フォワードモデル用の壁表・柱表・床基準値を生成する。

背景・設計は `research_notes/note_034_ir_sensor_model.md` を正とする
（`mouse/ir_sensor.py` の「高速フォワードモデル」節も参照）。

生成物は版管理下に置く（`mouse/ir_sensor.load_table()` で読める npz 形式）:

    mouse/data/ir_response_table.npz  壁の表（`meta["floor_baseline"]` に床の基準値も同梱）
    mouse/data/ir_post_table.npz      柱専用の表（柱がピーク距離付近にあるとき、
                                       無視すると2桁以上過小評価するために追加した。
                                       `build_post_table()` の docstring 参照）

壁・柱の仕様や迷路の寸法を変えたら、本スクリプトを再実行して作り直すこと:

    .venv/bin/python -m mouse.build_ir_response_table
"""
from __future__ import annotations

import time

from mouse.ir_sensor import (
    IrSensorSpec, SurfaceSpec, build_post_table, build_wall_table, floor_baseline, save_table,
)

# 表を作るときの基準センサ。位置・光軸は表の座標系に閉じているので原点でよい
# （`mouse/params.py` の LF/LS/RF/RS はいずれも取付位置以外の仕様が同じなので、
# この1枚の表を4本すべてで使い回せる）。gain は必ず 1.0 にする
# （`fast_response()` 側で実機ごとの gain を最後に1回だけ掛けるため。ここで
# 焼き込むと二重に掛かる）。
CANONICAL_SENSOR = IrSensorSpec(name="canonical", pos=(0.0, 0.0, 0.010), axis=(1.0, 0.0, 0.0), gain=1.0)
SURF = SurfaceSpec()

WALL_OUT_PATH = "mouse/data/ir_response_table.npz"
POST_OUT_PATH = "mouse/data/ir_post_table.npz"


def main() -> None:
    t0 = time.time()
    wall_table = build_wall_table(CANONICAL_SENSOR, SURF)
    dt_wall = time.time() - t0

    t1 = time.time()
    f0 = floor_baseline(CANONICAL_SENSOR, SURF)
    dt_floor = time.time() - t1

    t2 = time.time()
    post_table = build_post_table(CANONICAL_SENSOR, SURF)
    dt_post = time.time() - t2

    wall_table.meta["floor_baseline"] = f0
    save_table(wall_table, WALL_OUT_PATH)
    save_table(post_table, POST_OUT_PATH)

    print(f"[build_ir_response_table] wall table shape={wall_table.values.shape} "
          f"elements={wall_table.values.size} build={dt_wall:.1f}s -> {WALL_OUT_PATH}")
    print(f"[build_ir_response_table] floor_baseline={f0:.6e} build={dt_floor:.2f}s")
    print(f"[build_ir_response_table] post table shape={post_table.values.shape} "
          f"elements={post_table.values.size} build={dt_post:.1f}s -> {POST_OUT_PATH}")


if __name__ == "__main__":
    main()
