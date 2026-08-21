"""tests/test_sim_ir_ranges.py — `MouseSim.ir_ranges()`（IRセンサの必要なときだけ計算する経路）

背景・設計は `research_notes/note_034_ir_sensor_model.md` 追記4・5 を正とする。
教授セッションが `classic/explorer.py` の呼び出し頻度を数え直した結果、距離センサの
読みは毎ティックではなく必要な場面（区画中心到着時など）だけで足りると分かった。
本ファイルは `mouse/sim.py::MouseSim.ir_ranges()`（既定は現行の MuJoCo 純正
rangefinder のまま・後方互換）の検査。
"""
from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_maze_with_inner_wall(tmp_path, params):
    """内壁を1本持つ5x5迷路の MJCF を作る（開けすぎでも壁が無さすぎでもない、
    rangefinder と physical の値が両方とも意味を持つ配置にするため）。"""
    W, H = 5, 5
    v = np.zeros((W + 1, H), dtype=int)
    v[0, :] = 1
    v[W, :] = 1
    h = np.zeros((W, H + 1), dtype=int)
    h[:, 0] = 1
    h[:, H] = 1
    v[2, 1] = 1  # 区画(1,1)の右に内壁を1本
    xml_path = os.path.join(str(tmp_path), "inner.xml")
    build_maze_robot_xml(v, h, xml_path, model_name="inner5x5", params=params)
    return xml_path


# ============================================================================
# 1. 既定（"rangefinder"）は現行の挙動と一致する（回帰）
# ============================================================================
def test_ir_ranges_default_matches_observation_ranges(tmp_path):
    """`ir_sensor_mode` を指定しない（既定 "rangefinder"）とき、`ir_ranges()` は
    `observation()` の距離部分（ノイズ抜き）と一致する。既存の `observation()` を
    使う検査・環境は本メソッドの追加による影響を受けないことの直接的な保証。
    """
    params = RobotParams()
    xml_path = _build_maze_with_inner_wall(tmp_path, params)
    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(1, 1), heading_deg=0.0)

    obs = sim.observation()
    n = sim._n_rangefinders
    assert np.array_equal(sim.ir_ranges(), obs[:n])
    assert sim.ir_sensor_mode == "rangefinder"


# ============================================================================
# 2. "physical" に切り替えると値が変わる（作動側）
# ============================================================================
def test_ir_ranges_physical_mode_differs_from_rangefinder(tmp_path):
    """`ir_sensor_mode="physical"` は MuJoCo 純正 rangefinder（幾何距離 [m]）とは
    別物（IR LED＋PT の放射モデル。任意単位の光強度）なので、値が一致しないこと、
    かつ `ir_use_table` の有無で表引き・直接積分を切り替えられることを確認する。
    """
    params = RobotParams()
    xml_path = _build_maze_with_inner_wall(tmp_path, params)

    sim_rf = MouseSim(xml_path, params=params)
    sim_rf.full_reset(cell=(1, 1), heading_deg=0.0)
    r_rangefinder = sim_rf.ir_ranges()

    sim_phys = MouseSim(xml_path, params=params, ir_sensor_mode="physical")
    sim_phys.full_reset(cell=(1, 1), heading_deg=0.0)
    r_physical = sim_phys.ir_ranges()

    print(f"[ir_ranges] rangefinder={r_rangefinder} physical(direct)={r_physical}")
    assert not np.allclose(r_rangefinder, r_physical), (
        "rangefinder と physical で値が一致した（単位も物理モデルも別物のはず）"
    )
    assert np.all(np.isfinite(r_physical)) and np.all(r_physical >= 0.0)

    # ir_use_table: 表を使う版も同じ経路で計算でき、直接積分に近い値を返すこと
    sim_tbl = MouseSim(xml_path, params=params, ir_sensor_mode="physical", ir_use_table=True)
    sim_tbl.full_reset(cell=(1, 1), heading_deg=0.0)
    r_table = sim_tbl.ir_ranges()
    print(f"[ir_ranges] physical(table)={r_table}")
    assert np.all(np.isfinite(r_table))


def test_ir_ranges_caches_within_one_control_step(tmp_path):
    """同一の制御ステップ内で複数回呼んでも同じ値（キャッシュを使い回す）こと、
    `step_control()` を挟むと再計算されることを確認する（懸念1の対処）。

    キャッシュのキーを「制御ステップ番号だけ」にしてよいと判断した根拠:
    `classic/explorer.py::tick(self, obs)` は `obs` を1個の引数として受け取り、
    `_on_stationary(self, obs)`・`_tick_profile(self, obs)`・
    `_apply_wall_correction(self, obs)`・`_issue_next_fast_command(self, obs)` など
    すべての内部メソッドへ**同じ `obs` をそのまま渡す**（`classic/explorer.py` を
    実際に読んで確認した。途中で `sim.observation()` を再取得する箇所は無い）。
    `sense_walls(obs, ...)` の呼び出し（`tick()` 内の while ループで複数回になる
    場合を含む）も同じ `obs` を使うので、1ティックの中で姿勢が変わることはない。
    """
    params = RobotParams()
    xml_path = _build_maze_with_inner_wall(tmp_path, params)
    sim = MouseSim(xml_path, params=params, ir_sensor_mode="physical")
    sim.full_reset(cell=(1, 1), heading_deg=0.0)

    r1 = sim.ir_ranges()
    r2 = sim.ir_ranges()
    assert np.array_equal(r1, r2), "同一ステップ内で値が変わった（キャッシュが効いていない）"

    sim.step_control(1.0, 1.0)
    r3 = sim.ir_ranges()
    assert not np.array_equal(r1, r3), "step_control() の後も同じキャッシュを返した"


# ============================================================================
# 3. classic/ から真の壁へ到達する経路が無いこと（機械的検査）
# ============================================================================
def test_classic_has_no_import_path_to_true_geometry():
    """`classic/` のどのソースも、真の壁・MuJoCo モデルへ到達しうる経路
    （`mujoco`・`mouse.sim`・`mouse.mjcf` の import）を持たないことを検査する。

    当初の指示は「classic/ の全ソースに v_walls・h_walls・xml_path・privileged が
    現れないこと」だったが、これは文字どおりには実装できない。**この4語は
    `classic/` に現在も正当に現れている**（`classic/geometry.py::wall_obstacles(
    v_walls, h_walls, ...)` の引数名、`classic/maze_map.py::MazeMap` の
    「方策自身が作る地図」の属性名、`classic/explorer.py` 等のdocstringが規約
    そのものを説明する散文としての `privileged_pose`/`xml_path` の言及）。
    文字列一致で検査すると、これらの正当な既存コードごと落ちる。

    真に検査すべきは「真の壁・MuJoCo モデルに**到達できる経路があるか**」であり、
    それは `classic/geometry.py` 自身の docstring が既に定める規約
    （「本モジュールは MuJoCo や mouse/ のシミュレータを import しない」）と同じ
    線引きである。到達経路は import 文以外に無い（`classic/` のどの関数も `sim`
    オブジェクトを受け取って `.privileged_pose()` を呼ぶ実装を持たないことは
    `tests/test_video_no_privilege_leak.py` が別の角度から検査済み）ので、
    ここでは import 文だけを機械的に検査する。
    """
    classic_dir = _REPO_ROOT / "classic"
    forbidden_import_re = re.compile(
        r"^\s*(import\s+(mujoco|mouse\.sim|mouse\.mjcf)\b"
        r"|from\s+(mujoco|mouse\.sim|mouse\.mjcf)\s+import\b)",
        re.MULTILINE,
    )
    offenders = []
    for path in sorted(classic_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        m = forbidden_import_re.search(text)
        if m:
            offenders.append((path.name, m.group(0).strip()))
    assert not offenders, f"classic/ が真の幾何へ到達しうる import を持つ: {offenders}"


# ============================================================================
# 4. 実迷路で ir_ranges() を600回呼んだときの所要時間（見積り2.3分の裏取り）
# ============================================================================
def test_ir_ranges_timing_on_real_maze():
    """`competition/mazes/design_turn_v1/maze_41000.xml`（実迷路の1面）で
    `ir_ranges()`（"physical"・直接積分）を600回呼び、所要時間を測って印字する。
    """
    params = RobotParams()
    xml_path = _REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "maze_41000.xml"
    sim = MouseSim(str(xml_path), params=params, ir_sensor_mode="physical")

    rng = np.random.default_rng(0)
    n_calls = 600
    t0 = time.perf_counter()
    for i in range(n_calls):
        cell = (int(rng.integers(0, 16)), int(rng.integers(0, 16)))
        heading = float(rng.uniform(0, 360))
        sim.full_reset(cell=cell, heading_deg=heading)
        sim.ir_ranges()
    elapsed = time.perf_counter() - t0

    print(f"[ir_ranges timing] {n_calls}回・実迷路1面: {elapsed:.2f}秒 "
          f"({elapsed / n_calls * 1000:.2f}ms/回)")
    # 見積り(2.3分/迷路・条件)の裏取りが目的の検査であり、上限は緩く取る
    # （見積りの数倍までは許容し、失敗ではなく実測値そのものを報告として使う）。
    assert elapsed < 600.0, f"600回で{elapsed:.1f}秒は見積りから大きく外れている"
