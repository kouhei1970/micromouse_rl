"""
tests/test_explorer_belief.py
================
`classic/explorer.py`（`ClassicExplorer`）の地図差し替え（`map_source`、exp_035）の検査。

書き方は `tests/test_classic_pose.py`・`tests/test_wall_belief.py` に合わせる
（特に真値属性へのアクセスを検査するソース走査 — `tests/test_classic_pose.py:56`
と同じ手法）。事前登録は `experiments/exp_035_belief_driven/PREREG.md`、
設計の根拠は `research_notes/note_037_probabilistic_localization.md` §19・§20。

`map_source="threshold"`（既定）ではコード経路が本任務導入前と完全に同一である
ことが主張なので、ここでは主に `map_source="belief"` を切り替えたときの
挙動（`PoseEstimator`/`WallBelief` の生成・地図への反映・否定対照 N2）を固定する。
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

from mouse.params import RobotParams
from mouse.sim import MouseSim

from classic.explorer import ClassicExplorer
from classic.maze_map import Direction, WallState
from classic.pose import PoseEstimator
from classic.wall_belief import WallBelief

REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGN_V4_DIR = REPO_ROOT / "competition" / "mazes" / "design_v4"

# `tests/test_classic_policy.py` が「所要時間が短く済む」として選んだのと同じ面。
# 探索完了までは待たず、有限ティックで打ち切ってよい（任務指示）。
SHORT_RUN_MAZE_ID = "42134"

# `classic.wall_belief.WallBelief.update()` は exp_034 実測で約 3.9ms/周期
# かかる。300ティックなら演算だけで約1.2秒 + シミュレーション物理の分を
# 足しても、1テスト60秒以内に十分収まる（任務指示）。
SHORT_RUN_TICKS = 300


@pytest.fixture(scope="module")
def params():
    return RobotParams()


# ==========================================================================
# 1. map_source の検査（未知の値・既定値）
# ==========================================================================
def test_map_source_rejects_unknown_value():
    """既存の fast_mode の検査と同じ書き方（ValueError）。"""
    with pytest.raises(ValueError):
        ClassicExplorer(5, 5, map_source="bogus")


def test_default_map_source_is_threshold_without_belief_objects():
    """既定は従来どおり: map_source == "threshold"、wall_belief/pose_estimator
    はどちらも None、belief_update_count は 0。"""
    explorer = ClassicExplorer(5, 5)
    assert explorer.map_source == "threshold"
    assert explorer.wall_belief is None
    assert explorer.pose_estimator is None
    assert explorer.belief_update_count == 0


# ==========================================================================
# 2. map_source="belief" の構築
# ==========================================================================
def test_belief_map_source_builds_pose_estimator_and_wall_belief_with_outer_walls_declared():
    """map_source="belief" で構築すると PoseEstimator・WallBelief の両方が
    生成され、外周の柱間は WallBelief が l_max で初期化しているため最初から
    WALL と宣言される（`classic/wall_belief.py` の `WallBelief.__init__`
    参照）。内部の柱間は観測前なのでまだ UNKNOWN のまま。"""
    explorer = ClassicExplorer(5, 5, map_source="belief")
    assert explorer.map_source == "belief"
    assert isinstance(explorer.pose_estimator, PoseEstimator)
    assert isinstance(explorer.wall_belief, WallBelief)

    # 区画(0,0)の西壁・区画(4,4)の北壁はどちらも外周(構造上確定)。
    assert explorer.wall_belief.declare_at(0, 0, Direction.W) == WallState.WALL
    assert explorer.wall_belief.declare_at(4, 4, Direction.N) == WallState.WALL
    # 内部の柱間はまだ未知。
    assert explorer.wall_belief.declare_at(2, 2, Direction.E) == WallState.UNKNOWN


# ==========================================================================
# 3. classic/explorer.py が真値へアクセスしていないこと（ソース走査）
# ==========================================================================
def test_source_never_accesses_privileged_attributes():
    """静的検査: classic/explorer.py の AST を走査し、`privileged_pose`／
    `privileged_velocity` への属性アクセスが増えていないことを確認する
    （`tests/test_classic_pose.py:56`・`tests/test_wall_belief.py` と同じ手法）。"""
    src_path = os.path.join(os.path.dirname(__file__), "..", "classic", "explorer.py")
    with open(src_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=src_path)

    forbidden = {"privileged_pose", "privileged_velocity"}
    hits = [node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in forbidden]
    assert hits == [], f"classic/explorer.py が真値の属性へアクセスしている: {hits}"

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    assert "mouse.sim" not in imported_modules, "classic/explorer.py が mouse.sim を import している"


# ==========================================================================
# 4. design_v4 の迷路 1 面を短時間だけ実際に走らせる
# ==========================================================================
def _load_design_v4_maze_shape(maze_id: str):
    data = np.load(DESIGN_V4_DIR / f"maze_{maze_id}.npz")
    return int(data["width"]), int(data["height"])


def _run_short(params: RobotParams, maze_id: str, n_ticks: int, **explorer_kwargs) -> ClassicExplorer:
    """design_v4 の迷路 1 面を n_ticks だけ走らせ、ClassicExplorer を返す。
    ゴール到達・探索完了は待たず、指定ティック数で打ち切ってよい（任務指示）。"""
    xml_path = DESIGN_V4_DIR / f"maze_{maze_id}.xml"
    width, height = _load_design_v4_maze_shape(maze_id)
    sim = MouseSim(str(xml_path), params=params)
    sim.full_reset(cell=(0, 0), heading_deg=90.0)
    explorer = ClassicExplorer(width, height, params=params, **explorer_kwargs)
    for _ in range(n_ticks):
        obs = sim.observation()
        vl, vr, _plan_id = explorer.tick(obs)
        sim.step_control(vl, vr)
    return explorer


def test_belief_driven_short_run_updates_and_declares_walls(params):
    """map_source="belief" で design_v4 の迷路を短時間走らせても例外を出さず
    動き、belief_update_count > 0 かつ self.maze に UNKNOWN 以外の柱間が
    現れること。"""
    explorer = _run_short(params, SHORT_RUN_MAZE_ID, SHORT_RUN_TICKS, map_source="belief")
    assert explorer.belief_update_count > 0

    known_v = int(np.count_nonzero(explorer.maze.v_walls != int(WallState.UNKNOWN)))
    known_h = int(np.count_nonzero(explorer.maze.h_walls != int(WallState.UNKNOWN)))
    assert known_v + known_h > 0


def test_belief_update_every_tick_false_updates_strictly_less_often(params):
    """否定対照 N2（PREREG §4-2）の土台: belief_update_every_tick=False では、
    同じティック数での belief_update_count が True の場合より厳密に少ないこと。

    True のときは tick() の先頭で毎周期 1 回ずつ呼ぶので、
    belief_update_count は必ず SHORT_RUN_TICKS と一致する。False のときは
    区画中心（_on_stationary）でだけ 1 回呼ぶので、短時間の走行では
    区画中心への到達回数（高々数回〜十数回）しか積み上がらない。"""
    explorer_true = _run_short(params, SHORT_RUN_MAZE_ID, SHORT_RUN_TICKS,
                                map_source="belief", belief_update_every_tick=True)
    explorer_false = _run_short(params, SHORT_RUN_MAZE_ID, SHORT_RUN_TICKS,
                                 map_source="belief", belief_update_every_tick=False)

    assert explorer_true.belief_update_count == SHORT_RUN_TICKS
    assert explorer_false.belief_update_count < explorer_true.belief_update_count
