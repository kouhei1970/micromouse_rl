"""
tests/test_classic_sensing.py
================
classic/sensing.py の検査。**実際にシミュレータを動かして**距離センサの
生値を実測し、壁あり／壁なしが分離していること・判定関数が正しく判定する
ことを確認する。加えて「わざと閾値を壊すと落ちる」検査を含め、
本テストが空振りしないことを確認する（note_030 §5 の作法）。

🔴 2026-08-19 是正 (1)（note_029 §4「登録簿は働かない。実装された検査
だけが働いた」）: `classic/checks.py` の再発防止検査は実装した時点では
どこからも呼ばれていなかった。本ファイルは `classic.checks.plan_adherence`
を実際に呼び出す（型 C「計画した経路が走行に使われたか」を、「姿勢サンプル
群が意図した分類から外れていないか」という同型の問題に転用する）。

迷路の座標規約（v_walls[x,y] / h_walls[x,y]）は competition/evaluator.py の
モジュール docstring・mouse/mjcf.py の build_maze_robot_xml docstring に従う。
ロボット初期姿勢 heading_deg=90.0（+Y=北向き）のとき:
  前方壁 = h_walls[cx, cy+1]（北側）
  左方壁 = v_walls[cx, cy]（西側。左旋回で -X が左になる規約は
           mouse/sim.py の座標系そのもの）
  右方壁 = v_walls[cx+1, cy]（東側）
"""
import math
import os

import numpy as np
import pytest
import mujoco

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

from classic.checks import plan_adherence
from classic.sensing import (
    WallState, sense_walls,
    FRONT_WALL_MAX, FRONT_CLEAR_MIN, SIDE_WALL_MAX, SIDE_CLEAR_MIN,
)

W, H = 5, 5
CX, CY = 2, 2  # 判定対象のセル（境界の影響を避けるため迷路中央寄りを使う）


def _make_walls(open_front=False, open_left=False, open_right=False, deep=False):
    """CX,CY を囲む壁を全部立てた迷路をベースに、指定方向だけ開放する。
    deep=True なら 2 区画分連続で開放する（cutoff に張り付く「確実に壁なし」条件用）。"""
    v = np.ones((W + 1, H), dtype=int)
    h = np.ones((W, H + 1), dtype=int)
    if open_front:
        h[CX, CY + 1] = 0
        if deep:
            h[CX, CY + 2] = 0
    if open_left:
        v[CX, CY] = 0
        if deep:
            v[CX - 1, CY] = 0
    if open_right:
        v[CX + 1, CY] = 0
        if deep:
            v[CX + 2, CY] = 0
    return v, h


def _build_sim(v, h, tmp_path, label, params):
    xml_path = os.path.join(str(tmp_path), f"maze_{label}.xml")
    build_maze_robot_xml(v, h, xml_path, model_name=f"m_{label}", params=params)
    return MouseSim(xml_path, params=params)


def _set_pose(sim, dx, dy, dheading_deg):
    """区画中心からの位置ずれ dx,dy [m]・方位ずれ [deg] を与えて qpos を直接書く
    （reset_to_start はセル中心固定のため、位置ずれの再現に qpos を直接操作する）。"""
    cell = sim.params.cell_size
    cx0 = CX * cell + cell / 2
    cy0 = CY * cell + cell / 2
    qpos_adr = sim.model.jnt_qposadr[sim._root_joint_id]
    x = cx0 + dx
    y = cy0 + dy
    heading = math.radians(90.0 + dheading_deg)
    sim.data.qpos[qpos_adr:qpos_adr + 3] = [x, y, 0.002]
    sim.data.qpos[qpos_adr + 3] = math.cos(heading / 2.0)
    sim.data.qpos[qpos_adr + 4] = 0.0
    sim.data.qpos[qpos_adr + 5] = 0.0
    sim.data.qpos[qpos_adr + 6] = math.sin(heading / 2.0)
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


# 判定関数の運用前提（sensing.py docstring 参照）: 位置ずれ ±15mm・方位ずれ ±4°。
_DXS = np.linspace(-0.015, 0.015, 5)
_DYS = np.linspace(-0.015, 0.015, 5)
_DHEADS = np.linspace(-4.0, 4.0, 5)


def _sample_ranges(sim):
    """区画中心近傍の姿勢格子を振って距離センサ生値を集める。shape=(N, n_sensors)。"""
    rows = []
    for dx in _DXS:
        for dy in _DYS:
            for dh in _DHEADS:
                _set_pose(sim, dx, dy, dh)
                obs = sim.observation()
                rows.append(obs[:len(sim.params.sensors)].copy())
    return np.array(rows)


@pytest.fixture(scope="module")
def params():
    return RobotParams()


# ==========================================================================
# 1. 実測: 壁あり／壁なしで距離が分離していること（数値を報告する）
# ==========================================================================
def test_front_distance_separates_wall_present_from_absent(tmp_path_factory, params):
    """前方センサ (LF/RF): 壁ありと壁なし（1区画先のみ開放）の距離分布が
    重ならずに分離していることを実測する。数値は標準出力に報告する。"""
    tmp = tmp_path_factory.mktemp("front_sep")
    v_closed, h_closed = _make_walls()
    v_open, h_open = _make_walls(open_front=True, deep=False)

    sim_closed = _build_sim(v_closed, h_closed, tmp, "closed", params)
    sim_open = _build_sim(v_open, h_open, tmp, "open", params)

    closed = _sample_ranges(sim_closed)
    open_ = _sample_ranges(sim_open)

    # LF=0, RF=2 (params.sensors の並び順)
    idx_lf = [s['name'] for s in params.sensors].index('LF')
    idx_rf = [s['name'] for s in params.sensors].index('RF')
    front_closed = np.minimum(closed[:, idx_lf], closed[:, idx_rf])
    front_open = np.minimum(open_[:, idx_lf], open_[:, idx_rf])

    print(f"\n[実測] 前方距離(LF/RFの小さい方): 壁あり range=[{front_closed.min():.4f}, "
          f"{front_closed.max():.4f}]  壁なし range=[{front_open.min():.4f}, {front_open.max():.4f}]")

    # 分離: 壁ありの最大値 < 壁なしの最小値（重なりゼロ）
    assert front_closed.max() < front_open.min(), (
        "前方センサの壁あり/壁なしの距離分布が重なっている（実測で分離を確認できなかった）"
    )
    # しきい値がこの実測分布の隙間の中に収まっていることも確認する
    assert front_closed.max() < FRONT_WALL_MAX < FRONT_CLEAR_MIN < front_open.min()


def test_side_distance_separates_with_known_ambiguous_band(tmp_path_factory, params):
    """側方センサ (LS/RS): 壁あり／壁なし（1区画先のみ開放）の距離分布は
    **重なりが実在する**（センサが約76°傾いているため）。この重なりを
    握りつぶさず、しきい値がその重なり帯の外側（分離が確認できた領域）に
    正しく置かれていることを実測で確認する。"""
    tmp = tmp_path_factory.mktemp("side_sep")
    v_closed, h_closed = _make_walls()
    v_open, h_open = _make_walls(open_left=True, deep=False)

    sim_closed = _build_sim(v_closed, h_closed, tmp, "closed", params)
    sim_open = _build_sim(v_open, h_open, tmp, "open", params)

    closed = _sample_ranges(sim_closed)
    open_ = _sample_ranges(sim_open)

    idx_ls = [s['name'] for s in params.sensors].index('LS')
    side_closed = closed[:, idx_ls]
    side_open = open_[:, idx_ls]

    print(f"\n[実測] 左側方距離(LS): 壁あり range=[{side_closed.min():.4f}, {side_closed.max():.4f}]  "
          f"壁なし range=[{side_open.min():.4f}, {side_open.max():.4f}]")
    n_overlap = int(np.sum(side_closed > side_open.min()))
    print(f"[実測] 重なり帯 [{side_open.min():.4f}, {side_closed.max():.4f}] に入る"
          f"壁ありサンプル数: {n_overlap}/{len(side_closed)}")

    # 重なりが実在すること自体を確認する（無いはずのものを「無い」と言わない）
    assert side_closed.max() > side_open.min(), (
        "側方センサに重なりが実在するという前提が崩れている。"
        "しきい値設計（AMBIGUOUS 帯）の前提を再検証すること。"
    )
    # しきい値は「壁なしで一度も観測されなかった値」より下、
    # 「壁ありで一度も観測されなかった値」より上に置かれていること
    assert SIDE_WALL_MAX <= side_open.min()
    assert SIDE_CLEAR_MIN >= side_closed.max()
    assert SIDE_WALL_MAX < SIDE_CLEAR_MIN


# ==========================================================================
# 2. 判定関数が壁あり／なしを正しく返すこと（区画中心での代表ケース）
# ==========================================================================
@pytest.fixture(scope="module")
def center_observations(tmp_path_factory, params):
    """区画中心（ずれ無し）での 5 パターンの観測ベクトルをまとめて作る。"""
    tmp = tmp_path_factory.mktemp("center_obs")

    def obs_for(open_front, open_left, open_right, deep=False):
        v, h = _make_walls(open_front, open_left, open_right, deep=deep)
        label = f"f{int(open_front)}l{int(open_left)}r{int(open_right)}d{int(deep)}"
        sim = _build_sim(v, h, tmp, label, params)
        sim.full_reset(cell=(CX, CY), heading_deg=90.0)
        return sim.observation()

    return {
        "all_closed": obs_for(False, False, False),
        "front_open": obs_for(True, False, False, deep=True),
        "left_open": obs_for(False, True, False, deep=True),
        "right_open": obs_for(False, False, True, deep=True),
        "all_open": obs_for(True, True, True, deep=True),
    }


def test_sense_walls_all_closed(center_observations, params):
    r = sense_walls(center_observations["all_closed"], params)
    assert r.front == WallState.WALL
    assert r.left == WallState.WALL
    assert r.right == WallState.WALL
    assert not r.any_ambiguous


def test_sense_walls_front_open(center_observations, params):
    r = sense_walls(center_observations["front_open"], params)
    assert r.front == WallState.CLEAR
    assert r.left == WallState.WALL
    assert r.right == WallState.WALL


def test_sense_walls_left_open(center_observations, params):
    r = sense_walls(center_observations["left_open"], params)
    assert r.front == WallState.WALL
    assert r.left == WallState.CLEAR
    assert r.right == WallState.WALL


def test_sense_walls_right_open(center_observations, params):
    r = sense_walls(center_observations["right_open"], params)
    assert r.front == WallState.WALL
    assert r.left == WallState.WALL
    assert r.right == WallState.CLEAR


def test_sense_walls_all_open(center_observations, params):
    r = sense_walls(center_observations["all_open"], params)
    assert r.front == WallState.CLEAR
    assert r.left == WallState.CLEAR
    assert r.right == WallState.CLEAR
    assert not r.any_ambiguous


# ==========================================================================
# 3. 曖昧帯の実例で AMBIGUOUS が実際に返ること（握りつぶしていないことの確認）
# ==========================================================================
def test_sense_walls_returns_ambiguous_for_known_overlap_case(tmp_path_factory, params):
    """実測で重なりが確認された姿勢（左が浅く開放・位置と方位が重なり帯側に
    振れた姿勢）を再現し、AMBIGUOUS が実際に返ることを確認する。
    「どちらとも言えない範囲を明示的に返す」という要件そのものの検査。"""
    tmp = tmp_path_factory.mktemp("ambiguous")
    v_open, h_open = _make_walls(open_left=True, deep=False)
    sim = _build_sim(v_open, h_open, tmp, "ambig", params)
    # 実測で side_open の最小値(0.07873m 付近)を与えた姿勢
    _set_pose(sim, dx=-0.015, dy=0.015, dheading_deg=-4.0)
    obs = sim.observation()
    r = sense_walls(obs, params)
    print(f"\n[実測] 曖昧姿勢での左距離 = {r.left_dist:.4f} m → 判定 = {r.left.value}")
    assert r.left == WallState.AMBIGUOUS
    assert r.any_ambiguous


# ==========================================================================
# 4. わざと閾値を壊すと落ちること（空振りしない検査であることの確認）
# ==========================================================================
def test_broken_threshold_makes_classification_wrong(center_observations, params, monkeypatch):
    """しきい値を壊すと、壁なし(front_open)が WALL に誤判定されることを確認する。
    これは「本テストが常に成功するわけではない」＝空振りしない検査であることの
    直接的な証拠になる。"""
    import classic.sensing as sensing

    # 壊す前: 正しい判定が返ることを確認（前提）
    r_ok = sensing.sense_walls(center_observations["front_open"], params)
    assert r_ok.front == WallState.CLEAR

    # 前方のしきい値を、壁なし(front_open, 実測 0.300m = cutoff)より
    # 大きい側へ破壊する（正しい実装なら CLEAR になるはずの距離を
    # WALL/AMBIGUOUS に誤判定させる）
    monkeypatch.setattr(sensing, "FRONT_WALL_MAX", 0.35)
    monkeypatch.setattr(sensing, "FRONT_CLEAR_MIN", 0.40)

    r_broken = sensing.sense_walls(center_observations["front_open"], params)
    # 壊した結果、壁なしのはずが AMBIGUOUS や WALL に化ける
    # （少なくとも CLEAR ではなくなる = 誤判定が実際に起きることの確認）
    assert r_broken.front != WallState.CLEAR, (
        "しきい値を破壊しても判定が変わらなかった。"
        "この検査が判定ロジックを実際に通していない（空振り）疑いがある。"
    )


def test_threshold_ordering_is_enforced_by_construction():
    """しきい値の大小関係（wall_max < clear_min）が壊れていないことの直接確認。
    ここが崩れると全ての判定が意味を失う（AMBIGUOUS しか返らなくなる、
    または WALL/CLEAR が逆転しうる）。"""
    assert FRONT_WALL_MAX < FRONT_CLEAR_MIN
    assert SIDE_WALL_MAX < SIDE_CLEAR_MIN


# ==========================================================================
# 5. plan_adherence の応用: 姿勢ぶれ全域で意図した分類に乗っているか
#    是正 (1): classic.checks を実際に呼び出す。plan_adherence は本来
#    「走行がどの計画に乗っていたか」をティック列で測る検査だが、ここでは
#    「姿勢サンプル群のうち意図した分類に乗っていた割合」という同型の
#    問題に転用する。
# ==========================================================================
def test_front_classification_adherence_across_pose_grid(tmp_path_factory, params):
    """前方判定 (WALL/CLEAR) が、区画内の位置・方位ずれ全域（5×5×5=125 姿勢の
    格子）で意図した分類から一件も外れないことを plan_adherence で確認する
    （前方センサは実測で重なりゼロと分かっている。§1 の分離実測と対の検査）。"""
    tmp = tmp_path_factory.mktemp("front_adherence")
    v_closed, h_closed = _make_walls()
    v_open, h_open = _make_walls(open_front=True, deep=False)
    sim_closed = _build_sim(v_closed, h_closed, tmp, "closed_adh", params)
    sim_open = _build_sim(v_open, h_open, tmp, "open_adh", params)

    def classify_front_grid(sim):
        labels = []
        for dx in _DXS:
            for dy in _DYS:
                for dh in _DHEADS:
                    _set_pose(sim, dx, dy, dh)
                    obs = sim.observation()
                    labels.append(sense_walls(obs, params).front.value)
        return labels

    closed_adherence = plan_adherence(classify_front_grid(sim_closed), intended="wall")
    open_adherence = plan_adherence(classify_front_grid(sim_open), intended="clear")

    print(f"\n[実測] 前方判定の乗車率（壁あり側）:\n{closed_adherence.describe()}")
    print(f"[実測] 前方判定の乗車率（壁なし側）:\n{open_adherence.describe()}")

    assert closed_adherence.ok(1.0), "壁ありのはずの姿勢サンプルで WALL 以外の判定が出た"
    assert open_adherence.ok(1.0), "壁なしのはずの姿勢サンプルで CLEAR 以外の判定が出た"
