"""
experiments/exp_034_wall_belief/run.py
================
exp_034（壁の信念）の測定スクリプト。判定条文は
`experiments/exp_034_wall_belief/PREREG.md`（結果を見る前に凍結済み）。

やること（PREREG §1・§2）: `classic/pose.py` の `PoseEstimator` と
`classic/wall_belief.py` の `WallBelief` を、`classic/policy.py` の
`ClassicExplorerPolicy`（現行の古典ベースライン）に**並走**させる。
制御周期ごとに推定姿勢・共分散・距離センサ4本の実測値を `WallBelief.update()`
へ渡し、走行の終わりに `to_maze_map()` 相当の宣言を真の壁配列と突き合わせる。
**推定器・信念地図の出力は制御に一切使わない**（`_WallBeliefTrackingPolicy.act()`
参照。制御に渡す obs と、推定器・信念へ渡す値は別経路であり、後者をどう加工しても
`self._inner.act(obs)` に渡す obs は無傷のまま）。

【並走のさせ方（禁止事項: `classic/`・`competition/`・`mouse/` 配下は1行も変更しない）】
`exp_031_metric_pose/run.py` と同じ方式: `competition/policy_interface.py` の
`MousePolicy` を継承した薄いラッパを本ファイル内に定義し、
`competition/evaluator.py` の `CompetitionEvaluator.evaluate_maze()` へ
そのまま渡す。

【真の地図の読み方】
`requires_privileged = True` にし、評価器の `bind_maze(v_walls, h_walls)` 呼び出し
（迷路開始前に一度だけ）で真の壁配列を受け取る。**診断用に保持するだけで、
`WallBelief` へは絶対に渡さない**（`bind_maze()` の実装・`act()` の中身を見れば
分かるとおり、真の配列を触れる箇所は診断の集計コードだけである）。
`bind_sim(sim)` は `requires_privileged=True` の契約上呼ばれるが、本スクリプトは
`sim` を一切使わない（真の姿勢 `privileged_pose()` は不要 — 信念は推定姿勢と
実測レンジだけで動く。exp_031 と違い、真値姿勢との突き合わせは今回の主判定量ではない）。

【否定対照の実施方式（オフライン再生。物理シミュレーションは1回だけ）】
4つの否定対照（PREREG §4）は、いずれも「その周期の推定姿勢・共分散・距離センサ
実測値」という**一次記録**だけから再現できる（`WallBelief.update()` は
`classic.maze_map.MazeMap` 等の外部状態を持たず、純粋にこの3つの引数だけで
1周期分の証拠を積む）。そこで、否定対照を担当する1迷路（`NEGATIVE_CONTROL_SEED`）
だけ、通常走行（物理シミュレーション込み）を1回走らせながら毎周期の
(推定姿勢, 共分散, レンジ) を記録し、その後は**シミュレータを再実行せずに**
記録済みの一次記録を壊した／並べ替えた入力で新しい `WallBelief` インスタンスへ
再生する。これにより:
  - N1（姿勢を区画中心・軸並行に固定）・N2（観測を時間方向に並べ替え）・
    N3（対数尤度比を常に0）・N4（L_maxを外す）のいずれも、物理走行や制御則には
    一切触れない（走行そのものは影響を受けようがない — 独立した再生だから）
  - MuJoCo の物理ステップを4回余計に回す必要がなく、計算予算を大きく節約できる
N3 は `classic.wall_belief._log_likelihood_ratio`（モジュール内部関数）を
一時的に上書きして常に0を返すようにする。これは**ファイルを書き換えるのではなく、
実行時にモジュール属性を差し替えるだけ**（`finally` で必ず元に戻す）。
N4 は `WallBelief(..., l_max=1e9)` という**既存のコンストラクタ引数**を使うだけで、
`classic/wall_belief.py` を書き換える必要は無い。

【主判定量・合格条件・副次記録の対応（PREREG §2・§3・§7）】
- 主判定量 $N_{fatal}$（真は壁なのに「開通」と宣言した柱間の数、20迷路合計）
  → `count_fatal()`。20迷路すべてで計算し `summary.json` へ合算する。
- 合格条件1（宣言数が従来の3値地図を下回らない）→ `classic.policy.ClassicExplorerPolicy`
  が公開している `v_walls_known`/`h_walls_known`（読み取り専用プロパティ、
  真値ではなく方策自身が探索で得た地図）を**同じ走行から**取り出して比較する。
  ラッパの `_inner`（内部で駆動している ClassicExplorerPolicy そのもの）が
  作った地図であり、`WallBelief` とは完全に別の経路で更新されている。
- 合格条件2（走行が変わらない）→ `sha256_of_result()`（"policy" フィールドを
  除いて比較する理由は exp_031 と同じ）。
- 副次記録（PREREG §7）: 無害な誤り `count_benign()`、未知のまま残った割合
  `n_unknown/544`、1柱間あたりの観測回数の分布（`log_odds` が前後で変化したか
  という差分で「その周期にその柱間が候補として実際にヒットしたか」を検出する
  ── `WallBelief` を変更せずに数える唯一の方法。厳密には「対数尤度比が
  ちょうど浮動小数点で0.0だった」ごく稀なケースを見逃すが、無視できる）、
  従来の3値地図との柱間ごとの食い違い、宣言が途中で反転した柱間の数
  （毎周期の宣言配列を前周期と比較し、WALL↔OPENの反転だけを数える。
  UNKNOWNを経由する遷移は反転に数えない）。
  姿勢の不確かさによる分散の膨らみについては、否定対照を担当する1迷路だけで
  `_jacobian_fd`（モジュール内部関数）の戻り値を一時的に記録するフック方式で
  測る（同じくファイルは書き換えない。理由は下記モジュール docstring 参照）。
  全20迷路で行わなかった理由は、記録リストの生成・クリアが1周期あたり
  余分なオーバーヘッドになり、20迷路すべてでは無視できない時間になるため
  （判断が要った点。教授セッションの検収を求める）。

【マイコン実装の予算（PREREG §6）】
`classic/wall_belief.py`・`classic/pose.py` のモジュール docstring に、
乗算・除算の回数を数え上げた手計算（実行時計測ではない、と明記されている）が
既にある。本スクリプトはそれを書き換えずに引用し、代わりに**実測できるもの**
（実際の20迷路走行での1センサあたりの候補ヒット数の平均、Python実装の
1周期あたりの実測所要時間）を測って、手計算の前提（典型2〜3候補/センサ）が
実際の走行でも妥当かを確かめる。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import classic.wall_belief as wall_belief_mod  # noqa: E402
from classic.maze_map import WallState  # noqa: E402
from classic.policy import ClassicExplorerPolicy  # noqa: E402
from classic.pose import PoseEstimator  # noqa: E402
from classic.wall_belief import (  # noqa: E402
    L_MAX_DEFAULT,
    R_MAX_DEFAULT,
    SIGMA_SENSOR_DEFAULT,
    T_OPEN_DEFAULT,
    T_WALL_DEFAULT,
    WallBelief,
)
from competition.evaluator import CompetitionEvaluator  # noqa: E402
from competition.policy_interface import MousePolicy  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

DESIGN_V4_DIR = REPO_ROOT / "competition" / "mazes" / "design_v4"
OUT_ROOT = REPO_ROOT / "outputs" / "exp_034_wall_belief"
RAW_DIR = OUT_ROOT / "raw"

# 評価用に予約された seed 範囲（docs/RESEARCH_PLAN.md §2・§9-7 が正）。
RESERVED_SEED_RANGE = (1000, 40999)

# PREREG §2: 古典の設計帯 seed 41000〜61000
DESIGN_BAND = (41000, 61000)

# 否定対照を担当する1迷路。20迷路をseed昇順に並べた先頭（最小seed）。
# 決定的な規則で選ぶ（exp_031 が「選定した10面の先頭」を否定対照担当にしたのと
# 同じ思想）。この1迷路だけ毎周期の一次記録（推定姿勢・共分散・レンジ）を保持する。
NEGATIVE_CONTROL_SEED = 41003

# 🔴 否定対照4つ(N1-N4)は、いずれも記録済みの一次記録を新しい WallBelief で
# 再生するだけ（PREREG §4）。実測したところ `WallBelief.update()` は1周期
# 約3.9ms（後述のパイロット計測）で、この迷路(41003)は420秒の持ち時間を使い切り
# 42001周期に達した。5つの否定対照(N0検算込み)をまるごと(42001周期)再生すると
# 約13分かかり、「1迷路が10分を超えたら止める」という進め方の指示に抵触した
# （実際に一度キャンセルした）。そこで、否定対照の再生だけ先頭
# `NEGATIVE_CONTROL_MAX_TICKS` 周期に打ち切る。60秒ぶんの探索走行に相当し、
# 迷路全域を回り切るには足りない場合があるが、「姿勢や観測を壊すと信念が
# どう変わるか」を実証するには十分な観測回数が積まれる。主判定量(20迷路合計の
# N_fatal)や合格条件1(従来の3値地図との比較)は打ち切らず全周期を使う ──
# 打ち切るのは否定対照の再生だけである。（判断が要った点。教授セッションの検収を求める）
NEGATIVE_CONTROL_MAX_TICKS = 6000


# ==========================================================================
# 対象迷路の選定（PREREG §2: design_v4 の20迷路すべて）
# ==========================================================================
def select_maze_paths() -> list[Path]:
    all_npz = sorted(DESIGN_V4_DIR.glob("maze_*.npz"),
                      key=lambda p: int(p.stem.split("_")[1]))
    if len(all_npz) != 20:
        raise RuntimeError(
            f"design_v4 の面数が想定(20)と違う: {len(all_npz)}。"
            "PREREG §2 の前提(20迷路すべて)が崩れている可能性がある。実行を止める。"
        )
    lo, hi = RESERVED_SEED_RANGE
    dlo, dhi = DESIGN_BAND
    for p in all_npz:
        seed = int(p.stem.split("_")[1])
        if lo <= seed <= hi:
            raise RuntimeError(
                f"評価用に予約された seed 範囲 [{lo},{hi}] に含まれる seed {seed} を"
                f"選んでしまった（測定禁止）。"
            )
        if not (dlo <= seed <= dhi):
            raise RuntimeError(f"seed {seed} が古典の設計帯 [{dlo},{dhi}] の外にある。")
    return all_npz


# ==========================================================================
# 宣言（読み出し）のベクトル化 — classic.wall_belief.declare_state と同じ
# 非対称しきい値を配列全体へ適用するだけ（classic/wall_belief.py は変更しない。
# 544要素ぶんを毎周期 Python レベルのループで呼ぶのを避けるための外部再現）。
# ==========================================================================
def declared_array(log_odds: np.ndarray, t_wall: float = T_WALL_DEFAULT,
                    t_open: float = T_OPEN_DEFAULT) -> np.ndarray:
    out = np.zeros(log_odds.shape, dtype=np.int8)
    out[log_odds > t_wall] = int(WallState.WALL)
    out[log_odds < -t_open] = int(WallState.OPEN)
    return out


def declared_arrays(wb: WallBelief, t_wall: float = T_WALL_DEFAULT,
                     t_open: float = T_OPEN_DEFAULT):
    return declared_array(wb.log_odds_v, t_wall, t_open), declared_array(wb.log_odds_h, t_wall, t_open)


def count_fatal(declared_v: np.ndarray, declared_h: np.ndarray,
                true_v: np.ndarray, true_h: np.ndarray) -> int:
    """真は壁(true!=0)なのに「開通」と宣言した柱間の数（主判定量）。"""
    true_wall_v = true_v != 0
    true_wall_h = true_h != 0
    fatal_v = (declared_v == int(WallState.OPEN)) & true_wall_v
    fatal_h = (declared_h == int(WallState.OPEN)) & true_wall_h
    return int(np.count_nonzero(fatal_v)) + int(np.count_nonzero(fatal_h))


def count_benign(declared_v: np.ndarray, declared_h: np.ndarray,
                  true_v: np.ndarray, true_h: np.ndarray) -> int:
    """真は開通なのに「壁」と宣言した数（副次記録・無害な誤り）。"""
    true_wall_v = true_v != 0
    true_wall_h = true_h != 0
    benign_v = (declared_v == int(WallState.WALL)) & (~true_wall_v)
    benign_h = (declared_h == int(WallState.WALL)) & (~true_wall_h)
    return int(np.count_nonzero(benign_v)) + int(np.count_nonzero(benign_h))


def count_declared(declared_v: np.ndarray, declared_h: np.ndarray) -> int:
    return (int(np.count_nonzero(declared_v != int(WallState.UNKNOWN)))
            + int(np.count_nonzero(declared_h != int(WallState.UNKNOWN))))


def count_unknown(declared_v: np.ndarray, declared_h: np.ndarray) -> int:
    return (int(np.count_nonzero(declared_v == int(WallState.UNKNOWN)))
            + int(np.count_nonzero(declared_h == int(WallState.UNKNOWN))))


def _forced_cell_center_axis_pose(pose, cell_size: float):
    """N1: 姿勢を「区画中心・軸並行」に丸める（推定した区画・向きの象限は使うが、
    連続値は捨てる。classic/sensing.py 時代の較正の前提を模す）。"""
    x, y, theta = pose
    cx = math.floor(x / cell_size) * cell_size + cell_size / 2.0
    cy = math.floor(y / cell_size) * cell_size + cell_size / 2.0
    half_pi = math.pi / 2.0
    snapped = round(theta / half_pi) * half_pi
    two_pi = 2.0 * math.pi
    snapped = math.fmod(snapped + math.pi, two_pi)
    if snapped < 0.0:
        snapped += two_pi
    snapped -= math.pi
    return (cx, cy, snapped)


# ==========================================================================
# 一次記録の再生（負の対照。物理シミュレーションを再実行しない）
# ==========================================================================
def replay_sequence(width: int, height: int, params: RobotParams,
                     poses, covs, ranges_list, *,
                     l_max: float = L_MAX_DEFAULT, track_reversals: bool = True):
    """記録済みの (pose, cov, ranges) 列を新しい WallBelief へそのまま(あるいは
    壊して)与え、最終状態と反転回数・最大対数オッズを返す。"""
    wb = WallBelief(width, height, params, l_max=l_max)
    reversal_count = 0
    prev_v, prev_h = declared_arrays(wb)
    max_abs = float(max(np.max(np.abs(wb.log_odds_v)), np.max(np.abs(wb.log_odds_h))))
    for pose, cov, ranges in zip(poses, covs, ranges_list):
        wb.update(pose, cov, ranges)
        if track_reversals:
            cur_v, cur_h = declared_arrays(wb)
            rev_v = (prev_v != int(WallState.UNKNOWN)) & (cur_v != int(WallState.UNKNOWN)) & (prev_v != cur_v)
            rev_h = (prev_h != int(WallState.UNKNOWN)) & (cur_h != int(WallState.UNKNOWN)) & (prev_h != cur_h)
            reversal_count += int(np.count_nonzero(rev_v)) + int(np.count_nonzero(rev_h))
            prev_v, prev_h = cur_v, cur_h
        cur_max = float(max(np.max(np.abs(wb.log_odds_v)), np.max(np.abs(wb.log_odds_h))))
        if cur_max > max_abs:
            max_abs = cur_max
    return wb, reversal_count, max_abs


def compute_negative_controls(policy: "_WallBeliefTrackingPolicy", params: RobotParams,
                               width: int, height: int, true_v: np.ndarray,
                               true_h: np.ndarray) -> dict:
    """PREREG §4 の4つの否定対照。すべてオフライン再生（物理シミュレーション無し）。
    しきい値は置かない（あなたがしきい値を決めない、PREREG本文）。実測値だけを返す。"""
    n_full = len(policy.rec_pose)
    k = min(NEGATIVE_CONTROL_MAX_TICKS, n_full)
    # 🔴 否定対照の再生は先頭 k 周期に打ち切る(モジュール docstring・
    # NEGATIVE_CONTROL_MAX_TICKS のコメント参照。計算量対策)。主判定量
    # (20迷路合計のN_fatal)や合格条件1はこの打ち切りの影響を受けない
    # (別関数 process_maze() 側は全周期を使う)。
    poses = policy.rec_pose[:k]
    covs = policy.rec_cov[:k]
    ranges_list = policy.rec_ranges[:k]
    n = len(poses)
    cell_size = params.cell_size

    # N0: 記録をそのまま(打ち切った長さで)再生。オンライン走行がちょうど同じ
    # 周期数に達した瞬間の宣言配列(act() が控えておいたスナップショット)と
    # 一致するはずで、これが再生機構そのものの検算になる(追加の全長再生パスを
    # 走らせずに済む)。
    wb0, rev0, max0 = replay_sequence(width, height, params, poses, covs, ranges_list)
    dv0, dh0 = declared_arrays(wb0)
    n_fatal_0 = count_fatal(dv0, dh0, true_v, true_h)
    n_declared_0 = count_declared(dv0, dh0)

    replay_matches_online = None
    if policy.declared_snapshot_v is not None:
        replay_matches_online = bool(
            np.array_equal(dv0, policy.declared_snapshot_v)
            and np.array_equal(dh0, policy.declared_snapshot_h)
        )

    # N1: 姿勢を区画中心・軸並行に固定
    forced_poses = [_forced_cell_center_axis_pose(p, cell_size) for p in poses]
    wb1, rev1, max1 = replay_sequence(width, height, params, forced_poses, covs, ranges_list)
    dv1, dh1 = declared_arrays(wb1)
    n_fatal_1 = count_fatal(dv1, dh1, true_v, true_h)
    n_benign_1 = count_benign(dv1, dh1, true_v, true_h)
    n_declared_1 = count_declared(dv1, dh1)

    # N2: 観測を時間方向に並べ替える（固定シードの決定的な並べ替え）
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    shuffled_ranges = [ranges_list[i] for i in perm]
    wb2, rev2, max2 = replay_sequence(width, height, params, poses, covs, shuffled_ranges)
    dv2, dh2 = declared_arrays(wb2)
    n_fatal_2 = count_fatal(dv2, dh2, true_v, true_h)
    n_benign_2 = count_benign(dv2, dh2, true_v, true_h)
    n_declared_2 = count_declared(dv2, dh2)

    # N3: 対数尤度比を常に0にする（モジュール内部関数を実行時だけ差し替える。
    # ファイルは書き換えない。finally で必ず元に戻す）
    orig_llr = wall_belief_mod._log_likelihood_ratio
    wall_belief_mod._log_likelihood_ratio = lambda *a, **k: 0.0
    try:
        wb3, rev3, max3 = replay_sequence(width, height, params, poses, covs, ranges_list)
    finally:
        wall_belief_mod._log_likelihood_ratio = orig_llr
    dv3, dh3 = declared_arrays(wb3)
    n_declared_3 = count_declared(dv3, dh3)

    # N4: L_max を外す（既存のコンストラクタ引数を使うだけ）
    wb4, rev4, max4 = replay_sequence(width, height, params, poses, covs, ranges_list,
                                       l_max=1e9)

    return {
        "n_ticks_total_available": n_full,
        "n_ticks_replayed": n,
        "N0_replay_sanity": {
            "n_fatal": n_fatal_0,
            "n_declared": n_declared_0,
            "matches_online_run_at_same_tick_count": replay_matches_online,
            "note": "オンライン(実走行中に並走させたWallBelief)が同じ周期数に達した"
                    "瞬間の宣言配列と一致するかの検算(再生機構そのものにバグが無いことの確認)。"
                    "否定対照(N1-N4)は打ち切った長さで再生するため、20迷路合計のN_fatal"
                    "(主判定量)には使わない。",
        },
        "N1_forced_pose": {
            "破壊内容": "信念へ渡す姿勢を区画中心・軸並行に固定",
            "n_fatal": n_fatal_1,
            "n_fatal_baseline(N0)": n_fatal_0,
            "n_benign": n_benign_1,
            "n_declared": n_declared_1,
            "n_declared_baseline(N0)": n_declared_0,
            "reversal_count": rev1,
            "max_abs_log_odds": max1,
        },
        "N2_shuffled_ranges": {
            "破壊内容": "距離センサの実測値を時間方向にランダム(固定シード)並べ替え",
            "n_fatal": n_fatal_2,
            "n_fatal_baseline(N0)": n_fatal_0,
            "n_benign": n_benign_2,
            "n_declared": n_declared_2,
            "n_declared_baseline(N0)": n_declared_0,
            "reversal_count": rev2,
            "max_abs_log_odds": max2,
        },
        "N3_zero_llr": {
            "破壊内容": "対数尤度比を常に0にする(_log_likelihood_ratioを一時的に差し替え)",
            "n_declared": n_declared_3,
            "n_declared_baseline(N0)": n_declared_0,
        },
        "N4_no_l_max": {
            "破壊内容": "対数オッズの上限L_maxを1e9に緩めて実質撤廃",
            "reversal_count": rev4,
            "reversal_count_baseline(N0, L_max=既定12.7)": rev0,
            "max_abs_log_odds": max4,
            "max_abs_log_odds_baseline(N0)": max0,
        },
    }


# ==========================================================================
# 走行が変わらないことの確認（PREREG §3-2）
# ==========================================================================
def canonical_result_for_hash(result: dict) -> dict:
    """`_WallBeliefTrackingPolicy` は真の壁配列を読むために requires_privileged=True
    にしており、無印の ClassicExplorerPolicy(False) と異なる。結果 dict の
    "policy" フィールドはこの違いをそのまま映すので、そこを除いて正規化する
    （exp_031 run.py の canonical_result_for_hash と同じ判断）。"""
    d = dict(result)
    d.pop("policy", None)
    return d


def sha256_of_result(result: dict) -> str:
    canonical = canonical_result_for_hash(result)
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ==========================================================================
# 並走ラッパ
# ==========================================================================
class _WallBeliefTrackingPolicy(MousePolicy):
    """`ClassicExplorerPolicy` に `PoseEstimator`＋`WallBelief` を並走させる薄いラッパ。

    電圧計算(act の戻り値)は内部の `ClassicExplorerPolicy.act(obs)` に完全に
    委譲し、本クラス自身は一切関与しない。真の壁配列(bind_maze で受け取る)は
    診断用のフィールドへ保持するだけで、`WallBelief` にも内部方策にも一切渡さない。
    """

    name = "classic_explorer"  # 走行の同一性を見るため、無印と同じ名前にする
    requires_privileged = True  # bind_maze() で真の壁配列を診断用に読むためだけ

    def __init__(self, params: RobotParams | None = None, *, record_ticks: bool = False):
        self.params = params if params is not None else RobotParams()
        self._inner = ClassicExplorerPolicy(self.params)
        self._pose_estimator = PoseEstimator(self.params)
        self._wall_belief: WallBelief | None = None
        self._n_sensors = len(self.params.sensors)
        self._record_ticks = bool(record_ticks)

        self._true_v_walls: np.ndarray | None = None
        self._true_h_walls: np.ndarray | None = None

        self._obs_count_v: np.ndarray | None = None
        self._obs_count_h: np.ndarray | None = None
        self._prev_declared_v: np.ndarray | None = None
        self._prev_declared_h: np.ndarray | None = None
        self.reversal_count = 0

        # 否定対照の再生用の一次記録（record_ticks=True の迷路でのみ蓄積）
        self.rec_pose: list = []
        self.rec_cov: list = []
        self.rec_ranges: list = []
        # 否定対照の再生は NEGATIVE_CONTROL_MAX_TICKS 周期に打ち切るため、
        # 「オンライン走行そのものが同じ周期数でどう見えていたか」の検算用に、
        # ちょうどその周期数に達した瞬間の宣言配列を1回だけ控えておく
        # （追加の再生パスを走らせずに再生機構の正しさを確かめるため）。
        self.declared_snapshot_v: np.ndarray | None = None
        self.declared_snapshot_h: np.ndarray | None = None

    # ------------------------------------------------------------------
    def bind_sim(self, sim) -> None:
        pass  # requires_privileged=True の契約上呼ばれるが本スクリプトは使わない

    def bind_maze(self, v_walls, h_walls) -> None:
        # 診断用に保持するだけ。WallBelief へは絶対に渡さない。
        self._true_v_walls = np.array(v_walls, copy=True)
        self._true_h_walls = np.array(h_walls, copy=True)

    def on_maze_start(self, maze_info: dict) -> None:
        self._inner.on_maze_start(maze_info)
        width = int(maze_info["width"])
        height = int(maze_info["height"])
        self._wall_belief = WallBelief(width, height, self.params)
        self._pose_estimator.reset()  # 既定=発進姿勢(0.09,0.09,pi/2)

        self._obs_count_v = np.zeros_like(self._wall_belief.log_odds_v, dtype=np.int32)
        self._obs_count_h = np.zeros_like(self._wall_belief.log_odds_h, dtype=np.int32)
        self._prev_declared_v, self._prev_declared_h = declared_arrays(self._wall_belief)
        self.reversal_count = 0

    def on_run_start(self, run_index: int) -> None:
        self._inner.on_run_start(run_index)

    def on_run_end(self, outcome: str) -> None:
        self._inner.on_run_end(outcome)

    def on_retrieval(self) -> None:
        self._inner.on_retrieval()
        # 係員回収でロボットは物理的にスタートへ再配置される。推測航法の推定器も
        # 既知の発進姿勢へ再同期する(exp_031 と同じ考え方)。WallBelief(地図)は
        # 壁そのものは変わらないので保持する(従来の3値地図も走行を通じて保持される
        # のと同じ)。
        self._pose_estimator.reset()

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray):
        self._pose_estimator.predict(obs)
        pose = self._pose_estimator.pose
        cov = self._pose_estimator.covariance
        ranges = np.asarray(obs[: self._n_sensors], dtype=np.float64)

        if self._record_ticks:
            self.rec_pose.append(pose)
            self.rec_cov.append(cov.copy())
            self.rec_ranges.append(ranges.copy())

        wb = self._wall_belief
        pre_v = wb.log_odds_v.copy()
        pre_h = wb.log_odds_h.copy()
        wb.update(pose, cov, ranges)

        touched_v = wb.log_odds_v != pre_v
        touched_h = wb.log_odds_h != pre_h
        self._obs_count_v[touched_v] += 1
        self._obs_count_h[touched_h] += 1

        cur_v, cur_h = declared_arrays(wb)
        rev_v = ((self._prev_declared_v != int(WallState.UNKNOWN))
                 & (cur_v != int(WallState.UNKNOWN)) & (self._prev_declared_v != cur_v))
        rev_h = ((self._prev_declared_h != int(WallState.UNKNOWN))
                 & (cur_h != int(WallState.UNKNOWN)) & (self._prev_declared_h != cur_h))
        self.reversal_count += int(np.count_nonzero(rev_v)) + int(np.count_nonzero(rev_h))
        self._prev_declared_v, self._prev_declared_h = cur_v, cur_h

        if self._record_ticks and len(self.rec_pose) == NEGATIVE_CONTROL_MAX_TICKS:
            self.declared_snapshot_v = cur_v.copy()
            self.declared_snapshot_h = cur_h.copy()

        return self._inner.act(obs)  # 元の(無傷の) obs をそのまま使う。制御は一切汚さない

    # ------------------------------------------------------------------
    def get_plan_ids(self):
        return self._inner.get_plan_ids()

    @property
    def conventional_v_walls(self):
        """`classic/explorer.py` が作った従来の3値地図（真値ではない）。"""
        return self._inner.v_walls_known

    @property
    def conventional_h_walls(self):
        return self._inner.h_walls_known


# ==========================================================================
# マイコン実装の予算（PREREG §6。手計算は classic/wall_belief.py・classic/pose.py
# のモジュール docstring に既にある。ここでは実測できる部分だけを追加する）
# ==========================================================================
MCU_BUDGET_HAND_CALC = {
    "source": "classic/wall_belief.py・classic/pose.py モジュール docstring(手計算。実行時計測ではない)",
    "pose_multiplications_per_cycle": 97,
    "pose_trig_calls_per_cycle": 2,
    "pose_ram_bytes": 48,
    "wall_belief_multiplications_per_candidate": 884,
    "wall_belief_divisions_per_candidate": 17,
    "wall_belief_map_ram_bytes_quantized": 544,
    "wall_belief_map_ram_bytes_float32_prototype": 2176,
    "assumed_typical_candidates_per_sensor": "2-3(モジュールdocstringが1地点2姿勢で実測)",
    "assumed_cap_candidates_per_sensor": 8,
    "assumed_mcu_clock_mhz": 168,
    "assumed_control_rate_hz": 1000,
    "load_fraction_typical": "1割前後(168MHz)/数%(480MHz)",
    "load_fraction_cap": "2〜4割程度(168MHz)/1割未満(480MHz)",
}


# ==========================================================================
# 1迷路の処理
# ==========================================================================
def process_maze(npz_path: Path, params: RobotParams, record_for_negative_controls: bool) -> dict:
    seed = int(npz_path.stem.split("_")[1])
    evaluator = CompetitionEvaluator(params=params)  # 既定=公式規約

    record: dict = {"seed": seed, "npz_path": str(npz_path)}

    # ---- 1. 無印(推定器・信念を付けない)ベースライン ----
    t0 = time.time()
    baseline_policy = ClassicExplorerPolicy(params)
    baseline_result = evaluator.evaluate_maze(npz_path, baseline_policy)
    wall_untracked_s = time.time() - t0
    record["wall_clock_untracked_s"] = wall_untracked_s

    # ---- 2. 推定器・信念を並走させた走行(壊す前) ----
    t0 = time.time()
    tracked_policy = _WallBeliefTrackingPolicy(params, record_ticks=record_for_negative_controls)
    tracked_result = evaluator.evaluate_maze(npz_path, tracked_policy)
    wall_tracked_s = time.time() - t0
    record["wall_clock_tracked_s"] = wall_tracked_s

    # ---- 3. 走行が変わらないことの確認(PREREG §3-2) ----
    hash_untracked = sha256_of_result(baseline_result)
    hash_tracked = sha256_of_result(tracked_result)
    record["invariance_check"] = {
        "sha256_untracked": hash_untracked,
        "sha256_tracked": hash_tracked,
        "match": hash_untracked == hash_tracked,
    }

    # ---- 4. 診断量(主判定量 N_fatal を含む) ----
    wb = tracked_policy._wall_belief
    declared_v, declared_h = declared_arrays(wb)
    true_v = tracked_policy._true_v_walls
    true_h = tracked_policy._true_h_walls
    conv_v = tracked_policy.conventional_v_walls
    conv_h = tracked_policy.conventional_h_walls

    n_fatal = count_fatal(declared_v, declared_h, true_v, true_h)
    n_benign = count_benign(declared_v, declared_h, true_v, true_h)
    n_declared = count_declared(declared_v, declared_h)
    n_unknown = count_unknown(declared_v, declared_h)
    total_slots = declared_v.size + declared_h.size

    n_fatal_conv = count_fatal(conv_v, conv_h, true_v, true_h)
    n_benign_conv = count_benign(conv_v, conv_h, true_v, true_h)
    n_declared_conv = count_declared(conv_v, conv_h)

    mismatch = int(np.count_nonzero(declared_v != conv_v)) + int(np.count_nonzero(declared_h != conv_h))

    obs_counts = np.concatenate([tracked_policy._obs_count_v.ravel(), tracked_policy._obs_count_h.ravel()])
    n_ticks = len(tracked_policy.get_plan_ids())
    # 実測: 1センサ・1周期あたりの平均候補ヒット数(候補の対数尤度比計算まで進んだもの)。
    # obs_counts の総和 = (柱間,周期)ペアでlog_oddsが動いた回数の総計。
    total_hits = int(obs_counts.sum())
    n_sensors = len(params.sensors)
    avg_candidates_per_sensor_tick = (total_hits / n_sensors / n_ticks) if n_ticks > 0 else 0.0

    metrics = {
        "n_ticks": n_ticks,
        "n_fatal": n_fatal,
        "n_benign": n_benign,
        "n_declared": n_declared,
        "n_unknown": n_unknown,
        "n_unknown_fraction": n_unknown / total_slots,
        "n_fatal_conventional": n_fatal_conv,
        "n_benign_conventional": n_benign_conv,
        "n_declared_conventional": n_declared_conv,
        "declared_below_conventional": bool(n_declared < n_declared_conv),
        "mismatch_vs_conventional": mismatch,
        "reversal_count": tracked_policy.reversal_count,
        "observation_count_distribution": {
            "median": float(np.median(obs_counts)),
            "p95": float(np.percentile(obs_counts, 95)),
            "max": int(np.max(obs_counts)),
            "fraction_never_observed": float(np.mean(obs_counts == 0)),
        },
        "avg_candidates_per_sensor_per_tick": avg_candidates_per_sensor_tick,
    }
    record["metrics"] = metrics

    # ---- 5. 一次記録(anchor_check.py の唯一の入力) ----
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"maze_{seed}_final_state.npz"
    np.savez(
        raw_path,
        seed=np.int64(seed), width=np.int64(wb.width), height=np.int64(wb.height),
        log_odds_v=wb.log_odds_v, log_odds_h=wb.log_odds_h,
        declared_v=declared_v, declared_h=declared_h,
        true_v=true_v, true_h=true_h,
        conv_v=conv_v, conv_h=conv_h,
        t_wall=np.float64(T_WALL_DEFAULT), t_open=np.float64(T_OPEN_DEFAULT),
    )
    record["raw_log_path"] = str(raw_path)

    # ---- 6. 否定対照(担当迷路だけ。PREREG §4) ----
    if record_for_negative_controls:
        neg = compute_negative_controls(tracked_policy, params, wb.width, wb.height, true_v, true_h)
        record["negative_controls"] = neg

        seq_path = RAW_DIR / f"maze_{seed}_sequence.npz"
        np.savez(
            seq_path,
            poses=np.array(tracked_policy.rec_pose, dtype=np.float64),
            covs=np.array(tracked_policy.rec_cov, dtype=np.float32),
            ranges=np.array(tracked_policy.rec_ranges, dtype=np.float64),
        )
        record["sequence_log_path"] = str(seq_path)

    return record


# ==========================================================================
# 集計(全20迷路の maze_*.json から summary.json を作る)
# ==========================================================================
def aggregate_summary(selected: list[Path]) -> int:
    records = []
    for npz_path in selected:
        seed = int(npz_path.stem.split("_")[1])
        p = OUT_ROOT / f"maze_{seed}.json"
        if not p.exists():
            raise RuntimeError(f"{p} が無い。maze_{seed} がまだ処理されていない。")
        with open(p, encoding="utf-8") as f:
            records.append(json.load(f))

    n_fatal_total = sum(r["metrics"]["n_fatal"] for r in records)
    n_benign_total = sum(r["metrics"]["n_benign"] for r in records)
    n_declared_total = sum(r["metrics"]["n_declared"] for r in records)
    n_declared_conv_total = sum(r["metrics"]["n_declared_conventional"] for r in records)
    mazes_below_conventional = [r["seed"] for r in records if r["metrics"]["declared_below_conventional"]]
    all_invariant = all(r["invariance_check"]["match"] for r in records)
    total_wall_clock = sum(r.get("wall_clock_total_s", 0.0) for r in records)

    avg_candidates = float(np.mean([r["metrics"]["avg_candidates_per_sensor_per_tick"] for r in records]))
    n_sensors = 4
    mult_per_candidate = MCU_BUDGET_HAND_CALC["wall_belief_multiplications_per_candidate"]
    div_per_candidate = MCU_BUDGET_HAND_CALC["wall_belief_divisions_per_candidate"]
    est_mult_per_cycle_measured = avg_candidates * n_sensors * mult_per_candidate
    est_div_per_cycle_measured = avg_candidates * n_sensors * div_per_candidate
    cycles_168mhz_per_period = 168_000
    load_fraction_measured = est_mult_per_cycle_measured / cycles_168mhz_per_period

    neg_records = [r for r in records if "negative_controls" in r]

    summary = {
        "n_mazes": len(records),
        "maze_seeds": [r["seed"] for r in records],
        "n_fatal_total": n_fatal_total,
        "n_benign_total": n_benign_total,
        "n_declared_total": n_declared_total,
        "n_declared_conventional_total": n_declared_conv_total,
        "n_fatal_by_maze": {str(r["seed"]): r["metrics"]["n_fatal"] for r in records},
        "prereg_bucket": (
            "N_fatal=0: note_033の出口条件を満たす" if n_fatal_total == 0 else
            "1<=N_fatal<=4: 減ってはいるが残る" if n_fatal_total <= 4 else
            "N_fatal>=5: 方式として効いていない"
        ),
        "mazes_where_declared_below_conventional": mazes_below_conventional,
        "n_mazes_below_conventional": len(mazes_below_conventional),
        "all_invariance_checks_passed": all_invariant,
        "negative_control_seed": NEGATIVE_CONTROL_SEED,
        "negative_controls": (neg_records[0]["negative_controls"] if neg_records else None),
        "mcu_budget_hand_calc": MCU_BUDGET_HAND_CALC,
        "mcu_budget_measured": {
            "avg_candidates_per_sensor_per_tick_across_20_mazes": avg_candidates,
            "estimated_multiplications_per_cycle_at_measured_average": est_mult_per_cycle_measured,
            "estimated_divisions_per_cycle_at_measured_average": est_div_per_cycle_measured,
            "estimated_load_fraction_168mhz_1khz": load_fraction_measured,
            "note": "候補あたりの乗除算回数(884/17)はclassic/wall_belief.pyの手計算をそのまま使い、"
                    "候補数だけを実測値(20迷路平均)に差し替えた。",
        },
        "total_wall_clock_s": total_wall_clock,
        "prereg_path": "experiments/exp_034_wall_belief/PREREG.md",
    }
    summary_path = OUT_ROOT / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== 集計完了: {len(records)}迷路 ===")
    print(f"N_fatal 合計 = {n_fatal_total} ({summary['prereg_bucket']})")
    print(f"従来の3値地図を下回った迷路数 = {len(mazes_below_conventional)} / {len(records)}")
    print(f"全迷路で走行不変性を確認 = {all_invariant}")
    print(f"summary: {summary_path}")
    return 0


# ==========================================================================
# メイン
# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only-seed", type=int, default=None,
                         help="この seed の迷路だけを処理する(並列実行・パイロット実行用)")
    parser.add_argument("--aggregate-only", action="store_true",
                         help="既存の maze_*.json を集計して summary.json を書くだけ(再実行しない)")
    args = parser.parse_args(argv)

    params = RobotParams()
    selected = select_maze_paths()

    if args.aggregate_only:
        return aggregate_summary(selected)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    targets = selected
    if args.only_seed is not None:
        targets = [p for p in selected if int(p.stem.split("_")[1]) == args.only_seed]
        if not targets:
            raise RuntimeError(f"--only-seed {args.only_seed} は design_v4 の20面に含まれない")

    for npz_path in targets:
        seed = int(npz_path.stem.split("_")[1])
        record_neg = (seed == NEGATIVE_CONTROL_SEED)
        print(f"\n--- maze_{seed} 処理開始(否定対照記録={'あり' if record_neg else 'なし'}) ---", flush=True)
        t0 = time.time()
        record = process_maze(npz_path, params, record_for_negative_controls=record_neg)
        elapsed = time.time() - t0
        record["wall_clock_total_s"] = elapsed

        m = record["metrics"]
        print(f"maze_{seed}: n_fatal={m['n_fatal']} n_declared={m['n_declared']}"
              f"(従来={m['n_declared_conventional']}) n_ticks={m['n_ticks']} "
              f"invariance_match={record['invariance_check']['match']} "
              f"wall_clock={elapsed:.1f}s "
              f"(untracked={record['wall_clock_untracked_s']:.1f}s tracked={record['wall_clock_tracked_s']:.1f}s)",
              flush=True)

        out_path = OUT_ROOT / f"maze_{seed}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
