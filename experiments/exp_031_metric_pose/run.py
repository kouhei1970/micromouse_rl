"""
experiments/exp_031_metric_pose/run.py
================
exp_031（メトリック姿勢の並走）の測定スクリプト。判定条文は
`experiments/exp_031_metric_pose/PREREG.md`（結果を見る前に凍結済み）。

やること（PREREG §1・§2）: `classic/pose.py` の `PoseEstimator`（推測航法のみ・
観測更新なし）を、`classic/policy.py` の `ClassicExplorerPolicy`（現行の
古典ベースライン）に**並走**させ、真値姿勢（`MouseSim.privileged_pose()`）と
突き合わせて区画取り違え率 $r_{cell}$ を測る。**推定器の出力は制御に一切
使わない**。真値は診断だけに使い、推定器へは絶対に渡さない
（`_PoseTrackingPolicy.act()` 参照。推定器の `predict()` に渡す配列と、
内部の `ClassicExplorerPolicy.act()` に渡す配列は完全に分離している）。

【並走のさせ方（禁止事項: `competition/` 配下は 1 行も変更しない）】
`competition/policy_interface.py` の `MousePolicy` を継承した薄いラッパ
`_PoseTrackingPolicy` を本ファイル内に定義し、`competition/evaluator.py` の
`CompetitionEvaluator.evaluate_maze()` へ**そのまま**渡す。これは
`tests/test_classic_localization.py`（`_ToggleableExplorerPolicy`、
`DESIGN_V4_DIR` を使う `reach_comparison` フィクスチャ）に前例がある方式で、
`evaluator.evaluate_maze(maze_npz_path, policy)` を呼ぶだけなので
`competition/` を一切書き換えずに済む。

もう一つの候補（実験スクリプトが自前でシミュレータの繰り返しを書く）は
採らなかった。理由: `CompetitionEvaluator` は持ち時間・最大走行回数・
FREE/RUN_ACTIVE の状態機械・係員回収・ゴール停止判定など競技規約の細部
（`docs/RESEARCH_PLAN.md` §2）を実装しており、自前ループで再実装すると
**それ自体を検査し直す必要が生じる**（車輪の再発明であり、事前登録した
「1 実験 1 変更」の外側の変更を持ち込むリスクがある）。`MousePolicy` を
薄く包むだけなら、評価器のどの分岐からも `policy.act(obs)` は「観測を
渡して電圧を受け取るだけ」の口として一様に扱われる（ゴール停止後の
静止ホールド中の再呼び出し `competition/evaluator.py:645` 付近も含む）ので、
競技ハーネスの挙動を一切変えずに済む。

【真値の読み方】
`requires_privileged = True` にし、評価器の `bind_sim(sim)` 呼び出し
（`competition/evaluator.py:524-526`）で `MouseSim` の参照を受け取る。
`bind_sim`/`bind_maze` は評価器がシミュレータを構築した**直後・
走行開始前**に一度呼ばれるだけの通知であり、シミュレータ自体には何も
書き込まない（参照を保持するだけ）。したがって `requires_privileged` を
True にしても、`ClassicExplorerPolicy` に渡る観測列・返す電圧列は
一切変化しない（下記「走行が変わらないことの確認」で実測により確かめる）。

【局面の求め方(探索/帰還/最短走行)】
`ClassicExplorerPolicy` の内部状態（`_explorer.phase`)を直接読まず、
既存の公開 API `get_plan_ids()` が返す、ティックごとの計画識別子
（`classic/explorer.py` の `_phase_prefix()` が付ける接頭辞: "explore:"・
"return:"・"fast:"・"return2:"）を使う。これは `classic/checks.py` の
`plan_adherence()` が使っているのと同じ一次記録であり、私的属性への
アクセスを避けられる。

【「探索走行」の終わりの定義（判断が要った点。§進め方 手順どおり明記する）】
「最短走行の局面へ移った最初のティックまで」＝ **plan_id の接頭辞が
初めて "fast" になったティックの直前まで**、と定義する（1 度も "fast" に
ならなければ走行全体を採る）。"return2"（最短走行後の帰路）は "fast" より
後にしか起こらないので、この定義で境界は一意に決まる。

　🔴 判断が割れた理由: PREREG は局面を「探索・帰還・最短走行」の 3 つに
　分けているが（探索=explore、帰還=return、最短走行={fast, return2}）、
　主判定量の分母である「探索走行」が figure {explore} だけを指すのか
　{explore, return}（地図を作り切るまでの全期間、マイクロマウス業界の
　慣用で「探索走行」は帰路を含むことが多い）を指すのか、条文の字面からは
　一意に決まらない。本スクリプトは後者（{explore, return} を合わせた
　「最短走行が始まる前の全期間」）を採用した。教授セッションの検収を
   求める。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.policy import ClassicExplorerPolicy  # noqa: E402
from classic.pose import PoseEstimator  # noqa: E402
from competition.evaluator import CompetitionEvaluator  # noqa: E402
from competition.policy_interface import MousePolicy  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

DESIGN_V4_DIR = REPO_ROOT / "competition" / "mazes" / "design_v4"
OUT_ROOT = REPO_ROOT / "outputs" / "exp_031_metric_pose"
RAW_DIR = OUT_ROOT / "raw"

# 評価用に予約された seed 範囲（docs/RESEARCH_PLAN.md §2・§9-7 が正）。
# design_v4 は古典の設計帯 41000〜61000 のはずだが、念のため実測で検査する。
RESERVED_SEED_RANGE = (1000, 40999)

# PREREG §2: 古典の設計帯 seed 41000〜61000
DESIGN_BAND = (41000, 61000)


# ==========================================================================
# 対象迷路の選定（決定的規則。無作為に選ばない）
# ==========================================================================
def select_maze_paths() -> list[Path]:
    """design_v4 の全 20 面を seed 昇順に並べ、偶数番目（0,2,4,...,18 の
    10 個）を採る。

    先頭 10 個をそのまま採らない理由: design_v4 の seed は
    [41003, 41006, ..., 41284, 42073, ..., 50763] のように、前半に seed が
    密集している（41003〜41284 の 14 面が全体の 70%）。先頭 10 個だけを
    採ると 41003〜41049 の狭い範囲に偏り、42000 台・43000 台・49000〜50000台
    の面が 1 つも入らない。1 個おきに採れば、設計帯の広い範囲から
    満遍なく 10 面を拾える。
    """
    all_npz = sorted(DESIGN_V4_DIR.glob("maze_*.npz"),
                      key=lambda p: int(p.stem.split("_")[1]))
    if len(all_npz) != 20:
        raise RuntimeError(
            f"design_v4 の面数が想定(20)と違う: {len(all_npz)}。"
            "選定規則(偶数番目10面)の前提が崩れている可能性がある。実行を止める。"
        )
    selected = all_npz[0::2][:10]
    if len(selected) != 10:
        raise RuntimeError(f"選定した迷路が10面にならなかった: {len(selected)}")

    lo, hi = RESERVED_SEED_RANGE
    dlo, dhi = DESIGN_BAND
    for p in selected:
        seed = int(p.stem.split("_")[1])
        if lo <= seed <= hi:
            raise RuntimeError(
                f"評価用に予約された seed 範囲 [{lo},{hi}] に含まれる seed {seed} を"
                f"選んでしまった（測定禁止）。"
            )
        if not (dlo <= seed <= dhi):
            raise RuntimeError(
                f"seed {seed} が古典の設計帯 [{dlo},{dhi}] の外にある。"
            )
    return selected


# ==========================================================================
# 区画への変換（推定側・真値側で必ず同じ式を使う。PREREG §2）
# ==========================================================================
def poses_to_cells(xs: np.ndarray, ys: np.ndarray, cell_size: float):
    """floor(x/cell_size), floor(y/cell_size)。境界は下側を含む。"""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    cx = np.floor(xs / cell_size).astype(np.int64)
    cy = np.floor(ys / cell_size).astype(np.int64)
    return cx, cy


def wrap_angle_array(a: np.ndarray) -> np.ndarray:
    """角度を [-pi, pi) へ正規化する（`PoseEstimator._wrap` と同じ境界規約:
    ちょうど +pi は -pi 側に丸められる）。ベクトル化のため `np.mod` を使うが、
    `np.mod` は法が正なら常に非負を返すので、スカラー版の `math.fmod` 用の
    負側補正は不要（境界の実測一致は `tests/test_classic_pose.py` 側で
    スカラー版そのものが検査済み。ここでは同じ規約をベクトルへ広げるだけ）。
    """
    two_pi = 2.0 * math.pi
    a = np.asarray(a, dtype=np.float64)
    return np.mod(a + math.pi, two_pi) - math.pi


PHASE_LABELS = {
    "explore": "探索",
    "return": "帰還",
    "fast": "最短走行",
    "return2": "最短走行",
}


def phase_prefix(plan_id: str) -> str:
    prefix = plan_id.split(":", 1)[0]
    if prefix not in PHASE_LABELS:
        raise ValueError(
            f"未知の plan_id 接頭辞: {plan_id!r}(想定: explore/return/fast/return2)。"
            "classic/explorer.py の _phase_prefix() が変わった可能性がある。"
        )
    return prefix


def explore_window_length(plan_ids: list[str]) -> int:
    """「探索走行」の終わり = 接頭辞が初めて "fast" になったティックの直前まで。
    モジュール docstring の「探索走行の終わりの定義」参照。1 度も "fast" に
    ならなければ全長を返す。"""
    for i, pid in enumerate(plan_ids):
        if phase_prefix(pid) == "fast":
            return i
    return len(plan_ids)


# ==========================================================================
# 並走ラッパ（PoseEstimator を ClassicExplorerPolicy に並走させるだけ）
# ==========================================================================
class _PoseTrackingPolicy(MousePolicy):
    """`ClassicExplorerPolicy` に `PoseEstimator` を並走させる薄いラッパ。

    電圧計算(act の戻り値)は内部の `ClassicExplorerPolicy.act(obs)` に
    完全に委譲し、本クラス自身は一切関与しない。`PoseEstimator.predict()`
    に渡す配列と、内部方策に渡す配列は完全に別物であり
    （`corrupt` が None でない否定対照のときだけ前者を汚す。後者は常に
    無傷の `obs` を使う）、`requires_privileged=True` で得る真値
    (`sim.privileged_pose()`) は診断用のログへ積むだけで、推定器にも
    内部方策にも一切渡さない。
    """

    name = "classic_explorer"  # 走行の同一性を見るため、無印の ClassicExplorerPolicy と同じ名前にする
    requires_privileged = True  # 診断用真値(sim.privileged_pose())を読むためだけ。電圧計算には不使用

    def __init__(self, params: RobotParams | None = None, corrupt: str | None = None):
        if corrupt not in (None, "zero_wheel", "zero_gyro"):
            raise ValueError(f"未知の corrupt: {corrupt!r}")
        self.params = params if params is not None else RobotParams()
        self._inner = ClassicExplorerPolicy(self.params)  # 現行の古典ベースラインそのもの(既定構成)
        self._pose_estimator = PoseEstimator(self.params)
        self._corrupt = corrupt
        self._sim = None
        self._n_sensors = len(self.params.sensors)

        # ティックごとの一次記録(公開APIのみで構成: 局面は get_plan_ids() と
        # 突き合わせて後で付ける。act() の中では局面を読まない)。
        self.est_x: list[float] = []
        self.est_y: list[float] = []
        self.est_theta: list[float] = []
        self.true_x: list[float] = []
        self.true_y: list[float] = []
        self.true_theta: list[float] = []

    # ------------------------------------------------------------------
    def bind_sim(self, sim) -> None:
        self._sim = sim

    def bind_maze(self, v_walls, h_walls) -> None:
        pass  # 推定器は壁配列を一切参照しない設計(classic/pose.py docstring)

    def on_maze_start(self, maze_info: dict) -> None:
        self._inner.on_maze_start(maze_info)
        self._pose_estimator.reset()  # 既定=発進姿勢(0.09,0.09,pi/2)。評価器の full_reset と同じ前提

    def on_run_start(self, run_index: int) -> None:
        self._inner.on_run_start(run_index)

    def on_run_end(self, outcome: str) -> None:
        self._inner.on_run_end(outcome)

    def on_retrieval(self) -> None:
        self._inner.on_retrieval()
        # 係員回収でロボットは物理的にスタートへ再配置される
        # （`competition/evaluator.py` の `sim.reset_to_start(cell=(0,0), heading_deg=90.0)`）。
        # 実機の推測航法も、回収後は既知の発進姿勢へ再同期するのが自然
        # （運用上、回収は人手で行うので発進姿勢は既知）なので、ここで
        # 推定器もリセットする。この再同期は診断用の推定器の内部状態
        # だけに作用し、走行(電圧列)には一切影響しない。
        self._pose_estimator.reset()

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray):
        est_obs = obs
        if self._corrupt == "zero_wheel":
            # N1: 推定器に渡す車輪角速度だけを0に固定する(PREREG §4)。
            # 内部方策(self._inner)には無傷の obs をそのまま渡す。
            est_obs = np.array(obs, copy=True)
            est_obs[self._n_sensors + 6] = 0.0
            est_obs[self._n_sensors + 7] = 0.0
        elif self._corrupt == "zero_gyro":
            # N2: 推定器に渡すジャイロ z 成分だけを0に固定する(PREREG §4)。
            # `classic/pose.py predict()` が読むのは gyro_z(obs[n+5])だけなので
            # そこだけ汚す。
            est_obs = np.array(obs, copy=True)
            est_obs[self._n_sensors + 5] = 0.0

        self._pose_estimator.predict(est_obs)
        ex, ey, eth = self._pose_estimator.pose
        tx, ty, tth = self._sim.privileged_pose()

        self.est_x.append(ex)
        self.est_y.append(ey)
        self.est_theta.append(eth)
        self.true_x.append(tx)
        self.true_y.append(ty)
        self.true_theta.append(tth)

        return self._inner.act(obs)  # 元の(無傷の) obs をそのまま使う。制御は一切汚さない

    def get_plan_ids(self) -> list[str]:
        return self._inner.get_plan_ids()


# ==========================================================================
# 走行が変わらないことの確認（PREREG §3-1）
# ==========================================================================
def canonical_result_for_hash(result: dict) -> dict:
    """評価器の結果 dict から、方策の識別情報("policy"キー)を除いたものを返す。

    🔴 判断が要った点: `_PoseTrackingPolicy` は真値を読むために
    `requires_privileged=True` にしており、これは意図的に無印の
    `ClassicExplorerPolicy`（False）と異なる。結果 dict の "policy" フィールド
    (name/requires_privileged) はこの違いをそのまま映すので、そこを含めて
    ハッシュを取ると「ラッパを付けたかどうか」を比較しているだけになり、
    「走行(電圧列・結果)が変わっていないか」を見たいという本来の目的から
    外れる。走行そのもの(runs/incidents/timing/kpi)を比較するため、
    "policy" フィールドだけを除いて正規化する。教授セッションの検収を求める。
    """
    d = dict(result)
    d.pop("policy", None)
    return d


def sha256_of_result(result: dict) -> str:
    canonical = canonical_result_for_hash(result)
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ==========================================================================
# 診断量の計算（PREREG §7・主判定量 r_cell）
# ==========================================================================
def compute_metrics(policy: _PoseTrackingPolicy, cell_size: float) -> dict:
    plan_ids = policy.get_plan_ids()
    n = len(plan_ids)
    if n == 0:
        raise RuntimeError("走行が1ティックも記録されなかった(評価器が即座に終了した疑い)")
    if not (len(policy.est_x) == len(policy.true_x) == n):
        raise RuntimeError(
            f"一次記録の長さが一致しない: plan_ids={n} est={len(policy.est_x)} "
            f"true={len(policy.true_x)}(act() の呼び出し回数がずれている疑い)"
        )

    est_x = np.array(policy.est_x)
    est_y = np.array(policy.est_y)
    est_theta = np.array(policy.est_theta)
    true_x = np.array(policy.true_x)
    true_y = np.array(policy.true_y)
    true_theta = np.array(policy.true_theta)

    est_cx, est_cy = poses_to_cells(est_x, est_y, cell_size)
    true_cx, true_cy = poses_to_cells(true_x, true_y, cell_size)
    mismatch = (est_cx != true_cx) | (est_cy != true_cy)

    window = explore_window_length(plan_ids)
    if window == 0:
        raise RuntimeError("探索走行の長さが0ティックだった(境界の検出がおかしい)")

    r_cell = float(np.count_nonzero(mismatch[:window])) / float(window)

    dx = est_x[:window] - true_x[:window]
    dy = est_y[:window] - true_y[:window]
    th = true_theta[:window]
    lateral = -dx * np.sin(th) + dy * np.cos(th)
    longitudinal = dx * np.cos(th) + dy * np.sin(th)
    heading_err = wrap_angle_array(est_theta[:window] - th)

    lateral_rms = float(np.sqrt(np.mean(lateral ** 2)))
    longitudinal_rms = float(np.sqrt(np.mean(longitudinal ** 2)))
    heading_rms = float(np.sqrt(np.mean(heading_err ** 2)))

    # 誤差の時間発展(副次・PREREG §7): 探索走行の窓を5分割し、区間ごとのRMS
    n_bins = 5
    edges = np.linspace(0, window, n_bins + 1, dtype=int)
    time_evolution = []
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        time_evolution.append({
            "tick_start": int(a), "tick_end": int(b),
            "lateral_rms": float(np.sqrt(np.mean(lateral[a:b] ** 2))),
            "longitudinal_rms": float(np.sqrt(np.mean(longitudinal[a:b] ** 2))),
            "heading_rms": float(np.sqrt(np.mean(heading_err[a:b] ** 2))),
        })

    # 区画を取り違えた地点の分布(副次・真の区画側で集計。上位20区画のみ記録)
    mismatch_idx = np.nonzero(mismatch[:window])[0]
    cells = list(zip(true_cx[mismatch_idx].tolist(), true_cy[mismatch_idx].tolist()))
    top_cells = Counter(cells).most_common(20)

    # 局面ラベルの内訳(診断用)
    phase_counts = Counter(PHASE_LABELS[phase_prefix(pid)] for pid in plan_ids)

    return {
        "n_ticks_total": n,
        "n_ticks_explore_run": window,
        "r_cell": r_cell,
        "n_mismatch_ticks": int(np.count_nonzero(mismatch[:window])),
        "lateral_error_rms_m": lateral_rms,
        "longitudinal_error_rms_m": longitudinal_rms,
        "heading_error_rms_rad": heading_rms,
        "error_time_evolution": time_evolution,
        "mismatch_true_cells_top20": [
            {"cell": [int(c[0]), int(c[1])], "count": int(cnt)} for c, cnt in top_cells
        ],
        "phase_tick_counts": dict(phase_counts),
        "reached_fast_phase": bool(window < n),
    }


def total_displacement(xs: np.ndarray, ys: np.ndarray) -> float:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sum(np.hypot(dx, dy)))


# ==========================================================================
# 一次記録(生ログ)の保存 — anchor_check.py の唯一の入力
# ==========================================================================
def save_raw_log(policy: _PoseTrackingPolicy, seed: int, variant: str, cell_size: float) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"maze_{seed}_{variant}.npz"
    plan_ids = policy.get_plan_ids()
    np.savez(
        out_path,
        seed=np.int64(seed),
        cell_size=np.float64(cell_size),
        est_x=np.array(policy.est_x, dtype=np.float64),
        est_y=np.array(policy.est_y, dtype=np.float64),
        est_theta=np.array(policy.est_theta, dtype=np.float64),
        true_x=np.array(policy.true_x, dtype=np.float64),
        true_y=np.array(policy.true_y, dtype=np.float64),
        true_theta=np.array(policy.true_theta, dtype=np.float64),
        plan_id=np.array(plan_ids, dtype="<U40"),
    )
    return out_path


# ==========================================================================
# マイコン実装の予算(note_037 §13-4。classic/pose.py のコメントから転記。
# ここでは実測しない — 転記であることをキー名にも明示する)
# ==========================================================================
MCU_BUDGET = {
    "source": "classic/pose.py モジュール docstring(手計算。実行時計測ではない)",
    "multiplications_per_control_cycle": 97,
    "trig_calls_per_control_cycle": 2,
    "ram_bytes": 48,
    "assumed_mcu_clock_mhz": 168,
    "assumed_control_rate_hz": 1000,
    "estimated_load_fraction_low": 0.003,
    "estimated_load_fraction_high": 0.004,
    "note": "cos/sinをhalf=theta+dtheta/2の1点だけで評価し使い回す。"
            "共分散伝播はF/Vの疎性を使わない単純な密行列積として保守的に見積もった数え方"
            "(note_037 §13-2の見積もり方針)。",
}


# ==========================================================================
# 1 迷路の処理
# ==========================================================================
def process_maze(npz_path: Path, params: RobotParams, run_negative_controls: bool) -> dict:
    seed = int(npz_path.stem.split("_")[1])
    cell_size = params.cell_size
    evaluator = CompetitionEvaluator(params=params)  # 既定=公式規約(420秒・最大5走行)

    record: dict = {"seed": seed, "npz_path": str(npz_path)}

    # ---- 1. 無印(推定器を付けない)ベースライン ----
    t0 = time.time()
    baseline_policy = ClassicExplorerPolicy(params)
    baseline_result = evaluator.evaluate_maze(npz_path, baseline_policy)
    wall_untracked_s = time.time() - t0
    record["wall_clock_untracked_s"] = wall_untracked_s

    # ---- 2. 推定器を並走させた走行(壊す前) ----
    t0 = time.time()
    tracked_policy = _PoseTrackingPolicy(params, corrupt=None)
    tracked_result = evaluator.evaluate_maze(npz_path, tracked_policy)
    wall_tracked_s = time.time() - t0
    record["wall_clock_tracked_s"] = wall_tracked_s

    # ---- 3. 走行が変わらないことの確認(PREREG §3-1) ----
    hash_untracked = sha256_of_result(baseline_result)
    hash_tracked = sha256_of_result(tracked_result)
    record["invariance_check"] = {
        "sha256_untracked": hash_untracked,
        "sha256_tracked": hash_tracked,
        "match": hash_untracked == hash_tracked,
    }

    # ---- 4. 診断量(主判定量 r_cell を含む) ----
    metrics = compute_metrics(tracked_policy, cell_size)
    record["metrics"] = metrics

    raw_path = save_raw_log(tracked_policy, seed, "track", cell_size)
    record["raw_log_path"] = str(raw_path)

    # ---- 5. 否定対照(このメイン迷路でのみ実行。PREREG §4) ----
    if run_negative_controls:
        neg = {}

        t0 = time.time()
        n1_policy = _PoseTrackingPolicy(params, corrupt="zero_wheel")
        evaluator.evaluate_maze(npz_path, n1_policy)
        wall_n1_s = time.time() - t0
        n1_plan_ids = n1_policy.get_plan_ids()
        n1_window = explore_window_length(n1_plan_ids)
        n1_dist_m = total_displacement(
            np.array(n1_policy.est_x[:n1_window]), np.array(n1_policy.est_y[:n1_window]))
        save_raw_log(n1_policy, seed, "n1", cell_size)
        neg["N1_zero_wheel"] = {
            "wall_clock_s": wall_n1_s,
            "window_ticks": n1_window,
            "estimated_total_displacement_m": n1_dist_m,
            "condition": "total_displacement_m < 0.001",
            "fires": bool(n1_dist_m < 0.001),
        }

        t0 = time.time()
        n2_policy = _PoseTrackingPolicy(params, corrupt="zero_gyro")
        evaluator.evaluate_maze(npz_path, n2_policy)
        wall_n2_s = time.time() - t0
        n2_plan_ids = n2_policy.get_plan_ids()
        n2_window = explore_window_length(n2_plan_ids)
        n2_est_theta = np.array(n2_policy.est_theta[:n2_window])
        n2_true_theta = np.array(n2_policy.true_theta[:n2_window])
        n2_heading_err = wrap_angle_array(n2_est_theta - n2_true_theta)
        n2_heading_rms = float(np.sqrt(np.mean(n2_heading_err ** 2)))
        save_raw_log(n2_policy, seed, "n2", cell_size)
        baseline_heading_rms = metrics["heading_error_rms_rad"]
        ratio = (n2_heading_rms / baseline_heading_rms) if baseline_heading_rms > 0 else float("inf")
        neg["N2_zero_gyro"] = {
            "wall_clock_s": wall_n2_s,
            "window_ticks": n2_window,
            "heading_error_rms_rad_before_break": baseline_heading_rms,
            "heading_error_rms_rad_n2": n2_heading_rms,
            "ratio_to_before_break": ratio,
            "condition": "ratio_to_before_break >= 3 (暫定しきい値。PREREG §4。教授セッションが確定する)",
            "fires": bool(ratio >= 3.0),
        }

        record["negative_controls"] = neg

    return record


# ==========================================================================
# メイン
# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only-seed", type=int, default=None,
                         help="この seed の迷路だけを処理する(パイロット実行用)")
    parser.add_argument("--n-mazes", type=int, default=None,
                         help="選定した10面のうち先頭N面だけを処理する")
    parser.add_argument("--skip-negative-controls", action="store_true",
                         help="否定対照(N1/N2)をスキップする")
    args = parser.parse_args(argv)

    params = RobotParams()
    selected = select_maze_paths()

    if args.only_seed is not None:
        selected = [p for p in selected if int(p.stem.split("_")[1]) == args.only_seed]
        if not selected:
            raise RuntimeError(f"--only-seed {args.only_seed} は選定した10面に含まれない")
    elif args.n_mazes is not None:
        selected = selected[:args.n_mazes]

    negative_control_seed = int(selected[0].stem.split("_")[1])

    print(f"対象迷路: {[int(p.stem.split('_')[1]) for p in selected]}")
    print(f"否定対照(N1/N2)を実行する迷路: seed={negative_control_seed}")

    records = []
    t_all0 = time.time()
    for npz_path in selected:
        seed = int(npz_path.stem.split("_")[1])
        run_neg = (not args.skip_negative_controls) and (seed == negative_control_seed)
        print(f"\n--- maze_{seed} 処理開始(否定対照={'あり' if run_neg else 'なし'}) ---", flush=True)
        t0 = time.time()
        record = process_maze(npz_path, params, run_negative_controls=run_neg)
        elapsed = time.time() - t0
        record["wall_clock_total_s"] = elapsed
        records.append(record)

        m = record["metrics"]
        print(f"maze_{seed}: r_cell={m['r_cell']:.4f} "
              f"(探索走行 {m['n_ticks_explore_run']}/{m['n_ticks_total']} ティック) "
              f"invariance_match={record['invariance_check']['match']} "
              f"wall_clock={elapsed:.1f}s "
              f"(untracked={record['wall_clock_untracked_s']:.1f}s tracked={record['wall_clock_tracked_s']:.1f}s)",
              flush=True)

        out_path = OUT_ROOT / f"maze_{seed}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - t_all0

    r_cells = [r["metrics"]["r_cell"] for r in records]
    median_r_cell = float(np.median(r_cells)) if r_cells else None
    all_invariant = all(r["invariance_check"]["match"] for r in records)

    summary = {
        "n_mazes": len(records),
        "maze_seeds": [r["seed"] for r in records],
        "maze_selection_rule": "design_v4 の20面をseed昇順に並べ、偶数番目10個(0,2,4,...,18)を採る",
        "median_r_cell": median_r_cell,
        "r_cell_by_maze": {str(r["seed"]): r["metrics"]["r_cell"] for r in records},
        "all_invariance_checks_passed": all_invariant,
        "negative_control_seed": negative_control_seed,
        "negative_controls": next(
            (r["negative_controls"] for r in records if "negative_controls" in r), None),
        "mcu_budget": MCU_BUDGET,
        "total_wall_clock_s": total_elapsed,
        "prereg_path": "experiments/exp_031_metric_pose/PREREG.md",
        "explore_run_end_definition": (
            "plan_id の接頭辞が最初に 'fast' になったティックの直前まで"
            "(1度もfastにならなければ走行全体)。「探索走行」={explore, return} の合算。"
        ),
    }
    summary_path = OUT_ROOT / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== 完了: {len(records)}迷路 / 総実時間 {total_elapsed:.1f}s ===")
    print(f"median r_cell = {median_r_cell}")
    print(f"全迷路で走行不変性を確認 = {all_invariant}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
