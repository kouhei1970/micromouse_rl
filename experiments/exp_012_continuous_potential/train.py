"""
experiments/exp_012_continuous_potential/train.py
===================================================
exp_012（ポテンシャル Φ の連続化）の PPO 学習スクリプト。

`experiments/exp_010_m2_0/train.py` を土台にし、**exp_010 との差は
Φ の連続化 1 点のみ**（`Maze6Env(..., continuous_potential=True)`）。
1 実験 1 変更の原則（研究計画書 §9）を守り、他のハイパーパラメータは
**exp_010 と完全に同一**にする:

| 項目 | 値 |
|---|---|
| 行動の高周波成分への罰 | α=0.5・k=8.7e-3 |
| 並列環境数 | 1 |
| γ | 0.995 |
| 総ステップ | 200 万 |
| 定期評価の間隔 | 10 万ステップごと |

**背景（詳細は `experiments/exp_012_continuous_potential/design.md`）**:
exp_010/011 の学習ログ実測から、M2-0（迷路）で学習が定着しない主因は
「γ による割引の潰れ」でも「終点の報酬順序」でもなく、**進捗報酬（ポテンシャル
整形）が入るステップの時間分布**だと特定された。区画単位の階段関数 Φ では
進捗報酬が全ステップの 5.3%（区画境界を跨ぐ歩）にしか入らず、残り 94.7% には
「動くな」としか読めない密な負の信号（時間罰＋滑らかさの罰）だけが掛かる。
M1（廊下）は Φ が連続だったため進捗報酬が 100% のステップに一様に入っていた。
本実験は Φ を区画中心・壁開口部中点を結ぶ折れ線への射影による連続量へ置き換え、
この時間分布を M1 と同じ「密」な状態に戻す（`mouse/maze6_env.py` に実装済み。
既定 `continuous_potential=False` では階段版のまま bit 単位で不変）。

事前登録した予測・打ち切り基準・支持/棄却の判定基準は design.md §5・§6 を正とする。
**⚠️ design.md §6 の打ち切り基準（検証帯の定期評価で 100 万歩までの 10 点すべてが
ゴール率 < 0.05 なら打ち切る）は、土台にした exp_010 の train.py に実装されておらず、
本スクリプトにも実装していない**（1 実験 1 変更のため、無断で追加せず指示者へ
報告済み）。運用時は `validation_history.json` を人手で確認し、必要なら
学習プロセスを手動で止めること。

**帯の分割**（研究計画書 §9-7）:

| 用途 | maze seed |
|---|---|
| 学習 | 8000 以降（予約帯は `Maze6Env._next_maze_seed()` が決定的に読み飛ばす） |
| 日常判断・チェックポイント選択 | 検証帯 **7000-7019** |
| gate 判定 | 評価帯 **6000-6019**（本スクリプトでは**測らない**） |

使い方:
    .venv/bin/python experiments/exp_012_continuous_potential/train.py --smoke     # 疎通確認
    .venv/bin/python experiments/exp_012_continuous_potential/train.py \
        --action-highpass-penalty 8.7e-3 --seed 1 \
        --log-dir logs/exp_012_cont_phi_seed1 --model-out models/exp_012_cont_phi_seed1.zip
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402
from mouse.maze6_env import Maze6Env  # noqa: E402
from mouse.obs_history import ObsHistoryWrapper, parse_lags  # noqa: E402
from mouse.maze6_eval import VALIDATION_MAZE_DIR, evaluate_maze6  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

GAMMA = 0.995
TRAIN_BASE_SEED = 8000          # 学習用 maze seed の起点（予約帯は環境側が読み飛ばす）
WORKER_SEED_STRIDE = 1_000_000

SMOKE_TOTAL_STEPS = 100_000
FULL_TOTAL_STEPS = 2_000_000    # M1 の 2 倍。迷路はエピソードが長く探索も要るため
VALIDATION_EVERY_STEPS = 100_000


# 条件（裁定 R21・R24）→ Φ の実現方法。**変えるのはここだけ**で、k・γ・並列数・
# 総ステップ・学習 seed は 3 条件で完全に同一にする。
#   E  … 区画ごとの明示式 ＋ 全降下隣接の min（検証的。H の判定はこれのみ）
#   C  … 配置空間の測地距離場（探索的）
#   Cp … C の Φ を面ごとの定数 ρ 倍して総整形量を E に揃えたもの（探索的。C'）
CONDITION_FLAGS = {
    "E": dict(continuous_potential=True),
    "C": dict(geodesic_potential=True),
    "Cp": dict(geodesic_potential=True, geodesic_rho_scale=True),
}

# 学習環境の版（裁定 2026-08-14。`experiments/env_v2_design.md` が正）。
#   v1 … 従来（中心方式のゴール判定・衝突で終了・上限 6000 歩）
#   v2 … **規約終端（機体全体の内包）＋衝突リスポーン＋上限 2000 歩**
# **報酬の項と係数は両版で同一**（報酬契約は凍結。変えているのは環境の側だけ）。
ENV_VERSION_FLAGS = {
    "v1": {},
    "v2": dict(goal_rule_containment=True, collision_respawn=True,
               episode_limit_steps=2000),
}

# 🔴 **評価環境の版**（裁定 2026-08-14。学習環境とは別に持つ）。
# 出所: exp_019 の投入 35 分後に、**v2 のフラグを学習環境にだけ渡していて、
# 定期評価は既定（v1 の尺）のまま**だったことが分かった。カード §4 は
# 「v2 では規約判定で数える」と宣言していたので、**条文と測定が食い違っていた**
# （`post_3_3` と同じ「是正が呼び出し側まで届いていない」型の欠陥）。
#
# **評価にリスポーンは入れない。**立て直しは**学習の道具**であって競技の規則ではなく、
# 評価は**競技の単発試行の意味論**（衝突したらその試行は失敗）で測るためである。
# 上限歩数も据え置く（評価の尺を学習の都合で動かさない）。
# こうすると衝突で終わる規則が v1 と同じなので、**中心が初めてゴール区画へ入った歩**から
# **v1 のゴール事象が厳密に再現**でき、規約ゴール率と v1 互換値を 1 回の評価で両取りできる。
EVAL_ENV_FLAGS = {
    "v1": {},
    "v2": dict(goal_rule_containment=True),
}


def make_env(rank: int, gamma: float, log_dir: Path, args):
    def _init():
        env = Maze6Env(
            mode="generate",
            base_seed=TRAIN_BASE_SEED + rank * WORKER_SEED_STRIDE,
            gamma=gamma,
            maze_mode=args.maze_mode,
            visit_bonus=args.visit_bonus,
            collision_penalty=args.collision_penalty,
            action_smooth_penalty=args.action_smooth_penalty,
            action_highpass_penalty=args.action_highpass_penalty,
            action_highpass_alpha=args.action_highpass_alpha,
            # exp_012: exp_010 との差は「Φ の実現方法」1 点のみ
            **CONDITION_FLAGS[args.condition],
            # 環境の版（v2 は 3 つを同時に切り替える。既定 v1 では何も渡さないのと同じ）
            **ENV_VERSION_FLAGS[args.env_version],
        )
        # exp_021: 観測履歴の連結。**渡さなければラップしない**ので、
        # 履歴なしの経路は exp_019・exp_020 と完全に同一である（カード §2-3）。
        lags = parse_lags(getattr(args, "obs_history", None))
        if lags:
            env = ObsHistoryWrapper(env, lags,
                                    sham=bool(getattr(args, "obs_history_sham", False)))
        return Monitor(env, filename=str(log_dir / f"env_{rank}"))
    return _init


class EpisodeStatsCallback(BaseCallback):
    """行動振動の指標と、エピソードごとの maze seed を記録する。

    M1 と同じ様式。`ep_rew_mean` は性能の判断に使わない（研究計画書 §9）。
    """

    def __init__(self, n_envs: int, seed_log_path: Path = None, n_track: int = 100):
        super().__init__()
        self.n_envs = n_envs
        self._prev_sign = np.zeros((n_envs, 2))
        self._flips = np.zeros((n_envs, 2), dtype=int)
        self._steps = np.zeros(n_envs, dtype=int)
        self._diff_sq = np.zeros(n_envs)
        self._prev_action = np.zeros((n_envs, 2))
        self._ep_flip_rates, self._ep_diff_rms = [], []
        self._n_track = n_track
        self._seed_log_path = Path(seed_log_path) if seed_log_path else None
        self._seed_file = None
        self._written = 0

    def _on_training_start(self) -> None:
        if self._seed_log_path is not None:
            self._seed_file = open(self._seed_log_path, "w", encoding="utf-8")

    def _on_step(self) -> bool:
        acts = np.asarray(self.locals["actions"], dtype=np.float64).reshape(self.n_envs, 2)
        s = np.sign(acts)
        self._flips += (s * self._prev_sign < 0).astype(int)
        self._prev_sign = s
        self._diff_sq += ((acts - self._prev_action) ** 2).sum(axis=1)
        self._prev_action = acts
        self._steps += 1

        dt = RobotParams().control_dt
        for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            if not done:
                continue
            t = max(self._steps[i] * dt, 1e-9)
            self._ep_flip_rates.append(self._flips[i] / t)
            self._ep_diff_rms.append(np.sqrt(self._diff_sq[i] / max(self._steps[i], 1)))
            if len(self._ep_flip_rates) > self._n_track:
                self._ep_flip_rates.pop(0)
                self._ep_diff_rms.pop(0)
            if self._seed_file is not None:
                rec = dict(
                    step=int(self.num_timesteps),
                    maze_seed=int(info.get("maze_seed", -1)),
                    outcome=("goal" if info.get("goal") else
                             "collision" if info.get("collision") else "timeout"),
                    n_visited=int(info.get("n_visited", 0)),
                    odom_error_m=float(info.get("odom_error_m", float("nan"))),
                )
                # 条件 C・C' の記録の義務（裁定 R24-1）: 学習迷路ごとの 1/ρ を残す。
                # 値は**実行時の場**から出たもの（= 1/ρ_field。design.md の命名。R39-3）。
                if "goal_contained_rule" in info:
                    # 規約判定（機体全体の内包）の並記（裁定 R42-4・§9-19 強化）
                    rec["goal_contained_rule"] = bool(info["goal_contained_rule"])
                if "n_respawn" in info:
                    rec["n_respawn"] = int(info["n_respawn"])
                # 🔴 R51 系の記録列（exp_020・**記録のみ**。env 側で貯めた値を書き写すだけで、
                # 学習にも乱数にも一切触れない）。合格条件は「軌跡が bit 一致」＝ 項目 8 と同型。
                #   R51-1: 最後のリスポーン以降の min D（P5 を学習中の走行で判定できるように）
                #   R51-2: 走行距離・各区画の初訪問の歩・**区画境界を跨いだ回数**
                #          （「遅い」の正体・割引後の訪問の取り分・Q3 の steps_per_entry）
                if "min_d_since_respawn" in info:
                    rec["min_d_since_respawn"] = int(info["min_d_since_respawn"])
                if "path_len_m" in info:
                    rec["path_len_m"] = float(info["path_len_m"])
                if "visit_steps" in info:
                    rec["visit_steps"] = [int(v) for v in info["visit_steps"]]
                if "cell_entries" in info:
                    rec["cell_entries"] = int(info["cell_entries"])
                if "delta_t_containment" in info:
                    rec["delta_t_containment"] = int(info["delta_t_containment"])
                if "geo_inv_rho" in info:
                    rec["geo_inv_rho_field"] = float(info["geo_inv_rho"])
                    rec["geo_rho_applied"] = float(info["geo_rho_applied"])
                    rec["geo_g_start_m"] = float(info["geo_g_start_m"])
                self._seed_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._written += 1
                if self._written % 200 == 0:
                    self._seed_file.flush()
            self._flips[i] = 0
            self._steps[i] = 0
            self._diff_sq[i] = 0.0
            self._prev_sign[i] = 0.0
            self._prev_action[i] = 0.0
        return True

    def _on_training_end(self) -> None:
        if self._seed_file is not None:
            self._seed_file.close()
            self._seed_file = None

    def summary(self) -> dict:
        if not self._ep_flip_rates:
            return dict(action_diff_rms=None, sign_flip_rate_left_per_s=None,
                        sign_flip_rate_right_per_s=None)
        fr = np.mean(np.array(self._ep_flip_rates), axis=0)
        return dict(action_diff_rms=float(np.mean(self._ep_diff_rms)),
                    sign_flip_rate_left_per_s=float(fr[0]),
                    sign_flip_rate_right_per_s=float(fr[1]))


def parse_d0_schedule(text):
    """`"400000:4,700000:6,1000000:9"` → [(400000, 4), (700000, 6), (1000000, None)] 形式。

    **段の境界と $D_0$ 上限の対**を、**歩数の昇順**で返す。**最後の段は上限なし**
    （`None`）であり、**明示的に書かない**（schedule の最後の境界を過ぎたら外れる）。
    **成績には一切依存しない**（`experiments/exp_020_distance_curriculum/card.md` §3-1）。
    """
    if not text:
        return None
    out = []
    for part in text.split(","):
        step_s, d0_s = part.split(":")
        out.append((int(step_s), int(d0_s)))
    out.sort(key=lambda t: t[0])
    return out


class CurriculumCallback(BaseCallback):
    """距離カリキュラム（exp_020）: **歩数で決まる固定 schedule** で $D_0$ 上限を切り替える。

    🔴 **成績非依存**である（`num_timesteps` だけで決まる）。**方策の出来に関わらず
    schedule は同一**なので、「学習信号の分布を成績で変える」懸念（AUDIT_022 §4）を
    **構成的に回避する**。

    schedule の読み方（`[(400000, 4), (700000, 6), (1000000, 9)]` の例）:

    | 歩数 | $D_0$ 上限 |
    |---|---|
    | 0 〜 400,000 | **4** |
    | 400,000 〜 700,000 | 6 |
    | 700,000 〜 1,000,000 | 9 |
    | 1,000,000 〜 | **なし（学習帯の全分布）** |

    切り替えは `env_method("set_d0_max", ...)` で**全ワーカへ同時に通知**する。
    **ロールアウトの粒度でしか見ないので、境界から数百歩ずれうる**（カード §3-3 で許容済み・
    **ずれの実測はログに残す**）。
    """

    def __init__(self, schedule, log_dir: Path):
        super().__init__()
        self.schedule = list(schedule)
        self.log_dir = Path(log_dir)
        self._applied = None            # いま適用している上限（未設定 = None は「上限なし」と別）
        self._applied_set = False
        self.switch_log = []

    def _d0_for(self, step: int):
        """その歩数で適用すべき $D_0$ 上限。

        schedule の対は **(その段の終わりの歩数, その段の $D_0$ 上限)** である
        （`"400000:4"` = **40 万歩までは上限 4**）。
        **どの段にも当たらない ＝ 最後の境界を過ぎた**ら **`None`（上限なし）**を返す。
        """
        for boundary, d0 in self.schedule:
            if step < boundary:
                return d0
        return None

    def _apply(self, value) -> None:
        self.training_env.env_method("set_d0_max", value)
        self._applied, self._applied_set = value, True
        rec = dict(num_timesteps=int(self.num_timesteps), d0_max=value)
        self.switch_log.append(rec)
        with open(self.log_dir / "curriculum_switches.json", "w", encoding="utf-8") as f:
            json.dump(self.switch_log, f, indent=2, ensure_ascii=False)
        print(f"[curriculum] t={self.num_timesteps} → D0 上限 = {value}", flush=True)

    def _on_training_start(self) -> None:
        self._apply(self._d0_for(0))

    def _on_rollout_start(self) -> None:
        want = self._d0_for(self.num_timesteps)
        if not self._applied_set or want != self._applied:
            self._apply(want)

    def _on_step(self) -> bool:
        return True


class ValidationCallback(BaseCallback):
    """**検証帯 7000-7019** でのゴール到達率を定期記録する。

    gate 判定（評価帯 6000-6019）はここでは**行わない**（研究計画書 §9-7）。
    """

    def __init__(self, eval_every_steps: int, log_dir: Path, gamma: float,
                 maze_mode: str, n_trials: int = 1, save_on_goal_path: Path = None,
                 fine_updates: int = 4, coarse_k: int = 2,
                 eval_env_kwargs: dict = None, env_wrapper=None):
        super().__init__()
        # exp_021: 学習環境に掛けたのと同じ観測ラッパを評価にも掛ける
        # （**既定 None = 従来どおり**。学習と評価で観測の形が食い違う事故を防ぐ）
        self.env_wrapper = env_wrapper
        # 評価環境の版（既定 None = 従来どおり）。上の EVAL_ENV_FLAGS を参照
        self.eval_env_kwargs = dict(eval_env_kwargs or {})
        self.eval_every = int(eval_every_steps)
        self.log_dir = Path(log_dir)
        self.gamma = float(gamma)
        self.maze_mode = maze_mode
        self.n_trials = int(n_trials)
        self.history = []
        self._next_at = self.eval_every
        # 🔴 稀少事象の重み退避（研究計画書 §9-19。2026-08-13 追加）。
        # 出所: exp_012 条件 E の seed1 が 70/90/130 万歩でゴール率 0.05 を出したが、
        # チェックポイントは 40 万歩ごとで、その 4 点はいずれも 0.00 だった
        # ＝ **届いた方策の重みが谷間で失われ、事後に取り返せなかった**。
        # **記録側だけの追加で、学習の力学には一切介入しない**（1 実験 1 変更の
        # 「介入」には当たらない。裁定 2026-08-13。design.md「記録の非対称」も参照）。
        self.save_on_goal_path = Path(save_on_goal_path) if save_on_goal_path else None
        self.saved_goal_snapshots = []
        # 🔴 §9-19 強化（環境 v2 項目 4。准教授 AUDIT_022 指摘 1）:
        # **陽性の直後を細かい粒度でも保存する**。
        # 当初案（陽性の直後 K=2 回の**評価点**）は**評価点の間隔が 10 万歩**なので、
        # **捕らえたい現象（896 歩で「届く状態」が失われる）の 1000 倍粗く**、
        # 現行の記録で既に分かっていることしか増えなかった。
        # → **細粒度（直後 `fine_window` 歩を `fine_every` 歩ごと）と
        #    粗粒度（直後 `coarse_k` 回の評価点）の両建て**にする。
        # 🔴 R51-3（exp_020・裁定 2026-08-14）: **退避の単位を「歩」から「PPO 更新境界」へ**。
        # 旧: 200 歩ごとに 2000 歩ぶん（＝ 10 点）。
        # **PPO は n_steps ごとにしか重みを更新しない**ので、**歩で刻んでも解像度は上がらない**
        # （exp_019 では 19 本を保存して、実際に異なる重みは 2〜4 種だった — AUDIT_028）。
        # **更新境界で取れば、保存は約 1/10 になり、かつ「どの更新の重みか」まで特定できる。**
        self.fine_updates = int(fine_updates)     # 陽性の直後に押さえる更新の回数
        self.coarse_k = int(coarse_k)             # 直後 2 回の評価点
        self._n_update = 0                        # 完了した PPO 更新の回数
        self._n_update_started = False
        self._fine_updates_left = 0
        # （旧・歩を単位にした細粒度保存の状態は R51-3 で撤去した）
        self._coarse_left = 0                     # 残りの粗粒度保存回数

    def _evaluate(self):
        t0 = time.time()
        s = evaluate_maze6(
            lambda o: self.model.predict(o, deterministic=True)[0],
            maze_dir=VALIDATION_MAZE_DIR, n_trials=self.n_trials, seed=0,
            gamma=self.gamma, maze_mode=self.maze_mode,
            env_kwargs=(self.eval_env_kwargs or None),
            env_wrapper=self.env_wrapper)
        rec = dict(total_timesteps=int(self.num_timesteps),
                   goal_rate=s["goal_rate"],
                   # 🔴 どちらの尺で測ったかを記録から復元できるようにする。
                   # v2 では goal_rate = 規約判定・center_rule = v1 互換の参考値
                   goal_rate_center_rule=s.get("goal_rate_center_rule"),
                   mean_delta_t_containment=s.get("mean_delta_t_containment"),
                   collision_rate=s["collision_rate"],
                   timeout_rate=s["timeout_rate"],
                   mean_goal_time_s=s["mean_goal_time_s"],
                   mean_sec_per_cell=s["mean_sec_per_cell"],
                   mean_n_visited=s["mean_n_visited"],
                   mean_odom_error_m=s["mean_odom_error_m"],
                   sign_flip_rate_mean=s["sign_flip_rate_mean"],
                   i_rms_mean=s["i_rms_mean"],
                   failed_maze_seeds=s["failed_maze_seeds"],
                   eval_wall_time_s=time.time() - t0)
        self.history.append(rec)
        self.logger.record("validation/goal_rate", rec["goal_rate"])
        if rec["sign_flip_rate_mean"] is not None:
            self.logger.record("validation/sign_flip_rate_mean", rec["sign_flip_rate_mean"])
        with open(self.log_dir / "validation_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"[validation] t={rec['total_timesteps']} "
              f"ゴール率={rec['goal_rate']:.2f} "
              f"訪問={rec['mean_n_visited'] or float('nan'):.1f} 区画 "
              f"({rec['eval_wall_time_s']:.1f} s)", flush=True)
        # 稀少事象（ゴール率が非ゼロ）を記録した時点の重みを退避する（§9-19）。
        if self.save_on_goal_path is not None and rec["goal_rate"] > 0.0:
            self._save_snapshot("first_goal", rec["total_timesteps"],
                                goal_rate=rec["goal_rate"])
            # **陽性の直後を細かい粒度でも押さえる**（896 歩の現象を括るため）
            self._fine_updates_left = self.fine_updates   # R51-3: 次の N 回の更新境界で取る
            self._coarse_left = self.coarse_k
        elif self.save_on_goal_path is not None and self._coarse_left > 0:
            # 陽性の**直後 K 回の評価点**（「その後も届き続けるか」を見るため）
            self._coarse_left -= 1
            self._save_snapshot("after_goal_eval", rec["total_timesteps"],
                                goal_rate=rec["goal_rate"])
        return rec

    def _save_snapshot(self, tag: str, step: int, goal_rate: float = None,
                       n_update: int = None) -> None:
        """重みを 1 点退避して記録する（§9-19）。

        `n_update` は**完了した PPO 更新の回数**（R51-3。歩ではなく更新で追跡するため）。
        """
        p = self.save_on_goal_path.with_name(
            f"{self.save_on_goal_path.stem}_{tag}_{step}.zip")
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(p))
        self.saved_goal_snapshots.append(
            dict(total_timesteps=int(step), tag=tag, goal_rate=goal_rate, path=str(p),
                 n_update=(None if n_update is None else int(n_update))))
        print(f"[validation] 🔴 §9-19 退避（{tag}）: {p}", flush=True)

    def _on_rollout_start(self) -> None:
        """PPO 更新の境界（R51-3）。

        SB3 の順序は **collect_rollouts → on_rollout_end → train()** なので、
        **次のロールアウトの開始時点の重みは「直前の更新を終えた重み」**である。
        したがって**ここで取れば、更新 1 回ぶんの粒度で退避できる**。
        **評価は走らせない**（評価は重いので、ここでは重みを取るだけ）。
        """
        if self._n_update_started:
            self._n_update += 1          # 2 回目以降のロールアウト開始 = 更新が 1 回終わった
        self._n_update_started = True
        if (self.save_on_goal_path is not None and self._fine_updates_left > 0
                and self._n_update > 0):
            self._fine_updates_left -= 1
            self._save_snapshot(f"after_goal_update{self._n_update}", self.num_timesteps,
                                n_update=self._n_update)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_at:
            self._evaluate()
            while self._next_at <= self.num_timesteps:
                self._next_at += self.eval_every
        return True

    def _on_training_end(self) -> None:
        if not self.history or self.history[-1]["total_timesteps"] != self.num_timesteps:
            self._evaluate()


def main(argv=None):
    p = argparse.ArgumentParser(description="exp_012（Φ の連続化）PPO 学習")
    p.add_argument("--total-steps", type=int, default=FULL_TOTAL_STEPS)
    p.add_argument("--smoke", action="store_true", help="10 万ステップで疎通確認")
    p.add_argument("--seed", type=int, default=0, help="PPO 本体の乱数 seed")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--n-envs", type=int, default=1,
                   help="並列環境数。exp_008 より並列 1 が実時間で最速")
    p.add_argument("--maze-mode", choices=["loop", "full"], default="loop")
    p.add_argument("--log-dir", type=str, default="logs/exp_012_cont_phi")
    p.add_argument("--model-out", type=str, default="models/exp_012_cont_phi.zip")
    p.add_argument("--validation-every", type=int, default=VALIDATION_EVERY_STEPS)
    p.add_argument("--visit-bonus", type=float, default=0.02)
    p.add_argument("--collision-penalty", type=float, default=-1.0)
    p.add_argument("--action-smooth-penalty", type=float, default=0.0,
                   help="exp_006 の ‖Δa‖² 版（M1 では案 3 に劣る。既定 0）")
    p.add_argument("--action-highpass-penalty", type=float, default=8.7e-3,
                   help="案 3。M1 で 3 基準を達成した値（既定 8.7e-3）")
    p.add_argument("--action-highpass-alpha", type=float, default=0.5)
    p.add_argument("--init-model", type=str, default=None,
                   help="初期重みにする学習済みモデル（微調整）")
    p.add_argument("--condition", choices=["E", "C", "Cp"], default="E",
                   help="Φ の実現方法。E=明示式（既定・条件 E）／C=配置空間の測地距離場／"
                        "Cp=C を面ごとの定数 ρ 倍したもの（条件 C'）")
    p.add_argument("--env-version", choices=["v1", "v2"], default="v1",
                   help="学習環境の版。v2 = 規約終端（機体全体の内包）＋衝突リスポーン＋"
                        "エピソード上限 2000 歩（裁定 2026-08-14。既定 v1 は従来どおり）")
    p.add_argument("--d0-schedule", type=str, default=None,
                   help="距離カリキュラム（exp_020）。'400000:4,700000:6,1000000:9' の形で "
                        "「その段の終わりの歩数:その段の D0 上限」を並べる。"
                        "**渡さなければ既定 off ＝ カリキュラム導入前と同一の経路**")
    p.add_argument("--obs-history", type=str, default=None,
                   help="観測履歴の連結（exp_021）。'1,2,4,8,16,32,64,128' の形で"
                        "遅れ［歩］を並べる。制御周期 10 ms なので 128 歩 = 1.28 秒。"
                        "**渡さなければ既定 off ＝ 履歴導入前と同一の経路**")
    p.add_argument("--obs-history-sham", action="store_true",
                   help="にせ履歴（exp_022）。遅れの位置に**現在の観測を複製**する。"
                        "次元もパラメータ数も同じまま履歴の情報だけがゼロになる。"
                        "**--obs-history と併用する**（渡さなければ既定 off ＝ exp_021 と同一）")
    p.add_argument("--fine-updates", type=int, default=4,
                   help="§9-19 の退避: 陽性の直後に押さえる PPO 更新の回数（R51-3）")
    p.add_argument("--no-save-on-goal", action="store_true",
                   help="検証でゴール率が非ゼロになった時点の重み退避（§9-19）を止める")
    args = p.parse_args(argv)

    total_steps = SMOKE_TOTAL_STEPS if args.smoke else args.total_steps
    n_envs = 1 if args.smoke else args.n_envs
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(i, args.gamma, log_dir, args) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns) if n_envs == 1 else SubprocVecEnv(env_fns)
    # 🔴 内訳は **1 時点ぶんの観測（17 要素）** の説明である。exp_021 の観測履歴を
    # 有効にすると、この内訳が (1 + 遅れの数) 回くり返されたベクトルになる（下の行で印字）。
    print(f"[train] 観測空間 = {vec_env.observation_space}"
          f"（1 時点の内訳: 距離4 + 差分4 + ジャイロ1 + 加速度2 + 車輪2 + 前回行動2 + ゴール相対2 = 17）")
    print(f"[train] 迷路モード = {args.maze_mode} / 訪問報酬 = {args.visit_bonus} "
          f"/ 衝突罰 = {args.collision_penalty}")
    print(f"[train] 行動の高周波成分への罰 k = {args.action_highpass_penalty}"
          f"（α = {args.action_highpass_alpha}）")
    print(f"[train] 並列環境数 = {n_envs}")
    # exp_021: 観測履歴の連結。**評価にも同じラッパを掛ける**（学習と評価で観測の形が
    # 食い違う事故を防ぐ。exp_019 の「評価だけ v1 の尺だった」欠陥と同じ型を封じる）。
    _obs_lags = parse_lags(args.obs_history)
    _obs_sham = bool(args.obs_history_sham)
    _obs_wrapper = ((lambda e: ObsHistoryWrapper(e, _obs_lags, sham=_obs_sham))
                    if _obs_lags else None)
    if _obs_lags:
        _n_out = int(vec_env.observation_space.shape[0])
        _n_base = _n_out // (1 + len(_obs_lags))       # ラッパの定義より割り切れる
        print(f"[train] 観測履歴の遅れ = {_obs_lags}"
              f"（観測 {_n_base} → {_n_out} 次元・"
              f"窓 = {max(_obs_lags)} 歩 = {max(_obs_lags) * 0.01:.2f} 秒）", flush=True)
        if _obs_sham:
            print("[train] にせ履歴 = 有効（遅れの位置に現在の観測を複製・情報ゼロ）", flush=True)
        else:
            print("[train] にせ履歴 = 無効（既定・exp_021 と同一経路）", flush=True)
    else:
        print("[train] 観測履歴 = 無効（既定・exp_019 と同一経路）", flush=True)
    # 🔴 帯の明示と安全弁（裁定 R40 条件 4・R11 項目 7）。学習に使う maze seed は
    # TRAIN_BASE_SEED 以降で、環境側が予約帯を決定的に読み飛ばす。**ここでは起点が
    # 凍結帯に入っていないことを明示的に確かめる**（道具の側の歯止め）。
    _train_seeds = [TRAIN_BASE_SEED + i * WORKER_SEED_STRIDE for i in range(n_envs)]
    print(f"[train] {describe_seeds(_train_seeds, namespace='maze6')}")
    assert_seeds_allowed(_train_seeds, namespace="maze6", purpose="train")
    print(f"[train] 条件 = {args.condition}"
          f"（Φ: {CONDITION_FLAGS[args.condition]}）")
    print(f"[train] 環境の版 = {args.env_version}"
          f"（{ENV_VERSION_FLAGS[args.env_version] or '従来どおり'}）")
    print(f"[train] 評価環境の版 = {args.env_version}"
          f"（{EVAL_ENV_FLAGS[args.env_version] or '従来どおり'}）"
          f" ※ 評価にリスポーンは入れない（競技の単発試行の意味論）")
    # 版の同一性を記録で担保する（exp_019 の条件。申告ではなく記録）
    try:
        _git_rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, cwd=REPO_ROOT).stdout.strip()
    except Exception:  # noqa: BLE001
        _git_rev = "unknown"
    print(f"[train] 投入版 git rev = {_git_rev}")

    if args.init_model:
        model = PPO.load(args.init_model, env=vec_env, seed=args.seed, verbose=1)
        print(f"[train] 初期重みを {args.init_model} から読み込み（微調整）")
    else:
        # ハイパーパラメータは M1（exp_003〜006）・exp_010 と同一。1 実験 1 変更を守る
        model = PPO("MlpPolicy", vec_env,
                    learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
                    gamma=args.gamma, gae_lambda=0.95, ent_coef=0.0,
                    policy_kwargs=dict(net_arch=[128, 128]),
                    seed=args.seed, verbose=1)
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))

    stats_cb = EpisodeStatsCallback(n_envs=n_envs,
                                    seed_log_path=log_dir / "episode_seeds.jsonl")
    val_cb = ValidationCallback(eval_every_steps=args.validation_every, log_dir=log_dir,
                                gamma=args.gamma, maze_mode=args.maze_mode,
                                save_on_goal_path=(None if args.no_save_on_goal
                                                   else Path(args.model_out)),
                                eval_env_kwargs=EVAL_ENV_FLAGS[args.env_version],
                                fine_updates=args.fine_updates,
                                env_wrapper=_obs_wrapper)
    ckpt_cb = CheckpointCallback(save_freq=max(400_000 // n_envs, 1),
                                 save_path=str(log_dir))

    # 距離カリキュラム（exp_020）。**渡さなければ None ＝ コールバック自体を作らない**ので、
    # **カリキュラム導入前と経路が完全に同一**になる（カード §2-3「不活性の bit 一致」）。
    d0_schedule = parse_d0_schedule(args.d0_schedule)
    callbacks = [ckpt_cb, stats_cb, val_cb]
    if d0_schedule is not None:
        callbacks.insert(0, CurriculumCallback(d0_schedule, log_dir))
        print(f"[train] 距離カリキュラム = {d0_schedule}（最後の段以降は上限なし）", flush=True)
    else:
        print("[train] 距離カリキュラム = 無効（既定・exp_019 と同一経路）", flush=True)

    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=callbacks)
    elapsed = time.time() - t0
    print(f"[train] total_steps={total_steps} n_envs={n_envs} "
          f"elapsed={elapsed:.1f}s steps/s={total_steps / max(elapsed, 1e-9):.1f}")

    model_out = Path(args.model_out)
    if args.smoke:
        model_out = model_out.with_name(model_out.stem + "_smoke.zip")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))
    print(f"[train] model saved: {model_out}")

    with open(log_dir / ("smoke_run_summary.json" if args.smoke else "run_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(dict(
            experiment=f"exp_012_cond{args.condition}", smoke=bool(args.smoke), total_steps=total_steps,
            d0_schedule=d0_schedule, fine_updates=args.fine_updates,
            n_envs=n_envs, seed=args.seed, gamma=args.gamma, maze_mode=args.maze_mode,
            visit_bonus=args.visit_bonus, collision_penalty=args.collision_penalty,
            action_smooth_penalty=args.action_smooth_penalty,
            action_highpass_penalty=args.action_highpass_penalty,
            action_highpass_alpha=args.action_highpass_alpha,
            init_model=args.init_model, train_base_seed=TRAIN_BASE_SEED,
            condition=args.condition, condition_flags=CONDITION_FLAGS[args.condition],
            env_version=args.env_version,
            env_version_flags=ENV_VERSION_FLAGS[args.env_version],
            eval_env_flags=EVAL_ENV_FLAGS[args.env_version],
            git_rev=_git_rev,
            goal_snapshots=val_cb.saved_goal_snapshots,
            elapsed_s=elapsed, steps_per_sec=total_steps / max(elapsed, 1e-9),
            validation_history=val_cb.history, **stats_cb.summary()),
            f, indent=2, ensure_ascii=False)
    vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
