"""
mouse/corridor_eval.py
================
M1（廊下追従・exp_001_corridor）gate 測定ハーネス。

spec_m1_corridor.md §3 の確定仕様: 「以後の全 exp が同じ物差しを使うため、ここを
凍結扱いにする」（competition/ の評価器と同じ思想）。評価コース 20 本（既定
assets/corridor/eval、seed 3000-3019）× 各 5 試行（初期擾乱を試行ごとに変える）
を実行し、gate 本体である no_contact_completion_rate（壁接触なしで完走した
試行 / 100）ほかの指標を算出する。

--------------------------------------------------------------------------
実装上の判断（spec に明記のない詳細）
--------------------------------------------------------------------------
1. 試行 seed は (base_seed, course_seed, trial_index) から決定的に導出する
   （spec: 「試行 index から決定的に seed 化」。base_seed・course_seed も混ぜて
   コース間で乱数列が相関しないようにしている）。
2. 1 制御ステップ内で goal と collision が同時に真になった場合は goal を優先する
   （competition/evaluator.py の優先順位規約 goal > collision と同じ考え方）。
   no_contact_completion_rate は、その優先順位で goal 判定された試行のうち
   さらに info["collision"] が False だったものだけを数える
   （「壁接触なしで完走」の定義に忠実にするため）。
3. tipped（転倒）は CorridorEnv 側で info["collision"] に畳み込まれている
   （mouse/corridor_env.py 参照）ため、本ハーネスの失敗様式は spec 通り
   collision / timeout の2区分のみで分類する。
4. 出力先の実験名（OutputManager の phase 名）は引数 output_name で受ける
   （2026-08-10 教授承認）。exp_002 までは "exp_001_corridor" にハードコード
   されており、別実験の成果物が同じパスへ上書きされる事故が起きたため
   （研究計画書 §9-1）。既定値は後方互換のため従来どおり。
   **測定内容・gate の定義は一切変えていない。**
5. 観測構成（obs_dist_diff）は評価対象モデルの観測次元と一致させる必要がある
   ため引数で受ける。gate の定義には影響しない（同じコース・同じ試行 seed・
   同じ判定式）。
"""
import math
from pathlib import Path

import numpy as np

from mouse.clearance import ClearanceMeter, hf_energy_ratio
from mouse.corridor_env import CorridorEnv
from common.output_manager import OutputManager

DEFAULT_N_TRIALS = 5
DEFAULT_GAMMA = 0.995
DEFAULT_COURSE_DIR = "assets/corridor/eval"
VALIDATION_COURSE_DIR = "assets/corridor/validation"   # 日常の判断用（seed 5000-5019）
DEFAULT_OUTPUT_NAME = "exp_001_corridor"


def _trial_seed(base_seed: int, course_seed: int, trial_index: int) -> int:
    """(base_seed, course_seed, trial_index) から決定的に試行用 seed を作る。"""
    return int(base_seed) * 1_000_000 + int(course_seed) * 100 + int(trial_index)


def _run_one_trial(env: CorridorEnv, policy_fn, trial_seed: int,
                    keep_trace: bool = False) -> dict:
    """1 試行を走らせて指標を返す。

    Args:
        keep_trace: True なら行動列・車輪角速度・姿勢の時系列を `trace` キーに残す。
            **電流の再計算には `action` と `wheel_omega` の両方が要る**ので両方持つ。

    2026-08-11 追加（新キーのみ。既存キーの意味・型は変えない）:
      n_turns, hf_ratio, i_rms, i_dc, i_ac, min_wall_clearance_m
    追加の計算は評価ループ内で既に手に入る量だけを使うので、実測で走行あたり
    +3〜5% の時間しかかからない（最小壁余裕の幾何計算が主）。
    """
    obs, info = env.reset(seed=trial_seed)
    prev_action = None
    n_steps = 0
    diff_sq_sum = 0.0
    sign_flips = [0, 0]
    outcome = None
    collision_flag = False
    # 経路中心線からの横偏差（2026-08-10 追加。滑らかにしたぶん壁へ寄っていないかを見る）
    lat_max = float(info.get("lateral_m", 0.0))
    lat_sq_sum = float(info.get("lateral_m", 0.0)) ** 2
    lat_count = 1

    p = env.params
    sim = env.sim
    wheel_adr = (sim._left_wheel_qvel_adr, sim._right_wheel_qvel_adr)
    meter = ClearanceMeter(sim)
    acts, currents, omegas, poses, times = [], [], [], [], []
    # 旋回回数: ヨー角の累積変化が 1 区画ぶんの旋回（±45°）を超えるたびに 1 とする。
    # 旋回は時間の主要因（L0-a の回帰で 0.85 s/旋回 対 0.71 s/歩）なのに未記録だった
    _, _, yaw_prev = sim.privileged_pose()
    yaw_acc, n_turns = 0.0, 0

    while True:
        action = np.asarray(policy_fn(obs), dtype=np.float64).reshape(-1)
        clipped = np.clip(action, -1.0, 1.0)

        if prev_action is not None:
            diff = clipped - prev_action
            diff_sq_sum += float(np.sum(diff ** 2))
            for k in (0, 1):
                if prev_action[k] * clipped[k] < 0.0:
                    sign_flips[k] += 1
        prev_action = clipped
        n_steps += 1

        volt = clipped * p.voltage_limit
        w0 = np.array([sim.data.qvel[wheel_adr[0]], sim.data.qvel[wheel_adr[1]]])
        obs, reward, terminated, truncated, info = env.step(action)
        w1 = np.array([sim.data.qvel[wheel_adr[0]], sim.data.qvel[wheel_adr[1]]])
        # モータ電流 I = (V − K_e·N·ω_w)/R。制御周期内で V は一定、ω は前後の平均で代表
        currents.append((volt - p.motor_Ke * p.gear_ratio * 0.5 * (w0 + w1)) / p.motor_R)
        acts.append(clipped.copy())
        meter.update()

        x, y, yaw = sim.privileged_pose()
        dyaw = math.atan2(math.sin(yaw - yaw_prev), math.cos(yaw - yaw_prev))
        yaw_prev = yaw
        if yaw_acc * dyaw < 0.0:        # 向きが変わったら積算をやり直す
            yaw_acc = 0.0
        yaw_acc += dyaw
        if abs(yaw_acc) >= math.pi / 4:
            n_turns += 1
            yaw_acc = 0.0
        if keep_trace:
            omegas.append(0.5 * (w0 + w1))
            poses.append((x, y, yaw))
            times.append(float(info.get("sim_time", 0.0)))

        lat = float(info.get("lateral_m", 0.0))
        lat_max = max(lat_max, lat)
        lat_sq_sum += lat * lat
        lat_count += 1

        if terminated or truncated:
            goal_flag = bool(info.get("goal"))
            collision_flag = bool(info.get("collision"))
            if goal_flag:
                outcome = "goal"
            elif collision_flag:
                outcome = "collision"
            elif truncated:
                outcome = "timeout"
            else:
                outcome = "collision"  # 保険（本来ここには来ない）
            break

    sim_time = float(info.get("sim_time", 0.0))
    n_cells = int(info.get("n_cells", 0))
    progress_m = float(info.get("progress_m", 0.0))
    mean_speed = (progress_m / sim_time) if sim_time > 0 else 0.0
    sec_per_cell = (sim_time / n_cells) if n_cells > 0 else None

    complete = (outcome == "goal")
    no_contact_complete = complete and not collision_flag

    cur_arr = np.array(currents) if currents else np.empty((0, 2))
    n_transitions = max(n_steps - 1, 1)
    action_diff_rms = math.sqrt(diff_sq_sum / n_transitions)
    sign_flip_rate_left = (sign_flips[0] / sim_time) if sim_time > 0 else 0.0
    sign_flip_rate_right = (sign_flips[1] / sim_time) if sim_time > 0 else 0.0

    return dict(
        outcome=outcome,
        complete=complete,
        no_contact_complete=no_contact_complete,
        n_steps=n_steps,
        sim_time=sim_time,
        n_cells=n_cells,
        progress_m=progress_m,
        mean_speed=mean_speed,
        sec_per_cell=sec_per_cell,
        action_diff_rms=action_diff_rms,
        sign_flip_rate_left=sign_flip_rate_left,
        sign_flip_rate_right=sign_flip_rate_right,
        lateral_max_m=lat_max,
        lateral_rms_m=math.sqrt(lat_sq_sum / max(lat_count, 1)),
        # --- 以下 2026-08-11 追加（新キーのみ）---------------------------
        n_turns=int(n_turns),
        hf_ratio=float(hf_energy_ratio(acts)) if acts else None,
        i_rms=float(np.sqrt((cur_arr ** 2).mean())) if len(cur_arr) else None,
        i_dc=float(np.abs(cur_arr.mean(axis=0)).mean()) if len(cur_arr) else None,
        i_ac=float(np.sqrt(np.maximum(
            (cur_arr ** 2).mean(axis=0) - cur_arr.mean(axis=0) ** 2, 0.0)).mean())
        if len(cur_arr) else None,
        min_wall_clearance_m=(float(meter.worst)
                              if math.isfinite(meter.worst) else None),
        trace=(dict(t=[round(v, 4) for v in times],
                    action=[[round(float(a[0]), 5), round(float(a[1]), 5)] for a in acts],
                    wheel_omega=[[round(float(w[0]), 4), round(float(w[1]), 4)]
                                 for w in omegas],
                    pose=[[round(v, 5) for v in q] for q in poses])
               if keep_trace else None),
    )


def evaluate_corridor(policy_fn, course_dir=DEFAULT_COURSE_DIR, n_trials=DEFAULT_N_TRIALS,
                       deterministic=True, seed=0, gamma=DEFAULT_GAMMA,
                       save_output=True, output_name=DEFAULT_OUTPUT_NAME,
                       obs_dist_diff: bool = False,
                       keep_traces: bool = True, model_sha256: str = "unknown") -> dict:
    """M1 gate 測定を実行する。

    Args:
        policy_fn: obs(np.ndarray, shape(13,)) -> action(array-like, shape(2,))。
            SB3 モデルなら `lambda o: model.predict(o, deterministic=True)[0]`。
        course_dir: 評価コース npz+xml が入ったディレクトリ（既定: 評価用20本）。
        n_trials: コースあたりの試行数（既定5。gate は 20コース×5試行=100試行）。
        deterministic: 記録用メタデータ（policy_fn 自体が決定的かどうかは
            呼び出し側が保証する。ここでは分岐しない）。
        seed: 試行 seed 導出の基準値。
        gamma: CorridorEnv に渡す gamma（gate 測定では報酬値そのものは
            使わないが、環境構築に必要なため PPO の gamma と同じ既定値を使う）。
        save_output: True なら OutputManager 経由で outputs/<output_name>/ に保存。
        output_name: 出力先の実験名（実験ごとに必ず変えること。研究計画書 §9-1）。
        obs_dist_diff: 評価対象モデルの観測構成。距離センサの1階差分を含む
            モデル（exp_003 以降）なら True。

    Returns:
        dict: no_contact_completion_rate（gate 本体）ほかの指標を含む summary。
    """
    course_dir = Path(course_dir)
    npz_paths = sorted(course_dir.glob("corridor_*.npz"))
    if not npz_paths:
        raise ValueError(f"評価コースが見つかりません: {course_dir}")

    course_seeds = [int(np.load(p)["seed"]) for p in npz_paths]

    per_course = []
    all_trials = []

    for course_seed in course_seeds:
        env = CorridorEnv(course_dir=course_dir, course_seeds=[course_seed],
                           max_cache=2, gamma=gamma, obs_dist_diff=obs_dist_diff)
        trial_results = []
        for trial_idx in range(n_trials):
            tseed = _trial_seed(seed, course_seed, trial_idx)
            res = _run_one_trial(env, policy_fn, tseed, keep_trace=keep_traces)
            trial_results.append(res)
            all_trials.append(res)
        env.close()

        per_course.append(dict(
            course_seed=course_seed,
            n_trials=n_trials,
            n_complete=sum(1 for r in trial_results if r["complete"]),
            n_no_contact_complete=sum(1 for r in trial_results if r["no_contact_complete"]),
            n_collision=sum(1 for r in trial_results if r["outcome"] == "collision"),
            n_timeout=sum(1 for r in trial_results if r["outcome"] == "timeout"),
            trials=trial_results,
        ))

    n_total = len(all_trials)
    n_complete_total = sum(1 for r in all_trials if r["complete"])
    n_no_contact_total = sum(1 for r in all_trials if r["no_contact_complete"])
    n_collision_total = sum(1 for r in all_trials if r["outcome"] == "collision")
    n_timeout_total = sum(1 for r in all_trials if r["outcome"] == "timeout")

    speeds = [r["mean_speed"] for r in all_trials if r["mean_speed"] > 0]
    sec_per_cell_list = [r["sec_per_cell"] for r in all_trials if r["sec_per_cell"] is not None]
    diff_rms_list = [r["action_diff_rms"] for r in all_trials]
    flip_left_list = [r["sign_flip_rate_left"] for r in all_trials]
    flip_right_list = [r["sign_flip_rate_right"] for r in all_trials]
    lat_max_list = [r["lateral_max_m"] for r in all_trials]
    lat_rms_list = [r["lateral_rms_m"] for r in all_trials]

    summary = dict(
        n_courses=len(course_seeds),
        n_trials_per_course=n_trials,
        n_total_trials=n_total,
        no_contact_completion_rate=(n_no_contact_total / n_total) if n_total else None,
        completion_rate=(n_complete_total / n_total) if n_total else None,
        collision_rate=(n_collision_total / n_total) if n_total else None,
        timeout_rate=(n_timeout_total / n_total) if n_total else None,
        mean_forward_speed_mps=float(np.mean(speeds)) if speeds else None,
        mean_sec_per_cell=float(np.mean(sec_per_cell_list)) if sec_per_cell_list else None,
        action_diff_rms_mean=float(np.mean(diff_rms_list)) if diff_rms_list else None,
        sign_flip_rate_left_mean=float(np.mean(flip_left_list)) if flip_left_list else None,
        sign_flip_rate_right_mean=float(np.mean(flip_right_list)) if flip_right_list else None,
        lateral_max_m_max=float(np.max(lat_max_list)) if lat_max_list else None,
        lateral_max_m_mean=float(np.mean(lat_max_list)) if lat_max_list else None,
        lateral_rms_m_mean=float(np.mean(lat_rms_list)) if lat_rms_list else None,
        deterministic=bool(deterministic),
        seed=int(seed),
        gamma=float(gamma),
        course_dir=str(course_dir),
        obs_dist_diff=bool(obs_dist_diff),
        per_course=per_course,
    )

    # --- 来歴（2026-08-11 追加）------------------------------------------
    # 「再実行すれば確かめられる」が成立するには、**当時のコードが特定できる**必要がある。
    summary["git_rev"] = _git_rev()
    summary["model_sha256"] = model_sha256
    summary["eval_schema_version"] = 2      # v2: 走行ごとの駆動系指標と trace を追加

    # --- 代表走行の trace を決定的な規則で選ぶ ----------------------------
    # ①完走走行のうち反転率が中央値のもの ②同・最大のもの（**指標が壊れるのは最悪側**）
    # ③失敗走行の 1 本目。条件ごとに 3 本。
    if keep_traces:
        summary["traces"] = _select_traces(per_course)
    for pc in per_course:                    # trace 本体は summary から落とす（肥大化を防ぐ）
        for tr in pc["trials"]:
            tr.pop("trace", None)

    if save_output:
        out_mgr = OutputManager(output_name)
        out_mgr.save_metrics(summary)
        out_mgr.finalize()
        summary["output_dir"] = str(out_mgr.archive_dir)

    return summary


def _git_rev() -> str:
    """現在の HEAD（dirty なら末尾に -dirty）。特定できなければ 'unknown'。"""
    import subprocess
    try:
        root = Path(__file__).resolve().parents[1]
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return "unknown"
        h = rev.stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                            capture_output=True, text=True, timeout=10)
        return h + ("-dirty" if st.stdout.strip() else "")
    except Exception:  # noqa: BLE001  来歴が取れなくても評価は続ける
        return "unknown"


def file_sha256(path) -> str:
    """モデルファイルの SHA-256（先頭 16 桁）。特定できなければ 'unknown'。"""
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


def _select_traces(per_course) -> dict:
    """代表走行 3 本を決定的な規則で選ぶ。

    ① 完走走行のうち符号反転率が**中央値**のもの（典型）
    ② 同**最大**のもの（**指標が壊れるのは最悪側**なので必ず残す）
    ③ 失敗走行の 1 本目（コース seed → 試行番号の順で最初のもの）
    """
    flat = [(pc["course_seed"], i, tr)
            for pc in per_course for i, tr in enumerate(pc["trials"])]
    ok = [(cs, i, tr) for cs, i, tr in flat if tr["no_contact_complete"]]
    out = {}
    if ok:
        ok_sorted = sorted(ok, key=lambda r: 0.5 * (r[2]["sign_flip_rate_left"]
                                                    + r[2]["sign_flip_rate_right"]))
        for tag, rec in (("median_flip", ok_sorted[len(ok_sorted) // 2]),
                         ("max_flip", ok_sorted[-1])):
            cs, i, tr = rec
            if tr.get("trace"):
                out[tag] = dict(course_seed=cs, trial_index=i, **tr["trace"])
    ng = [(cs, i, tr) for cs, i, tr in flat if not tr["no_contact_complete"]]
    if ng and ng[0][2].get("trace"):
        cs, i, tr = ng[0]
        out["first_failure"] = dict(course_seed=cs, trial_index=i, **tr["trace"])
    return out
