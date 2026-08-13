"""
tests/test_exp012_goal_snapshot.py
===================================
exp_012 の train.py に入れた**稀少事象の重み退避**（研究計画書 §9-19）が、
実際に発火して重みを保存するかを確認する単体テスト。

出所: 条件 E の seed1 が 70 / 90 / 130 万歩でゴール率 0.05 を出したが、
チェックポイントは 40 万歩ごとで**その 4 点はいずれも 0.00**だったため、
**届いた方策の重みが谷間で失われ、事後に取り返せなかった**（design.md §6-2）。

**閾値を触らずに発火させる**ために、`evaluate_maze6` を**このテストの中でだけ**
差し替えて「ゴール率が非ゼロ」を返させる（本番の条件式 `goal_rate > 0.0` は
そのまま通る ＝ 検査が本番の経路を通ることを保証する）。

検査:
  (1) ゴール率が非ゼロの評価点で `<model_out の名前>_first_goal_{step}.zip` ができる
  (2) 保存されたファイルが PPO.load で読める（＝ 壊れていない）
  (3) run_summary.json に goal_snapshots が記録される
  (4) ゴール率が 0 のときは**保存されない**（＝ 常時保存になっていない）
  (5) --no-save-on-goal を渡すと保存されない

実行:
    .venv/bin/python tests/test_exp012_goal_snapshot.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402

TRAIN_PY = REPO_ROOT / "experiments/exp_012_continuous_potential/train.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("exp012_train", TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_stats(goal_rate: float) -> dict:
    """evaluate_maze6 の戻り値のうち ValidationCallback が読む項目だけを埋める。"""
    return dict(goal_rate=goal_rate, collision_rate=1.0 - goal_rate, timeout_rate=0.0,
                mean_goal_time_s=None, mean_sec_per_cell=None, mean_n_visited=1.0,
                mean_odom_error_m=0.0, sign_flip_rate_mean=0.0, i_rms_mean=0.0,
                failed_maze_seeds=[])


def run_case(goal_rates, extra_argv=()) -> dict:
    """goal_rates を順に返す評価器で短い学習を回し、生成物を集めて返す。"""
    mod = load_train_module()
    calls = {"n": 0}

    def stub_evaluate(policy_fn, **kwargs):
        gr = goal_rates[min(calls["n"], len(goal_rates) - 1)]
        calls["n"] += 1
        return fake_stats(gr)

    mod.evaluate_maze6 = stub_evaluate      # 本番の条件式はそのまま通す
    tmp = Path(tempfile.mkdtemp(prefix="exp012_snap_"))
    try:
        log_dir = tmp / "log"
        model_out = tmp / "models" / "cond_test.zip"
        argv = ["--total-steps", "2048", "--validation-every", "1024",
                "--n-envs", "1", "--seed", "0",
                "--log-dir", str(log_dir), "--model-out", str(model_out),
                *extra_argv]
        rc = mod.main(argv)
        assert rc == 0, f"train.main が {rc} を返した"
        snaps = sorted((model_out.parent).glob("cond_test_first_goal_*.zip"))
        summary = json.loads((log_dir / "run_summary.json").read_text(encoding="utf-8"))
        loadable = []
        for p in snaps:
            PPO.load(str(p), device="cpu")   # 壊れていれば例外
            loadable.append(p.name)
        return dict(snapshots=[p.name for p in snaps], loadable=loadable,
                    summary_snaps=summary.get("goal_snapshots", []),
                    n_eval_calls=calls["n"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ok = True
    print("=" * 74)
    print("exp_012 稀少事象の重み退避（§9-19）の発火確認")
    print("=" * 74)

    # (1)(2)(3) ゴール率が非ゼロなら保存され、読み込め、summary に載る
    r = run_case([0.05, 0.05])
    print(f"  非ゼロのとき: 保存 {r['snapshots']}  読込成功 {r['loadable']}  "
          f"summary の記録 {len(r['summary_snaps'])} 件（評価 {r['n_eval_calls']} 回）")
    if not r["snapshots"]:
        ok = False
        print("  🔴 FAIL: ゴール率が非ゼロなのに重みが保存されていない")
    if len(r["loadable"]) != len(r["snapshots"]):
        ok = False
        print("  🔴 FAIL: 保存された重みが読み込めない")
    if len(r["summary_snaps"]) != len(r["snapshots"]):
        ok = False
        print("  🔴 FAIL: run_summary.json の goal_snapshots と保存数が食い違う")
    if r["summary_snaps"]:
        s0 = r["summary_snaps"][0]
        if not (s0.get("goal_rate", 0) > 0 and "total_timesteps" in s0):
            ok = False
            print(f"  🔴 FAIL: goal_snapshots の中身が不足: {s0}")

    # (4) ゴール率 0 のときは保存されない（常時保存になっていないことの確認）
    r0 = run_case([0.0, 0.0])
    print(f"  ゼロのとき  : 保存 {r0['snapshots']}（空であること）")
    if r0["snapshots"]:
        ok = False
        print("  🔴 FAIL: ゴール率 0 でも保存されている（常時保存になっている）")

    # (5) --no-save-on-goal で止まる
    rn = run_case([0.05, 0.05], extra_argv=("--no-save-on-goal",))
    print(f"  --no-save-on-goal: 保存 {rn['snapshots']}（空であること）")
    if rn["snapshots"]:
        ok = False
        print("  🔴 FAIL: --no-save-on-goal を渡しても保存されている")

    print("=" * 74)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
