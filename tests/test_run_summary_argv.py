#!/usr/bin/env python3
"""`run_summary.json` に投入の設定が残ることの検査（R11・保守バッチ）。

Verify that the training run summary records the full invocation (argv and resolved args).

## なぜ要るか（欠落の経緯）

**exp_021（観測履歴）・exp_022（にせ履歴）・exp_023（再帰型方策）で追加した
`--obs-history` `--obs-history-sham` `--recurrent` `--lstm-hidden` `--respawn-reset` は、
`run_summary.json` に 1 つも記録されていなかった。**
そのため **「この走行がどの群か」の機械可読な記録が存在せず**、判定側は
**`train.log` の日本語 1 行**を読むしかなかった（脆い経路）。

**是正の設計（教授裁定 2026-08-15・准教授案）**: **項目を 1 つずつ足すのではなく
`argv` を丸ごと記録する。****将来フラグが増えても自動で残る。**

| # | 検査 | 要点 |
|---|---|---|
| T-S1 | `argv` に実際に渡した引数が残る | 記録の存在 |
| T-S2 | `args_resolved` に第 2 弾で足したフラグが**確定値**で残る | 既定値込みで復元できる |
| **T-S3** | **フラグを渡さない走行と渡した走行が区別できる** | **空振り防止**（記録があっても群を判別できなければ意味が無い） |
| **T-S4** | **`args_resolved` が argparse の全フラグを覆う** | **将来のフラグ増に自動対応**していること自体の検査 |

実行方法（リポジトリルートで。**学習を 2 回 × 2048 歩だけ回すので 1 分ほどかかる**）:
    .venv/bin/python tests/test_run_summary_argv.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO_ROOT, "experiments", "exp_012_continuous_potential", "train.py")
PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))
    return bool(cond)


def run_train(extra_args, tag):
    """最小構成（2048 歩）で 1 本走らせ、書かれた run_summary.json を返す。"""
    d = tempfile.mkdtemp(prefix=f"r11_{tag}_")
    cmd = [PY, TRAIN, "--condition", "E", "--env-version", "v2",
           "--total-steps", "2048", "--seed", "1",
           "--log-dir", d, "--model-out", os.path.join(d, "m.zip")] + extra_args
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[{tag}] 学習が失敗した（終了コード {r.returncode}）:\n{r.stdout[-2000:]}")
    with open(os.path.join(d, "run_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    shutil.rmtree(d, ignore_errors=True)
    return summary


print("[T-S1・T-S2] 群 2 相当（再帰型・リセットあり・観測履歴あり）で走らせる")
G2_FLAGS = ["--recurrent", "--lstm-hidden", "32", "--respawn-reset", "--obs-history", "1,2"]
g2 = run_train(G2_FLAGS, "g2")

check("T-S1 argv が記録されている", isinstance(g2.get("argv"), list) and len(g2["argv"]) > 3,
      " ".join(g2.get("argv", [])[:3]) + " …")
for flag in ("--recurrent", "--respawn-reset", "--obs-history", "--lstm-hidden"):
    check(f"T-S1 argv に {flag} が残る", flag in g2["argv"])

res = g2.get("args_resolved", {})
for key, want in (("recurrent", True), ("respawn_reset", True),
                  ("lstm_hidden", 32), ("obs_history", "1,2"),
                  ("obs_history_sham", False)):
    check(f"T-S2 args_resolved の {key} = {want!r}", res.get(key) == want, repr(res.get(key)))

print("\n[T-S3] 🔴 空振り防止 — フラグを渡さない走行と区別できるか")
g1 = run_train([], "g1")
check("T-S3 群 1 相当では recurrent = False",
      g1["args_resolved"]["recurrent"] is False, repr(g1["args_resolved"]["recurrent"]))
check("T-S3 群 1 相当では respawn_reset = False",
      g1["args_resolved"]["respawn_reset"] is False)
check("T-S3 群 1 相当の argv に --respawn-reset が無い", "--respawn-reset" not in g1["argv"])
check("🔴 T-S3 2 本の記録が実際に違う（同じ内容なら群を判別できない）",
      g1["args_resolved"] != g2["args_resolved"]
      and g1["args_resolved"]["respawn_reset"] != g2["args_resolved"]["respawn_reset"])

print("\n[T-S4] 🔴 将来のフラグ増に自動対応しているか（argparse の全フラグを覆う）")
helptext = subprocess.run([PY, TRAIN, "--help"], capture_output=True, text=True,
                          cwd=REPO_ROOT).stdout
flags = {m.replace("--", "").replace("-", "_")
         for m in re.findall(r"--[a-z0-9][a-z0-9-]+", helptext)} - {"help"}
missing = sorted(f for f in flags if f not in res)
check("T-S4 argparse の全フラグが args_resolved に入っている",
      not missing, f"覆っている数 {len(flags) - len(missing)} / {len(flags)}"
      + (f"・欠け {missing}" if missing else ""))
# 空振り防止: フラグを 1 つも拾えていなければ、この検査自体が無意味
check("T-S4 空振り防止: --help から 10 個以上のフラグを拾えている", len(flags) >= 10,
      f"{len(flags)} 個")

print("\n" + "=" * 78)
n_ok = sum(1 for _, ok in results if ok)
for name, ok in results:
    if not ok:
        print(f"  ❌ FAIL  {name}")
print(f"  {n_ok} / {len(results)} PASS")
sys.exit(0 if n_ok == len(results) else 1)
