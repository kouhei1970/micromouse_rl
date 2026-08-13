"""条件 C' の進捗の**独立観測**を時刻つきで記録する（研究計画書 §9-16）。

§9-16（2026-08-14 制定）: 効力が「**誰も結果を見ていない時点で確定**」に依存する
裁定・登録は、**自己申告に加えて独立の観測を意図的に取る**。

前例（`AUDIT_019` / 裁定 R46）では、准教授の独立観測（06:59:24）が
学生B のコミット（06:59:39）と 15 秒違いで**偶然**存在した。**本スクリプトはそれを仕組みにする。**

記録するのは「**その時刻に何が観測可能だったか**」だけである（判定はしない）。
打ち切り基準は 100 万歩までの 10 点なので、**10 点が揃った時刻**が本記録の要点になる。
"""
import json
import os
import sys
import time
from datetime import datetime

REPO = "/Users/kouhei/tmp/github/micromouse_rl"
OUT = f"{REPO}/verification/out/cp_observations.jsonl"
SEEDS = (1, 2, 3)


def snapshot():
    rec = {"observed_at": datetime.now().isoformat(timespec="seconds"), "seeds": {}}
    for s in SEEDS:
        p = f"{REPO}/logs/exp_012_condCp_seed{s}/validation_history.json"
        if not os.path.exists(p):
            rec["seeds"][s] = None
            continue
        try:
            rows = json.load(open(p))
        except (ValueError, OSError):
            continue                       # 書き込み中は次回に回す
        pts = [r for r in rows if r["total_timesteps"] <= 1_000_000]
        rec["seeds"][s] = {
            "n_points": len(rows), "n_points_le_1M": len(pts),
            "latest_steps": rows[-1]["total_timesteps"] if rows else None,
            "abort_decidable": len(pts) >= 10,          # 10 点が揃ったか
            "all_below_005": all(r["goal_rate"] < 0.05 for r in pts) if pts else None,
        }
    return rec


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    deadline = time.time() + float(sys.argv[2]) if len(sys.argv) > 2 else time.time() + 4 * 3600
    while time.time() < deadline:
        rec = snapshot()
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done = all(v and v["latest_steps"] and v["latest_steps"] >= 2_000_000
                   for v in rec["seeds"].values())
        if done:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
