"""R38-4 の検証 — 学習の生ログ（monitor）から各 seed が何時に何歩に達したかを復元する。

`validation_history.json` を使わない経路で時刻を出すことが要件（判定文書の主張と独立）。
`env_0.monitor.csv` の先頭行の `t_start`（絶対時刻）とエピソード長の累積を使う。
"""
import csv
import datetime
import io
import json
import sys

REPO_ROOT = "/Users/kouhei/tmp/github/micromouse_rl"
MARKS = (900_000, 1_000_000, 2_000_000)


def main():
    for s in (1, 2, 3):
        base = f"{REPO_ROOT}/logs/exp_012_condE_seed{s}"
        lines = open(f"{base}/env_0.monitor.csv").read().splitlines()
        t_start = json.loads(lines[0][1:])["t_start"]
        rows = [(int(float(r["l"])), float(r["t"]))
                for r in csv.DictReader(io.StringIO("\n".join(lines[1:])))]
        n_env = json.load(open(f"{base}/run_summary.json")).get("n_envs") or 1
        print(f"seed{s}: n_envs={n_env}  開始 "
              f"{datetime.datetime.fromtimestamp(t_start).strftime('%H:%M:%S')}")
        steps = 0
        hit = {}
        for L, T in rows:
            prev, steps = steps, steps + L
            for m in MARKS:
                if prev < m // n_env <= steps and m not in hit:
                    hit[m] = T
        for m, T in sorted(hit.items()):
            at = datetime.datetime.fromtimestamp(t_start + T)
            print(f"    総 {m:>9} 歩 → {at.strftime('%H:%M:%S')} JST（投入から {T/60:.1f} 分）")


if __name__ == "__main__":
    main()
