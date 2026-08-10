#!/usr/bin/env python3
"""independent_kpi.json から報告用の Markdown 表を生成する（転記ミス防止のため手打ちしない）。"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = json.loads((REPO / "verification" / "out" / "independent_kpi.json").read_text())
C = D["conditions"]

BANDS = [("eval", "評価帯 seed 1000-1019"), ("validation", "検証帯 seed 4000-4019"),
         ("contest_reference", "大会実迷路")]
METHODS = [("adachi_classical", "L0-a 区画ごと停止"), ("l0b_straightrun", "L0-b 直進連続"),
           ("l0c_slalom", "L0-c スラローム")]


def key_of(band, method):
    for k in C:
        if k.startswith(band + "/") and k.split("/")[1].startswith(method):
            return k
    return None


def f(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


out = []
for band, blabel in BANDS:
    out.append(f"\n#### {blabel}（`{band}`）\n")
    out.append("| 指標 | 単位 | " + " | ".join(m[1] for m in METHODS) + " |")
    out.append("|---|---|---|---|---|")
    rows = [(key_of(band, m[0])) for m in METHODS]
    cs = [C[k] if k else None for k in rows]

    def line(label, unit, fn):
        out.append(f"| {label} | {unit} | " + " | ".join(fn(c) if c else "—" for c in cs) + " |")

    line("n（面数）", "面", lambda c: str(c["n_mazes"]))
    line("**(a) ゴール到達率**", "%", lambda c: f"**{c['a_goal_rate']*100:.0f}%** ({c['a_count']})")
    line("**(b) 最短走行成立率**", "%", lambda c: f"**{c['b_fast_done_rate']*100:.0f}%** ({c['b_count']})")
    line("**(c) 有効最短走行率**〔分母=全面〕", "%",
         lambda c: f"**{c['c_effective_rate_over_all']*100:.0f}%** ({c['c_count_over_all']})")
    line("　(c) 〔分母=(b)該当面〕", "%",
         lambda c: (f"{c['c_effective_rate_over_b']*100:.0f}% ({c['c_count_over_b']})"
                    if c["c_effective_rate_over_b"] is not None else "—"))
    line("**(d) 最速タイム 中央値**", "s", lambda c: f"**{f(c['d_best_time']['median'])}**")
    line("　(d) 最速タイム 最小 / 最大", "s",
         lambda c: f"{f(c['d_best_time']['min'])} / {f(c['d_best_time']['max'])}")
    line("　探索走行タイム 中央値", "s", lambda c: f"{f(c['d_explore_time']['median'])} (n={c['d_explore_time']['n']})")
    line("　最短走行タイム 中央値", "s", lambda c: f"{f(c['d_fast_time']['median'])} (n={c['d_fast_time']['n']})")
    line("**(e) 初回最短走行効率 中央値**", "—",
         lambda c: f"**{f(c['e_first_fast_efficiency_B']['median'], 4)}**")
    line("　(e) 最大 / 未定義面数", "—",
         lambda c: f"{f(c['e_first_fast_efficiency_B']['max'], 4)} / {c['e_undefined_count']} 面")
    line("　(e) が構造的に 1.00 に固定される面", "面",
         lambda c: f"{c['e_forced_one_count']} / {c['b_count'].split('/')[0]}")
    out.append("| — | — | — | — | — |")
    line("走行回数/面 中央値 (min–max)", "回",
         lambda c: f"{c['runs_per_maze']['median']:.0f} ({c['runs_per_maze']['min']:.0f}–{c['runs_per_maze']['max']:.0f})")
    line("探索後に成立した走行数 中央値 (min–max)", "回",
         lambda c: f"{c['fast_runs_per_maze']['median']:.0f} ({c['fast_runs_per_maze']['min']:.0f}–{c['fast_runs_per_maze']['max']:.0f})")
    line("探索/最短 タイム比 中央値 [Q1, Q3]", "—",
         lambda c: (f"{f(c['explore_over_fast']['median'],3)} [{f(c['explore_over_fast']['q1'],3)}, "
                    f"{f(c['explore_over_fast']['q3'],3)}]" if c["explore_over_fast"]["n"] else "—"))
    line("　└ 経路比 L_探索/L_最短 中央値", "—",
         lambda c: f(c["path_ratio"]["median"], 3) if c["path_ratio"]["n"] else "—")
    line("　└ 速度比 v_最短/v_探索 中央値", "—",
         lambda c: f(c["speed_ratio"]["median"], 3) if c["speed_ratio"]["n"] else "—")
    line("使用した持ち時間 中央値", "s", lambda c: f(c["time_used"]["median"], 1))

print("\n".join(out))

print("\n\n### 走行回数のヒストグラム（面数）\n")
print("| 条件 | 1回 | 2回 | 3回 | 4回 | 5回 |")
print("|---|---|---|---|---|---|")
for band, blabel in BANDS:
    for meth, mlabel in METHODS:
        k = key_of(band, meth)
        if not k:
            continue
        h = C[k]["runs_per_maze_hist"]
        print(f"| {band} / {mlabel} | " + " | ".join(str(h.get(str(i), 0)) for i in range(1, 6)) + " |")

print("\n\n### 探索走行の後に成立した走行の回数（面数）\n")
print("| 条件 | 0回 | 1回 | 2回 | 3回 | 4回 |")
print("|---|---|---|---|---|---|")
for band, blabel in BANDS:
    for meth, mlabel in METHODS:
        k = key_of(band, meth)
        if not k:
            continue
        h = C[k]["fast_runs_per_maze_hist"]
        print(f"| {band} / {mlabel} | " + " | ".join(str(h.get(str(i), 0)) for i in range(0, 5)) + " |")

print("\n\n### (c) の 10% 閾値の感度（該当面数 / n）\n")
print("| 条件 | n | ≥0% | ≥5% | ≥8% | **≥10%（現行）** | ≥12% | ≥15% | ≥20% |")
print("|---|---|---|---|---|---|---|---|---|")
for band, blabel in BANDS:
    for meth, mlabel in METHODS:
        k = key_of(band, meth)
        if not k:
            continue
        s = C[k]["c_threshold_sensitivity"]
        n = C[k]["n_mazes"]
        cells = " | ".join(
            (f"**{s[t]}**" if t == "0.10" else str(s[t]))
            for t in ["0.00", "0.05", "0.08", "0.10", "0.12", "0.15", "0.20"])
        print(f"| {band} / {mlabel} | {n} | {cells} |")
