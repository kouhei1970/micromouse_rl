#!/usr/bin/env python3
"""監査: exp_019 の打ち切り 6/6 発火の**時系列の独立復元**（准教授セッション）

背景
----
`AUDIT_020` の照合で登録した **作法 20**:
「**観測の間隔は、括りたい事象の時間尺度より細かくする**」「**消える証拠は気づいた時点で即時保全する**」。

**`validation_history.json` の mtime は、次の評価が書かれた瞬間に上書きされる。**
走行は完走まで継続するので、**待てば必ず消える**。実際、保全時刻 14:55:00 の時点で
**6 seed 中 4 seed は既に 11 点目以降が書かれており、10 点目の mtime は失われていた**。

そこで**追記型（append-only）の証拠**から絶対時刻を復元する:

1. **`env_0.monitor.csv`** — 先頭行に `t_start`（絶対 unix 時刻）、各行に
   `l`（エピソード長）と `t`（`t_start` からの経過秒）。**n_envs = 1** なので
   `cumsum(l)` がそのまま `total_timesteps`。**`AUDIT_016` と同じ経路**
2. **TensorBoard の events ファイル** — 各スカラーに `wall_time`（絶対 unix 時刻）。
   **追記型で上書きされない**

**この 2 つは互いに独立な経路**（片方は Monitor ラッパ、もう片方は SB3 の logger）。

判定するもの
------------
- 各 seed で **10 点目（total_timesteps = 1,000,000）が書かれた絶対時刻**
- それが**打ち切りの検知時刻（2026-08-14 14:52:27 JST）より前**か
- **2 経路が互いに一致する**か

使い方: `.venv/bin/python verification/audit_exp019_cutoff_timeline.py`
"""
from __future__ import annotations

import csv
import glob
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))
SEEDS = [1, 2, 3, 4, 5, 6]
CUTOFF_STEP = 1_000_000          # 打ち切り条文の窓の終端（10 点目）
DETECTED = datetime(2026, 8, 14, 14, 52, 27, tzinfo=JST)   # 学生B の検知報告
PRESERVED = Path("verification/out/preserved_20260814_cutoff")


def ts(unix: float) -> datetime:
    return datetime.fromtimestamp(unix, tz=JST)


def from_monitor(log_dir: Path) -> dict:
    """monitor.csv から、累計歩数が 1,000,000 を超えた瞬間の絶対時刻を復元。"""
    p = log_dir / "env_0.monitor.csv"
    with p.open() as f:
        header = f.readline()
        t_start = json.loads(header.lstrip("#"))["t_start"]
        rows = list(csv.DictReader(f))
    cum = 0
    for r in rows:
        cum += int(r["l"])
        if cum >= CUTOFF_STEP:
            return dict(t_start=t_start, cross_steps=cum,
                        cross_time=ts(t_start + float(r["t"])),
                        n_episodes=len(rows), total_steps=None)
    total = sum(int(r["l"]) for r in rows)
    return dict(t_start=t_start, cross_steps=None, cross_time=None,
                n_episodes=len(rows), total_steps=total)


def from_tensorboard(log_dir: Path) -> dict:
    """TB events から、評価スカラーの step=1,000,000 の wall_time を読む。"""
    from tensorboard.backend.event_processing import event_accumulator
    files = sorted(glob.glob(str(log_dir / "events.out.tfevents.*")))
    if not files:
        return dict(tag=None, wall=None)
    ea = event_accumulator.EventAccumulator(
        files[-1], size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    # 評価に関係するタグを優先して探す（無ければ任意のタグで時刻だけ取る）
    pref = [t for t in tags if "goal" in t.lower() or "valid" in t.lower()
            or "eval" in t.lower()]
    # ⚠️ TB の step は SB3 の `num_timesteps` の実値（例 1,001,472）であり、
    # `validation_history.json` の丸めた `total_timesteps`（1,000,000）とは一致しない。
    # したがって「**1,000,000 以上で最初のもの**」＝ 10 点目 を採る。
    for tag in (pref + tags):
        for e in ea.Scalars(tag):
            if e.step >= CUTOFF_STEP:
                return dict(tag=tag, wall=ts(e.wall_time), step=e.step,
                            n_tags=len(tags))
    return dict(tag=None, wall=None, n_tags=len(tags), tags=tags[:8])


def main() -> int:
    print("=" * 96)
    print("監査: exp_019 打ち切り 6/6 発火の時系列の独立復元（追記型の証拠から）")
    print(f"  検知報告（学生B）: {DETECTED:%Y-%m-%d %H:%M:%S %Z}")
    print("=" * 96)
    print(f"{'seed':>5}{'点数':>5}{'10 点目 (monitor)':>22}{'10 点目 (TB)':>20}"
          f"{'差[s]':>8}{'検知前':>7}")
    out = []
    for n in SEEDS:
        ld = Path(f"logs/exp_019_v2_seed{n}")
        hist = json.loads((PRESERVED / f"validation_history_seed{n}.json").read_text())
        mon = from_monitor(ld)
        tb = from_tensorboard(ld)
        mt = mon["cross_time"]
        wt = tb["wall"]
        diff = (wt - mt).total_seconds() if (mt and wt) else None
        before = (wt or mt) is not None and (wt or mt) < DETECTED
        print(f"{n:>5}{len(hist):>5}"
              f"{(mt.strftime('%H:%M:%S') if mt else 'n/a'):>22}"
              f"{(wt.strftime('%H:%M:%S') if wt else 'n/a'):>20}"
              f"{(f'{diff:+.1f}' if diff is not None else 'n/a'):>8}"
              f"{('✅' if before else '🔴'):>7}")
        out.append(dict(seed=n, n_points=len(hist),
                        monitor_cross=mt.isoformat() if mt else None,
                        tb_wall=wt.isoformat() if wt else None,
                        tb_tag=tb.get("tag"), diff_s=diff,
                        before_detection=bool(before),
                        goal_rates=[x["goal_rate"] for x in hist],
                        steps=[x["total_timesteps"] for x in hist]))

    print("-" * 96)
    # 打ち切り条文の再現: 100 万歩までの 10 点すべてが < 0.05（厳密不等号）
    print("打ち切り条文の独立再現（100 万歩までの 10 点すべて goal_rate < 0.05・厳密不等号）:")
    n_fire = 0
    for r in out:
        pts = [g for s, g in zip(r["steps"], r["goal_rates"]) if s <= CUTOFF_STEP]
        fire = len(pts) >= 10 and all(g < 0.05 for g in pts)
        n_fire += fire
        print(f"  seed{r['seed']}: 窓内 {len(pts)} 点・最大 {max(pts):.4f} → "
              f"{'発火' if fire else '非発火'}")
    print(f"\n  **発火した seed: {n_fire} / {len(SEEDS)}**"
          f"  → P2（4 seed 以上で発火）は {'成立' if n_fire >= 4 else '不成立'}")
    print("=" * 96)

    p = Path("verification/out/exp019_cutoff_timeline.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"detected_report": DETECTED.isoformat(),
                             "n_fire": n_fire, "seeds": out},
                            ensure_ascii=False, indent=2))
    print(f"出力: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
