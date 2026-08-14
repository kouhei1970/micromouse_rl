#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
保全された学習済み重みの出所照合（規約 §9-21 (c) の履行確認）

准教授セッション（8 代目）・2026-08-14
`AUDIT_042`（exp_021 の完走後判定の独立再計算）の **L1（出所）** で使う道具。

## 何をするか

`experiments/preserved_weights.md` に載っている重みについて:

  1. **現物が存在するか**
  2. **SHA-256 が一覧の記載と一致するか**（改竄・取り違えの検出）
  3. **版管理下にあるか**（§9-21 (c) の要求。`models/*.zip` と `logs/` は
     `.gitignore` で除外されているので `git add -f` されている必要がある）
  4. **判定が依存する重みが漏れていないか**（引数で必要な一覧を渡して検査する）

## なぜ要るか

規約 §9-21 (c)（2026-08-14 新設）は「**判定が依存する学習済み重みは、判定と同じ寿命で
保全する = 版管理下へ入れる**」と定める。**出力の SHA-256 記録では代替できない** —
ハッシュは「変わっていないこと」しか保証せず、**重みが失われた時点で
RL 側の判定は再生成不能になる**。本スクリプトは、その履行を機械的に確かめる。

**⚠️ 一覧に載っていることと、実際に保全されていることは別である。**
**一覧は宣言であって検査ではない**（作法 18: 「印を付けた」と宣言したら全数検査とセットにする）。
"""

import hashlib
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST = os.path.join(REPO, "experiments/preserved_weights.md")

# exp_021 の判定に要る重み（対照群 = exp_019 の最終方策 6 本 ＋ 80 万歩の退避重み 6 本）。
# カード §4-1・§4-2-bis（是正版）・§9 手順 4。
REQUIRED_EXP021 = ([f"models/exp_019_v2_seed{n}.zip" for n in range(1, 7)]
                   + [f"logs/exp_019_v2_seed{n}/rl_model_800000_steps.zip"
                      for n in range(1, 7)])


def parse_list(path):
    """一覧から (パス, SHA-256) の組を拾う。表の列順に依存しないようにする。"""
    txt = open(path, encoding="utf-8").read()
    rows = re.findall(r"([\w./-]+\.zip)[^\n]*?\b([0-9a-f]{64})\b", txt)
    if not rows:
        rows = [(p, h) for h, p in
                re.findall(r"\b([0-9a-f]{64})\b[^\n]*?([\w./-]+\.zip)", txt)]
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO,
                         capture_output=True, text=True, timeout=60)
    return set(out.stdout.split())


def main(required=None):
    if not os.path.exists(LIST):
        print(f"一覧が無い: {LIST}")
        return 2
    rows = parse_list(LIST)
    tracked = tracked_files()
    print(f"一覧 {os.path.relpath(LIST, REPO)}: {len(rows)} 件\n")

    ok = miss = bad = untracked = 0
    for p, h in rows:
        full = os.path.join(REPO, p)
        if not os.path.exists(full):
            miss += 1
            print(f"  🔴 現物なし: {p}")
            continue
        d = sha256(full)
        if d != h:
            bad += 1
            print(f"  🔴 SHA-256 不一致: {p}\n     一覧 {h[:16]}… / 実測 {d[:16]}…")
        else:
            ok += 1
        if p not in tracked:
            untracked += 1
            print(f"  🔴 版管理下にない（§9-21 (c) 違反）: {p}")

    print(f"\n  SHA-256 一致 {ok} / 不一致 {bad} / 現物なし {miss} / 版管理外 {untracked}")

    req = required if required is not None else REQUIRED_EXP021
    listed = {p for p, _ in rows}
    n_listed = sum(1 for p in req if p in listed)
    n_tracked = sum(1 for p in req if p in tracked)
    print(f"\n  判定が依存する重み {len(req)} 本:")
    print(f"    一覧に載っている: {n_listed}/{len(req)}")
    print(f"    版管理下にある  : {n_tracked}/{len(req)}")
    for p in req:
        if p not in listed or p not in tracked:
            print(f"      🔴 {p}"
                  f"（一覧 {'○' if p in listed else '×'} / 版管理 {'○' if p in tracked else '×'}）")

    fails = bad + miss + untracked + (len(req) - n_listed) + (len(req) - n_tracked)
    print("\n" + "=" * 60)
    print(f"総括: 不合格 {fails} 件")
    print("""
⚠️ 本検査が確かめるのは**保全されていること**であって、
   **重みの中身が正しいこと**ではない。学習が正しく走ったかは別の検査による。
""")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
