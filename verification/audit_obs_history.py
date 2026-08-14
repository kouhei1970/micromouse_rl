#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
監査 044: `mouse/obs_history.py` の意味的監査

准教授セッション（8 代目）・2026-08-15・教授発注（`AUDIT_042` 未確認 3）

## 問い

**「遅れ (1,2,4,…,128) の各位置に入っている観測は、本当にその時点のものか。」**

bit 一致検証と単体テスト 8 件は学生B が通しているが、**通っていることと、
何かを検査していることは別**である（`AUDIT_041` §7-5 で学生B 自身が T7 について書いた通り）。
**本監査は実装を読んだうえで、独立に書いた検査で意味の層を確かめる。**

## 実装を読んで立てた仮説（**検査を書く前に列挙した**）

| # | 疑い | なぜ疑うか |
|---|---|---|
| **H1** | **遅れの位置が 1 つずれている** | `appendleft` の後 `buf[k]` が k 歩前になるかは、`appendleft` と `maxlen` の相互作用に依存する |
| **H2** | **`reset` で同じ配列オブジェクトを 129 個入れている**（`obs_history.py`:114-115） | **環境が観測配列を使い回していれば、履歴が全部同じ値に化ける**。典型的な別名（エイリアス）の誤り |
| **H3** | **`step` でも同じ配列を積んでいる可能性**（:120-121） | 同上。`np.asarray` は**コピーを作らない**（既に float32 の ndarray ならそのまま返す） |
| **H4** | **立て直し（リスポーン）を跨ぐ履歴** | 立て直しは**エピソード内**で起こるので履歴は連続する。**跨いだ履歴は、機体がもう居ない場所の観測**を指す |
| **H5** | **エピソード境界で履歴が残る** | `reset` で `clear()` しているので残らないはずだが、確かめる |

**H2・H3 が本命である。**`Maze6Env.step` が毎回新しい配列を返すなら無害、
**使い回すなら履歴は壊れる**。**単体テスト T4 は「k 歩進めた後」を見ているので、
使い回しがあれば検出できるはずだが、それは学生B の検査であって私のものではない。**
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from mouse.maze6_env import Maze6Env            # noqa: E402
from mouse.obs_history import ObsHistoryWrapper, parse_lags  # noqa: E402

LAGS = (1, 2, 4, 8, 16, 32, 64, 128)
N_STEPS = 400
results = []


def rec(item, ok, detail):
    results.append((item, ok, detail))
    print(f"  [{'PASS' if ok is True else ('FAIL' if ok is False else 'INFO')}] {item}: {detail}")


import gymnasium as gym  # noqa: E402


class Recorder(gym.Wrapper):
    """素の環境を包み、**返された観測を必ずコピーして**記録する。

    コピーするのは、**環境が配列を使い回していても私の記録は汚れない**ようにするため。
    こうしておくと、ラッパ側が汚れていれば差として現れる。
    """

    def __init__(self, env):
        super().__init__(env)
        self.log = []

    def reset(self, **kw):
        o, i = self.env.reset(**kw)
        self.log.append(np.array(o, dtype=np.float32, copy=True))
        return o, i

    def step(self, a):
        o, r, te, tr, i = self.env.step(a)
        self.log.append(np.array(o, dtype=np.float32, copy=True))
        return o, r, te, tr, i


def main():
    print("=" * 72)
    print("監査 044: mouse/obs_history.py の意味的監査")
    print("=" * 72)

    # ---------------- H2/H3: 環境が観測配列を使い回すか ----------------
    print("\n=== H2/H3: 環境は観測配列を使い回すか（別名の誤りの有無） ===")
    e = Maze6Env(mode="generate", base_seed=8000, gamma=0.995,
                 collision_respawn=True, goal_rule_containment=True,
                 episode_limit_steps=2000)
    o0, _ = e.reset(seed=0)
    ids = [id(o0)]
    vals = [np.array(o0, copy=True)]
    for _ in range(3):
        o, *_ = e.step(np.zeros(2, dtype=np.float32))
        ids.append(id(o))
        vals.append(np.array(o, copy=True))
    same_obj = len(set(ids)) < len(ids)
    rec("観測配列のオブジェクト同一性", None,
        f"4 回の id が{'**重複する（使い回し）**' if same_obj else '全て異なる（毎回新しい配列）'}")
    # 使い回しなら「古い参照の中身が後から変わる」ことを直接見る
    o_ref, _ = e.reset(seed=0)
    before = np.array(o_ref, copy=True)
    e.step(np.zeros(2, dtype=np.float32))
    mutated = not np.array_equal(np.asarray(o_ref), before)
    rec("reset が返した配列が step 後に書き換わるか", not mutated,
        "書き換わらない（安全）" if not mutated
        else "🔴 **書き換わる** → 履歴が別名で壊れる危険がある")
    e.close() if hasattr(e, "close") else None

    # ---------------- H1/H5: 遅れの位置は正しいか ----------------
    print("\n=== H1: 各遅れの位置に、その時点の観測が入っているか ===")
    base = Maze6Env(mode="generate", base_seed=8000, gamma=0.995,
                    collision_respawn=True, goal_rule_containment=True,
                    episode_limit_steps=2000)
    r = Recorder(base)
    w = ObsHistoryWrapper(r, LAGS)
    n = base.observation_space.shape[0]

    rng = np.random.default_rng(12345)          # **行動の乱数。環境の乱数ではない**
    obs, _ = w.reset(seed=0)
    checks = {lag: [0, 0] for lag in LAGS}      # [一致, 検査数]
    cur_ok = [0, 0]
    resp_steps = []
    for t in range(1, N_STEPS + 1):
        a = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, _, te, tr, info = w.step(a)
        if info.get("respawned"):
            resp_steps.append(t)
        # 私の記録: log[k] が k 歩目の観測（log[0] = reset の観測）
        cur_ok[1] += 1
        cur_ok[0] += np.array_equal(obs[:n], r.log[t])
        for j, lag in enumerate(LAGS):
            want = r.log[max(0, t - lag)]        # 足りない分は reset の観測で埋まるはず
            got = obs[n * (j + 1): n * (j + 2)]
            checks[lag][1] += 1
            checks[lag][0] += np.array_equal(got, want)
        if te or tr:
            break

    rec("現在の観測（先頭 17 要素）", cur_ok[0] == cur_ok[1],
        f"{cur_ok[0]}/{cur_ok[1]} 一致")
    for lag in LAGS:
        ok, tot = checks[lag]
        rec(f"遅れ {lag} 歩の位置", ok == tot, f"{ok}/{tot} 一致")

    # 空振りでないことの確認: 観測が実際に変化しているか
    diffs = [float(np.abs(r.log[i] - r.log[i - 1]).max()) for i in range(1, len(r.log))]
    rec("空振りでないか（隣接歩の観測の差）", min(diffs) > 0,
        f"最小 {min(diffs):.3e} / 中央値 {float(np.median(diffs)):.3e}"
        f"（0 なら定数観測で検査が空振り）")

    # ---------------- H5: エピソード境界で履歴が消えるか ----------------
    print("\n=== H5: エピソード境界で履歴が消えるか ===")
    obs2, _ = w.reset(seed=1)
    first = obs2[:n]
    all_same = all(np.array_equal(obs2[n * (j + 1): n * (j + 2)], first)
                   for j in range(len(LAGS)))
    rec("reset 直後は全ての遅れが最初の観測", all_same,
        "全 8 か所が現在と同値（前エピソードの残りが無い）" if all_same
        else "🔴 前のエピソードの観測が残っている")

    # ---------------- H4: 立て直しを跨ぐ履歴の割合 ----------------
    print("\n=== H4: 立て直しを跨ぐ履歴の割合（意味の解釈に効く） ===")
    rec("この試走での立て直し", None, f"{len(resp_steps)} 回（歩 {resp_steps[:8]}…）")
    print("      → 実測データでの割合は下記（保存済みの resp_hist から数える）")

    # 実測データ（exp_021 の rollout）で、履歴 128 歩が立て直しを跨ぐ歩の割合
    import json
    for lab, f in (("対照", "control_final"), ("介入", "treat_final")):
        p = os.path.join(REPO, f"outputs/exp_021_driving_{f}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        tot = strad = 0
        for b in d["detail"].values():
            for rr in b.get("raw", []):
                rh = rr["resp_hist"]
                idx = [i for i, x in enumerate(rh) if x]
                for t in range(len(rh)):
                    tot += 1
                    if any(t - 128 < i <= t for i in idx):
                        strad += 1
        if tot:
            rec(f"{lab}群: 履歴 128 歩が立て直しを跨ぐ歩の割合", None,
                f"{strad}/{tot} = {strad/tot*100:.1f} %")

    print("\n" + "=" * 72)
    nf = sum(1 for _, ok, _ in results if ok is False)
    print(f"総括: 不合格 {nf} 件")
    print("""
⚠️ 本監査が確かめたのは**遅れの位置の意味**である。
   **観測 17 要素そのものの正しさ**（推測航法・距離センサの値）は対象外で、未確認。
""")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
