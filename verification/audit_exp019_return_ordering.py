#!/usr/bin/env python3
"""監査: ⑥（§9-8 の総収益順序を v2 仕様で再実行）の独立再計算 — 手順 ③（上限 2000 歩・机上）

判定するもの
------------
学生B が報告した **「検証帯 20 面中 18 面で総収益順序が成立」** を、
**独立に実装した式で再現するか**。

**独立性の作り**
----------------
- $D_0$ は **`assets/maze6/loop/validation/*.npz` の壁配列から自前の BFS** で出す
  （`shortest_distances()` も `generate_maze()` も呼ばない）
- 収益の式は **§9-8 のシナリオ定義**（ゴール／探索／滞留／衝突）から**自分で書き直す**
- 報酬契約の定数は `mouse.maze6_env` から読み、**値を assert** する（作法 11）

**🔴 本監査の主眼**
-------------------
`AUDIT_024` で**リスポーン整形の恒等式**（区間の割引整形総和 = 0）を実測で確定させた。
**その恒等式を、衝突シナリオの収益の式に当てるとどうなるか**を検査する。

  リスポーンでは「区間で稼いだ整形 $\\gamma^{T}\\Phi_c$」と「リスポーンの整形 $-\\gamma^{T}\\Phi_c$」が
  **厳密に打ち消す**。したがって **N 回の衝突の総収益に $\\Phi_c$ は残らない**。

使い方: `.venv/bin/python verification/audit_exp019_return_ordering.py`
"""
from __future__ import annotations

import glob
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mouse.maze6_env import (  # noqa: E402
    _COLLISION_PENALTY,
    _GOAL_BONUS,
    _TIME_PENALTY,
    _VISIT_BONUS,
)

# --- 定数（§9-8 と同じ前提。値は assert する）---
GAMMA = 0.995
CELL = 0.18          # m
DT = 0.01            # s（制御周期）
SPC = CELL / (0.96 * DT)     # 1 区画あたりの歩数 = 18.75
N_CELLS_TOTAL = 36
LIMIT = 2000         # 裁定 2026-08-14: v2 のエピソード上限
DT_CONTAIN = 7.0     # 規約終端の遅れ（実装での実測値・学生B 報告）
T_OBS = 608.0        # 実走 regime の衝突までの歩数
ALPHA_OBS = 0.133    # 実走 regime の進捗率
VISIT_OBS = 0.055    # 実走 regime の訪問ボーナス回収量
VALID_DIR = "assets/maze6/loop/validation"


def _check_contract() -> None:
    assert _TIME_PENALTY == 0.001 and _GOAL_BONUS == 1.0
    assert _COLLISION_PENALTY == -1.0 and _VISIT_BONUS == 0.02
    assert abs(SPC - 18.75) < 1e-12, SPC


def S(T: float) -> float:
    """時間罰の等比和 Σ_{t<T} γ^t。"""
    return (1.0 - GAMMA ** T) / (1.0 - GAMMA)


def visit_disc(n_cells: int) -> float:
    """訪問報酬の割引後総和（i 区画目は i·SPC 歩目に入るとみなす）。"""
    return _VISIT_BONUS * sum(GAMMA ** (i * SPC) for i in range(n_cells))


# ---------------------------------------------------------------- 自前 BFS
def d0_from_npz(path: str) -> tuple[int, int]:
    """壁配列から自前の BFS で D₀（スタート→ゴールの区画距離）を出す。"""
    d = np.load(path)
    v, h = d["v_walls"], d["h_walls"]
    w, hh = int(d["width"]), int(d["height"])
    start = tuple(int(x) for x in d["start"])
    goals = {tuple(int(x) for x in g) for g in d["goal_cells"]}

    def connects(x, y, nx, ny) -> bool:
        if nx == x + 1:
            return v[x + 1, y] == 0
        if nx == x - 1:
            return v[x, y] == 0
        if ny == y + 1:
            return h[x, y + 1] == 0
        return h[x, y] == 0

    dist = {start: 0}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < hh and (nx, ny) not in dist \
                    and connects(x, y, nx, ny):
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return int(d["seed"]), min(dist[g] for g in goals if g in dist)


# ------------------------------------------------------- シナリオの総収益
def scenarios(d0: int) -> dict:
    """1 面の 4 挙動の割引後総収益（k=0・上限 2000 歩・規約終端・リスポーン）。"""
    D0_m = d0 * CELL
    Tg = d0 * SPC
    Tg_pay = Tg + DT_CONTAIN

    goal = (GAMMA ** Tg_pay * D0_m
            - _TIME_PENALTY * S(Tg_pay)
            + GAMMA ** (Tg_pay - 1) * _GOAL_BONUS
            + visit_disc(d0))
    explore = -_TIME_PENALTY * S(LIMIT) + visit_disc(N_CELLS_TOTAL)
    stay = -_TIME_PENALTY * S(LIMIT)

    # --- 衝突（リスポーンで上限まで繰り返す）---
    phi_c = ALPHA_OBS * D0_m               # 1 区間で稼ぐ整形
    n_crash = int(LIMIT // T_OBS)          # = 3

    # (I) 学生B の式（照合用に再現）: 1 回あたり −(1.0 + Φ_c) を課す
    crash_B = (-_TIME_PENALTY * S(LIMIT)
               - sum(GAMMA ** (i * T_OBS - 1) * (1.0 + phi_c)
                     for i in range(1, n_crash + 1))
               + VISIT_OBS)

    # (II) 恒等式（AUDIT_024）に整合させた式:
    #      区間で稼ぐ整形 +γ^(iT)·Φ_c と リスポーンの整形 −γ^(iT)·Φ_c は**厳密に打ち消す**。
    #      したがって残るのは衝突罰 −1.0 だけである。
    crash_id = (-_TIME_PENALTY * S(LIMIT)
                - sum(GAMMA ** (i * T_OBS - 1) * 1.0
                      for i in range(1, n_crash + 1))
                + VISIT_OBS)

    return dict(d0=d0, goal=goal, explore=explore, stay=stay,
                crash_B=crash_B, crash_id=crash_id,
                delta=crash_id - crash_B, n_crash=n_crash, phi_c=phi_c)


def ok(r: dict, key: str) -> bool:
    return r["goal"] > r["explore"] > r["stay"] > r[key]


def main() -> int:
    _check_contract()
    faces = sorted(d0_from_npz(p) for p in glob.glob(f"{VALID_DIR}/*.npz"))
    rows = []
    for seed, d0 in faces:
        r = scenarios(d0)
        r["seed"] = seed
        rows.append(r)

    n_B = sum(ok(r, "crash_B") for r in rows)
    n_id = sum(ok(r, "crash_id") for r in rows)

    print("=" * 92)
    print("監査: ⑥ §9-8 総収益順序の独立再計算（手順 ③・上限 2000 歩・k=0・規約終端・リスポーン）")
    print("=" * 92)
    print(f"{'面':>6}{'D₀':>4}{'ゴール':>9}{'探索':>9}{'滞留':>9}"
          f"{'衝突(B)':>10}{'衝突(恒等式)':>13}{'差':>9}  B  恒等式")
    for r in rows:
        print(f"{r['seed']:>6}{r['d0']:>4}{r['goal']:>9.3f}{r['explore']:>9.3f}"
              f"{r['stay']:>9.3f}{r['crash_B']:>10.3f}{r['crash_id']:>13.3f}"
              f"{r['delta']:>9.4f}"
              f"  {'✅' if ok(r,'crash_B') else '🔴'}"
              f"   {'✅' if ok(r,'crash_id') else '🔴'}")
    print("-" * 92)
    print(f"  学生B の式での成立面   : **{n_B} / {len(rows)}**")
    print(f"  恒等式に整合させた式   : **{n_id} / {len(rows)}**")
    fails_id = [r["seed"] for r in rows if not ok(r, "crash_id")]
    fails_B = [r["seed"] for r in rows if not ok(r, "crash_B")]
    print(f"  崩れる面（B の式）     : {fails_B}")
    print(f"  崩れる面（恒等式）     : {fails_id}")
    print("=" * 92)

    out = Path(__file__).resolve().parent / "out" / "exp019_return_ordering.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n_ok_studentB_form": n_B, "n_ok_identity_form": n_id,
                               "n_faces": len(rows), "limit": LIMIT,
                               "fails_studentB": fails_B, "fails_identity": fails_id,
                               "rows": rows}, ensure_ascii=False, indent=2))
    print(f"出力: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
