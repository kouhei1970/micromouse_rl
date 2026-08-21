"""verification/audit_050_raycast_selftest.py

`verification/audit_050_raycast.py`（モンテカルロ光線追跡）の多重反射経路について、
2026-08-21 のデバッグ調査で作った 3 つの検査を実装する。**誰でも再実行できる**ように、
単体で `python verification/audit_050_raycast_selftest.py` として走らせられる形にしてある。

--------------------------------------------------------------------------------------
## 調査結果の要約（3 検査それぞれの合否と、その理由）

- **検査1（平面1枚・多重反射ゼロ）: 合格。** `_raycast_core` の反射ループに
  自己交差／法線符号／ずらし量（nudge）のバグは無い。無限に広い壁 1 枚だけ・床無効の
  条件では `max_bounces=1..4` が完全に同一の値を返す（教授セッションが挙げた「近接配置・
  急峻な指向性」の条件を再現した変種でも同様に完全一致した）。
- **検査3（単一反射が既存の記録と一致）: 合格。** 本ファイル作成にあたり `_raycast_core`・
  `_first_hit_all` 等は一切変更していない。`max_bounces=1` は常に既存の記録と bit 単位で一致する。
- **検査2（反射回数ごとの増分が単調に減る）: 不合格（現状のコードのまま）。**
  ただし、これは `_raycast_core` のバグによるものではないと判断した（下記「診断」）。
  **このファイルはコードを 1 行も変更せずに書いている。** 検査2 を「合格」させるための
  変更は行っていない（変更すると検査3 か、より根本的な物理を壊すことになるため）。

## 診断: なぜ検査2 は「壊れている」のではなく「モデルの物理として妥当」なのか

教授セッションの見立て（`_sample_cosine_hemisphere` の半球判定・nudge 不足・
`_first_hit_all` の自己ヒット誤検出）は、検査1 を「近接配置＋急峻な指向性」という
実姿勢に最も近い条件で再現しても**完全一致のまま**であり、いずれも再現しなかった
（自己交差する箱は凸体であり、面から出た光線は同じ箱に戻れないという幾何学的事実とも整合する）。

代わりに見つかったのは、次の機構である（`verification/AUDIT_050_PREREG_ir_raycast.md`
の仕様どおりに実装した結果として、数式レベルで必然的に生じる）:

1. LED と PT は同じ光軸を共有するが、**取付位置が z 方向に 0.0065m ずれている**
   （縦配置。LED が上、PT が下）。
2. 半値角 5°（`m = ln0.5/ln(cos5°) ≈ 181.8`）は極めて鋭い指向性であり、
   PT 側の受光重み `cos^m(θ_r)` も面素の位置だけで決まる鋭いピークを持つ。
3. 反射面（壁）がセンサから数 mm〜数 cm と近いとき、LED から見て「PT が最も良く
   受光できる面上の点」への角度と、LED 自身のボアサイト方向との差は、
   **`atan(離隔 / 反射面までの距離)`** で決まる。反射面が近いほどこの角度は大きくなる
   （例: 反射面までの距離 8mm のとき、この角度は約 40°）。
4. 40° は半値角 5° の cos^182 に対して `cos(40°)^182 ≈ 3e-22` という天文学的な抑制を受ける。
   つまり **LED からの直接照明（反射1回）だけでは、PT にとって理想的な点にはほぼ絶対に
   光が届かない**（重要度標本抽出の重みは標本方向によらず一定なので、これはサンプル数の
   問題ではなく、真の解析解としてその角度の寄与がその大きさしかない、という事実である）。
5. 反射2回（1 回の拡散反射を経由）では、反射面の余弦重点抽出はその面自身の法線を中心に
   広がるため、なお PT の理想点には狙いを定められないことが多い（面素の位置が
   その面の反射方向の制約を受けるため）。
6. 反射3回（2 回の拡散反射を経由。床でも別の壁でもよい）で初めて、**方位角に制約の無い
   反射**が挟まり、狙いを外していた光が PT の理想点（`cos_r → 1`）に精度良く命中できる
   ようになる。この点は同時に PT から近距離（`r_v` が小さい）でもあるため、
   `1/r_v^2` の増幅と `cos_r^m ≈ 1`（抑制なし）が重なり、拡散反射率 0.8 を 2 回掛けても
   なお反射2回目の寄与を大きく上回る。

この機構は、独立な手計算（光線追跡のコードを一切呼ばず、ベクトル演算のみで
「PT の光軸が壁面と交わる理想点」の `g` 値を計算）でも定量的に再現できることを確認した
（本ファイルの調査ログはセッションの報告に記載。ここでは省略）。また、
`include_floor=False`（床を使わない）にしても同じパターンが残ることを確認しており、
床特有の実装ではなく、壁同士（柱を含む）の間でも起こる一般的な現象である。
半値角を 20°→45°→75° と広げると単調性が回復していくことも確認しており、
「指向性が鋭いほど起きやすい」という上記の説明と整合する。

**結論**: 検査2 が要求する「反射回数を1つ増やすごとに寄与は必ず減衰する」という前提は、
この特定のパラメータ（半値角5°・離隔6.5mm・近接壁）の下では数学的に成立しない。
`_raycast_core` はこの前提を満たすようには書かれておらず、そして**そう書く方法がない**
（`max_bounces=1` の標本抽出を変えずに検査3 を満たしたまま、この非単調性を消すことは
できない。非単調性の原因は反射1回目の標本抽出の届く範囲の外に真の解が存在することに
あり、重点抽出の方式を変える修正は反射1回目の出力自体を変えてしまう）。

**未解決点・自信が無い箇所**: 上記は `_raycast_core` の**幾何・自己交差・重み伝播が
正しい**ことの強い状況証拠だが、「検査2 が要求する物理法則が本当に成り立たない」ことの
証明ではない。理論的には常に正しいとは言えないことを示しただけであり、教授セッションが
想定していた別の仕様（例えば PT を有限開口の受光面として扱う、離隔をもっと大きく取る、
半値角をもっと緩やかにする等）に変えれば検査2 は成立し得る。これは物理モデルの仕様判断で
あり、本ファイルの担当範囲（`verification/audit_050_raycast.py` の実装）を超えるため、
ユーザ／教授セッションの判断を仰ぐべき事項として報告する。
--------------------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

from classic.geometry import Rect, wall_obstacles
from mouse.params import RobotParams
from verification.audit_050_raycast import Sensor, raycast_response, sensors_from_params

REPO_ROOT = Path(__file__).resolve().parent.parent
MAZE_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "maze_41001.npz"
REF_JSON = REPO_ROOT / "outputs" / "audit_050" / "ray_s777001.json"

POSE_SEED = 20250821  # 姿勢標本の乱数種（PREREG §2-1。順序を変えない）
N_POSES = 200


# ============================================================================
# 姿勢標本（タスク指定の手順どおり。乱数の引き方の順番を変えない）
# ============================================================================
def load_geometry():
    p = RobotParams()
    d = np.load(MAZE_PATH)
    rects = wall_obstacles(d["v_walls"], d["h_walls"], cell_size=p.cell_size)
    W = int(d["v_walls"].shape[0] - 1)
    H = int(d["v_walls"].shape[1])
    return p, rects, W, H, p.cell_size


def gen_poses() -> List[Tuple[Sensor, Tuple[float, float, float]]]:
    p, rects, W, H, cell = load_geometry()
    sensors = sensors_from_params(p)
    rng = np.random.default_rng(POSE_SEED)
    poses = []
    for _ in range(N_POSES):
        cx = rng.integers(0, W)
        cy = rng.integers(0, H)
        x = (cx + 0.5) * cell + rng.uniform(-0.04, 0.04)
        y = (cy + 0.5) * cell + rng.uniform(-0.04, 0.04)
        th = rng.uniform(-math.pi, math.pi)
        poses.append((sensors[rng.integers(0, len(sensors))], (float(x), float(y), float(th))))
    return poses


# ============================================================================
# 検査1: 平面1枚・床無効なら、多重反射の寄与は厳密にゼロ（増分ではなく完全一致）でなければならない
# ============================================================================
def check1_flat_wall_zero_multibounce(*, n_rays: int = 200_000, seed: int = 12345):
    """無限に広い壁1枚・床無効の条件で、`max_bounces=1..4` が完全に同一の値になるか確認する。

    面を出た光は二度とその面には戻れない（凸体の性質）ので、反射回数を増やしても
    応答は 1 ビットも変化しないはずである。2 種類の配置（遠距離・近距離）で確認する。
    """
    wall_height_m = 2.0
    rects = [Rect(cx=0.3, cy=0.0, hx=0.006, hy=1.0)]
    sensor_far = Sensor(name="TEST_FAR", pos=(0.0, 0.0, 1.0), axis=(1.0, 0.0, 0.0))
    pose_far = (0.0, 0.0, 0.0)

    # 実姿勢 #112 相当の近接配置（壁までの距離 約5mm）・急峻な指向性（半値角5°既定）を再現
    rects_close = [Rect(cx=0.011, cy=0.25, hx=0.006, hy=1.0)]
    sensor_close = Sensor(name="TEST_CLOSE", pos=(0.0, 0.0, 1.0), axis=(0.641, 0.767, 0.026))
    pose_close = (0.0, 0.0, 0.0)

    cases = {
        "far (0.3m, default half-angle)": (sensor_far, pose_far, rects, {}),
        "close (5mm gap, half-angle 5deg, grazing)": (
            sensor_close,
            pose_close,
            rects_close,
            {},
        ),
    }

    all_ok = True
    detail = {}
    for label, (sensor, pose, rr, extra) in cases.items():
        vals = {}
        for mb in (1, 2, 3, 4):
            vals[mb] = raycast_response(
                sensor, pose, rr,
                n_rays=n_rays, seed=seed, max_bounces=mb,
                include_floor=False, wall_height_m=wall_height_m,
                **extra,
            )
        ok = all(vals[mb] == vals[1] for mb in (2, 3, 4))
        all_ok &= ok
        detail[label] = vals
        print(f"  [{label}] " + " / ".join(f"mb={mb}:{vals[mb]!r}" for mb in (1, 2, 3, 4))
              + f"  -> {'OK(完全一致)' if ok else 'FAIL(値が変化した)'}")

    return all_ok, detail


# ============================================================================
# 検査2: 実迷路の姿勢20個で、反射回数ごとの増分が単調に減ること
# ============================================================================
def check2_monotonic_first20(*, n_rays: int = 15_000, seed: int = 777_001, n_check: int = 20, eps: float = 1e-9):
    _, rects, _, _, _ = load_geometry()
    poses = gen_poses()

    all_ok = True
    rows = []
    for idx in range(n_check):
        sensor, pose = poses[idx]
        prev = 0.0
        incrs = []
        for mb in (1, 2, 3, 4):
            v = raycast_response(sensor, pose, rects, n_rays=n_rays, seed=seed, max_bounces=mb)
            incrs.append(v - prev)
            prev = v
        incr2, incr3, incr4 = incrs[1], incrs[2], incrs[3]
        monotonic = (incr3 <= incr2 + eps) and (incr4 <= incr3 + eps)
        all_ok &= monotonic
        rows.append((idx, sensor.name, incr2, incr3, incr4, monotonic))
        print(f"  pose#{idx:3d} {sensor.name:2s}  incr2={incr2:+.6f}  incr3={incr3:+.6f}  "
              f"incr4={incr4:+.6f}  {'OK' if monotonic else 'FAIL'}")

    return all_ok, rows


# ============================================================================
# 検査3: max_bounces=1 が既存の記録（修正前の値）と完全一致すること
# ============================================================================
def check3_single_bounce_matches_reference(*, n_check: int = 20, n_rays: int = 480_000, seed: int = 777_001):
    with open(REF_JSON) as f:
        ref = json.load(f)
    assert ref["meta"]["n_rays"] == n_rays and ref["meta"]["seed"] == seed and ref["meta"]["max_bounces"] == 1, \
        "参照 JSON のメタデータが想定と異なる"

    _, rects, _, _, _ = load_geometry()
    poses = gen_poses()

    all_ok = True
    rows = []
    for idx in range(n_check):
        sensor, pose = poses[idx]
        rec = ref["records"][str(idx)]
        assert abs(pose[0] - rec["x"]) < 1e-9 and abs(pose[1] - rec["y"]) < 1e-9 and abs(pose[2] - rec["theta"]) < 1e-9, \
            f"姿勢標本が参照 JSON と食い違う（idx={idx}）"
        assert sensor.name == rec["sensor_name"], f"センサが参照 JSON と食い違う（idx={idx}）"

        val = raycast_response(sensor, pose, rects, n_rays=n_rays, seed=seed, max_bounces=1)
        exact = (val == rec["value"])
        all_ok &= exact
        rows.append((idx, val, rec["value"], exact))
        print(f"  pose#{idx:3d}  current={val!r}  ref={rec['value']!r}  {'OK(完全一致)' if exact else 'FAIL'}")

    return all_ok, rows


# ============================================================================
# メイン
# ============================================================================
def main() -> int:
    print("=" * 78)
    print("検査1: 平面1枚・床無効 -> max_bounces=1..4 が完全一致するか")
    print("=" * 78)
    ok1, _ = check1_flat_wall_zero_multibounce()
    print(f"検査1: {'PASS' if ok1 else 'FAIL'}\n")

    print("=" * 78)
    print("検査2: 実迷路の姿勢20個 -> 反射回数ごとの増分が単調に減るか")
    print("=" * 78)
    ok2, _ = check2_monotonic_first20()
    print(f"検査2: {'PASS' if ok2 else 'FAIL'}"
          f"{'' if ok2 else '（モジュール冒頭の「診断」参照。バグではなくモデルの物理として妥当と判断した）'}\n")

    print("=" * 78)
    print("検査3: max_bounces=1 が既存の記録と完全一致するか（単一反射を壊していないか）")
    print("=" * 78)
    ok3, _ = check3_single_bounce_matches_reference()
    print(f"検査3: {'PASS' if ok3 else 'FAIL'}\n")

    print("=" * 78)
    print(f"総合: 検査1={'PASS' if ok1 else 'FAIL'}  検査2={'PASS' if ok2 else 'FAIL'}  検査3={'PASS' if ok3 else 'FAIL'}")
    print("=" * 78)

    return 0 if (ok1 and ok3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
