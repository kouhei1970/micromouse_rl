#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0-c+E1 の**経路選択だけ**を「最短歩数」から「最短時間」へ替えた版（TR）。

2026-08-13 新設（exp_015・レバー ①）。**`competition/baseline_slalom_e1.py`
（対照）は変更しない** — exp_014 で測った L0-c+E1 は本実験の対照であり、
再現できなくなると面ごとの対応差が取れなくなるため（この系列で繰り返し
効いている作法）。

--------------------------------------------------------------------------
1 実験 1 変更 — 変えるのは「既知地図の上でどの経路を引くか」だけ
--------------------------------------------------------------------------
親（L0-c+E1）は足立法の距離場（`_flood_fill`）で**歩数**最短の経路を引く。
本クラスは**計時される最短走行（ゴールへ向かう走行）に限って**、距離場を
`competition/route_planner.py` の**時間の場**（状態 = 区画 × 進行方向、
費用 = a·移動 + b·旋回）へ差し替える。

**探索戦略・制御ゲイン・速度プロファイル・軌道生成・ロボットパラメータは
いっさい変えない。**

### 適用範囲を「ゴールへ向かう走行」に限る理由

| 走行 | 適用 | 理由 |
|---|---|---|
| 初回探索（`_explored_once` が False） | **しない** | ここを替えると**探索の道順が変わり地図の育ち方が変わる**。(a)(b)(c) と初回最短走行効率 (e) に影響が出て、「経路選択だけを変えた」比較でなくなる |
| E1 の追加探索（`verify`） | **しない** | 同上。E1 は exp_014 で受理済みの土台であり、本実験の変更対象ではない |
| 帰路（`to_start`） | **しない** | 帰路は評価器の指標で計時されない。替えても得点にならない一方、走行時間が変わると持ち時間の消費が動き、走行回数を通じて (b)(d) に間接的に効きうる（交絡） |
| **最短走行（`to_goal` かつ探索済み）** | **する** | **ここだけが計時される。**本実験の効果はここに出る |

この切り分けは `tr_modes` で変更できる（既定 `("to_goal",)`）。**exp_015 では
既定から動かさないこと。**

--------------------------------------------------------------------------
⚠️ 未知壁の扱いを親と揃えてある
--------------------------------------------------------------------------
時間の場にも親と同じ `_connects_known`（**未知壁は通行可＝楽観的**）を渡す。
揃えないと歩数最短と時間最短が別の地図の上で引かれ、比較が成立しない
（exp_015 カード §4-1「T1 が外れて発動 0」の読み方）。

--------------------------------------------------------------------------
係数はハードコードしない
--------------------------------------------------------------------------
`research_notes/data/time_model_l0c_design.json`（**設計帯** `design_v4` の
L0-c+E1 の最短走行 80 本から回帰）を読む。評価帯由来の係数を使うと評価帯の
情報が設計へ漏れる（教授指示）。**この係数は L0-c 専用**で、超信地旋回の
方式（L0-a/L0-b）とは turn_cost が大きく違う（同 JSON の caveat）。

使い方:
    .venv/bin/python -m competition.evaluator \\
        --policy competition.baseline_slalom_e1_tr:SlalomE1TRPolicy \\
        --maze-dir competition/mazes/eval \\
        --out-dir outputs/exp_015_time_optimal_route/l0c_e1_tr
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition import baseline_slalom as _slalom  # noqa: E402
from competition import route_planner  # noqa: E402
from competition.baseline_slalom_e1 import SlalomE1Policy  # noqa: E402

# 方位の規約が親と食い違ったまま動くと、経路が静かに壊れる（例外は出ない）。
# 別モジュールに定数を置いた以上、取り込み時に一度だけ突き合わせる。
assert route_planner.DIRS == _slalom.DIRS, "方位の並びが親と食い違っている"
assert route_planner.DELTA == _slalom._DELTA, "方位ベクトルが親と食い違っている"

TIME_MODEL_JSON = REPO_ROOT / "research_notes" / "data" / "time_model_l0c_design.json"


def load_time_model(path=TIME_MODEL_JSON):
    """時間モデルの係数を実測の回帰結果から読む（**ハードコードしない**）。"""
    j = json.load(open(path, encoding="utf-8"))
    return float(j["a"]), float(j["b"])


class SlalomE1TRPolicy(SlalomE1Policy):
    """L0-c + E1 + TR（時間最短経路）。"""

    name = "L0-c+E1+TR (slalom, E1, time-optimal route)"

    def __init__(self, *args, time_model_path=None, tr_modes=("to_goal",), **kwargs):
        super().__init__(*args, **kwargs)
        self.time_model_path = Path(time_model_path) if time_model_path else TIME_MODEL_JSON
        a, b = load_time_model(self.time_model_path)
        self.time_a, self.time_b = a, b
        # 裁定 R29: コストモデルは差し替えられる形にする（exp_016 の斜め走行で
        # 状態空間ごと差し替える）。ここを別のモデルに替えれば方策側は無改造。
        self.cost_model = route_planner.StraightGridModel(a, b)
        self.tr_modes = tuple(tr_modes)

    # ------------------------------------------------------------------
    # 適用範囲の判定
    # ------------------------------------------------------------------
    def _use_time_route(self) -> bool:
        """いま引く経路を「時間最短」にするか。計時される最短走行だけ True。"""
        return self._explored_once and self.target_mode in self.tr_modes

    # ------------------------------------------------------------------
    # 距離場 → 時間の場（`_build_chain` からの呼び出しをここで差し替える）
    # ------------------------------------------------------------------
    def _flood_fill(self, targets):
        if not self._use_time_route():
            return super()._flood_fill(targets)
        return route_planner.value_field(targets, self.width, self.height,
                                          self._connects_known, self.cost_model)

    # ------------------------------------------------------------------
    # 方向の選択 — 時間の場のときだけ「旋回費用 + 残り時間」で選ぶ
    # ------------------------------------------------------------------
    def _select_direction(self, cell, prev_dir: str, dist_field):
        states = getattr(dist_field, "states", None)
        if states is None:                       # 親の距離場（探索走行・帰路）
            return super()._select_direction(cell, prev_dir, dist_field)

        candidates = []
        for d_out, _nxt, state, w in self.cost_model.successors(
                cell, prev_dir, self.width, self.height, self._connects_known):
            v = states.get(state)
            if v is None:                        # 目標へ到達できない向き
                continue
            candidates.append((d_out, w + v))
        if not candidates:
            return None

        best = min(c for _, c in candidates)
        # 同点は親と同じ決定的タイブレーク（直進優先 → 時計回り）。費用差が
        # 浮動小数の誤差の桁（TIE_EPS）なら同点として扱う。
        best_set = {d for d, c in candidates if c <= best + route_planner.TIE_EPS}
        return next(d for d in self._tiebreak_order(prev_dir) if d in best_set)

    # ------------------------------------------------------------------
    # 発動（歩数最短と時間最短が食い違ったか）は**走行中に数えない**
    # ------------------------------------------------------------------
    # 走行のたびに対照側の距離場も計算して突き合わせることは可能だが、
    #   (1) 1 手ごとに BFS を余分に回すことになり、
    #   (2) 親の `_target_cells` は `verify` → `to_start` の遷移という副作用を
    #       持つので、判定のためだけに呼ぶと状態を壊しかねない。
    # 発動の判定は**事後に集計器で**行う（`traj/*.npz` から実際に走った区画列を
    # 復元し、同じ地図の上の歩数最短経路と突き合わせる）。実際に走った経路を
    # 見るので、方策の内部カウンタより強い証拠になる。
