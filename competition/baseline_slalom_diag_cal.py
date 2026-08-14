#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**斜め方策を新既定の積み上げへ載せ替えたもの** — `card_016diag_switch.md` §2-1。

2026-08-14 新設（教授裁定 = `card_016g.md` §9-1 の載せ替えカードの起票と GO）。

--------------------------------------------------------------------------
何を載せ替えるのか
--------------------------------------------------------------------------
`competition/baseline_slalom_diag.py`（016-D 版）は、**新既定の積み上げより前の版**で、
**F0・F0-b・校正済み安全率 0.75・45° クロソイドのいずれも入っていない**。
本モジュールはその 4 つを載せた版である。

| 段階 | 中身 | 入れ方 |
|---|---|---|
| **F0** | 加速度の前置補償 | `VelocityLoopMixin`（`_wheel_targets_to_voltage`） |
| **F0-b** | 参照の弧長内挿 | `ReferenceInterpMixin`（`_do_drive_control` の参照の読み方だけ） |
| **cal** | 校正済み旋回安全率 **0.75** | `safety_factor` の既定値 |
| **016-G** | **45° の遷移にクロソイド**（$L_c$ = 47.1239 mm） | 参照経路を作る関数の差し替え（下記） |

**混ぜ込みの順序は `baseline_slalom_e1t_tr_f0b.py` と同じ**にしてある
（**新既定と同じ制御が乗っていることを、継承の連なりで示せる形にするため**）。

--------------------------------------------------------------------------
なぜ新しいファイルを作るのか（作法 1）
--------------------------------------------------------------------------
**`baseline_slalom_diag.py`（016-D 版）は変更しない。**
016-D の記録（確保済みの評価用 20 迷路で +35.6 %）は本カードの比較の起点であり、
**書き換えると再現できなくなる。**

--------------------------------------------------------------------------
参照経路の差し替え方（**親の `_replan` を写さない**）
--------------------------------------------------------------------------
親の `_replan` は**モジュール大域の `build_diagonal_path` を呼ぶ**。
**親の `_replan` を写して 1 箇所だけ変えると、写しが親からずれても気づけない**ので、
**呼び出しの間だけモジュール属性を差し替えて親をそのまま呼ぶ**:

    saved = mod.build_diagonal_path
    mod.build_diagonal_path = <クロソイド版>
    try:    return super()._replan(...)
    finally: mod.build_diagonal_path = saved

**`finally` で必ず戻す**ので、例外が出ても元の関数が残ることはない。
⚠️ **この方式は「1 プロセスの中で単一スレッドが走る」ことを前提にしている。**
本リポジトリの評価は 1 プロセス 1 方策 1 迷路を順に回すので前提は満たされる
（**並列化するときは方式を見直すこと**）。

--------------------------------------------------------------------------
⚠️ 載せ替えていないもの（**1 実験 1 変更を守るため**）
--------------------------------------------------------------------------
- **経路選択の費用モデル `r_speed` = 0.814 は 016-D のまま**である。
  `card_016d.md` §7-4 は「**問いに答える定義なら 0.56**」（対照の方策が実際に出す
  速度 0.806 m/s を分母にした値）と記録しているが、**ここを変えると
  「制御の載せ替え」と「経路選択の変更」の 2 変更になる**ので触らない。
  **限定として持ち回る**（`card_016diag_switch.md` §7）
- **斜め区間の速度上限 `v_diag` = 0.45 m/s も 016-D のまま**
  （**016-G の主判定と同じ値**。探索しない）
"""
import sys

# ⚠️ **この import は `clothoid_path` より先に置く**。
# `baseline_slalom_diag` が `experiments/exp_016_diagonal` を `sys.path` へ足すので、
# その後でないと `clothoid_path` が見つからない。
from competition.baseline_slalom_diag import SlalomDiagPolicy  # noqa: I001
from competition.baseline_slalom_e1t_tr_f0b_cal import SAFETY_FACTOR_CALIBRATED
from competition.reference_interp import ReferenceInterpMixin
from competition.velocity_loop import VelocityLoopMixin

from clothoid_path import build_clothoid_path  # noqa: E402

# 016-G の採用値（`card_016g.md` §6-4）。**幾何が許す全量 = R·θ**であり、
# 係数の探索はしていない。45° の遷移にだけ入れる（教授裁定 (a)）
L_C_CLOTHOID_M = 0.0471239
CLOTHOID_TURNS_DEG = (45,)


class SlalomDiagCalPolicy(ReferenceInterpMixin, VelocityLoopMixin, SlalomDiagPolicy):
    """斜めあり ＋ 新既定の積み上げ（F0 ＋ F0-b ＋ 校正 0.75 ＋ 45° クロソイド）。

    **`L_c` = 0 を渡すと、参照経路は 016-D 版とビット単位で同じ**になる
    （`clothoid_path.build_clothoid_path` の性質。`tests/test_clothoid_path.py` が検査済み）。
    **無害性の確認はこれを使う。**
    """

    name = "L0-c+E1T+TR+DIAG+F0+F0b+cal0.75+clothoid45"

    def __init__(self, *args, L_c: float = L_C_CLOTHOID_M,
                 clothoid_turns=CLOTHOID_TURNS_DEG, **kw):
        kw.setdefault("safety_factor", SAFETY_FACTOR_CALIBRATED)
        kw.setdefault("k_acc_ff", 1.0)          # F0: 物理から導いた全量
        kw.setdefault("ref_interp", True)       # F0-b: 参照を弧長で内挿して読む
        super().__init__(*args, **kw)
        self.L_c = float(L_c)
        self.clothoid_turns = tuple(clothoid_turns)
        self._diag_mod = sys.modules[SlalomDiagPolicy.__module__]

    # ------------------------------------------------------------------
    def _build_path(self, nodes, dirs, cell_size, R, **kw):
        """親が呼ぶ `build_diagonal_path` と同じ形。クロソイド版へ委ねる。"""
        kw.setdefault("L_c", self.L_c)
        kw.setdefault("turns", self.clothoid_turns)
        return build_clothoid_path(nodes, dirs, cell_size, R, **kw)

    def _replan(self, x: float, y: float, yaw: float) -> None:
        """**親をそのまま呼ぶ。**参照経路を作る関数だけを呼び出しの間だけ差し替える。

        **親の `_replan` を 1 行も写していない**ので、親が変わっても追随する。
        """
        mod = self._diag_mod
        saved = mod.build_diagonal_path
        mod.build_diagonal_path = self._build_path
        try:
            return super()._replan(x, y, yaw)
        finally:
            mod.build_diagonal_path = saved
