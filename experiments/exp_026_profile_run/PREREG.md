# exp_026 事前登録 — 最短走行のプロファイル追従化と η の実測

- 起票: 2026-08-20（教授セッション・単独運用）
- 根拠: `research_notes/note_031_profile_planner_and_eta.md`（判定条文・段 3/段 4 の結果）
- 規約: `docs/RESEARCH_PLAN.md` §12（1 実験 1 変更・判定条件は結果を見る前に確定）
- 🔴 **判定条文（`η = T_ideal / T_measured` とその分解、目標水準、完走率の扱い）は
  `note_031` §「判定条文」に確定済みであり、本文書では写さない。参照するだけとする。**
  本文書には**この実験に固有の事項**（対象・条件・一次記録・否定対照・限界）だけを書く。

## 1. 変更点（1 実験 1 変更）

**最短走行（Phase.FAST）と最短経路での帰還（Phase.RETURN2）だけ**を、現行の
コマンド方式（1コマンド発行→完了待ち→次）から速度プロファイル追従
（`classic/fast_planner.py` の `plan_fast_run()` が作る計画を `classic/tracker.py`
の `ProfileTracker` で追従する）へ差し替える。

- `classic/explorer.py` に `fast_mode: str = "command"` を追加した。
  `"command"`（既定）は変更前と完全に同一のコード経路。`"profile"` のとき
  Phase.FAST/RETURN2 の実行方式が変わる。
- `classic/policy.py` の `ClassicExplorerPolicy` にも `fast_mode` を通す。

**変えないもの（同時に動かさない）**:

- 探索（Phase.EXPLORE）と帰還（Phase.RETURN）— 現行のコマンド方式のまま
  （`fast_mode` に関わらず不変）
- 壁センサによる位置補正（S2）— 本実験は**推測航法のみ**で最短走行を走らせる
  （`ProfileTracker` の `kp_lat` は既定 0 のまま配線しない。任務指示: 「どこまで
  持つのかを先に測るため。補正は次の段で足す」）
- `classic/profile.py`・`classic/geometry.py`・`classic/ideal.py`・`classic/tracker.py`・
  `classic/motion.py` — 触らない

## 2. 対象迷路

`competition/mazes/design_turn_v1/` の全 10 迷路
（seed = 41000, 41001, 41002, 41004, 41005, 41007, 41008, 41009, 41010, 41012）。
`experiments/exp_025_s4_slalom/ideal_table.json` が同じ 10 迷路の `T_ideal`
（`t_ideal_slalom`。start=(0,0)・goals=中央2x2・`mode="slalom"`・`allocation="best"`）
を既に版管理下に持っているので、そのまま η の分母に使う（作り直さない）。

## 3. 条件

`fast_mode` の 2 値。他の設定（`extend_straights=True`・`localization_enabled=True`
— 探索・帰還にのみ効く。FAST/RETURN2 では `fast_mode="profile"` のとき参照されない）
は両条件で同一に固定する。

| 条件キー | `ClassicExplorerPolicy(...)` |
|---|---|
| `command` | `fast_mode="command"`（対照。現行のコマンド方式） |
| `profile` | `fast_mode="profile"`（作動側。速度プロファイル追従） |

`CompetitionEvaluator(time_budget=1500.0, max_runs=5)`。競技規約の 420 秒ではなく
広い持ち時間を使う（`exp_025` PREREG 追記1 と同じ理由: 探索・帰還が S1 の限界で
持ち時間を使い切り、最短走行の標本が失われるのを防ぐため。この広い持ち時間は
判定の対象である FAST/RETURN2 の**タイム自体**には影響しない — タイムは
`runs[].run_time` で走行ごとに個別に確定するため）。

## 4. 一次記録

`outputs/exp_026/<condition>/maze_<seed>.json`。`competition.evaluator.
CompetitionEvaluator.evaluate_maze()` の結果 dict に、`policy.get_plan_ids()`
（`plan_ids`）・`policy.get_run_phases()`（`run_phases`）を足したもの
（`exp_024`/`exp_025` の一次記録と同じ形）。

`judge.py` はこの一次記録**だけ**を入力に、`note_031` の判定条文どおり

```
T_measured(迷路) = 「走行が始まった瞬間の段階が FAST であり、かつゴールした
                     走行」のタイム（run_time）の最小値
                    （exp_024 recompute_anchors.py の compute_t_fast と同じ定義）
η       = T_ideal / T_measured
η_track = t_plan   / T_measured   （t_plan は `plan_fast_run()` が積んだ理想時間の
                                     合計。一次記録には無いので、`judge.py` が
                                     `command` 条件で成立した FAST 走行と**同一の
                                     学習地図**を条件 `profile` の走行から独立に
                                     再現し、`plan_fast_run()` を呼び直して求める
                                     — 走行データそのものは使わず、地図だけから
                                     計算し直す再計算である）
η_map   = T_ideal / t_plan
```

を迷路ごとに計算し、中央値と最小値を印字する。

## 5. `t_plan` の再計算について（🔴 一次記録に無い値の扱い）

`plan_fast_run()` が返す `FastPlan.t_plan`（速度プロファイルの理想時間の合計）は
走行中に捨てられ、`competition.evaluator.CompetitionEvaluator` の一次記録
（`runs[]`/`plan_ids`）には残らない。これを失わず記録するため、`run_exp026.py`
は `evaluate_maze()` の完了後（`profile` 条件のみ）に
`policy._explorer.maze.v_walls`/`h_walls`（方策自身が学習した地図。真値ではない）
を読み出し、一次記録 JSON に `"maze_v_walls"`/`"maze_h_walls"` として書き出す。

🔴 これは走行そのものへ介入しない**読み取り専用の後処理**である
（`act()`/`tick()` の電圧計算には一切関与しない）。**最短走行中は地図を
書き換えない**（`classic/explorer.py` モジュール docstring「S3: 最短走行」）ため、
評価器が完了した時点の地図は、最初の FAST 走行が計画に使った地図と厳密に一致する
（Phase.FAST に一度入ると、探索(EXPLORE)・帰還(RETURN) — 地図を書き換える唯一の
2 段階 — へは二度と戻らない設計であるため）。`judge.py` はこの地図に対して
`classic.fast_planner.plan_fast_run(maze, start=(0,0), goals=中央2x2,
start_heading=Direction.N)` を呼び直して `FastPlan.t_plan` を得る
（走行データそのものではなく、地図だけを入力にした決定的な再計算）。

## 6. 否定対照（対で）

| # | 壊し方 | 期待 | 何の確認か |
|---|---|---|---|
| N1 | 壊さない（`fast_mode="profile"`） | `plan_ids` の FAST/RETURN2 区間に `"profile"` を含む plan_id が現れる | profile 追従が実際に使われている（作動側） |
| N2 | 壊さない（`fast_mode="command"`） | `plan_ids` の FAST/RETURN2 区間に `"profile"` を含む plan_id が**一切現れない**（変更前と同じ語彙のまま） | 既定は完全に元のまま（空振り側） |
| N3 | `ProfileTracker` の電圧前置き（`TrackerGains.use_voltage_feedforward`）を `False` にする | 同一迷路の `profile` 条件の `T_measured` が既定（`True`）より**悪化する**（`note_031` 段4の実測 — 前置き無しでは終端角速度が大幅に悪化する — と整合する方向に動く） | 前置きが結果に効いている（作動側） |
| N4 | N3 の壊し方を適用しない（既定 `use_voltage_feedforward=True`） | N1 と同一の `T_measured` が出る（同一条件の再実行なので一致） | 検査自体の空振り側 |

N3/N4 は 10 迷路のうち代表 1 迷路（`design_turn_v1` の最短歩数最小・maze_41001）
だけで確かめる（10 迷路全部を 2 系統で回すのは本実験の主目的ではないため）。

## 7. 限界

1. 主判定（η の表）は競技の持ち時間の中で成立した走行ではない
   （`time_budget=1500.0` で成立した最初の FAST 走行を取る。`exp_025` PREREG §11-1 と同じ限界）。
2. 本実験は S2（壁センサ位置補正）を profile 経路に配線しない。推測航法だけで
   どこまで走れるかを測るのが目的であり、衝突・スタックで η が定義できない迷路が
   出ることは想定内（判定の合否条件にはしない — `note_031` 判定条文「完走率は
   合格の門にしない」）。
3. `t_plan` は §5 のとおり地図を保存し直す形で再計算する。走行そのものとは
   独立な計算だが、`plan_fast_run()` が使う `RobotParams` は評価器の既定値と
   同一である前提に立つ（両者とも `RobotParams()` を明示せず既定値のまま使う）。
4. 迷路は `design_turn_v1`（ターンの多い調整用迷路）10 面のみ。実戦帯の迷路では測らない。
5. `η_track`/`η_map` の分解は探索・帰還が両条件で完全に同一であること
   （§6 N1/N2 の空振り側が保証する部分集合）を前提にする。もし探索経路が
   条件間で食い違う迷路が出たら、その迷路は η の表にそのまま出し、
   食い違いがあったことを報告に明記する（黙って揃えない）。
