# exp_027 事前登録 — 摩擦円の使用率 u の掃引と、壊れずに走り切れる上限の測定

- 起票: 2026-08-20（教授セッション・単独運用）
- 根拠: `research_notes/note_031_profile_planner_and_eta.md`
  §「段 5 の結果（2026-08-20・exp_026）: 計画は当たり、実行が持たなかった」の
  「次にやること」1・2・3。exp_026 は摩擦円を 100% 使う計画（`u=1.00`）・
  推測航法のみ（壁センサ補正なし）で 10 迷路すべてが衝突した
  （`η_track=0`）ことを実測した。本実験はその是正 2 点
  （摩擦円の使用率 `u` を下げる／壁センサによる位置補正を足す）を導入し、
  **どこまで `u` を上げると壊れるか**を条件（壁センサ補正あり／なし）ごとに測る。
- 規約: `docs/RESEARCH_PLAN.md` §12（1 実験 1 変更・判定条件は結果を見る前に確定）。
  🔴 本実験は「1 変更」ではなく `u`（6 水準）×壁センサ補正（2 条件）の
  **2 要因の掃引**である。これは実験名（`friction_sweep`）どおり、
  「どこまで上げると壊れるか」という**単一の問い**に答えるために必要な
  最小の掃引であり（`u` だけを掃っても、壁センサ補正の有無で壊れる `u` が
  変わるかどうかが分からない）、複数の独立した問いを同時に検証しているの
  ではない。
- 🔴 **判定条文（`η = T_ideal / T_measured` とその分解 `η_map`・`η_track`、
  目標水準、完走率の扱い）は `note_031` §「判定条文」に確定済みであり、
  本文書では写さない。参照するだけとする。**
  本文書には**この実験に固有の事項**（対象・条件・一次記録・「壊れずに
  走り切れた」の定義・否定対照・限界）だけを書く。

## 1. 変更点

- `classic/fast_planner.py::plan_fast_run` に足した `friction_use`（記号 `u`）
  引数（既定 `1.0`）で、速度計画が使う摩擦円の割合を下げる
  （`vehicle_limits()` の `A_TR`・`A_LAT` の両方に `u` を掛けてから
  `min_time` へ渡す。`V_TOP`・`alpha_yaw_max` は変えない）。
  🔴 `classic/ideal.py::ideal_time_for_path`（判定の分母 `T_ideal` の計算元）
  は本引数の影響を受けない（`u=1.00` 固定のまま）。
- `classic/tracker.py` の `apply_lateral_correction()` に呼び出し元を作った
  （`classic/explorer.py::ClassicExplorer._apply_wall_correction`）。
  直線区間でのみ、壁センサから (a) 横位置・(b) 前後位置のずれを推定して
  推測航法へ補正を差し込む（詳細は同メソッドの docstring）。
  `ClassicExplorer(..., wall_correction=True)` で有効化する
  （既定 `False` = exp_026 と同一の推測航法のみ）。

**変えないもの（同時に動かさない）**:

- `fast_mode="profile"` 固定（`"command"` 条件は本実験の対象外。
  コマンド方式との比較は exp_026 で確定済み）
- 探索（Phase.EXPLORE）と帰還（Phase.RETURN）— 現行のコマンド方式のまま
  （`fast_mode`/`friction_use`/`wall_correction` に関わらず不変。
  `plan_fast_run`/`apply_lateral_correction` は Phase.FAST・Phase.RETURN2
  でのみ参照される）
- `extend_straights=True`・半径配分 `allocation="best"`・`margin=0.005`
  （`plan_fast_run` の既定のまま。exp_026 と同一）

## 2. 対象迷路

`competition/mazes/design_turn_v1/` の全 10 迷路
（seed = 41000, 41001, 41002, 41004, 41005, 41007, 41008, 41009, 41010, 41012）。
`T_ideal` は `experiments/exp_025_s4_slalom/ideal_table.json` の
`t_ideal_slalom`（`u=1.00` の物理限界。exp_026 と同じ、作り直さない）。

## 3. 条件

- **摩擦円の使用率 `u`**: `{0.50, 0.60, 0.70, 0.80, 0.90, 1.00}` の6水準
  （任務指示どおり）。
- **壁センサ補正**: `wall_correction ∈ {False, True}` の2条件。
- 計 6×2=12 条件 ×10 迷路 = **120 走行**（`CompetitionEvaluator(
  time_budget=1500.0, max_runs=5)`。exp_026 と同じ広い持ち時間 — 理由も
  同じ: 探索・帰還が持ち時間を使い切り最短走行の標本が失われるのを防ぐ。
  最短走行のタイム自体には影響しない）。

| 条件キー | `ClassicExplorerPolicy(...)` |
|---|---|
| `u=<値>` × `wc=off` | `fast_mode="profile", friction_use=<値>, wall_correction=False` |
| `u=<値>` × `wc=on`  | `fast_mode="profile", friction_use=<値>, wall_correction=True` |

🔴 **測定は前景で完走させること**（`run_exp027.py` を参照。過去に評価途中で
バックグラウンドへ投げてプロセスが落ちた事故があるため、1回のコマンド
実行が長時間になりすぎないよう `--only-u`/`--only-wall-correction` で
チャンクに分けて逐次実行できるようにしてある）。

## 4. 一次記録

`outputs/exp_027/u_<u:.2f>/wc_<off|on>/maze_<seed>.json`。
`competition.evaluator.CompetitionEvaluator.evaluate_maze()` の結果 dict に、
`policy.get_plan_ids()`（`plan_ids`）・`policy.get_run_phases()`（`run_phases`）・
方策が学習した地図（`maze_v_walls`/`maze_h_walls`。真値ではない。読み取り専用の
後処理）・`friction_use`・`wall_correction` を足したもの（exp_026 の一次記録と
同じ形に、掃引の条件値2つを追記しただけ）。

## 5. `t_plan` の再計算について

`t_plan`（速度プロファイルの理想時間の合計）は一次記録に残らない（exp_026
PREREG §5 と同じ理由）。`judge.py` は一次記録に保存された学習地図に対して
`classic.fast_planner.plan_fast_run(maze, start=(0,0), goals=中央2x2,
start_heading=Direction.N, friction_use=<その走行と同じu>)` を呼び直して
`t_plan` を独立に再計算する。🔴 **`friction_use` はその走行が実際に使った値と
必ず一致させる**（`u` を下げた計画の `t_plan` を `u=1.00` で計算し直すと
`η_track`/`η_map` の分解が意味を失う）。

## 6. 「壊れずに走り切れた」の定義（本実験固有・結果を見る前に確定）

judge.py は水準（`u`, `wall_correction`）ごとに、10 迷路それぞれについて
`η` が計算できたか（= その迷路で「段階が FAST でゴールした走行」が
5 走行のうち最低 1 本でも成立したか）を見る。**衝突した迷路数**は
「`η` を計算できなかった迷路の数」と定義する（note_031 の判定条文どおり、
完走率そのものは合格の門にしないが、「壊れた/壊れていない」を判定するための
本実験固有の指標として使う）。

**「壊れずに走り切れた」水準 = 10 迷路すべてで `η` が計算できた水準**
（衝突した迷路数 = 0）と定義する。各条件（`wall_correction` あり/なし）に
ついて、この定義を満たす水準のうち**最大の `u`** を印字する
（満たす水準が1つも無ければ「無し」と印字する）。

## 7. 否定対照（対で。結果を見る前に確定）

1. **回帰**: `u=1.00` かつ `wall_correction=False` の一次記録が、`exp_026`
   の `profile` 条件の一次記録と**同一**になること（`plan_ids`・`runs[].
   outcome`・`runs[].run_time` を突き合わせる）。挙動を何も変えていない
   はずの組み合わせで数値が変わっていたら、実装のどこかに副作用がある
   ということなので、これを崩さないことを直接検査する。
2. **壁センサ補正の対**: 同じ `u` で `wall_correction=True` と `False` の
   一次記録を比べ、`True` 側は `False` 側と異なる（`plan_ids`・`runs[].
   run_time` のいずれかが変わる）こと。`False` 側は exp_026 と一致する
   （対照1と同じ）。

## 8. 単調性（結果を見る前に確定）

同一の学習地図（迷路・`wall_correction` の条件が同じ一次記録から再計算した
`t_plan`）について、`u` を下げるほど `t_plan` が単調に増えること
（`classic/fast_planner.py` 側の単体検査 `tests/test_fast_planner.py::
test_friction_use_lowers_speed_without_changing_the_route` と同じ性質を、
実際に測定に使った学習地図でも確認する）。

## 9. 限界（結果を見る前に明記しておく）

- 学習地図は `wall_correction`/`u` によって走行そのものが変わるため
  迷路ごとに異なりうる（同一迷路でも条件間で経路選択が変わる可能性がある。
  `η_map` はその条件の学習地図に対する値であり、条件をまたいで単純比較
  できるとは限らない）。
- 壁センサによる前後位置補正（`_apply_wall_correction` (b)）は「側方壁の
  確定状態が反転した区画境界でしか働かない」ため、側方に開口部の無い
  区画（両側とも壁が最初から最後まで続く直線）では前後補正が一度も
  発火しない場合がある（横位置補正は両側/片側どちらかが確定していれば
  毎ティック働くので影響は横位置ほど大きくない）。
