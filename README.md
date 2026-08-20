# Micromouse RL — センサ→モータ直接学習によるマイクロマウス

MuJoCo 物理シミュレーションの上で、マイクロマウスの制御を
**センサ入力から左右モータ電圧まで一貫して学習で獲得する**ことを目指す研究プログラム。
併せて、その到達度を測るための**古典実装と物理限界の物差し**を整備している。

⚠️ **本ファイルに数値を書くときは、必ず出典を併記すること。**
正本と食い違ったときは**常に正本が勝つ**。2026-08-20 の棚卸しで、本ファイルには
持ち時間・センサ本数・学習用 seed など 12 件の失効した記述が残っていた
（経緯は `research_notes/note_032`）。

## 目的

センサ入力（距離センサ・ジャイロ・加速度計・車輪角速度）から左右 2 モータの電圧指令を
直接出力する方策を強化学習で獲得し、マイクロマウスクラシック競技の公式ルールで評価する。
「目標速度」のような中間層や、あらかじめプログラムされた直進・旋回動作は**最終方策には含めない**。
入出力契約・評価規約は `docs/RESEARCH_PLAN.md` §1・§2 が正本。

**過程では何を試してもよい**（2026-08-20 のユーザ決定）。蒸留・模倣・報酬の変更も実験としては可で、
試したことと、できなかった理由は `research_notes/attempts_register.md` に残す。
最終形の入出力契約だけは変わらない。

## 判定基準 — 物理限界からの肉薄度 η

本プログラムの判定量は 1 つだけである（条文は `research_notes/note_031`）。

```
η = T_ideal / T_measured
```

`T_ideal` は**迷路ごと・経路ごと・走行方式ごとに計算し直した物理限界の最小時間**。
摩擦円・トラクション・モータの力-速度特性の下での最小時間速度計画から求める（`classic/profile.py`）。
実装に依存しないので、段階をまたいでも比較できる。

- 行き先は **η → 1**。段階的に 0.60 → 0.70 → 0.80
- 分解して報告する: `η = η_map × η_track`（経路選択の質 × 計画への追従）
- **迷路ごとに出す**（合計や平均だけで語らない）
- **完走率は合格の門にしない**。究極の速度を競うものであり、衝突は走行 1 本を失う代償である

## 正本ドキュメントの読み順

1. `docs/RESEARCH_PLAN.md` — 研究計画書（ミッション・評価規約・マイルストーン・実験規律・体制）
2. `docs/MODEL_VERIFICATION_PLAN.md` — 物理モデルの確定仕様（**パラメータの正本**）
3. `docs/ROBOT_SPEC.md` — 確定済みロボット仕様・較正結果
4. `research_notes/` — 最新番号のノートに直近の意思決定がある。
   現況の中心は `note_029`（旧実装の総括）・`note_030`（再構築計画）・
   `note_031`（判定基準の全面是正）・`note_032`（白紙の AI が外したこと）
5. `research_notes/attempts_register.md` — 試行台帳（何を試し、できなかった理由は何か）

`docs/JA_ENGINEERING_TERMS.md` は日本語文章が従う用語辞書。**§6 のチェックリストを
報告・コミット・文書の前に毎回適用する**。

## 体制

**2026-08-18 から単独運用**（`docs/RESEARCH_PLAN.md` §11・§12-9）。
教授セッション 1 本が方針決定・設計・検収・判定を行い、実装・測定・調査は subagent へ委譲する。
旧マルチセッション体制（学生 A・学生 B・准教授）は停止した。

## 現在地（2026-08-20）

古典実装は 2026-08-19 にゼロから再構築中（旧実装は `dc01c51` で全破棄）。
速度計画・幾何・理想時間の 3 層が完成し、10 迷路の理想時間表が版管理下にある
（`experiments/exp_025_s4_slalom/ideal_table.json`・走行を 1 本も回さずに算出）。

| 要素 | 物理限界 | 現行実装の実測 | η |
|---|---|---|---|
| 直線 1 区画（停止→停止） | 0.360 s | 2.030 s | 17.7% |
| その場 90° 旋回 | 0.173 s | 2.000 s | 8.6% |
| 最短走行 1 本（maze_41001） | 9.175 s | 154.24 s | 5.9% |

実測 154.24 秒の分解: `9.17 s（物理限界）× 2.17（ターン方式）× 4.23（速度上限）× 1.83（制御）`。
損失のほとんどは 2 つの設計上の選択で、制御そのものは直線で 88.9% まで来ている。
現在は速度プロファイル追従への作り替え中（`classic/tracker.py`）。

強化学習トラックは温存中。確定しているのは「短期文脈でも隠れ 32 次元の再帰構造でも届かない」
ことだけで、記憶を持つ構成は未検証（`research_notes/note_026`）。

## ディレクトリ構成

```
micromouse_rl/
├── classic/          # 古典実装（2026-08-19 からゼロ再構築。requires_privileged = False）
│                     #   maze_map flood route sensing motion localization explorer policy checks
│                     #   profile(速度計画) geometry(幾何・干渉判定) ideal(経路→理想時間) tracker(追従)
├── competition/      # 凍結評価ハーネス（評価迷路 npz・evaluator・迷路生成器・model_verification）
├── mouse/            # ロボット v2（物理パラメータ・MJCF ビルダー・Gymnasium 環境）
├── assets/           # MuJoCo XML アセット（mouse_v2.xml が現行ロボットモデル）
├── experiments/      # 実験カード（1 実験 1 変更、exp_NNN_<短い名前>/）
├── research_notes/   # 研究ノート（note_NNN_<題名>.md）と試行台帳
├── docs/             # 正本ドキュメント
├── handover/         # 引き継ぎメモ（professor.md 1 通・80 行以内）
├── verification/     # 検証記録
├── tests/            # 新系統のテスト
├── common/           # 共通ユーティリティ（OutputManager・可視化）
└── legacy/           # 凍結レガシー（旧 Phase1〜4 一式。参照・流用禁止）
```

## クイックスタート

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # 再現には requirements-lock.txt
```

### 検証

```bash
.venv/bin/python -m pytest competition/model_verification.py -q  # 物理モデル検証
.venv/bin/python -m pytest tests/ -q                             # 全テスト
.venv/bin/python -m pytest tests/test_classic_motion.py -q        # 古典の動作生成
.venv/bin/python -m pytest tests/test_profile.py tests/test_geometry.py tests/test_ideal.py -q
```

`tests/test_ideal.py` は幾何探索を伴うため数分かかる。

### 物理限界の確認

```bash
.venv/bin/python research_notes/scripts/check_physical_limits_and_ideal_lap.py
.venv/bin/python experiments/exp_025_s4_slalom/geometry_anchor.py
.venv/bin/python experiments/exp_025_s4_slalom/ideal_table.py   # 10 迷路の理想時間表
```

### 古典実装の評価

```bash
.venv/bin/python -m competition.evaluator \
  --policy classic.policy:ClassicExplorerPolicy \
  --maze-dir competition/mazes/design_turn_v1
```

持ち時間・最大走行数の既定値は `competition/evaluator.py` の `--time-budget` / `--max-runs`
（規約上の値は `docs/RESEARCH_PLAN.md` §2 が正本）。

## seed の規約（重要）

**評価用に予約された seed 帯を学習・調整に使ってはならない。**一度汚染すると復元できない。

- 予約プールは**候補プール全体** `[1000, 40999]`。採用された 20 seed だけではない
  （選ばれなかった seed も同じ生成過程にあり、帯の再作成で使われうる）
- `41000〜61000` は古典高速化トラックの設計帯として消費済み
- **16x16 の学習用迷路 seed は 61001 以降**を使う

正本は `docs/RESEARCH_PLAN.md` §2・§9-7。実装上の安全弁も同 §9-7 に規定がある。
迷路生成器の現行版は `competition/maze_gen_v3.py` 系。`competition/maze_gen.py` の
`EvalMazeGenerator` は規定違反と判定されて退避された旧版であり、新規に使わない。

評価迷路の壁配列 npz は確保済み成果物で、再生成・上書きしない。
MuJoCo XML はその npz からの派生物なので作り直せる。

```bash
.venv/bin/python -m competition.regenerate_maze_xml --maze-dir competition/mazes/eval
```

## legacy/ について

`legacy/` 配下（旧 Phase1〜4・旧ロボットモデル・旧学習済みモデル）は過去実験の再現性保全のために
凍結された参照専用ディレクトリ。総質量が非現実的・キャスタ実効摩擦 μ=1・壁のすり抜けなどの
物理モデル欠陥（`docs/MODEL_VERIFICATION_PLAN.md` §2）を含むため、
**新規の実装・調査・報告のいずれでも参照・流用しない。**
