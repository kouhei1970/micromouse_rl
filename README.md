# Micromouse RL — センサ→モータ直接学習によるマイクロマウス

MuJoCo 物理シミュレーションと強化学習を用いて、マイクロマウスロボットの制御方策を
センサ入力から左右モータ電圧まで一貫して学習で獲得する研究プログラム。

## プロジェクトの目的

センサ入力（距離センサ 6 本・ジャイロ・加速度計・車輪角速度）から**左右 2 モータの電圧指令**を
直接出力する方策を強化学習で獲得し、マイクロマウスクラシック競技公式ルール（16x16 の未知迷路、
持ち時間 300 秒、最大 5 走行、完走した走行のうち最速タイムが成績）で評価することを最終目標とする。
「目標速度」のような中間層や、あらかじめプログラムされた直進・旋回動作は最終方策に含めない。

学習手段はまず報酬のみの純粋な強化学習から始め、停滞が確認された場合に限り模倣学習・蒸留を
段階的に解禁する（詳細は `docs/RESEARCH_PLAN.md` §1, §8）。教育・学術の両面を副次目的とし、
「どこまでが古典制御で、どこからが学習による制御か」を自律度ラダー（下表）で可視化する。

## 正本ドキュメントの読み順

新規に参加する場合、以下の順に読むこと。

1. **`docs/RESEARCH_PLAN.md`** — プログラム憲章（ミッション、評価規約、マイルストーン、実験規律、体制）
2. **`docs/MODEL_VERIFICATION_PLAN.md`** — ロボット物理モデルの確定仕様と検証プロトコル（パラメータの正本）
3. **`docs/ROBOT_SPEC.md`** — 確定済みロボット仕様・較正結果
4. **`research_notes/`** — 研究ノート（通し番号順。最新ノートに直近の意思決定と経緯がある）

`docs/JA_ENGINEERING_TERMS.md` は本プロジェクトの日本語文章（報告・ノート・コミットメッセージ）が
従う用語辞書であり、文章を書く際はあわせて参照する。

## 自律度ラダー（L0〜L4）と現在地

同一課題・同一評価規約（`docs/RESEARCH_PLAN.md` §2）の下で、運動制御・経路計画・地図記憶の
どこまでを学習が担うかを 5 段階で定義し、教材・比較の物差しとする（詳細は同 §7）。

| Level | 運動制御 | 経路計画・探索 | 地図・記憶 | 実装 |
|---|---|---|---|---|
| L0 | 古典（プログラム動作 + PID） | 古典（足立法） | 古典（明示的な壁地図） | `competition/baseline_classical.py` |
| L1 | 学習（速度追従 NN） | 古典 | 古典 | 旧 Phase1〜3 相当（優先度低） |
| L2 | 学習（センサ→モータ電圧） | 古典（特権情報使用） | 古典 | 中間ベースライン |
| L3 | 学習 | 学習（単走・走行間記憶なし） | なし | M2〜M3 の主対象 |
| L4 | 学習 | 学習 | 学習（方策の内部記憶） | M4〜M5 の主対象・最終目標 |

**現在地: M0 完了・L0 基準確立**（2026-08-10）。物理モデル検証スイート 15/15 全合格、
評価迷路 20 面（seed 1000〜1019）を L0（古典ベースライン）で完走 **20/20**、
完走タイムの中央値 **74.6 秒**（詳細は `research_notes/note_004_model_verification.md`）。
次のマイルストーンは M1（低レベル運動の直接学習、`docs/RESEARCH_PLAN.md` §5）。

## ディレクトリ構成

```
micromouse_rl/
├── competition/      # 凍結評価ハーネス（評価迷路 npz・evaluator・古典ベースライン・model_verification）
├── mouse/            # ロボット v2（物理パラメータ・MJCF ビルダー・Gymnasium 環境）
├── assets/           # MuJoCo XML アセット（mouse_v2.xml が現行ロボットモデル）
├── experiments/      # 実験カード（1 実験 1 変更、exp_NNN_<短い名前>/card.md）
├── research_notes/   # 研究ノート（意思決定・考察の記録、note_NNN_<題名>.md）
├── docs/             # 正本ドキュメント（研究計画・モデル検証計画・ロボット仕様・用語辞書）
├── tests/            # 新系統のテスト（mouse・環境フック・評価器・古典ベースライン）
├── common/           # 継続使用の共通ユーティリティ（OutputManager・可視化）
└── legacy/           # 凍結レガシー（旧 Phase1〜4 一式。新規作業で参照・流用禁止、legacy/README.md 参照）
```

## クイックスタート

### 環境構築

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 固定バージョンで再現する場合: pip install -r requirements-lock.txt
```

### 検証スイートの実行

物理モデル・環境フック・評価器・古典ベースラインの退行がないことを確認する。

```bash
.venv/bin/python -m pytest competition/model_verification.py -q   # ロボット物理モデル検証 15項目
.venv/bin/python tests/test_mouse_v2.py                           # ロボット v2 基本検証
.venv/bin/python tests/test_env_hooks.py                          # 観測フック（量子化・遅延・ノイズ）
.venv/bin/python tests/test_evaluator.py                          # 評価器（走行境界・スタック判定 等）
.venv/bin/python tests/test_baseline.py                           # 古典ベースライン（足立法・追従制御）
.venv/bin/python competition/verify_eval_mazes.py                 # 評価迷路20面の仕様適合・再現性
```

### ベースライン評価の実行

古典ベースライン（L0、足立法 + PID）を評価迷路 20 面で評価する。

```bash
.venv/bin/python -m competition.evaluator \
  --policy competition.baseline_classical:AdachiPolicy
```

結果 JSON は既定で `competition/results/` に保存される。

### 迷路の再生成

評価迷路（seed 1000〜1019）の壁配列 npz は凍結成果物であり再生成・上書きしない。
MuJoCo XML はその npz から再生成可能な派生物であり、以下で作り直せる。

```bash
.venv/bin/python -m competition.regenerate_maze_xml --maze-dir competition/mazes/eval
```

学習用迷路（seed 2000 以降）を新規生成する場合は `competition/maze_gen.py` の
`EvalMazeGenerator`（内部の DFS 生成器 `_dfs_perfect_maze` を含む）を参照。

## legacy/ について

`legacy/` 配下（旧 Phase1〜4、旧ロボットモデル、旧学習済みモデル）は過去実験の再現性保全のために
凍結された参照専用ディレクトリである。総質量が非現実的、キャスタ実効摩擦 μ=1、壁のすり抜け等の
重大な物理モデル欠陥（`docs/MODEL_VERIFICATION_PLAN.md` §2、要約は `legacy/README.md`）を含むため、
**新規の実装・調査・報告書のいずれでも参照・流用しないこと**。
