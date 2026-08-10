# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 言語設定

- 応答は日本語で行うこと
- コードのコメントも日本語で記述すること

## ⚠️ 最初に読むこと（2026-08-10 プログラム再編）

本リポジトリは 2026-08-10 に研究プログラムとして白紙から再編された。**旧 Phase1〜4 系のコード・ドキュメントは凍結レガシーであり、新規作業の参照先ではない**（詳細は下記「レガシー」節）。

必読の正本（この順で読む）:

1. `docs/RESEARCH_PLAN.md` — プログラム研究計画書（ミッション、評価規約、マイルストーン、実験規律、体制）
2. `docs/MODEL_VERIFICATION_PLAN.md` — 物理モデルの確定仕様と検証プロトコル（ロボットパラメータの正本、改訂 r3 現在）
3. `research_notes/` — 研究ノート（最新の番号のノートに直近の意思決定と経緯がある）

## ミッション（要約）

センサ入力（距離センサ 6 本・ジャイロ・加速度計・車輪角速度）→ 左右 2 モータ電圧を**直接**出力する方策を強化学習で獲得し、クラシック競技公式ルール（16x16 未知迷路・持ち時間 300 s・最大 5 走行・最速走行タイム）で評価する。目標速度の中間層やプログラム済み動作プリミティブは最終方策に含めない。学習手段は段階的解禁（純 RL → 停滞時に蒸留解禁、RESEARCH_PLAN §8）。

## 体制（マルチセッション運用）

- **教授セッション**: 方針決定・指示書発行・検収・マイルストーン判定・コミット判断
- **学生セッション**: 実装・学習実行・結果の定量報告
- セッション間連絡は SendMessage、成果共有は git コミット。仕様と矛盾する事実を見つけたら勝手に仕様を変えず教授セッションへ相談すること

## アクティブなディレクトリ（新系統）

- `competition/` — 凍結評価ハーネス（評価迷路 npz・evaluator・古典ベースライン・model_verification）
- `mouse/` — ロボット v2（params・ビルダー・環境）
- `assets/mouse_v2.xml` — 新ロボットモデル（パラメータの正本は MODEL_VERIFICATION_PLAN §4）
- `experiments/` — 実験カード（1 実験 1 変更、RESEARCH_PLAN §9）
- `research_notes/` — 研究ノート（意思決定・考察の記録、テンプレートあり)
- `tests/` — 新系統のテスト

**鉄則**: 評価迷路 seed 1000〜1019 は学習に使用禁止。ロボット物理パラメータの変更は MODEL_VERIFICATION_PLAN の改訂＋検証スイート再実行が必須。ドキュメントとコードの数値を乖離させない。

## レガシー（凍結・変更禁止）

旧 Phase1〜4 系一式は `legacy/`（`legacy/phase1_open/ legacy/phase2_slalom/ legacy/phase3_maze/ legacy/phase4_speed/`、旧 `assets/micromouse_*.xml`、`models/*.zip`、旧 README.md の記述）へ移設済み（2026-08-10）。**過去実験の再現性保全のため凍結**。既知の重大欠陥（質量が非現実的・キャスタ実効摩擦 μ=1・壁すり抜け等、MODEL_VERIFICATION_PLAN §2、要約は `legacy/README.md`）を含むため、**新規作業で流用・参照しないこと**。

## セッション再開手順

1. **自分の引き継ぎメモ `handover/<役割>.md` を読む**（規約は `docs/SESSION_HANDOVER.md`）
2. `git log -5 --oneline` と直近コミットの Next steps を確認
3. `research_notes/` の最新ノートを読む
4. `git status` で未コミット変更を確認（他セッションの作業中ファイルの可能性あり — 勝手に消さない・コミットしない）
5. 役割が不明なら、まず教授セッション宛て（ListAgents で確認）に SendMessage で確認

## 引き継ぎメモの維持（必須）

長時間稼働すると動作が重くなり作業途中で停止することがあるため、**定期的にまとめてセッションを入れ替える**運用とする。各セッションは `handover/<役割>.md` に現況（進行中タスク・未コミット変更・直近の裁定・詰まっている点・次にやること）を**常時維持**すること。更新の契機はコミット時・作業の区切り・**中断するとき**。詳細は `docs/SESSION_HANDOVER.md`。

## 環境

- Python は `.venv`（mujoco / gymnasium / stable-baselines3 / torch 導入済み）。バージョンは requirements.txt を参照
- Gymnasium API・OutputManager（`common/output_manager.py`、これは新系統でも継続使用）・TensorBoard ログの慣例は踏襲
