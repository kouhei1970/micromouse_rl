# 研究ノート（Research Notes）

運用ルールは `docs/RESEARCH_PLAN.md` §10 に従う。

- ファイル名: 通し番号 `note_NNN_<短い題名>.md`。テンプレートは [template.md](template.md)。
- 書く契機: (a) 方針決定・仕様変更の当日、(b) 実験カード完了時、(c) マイルストーン到達時。
- 実験カード（`experiments/`）が生データ寄りの記録、研究ノートは解釈・意思決定寄りの記録。
- 各ノートは関連するコミットハッシュ・実験カード・前後のノートへ必ずリンクする。
- 最終的にこれらのノートが教育資料と論文体報告書の素材になる。

## 目次

- [note_001: 研究プログラム発足 — 現状監査と憲章制定](note_001_program_kickoff.md)
- [note_002: 関連研究調査 — 新規性の評価](note_002_related_work.md)
- [note_003: キャスタ摩擦欠陥の発見 — MuJoCo 摩擦合成規則の罠](note_003_caster_friction.md)
- [note_004: 物理モデルの検証 — 監査から全 green まで](note_004_model_verification.md)
- [note_005: M0 総括 — 測る仕組みを先に作る](note_005_m0_summary.md)

## 補助資料

- `data/` — 各時点の実測記録（検証スイートの結果 JSON 等）。ノート本文の数値の裏付け。
- `scripts/` — ノートの数値を再生成する検証・診断スクリプト（憲章 §10「図表は再生成可能なスクリプトとともに保存する」）。
