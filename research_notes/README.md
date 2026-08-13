# 研究ノート（Research Notes）

運用ルールは `docs/RESEARCH_PLAN.md` §10 に従う。

- ファイル名: 通し番号 `note_NNN_<短い題名>.md`。テンプレートは [template.md](template.md)。
- 書く契機: (a) 方針決定・仕様変更の当日、(b) 実験カード完了時、(c) マイルストーン到達時。
- 実験カード（`experiments/`）が生データ寄りの記録、研究ノートは解釈・意思決定寄りの記録。
- 各ノートは関連するコミットハッシュ・実験カード・前後のノートへ必ずリンクする。
- 最終的にこれらのノートが教育資料と論文体報告書の素材になる。

## 目次

- [note_001: 研究プログラム発足 — 現状監査と研究計画書制定](note_001_program_kickoff.md)
- [note_002: 関連研究調査 — 新規性の評価](note_002_related_work.md)
- [note_003: キャスタ摩擦欠陥の発見 — MuJoCo 摩擦合成規則の罠](note_003_caster_friction.md)
- [note_004: 物理モデルの検証 — 監査から全 green まで](note_004_model_verification.md)
- [note_005: M0 総括 — 測る仕組みを先に作る](note_005_m0_summary.md)
- [note_006: 探索戦略 — 足立法の実際と「最短経路の確定」](note_006_exploration_strategy.md)
- [note_007: 隠れていたものは、条件を変えると見える — 指標と動作点の教訓](note_007_hidden_by_metrics_and_operating_point.md)
- [note_008: 走っていない検査と、合っている指標 — 検証の穴の二つの形](note_008_unrun_checks_and_matching_metrics.md)
- [note_009: 決定は引き継がれるが、根拠は引き継がれない](note_009_decisions_outlive_their_grounds.md)
- [note_010: ある軸で多様であることは、効く軸で多様であることを意味しない](note_010_diverse_on_one_axis.md)
- [note_011: 正準な代表を 1 つ選んで描くと、同値な別の解が「誤り」に見える](note_011_canonical_representative_hides_equivalents.md)
- [note_012: 測れていることと、意味が分かっていることは別](note_012_measured_but_not_understood.md)
- [note_013: 対象の制約が、指標を退化させることがある](note_013_constraints_can_degenerate_metrics.md)
- [note_014: 同じ役割を 2 セッションが同時に担うと、指示の出所が失われる](note_014_one_role_two_sessions.md)
- [note_015: 同じ名前の指標が、実装の副次的な規則によって別の量になる](note_015_same_name_different_quantity.md)
- [note_016: 中央値は悪化を過大に見せ、裾の効果をゼロに見せる](note_016_median_hides_the_tail.md)

## 補助資料

- `data/` — 各時点の実測記録（検証スイートの結果 JSON 等）。ノート本文の数値の裏付け。
- `scripts/` — ノートの数値を再生成する検証・診断スクリプト（研究計画書 §10「図表は再生成可能なスクリプトとともに保存する」）。
