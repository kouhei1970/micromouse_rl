# Phase 2: Slalom & Maze Navigation (スラローム・迷路走行)

## 目的 (Purpose)
Phase 1で学習した「Low-Level Controller」を利用し、階層型強化学習（Hierarchical RL）によって迷路内を走行する「High-Level Policy」を学習すること。
具体的には、区画ごとの移動やスラローム走行（直進からスムーズにカーブして次の区画へ入るなど）を実現し、壁や柱に衝突せずにゴールを目指す。

## 環境仕様 (Environment Specifications)
*   **空間:** 16x16区画のマイクロマウス迷路。
*   **区画サイズ:** 1区画 0.18m (180mm)。
*   **床 (Floor):**
    *   テクスチャ: 市松模様 (Checkerboard)。
    *   グリッドサイズ: 0.18m (180mm)。1区画に市松模様が1つ収まる（`texrepeat="20 20"` for 3.6m width）。
    *   位置合わせ: グリッドの交点が柱の中心（原点含む）と一致するように配置。
*   **壁 (Walls):**
    *   配置: 迷路データに基づき配置。
    *   サイズ: 厚さ12mm、高さ50mm。
    *   ビジュアル: 白色のボディ、上端（Top）は赤色。
*   **柱 (Posts):**
    *   配置: グリッドの交点（区画の四隅）。
    *   サイズ: 12mm x 12mm x 50mm。
    *   ビジュアル: 白色。

## タスク (Tasks)
1.  **階層型制御:** High-Level Policyが「次の区画へ移動」「右折」などの抽象的なアクションを出力し、Low-Level Controllerがそれを実行する。
2.  **衝突回避:** 壁や柱にぶつからずに走行する。
3.  **迷路探索/走行:** スタートからゴールまでの経路を走行する。

## 成果物 (Deliverables)
*   **モデルファイル:** `models/phase2_slalom.zip` (High-Level Policy)
*   **迷路定義:** `assets/micromouse_slalom.xml` (自動生成されるMuJoCoモデル)

## 注意事項 (Notes)
*   **Phase 1との連携:** Phase 1のモデル(`models/phase1_open.zip`)をロードして使用する。Phase 1のモデル再学習が必要になった場合、インターフェース（観測・アクション空間）が変わらないように注意する。
*   **ビジュアル仕様:** 壁の「白ボディ・赤トップ」、床の「0.18m市松模様」はこのPhaseの重要な視覚的特徴である。
