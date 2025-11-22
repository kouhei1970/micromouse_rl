# Phase 1: オープンフィールドにおける走行制御と安定化 (Open Field Control & Stabilization)

**記録日:** 2025年11月22日
**担当:** GitHub Copilot (Gemini 3 Pro)

## 1. 目的 (Objective)
壁のない無限のオープンフィールドにおいて、指定された目標速度（直進速度 $v$、角速度 $\omega$）に対して、定常偏差なく、かつ物理的に安定して追従できる制御則を強化学習（PPO）によって獲得する。

## 2. 直面した課題 (Challenges)
1.  **ピッチング（前後の傾き）:** 急加減速時にロボットが前後に激しく傾き、転倒したり挙動が不安定になる現象が発生。
2.  **定常偏差 (Steady-state Error):** 目標速度に対して、実際の速度がわずかに届かない（あるいは超過する）状態が続き、誤差がゼロに収束しない。
3.  **センサーの床誤検知:** 走行中の振動や傾きにより、距離センサーが床を壁として誤検知し、観測値がチラつく（ノイズが乗る）。
4.  **学習のトレードオフ:** 直進性能を上げると旋回性能が下がる（またはその逆）現象。

## 3. 実施した対策 (Solutions Implemented)

### 3.1 物理モデルの改修 (`generate_open_maze.py`)
*   **キャスタの追加:**
    *   シャーシの前方 (`caster_front`) と後方 (`caster_back`) に、摩擦係数ゼロの球体（サイズ 0.002）を追加。
    *   **効果:** 接地点を増やすことでピッチングを物理的に抑制し、転倒を防止。
*   **センサー角度の調整:**
    *   4つの距離センサーの `zaxis` を `0.026` (約1.5度上向き) に設定。
    *   **効果:** 15cm先で床から約4mm浮く計算となり、振動による床の誤検知（チラつき）を解消。

### 3.2 報酬関数の改良 (`micromouse_env.py`)
*   **積分項 (Integral Term) の導入:**
    *   速度誤差の累積値（積分値）を報酬のペナルティ項に追加。
    *   **数式:** `Reward = 1.0 - (2.0 * |lin_err| + 1.0 * |lin_integral| + 1.0 * |ang_err| + 1.0 * |ang_integral|)`
    *   **係数調整:** 当初 `5.0` であった積分項の係数を `1.0` に緩和し、過剰な振動（オーバーシュート）を抑制。

## 4. 現在のステータス (Current Status)

*   **最新モデル:** `ppo_micromouse_open.zip`
*   **環境ファイル:** `micromouse_open.xml` (生成スクリプト: `generate_open_maze.py`)
*   **学習スクリプト:** `train_open_field.py`

### 性能評価 (Performance Metrics)
直近の追加学習（約60万ステップ）後の評価値:
*   **Linear Velocity MSE:** 0.0289 (直進安定性は良好)
*   **Angular Velocity MSE:** 0.8373 (旋回追従に若干の遅れあり)

### 挙動
*   キャスタのおかげで急停止・急発進でも転倒しない。
*   目標速度に対してスムーズに追従するが、直進と旋回の精度にはトレードオフが見られる。

## 5. 再開時の手順 (How to Resume)

このフェーズの学習を再開、または微調整する場合は以下の手順で行う。

1.  **環境の準備:**
    ```bash
    python generate_open_maze.py  # 最新の物理設定でXMLを生成
    ```

2.  **学習の実行:**
    ```bash
    python train_open_field.py
    ```
    *   `train_open_field.py` は既存の `ppo_micromouse_open.zip` があればそれをロードして学習を継続するように記述されているか確認すること（現在は新規学習または上書き設定になっている場合は、ロード処理を追加する必要があるかもしれません）。
    *   *現状のスクリプトは `model.save` で上書き保存するため、バックアップが必要なら手動で行うこと。*

3.  **パラメータ調整:**
    *   **直進重視にしたい場合:** `micromouse_env.py` の `lin_err` や `lin_integral` の係数を上げる。
    *   **旋回重視にしたい場合:** `ang_err` や `ang_integral` の係数を上げる。

4.  **結果の確認:**
    ```bash
    python visualize_results.py
    ```
    *   `agent_behavior.mp4` で動画を確認。
    *   `tracking_performance.png` でグラフを確認。
