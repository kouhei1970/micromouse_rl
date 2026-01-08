# Stable-Baselines3 + MuJoCo 強化学習入門

このチュートリアルでは、MuJoCo物理シミュレーションとStable-Baselines3を使った強化学習の基礎を解説します。本プロジェクト（マイクロマウス）を題材に、実践的な強化学習の始め方を学びます。

## 目次

1. [強化学習とは](#強化学習とは)
2. [環境構築](#環境構築)
3. [Gymnasium環境の基本](#gymnasium環境の基本)
4. [MuJoCo環境の作成](#mujoco環境の作成)
5. [Stable-Baselines3で学習](#stable-baselines3で学習)
6. [学習結果の評価と可視化](#学習結果の評価と可視化)
7. [本プロジェクトで実践](#本プロジェクトで実践)
8. [次のステップ](#次のステップ)

---

## 強化学習とは

### 直感的な理解：試行錯誤で学ぶ

強化学習（Reinforcement Learning, RL）は、**試行錯誤を通じて学ぶ**機械学習の手法です。

身近な例で考えてみましょう：

**例：自転車の乗り方を覚える**
```
1. 最初はバランスを崩して転ぶ（失敗 → 悪い結果）
2. 少しずつコツをつかむ（成功 → 良い結果）
3. 何度も試すうちに、上手に乗れるようになる
```

このように、「やってみて、結果を見て、やり方を改善する」というサイクルを繰り返すのが強化学習です。コンピュータに「正解」を直接教えるのではなく、**良い結果には報酬を、悪い結果にはペナルティを与え**、コンピュータ自身に最適な行動を見つけさせます。

---

### 他の機械学習との違い

機械学習には大きく3つのアプローチがあります：

| 手法 | 学習方法 | 例 |
|------|---------|-----|
| **教師あり学習** | 正解データから学ぶ | 画像分類（これは猫、これは犬） |
| **教師なし学習** | データの構造を見つける | 顧客のグループ分け |
| **強化学習** | 試行錯誤で報酬を最大化 | ゲームAI、ロボット制御 |

**強化学習が適しているケース：**
- 正解を定義しにくいが、良し悪しは判断できる
- 連続的な意思決定が必要
- シミュレーションで大量の試行が可能

**本プロジェクトの例：**
マイクロマウスをゴールに導く「正解の動き」を人間が定義するのは困難です。しかし「ゴールに近づいたら良い」「壁にぶつかったら悪い」という報酬は簡単に定義できます。強化学習なら、ロボット自身が最適な動きを発見できます。

---

### 基本用語を理解する

強化学習を理解するための重要な概念を、ゲームに例えて説明します。

#### エージェント（Agent）
**学習する主体**です。ゲームで言えば「プレイヤー」に相当します。

```
本プロジェクトでは → マイクロマウスロボット
```

#### 環境（Environment）
**エージェントが行動する世界**です。ゲームで言えば「ゲームの世界」そのものです。

```
本プロジェクトでは → MuJoCoで再現した迷路（壁、床、物理法則を含む）
```

#### 状態・観測（State / Observation）
**環境の現在の情報**です。エージェントが「今どういう状況か」を知るためのデータです。

```
本プロジェクトでは：
- 距離センサーの値（前方の壁までの距離など）
- 現在の速度（どのくらいの速さで動いているか）
- 目標速度（どのくらいの速さで動くべきか）
```

#### 行動（Action）
**エージェントが選択する操作**です。ゲームで言えば「コントローラーの入力」に相当します。

```
本プロジェクトでは：
- 左モーターの電圧（-3V 〜 +3V）
- 右モーターの電圧（-3V 〜 +3V）
```

行動空間には2種類あります：
- **離散行動**: 選択肢から1つ選ぶ（例：上・下・左・右）
- **連続行動**: 実数値で指定（例：モーター電圧 2.5V）← 本プロジェクトはこちら

#### 報酬（Reward）
**行動の良し悪しを示す数値**です。エージェントはこの報酬を最大化するように学習します。

```
本プロジェクトでは：
- 目標速度に近い → 高い報酬（+1.0）
- 目標速度から外れる → 低い報酬（0.0 や負の値）
- 壁にぶつかる → ペナルティ（-10）
- ゴールに到達 → 大きなボーナス（+1000）
```

**報酬設計のポイント：**
報酬の設計は強化学習で最も重要な要素です。報酬が適切でないと、意図しない行動を学習してしまうことがあります。

```
悪い例：「ゴール到達時のみ +1」
→ ゴールに辿り着くまで何も学習できない（報酬がスパースすぎる）

良い例：「ゴールに近づくほど報酬が増える」
→ 少しずつ改善する方向がわかる
```

#### ポリシー（Policy）
**状態から行動への対応規則**です。「この状況ではこう動く」というルールブックのようなものです。

```
ポリシー π: 状態 s → 行動 a

例：
- 前方に壁が近い → 減速する
- 左に曲がりたい → 右モーターを強く回す
```

強化学習の目標は、**最適なポリシーを見つけること**です。

#### エピソード（Episode）
**開始から終了までの一連の流れ**です。ゲームで言えば「1プレイ」に相当します。

```
本プロジェクトでは：
- 開始：ロボットがスタート位置に配置される
- 終了：ゴール到達、壁に衝突、または時間切れ

1エピソードで数百〜数千ステップの行動を実行
学習には数千〜数百万エピソードが必要
```

---

### 強化学習のループ（学習の仕組み）

強化学習は以下のサイクルを繰り返して学習します：

```
    ┌────────────────────────────────────────────────────┐
    │                                                    │
    │   ① 観測                                           │
    │   エージェントが環境の状態を観測する                     │
    │   （センサー値、速度など）                             │
    │                     │                              │
    │                     ▼                              │
    │   ② 行動選択                                        │
    │   ポリシーに従って行動を決定する                        │
    │   （モーター電圧を決める）                             │
    │                     │                              │
    │                     ▼                              │
    │   ③ 行動実行                                        │
    │   環境に行動を適用する                                │
    │   （モーターを回す）                                  │
    │                     │                              │
    │                     ▼                              │
    │   ④ 結果の観測                                      │
    │   新しい状態と報酬を受け取る                           │
    │   （速度が変化、報酬 +0.8）                           │
    │                     │                              │
    │                     ▼                              │
    │   ⑤ 学習                                           │
    │   報酬をもとにポリシーを改善する                        │
    │   （良い行動の確率を上げる）                           │
    │                     │                              │
    └─────────────────────┴──────────────────────────────┘
                          │
                          └──→ ①に戻って繰り返す
```

**具体例で追ってみましょう：**

```
ステップ1:
  観測: 前方の壁まで0.1m、目標速度0.5m/s、現在速度0.3m/s
  行動: 左モーター2.0V、右モーター2.0V（加速）
  結果: 速度が0.45m/sに上昇
  報酬: +0.9（目標に近づいた！）

ステップ2:
  観測: 前方の壁まで0.05m、目標速度0.5m/s、現在速度0.45m/s
  行動: 左モーター2.0V、右モーター2.0V（そのまま加速）
  結果: 壁に衝突！
  報酬: -10（失敗）
  → エピソード終了

学習:
  「壁が近いときに加速するのは良くない」と学ぶ
  → 次回は壁が近いときに減速するようポリシーを更新
```

---

### 価値関数とは（発展的内容）

強化学習では「今の報酬」だけでなく「将来の報酬」も考慮します。

**例：チェスで考える**
```
今すぐ相手の駒を取る（即時報酬: +1）
  → しかし次のターンでクイーンを取られる（将来の報酬: -9）
  → トータルでは損（-8）

駒を取らずに守る（即時報酬: 0）
  → 数手後にチェックメイト（将来の報酬: +100）
  → トータルでは得（+100）
```

このように、**長期的な報酬の合計**を最大化することが重要です。これを評価するのが**価値関数**です。

**割引率（γ: ガンマ）:**
将来の報酬は不確実なので、現在に近いほど重視します。

```
γ = 0.99 の場合:
  1ステップ後の報酬 × 0.99
  2ステップ後の報酬 × 0.99²
  3ステップ後の報酬 × 0.99³
  ...

γが1に近い → 長期的な報酬を重視
γが0に近い → 即時の報酬を重視
```

---

### まとめ：強化学習の全体像

```
┌─────────────────────────────────────────────────────────────┐
│                        強化学習                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【目標】報酬を最大化する最適なポリシーを見つける                   │
│                                                             │
│  【方法】                                                    │
│   1. 環境と相互作用して経験を集める                              │
│   2. 経験から「何が良い行動か」を学ぶ                            │
│   3. ポリシーを改善する                                        │
│   4. 繰り返す（数百万回）                                      │
│                                                             │
│  【本プロジェクト】                                            │
│   エージェント: マイクロマウス                                  │
│   環境: MuJoCo物理シミュレーション                              │
│   行動: モーター電圧（連続値）                                  │
│   報酬: 速度追従、ゴール到達、衝突回避                            │
│   アルゴリズム: PPO                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 環境構築

### 必要なパッケージのインストール

```bash
# 仮想環境の作成（推奨）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 必須パッケージ
pip install mujoco          # 物理エンジン
pip install gymnasium        # 環境インターフェース
pip install stable-baselines3 # 強化学習アルゴリズム

# 可視化用（オプション）
pip install matplotlib opencv-python imageio
```

### インストール確認

```python
import mujoco
import gymnasium as gym
import stable_baselines3 as sb3

print(f"MuJoCo: {mujoco.__version__}")
print(f"Gymnasium: {gym.__version__}")
print(f"Stable-Baselines3: {sb3.__version__}")
```

---

## Gymnasium環境の基本

Gymnasium（旧OpenAI Gym）は、強化学習環境の標準インターフェースを提供するライブラリです。

### 環境クラスの構造

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyEnv(gym.Env):
    """カスタム環境のテンプレート"""

    # レンダリング設定
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # 行動空間の定義
        # 例: 2次元連続値 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 観測空間の定義
        # 例: 4次元連続値
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """環境を初期状態にリセット"""
        super().reset(seed=seed)

        # 初期状態を設定
        observation = np.zeros(4, dtype=np.float32)
        info = {}

        return observation, info

    def step(self, action):
        """1ステップ実行"""
        # 行動を適用して環境を更新

        observation = np.zeros(4, dtype=np.float32)  # 新しい観測
        reward = 0.0                                  # 報酬
        terminated = False                            # 終了条件（ゴール到達等）
        truncated = False                             # 打ち切り（時間切れ等）
        info = {}                                     # 追加情報

        return observation, reward, terminated, truncated, info

    def render(self):
        """環境を描画"""
        pass

    def close(self):
        """リソースを解放"""
        pass
```

### 空間の種類

```python
from gymnasium import spaces

# 離散空間（選択肢から1つ選ぶ）
discrete = spaces.Discrete(4)  # 0, 1, 2, 3 のいずれか

# 連続空間（範囲内の実数値）
box = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

# 多次元離散空間
multi_discrete = spaces.MultiDiscrete([3, 4])  # [0-2, 0-3] の組み合わせ
```

---

## MuJoCo環境の作成

MuJoCo（Multi-Joint dynamics with Contact）は、高速で正確な物理シミュレーションエンジンです。

### MuJoCoの基本概念

1. **モデル（MjModel）**: シミュレーション世界の定義（XML形式）
2. **データ（MjData）**: シミュレーションの現在状態
3. **制御（ctrl）**: アクチュエータへの入力
4. **センサー（sensordata）**: センサー出力値

### XMLモデルファイルの例

```xml
<mujoco model="simple_robot">
  <!-- ワールド設定 -->
  <worldbody>
    <!-- 地面 -->
    <geom type="plane" size="1 1 0.1"/>

    <!-- ロボット本体 -->
    <body name="robot" pos="0 0 0.1">
      <joint type="free"/>  <!-- 自由に動ける -->
      <geom type="box" size="0.1 0.05 0.02"/>

      <!-- 左車輪 -->
      <body name="left_wheel" pos="-0.05 0.06 0">
        <joint name="left_motor" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.02 0.01"/>
      </body>

      <!-- 右車輪 -->
      <body name="right_wheel" pos="-0.05 -0.06 0">
        <joint name="right_motor" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.02 0.01"/>
      </body>
    </body>
  </worldbody>

  <!-- アクチュエータ（モーター） -->
  <actuator>
    <motor name="left_motor" joint="left_motor" gear="1"/>
    <motor name="right_motor" joint="right_motor" gear="1"/>
  </actuator>

  <!-- センサー -->
  <sensor>
    <velocimeter name="velocity" site="robot_site"/>
    <gyro name="gyro" site="robot_site"/>
  </sensor>
</mujoco>
```

### MuJoCo環境クラスの実装

```python
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MuJoCoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, xml_file="model.xml"):
        self.render_mode = render_mode

        # MuJoCoモデルの読み込み
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # 空間の定義
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,),  # nu = アクチュエータ数
            dtype=np.float32
        )

        obs_dim = 4  # 観測次元数（設計による）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.viewer = None
        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # MuJoCoをリセット
        mujoco.mj_resetData(self.model, self.data)

        # 初期状態を設定（位置など）
        self.data.qpos[0] = 0  # x位置
        self.data.qpos[1] = 0  # y位置

        # 物理計算を1回実行してセンサー値を更新
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        # アクチュエータに制御入力を設定
        self.data.ctrl[:] = action * 3.0  # スケーリング例

        # シミュレーションを複数ステップ実行（制御周期調整）
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()
        reward = self._compute_reward(observation, action)
        terminated = self._check_termination(observation)
        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, {}

    def _get_obs(self):
        """センサーデータから観測を構築"""
        # 例: センサーデータの最初の4要素を使用
        return self.data.sensordata[:4].astype(np.float32)

    def _compute_reward(self, obs, action):
        """報酬関数（タスクに応じて設計）"""
        return 0.0

    def _check_termination(self, obs):
        """終了条件のチェック"""
        return False

    def _render_frame(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
```

---

## Stable-Baselines3で学習

Stable-Baselines3は、PyTorchベースの高品質な強化学習アルゴリズム実装を提供するライブラリです。

### 対応アルゴリズム

| アルゴリズム | 行動空間 | 特徴 |
|-------------|---------|------|
| **PPO** | 連続/離散 | 安定性が高く、汎用的。最も推奨 |
| **SAC** | 連続 | サンプル効率が高い |
| **TD3** | 連続 | 高性能だがチューニングが難しい |
| **A2C** | 連続/離散 | シンプルで高速 |
| **DQN** | 離散のみ | 離散行動空間の標準手法 |

---

### PPO (Proximal Policy Optimization)

**推奨度: ★★★★★（初心者に最もおすすめ）**

PPOは現在最も広く使われている強化学習アルゴリズムです。OpenAIが開発し、安定性と性能のバランスが優れています。

**仕組み:**
- ポリシー勾配法の一種で、「ポリシー（行動方針）」を直接最適化
- クリッピングにより、1回の更新でポリシーが大きく変わりすぎることを防止
- Actor-Critic構造：Actor（行動選択）とCritic（状態価値評価）を同時に学習

**メリット:**
- ハイパーパラメータに対してロバスト（デフォルト値でも動くことが多い）
- 連続・離散どちらの行動空間にも対応
- 並列環境での学習が容易

**デメリット:**
- サンプル効率はSACやTD3より劣る（より多くの経験が必要）
- オンポリシー型のため、過去の経験を再利用できない

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,       # 更新前に収集するステップ数
    batch_size=64,
    n_epochs=10,        # 収集したデータを何回学習に使うか
    gamma=0.99,         # 割引率
    clip_range=0.2,     # ポリシー更新のクリップ範囲
    verbose=1
)
```

---

### SAC (Soft Actor-Critic)

**推奨度: ★★★★☆（連続行動空間で高性能）**

SACは最大エントロピー強化学習に基づくオフポリシーアルゴリズムです。探索と活用のバランスを自動調整します。

**仕組み:**
- 報酬の最大化だけでなく、行動のエントロピー（ランダム性）も最大化
- 経験再生バッファを使用し、過去の経験を効率的に再利用
- 2つのQ関数を使い、過大評価を防止

**メリット:**
- サンプル効率が高い（少ない経験で学習可能）
- 探索が自動的に行われる（エントロピー項のおかげ）
- 温度パラメータの自動調整機能

**デメリット:**
- 連続行動空間のみ対応（離散には使えない）
- PPOより実装が複雑で、デバッグが難しい

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1_000_000,  # 経験再生バッファのサイズ
    learning_starts=10000,   # 学習開始前に収集する経験数
    batch_size=256,
    tau=0.005,               # ターゲットネットワークの更新率
    gamma=0.99,
    verbose=1
)
```

---

### TD3 (Twin Delayed DDPG)

**推奨度: ★★★☆☆（上級者向け、高性能）**

TD3はDDPG（Deep Deterministic Policy Gradient）の改良版で、Q値の過大評価問題に対処しています。

**仕組み:**
- 決定論的ポリシー（同じ状態では常に同じ行動を選択）
- Twin：2つのQ関数を使い、小さい方の値を採用
- Delayed：ポリシーの更新頻度をQ関数より低くする
- ターゲットポリシーにノイズを追加（スムージング）

**メリット:**
- 連続制御タスクで高い性能
- DDPGの不安定性を大幅に改善

**デメリット:**
- 連続行動空間のみ
- ハイパーパラメータの調整が必要な場合がある
- 探索のためのノイズ設計が重要

```python
from stable_baselines3 import TD3

model = TD3(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=10000,
    batch_size=256,
    tau=0.005,
    policy_delay=2,          # ポリシー更新の遅延
    target_policy_noise=0.2, # ターゲットポリシーのノイズ
    verbose=1
)
```

---

### A2C (Advantage Actor-Critic)

**推奨度: ★★★☆☆（シンプルで高速）**

A2CはPPOの前身となるアルゴリズムで、よりシンプルな実装です。

**仕組み:**
- Actor-Critic構造
- Advantage（アドバンテージ）を使って分散を削減
- 同期的に複数環境で学習（A3Cの同期版）

**メリット:**
- 実装がシンプル
- 学習が高速（PPOより軽量）
- 連続・離散どちらにも対応

**デメリット:**
- PPOより不安定になりやすい
- ハイパーパラメータに敏感

```python
from stable_baselines3 import A2C

model = A2C(
    "MlpPolicy",
    env,
    learning_rate=7e-4,
    n_steps=5,          # PPOより短いステップで更新
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.01,      # エントロピー係数（探索促進）
    vf_coef=0.5,
    verbose=1
)
```

---

### DQN (Deep Q-Network)

**推奨度: ★★★★☆（離散行動空間の定番）**

DQNはDeepMindが開発した、深層学習とQ学習を組み合わせた手法です。Atariゲームで人間を超える性能を達成し、深層強化学習ブームの火付け役となりました。

**仕組み:**
- Q関数（状態-行動価値関数）をニューラルネットワークで近似
- 経験再生バッファで過去の経験を再利用
- ターゲットネットワークで学習を安定化

**メリット:**
- 離散行動空間で実績のある手法
- サンプル効率が比較的高い
- 理論的に理解しやすい

**デメリット:**
- 離散行動空間のみ（連続には使えない）
- 行動空間が大きいと性能低下

```python
from stable_baselines3 import DQN

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=1_000_000,
    learning_starts=50000,
    batch_size=32,
    tau=1.0,                    # ターゲットネットワーク更新率
    target_update_interval=10000, # ターゲット更新間隔
    exploration_fraction=0.1,   # 探索率の減衰期間
    exploration_final_eps=0.05, # 最終的な探索率
    verbose=1
)
```

---

### アルゴリズム選択ガイド

```
タスクの行動空間は？
├── 離散（上下左右など選択肢から選ぶ）
│   └── DQN を使用
│
└── 連続（モーター電圧など実数値）
    │
    ├── 初心者 or 安定性重視
    │   └── PPO を使用 ★おすすめ
    │
    ├── サンプル効率重視（経験を集めるのが高コスト）
    │   └── SAC を使用
    │
    └── 最高性能を追求（チューニング可能）
        └── TD3 を使用
```

**本プロジェクト（マイクロマウス）での選択理由:**

モーター電圧という連続行動空間を扱い、シミュレーションで大量の経験を集められるため、安定性の高い**PPO**を採用しています。実機での学習など経験収集コストが高い場合は、SACやTD3がより適しています。

---

### 基本的な学習コード

```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# 1. 環境の作成とラップ
env = MyEnv(render_mode=None)

# Monitor: 学習ログを記録
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, filename=os.path.join(log_dir, "monitor"))

# 2. モデルの作成
model = PPO(
    "MlpPolicy",     # 多層パーセプトロンポリシー
    env,             # 環境
    verbose=1,       # ログ出力
    learning_rate=3e-4,
    n_steps=2048,    # バッチサイズ
    batch_size=64,
    n_epochs=10,
    gamma=0.99,      # 割引率
)

# 3. コールバックの設定（チェックポイント保存）
checkpoint_callback = CheckpointCallback(
    save_freq=50000,      # 50000ステップごとに保存
    save_path='./logs/',
    name_prefix='ppo_model'
)

# 4. 学習の実行
model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_callback
)

# 5. モデルの保存
model.save("models/my_model")
```

### 学習済みモデルの読み込みと継続学習

```python
# 既存モデルの読み込み
model = PPO.load("models/my_model", env=env)

# 追加学習
model.learn(total_timesteps=500_000)

# 保存
model.save("models/my_model_v2")
```

### PPOのハイパーパラメータ

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,    # 学習率（デフォルト: 3e-4）
    n_steps=2048,          # 更新前に収集するステップ数
    batch_size=64,         # ミニバッチサイズ
    n_epochs=10,           # 各更新でのエポック数
    gamma=0.99,            # 割引率（将来報酬の重み）
    gae_lambda=0.95,       # GAEのλパラメータ
    clip_range=0.2,        # PPOクリップ範囲
    ent_coef=0.0,          # エントロピー係数
    vf_coef=0.5,           # 価値関数係数
    max_grad_norm=0.5,     # 勾配クリッピング
    verbose=1,
    tensorboard_log="./tb_logs/"  # TensorBoard用
)
```

---

## 学習結果の評価と可視化

### 学習済みモデルでのテスト

```python
from stable_baselines3 import PPO

# モデルの読み込み
model = PPO.load("models/my_model")

# テスト用環境（レンダリングあり）
env = MyEnv(render_mode="human")

# テスト実行
obs, _ = env.reset()
for _ in range(1000):
    # 決定論的に行動を選択
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
```

### 動画の保存

```python
import imageio
import numpy as np

env = MyEnv(render_mode="rgb_array")
model = PPO.load("models/my_model")

frames = []
obs, _ = env.reset()

for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # フレームを取得
    frame = env.render()
    frames.append(frame)

    if terminated or truncated:
        break

env.close()

# MP4で保存
imageio.mimsave("evaluation.mp4", frames, fps=30)
```

### 学習曲線のプロット

```python
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results

# Monitorのログを読み込み
results = load_results("logs/")

# エピソード報酬をプロット
plt.figure(figsize=(10, 5))
plt.plot(results["timesteps"], results["rewards"])
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Learning Curve")
plt.savefig("learning_curve.png")
plt.show()
```

---

## 本プロジェクトで実践

このマイクロマウスプロジェクトを使って、実際に強化学習を体験してみましょう。

### Phase 1: 低レベル速度制御

最も基本的なタスクです。目標速度を追従するモーター制御を学習します。

```bash
# 学習の実行
python phase1_open/train.py
```

**タスク内容:**
- 入力（観測）: 距離センサー値、現在速度、目標速度
- 出力（行動）: 左右モーター電圧
- 報酬: 目標速度との誤差が小さいほど高い

**コードのポイント（`phase1_open/env.py`）:**

```python
# 報酬関数の例
lin_error = self.target_linear_velocity - linear_vel
ang_error = self.target_angular_velocity - angular_vel

reward = 1.0 - (2.0 * abs(lin_error) + 1.0 * abs(ang_error))
```

### Phase 3: 迷路ナビゲーション

Phase 1で学習した低レベル制御を使い、迷路内のゴールを目指します。

```bash
# Phase 1のモデルが必要
python phase3_maze/train.py
```

**階層型RLの構造:**
```
Phase 3 Policy
  → 目標速度（線速度、角速度）を出力
  → Phase 1 Controller
    → モーター電圧を出力
    → MuJoCo Physics
```

### 学習のモニタリング

```bash
# TensorBoardで学習を可視化
tensorboard --logdir=./tb_logs/
```

### 評価の実行

```bash
# Phase 3の評価
python phase3_maze/evaluate.py
```

---

## 次のステップ

### 報酬関数の設計

報酬関数は強化学習で最も重要な設計要素です。

**良い報酬関数の特徴:**
- 目標に近づくほど高い報酬
- スパースすぎない（学習信号が得られる）
- 意図しない抜け道がない

**報酬シェーピングの例:**
```python
# スパースな報酬（学習が難しい）
reward = 1.0 if goal_reached else 0.0

# シェーピングされた報酬（学習しやすい）
distance_to_goal = np.linalg.norm(goal_pos - current_pos)
reward = -distance_to_goal  # ゴールに近いほど報酬が高い
if goal_reached:
    reward += 100  # ボーナス
```

### 観測空間の設計

エージェントに何を見せるかで学習効率が大きく変わります。

```python
# 必要最小限の情報に絞る
observation = np.array([
    distance_to_goal,      # スカラー
    angle_to_goal,         # スカラー
    current_velocity,      # スカラー
    *sensor_readings,      # 距離センサー
])
```

### ハイパーパラメータチューニング

学習がうまくいかない場合:
1. `learning_rate`を下げる（1e-4 → 3e-5）
2. `n_steps`を増やす（2048 → 4096）
3. 報酬のスケールを調整

### 参考リンク

- [Stable-Baselines3 ドキュメント](https://stable-baselines3.readthedocs.io/)
- [MuJoCo ドキュメント](https://mujoco.readthedocs.io/)
- [Gymnasium ドキュメント](https://gymnasium.farama.org/)
- [Spinning Up (OpenAI)](https://spinningup.openai.com/) - 強化学習の理論

---

## まとめ

1. **Gymnasium**で環境インターフェースを実装
2. **MuJoCo**で物理シミュレーションを構築
3. **Stable-Baselines3**のPPOで学習
4. **報酬関数**の設計がタスク成功の鍵

このプロジェクトのPhase 1から始めて、段階的に複雑なタスクに挑戦してみてください。
