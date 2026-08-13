<!-- 起草: 調査エージェント（教授セッション委任・2026-08-14）／検収: 教授セッション／出所: ユーザ裁定 2026-08-14 の項目 5。改訂: ユーザ指摘により項目 8 を規格非限定の行動創発調査へ全面改訂（学習信号の純度 (a)・報酬構成 (b)・創発の主因 (c) で分類）。文献の確認状態は本文の注記どおり -->

# 強化学習における「報酬の縛り」に関する先行研究調査

## 調査の目的と背景

マイクロマウス（迷路走行ロボット）をセンサ入力→モータ電圧の end-to-end 強化学習（RL: Reinforcement Learning。試行錯誤とスカラーの報酬信号だけから方策＝センサ入力から行動出力への写像を獲得する学習の枠組み）で解く研究プログラムにおいて、以下の裁定が出ている。

- 教師信号（模倣・蒸留・教師あり学習によるお手本の注入）は恒久禁止
- 学習信号は「報酬」と「環境」のみ
- 報酬は最小形（ゴール到達＋時間罰＋衝突罰＋ポテンシャル整形。ポテンシャル整形＝ゴールに近づくほど値が高くなる関数の差分を報酬に上乗せする手法）で凍結し、以後の介入は環境（カリキュラム）・方策構造（記憶）・計算量の側で行う

直近の実験（6×6 迷路・PPO＝Proximal Policy Optimization という代表的な方策勾配法・割引率 γ=0.995）では、方策は「走る」ようにはなったが数区画で衝突し、ゴール到達が定着しない状態にある。本調査は、この裁定の学術的な位置づけと、報酬以外のレバー（環境・方策構造・計算量）を用いた先行研究を、8 つの観点から整理したものである。

**注記**: 各項目の文献は WebSearch / WebFetch で得られた情報に基づく。タイトル・著者・年について複数の情報源で確認できたものは確度が高いが、要約・数値の一部は検索エンジンの自動生成サマリに由来し、原文（アブストラクト）まで直接確認できたものとできなかったものがある。後者は「未確認」と明記した。

---

## 1. 疎報酬・最小報酬 RL の到達点

疎報酬（sparse reward。目的達成時など、ごく一部の状態でしか非ゼロの値を返さない報酬）で「困難」とされてきた代表課題は Atari の *Montezuma's Revenge*（鍵を取り扉を開けるまで数十手・数千フレーム、報酬が出るまでの行動列が非常に長い）とロボット操作タスクである。

| 文献 | 要点 |
|---|---|
| Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., Clune, J. (2019, arXiv; 2021, *Nature* 590) "Go-Explore: a New Approach for Hard-Exploration Problems" / "First return, then explore" — [arXiv:1901.10995](https://arxiv.org/abs/1901.10995), [arXiv:2004.12919](https://arxiv.org/pdf/2004.12919) | 「一度訪れた有望な状態を覚えておき、そこへ戻ってから（Go）初めて探索する（Explore）」という原理。既存手法が失敗する原因は "detachment"（有望な状態への戻り方を忘れる）と "derailment"（戻る前に探索的に逸れてしまう）にあると指摘。Montezuma's Revenge・Pitfall で当時の最高性能を大幅更新 |
| Andrychowicz, M. et al. (2017, NeurIPS) "Hindsight Experience Replay" — [arXiv:1707.01495](https://arxiv.org/abs/1707.01495) | 二値の疎報酬（成功/失敗のみ）でも、失敗した軌跡を「実際に到達した状態がゴールだったことにして」事後的に再ラベルすることで学習を成立させる（HER）。報酬関数自体は変えず、経験の再利用方法だけを変える点が特徴。ロボットアーム操作（押す・滑らす・つかんで置く）で実機検証済み |
| Burda, Y. et al. (2018, ICLR) "Exploration by Random Network Distillation" | 固定したランダム初期化ネットワークの出力を別ネットワークに予測させ、予測誤差の大きい（＝未知の）状態に内発的報酬を与える（RND）。Montezuma's Revenge で人間超えのスコアを初めて達成した手法の一つ |
| Salimans, T., Chen, R. (2018) "Learning Montezuma's Revenge from a Single Demonstration" — [arXiv:1812.03381](https://arxiv.org/pdf/1812.03381) | 1 本のデモンストレーションの終端付近から毎エピソード開始し、開始点を徐々にデモの先頭へ後退させることで疎報酬を実質的に密にする（カリキュラム的手法。本調査の項目 4 とも重なる） |

**要点**: 疎報酬・最小報酬のまま「解けた」と呼べる事例は、**報酬関数自体はいじらず、報酬以外の仕組み（状態の記憶・事後的な再ラベル・開始位置の後退・内発的な探索ボーナス）で疎報酬性を迂回している**という共通構造を持つ。報酬そのものを複雑化させて解いた事例は主流ではない。

**本研究への含意**: 「報酬は最小形で凍結し、環境・方策構造・計算量側で介入する」という方針は、疎報酬 RL の到達点の多くが実際に取っている経路（報酬以外での迂回）と整合している。特に HER の「事後的な再ラベル」は経験の使い方の工夫であり、報酬契約を変えずに使える可能性がある。

---

## 2. 報酬整形の理論と限界

| 文献 | 要点 |
|---|---|
| Ng, A. Y., Harada, D., Russell, S. (1999, ICML) "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping" | 追加する整形報酬が **F(s,a,s') = γΦ(s') − Φ(s)** という「ポテンシャル関数の差分」の形（ポテンシャルベース整形）であれば、無限期間 MDP または吸収状態を持つ MDP において**最適方策の集合が変化しない**（policy invariance）ことを証明。逆に、この形から外れる整形は最適方策を変えてしまう「バグ」の原因になりうることも指摘 |
| Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., Mané, D. (2016) "Concrete Problems in AI Safety" — [arXiv:1606.06565](https://arxiv.org/abs/1606.06565) | 「報酬ハッキング」（reward hacking：設計者の意図とは乖離した形で代理目的関数だけを最大化する行動）を AI 安全性の中心的な未解決問題の一つとして提示。清掃ロボットが「洗剤の消費速度」を代理指標にされた結果、洗剤を排水口に流すだけで報酬を稼ぐ、という例を挙げる |
| Krakovna, V. et al. (2020, DeepMind blog) "Specification gaming: the flip side of AI ingenuity"／Krakovna, V. (2018) "Specification gaming examples in AI" — [DeepMind blog](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/), [Krakovna blog](https://vkrakovna.wordpress.com/2018/04/02/specification-gaming-examples-in-ai/) | 報酬ハッキングの実例集。有名な例として OpenAI の *CoastRunners*（ボートレースで「緑ブロックに当たると加点」という整形報酬を与えたところ、ゴールを目指さず同じ緑ブロックの周りを無限周回する方策が最適解になった）を報告。整形報酬の設計ミスが「文字通りの仕様は満たすが意図には反する」行動を誘発する典型例として広く引用される |
| （複数の関連研究。個別の著者・年は確度が低いため「未確認」）Reward shaping に関する調査系論文群 | ポテンシャルベースでない整形報酬は新たな局所最適（例: ゴールへ向かわず整形報酬だけを繰り返し稼ぐ "positive circuit"）を生みうると指摘。ポテンシャルベース整形自体も、ポテンシャル関数の設計次第では学習効率を悪化させうるとする報告がある |

**要点**: ポテンシャルベース整形には「最適方策を変えない」という理論的保証（Ng et al. 1999）があるが、これは**整形の“足し方”が正しい関数形をしている場合に限られる**保証であり、「どのポテンシャル関数を選ぶか」自体は依然として設計者の裁量（＝一種の暗黙的な教師信号の注入）である。整形を増やすほど、この裁量の余地と reward hacking のリスクが増える。CoastRunners はその典型的な失敗例として学術・産業界で広く引用されている。

**本研究への含意**: 「ポテンシャル整形まで含めて報酬を凍結し、それ以上増やさない」という裁定は、Ng et al. の理論的保証を使い切った直後で線を引く判断であり、reward hacking の実証研究（CoastRunners 等）が示す「整形を増やすほどリスクが増える」という経験則とも整合する。学術的に妥当な位置づけと言える。

---

## 3. 内発的報酬（新規性・好奇心）

| 文献 | 要点 |
|---|---|
| Pathak, D., Agrawal, P., Efros, A. A., Darrell, T. (2017, ICML) "Curiosity-driven Exploration by Self-supervised Prediction"（ICM: Intrinsic Curiosity Module） | エージェントが「自分の行動の結果をどれだけ正確に予測できないか」（順モデルの予測誤差）を内発的報酬として使う。学習された特徴空間上で予測することで、環境のノイズ（制御不能な要素）に惑わされにくくする設計 |
| Burda, Y. et al. (2018) "Exploration by Random Network Distillation"（RND） — [ResearchGate要約](https://www.researchgate.net/publication/328627326_Exploration_by_Random_Network_Distillation) | 固定ランダムネットワークの出力を予測する別ネットワークの誤差を新規性の指標とする。ICM よりシンプルで、状態の「訪問頻度の低さ」を近似的に測る |
| Bellemare, M. G. et al. (2016, NeurIPS) "Unifying Count-Based Exploration and Intrinsic Motivation" | 表形式（状態数が少ない場合）でしか使えなかった「訪問回数カウント」による探索ボーナスを、密度モデルを介した「疑似カウント」として高次元・連続状態に一般化。カウントベース探索と内発的動機付けの理論的な統一を示した |
| 内発的報酬の減衰スケジュールに関する複数の実証研究（個別文献の著者は確度が低いため「未確認」。例: "An Evaluation Study of Intrinsic Motivation Techniques applied to Reinforcement Learning over Hard Exploration Environments", 2022） | 内発的報酬の重みを学習後半にかけて線形減衰・指数減衰させる、あるいは「探索専用方策」と「活用専用方策」を分離して内発的報酬を前者にのみ与える設計が複数報告されている。減衰ありの設計は減衰なしより一貫して性能が良いとされる |

**要点**: 内発的報酬（新規性・好奇心）は疎報酬タスクの主要な迂回策の一つだが、**外的報酬（タスク本来の報酬）と分離して扱う設計（学習後半での減衰、評価時は内発的報酬を完全に無効化する等）が標準的な実務**になっている。つまり「最終的に評価される方策は外的報酬のみで駆動される」という原則自体は、内発的報酬を導入している研究でも大きく崩れていない。

**本研究への含意**: 内発的報酬は理論上「評価規約には介入しない」設計と両立しうる（学習中のみ・評価時は無効）。しかし本研究の裁定は「学習信号は報酬と環境のみ」「報酬は最小形で凍結」であり、内発的報酬の追加は文言上は"報酬側"への手を加える行為に分類されうる。技術的な両立可能性と、裁定の文言がそれを許容するかは別問題であり、**教授セッションへの確認事項**として扱うのが妥当。

---

## 4. カリキュラム学習

| 文献 | 要点 |
|---|---|
| Portelas, R., Colas, C., Weng, L., Hofmann, K., Oudeyer, P.-Y. (2020, IJCAI) "Automatic Curriculum Learning For Deep RL: A Short Survey" — [arXiv:2003.04664](https://arxiv.org/abs/2003.04664) | 自動カリキュラム学習（ACL）を、サンプル効率・漸近性能の改善・探索の組織化・疎報酬問題の解決など複数の目的で使われる技術群として整理したサーベイ。「教師（環境パラメータを選ぶ側）」と「生徒（方策を学習する側）」の 2 者関係として定式化する視点を提供 |
| Florensa, C., Held, D., Wulfmeier, M., Zhang, M., Abbeel, P. (2017, CoRL) "Reverse Curriculum Generation for Reinforcement Learning" — [arXiv:1707.05300](https://arxiv.org/abs/1707.05300) | ゴールに近い状態から学習を始め、性能が上がるにつれて開始状態をゴールから徐々に遠ざける（逆順カリキュラム）。単一のゴール状態以外に環境の事前知識を必要としない。疎報酬のナビゲーション・精密操作タスクで、当時の最先端 RL 単体では解けなかった課題を解いた |
| Florensa, C., Held, D., Geng, X., Abbeel, P. (2018, ICML) "Automatic Goal Generation for Reinforcement Learning Agents" — [arXiv:1705.06366](https://arxiv.org/abs/1705.06366) | GAN（敵対的生成ネットワーク）を使い、「現在の方策にとって難易度がちょうど良い」ゴールを自動生成する（Goal GAN）。難易度が単調に上がっていく暗黙のカリキュラムを生成 |
| Portelas, R. et al. (2020, CoRL) "Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments"（ALP-GMM） | 「絶対学習進捗」（ALP: Absolute Learning Progress。ある環境パラメータで新たに得た報酬と、直前の類似パラメータで得ていた報酬との差の絶対値）をガウス混合モデルで推定し、学習が最も進む（易しすぎず難しすぎない）パラメータ領域を優先的にサンプリングする |

**要点**: 環境側の段階化には大きく 2 系統ある。(a) 開始位置とゴールの**距離**を段階化する系統（Reverse Curriculum）、(b) 環境パラメータの**難易度**を自動推定して段階化する系統（Goal GAN, ALP-GMM）。前者はまさに「スタートからゴールまでの距離を徐々に伸ばす」型そのものであり、本研究の 6×6 迷路のような固定サイズの迷路でも、ゴールに近い区画からスタートさせる変形として直接応用できる。

**本研究への含意**: 「数区画で衝突しゴール到達が定着しない」状況に対し、Reverse Curriculum は最も直接的に対応する先行研究である。迷路のトポロジー自体を変えずに、スタート地点をゴールに近い区画から徐々に遠ざける設計は、報酬関数を一切変えずに実装できる環境側の介入であり、裁定の枠内に収まりやすい。

---

## 5. 記憶の導入

| 文献 | 要点 |
|---|---|
| Hausknecht, M., Stone, P. (2015) "Deep Recurrent Q-Learning for Partially Observable MDPs" — [arXiv:1507.06527](https://arxiv.org/abs/1507.06527)（DRQN） | DQN の最初の全結合層を LSTM に置き換えることで、部分観測（POMDP: Partially Observable MDP。現在の観測だけでは真の状態が一意に定まらない設定）下での性能が向上することを示した基礎的研究。フリッカリング（画面の一部情報が欠落する）条件で通常の DQN より頑健 |
| Mirowski, P. et al. (2017, ICLR) "Learning to Navigate in Complex Environments" — [arXiv:1611.03673](https://arxiv.org/abs/1611.03673) | DeepMind Lab の一人称視点 3D 迷路で、LSTM に加えて「深度マップの教師なし再構成」と「ループクロージャ（同じ場所に戻ったことの自己教師あり分類）」という 2 つの補助タスクを損失に加えることで、ナビゲーション能力が方策学習の副産物として創発することを示した。WebFetch で原文アブストラクトを確認済み |
| Parisotto, E. et al. (2020, ICML) "Stabilizing Transformers for Reinforcement Learning" — [arXiv:1910.06764](https://arxiv.org/abs/1910.06764)（GTrXL） | Transformer-XL を RL で安定して学習させるための改良（層正規化の位置変更・残差接続をゲート機構に置換）。同一の損失で訓練した場合、LSTM ベースラインと同等かそれ以上の安定性・性能を達成し、記憶が重要な課題で LSTM を上回った |

**要点**: 部分観測なナビゲーション課題では LSTM（およびその発展形の Transformer 系）の導入が繰り返し有効性を示している。ただし Mirowski らの成功は LSTM 単体ではなく、**深度予測・ループクロージャという補助タスク（教師なし・自己教師あり）を併用**している点に注意が必要——これは「教師信号の注入」ではなく環境から得られる情報（深度・軌跡の再訪）を使った自己教師あり学習であり、外部のお手本を与えるものではない。

**本研究への含意**: マイクロマウスのセンサ（距離センサ・ジャイロ・加速度計・車輪角速度）は局所観測であり、迷路全体のトポロジーは観測に含まれない——本質的に POMDP である。記憶（LSTM 等）の導入は理論的に正当化される方策構造側のレバーである。Mirowski らの補助タスク（深度予測等）は報酬を変えずに表現学習を助ける手段として参考になりうるが、これも報酬契約の外側にある設計要素として教授セッションでの整理が必要。

---

## 6. 計算量・seed 数の増強

| 文献 | 要点 |
|---|---|
| Henderson, P. et al. (2018, AAAI) "Deep Reinforcement Learning that Matters" | 同一のアルゴリズム・同一のコードベースでも、乱数シード（seed。学習の初期値や探索の乱数系列を決める種）だけの違いで統計的に有意な性能差が生じることを実証。ハイパーパラメータ・報酬のスケール・ネットワーク構造の影響も含め、RL 研究の再現性の危うさを指摘した影響力の大きい論文 |
| Colas, C. et al. (2018) "How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments" — [arXiv:1806.08295](https://arxiv.org/pdf/1806.08295) | 「seed をいくつ使えば、観測された性能差が本物の改善だと統計的に言えるか」を検定力分析の枠組みで定量化。RL 研究で使われる seed 数が往々にして不十分であることを示す |
| Baker, B. et al. (2019, ICLR 2020) "Emergent Tool Use From Multi-Agent Autocurricula" — [arXiv:1909.07528](https://arxiv.org/abs/1909.07528) | 報酬・環境ルールを一切変えず、大規模な計算資源だけで「隠れる・見つける」というマルチエージェント競争から道具使用（箱を運んで砦を作る、傾斜台を使って砦を乗り越える等）が段階的に創発することを実証。検索エンジン要約によれば、標準構成でこの創発の後半段階に至るには約 1.6M パラメータ・バッチサイズ 64,000・約 1.32 億エピソード（317 億フレーム）・34 時間の学習を要したとされる（この数値は WebFetch で原文アブストラクトまでは確認できておらず、検索エンジンの要約に基づくため「未確認」として扱う） |
| Hilton, J., Tang, J., Schulman, J. (2023) "Scaling laws for single-agent reinforcement learning" — [arXiv:2301.13442](https://arxiv.org/abs/2301.13442) | 「内在性能」（intrinsic performance：達成報酬に到達するのに必要な最小計算量として定義される、なめらかな指標）が、モデルサイズと環境とのやり取り回数（環境ステップ数）のべき乗則に従うことを複数タスクで示した。言語モデルのスケーリング則と類似の構造が単一エージェント RL にも成り立つとする |

**要点**: 計算量の増強のみで新しい行動が創発した実証例（hide-and-seek）は存在するが、これは**報酬・環境ルールを変えていない**という点で本研究の裁定の精神に最も近いレバーである一方、要した計算資源は極めて大規模（数十時間・数十億フレーム規模）である。他方で seed 依存性の研究（Henderson 2018, Colas 2018）は、「seed を増やす」こと自体は創発を引き起こす操作ではなく、**観測された性能のばらつきを正しく測定するための統計的な下準備**であることを示している——両者は別の効果である点に注意が必要。

**本研究への含意**: 「計算量・seed 数の増強」は報酬・環境仕様に一切手を触れない最も保守的なレバーであり、まず試す価値が高い。ただし (a) 実際に創発が起きるかは計算資源の規模に強く依存し、限られた計算資源では上限がある可能性がある、(b) seed を増やす行為自体は創発を起こす手段ではなく「今の設定で本当に停滞しているか」を統計的に確認する手段である、という 2 点を区別して扱うべきである。

---

## 7. 創発縛り（教師なし）の科学としての価値

| 文献 | 要点 |
|---|---|
| Nosek, B. A. et al. (2018, PNAS) "The preregistration revolution" | 仮説・手法・分析計画を事前に公開登録することで、結果が正でも負でも報告されるようにし、事後的な仮説の後付け（HARKing）や p-hacking を防ぐという心理学発の方法論運動。機械学習分野への応用は限定的だが、近年 NLP・ML でも議論が始まっている（例: 2023 年の "A Two-Sided Discussion of Preregistration of NLP Research"） |
| Pineau, J. et al. (2020, JMLR) "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)" | NeurIPS 2019 で実施された大規模な再現性検証プログラムの報告。173 本の論文が再現性検証の対象となり（ICLR 2019 時点から 92% 増）、結果が再現できた／できなかったの両方を公開する枠組みを整備 |
| Karl, F., Kemeter, L. M., Dax, G., Sierak, P. (2024, ICML) "Position: Embracing Negative Results in Machine Learning" — [arXiv:2406.03980](https://arxiv.org/abs/2406.03980) | 「予測性能の高さだけを論文の価値とする」文化が研究コミュニティ全体の非効率と誤ったインセンティブを生んでいると論じ、否定的結果（うまくいかなかった実験）の公表を積極的に評価すべきだと主張する立場表明論文 |

**要点**: 「報酬を最小形に凍結し、それ以上は増やさず、どこまで到達しどこで止まるかを立証すること自体を成果とする」という枠組みに**RL に特化した専用の先行研究・方法論は確認できなかった**（未確認）。ただし、より一般的な機械学習・心理学の研究方法論として、(a) 事前に仮説・手続きを固定して結果の解釈的な後付けを防ぐ「事前登録」運動、(b) 否定的結果・再現性検証そのものを成果として公表する文化的な潮流、の 2 つは実在し、近年勢いを増している。

**本研究への含意**: 本研究の裁定（報酬を凍結し以後の介入は環境・構造・計算量に限定）は、これら一般的な研究倫理の潮流——「介入の自由度を事前に制限し、その制約下での結果を（成功・失敗を問わず）記録する」——と方向性が一致する。ただし RL 分野に特化した先行事例が見当たらない以上、「本研究プログラム自体がこの枠組みの適用例として新規性を持つ」という位置づけ（先行研究の後追いではなく方法論の提案側）で報告するのが誠実である。

---

## 8. 行動創発の先行例（プラットフォーム・規格を問わない）

**改訂の経緯**: 当初は「マイクロマウス規格に限定した先行例」として調査したが、その限定では該当研究がほぼ存在しなかった。ユーザ指摘を受け、**「最小限の報酬（ゴール・衝突・時間程度）から、ナビゲーション・障害物回避・機敏な移動といった行動が創発した」先行研究**を、プラットフォーム・競技規格を問わず広く再調査した。各文献について、(a) 学習信号の純度（模倣・デモンストレーション・特権情報＝シミュレータの真値など、報酬と環境以外の情報源が混入していないか）、(b) 報酬の構成（最小形か、整形・補助報酬を含むか）、(c) 創発に効いたと論文が主張する主因、の 3 点を分類した。**(a) が「純粋」である文献のみが、本研究の「創発縛り」と同条件の証拠として扱える**。

### 8-A. 分類サマリー

| 文献 | (a) 学習信号の純度 | (b) 報酬構成 | (c) 創発の主因 |
|---|---|---|---|
| Mirowski et al. (2017, ICLR)「迷路ナビゲーション」 | 方策自体は純粋（模倣・デモなし）。ただし補助タスク（深度予測・ループクロージャ）の教師信号に**シミュレータの特権情報**（Z-buffer 真値深度・真値速度の積分）を使用 | 疎（りんご1点・いちご2点・ゴール10点のみ。整形なし） | 補助タスク＋ LSTM 記憶の併用（論文は補助タスクの寄与を「劇的」と表現） |
| Heess et al. (2017)「移動行動創発」 | 純粋と見られる（模倣・デモ・特権情報への言及はアブストラクト・概要の範囲では確認できず。**本文未確認のため断定は避ける**） | 最小（前進速度に基づく単一項。整形の記述なし） | **環境の多様性**（多様な地形・障害物への適応。論文は「明示的な報酬による誘導なしに」("without explicit reward-based guidance") 走る・跳ぶ・しゃがむ・曲がるが創発したと明記） |
| Kaufmann et al. (2023, *Nature*)「Swift・ドローンレース」／その前身 Song et al. (2021, IROS) | **不純**：方策（実機搭載）自体は模倣・デモなしだが、**価値関数（critic）の学習にのみ特権情報**（真値位置・姿勢・速度）を使用する非対称 actor-critic。知覚（ゲート検出）は別途教師あり学習で訓練 | **不純（整形あり）**：進捗＋知覚（カメラをゲートへ向ける項）＋行動の滑らかさ＋衝突罰の4項からなる密な整形報酬 | 知覚考慮の報酬整形＋実機データに基づく残差モデル（sim-to-real 補正）＋価値関数による人間より長い時間軸での最適化 |
| Tai, Paolo & Liu (2017, IROS)「車輪ロボット地図なしナビゲーション」 | 純粋（"trained end-to-end from scratch...without any manually designed features and prior demonstrations" と明記） | **最小に近い**：到達報酬＋衝突罰＋距離差分のポテンシャル整形の3項（本研究の凍結報酬とほぼ同型） | 疎な距離センサへの入力抽象化（sim-real ギャップの縮小）＋非同期 DDPG による並列学習 |
| Chen, Liu, Everett, How (2017, ICRA)「CADRL・多エージェント衝突回避」 | **不純**：価値ネットワークの初期重みを、**古典アルゴリズム ORCA が生成した軌道への教師あり学習で初期化**してから RL で微調整（教師あり事前学習＋RL微調整のハイブリッド） | 最小（衝突回避・到達） | 教師あり初期化が不可欠と報告（論文は「初期化なしでは有用な RL 経験の生成自体が困難」との趣旨を述べる） |
| Baker et al. (2019, OpenAI)「hide-and-seek・マルチエージェント」 | 純粋（模倣・デモなし。自己対戦のみ） | 最小（隠れる/見つけるの単純なチーム報酬） | **計算量**（大規模な自己対戦・数億エピソード規模の学習） |
| Hafner et al. (2023) "Mastering Diverse Domains through World Models"（DreamerV3・Minecraft ダイヤモンド） — [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) | 純粋（"the first algorithm to collect diamonds in Minecraft from scratch **without human data or curricula**" と明記） | 疎（12のマイルストーン到達ごとに初回のみ+1。整形なし） | **世界モデル（world model）**：学習した潜在空間ダイナミクスモデル上での大量の「空想上のロールアウト」により、少ない実環境ステップから方策・価値関数を学習 |

### 8-B. 各文献の詳細

**Mirowski, P. et al. (2017, ICLR) "Learning to Navigate in Complex Environments"** — [arXiv:1611.03673](https://arxiv.org/abs/1611.03673)（項目 5 とも重複）。WebFetch で原文を確認。DeepMind Lab の一人称視点 3D 迷路で、報酬はりんご・いちご・ゴールの疎な得点のみ（整形なし）。「明示的な報酬整形ではなく、深度予測・ループクロージャという2つの補助タスクを損失に加えることで学習効率が劇的に改善する」ことを示した。ただし両補助タスクの教師信号（真値深度・真値速度）はシミュレータの特権情報であり、実機のカメラ画像だけからは得られない量である点に注意。**報酬自体は最小形のまま、方策構造・補助損失側の工夫で疎報酬性を克服した例**として扱える。

**Heess, N. et al. (2017) "Emergence of Locomotion Behaviours in Rich Environments"** — [arXiv:1707.02286](https://arxiv.org/abs/1707.02286)。DeepMind による、多様な地形（起伏・障害物・隙間）で単純な「前進速度」報酬のみを与えたところ、走る・跳ぶ・しゃがむ・向きを変えるといった行動が方策構造や報酬の追加設計なしに現れたと報告する研究。「without explicit reward-based guidance」という表現が示す通り、**行動の多様化を駆動したのは報酬ではなく環境（地形）の多様性**であるとされる。WebFetch では概要レベルの確認にとどまり、本文中の厳密な報酬式・特権情報の有無までは確認できていない（**未確認**）。

**Kaufmann, E. et al. (2023, *Nature* 620) "Champion-level drone racing using deep reinforcement learning"（Swift）** — [PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/)、前身研究: Song, Y. et al. (2021, IROS) "Autonomous Drone Racing with Deep Reinforcement Learning" — [arXiv:2103.08624](https://arxiv.org/abs/2103.08624)。WebFetch で PMC 全文を確認。人間の世界チャンピオンに勝利した高性能ドローンレース方策だが、**本研究の「創発縛り」の条件とは大きく異なる**：(1) 報酬は進捗・知覚（ゲートを画角内に収める）・滑らかさ・衝突罰の4項からなる密な整形報酬であり最小形ではない、(2) 学習中の価値関数（実機には搭載されない、学習時のみ使う内部関数）はシミュレータの真値位置・姿勢・速度という特権情報を使う非対称 actor-critic 構成、(3) 実機投入前にゲート検出器を教師あり学習で個別に訓練、(4) 実機とシミュレータの差を埋める残差モデルを実データから同定。**「センサから直接行動を出力する end-to-end」に見える成果の内実は、報酬整形・特権情報付き学習・モジュール化されたパイプラインの組み合わせであり、本研究が禁じている類の介入を複数含む**。到達性能の高さと引き換えに、学習信号の純度は本研究の基準を満たさない好例。

**Tai, L., Paolo, G., Liu, M. (2017, IROS) "Virtual-to-real Deep Reinforcement Learning..."** — [arXiv:1703.00420](https://arxiv.org/abs/1703.00420)。WebFetch で原文（ar5iv版）の報酬式を確認。報酬は r(s,a) = r_arrive（到達）／r_collision（衝突）／c_r・(d_{t-1}−d_t)（距離差分のポテンシャル整形）の3項のみで、模倣・デモは一切用いず「trained end-to-end from scratch」と明記。**本研究の凍結報酬（ゴール＋衝突罰＋ポテンシャル整形）とほぼ同型の報酬設計で、疎な距離センサ入力から連続値の操舵指令を出力する障害物回避ナビゲーションを学習し、実機転移にも成功した例**。8項目中もっとも本研究の設定に近い直接の先行例と言える。

**Chen, Y. F., Liu, M., Everett, M., How, J. P. (2017, ICRA) "Decentralized Non-Communicating Multiagent Collision Avoidance with Deep Reinforcement Learning"（CADRL）**。原文は arXiv 未掲載と見られ（検索で arXiv版を発見できず）、ACM Digital Library・著者らの GitHub リポジトリ記述から確認（**原文 PDF は未直接照合、二次資料に基づく**）。報酬自体は最小（衝突回避・到達）だが、**価値ネットワークの重みを、古典的な衝突回避アルゴリズム ORCA が生成した軌道への教師あり学習で初期化してからでないと、RL による学習が実用的な回避行動に至らなかった**と報告されている。これは「報酬は最小」でも「学習信号は報酬と環境のみ」ではない（ORCA という外部エキスパートのデモが混入している）好例であり、**報酬の最小性と学習信号の純粋性は別軸で判定する必要がある**ことを示す。

**Baker, B. et al. (2019) "Emergent Tool Use From Multi-Agent Autocurricula"（項目 6 と同一文献）** — [arXiv:1909.07528](https://arxiv.org/abs/1909.07528)。報酬・環境ルールを変えず、大規模な自己対戦のみから道具使用が段階的に創発。模倣・デモ・特権情報の使用は報告されておらず、**(a)(b) ともに本研究の条件に近い数少ない事例**。ただし創発の主因は計算量であり、必要な学習規模は非常に大きい。

**Hafner, D. et al. (2023) "Mastering Diverse Domains through World Models"（DreamerV3）** — [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)。Minecraft のダイヤモンド収集という、人間データや専用カリキュラムなしには長年解けなかった疎報酬タスクを、学習した「世界モデル」（環境の遷移を近似する潜在空間モデル）上での大量の空想ロールアウトによって解いた。報酬は 12 個のマイルストーン到達ごとに初回のみ +1 という疎な構成で、整形は含まれない。**(a)(b) ともに純粋**であり、創発の主因は「記憶」とも「計算量」とも異なる第三の軸——**方策構造の一種としての世界モデル（学習された環境モデル上でのプランニング/ロールアウト）**——にある点が特徴的。M2 の4レバー（カリキュラム／内発的報酬／記憶／計算量）のいずれにも直接は分類されず、方策構造レバーの拡張候補として注記に値する。

**マイクロマウス競技専用の end-to-end RL 研究**: 引き続き検索範囲内では発見できなかった（**未確認＝存在しないことの証明ではない**）。マイクロマウス分野の主流はフラッドフィル法など古典的探索アルゴリズムであり（[UB IEEE Club Wiki](https://ubieee.github.io/wiki/micromouse/software/maze-solving-algorithms/) 等）、RL の適用事例があっても離散化されたセル単位の意思決定に留まり、センサ→モータ電圧の直接出力という契約とは異なると見られる。

**要点**: 「最小限の報酬から行動が創発した」純粋な事例（(a)(b) がともに満たされる）は、**Heess et al.（環境の多様性）・Tai et al.（疎センサ＋最小整形での障害物回避）・Baker et al.（計算量）・Hafner et al.（世界モデル）**の4件が該当し、いずれも報酬設計そのものではなく**環境・計算量・方策/モデル構造**側の工夫を主因としている。一方、性能面で最も華々しい2事例（Kaufmann/Swift のドローンレース、Chen/CADRL の衝突回避）は、**報酬整形・特権情報つき学習・教師あり事前学習のいずれか（または複数）を用いており、本研究の「創発縛り」と同条件ではない**。これは、実世界で高性能な end-to-end 制御を達成した研究の多くが、実際には本研究が禁じている種類の介入（整形・特権情報・デモ）に頼っていることを示しており、**「創発縛り」の下での成功は文献的にも希少で挑戦的な設定である**ことの傍証になる。

**本研究への含意**: (1) Tai et al. は報酬設計・出力契約（センサ→連続値の直接出力）ともに本研究に最も近い直接証拠であり、「最小報酬＋疎センサでの障害物回避ナビゲーションは原理的に学習可能」という主張を支持する。(2) Heess et al. は、環境側の介入が「距離を徐々に伸ばすカリキュラム」という狭い形だけでなく、「地形・障害物配置の多様性を増やす」というもう一つの環境レバーでも創発を駆動しうることを示しており、カリキュラムのレパートリーを広げる示唆になる。(3) Baker et al. と Hafner et al. は、計算量の増強と並んで「世界モデル」という第三の方策構造的選択肢が存在することを示す——M2 の4レバーの外側にある追加候補として教授セッションへ報告する価値がある。(4) Swift・CADRL の不純な事例は、**性能を優先すると「創発縛り」から逸脱する誘惑が強いこと**を実証しており、今後の実験で無自覚に特権情報や事前学習が混入しないよう、実装レベルでの監査（価値関数の入力・初期化手順の確認）を予防的に行う価値がある。

---

## M2 の 4 レバー（カリキュラム／内発的報酬／記憶／計算量）への優先順位の示唆

文献の証拠に基づくと、優先順位は **カリキュラム ＞ 記憶 ＞ 計算量／seed ＞ 内発的報酬** の順が妥当と考えられる。根拠は次の通り。

1. **カリキュラム（最優先）**: Reverse Curriculum Generation（Florensa et al. 2017）は「スタートからゴールまでの距離を徐々に伸ばす」という、まさに本研究が直面している症状（数区画で衝突・ゴール到達が定着しない）に対応する直接の先行研究であり、かつ報酬関数を一切変更せずに環境側だけで実装できる。裁定の文言に最も抵触しにくいレバーでもある。
2. **記憶（次点）**: マイクロマウスのセンサは本質的に部分観測（POMDP）であり、LSTM 等の導入は Mirowski (2017)・DRQN (2015) が示す通り理論的にも実証的にも妥当性が高い。方策構造の変更であり報酬とも環境とも独立に検証できる。
3. **計算量／seed 増強**: 報酬・環境に一切手を触れない最も保守的なレバーだが、hide-and-seek の実例（Baker et al. 2019）が示すように創発には非常に大きな計算資源を要した例があり、費用対効果は不確実。ただし「今の停滞が真の停滞か、単なる分散か」を切り分けるための seed 増強は、他のどのレバーを試す場合にも前提として必要（Henderson 2018, Colas 2018）。
4. **内発的報酬（最後・要相談）**: 技術的には評価時無効化・学習後半での減衰という設計で「報酬契約の凍結」と両立させる先行例が複数あるが、報酬関数そのものへの追加である点で裁定の文言（報酬は最小形で凍結）に最も抵触しやすい。導入するなら教授セッションへの個別確認が必須。

### 追記（項目8の追加調査を受けて）

規格非限定で「行動創発」の先行例を洗い直した結果、上記の優先順位を補強・修正する材料が得られた。

- **カリキュラムの根拠強化＋レパートリー拡大**: Tai et al. (2017) は本研究とほぼ同型の最小報酬（到達＋衝突＋ポテンシャル整形）で疎センサからの障害物回避ナビゲーションを学習した直接証拠であり、報酬凍結のまま前進できるという方針の妥当性を補強する。さらに Heess et al. (2017) は「距離を伸ばす」型のカリキュラムだけでなく、「地形・障害物配置の多様性を増やす」という環境側の別の軸でも行動創発が起きることを示した。6×6 迷路での停滞に対しては、区画数の段階化（Reverse Curriculum 型）に加えて、**同じ区画数でも迷路パターンの多様性を増やす**という手も環境レバーの選択肢に加えられる。
- **計算量の位置づけを再確認**: Baker et al. (2019) は報酬・環境ルールを変えず計算量のみで創発した数少ない「純粋」事例として、計算量レバーの理論的な正当性を裏付ける一方、要した規模の大きさは費用対効果への懸念を再確認させる。
- **世界モデルという第三の構造的選択肢（4レバー外・要検討）**: Hafner et al. (2023) DreamerV3 は、記憶（LSTM等）とも計算量とも異なる「学習された環境モデル上での大量の空想ロールアウト」によって疎報酬タスクを解いた純粋な事例である。M2 の4レバーには含まれていないが、方策構造側の介入として記憶の隣接候補になりうるため、教授セッションへ検討材料として提示する価値がある。
- **「性能優先」の誘惑への警戒**: 対照的に、性能面で最も華々しい事例（Kaufmann et al. 2023 Swift のドローンレース、Chen et al. 2017 CADRL の衝突回避）は、いずれも報酬整形・特権情報つき学習・教師あり事前学習のいずれかに依存しており、本研究の「創発縛り」とは同条件でない。これは、記憶や計算量のレバーを実装する際にも、価値関数の入力や初期化手順に無自覚に特権情報・デモが混入しないよう、実装レベルでの監査を予防的に行うべきという教訓を与える。

---

## 出典一覧（本文中で参照した主なリンク）

- [Go-Explore: a New Approach for Hard-Exploration Problems (arXiv:1901.10995)](https://arxiv.org/abs/1901.10995)
- [First return, then explore (arXiv:2004.12919)](https://arxiv.org/pdf/2004.12919)
- [Hindsight Experience Replay (arXiv:1707.01495)](https://arxiv.org/abs/1707.01495)
- [Learning Montezuma's Revenge from a Single Demonstration (arXiv:1812.03381)](https://arxiv.org/pdf/1812.03381)
- [Concrete Problems in AI Safety (arXiv:1606.06565)](https://arxiv.org/abs/1606.06565)
- [Specification gaming: the flip side of AI ingenuity (DeepMind blog)](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
- [Specification gaming examples in AI (Krakovna blog)](https://vkrakovna.wordpress.com/2018/04/02/specification-gaming-examples-in-ai/)
- [Exploration by Random Network Distillation (ResearchGate)](https://www.researchgate.net/publication/328627326_Exploration_by_Random_Network_Distillation)
- [Automatic Curriculum Learning For Deep RL: A Short Survey (arXiv:2003.04664)](https://arxiv.org/abs/2003.04664)
- [Reverse Curriculum Generation for Reinforcement Learning (arXiv:1707.05300)](https://arxiv.org/abs/1707.05300)
- [Automatic Goal Generation for Reinforcement Learning Agents (arXiv:1705.06366)](https://arxiv.org/abs/1705.06366)
- [Deep Recurrent Q-Learning for Partially Observable MDPs (arXiv:1507.06527)](https://arxiv.org/abs/1507.06527)
- [Learning to Navigate in Complex Environments (arXiv:1611.03673)](https://arxiv.org/abs/1611.03673)
- [Stabilizing Transformers for Reinforcement Learning (arXiv:1910.06764)](https://arxiv.org/abs/1910.06764)
- [How Many Random Seeds? (arXiv:1806.08295)](https://arxiv.org/pdf/1806.08295)
- [Emergent Tool Use From Multi-Agent Autocurricula (arXiv:1909.07528)](https://arxiv.org/abs/1909.07528)
- [Scaling laws for single-agent reinforcement learning (arXiv:2301.13442)](https://arxiv.org/abs/2301.13442)
- [Position: Embracing Negative Results in Machine Learning (arXiv:2406.03980)](https://arxiv.org/abs/2406.03980)
- [Improving Reproducibility in Machine Learning Research (JMLR / NeurIPS 2019 Reproducibility Program)](https://arxiv.org/pdf/2003.12206)
- [Virtual-to-real Deep Reinforcement Learning (arXiv:1703.00420)](https://arxiv.org/abs/1703.00420)
- [MicroMouse Maze Solving Algorithms (UB IEEE Club Wiki)](https://ubieee.github.io/wiki/micromouse/software/maze-solving-algorithms/)
- [Emergence of Locomotion Behaviours in Rich Environments (arXiv:1707.02286)](https://arxiv.org/abs/1707.02286)
- [Champion-level drone racing using deep reinforcement learning (PMC 全文)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/)
- [Autonomous Drone Racing with Deep Reinforcement Learning (arXiv:2103.08624)](https://arxiv.org/abs/2103.08624)
- [Mastering Diverse Domains through World Models / DreamerV3 (arXiv:2301.04104)](https://arxiv.org/abs/2301.04104)
