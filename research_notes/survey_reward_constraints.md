<!-- 起草: 調査エージェント（教授セッション委任・2026-08-14）／検収: 教授セッション／出所: ユーザ裁定 2026-08-14 の項目 5（報酬の縛りの先行研究調査）。文献の確認状態は本文の注記どおり -->

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

## 8. 迷路・ナビゲーション RL の先行例

| 文献 | 要点 |
|---|---|
| Mirowski, P. et al. (2017, ICLR) "Learning to Navigate in Complex Environments"（項目 5 と同一文献） | DeepMind Lab の一人称視点 3D 迷路で、生の視覚入力から LSTM ベースの方策が経路探索能力を獲得することを示した代表例。ただし迷路は連続空間中の視覚ナビゲーションであり、車輪付きロボットの二輪独立駆動・電圧出力という本研究の出力契約とは異なる |
| Tai, L., Paolo, G., Liu, M. (2017, IROS) "Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation" — [arXiv:1703.00420](https://arxiv.org/abs/1703.00420) | 10 次元の疎な距離センサ値とゴールの相対位置を入力とし、連続的な操舵指令を出力する「地図なしナビゲーション」を学習。実機（差動駆動ロボット）へのシミュレーションからの転移も検証。**WebFetch で原文アブストラクトを確認したところ、当初の検索エンジン要約にあった「生の感覚入力による複雑な3D迷路でのナビゲーション」という記述は誤りで、実際は疎な距離センサ値を入力とする単純化されたマップレスナビゲーション課題であった（Mirowski論文の記述との混同と判断し、本報告では訂正済み）** |
| "Solving Maze Problem with Reinforcement Learning by a Mobile Robot"（著者・年ともに未確認） | 移動ロボットによる迷路問題への RL 適用に関する研究として検索結果に表れたが、原文未確認のため詳細不明 |
| マイクロマウス競技専用の end-to-end RL 研究 | 検索範囲内では**発見できなかった**。マイクロマウス分野の主流はフラッドフィル法など古典的探索アルゴリズムであり（[UB IEEE Club Wiki](https://ubieee.github.io/wiki/micromouse/software/maze-solving-algorithms/) 等）、RL の適用事例は限定的で、多くは「離散化されたセル単位の意思決定」に RL を使うもの（センサ→モータ電圧の直接出力ではない）に留まると見られる |

**要点**: 迷路・ナビゲーション課題への RL 適用自体は複数の先行研究があるが、(a) 視覚ベースの 3D 迷路（Mirowski）、(b) 疎な距離センサ＋連続操舵指令の地図なしナビゲーション（Tai et al.）のいずれも、**センサから直接モータ電圧を出力する end-to-end 契約・かつマイクロマウス競技規格（16×18cm 格子・公式ルール）**という組み合わせの先行研究は見当たらなかった。

**本研究への含意**: 本研究プログラムは、少なくとも公開文献で確認できる範囲では、マイクロマウス競技規格下でのセンサ→モータ電圧直接出力 end-to-end RL という点で先行研究の空白を埋める位置づけにある。関連研究としては Tai et al.（疎センサ＋連続制御の地図なしナビゲーション）が最も近い出力契約（連続値の操舵/速度指令）を持つが、迷路の難度・センサ構成は異なる。この位置づけ自体は、成果を発表する際に「新規性の主張」として使える一方、「比較対象となる直接の先行ベースラインが存在しない」という研究上の制約でもある。

---

## M2 の 4 レバー（カリキュラム／内発的報酬／記憶／計算量）への優先順位の示唆

文献の証拠に基づくと、優先順位は **カリキュラム ＞ 記憶 ＞ 計算量／seed ＞ 内発的報酬** の順が妥当と考えられる。根拠は次の通り。

1. **カリキュラム（最優先）**: Reverse Curriculum Generation（Florensa et al. 2017）は「スタートからゴールまでの距離を徐々に伸ばす」という、まさに本研究が直面している症状（数区画で衝突・ゴール到達が定着しない）に対応する直接の先行研究であり、かつ報酬関数を一切変更せずに環境側だけで実装できる。裁定の文言に最も抵触しにくいレバーでもある。
2. **記憶（次点）**: マイクロマウスのセンサは本質的に部分観測（POMDP）であり、LSTM 等の導入は Mirowski (2017)・DRQN (2015) が示す通り理論的にも実証的にも妥当性が高い。方策構造の変更であり報酬とも環境とも独立に検証できる。
3. **計算量／seed 増強**: 報酬・環境に一切手を触れない最も保守的なレバーだが、hide-and-seek の実例（Baker et al. 2019）が示すように創発には非常に大きな計算資源を要した例があり、費用対効果は不確実。ただし「今の停滞が真の停滞か、単なる分散か」を切り分けるための seed 増強は、他のどのレバーを試す場合にも前提として必要（Henderson 2018, Colas 2018）。
4. **内発的報酬（最後・要相談）**: 技術的には評価時無効化・学習後半での減衰という設計で「報酬契約の凍結」と両立させる先行例が複数あるが、報酬関数そのものへの追加である点で裁定の文言（報酬は最小形で凍結）に最も抵触しやすい。導入するなら教授セッションへの個別確認が必須。

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
