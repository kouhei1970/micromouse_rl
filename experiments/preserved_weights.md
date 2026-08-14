# 判定が依存する学習済み重みの保全記録

- 制定: 2026-08-14（教授裁定。研究計画書 §9-21 (c)「**判定が依存する学習済み重みは判定と同じ寿命で保全する ＝ 版管理下へ置く**」の実施）
- 発端: 准教授 AUDIT_042 §3 — **判定が依存するモデルが版管理外**であり、**出力の SHA-256 は「変わっていないこと」しか保証せず、モデルが失われたら再生成できない**（AUDIT_039 §3-1 と同じ構造）
- 記録者: 学生B

## 0. 方針

**判定文書が値を引いている重みは、`.gitignore` の除外に対して `git add -f` でパスを明示して版管理下へ入れる。**

- **重みは測定出力より小さい**（本記録の 18 本で合計 **8.4 MB**）うえ、**真の単一障害点**である（**重みがあれば下流はすべて決定的に再生成できる**）
- **毎歩の生データ（数 MB〜）は版管理下に置かず、SHA-256 を実験カードへ記録する**
- **今回入れるのは判定が現役のものだけ**である。**それより古い実験（exp_012 等）の棚卸しは R11 へ登録**した（教授裁定）

## 1. 保全した重みの一覧（SHA-256 つき）

### exp_019（環境 v2 の基準実験）の最終方策（200 万歩）

**判定での用途**: exp_021 の対照群（Q1・Q2・Q4）／exp_019 自身の P5 判定

| パス | 実歩数 | 大きさ | SHA-256 |
|---|---|---|---|
| `models/exp_019_v2_seed1.zip` | 2,000,896 | 478 KB | `8d690f2e65ae13daabc9c7fe71d557ba5a8c01dd5991cfc854a97dd08246dbf7` |
| `models/exp_019_v2_seed2.zip` | 2,000,896 | 478 KB | `71f918bb728811b1405091254b30180d2dc621d1b4cc904bc1c5264271733197` |
| `models/exp_019_v2_seed3.zip` | 2,000,896 | 478 KB | `000f400629e4305b7fd7feffb748792ac391962b5f7ad73cf8a680fcb6ec8f88` |
| `models/exp_019_v2_seed4.zip` | 2,000,896 | 478 KB | `4f1115b29f1844aac369ba72e63eda9e57dcbd6d6595349cefa78f8d5aebfcfb` |
| `models/exp_019_v2_seed5.zip` | 2,000,896 | 478 KB | `85e9daa96a568b4b1221d2ee07317b326a855b861055cbff940899663f0e434d` |
| `models/exp_019_v2_seed6.zip` | 2,000,896 | 478 KB | `924fc3d22466bcc538200eb943043f7b47d6ac48e2ab2eda86a056d64d571ef0` |

### exp_019 の 80 万歩の退避重み（80 万歩）

**判定での用途**: exp_021 の報告トリガーの対照（§4-2-bis）

| パス | 実歩数 | 大きさ | SHA-256 |
|---|---|---|---|
| `logs/exp_019_v2_seed1/rl_model_800000_steps.zip` | 800,000 | 478 KB | `9c1e727175920c788b4869f20da9e26faab4d9f3a9ac10fd902e6127f27a225b` |
| `logs/exp_019_v2_seed2/rl_model_800000_steps.zip` | 800,000 | 478 KB | `3c470e0bd70b1a0bc388aaa2f99c0e6e88e8c4081b40938fb53d50b31b0b79e2` |
| `logs/exp_019_v2_seed3/rl_model_800000_steps.zip` | 800,000 | 478 KB | `0bedc21d412df62fd21d1a8e0cdac2ea5a7f234708b9bd83631fa17c0315fa26` |
| `logs/exp_019_v2_seed4/rl_model_800000_steps.zip` | 800,000 | 478 KB | `0dd2fe9b0eaf966ae7844f60d8f6f25be2e0d565d54bb60a3148ea1b5584c1c7` |
| `logs/exp_019_v2_seed5/rl_model_800000_steps.zip` | 800,000 | 478 KB | `04e514b16d19c2e349763f516475eebf4cef16a005f25410086e7dd1be8a5bfe` |
| `logs/exp_019_v2_seed6/rl_model_800000_steps.zip` | 800,000 | 478 KB | `948cf4d464d8baf34f9f2bba0832d830c76e8db6a3ca39051edcfc74cb4d9f19` |

### exp_020（距離カリキュラム実験）の最終方策（200 万歩）

**判定での用途**: exp_020 の Q1・Q4 の判定

| パス | 実歩数 | 大きさ | SHA-256 |
|---|---|---|---|
| `models/exp_020_seed1.zip` | 2,000,896 | 478 KB | `d27929d3c3ebf9bfcf88ca12eef609c971fe72f050da21c1f4390095a15d46dc` |
| `models/exp_020_seed2.zip` | 2,000,896 | 478 KB | `69d6c7afcf884598a29c64fe2380c1b5191b86e84a0f6112acbbae3db854c391` |
| `models/exp_020_seed3.zip` | 2,000,896 | 478 KB | `b7a750be0bf72eade3d47e5487091ff8738f83f965a24556fab14714edcffe6c` |
| `models/exp_020_seed4.zip` | 2,000,896 | 478 KB | `853762991b7c68a349d5b4c81f1ef78848d715d3da585589f71a28b9a4735830` |
| `models/exp_020_seed5.zip` | 2,000,896 | 478 KB | `9d8bd056e1795eaf56979648894edcd5ce7fe8fa1ae214176ce34b45645fdab1` |
| `models/exp_020_seed6.zip` | 2,000,896 | 478 KB | `ab4c614a6df11cfd09c6b3240196f5d6eb657ec7c751936250af1d732fcd6130` |

**合計 18 本・8.41 MB。****実歩数は 200 万歩の 12 本がすべて 2,000,896・80 万歩の 6 本がすべて 800,000 で、群の中で完全に揃っている**（`n_steps` = 2048 のロールアウト粒度による決定的な超過）。


### exp_021（観測履歴の連結）の重み — 2026-08-15 追加

#### exp_021（観測履歴の連結）の最終方策

**判定での用途**: exp_021 の Q1・Q2・Q4 の判定

| パス | 実歩数 | 大きさ | SHA-256 |
|---|---|---|---|
| `models/exp_021_seed1.zip` | 2,000,896 | 892 KB | `b6cef28d9117a7707d024ade985ac8763e81d54fff85e96ddc15f53732cc7b11` |
| `models/exp_021_seed2.zip` | 2,000,896 | 892 KB | `a41f68d62d9a291cb53e0325b71d6103798724a9b8b0fd6adde5cb80b2734527` |
| `models/exp_021_seed3.zip` | 2,000,896 | 892 KB | `d9c2416d38602c6b22d1006aa1e25247e5817a47f8e54031ca382dca75a46751` |
| `models/exp_021_seed4.zip` | 2,000,896 | 892 KB | `b5ca1be385df7292a3116904b7c621f424092fa79b2beae22915deef8dc3fcee` |
| `models/exp_021_seed5.zip` | 2,000,896 | 892 KB | `a8b251cb7206e90ff329925a8bf4fc5958e4e20482265788a4e2bbd241b988a4` |
| `models/exp_021_seed6.zip` | 2,000,896 | 892 KB | `484af33bf29dbb68d6fe4470a7654940326b9c3611a85f98e5c00c657a013802` |

#### exp_021 の 80 万歩の退避重み

**判定での用途**: exp_021 の報告トリガー（§4-2-bis）の実行記録

| パス | 実歩数 | 大きさ | SHA-256 |
|---|---|---|---|
| `logs/exp_021_seed1/rl_model_800000_steps.zip` | 800,000 | 892 KB | `cd18f34d2516595cb5f4a77ba9d1b3cdf83bd6a71cda30fb3208767d9d361390` |
| `logs/exp_021_seed2/rl_model_800000_steps.zip` | 800,000 | 892 KB | `1788eff90f713c5a1115409412df5e7ad208447c383a340a8ed6bffb6825f9b5` |
| `logs/exp_021_seed3/rl_model_800000_steps.zip` | 800,000 | 892 KB | `0d5c418dfa335fc8ea80d83a8844dea5f514f580a2884aa2952baf1c626386e7` |
| `logs/exp_021_seed4/rl_model_800000_steps.zip` | 800,000 | 892 KB | `19a6a2b9b7c4fb7ad59160f06ed5c669ecdb8bba0be491237137c3bf2d57d9be` |
| `logs/exp_021_seed5/rl_model_800000_steps.zip` | 800,000 | 892 KB | `626b205bb862404137f25ed8915a566fd8802fbd8a2f832151de3d37473013c8` |
| `logs/exp_021_seed6/rl_model_800000_steps.zip` | 800,000 | 892 KB | `7300db6a0acd1006c1c33a22193415840597cd8ae1e5a9549ff095149296166d` |

**合計 12 本・10.45 MB。****最終方策 6 本はすべて 2,000,896 歩・退避重み 6 本はすべて 800,000 歩で完全に揃っている。****対照群（exp_019）と学習量が厳密に一致している**（`n_steps` = 2048 の粒度による決定的な超過）。

## 2. 未実施（R11 へ登録）

- **exp_012 以前の実験の重みの棚卸し**。**判定が現役でないもの**なので今回は入れていない。**過去の判定文書が値を引いている重みが失われていないか**を、R11 バッチで確認する
- ~~exp_021 の重み 12 本は完走後に保全する~~ → **2026-08-15 実施済み**（上記 §1 の末尾）

