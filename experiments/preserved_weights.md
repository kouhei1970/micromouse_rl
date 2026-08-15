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


### exp_022（にせ履歴・帰無の原因の分離）の重み — 2026-08-15 追加

**判定での用途**: exp_022 の P1〜P4 の判定（最終方策）／報告トリガーの実行記録（80 万歩）

| パス | 実歩数 | 大きさ | SHA-256 |
|---|---|---|---|
| `models/exp_022_seed1.zip` | 2,000,896 | 892 KB | `6862fada8d3af7ed95bbcb8bb0856247c557f545f646745e1c5ab3aa9091ac68` |
| `models/exp_022_seed2.zip` | 2,000,896 | 892 KB | `b50bf7eeef419e2c36790e2f7921e3d7a84c110d6e31e0853212835f44dc8ef8` |
| `models/exp_022_seed3.zip` | 2,000,896 | 892 KB | `2d04683b2e3e5c16d2ac8d9497d4c34326d1c0885772d6eb77207b6d63930113` |
| `models/exp_022_seed4.zip` | 2,000,896 | 892 KB | `80eda1822a622df46f11eceaa5451ed637af5d0176467f4149f6bc65a2005edb` |
| `models/exp_022_seed5.zip` | 2,000,896 | 892 KB | `ce75b2f28a9f0c0c77cdaf80b11b29667316e69a0ac807872a9cfe1db6daa7e1` |
| `models/exp_022_seed6.zip` | 2,000,896 | 892 KB | `4d2a5c05d8bc803c069bfb9c43dd15010aeed5dca6a0cee2023a617b0f521ff9` |
| `logs/exp_022_seed1/rl_model_800000_steps.zip` | 800,000 | 892 KB | `7e91ea33cb1fa304ab0034250bb48ca2e5904656b2b0a7e0036c7f52e025d370` |
| `logs/exp_022_seed2/rl_model_800000_steps.zip` | 800,000 | 892 KB | `da53ac03f677c08bf2aec776ee24f2d1f665aa8dc986b7a6158f4b6c658e41c3` |
| `logs/exp_022_seed3/rl_model_800000_steps.zip` | 800,000 | 892 KB | `66a40f4e1791c657e28b5d656a55a79e24d4b7c201d75c06d53bc3fe9f208fe0` |
| `logs/exp_022_seed4/rl_model_800000_steps.zip` | 800,000 | 892 KB | `abcb4d484f9d73d64ba3759463515a0b2f7550f8a088c2375321727abc2cb704` |
| `logs/exp_022_seed5/rl_model_800000_steps.zip` | 800,000 | 892 KB | `33345c660c1c46aaa5fbe4d47714ee8055496bd07074ed9d820803c8920e8a5c` |
| `logs/exp_022_seed6/rl_model_800000_steps.zip` | 800,000 | 892 KB | `5745fb1fb1c03dbee0da15bdad03c1b2c9c6c58087d448b28bcb192768d50620` |

**合計 12 本・10.45 MB。****最終方策 6 本はすべて 2,000,896 歩・退避重み 6 本はすべて 800,000 歩で完全に揃っている**（**exp_019・exp_021 とも厳密に一致 ＝ 3 群の学習量が揃っている**）。

## 2. 未実施（R11 へ登録）

- **exp_012 以前の実験の重みの棚卸し**。**判定が現役でないもの**なので今回は入れていない。**過去の判定文書が値を引いている重みが失われていないか**を、R11 バッチで確認する
- ~~exp_021 の重み 12 本は完走後に保全する~~ → **2026-08-15 実施済み**（上記 §1 の末尾）


## exp_023（再帰型方策実験・第 2 弾 B）2026-08-15

| 群 | ファイル | SHA-256 | 大きさ |
|---|---|---|---|
| 群1 最終 | `models/exp_023a_seed1.zip` | `c5ae9d5293ca8c79b01be56a775d1d64ae7aeb2dd256e395cf594ec0f70a3f89` | 1,122,498 |
| 群1 最終 | `models/exp_023a_seed2.zip` | `1583fa055f567fcda140439be9836bc8d29938815d8460974d202ea2d6b88afb` | 1,122,498 |
| 群1 最終 | `models/exp_023a_seed3.zip` | `ebefc44bffe8d99219a4efc2190f93ce0cebbcf1847d9102f1132a5365a78b93` | 1,122,498 |
| 群1 最終 | `models/exp_023a_seed4.zip` | `0e43b1fce0df270fb874d041456d2e1d1366c58f1e3c7d82f7cec73ef96dea36` | 1,122,494 |
| 群1 最終 | `models/exp_023a_seed5.zip` | `fcdd280a22c549ab28d804a3a784f28e09e394dbc5fe1fd728351030a27bc43b` | 1,122,498 |
| 群1 最終 | `models/exp_023a_seed6.zip` | `1c837db9e608c34da0e8fb2b66462f174885da62832bbb99c51961e3f66c1334` | 1,122,498 |
| 群2 最終 | `models/exp_023b_seed1.zip` | `f38c40f7c2bff6acf0efad231886cc09b2c2cb1b84af73de9781aac2df7c7077` | 1,122,521 |
| 群2 最終 | `models/exp_023b_seed2.zip` | `0621143a4b2827546bee726bb020398a63caccc6de4cf0a677c98a7efde4d623` | 1,122,521 |
| 群2 最終 | `models/exp_023b_seed3.zip` | `ac566ffdfadb54499a185582f1240c78cf7f7301662bf2eb188bbb3a7bd0dc73` | 1,122,521 |
| 群2 最終 | `models/exp_023b_seed4.zip` | `46e77373ee26c7dd88029fe48d12c37ad4ed95596a591c4a3d0bbced376c8a4b` | 1,122,521 |
| 群2 最終 | `models/exp_023b_seed5.zip` | `fb6f309ef8a2e535f3f0dc185008afbb87a2763d388d774991c0bdb8b5e82318` | 1,122,521 |
| 群2 最終 | `models/exp_023b_seed6.zip` | `2f2d98d1fd9980d566d9904387c99fba3afc060a5f3bf7cac1f0aba856058283` | 1,122,521 |
| 群1 80万歩 | `logs/exp_023a_seed1/rl_model_800000_steps.zip` | `37f5ea5c64b6f428ac3c5c534a321f0a3ad0634a36bb36fde867c985fdb64333` | 1,122,492 |
| 群1 80万歩 | `logs/exp_023a_seed2/rl_model_800000_steps.zip` | `42d9851ac377d64e0c96f345e5ce5533f30e6210cd3b1478fb9a03f9e23efb77` | 1,122,492 |
| 群1 80万歩 | `logs/exp_023a_seed3/rl_model_800000_steps.zip` | `c7d0d3a489ec2fa3e7257aefb5863a1c7edc3bb8dc2933947e14c789aad15b50` | 1,122,492 |
| 群1 80万歩 | `logs/exp_023a_seed4/rl_model_800000_steps.zip` | `cc69a271160e10c74b6609d1a97269bd6ff0624cb4ddb203f315405521e05585` | 1,122,492 |
| 群1 80万歩 | `logs/exp_023a_seed5/rl_model_800000_steps.zip` | `5fbecdf63f0adf64915d7c9b7beddaf95b85a45435c264f1c36bf8f0f6886edd` | 1,122,492 |
| 群1 80万歩 | `logs/exp_023a_seed6/rl_model_800000_steps.zip` | `50f0b179156c5095ff26b44eec5b2e05e4c7c63220b1a98f78d2c0677374c44e` | 1,122,492 |
