# 実装の詳細と学習戦略

## アーキテクチャ概要
CARIM は **Text-to-Text Matching** に基づく検索モデルです。
従来の Image-Text Retrieval (CLIPなど) と異なり、画像を直接エンコードするのではなく、画像からVLMによって生成された詳細なキャプションから「要素 (Elements)」を抽出し、その要素集合とクエリとの類似度を計算します。

### パイプライン
1.  **Image Captioning (VLM)**: Qwen-VL を使用して、画像から詳細な説明文（Dense Caption）を生成します。
2.  **Element Extraction (LLM)**: Qwen-2-Instruct を使用して、説明文から検索可能な「要素（名詞句、動作、状態）」をリスト形式で抽出します。
    *   例: `["red car", "parked", "near intersection", "sunny day"]`
3.  **Encoding**: クエリと各要素を、同じ Text Encoder (Qwen-2-1.5B) でベクトル化します。
4.  **Scoring**: Inclusive Text Matching (下記参照) によりスコアを算出します。

## モデル構造 (`models/carim_scorer.py`)
- **Backbone**: `Qwen/Qwen2-1.5B-Instruct` (Feature Extractionのみ, 重みは凍結)
- **Projection Layer**: 次元数を削減するための線形層（学習対象）。Text Encoderの出力を、より判別性の高い空間へ写像します。

### Inclusive Text Matching
クエリ $Q$ と、シーンに含まれる要素集合 $E = \{e_1, e_2, ..., e_m\}$ との類似度 $S(Q, E)$ は以下のように計算されます。

1.  クエリの各トークン $q_i$ について、最も類似度の高い要素 $e_j$ を探します（Max Sim）。
    $$ s_i = \max_{j} \text{cos\_sim}(q_i, e_j) $$
2.  全クエリトークンについて $s_i$ を平均します。
    $$ S(Q, E) = \frac{1}{|Q|} \sum_{i} s_i $$

これにより、「クエリに含まれるすべての単語が、シーン内のいずれかの要素によって説明されているか」を定量化します。

## 学習戦略 (`train.py`)
学習には **Adaptive Negative Injection (ANI)** という独自の手法を採用しています。

### 課題
Text-to-Text の学習において、単にランダムなシーンをネガティブサンプルとするだけでは、「歩行者がいる」シーンに対して「歩行者がいない」シーンを区別するような、細かい識別能力が得られにくい問題がありました。

### 解決策: ANI
学習バッチごとに、以下の3種類のクエリを動的に生成し、損失関数を計算します。

1.  **Positive Query ($L_{pos}$)**:
    *   正解シーンの要素集合から、一部をランダムにサンプリングしてクエリとします。
    *   目標: スコアを最大化 (1.0に近づける)。
2.  **Synthetic Negative Query ($L_{neg}$)**:
    *   正解シーンには存在しないが、意味的に紛らわしい要素（Hard Negative）を意図的に混入させたクエリを生成します。
    *   目標: スコアを最小化 (0.0に近づける)。
    *   *実装詳細*: 全データセットから抽出した「要素プール」からサンプリングし、現在のシーン要素と埋め込み類似度が高いが閾値以下のもの（似ているが違うもの）を選びます。
3.  **Retrieval Loss ($L_{contrastive}$)**:
    *   バッチ内の他のシーンをネガティブとする、標準的な対照学習損失。

$$ L_{total} = L_{contrastive} + L_{neg} + L_{pos} $$

## インデックスと検索
学習完了後、全シーンの要素を事前にエンコードし、`text_index.pt` として保存します。
推論時（Viewer）は、入力されたクエリのみをエンコードし、保存されたインデックスとの類似度を高速に計算します。
