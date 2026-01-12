# CARIM: Context-Aware Retrieval of Images for Merging
## 概要
CARIM (Context-Aware Retrieval of Images for Merging) は、自律走行車の走行シーン画像（nuScenesデータセット等）に対して、自然言語クエリによる高精度な検索を行うためのシステムです。
最新の VLM (Qwen-VL) と LLM (Qwen-2) を組み合わせた **Text-to-Text Architecture** を採用し、従来の画像-テキスト対照学習（CLIP等）よりも詳細な「文脈（Context）」を考慮した検索を可能にします。

## 特徴
- **Text-to-Text Retrieval**: 画像そのものではなく、画像から生成・抽出された「テキスト要素（Elements）」に対して検索を行います。
- **Inclusive Text Matching**: クエリに含まれるキーワードが、シーン内の要素にどれだけ含まれているかを評価する独自スコアリング。
- **Adaptive Negative Injection (ANI)**: 学習時に動的に「似て非なる」ネガティブクエリを生成し、微細な違い（"Pedestrian" vs "Rider" 等）を識別できるようにします。
- **Interactive Viewer**: 検索結果を動画として閲覧できる Web UI (Streamlit製)。

## 環境構築
本プロジェクトは Singularity コンテナ上で動作することを前提としています。

### 1. Singularity イメージのビルド
```bash
./singularity/build_sif.sh
# または
sudo singularity build carim_qwen.sif singularity/carim_qwen.def
```

### 2. データセット準備
nuScenesデータセットを使用します。以下のスクリプトで物理画像を収集し、キャプション生成パイプラインを実行します。

```bash
# 1. 画像パスのリスト作成
python3 scripts/build_dataset_from_images.py

# 2. VLMによるキャプション生成 (Qwen-VL)
sbatch slurm/generate_full.sbatch

# 3. LLMによる要素抽出 (Refinement)
sbatch slurm/refine_full.sbatch

# 4. データセットのマージ (train.jsonl の作成)
sbatch slurm/merge_full.sbatch
```

## 学習 (Training)
モデルの学習は `train.py` で行います。Slurm環境での実行を推奨します。

```bash
sbatch slurm/train_full.sbatch
```
- **入力**: `train_full.jsonl` (VLMキャプションと要素リストを含む)
- **出力**: `runs/carim_text_model_full.pt`
- **設定**: 3-4 GPUでの分散学習 (DataParallel) をサポート。

詳細は [実装の詳細と学習戦略](docs/architecture.md) を参照してください。

## インデックス作成 (Indexing)
学習済みモデルを使用して、全データセットのテキスト要素をベクトル化し、検索用のインデックスを作成します。

```bash
sbatch slurm/index_full.sbatch
```
- **出力**: `datasets/nuscenes_vlm/processed/text_index_full.pt`

## Viewerの起動
学習済みモデルとインデックスを使用して、検索Viewerを起動します。

```bash
sbatch slurm/run_viewer_trained.sbatch
```
起動後、ブラウザで `http://localhost:8501` (ポートは設定依存) にアクセスしてください。

## ディレクトリ構成
- `app.py`: Viewer アプリケーション (Streamlit)
- `train.py`: 学習スクリプト
- `models/`: モデル定義 (CARIMScorer, TextEncoder)
- `scripts/`: データ処理・補助スクリプト
- `slurm/`: ジョブ投入用スクリプト
- `docs/`: ドキュメント

## ライセンス
Proprietary / Research Use Only
