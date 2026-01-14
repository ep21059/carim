# CARIM: Context-Aware Retrieval of Images for Merging

![CARIM Viewer](https://via.placeholder.com/800x400?text=CARIM+Viewer+Dashboard)

## 概要
CARIM (Context-Aware Retrieval of Images for Merging) は、自律走行車の走行シーン画像（nuScenesデータセット）に対して、自然言語クエリによる高精度な検索を行うためのシステムです。

最新の **Text-to-Text Architecture** (Qwen-1.5B) を採用し、画像から抽出された「テキスト要素（Elements）」とクエリの意味的類似度を学習することで、"A red car crossing intersection at night" のような複雑な文脈検索を実現します。

## 主な機能 (Ver 1.0)
- **Natural Language Search**: 自然言語でのシーン検索（例: "Pedestrian crossing in rain"）。
- **Browse & Filter**: 天候や時間帯によるフィルタリングと、グリッドビューでの全シーン閲覧。
- **Video Playback**: タイムラインスライダー付きの動画プレイヤーで、シーンの前後関係を確認可能。
- **Explainability (XAI)**: なぜそのシーンがヒットしたのか、どの単語（要素）が寄与したかを可視化 ("Why This Match?")。
- **High Stability**: 14.5k件のフルデータセットに対する安定した検索インデックス。

---

## クイックスタート (Viewerの利用)

学習済みモデルを使用して、検索Viewerをすぐに起動できます。

### 1. Viewerジョブの投入
```bash
sbatch slurm/run_viewer_trained.sbatch
```

### 2. アクセス手順 (ポート転送)
ジョブが起動したら、ローカルPC（手元のPC）のターミナルで以下を実行し、SSHポート転送を行います。
※ `NODE_IP` は `runs/CARIM/viewer_trained_*.out` ログファイルで確認してください（例: `192.168.170.xx`）。

```bash
# ローカルPCで実行
ssh -L 9991:NODE_IP:9991 ryoc1220@mprg.cs.chubu.ac.jp
```

ブラウザで **[http://localhost:9991](http://localhost:9991)** にアクセスします。

---

## 学習パイプライン (Training Pipeline)

ゼロからデータセットを作成し、モデルを学習する手順です。

### 1. データセット準備
VLM (Qwen-VL) によるキャプション生成と、LLMによる要素抽出を行います。
```bash
# 画像パスリスト作成
python3 scripts/build_dataset_from_images.py

# キャプション生成 & 要素抽出 & マージ (Slurmジョブ)
sbatch slurm/generate_full.sbatch
sbatch slurm/refine_full.sbatch
sbatch slurm/merge_full.sbatch
```
-> `datasets/nuscenes_vlm/processed/train_full.jsonl` が作成されます。

### 2. モデル学習 (Single-GPU)
安定性を重視し、シングルGPU構成 (A6000) で学習を行います。
```bash
sbatch slurm/train_full.sbatch
```
- **設定**: Batch Size 24, Epochs 5
- **出力**: `runs/carim_text_model_full.pt`

### 3. インデックス作成
全データの検索インデックスを構築します。
```bash
sbatch slurm/index_full.sbatch
```
- **出力**: `datasets/nuscenes_vlm/processed/text_index_full.pt`

---

## ディレクトリ構成

```text
carim_ver1/
├── app.py                # Viewer アプリケーション (Streamlit)
├── train.py              # 学習スクリプト
├── models/               # モデル定義
│   └── carim_scorer.py   # CARIMScorer (Qwen base)
├── scripts/              # ユーティリティ
│   └── indexer.py        # インデックス作成スクリプト
├── slurm/                # Slurmジョブスクリプト
│   ├── train_full.sbatch
│   ├── index_full.sbatch
│   └── run_viewer_trained.sbatch
└── docs/                 # ドキュメント
```

## 技術仕様
- **Base Model**: Qwen/Qwen2-1.5B-Instruct
- **Embedding Dim**: 256 (Projected from 1536)
- **Loss Function**: Custom Contrastive Loss with Adaptive Negative Injection (ANI)
- **Framework**: PyTorch, Hugging Face Transformers, Streamlit

## ライセンス
Proprietary / Research Use Only
