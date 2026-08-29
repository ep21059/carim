# 金沢データセット追加学習（パターンA）実装計画

金沢データセット（`/home/ryoc1220/carim_ver1/datasets/kanazawa/20250127_151151`）専用のモデル（パターンA）を追加学習するためのパイプラインを構築します。
本家実装と環境を分離するため、`carim_ver1/kanazawa_ver/` 以下に独立したディレクトリ構成を作成します。

## 概要

`kanazawa_ver` ディレクトリ配下に `scripts`, `slurm` ディレクトリなどをコピー・配置し、金沢データセット向けのパス設定に書き換えます。その後、キャプション生成から追加学習までのパイプラインを実行します。

## Proposed Changes

### 1. 新規ディレクトリ・環境構築 (`/home/ryoc1220/carim_ver1/kanazawa_ver/`)
`kanazawa_ver` 直下に以下を準備します。
- `models/`, `losses/`, `carim_qwen.sif` をシンボリックリンクまたはコピーで配置
- `scripts/` ディレクトリ配下に以下のファイル群を配置

#### [NEW] `kanazawa_ver/scripts/build_dataset_from_images.py`
画像ディレクトリ (`/home/ryoc1220/carim_ver1/datasets/kanazawa/20250127_151151`) の画像を読み込み、空のメタデータを持つ `train.jsonl` を作成するようパスを変更します。

#### [NEW] `kanazawa_ver/scripts/generate_captions_qwen.py`, `refine_captions_llm.py`, `merge_full_dataset.py`, `indexer.py`
本家の `scripts/` サブモジュールをコピーし、必要に応じて変更。

### 2. Slurmジョブスクリプト（キャプション生成・精製・マージ）
キャプション生成から要素抽出までを行うパイプラインのSlurmスクリプトを作成します。

#### [NEW] `kanazawa_ver/slurm/generate_kanazawa.sbatch`
- `scripts/generate_captions_qwen.py` を実行（Qwen-VL-Chat）。出力先は `kanazawa_ver/datasets/...`

#### [NEW] `kanazawa_ver/slurm/refine_kanazawa.sbatch`
- `scripts/refine_captions_llm.py` を実行（Qwen2-1.5B）

#### [NEW] `kanazawa_ver/slurm/merge_kanazawa.sbatch`
- `scripts/merge_full_dataset.py` を実行。

### 3. 追加学習ジョブ
`carim_ver1` の重みをベースにファインチューニングを行うスクリプトです。

#### [MODIFY] `kanazawa_ver/train.py`
本家の `train.py` をコピーし、Argparseに `--pretrained` 引数を追加。指定されたら学習ループに入る前に `runs/carim_text_model_full.pt` を `model.load_state_dict(...)` で読み込むように改修します。

#### [NEW] `kanazawa_ver/slurm/train_kanazawa.sbatch`
`kanazawa_ver/train.py` を実行するジョブスクリプト。

## Verification Plan
1. `kanazawa_ver/scripts/build_dataset_from_images.py` 等のファイルの準備完了をユーザーに報告。
2. ユーザー様に `sbatch kanazawa_ver/slurm/generate_kanazawa.sbatch` 等を順次実行していただく。
