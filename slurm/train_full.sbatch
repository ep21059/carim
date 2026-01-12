#!/bin/bash
#SBATCH --job-name=TrainFull
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/ryoc1220/carim_ver1/runs/train_full_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/runs/train_full_%j.err

set -e

singularity exec --nv \
  --bind /home/ryoc1220/carim_ver1:/workspace \
  /home/ryoc1220/carim_ver1/carim_qwen.sif \
  bash -c '
  export PYTHONPATH=/workspace:$PYTHONPATH
  export TOKENIZERS_PARALLELISM=false
  cd /workspace
  
  echo "Training on Full Data (14.5k items) with 3 GPUs..."
  # Batch Size 16 (effective ~5 per GPU)
  # Elements encoding is flattened: 16 * 32 = 512 elements total batch.
  # Split across 3 GPUs = 170 elements per GPU.
  # Should fit A6000 easily.
  
  python3 train.py \
    --jsonl_path datasets/nuscenes_vlm/processed/train_full.jsonl \
    --elements_path datasets/nuscenes_vlm/captions_elements.json \
    --save_path runs/carim_text_model_full.pt \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5
'
