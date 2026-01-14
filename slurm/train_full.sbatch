#!/bin/bash
#SBATCH --job-name=TrainFull
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/ryoc1220/carim_ver1/runs/train_full_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/runs/train_full_%j.err

set -e

singularity exec --nv \
  --bind /home/ryoc1220/carim_ver1:/workspace \
  /home/ryoc1220/carim_ver1/carim_qwen.sif \
  bash -c '
  export PYTHONPATH=/workspace:$PYTHONPATH
  export CUDA_LAUNCH_BLOCKING=1
  export TOKENIZERS_PARALLELISM=false
  cd /workspace
  
  echo "Training on Full Data (14.5k items) with 1 GPU (Single Process)..."
  
  python3 train.py \
    --jsonl_path datasets/nuscenes_vlm/processed/train_full.jsonl \
    --elements_path datasets/nuscenes_vlm/captions_elements.json \
    --save_path runs/carim_text_model_full.pt \
    --epochs 5 \
    --batch_size 24 \
    --num_workers 0 \
    --lr 2e-5
'
