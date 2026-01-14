#!/bin/bash
#SBATCH --job-name=IndexFull
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ryoc1220/carim_ver1/runs/index_full_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/runs/index_full_%j.err

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error

set -e

singularity exec --nv \
  --bind /home/ryoc1220/carim_ver1:/workspace \
  /home/ryoc1220/carim_ver1/carim_qwen.sif \
  bash -c '
  export PYTHONPATH=/workspace:$PYTHONPATH
  cd /workspace
  
  echo "Indexing Full Data (14.5k items)..."
  # Indexing uses the model trained by train_full.sbatch
  
  python3 scripts/indexer.py \
    --captions_file datasets/nuscenes_vlm/processed/train_full.jsonl \
    --output_file datasets/nuscenes_vlm/processed/text_index_full.pt \
    --checkpoint runs/carim_text_model_full.pt \
    --model_name Qwen/Qwen2-1.5B-Instruct
'
