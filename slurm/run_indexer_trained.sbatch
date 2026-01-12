#!/bin/bash
#SBATCH --job-name=TextIndexTrained
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ryoc1220/carim_ver1/runs/CARIM/text_index_trained_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/runs/CARIM/text_index_trained_%j.err

set -e

singularity exec --nv \
  --bind /home/ryoc1220/carim_ver1:/workspace \
  /home/ryoc1220/carim_ver1/carim_qwen.sif \
  bash -c '
  set -e
  cd /workspace
  export PYTHONPATH=/workspace:$PYTHONPATH

  echo "=== Running Text Indexing (Trained Model) ==="
  mkdir -p datasets/nuscenes_vlm/processed

  # Output to a NEW index file to differentiate from zero-shot
  python3 scripts/indexer.py \
    --captions_file datasets/nuscenes_vlm/captions_elements.json \
    --output_file datasets/nuscenes_vlm/processed/text_index_trained.pt \
    --model_name Qwen/Qwen2-1.5B-Instruct \
    --checkpoint runs/CARIM/carim_text_trained.pt
  
  echo "=== Done ==="
'
