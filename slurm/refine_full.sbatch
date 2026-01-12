#!/bin/bash
#SBATCH --job-name=RefineFull
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ryoc1220/carim_ver1/runs/refine_full_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/runs/refine_full_%j.err

set -e

singularity exec --nv \
  --bind /home/ryoc1220/carim_ver1:/workspace \
  /home/ryoc1220/carim_ver1/carim_qwen.sif \
  bash -c '
  export PYTHONPATH=/workspace:$PYTHONPATH
  cd /workspace
  
  echo "Refining Full Captions (14.5k images)..."
  python3 scripts/refine_captions_llm.py \
    --json_path datasets/nuscenes_vlm/captions_inclusive.json \
    --output_file datasets/nuscenes_vlm/captions_elements.json \
    --model_path Qwen/Qwen2-1.5B-Instruct
'
