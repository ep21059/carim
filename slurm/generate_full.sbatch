#!/bin/bash
#SBATCH --job-name=GenFullCaptions
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ryoc1220/carim_ver1/runs/gen_full_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/runs/gen_full_%j.err

set -e

singularity exec --nv \
  --bind /home/ryoc1220/carim_ver1:/workspace \
  /home/ryoc1220/carim_ver1/carim_qwen.sif \
  bash -c '
  export PYTHONPATH=/workspace:$PYTHONPATH
  cd /workspace
  
  echo "Generating Full Captions for 14.5k images..."
  python3 scripts/generate_captions_qwen.py \
    --data_dir datasets/nuscenes_vlm/samples/CAM_FRONT \
    --output_file datasets/nuscenes_vlm/captions_inclusive.json \
    --model_path Qwen/Qwen-VL-Chat \
    --batch_size 1
'
