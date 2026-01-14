#!/bin/bash
#SBATCH --job-name=CarimViewerTrained
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ryoc1220/carim_ver1/runs/CARIM/viewer_trained_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/runs/CARIM/viewer_trained_%j.err

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error

set -e

singularity exec --nv \
  --bind /home/ryoc1220/carim_ver1:/workspace \
  /home/ryoc1220/carim_ver1/carim_qwen.sif \
  bash -c '
  set -e
  cd /workspace
  
  # Prevent loading conflicting local packages
  export PYTHONNOUSERSITE=1
  export PYTHONPATH=/workspace:$PYTHONPATH

  echo "=== Running CARIM Viewer (Trained Model) ==="
  HOST_IP=$(hostname -I | awk "{print \$1}")
  PORT=9991
  
  echo "Starting Streamlit on $HOST_IP:$PORT..."
  
  # IMPORTANT: Loading the TRAINED index and TRAINED checkpoint (FULL DATASET)
  streamlit run app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    -- \
    --index_path datasets/nuscenes_vlm/processed/text_index_full.pt \
    --model_name Qwen/Qwen2-1.5B-Instruct \
    --checkpoint_path runs/carim_text_model_full.pt \
    --jsonl_path datasets/nuscenes_vlm/processed/train_full.jsonl
'
