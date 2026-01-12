#!/bin/bash
#SBATCH --job-name=MergeFull
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=a6000_ada
#SBATCH --time=00:10:00
#SBATCH --mem=8G

python3 scripts/merge_full_dataset.py \
    --train_jsonl datasets/nuscenes_vlm/processed/train.jsonl \
    --captions_json datasets/nuscenes_vlm/captions_inclusive.json \
    --elements_json datasets/nuscenes_vlm/captions_elements.json \
    --output_jsonl datasets/nuscenes_vlm/processed/train_full.jsonl
