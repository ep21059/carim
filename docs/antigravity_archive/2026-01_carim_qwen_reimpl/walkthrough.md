# CARIM Re-implementation Walkthrough (Qwen Text-to-Text)

## Overview
We have successfully re-implemented CARIM using a Text-to-Text architecture with Qwen-1.5B-Instruct. The system now utilizes the full dataset (14.5k samples) and supports rapid retrieval via text queries and filtered browsing.

## Completed Work

### 1. Dataset Rebuild
- **Source**: Consolidated 14,540 physical images from consistent source directories.
- **Captions**: Generated VLM captions for all images.
- **Data Integration**: Merged elements extracted via LLM into `train_full.jsonl`.
- **Indexing**: Created `text_index_full.pt` containing embeddings for all 14.5k scenes.

### 2. Model Architecture (Text-to-Text)
- **Model**: `CARIMScorer` (wraps Qwen-1.5B).
- **Training Objective**: Projection Layer training using Contrastive Loss + ANI (Adaptive Negative Injection).
- **Optimization**: Switched from distributed multi-GPU to robust Single-GPU training (A6000 Ada/A6000) with extended runtime to prevent deadlocks.
- **Training Data**: Full dataset (14.5k samples).
- **Artifact**: `runs/carim_text_model_full.pt`.

### 3. Viewer Application
- **Path**: `app.py`
- **Features**:
    - **Search Mode**: Free-text query ("A red car crossing intersection").
    - **Browse Mode**: Filter by Time (Day/Night) and Weather (Rain/Sunny).
    - **Visualization**: Slider-based video playback, Token Heatmap Analysis, Matched Frame Thumbnail.
    - **Performance**: Cached model loading and efficient index search.

## Verification
### Training
- Metric: Loss converged (~0.12).
- Stability: Single-GPU training completed 5 epochs on 14.5k samples without error.

### Interface
1. **Launch**:
   ```bash
   sbatch slurm/run_viewer_trained.sbatch
   ```
2. **Access**:
   - Establish SSH tunnel: `ssh -L 9991:NODE_IP:9991 user@gateway`
   - Open Browser: `http://localhost:9991`

## Next Steps
- User verification of search quality on the full dataset.
- Potential fine-tuning of ANI thresholds if "hallucinations" (false positives) occur.
