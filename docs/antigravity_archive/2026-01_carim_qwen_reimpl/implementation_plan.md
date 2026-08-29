# CARIM Architecture Re-implementation Plan (Text-to-Text)

The goal is to align the codebase with the actual CARIM paper (ICCV 2025) methodology, specifically transitioning from Image-Text retrieval to **Text-to-Text retrieval via Inclusive Text Matching**.

## User Review Required
> [!IMPORTANT]
> This is a fundamental architectural change. The previous Image-Text Contrastive (CLIP-like) model will be replaced by a Text-to-Text Attention model.
> - **Inputs**: Query Text + Generated Captions (instead of Images).
> - **Inference**: Matches Query against Caption Embeddings.
> - **Requirement**: Requires an LLM for Step 2 (Element Extraction). We will use Qwen (already available) for this.

## Proposed Changes

### 1. Dataset Rebuild (Executed 2026-01-10)
Due to metadata mismatches, the dataset is being rebuilt from physical images.
1.  **Delete Inconsistent Data**: `train.jsonl` and partial captions.
2.  **Image Census**: Identified 14,540 images in `CAM_FRONT`.
3.  **Metadata Regeneration**: Created new `train.jsonl` mapping these 14.5k images.
4.  **VLM Captioning**: Run Qwen-VL on all 14.5k images (Job 11955 Running).
5.  **LLM Refinement**: Extract elements from new captions.

### 2. Partial Pipeline Execution (Completed 2026-01-11)
To accelerate development, a partial pipeline was run on ~6,000 generated captions.
1. **Refinement**: `slurm/refine_partial.sbatch` -> `captions_elements.json` (6k items).
2. **Dataset Prep**: `scripts/prepare_partial_dataset.py` -> `train_partial.jsonl`.
3. **Training**: `slurm/train_partial.sbatch` -> `runs/carim_text_model.pt`.
    - Loss converged (Epoch 1..5).
4. **Indexing**: `slurm/index_partial.sbatch` -> `text_index_trained.pt`.
5. **Viewer**: `slurm/run_viewer_trained.sbatch` -> Hosted on Port 8525.

### 2. Data Preparation (Legacy/Refinement)
Implement "Step 2: Extract partial elements" from the paper diagram.
*   **Input**: `captions_inclusive.json` (Dense captions).
*   **Process**: Use Qwen-1.5B-Instruct to extract specific "elements" (objects, actions, environment) from dense captions.
*   **Output**: `captions_elements.json` (List of keywords/phrases per scene).

#### [NEW] `scripts/refine_captions_llm.py`
- script to iterate over dense captions and prompt LLM to extract elements.

### 2. Model Architecture (Inclusive Text Matcher)
Implement the architecture in Figure 3.
*   **Query Encoder**: Encodes user query into `Q` (Sequence of tokens?).
*   **Caption Encoder**: Encodes extracted elements into `K` (Keys) and `V` (Values).
*   **Mechanism**:
    1.  Compute Attention Matrix between $Q$ and $K$.
    2.  Select Max Key Features.
    3.  Compute Similarity $\sigma$.
    4.  Average for final score $s$.

#### [MODIFY] `models/carim_scorer.py`
- Rewrite `CARIMScorer` to accept `query_text` and `candidate_text_features` (instead of images).
- Implement the Attention/Softmax logic described in the paper.

#### [MODIFY] `models/text_encoder.py`
- Ensure it can handle batch encoding of caption lists.

### 3. Indexing
Index the *text* features of the captions/elements.

#### [MODIFY] `scripts/indexer.py`
- Load `captions_elements.json`.
- Encode elements using the Text Encoder.
- Save as `text_index.pt`.

### 4. Viewer / Retrieval
Update the search logic to use the new Text-to-Text scorer.

#### [MODIFY] `app.py`
- Load `text_index.pt`.
- Run retrieval using `model.compute_text_similarity(query, caption_index)`.
- (The video player logic remains the same, just the retrieval source changes).

### 5. Adaptive Negative Injection (ANI)
Implement the ANI strategy to robustly train the model against spurious correlations.

#### [NEW] `scripts/ani_utils.py`
- **Global Negative Pool**: Manage a list of unique elements seen across the dataset.
- **Semantic Filter**: Before using a negative, compute cosine similarity with positive elements. If sim > threshold (e.g. 0.7), discard it (it's likely a synonym like "pedestrian" vs "person").
- **Synthetic Query Generator**:
    - **Hard Negative**: `Positive Set` + `1 Negative Element`.
    - **Easy Negative**: `Multiple Negatives` + `Subset of Positives`.

#### [MODIFY] `train.py`
- Integrate `ANI Sampler`.
- For each batch, generate synthetic negative queries.
- **Loss**: Add a loss term (e.g., MSE or Margin Loss) forcing the score of (Synthetic Query, Positive Scene) to be **0** (or minimized), because the Synthetic Query contains a negative constraint that the Positive Scene does not satisfy.

## Verification Plan

### Automated Verification
1.  **Refinement Test**: Run `refine_captions_llm.py` on a small subset (5 scenes) and verify output JSON structure.
2.  **Scoring Test**: Create dummy Query and dummy Caption Elements. Verify standard "Inclusive" behavior (e.g. Query "Red Car" matches Caption ["Red Car", "Street"] better than ["Blue Car", "Street"]).
3.  **ANI Test**: confirm synthetic queries are generated and training loss decreases.

### Manual Verification
1.  **Viewer**: Run `app.py`, input complex queries (e.g. "Pedestrian crossing near white truck"), and verify if retrieved scenes match better than the previous version.
