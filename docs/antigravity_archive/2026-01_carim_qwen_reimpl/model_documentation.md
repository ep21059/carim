# CARIM Implementation Summary

This document details the current implementation of the CARIM (Caption-based Autonomous Driving scene Retrieval via Inclusive text Matching) model, reflecting the Text-to-Text architecture and Adaptive Negative Injection (ANI) mechanisms.

## 1. Architecture: Inclusive Text-to-Text Retrieval

The model has shifted from traditional Image-Text retrieval to a **Text-to-Text** approach to resolve domain misalignment.

*   **Query Encoder**: Encodes user query text into a sequence of token embeddings.
*   **Scene Representation**: Videos are represented not by raw frames, but by **Refined Elements** extracted from VLM-generated captions.
*   **Element Encoder**: Encodes scene elements (keywords/phrases) into embeddings.
*   **Projection Layer**: Projects both Query and Element embeddings into a shared latent space ($D=256$).

## 2. Scoring Mechanism (Inclusive Matching)

The scoring function is designed to support "Hard Constraints" (all query conditions must be met) while allowing "Inclusive" content (extra info in video is ignored).

**Formula:**
$$ S(Q, E) = \frac{1}{|Q_{valid}|} \sum_{i=1}^{|Q|} \max_{j} \text{sim}(q_i, e_j) $$

*   **Local Alignment (MaxSim)**: For each token $q_i$ in the query, we find the **single most similar** element $e_j$ in the scene (using `argmax` equivalent `torch.max`). This ensures that unrelated video elements do not dilute the score.
*   **Global Aggregation (Average)**: The individual best-match scores are averaged. This represents the **Satisfaction Ratio** of the query conditions.
    *   *Verified:* A query with 3 matching conditions out of 4 results in a score of $\approx 0.75$.

## 3. ANI: Adaptive Negative Injection

To prevent the model from learning spurious correlations and to enforce strict adherence to negative constraints.

### 3.1. Negative Selection Logic
*   **Global Pool**: A collection of unique elements extracted from the entire dataset.
*   **Semantic Filtering**:
    *   Before selecting a negative element $n$ for a positive set $P$, we calculate the cosine similarity: $\max_{p \in P} \text{sim}(n, p)$.
    *   **Threshold**: If similarity $> 0.65$, the candidate is rejected. This prevents "Pedestrian" being used as a negative for "Person".

### 3.2. Synthetic Query Generation (Balanced Sampling)
The training batch includes synthetic queries generated with the following probabilities:

| Type | Composition | Probability | Purpose |
| :--- | :--- | :--- | :--- |
| **Pure Negative** | 2 Negative Elements (No Positives) | 20% | Teach model to score 0 when nothing matches (Balanced Sampling). |
| **Hard Negative** | All Positive Elements + 1 Negative | 40% | Enforce strict boundary; 1 mismatch should penalty score. |
| **Easy Negative** | Subset of Positives + 1 Negative | 40% | General robustness. |

*   **Positive Subsets**: Additionally, random subsets of positive elements are generated to train $L_{pos}$.

## 4. Loss Functions

The total loss $L$ combines three objectives to ensure precise retrieval:

$$ L = L_{self} + L_{neg} + L_{pos} $$

### 4.1. $L_{self}$ (Self-Alignment / Contrastive)
*   **Input**: Ground Truth Caption vs Scene Elements.
*   **Goal**: Standard contrastive learning to align the global caption with its constituent elements.

### 4.2. $L_{neg}$ (Negative Rejection)
*   **Input**: Synthetic Negative Query vs Scene Elements.
*   **Goal**: **Minimize** the score ($S \to 0$).
*   **Effect**: Penalizes the model if it "hallucinates" a match for the injected negative element.

### 4.3. $L_{pos}$ (Positive Reinforcement)
*   **Input**: Synthetic Positive Query (Subset) vs Scene Elements.
*   **Goal**: **Maximize** the score ($S \to 1$).
*   **Effect**: Ensures that even a partial query (e.g., "Red Car only") scores perfect 1.0 if that element exists, reinforcing the "Inclusive" nature.

## 5. Data Pipeline Implementation

1.  **VLM Captioning**: `Qwen-VL` generates detailed descriptions.
2.  **LLM Refinement**: `Qwen-1.5B` extracts "Active Elements" and removes negative constraints (e.g., "no pedestrians").
3.  **Training**: `train.py` loads these elements and applies ANI on-the-fly.

## 6. Risks and Tuning Points (User Feedback)

### 6.1. Risks
1.  **Embedding Filter False Positives**: High similarity between antonyms (e.g., "Red" vs "Blue") might cause important negatives to be filtered out.
2.  **Loss Imbalance**: High ratio of negatives (80%) might make the model too conservative, lowering Recall.
3.  **VLM Dependency**: Missed objects in VLM captions result in permanent retrieval loss.

### 6.2. Monitoring Metrics
*   **Score Histogram**: Verify clear separation between Positive (>0.95) and 1-Mismatch Negative (~0.75).
*   **Attribute Discriminability**: Explicitly test if the model distinguishes attributes like color or state (e.g., "Stopped" vs "Moving").
