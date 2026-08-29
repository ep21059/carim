# Task Checklist
- [ ] Understand the objective: Explain the implementation details of `carim_ver1`.
- [x] Read key source files to understand the implementation:
  - `README.md` (Done)
  - `models/carim_scorer.py` (Done)
  - `losses/itc_loss.py` (Done)
  - `train.py` (Done)
  - `app.py` (Done)
- [x] Read data processing scripts (`scripts/indexer.py`, etc.).
- [x] Synthesize findings into a detailed Japanese explanation for the user.
- [x] Create `kanazawa_ver` directory structure.
- [x] Copy and modify `scripts/` to `kanazawa_ver/scripts/` to target new dataset.
- [x] Create appropriate slurm scripts in `kanazawa_ver/slurm/` (generate, refine, merge, train).
- [x] Modify `kanazawa_ver/train.py` to allow loading pretrained weights (`--pretrained` argument).
[x] Pipeline is ready. Users can now execute the scripts in order.

- [x] Adapt `app.py` for Kanazawa dataset format and locations.
- [x] Create `slurm/index_kanazawa.sbatch` and submit it to generate `text_index.pt`.
- [ ] Wait for the index generation to complete.
- [x] Notify user and provide a `run_app.sbatch` or Streamlit command to test the search feature.

- [x] Write an experiment `REPORT.md` summarizing the Kanazawa fine-tuning logic, results, and time required.
- [x] Write a `README.md` for the `kanazawa_ver/` directory.
- [x] Commit the Kanazawa scripts and markdown files to Git (push requires user's Github Token).
