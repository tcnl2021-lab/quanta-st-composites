# Quanta — standardised-test composite prediction

Streamlit + static landing page visualising the held-out prediction of seven
composite standardised-test scores from the TCNL Quanta multimodal feature
stack. Sibling demo to `brainage-heldout-demo`.

- **Cohort:** 427 healthy adults (TCNL Quanta multimodal ageing study)
- **Split:** 341 train / 86 held-out, age × sex stratified, frozen
  (inherited from the brain-age held-out repo)
- **Targets (7):** `LANGUAGE_ST_NORM_VCI`, `MEMORY_ST_NORM_{AudImm, VisImm, WorMem}`,
  `MOTOR_ST_SCALED_{FineMotor, Balance, ProcessingSpeed}` — publisher-canonical
  norm / scaled composite indices, pruned for linear redundancies
- **Held-out highlights:** Motor R² 0.48–0.59, ρ 0.70–0.77; Memory / Language
  R² 0.07–0.25; all seven beat the training-mean baseline

The app reads de-identified artifacts in `data/` produced by the internal
`Quanta_st_composites` pipeline. **No raw subject features ship in this repo.**

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (`tcnl2021-lab/quanta-st-composites`).
2. At <https://share.streamlit.io>, **New app** → pick the repo, branch
   `main`, entry file `app.py`. Free tier is sufficient.
3. Artifacts in `data/` are committed — no secrets, no env vars, no DB.

## What's in this repo

- `index.html` — static landing page with results table, composite-score
  design notes, modality contribution table, deviation summary.
- `app.py` — Streamlit browser app: target selector + five tabs
  (predicted-vs-true scatter, per-target overview, modality ablation,
  Zhang-corrected deviation explorer with FDR filter, about).
- `data/`
  - `holdout_predictions.npz` — `y_true`, `cv_oof`, `test_pred` per target × subject
  - `holdout_summary.csv` / `.json` — per-target CV + held-out metrics
  - `modality_ablation.csv` — leave-one-block-out: 7 targets × 7 blocks
  - `deviation_correlations.csv` — Zhang-corrected deviation Pearson r vs 30
    `BASIC_Q_*` + 70 unused `_ST_` features (with Fisher-z CIs + BH-FDR)
  - `deviation_summary.csv` — per-target deviation SD + sig-count summary
  - `deviation_zhang.npz` — per-target prediction + deviation vectors
  - `st_targets_meta.json` — target list, dropped candidates, missingness
- `requirements.txt` — `streamlit`, `numpy`, `pandas`, `plotly`.

## Refreshing the artifacts

In the internal `Quanta_st_composites` working repo:

```bash
micromamba run -n base python codes/st_composites/build_targets.py
micromamba run -n base python codes/st_composites/assemble_features.py
micromamba run -n base python codes/st_composites/train_holdout.py
micromamba run -n base python codes/st_composites/modality_ablation.py
micromamba run -n base python codes/st_composites/deviation_analysis.py
```

Then copy the regenerated files in `reports/` and `Data/st_targets_meta.json`
into `data/` here and commit.

## Provenance

- Source repo: internal `Quanta_st_composites` (commit `1b338b8`)
- Frozen split: inherited from internal `Quanta_heldout`
  (`codes/brain_age_2026/tuning/cache/holdout_split.npz`)
- Composite-score pruning rule, modality-ablation method, and Zhang
  bias correction are documented in `RESULTS.md` of the internal working repo.
