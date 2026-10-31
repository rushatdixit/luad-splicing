# Analysis plan (frozen 2026-10-29, before any EDA or modelling)

Frozen before any PSI-dependent analysis. The train/test and CV splits were generated at the same time,
from sample metadata only (`data/processed/splits.csv`). Any later change must be logged in `docs/decisions.md`
with the reason, and must not be motivated by results.

Research question, hypotheses and leakage/honesty rules: see CLAUDE.md. Data facts: see `docs/data_inventory.md`.

## A. Frozen decisions (set by the project owner)

| Item | Decision |
|---|---|
| Outcome | tumour (1) vs non-tumour (0), label from the `_Norm` suffix only |
| Primary population | all 573 samples (514 tumour, 59 normal; 516 patients) |
| Pre-specified sensitivity population | samples from the 7 sites that contribute normals: 44, 55, 50, 49, 91, 38, 73 (234 tumour + 59 normal), a control for tissue-source-site confounding |
| Main statistics | paired Wilcoxon signed-rank test per event on the 57 matched tumour/normal pairs, BH-FDR across tested events |
| Secondary statistics | unpaired tumour vs normal (Mann-Whitney U) on all samples, BH-FDR |
| Evaluation | stratified, patient-grouped repeated CV (5 folds × 10 repeats) on the training patients, plus one patient-level held-out test set (20% of patients) used **once** at the end |
| Primary metric | ROC-AUC on the held-out test set |
| Secondary metrics | PR-AUC (classes are imbalanced, ~10% normal), confusion matrix, precision, recall, F1 |
| Baselines | majority-class predictor, then L2 and L1 logistic regression; random forest optional |
| Features | PSI values only. No clinical fields, no annotation columns (`pct_with_values`, `psi_range`, `std_psi`, `avg_exp_pct`), no site |
| Fitted inside training folds only | coverage/variability filters, imputation, scaling, feature selection, hyperparameter tuning |
| Seed | `SEED = 42` everywhere |

## B. Splits (generated now, `src/splits.py`)

- Unit of splitting: the **patient**. A tumour and its matched normal always land in the same set and fold.
- Patient strata (stratification keeps class balance and site mix comparable across sets):
  `paired` (57), `normal_only` (2 unpaired normals), `tumour_normal_site` (tumour-only patients at the 7 normal sites, 177),
  `tumour_other_site` (tumour-only patients at the other 26 sites, 280).
- Within each stratum: random shuffle → stable sort by site → assign group = (position + random offset) mod k.
  This spreads each stratum and each site evenly across groups.
- Held-out test = one group of a 5-way assignment (≈ 20% of patients). CV = 10 repeats × 5 folds on the training patients only,
  with a fresh shuffle per repeat from the same seeded generator.
- The same splits are used for every model and for the 7-site sensitivity analysis (by subsetting rows).

## C. Statistics details (confirmed by the owner before Phase 3; see decisions.md #10-13)

- Paired test: `scipy.stats.wilcoxon(tumour − normal, zero_method="wilcox")` per event, on complete pairs only.
  An event is tested only if it has **≥ 30 complete pairs**. Untestable events (too few pairs, or all differences zero) are counted and reported, not dropped silently.
- Effect size: **median paired ΔPSI** (tumour − normal) with a 95% percentile bootstrap CI over that event's complete pairs (2,000 resamples, seed 42).
  The mean paired ΔPSI is reported as an extra column.
- Paired analysis in two versions: (a) all 57 pairs, "descriptive, full cohort"; (b) the 46 training-patient pairs only.
  **Only (b) may feed later feature ranking.**
- Unpaired (secondary): Mann-Whitney U per event on non-missing values. Effect size = difference in median PSI (tumour − normal) with a 95% bootstrap CI
  (resampled within each group), plus the difference in means. Run on the full cohort and on the 7-site sensitivity population.
- BH-FDR applied separately within each analysis family. Adjusted p-values reported.
- These whole-cohort tests are EDA only. They never feed model feature selection (except (b), which uses training patients only).

## D. Modelling details (confirmed by the owner before Phase 4; see decisions.md #15-18)

- Outer CV: the frozen `splits.csv` folds (5 × 10, patient-grouped), training patients only. The test set is untouched until Stage 2.
- Inside every outer training fold, in this order:
  1. coverage filter (≤ 20% missing in the fold's training samples, ≥ 25 complete training pairs);
  2. paired Wilcoxon on the fold's training pairs, keep BH-FDR < 0.05, rank by |median ΔPSI| (top-up if fewer than k pass);
  3. top-k events (k = 10, 25, 50, 100);
  4. Pipeline: median imputation → standard scaling → classifier.
  No missingness indicators, clinical fields or site.
- Classifiers:
  - majority class;
  - L2 and L1 logistic regression (balanced class weights; C tuned by 3-fold patient-grouped inner CV);
  - random forest (500 trees, untuned).
- Threshold 0.5 on P(tumour) for confusion matrix, precision, recall and F1. Never tuned on validation or test data.
- Leakage audit: shuffled training labels with pseudo-pairs; expected ROC-AUC ≈ 0.5. Any AUC > 0.98 is flagged and audited.
- Sensitivity: 7 normal-contributing sites only; without the 8 site-44 cluster tumours.
- Stage 2 (later): the final model trained on all training patients, ranking from Phase 3 (b) / all 46 training pairs, evaluated once on the test set.

## E. Final test protocol (FROZEN 2026-10-29, before any test-set access; see decisions.md #20-21)

| Item | Frozen choice |
|---|---|
| Final model | LR-L2 (balanced), k = 25, threshold 0.5; C tuned by 3-fold patient-grouped inner CV on training data |
| Ranking | 46 training pairs; ≤ 20% missing over 460 training samples and ≥ 30 complete pairs; paired Wilcoxon; BH-FDR < 0.05 gate; rank by \|median ΔPSI\| (tie-break \|mean ΔPSI\|) |
| Fitted on training only | coverage filter, ranking, median imputation, scaling, C |
| Primary result | test ROC-AUC (113 samples) with a patient-level bootstrap 95% CI (2,000 resamples, seed 42) |
| Secondary (always reported) | PR-AUC (normal) with CI; confusion matrix at 0.5; majority-class baseline; RF (k = 25) |
| Sensitivity | refit without the 7 training site-44 cluster tumours; evaluate on the 112 test samples without TCGA_44_6146 (main model on the same 112 also reported) |
| One-shot rule | `src/final_test.py` creates `results/test_lock.json` before touching test data and refuses to run if it exists |
| Shortlist | ≥ 80% CV selection frequency at k = 25 AND in the final top-25 |
