# Phase 4, Stage 1 report: cross-validated models (training patients only)

Produced by `python -m src.run_models_cv` (env `luad-splicing`, seed 42), on 2026-10-29. Settings: `docs/decisions.md` #15-18 and `docs/analysis_plan.md` §D,
fixed before any model was fit. **The held-out test set was not used.** All numbers are means ± SD over 50 outer validation folds
(5 folds × 10 repeats, patient-grouped, from the frozen `splits.csv`).

---

## ⚠ Flag: AUC > 0.98

**Every real-label model in every configuration exceeds ROC-AUC 0.98.** For each of L1, L2 and RF, at every k:
- main: 50 of 50 folds above 0.98;
- 7-site: 50 of 50 folds;
- without site 44: 49-50 of 50 folds.

CLAUDE.md says to assume leakage first. The audit in §3 found no evidence of pipeline leakage. Its findings, briefly:
- the shuffled-label control collapses toward chance;
- what remains of the null signal is explained by chance overlap with the true labels;
- the per-fold assertions passed in all 200 fold runs.

The very high AUC is consistent with the very large paired effects seen in Phase 3 (median ΔPSI up to 0.79). **A near-perfect within-cohort CV AUC is not
evidence that the result generalises.** It does not rule out confounding by tissue composition or by how the samples were collected and processed (§4).

---

## 1. Observations

### 1.1 What each fold contained (`cv_selected_events_per_fold.csv`)

| config | training samples per fold | training normals | pairs used for ranking | events passing coverage | events passing in-fold FDR < 0.05 |
|---|---|---|---|---|---|
| main | ~368 | 37-39 | 36-37 | ~25,560 | 3,782-4,470 (mean 4,126) |
| 7 sites only | ~189 | 37-39 | 36-37 | ~25,560 | mean 4,125 |
| without site-44 cluster | ~362 | 37-39 | 33-36 | ~25,260 | mean 4,346 |
| shuffled labels (null) | ~368 | 37-39 | 37-39 pseudo-pairs | ~26,440 | 0-1 |

In every real-label fold at least 3,782 events passed the FDR gate, so no top-up was needed. In the null, almost nothing passed (0-1 events),
so the top-k was always filled by effect size alone. That is the intended behaviour of the pre-specified rule.

### 1.2 Selection stability (`cv_selection_frequency_by_event.csv`, `cv_selection_frequency_by_gene.csv`; figure `cv_selection_frequency_main_k25`)

At k = 25 (main), across the 50 folds:
- **Only 44 distinct events were ever selected.** 19 were selected in ≥ 80% of folds, and **11 in every fold**:
  SCEL ES, MYH14 ES, SLC19A1 AP, SPTBN1 AP, KIF4A AT, NEIL3 AT, NUMB ES, SH3GL3 ES, FBLN2 ES, MTMR1 ES and HYAL1 AP.
- **24 of the 25 most frequently selected events are also in the Phase 3 (b) top-25.** (b) was not used in CV; this is a descriptive comparison.
- The sensitivity configs select almost the same set: 7-site, 45 distinct and 13 always selected; without site 44, 43 distinct and 10 always selected.
- Null: 679 distinct events were selected, none in more than 80% of folds. This is what a pipeline choosing noise looks like.
- **By gene:** each frequently selected gene contributes a single event, except FAM72A (two events, an AP and an AT, with max frequency 0.98).
  The stable set is spread across different genes rather than concentrated in a few.

### 1.3 Misclassified samples (main, k = 25; `cv_oof_mean_prob_main_k25.csv`)

Mean out-of-fold P(tumour) across the 10 repeats. A sample counts here if it falls on the wrong side of 0.5 for at least one of LR-L2, LR-L1 and RF:
- **3 normals** (`TCGA_49_4512_Norm`, `TCGA_55_6983_Norm`, `TCGA_91_6835_Norm`). All three are paired, and each is misclassified by only some of the three models.
- **12 tumours**:
  - 8 of the 12 come from sites that contributed no normals (75, 86, 93, 97 ×3, 99, L4). Roughly 54% of training tumours come from such sites.
    The other 4 are from sites 44 (`TCGA_44_6148`, not part of the site-44 cluster), 49 and 55 (×2).
  - 9 of the 12 have no purity value (32% of all tumours lack one).
- Across tumours with purity data, purity does not correlate with P(tumour) (Spearman ρ = 0.00, p = 0.96).

---

## 2. Model results (`cv_metrics_summary.csv`, `cv_metrics_per_fold.csv`; figures `cv_roc_auc_by_model_k`, `cv_pr_auc_normal_by_model_k`, `cv_mean_roc_main_k25`)

Main config (all training patients). PR-AUC (normal) treats the minority class as positive; its chance level equals the normal prevalence, about 0.10.
Threshold 0.5 on P(tumour).

| model | k | ROC-AUC | PR-AUC (normal) | PR-AUC (tumour) | precision (tumour) | recall (tumour) | F1 (tumour) | recall, normal (specificity) |
|---|---|---|---|---|---|---|---|---|
| majority class | any | 0.500 ± 0.000 | 0.104 ± 0.006 | 0.896 | 0.896 | 1.000 | 0.945 | 0.000 |
| LR-L2 | 10 | 0.997 ± 0.004 | 0.969 ± 0.047 | 1.000 | 0.997 | 0.980 ± 0.015 | 0.988 | 0.971 ± 0.062 |
| LR-L2 | 25 | 0.998 ± 0.003 | 0.978 ± 0.036 | 1.000 | 0.999 | 0.970 ± 0.018 | 0.984 | 0.990 ± 0.058 |
| LR-L2 | 50 | 0.999 ± 0.002 | 0.988 ± 0.016 | 1.000 | 0.999 | 0.982 ± 0.014 | 0.991 | 0.994 ± 0.024 |
| LR-L2 | 100 | 0.999 ± 0.001 | 0.995 ± 0.008 | 1.000 | 1.000 | 0.989 ± 0.009 | 0.994 | 0.996 ± 0.021 |
| LR-L1 | 10 | 0.996 ± 0.004 | 0.963 ± 0.044 | 1.000 | 0.996 | 0.981 ± 0.014 | 0.988 | 0.966 ± 0.060 |
| LR-L1 | 25 | 0.996 ± 0.003 | 0.964 ± 0.038 | 1.000 | 0.996 | 0.975 ± 0.015 | 0.985 | 0.964 ± 0.060 |
| LR-L1 | 50 | 0.998 ± 0.003 | 0.981 ± 0.021 | 1.000 | 0.997 | 0.979 ± 0.015 | 0.988 | 0.974 ± 0.045 |
| LR-L1 | 100 | 0.999 ± 0.001 | 0.995 ± 0.011 | 1.000 | 1.000 | 0.987 ± 0.011 | 0.994 | 0.998 ± 0.014 |
| RF | 10 | 0.995 ± 0.005 | 0.948 ± 0.061 | 0.999 | 0.994 | 0.988 ± 0.009 | 0.991 | 0.945 ± 0.073 |
| RF | 25 | 0.997 ± 0.003 | 0.972 ± 0.037 | 1.000 | 0.994 | 0.990 ± 0.008 | 0.991 | 0.945 ± 0.062 |
| RF | 50 | 0.998 ± 0.003 | 0.981 ± 0.033 | 1.000 | 0.993 | 0.990 ± 0.008 | 0.992 | 0.945 ± 0.059 |
| RF | 100 | 0.999 ± 0.002 | 0.991 ± 0.018 | 1.000 | 0.995 | 0.990 ± 0.007 | 0.993 | 0.959 ± 0.055 |

SDs for every metric and the mean confusion-matrix counts (TN/FP/FN/TP) are in `cv_metrics_summary.csv`.
PR-AUC (tumour) is near 1 even for weak models, because about 90% of samples are tumours. PR-AUC (normal) is the more informative of the two.

Sensitivity configs (ROC-AUC mean ± SD, range over models and k):

| config | LR-L2 | LR-L1 | RF | majority PR-AUC (normal) |
|---|---|---|---|---|
| **7 normal-contributing sites only** | 1.000 ± 0.000 (all k) | 0.999-1.000 | 1.000 | 0.203 |
| **without site-44 cluster** | 0.997-0.999 | 0.996-0.999 | 0.995-0.999 | 0.106 |

- Differences between models and between values of k are tiny (≤ 0.004 in mean AUC). They are within one SD and should not be read as a ranking of models.
- Larger k improves specificity and normal-class PR-AUC slightly.
- Random forest has the lowest specificity at threshold 0.5 (0.945, i.e. about 1 in 18 normals called tumour, roughly 0.5 per validation fold). Its AUC is the same as the others, so this is a threshold and calibration effect, not worse ranking.

---

## 3. Leakage audit

### 3.1 Shuffled training labels (`null_shuffled`; figure `cv_null_audit_auc`; `cv_null_audit_label_overlap.csv`)
In each fold, training labels were permuted (class counts kept), normals were pseudo-paired with tumours, and the identical pipeline ran. Validation labels were real.

| model | k = 10 | k = 25 | k = 50 | k = 100 |
|---|---|---|---|---|
| LR-L2 | 0.566 ± 0.257 | 0.584 ± 0.258 | 0.583 ± 0.251 | 0.587 ± 0.231 |
| LR-L1 | 0.559 ± 0.220 | 0.578 ± 0.196 | 0.574 ± 0.178 | 0.594 ± 0.142 |
| RF | 0.520 ± 0.196 | 0.546 ± 0.207 | 0.560 ± 0.210 | 0.562 ± 0.223 |

- **The null collapses from about 0.998 to about 0.52-0.59, with a wide spread** (fold AUCs from about 0.05 to 0.95; 0 of 600 null model-folds above 0.98).
  This is the expected pattern if the real-label performance comes from real label information, not from a leak in the pipeline.
- **Why the mean sits slightly above 0.5:** a random permutation leaves some real normals in the shuffled "normal" class by chance (1-9 per fold, expected ≈ 4.0).
  Because the true signal is so strong, the model learns from that chance overlap. The null fold AUC correlates with the excess of real normals over expectation
  (Spearman ρ = +0.54 to +0.72, all p ≤ 5e-5). In these 50 permutations the realised overlap was slightly above expectation (mean 4.28 vs 4.01), which shifts the mean
  above 0.5. This is a known property of permutation controls when the signal is strong. It is not a constant leak, and **no further action seems needed**.
  A stricter null that keeps exactly zero overlap with the true labels is possible if you want one.

### 3.2 Structural checks (all passed in all 200 fold runs, and in `tests/test_models.py`)

| check | where |
|---|---|
| No patient in both the training and validation fold | `src/models.py:116` |
| Ranking pairs come only from the training fold, and contain no validation patient | `src/models.py:121-122` |
| Coverage filter, ranking, imputation and scaling are fit on training-fold data only | `src/models.py` `run_fold`, `coverage_filter`, `rank_events`, `make_models` (sklearn `Pipeline`, refit inside the inner tuning folds as well) |
| Only event IDs used as features (no clinical fields, site or missingness indicators) | `src/models.py:136` |
| No NaN reaches a classifier | `src/models.py:147` |
| No test-set sample enters CV | `src/models.py:160`; `src/run_models_cv.py` (asserts no test sample in out-of-fold predictions); `tests/test_models.py::test_no_test_sample_in_cv_outputs` |
| The Phase 3 (b) ranking is not used in CV | (b) is only read *after* CV, in `run_models_cv.main`, for the descriptive overlap column `in_phase3b_topk` |

**Files to review for leakage:** `src/models.py` lines 113-151 (`run_fold`) and 154-161 (`fold_indices`), and `src/run_models_cv.py` `run_config`.

### 3.3 Known limitations of the procedure (not leakage of validation data)
- Ranking is done once per outer fold, not inside the inner folds used to tune C. This can bias only the choice of C, not the outer estimate.
- A bug was found and fixed *before* the full run: with sklearn 1.9, `penalty="l1"` combined with the default `l1_ratio=0` silently fits an L2 model.
  The code now sets `l1_ratio` explicitly, and a check confirmed that L1 gives sparse coefficients. All reported L1 results come from the corrected code.

---

## 4. Interpretation

- **Within this cohort, a few PSI events separate tumour from adjacent non-tumour lung almost perfectly in patient-grouped CV.** This holds with
  10 events, holds when the analysis is restricted to the 7 sites that contribute normals, and holds without the site-44 cluster. The secondary hypothesis
  (a small subset keeps performance on held-out samples) is supported **at the CV level only**. The test set has not yet been used, and no external cohort has been tested.
- **The site imbalance does not appear to drive the classification.** Restricted to the 7 sites with normals, where tumours and normals share the same sites, performance stays near 1.0.
  Most misclassified tumours (8 of 12) come from sites without normals. That fits a modest site shift, not a site shortcut.
- **The site-44 cluster does not drive the result** (AUC and selected events essentially unchanged without it).
- **What the high AUC does not show:** it does not show that these events are tumour-specific splicing changes, rather than markers of tissue composition
  (tumour epithelium vs the cell mix of normal lung). It also does not show that they would hold in other cohorts, platforms or alignment versions. Tumour vs
  *adjacent normal* tissue is also an easier task than any clinically relevant comparison. None of this implies clinical usefulness.
- **The task looks close to linearly separable with a few features**, so the specific model choice barely matters. In Stage 2, a simple model (LR-L2) with a small k
  will be easier to interpret, at no measurable cost in performance.

## 5. Speculation (unverified)

- Many of the most stable events are alternative promoter or terminator events (SLC19A1, SPTBN1, HYAL1, NTM, FAM72A, TEF AP; KIF4A, NEIL3 AT). Some top genes look
  proliferation-related by name (e.g. KIF4A, NEIL3, FAM72A). **That is not verified here.** If confirmed, they could mark differences in proliferation or cell-type mix
  rather than splicing within the same cells. This needs checking against sourced annotation in Phase 5.
- The few misclassified tumours, mostly from sites without normals and mostly without purity values, could be low-purity or atypical samples. That is untested.

## 6. Surprises (flagged for the project owner)

1. **Every real-label AUC is above 0.98** (see the flag at the top).
2. **The shuffled-label null averages 0.52-0.59, not exactly 0.5.** It is explained by chance overlap between shuffled and true labels (§3.1), and no fold comes close to the real AUCs.
3. **The 7-site restriction makes the task slightly easier**, not harder (AUC 1.000). The extra difficulty in the main config comes from tumours from sites without normals.
4. **Selection is very stable** (11 events in every fold at k = 25) and almost the same as the Phase 3 (b) top list, even though (b) was never used in CV.
5. **An sklearn 1.9 API change silently turned L1 into L2.** It was caught by its warning on a timing run and fixed before the reported run (§3.3).

## 7. What Stage 2 needs from you
- Choose the final model and k. My suggestion: LR-L2 with k = 25 or smaller. All options are within noise, so this choice cannot be tuned to the test set.
- Confirm that the final ranking uses all 46 training pairs (the Phase 3 (b) ranking with the same gate and ordering rule). Then: one fit on all training patients, and one evaluation on the 102 test patients (11 pairs).
  With 11 test normals, the test AUC confidence interval will be wide.
- Decide whether the site-44 sensitivity analysis should also be evaluated on the test set, excluding `TCGA_44_6146`.

## 8. Reproducibility
- Code: `src/models.py`, `src/run_models_cv.py` (plus the `n_boot=0` option added to `src/eda_stats.py::paired_tests`). Tests: `tests/test_models.py`. All 18 tests pass.
- Runtime: about 75-115 s per config with joblib. Cached per config in `results/tables/cv_runs/` (`--force` reruns).
- Two descriptive tables (`cv_null_audit_label_overlap.csv` and `cv_oof_mean_prob_main_k25.csv`) were produced with one-off analysis commands after the main run.
  They are not yet part of `run_models_cv.py`. I can move them into the script if you want them regenerated automatically.
