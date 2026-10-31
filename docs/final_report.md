# Final report: alternative-splicing events distinguishing LUAD tumour from adjacent non-tumour lung (TCGA SpliceSeq)

**Candidate-biomarker discovery only.** Nothing here is a diagnostic, treatment recommendation or validated biomarker.
All results come from one cohort (TCGA-LUAD, SpliceSeq PSI, 2016 alignment). No independent cohort was tested.

Supporting documents: `docs/data_inventory.md`, `docs/qc_report.md`, `docs/eda_report.md`, `docs/model_report.md`,
`docs/analysis_plan.md` (frozen protocol, §E) and `docs/decisions.md` (#1-21).

---

## 0. Research question and design (recap)
- **Question:** can alternative-splicing events distinguish tumour from non-tumour lung samples, and which events give the strongest, most reproducible signal?
- **Data:** 573 samples (514 tumour, 59 non-tumour) from 516 patients, with 57 matched pairs and 27,836 events.
- **Labels:** taken only from the `_Norm` suffix of each sample column.
- **Splits:** fixed before any data analysis. The held-out test set is 102 patients (113 samples; 11 normals, all paired). Cross-validation used the 414 training patients
  (5 folds × 10 repeats, grouped by patient).
- **Protocol:** written into `docs/decisions.md` / `analysis_plan.md` at 14:35:20 (+05:30). The one-time test lock was created at 14:37:16 (+05:30); see `results/test_lock.json`.
  The lock records SHA-256 hashes of the code and data files used.

---

## 1. Observations (data facts, not model outputs)

- **Site imbalance.** Normals come from only 7 of 33 tissue source sites. Site is the main driver of global PSI variation: it explains 20-30% of PC1, even among tumours alone.
  Tumour vs normal appears on lower principal components (`eda_report.md` §1.4).
- **Outlier cluster.** An 8-tumour cluster from site 44 dominates PC1. It contains high-missingness samples, and the cause is not known.
- **Missingness differs by class and by site.**
- **Many events differ within patients.** In the training-only paired analysis, 5,144 of 25,176 tested events are at FDR < 0.05 (46 pairs). About 1,060 of these have |median ΔPSI| ≥ 0.1.
- **The strongest differences are large** (|median ΔPSI| up to 0.80) and come mostly from exon-skipping (ES), alternative-promoter (AP) and alternative-terminator (AT) events.
- **Clinical variables are weak correlates.** Purity, age, gender and stage relate only weakly to PC1/PC2 and to the top events (|ρ| ≤ 0.18). Purity is missing for 32% of tumours.
  Smoking, race and sequencing batch are not available in this download.

## 2. Model predictions

### 2.1 Cross-validation (training patients only; `cv_metrics_summary.csv`, `cv_robust_lowmissing.csv`)

| model (k = 25) | CV ROC-AUC, main (mean ± SD, 50 folds) | CV ROC-AUC, events ≤ 5% missing | CV PR-AUC (normal), main | CV PR-AUC (normal), ≤ 5% missing |
|---|---|---|---|---|
| LR-L2 | 0.998 ± 0.003 | 0.998 ± 0.002 | 0.978 ± 0.036 | 0.987 ± 0.022 |
| LR-L1 | 0.996 ± 0.003 | 0.997 ± 0.003 | 0.964 ± 0.038 | 0.980 ± 0.023 |
| RF | 0.997 ± 0.003 | 0.997 ± 0.003 | 0.972 ± 0.037 | 0.976 ± 0.033 |
| majority | 0.500 | 0.500 | 0.104 | 0.104 |

- **Robustness:** restricting to events with ≤ 5% missing (about 21,600 eligible per fold, vs about 25,600 in the main run) changes nothing material. The signal does not come from
  poorly covered events or from missingness patterns.
- **Shuffled-label control:** CV ROC-AUC 0.52-0.59, explained by chance overlap with the true labels (`model_report.md` §3).

### 2.2 Held-out test set, evaluated once (`test_metrics.csv`, `test_predictions.csv`)

Final model: LR-L2, k = 25 events, ranked on the 46 training pairs. C = 0.001 was chosen by inner CV, which is the smallest value in the grid (see §6). Threshold 0.5.

| analysis | n (normal) | ROC-AUC [95% patient-bootstrap CI] | PR-AUC normal [95% CI] | TN / FP / FN / TP | specificity | recall (tumour) |
|---|---|---|---|---|---|---|
| **Primary: LR-L2** | 113 (11) | **1.000 [1.000, 1.000]** | 1.000 [1.000, 1.000] | 11 / 0 / 1 / 101 | 1.00 | 0.99 |
| Secondary: RF (k = 25) | 113 (11) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 11 / 0 / 0 / 102 | 1.00 | 1.00 |
| Baseline: majority class | 113 (11) | 0.500 | 0.097 [0.047, 0.150] | 0 / 11 / 0 / 102 | 0.00 | 1.00 |
| Sensitivity: main LR-L2, test without TCGA_44_6146 | 112 (11) | 1.000 [1.000, 1.000] | 1.000 | 11 / 0 / 1 / 100 | 1.00 | 0.99 |
| Sensitivity: refit without the site-44 cluster, same 112 | 112 (11) | 1.000 [1.000, 1.000] | 1.000 | 11 / 0 / 1 / 100 | 1.00 | 0.99 |

- No bootstrap resample was skipped. Every one of the 2,000 resamples contained both classes and gave AUC = 1.000.
  The CI of [1.000, 1.000] means all test tumours scored above all test normals. It does **not** mean the true AUC is exactly 1.
  With only 11 test normals, a single future misranking would change the estimate noticeably.
- **The only error at 0.5 is `TCGA_78_7162`** (tumour, site 78, which has no normals; purity 0.29; P(tumour) = 0.499). It is still ranked above every normal:
  the highest normal scored 0.315. RF called it tumour (P = 0.914).
- **The test AUC is not below the CV range**; it sits at its upper edge. The sensitivity analyses give identical results.
- Figures: `results/figures/final_test_predicted_probabilities` (held-out probabilities by class, LR-L2 and RF) and
  `results/figures/final_cv_vs_test_auc` (CV fold AUCs next to the single test AUC).

### 2.3 Candidate shortlist (`candidate_shortlist.csv`; training data only)
Rule (frozen, decisions #21): selected in ≥ 80% of CV folds at k = 25 **and** among the final model's 25 features. **19 candidate events** qualify.
Median ΔPSI is tumour − normal on the 46 training pairs, with a 95% bootstrap CI. BH-FDR from the final ranking ranges from 4e-11 to 3e-6.

| rank | event | gene | type | direction | median ΔPSI [95% CI] | pairs | CV freq. |
|---|---|---|---|---|---|---|---|
| 1 | 26087 | SCEL | ES | higher in tumour | +0.80 [0.75, 0.83] | 40 | 1.00 |
| 2 | 60878 | SLC19A1 | AP | higher in tumour | +0.72 [0.61, 0.78] | 40 | 1.00 |
| 3 | 44845 | LAMA3 | AP | higher in tumour | +0.71 [0.41, 0.85] | 36 | 0.90 |
| 4 | 51169 | MYH14 | ES | higher in tumour | +0.69 [0.64, 0.75] | 43 | 1.00 |
| 5 | 53577 | SPTBN1 | AP | higher in tumour | +0.53 [0.49, 0.58] | 46 | 1.00 |
| 6 | 90356 | MTMR1 | ES | lower in tumour | −0.53 [−0.66, −0.46] | 38 | 1.00 |
| 7 | 64992 | HYAL1 | AP | lower in tumour | −0.51 [−0.58, −0.42] | 42 | 1.00 |
| 8 | 71227 | NEIL3 | AT | higher in tumour | +0.50 [0.41, 0.56] | 45 | 1.00 |
| 9 | 32281 | SH3GL3 | ES | higher in tumour | +0.50 [0.44, 0.53] | 46 | 1.00 |
| 10 | 28288 | NUMB | ES | higher in tumour | +0.50 [0.36, 0.55] | 46 | 1.00 |
| 12 | 89373 | KIF4A | AT | higher in tumour | +0.49 [0.38, 0.57] | 46 | 1.00 |
| 13 | 62421 | TEF | AP | higher in tumour | +0.48 [0.38, 0.58] | 41 | 0.90 |
| 15 | 9576 | FAM72A | AP | higher in tumour | +0.48 [0.37, 0.59] | 46 | 0.98 |
| 16 | 1493 | COL16A1 | ES | higher in tumour | +0.47 [0.36, 0.58] | 38 | 0.80 |
| 17 | 63511 | FBLN2 | ES | lower in tumour | −0.47 [−0.52, −0.44] | 45 | 1.00 |
| 20 | 87357 | TNC | ES | higher in tumour | +0.46 [0.35, 0.53] | 44 | 0.80 |
| 21 | 19519 | NTM | AP | higher in tumour | +0.46 [0.44, 0.61] | 41 | 0.98 |
| 22 | 18299 | SMCO4 | AP | higher in tumour | +0.46 [0.37, 0.52] | 44 | 0.92 |
| 24 | 50162 | CEACAM1 | ES | lower in tumour | −0.45 [−0.50, −0.41] | 43 | 0.92 |

- **Composition:** 15 events have higher PSI in tumour and 4 lower. By type: 9 ES, 8 AP, 2 AT.
- **Not shortlisted:** six of the final 25 features were selected in only 50-74% of CV folds: MELK ES, HIGD1B AP, FGF13 AP, STRA6 AD, ADRA1A AT and MACF1 ES.
- No gene functions are stated here, on purpose.
- Figure: `results/figures/final_candidate_shortlist_effects` (median ΔPSI with 95% CI, and CV selection frequency).

## 3. Interpretation (what the evidence supports)
- **H0 is rejected for this cohort.** Splicing measurements carry strong discriminatory information for tumour vs adjacent non-tumour lung, at the univariate level
  (thousands of events at FDR < 0.05 within patients) and the multivariate level (CV AUC about 0.998, test AUC 1.000).
- **The secondary hypothesis is supported within this cohort.** A small set of 25 events, chosen on training patients only, ranked all held-out samples correctly.
- **Leakage was checked repeatedly and not found:**
  - the shuffled-label control collapses toward chance;
  - all in-fold assertions passed;
  - the test set was opened once, under a lock, with a protocol frozen beforehand;
  - the result holds with stricter coverage, in the 7-site population, and without the site-44 cluster.
- **What remains unresolved:**
  - *Site:* the 7-site restriction argues against a site shortcut, but site and label are not fully separable in this design.
  - *Tissue composition:* adjacent normal lung and tumour differ in which cell types are present. These events could reflect cell mix as much as splicing changes within the same cells. Nothing here distinguishes the two.
  - *Generalisability:* a single cohort, one quantification pipeline and an old alignment. **This is not validation.** The events are **candidate events**.
- **Tumour vs adjacent normal tissue is a comparatively easy contrast.** Near-perfect separation here says nothing about harder, clinically relevant distinctions
  such as early detection, benign lesions, other cancer types, or liquid biopsy.

## 4. Speculation (unverified, for discussion only)
- Over half the shortlisted events are alternative promoter or terminator events (10 of 19). Some of the separation may therefore come from differences in transcript start/end usage
  or in which cell types are present, rather than from classical cassette-exon splicing. This is not tested.
- The very large, consistent ΔPSI values suggest that a few events with near on/off behaviour between tissues drive the classifier. That would fit with tissue-identity markers.
- Any gene-level biological context for the shortlisted genes still needs to be sourced and verified. None is asserted here.

## 5. Limitations and alternative explanations
1. **Single cohort, no external validation.** The SpliceSeq resource dates from 2016 (older TCGA alignment).
2. **Few normals.** There are 59 in total and 11 in the test set, so the test AUC CI is degenerate and gives little information about precision.
3. **Normals are adjacent tissue, not healthy lung.** "Normal-adjacent" may already differ from healthy tissue.
4. **Site confounding:** normals come from 7 sites. Mitigated by paired analysis, grouped CV and the 7-site sensitivity analysis, but not eliminated.
5. **Composition and purity:** purity is missing for 32% of tumours and correlates only weakly with the events. Cell-type composition was not measured.
6. **Unmeasured confounders:** smoking, batch/plate, RNA quality and ancestry are not available.
7. **Portal pre-filters** (≥ 75% of samples with PSI, expression filter) were applied across all samples before download. They ignore labels, but they cannot be undone.
8. **Effect-size measure:** the median ΔPSI understates shifts for events piled up at PSI = 1 (`eda_report.md` §5).
9. **LR-L2 chose C = 0.001, at the edge of the grid** (both final fits, and 42 of 50 main-config CV folds at k = 25). Its probabilities are compressed toward 0.5, which explains the single error at threshold 0.5
   despite perfect ranking. A wider grid might change calibration, but would hardly change ranking.

## 6. Surprises (flagged)
1. **The test AUC is 1.000 for every model and every sensitivity analysis.** It is at the top of the CV range, not below it. Given the effect sizes, this matches the CV result,
   but it should be read as evidence of an easy contrast in this cohort, not as proof of a biomarker.
2. **The patient-bootstrap CI is [1.000, 1.000].** A percentile bootstrap cannot show uncertainty when the observed separation is perfect. An exact or prior-based interval
   (for example, a Clopper-Pearson-style bound on the probability of misranking with 11 × 102 comparisons) would be more informative. That was not pre-specified, so it is not reported as a result.
3. **The one misclassified tumour sits exactly at the threshold (0.499)**, because the chosen C sits at the edge of the grid (§5.9).
4. **Earlier in the project:** an sklearn 1.9 change silently turned L1 into L2 (caught and fixed before Stage 1 results; `model_report.md` §3.3).

## 7. What remains unknown
- Whether the events separate tumour from normal in independent cohorts, with current alignments and other PSI quantification tools.
- Whether the signal is tumour-cell-intrinsic or reflects tissue composition (this would need single-cell or deconvolution data, or purity-matched comparisons).
- Whether any event separates harder contrasts (stage, subtype, other lung cancers, non-cancer lung disease).
- Biological roles of the shortlisted splice events (not investigated).

## 8. Reproducibility and audit
- **Environment:** `luad-splicing` (`environment.yml`), seed 42.
- **Pipeline order:**
  1. `src.build_inventory`
  2. `src.build_matrix`
  3. `src.splits`
  4. `src.qc_report`
  5. `src.run_eda`
  6. `src.run_models_cv`
  7. `src.run_robust_lowmissing`
  8. `src.final_test` (one-shot; locked)
  9. `src.build_shortlist`
  10. `src.report_figures` (reads saved outputs only; no refitting, no test re-evaluation)
- **Tests:** `python -m pytest -q`, 21 passing.
- **Files to audit for leakage:**
  - `src/final_test.py`: `fit_final` (training-only fitting) and the lock block in `main` (test rows are indexed only after `acquire_lock`);
  - `src/models.py`: `run_fold` and `fold_indices`;
  - `src/splits.py`: `make_splits` and `check_splits`;
  - `tests/test_final.py`, `tests/test_models.py`, `tests/test_phase2.py`.
- **Test-set access:** exactly once, recorded in `results/test_lock.json`. A second invocation was confirmed to exit with "REFUSING TO RUN".
