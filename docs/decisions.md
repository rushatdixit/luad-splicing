# Decisions log

## 2026-10-29: Phase 1 (data inventory)

1. **Correction of fact, not a plan change.** In `PSI_download_LUAD.txt` the 13 clinical rows sit at the TOP of the file
   (lines 2-14, directly under the header), not appended below the PSI rows as CLAUDE.md states. There are also two
   trailing per-event columns (`avg_exp_pct`, `annotation`) that CLAUDE.md does not list. The loader detects clinical rows by
   content (blank `splice_type` and `as_id`). CLAUDE.md should be updated by the project owner.
2. **Processed output layout (owner's choice).** Three files instead of two: the PSI matrix (`psi_events_x_samples.csv.gz`,
   events × samples, numeric only), the event annotation (`event_annotation.csv`, which holds all-sample statistics that must not be used
   before the split), and the clinical table (`clinical_by_sample.csv`, one row per sample, matched normals get their patient's values
   copied with a flag).
3. **Environment (owner's choice).** Phase 1 runs in the existing conda env `data_science`, with pandas/numpy pinned in
   `requirements.txt`. The full pinned environment is deferred until before Phase 3/4.
4. **Missing-value coding.** PSI: only `null` → NaN; exact 0 is kept as a real PSI value. Clinical: blank/`null` → NaN, and
   `0`/`0.0` → NaN for Age and Absolute_Purity_Value. Height_cm, Weight_kg and Tissue_Specific_Attribute_1-3 are dropped (100% missing).
   `NX`/`TX` are kept as categories.

## 2026-10-29: Phase 2 (matrix, frozen plan, splits)

5. **Analysis plan frozen** in `docs/analysis_plan.md` before any EDA. Owner decisions:
   - primary population = all 573 samples;
   - pre-specified sensitivity population = the 7 normal-contributing sites;
   - features = PSI only;
   - filters and imputation are fitted inside training folds only.
6. **Change vs CLAUDE.md (owner's decision):** the **paired** Wilcoxon test on the 57 pairs (+ BH-FDR) is now the **main** statistical
   analysis, and the unpaired comparison is secondary. CLAUDE.md "Statistics rules" (and one bullet in "File structure")
   still describe the paired analysis as a *sensitivity check*. Reason: site confounding (normals come from only 7 of 33 sites),
   which the within-patient comparison controls for. This decision was made before any PSI-dependent results were seen.
7. **Split parameters (owner's choice):**
   - held-out test = 20% of patients (one of 5 stratified patient groups);
   - CV = 5 folds × 10 repeats on the training patients;
   - SEED = 42.
   The splits were generated from metadata only, before EDA, and saved to `data/processed/splits.csv`.
8. **Environment (owner's choice):** dedicated conda env `luad-splicing`, exported to `environment.yml`.
   This supersedes decision 3 (deferring the env setup).
9. **Analysis-ready matrix:** samples × events with no event filtering, imputation or scaling (all 27,836 events kept).
   Metadata is stored separately in `sample_metadata.csv`. `normal_site` is derived from labels and is therefore never a feature.

## 2026-10-29: Phase 3 parameters (set by the owner BEFORE any test was run)

10. **Statistics details confirmed with changes vs the §C proposal** (no PSI test results had been seen):
    - minimum **30** complete pairs per event (proposal: 10);
    - primary effect size = **median** paired ΔPSI with a 95% bootstrap CI (proposal: the mean; the mean is kept as an extra column).
    Counted from missingness alone, the 30-pair rule drops 748 events (all 57 pairs) and 1,780 events (46 training pairs).
11. **Two versions of the paired analysis:** (a) all 57 pairs, "descriptive, full cohort"; (b) the 46 training-patient pairs only.
    **Only (b) may feed later feature ranking.**
12. **Sensitivity population (7 normal-contributing sites):** all 57 pairs already lie within these sites, so the paired 7-site
    analysis is identical to (a). The restriction changes only the unpaired analysis and the PCA.
13. **EDA display choices (made before running):**
    - "top events" = FDR < 0.05 ranked by |median ΔPSI| from (a), for figures only (top 50 heatmap, top 10 confounder checks);
    - PCA uses events complete in all samples, centred and not scaled, for visualization only;
    - example events = one random event per splice type (seed 42).
14. **Implementation detail added before running:** the unpaired Mann-Whitney test requires ≥ 10 non-missing values in EACH class.
    Some events are > 90% missing in normals, so without this rule a "test" could rest on 2-5 normal values.
    Events that fail are reported as untestable, not dropped silently. Events with no variation are also flagged untestable.

## 2026-10-29: Phase 4 Stage 1 (set by the owner BEFORE any model was fit)

15. **Site-44 cluster** (8 tumours: TCGA_44_2656, _2662, _2666, _2668, _3917, _4112, _6146, _6147): kept in the main analysis.
    The pre-specified sensitivity run excludes these 8 tumours from train and test. Their matched normals stay in as unpaired normals.
16. **In-fold feature ranking.** Ranking is redone inside every CV fold from that fold's training patients only:
    - paired Wilcoxon on the fold's training pairs, then keep events with in-fold BH-FDR < 0.05;
    - rank by |median ΔPSI| (tie-break |mean ΔPSI|), not by p;
    - if fewer than k events pass, top up by |median ΔPSI| from the rest, and log it.
    The Phase 3 (b) ranking is NOT used in CV. It may only feed the final model trained on all training patients (Stage 2).
17. **In-fold preprocessing:**
    - coverage filter from the training fold: ≤ 20% missing across the fold's training samples AND ≥ 25 complete training pairs;
    - median imputation and standard scaling fitted on the training fold (sklearn Pipeline, so they are also refit inside inner CV);
    - no missingness indicators, no clinical features, no site.
18. **Models and evaluation:**
    - majority class; L1 and L2 logistic regression (`class_weight="balanced"`, C ∈ 10^-3..10^2 tuned by 3-fold patient-grouped inner CV);
      random forest (500 trees, sqrt features, balanced, untuned); each with k = 10, 25, 50, 100.
    - Classification threshold 0.5 on P(tumour), never tuned.
    - PR-AUC is reported with both tumour and normal as the positive class.
    - Leakage audit: within each training fold, labels are permuted (class counts kept), "normals" are randomly pseudo-paired with "tumours",
      and the identical pipeline runs. Validation labels stay real.
    - Sensitivity runs: 7 normal-contributing sites only, and without the site-44 cluster.

## 2026-10-29: Robustness check and FINAL TEST PROTOCOL (frozen before any test-set access)

19. **Robustness CV (done; CV only):** main config, k = 25, in-fold coverage filter tightened to ≤ 5% missing (≥ 25 pairs).
    ROC-AUC was unchanged (LR-L2 0.998 vs 0.998 in the main run). Saved to `results/tables/cv_robust_lowmissing.csv`. This changes nothing in the protocol below.
20. **Final test protocol (frozen once written; set by the owner):**
    - **Final model:** LR-L2 with balanced class weights, k = 25, threshold 0.5 on P(tumour). C is tuned by 3-fold patient-grouped inner CV on the training
      set (grid 10^-3..10^2, as in Stage 1).
    - **Ranking:** the Phase 3 (b) rule on all 46 training pairs:
      - ≤ 20% missing across the 460 training samples AND ≥ 30 complete pairs;
      - paired Wilcoxon, BH-FDR within the eligible events, keep FDR < 0.05;
      - rank by |median ΔPSI| (tie-break |mean ΔPSI|).
      Median ΔPSI 95% bootstrap CIs use 2,000 resamples, seed 42.
    - Coverage filter, median imputation and scaling are fitted on the 460 training samples only.
    - **Primary result:** ROC-AUC on the 113 test samples with a 95% CI from a patient-level bootstrap (102 test patients resampled with all their samples,
      2,000 resamples, seed 42; resamples without a normal are skipped and counted).
    - **Secondary results, all reported whatever the outcome:**
      - PR-AUC (normal as positive), with the same bootstrap CI;
      - confusion matrix at 0.5;
      - majority-class baseline;
      - one pre-declared secondary model, RF (k = 25, same features, 500 trees, balanced, seed 42).
    - **Sensitivity:** refit ranking and model without the 7 site-44 cluster tumours in training (their normals stay); evaluate on the 112 test samples
      without TCGA_44_6146. The main model scored on the same 112 samples is reported alongside.
    - The test set is opened exactly once by `src/final_test.py`, which creates `results/test_lock.json` and refuses to run again.
      Nothing is changed after seeing test results.
21. **Candidate shortlist rule:** events selected in ≥ 80% of the main-config CV folds at k = 25 AND in the final model's top-25 features.
    Columns are statistical only; no gene functions or biological claims.
