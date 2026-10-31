# Phase 3 report: EDA and statistics (no models)

Produced by `python -m src.run_eda` (env `luad-splicing`, seed 42), on 2026-10-29. The numbers below are copied from the tables in
`results/tables/`, and the figures are in `results/figures/`. The parameters were fixed before running (`docs/decisions.md` #10-14, `docs/analysis_plan.md` §C).

Labelling convention used throughout:
- **(a) "descriptive, full cohort"** uses all 57 pairs or all samples. It may NOT feed feature ranking.
- **(b) "training pairs only"** uses the 46 pairs whose patients are in the training set. This is the only analysis allowed to feed later ranking.
  The patient list is saved in `stats_paired_train_pairs_used.csv`, and the code and tests assert that it contains no test patient.

---

## 1. Observations

### 1.1 Cohort (`eda_cohort_by_set.csv`, `eda_cohort_site_by_set.csv`, `eda_clinical_by_set.csv`)

| set | patients | samples | tumour | normal | pairs | samples from the 7 normal sites |
|---|---|---|---|---|---|---|
| train | 414 | 460 | 412 | 48 | 46 | 236 |
| test | 102 | 113 | 102 | 11 | 11 | 57 |
| all | 516 | 573 | 514 | 59 | 57 | 293 |

Train and test tumours have similar clinical profiles:

| | median age (IQR) | female | median purity (n available) | stage I / II / III / IV |
|---|---|---|---|---|
| train | 66 (59-72) | 53.9% | 0.44 (280) | 225 / 95 / 68 / 19 |
| test | 67 (60-74) | 52.9% | 0.44 (71) | 49 / 27 / 16 / 7 |

### 1.2 Missingness (`eda_missingness_*`; figures `eda_missingness_per_sample_by_class_and_site`, `eda_missingness_per_event_tumour_vs_normal`)

- Per sample:
  - The median sample misses 1.9% of events. Normals miss slightly more on average (median 3.3% vs 1.9%; Kruskal-Wallis p = 0.004).
  - Missingness differs strongly by site (Kruskal-Wallis p = 3e-8 over all samples; p = 2e-6 within tumours only).
  - Most high-missingness tumours come from sites that also contribute normals (55, 44, 50, 49, 91).
- Per event:
  - Within tumours, missingness is spread fairly evenly (75th percentile 3.9%, maximum 28%).
  - Within normals it is all-or-nothing: 75% of events are complete in every normal, but a minority are mostly missing (maximum 96.6%).
    The pattern of which values are missing is itself class-dependent.
- PSI distribution (`eda_psi_distribution_by_class`): values pile up at exactly 1 (about 30% of all values), with a smaller pile at 0, in both classes.

### 1.3 Random example events (one per splice type, drawn before testing; `eda_example_events_random.csv`, figure `eda_example_events_random`)

Of the 7 random events, 5 have a median PSI of 1.0 in both classes, with 64-100% of values exactly 1 (RIN1 AA, COMMD10 AD,
GTPBP3 AT, MBD1 ES, and LRRC69 ME close to it). The GTPBP3 AT event is 1.0 in every sample. The other two (GNAI2 AP and EFEMP2 RI) are continuous,
centred around 0.77-0.90, and slightly lower and more spread in tumours (tumour SD about 2-3× the normal SD).
LRRC69 ME is 25% missing in tumours but only 8% in normals, and has a zero pile in tumours (15%) that is nearly absent in normals (2%).
**Typical event:** close to the ceiling with little variance. Informative events are the minority.

### 1.4 PCA (visualization only; `eda_pca_r2.csv`, figures `eda_pca_*`)

How NaNs were handled:
- **Only events with no missing value in any sample were used**, minus constant events: 10,041 events. Nothing was imputed.
- Values were centred but not scaled.
- This uses the whole cohort and never feeds a model.

How much of each PC is explained by class (R² label) and by site (R² site):

| PCA | PC1 (var) | R² label | R² site | PC2 (var) | R² label | R² site | PC3 (var) | R² label | R² site |
|---|---|---|---|---|---|---|---|---|---|
| all 573 samples | 20.9% | **0.02** | 0.20 | 6.8% | 0.16 | 0.24 | 5.9% | **0.27** | 0.12 |
| tumours only (514) | 21.8% | n/a | **0.23** | 7.0% | n/a | 0.24 | 4.6% | n/a | 0.25 |
| 7-site population (293) | 26.7% | 0.03 | 0.13 | 7.4% | **0.49** | 0.12 | 6.2% | 0.16 | 0.08 |

- **PC1 is not tumour vs normal.** Its largest single feature is a tight cluster of **8 tumours, all from site 44**, sitting far from everything else.
  Those 8 are `TCGA_44_2656, _2662, _2666, _2668, _3917, _4112, _6146, _6147`. Seven are in training and `TCGA_44_6146` is in the test set. Four are paired tumours (2662, 2668, 6146, 6147).
  Several are among the highest-missingness samples (6147: 27%, 2662: 22%). Their matched normals are not in the cluster.
  Even with these 8 removed, site still explains 30% of PC1's variance (computed post hoc, not saved as a table).
- Tumour vs normal separates along PC2/PC3 (all samples) and PC2 (7-site population, R² 0.49).
  In the PC1-PC3 projection, normals sit at the low-PC3 edge of the tumour cloud and overlap it partly. They are not a separate cluster.
- **Tumours only:** site explains 23-25% of the variance of each of PC1-PC3. Site is a major source of variation even without any normals present.
- Robustness (`eda_pca_robustness.csv`): repeating the PCA on 23,926 events (≤ 10% missing, median-imputed) reproduces PC1 (|r| = 0.99).
  PC2 and PC3 only partly agree (|r| = 0.67 and 0.71), and in that version the label lands on PC2 (R² 0.51). The ordering of the minor components depends on the NaN handling.
- Purity (`eda_pca_all_by_purity`): no visible purity gradient along PC1 or PC2 (see §1.6).

### 1.5 Missing data: what was dropped (`stats_dropped_events_summary.csv`)

| analysis | events | dropped: < 30 pairs (paired) / < 10 per group (unpaired) | untestable: all differences 0 / no variation | tested |
|---|---|---|---|---|
| (a) paired, 57 pairs | 27,836 | 748 | 798 | 26,290 |
| (b) paired, 46 training pairs | 27,836 | 1,780 | 880 | 25,176 |
| unpaired, all samples | 27,836 | 45 | 332 | 27,459 |
| unpaired, 7 sites | 27,836 | 45 | 460 | 27,331 |

### 1.6 Confounders (tumours only; `eda_confounders.csv`, `eda_purity_direction_top10.csv`, figures `eda_confounders_rho_heatmap`, `eda_purity_vs_pc1_tumours`)

BH correction was applied across the 70 confounder tests (14 targets × 5 confounders).

| target | purity ρ (n = 351) | age ρ (n = 495) | gender | stage I-IV ε² | site ε² |
|---|---|---|---|---|---|
| PC1 (all) | −0.12 (FDR 0.06) | 0.05 (ns) | ns | 0.03 (FDR 0.004) | **0.24 (FDR 5e-16)** |
| PC2 (all) | 0.18 (FDR 0.004) | −0.14 (FDR 0.006) | ns | ns | **0.16 (FDR 1e-9)** |
| PC1 (tumours only) | 0.12 (FDR 0.07) | ns | ns | 0.03 (FDR 0.003) | **0.24 (FDR 5e-16)** |

- Site is by far the strongest correlate of the main axes of variation. Purity, age and stage are weak (|ρ| ≤ 0.18, ε² ≤ 0.03). Gender shows nothing.
- Top-10 events (from (a)):
  - Only 1 of 10 correlates with purity at FDR < 0.05 (LAMA3 AP, ρ = −0.17). Its direction is *opposite* to its tumour − normal shift.
  - 6 of 10 move with purity in the same direction as tumour − normal, but all |ρ| ≤ 0.17.
- Smoking history, race and sequencing batch/plate are not in this download, so they cannot be checked.
- Purity is missing for 32% of tumours. Missingness was not tested for association with site.

---

## 2. Statistical results (`stats_summary.csv`, `stats_agreement.csv`; figure `eda_volcano_paired_and_unpaired`)

Tests used:
- Paired: Wilcoxon signed-rank (`zero_method="wilcox"`, non-parametric; PSI is bounded and piles up at 0/1, so normality cannot be assumed).
- Unpaired: Mann-Whitney U.
- Effect size: median ΔPSI (tumour − normal) with a 95% bootstrap CI (2,000 resamples). The mean ΔPSI is given too.
- BH-FDR within each analysis.

| analysis | tested | FDR < 0.05 | of which \|median ΔPSI\| ≥ 0.1 | significant but median ΔPSI exactly 0 | higher / lower in tumour (by mean) |
|---|---|---|---|---|---|
| **(a) paired, 57 pairs, descriptive** | 26,290 | **6,101 (23%)** | 1,095 | 375 | 2,752 / 3,349 |
| **(b) paired, 46 training pairs** | 25,176 | **5,144 (20%)** | 1,062 | 218 | 2,287 / 2,857 |
| unpaired, all samples, descriptive | 27,459 | 11,292 (41%) | 1,522 | 1,256 | 5,280 / 6,012 |
| unpaired, 7 sites (sensitivity) | 27,331 | 10,175 (37%) | 1,447 | 1,094 | 4,739 / 5,436 |

Agreement between analyses (events testable in both):

| comparison | Spearman ρ of median ΔPSI | Spearman ρ of mean ΔPSI | both FDR < 0.05 | Jaccard | same direction (both sig.) |
|---|---|---|---|---|---|
| (a) vs (b) | 0.93 | 0.97 | 4,913 | 0.79 | 99.7% |
| (a) vs unpaired full | 0.73 | 0.85 | 5,594 | 0.48 | 98.9% |
| unpaired full vs 7-site | 0.96 | 0.98 | 9,830 | 0.84 | 99.1% |
| (a) vs unpaired 7-site | 0.76 | 0.89 | 5,595 | 0.53 | 99.3% |

- Of the top-50 events in (b), 44 are also in the top-50 of (a).
- **Top events in (a)** (`eda_top50_paired_full_descriptive.csv`; heatmap `eda_heatmap_top50_paired_full`):

  | event | type | median ΔPSI | 95% CI | pairs | FDR |
  |---|---|---|---|---|---|
  | SCEL 26087 | ES | +0.79 | 0.75-0.83 | 50 | 5e-11 |
  | LAMA3 44845 | AP | +0.77 | 0.45-0.86 | 44 | 3e-6 |
  | MYH14 51169 | ES | +0.71 | 0.64-0.76 | 54 | 4e-8 |
  | SLC19A1 60878 | AP | +0.70 | 0.64-0.77 | 50 | 2e-10 |
  | SPTBN1 53577 | AP | +0.53 | 0.49-0.57 | 57 | 3e-8 |

  The heatmap shows these top events cleanly separating tumours from normals, including tumours from sites that contributed no normals.
- **Splice types:** the top-50 are mostly alternative promoter (AP, 27) and alternative terminator (AT, 6), with 15 exon-skipping (ES).
  The fraction of tested events reaching significance is highest for AT (39%), AP (33%) and ME (31%), and lowest for AA (15%).
- **Volcano ceiling:** in the paired volcano, p-values cannot go below roughly 1e-10 to 1e-14. With 30-57 pairs the smallest achievable Wilcoxon p is limited,
  so many events stack at the ceiling. Ranking by p therefore does not separate the strongest events. Effect size does.

---

## 3. Interpretation (what the observations support)

- **H0 (no tumour-vs-normal signal) is not supported, at the univariate level.** About a fifth of testable events differ within patients at FDR < 0.05.
  About 1,100 of them have a median within-patient shift of ≥ 0.1 PSI. The training-pairs-only result (b) closely matches (a), so the signal does not depend on
  the 11 held-out pairs. This is an association in a single cohort. It is not a classifier result and not validation.
- **The within-patient (paired) design matters.** The unpaired test flags nearly twice as many events (11,292 vs 6,101), and its effects agree less well with the paired ones
  (ρ 0.73). Restricting the unpaired comparison to the 7 normal sites changes it little (ρ 0.96 with the full unpaired). So the extra unpaired hits are not explained only by
  the 26 tumour-only sites. More likely they come from larger n (514 tumours) and from between-patient variation. The paired analysis remains the most site-robust evidence.
- **Site is the dominant source of global variation** (PC1, R² 0.20-0.30, also within tumours). Tumour status shows up on lower components.
  A classifier could therefore pick up site structure, especially because normals come from only 7 sites. The 7-site sensitivity analysis and grouped CV
  are needed in Phase 4 precisely because of this.
- **Purity, age, gender and stage are weak correlates** of the main axes and of the top events in this cohort. Purity is missing for 32% of tumours, so it is only partly controlled.

## 4. Speculation (unverified; for discussion only)

- The over-representation of AP and AT events among the strongest differences suggests that some of the "splicing" signal may reflect a shift in
  **transcript start/end usage, or in which cell types are present** (tumour epithelium vs the cell mix of normal lung), rather than splicing changes within the same cells.
  The purity analysis cannot tell these apart: a tumour-cell-intrinsic change and simple dilution by non-tumour cells both predict PSI moving with purity in the tumour − normal direction.
- The site-44 cluster could be a batch or library-quality artefact (high missingness, tight clustering). This is not verified; SpliceSeq does not give plate or batch.
- No gene functions are claimed here. Any biological context for the top genes is deferred to Phase 5 and will be marked as unverified unless sourced.

---

## 5. Surprising or uncertain findings (flagged for the project owner)

1. **An 8-tumour site-44 outlier cluster** dominates PC1. Four of the 8 are paired tumours, and one (`TCGA_44_6146`) is in the held-out test set.
   I have **not** excluded anything. Excluding them, or adding a sensitivity analysis without them, is a decision for you. If made, it should be pre-specified now, before Phase 4.
2. **The effect sizes are very large** for the top events (median ΔPSI up to 0.79, with tight CIs). This is not an artefact of the pipeline, since the paired values come straight from the matrix.
   It is still unusual for splicing proper. See the AP/AT point in §4.
3. **The median ΔPSI is exactly 0 for 375 significant events in (a), and for 1,256 in the unpaired analysis.** These come from events piled up at PSI = 1.
   The Wilcoxon test (which drops zero differences) and the Mann-Whitney test (which uses ranks) can still detect a shift in the minority of non-ceiling values. For these events,
   the frozen effect-size measure (the median) understates the effect. The mean ΔPSI is in the tables.
4. **Missingness depends on class and on site** (§1.2). This supports fitting coverage filters and imputation inside training folds only. Missingness itself may carry label information.
5. **The paired 7-site sensitivity analysis is identical to (a)**, since all 57 pairs are within those sites. Only the unpaired analysis and the PCA were re-run on the 7-site population.
6. **The minor PCA components are not stable** to the NaN-handling choice (PC2/PC3 |r| ≈ 0.7). Only PC1 is robust.

## 6. Reproducibility

- Code: `src/run_eda.py`, `src/eda_stats.py`, `src/plotting.py`. Tests: `tests/test_eda_stats.py`. All 13 tests pass.
- The heatmap's axis labels were fixed after the main run and the figure re-rendered from the saved statistics. A full rerun gives the same output.
- No model was trained. The test set was not used for anything except being labelled in figures and being excluded from (b).
