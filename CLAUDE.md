# CLAUDE.md: LUAD RNA-Splicing Biomarker Project (Computational Analysis)

## Project goal
Test whether alternative RNA-splicing events in public lung adenocarcinoma (LUAD) transcriptomic data can distinguish tumour samples from non-tumour lung samples, and identify which events show the strongest, most reproducible signal.

This is a **candidate-biomarker discovery** project. It is NOT diagnosis, treatment recommendation or a cancer cure. Never write results in a way that implies clinical usefulness.

## Research question
Can alternative RNA-splicing events in public LUAD data distinguish tumour from non-tumour lung samples, and which events show the strongest, most reproducible signal?

## Hypotheses
- **H0:** The selected splicing features carry no meaningful predictive information for tumour vs non-tumour in the chosen dataset.
- **H1:** At least some splicing features differ between groups and carry measurable discriminatory information.
- **Secondary:** A small subset of events keeps useful performance on held-out samples, not only on the samples used to select them.

## Frozen analysis plan (do not change after seeing results)
- **Primary outcome:** tumour (1) vs non-tumour (0) classification.
- **Main comparison:** tumour vs non-tumour splicing measurements per event.
- **Primary metric:** ROC-AUC on the held-out test set.
- **Secondary metrics:** confusion matrix, precision, recall, F1, PR-AUC (important if classes are imbalanced).
- **Baselines:** majority-class predictor, then L2/L1 logistic regression.
- **Comparison model (optional):** random forest, only if sample size and feature count support it.
- If any of the above must change, STOP and tell me. Log the change and the reason in `docs/decisions.md`. Do not silently adjust the plan.

## Data (to be confirmed in Phase 1, do not assume)
- Candidate source: TCGA-LUAD, or another documented cohort with RNA-seq and tumour/non-tumour labels.
- Prefer **precomputed event-level splicing measurements**. Do not process raw reads unless I ask.
- Do NOT invent dataset URLs, file names, column names or sample counts. If you cannot find or verify something, say so and ask.
- Record in `docs/data_inventory.md`: source, access date, access terms, sample counts per class, inclusion/exclusion criteria, how tumour and non-tumour are defined, and any matched tumour/normal pairs.
- Raw data is read-only: never edit files in `data/raw/`. Write derived data to `data/processed/`.

### Source and download settings (record these in docs/data_inventory.md)
- Source: TCGA SpliceSeq (MD Anderson), PSI Data Download page: https://bioinformatics.mdanderson.org/TCGASpliceSeq/PSIdownload.jsp
- Downloaded 29 Oct 2026 as `PSI_download_LUAD.zip` (single txt file, ~94.9 MB), tumour type LUAD.
- Settings: all genes, all samples, all 7 event types (AA, AD, ES, RI, AP, AT, ME), Percentage of Samples with PSI Value = 75, Minimum Average Expression Percentage = 50, Minimum PSI Range = 0, Minimum PSI Standard Deviation = 0, Gene RPKM Matrix = off, Gene Descriptions = on, Clinical Data = on.
- Note: this resource dates from 2016 (older TCGA alignment). Mention as a limitation in the report.

### File structure (PSI_download_LUAD.txt), confirmed in the Phase 1 inventory (see docs/data_inventory.md)
- Tab-delimited, CRLF line endings. The header and the clinical rows end with a trailing tab (an empty extra field, dropped on load).
- Columns 1-10 are annotation: `symbol, as_id, splice_type, exons, from_exon, to_exon, novel_splice, pct_with_values, psi_range, std_psi`. These are NOT features. `pct_with_values`, `psi_range` and `std_psi` are computed across ALL samples, so using them to filter events before the train/test split is leakage. Recompute any variability filter inside training data only.
- Sample columns follow, named like `TCGA_38_4625` (tumour) and `TCGA_38_4625_Norm` (matched adjacent normal). **A column ending `_Norm` is a normal sample; all others are tumour. The sample label comes ONLY from this suffix.**
- After the sample columns come two more per-event columns: `avg_exp_pct` and `annotation` (free-text gene description). These are NOT features. `avg_exp_pct` is also computed across ALL samples, so the same leakage rule applies.
- Patient ID = column name with `_Norm` removed (e.g. `TCGA_38_4625`). Use patient ID as the group key for every split. A tumour column and its `_Norm` column MUST end up in the same fold/split.
- Some `_Norm` columns have no tumour column for the same patient (e.g. `TCGA_44_5645_Norm`, `TCGA_44_6144_Norm` in the header). Count these, keep them as unpaired normals, and exclude them from the paired sensitivity analysis.
- PSI values are between 0 and 1. Missing (unquantified) PSI is coded ONLY as the string `null` (no blanks occur in PSI rows), which is recoded to NaN. **An exact PSI of 0 is a real value (event fully skipped) and must be kept, never recoded to NaN.**
- Clinical rows are at the TOP of the file (lines 2-14, directly under the header, before the PSI rows), one value per sample column. They are identified by content: `splice_type` and `as_id` are blank. The 13 row labels, verbatim: `Gender`, `Age`, `Height_cm`, `Weight_kg`, `Tumor_Status`, `Vital_Status`, `Pathologic_Tumor_Stage`, `Pathologic_N`, `Pathologic_T`, `Absolute_Purity_Value`, `Tissue_Specific_Attribute_1`, `Tissue_Specific_Attribute_2`, `Tissue_Specific_Attribute_3`. They are split into a separate clinical table (`src/load_data.py`).
- **Clinical values are blank for `_Norm` columns. NEVER use clinical columns as model features** (blank vs filled would leak the label). For EDA, copy a patient's clinical values onto their matched normal by patient ID.
- Missing CLINICAL values appear in three forms: blank (every `_Norm` column), the string `null`, and `0` (Height_cm and Weight_kg are all 0; Age has 19 zeros; Absolute_Purity_Value has 163 zeros). Recode all of them to NaN. **The 0 -> NaN rule applies to clinical fields only, never to PSI.** Height, weight and Tissue_Specific_Attribute_1-3 are entirely missing, so they are dropped.
- `Tumor_Status` (TUMOR FREE / WITH TUMOR) is the patient's follow-up disease status, NOT the sample label. Do not use it as the label.

### Confounders to check explicitly
- **Tissue source site** = 2nd part of the ID (e.g. 05, 35, 38, 44, 49). Normals appear concentrated in some sites. Tabulate tumour/normal counts per site, include site in EDA and PCA colouring, and report whether site could explain tumour-vs-normal separation. The matched-pair analysis is the main control for this.
- **Tumour purity** (`Absolute_Purity_Value`) and clinical stage: check and report.
- Unpaired normals vs paired normals: check if they differ in site or PSI distribution.

## Pipeline order (do not skip ahead)
1. Data feasibility and inventory
2. Sample-by-event matrix (rows = samples, columns = events) + QC report
3. EDA and statistical testing
4. Baseline and candidate models
5. Candidate event ranking, stability, gene mapping
6. Report-ready tables and figures

**Do not train any classifier until steps 1-3 are done and I have confirmed the data and labels look trustworthy.**

## Data leakage rules (critical)
- Create the train/test split **before** any data-dependent step.
- Filtering by variance/coverage, imputation, scaling, feature selection and hyperparameter tuning must be fitted on **training data only** (use sklearn `Pipeline` or the R equivalent so this happens inside each CV fold).
- **Keep related samples together:** if a patient has both tumour and matched normal samples, all samples from that patient must land in the same split. Use grouped splits (e.g. `GroupKFold`, `StratifiedGroupKFold`) keyed on patient ID.
- The test set is used **once**, at the end, for the final reported numbers. Never use it to choose features, tune thresholds or compare designs.
- Never select the top events on the full dataset and then evaluate a model on that same dataset.
- Use a fixed random seed (`SEED = 42`) everywhere and record it.
- Statistical testing across the full cohort for EDA is allowed, but results from it must NOT feed into model feature selection. Model feature selection happens inside training folds only.

## Statistics rules
- Report effect sizes (e.g. difference in mean PSI, or the equivalent measure for the chosen quantification) with confidence intervals, not only p-values.
- Apply multiple-testing correction (Benjamini-Hochberg FDR) across all tested events. Report adjusted p-values.
- If matched pairs exist, run a paired analysis as a sensitivity check alongside the unpaired one.
- State which test was used and why. Check assumptions or use a non-parametric test.
- Check confounders (age, sex, smoking, batch, sample source, tumour purity) where metadata is available, and report where it is not.

## Modelling rules
- Evaluate all models on **identical splits**.
- Report mean and standard deviation across CV folds, plus the final held-out test result.
- Report baselines next to every model result.
- Stability: record which events are selected in each CV fold and report selection frequency.
- If class imbalance is large, use stratified splits, report PR-AUC, and consider class weights. Do not just report accuracy.
- If performance looks surprisingly high (e.g. AUC above 0.98), assume leakage first and audit the pipeline before reporting it.

## Interpretation and honesty rules
- Never describe an event as a "validated biomarker". Use "candidate event".
- A small p-value or a high model ranking is not validation.
- Do not claim generalization unless an independent cohort was actually tested.
- Keep four things separate in every write-up: **observations**, **model predictions**, **biological interpretation**, **speculation**.
- Negative or weak results are valid results. Report them plainly. Do not tune or re-slice until something looks significant.
- Do not make up literature citations or gene functions. If asked for biological context, mark anything unverified as such.

## Repository layout
```
data/raw/          # read-only originals
data/processed/    # derived matrices
docs/              # data_inventory.md, decisions.md, analysis_plan.md
src/               # reusable code (loading, qc, stats, models)
notebooks/         # exploratory only, no final results live here only
results/figures/   # plots
results/tables/    # stats and metrics tables (CSV)
reports/           # report drafts
environment.yml or requirements.txt   # pinned dependencies
```

## Engineering conventions
- Language: Python (pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib/seaborn). Use R only if I say so.
- Every result must be reproducible by running a script from the repo root. Put logic in `src/`, not only in notebooks.
- Save every figure and table to `results/` with a descriptive file name. Do not only display them.
- Write short docstrings and comments explaining *why*, especially in the split and feature-selection code, so I can audit it.
- Add simple tests or assertions for: no patient ID appears in both train and test, no NaNs entering models unexpectedly, expected matrix shape and sample counts.
- Pin package versions and log them.

## How to work with me
- Before starting each phase, state in 2-3 lines what you will do and what the outputs will be.
- After each phase, summarize: what was produced, key numbers (sample counts, number of events kept or dropped and why), and anything surprising or uncertain.
- Flag assumptions explicitly. If something is ambiguous (label definitions, which measurement to use), ask instead of guessing.
- Do not delete data or overwrite results without asking.
- Point me to the exact files and lines I should review for leakage risk.

## Milestones (done when)
1. **Data feasibility:** documented dataset, criteria, labels, sample counts
2. **Clean matrix:** reproducible sample-by-event table with QC and missingness documented
3. **EDA:** distributions, plots, statistical tests with multiple-testing correction
4. **Baseline model:** leakage-aware evaluation of baseline and logistic regression
5. **Shortlist:** small set of candidate events backed by effect sizes and stability checks
6. **Report:** biological context, limitations, alternative explanations, what remains unknown
