# luad-splicing

Candidate-biomarker discovery: can alternative RNA-splicing events in public lung adenocarcinoma (LUAD) data
distinguish tumour from adjacent non-tumour lung samples, and which events give the strongest, most reproducible signal?

> **Scope.** This is an exploratory research analysis of one public cohort. It is **not** a diagnostic test, a
> treatment recommendation or a validated biomarker. The events reported here are **candidate events**. No independent
> cohort has been tested.

## Data

| | |
|---|---|
| Source | [TCGA SpliceSeq](https://bioinformatics.mdanderson.org/TCGASpliceSeq/PSIdownload.jsp) (MD Anderson), PSI Data Download, tumour type LUAD |
| Measurement | Percent-spliced-in (PSI) per event, 7 event types (AA, AD, ES, RI, AP, AT, ME) |
| Samples | 573 (514 tumour, 59 adjacent non-tumour) from 516 patients; 57 matched tumour/normal pairs |
| Events | 27,836 |
| Label | from the sample column name only: `_Norm` suffix = non-tumour, otherwise tumour |

**The data is not included in this repository.** To reproduce, download it from the SpliceSeq PSI download page with
the settings listed in [`docs/data_inventory.md`](docs/data_inventory.md) (all genes, all samples, all 7 event types,
Percentage of Samples with PSI Value = 75, Minimum Average Expression Percentage = 50, Minimum PSI Range = 0,
Minimum PSI Standard Deviation = 0, Gene RPKM Matrix off, Gene Descriptions on, Clinical Data on). Unzip it to
`data/raw/PSI_download_LUAD.txt`. The SHA-256 of the file used here is recorded in the inventory, so you can check
your copy matches. The portal's terms of use have not been reviewed here; check them before you use the data.

The SpliceSeq resource dates from 2016 and uses an older TCGA alignment.

## Setup

```bash
conda env create -f environment.yml    # env name: luad-splicing (Python 3.12)
conda activate luad-splicing
# or: pip install -r requirements.txt
```

All randomness uses `SEED = 42`.

## Running the pipeline

Run each step from the repository root, in this order:

```bash
python -m src.build_inventory        # 1. parse raw file, write inventory + processed matrices
python -m src.build_matrix           # 2. sample-by-event matrix and sample metadata
python -m src.splits                 #    frozen patient-grouped train/test + 5x10 CV folds
python -m src.qc_report              #    QC report
python -m src.run_eda                # 3. EDA and per-event statistics (no models)
python -m src.run_models_cv          # 4. grouped CV on training patients only
python -m src.run_robust_lowmissing  #    robustness CV (events <= 5% missing)
python -m src.final_test             #    one-shot held-out test evaluation (locked, see below)
python -m src.build_shortlist        # 5. candidate-event shortlist
python -m src.report_figures         # 6. report figures from saved outputs
python -m pytest -q                  # tests
```

**The held-out test set can be used only once.** `src.final_test` creates `results/test_lock.json` and refuses to
run again while it exists. That lock is committed, so the test step will not re-run on a fresh clone. Use
`python -m src.final_test --dry-run` to run the same code path on a CV fold instead.

## Methods in brief

- **Splits first.** A train/test split (102 patients / 113 samples held out) and repeated 5-fold CV (×10) are fixed
  from sample metadata alone, before any data-dependent step. Splits are grouped by patient, so a tumour and its
  matched normal always land together.
- **No leakage.** Coverage filtering, event ranking (paired Wilcoxon on training-fold pairs), imputation, scaling
  and hyperparameter tuning all happen inside each training fold (sklearn `Pipeline`s). Clinical fields are never
  used as features.
- **Statistics.** Paired Wilcoxon signed-rank and unpaired Mann-Whitney U, with Benjamini-Hochberg FDR and
  bootstrap CIs on median ΔPSI.
- **Models.** Majority-class baseline, L2 and L1 logistic regression, and random forest, all evaluated on identical
  splits. The CV controls include shuffled labels, a 7-site restriction, removal of an outlier cluster, and a
  well-covered-events-only rerun.
- **Confounders checked.** Tissue source site, tumour purity, age, sex and stage. Smoking, batch and race are not
  available in this download.

The full frozen protocol is in [`docs/analysis_plan.md`](docs/analysis_plan.md), and every deviation is logged in
[`docs/decisions.md`](docs/decisions.md).

## Results summary

See [`docs/final_report.md`](docs/final_report.md) for the full write-up, which keeps observations, model predictions,
interpretation and speculation separate.

| model (k = 25 events) | CV ROC-AUC (mean ± SD, 50 folds) | held-out test ROC-AUC |
|---|---|---|
| LR-L2 (primary) | 0.998 ± 0.003 | 1.000 |
| Random forest | 0.997 ± 0.003 | 1.000 |
| Majority class | 0.500 | 0.500 |

- Within patients, 5,144 of 25,176 tested events differ at FDR < 0.05 (46 training pairs).
- **19 candidate events** were selected in ≥ 80% of CV folds and also appear in the final model
  ([`results/tables/candidate_shortlist.csv`](results/tables/candidate_shortlist.csv)).
- The shuffled-label control falls to CV AUC 0.52-0.59.

**How to read this.** Near-perfect separation of tumour from *adjacent* normal tissue is a comparatively easy
contrast. It does not imply usefulness for any clinical question. Key limitations:
- **Few test normals.** There are only 11, so the bootstrap CI is degenerate.
- **Site confounding.** Normals come from 7 of 33 tissue source sites.
- **Composition vs splicing.** The signal may reflect tissue cell-type composition rather than splicing changes.
- **One cohort.** A single cohort with an old alignment, and no external validation.

## Repository layout

```
src/              pipeline code (loading, splits, stats, models, figures)
tests/            pytest checks (no patient in both train and test, no NaNs into models, expected shapes)
docs/             data inventory, analysis plan, decision log, QC / EDA / model / final reports
results/tables/   statistics and metrics (CSV)
results/figures/  figures (PNG + PDF)
data/raw/         original download (not tracked; read-only)
data/processed/   derived matrices (not tracked; regenerated by the pipeline)
```

Some documents in `docs/` (`data_inventory.md` and `qc_report.md`) are generated by scripts. Edit the script that
produces them, not the document.
