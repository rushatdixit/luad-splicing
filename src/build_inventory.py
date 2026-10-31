"""Phase 1: data inventory. Run from the repo root:  python -m src.build_inventory

Reads data/raw/PSI_download_LUAD.txt (read-only) and writes:
  data/processed/psi_events_x_samples.csv.gz   PSI matrix (as_id x sample), NaN = unquantified
  data/processed/event_annotation.csv          per-event annotation (NOT features, see leakage note)
  data/processed/clinical_by_sample.csv        sample sheet + clinical fields (EDA only, never features)
  results/tables/inventory_*.csv               counts and missingness tables
  docs/data_inventory.md                       generated inventory document
No modelling, feature selection or statistical testing happens here.
"""
import hashlib
import os
import platform
import sys
from datetime import date

import numpy as np
import pandas as pd

from src.load_data import (RAW_PATH, clean_clinical, clean_psi, clinical_coding, read_raw,
                           sample_columns, sample_sheet, split_psi_clinical)

SEED = 42  # not used in this phase (nothing random); recorded for consistency with later phases

# Expected values, verified independently with awk on the raw file before this script was written.
EXPECTED = dict(n_events=27836, n_samples=573, n_tumour=514, n_normal=59, n_paired=57,
                n_clin_rows=13, unpaired=["TCGA_44_5645_Norm", "TCGA_44_6144_Norm"])

PROC, TABLES, DOC = "data/processed", "results/tables", "docs/data_inventory.md"


def md_table(df, index=False):
    """Minimal markdown table writer (avoids a dependency on `tabulate`)."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * d.shape[1]) + "|"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}" if not np.isnan(v) else "NaN"
    return str(v)


def missing_summary(frac):
    """Distribution of a per-event (or per-sample) missing fraction."""
    q = frac.quantile([0, .05, .25, .5, .75, .95, 1]).values
    return {"n": len(frac), "mean": frac.mean(), "min": q[0], "p05": q[1], "p25": q[2], "median": q[3],
            "p75": q[4], "p95": q[5], "max": q[6],
            "n_complete": int((frac == 0).sum()), "n_gt_0.10": int((frac > .10).sum()),
            "n_gt_0.25": int((frac > .25).sum()), "n_gt_0.50": int((frac > .50).sum()),
            "n_all_missing": int((frac == 1).sum())}


def main():
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)

    # ---- load & split -------------------------------------------------------------------------
    raw_bytes = os.path.getsize(RAW_PATH)
    with open(RAW_PATH, "rb") as fh:
        sha256 = hashlib.sha256(fh.read()).hexdigest()
    df = read_raw(RAW_PATH)
    samples = sample_columns(df)
    psi_rows, clin_rows = split_psi_clinical(df)
    clin_first_line = int(clin_rows.index.min()) + 2  # +1 header, +1 for 1-based line numbers
    clin_last_line = int(clin_rows.index.max()) + 2
    sheet = sample_sheet(samples)
    psi, annot = clean_psi(psi_rows, samples)
    clin = clean_clinical(clin_rows, sheet)
    coding = clinical_coding(clin_rows, sheet)

    # ---- assertions (CLAUDE.md: expected shape and sample counts) --------------------------------
    n_tum, n_norm = int(sheet["label"].sum()), int((sheet["label"] == 0).sum())
    paired_norm = sheet[(sheet["label"] == 0) & sheet["is_paired"]]
    unpaired_norm = sheet[(sheet["label"] == 0) & ~sheet["is_paired"]]
    assert psi.shape == (EXPECTED["n_events"], EXPECTED["n_samples"]), psi.shape
    assert (n_tum, n_norm) == (EXPECTED["n_tumour"], EXPECTED["n_normal"])
    assert len(paired_norm) == EXPECTED["n_paired"]
    assert sorted(unpaired_norm["sample_id"]) == EXPECTED["unpaired"]
    assert len(clin_rows) == EXPECTED["n_clin_rows"]
    assert list(psi.columns) == list(sheet["sample_id"]) == list(clin["sample_id"])
    # Every normal's patient either has a tumour column (paired) or is in the unpaired list.
    tum_pids = set(sheet.loc[sheet["label"] == 1, "patient_id"])
    assert set(paired_norm["patient_id"]) <= tum_pids
    assert not set(unpaired_norm["patient_id"]) & tum_pids

    # ---- write processed data -----------------------------------------------------------------
    psi.to_csv(f"{PROC}/psi_events_x_samples.csv.gz", index_label="as_id", float_format="%.4g")
    annot.to_csv(f"{PROC}/event_annotation.csv", index=False)
    clin.to_csv(f"{PROC}/clinical_by_sample.csv", index=False)

    # ---- inventory tables ---------------------------------------------------------------------
    ev_type = (annot.groupby("splice_type").size().rename("n_events")
               .sort_values(ascending=False).to_frame())
    ev_type.loc["TOTAL"] = ev_type["n_events"].sum()
    ev_type.to_csv(f"{TABLES}/inventory_events_by_type.csv")

    site = (sheet.assign(paired_normal=(sheet["label"] == 0) & sheet["is_paired"])
            .groupby("site").agg(tumour=("label", "sum"),
                                 normal=("label", lambda x: int((x == 0).sum())),
                                 paired_normal=("paired_normal", "sum"))
            .sort_values(["normal", "tumour"], ascending=False))
    site["frac_normal"] = site["normal"] / (site["tumour"] + site["normal"])
    site.to_csv(f"{TABLES}/inventory_samples_by_site.csv")

    isna = psi.isna()
    tum_cols = sheet.loc[sheet["label"] == 1, "sample_id"]
    nrm_cols = sheet.loc[sheet["label"] == 0, "sample_id"]
    ev_frac = isna.mean(axis=1)
    ev_summary = {"all_events__all_samples": missing_summary(ev_frac),
                  "all_events__tumour_only": missing_summary(isna[tum_cols].mean(axis=1)),
                  "all_events__normal_only": missing_summary(isna[nrm_cols].mean(axis=1))}
    types = annot.set_index("as_id")["splice_type"].reindex(psi.index)
    for t in ev_type.index.drop("TOTAL"):
        ev_summary[f"{t}__all_samples"] = missing_summary(ev_frac[types == t])
    ev_summary = pd.DataFrame(ev_summary).T
    ev_summary.index.name = "subset"
    ev_summary.to_csv(f"{TABLES}/inventory_missingness_per_event_summary.csv")

    smp = sheet.set_index("sample_id")[["patient_id", "site", "sample_type", "is_paired"]].copy()
    smp["frac_missing"] = isna.mean(axis=0)
    smp.to_csv(f"{TABLES}/inventory_missingness_per_sample.csv")
    smp_summary = pd.DataFrame({k: missing_summary(g["frac_missing"])
                                for k, g in [("all", smp), *smp.groupby("sample_type")]}).T

    coding.to_csv(f"{TABLES}/inventory_clinical_coding.csv", index=False)

    # Categorical value counts (tumours only, after recoding) for the doc.
    tum_clin = clin[clin["label"] == 1]
    cat_counts = {c: tum_clin[c].value_counts(dropna=False).to_dict()
                  for c in ["Gender", "Tumor_Status", "Vital_Status", "Pathologic_Tumor_Stage",
                            "Pathologic_N", "Pathologic_T"]}

    # Are the 0..1 bounds actually attained, and how common is exact 0/1?
    v = psi.values[~np.isnan(psi.values)]
    psi_stats = dict(min=float(v.min()), max=float(v.max()), n_nan=int(isna.values.sum()),
                     n_cells=int(psi.size), n_exact0=int((v == 0).sum()), n_exact1=int((v == 1).sum()))

    versions = dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__)

    write_doc(locals())
    print_summary(locals())


def print_summary(L):
    print(f"raw file: {RAW_PATH} ({L['raw_bytes']:,} bytes) sha256={L['sha256']}")
    print(f"clinical rows at file lines {L['clin_first_line']}-{L['clin_last_line']}:")
    for name in L["clin_rows"]["symbol"]:
        print(f"   {name}")
    print(f"events: {L['psi'].shape[0]:,}  samples: {L['psi'].shape[1]}  tumour: {L['n_tum']}  normal: {L['n_norm']}")
    print(f"paired normals: {len(L['paired_norm'])}  unpaired: {list(L['unpaired_norm']['sample_id'])}")
    print(L["ev_type"].to_string())
    print(L["site"][L["site"]["normal"] > 0].to_string())
    print(f"PSI: {L['psi_stats']}")
    print(L["ev_summary"].loc[["all_events__all_samples", "all_events__tumour_only", "all_events__normal_only"]].to_string())
    print(L["smp_summary"].to_string())
    print(L["coding"].to_string())
    print(f"versions: {L['versions']}")


def write_doc(L):
    site, coding, ev_summary, smp_summary = L["site"], L["coding"], L["ev_summary"], L["smp_summary"]
    ps = L["psi_stats"]
    no_norm_sites = site[site["normal"] == 0]
    unpaired = L["unpaired_norm"]
    sum_cols = ["n", "mean", "min", "p25", "median", "p75", "p95", "max",
                "n_complete", "n_gt_0.10", "n_gt_0.25", "n_gt_0.50", "n_all_missing"]
    ev_tbl = ev_summary[sum_cols].copy()
    for c in ["n", "n_complete", "n_gt_0.10", "n_gt_0.25", "n_gt_0.50", "n_all_missing"]:
        ev_tbl[c] = ev_tbl[c].astype(int)
    sm_tbl = smp_summary[sum_cols].copy()
    for c in ["n", "n_complete", "n_gt_0.10", "n_gt_0.25", "n_gt_0.50", "n_all_missing"]:
        sm_tbl[c] = sm_tbl[c].astype(int)
    worst = L["smp"].sort_values("frac_missing", ascending=False).head(5)

    cat_lines = "\n".join(f"- **{k}**: " + ", ".join(f"{_fmt(a) if not (isinstance(a, float) and np.isnan(a)) else 'NaN'} = {b}"
                                                      for a, b in v.items())
                          for k, v in L["cat_counts"].items())
    age = L["tum_clin"]["Age"].dropna()
    pur = L["tum_clin"]["Absolute_Purity_Value"].dropna()

    doc = f"""# Data inventory: TCGA SpliceSeq LUAD PSI

> **Generated by `src/build_inventory.py`. Do not edit by hand; change the script and rerun**
> (`python -m src.build_inventory` from the repo root). Generated on {date.today().isoformat()}.

## 1. Source and download

| Item | Value |
|---|---|
| Source | TCGA SpliceSeq (MD Anderson), PSI Data Download page: https://bioinformatics.mdanderson.org/TCGASpliceSeq/PSIdownload.jsp |
| Downloaded | 29 Oct 2026, as `PSI_download_LUAD.zip` (single txt file) |
| Raw file | `data/raw/PSI_download_LUAD.txt` (name matches CLAUDE.md) |
| Size | {L['raw_bytes']:,} bytes |
| SHA-256 | `{L['sha256']}` |
| Format | tab-delimited ASCII, CRLF line endings, 1 header line + {len(L['clin_rows'])} clinical rows + {L['psi'].shape[0]:,} PSI rows |
| Access terms | **Not verified.** I did not read the portal's terms of use. Please confirm and record them here. |
| Resource age | SpliceSeq resource dates from 2016 (older TCGA alignment). This is a limitation to report. |

**Download settings** (from CLAUDE.md): tumour type LUAD; all genes; all samples; all 7 event types
(AA, AD, ES, RI, AP, AT, ME); Percentage of Samples with PSI Value = 75; Minimum Average Expression Percentage = 50;
Minimum PSI Range = 0; Minimum PSI Standard Deviation = 0; Gene RPKM Matrix = off; Gene Descriptions = on; Clinical Data = on.

## 2. File structure (as found)

- Columns 1-10: `symbol, as_id, splice_type, exons, from_exon, to_exon, novel_splice, pct_with_values, psi_range, std_psi`.
- Columns 11-583: {L['psi'].shape[1]} sample columns (`TCGA_XX_XXXX` or `TCGA_XX_XXXX_Norm`). No duplicate names.
- Columns 584-585: **`avg_exp_pct`, `annotation`** (gene description text). *Not listed in CLAUDE.md.*
- The header and clinical rows end with a trailing tab (an empty 586th field). It is dropped on load.
- **Clinical rows are at file lines {L['clin_first_line']}-{L['clin_last_line']}, i.e. at the TOP of the file, directly under the header,
  not appended below the PSI rows as CLAUDE.md expected.** The loader identifies them by content (blank `splice_type`
  and `as_id`), not by position. Their first-column labels, verbatim:
  {", ".join(f"`{s}`" for s in L['clin_rows']['symbol'])}.

## 3. Labels and samples

- **Label rule:** a column ending `_Norm` is non-tumour (label 0); every other sample column is tumour (label 1).
  The label comes only from this suffix. `Tumor_Status` is NOT used as a label.
- **Patient ID** = column name with `_Norm` removed. This is the group key for every later split.
- **Site** = 2nd field of the ID (tissue source site).

| | Count |
|---|---|
| Sample columns | {L['psi'].shape[1]} |
| Tumour (label 1) | {L['n_tum']} |
| Non-tumour `_Norm` (label 0) | {L['n_norm']} |
| Unique patients | {L['sheet']['patient_id'].nunique()} |
| Matched tumour + normal pairs | {len(L['paired_norm'])} |
| Unpaired normals (no tumour column) | {len(unpaired)}: {", ".join(f"`{s}`" for s in unpaired['sample_id'])} (both site {", ".join(sorted(unpaired['site'].unique()))}) |
| Tumours without a normal | {L['n_tum'] - len(L['paired_norm'])} |

Class balance: {L['n_norm']}/{L['psi'].shape[1]} = {L['n_norm'] / L['psi'].shape[1]:.1%} non-tumour, so the classes are **imbalanced** (PR-AUC and stratified splits are needed later).

### Samples per tissue source site

Normals come from **{int((site['normal'] > 0).sum())} of {len(site)} sites**. The other {len(no_norm_sites)} sites contribute
{int(no_norm_sites['tumour'].sum())} tumours and **no normals at all**. Site is therefore strongly associated with the label
and is a major potential confounder (see §7).

Sites with at least one normal:

{md_table(site[site['normal'] > 0], index=True)}

Sites with no normals (tumour count): {", ".join(f"{s} ({int(t)})" for s, t in no_norm_sites['tumour'].items())}.

Full table: `results/tables/inventory_samples_by_site.csv`.

## 4. Events

{md_table(L['ev_type'], index=True)}

- `as_id` is unique ({L['psi'].shape[0]:,} events, {L['annot']['symbol'].nunique():,} gene symbols).
- `novel_splice` = 1 for {int((L['annot']['novel_splice'] == '1').sum())} events.
- **Leakage note:** `pct_with_values`, `psi_range`, `std_psi` and `avg_exp_pct` are computed across ALL samples.
  They live only in `event_annotation.csv` and must not be used to filter or select events before the train/test split.
  The portal's download filters (≥75% samples with PSI, ≥50% min. average expression) were also applied across all samples.
  These filters ignore labels, but they cannot be undone, so this is noted as an unavoidable minor limitation.

## 5. PSI values and missingness

- Missing PSI is coded **only** as the string `null` (no blanks were found). It is recoded to NaN.
  NaN cells: {ps['n_nan']:,} of {ps['n_cells']:,} ({ps['n_nan'] / ps['n_cells']:.2%}).
- **Range check: min = {ps['min']}, max = {ps['max']}. All values lie within [0, 1].**
- Exact 0: {ps['n_exact0']:,} cells; exact 1: {ps['n_exact1']:,} cells. **An exact 0 is a real PSI value and is kept.** The
  "0 means missing" rule in CLAUDE.md applies to clinical fields only.

### Fraction missing per event (across samples)

{md_table(ev_tbl, index=True)}

### Fraction missing per sample (across events)

{md_table(sm_tbl, index=True)}

Five samples with the most missing values: {", ".join(f"`{i}` ({r.frac_missing:.1%}, {r.sample_type})" for i, r in worst.iterrows())}.
Per-sample table: `results/tables/inventory_missingness_per_sample.csv`.

## 6. Clinical rows

Values are blank for every `_Norm` column. All counts below are over the {L['n_tum']} tumour columns.
**Clinical columns are never model features** (blank vs filled would leak the label).
In `clinical_by_sample.csv`, matched normals receive their patient's values (flag `clinical_copied_from_tumour`),
and the 2 unpaired normals stay NaN.

{md_table(coding)}

Recoding applied: blank → NaN; `null` → NaN; `0`/`0.0` → NaN for `Age` and `Absolute_Purity_Value`.
Dropped as entirely missing: `Height_cm`, `Weight_kg` (all 0), `Tissue_Specific_Attribute_1/2/3` (all `null`).

Usable fields (tumours, after recoding):
- **Age**: n = {len(age)}, median {age.median():.0f}, range {age.min():.0f}-{age.max():.0f}
- **Absolute_Purity_Value**: n = {len(pur)} ({len(pur) / L['n_tum']:.0%} of tumours), median {pur.median():.2f}, range {pur.min():.2f}-{pur.max():.2f}
{cat_lines}

`NX` / `TX` mean "cannot be assessed". They are kept as their own categories, not recoded to NaN.
Smoking history, race and batch/plate are **not available** in this download.

## 7. Inclusion / exclusion

- Phase 1 applied **no** sample or event exclusions: all {L['psi'].shape[1]} samples and {L['psi'].shape[0]:,} events are kept.
- The 2 unpaired normals stay in the main tumour-vs-normal analysis and are excluded only from the paired sensitivity analysis.

## 8. Assumptions and oddities

1. **Clinical rows are at the top of the file, not the bottom** (contradicts CLAUDE.md; detected by content).
2. **Two extra trailing per-event columns** (`avg_exp_pct`, `annotation`) are not described in CLAUDE.md. Both are annotation, not features.
3. **Site confounding:** all {L['n_norm']} normals come from {int((site['normal'] > 0).sum())} sites; {int(no_norm_sites['tumour'].sum())} tumours come from sites with no normals.
   A classifier could learn site rather than tumour status. The matched-pair analysis is the main control.
   Restricting to the normal-contributing sites is also worth considering as a sensitivity check. That is a plan decision for you.
4. Tumour purity is missing (coded 0) for {int(coding.set_index('field').loc['Absolute_Purity_Value', 'n_zero'])} of {L['n_tum']} tumours, which limits the purity confounder check.
5. **Missingness differs by class.** {int(ev_summary.loc['all_events__normal_only', 'n_gt_0.50'])} events are >50% missing within normals
   (max {ev_summary.loc['all_events__normal_only', 'max']:.1%}), but {int(ev_summary.loc['all_events__tumour_only', 'n_gt_0.50'])} are within tumours.
   The missingness pattern itself may therefore carry label (or site) information. Coverage filters and imputation in later phases must be fitted
   inside training folds, and missingness indicators should not be used as features without an explicit decision.
6. Exact PSI 0 is treated as a real value; only `null` is missing.
7. Patient ID is assumed to be the first three ID fields (`TCGA_<site>_<participant>`). No patient has more than one tumour or one normal column (asserted).
8. The SpliceSeq resource dates from 2016 and uses an older TCGA alignment.
9. Access terms were not verified (see §1).

## 9. Reproducibility

- Code: `src/load_data.py` (loading/cleaning), `src/build_inventory.py` (this report + outputs).
- Outputs: `data/processed/psi_events_x_samples.csv.gz` ({L['psi'].shape[0]:,} events × {L['psi'].shape[1]} samples, rows = events;
  the samples × events transpose is a Phase 2 step), `data/processed/event_annotation.csv`, `data/processed/clinical_by_sample.csv`,
  `results/tables/inventory_*.csv`.
- No randomness in this phase (SEED = {SEED} recorded for later phases).
- Versions: Python {L['versions']['python']}, pandas {L['versions']['pandas']}, numpy {L['versions']['numpy']} (conda env `luad-splicing`, see `environment.yml`).
"""
    with open(DOC, "w") as fh:
        fh.write(doc)


if __name__ == "__main__":
    sys.exit(main())
