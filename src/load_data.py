"""Loading and cleaning of the TCGA SpliceSeq LUAD PSI download.

The raw file mixes three things in one tab-delimited table:
  * 13 clinical rows (one value per sample column) at the TOP of the file,
  * ~27.8k PSI event rows (annotation columns + one PSI value per sample),
  * two trailing per-event columns (`avg_exp_pct`, `annotation`) after the samples.
These functions pull them apart without silently coercing any value.
Nothing here ever writes to data/raw/.
"""
import csv
import re

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/PSI_download_LUAD.txt"

# Columns 1-10 of the header (before the sample columns), as confirmed from the file.
LEADING_ANNOT = ["symbol", "as_id", "splice_type", "exons", "from_exon", "to_exon",
                 "novel_splice", "pct_with_values", "psi_range", "std_psi"]
# Columns after the sample columns (not documented in CLAUDE.md; found in the header).
TRAILING_ANNOT = ["avg_exp_pct", "annotation"]
SPLICE_TYPES = {"AA", "AD", "ES", "RI", "AP", "AT", "ME"}
SAMPLE_RE = re.compile(r"^TCGA_[A-Z0-9]{2}_[A-Z0-9]{4}(_Norm)?$")
NORM_SUFFIX = "_Norm"

# Clinical rows that are entirely missing in this download (verified in build_inventory).
CLIN_DROP = ["Height_cm", "Weight_kg", "Tissue_Specific_Attribute_1",
             "Tissue_Specific_Attribute_2", "Tissue_Specific_Attribute_3"]
# Fields where a literal 0 means "missing" (an age or purity of 0 is impossible).
CLIN_ZERO_IS_MISSING = ["Age", "Absolute_Purity_Value"]
CLIN_NUMERIC = ["Age", "Absolute_Purity_Value"]


def read_raw(path=RAW_PATH):
    """Read the whole file as strings, so that no 'null', blank or 0 gets reinterpreted on read.

    QUOTE_NONE: the free-text `annotation` column must not be parsed as quoted fields.
    The header (and the clinical rows) end in a trailing tab, so pandas creates one extra
    unnamed column. We assert it is empty and then drop it.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
                     quoting=csv.QUOTE_NONE)
    # The file has CRLF line endings; the C parser handles them, but check that no CR leaked into the data.
    assert not any("\r" in c for c in df.columns), "CR left in header"
    assert not df.iloc[:, -2:].fillna("").apply(lambda s: s.str.contains("\r")).any().any(), "CR left in values"
    extra = [c for c in df.columns if c == "" or c.startswith("Unnamed:")]
    assert len(extra) == 1, f"expected exactly one empty trailing column, got {extra}"
    assert df[extra[0]].fillna("").eq("").all(), "trailing column is not empty"
    df = df.drop(columns=extra)
    assert list(df.columns[:10]) == LEADING_ANNOT, "unexpected leading annotation columns"
    assert list(df.columns[-2:]) == TRAILING_ANNOT, "unexpected trailing annotation columns"
    return df


def sample_columns(df):
    """Sample columns are everything between the leading and trailing annotation columns."""
    cols = list(df.columns[len(LEADING_ANNOT):-len(TRAILING_ANNOT)])
    bad = [c for c in cols if not SAMPLE_RE.match(c)]
    assert not bad, f"unexpected sample column names: {bad[:5]}"
    assert len(cols) == len(set(cols)), "duplicate sample column names"
    return cols


def split_psi_clinical(df):
    """Split the clinical rows from the PSI rows.

    A clinical row has a blank `splice_type` (and a blank as_id). We detect these rows by content,
    not position: CLAUDE.md expected them at the bottom, but in this file they are at the top.
    """
    is_clin = df["splice_type"].eq("") & df["as_id"].eq("")
    idx = np.flatnonzero(is_clin.values)
    assert len(idx) > 0, "no clinical rows found"
    assert (np.diff(idx) == 1).all(), "clinical rows are not one contiguous block"
    clin, psi = df[is_clin].copy(), df[~is_clin].copy()
    bad = set(psi["splice_type"]) - SPLICE_TYPES
    assert not bad, f"unexpected splice_type values in PSI rows: {bad}"
    assert psi["as_id"].is_unique, "duplicate as_id"
    return psi, clin


def sample_sheet(sample_cols):
    """One row per sample column. The label comes ONLY from the `_Norm` suffix (1 = tumour, 0 = normal).

    patient_id (the column name without `_Norm`) is the grouping key for every later split,
    so a tumour and its matched normal can never be separated.
    """
    s = pd.DataFrame({"sample_id": sample_cols})
    is_norm = s["sample_id"].str.endswith(NORM_SUFFIX)
    s["patient_id"] = s["sample_id"].str.removesuffix(NORM_SUFFIX)
    s["site"] = s["patient_id"].str.split("_").str[1]
    s["label"] = (~is_norm).astype(int)
    s["sample_type"] = np.where(is_norm, "normal", "tumour")
    n_per_patient = s.groupby("patient_id")["sample_id"].transform("size")
    assert n_per_patient.max() <= 2, "a patient has more than one tumour or normal column"
    # Paired = this patient has both a tumour and a normal column.
    s["is_paired"] = n_per_patient.eq(2)
    return s


def clean_psi(psi_rows, sample_cols):
    """Return (PSI matrix events x samples as float, event annotation table).

    Only the string 'null' marks a missing PSI. An exact 0 is a real PSI value (the event is fully
    excluded) and is deliberately NOT recoded. The '0 = missing' rule applies to clinical fields only.
    """
    raw = psi_rows.set_index("as_id")[sample_cols]
    vals = raw.replace({"null": np.nan, "": np.nan})
    psi = vals.apply(pd.to_numeric, errors="coerce")
    # Anything that became NaN here but wasn't 'null'/blank would be a parse problem.
    stray = psi.isna() & vals.notna()
    assert not stray.values.any(), "non-numeric PSI values other than 'null'/blank"
    assert np.nanmin(psi.values) >= 0 and np.nanmax(psi.values) <= 1, "PSI outside [0, 1]"
    psi.index = psi.index.astype(int)
    annot = psi_rows[LEADING_ANNOT + TRAILING_ANNOT].copy()
    annot["as_id"] = annot["as_id"].astype(int)
    return psi, annot


def clinical_coding(clin_rows, sheet):
    """Count how each clinical row encodes missing values, over tumour columns only.

    Normal columns are blank in every clinical row (checked here), so including them would only inflate 'blank'.
    """
    tum = sheet.loc[sheet["label"] == 1, "sample_id"]
    nrm = sheet.loc[sheet["label"] == 0, "sample_id"]
    rows = []
    for _, r in clin_rows.iterrows():
        v = r[tum]
        n_zero = v.isin(["0", "0.0"]).sum()
        n_blank, n_null = v.eq("").sum(), v.eq("null").sum()
        missing = n_blank + n_null + (n_zero if r["symbol"] in CLIN_ZERO_IS_MISSING + ["Height_cm", "Weight_kg"] else 0)
        rows.append({
            "field": r["symbol"],
            "n_tumour": len(v), "n_blank": n_blank, "n_null": n_null, "n_zero": n_zero,
            "n_valid": len(v) - missing,
            "zero_means_missing": r["symbol"] in CLIN_ZERO_IS_MISSING + ["Height_cm", "Weight_kg"],
            "normals_nonblank": int(r[nrm].ne("").sum()),
            "distinct_valid_values": v[~v.isin(["", "null"])].nunique(),
            "usable": r["symbol"] not in CLIN_DROP,
        })
    return pd.DataFrame(rows)


def clean_clinical(clin_rows, sheet):
    """Clinical table with one row per sample. For EDA/confounder checks only, NEVER model features.

    Raw clinical values are blank for every `_Norm` column. If they were used as features, blank vs filled
    would encode the label. For EDA we copy each patient's values onto their matched normal and flag it;
    the 2 unpaired normals have no tumour to copy from and stay NaN.
    """
    t = clin_rows.set_index("symbol")[sheet["sample_id"]].T
    t.columns.name = None
    t = t.drop(columns=CLIN_DROP)
    t = t.replace({"": np.nan, "null": np.nan})
    for c in CLIN_ZERO_IS_MISSING:
        t[c] = t[c].mask(t[c].isin(["0", "0.0"]))
    for c in CLIN_NUMERIC:
        t[c] = pd.to_numeric(t[c], errors="raise")
    out = sheet.set_index("sample_id").join(t)
    assert out.loc[out["label"] == 0, t.columns].isna().all().all(), "normals unexpectedly have clinical values"
    tumour_vals = out.loc[out["label"] == 1].set_index("patient_id")[t.columns]
    is_pn = (out["label"] == 0) & out["is_paired"]
    for c in t.columns:
        out.loc[is_pn, c] = out.loc[is_pn, "patient_id"].map(tumour_vals[c])
    out["clinical_copied_from_tumour"] = is_pn
    return out.reset_index()
