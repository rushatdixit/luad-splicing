"""Phase 5 checks on the 19 shortlisted candidate events. Run from the repo root:  python -m src.phase5_checks

Descriptive and post hoc. TRAINING samples only (set == "train" in data/processed/splits.csv); no model is fitted
and the test set is never read. Writes three NEW tables and refuses to overwrite them if they already exist:
  results/tables/shortlist_event_annotation.csv   what the SpliceSeq download says about each event
  results/tables/shortlist_event_behaviour.csv    missingness, median PSI per class, PSI pinned at exactly 0/1
  results/tables/shortlist_purity_corr.csv        Spearman PSI vs Absolute_Purity_Value in training tumours, BH-FDR

Step 1 uses the local download only (owner decision): fields the download does not contain (chromosome, strand,
coordinates, transcripts) are marked as such rather than looked up or guessed.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from src.build_matrix import PROC, load_features, load_metadata
from src.eda_stats import bh, paired_table

T = "results/tables"
OUT_ANNOT = f"{T}/shortlist_event_annotation.csv"
OUT_BEHAV = f"{T}/shortlist_event_behaviour.csv"
OUT_PURITY = f"{T}/shortlist_purity_corr.csv"

N_TRAIN, N_TRAIN_NORMAL, N_TRAIN_PAIRS = 460, 48, 46  # from results/test_lock.json fit_info and docs/final_report.md
MAX_NORMAL_MISSING = 0.20   # "normals more than 20% missing" (request)
PINNED_FRAC = 0.50          # "mostly pinned" = more than half of non-missing normals at exactly 0 or 1 (assumption, stated in the doc)
NOT_IN_DOWNLOAD = "not in SpliceSeq PSI download"
CANNOT_DECIDE = ("cannot be decided from the annotation: the download gives only SpliceSeq splice-graph exon IDs, "
                 "with no mapping to RefSeq/Ensembl exon numbers, coordinates or transcript isoforms")
ISOFORM_QUESTIONS = {
    "NUMB": "Is this ES event the exon known as exon 9 (the PRRL/PRRS isoform exon)?",
    "CEACAM1": "Which exon is this, and is it the exon that distinguishes the long and short cytoplasmic-tail isoforms?",
}


def training_samples(meta, splits):
    """Training-set metadata only. The assertions are the guard that no test sample or patient can enter this task."""
    s = splits.set_index("sample_id")
    m = meta.copy()
    m["set"] = m.sample_id.map(s["set"])
    assert m["set"].notna().all(), "every sample must have a split assignment"
    tr = m[m["set"] == "train"]
    te_patients = set(m.loc[m["set"] == "test", "patient_id"])
    assert len(tr) == N_TRAIN, f"expected {N_TRAIN} training samples, got {len(tr)}"
    assert not (set(tr.patient_id) & te_patients), "a training patient also appears in the test set"
    assert (tr.label == 0).sum() == N_TRAIN_NORMAL
    return tr


def annotation_table(short, ann):
    """Step 1: verbatim fields from the download, plus explicit 'not available' markers."""
    a = ann.set_index("as_id").loc[short.as_id]
    out = pd.DataFrame({
        "final_rank": short.final_rank.values,
        "as_id": short.as_id.values,
        "gene": a.symbol.values,
        "splice_type": a.splice_type.values,
        "spliceseq_exons": a.exons.values,
        "from_exon": a.from_exon.values,
        "to_exon": a.to_exon.values,
        "novel_splice": a.novel_splice.values,
        "chromosome": NOT_IN_DOWNLOAD,
        "strand": NOT_IN_DOWNLOAD,
        "genomic_coordinates": NOT_IN_DOWNLOAD,
        "psi_alternative_transcripts": NOT_IN_DOWNLOAD,
        "annotation_portal_gene_description": a.annotation.fillna("").values,  # verbatim; not verified by us
    })
    out["isoform_question"] = out.gene.map(ISOFORM_QUESTIONS).fillna("")
    out["isoform_answer"] = np.where(out.isoform_question != "", CANNOT_DECIDE, "")
    return out


def pct(x):
    return round(100 * float(x), 2)


def behaviour_table(short, X, tr):
    """Step 2: per-class missingness, median PSI and PSI exactly 0/1, training samples only.

    Exact equality (== 0, == 1) is deliberate: PSI = 0 is a real value (CLAUDE.md), and values were parsed from text.
    """
    tum_ids = tr.loc[tr.label == 1, "sample_id"]
    nrm_ids = tr.loc[tr.label == 0, "sample_id"]
    paired_patients = sorted(set(tr.loc[tr.label == 0, "patient_id"]) & set(tr.loc[tr.label == 1, "patient_id"]))
    assert len(paired_patients) == N_TRAIN_PAIRS, f"expected {N_TRAIN_PAIRS} training pairs, got {len(paired_patients)}"
    Tp, Np = paired_table(X, tr, paired_patients)

    rows = []
    for _, s in short.iterrows():
        e = s.as_id
        t, n = X.loc[tum_ids, e], X.loc[nrm_ids, e]
        t_ok, n_ok = t.dropna(), n.dropna()
        pt, pn = Tp[e], Np[e]
        complete = pt.notna() & pn.notna()
        pinned_pair = (pt[complete].isin([0.0, 1.0]) | pn[complete].isin([0.0, 1.0]))
        r = {
            "final_rank": s.final_rank, "as_id": e, "gene": s.gene, "splice_type": s.splice_type,
            "n_train_tumour": len(t), "n_train_normal": len(n),
            "pct_missing_tumour": pct(t.isna().mean()), "pct_missing_normal": pct(n.isna().mean()),
            "median_psi_tumour": t_ok.median(), "median_psi_normal": n_ok.median(),
            "pct_tumour_at_0_or_1": pct(t_ok.isin([0.0, 1.0]).mean()),
            "pct_normal_at_0_or_1": pct(n_ok.isin([0.0, 1.0]).mean()),
            "n_complete_pairs": int(complete.sum()),
            "pct_pairs_either_0_or_1": pct(pinned_pair.mean()),
            "shortlist_median_dPSI_paired": s.median_dPSI_tumour_minus_normal,
        }
        r["flag_normal_missing_gt20"] = n.isna().mean() > MAX_NORMAL_MISSING
        r["flag_normal_pinned"] = n_ok.isin([0.0, 1.0]).mean() > PINNED_FRAC
        r["any_flag"] = r["flag_normal_missing_gt20"] or r["flag_normal_pinned"]
        rows.append(r)
    return pd.DataFrame(rows)


def purity_table(short, X, tr, clin):
    """Step 3: Spearman PSI vs purity in training tumours with a purity value; BH across the 19 events."""
    pur = clin.set_index("sample_id")["Absolute_Purity_Value"]
    tum = tr.loc[tr.label == 1, "sample_id"]
    tum = tum[pur.reindex(tum).notna().values]
    rows = []
    for _, s in short.iterrows():
        e = s.as_id
        d = pd.DataFrame({"psi": X.loc[tum, e].values, "purity": pur.loc[tum].values}).dropna()
        rho, p = stats.spearmanr(d.psi, d.purity)
        rows.append({"final_rank": s.final_rank, "as_id": e, "gene": s.gene, "splice_type": s.splice_type,
                     "n_tumours_with_purity": len(tum), "n": len(d), "rho": rho, "p": p,
                     "shortlist_median_dPSI_paired": s.median_dPSI_tumour_minus_normal})
    out = pd.DataFrame(rows)
    out["fdr_bh"] = bh(out.p)
    # Same convention as eda_purity_direction_top10.csv: does higher purity move PSI the same way as tumour - normal?
    out["same_direction_as_tumour_minus_normal"] = np.sign(out.rho) == np.sign(out.shortlist_median_dPSI_paired)
    return out, len(tum)


def main():
    for p in (OUT_ANNOT, OUT_BEHAV, OUT_PURITY):
        if os.path.exists(p):
            raise SystemExit(f"REFUSING TO OVERWRITE existing output {p}")

    short = pd.read_csv(f"{T}/candidate_shortlist.csv", dtype={"as_id": str})
    ann = pd.read_csv(f"{PROC}/event_annotation.csv", dtype={"as_id": str}, keep_default_na=False,
                      na_values=[""])  # keep the literal 'null' in from_exon/to_exon as written in the download
    X = load_features()
    meta = load_metadata()
    splits = pd.read_csv(f"{PROC}/splits.csv", dtype={"site": str})
    clin = pd.read_csv(f"{PROC}/clinical_by_sample.csv", dtype={"site": str})

    assert len(short) == 19
    assert set(short.as_id) <= set(X.columns) and set(short.as_id) <= set(ann.as_id), "shortlist event missing"

    tr = training_samples(meta, splits)
    Xtr = X.loc[tr.sample_id]  # from here on only training rows exist in memory for this task

    a = annotation_table(short, ann)
    b = behaviour_table(short, Xtr, tr)
    c, n_tum_pur = purity_table(short, Xtr, tr, clin)
    for df in (a, b, c):
        assert len(df) == 19

    a.to_csv(OUT_ANNOT, index=False)
    b.to_csv(OUT_BEHAV, index=False)
    c.to_csv(OUT_PURITY, index=False)
    print(f"training samples {len(tr)} ({(tr.label == 1).sum()} tumour, {(tr.label == 0).sum()} normal); "
          f"training tumours with purity {n_tum_pur}")
    print(f"flagged events (Step 2): {int(b.any_flag.sum())}; purity FDR < 0.05 (Step 3): {int((c.fdr_bh < 0.05).sum())}")


if __name__ == "__main__":
    main()
