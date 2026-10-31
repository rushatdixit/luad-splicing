"""Phase 2: frozen train/test and repeated-CV splits. Run from the repo root:  python -m src.splits

Writes data/processed/splits.csv (one row per sample):
  set          'train' or 'test' (held-out, used ONCE at the very end)
  cv_rep0..9   validation fold 0-4 for training samples in each CV repeat; -1 for test samples

Why it is built this way (audit notes):
  * Only sample METADATA (label, patient, site) is used, never PSI values. So the split is fixed before
    any data-dependent step and cannot be tuned to the results.
  * The unit of assignment is the PATIENT. A tumour and its matched normal therefore always share a set and a fold.
  * Patients are stratified (paired / normal-only / tumour-only at a normal site / tumour-only elsewhere),
    and within each stratum sorted by site before a round-robin assignment. This spreads the ~10% normals,
    the 57 pairs and the tissue source sites evenly across test/train and across folds.
"""
import numpy as np
import pandas as pd

from src.build_matrix import load_metadata

SEED = 42
TEST_GROUPS = 5   # held-out test = 1 of 5 patient groups, i.e. ~20% of patients
N_FOLDS = 5
N_REPEATS = 10
OUT = "data/processed/splits.csv"


def patient_table(meta):
    """One row per patient with its stratum. Strata are defined from labels and site only."""
    p = meta.groupby("patient_id").agg(site=("site", "first"),
                                       n_tumour=("label", "sum"),
                                       n_normal=("label", lambda s: int((s == 0).sum())),
                                       normal_site=("normal_site", "first"))
    assert meta.groupby("patient_id")["site"].nunique().eq(1).all(), "a patient spans two sites"
    p["stratum"] = np.select(
        [(p.n_tumour == 1) & (p.n_normal == 1), p.n_tumour == 0, p.normal_site],
        ["paired", "normal_only", "tumour_normal_site"], default="tumour_other_site")
    return p


def assign_groups(p, k, rng):
    """Assign each patient in `p` to one of k groups, balanced within stratum and spread across sites.

    Shuffle first so the order within a site is random, then stable-sort by site so that consecutive
    positions walk through the sites, then deal positions round-robin from a random offset.
    """
    groups = pd.Series(-1, index=p.index, dtype=int)
    for _, sub in p.groupby("stratum", sort=True):
        ids = sub.index.to_numpy()[rng.permutation(len(sub))]
        ids = ids[np.argsort(sub.loc[ids, "site"].to_numpy(), kind="stable")]
        groups[ids] = (np.arange(len(ids)) + rng.integers(k)) % k
    return groups


def make_splits(meta, seed=SEED):
    """Return the per-sample split table. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    p = patient_table(meta)
    p["set"] = np.where(assign_groups(p, TEST_GROUPS, rng) == 0, "test", "train")
    train = p[p["set"] == "train"]
    for r in range(N_REPEATS):
        # Folds are drawn on TRAINING patients only; test patients never enter CV.
        p[f"cv_rep{r}"] = assign_groups(train, N_FOLDS, rng).reindex(p.index).fillna(-1).astype(int)
    cols = ["stratum", "set"] + [f"cv_rep{r}" for r in range(N_REPEATS)]
    out = meta[["sample_id", "patient_id", "site", "label", "is_paired", "normal_site"]].merge(
        p[cols], left_on="patient_id", right_index=True, how="left", validate="many_to_one")
    return out


def check_splits(s):
    """Leakage and balance assertions. Raises AssertionError on any violation."""
    cv = [f"cv_rep{r}" for r in range(N_REPEATS)]
    # 1. No patient on both sides, and each patient's samples share set and folds.
    assert s.groupby("patient_id")["set"].nunique().eq(1).all(), "patient split across train/test"
    assert not set(s.loc[s.set == "train", "patient_id"]) & set(s.loc[s.set == "test", "patient_id"])
    for c in cv:
        assert s.groupby("patient_id")[c].nunique().eq(1).all(), f"patient split across folds in {c}"
    # 2. Test samples are never in CV; every training sample is in exactly one fold 0..k-1 per repeat.
    assert (s.loc[s.set == "test", cv] == -1).all().all()
    assert s.loc[s.set == "train", cv].isin(range(N_FOLDS)).all().all()
    # 3. Every fold and the test set contain normals and pairs, also within the 7-site subset.
    for name, sub in [("all", s), ("7-site", s[s.normal_site])]:
        assert ((sub.set == "test") & (sub.label == 0)).any(), f"no normals in test ({name})"
        assert ((sub.set == "test") & sub.is_paired & (sub.label == 0)).any(), f"no pairs in test ({name})"
        for c in cv:
            for f in range(N_FOLDS):
                v = sub[sub[c] == f]
                assert (v.label == 0).any() and (v.is_paired & (v.label == 0)).any(), f"{c} fold {f} ({name})"


def main():
    meta = load_metadata()
    s = make_splits(meta)
    check_splits(s)
    assert s.equals(make_splits(meta)), "splits are not deterministic for the fixed seed"
    assert len(s) == 573 and s["patient_id"].nunique() == 516
    s.to_csv(OUT, index=False)

    pt = s.drop_duplicates("patient_id")
    print("patients by set x stratum:\n" + pd.crosstab(pt["stratum"], pt["set"], margins=True).to_string())
    print("samples by set x label:\n" + pd.crosstab(s["set"], s["label"], margins=True).to_string())
    print("test patients:", pt.loc[pt.set == "test", "patient_id"].nunique(),
          "| train/test patient overlap:",
          len(set(pt.loc[pt.set == "train", "patient_id"]) & set(pt.loc[pt.set == "test", "patient_id"])))


if __name__ == "__main__":
    main()
