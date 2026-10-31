"""Per-event tumour-vs-normal statistics (Phase 3). Pure functions, no file I/O.

Paired:   Wilcoxon signed-rank on complete pairs (tumour - normal), zero_method="wilcox".
Unpaired: Mann-Whitney U (two-sided) on non-missing values.
Effect sizes: median difference (primary) with a 95% percentile bootstrap CI, plus mean difference.
Multiple testing: Benjamini-Hochberg over the testable events of one analysis family.

Bootstrap seeds are derived per event as default_rng([seed, j]). Results therefore do not depend on
chunking or parallelism, and every event's CI can be reproduced on its own.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

N_BOOT = 2000
SEED = 42


def paired_table(X, meta, patient_ids):
    """Tumour and normal PSI matrices (rows = patient_id, same order), for the given paired patients."""
    m = meta.set_index("sample_id")
    tum = m[(m.label == 1) & m.patient_id.isin(patient_ids)]
    nrm = m[(m.label == 0) & m.patient_id.isin(patient_ids)]
    T = X.loc[tum.index].set_axis(tum.patient_id.values).loc[list(patient_ids)]
    N = X.loc[nrm.index].set_axis(nrm.patient_id.values).loc[list(patient_ids)]
    assert len(T) == len(N) == len(patient_ids), "every patient must have exactly one tumour and one normal"
    return T, N


def _paired_one(j, d, min_pairs, n_boot, seed):
    d = d[~np.isnan(d)]
    n, nz = len(d), int((d != 0).sum())
    row = {"n_pairs": n, "n_nonzero": nz,
           "median_diff": np.median(d) if n else np.nan, "mean_diff": d.mean() if n else np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "W": np.nan, "p": np.nan, "testable": False, "reason": ""}
    if n < min_pairs:
        row["reason"] = f"fewer_than_{min_pairs}_complete_pairs"
        return row
    if nz == 0:
        row["reason"] = "all_differences_zero"
        return row
    res = stats.wilcoxon(d, zero_method="wilcox")
    row.update(W=res.statistic, p=res.pvalue, testable=True)
    if n_boot > 0:  # n_boot=0: ranking inside CV folds needs no CI
        rng = np.random.default_rng([seed, j])
        boots = np.median(d[rng.integers(0, n, (n_boot, n))], axis=1)
        row.update(ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5))
    return row


def paired_tests(T, N, min_pairs=30, n_boot=N_BOOT, seed=SEED, n_jobs=-1):
    """Per-event paired Wilcoxon + bootstrap CI of the median paired difference (tumour - normal)."""
    assert list(T.index) == list(N.index) and list(T.columns) == list(N.columns)
    D = T.values - N.values
    rows = Parallel(n_jobs=n_jobs, batch_size=512)(
        delayed(_paired_one)(j, D[:, j], min_pairs, n_boot, seed) for j in range(D.shape[1]))
    out = pd.DataFrame(rows, index=pd.Index(T.columns, name="as_id"))
    out["fdr"] = bh(out["p"])
    return out


def _unpaired_one(j, t, n, min_group, n_boot, seed):
    t, n = t[~np.isnan(t)], n[~np.isnan(n)]
    row = {"n_tumour": len(t), "n_normal": len(n),
           "median_tumour": np.median(t) if len(t) else np.nan, "median_normal": np.median(n) if len(n) else np.nan,
           "median_diff": np.nan, "mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
           "U": np.nan, "p": np.nan, "testable": False, "reason": ""}
    if len(t) < min_group or len(n) < min_group:
        row["reason"] = f"fewer_than_{min_group}_values_in_a_group"
        return row
    row["median_diff"] = row["median_tumour"] - row["median_normal"]
    row["mean_diff"] = t.mean() - n.mean()
    if np.ptp(np.concatenate([t, n])) == 0:
        row["reason"] = "no_variation"
        return row
    res = stats.mannwhitneyu(t, n, alternative="two-sided")
    rng = np.random.default_rng([seed, j])
    bt = np.median(t[rng.integers(0, len(t), (n_boot, len(t)))], axis=1)
    bn = np.median(n[rng.integers(0, len(n), (n_boot, len(n)))], axis=1)
    row.update(U=res.statistic, p=res.pvalue, testable=True,
               ci_low=np.percentile(bt - bn, 2.5), ci_high=np.percentile(bt - bn, 97.5))
    return row


def unpaired_tests(X, labels, min_group=10, n_boot=N_BOOT, seed=SEED, n_jobs=-1):
    """Per-event Mann-Whitney U, tumour (label 1) vs normal (label 0), + bootstrap CI of the median difference."""
    labels = np.asarray(labels)
    Tm, Nm = X.values[labels == 1], X.values[labels == 0]
    rows = Parallel(n_jobs=n_jobs, batch_size=256)(
        delayed(_unpaired_one)(j, Tm[:, j], Nm[:, j], min_group, n_boot, seed) for j in range(X.shape[1]))
    out = pd.DataFrame(rows, index=pd.Index(X.columns, name="as_id"))
    out["fdr"] = bh(out["p"])
    return out


def bh(p):
    """Benjamini-Hochberg over the non-NaN p-values; NaN (untestable) stays NaN."""
    p = pd.Series(p, dtype=float)
    q = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.any():
        q[ok] = multipletests(p[ok].values, method="fdr_bh")[1]
    return q.values


def eta_squared(y, groups):
    """Share of the variance of y explained by a categorical grouping (one-way ANOVA R^2)."""
    y = pd.Series(np.asarray(y, dtype=float))
    g = pd.Series(np.asarray(groups))
    ok = y.notna() & g.notna()
    y, g = y[ok], g[ok]
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_between = y.groupby(g).agg(lambda v: len(v) * (v.mean() - y.mean()) ** 2).sum()
    return ss_between / ss_tot if ss_tot > 0 else np.nan


def top_events(res, n, fdr=0.05):
    """Significant events ranked by |median diff|, ties broken by p. For DISPLAY only."""
    sig = res[res["fdr"] < fdr].copy()
    sig["abs_eff"] = sig["median_diff"].abs()
    return sig.sort_values(["abs_eff", "p"], ascending=[False, True]).head(n).drop(columns="abs_eff")
