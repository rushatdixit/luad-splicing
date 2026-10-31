"""Robustness CV: main config restricted to well-covered events. Run from the repo root:
    python -m src.run_robust_lowmissing

Same 50 outer folds and fold seeds as Stage 1 (src/run_models_cv.py). The only change: the in-fold coverage
filter keeps events with <= 5% missing across the fold's training samples (Stage 1: <= 20%), still with
>= 25 complete training pairs. k = 25. Test set not used (models.fold_indices asserts it).
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.build_matrix import load_features, load_metadata
from src.models import fold_indices, run_fold

SEED = 42
MAX_MISSING_ROBUST = 0.05
K = 25
MODELS_ROBUST = ("majority", "logreg_l2", "logreg_l1", "random_forest")
T = "results/tables"


def main():
    X, meta = load_features(), load_metadata()
    splits = pd.read_csv("data/processed/splits.csv", dtype={"site": str})
    assert list(X.index) == list(meta.sample_id) == list(splits.sample_id)
    Xv, cols = X.values, np.asarray(X.columns)
    label, patient = meta.label.values, meta.patient_id.values
    mask = np.ones(len(meta), bool)
    jobs = [(r, f, *fold_indices(meta, splits, mask, r, f)) for r in range(10) for f in range(5)]
    out = Parallel(n_jobs=-1)(
        delayed(run_fold)(Xv, cols, label, patient, tr, va, False,
                          np.random.SeedSequence([SEED, r, f]).generate_state(1)[0],   # identical to Stage 1
                          max_missing=MAX_MISSING_ROBUST, ks=(K,), models=MODELS_ROBUST)
        for r, f, tr, va in jobs)
    rows = []
    for (r, f, tr, va), (m, _, sel) in zip(jobs, out):
        rows += [{"repeat": r, "fold": f, "n_eligible": sel["n_eligible"], "n_passed_fdr": sel["n_passed_fdr"], **x} for x in m]
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(f"{T}/cv_runs/robust_lowmissing_metrics.csv", index=False)

    main_s = pd.read_csv(f"{T}/cv_metrics_summary.csv")
    main_s = main_s[(main_s.config == "main") & (main_s.k == K)].set_index("model")
    main_sel = pd.read_csv(f"{T}/cv_runs/main_selected.csv").drop_duplicates(["repeat", "fold"])
    g = per_fold.groupby("model")
    res = pd.DataFrame({
        "k": K,
        "robust_max_missing": MAX_MISSING_ROBUST,
        "robust_roc_auc_mean": g.roc_auc.mean(), "robust_roc_auc_sd": g.roc_auc.std(),
        "robust_pr_auc_normal_mean": g.pr_auc_normal.mean(), "robust_pr_auc_normal_sd": g.pr_auc_normal.std(),
        "robust_n_folds_auc_gt_0.98": g.roc_auc.apply(lambda v: int((v > 0.98).sum())),
        "robust_mean_eligible_events": g.n_eligible.mean(),
        "main_max_missing": 0.20,
        "main_roc_auc_mean": main_s.roc_auc_mean, "main_roc_auc_sd": main_s.roc_auc_sd,
        "main_pr_auc_normal_mean": main_s.pr_auc_normal_mean, "main_pr_auc_normal_sd": main_s.pr_auc_normal_sd,
        "main_mean_eligible_events": main_sel.n_eligible.mean(),
    })
    res.index.name = "model"
    res.to_csv(f"{T}/cv_robust_lowmissing.csv")
    pd.set_option("display.width", 250)
    print(res.round(4).to_string())


if __name__ == "__main__":
    main()
