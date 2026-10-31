"""ONE-SHOT final test evaluation (protocol frozen in docs/analysis_plan.md §E, decisions.md #20).

    python -m src.final_test --dry-run   # identical code path on CV fold (rep 0, fold 0); no lock; output to --out-dir
    python -m src.final_test             # THE real run: creates results/test_lock.json and refuses to ever run again

Audit notes:
  * Everything is fitted first, on training samples only: coverage filter, paired ranking on the 46 training pairs,
    imputation/scaling (Pipeline), C (inner grouped CV), RF, and the sensitivity refit.
  * Only then is the lock created atomically (open(..., "x")). Test rows are indexed only after that line.
  * If the lock exists, the script exits before fitting anything.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.build_matrix import load_features, load_metadata
from src.models import coverage_filter, fold_indices, make_models, metrics, pairs_for_ranking, rank_events

SEED = 42
K = 25
MAX_MISSING, MIN_PAIRS = 0.20, 30          # Phase 3 (b) rule, frozen in decisions.md #20
N_BOOT = 2000
LOCK = "results/test_lock.json"
T = "results/tables"
SITE44_CLUSTER_TRAIN = ["TCGA_44_2656", "TCGA_44_2662", "TCGA_44_2666", "TCGA_44_2668",
                        "TCGA_44_3917", "TCGA_44_4112", "TCGA_44_6147"]
SITE44_CLUSTER_TEST = "TCGA_44_6146"
HASHED = ["src/final_test.py", "src/models.py", "src/eda_stats.py",
          "data/processed/splits.csv", "data/processed/psi_samples_x_events.csv.gz"]


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def acquire_lock(path, info):
    """Atomically create the lock file. Raises FileExistsError if the test set was already opened."""
    with open(path, "x") as fh:
        json.dump(info, fh, indent=2)


def patient_bootstrap(y, p, patients, metric, n_boot=N_BOOT, seed=SEED):
    """95% percentile CI, resampling PATIENTS with replacement (all of a patient's samples move together).

    Resamples without both classes cannot give an AUC; they are skipped and counted.
    """
    rng = np.random.default_rng(seed)
    pats = np.unique(patients)
    idx_of = {q: np.flatnonzero(patients == q) for q in pats}
    vals, skipped = [], 0
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[q] for q in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(y[idx])) < 2:
            skipped += 1
            continue
        vals.append(metric(y[idx], p[idx]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5), skipped


def fit_final(Xv, cols, label, patient, tr_idx):
    """Fit ranking + LR-L2 + RF + majority on training samples only. Returns features, ranking table and models."""
    rng = np.random.default_rng(SEED)
    t_idx, n_idx = pairs_for_ranking(tr_idx, label[tr_idx], patient, False, rng)
    assert set(t_idx) <= set(tr_idx) and set(n_idx) <= set(tr_idx)
    Xtr = Xv[tr_idx]
    keep = coverage_filter(Xtr, Xv[t_idx], Xv[n_idx], max_missing=MAX_MISSING, min_pairs=MIN_PAIRS)
    ranked, n_passed, table = rank_events(Xv[t_idx][:, keep], Xv[n_idx][:, keep], cols[keep], k_max=K,
                                          min_pairs=MIN_PAIRS, n_boot=N_BOOT, return_table=True)
    assert n_passed >= K, "fewer than k events passed the FDR gate"
    col_pos = {c: i for i, c in enumerate(cols)}
    f = [col_pos[c] for c in ranked]
    A, y = Xtr[:, f], label[tr_idx]
    fac = make_models(SEED)
    fitted = {}
    for name in ("logreg_l2", "random_forest", "majority"):
        m = fac[name]()
        if name == "logreg_l2":
            m.fit(A, y, groups=patient[tr_idx])
            fitted[name], best_c = m.best_estimator_, m.best_params_["clf__C"]
        else:
            fitted[name] = m.fit(A, y)
    info = {"n_train_samples": len(tr_idx), "n_train_normals": int((y == 0).sum()), "n_pairs": len(t_idx),
            "n_eligible": int(keep.sum()), "n_passed_fdr": n_passed, "best_C_lr_l2": float(best_c)}
    return {"features": ranked, "feat_pos": f, "table": table, "models": fitted, "info": info}


def predict(fit, Xv, idx):
    B = Xv[idx][:, fit["feat_pos"]]
    out = {}
    for name, pipe in fit["models"].items():
        assert not np.isnan(pipe[:-1].transform(B)).any(), "NaN reached a model"
        out[name] = pipe.predict_proba(B)[:, list(pipe.classes_).index(1)]
    return out


def evaluate(name, y, p, patients, model_desc):
    m = metrics(y, p)
    lo, hi, sk = patient_bootstrap(y, p, patients, roc_auc_score)
    plo, phi, _ = patient_bootstrap(y, p, patients, lambda a, b: average_precision_score(1 - a, 1 - b))
    return {"analysis": name, "model": model_desc, "n": len(y), "n_normal": int((y == 0).sum()),
            "n_patients": len(np.unique(patients)),
            "roc_auc": m["roc_auc"], "roc_auc_ci_low": lo, "roc_auc_ci_high": hi,
            "pr_auc_normal": m["pr_auc_normal"], "pr_auc_normal_ci_low": plo, "pr_auc_normal_ci_high": phi,
            "pr_auc_tumour": m["pr_auc_tumour"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
            "precision_tumour": m["precision_tumour"], "recall_tumour": m["recall_tumour"], "f1_tumour": m["f1_tumour"],
            "specificity": m["recall_normal"], "bootstrap_resamples_skipped": sk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=T)
    a = ap.parse_args()
    if not a.dry_run and os.path.exists(LOCK):
        sys.exit(f"REFUSING TO RUN: {LOCK} exists. The test set has already been opened once.")
    out = a.out_dir if a.dry_run else T
    os.makedirs(out, exist_ok=True)

    X, meta = load_features(), load_metadata()
    splits = pd.read_csv("data/processed/splits.csv", dtype={"site": str})
    assert list(X.index) == list(meta.sample_id) == list(splits.sample_id)
    Xv, cols = X.values, np.asarray(X.columns)
    label, patient, sid = meta.label.values, meta.patient_id.values, meta.sample_id.values

    if a.dry_run:   # identical code path; fit on a CV training part, "evaluate" on its validation fold
        tr, ev = fold_indices(meta, splits, np.ones(len(meta), bool), 0, 0)
    else:
        tr = np.flatnonzero(splits.set.values == "train")
    assert (splits.set.values[tr] == "train").all() and len(tr) == (460 if not a.dry_run else len(tr))

    # ---- fit everything on training data only --------------------------------------------------
    main_fit = fit_final(Xv, cols, label, patient, tr)
    tr_sens = tr[~np.isin(sid[tr], SITE44_CLUSTER_TRAIN)]
    assert len(tr) - len(tr_sens) == (7 if not a.dry_run else len(tr) - len(tr_sens))
    sens_fit = fit_final(Xv, cols, label, patient, tr_sens)

    annot = pd.read_csv("data/processed/event_annotation.csv", usecols=["as_id", "symbol", "splice_type"])
    annot["as_id"] = annot["as_id"].astype(str)
    rt = main_fit["table"].join(annot.set_index("as_id")).sort_values("rank")
    rt["in_final_top25"] = rt["rank"] <= K
    rt.to_csv(f"{out}/final_ranking_train.csv")
    pd.DataFrame({"rank": range(1, K + 1), "main_features": main_fit["features"],
                  "sensitivity_features": sens_fit["features"]}).to_csv(f"{out}/final_features.csv", index=False)

    # ---- open the test set: lock first --------------------------------------------------------
    if a.dry_run:
        print("DRY RUN: evaluating on CV rep 0 / fold 0 validation samples; no lock created.")
    else:
        lock_info = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
                     "protocol": "docs/analysis_plan.md §E; docs/decisions.md #20",
                     "sha256": {p: sha256(p) for p in HASHED},
                     "fit_info": {"main": main_fit["info"], "sensitivity": sens_fit["info"]}}
        acquire_lock(LOCK, lock_info)   # raises if it already exists
        ev = np.flatnonzero(splits.set.values == "test")
        assert len(ev) == 113
    assert not set(patient[ev]) & set(patient[tr]), "evaluation patient also in training"

    y_ev, pat_ev = label[ev], patient[ev]
    pm, ps = predict(main_fit, Xv, ev), predict(sens_fit, Xv, ev)
    keep112 = sid[ev] != SITE44_CLUSTER_TEST

    rows = [evaluate("primary: main LR-L2", y_ev, pm["logreg_l2"], pat_ev, "LR-L2 k=25"),
            evaluate("secondary: main RF", y_ev, pm["random_forest"], pat_ev, "RF k=25"),
            evaluate("baseline: majority class", y_ev, pm["majority"], pat_ev, "majority"),
            evaluate("sensitivity: main LR-L2, test without TCGA_44_6146", y_ev[keep112], pm["logreg_l2"][keep112],
                     pat_ev[keep112], "LR-L2 k=25"),
            evaluate("sensitivity: refit without site-44 cluster, test without TCGA_44_6146", y_ev[keep112],
                     ps["logreg_l2"][keep112], pat_ev[keep112], "LR-L2 k=25 (refit)")]
    mt = pd.DataFrame(rows)
    mt.to_csv(f"{out}/test_metrics.csv", index=False)

    pred = pd.DataFrame({"sample_id": sid[ev], "patient_id": pat_ev, "site": meta.site.values[ev], "label": y_ev,
                         "p_tumour_lr_l2": pm["logreg_l2"], "pred_lr_l2_at_0.5": (pm["logreg_l2"] >= 0.5).astype(int),
                         "p_tumour_rf": pm["random_forest"],
                         "p_tumour_lr_l2_sensitivity": np.where(keep112, ps["logreg_l2"], np.nan)})
    pred.to_csv(f"{out}/test_predictions.csv", index=False)

    if not a.dry_run:
        with open(LOCK) as fh:
            info = json.load(fh)
        info.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                    outputs_sha256={p: sha256(p) for p in [f"{T}/test_metrics.csv", f"{T}/test_predictions.csv"]})
        with open(LOCK, "w") as fh:
            json.dump(info, fh, indent=2)

    pd.set_option("display.width", 250)
    print(json.dumps({"main": main_fit["info"], "sensitivity": sens_fit["info"]}, indent=1))
    print(mt[["analysis", "n", "n_normal", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc_normal",
              "pr_auc_normal_ci_low", "pr_auc_normal_ci_high", "tn", "fp", "fn", "tp", "specificity",
              "bootstrap_resamples_skipped"]].round(3).to_string())


if __name__ == "__main__":
    main()
