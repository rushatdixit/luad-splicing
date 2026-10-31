"""Phase 4 Stage 1: leakage-aware cross-validation on TRAINING patients only.

Audit map (every data-dependent step happens inside run_fold, using training-fold samples only):
  * pairs_for_ranking: tumour/normal pairs drawn ONLY from the training fold (asserted).
  * coverage_filter:   missingness and pair counts computed on the training fold only.
  * rank_events:       paired Wilcoxon on the training-fold pairs. Keep in-fold BH-FDR < 0.05, rank by
                       |median dPSI| (not p), top up if fewer than k pass.
  * make_models:       sklearn Pipelines (median imputer -> scaler -> classifier), so imputation and scaling
                       are also refit inside every inner tuning fold.
  * The test set never enters: fold_indices returns only samples with set == "train" (asserted).
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eda_stats import paired_tests

SEED = 42
KS = (10, 25, 50, 100)
MAX_MISSING = 0.20      # coverage filter: max fraction missing among training-fold samples
MIN_PAIRS = 25          # coverage filter: min complete training-fold pairs
FDR_GATE = 0.05
THRESHOLD = 0.5         # on P(tumour); fixed, never tuned
C_GRID = np.logspace(-3, 2, 6)
MODELS = ("majority", "logreg_l2", "logreg_l1", "random_forest")


def pairs_for_ranking(tr_idx, y_tr, patient, shuffled, rng):
    """Return (tumour_idx, normal_idx) of pairs used for ranking, all drawn from the training fold.

    Real labels: patients in the training fold with exactly one tumour and one normal sample.
    Shuffled labels (null audit): every pseudo-'normal' is paired with a random pseudo-'tumour'
    (without replacement), so the identical paired ranking code runs under the null.
    """
    if shuffled:
        nrm = tr_idx[y_tr == 0]
        tum = rng.choice(tr_idx[y_tr == 1], size=len(nrm), replace=False)
        return tum, nrm
    df = pd.DataFrame({"i": tr_idx, "y": y_tr, "p": patient[tr_idx]})
    both = df.groupby("p")["y"].agg(lambda s: sorted(s) == [0, 1])
    pids = both.index[both]
    sub = df[df.p.isin(pids)].sort_values(["p", "y"])
    return sub.loc[sub.y == 1, "i"].values, sub.loc[sub.y == 0, "i"].values


def coverage_filter(Xtr, Xt, Xn, max_missing=MAX_MISSING, min_pairs=MIN_PAIRS):
    """Boolean mask of events: <= max_missing missing in the training data AND >= min_pairs complete pairs."""
    ok_missing = np.isnan(Xtr).mean(axis=0) <= max_missing
    ok_pairs = (~np.isnan(Xt) & ~np.isnan(Xn)).sum(axis=0) >= min_pairs
    return ok_missing & ok_pairs


def rank_events(Xt, Xn, cols, k_max=max(KS), min_pairs=MIN_PAIRS, n_boot=0, return_table=False):
    """In-fold ranking. Gate: BH-FDR < 0.05; order: |median dPSI| then |mean dPSI| (effect size, not p).

    Returns (ranked event ids, n events passing the gate). If fewer than k_max pass, the list is topped
    up with the remaining events in the same effect-size order. Callers log when top-up affects a given k.
    return_table=True also returns the full per-event statistics (used by the final model, with n_boot CIs).
    """
    T = pd.DataFrame(Xt, columns=cols)
    N = pd.DataFrame(Xn, columns=cols)
    res = paired_tests(T, N, min_pairs=min_pairs, n_boot=n_boot, n_jobs=1 if n_boot == 0 else -1)
    res["abs_med"], res["abs_mean"] = res.median_diff.abs(), res.mean_diff.abs()
    passed = res.testable & (res.fdr < FDR_GATE)
    order = ["abs_med", "abs_mean"]
    top = res[passed].sort_values(order, ascending=False, kind="stable")
    rest = res[~passed].sort_values(["testable", *order], ascending=False, kind="stable")
    ranked = list(top.index) + list(rest.index)
    if return_table:
        res["passed_fdr_gate"] = passed
        res["rank"] = pd.Series(range(1, len(ranked) + 1), index=ranked)
        return ranked[:k_max], int(passed.sum()), res
    return ranked[:k_max], int(passed.sum())


def make_models(groups_inner_seed=SEED):
    """Model factories. Each is a Pipeline, so the imputer/scaler are fit on training data only."""
    pre = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=groups_inner_seed)

    def logreg(penalty):
        # sklearn >= 1.8: the penalty is set via l1_ratio (0 = L2, 1 = L1); `penalty=` is deprecated and,
        # if combined with the default l1_ratio=0, silently gives L2. Set it explicitly.
        l1_ratio, solver = (0.0, "lbfgs") if penalty == "l2" else (1.0, "liblinear")
        pipe = Pipeline(pre + [("clf", LogisticRegression(l1_ratio=l1_ratio, solver=solver, class_weight="balanced",
                                                          max_iter=5000))])
        return GridSearchCV(pipe, {"clf__C": C_GRID}, scoring="roc_auc", cv=inner, n_jobs=1)

    return {
        "majority": lambda: Pipeline(pre + [("clf", DummyClassifier(strategy="most_frequent"))]),
        "logreg_l2": lambda: logreg("l2"),
        "logreg_l1": lambda: logreg("l1"),
        "random_forest": lambda: Pipeline(pre + [("clf", RandomForestClassifier(
            n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=1))]),
    }


def metrics(y, p):
    """Fold metrics. Positive class = tumour (1); PR-AUC is also given with normal as positive."""
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": roc_auc_score(y, p), "pr_auc_tumour": average_precision_score(y, p),
            "pr_auc_normal": average_precision_score(1 - y, 1 - p),
            "precision_tumour": precision_score(y, pred, zero_division=0),
            "recall_tumour": recall_score(y, pred, zero_division=0),
            "f1_tumour": f1_score(y, pred, zero_division=0),
            "recall_normal": recall_score(1 - y, 1 - pred, zero_division=0),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "n_val": len(y), "n_val_normal": int((y == 0).sum())}


def run_fold(Xv, cols, label, patient, tr_idx, va_idx, shuffled, fold_seed,
             max_missing=MAX_MISSING, ks=KS, models=MODELS):
    """One outer fold. Returns (metric rows, oof rows, selection rows). Uses training-fold data only."""
    rng = np.random.default_rng(fold_seed)
    assert not set(patient[tr_idx]) & set(patient[va_idx]), "patient in both training and validation fold"
    y_tr, y_va = label[tr_idx].copy(), label[va_idx]
    if shuffled:
        y_tr = rng.permutation(y_tr)    # class counts preserved; validation labels stay real
    t_idx, n_idx = pairs_for_ranking(tr_idx, y_tr, patient, shuffled, rng)
    assert set(t_idx) <= set(tr_idx) and set(n_idx) <= set(tr_idx), "ranking pair outside training fold"
    assert not set(patient[t_idx]) & set(patient[va_idx]) and not set(patient[n_idx]) & set(patient[va_idx])

    Xtr = Xv[tr_idx]
    keep = coverage_filter(Xtr, Xv[t_idx], Xv[n_idx], max_missing=max_missing)
    ecols = cols[keep]
    ranked, n_passed = rank_events(Xv[t_idx][:, keep], Xv[n_idx][:, keep], ecols)
    col_pos = {c: i for i, c in enumerate(cols)}

    mrows, orows = [], []
    sel = {"n_pairs_ranking": len(t_idx), "n_eligible": int(keep.sum()), "n_passed_fdr": n_passed, "ranked": ranked}
    factories = {m: f for m, f in make_models(fold_seed).items() if m in models}
    groups_tr = patient[tr_idx]
    for k in ks:
        f = [col_pos[c] for c in ranked[:k]]
        assert all(str(c).isdigit() for c in ranked[:k]), "non-event feature"
        A, B = Xtr[:, f], Xv[va_idx][:, f]
        for name, make in factories.items():
            m = make()
            if isinstance(m, GridSearchCV):
                m.fit(A, y_tr, groups=groups_tr)
                pipe, best_c = m.best_estimator_, m.best_params_["clf__C"]
            else:
                m.fit(A, y_tr)
                pipe, best_c = m, np.nan
            pre = pipe[:-1]
            assert not np.isnan(pre.transform(A)).any() and not np.isnan(pre.transform(B)).any(), "NaN reached a model"
            p = pipe.predict_proba(B)[:, list(pipe.classes_).index(1)]
            mrows.append({"model": name, "k": k, "topped_up": n_passed < k, "best_C": best_c, **metrics(y_va, p)})
            orows += [{"model": name, "k": k, "idx": int(i), "p_tumour": float(pp)} for i, pp in zip(va_idx, p)]
    return mrows, orows, sel


def fold_indices(meta, splits, mask, rep, fold):
    """Training/validation sample indices for one outer fold, restricted to `mask` and to set == 'train'."""
    in_cv = mask & (splits["set"].values == "train")
    col = splits[f"cv_rep{rep}"].values
    tr = np.flatnonzero(in_cv & (col != fold))
    va = np.flatnonzero(in_cv & (col == fold))
    assert (splits["set"].values[np.r_[tr, va]] == "train").all(), "test-set sample entered CV"
    return tr, va
