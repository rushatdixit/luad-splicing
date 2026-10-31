"""Phase 4 checks: in-fold ranking, fold isolation, null-audit label handling, no test leakage."""
import os

import numpy as np
import pandas as pd
import pytest

from src.models import fold_indices, pairs_for_ranking, rank_events, run_fold


def _synthetic(n_pairs=40, n_tum_only=60, n_events=60, planted=(3, 7, 11), seed=0):
    """Paired + tumour-only patients; only the planted events differ tumour vs normal."""
    rng = np.random.default_rng(seed)
    pids = [f"P{i}" for i in range(n_pairs)] * 2 + [f"Q{i}" for i in range(n_tum_only)]
    label = np.r_[np.ones(n_pairs), np.zeros(n_pairs), np.ones(n_tum_only)].astype(int)
    X = rng.uniform(0.3, 0.7, (len(label), n_events))
    for j in planted:
        X[label == 1, j] += 0.25
    return X, np.asarray([str(c) for c in range(n_events)]), label, np.asarray(pids)


def test_ranking_finds_planted_events():
    X, cols, label, pid = _synthetic()
    t, n = pairs_for_ranking(np.arange(len(label)), label, pid, False, np.random.default_rng(0))
    ranked, n_pass = rank_events(X[t], X[n], cols, k_max=3)
    assert set(ranked) == {"3", "7", "11"} and n_pass >= 3


def test_fold_uses_training_patients_only():
    X, cols, label, pid = _synthetic()
    pats = np.unique(pid)
    va_p = set(pats[::5])
    va = np.flatnonzero(np.isin(pid, list(va_p)))
    tr = np.flatnonzero(~np.isin(pid, list(va_p)))
    t, n = pairs_for_ranking(tr, label[tr], pid, False, np.random.default_rng(0))
    assert not set(pid[t]) & va_p and not set(pid[n]) & va_p
    # Pairs really are the same patient, tumour vs normal.
    assert (pid[t] == pid[n]).all() and (label[t] == 1).all() and (label[n] == 0).all()
    # A fold that leaks a patient across train/validation must fail loudly.
    with pytest.raises(AssertionError):
        run_fold(X, cols, label, pid, np.r_[tr, va[:1]], va, False, 0)


def test_shuffled_pseudo_pairs_keep_class_counts():
    X, cols, label, pid = _synthetic()
    rng = np.random.default_rng(1)
    tr = np.arange(len(label))
    y_shuf = rng.permutation(label[tr])
    assert (np.bincount(y_shuf) == np.bincount(label[tr])).all()
    t, n = pairs_for_ranking(tr, y_shuf, pid, True, rng)
    assert len(t) == len(n) == (y_shuf == 0).sum() and len(set(t)) == len(t)


def test_fold_indices_exclude_test_set():
    meta = pd.read_csv("data/processed/sample_metadata.csv")
    s = pd.read_csv("data/processed/splits.csv")
    for r in range(10):
        for f in range(5):
            tr, va = fold_indices(meta, s, np.ones(len(meta), bool), r, f)
            assert (s.set.values[np.r_[tr, va]] == "train").all()
            assert not set(s.patient_id.values[tr]) & set(s.patient_id.values[va])


@pytest.mark.skipif(not os.path.exists("results/tables/cv_oof_predictions.csv.gz"),
                    reason="run python -m src.run_models_cv first")
def test_no_test_sample_in_cv_outputs():
    s = pd.read_csv("data/processed/splits.csv")
    o = pd.read_csv("results/tables/cv_oof_predictions.csv.gz", usecols=["sample_id"])
    assert not set(o.sample_id) & set(s.loc[s.set == "test", "sample_id"])
