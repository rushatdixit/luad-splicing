"""Phase 3 checks: statistics functions on synthetic data, and the training-only pair file on disk."""
import os

import numpy as np
import pandas as pd
import pytest

from src.eda_stats import bh, eta_squared, paired_tests, unpaired_tests


def _pairs(n_pairs, shift, seed=0):
    rng = np.random.default_rng(seed)
    N = pd.DataFrame({"1": rng.uniform(0.2, 0.6, n_pairs)})
    T = N + shift + rng.normal(0, 0.02, (n_pairs, 1))
    return T, N


def test_paired_detects_known_shift():
    T, N = _pairs(40, 0.15)
    r = paired_tests(T, N, min_pairs=30, n_boot=200, n_jobs=1).loc["1"]
    assert r.testable and r.p < 1e-6
    assert 0.1 < r.median_diff < 0.2 and r.ci_low < r.median_diff < r.ci_high


def test_min_pairs_rule_and_zero_differences():
    T, N = _pairs(40, 0.15)
    T.iloc[:15, 0] = np.nan                            # only 25 complete pairs
    r = paired_tests(T, N, min_pairs=30, n_boot=50, n_jobs=1).loc["1"]
    assert not r.testable and r.reason.startswith("fewer_than_30") and np.isnan(r.p) and np.isnan(r.fdr)
    T2, N2 = _pairs(40, 0.0)
    T2[:] = N2.values                                  # identical -> all differences 0
    r2 = paired_tests(T2, N2, min_pairs=30, n_boot=50, n_jobs=1).loc["1"]
    assert not r2.testable and r2.reason == "all_differences_zero"


def test_unpaired_sign_and_min_group():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"1": np.r_[rng.uniform(0.6, 0.8, 50), rng.uniform(0.2, 0.4, 20)],
                      "2": np.r_[rng.uniform(0, 1, 50), [np.nan] * 15, rng.uniform(0, 1, 5)]})
    lab = np.r_[np.ones(50), np.zeros(20)]
    r = unpaired_tests(X, lab, min_group=10, n_boot=100, n_jobs=1)
    assert r.loc["1", "testable"] and r.loc["1", "median_diff"] > 0.2 and r.loc["1", "p"] < 1e-6
    assert not r.loc["2", "testable"]                 # only 5 normal values


def test_bh_keeps_nan_and_is_monotone():
    q = bh([0.01, np.nan, 0.04, 0.03])
    assert np.isnan(q[1]) and (q[[0, 2, 3]] >= [0.01, 0.04, 0.03]).all()


def test_eta_squared_bounds():
    assert eta_squared([1, 1, 5, 5], ["a", "a", "b", "b"]) == pytest.approx(1.0)
    assert eta_squared([1, 5, 1, 5], ["a", "a", "b", "b"]) == pytest.approx(0.0)


@pytest.mark.skipif(not os.path.exists("results/tables/stats_paired_train_pairs_used.csv"),
                    reason="run python -m src.run_eda first")
def test_train_only_pairs_contain_no_test_patient():
    used = pd.read_csv("results/tables/stats_paired_train_pairs_used.csv")["patient_id"]
    s = pd.read_csv("data/processed/splits.csv")
    test_pids = set(s.loc[s.set == "test", "patient_id"])
    assert len(used) == 46 and not set(used) & test_pids
