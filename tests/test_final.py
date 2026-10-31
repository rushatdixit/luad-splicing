"""Checks for the one-shot test script. These never run the real evaluation."""
import os

import numpy as np
import pandas as pd
import pytest

from src.final_test import acquire_lock, patient_bootstrap


def test_lock_refuses_second_acquire(tmp_path):
    p = tmp_path / "lock.json"
    acquire_lock(p, {"status": "started"})
    with pytest.raises(FileExistsError):
        acquire_lock(p, {"status": "started"})


def test_patient_bootstrap_keeps_patients_together():
    y = np.array([1, 0, 1, 0, 1, 1, 1])
    pats = np.array(["a", "a", "b", "b", "c", "d", "e"])
    seen = []

    def metric(yy, pp):
        # each resample must contain whole patients: 'a' and 'b' contribute a tumour AND a normal
        seen.append(len(yy))
        return 0.5

    lo, hi, skipped = patient_bootstrap(y, np.zeros(len(y)), pats, metric, n_boot=200)
    assert lo == hi == 0.5 and skipped + len(seen) == 200 and min(seen) >= 6


@pytest.mark.skipif(not os.path.exists("results/tables/test_predictions.csv"), reason="final test not run yet")
def test_test_predictions_are_test_patients_only():
    pred = pd.read_csv("results/tables/test_predictions.csv")
    s = pd.read_csv("data/processed/splits.csv")
    assert len(pred) == 113
    assert set(pred.sample_id) == set(s.loc[s.set == "test", "sample_id"])
    assert not set(pred.patient_id) & set(s.loc[s.set == "train", "patient_id"])
