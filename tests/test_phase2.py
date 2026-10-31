"""Phase 2 checks on the saved outputs. Run from the repo root:  python -m pytest -q

These re-check the saved files (not just the in-memory objects), so they catch a stale or hand-edited output.
"""
import pandas as pd
import pytest

from src.build_matrix import load_features, load_metadata
from src.splits import N_FOLDS, N_REPEATS, check_splits, make_splits

CV = [f"cv_rep{r}" for r in range(N_REPEATS)]


@pytest.fixture(scope="module")
def X():
    return load_features()


@pytest.fixture(scope="module")
def meta():
    return load_metadata()


@pytest.fixture(scope="module")
def splits():
    return pd.read_csv("data/processed/splits.csv", dtype={"site": str})


def test_matrix_shape_and_alignment(X, meta, splits):
    assert X.shape == (573, 27836)
    assert list(X.index) == list(meta.sample_id) == list(splits.sample_id)
    assert (meta.label == 1).sum() == 514 and (meta.label == 0).sum() == 59


def test_features_are_psi_only(X):
    # No label/patient/site/clinical column in the feature table, and values are valid PSI.
    assert all(c.isdigit() for c in X.columns)
    assert X.dtypes.map(pd.api.types.is_float_dtype).all()
    assert X.min().min() >= 0 and X.max().max() <= 1


def test_nan_count_unchanged(X):
    # NaN is expected (no imputation yet); the count must match the raw-file inventory exactly.
    assert int(X.isna().sum().sum()) == 540861


def test_no_patient_in_train_and_test(splits):
    train = set(splits.loc[splits.set == "train", "patient_id"])
    test = set(splits.loc[splits.set == "test", "patient_id"])
    assert train and test and not (train & test)


def test_pairs_kept_together(splits):
    for col in ["set", *CV]:
        assert splits.groupby("patient_id")[col].nunique().eq(1).all(), col


def test_test_samples_never_in_cv(splits):
    assert (splits.loc[splits.set == "test", CV] == -1).all().all()
    assert splits.loc[splits.set == "train", CV].isin(range(N_FOLDS)).all().all()


def test_saved_splits_match_regeneration(meta, splits):
    # The file on disk must equal what the seeded generator produces (no hand edits, no drift).
    regen = make_splits(meta)
    check_splits(regen)
    pd.testing.assert_frame_equal(regen.reset_index(drop=True), splits, check_dtype=False)
