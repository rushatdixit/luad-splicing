"""Phase 2: analysis-ready table. Run from the repo root:  python -m src.build_matrix

Writes:
  data/processed/psi_samples_x_events.csv.gz   rows = samples, columns = events (as_id), PSI only
  data/processed/sample_metadata.csv           label, patient, site, pair flag (kept OUT of the feature table)

Deliberately applies NO event filtering, imputation or scaling. Any such step uses the data, so it must be
fitted inside training folds later (CLAUDE.md leakage rules). Only a transpose and a metadata join happen here.
"""
import numpy as np
import pandas as pd

from src.load_data import sample_sheet

PROC = "data/processed"
N_EVENTS, N_SAMPLES, N_NAN = 27836, 573, 540861  # from the Phase 1 inventory


def load_features(path=f"{PROC}/psi_samples_x_events.csv.gz"):
    """Load the samples x events PSI table (index sample_id, columns as_id as str), all float.

    Forcing float matters: events that are PSI = 1 in every sample (no NaN) would otherwise load as int.
    """
    return pd.read_csv(path, index_col="sample_id").astype(float)


def load_metadata(path=f"{PROC}/sample_metadata.csv"):
    return pd.read_csv(path, dtype={"site": str})


def main():
    psi = pd.read_csv(f"{PROC}/psi_events_x_samples.csv.gz", index_col="as_id")
    X = psi.T
    X.index.name = "sample_id"
    X.columns = X.columns.astype(str)
    X.columns.name = None

    meta = sample_sheet(list(X.index))
    # normal_site: the site contributes at least one normal. This defines the pre-specified 7-site
    # sensitivity population. It is derived from labels, so it must NEVER be used as a feature.
    meta["normal_site"] = meta.groupby("site")["label"].transform(lambda s: (s == 0).any())

    assert X.shape == (N_SAMPLES, N_EVENTS), X.shape
    assert int(X.isna().sum().sum()) == N_NAN
    assert np.nanmin(X.values) >= 0 and np.nanmax(X.values) <= 1
    assert list(meta["sample_id"]) == list(X.index), "metadata and feature rows are not aligned"
    # Feature columns must be event IDs only: no label, patient, site or clinical field slipped in.
    assert all(c.isdigit() for c in X.columns), "non-event column in the feature table"
    assert sorted(meta.loc[meta["normal_site"], "site"].unique()) == sorted(["38", "44", "49", "50", "55", "73", "91"])

    X.to_csv(f"{PROC}/psi_samples_x_events.csv.gz", float_format="%.4g")
    meta.to_csv(f"{PROC}/sample_metadata.csv", index=False)
    print(f"features: {X.shape[0]} samples x {X.shape[1]} events; NaN cells {int(X.isna().sum().sum()):,}")
    print(meta.groupby(["normal_site", "sample_type"]).size().to_string())


if __name__ == "__main__":
    main()
