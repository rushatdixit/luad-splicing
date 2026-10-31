"""Candidate-event shortlist (decisions.md #21). Run from the repo root:  python -m src.build_shortlist

Rule: selected in >= 80% of main-config CV folds at k = 25 AND among the final model's top-25 features.
Uses training-derived tables only (final ranking on the 46 training pairs, CV selection frequency); no test data.
Statistical columns only: no gene functions or biological claims.
"""
import pandas as pd

T = "results/tables"
K, MIN_FREQ = 25, 0.80


def main():
    rank = pd.read_csv(f"{T}/final_ranking_train.csv", dtype={"as_id": str})
    freq = pd.read_csv(f"{T}/cv_selection_frequency_by_event.csv", dtype={"as_id": str})
    freq = freq[(freq.config == "main") & (freq.k == K)][["as_id", "freq"]].rename(columns={"freq": "cv_selection_freq_k25"})
    s = rank[rank.in_final_top25].merge(freq, on="as_id", how="left").fillna({"cv_selection_freq_k25": 0.0})
    s = s[s.cv_selection_freq_k25 >= MIN_FREQ].copy()
    s["direction"] = s.median_diff.map(lambda d: "higher PSI in tumour" if d > 0 else "lower PSI in tumour")
    s = s.rename(columns={"symbol": "gene", "rank": "final_rank", "n_pairs": "n_training_pairs",
                          "median_diff": "median_dPSI_tumour_minus_normal", "ci_low": "median_dPSI_ci95_low",
                          "ci_high": "median_dPSI_ci95_high", "mean_diff": "mean_dPSI", "fdr": "fdr_bh_training_pairs"})
    cols = ["final_rank", "as_id", "gene", "splice_type", "direction", "median_dPSI_tumour_minus_normal",
            "median_dPSI_ci95_low", "median_dPSI_ci95_high", "mean_dPSI", "n_training_pairs", "fdr_bh_training_pairs",
            "cv_selection_freq_k25"]
    s = s[cols].sort_values("final_rank")
    assert (s.median_dPSI_ci95_low <= s.median_dPSI_tumour_minus_normal).all()
    s.to_csv(f"{T}/candidate_shortlist.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"{len(s)} candidate events (of {int(rank.in_final_top25.sum())} final features)")
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
