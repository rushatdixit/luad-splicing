"""Report-ready figures for the final results. Run from the repo root:  python -m src.report_figures

Reads saved outputs only (test_predictions.csv, test_metrics.csv, candidate_shortlist.csv, cv_metrics_per_fold.csv).
It does NOT refit anything or re-evaluate the test set; the one-shot lock is untouched.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plotting import CLASS_COLOR, CLASS_MARKER, INK2, MUTED, TUMOUR, savefig

T = "results/tables"
SEED = 42


def fig_test_scores(pred):
    """Held-out P(tumour) by class, LR-L2 (primary) and RF (secondary), with the fixed 0.5 threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    rng = np.random.default_rng(SEED)
    for ax, (col, title) in zip(axes, [("p_tumour_lr_l2", "Primary: LR-L2, k = 25"), ("p_tumour_rf", "Secondary: RF, k = 25")]):
        for i, (c, lab) in enumerate([("normal", 0), ("tumour", 1)]):
            v = pred.loc[pred.label == lab, col]
            ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=18 if lab == 0 else 10, c=CLASS_COLOR[c],
                       marker=CLASS_MARKER[c], alpha=0.75, linewidths=0.3, edgecolors="white")
        ax.axhline(0.5, ls="--", lw=0.8, c=INK2)
        ax.text(1.45, 0.5, "threshold 0.5", va="bottom", ha="right", fontsize=7, color=INK2)
        wrong = pred[(pred[col] >= 0.5).astype(int) != pred.label]
        for r in wrong.itertuples():
            ax.annotate(r.sample_id, (r.label, getattr(r, col)), xytext=(8, -12), textcoords="offset points",
                        fontsize=7, color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
        n0, n1 = int((pred.label == 0).sum()), int((pred.label == 1).sum())
        ax.set_xticks([0, 1], [f"normal (n={n0})", f"tumour (n={n1})"])
        ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.02, 1.02)
        ax.set_title(title)
    axes[0].set_ylabel("predicted P(tumour), held-out test set")
    fig.suptitle("Held-out test set (102 patients, evaluated once): predicted probabilities", y=1.02)
    fig.tight_layout()
    savefig(fig, "final_test_predicted_probabilities")


def fig_cv_vs_test(m, tm):
    """CV fold ROC-AUCs (main config, k = 25) next to the single held-out test ROC-AUC, per model."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    rng = np.random.default_rng(SEED)
    rows = [("logreg_l2", "LR-L2", "primary: main LR-L2"), ("random_forest", "RF", "secondary: main RF"),
            ("majority", "majority", "baseline: majority class")]
    for i, (model, lab, test_name) in enumerate(rows):
        v = m[(m.config == "main") & (m.model == model) & (m.k == 25)].roc_auc
        ax.scatter(i - 0.12 + rng.uniform(-0.08, 0.08, len(v)), v, s=8, c=MUTED, alpha=0.7, linewidths=0,
                   label="CV validation folds (50)" if i == 0 else None)
        t = tm.set_index("analysis").loc[test_name]
        ax.errorbar(i + 0.15, t.roc_auc, yerr=[[t.roc_auc - t.roc_auc_ci_low], [t.roc_auc_ci_high - t.roc_auc]],
                    fmt="D", ms=6, c=TUMOUR, capsize=3, label="held-out test (95% patient bootstrap CI)" if i == 0 else None)
    ax.axhline(0.5, ls="--", lw=0.8, c=INK2)
    ax.set_xticks(range(len(rows)), [r[1] for r in rows])
    ax.set_ylim(0.45, 1.03); ax.set_ylabel("ROC-AUC")
    ax.set_title("Cross-validation (training patients) vs held-out test, k = 25")
    ax.legend(loc="center right", fontsize=7)
    savefig(fig, "final_cv_vs_test_auc")


def fig_shortlist(s):
    """Median paired dPSI (tumour - normal, 46 training pairs) with 95% bootstrap CI for the candidate events."""
    s = s.sort_values("median_dPSI_tumour_minus_normal")
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(s) + 1.4))
    y = np.arange(len(s))
    ax.errorbar(s.median_dPSI_tumour_minus_normal, y,
                xerr=[s.median_dPSI_tumour_minus_normal - s.median_dPSI_ci95_low,
                      s.median_dPSI_ci95_high - s.median_dPSI_tumour_minus_normal],
                fmt="o", ms=4, c=TUMOUR, ecolor=TUMOUR, elinewidth=1.2, capsize=2)
    ax.axvline(0, c=INK2, lw=0.8)
    ax.set_yticks(y, [f"{r.gene} {r.splice_type} {r.as_id}" for r in s.itertuples()], fontsize=7)
    for yy, r in zip(y, s.itertuples()):
        ax.text(1.02, yy, f"{r.cv_selection_freq_k25:.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=INK2)
    ax.text(1.02, len(s) - 0.3, "CV freq.", transform=ax.get_yaxis_transform(), fontsize=7, color=INK2, va="bottom")
    ax.set_xlim(-1, 1)
    ax.set_xlabel(f"median paired ΔPSI, tumour − normal (training pairs, {s.n_training_pairs.min()}-"
                  f"{s.n_training_pairs.max()} complete per event; 95% bootstrap CI)")
    ax.set_title(f"{len(s)} candidate events (≥ 80% CV selection and in final top-25)", fontsize=9)
    savefig(fig, "final_candidate_shortlist_effects")


def main():
    pred = pd.read_csv(f"{T}/test_predictions.csv")
    tm = pd.read_csv(f"{T}/test_metrics.csv")
    m = pd.read_csv(f"{T}/cv_metrics_per_fold.csv")
    s = pd.read_csv(f"{T}/candidate_shortlist.csv", dtype={"as_id": str})
    fig_test_scores(pred)
    fig_cv_vs_test(m, tm)
    fig_shortlist(s)
    print("wrote final_test_predicted_probabilities, final_cv_vs_test_auc, final_candidate_shortlist_effects")


if __name__ == "__main__":
    main()
