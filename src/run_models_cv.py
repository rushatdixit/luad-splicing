"""Phase 4 Stage 1: run patient-grouped CV on training patients only. Run from the repo root:
    python -m src.run_models_cv            (reuses cached per-config results if present)
    python -m src.run_models_cv --force    (recompute everything)

Configs share the frozen outer folds in data/processed/splits.csv:
  main            all training patients
  sens_7site      training/validation samples from the 7 normal-contributing sites only
  sens_no_site44  without the 8 site-44 cluster tumours (their normals stay, as unpaired normals)
  null_shuffled   main, but training labels permuted within each fold (pseudo-pairs); validation labels real
The test set is never used here (asserted in models.fold_indices).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from src.build_matrix import load_features, load_metadata
from src.eda_stats import top_events
from src.models import KS, MODELS, fold_indices, run_fold
from src.plotting import INK2, MUTED, NORMAL, THIRD, TUMOUR, savefig

SEED = 42
N_REPEATS, N_FOLDS = 10, 5
SITE44_CLUSTER = ["TCGA_44_2656", "TCGA_44_2662", "TCGA_44_2666", "TCGA_44_2668",
                  "TCGA_44_3917", "TCGA_44_4112", "TCGA_44_6146", "TCGA_44_6147"]
T, RUNS = "results/tables", "results/tables/cv_runs"
CONFIGS = ["main", "sens_7site", "sens_no_site44", "null_shuffled"]
MODEL_LABEL = {"majority": "majority", "logreg_l2": "LR-L2", "logreg_l1": "LR-L1", "random_forest": "RF"}


def config_mask(name, meta):
    if name == "sens_7site":
        return meta.normal_site.values
    if name == "sens_no_site44":
        m = ~meta.sample_id.isin(SITE44_CLUSTER).values
        assert (~m).sum() == 8 and (meta.label.values[~m] == 1).all()
        return m
    return np.ones(len(meta), bool)


def run_config(name, Xv, cols, meta, splits):
    mask = config_mask(name, meta)
    label, patient = meta.label.values, meta.patient_id.values
    jobs = []
    for r in range(N_REPEATS):
        for f in range(N_FOLDS):
            tr, va = fold_indices(meta, splits, mask, r, f)
            jobs.append((r, f, tr, va))
    t0 = time.time()
    out = Parallel(n_jobs=-1, verbose=0)(
        delayed(run_fold)(Xv, cols, label, patient, tr, va, name == "null_shuffled",
                          np.random.SeedSequence([SEED, r, f]).generate_state(1)[0])
        for r, f, tr, va in jobs)
    print(f"  {name}: {len(jobs)} folds in {time.time() - t0:.0f}s", flush=True)
    mrows, orows, srows = [], [], []
    for (r, f, tr, va), (m, o, s) in zip(jobs, out):
        mrows += [{"config": name, "repeat": r, "fold": f, "n_train": len(tr),
                   "n_train_normal": int((label[tr] == 0).sum()), **x} for x in m]
        orows += [{"config": name, "repeat": r, "fold": f, **x} for x in o]
        srows += [{"config": name, "repeat": r, "fold": f, "rank": i + 1, "as_id": e,
                   "n_pairs_ranking": s["n_pairs_ranking"], "n_eligible": s["n_eligible"],
                   "n_passed_fdr": s["n_passed_fdr"]} for i, e in enumerate(s["ranked"])]
    o = pd.DataFrame(orows)
    o.insert(3, "sample_id", meta.sample_id.values[o.idx.values])
    o["label"] = label[o.idx.values]
    return pd.DataFrame(mrows), o.drop(columns="idx"), pd.DataFrame(srows)


def summarize(m):
    cols = ["roc_auc", "pr_auc_tumour", "pr_auc_normal", "precision_tumour", "recall_tumour", "f1_tumour",
            "recall_normal", "tn", "fp", "fn", "tp"]
    g = m.groupby(["config", "model", "k"])[cols]
    s = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    s["n_folds"] = m.groupby(["config", "model", "k"]).size()
    s["n_folds_auc_gt_0.98"] = m.assign(h=m.roc_auc > 0.98).groupby(["config", "model", "k"]).h.sum()
    s["n_folds_topped_up"] = m.groupby(["config", "model", "k"]).topped_up.sum()
    order = [c for pair in zip([f"{c}_mean" for c in cols], [f"{c}_sd" for c in cols]) for c in pair]
    return s[order + ["n_folds", "n_folds_auc_gt_0.98", "n_folds_topped_up"]].reset_index()


def selection_frequency(sel, annot, b_ranked):
    ev_rows, gene_rows = [], []
    for cfg, s in sel.groupby("config"):
        n_folds = s.groupby(["repeat", "fold"]).ngroups
        for k in KS:
            top = s[s["rank"] <= k]
            freq = (top.groupby("as_id").size() / n_folds).rename("freq").reset_index()
            freq = freq.merge(annot, on="as_id", how="left")
            freq["in_phase3b_topk"] = freq.as_id.isin(b_ranked[:k])
            freq.insert(0, "k", k)
            freq.insert(0, "config", cfg)
            ev_rows.append(freq.sort_values("freq", ascending=False))
            per_fold_gene = top.merge(annot, on="as_id").groupby(["repeat", "fold"]).symbol.apply(set)
            genes = freq.groupby("symbol").agg(n_events_ever=("as_id", "nunique"), max_event_freq=("freq", "max"))
            genes["any_event_freq"] = [sum(g in st for st in per_fold_gene) / n_folds for g in genes.index]
            genes = genes.reset_index().sort_values(["any_event_freq", "max_event_freq"], ascending=False)
            genes.insert(0, "k", k)
            genes.insert(0, "config", cfg)
            gene_rows.append(genes)
    return pd.concat(ev_rows), pd.concat(gene_rows)


# ---- figures ------------------------------------------------------------------------------------
def fig_metric(m, metric, ylabel, name, baseline=None):
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(16, 4), sharey=True)
    rng = np.random.default_rng(SEED)
    for ax, cfg in zip(axes, CONFIGS):
        sub = m[m.config == cfg]
        x, labels = 0, []
        for model in MODELS:
            for k in KS:
                v = sub[(sub.model == model) & (sub.k == k)][metric].values
                col = MUTED if model == "majority" else TUMOUR
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=6, c=col, alpha=0.45, linewidths=0)
                ax.plot([x - 0.3, x + 0.3], [v.mean()] * 2, c=INK2, lw=2)
                labels.append(f"{MODEL_LABEL[model]} k={k}")
                x += 1
            x += 0.6
        pos = [i + 0.6 * (i // len(KS)) for i in range(len(labels))]
        ax.set_xticks(pos, labels, rotation=90, fontsize=7)
        if baseline is not None:
            ax.axhline(baseline, ls="--", lw=0.8, c=INK2)
        ax.axhline(0.98, ls=":", lw=0.8, c=NORMAL)
        ax.set_title(cfg, fontsize=9)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{ylabel} per validation fold (dots) and mean (bar); dotted line = 0.98 flag", y=1.02)
    fig.tight_layout()
    savefig(fig, name)


def fig_null(m):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, model in zip(axes, ["logreg_l2", "logreg_l1", "random_forest"]):
        for cfg, col, lab in [("null_shuffled", MUTED, "shuffled training labels"), ("main", TUMOUR, "real labels")]:
            v = m[(m.config == cfg) & (m.model == model) & (m.k == 25)].roc_auc
            ax.hist(v, bins=bins, color=col, alpha=0.85, label=f"{lab} (mean {v.mean():.2f})")
        ax.axvline(0.5, ls="--", lw=0.8, c=INK2)
        ax.set_title(f"{MODEL_LABEL[model]}, k=25"); ax.set_xlabel("validation ROC-AUC")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("folds")
    fig.suptitle("Leakage audit: ROC-AUC with shuffled training labels vs real labels (50 folds each)", y=1.03)
    fig.tight_layout()
    savefig(fig, "cv_null_audit_auc")


def fig_selection(freq, annot):
    f = freq[(freq.config == "main") & (freq.k == 25)].head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 7))
    ax.barh(range(len(f)), f.freq, color=TUMOUR, height=0.7)
    ax.set_yticks(range(len(f)), [f"{r.symbol} {r.splice_type} {r.as_id}" for r in f.itertuples()], fontsize=7)
    ax.set_xlim(0, 1); ax.set_xlabel("fraction of the 50 outer folds where selected (top-25)")
    ax.set_title("Selection stability, main config, k=25 (top 30 events)")
    savefig(fig, "cv_selection_frequency_main_k25")


def fig_roc(o):
    from sklearn.metrics import roc_curve
    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for model, col in [("logreg_l2", TUMOUR), ("logreg_l1", NORMAL), ("random_forest", THIRD)]:
        tprs = []
        for (r, f), g in o[(o.config == "main") & (o.model == model) & (o.k == 25)].groupby(["repeat", "fold"]):
            fpr, tpr, _ = roc_curve(g.label, g.p_tumour)
            tprs.append(np.interp(grid, fpr, tpr))
        mt = np.mean(tprs, axis=0)
        ax.plot(grid, mt, lw=2, c=col, label=MODEL_LABEL[model])
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, c=INK2)
    ax.set_xlabel("false positive rate (normals called tumour)"); ax.set_ylabel("true positive rate (tumours)")
    ax.set_title("Mean validation ROC, main config, k=25"); ax.legend(loc="lower right")
    savefig(fig, "cv_mean_roc_main_k25")


def null_label_overlap(meta, splits, m):
    """Null-audit diagnostic: how many REAL normals each fold's permutation left in the shuffled 'normal' class.

    Re-derives each fold's permutation with the same seed as run_fold. A strong true signal plus chance
    overlap with the true labels explains null AUCs drifting above 0.5 (see model_report.md §3.1).
    """
    lab, rows = meta.label.values, []
    for r in range(N_REPEATS):
        for f in range(N_FOLDS):
            tr, _ = fold_indices(meta, splits, np.ones(len(meta), bool), r, f)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, r, f]).generate_state(1)[0])
            y = rng.permutation(lab[tr])     # first draw of the fold rng, exactly as in models.run_fold
            rows.append({"repeat": r, "fold": f,
                         "real_normals_in_shuffled_normal": int(((y == 0) & (lab[tr] == 0)).sum()),
                         "expected": (y == 0).sum() * (lab[tr] == 0).mean()})
    d = pd.DataFrame(rows)
    nm = m[m.config == "null_shuffled"].pivot_table(index=["repeat", "fold"], columns=["model", "k"], values="roc_auc")
    nm.columns = [f"null_auc_{a}_k{b}" for a, b in nm.columns]
    return d.merge(nm.reset_index(), on=["repeat", "fold"])


def oof_mean_prob(o):
    """Mean out-of-fold P(tumour) per sample across the 10 repeats (main config, k = 25), with clinical context."""
    clin = pd.read_csv("data/processed/clinical_by_sample.csv", dtype={"site": str})
    s = o[(o.config == "main") & (o.k == 25) & (o.model != "majority")]
    p = s.groupby(["sample_id", "model"]).p_tumour.mean().unstack()
    p = p.join(clin.set_index("sample_id")[["label", "site", "is_paired", "Absolute_Purity_Value", "Pathologic_Tumor_Stage"]])
    p["wrong_any"] = (p[["logreg_l2", "logreg_l1", "random_forest"]] >= 0.5).astype(int).ne(p.label, axis=0).any(axis=1)
    return p


def main():
    force = "--force" in sys.argv
    os.makedirs(RUNS, exist_ok=True)
    X, meta = load_features(), load_metadata()
    splits = pd.read_csv("data/processed/splits.csv", dtype={"site": str})
    assert list(X.index) == list(meta.sample_id) == list(splits.sample_id)
    annot = pd.read_csv("data/processed/event_annotation.csv", usecols=["as_id", "symbol", "splice_type"])
    annot["as_id"] = annot["as_id"].astype(str)
    Xv, cols = X.values, np.asarray(X.columns)

    ms, os_, ss = [], [], []
    for cfg in CONFIGS:
        paths = [f"{RUNS}/{cfg}_{t}.csv" for t in ("metrics", "oof", "selected")]
        if not force and all(os.path.exists(p) for p in paths):
            m, o, s = (pd.read_csv(p, dtype={"as_id": str}) for p in paths)
            print(f"  {cfg}: loaded cached results")
        else:
            m, o, s = run_config(cfg, Xv, cols, meta, splits)
            for df, p in zip((m, o, s), paths):
                df.to_csv(p, index=False)
        ms.append(m); os_.append(o); ss.append(s)
    m, o, s = pd.concat(ms), pd.concat(os_), pd.concat(ss)

    # Test patients must not appear anywhere in CV outputs.
    test_ids = set(splits.loc[splits.set == "test", "sample_id"])
    assert not set(o.sample_id) & test_ids, "test sample in CV predictions"

    m.to_csv(f"{T}/cv_metrics_per_fold.csv", index=False)
    summ = summarize(m)
    summ.to_csv(f"{T}/cv_metrics_summary.csv", index=False)
    o.to_csv(f"{T}/cv_oof_predictions.csv.gz", index=False)
    s.to_csv(f"{T}/cv_selected_events_per_fold.csv", index=False)

    b = pd.read_csv(f"{T}/stats_paired_train_only.csv", dtype={"as_id": str}).set_index("as_id")
    b_ranked = list(top_events(b, max(KS)).index)  # Phase 3 (b): descriptive comparison only, never used in CV
    ev, gene = selection_frequency(s, annot, b_ranked)
    ev.to_csv(f"{T}/cv_selection_frequency_by_event.csv", index=False)
    gene.to_csv(f"{T}/cv_selection_frequency_by_gene.csv", index=False)

    fig_metric(m, "roc_auc", "ROC-AUC", "cv_roc_auc_by_model_k", baseline=0.5)
    fig_metric(m, "pr_auc_normal", "PR-AUC (normal = positive)", "cv_pr_auc_normal_by_model_k",
               baseline=float((m.n_val_normal / m.n_val).mean()))
    fig_null(m)
    fig_selection(ev, annot)
    fig_roc(o)
    null_label_overlap(meta, splits, m).to_csv(f"{T}/cv_null_audit_label_overlap.csv", index=False)
    oof_mean_prob(o).to_csv(f"{T}/cv_oof_mean_prob_main_k25.csv")

    pd.set_option("display.width", 250)
    show = ["config", "model", "k", "roc_auc_mean", "roc_auc_sd", "pr_auc_normal_mean", "pr_auc_normal_sd",
            "f1_tumour_mean", "recall_normal_mean", "n_folds_auc_gt_0.98", "n_folds_topped_up"]
    print(summ[show].round(3).to_string())
    print(s.drop_duplicates(["config", "repeat", "fold"]).groupby("config")[["n_pairs_ranking", "n_eligible", "n_passed_fdr"]]
          .describe().round(0).T.to_string())


if __name__ == "__main__":
    main()
