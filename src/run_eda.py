"""Phase 3: EDA and statistics. Run from the repo root:  python -m src.run_eda

No models are trained here. Everything is descriptive of the full cohort EXCEPT the paired analysis (b),
which uses training-patient pairs only. (b) is the only output allowed to feed later feature ranking.
Every table goes to results/tables/ and every figure to results/figures/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.decomposition import PCA

from src.build_matrix import load_features, load_metadata
from src.eda_stats import bh, eta_squared, paired_table, paired_tests, top_events, unpaired_tests
from src.plotting import (CLASS_COLOR, CLASS_MARKER, DIV, INK2, MUTED, NAN_COLOR, NORMAL, SEQ, THIRD,
                          TUMOUR, savefig, scatter_by_class)

SEED = 42
MIN_PAIRS = 30
MIN_GROUP = 10          # unpaired: minimum non-missing values per class
EFF = 0.10              # |delta PSI| reporting threshold
T = "results/tables"
STAGE4 = {"Stage I": "I", "Stage IA": "I", "Stage IB": "I", "Stage II": "II", "Stage IIA": "II",
          "Stage IIB": "II", "Stage IIIA": "III", "Stage IIIB": "III", "Stage IV": "IV"}


# ------------------------------------------------------------------------------------------------
def load_all():
    X, meta = load_features(), load_metadata()
    s = pd.read_csv("data/processed/splits.csv", dtype={"site": str})
    clin = pd.read_csv("data/processed/clinical_by_sample.csv", dtype={"site": str})
    annot = pd.read_csv("data/processed/event_annotation.csv", usecols=["as_id", "symbol", "splice_type"])
    annot["as_id"] = annot["as_id"].astype(str)
    assert list(X.index) == list(meta.sample_id) == list(s.sample_id) == list(clin.sample_id)
    meta = meta.merge(s[["sample_id", "set"]], on="sample_id")
    meta = meta.merge(clin[["sample_id", "Gender", "Age", "Pathologic_Tumor_Stage", "Absolute_Purity_Value"]],
                      on="sample_id")
    meta["stage4"] = meta["Pathologic_Tumor_Stage"].map(STAGE4)
    # A normal has no purity/age/stage of its own; the copied patient values are only meaningful for tumours.
    for c in ["Gender", "Age", "stage4", "Absolute_Purity_Value"]:
        meta.loc[meta.label == 0, c] = np.nan
    return X, meta, annot.set_index("as_id")


def site_order(meta):
    """Normal-contributing sites first (by size), then the rest (by size)."""
    n = meta.groupby("site").agg(n=("label", "size"), ns=("normal_site", "first"))
    return list(n.sort_values(["ns", "n"], ascending=False).index)


# ---- 1. cohort tables ---------------------------------------------------------------------------
def cohort_tables(meta):
    ct = pd.crosstab(meta.site, [meta.set, meta.sample_type])
    ct.columns = [f"{a}_{b}" for a, b in ct.columns]
    ct["total_tumour"] = ct.filter(like="_tumour").sum(axis=1)
    ct["total_normal"] = ct.filter(like="_normal").sum(axis=1)
    ct["normal_site"] = ct["total_normal"] > 0
    ct = ct.loc[site_order(meta)]
    ct.to_csv(f"{T}/eda_cohort_site_by_set.csv")

    by_set = meta.groupby("set").apply(lambda g: pd.Series({
        "patients": g.patient_id.nunique(), "samples": len(g), "tumour": int(g.label.sum()),
        "normal": int((g.label == 0).sum()), "pairs": int(((g.label == 0) & g.is_paired).sum()),
        "samples_7site": int(g.normal_site.sum())}), include_groups=False)
    by_set.loc["all"] = by_set.sum()
    by_set.loc["all", "patients"] = meta.patient_id.nunique()
    by_set.to_csv(f"{T}/eda_cohort_by_set.csv")

    tum = meta[meta.label == 1]
    rows = []
    for st, g in [("train", tum[tum.set == "train"]), ("test", tum[tum.set == "test"]), ("all", tum)]:
        a, p = g.Age.dropna(), g.Absolute_Purity_Value.dropna()
        r = {"set": st, "n_tumours": len(g), "age_n": len(a), "age_median": a.median(),
             "age_q1": a.quantile(.25), "age_q3": a.quantile(.75), "female_frac": (g.Gender == "FEMALE").mean(),
             "purity_n": len(p), "purity_median": p.median()}
        for s4 in ["I", "II", "III", "IV"]:
            r[f"stage_{s4}"] = int((g.stage4 == s4).sum())
        r["stage_missing"] = int(g.stage4.isna().sum())
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{T}/eda_clinical_by_set.csv", index=False)
    return ct


# ---- 2. missingness & distributions -------------------------------------------------------------
def missingness(X, meta):
    smp = meta[["sample_id", "site", "sample_type", "normal_site"]].copy()
    smp["frac_missing"] = X.isna().mean(axis=1).values
    order = site_order(meta)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 6]})
    for i, c in enumerate(["tumour", "normal"]):
        v = smp.loc[smp.sample_type == c, "frac_missing"]
        a0.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops={"color": INK2})
        a0.scatter(i + np.random.default_rng(SEED).uniform(-0.18, 0.18, len(v)), v, s=8,
                   c=CLASS_COLOR[c], marker=CLASS_MARKER[c], alpha=0.6, linewidths=0)
    a0.set_xticks([0, 1], ["tumour", "normal"])
    a0.set_ylabel("fraction of events missing (per sample)")
    a0.set_title("By class")
    xpos = {s: i for i, s in enumerate(order)}
    rng = np.random.default_rng(SEED)
    for c in ["tumour", "normal"]:
        g = smp[smp.sample_type == c]
        a1.scatter(g.site.map(xpos) + rng.uniform(-0.25, 0.25, len(g)), g.frac_missing, s=9 if c == "tumour" else 16,
                   c=CLASS_COLOR[c], marker=CLASS_MARKER[c], alpha=0.65, linewidths=0, label=c)
    n7 = int(meta.groupby("site").normal_site.first().sum())
    a1.axvspan(-0.5, n7 - 0.5, color="#f4f3ef", zorder=0)
    a1.text(n7 / 2 - 0.5, a1.get_ylim()[1] * 0.97, "sites with normals", ha="center", va="top", color=INK2)
    a1.set_xticks(range(len(order)), order, rotation=90)
    a1.set_xlabel("tissue source site")
    a1.set_title("By site")
    a1.legend(loc="upper right")
    savefig(fig, "eda_missingness_per_sample_by_class_and_site")

    ev = pd.DataFrame({"tumour": X[meta.label.values == 1].isna().mean(), "normal": X[meta.label.values == 0].isna().mean()})
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    hb = ax.hexbin(ev.tumour, ev.normal, gridsize=40, bins="log", cmap=SEQ, mincnt=1, linewidths=0.2)
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, c=INK2)
    ax.set_xlim(-0.02, 1); ax.set_ylim(-0.02, 1)
    ax.set_xlabel("fraction missing in tumours"); ax.set_ylabel("fraction missing in normals")
    ax.set_title("Per-event missingness by class")
    fig.colorbar(hb, ax=ax, label="events (log)")
    savefig(fig, "eda_missingness_per_event_tumour_vs_normal")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(0, 1, 51)
    for c, lab in [("tumour", 1), ("normal", 0)]:
        v = X.values[meta.label.values == lab].ravel()
        ax.hist(v[~np.isnan(v)], bins=bins, density=True, histtype="step", lw=2, color=CLASS_COLOR[c], label=c)
    ax.set_yscale("log"); ax.set_xlabel("PSI"); ax.set_ylabel("density (log)")
    ax.set_title("All PSI values by class"); ax.legend()
    savefig(fig, "eda_psi_distribution_by_class")

    tests = [{"comparison": "per-sample missingness ~ site (all samples)",
              **dict(zip(["H", "p"], stats.kruskal(*[g.frac_missing for _, g in smp.groupby("site")])))},
             {"comparison": "per-sample missingness ~ class",
              **dict(zip(["H", "p"], stats.kruskal(*[g.frac_missing for _, g in smp.groupby("sample_type")])))},
             {"comparison": "per-sample missingness ~ site (tumours only)",
              **dict(zip(["H", "p"], stats.kruskal(*[g.frac_missing for _, g in smp[smp.sample_type == 'tumour'].groupby("site")])))}]
    tests = pd.DataFrame(tests)
    tests.to_csv(f"{T}/eda_missingness_tests.csv", index=False)
    ev.describe().to_csv(f"{T}/eda_missingness_per_event_by_class_summary.csv")
    return smp, ev, tests


def describe_events(X, meta, ids):
    rows = []
    for e in ids:
        for c, lab in [("tumour", 1), ("normal", 0)]:
            v = X.loc[meta.label.values == lab, e]
            nn = v.dropna()
            rows.append({"as_id": e, "class": c, "n": len(v), "n_missing": int(v.isna().sum()),
                         "median": nn.median(), "q1": nn.quantile(.25), "q3": nn.quantile(.75),
                         "frac_exact_0": (nn == 0).mean(), "frac_exact_1": (nn == 1).mean(), "sd": nn.std()})
    return pd.DataFrame(rows)


def plot_events(X, meta, annot, ids, name, title, res=None):
    n = len(ids)
    ncol = min(n, 4 if n > 4 else n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.8 * nrow), squeeze=False)
    rng = np.random.default_rng(SEED)
    for ax, e in zip(axes.ravel(), ids):
        for i, (c, lab) in enumerate([("tumour", 1), ("normal", 0)]):
            v = X.loc[meta.label.values == lab, e].dropna()
            ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops={"color": INK2})
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=6 if lab else 12, c=CLASS_COLOR[c],
                       marker=CLASS_MARKER[c], alpha=0.5, linewidths=0)
        a = annot.loc[e]
        sub = f"\nmedian ΔPSI {res.loc[e, 'median_diff']:+.2f}, FDR {res.loc[e, 'fdr']:.1e}" if res is not None else ""
        ax.set_title(f"{a.symbol} {a.splice_type} ({e}){sub}", fontsize=8)
        ax.set_xticks([0, 1], ["tumour", "normal"]); ax.set_ylim(-0.05, 1.05)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].set_ylabel("PSI")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    savefig(fig, name)


def example_events(X, meta, annot):
    """One event per splice type, drawn at random (seed 42) BEFORE any test is run: shows typical shapes."""
    rng = np.random.default_rng(SEED)
    ids = [rng.choice(sorted(annot.index[annot.splice_type == t])) for t in sorted(annot.splice_type.unique())]
    d = describe_events(X, meta, ids).merge(annot, left_on="as_id", right_index=True)
    d.to_csv(f"{T}/eda_example_events_random.csv", index=False)
    plot_events(X, meta, annot, ids, "eda_example_events_random", "Random example events (one per splice type, seed 42)")
    return d


# ---- 3. PCA -------------------------------------------------------------------------------------
def run_pca(Xsub, n=10):
    """Centred, unscaled PCA on complete, non-constant events (visualization only)."""
    keep = Xsub.notna().all() & (Xsub.std() > 0)
    Z = Xsub.loc[:, keep]
    p = PCA(n_components=n, svd_solver="full", random_state=SEED).fit(Z.values)
    sc = pd.DataFrame(p.transform(Z.values), index=Xsub.index, columns=[f"PC{i + 1}" for i in range(n)])
    return sc, p.explained_variance_ratio_, int(keep.sum())


def r2_table(name, sc, var, meta_sub, n_events):
    rows = []
    for k in range(5):
        pc = sc.iloc[:, k].values
        rows.append({"analysis": name, "n_samples": len(sc), "n_events": n_events, "PC": f"PC{k + 1}",
                     "var_explained": var[k], "R2_label": eta_squared(pc, meta_sub.label.values)
                     if meta_sub.label.nunique() > 1 else np.nan, "R2_site": eta_squared(pc, meta_sub.site.values)})
    return rows


def pca_site_facets(sc, meta_sub, order, name, title, n_sites=12):
    sites = [s for s in order if s in set(meta_sub.site)][:n_sites]
    fig, axes = plt.subplots(3, 4, figsize=(11, 8), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), sites):
        m = (meta_sub.site == s).values
        ax.scatter(sc.PC1[~m], sc.PC2[~m], s=5, c=MUTED, alpha=0.5, linewidths=0)
        cls = meta_sub.sample_type.values[m]
        scatter_by_class(ax, sc.PC1.values[m], sc.PC2.values[m], cls, s=12, alpha=0.9)
        tag = " (has normals)" if meta_sub.normal_site.values[m][0] else ""
        ax.set_title(f"site {s}{tag}, n={int(m.sum())}", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("PC1")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    savefig(fig, name)


def pca_all(X, meta):
    order = site_order(meta)
    rows = []
    # Main: all 573 samples, events complete in every sample.
    sc, var, ne = run_pca(X)
    rows += r2_table("all_samples", sc, var, meta, ne)
    cls = meta.sample_type.values

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (a, b) in zip(axes[:2], [("PC1", "PC2"), ("PC1", "PC3")]):
        scatter_by_class(ax, sc[a].values, sc[b].values, cls)
        ax.set_xlabel(f"{a} ({var[int(a[2:]) - 1]:.1%})"); ax.set_ylabel(f"{b} ({var[int(b[2:]) - 1]:.1%})")
    axes[0].legend(loc="best")
    axes[0].set_title("Coloured by class")
    ax = axes[2]
    ns = meta.normal_site.values
    for flag, col, lab in [(False, MUTED, "tumour, site without normals"), (True, THIRD, "tumour, site with normals")]:
        m = (ns == flag) & (cls == "tumour")
        ax.scatter(sc.PC1[m], sc.PC2[m], s=10, c=col, alpha=0.75, linewidths=0.3, edgecolors="white", label=lab)
    m = cls == "normal"
    ax.scatter(sc.PC1[m], sc.PC2[m], s=22, c=NORMAL, marker="^", alpha=0.85, linewidths=0.3, edgecolors="white", label="normal")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Coloured by site group"); ax.legend(loc="best", fontsize=7)
    fig.suptitle(f"PCA, all 573 samples ({ne:,} complete events, centred, unscaled)", y=1.02)
    fig.tight_layout()
    savefig(fig, "eda_pca_all_by_class_and_site_group")

    pca_site_facets(sc, meta, order, "eda_pca_all_by_site_facets",
                    "PCA (all samples): each panel highlights one site (12 largest sites; normals = triangles)")

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    pur = meta.Absolute_Purity_Value.values
    mt, mmiss = (cls == "tumour") & ~np.isnan(pur), (cls == "tumour") & np.isnan(pur)
    ax.scatter(sc.PC1[cls == "normal"], sc.PC2[cls == "normal"], s=16, marker="^", c=MUTED, alpha=0.8, linewidths=0, label="normal (no purity)")
    ax.scatter(sc.PC1[mmiss], sc.PC2[mmiss], s=12, facecolors="none", edgecolors="#8d8c86", linewidths=0.6, label="tumour, purity missing")
    s_ = ax.scatter(sc.PC1[mt], sc.PC2[mt], s=14, c=pur[mt], cmap=SEQ, vmin=0, vmax=1, linewidths=0.2, edgecolors="white", label="tumour, purity")
    fig.colorbar(s_, ax=ax, label="ABSOLUTE purity (tumours)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA coloured by tumour purity"); ax.legend(fontsize=7, loc="best")
    savefig(fig, "eda_pca_all_by_purity")

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(range(1, 11), var * 100, color=TUMOUR, width=0.7)
    ax.set_xlabel("component"); ax.set_ylabel("% variance explained"); ax.set_title("Scree (all samples)")
    savefig(fig, "eda_pca_all_scree")

    # Tumours only: does site alone drive variation?
    tm = meta.label.values == 1
    sct, vart, net = run_pca(X[tm])
    rows += r2_table("tumours_only", sct, vart, meta[tm], net)
    pca_site_facets(sct, meta[tm].reset_index(drop=True), order, "eda_pca_tumours_only_by_site_facets",
                    f"PCA, tumours only ({net:,} complete events): each panel highlights one site")

    # 7-site sensitivity population.
    m7 = meta.normal_site.values
    sc7, var7, ne7 = run_pca(X[m7])
    rows += r2_table("7_site_population", sc7, var7, meta[m7], ne7)
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter_by_class(ax, sc7.PC1.values, sc7.PC2.values, meta[m7].sample_type.values)
    ax.set_xlabel(f"PC1 ({var7[0]:.1%})"); ax.set_ylabel(f"PC2 ({var7[1]:.1%})"); ax.legend()
    ax.set_title(f"PCA, 7 sites with normals ({ne7:,} complete events)")
    savefig(fig, "eda_pca_7site_by_class")

    # Robustness: events <= 10% missing, per-event median imputation (whole data; visualization only).
    keep = X.isna().mean() <= 0.10
    Xi = X.loc[:, keep].fillna(X.loc[:, keep].median())
    sci, vari, nei = run_pca(Xi)
    rows += r2_table("robustness_median_imputed_le10pct_missing", sci, vari, meta, nei)
    robust = {f"abs_corr_PC{k}": abs(np.corrcoef(sc[f"PC{k}"], sci[f"PC{k}"])[0, 1]) for k in (1, 2, 3)}

    r2 = pd.DataFrame(rows)
    r2.to_csv(f"{T}/eda_pca_r2.csv", index=False)
    pd.Series(robust).to_csv(f"{T}/eda_pca_robustness.csv", header=["value"])
    scores = sc.iloc[:, :5].add_prefix("all_")
    scores = scores.join(sct.iloc[:, :2].add_prefix("tumouronly_"))
    scores.index.name = "sample_id"
    scores.to_csv(f"{T}/eda_pca_scores.csv")
    return sc, sct, r2, robust


# ---- 4/5. statistics ----------------------------------------------------------------------------
def summarize(res, name):
    t = res[res.testable]
    sig = t[t.fdr < 0.05]
    return {"analysis": name, "n_events": len(res),
            "n_dropped_min_n": int(res.reason.str.startswith("fewer").sum()),
            "n_untestable_other": int((~res.testable & ~res.reason.str.startswith("fewer")).sum()),
            "n_tested": len(t), "n_fdr05": len(sig),
            "n_fdr05_absmedian_ge_0.1": int((sig.median_diff.abs() >= EFF).sum()),
            "n_fdr05_absmean_ge_0.1": int((sig.mean_diff.abs() >= EFF).sum()),
            "n_fdr05_median_diff_exactly_0": int((sig.median_diff == 0).sum()),
            "n_fdr05_higher_in_tumour": int((sig.mean_diff > 0).sum()),
            "n_fdr05_lower_in_tumour": int((sig.mean_diff < 0).sum()),
            "min_fdr": t.fdr.min()}


def agreement(r1, r2, n1, n2):
    both = r1.testable & r2.testable
    a, b = r1[both], r2[both]
    s1, s2 = a.fdr < 0.05, b.fdr < 0.05
    sb = s1 & s2
    return {"comparison": f"{n1} vs {n2}", "n_common_testable": int(both.sum()),
            "spearman_median_diff": stats.spearmanr(a.median_diff, b.median_diff).statistic,
            "spearman_mean_diff": stats.spearmanr(a.mean_diff, b.mean_diff).statistic,
            f"n_fdr05_{n1}": int(s1.sum()), f"n_fdr05_{n2}": int(s2.sum()), "n_fdr05_both": int(sb.sum()),
            "jaccard_fdr05": sb.sum() / max((s1 | s2).sum(), 1),
            "sign_agreement_mean_diff_both_sig": (np.sign(a.mean_diff[sb]) == np.sign(b.mean_diff[sb])).mean()
            if sb.any() else np.nan}


def run_stats(X, meta, annot, splits_set):
    paired = meta[(meta.label == 0) & meta.is_paired]
    pids_all = list(paired.patient_id)
    # Sensitivity population note: every pair lies within the 7 normal-contributing sites, so the
    # paired 7-site analysis is identical to (a). Asserted rather than recomputed.
    assert paired.normal_site.all()
    # (b) training-patient pairs ONLY. This is the only analysis allowed to feed later ranking.
    pids_train = list(paired.loc[paired.set == "train", "patient_id"])
    test_pids = set(splits_set.loc[splits_set.set == "test", "patient_id"])
    assert not set(pids_train) & test_pids, "test patient in training-only paired analysis"
    assert len(pids_all) == 57 and len(pids_train) == 46
    pd.DataFrame({"patient_id": pids_train}).to_csv(f"{T}/stats_paired_train_pairs_used.csv", index=False)

    out = {}
    Ta, Na = paired_table(X, meta, pids_all)
    out["paired_full_descriptive"] = paired_tests(Ta, Na, MIN_PAIRS)
    Tb, Nb = paired_table(X, meta, pids_train)
    out["paired_train_only"] = paired_tests(Tb, Nb, MIN_PAIRS)
    out["unpaired_full_descriptive"] = unpaired_tests(X, meta.label.values, MIN_GROUP)
    m7 = meta.normal_site.values
    out["unpaired_7site_sensitivity"] = unpaired_tests(X[m7], meta.label.values[m7], MIN_GROUP)

    for k, r in out.items():
        r.join(annot).to_csv(f"{T}/stats_{k}.csv")
    summ = pd.DataFrame([summarize(r, k) for k, r in out.items()])
    summ.to_csv(f"{T}/stats_summary.csv", index=False)
    dropped = pd.concat({k: r.loc[~r.testable, "reason"].value_counts() for k, r in out.items()}).rename("n_events")
    dropped.index.names = ["analysis", "reason"]
    dropped.to_csv(f"{T}/stats_dropped_events_summary.csv")

    ag = pd.DataFrame([
        agreement(out["paired_full_descriptive"], out["paired_train_only"], "paired_full", "paired_train"),
        agreement(out["paired_full_descriptive"], out["unpaired_full_descriptive"], "paired_full", "unpaired_full"),
        agreement(out["unpaired_full_descriptive"], out["unpaired_7site_sensitivity"], "unpaired_full", "unpaired_7site"),
        agreement(out["paired_full_descriptive"], out["unpaired_7site_sensitivity"], "paired_full", "unpaired_7site")])
    ag.to_csv(f"{T}/stats_agreement.csv", index=False)
    return out, summ, ag


# ---- 6. figures ---------------------------------------------------------------------------------
def volcanoes(out, annot):
    titles = {"paired_full_descriptive": "(a) Paired, all 57 pairs (descriptive, full cohort)",
              "paired_train_only": "(b) Paired, 46 training pairs only",
              "unpaired_full_descriptive": "Unpaired, all samples (descriptive)",
              "unpaired_7site_sensitivity": "Unpaired, 7 sites with normals (sensitivity)"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, (k, r) in zip(axes.ravel(), out.items()):
        t = r[r.testable]
        y = -np.log10(t.p.clip(lower=1e-300))
        sig = (t.fdr < 0.05).values
        ax.scatter(t.median_diff[~sig], y[~sig], s=4, c=MUTED, alpha=0.5, linewidths=0, label="FDR ≥ 0.05")
        ax.scatter(t.median_diff[sig], y[sig], s=5, c=TUMOUR, alpha=0.6, linewidths=0, label="FDR < 0.05")
        for v in (-EFF, EFF):
            ax.axvline(v, ls="--", lw=0.7, c=INK2)
        top = top_events(t, 5)
        for e, row in top.iterrows():
            ax.annotate(annot.loc[e, "symbol"], (row.median_diff, -np.log10(max(row.p, 1e-300))), fontsize=7,
                        xytext=(3, 2), textcoords="offset points", color=INK2)
        ax.set_title(f"{titles[k]}\n{len(t):,} tested, {int(sig.sum()):,} FDR < 0.05", fontsize=9)
        ax.set_xlabel("median ΔPSI (tumour − normal)"); ax.set_ylabel("−log10 p")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    savefig(fig, "eda_volcano_paired_and_unpaired")


def heatmap(X, meta, annot, res, n=50):
    top = top_events(res, n)
    top = top.assign(up=top.median_diff > 0).sort_values(["up", "median_diff"], ascending=[False, False])
    cols = meta.assign(o=meta.label, ns=~meta.normal_site).sort_values(["o", "ns", "site", "sample_id"])
    M = X.loc[cols.sample_id, top.index].T.values
    fig = plt.figure(figsize=(14, 0.2 * len(top) + 2.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.25, 0.25, 0.25, 0.2 * len(top)], width_ratios=[60, 1],
                          hspace=0.05, wspace=0.02)
    bars = [("class", np.where(cols.label == 1, 0, 1), ListedColormap([TUMOUR, NORMAL])),
            ("site has normals", np.where(cols.normal_site, 1, 0), ListedColormap([MUTED, THIRD])),
            ("set", np.where(cols.set == "test", 1, 0), ListedColormap(["#dcdbd5", "#3e3d3a"]))]
    for i, (lab, v, cm) in enumerate(bars):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(v[None, :], aspect="auto", cmap=cm, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([0], [lab]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    ax = fig.add_subplot(gs[3, 0])
    cm = SEQ.copy(); cm.set_bad(NAN_COLOR)
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cm, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(top)), [f"{annot.loc[e, 'symbol']} {annot.loc[e, 'splice_type']} {e} ({r.median_diff:+.2f})"
                                    for e, r in top.iterrows()], fontsize=6)
    # Columns are sorted with normals (label 0) first.
    nn = int((cols.label == 0).sum())
    ax.axvline(nn - 0.5, c="white", lw=1.5)
    ax.set_xticks([nn / 2, nn + (len(cols) - nn) / 2], [f"normal (n={nn})", f"tumour (n={len(cols) - nn})"])
    ax.grid(False)
    cax = fig.add_subplot(gs[3, 1])
    fig.colorbar(im, cax=cax, label="PSI (grey = missing)")
    fig.suptitle(f"Top {len(top)} events from paired analysis (a), descriptive, full cohort. "
                 "Columns: class, then site group, then site. Bars: class (blue tumour / orange normal), "
                 "site has normals (aqua), set (dark = test)", fontsize=8, y=0.92)
    savefig(fig, "eda_heatmap_top50_paired_full")
    return top


# ---- 7. confounders -----------------------------------------------------------------------------
def confounders(X, meta, sc, sct, top10):
    tm = (meta.label == 1).values
    mt = meta[tm].reset_index(drop=True)
    targets = {"PC1_all": sc.PC1.values[tm], "PC2_all": sc.PC2.values[tm],
               "PC1_tumouronly": sct.PC1.values, "PC2_tumouronly": sct.PC2.values}
    for e in top10.index:
        targets[f"event_{e}"] = X.loc[tm, e].values
    rows = []
    for tname, y in targets.items():
        y = np.asarray(y, dtype=float)
        for var in ["Absolute_Purity_Value", "Age"]:
            x = mt[var].values.astype(float)
            ok = ~np.isnan(x) & ~np.isnan(y)
            r = stats.spearmanr(x[ok], y[ok])
            rows.append({"target": tname, "confounder": var, "test": "Spearman", "n": int(ok.sum()),
                         "effect": r.statistic, "effect_measure": "rho", "p": r.pvalue})
        ok = mt.Gender.notna().values & ~np.isnan(y)
        f, m_ = y[ok & (mt.Gender == "FEMALE").values], y[ok & (mt.Gender == "MALE").values]
        rows.append({"target": tname, "confounder": "Gender", "test": "Mann-Whitney", "n": int(ok.sum()),
                     "effect": np.median(m_) - np.median(f), "effect_measure": "median(male) - median(female)",
                     "p": stats.mannwhitneyu(m_, f).pvalue})
        for var in ["stage4", "site"]:
            ok = mt[var].notna().values & ~np.isnan(y)
            grp = [y[ok & (mt[var] == g).values] for g in sorted(mt.loc[ok, var].unique())]
            grp = [g for g in grp if len(g) >= 2]
            H, p = stats.kruskal(*grp)
            nn, k = sum(len(g) for g in grp), len(grp)
            rows.append({"target": tname, "confounder": var, "test": "Kruskal-Wallis", "n": nn,
                         "effect": (H - k + 1) / (nn - k), "effect_measure": "epsilon^2", "p": p})
    cf = pd.DataFrame(rows)
    cf["fdr"] = bh(cf.p)
    cf.to_csv(f"{T}/eda_confounders.csv", index=False)

    # Purity direction check for the top events: within tumours, does higher purity move PSI in the same direction
    # as tumour - normal? Either tumour-cell-intrinsic splicing or plain mixture dilution would produce that.
    comp = []
    for e, r in top10.iterrows():
        rho = cf[(cf.target == f"event_{e}") & (cf.confounder == "Absolute_Purity_Value")].iloc[0]
        comp.append({"as_id": e, "median_diff_paired": r.median_diff, "mean_diff_paired": r.mean_diff,
                     "rho_purity": rho.effect, "p_purity": rho.p, "fdr_purity": rho.fdr,
                     "same_direction": np.sign(rho.effect) == np.sign(r.mean_diff)})
    comp = pd.DataFrame(comp)
    comp.to_csv(f"{T}/eda_purity_direction_top10.csv", index=False)

    rho = cf[cf.effect_measure == "rho"].pivot(index="confounder", columns="target", values="effect")
    rho = rho[list(targets)]
    fdr = cf[cf.effect_measure == "rho"].pivot(index="confounder", columns="target", values="fdr")[list(targets)]
    fig, ax = plt.subplots(figsize=(0.62 * len(targets) + 2, 2.6))
    im = ax.imshow(rho.values, cmap=DIV, vmin=-0.6, vmax=0.6, aspect="auto")
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            ax.text(j, i, f"{rho.values[i, j]:+.2f}{'*' if fdr.values[i, j] < 0.05 else ''}", ha="center",
                    va="center", fontsize=7)
    ax.set_xticks(range(rho.shape[1]), rho.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(rho.shape[0]), ["purity", "age"]); ax.grid(False)
    fig.colorbar(im, ax=ax, label="Spearman ρ (tumours)")
    ax.set_title("Purity and age vs PCs and top-10 events (tumours only; * FDR < 0.05)")
    savefig(fig, "eda_confounders_rho_heatmap")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    pur = mt.Absolute_Purity_Value.values.astype(float)
    ok = ~np.isnan(pur)
    for ax, (k, lab) in zip(axes, [("PC1_all", "PC1 (all-sample PCA)"), ("PC1_tumouronly", "PC1 (tumour-only PCA)")]):
        ax.scatter(pur[ok], targets[k][ok], s=10, c=TUMOUR, alpha=0.6, linewidths=0)
        r = cf[(cf.target == k) & (cf.confounder == "Absolute_Purity_Value")].iloc[0]
        ax.set_title(f"{lab}: ρ = {r.effect:+.2f}, n = {r.n}")
        ax.set_xlabel("ABSOLUTE purity"); ax.set_ylabel(lab)
    fig.tight_layout()
    savefig(fig, "eda_purity_vs_pc1_tumours")
    return cf, comp


# ------------------------------------------------------------------------------------------------
def main():
    X, meta, annot = load_all()
    splits_set = pd.read_csv("data/processed/splits.csv")[["patient_id", "set"]]
    print("1. cohort tables"); cohort_tables(meta)
    print("2. missingness + random example events"); missingness(X, meta); example_events(X, meta, annot)
    print("3. PCA"); sc, sct, r2, robust = pca_all(X, meta)
    print("4/5. statistics (bootstrap; several minutes)"); out, summ, ag = run_stats(X, meta, annot, splits_set)
    print(summ.to_string()); print(ag.to_string())
    print("6. figures"); volcanoes(out, annot)
    res_a = out["paired_full_descriptive"]
    top50 = heatmap(X, meta, annot, res_a)
    top50.join(annot).to_csv(f"{T}/eda_top50_paired_full_descriptive.csv")
    top10 = top_events(res_a, 10)
    d = describe_events(X, meta, list(top10.index[:6])).merge(annot, left_on="as_id", right_index=True)
    d.to_csv(f"{T}/eda_example_events_top6_paired_full.csv", index=False)
    plot_events(X, meta, annot, list(top10.index[:6]), "eda_example_events_top6_paired_full",
                "Top 6 events, paired analysis (a): descriptive, full cohort", res=res_a)
    print("7. confounders"); cf, comp = confounders(X, meta, sc, sct, top10)
    print(r2.to_string()); print(robust); print(comp.to_string())
    print(cf[cf.target.str.startswith("PC")].to_string())


if __name__ == "__main__":
    main()
