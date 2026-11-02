# Phase 5 checks: the 19 candidate events (training data only)

Produced by `python -m src.phase5_checks` (tables) and written by hand from those tables (this document), 2026-11-02.

## 0. Scope
- **Post hoc and descriptive.** These checks were not part of the frozen protocol (`docs/analysis_plan.md` §E) and change
  nothing in it. No model was fitted.
- **Training data only:**
  - 460 samples: 412 tumours, 48 normals (46 matched pairs + 2 unpaired normals);
  - the script asserts that no test sample or test patient is used;
  - the test set was not read.
- **Annotation source:** the local SpliceSeq PSI download only (owner decision). Nothing was looked up externally.
- **Existing outputs were not modified.** The script writes three new tables and refuses to overwrite them:
  - `results/tables/shortlist_event_annotation.csv` (Step 1)
  - `results/tables/shortlist_event_behaviour.csv` (Step 2)
  - `results/tables/shortlist_purity_corr.csv` (Step 3)
- **Definitions:**
  - "missing" = PSI `null` in the download;
  - "pinned" = PSI exactly 0 or exactly 1 (exact equality, because PSI = 0 is a real value);
  - **"mostly pinned" = more than 50% of non-missing training normals at exactly 0 or 1.** That threshold is my
    assumption; the request did not define "mostly".

---

## 1. Observations

### 1.1 Annotation (Step 1)
- **Missing fields.** The download has no chromosome, strand, genomic coordinates or transcript identifiers for any
  event. Those columns read "not in SpliceSeq PSI download".
- **Exon IDs are SpliceSeq splice-graph IDs**, for example `6.1`, `4.1`, `45.1`, `12:13:14:15:16:19`.
  - For ES events, `from_exon`/`to_exon` give the flanking SpliceSeq exons.
  - For AP and AT events, both are `null` in the download.
- **No event is flagged as novel** (`novel_splice` = 0 for all 19).
- **Multi-exon ES events:** two events skip more than one SpliceSeq exon together.
  - SH3GL3: exons 2:3:4, between 1 and 5.
  - TNC: exons 12:13:14:15:16:19, between 11 and 20.
- **Gene descriptions:** the portal's description field is copied verbatim into
  `annotation_portal_gene_description`. It has not been verified here and is not used in any claim. It is empty for
  SH3GL3, FAM72A and SMCO4.
- **NUMB (as_id 28288):** ES of SpliceSeq exon 13 (between 12 and 14). Is it "exon 9", the PRRL/PRRS isoform exon?
  **This cannot be decided from the annotation.** The download gives no mapping from SpliceSeq exon IDs to RefSeq or
  Ensembl exon numbers, coordinates or isoforms.
- **CEACAM1 (as_id 50162):** ES of SpliceSeq exon 9 (between 8 and 10). Is it the exon that distinguishes the long and
  short cytoplasmic-tail isoforms? **This cannot be decided from the annotation**, for the same reason. The
  SpliceSeq exon number should not be read as a transcript exon number.

### 1.2 Event behaviour in training samples (Step 2)

| rank | gene | type | % missing T / N | median PSI T / N | % at 0 or 1, T / N | complete pairs (of 46) | % pairs with a 0 or 1 | flag |
|---|---|---|---|---|---|---|---|---|
| 1 | SCEL | ES | 11.4 / 0.0 | 0.980 / 0.166 | 32.9 / 0.0 | 40 | 45.0 | – |
| 2 | SLC19A1 | AP | 11.7 / 0.0 | 0.906 / 0.193 | 16.5 / 6.3 | 40 | 20.0 | – |
| 3 | LAMA3 | AP | 21.1 / 0.0 | 0.424 / 0.023 | 21.5 / 43.8 | 36 | 55.6 | – |
| 4 | MYH14 | ES | 4.4 / 0.0 | 0.911 / 0.176 | 14.7 / 0.0 | 43 | 14.0 | – |
| 5 | SPTBN1 | AP | 1.5 / 0.0 | 0.910 / 0.331 | 3.0 / 0.0 | 46 | 2.2 | – |
| 6 | MTMR1 | ES | 10.4 / 12.5 | 0.276 / 0.829 | 4.9 / 16.7 | 38 | 18.4 | – |
| 7 | HYAL1 | AP | 2.7 / 0.0 | 0.248 / 0.801 | 1.5 / 0.0 | 42 | 0.0 | – |
| 8 | NEIL3 | AT | 0.2 / 2.1 | 0.680 / 0.176 | 0.0 / 10.6 | 45 | 11.1 | – |
| 9 | SH3GL3 | ES | 0.7 / 0.0 | 1.000 / 0.493 | 80.0 / 2.1 | 46 | 89.1 | – |
| 10 | NUMB | ES | 0.0 / 0.0 | 0.627 / 0.107 | 0.0 / 0.0 | 46 | 0.0 | – |
| 12 | KIF4A | AT | 0.0 / 0.0 | 0.759 / 0.238 | 0.0 / 0.0 | 46 | 0.0 | – |
| 13 | TEF | AP | 16.3 / 0.0 | 0.710 / 0.241 | 7.0 / 0.0 | 41 | 19.5 | – |
| 15 | FAM72A | AP | 0.0 / 0.0 | 0.740 / 0.261 | 0.7 / 0.0 | 46 | 0.0 | – |
| 16 | COL16A1 | ES | 6.3 / 12.5 | 0.670 / 0.177 | 1.0 / 4.8 | 38 | 5.3 | – |
| 17 | FBLN2 | ES | 1.7 / 0.0 | 0.222 / 0.680 | 1.5 / 0.0 | 45 | 4.4 | – |
| 20 | TNC | ES | 0.7 / 2.1 | 0.660 / 0.248 | 1.5 / 6.4 | 44 | 6.8 | – |
| 21 | NTM | AP | 2.9 / 6.3 | 0.772 / 0.031 | 15.3 / 2.2 | 41 | 17.1 | – |
| 22 | SMCO4 | AP | 1.0 / 0.0 | 0.725 / 0.256 | 2.5 / 0.0 | 44 | 2.3 | – |
| 24 | CEACAM1 | ES | 2.2 / 2.1 | 0.172 / 0.652 | 0.5 / 0.0 | 43 | 2.3 | – |

T = 412 training tumours, N = 48 training normals. Percentages at 0 or 1 are of non-missing values.

- **No event is flagged.**
  - The most-missing normals are MTMR1 and COL16A1, at 12.5% (6 of 48); the 20% threshold is not reached.
  - The most-pinned normals are LAMA3's, at 43.8% exactly 0 or 1; the 50% threshold is not reached.
- **Direction check:** for all 19 events, the sign of (median tumour PSI − median normal PSI) matches the sign of the
  shortlist's paired median ΔPSI.
- **Tumour-side missingness:**
  - highest for LAMA3 (21.1%), TEF (16.3%), SLC19A1 (11.7%), SCEL (11.4%) and MTMR1 (10.4%);
  - for the first four, normals are fully quantified (0% missing);
  - MTMR1 is the exception: its normals are more missing (12.5%) than its tumours.
- **Pinning is mostly on the tumour side, not the normal side:**
  - SH3GL3: median tumour PSI 1.000, 80.0% of tumours at exactly 0 or 1, and 89.1% of complete pairs contain a 0 or 1;
  - SCEL: 32.9% of tumours pinned (45.0% of pairs).
- **Normals near zero without being pinned:** LAMA3 and NTM have median normal PSI 0.023 and 0.031.

### 1.3 Purity correlation in training tumours (Step 3)
- 280 of 412 training tumours have a purity value (32% missing).
- Per-event *n* is 220-280, depending on PSI missingness.
- Spearman ρ, with BH-FDR across the 19 events:

| rank | gene | type | n | ρ | FDR | same direction as tumour − normal? |
|---|---|---|---|---|---|---|
| 22 | **SMCO4** | AP | 277 | **+0.55** | **< 0.0001** | yes |
| 20 | **TNC** | ES | 278 | **−0.24** | **0.0006** | no |
| 10 | **NUMB** | ES | 280 | **+0.22** | **0.0011** | yes |
| 1 | **SCEL** | ES | 253 | **+0.19** | **0.014** | yes |
| 3 | LAMA3 | AP | 220 | −0.16 | 0.064 | no |
| 16 | COL16A1 | ES | 258 | −0.14 | 0.073 | no |
| 17 | FBLN2 | ES | 276 | +0.13 | 0.078 | no |
| others (12 events) | | | | \|ρ\| ≤ 0.09 | ≥ 0.33 | 8 yes, 4 no |

- **Four events pass FDR < 0.05:**
  - SMCO4, NUMB and SCEL: higher purity goes with a PSI shift in the *same* direction as tumour vs normal;
  - TNC: the *opposite* direction.
- Across all 19 events, 11 have a purity correlation in the same direction as tumour − normal and 8 the opposite.
- LAMA3 (ρ = −0.16, opposite direction) matches the full-cohort EDA result (ρ = −0.17; `eda_purity_direction_top10.csv`).
  It does not pass FDR here (0.064), with fewer samples (training only).

---

## 2. Interpretation (what the observations do and do not support)
- **Missingness and pinning artefacts:**
  - No candidate's tumour-vs-normal difference appears to be produced by normals being mostly missing, or mostly
    stuck at 0 or 1.
  - For every event, the normals are well quantified (≤ 12.5% missing) and sit at an intermediate PSI.
  - The medians differ in the same direction as the paired ΔPSI.
- **Where the extremes are:**
  - The events that do touch the 0/1 boundaries do so on the tumour side: SH3GL3, and partly SCEL. There, median ΔPSI
    mostly reflects how far normals sit from full inclusion.
  - As already noted in `docs/eda_report.md` §5, median ΔPSI understates shifts for events piled up at PSI = 1.
    SH3GL3 is the clearest example among the candidates.
- **Missing tumour values:**
  - For LAMA3, TEF, SLC19A1 and SCEL, 11-21% of tumour values are missing, while normals are fully quantified.
  - Median imputation inside the model pipeline would fill these tumours with a training median. Missingness could
    therefore carry some class information for these events.
  - This was not tested here, and no model was fitted. The earlier ≤ 5%-missing CV robustness run
    (`cv_robust_lowmissing.csv`) showed that overall performance does not depend on poorly covered events. It did not
    test these events individually.
- **Purity:**
  - **SMCO4 is the one candidate whose within-tumour PSI tracks purity substantially** (ρ = +0.55, same direction as
    tumour − normal).
  - This fits SMCO4 partly reflecting the proportion of tumour cells in the sample, i.e. tissue composition. It does
    not establish that, and it does not rule out a tumour-intrinsic change.
  - NUMB and SCEL show weak same-direction associations (ρ ≈ 0.2).
  - TNC's weak *opposite* association (ρ = −0.24) means purer tumours show a smaller shift away from normal. That
    would fit a contribution from non-tumour (e.g. stromal) cells. This is a possibility, not a finding.
  - For the other 15 events, purity explains little (|ρ| < 0.17).
  - These are correlations within tumours only. They cannot separate composition from cell-intrinsic splicing, and
    purity is missing for 32% of training tumours.
- **Isoforms:** nothing here identifies isoforms. The NUMB and CEACAM1 questions remain open (§3).

## 3. What could not be determined
- Chromosome, strand, genomic coordinates, and which transcripts each PSI alternative corresponds to, for all 19 events.
  The download does not contain them.
- Whether NUMB SpliceSeq exon 13 is the "exon 9" PRRL/PRRS exon.
- Which transcript exon CEACAM1 SpliceSeq exon 9 is, and whether it distinguishes the long and short cytoplasmic-tail isoforms.
- Answering these needs an external mapping of SpliceSeq exon IDs to genome coordinates and transcripts, for example
  SpliceSeq's splice-graph exon coordinates, or a matching gene model. Any such source should be documented in
  `docs/data_inventory.md` before use.

## 4. Surprising findings
1. **SMCO4 correlates with purity at ρ = +0.55.** That is about three times stronger than any purity correlation seen
   for the top-10 events in the EDA (|ρ| ≤ 0.18). SMCO4 was not among those top 10, so this is the first check of it.
   It also has no gene description in the download.
2. **SH3GL3 is pinned at PSI = 1 in 80% of training tumours** (median 1.000), and 89% of training pairs contain a
   0 or 1. Its ranking depends on an event that tumours almost always fully include.
3. **No event triggers either requested flag.** I expected at least one candidate to have poorly quantified normals.
   Normals have less missingness than tumours for 11 of 19 events, the same for 3, and more for 5, all at ≤ 12.5%.
4. **TNC's within-tumour purity association runs opposite to its tumour − normal shift**, and passes FDR.
