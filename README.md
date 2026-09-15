# CGFS: Coreset Greedy-Based Feature Selection

Official implementation and experiment notebooks for the paper:

> **Submodular Set Cover-Based Coreset Feature Selection for Efficient Intrusion Detection in IoT Networks**
> Mohammed Nagah Amr, Ahmed Elliethy, Tamer Mekkawy, Ashraf Mahran

CGFS formulates feature selection as a **submodular set cover** problem. It selects the features that best *represent* the full feature space, rather than those that score highest in isolation, and it comes with provable approximation guarantees.

---

## Table of Contents

- [Overview](#overview)
- [Method in Brief](#method-in-brief)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Reproducing the Paper](#reproducing-the-paper)
  - [Section 4.2: Sensitivity Analysis (choosing k)](#section-42-sensitivity-analysis-choosing-k)
  - [Sections 4.3, 4.4, 4.5: Detection Performance, SOTA Comparison, Runtime](#sections-43-44-45-detection-performance-sota-comparison-runtime)
  - [Section 4.6, Figure 5: Coverage Trajectory](#section-46-figure-5-coverage-trajectory)
  - [Section 4.6, Figure 6: Marginal F1 Gains](#section-46-figure-6-marginal-f1-gains)
- [Using CGFS on Your Own Data](#using-cgfs-on-your-own-data)
- [Mapping Notebook Names to Paper Names](#mapping-notebook-names-to-paper-names)
- [Notes and Known Caveats](#notes-and-known-caveats)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

High-dimensional IoT traffic makes lightweight intrusion detection hard. CGFS addresses this with the following properties:

- **Model-agnostic.** No classifier is trained during selection. One selected subset is reused unchanged by every downstream model.
- **Redundancy-aware.** A facility-location coverage objective rewards features that cover parts of the feature space not already covered.
- **Relevance-weighted.** Mutual-information weights direct coverage toward label-informative attributes. Uninformative columns receive zero weight.
- **Provable.** The greedy solution satisfies a $(1-1/e)$ approximation at any budget. The same greedy run also reaches any target coverage error using at most a logarithmic factor more features than the optimal cover.
- **Fast.** The implementation uses an exact flat FAISS inner-product index, incremental coverage tracking, and lazy greedy evaluation. Selection takes **0.285–0.538 s** on the evaluated benchmarks, which is **4.5–133.5×** faster than the evaluated mRMR, Mutual Information, XGBoost-importance, and RFE implementations.

Results are reported on three IoT/IIoT benchmarks: **RT-IoT2022**, **Edge-IIoTset**, and **CICIoT2023**.

## Method in Brief

**Similarity.** For two standardized and $\ell_2$-normalized features, the rectified cosine similarity is

$$s(f_i, f_j) = \left|\langle \tilde f_i, \tilde f_j \rangle\right| \in [0, 1].$$

**Relevance weights.** Each feature is weighted by its mutual information with the label, normalized to sum to $d$:

$$w_i = d \cdot \frac{I(f_i; y)}{\sum_l I(f_l; y)}.$$

**Coverage.** The objective is the weighted facility-location function, and the coverage error is its shortfall from the maximum:

$$\mathcal{G}(F_s) = \sum_{i} w_i \max_{j \in F_s} s(f_i, f_j), \qquad E(F_s) = d - \mathcal{G}(F_s).$$

**Selection.** Features are added greedily by largest marginal gain until the budget $k$ is reached. Because the iterates are nested, one run yields a coreset for every budget up to $k$, together with its full coverage-error trace.

---

## Repository Structure

```
.
├── CGFS_weighted_benchmark_experiment.ipynb   # Sec. 4.3, 4.4, 4.5 (Tables 5–11)
├── CGFS_sample_vs_full_experiment.ipynb       # Sec. 4.2 (Tables 3–4, Figure 4)
├── CGFS_coverage_gain_experiment.ipynb        # Sec. 4.6 (Figure 5)
├── CGFS_f1_marginal_gains_figure.ipynb        # Sec. 4.6 (Figure 6)
├── cgfs_f1_marginal_gains.csv                 # Input data for Figure 6
└── README.md
```

Every notebook is self-contained. The core routines are defined identically in each experiment notebook: `load_dataset`, `screen_columns`, `zscore`, `rectified_cosine_matrix`, `mi_weights`, and `cgfs`.

---

## Installation

The experiments were run with Python 3.10+ (the paper reports Python 3.12). The recorded notebook runs used numpy 1.26.4, pandas 2.1.4, and matplotlib 3.8.2.

```bash
git clone https://github.com/ahmed-elliethy/CGFS.git
cd CGFS

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy pandas scikit-learn matplotlib jupyter
pip install faiss-cpu xgboost      # recommended (see note below)
```

> **Optional dependencies.**
> - **`faiss`**: without it, the notebooks use an exact NumPy/BLAS fallback that produces the same similarities.
> - **`xgboost`**: without it, the XGBoost selector falls back to random-forest importance, and the gradient-boosting classifier falls back to scikit-learn's `HistGradientBoostingClassifier`.
>
> Install both packages to reproduce the paper's numbers. Each notebook prints which backends it detected when it starts.

---

## Datasets

The datasets are **not** redistributed here. Download them from their official sources:

| Dataset | Samples used | Raw features | Classes | Target column | Source |
|---|---|---|---|---|---|
| RT-IoT2022 | 123,117 | 83 | 12 | `Attack_type` | UCI ML Repository |
| Edge-IIoTset | 157,800 | 60 | 15 | `Attack_type` | Ferrag et al. (IEEE Access, 2022) |
| CICIoT2023 | 148,538 (per-class 1/7 sample) | 46 | 23 | `label` | Canadian Institute for Cybersecurity |

The notebooks expect these file names:

| `DATASET_KEY` | Expected file | Columns dropped before selection |
|---|---|---|
| `rt-iot` | `RT_IOT2022.csv` | `Unnamed: 0`, `Attack_type`, `Attack_Label` |
| `edge-iiot` | `ML-EdgeIIoT-dataset.csv` | `Attack_label`, `Attack_type`, plus the leaky host/payload/timestamp columns (see paper, Sec. 4.1.1) |
| `cic-iot` | `CICIoT2023.csv` | `label`, `Binary Label` |

For CICIoT2023, the loader expects a **single merged CSV** that contains a multiclass `label` column and a `Binary Label` column.

### Where to place the files

Each dataset path is built as `os.path.join(DATA_DIR, "../<file>.csv")`. With the default `DATA_DIR = "."`, the CSVs must therefore sit **one level above** the notebook folder:

```
parent/
├── RT_IOT2022.csv
├── ML-EdgeIIoT-dataset.csv
├── CICIoT2023.csv
└── CGFS/                 # this repository
    └── *.ipynb
```

To use a different layout, change `DATA_DIR` in the configuration cell, or edit the `path` field of the `DATASETS` dictionary.

---

## Reproducing the Paper

All notebooks share the same protocol:

1. Stratified 70/30 train/test split.
2. Screening of constant and near-constant columns (modal value ≥ 99.5%) on the training split only.
3. z-scoring with training statistics.
4. Quantile-binned mutual-information weights (`MI_BINS = 32`).
5. Weighted CGFS selection.

Results are reported as mean ± std over seeds.

### Section 4.2: Sensitivity Analysis (choosing k)

**Notebook:** `CGFS_sample_vs_full_experiment.ipynb`
**Produces:** Table 3 (full-data reference), Table 4, and Figure 4.

For each dataset, the notebook draws a class-wise stratified sample, runs weighted CGFS, trains a Random Forest on the top $k \in \{5, 10, 15, 20, 25, 30\}$ features, and compares the resulting weighted-F1 curve with the full-dataset curve.

Key configuration (cell 1):

| Setting | Paper value | Meaning |
|---|---|---|
| `DATASETS_TO_RUN` | `["rt-iot", "edge-iiot", "cic-iot"]` | Datasets to process |
| `SAMPLE_SIZE` | `0.2` for each dataset | 20% stratified sample (an int, fraction, `None`, or per-dataset dict) |
| `SAMPLE_SEEDS` | `[40, 41, 42]` | Three independent sample draws |
| `SEEDS` | `[0, 1, 2, 3, 4]` | Five split/forest seeds per draw (15 runs per budget) |
| `BASE_SAMPLE` | `{"cic-iot": {"frac": 1/7, "random_state": 42}}` | Per-class 1/7 base sample for CICIoT2023, applied automatically |
| `F1_AVERAGE` | `"weighted"` | Must match `FULL_REFERENCE` |
| `RUN_FULL_TOO` | `False` | If `True`, re-runs the full-data pipeline instead of using the stored constants (slow) |
| `FULL_REFERENCE` | pre-filled | Full-dataset RF weighted-F1 (mean, std) from the benchmark run (Table 3) |

The notebook reports two separate spread components. `σ_sampling` is the spread across sample draws and is the shaded band in Figure 4. `σ_split` is the spread across split/forest seeds.

**Outputs:**

- `cgfs_sample_vs_full.png` and `cgfs_sample_vs_full.pdf` (Figure 4)
- `cgfs_sample_raw.csv`, `cgfs_sample_per_draw.csv`, `cgfs_sample_vs_full.csv`
- `cgfs_sample_vs_full.tex` (paste-ready LaTeX tables)

To select $k^\star$, apply the tolerance rule of Eq. (25) with $\tau = 1\%$ to the resulting curves. This gives $k^\star = 25, 25, 10$ on the full data for RT-IoT2022, Edge-IIoTset, and CICIoT2023.

### Sections 4.3, 4.4, 4.5: Detection Performance, SOTA Comparison, Runtime

**Notebook:** `CGFS_weighted_benchmark_experiment.ipynb`
**Produces:** Tables 5–7 (full vs. CGFS subset), Tables 8–10 (comparison with 10 selectors), and Table 11 (feature selection time).

This notebook runs one dataset at a time. For each seed, it runs every selector at every budget. Each selected subset is then used to train five classifiers (RF, DT, histogram GB, linear SVM, and a custom brute-force k-NN), and the notebook records accuracy, precision, recall, F1, FPR, and selection, fit, and inference times. A full-feature "ceiling" run is recorded for each classifier and seed.

#### Configuring a run

Edit the configuration cell (cell 1):

| Setting | Paper value | Notes |
|---|---|---|
| `DATASET_KEY` | `"rt-iot"`, `"edge-iiot"`, or `"cic-iot"` | One dataset per run |
| `DATA_DIR`, `OUT_DIR` | `"."` | Input location and results location |
| `SEEDS` | `[0, 1, 2]` | Three seeds |
| `K_LIST` | `[5, 10, 15, 20, 25, 30]` | The paper tables use k = 25 (RT-IoT2022, Edge-IIoTset) and k = 10 (CICIoT2023) |
| `NORMAL_CLASS` | **dataset-specific (see below)** | Benign class(es) used for the binary false-alarm rate |
| `INCLUDE_MRMR` | `True` | Adds the mRMR baseline |
| `MI_BASELINE` | `"sklearn"` | The Mutual Information baseline uses `mutual_info_classif` (kNN estimator) |
| `CLASSIFIERS_TO_RUN` | `["rf", "dt", "gb", "svm", "knn"]` | `"logreg"` is also available |
| `KNN_MAX_TRAIN` | `20_000` | Reference-set cap for k-NN |
| `SVM_KIND` | `"linear"` | `LinearSVC`, C = 1.0, tol = 1e-3, max_iter = 2000, parallel one-vs-rest |
| `SELECT_PER_K` | `True` | Re-runs every selector at each budget, so that timings are honest |
| `RFE_SUBSAMPLE`, `MI_SUBSAMPLE` | `None` | Use the full training split |

Set `NORMAL_CLASS` to match the dataset:

```python
# RT-IoT2022
NORMAL_CLASS = ["MQTT_Publish", "Thing_Speak", "Wipro_bulb"]
# Edge-IIoTset
NORMAL_CLASS = ["Normal"]
# CICIoT2023
NORMAL_CLASS = ["BenignTraffic"]
```

#### ⚠️ Required step for CICIoT2023

The paper uses a class-wise stratified **1/7** sample of CICIoT2023 (148,538 rows). In this notebook the sampling step is **commented out** inside `load_dataset` (cell 3). Before running with `DATASET_KEY = "cic-iot"`, uncomment these lines:

```python
    df = pd.read_csv(path, low_memory=False)
    # for CIC-IoT2023 dataset only
    sample_fraction = 1/7
    df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=sample_fraction,random_state=42))
```

**Comment them out again** before running RT-IoT2022 or Edge-IIoTset, because those datasets are used in full.

The other three notebooks apply this sample automatically through their `BASE_SAMPLE` setting.

#### Runtime

The full grid is `12 selectors × 6 budgets × 5 classifiers × 3 seeds = 1080` model fits, plus 216 selection runs. On the authors' CPU machine, the recorded RT-IoT2022 run took about **2.6 hours**. The linear SVM, k-NN, and RFE account for most of that time.

To shorten a run, you can:

- Set `K_LIST` to the paper budget only (e.g. `[25]` or `[10]`).
- Reduce `CLASSIFIERS_TO_RUN`.
- Set `SELECT_PER_K = False`.
- Set `RFE_SUBSAMPLE` or `MI_SUBSAMPLE` (e.g. `20_000`).

Some of these changes alter timings or results relative to the paper.

A built-in self-check (`RUN_SELF_CHECK = True`) runs before the experiment. It verifies similarity symmetry and range, $\sum w = d$, that lazy greedy matches standard greedy exactly, submodularity on random pairs, monotone coverage error, and zero weight for constant columns.

#### Outputs

Files are named by dataset key. With `DATASET_KEY = "rt-iot"`:

| File | Content |
|---|---|
| `cgfs_rt-iot_raw.csv` | One row per (seed, selector, k, classifier), including the selected feature names |
| `cgfs_rt-iot_agg.csv` | Mean and std over seeds per (classifier, method, k). This file is used to build the Figure 6 CSV |
| `cgfs_rt-iot_full_ceiling.csv` | Full-feature results (the "Full" rows of Tables 5–7) |
| `cgfs_rt-iot_tables.tex` | LaTeX tables per classifier and metric |

The notebook also prints several extra analyses that are not part of the paper tables:

- selection stability (Jaccard overlap across seeds),
- weighted-vs-unweighted overlap,
- coverage-error traces,
- a timing breakdown,
- an optional per-class report.

### Section 4.6, Figure 5: Coverage Trajectory

**Notebook:** `CGFS_coverage_gain_experiment.ipynb`
**Produces:** Figure 5, which has two panels: the absolute coverage $\mathcal{G}(F_i)$ per dataset, and the fraction $\mathcal{G}(F_i)/d$ of attainable coverage.

The notebook runs the greedy for `KMAX = 25` steps on each dataset with `SEEDS = [0, 1, 2]` and reads the coverage values directly from the returned error trace, using $\mathcal{G}(F_i) = d - E_i$. Because the ceiling $d$ differs per dataset (77, 35, and 34 after screening), only the normalized curve is averaged across datasets.

Useful options:

- `INCLUDE_UNWEIGHTED = True` also traces the unweighted objective.
- `COMPUTE_ONLINE_BOUND = True` reports Minoux's data-dependent bound on $\mathrm{OPT}_k$. This gives an empirical lower bound on the approximation ratio that was actually achieved.
- The notebook checks that the greedy marginal gains are non-increasing for every dataset and seed.

**Outputs:**

- `cgfs_coverage_gain.png` and `cgfs_coverage_gain.pdf` (**Figure 5**)
- `cgfs_marginal_gain.png` and `cgfs_marginal_gain.pdf` (supplementary: log-scale marginal gains, weighted vs. unweighted)
- `cgfs_coverage_trajectory_raw.csv`, `cgfs_coverage_per_dataset.csv`, `cgfs_coverage_across_datasets.csv`
- `cgfs_coverage_gain.tex`

### Section 4.6, Figure 6: Marginal F1 Gains

**Notebook:** `CGFS_f1_marginal_gains_figure.ipynb`
**Input:** `cgfs_f1_marginal_gains.csv` (included in the repository)
**Produces:** Figure 6, a box plot of the marginal weighted-F1 gain over the intervals 0→5, 5→10, 10→15, 15→20, and 20→25.

The CSV has **75 rows**: 3 datasets × 5 classifiers × 5 intervals, so each box summarizes 15 values. Its columns are:

```
dataset, classifier, k_from, k_to, interval, f1_from, f1_to, gain
```

The first interval is measured from a baseline of `f1_from = 0`, which represents a model with no features. The gains in each series add up exactly to the F1 score at k = 25. The F1 values are seed means of weighted CGFS.

To produce the figure, open the notebook and run all cells.

Configuration options:

- `AS_PERCENT` plots percentage points instead of the 0–1 scale.
- `SHOW_POINTS` overlays the individual values.
- `ANNOTATE_MEANS` labels each box with its mean.

**Outputs:**

- `cgfs_f1_marginal_gains.png` and `cgfs_f1_marginal_gains.pdf` (**Figure 6**)
- `cgfs_f1_marginal_gains.tex` (summary statistics table)

**Regenerating the CSV.** The appendix cell of this notebook rebuilds the CSV from the aggregated result files `cgfs_rt-iot_agg.csv`, `cgfs_edge-iiot_agg.csv`, and `cgfs_cic-iot_agg.csv`. To use it, set `REGENERATE = True`. The rebuild keeps the `CGFS-weighted` rows and takes differences of F1 between consecutive budgets. If you point it at the per-run raw files and also group by seed, you get 45 values per box instead of 15.

---

## Using CGFS on Your Own Data

The core routines can be copied out of any experiment notebook (sections 3–6) and used directly:

```python
import numpy as np
from sklearn.model_selection import train_test_split

# X: (n_samples, n_features) numeric array, y: integer-encoded labels
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# 1) Screen constant / near-constant columns (training split only)
keep, kept_names = screen_columns(Xtr, feature_names, enabled=True,
                                  near_const_thresh=0.995)
Xtr, Xte = Xtr[:, keep], Xte[:, keep]

# 2) Standardize with training statistics
Ztr, Zte = zscore(Xtr, Xte)

# 3) Rectified cosine similarity (exact flat FAISS index) and MI weights (sum = d)
sim = rectified_cosine_matrix(Ztr)
w, mi, codes = mi_weights(Xtr, ytr, 32, weighted=True)

# 4) Lazy-greedy CGFS
k = 10
selected, errors = cgfs(sim, w, k)   # errors[i-1] = E_i = d - G(F_i)

print("Selected features:", [kept_names[i] for i in selected])
print("Coverage fraction at k:", 1 - errors[-1] / len(w))
```

- The selection order is nested: `selected[:j]` is the CGFS coreset for any budget `j ≤ k`.
- The coverage-error trace `errors` lets you read off the smallest prefix that meets a coverage tolerance (see Proposition 3.2 in the paper).
- Passing `w = np.ones(d)` gives the unweighted variant.

The dense similarity matrix needs $O(d^2)$ memory and $O(nd^2)$ time to build. CGFS is therefore designed for large sample counts with a moderate number of features.

---

## Mapping Notebook Names to Paper Names

| Notebook identifier | Paper name |
|---|---|
| `CGFS-weighted` | **CGFS** (the proposed method) |
| `CGFS-unweighted` | Ablation, not reported in the paper tables |
| `MutualInfo` | Mutual Information |
| `Chi2` | Chi-Square |
| `XGBoost` | XGBoost importance |
| `L1` | L1 logistic regression |
| `RFE`, `ReliefF`, `mRMR`, `ANOVA`, `Correlation`, `Random` | Same names |
| `rf`, `dt`, `gb`, `svm`, `knn` | RF, DT, GB, SVM, KNN |
| `select_total_s` | **FST**: selector time plus the shared preparation it depends on (similarity matrix and MI weights for CGFS) |
| `fpr_binary` | **FPR** as reported in the paper (benign vs. attack, using `NORMAL_CLASS`) |
| `f1` / `F1(w)` | Support-weighted F1 |

---

## Notes and Known Caveats

- **Leakage-free protocol.** Screening, standardization, MI weights, and feature selection all use the training partition only.
- **Timing.** Reported times depend on hardware, BLAS threading, and library versions. The paper's timings were measured on an Intel Core i7-10750H with 32 GB RAM (CPU only).
- **pandas ≥ 3.0.** The commented `groupby(...).apply(...)` sampling line in the benchmark notebook may drop the label column under pandas 3.0. The other notebooks use an equivalent `apply_base_sample` helper that avoids this problem. If you use pandas 3.x, use that helper for CICIoT2023 in the benchmark notebook as well.
- **Linear SVM convergence.** `LinearSVC` may emit `ConvergenceWarning` at `max_iter = 2000`. This matches the paper's configuration.
- **k-NN.** The reference set is capped at 20,000 stratified instances for every method, so the absolute k-NN accuracies are lower bounds.

---

## Citation

If you use this code, please cite:

```bibtex
@article{amr2026cgfs,
  title   = {Submodular Set Cover-Based Coreset Feature Selection for Efficient
             Intrusion Detection in IoT Networks},
  author  = {Amr, Mohammed Nagah and Elliethy, Ahmed and Mekkawy, Tamer and Mahran, Ashraf},
  journal = {TBD},
  year    = {2026}
}
```

*(Citation details will be updated upon publication.)*

## License

Add your license here (e.g., MIT). The datasets remain subject to the licenses and terms of their original providers.

## Contact

- **Ahmed Elliethy** (corresponding author), Military Technical College, Cairo: a.s.elliethy@mtc.edu.eg
- **Mohammed Nagah Amr**, School of Information Technology, Newgiza University, Cairo

Questions and issues are welcome through the GitHub issue tracker.
