# TabPFN-Anchor

Reference implementation of

> **TabPFN-Anchor: Anchor-Conditioned Ensemble Adaptation of a Tabular
> Foundation Model for High-Dimensional Cancer Subtype Classification**
> Hongryul Ahn, Division of AI and Data Science, The University of Suwon

A transcriptome has around twenty thousand genes, far more than the number of
features a TabPFN estimator handles well at once. TabPFN-Anchor uses the subtype
labels to decide *which* genes each fixed-width model receives, rather than
leaving that to the backbone: a cascaded ranking produces a 100-gene **anchor
set** that can be reported on its own, and the genes ranked below it supply
ordered additional context to nine 500-gene views whose predictions are combined
by a stacked meta-learner. The 500-gene view width is a setting of this
procedure, not a property of any TabPFN release.

The estimator is an ordinary scikit-learn classifier.

---

## Install

```bash
pip install -e .
```

A GPU is strongly recommended. The TabPFN backbone checkpoint is downloaded by
the `tabpfn` package on first use.

## Running it on real data

```bash
tabpfn-anchor download-tcga --cohort UCEC --data-dir ./data
tabpfn-anchor tcga-example  --cohort UCEC --repeat 1 --fold 1 --data-dir ./data
```

`download-tcga` fetches the release archive from the NCI Genomic Data Commons,
checks it against the published size and MD5, and extracts one cohort. **No
data ships with this repository**; everything it writes goes to a local
directory that `.gitignore` excludes. See [DATA.md](DATA.md).

`tcga-example` reads one column of the cross-validation assignment distributed
with that release rather than calling `train_test_split`, so the partition is
not ours and another method can be run on the same column. Selection is fitted
on the training partition alone. It writes the anchor set, the full ranking,
the metrics and the resolved configuration to `results/`.

**This is one split, not the paper.** The reported numbers are means over
twelve cohorts and many splits with a paired test across them. A single split
has a standard error wide enough that comparing its macro-F1 against a number
in the paper says nothing. Treat it as a demonstration that the method runs end
to end on real data.

The example preset cuts the attribution budget so the command finishes; the
view construction is the published one (100 anchors, 400 residual genes per
view, K = 9, 3,700 distinct genes). `--paper-settings` restores the published
budgets, which are far slower.

If you only want to check the installation, the Python example below runs on
synthetic data and needs no download.

### The backbone

The `tabpfn` package downloads the checkpoint on first use: it opens a browser
to log in with Prior Labs, requires you to accept the model licence, and
fetches about 203 MB. On a headless machine, accept the licence through the web
interface and set `TABPFN_TOKEN` first. **The TabPFN-3 weights are under a
non-commercial licence** — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Almost all of the runtime is the OddSHAP attribution stage; the screen and the
Random Forest refinement together take under a second even at transcriptome
scale.

**Name a single device.** With several GPUs visible, the backbone's `auto`
setting was measured about fifty times slower than naming one device (68 s
against 1.4 s to fit one view model on 240 samples by 60 features), and it
places a context on every visible GPU. The CLI therefore selects `cuda:0`
itself and tells you so; override with `--device`. In Python, pass
`TabPFNAnchorClassifier(device="cuda:0")` rather than leaving the default.

## Python API

### A complete example you can paste and run

The data are synthetic, straight from scikit-learn. They contain nothing from
any cohort; the values are not on an expression scale and the columns are not
genes. What they reproduce is the shape of the task — far more features than
samples, a few informative features, many correlated with those, imbalanced
classes.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from tabpfn_anchor import TabPFNAnchorClassifier

X, y = make_classification(
    n_samples=300, n_features=3_000, n_informative=20, n_redundant=80,
    n_classes=4, n_clusters_per_class=2, weights=[.40, .28, .20, .12],
    class_sep=2.5, flip_y=0.01, random_state=0)
X = pd.DataFrame(X, columns=[f"feature_{j:05d}" for j in range(X.shape[1])])
y = pd.Series([f"Class_{chr(65 + c)}" for c in y])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

clf = TabPFNAnchorClassifier(device="cuda:0")   # defaults = published configuration
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)
print(f1_score(y_test, pred, average="macro"))

clf.anchor_features_    # the 100 anchor genes, with attribution rank
clf.feature_ranking_    # every ranked gene, with the stage it was ranked in
clf.blocks_             # the gene indices given to each view
```

That runs the published configuration, which takes minutes rather than seconds;
`example/run.py` is the same thing with the budgets cut. Do not read anything
into the score: the difficulty of `make_classification` output is whatever its
arguments make it.

### Loading your own data

`fit` takes anything scikit-learn takes. A DataFrame is worth preferring: the
classifier slices features by position, remembers the column names, and refuses
a differently ordered frame at predict time — with a bare array it cannot check.

```python
import numpy as np
import pandas as pd

# Rows are samples, columns are genes, plus one column holding the label.
df = pd.read_csv("expression.tsv", sep="\t", index_col=0)

y = df["subtype"]                       # whatever your label column is called
X = df.drop(columns=["subtype"])

# RNA-seq counts need a log scale; skip this if your matrix is already logged
# (microarray usually is, and it may contain negative values).
if X.to_numpy().min() >= 0 and X.to_numpy().max() >= 50:
    X = np.log2(X + 1.0)
```

Requirements, all of which raise a clear error rather than failing quietly:

| | |
|---|---|
| `X` | numeric, no missing values, unique column names |
| `y` | one label per row, at least two classes |
| class sizes | the smallest class needs at least two members for the stacking folds |
| features | see the note below on how many |

### How many features you need

The default `mode="auto"` skips the whole anchor construction when the input
has **2,000 features or fewer** and calls TabPFN directly, because building
nine 500-gene views out of 1,000 genes is not a meaningful decomposition. That
is the right default for a transcriptome, but it surprises people testing on a
small table:

```python
X, y = make_classification(n_samples=300, n_features=1_000, n_informative=20)
clf = TabPFNAnchorClassifier().fit(X, y)
clf.anchor_features_                    # AttributeError: no ranking was built
```

Pass `mode="anchor"` to force the full cascade on a narrow input, or use data
with more than 2,000 features. The error message says which case you are in.

## Method

| Step | What it does | Keeps |
|---|---|---|
| 1.1 | one-way ANOVA *F* screen of the transcriptome | 2,000 genes |
| 1.2 | Random Forest feature importance (mean decrease in impurity) | 500 genes |
| 1.3 | OddSHAP attribution against the frozen backbone | **100 anchors** |
| 2.1 | build *K* = 9 views, each the anchors plus a disjoint block of 400 lower-ranked genes | 3,700 distinct genes |
| 2.2 | fit one view model per view; combine out-of-fold probabilities with an L2 logistic meta-learner | |

The expensive stage is 1.3, which is about four fifths of the cost. What it
buys is not accuracy but a compact gene set that can be reported and evaluated
independently of the ensemble.

This repository implements the published method. The ablation variants used
in the paper's ablation tables (an earlier Step 1.1 ranker, an earlier
attribution backend) are not included: publishing two implementations invites
confusion about which one produced the results.

## Published configuration and data provenance

**This is a reference implementation of the method, not the benchmark harness.**
It does not ship the code that ran twelve cohorts across repeated
cross-validation, aggregated macro-F1, applied the multiple-comparison
correction, or generated the paper's tables and figures. There is no single
command here that reproduces the reported numbers, and you should not expect
one.

What this repository does give you is the method itself, the exact
configuration it was run under, and the provenance of the data:

- `configs/paper/tabpfn_anchor.yaml` records the configuration that produced
  the reported results, transcribed from the configuration file the run itself
  wrote. Every value in it also equals the estimator's default, and
  `tests/test_api_contract.py` compares the whole file against
  `get_params()`, so the two cannot drift apart.
- `--paper-settings` reads that file rather than relying on the defaults.
- [DATA.md](DATA.md) identifies the source archive by GDC UUID and MD5, and
  the cross-validation splits distributed with it. Those splits are not ours;
  the paper uses the first three repeats for its 15-split analyses and the
  first ten for its 50-split analyses.
- `configs/legacy/` records defaults that were superseded on 2026-09-02. They
  were never the published method: every configuration that produced a reported
  result set those values explicitly, so no reported number depends on them.

Running the published configuration on real data is slow — roughly eighteen
minutes per outer split on one GPU for a TCGA-sized cohort, about four fifths
of it in the OddSHAP stage.

## scikit-learn compliance

`clone`, `get_params`, `set_params`, `classes_`, `n_features_in_`,
`feature_names_in_`, `predict`, `predict_proba` and the inherited `score`
behave as scikit-learn requires, and `tests/test_api_contract.py` asserts each
of them.

`sklearn.utils.estimator_checks.check_estimator` is **not** run as a whole. It
refits the estimator many times on small low-dimensional arrays, whereas this
estimator needs a downloaded backbone checkpoint, a GPU to be practical, and an
input with more features than samples — a regime the generic checks do not
produce. The individually meaningful checks are asserted instead.

```bash
pip install -e ".[test]"
pytest -q
```

Tests that require the backbone checkpoint skip themselves when it is absent.

## Citation

See `CITATION.cff`.

## License

The TabPFN-Anchor source code in this repository is licensed under the MIT
License (`LICENSE`).

This repository distributes no TabPFN source code and no model weights. The
default configuration requires a separately distributed TabPFN-3 checkpoint,
which is licensed under its own terms; those permit non-commercial research and
limited internal evaluation and restrict commercial and production use. You
must obtain the checkpoint from Prior Labs, accept its licence and comply with
it.

**The MIT License here grants no rights to the TabPFN package, the model
weights, the model outputs, or any third-party dataset.** A repository marked
MIT does not make TabPFN-3 usable commercially. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [DATA.md](DATA.md).
