# TabPFN-Anchor

**Anchor-based TabPFN for high-dimensional tabular classification,
with an application to cancer-subtype classification from gene-expression data.**

TabPFN is a pretrained tabular foundation model that works well on small
datasets, but each of its estimators reads only a limited number of features
(500 for the TabPFN-3 checkpoint used here). TabPFN-Anchor lets a frozen TabPFN
backbone use inputs with thousands to tens of thousands of features:

1. **Anchor selection.** A cascaded feature ranking (one-way ANOVA *F* →
   Random Forest importance → TabPFN-based OddSHAP attribution) keeps the top
   100 features as *anchors*.
2. **Stacked multi-view TabPFN ensemble.** Each of *K* = 9 views contains the
   anchors plus a different block of 400 next-ranked features. A frozen TabPFN
   model is fitted to each view, and a logistic-regression meta-learner stacks
   their out-of-fold probabilities.

The result is a scikit-learn classifier you can drop into any benchmark, plus a
compact anchor feature set you can report and evaluate on its own. The method
needs only a numeric feature matrix and class labels; it was developed and
evaluated on cancer gene-expression data (12 TCGA cohorts, about 20,000 genes
each), and its behaviour on other kinds of data has not been evaluated.

![TabPFN-Anchor graphical abstract](docs/graphical_abstract.png)

This is the reference implementation of the following paper, **accepted for
publication in *IEEE Access* (2026)**:

> H. Ahn, "TabPFN-Anchor: Anchor-Conditioned Ensemble Adaptation of a Tabular
> Foundation Model for High-Dimensional Cancer Subtype Classification,"
> *IEEE Access*, 2026. Accepted for publication.

## Installation

Requires **Python 3.12 or later** and, in practice, a GPU.

```bash
git clone https://github.com/aipply-ahn/tabpfn-anchor.git
cd tabpfn-anchor
pip install -e .
```

On first use the `tabpfn` package downloads the TabPFN-3 checkpoint (about
203 MB) after you log in with Prior Labs and accept the model licence; on a
headless machine, accept it on the web and set `TABPFN_TOKEN`. **The TabPFN-3
weights are under a non-commercial licence** — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Pass a single device such as
`device="cuda:0"`.

## Quick start

### A complete example you can paste and run

The data are synthetic, straight from scikit-learn: far more features than
samples, a few informative features, many correlated with them, three classes.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from tabpfn_anchor import TabPFNAnchorClassifier

X, y = make_classification(
    n_samples=300, n_features=4000, n_informative=20, n_redundant=80,
    n_classes=3, random_state=0)
X = pd.DataFrame(X, columns=[f"feature_{j:05d}" for j in range(X.shape[1])])
y = pd.Series([f"Class_{chr(65 + c)}" for c in y])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

clf = TabPFNAnchorClassifier(device="cuda:0")   # defaults = published configuration
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)
print(f1_score(y_test, pred, average="macro"))

clf.anchor_features_    # the 100 anchor features, with attribution rank
clf.feature_ranking_    # every ranked feature and the stage that ranked it
clf.blocks_             # the feature indices given to each view
```

The same code runs as a script with `python -m tabpfn_anchor.demo.sklearn_toy`.
It takes about nine minutes on one GPU, mostly in the OddSHAP step.

## Using your own data

```python
import pandas as pd
from sklearn.model_selection import train_test_split

from tabpfn_anchor import TabPFNAnchorClassifier

# Rows are samples, columns are features (e.g. genes), plus one label column.
df = pd.read_csv("expression.tsv", sep="\t", index_col=0)
y = df["subtype"]
X = df.drop(columns=["subtype"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# preprocessing="rnaseq" applies log2(x + 1) inside fit and predict
# (log transformation only; no library-size normalisation).
# Use "microarray" or "none" for data that are already on a log scale.
clf = TabPFNAnchorClassifier(device="cuda:0", preprocessing="rnaseq")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
```

### Real-data example: TCGA

```bash
tabpfn-anchor download-tcga --cohort UCEC --data-dir ./data
tabpfn-anchor tcga-example  --cohort UCEC --repeat 1 --fold 1 --data-dir ./data
```

`download-tcga` fetches the TCGA release used in the paper from the NCI Genomic
Data Commons, verifies it and extracts one cohort (no data ship with this
repository; see [DATA.md](DATA.md)). `tcga-example` trains and evaluates on
one of the cross-validation splits distributed with that release, so other
methods can be run on exactly the same split. It uses shortened attribution
settings; add `--paper-settings` for the published ones.

## Execution path by number of features

With the default `mode="auto"`, what runs depends on the number of input
features (default settings shown):

| Number of features | What runs | Feature ranking | Views *K* |
|---|---|---|---|
| ≤ 500 | TabPFN-3 directly | none | — |
| 501 – 2,000 | TabPFN-3 directly (each estimator samples 500 features at random) | none | — |
| 2,001 – 5,000 | TabPFN-Anchor | Random Forest → OddSHAP | 4 – 9 |
| > 5,000 | TabPFN-Anchor | ANOVA-*F* → Random Forest → OddSHAP | 9 |

*K* = min(9, ⌊(number of features − 100) / 400⌋), so nine views need at least
3,700 features.

**With 2,000 features or fewer, `mode="auto"` runs TabPFN-3 itself, not the
anchor ensemble, and no anchor signature is produced.** Set `mode="anchor"` to
always run the anchor ensemble, and check `clf.execution_plan_` after fitting.

## Documentation

- [docs/examples.md](docs/examples.md) — worked examples: preprocessing,
  adjusting the cascaded ranking, the number of anchors and the number of views,
  execution modes, reading the outputs, cross-validated benchmarking, shorter
  run times.
- [docs/configuration.md](docs/configuration.md) — method, all main options,
  outputs, input requirements and tests.

## Paper and reproducibility

TabPFN-Anchor was evaluated on 12 TCGA cancer-subtype classification tasks
comprising 6,537 tumours, against four other high-dimensional TabPFN
configurations and 17 general classifiers. The configuration that produced the
reported results, the macro-F1 of every comparison at the level of each outer
split and the script that reproduces the reported statistics are in
[paper/](paper/). See the paper for the full evaluation design.

## Citation

If you use TabPFN-Anchor, please cite the paper (accepted for publication in
*IEEE Access*; volume, pages and DOI will be added on publication):

```bibtex
@article{ahn2026tabpfnanchor,
  author  = {Ahn, Hongryul},
  title   = {{TabPFN-Anchor}: Anchor-Conditioned Ensemble Adaptation of a
             Tabular Foundation Model for High-Dimensional Cancer Subtype
             Classification},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Accepted for publication}
}
```

## License

The TabPFN-Anchor source code is released under the MIT License
([LICENSE](LICENSE)). This repository contains no TabPFN code or model weights;
the TabPFN-3 checkpoint is licensed separately by Prior Labs (non-commercial
research and limited internal evaluation; commercial use restricted). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [DATA.md](DATA.md).
