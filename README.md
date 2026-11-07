# TabPFN-Anchor

High-dimensional classification with anchor-conditioned TabPFN.

![TabPFN-Anchor graphical abstract](docs/graphical_abstract.png)

| 100 | 9 × 500 | 12 |
|:---:|:---:|:---:|
| shared anchor features | feature views on a frozen TabPFN-3 | TCGA cancer-subtype tasks · 6,537 tumours |

<sub>Published configuration. Evaluated on cancer gene-expression data; other domains have not been evaluated.
Paper: <i>IEEE Access</i>, 2026 (accepted) · <a href="#citation">citation</a></sub>

## Installation

Python 3.12+ · NVIDIA GPU recommended

```bash
git clone https://github.com/aipply-ahn/tabpfn-anchor.git
cd tabpfn-anchor
pip install -e .
```

### TabPFN-3 checkpoint

[Model access](https://huggingface.co/Prior-Labs/tabpfn_3) · [Setup guide](docs/installation.md)

Sign in with Prior Labs and accept the licence once. The checkpoint then downloads automatically on the first run.

Already downloaded the file? Put it in place:

```bash
tabpfn-anchor checkpoint --install /path/to/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt
tabpfn-anchor checkpoint      # show the expected location and status
```

Model weights are under a non-commercial licence ([details](THIRD_PARTY_NOTICES.md)).

## Quick start

### 01 — Toy data

Run a synthetic classification example (~9 min on one GPU).

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

### 02 — Your data

Load a CSV/TSV → fit → predict.

| sample_id | GENE_A | GENE_B | GENE_C | … | subtype |
|---|---|---|---|---|---|
| Sample_001 | 12.5 | 38.2 | 0.0 | … | A |
| Sample_002 | 5.1 | 10.7 | 24.3 | … | B |
| Sample_003 | 32.8 | 7.4 | 16.2 | … | C |

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn_anchor import TabPFNAnchorClassifier

df = pd.read_csv("expression.tsv", sep="\t", index_col=0)
y, X = df["subtype"], df.drop(columns=["subtype"])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

clf = TabPFNAnchorClassifier(device="cuda:0", preprocessing="rnaseq")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
```

### 03 — TCGA data

Run one cross-validation split from the paper, with shortened settings (`--paper-settings` for the published ones).

```bash
tabpfn-anchor download-tcga --cohort UCEC --data-dir ./data
tabpfn-anchor tcga-example  --cohort UCEC --repeat 1 --fold 1 --data-dir ./data
```

## Execution

### What runs, by number of features

| Input features | Model | Feature ranking | Views |
|---|---|---|---|
| ≤ 500 | native TabPFN-3 | — | — |
| 501 – 2,000 | native TabPFN-3 | — | — |
| 2,001 – 5,000 | TabPFN-Anchor | Random Forest → OddSHAP | 4 – 9 |
| > 5,000 | TabPFN-Anchor | ANOVA-*F* → Random Forest → OddSHAP | 9 |

**Up to 2,000 features**, the default `mode="auto"` runs native TabPFN-3.
Use `mode="anchor"` to run the anchor ensemble instead.

### Input preprocessing

- `rnaseq` — log₂(x + 1), for non-negative values; no library-size normalisation
- `microarray` — values used as given
- `none` — values used as given (default)

Applied in both `fit` and `predict`.

More options: [configuration](docs/configuration.md) · [examples](docs/examples.md)

## Documentation

| | |
|---|---|
| [Installation](docs/installation.md) | environment, checkpoint, troubleshooting |
| [Examples](docs/examples.md) | preprocessing, anchors, views, modes, benchmarking |
| [Configuration](docs/configuration.md) | all options and outputs |
| [Paper materials](paper/) | configuration, split-level results, statistics |

## Citation

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

Code: MIT ([LICENSE](LICENSE)) · TabPFN-3 weights: Prior Labs, non-commercial ([notices](THIRD_PARTY_NOTICES.md)) · Data: none included ([DATA.md](DATA.md))
