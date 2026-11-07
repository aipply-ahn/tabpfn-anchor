# TabPFN-Anchor

High-dimensional classification with anchor-conditioned TabPFN.

![TabPFN-Anchor graphical abstract](docs/graphical_abstract.png)

| 100 | 9 × 500 | 12 |
|:---:|:---:|:---:|
| shared anchor features | feature views on a frozen TabPFN-3 | TCGA cancer-subtype tasks · 6,537 tumours |

<sub>Published configuration. Evaluated on cancer gene-expression data; other domains have not been evaluated.
Paper: <i>IEEE Access</i>, 2026 (accepted) · <a href="#citation">citation</a></sub>

## Installation

### Package installation

Python 3.12+ · NVIDIA GPU recommended

```bash
git clone https://github.com/aipply-ahn/tabpfn-anchor.git
cd tabpfn-anchor
pip install -e .
```

### TabPFN-3 checkpoint installation

TabPFN-Anchor runs on a frozen TabPFN-3 backbone. By default it loads the multiclass
checkpoint used in the paper, `tabpfn-v3-classifier-v3_20260417_multiclass.ckpt`.
The checkpoint can be obtained in one of two ways:

**1) Automatic** — sign in with Prior Labs and accept the TabPFN-3 licence
([model page](https://huggingface.co/Prior-Labs/tabpfn_3)). The checkpoint then downloads
automatically on the first run. On a server without a browser, set `TABPFN_TOKEN`
([setup guide](docs/installation.md)).

**2) Manual** — download the checkpoint file from the
[model page](https://huggingface.co/Prior-Labs/tabpfn_3), then run the script below to install it:

```bash
tabpfn-anchor checkpoint --install /path/to/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt
tabpfn-anchor checkpoint      # check: expected location and status
```

Model weights are under a non-commercial licence ([details](THIRD_PARTY_NOTICES.md)).

### Installation test

Once the package and the checkpoint are installed, run:

```bash
tabpfn-anchor check
```

Both installations are confirmed when you see:

```text
[1/3] Package installation       OK  (tabpfn-anchor 0.1.0, tabpfn 8.4.0)
[2/3] Checkpoint installation    OK  (~/.cache/tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt)
[3/3] Backbone test              OK  (TabPFN-3 loaded on cuda:0)

TabPFN-Anchor is installed and ready.
```

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

## Parameters

### Feature selection — `mode`

- `"auto"` (default) — chooses the route by the number of input features (table below)
- `"anchor"` — always the anchor ensemble (nine views need ≥ 3,700 features)
- `"tabpfn"` — always native TabPFN-3, no feature selection

| Input features | `mode="auto"` runs | Feature ranking | Views |
|---|---|---|---|
| ≤ 2,000 | native TabPFN-3 | — | — |
| 2,001 – 5,000 | TabPFN-Anchor | Random Forest → OddSHAP | 4 – 9 |
| > 5,000 | TabPFN-Anchor | ANOVA-*F* → Random Forest → OddSHAP | 9 |

### Preprocessing — `preprocessing`

- `"rnaseq"` — log₂(x + 1), for non-negative values; no library-size normalisation
- `"microarray"` — values used as given
- `"none"` (default) — values used as given

Applied in both `fit` and `predict`.

### Others

| Parameter | Default | Controls |
|---|---|---|
| `stack_size` | 9 | number of views |
| `n_anchor` · `n_residual` | 100 · 400 | anchors and view-specific features per view (sum = 500) |
| `n_stage1` · `n_stage2` | 2000 · 500 | features kept by the ANOVA-*F* and Random Forest stages |
| `anchor_source` | `"shap"` | `"rf"` skips OddSHAP (faster) |
| `device` | `"auto"` | set one GPU, e.g. `"cuda:0"` |

All parameters: [configuration](docs/configuration.md) · worked examples: [examples](docs/examples.md)

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
