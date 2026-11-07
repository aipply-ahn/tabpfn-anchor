# TabPFN-Anchor

**Anchor-conditioned TabPFN for high-dimensional classification.**

![TabPFN-Anchor graphical abstract](docs/graphical_abstract.png)

**Problem**\
High-dimensional classification under the TabPFN-3 feature budget (≤ 500 features per estimator)

**Approach**\
Three-stage feature ranking → 100 shared anchors → nine 500-feature views → stacked predictions

**Outputs**\
Class predictions + compact 100-feature signature

**Evaluation**\
12 TCGA cancer-subtype tasks · 6,537 tumours · ~20,000 genes per sample

**Scope**\
Evaluated on cancer gene-expression data. Performance on other data domains has not been evaluated.

**Paper**\
*IEEE Access*, 2026 — accepted for publication ([citation](#citation))

## Installation

**Requirements:** Python 3.12+ · NVIDIA GPU recommended

```bash
git clone https://github.com/aipply-ahn/tabpfn-anchor.git
cd tabpfn-anchor
pip install -e .
```

**TabPFN-3 checkpoint** (licence-gated; one-time setup)

1. **Automatic** — run any example; sign in with Prior Labs and accept the licence when the browser opens.
2. **Server without a browser** — accept the licence on the web, then `export TABPFN_TOKEN=<your-token>`.
3. **Manual** — download `tabpfn-v3-classifier-v3_20260417_multiclass.ckpt` from
   [huggingface.co/Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3), then:

```bash
tabpfn-anchor checkpoint --install /path/to/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt
tabpfn-anchor checkpoint      # shows the expected location and whether the file is present
```

※ Location: `$TABPFN_MODEL_CACHE_DIR` if set, otherwise `~/.cache/tabpfn/`.

※ The TabPFN-3 weights are non-commercial ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Quick start

### [1] Toy data

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

※ ~9 min on one GPU (paper settings). Same code as a script: `python -m tabpfn_anchor.demo.sklearn_toy`

### [2] Your data

`expression.tsv` — one row per sample, one column per feature, plus the label column:

| sample_id | GENE_A | GENE_B | GENE_C | … | subtype |
|---|---|---|---|---|---|
| Sample_001 | 12.5 | 38.2 | 0.0 | … | A |
| Sample_002 | 5.1 | 10.7 | 24.3 | … | B |
| Sample_003 | 32.8 | 7.4 | 16.2 | … | C |

*(illustrative values)*

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

※ `preprocessing` — see [Execution notes](#execution-notes).

### [3] TCGA data (paper)

```bash
tabpfn-anchor download-tcga --cohort UCEC --data-dir ./data
tabpfn-anchor tcga-example  --cohort UCEC --repeat 1 --fold 1 --data-dir ./data
```

※ Runs one of the paper's cross-validation splits; it does not regenerate the full paper results.

※ Uses shortened attribution settings; add `--paper-settings` for the published ones.

※ No data ship with this repository ([DATA.md](DATA.md)).

## Execution notes

### Preprocessing

※ `preprocessing="rnaseq"` — applies `log2(x + 1)` inside `fit` and `predict`. Non-negative values only. No library-size normalisation.

※ `preprocessing="microarray"` or `"none"` (default) — values are used as given.

※ Do not log-transform the same data twice.

### Automatic execution (`mode="auto"`)

| Input features | What runs | Ranking | Views *K* |
|---|---|---|---|
| ≤ 500 | TabPFN-3 (native) | — | — |
| 501 – 2,000 | TabPFN-3 (native, random feature sampling) | — | — |
| 2,001 – 5,000 | TabPFN-Anchor | RF → OddSHAP | 4 – 9 |
| > 5,000 | TabPFN-Anchor | ANOVA-*F* → RF → OddSHAP | 9 |

※ ≤ 2,000 features: native TabPFN-3 runs, not the anchor ensemble.

※ To force the anchor ensemble: `mode="anchor"` (nine views need ≥ 3,700 features; lower `stack_size` otherwise).

※ Check what ran: `clf.execution_plan_`.

More options: [configuration reference](docs/configuration.md) · [usage examples](docs/examples.md)

## Documentation

| Doc | For |
|---|---|
| [docs/installation.md](docs/installation.md) | environment, checkpoint, troubleshooting |
| [docs/examples.md](docs/examples.md) | 8 runnable scenarios: preprocessing, anchors, views, modes, benchmarking |
| [docs/configuration.md](docs/configuration.md) | all options, execution modes, outputs |
| [paper/](paper/) | paper configuration, split-level results, statistics script |

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

- **Code:** MIT ([LICENSE](LICENSE)).
- **TabPFN-3 weights:** separate Prior Labs licence, non-commercial ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
- **Data:** none included ([DATA.md](DATA.md)).
