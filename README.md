# TabPFN-Anchor

**Anchor-conditioned TabPFN for high-dimensional classification.**

- **Select** — rank features in three stages and keep 100 shared *anchors*.
- **Adapt** — fit a frozen TabPFN-3 on nine 500-feature views (anchors + ranked features).
- **Predict** — stack the view predictions; keep the anchors as a compact feature signature.

Evaluated on 12 TCGA cancer-subtype tasks (6,537 tumours, ~20,000 genes each).
Performance on other data domains has not been evaluated.

![TabPFN-Anchor graphical abstract](docs/graphical_abstract.png)

Paper: *IEEE Access*, 2026 — accepted for publication ([citation](#citation)).

## Installation

**Requirements:** Python 3.12+ · NVIDIA GPU recommended

```bash
git clone https://github.com/aipply-ahn/tabpfn-anchor.git
cd tabpfn-anchor
pip install -e .
```

**TabPFN-3 checkpoint** — downloads on first use after you sign in with Prior Labs
and accept the licence (non-commercial). Details: [docs/installation.md](docs/installation.md).

## Quick start

### A complete example you can paste and run

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

> **Runtime:** ~9 min on one GPU (paper settings)
> **Outputs:** predictions · 100 anchor features · full feature ranking
> **As a script:** `python -m tabpfn_anchor.demo.sklearn_toy`

## Using your own data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn_anchor import TabPFNAnchorClassifier

df = pd.read_csv("expression.tsv", sep="\t", index_col=0)   # rows = samples, columns = genes
y, X = df["subtype"], df.drop(columns=["subtype"])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

clf = TabPFNAnchorClassifier(device="cuda:0", preprocessing="rnaseq")   # applies log2(x + 1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
```

| `preprocessing` | Use for | Transformation |
|---|---|---|
| `"rnaseq"` | non-negative RNA-seq values | `log2(x + 1)` |
| `"microarray"` | normalised, log-scale arrays | none |
| `"none"` (default) | data you preprocessed yourself | none |

**Real data (TCGA, paper splits):**

```bash
tabpfn-anchor download-tcga --cohort UCEC --data-dir ./data
tabpfn-anchor tcga-example  --cohort UCEC --repeat 1 --fold 1 --data-dir ./data
```

## Execution by feature count

| Features | What runs | Ranking | Views *K* |
|---|---|---|---|
| ≤ 500 | TabPFN-3 (native) | — | — |
| 501 – 2,000 | TabPFN-3 (native, random feature sampling) | — | — |
| 2,001 – 5,000 | TabPFN-Anchor | RF → OddSHAP | 4 – 9 |
| > 5,000 | TabPFN-Anchor | ANOVA-*F* → RF → OddSHAP | 9 |

> ⚠️ **≤ 2,000 features:** native TabPFN-3 runs, not TabPFN-Anchor. Use `mode="anchor"` to force it.

## Documentation

| Doc | For |
|---|---|
| [docs/installation.md](docs/installation.md) | environment, TabPFN-3 sign-in, troubleshooting |
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
