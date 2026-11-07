# TabPFN-Anchor — usage examples

Each example is collapsed; click its title to expand it. All examples assume

```python
from tabpfn_anchor import TabPFNAnchorClassifier
```

and a feature matrix `X` (samples × features, ideally a pandas DataFrame) with
labels `y`. The defaults are the published configuration; every example below
changes only what it names.

---

<details>
<summary><b>1. Preprocessing: RNA-seq, microarray or already preprocessed data</b></summary>

The `preprocessing` option is applied inside `fit` and `predict`, so training
and test data always receive the same transformation.

```python
# RNA-seq expression values that are non-negative and not yet log-transformed:
# X is replaced by log2(X + 1) inside fit and predict.
clf = TabPFNAnchorClassifier(device="cuda:0", preprocessing="rnaseq")

# Microarray data, already normalised and on a log scale: used as given.
clf = TabPFNAnchorClassifier(device="cuda:0", preprocessing="microarray")

# Data you have preprocessed yourself (the default): used as given.
clf = TabPFNAnchorClassifier(device="cuda:0", preprocessing="none")
```

- `"rnaseq"` applies only the log transformation; it does not perform
  library-size normalisation or any other RNA-seq preprocessing.
- Negative inputs are not accepted by `"rnaseq"` and raise an error.
- Do not apply `log2(x + 1)` yourself **and** pass `preprocessing="rnaseq"`;
  that transforms the data twice.
- The published experiments applied `log2(x + 1)` to the RNA-seq cohorts before
  calling the estimator, so the paper configuration uses `"none"`.

</details>

<details>
<summary><b>2. Cascaded gene ranking: how many features each stage keeps</b></summary>

Step 1 ranks the features in three stages. Each stage keeps a number of
features that the next stage re-ranks:

| Parameter | Stage | Default |
|---|---|---|
| `n_stage1` | one-way ANOVA *F* screen keeps this many | 2,000 |
| `n_stage2` | Random Forest importance keeps this many | 500 |
| `n_anchor` | TabPFN-based OddSHAP keeps this many anchors | 100 |

They must satisfy `n_anchor ≤ n_stage2 ≤ n_stage1`.

```python
# A wider ANOVA-F screen before the Random Forest:
clf = TabPFNAnchorClassifier(device="cuda:0", n_stage1=3000)
```

`n_stage2` is also the width of every view (see example 3), so it is best kept
at 500 for the TabPFN-3 checkpoint, which uses at most 500 features per
estimator; a wider view would be subsampled at random inside the backbone.

To skip the OddSHAP stage and use the two-stage ranking (ANOVA-*F* → Random
Forest) directly, which is much faster:

```python
clf = TabPFNAnchorClassifier(device="cuda:0", anchor_source="rf")
```

The anchors are then the top Random Forest features. In the paper this
two-stage ranking gave comparable ensemble macro-F1 at about one fifth of the
training time; OddSHAP is what defines the TabPFN-contextual anchor signature.

</details>

<details>
<summary><b>3. Anchors: how many features every view shares</b></summary>

Every view contains the same `n_anchor` anchors plus `n_residual` further
features of its own. The view width is `n_anchor + n_residual`, which must
equal `n_stage2`:

```python
# Fewer shared anchors, more view-specific features (50 + 450 = 500):
clf = TabPFNAnchorClassifier(device="cuda:0", n_anchor=50, n_residual=450)

# More shared anchors (200 + 300 = 500):
clf = TabPFNAnchorClassifier(device="cuda:0", n_anchor=200, n_residual=300)
```

`n_anchor=50` with the default `n_residual=400` raises an error, because the
view would no longer be 500 features wide.

The number of distinct features used by the ensemble is
`n_stage2 + n_residual × (K − 1)`: 3,700 by default, 4,100 with 50 anchors and
2,900 with 200 anchors (nine views each).

</details>

<details>
<summary><b>4. Number of views K</b></summary>

`stack_size` sets how many views (and TabPFN view models) are stacked:

```python
clf = TabPFNAnchorClassifier(device="cuda:0", stack_size=5)    # 2,100 distinct features
clf = TabPFNAnchorClassifier(device="cuda:0", stack_size=15)   # 6,100 distinct features
```

- Each view adds one TabPFN fit, so training and prediction time grow roughly
  in proportion to K; the ranking step does not depend on K.
- With `mode="auto"`, K is reduced automatically when the input is too narrow:
  at most ⌊(number of features − `n_anchor`) / `n_residual`⌋ complete views fit
  (9 views need 3,700 features with the defaults). A warning says so, and
  `clf.execution_plan_["effective_n_views"]` records the value used.
- In the paper, K between 1 and 30 changed mean macro-F1 only slightly; K = 9
  was fixed on a separate development cohort.

</details>

<details>
<summary><b>5. Execution mode: automatic, always the anchor ensemble, or plain TabPFN</b></summary>

```python
clf = TabPFNAnchorClassifier(device="cuda:0", mode="auto")     # default
clf = TabPFNAnchorClassifier(device="cuda:0", mode="anchor")   # always the anchor ensemble
clf = TabPFNAnchorClassifier(device="cuda:0", mode="tabpfn")   # always TabPFN-3 directly
```

| `mode` | Behaviour |
|---|---|
| `"auto"` | TabPFN-3 directly up to 2,000 features; the anchor ensemble above (the ANOVA-*F* screen is skipped up to 5,000 features); K reduced if the input is too narrow |
| `"anchor"` | always the full anchor ensemble; raises an error instead of adapting (with nine views it needs at least 3,700 features — lower `stack_size` for narrower inputs) |
| `"tabpfn"` | always TabPFN-3 directly, no ranking and no anchors |

Check which path actually ran:

```python
clf.fit(X_train, y_train)
clf.execution_plan_["route"]               # "direct_tabpfn", "rf_shap_anchor" or "screen_rf_shap_anchor"
clf.execution_plan_["effective_n_views"]   # K actually used
clf.resolved_config_                       # every resolved setting, including preprocessing
```

Results from the `"direct_tabpfn"` route are plain TabPFN-3 results and should
not be reported as TabPFN-Anchor results.

</details>

<details>
<summary><b>6. Reading the outputs: anchor signature, ranking and views</b></summary>

```python
clf.fit(X_train, y_train)

clf.anchor_features_      # DataFrame: anchor_rank, feature, importance, source_stage
clf.feature_ranking_      # DataFrame: rank, feature, source_stage, score, is_anchor
clf.blocks_               # list of feature-index arrays, one per view

# Save the anchor signature:
clf.anchor_features_.to_csv("anchor_signature.csv", index=False)

# Evaluate the 100 anchors on their own with any other classifier:
from sklearn.linear_model import LogisticRegression
anchors = clf.anchor_features_["feature"].tolist()
LogisticRegression(max_iter=5000).fit(X_train[anchors], y_train)
```

Feature names are reported when `X` is a DataFrame; with a NumPy array the
columns are identified by index.

</details>

<details>
<summary><b>7. Benchmarking against other classifiers with cross-validation</b></summary>

The estimator follows the scikit-learn contract, so it fits into the same
cross-validation loop as any other classifier. Feature selection happens inside
`fit`, on each training fold only.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
models = {
    "TabPFN-Anchor": TabPFNAnchorClassifier(device="cuda:0", preprocessing="none"),
    "Random forest": RandomForestClassifier(n_estimators=500, random_state=0),
}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
```

With the published configuration each TabPFN-Anchor fold takes on the order of
minutes to tens of minutes on one GPU, depending on cohort size and the number
of classes (about 18 minutes per split for a TCGA-sized cohort).

</details>

<details>
<summary><b>8. Shorter run time</b></summary>

Almost all of the run time is the OddSHAP attribution step, whose cost grows
with (explained samples) × (classes) × (coalition budget). These settings
reduce it:

```python
clf = TabPFNAnchorClassifier(
    device="cuda:0",
    shap_max_samples=16,    # explained samples (default 32)
    shap_budget=256,        # coalition budget per explanation (default 1024)
    rf_n_estimators=500,    # Random Forest trees (default 2000)
)
```

or skip OddSHAP entirely with `anchor_source="rf"` (example 2). These are not
the published settings, so results should not be reported as reproductions of
the paper.

</details>

---

See the [README](../README.md) for installation and the backbone licence, and
[configuration.md](configuration.md) for all options and outputs.
