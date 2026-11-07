# Examples

Click a title to expand it. All examples assume:

```python
from tabpfn_anchor import TabPFNAnchorClassifier
```

- `X`: samples × features (pandas DataFrame recommended)
- `y`: class labels
- Defaults = the published configuration. Each example changes only what it shows.

Options reference: [configuration.md](configuration.md)

---

<details>
<summary><b>1. Preprocessing — RNA-seq, microarray, already processed</b></summary>

```python
TabPFNAnchorClassifier(device="cuda:0", preprocessing="rnaseq")      # X -> log2(X + 1)
TabPFNAnchorClassifier(device="cuda:0", preprocessing="microarray")  # X as given
TabPFNAnchorClassifier(device="cuda:0", preprocessing="none")        # X as given (default)
```

- Applied inside both `fit` and `predict`.
- `"rnaseq"`: log transform only — no library-size normalisation.
- `"rnaseq"` rejects negative values.
- Do not log-transform yourself *and* pass `"rnaseq"`.
- Paper runs log-transformed RNA-seq beforehand, so they use `"none"`.

</details>

<details>
<summary><b>2. Cascaded feature ranking — how many features each stage keeps</b></summary>

| Parameter | Stage | Default |
|---|---|---|
| `n_stage1` | ANOVA-*F* screen | 2,000 |
| `n_stage2` | Random Forest importance | 500 |
| `n_anchor` | TabPFN-3 + OddSHAP | 100 |

```python
TabPFNAnchorClassifier(device="cuda:0", n_stage1=3000)       # wider first screen
TabPFNAnchorClassifier(device="cuda:0", anchor_source="rf")  # skip OddSHAP (faster)
```

- Rule: `n_anchor ≤ n_stage2 ≤ n_stage1`.
- Keep `n_stage2 = 500`: it is the view width, and TabPFN-3 reads ≤ 500 features per estimator.
- `anchor_source="rf"`: anchors = top Random Forest features. In the paper: similar
  ensemble macro-F1, about 1/5 of the training time.

</details>

<details>
<summary><b>3. Anchors — how many features every view shares</b></summary>

```python
TabPFNAnchorClassifier(device="cuda:0", n_anchor=50,  n_residual=450)  # 50 + 450 = 500
TabPFNAnchorClassifier(device="cuda:0", n_anchor=200, n_residual=300)  # 200 + 300 = 500
```

- Rule: `n_anchor + n_residual = n_stage2` (500). Otherwise an error is raised.
- Distinct features used (9 views): 4,100 with 50 anchors · 3,700 default · 2,900 with 200 anchors.

</details>

<details>
<summary><b>4. Number of views K</b></summary>

```python
TabPFNAnchorClassifier(device="cuda:0", stack_size=5)    # 2,100 distinct features
TabPFNAnchorClassifier(device="cuda:0", stack_size=15)   # 6,100 distinct features
```

- Run time grows roughly with *K* (one TabPFN fit per view).
- Too few features → *K* is reduced automatically (`mode="auto"`), with a warning.
- Used value: `clf.execution_plan_["effective_n_views"]`.
- In the paper, *K* = 1–30 changed mean macro-F1 only slightly.

</details>

<details>
<summary><b>5. Execution mode — auto, always anchor, or native TabPFN</b></summary>

```python
TabPFNAnchorClassifier(device="cuda:0", mode="auto")     # default: depends on feature count
TabPFNAnchorClassifier(device="cuda:0", mode="anchor")   # always the anchor ensemble
TabPFNAnchorClassifier(device="cuda:0", mode="tabpfn")   # always native TabPFN-3
```

Check what ran:

```python
clf.fit(X_train, y_train)
clf.execution_plan_["route"]   # "direct_tabpfn" · "rf_shap_anchor" · "screen_rf_shap_anchor"
```

- `"anchor"` with 9 views needs ≥ 3,700 features (else lower `stack_size`).
- `"direct_tabpfn"` results are native TabPFN-3, not TabPFN-Anchor.

</details>

<details>
<summary><b>6. Outputs — anchor signature, ranking, views</b></summary>

```python
clf.fit(X_train, y_train)

clf.anchor_features_      # anchor_rank, feature, importance, source_stage
clf.feature_ranking_      # rank, feature, source_stage, score, is_anchor
clf.blocks_               # feature indices per view

clf.anchor_features_.to_csv("anchor_signature.csv", index=False)

# Evaluate the anchors alone with another classifier
from sklearn.linear_model import LogisticRegression
anchors = clf.anchor_features_["feature"].tolist()
LogisticRegression(max_iter=5000).fit(X_train[anchors], y_train)
```

- Feature names appear when `X` is a DataFrame; otherwise column indices.

</details>

<details>
<summary><b>7. Benchmark with cross-validation</b></summary>

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
models = {
    "TabPFN-Anchor": TabPFNAnchorClassifier(device="cuda:0"),
    "Random forest": RandomForestClassifier(n_estimators=500, random_state=0),
}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
```

- Feature selection runs inside `fit`, on each training fold only.
- Time per fold: ~18 min for a TCGA-sized cohort (paper settings, one GPU).

</details>

<details>
<summary><b>8. Shorter run time</b></summary>

```python
TabPFNAnchorClassifier(
    device="cuda:0",
    shap_max_samples=16,   # default 32
    shap_budget=256,       # default 1024
    rf_n_estimators=500,   # default 2000
)
```

- Most of the time is OddSHAP: cost ∝ samples × classes × budget.
- Fastest: `anchor_source="rf"` (example 2).
- Not the published settings — do not report as a paper reproduction.

</details>
