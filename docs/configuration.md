# TabPFN-Anchor — method, options and outputs

Reference for the estimator `TabPFNAnchorClassifier`. For worked examples see
[examples.md](examples.md); for installation see the [README](../README.md).

## Method

| Step | What it does | Keeps (default) | Parameter |
|---|---|---|---|
| 1.1 | one-way ANOVA *F* screen of all features | 2,000 | `n_stage1` |
| 1.2 | Random Forest feature importance (mean decrease in impurity) | 500 | `n_stage2` |
| 1.3 | OddSHAP attribution with the frozen TabPFN-3 backbone | 100 anchors | `n_anchor` |
| 2.1 | *K* views: the anchors plus a disjoint block of next-ranked features | 3,700 distinct features | `stack_size`, `n_residual` |
| 2.2 | one TabPFN view model per view; out-of-fold probabilities stacked by ℓ2 logistic regression | | `oof_folds`, `meta_C` |

Constraints: `n_anchor ≤ n_stage2 ≤ n_stage1`, and `n_anchor + n_residual =
n_stage2` (the width of every view). The number of distinct features used is
`n_stage2 + n_residual × (K − 1)`. Step 1.3 accounts for most of the run time;
on a TCGA-sized cohort the published configuration takes about eighteen minutes
per outer split on one GPU.

## Execution modes

| `mode` | Behaviour |
|---|---|
| `"auto"` (default) | follows the feature-count table in the README: TabPFN-3 directly up to 2,000 features, the anchor ensemble above (the ANOVA-*F* screen is skipped up to 5,000 features), *K* reduced if the input is too narrow |
| `"anchor"` | always the full anchor ensemble; raises an error instead of adapting (nine views need at least 3,700 features — lower `stack_size` for narrower inputs) |
| `"tabpfn"` | always TabPFN-3 directly, with no ranking and no anchors |

After fitting, `clf.execution_plan_` records the route that ran
(`"direct_tabpfn"`, `"rf_shap_anchor"` or `"screen_rf_shap_anchor"`), the
ranking stages used and the number of views and folds; `clf.resolved_config_`
records every resolved setting.

## Preprocessing

`preprocessing` is applied inside `fit` and `predict`, so training and test data
always receive the same transformation.

| `preprocessing` | Transformation | Use for |
|---|---|---|
| `"none"` (default) | none | data you have already preprocessed |
| `"rnaseq"` | `log2(x + 1)` | non-negative RNA-seq expression values that are not yet log-transformed |
| `"microarray"` | none | microarray data, usually already normalised and on a log scale |

- `"rnaseq"` applies only the log transformation. It does not perform
  library-size normalisation or any other RNA-seq preprocessing.
- Negative inputs are not accepted by `"rnaseq"` and raise an error.
- Do not apply `log2(x + 1)` yourself and also pass `"rnaseq"`; that transforms
  the data twice.

## Input requirements

| Input | Requirement |
|---|---|
| `X` | numeric, no missing values; with a DataFrame, unique column names |
| `y` | one label per row, at least two classes |
| class sizes | at least two samples in the smallest class |
| features | more than 2,000 for the anchor ensemble under `mode="auto"` |

A pandas DataFrame is preferable: the classifier remembers the column names and
refuses a differently ordered frame at prediction time.

## Outputs after `fit`

| Attribute | Contents |
|---|---|
| `anchor_features_` | DataFrame: `anchor_rank`, `feature`, `importance`, `source_stage` |
| `feature_ranking_` | DataFrame: `rank`, `feature`, `source_stage`, `score`, `is_anchor` |
| `blocks_` | list of feature-index arrays, one per view |
| `execution_plan_`, `resolved_config_` | the route and settings actually used |
| `classes_`, `n_features_in_`, `feature_names_in_` | as in scikit-learn |

The ranking attributes exist only when the anchor ensemble ran.

## Main parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `"auto"` | pass a single device such as `"cuda:0"`; `"auto"` with several GPUs is much slower |
| `preprocessing` | `"none"` | see above |
| `mode` | `"auto"` | see above |
| `n_stage1`, `n_stage2`, `n_anchor`, `n_residual` | 2000, 500, 100, 400 | cascade and view sizes |
| `stack_size` | 9 | number of views *K* |
| `anchor_source` | `"shap"` | `"rf"` uses the top Random Forest features as anchors and skips OddSHAP |
| `rf_n_estimators` | 2000 | trees in the Random Forest step |
| `shap_max_samples`, `shap_budget` | 32, 1024 | explained samples and coalition budget of OddSHAP |
| `tabpfn_n_estimators` | 8 | internal estimators per TabPFN view model |
| `oof_folds` | 5 | stacking folds (reduced automatically if a class is smaller) |
| `random_state` | 42 | seed |

## Paper configuration and results

- [`paper_results/configs/tabpfn_anchor.yaml`](../paper_results/configs/tabpfn_anchor.yaml)
  is the configuration that produced the reported results. Its values equal the
  estimator defaults, and a test checks that they stay equal. The CLI option
  `--paper-settings` reads this file from a source checkout.
- [`paper_results/`](../paper_results/) also holds the macro-F1 of every method
  in every predictive comparison of the paper and `aggregate_stats.py`, which
  reproduces the reported statistics. These files contain results only: no
  expression values, sample identifiers or split assignments.
- [DATA.md](../DATA.md) identifies the source data by GDC UUID and MD5.

The repository provides the method and its configuration, not the harness that
ran the full twelve-cohort benchmark.

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

The estimator follows the scikit-learn contract (`fit`, `predict`,
`predict_proba`, `get_params`, `set_params`, `clone`). Tests that need the
backbone checkpoint skip themselves when it is absent.
