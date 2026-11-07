# Configuration

Options and outputs of `TabPFNAnchorClassifier`.
Examples: [examples.md](examples.md) · Setup: [installation.md](installation.md)

## Method

| Step | What it does | Keeps | Parameter |
|---|---|---|---|
| 1.1 | ANOVA-*F* screen | 2,000 | `n_stage1` |
| 1.2 | Random Forest importance | 500 | `n_stage2` |
| 1.3 | TabPFN-3 + OddSHAP attribution | 100 anchors | `n_anchor` |
| 2.1 | *K* views = anchors + next-ranked block | 3,700 distinct features | `stack_size`, `n_residual` |
| 2.2 | one TabPFN view model per view, stacked by logistic regression | — | `oof_folds`, `meta_C` |

**Rules**

- `n_anchor ≤ n_stage2 ≤ n_stage1`
- `n_anchor + n_residual = n_stage2` (= view width; keep 500 for TabPFN-3)
- distinct features = `n_stage2 + n_residual × (K − 1)`

## Execution by feature count (`mode="auto"`)

| Features | Route (`execution_plan_["route"]`) | Ranking | *K* |
|---|---|---|---|
| ≤ 500 | `direct_tabpfn` | — | — |
| 501 – 2,000 | `direct_tabpfn` (random feature sampling inside TabPFN) | — | — |
| 2,001 – 5,000 | `rf_shap_anchor` | RF → OddSHAP | 4 – 9 |
| > 5,000 | `screen_rf_shap_anchor` | ANOVA-*F* → RF → OddSHAP | 9 |

*K* = min(9, ⌊(features − 100) / 400⌋):

| Features | 2,001–2,099 | 2,100–2,499 | 2,500–2,899 | 2,900–3,299 | 3,300–3,699 | ≥ 3,700 |
|---|---|---|---|---|---|---|
| *K* | 4 | 5 | 6 | 7 | 8 | 9 |

## Execution modes

| `mode` | Behaviour |
|---|---|
| `"auto"` (default) | follows the table above |
| `"anchor"` | always the anchor ensemble; errors if the input is too narrow (9 views need ≥ 3,700 features) |
| `"tabpfn"` | always native TabPFN-3; no ranking, no anchors |

Check what ran: `clf.execution_plan_` · all settings: `clf.resolved_config_`

## Preprocessing

| `preprocessing` | Transformation | Use for |
|---|---|---|
| `"none"` (default) | none | data you preprocessed yourself |
| `"rnaseq"` | `log2(x + 1)` | non-negative RNA-seq values, not yet logged |
| `"microarray"` | none | normalised, log-scale arrays |

- Applied inside both `fit` and `predict`.
- `"rnaseq"` is a log transform only — no library-size normalisation.
- `"rnaseq"` rejects negative values.
- Do not log-transform yourself *and* pass `"rnaseq"`.

## Input requirements

| Input | Requirement |
|---|---|
| `X` | numeric, no missing values; DataFrame column names unique |
| `y` | ≥ 2 classes |
| class size | ≥ 2 samples in the smallest class |
| features | > 2,000 for the anchor ensemble under `"auto"` |

A DataFrame is recommended: column order is checked at prediction time.

## Outputs after `fit`

| Attribute | Contents |
|---|---|
| `anchor_features_` | DataFrame: `anchor_rank`, `feature`, `importance`, `source_stage` |
| `feature_ranking_` | DataFrame: `rank`, `feature`, `source_stage`, `score`, `is_anchor` |
| `blocks_` | feature indices per view |
| `execution_plan_` | route, ranking stages, number of views and folds |
| `resolved_config_` | every resolved setting |

Ranking outputs exist only when the anchor ensemble ran.

## Main parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `"auto"` | set one device, e.g. `"cuda:0"` |
| `preprocessing` | `"none"` | see above |
| `mode` | `"auto"` | see above |
| `n_stage1` · `n_stage2` · `n_anchor` · `n_residual` | 2000 · 500 · 100 · 400 | cascade and view sizes |
| `stack_size` | 9 | number of views *K* |
| `anchor_source` | `"shap"` | `"rf"` skips OddSHAP (faster) |
| `rf_n_estimators` | 2000 | Random Forest trees |
| `shap_max_samples` · `shap_budget` | 32 · 1024 | OddSHAP cost controls |
| `tabpfn_n_estimators` | 8 | estimators per TabPFN view model |
| `oof_folds` | 5 | stacking folds (auto-reduced for small classes) |
| `random_state` | 42 | seed |

## Paper configuration

- Settings: [`paper/configs/tabpfn_anchor.yaml`](../paper/configs/tabpfn_anchor.yaml) (= the defaults; checked by a test)
- CLI: `tabpfn-anchor tcga-example --paper-settings` reads it from a source checkout
- Results and statistics: [`paper/`](../paper/)
