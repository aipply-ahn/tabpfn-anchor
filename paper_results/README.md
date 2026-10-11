# Numerical results of the paper

These files hold the macro-F1 of every predictive comparison reported in the
paper, at the level of one outer cross-validation split, together with the
script that turns them into the reported statistics. They are results, not
data: no expression value, sample identifier or cross-validation assignment is
included.

| File | Content |
|---|---|
| `split_level_macroF1.csv` | One row per (family, method, cohort, outer split): macro-F1 on that outer test fold. Columns: `family`, `role` (`reference` or `comparator`), `method`, `cohort`, `split` (`R<repeat>:F<fold>`), `macro_f1`. |
| `aggregate_stats.py` | Reads the file above, averages the splits of each cohort and, for every reference-comparator pair, computes the mean paired difference over the 12 cohorts, its 95% bootstrap interval (12 cohorts resampled, 10,000 resamples), the numbers of cohorts won and lost, the exact two-sided Wilcoxon signed-rank test and the Holm correction within the family. |
| `cohort_level_statistics.csv` | Output of `aggregate_stats.py`; the values reported in the paper. |

Families in `split_level_macroF1.csv`:

| Family | Reported in | Splits |
|---|---|---|
| `F1_highdim_tabpfn` | Table 1 | 12 cohorts x 50 |
| `F2_general` | Fig. 2 | 12 cohorts x 15 |
| `F4_same_backbone` | Table 2 | 12 cohorts x 15 |
| `F3_same_stack` | Table 3 | 12 cohorts x 15 |
| `REC_recurrence` | Table 4 (anchor-conditioned vs distributed recurrence) | 12 cohorts x 15 |
| `ABL_view_construction` | Table 4 (descriptive rows) | 12 cohorts x 15 |
| `SIG_selector_x_classifier[...]` | Fig. 3A and Supplementary Table S10 (descriptive) | 12 cohorts x 15 |

To reproduce the reported statistics (requires numpy, pandas and scipy):

```bash
cd paper_results
python aggregate_stats.py
```

The script rewrites `cohort_level_statistics.csv`; the result is identical to
the file committed here.
