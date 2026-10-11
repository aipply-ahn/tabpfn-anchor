#!/usr/bin/env python3
"""Reproduce the reported cohort-level tests from split_level_macroF1.csv.

Unit of inference: the cancer task (n = 12).  For each method, macro-F1 is
averaged over all outer splits of a cohort.  Within a family, the reference is
compared with each comparator through the 12 paired cohort means:

  * two-sided Wilcoxon signed-rank test, exact null distribution when there is
    no zero or tied difference;
  * Holm correction within the family (families are never merged);
  * 95% percentile-bootstrap interval of the mean paired difference, resampling
    the 12 cohorts (10,000 resamples, seed 20261007, cohorts in alphabetical order).

Confirmatory families: F1_highdim_tabpfn, F2_general, F3_same_stack,
F4_same_backbone (rows marked "descriptive" excluded) and REC_recurrence.
ABL_* and SIG_* are descriptive: effect sizes and intervals only.

    python aggregate_stats.py            # needs numpy, pandas, scipy
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
B, SEED = 10_000, 20261007


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty(len(p)); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k]); out[k] = min(run, 1.0)
    return out


def main():
    d = pd.read_csv(HERE / "split_level_macroF1.csv")
    cm = d.groupby(["family", "role", "method", "cohort"]).macro_f1.mean().reset_index()
    out = []
    for fam, g in cm.groupby("family", sort=False):
        ref = g[g.role == "reference"].set_index("cohort").macro_f1.sort_index()
        assert len(ref) == 12, fam
        rows = []
        for m, c in g[g.role == "comparator"].groupby("method", sort=False):
            x = (ref - c.set_index("cohort").macro_f1.sort_index()).values
            assert len(x) == 12 and not np.isnan(x).any(), (fam, m)
            exact = (x != 0).all() and len(np.unique(np.abs(x))) == len(x)
            rng = np.random.default_rng(SEED)
            bm = x[rng.integers(0, 12, size=(B, 12))].mean(axis=1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            rows.append(dict(family=fam, comparator=m, ref_mean=ref.mean(), comp_mean=ref.mean() - x.mean(),
                             mean_diff=x.mean(), median_diff=np.median(x), ci_lo=lo, ci_hi=hi,
                             wins=int((x > 0).sum()), losses=int((x < 0).sum()), ties=int((x == 0).sum()),
                             p=wilcoxon(x, method="exact" if exact else "approx").pvalue,
                             confirmatory=not (fam.startswith(("ABL", "SIG")) or "descriptive" in m)))
        r = pd.DataFrame(rows)
        r["p_holm"] = np.nan
        k = r.confirmatory
        if k.any():
            r.loc[k, "p_holm"] = holm(r.loc[k, "p"].values)
        r.loc[~k, "p"] = np.nan
        out.append(r)
    out = pd.concat(out, ignore_index=True)
    out.to_csv(HERE / "cohort_level_statistics.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(out.drop(columns="confirmatory").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
