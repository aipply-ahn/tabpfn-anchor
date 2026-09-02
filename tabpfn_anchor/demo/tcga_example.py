#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fit and evaluate on one official outer split of one real cohort.

    tabpfn-anchor download-tcga --cohort UCEC --data-dir ./data
    tabpfn-anchor tcga-example  --cohort UCEC --repeat 1 --fold 1 --data-dir ./data

This is the third and last demo stage. The first two run on synthetic data and
show that the code works; this one runs on the data the paper used, so that the
numbers you see come from the real distribution.

**It is one split, not the paper.** The reported results are means over twelve
cohorts and fifteen or fifty outer splits, with a paired test across cohorts
and a multiple-comparison correction. None of that is here. A single split has
a standard error large enough that comparing its macro-F1 against a number in
the paper says nothing.

What it does guarantee is that the split is not ours. The release ships a
100 x 5 cross-validation assignment, and this module reads one column of it
rather than calling ``train_test_split``. A comparison against another method
is meaningful only if both read the same column.

Leakage
-------
Selection is fitted on the outer-training partition alone. The cascaded
ranking, the anchor set and the view blocks are all produced inside
``fit(X_train, y_train)``; the test partition is touched only by ``predict``.
The one preprocessing step applied outside the estimator, ``log2(x + 1)``, is a
fixed monotone map with no fitted parameter, so applying it to the whole matrix
is not leakage. Microarray cohorts are already on a log scale and are left
alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .download_tcga import SOURCE

#: The expression block of the multi-omics table. The same file also carries
#: `B:MUTA:`, `N:MIRN:` and `N:METH:` columns, which this work does not use.
GEXP_PREFIX = "N:GEXP:"

#: Cross-validation coding used by the released assignment files.
TRAIN_VALUE, TEST_VALUE = 0, 1


def find_files(data_dir: Path, cohort: str) -> tuple[Path, Path]:
    """Locate the cohort matrix and its cross-validation assignment."""
    rel = SOURCE["release"]
    matrix = data_dir / "source" / f"{cohort}_{rel}.tsv"
    cv = data_dir / "source" / f"{cohort}_CVfolds_5FOLD_{rel}.tsv"
    missing = [p for p in (matrix, cv) if not p.is_file()]
    if missing:
        raise SystemExit(
            "not found:\n  " + "\n  ".join(str(p) for p in missing) + "\n\n"
            f"Get them with:  tabpfn-anchor download-tcga --cohort {cohort} "
            f"--data-dir {data_dir}\nSee DATA.md.")
    return matrix, cv


def load(matrix: Path, cv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    """Read the expression block, the labels and the fold assignment.

    Returns the platform as well, because it decides whether log2(x + 1) is
    applied: a microarray matrix is already on a log scale and carries negative
    values, and taking the log again turns every entry at or below -1 into NaN.
    """
    df = pd.read_csv(matrix, sep="\t", index_col=0)
    y = df["Labels"].astype(str)
    gexp = [c for c in df.columns if c.startswith(GEXP_PREFIX)]
    if not gexp:
        raise SystemExit(f"no {GEXP_PREFIX} columns in {matrix}")
    X = df[gexp].astype(np.float32)

    finite = X.to_numpy()
    finite = finite[np.isfinite(finite)]
    platform = ("microarray"
                if (finite.min() < 0.0 or finite.max() < 50.0) else "rnaseq")

    cv = pd.read_csv(cv_path, sep="\t", index_col=0).drop(
        columns=["Labels"], errors="ignore")
    cv = cv.reindex(X.index)
    if cv.isna().all(axis=1).any():
        raise SystemExit(f"{cv_path} does not cover every sample of {matrix}")
    return X, y, cv, platform


def split(X: pd.DataFrame, y: pd.Series, cv: pd.DataFrame, column: str):
    if column not in cv.columns:
        raise SystemExit(
            f"{column} is not a column of the assignment file. "
            f"It offers {cv.columns[0]} .. {cv.columns[-1]}.")
    v = pd.to_numeric(cv[column], errors="coerce").to_numpy()
    tr, te = v == TRAIN_VALUE, v == TEST_VALUE
    if not tr.any() or not te.any():
        raise SystemExit(f"{column} yields an empty partition")
    return X[tr], X[te], y[tr], y[te]


def run(cohort: str, data_dir: Path, repeat: int, fold: int, params: dict,
        out_dir: Path | None, seed: int = 42, verbose: bool = False) -> dict:
    from sklearn.metrics import f1_score, balanced_accuracy_score

    from ..estimator import TabPFNAnchorClassifier

    matrix, cv_path = find_files(data_dir, cohort)
    X, y, cv, platform = load(matrix, cv_path)
    column = f"R{repeat}:F{fold}"

    if platform == "rnaseq":
        X = np.log2(X + 1.0)
        scale = "log2(x + 1)"
    else:
        scale = "left as distributed (already log scale)"

    Xtr, Xte, ytr, yte = split(X, y, cv, column)
    print(f"cohort        {cohort}   release {SOURCE['release']}")
    print(f"platform      {platform}; {scale}")
    print(f"split         {column}   (official assignment, not resampled here)")
    print(f"train/test    {len(Xtr)} / {len(Xte)}")
    print(f"features      {X.shape[1]:,}")
    print(f"classes       {ytr.nunique()}   "
          + ", ".join(f"{k}={v}" for k, v in ytr.value_counts().sort_index().items()))

    clf = TabPFNAnchorClassifier(random_state=seed, verbose=verbose, **params)
    print("\nfitting on the training partition only ...")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    metrics = {
        "cohort": cohort,
        "release": SOURCE["release"],
        "split": column,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "n_features": int(X.shape[1]),
        "platform": platform,
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "n_views": int(clf.effective_n_views_),
        "n_genes_used": int(clf.n_genes_used_),
    }
    print(f"\nmacro-F1      {metrics['macro_f1']:.3f}   "
          f"balanced accuracy {metrics['balanced_accuracy']:.3f}")
    print(f"views         {metrics['n_views']}   "
          f"distinct genes used {metrics['n_genes_used']:,}")
    print("\nOne split of one cohort. This is not the reported result, which is a "
          "mean\nover twelve cohorts and many splits; see README.")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        clf.anchor_features_.to_csv(out_dir / "anchors.tsv", sep="\t", index=False)
        clf.feature_ranking_.to_csv(out_dir / "ranking.tsv", sep="\t", index=False)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        (out_dir / "run_config.json").write_text(
            json.dumps(clf.get_params(), indent=2, default=str) + "\n",
            encoding="utf-8")
        print(f"\nwrote         {out_dir}/"
              "{anchors.tsv,ranking.tsv,metrics.json,run_config.json}")
    return metrics
