#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal example: TabPFN-Anchor as a scikit-learn classifier.

    python example/run.py

The data come straight from ``sklearn.datasets.make_classification``. They are
synthetic and contain nothing from any cohort; they reproduce the shape of the
task -- far more features than samples, a few informative features, many
correlated with those, imbalanced classes -- and nothing else. The values are
not on an expression scale and the columns are not genes.

The backbone is separate: on first use the ``tabpfn`` package needs network
access to obtain the checkpoint, and that checkpoint is under its own
non-commercial licence -- see THIRD_PARTY_NOTICES.md.

What the example shows
    1. the estimator is an ordinary scikit-learn classifier
    2. fitting produces a ranking of every feature and an anchor set
    3. the anchor set is a plain list of feature names you can report

The settings below are reduced so the example finishes quickly.  They are not
the configuration used in the paper; see configs/paper/tabpfn_anchor.yaml.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from tabpfn_anchor import TabPFNAnchorClassifier
from tabpfn_anchor.cli import _pick_device

N_INFORMATIVE = 20


def main() -> None:
    # 3,000 features keeps this above the 2,000-feature threshold at which
    # mode="auto" would hand the input straight to TabPFN and build no ranking.
    X, y = make_classification(
        n_samples=300, n_features=3_000,
        n_informative=N_INFORMATIVE, n_redundant=80, n_repeated=0,
        n_classes=4, n_clusters_per_class=2, weights=[.40, .28, .20, .12],
        class_sep=2.5, flip_y=0.01, shuffle=False, random_state=20260902)
    # shuffle=False leaves the informative block first, so the example can say
    # afterwards how many of the planted features the anchors recovered. Real
    # data comes with no such key.
    informative = {f"feature_{j + 1:05d}" for j in range(N_INFORMATIVE)}
    X = pd.DataFrame(X, columns=[f"feature_{j + 1:05d}" for j in range(X.shape[1])])
    y = pd.Series([f"Class_{chr(65 + c)}" for c in y], name="subtype")
    print(f"synthetic data: {X.shape[0]} samples x {X.shape[1]:,} features, "
          f"{y.nunique()} classes, {len(informative)} of them informative")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    clf = TabPFNAnchorClassifier(
        # reduced for the example; the published configuration is much larger
        n_stage1=1000, n_stage2=150, n_anchor=50, n_residual=100, stack_size=3,
        rf_n_estimators=300, shap_budget=192, shap_max_samples=16,
        tabpfn_n_estimators=2, shap_tabpfn_n_estimators=2,
        random_state=42,
        # Name one device. With several GPUs visible, the backbone's "auto"
        # setting was measured about fifty times slower.
        device=_pick_device(),
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    pred = clf.predict(X_test)
    print(f"\nmacro-F1: {f1_score(y_test, pred, average='macro'):.3f}   "
          f"predict_proba shape: {proba.shape}   classes: {list(clf.classes_)}")

    print(f"views fitted: {clf.effective_n_views_}   "
          f"distinct features used: {clf.n_genes_used_}")

    anchors = clf.anchor_features_
    print(f"\nanchor set ({len(anchors)} features), first 10 by attribution rank:")
    print(anchors.head(10).to_string(index=False))

    hit = informative & set(anchors["feature"])
    print(f"\n{len(hit)} of the {len(informative)} informative features are anchors "
          f"(a random draw of {len(anchors)} would contain about "
          f"{len(anchors) * len(informative) / X.shape[1]:.1f})")

    print("\nfull ranking of every feature is in clf.feature_ranking_")


if __name__ == "__main__":
    main()
