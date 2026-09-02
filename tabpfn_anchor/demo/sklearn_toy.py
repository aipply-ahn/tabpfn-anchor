#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check an installation by running the example from README.md.

    python -m tabpfn_anchor.demo.sklearn_toy

The code between the markers below is the README's "A complete example you can
paste and run", character for character. `tests/test_api_contract.py` extracts
both and compares them, so the page and this file cannot drift apart: if you
edit one, edit the other.

That is the only reason this file exists. It is not a data generator with
options -- there is nothing here you could not type yourself, and no preset
name standing in for arguments you should be reading. The data come from one
`sklearn.datasets.make_classification` call whose arguments are in plain sight.

They are synthetic and contain nothing from any cohort. The values are not on
an expression scale and the columns are not genes; what they reproduce is the
shape of the task -- far more features than samples, a few informative
features, many correlated with those, imbalanced classes -- and nothing else.
Do not read anything into the score: the difficulty of make_classification
output is whatever its arguments make it.

This runs the published configuration, so it takes minutes rather than seconds,
and it needs the backbone checkpoint. On first use the tabpfn package fetches
that over the network under its own non-commercial licence; see
THIRD_PARTY_NOTICES.md. Pass a different device if you have no GPU at cuda:0.
"""

from __future__ import annotations


def main() -> None:
    # --- README EXAMPLE BEGIN (keep identical to README.md) ---
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

    clf.anchor_features_    # the 100 anchor genes, with attribution rank
    clf.feature_ranking_    # every ranked gene, with the stage it was ranked in
    clf.blocks_             # the gene indices given to each view
    # --- README EXAMPLE END ---

    # Not part of the README block: say what was produced, so that running this
    # as an installation check reports something rather than one bare number.
    print(f"predict_proba {proba.shape}   classes {list(clf.classes_)}")
    print(f"views {clf.effective_n_views_}   "
          f"distinct features used {clf.n_genes_used_:,}")
    print(f"anchors {len(clf.anchor_features_)}   "
          f"ranking rows {len(clf.feature_ranking_):,}")


if __name__ == "__main__":
    main()
