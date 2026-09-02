# -*- coding: utf-8 -*-
"""TabPFN-Anchor — anchor-conditioned TabPFN stacking for high-dimensional data.

    from tabpfn_anchor import TabPFNAnchorClassifier

    clf = TabPFNAnchorClassifier(device="cuda:0")   # defaults = published config
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    pred = clf.predict(X_test)

    clf.feature_ranking_     # every feature, in the pipeline's unified ranking
    clf.anchor_features_     # the anchor set alone

Exactly one name is public: the classifier. The individual stage functions
(stage1_logistic_pi and the like) were public once, and their defaults differed
from the classifier's -- budget 2048 against 1024, approximator None against
OddSHAP -- so **two public entry points in one package ran two different
methods**. They are internal again.
"""

from .estimator import TabPFNAnchorClassifier

__version__ = "0.1.0"
__all__ = ["TabPFNAnchorClassifier"]
