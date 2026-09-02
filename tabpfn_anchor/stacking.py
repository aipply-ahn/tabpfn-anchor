# -*- coding: utf-8 -*-
"""STEP 2 -- stepwise stacking ensemble.

The 100 anchor genes go into every view model, and the remaining 400 slots are
filled by cutting successive blocks from further down the unified ranking
(R_pool), giving K TabPFN models. A logistic regression sits on top of the
meta-features formed by concatenating the view models' probabilities.

The meta-features must be built out of fold. Letting a view model predict the
same samples it was fitted on makes its probabilities far too confident; the
meta-learner then fits those values and collapses at test time.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

__all__ = ["build_blocks", "oof_and_full_probs", "stack_meta_features"]


def build_blocks(rank_pool, anchor_idx, n_stage2=500, n_residual=400, k=9):
    """Build the list of feature indices each of the k view models will use.

    Two invariants hold for every anchor_source:
      (1) every view contains the complete anchor set
          -> the intersection across views equals |anchor_idx|
      (2) every view has exactly n_stage2 features
          -> the feature budget is the same for every learner

    k=1  : anchors plus enough non-anchors from the top of the pool to reach
           n_stage2. With anchor_source="shap" (the published setting) the
           anchors already lie inside the top n_stage2, so the result equals
           R_pool[:n_stage2].
    k>=2 : anchors plus the next n_residual non-anchors; blocks do not overlap.

    An earlier version assumed the anchors were inside the top n_stage2 and
    merely subtracted them from the residual. Ablations that change
    anchor_source break that assumption: views came out smaller than 500 and
    the intersection fell short of 100.

    If the pool runs out before a block can be filled **exactly**, this stops
    and returns only the blocks it built. It never builds a partial view --
    better that the caller can see the shortfall than that it shrinks quietly.
    """
    rank_pool = np.asarray(rank_pool)
    anchor_idx = np.asarray(anchor_idx)
    n_anchor = len(anchor_idx)

    # Build the anchor-free ordering once, then cut blocks off the front.
    non_anchor = rank_pool[~np.isin(rank_pool, anchor_idx)]

    n_first = max(0, n_stage2 - n_anchor)
    blocks = [np.concatenate([anchor_idx, non_anchor[:n_first]])]

    cur = n_first
    for _ in range(2, k + 1):
        residual = non_anchor[cur:cur + n_residual]
        # Invariant (2). If fewer than n_residual genes remain, do not build a
        # view from them: that would create a learner of a different size and
        # break the method's definition of one feature budget per learner.
        if len(residual) < n_residual:
            break
        blocks.append(np.concatenate([anchor_idx, residual]))
        cur += n_residual
    return blocks


def oof_and_full_probs(make_model, X, y, n_classes, n_folds=5, random_state=42,
                       recorder=None, view=None):
    """For one block, produce (out-of-fold probabilities, full-data model).

    - out-of-fold probabilities: for training the meta-learner. Each sample is
      predicted by a model fitted without it.
    - full-data model: what predicts at test time, refitted on all of the
      training data.

    Some cohorts have very small classes, so n_folds is lowered to at most the
    smallest class size; otherwise a fold can lose a class entirely.

    Given a recorder, fitting and prediction are timed separately for each
    inner fold. TabPFN's fit is lazy and the real computation happens in
    predict (measured on BRCA: fit 42.7 s, predict 786.9 s), so timing them
    together would hide where the cost is.
    """
    from .selection import NULL_RECORDER
    rec = recorder or NULL_RECORDER
    y = np.asarray(y)
    counts = np.bincount(y, minlength=n_classes)
    nonzero = counts[counts > 0]
    # Raising this with max(2, ...) would carry on even when a class has a
    # single sample, where two-fold stratification does not hold. Better to
    # stop here than to produce a quietly wrong out-of-fold estimate.
    if len(nonzero) and nonzero.min() < 2:
        raise ValueError(
            "out-of-fold estimation requires at least 2 samples per class; "
            f"smallest class has {int(nonzero.min())}")
    n_splits = int(min(n_folds, nonzero.min())) if len(nonzero) else int(n_folds)
    n_splits = max(2, n_splits)

    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for j, (tr, va) in enumerate(skf.split(X, y), start=1):
        m = make_model()
        with rec.stage("view_oof_fit", view=view, inner_fold=j, gpu=True):
            m.fit(X[tr], y[tr])
        with rec.stage("view_oof_predict", view=view, inner_fold=j, gpu=True):
            oof[va] = _align(m.predict_proba(X[va]), m.classes_, n_classes)

    full = make_model()
    with rec.stage("view_refit", view=view, gpu=True):
        full.fit(X, y)
    return oof, full, n_splits


def _align(proba, model_classes, n_classes):
    """Pad columns for classes the fold never saw, keeping column order."""
    if proba.shape[1] == n_classes:
        return proba.astype(np.float32)
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(model_classes, dtype=int)] = proba
    return out


def stack_meta_features(prob_list):
    """Concatenate the view models' probabilities along the feature axis."""
    return np.concatenate([np.asarray(p, dtype=np.float32) for p in prob_list], axis=1)
