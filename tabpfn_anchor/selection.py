# -*- coding: utf-8 -*-
"""STEP 1 -- anchor signature construction.

Three steps run in order, each using a model that can see a higher order of
feature--label interaction than the one before it.

    1.1  one-way ANOVA F statistic                      P     -> 2000   (marginal)
    1.2  Random Forest feature importance (MDI)         2000  ->  500   (low-order N:1)
    1.3  TabPFN + OddSHAP attribution                    500  ->  100   (high-order N:1)

Step 1.1 was logistic regression with permutation importance in an earlier
version; see stage1_anova_f for why it was replaced.

Every step returns global indices (0..P-1), so the caller carries indices only
and never has to track a column-name mapping.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path as _Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

__all__ = [
    "stage1_anova_f",
    "STAGE1_SOURCES",
    "SHAP_BACKENDS",
    "run_stage1",
    "stage2_random_forest",
    "stage3_tabpfn_shap",
    "build_rank_pool",
]


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
# The imputation scheme for Step 1.3. It is a fixed constant because it goes
# into the cache key.
# "baseline": a masked feature is replaced by that feature's mean over the
# background. This is a baseline game, not a conditional expectation. The
# Methods section states that definition verbatim, and switching to a joint
# marginal would require redoing the OddSHAP stability check -- the confirmed
# IoU of 0.905 was measured under this setting.
SHAP_IMPUTER = "baseline"


class NullRecorder:
    """Default recorder that measures nothing.

    It exists so that the package does not depend on the experiment harness.
    When the runner passes a real recorder (StageRecorder in
    runner/instrument.py), per-stage time and memory accumulate in events.tsv;
    when nothing is passed this one is used and costs nothing. The interface is
    the single method stage().
    """

    @contextmanager
    def stage(self, name, **kw):
        yield


NULL_RECORDER = NullRecorder()


# ---------------------------------------------------------------------------
# Step 1.1
# ---------------------------------------------------------------------------
def stage1_anova_f(X, y, n_keep=2000, logger=None, recorder=None, **_):
    """Step 1.1 -- the plain marginal F test (ANOVA-F).

    Why this one
    ------------
    This is Step 1.1 of the published method. The earlier version
    (`legacy_lr_pi.stage1_logistic_pi`) used ridge logistic regression with
    permutation importance, and was replaced for the following reason. With
    P (20,531) >> N, ridge **spreads weight evenly across a correlated block of
    genes.** In a block of size k each gene gets a coefficient of roughly 1/k,
    so permuting one gene moves the linear predictor by roughly 1/k as well and
    the permutation importance is diluted in inverse proportion to block size.
    The consequence is that **the strongest marginal signals get pushed out of
    the pool.** Measured on eight cohorts (how many of the limma-F top 100 fall
    inside the top 2,000 of Step 1.1; the value expected by chance is 9.7):

        LR+PI     0.2 ~ 24.7      (LUAD 1.3, KIRCKICH 0.2)
        RF-FI    92.0 ~ 100.0     <- multivariate, yet free of the effect
        ANOVA-F 100.0             <- all of them, by definition

    So this is not a consequence of being multivariate; it is specific to the
    ridge-plus-permutation-importance combination. ANOVA-F looks at each gene
    on its own, so nothing is diluted, and it is far more reproducible across
    splits (IoU at depth 100: 0.49-0.88 for ANOVA-F against 0.15-0.27 for
    LR+PI).

    Why ANOVA-F rather than limma-F
    -------------------------------
    The only difference is empirical Bayes variance shrinkage, and at
    n = 444-995 the two rankings were effectively the same (top-100 IoU
    0.978-0.997 over eight cohorts). ANOVA-F is not domain-specific, so it does
    not invite the objection that a transcriptomics-specific tool gave us an
    advantage, and it carries no dependency on R. Do not generalise this
    equivalence: in cohorts of a few dozen samples the shrinkage does real work.

    The contract is the one shared by every Step 1.1 implementation: return
    (rank[P], importance[P]) where rank holds global indices in decreasing
    order of importance. n_keep is used only for logging -- truncation is the
    caller's job. recorder and the other arguments are accepted and ignored so
    that call sites stay uniform.
    """
    from sklearn.feature_selection import f_classif

    rec = recorder or NULL_RECORDER
    with rec.stage("ANOVA_F"):
        F, _ = f_classif(X, np.asarray(y))
    # nan and +inf must be handled differently.
    #   nan  : a gene with zero variance. It carries no information, so push it
    #          to the back by flooring it at 0. Left alone, the argsort result
    #          would be implementation-defined.
    #   +inf : zero within-class variance but differing class means. That is
    #          **perfect separation**, the strongest signal there is. Flooring
    #          it at 0 would send it to the back, the exact opposite treatment.
    #          Map it to the largest finite value so it sorts to the front
    #          without breaking the ordering.
    imp = np.nan_to_num(np.asarray(F, dtype=np.float64),
                        nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    rank = np.argsort(-imp, kind="stable")
    if logger:
        logger.info(f"[1-1] ANOVA-F  P={X.shape[1]} -> {min(n_keep, X.shape[1])}")
    return rank, imp


# The swap point for Step 1.1. A value is a callable returning
# (rank[P], importance[P]) and must honour the same contract -- everything
# downstream (Steps 1.2 and 1.3, build_rank_pool) assumes only that rank holds
# global indices.
def _legacy_lr_pi(*args, **kwargs):
    """[ablation only] the former Step 1.1. Not used on the main path.

    legacy_lr_pi uses this module's NULL_RECORDER, so importing it at module
    level would be circular. It is imported only when actually called.
    """
    from .legacy_lr_pi import stage1_logistic_pi
    return stage1_logistic_pi(*args, **kwargs)


# Step 1.1 of the published method is anova_f. lr_pi is kept for ablations only,
# and it is registered only where its implementation is actually present. The
# public release does not ship legacy_lr_pi.py, so offering "lr_pi" there would
# advertise an option that raises ModuleNotFoundError when selected.
STAGE1_SOURCES = {
    "anova_f": stage1_anova_f,
}
if (_Path(__file__).with_name("legacy_lr_pi.py")).exists():
    STAGE1_SOURCES["lr_pi"] = _legacy_lr_pi


# Step 1.3 backends, gated the same way. "kernelshap"/"batched" reach an
# implementation the public release does not ship, so they are offered only
# where that file is present rather than failing with ModuleNotFoundError at
# the point of use.
SHAP_BACKENDS = {"auto", "shapiq"}
if (_Path(__file__).with_name("kernelshap_legacy.py")).exists():
    SHAP_BACKENDS |= {"kernelshap", "batched"}


def run_stage1(source, X, y, **kw):
    """Select Step 1.1 by name, stopping on an unknown name.

    There is deliberately no silent fallback. Step 1.1 is part of the
    definition of the method, and a single typo silently running a different
    method is hard to notice from the logs alone.
    """
    fn = STAGE1_SOURCES.get(source)
    if fn is None:
        raise ValueError(
            f"unknown stage1_source: {source!r}. "
            f"available: {sorted(STAGE1_SOURCES)}")
    return fn(X, y, **kw)


# ---------------------------------------------------------------------------
# Step 1.2
# ---------------------------------------------------------------------------
def stage2_random_forest(X, y, n_keep=500, n_estimators=2000,
                         random_state=42, n_jobs=-1, logger=None,
                         recorder=None):
    """Reduce the survivors of Step 1.1 again, by Gini importance.

    Returns (rank, importance), both indexed into the columns of the X passed
    in, that is, local indices.
    """
    rec = recorder or NULL_RECORDER
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                random_state=random_state, n_jobs=n_jobs)
    with rec.stage("RF_fit"):
        rf.fit(X, y)
    with rec.stage("RF_importance"):
        imp = rf.feature_importances_
    rank = np.argsort(imp)[::-1]
    if logger:
        logger.info(f"[1-2] RF(n={n_estimators})  {X.shape[1]} -> "
                    f"{min(n_keep, X.shape[1])}")
    return rank, imp


# ---------------------------------------------------------------------------
# Step 1.3
# ---------------------------------------------------------------------------
def stage3_tabpfn_shap(model, X, y, n_keep=100, max_samples=200, budget=2048,
                       backend="auto", class_mode="all", random_state=42,
                       max_batch_rows=2048, approximator=None, logger=None):
    """Compute Shapley values against a fitted TabPFN and take the top n_keep.

    This is the bottleneck of the pipeline. At P=500, KernelSHAP's default
    settings need 2*500 + 2048 = 3048 model calls per sample, which is 600,000
    calls for 200 samples. That is not practical.

    backend
        "shapiq" : shapiq's imputation explainer, used against TabPFN. The
                   training set stays fixed and each coalition costs a single
                   forward pass, so TabPFN's KV cache still applies. The cost
                   is controlled directly by budget.
        "kernelshap" : shap.KernelExplainer. A fallback path; nsamples is
                   bounded by budget.
        "auto"   : the shapiq path. **Stops** on failure -- no silent
                   substitution.

    Only the importance ordering is needed, so a stratified subsample of
    max_samples points is explained.

    Returns (rank, importance), indexed into the columns of X.
    """
    idx = _balanced_subsample(y, max_samples, random_state)
    Xe = np.ascontiguousarray(X[idx])
    n_features = X.shape[1]

    n_classes = len(np.unique(y))
    used = None
    if backend not in SHAP_BACKENDS:
        raise ValueError(
            f"unknown shap_backend: {backend!r}. available: "
            f"{sorted(SHAP_BACKENDS)}")
    if backend in ("kernelshap", "batched"):
        # The estimator the manuscript describes. It is a rewrite that fixes
        # the validity problems of the old implementation: explaining only one
        # class, no efficiency constraint, |mean phi|, resampling the data for
        # every coalition, and shuffling each column independently. See
        # kernelshap.py.
        from .kernelshap_legacy import batched_kernel_shap
        return batched_kernel_shap(
            model, X, y, n_keep=n_keep, n_coalitions=budget,
            n_eval=max_samples, max_batch_rows=max_batch_rows,
            random_state=random_state, logger=logger)
    if backend in ("auto", "shapiq"):
        # No silent fallback here. "Regression-based KernelSHAP was unstable,
        # so we moved to OddSHAP" is a central decision of this method; if a
        # dependency problem quietly ran KernelSHAP instead, that decision
        # would be void and the logs alone would not reveal it. Better to stop
        # when it is unavailable.
        imp = _shapiq_importance(model, X, Xe, budget, random_state,
                                 n_classes, class_mode, approximator,
                                 ye=np.asarray(y)[idx])
        used = "shapiq"
    if used is None:
        raise RuntimeError(f"unknown backend: {backend}")

    rank = np.argsort(imp)[::-1]
    if logger:
        logger.info(f"[1-3] TabPFN+SHAP({used})  {n_features} -> {min(n_keep, n_features)} "
                    f"(explained {len(idx)} samples, budget={budget}, "
                    f"class_mode={class_mode})")
    return rank, imp


def _make_approximator(name, n_features, random_state):
    """Fetch the approximator from the documented namespace.

    getattr(shapiq, name) reports a renamed or removed class only as an
    AttributeError, and a typo could quietly pick up something else. As of
    shapiq 1.6, OddSHAP is documented at shapiq.approximator.OddSHAP.
    """
    import shapiq.approximator as A
    cls = getattr(A, name, None)
    if cls is None:
        raise RuntimeError(
            f"shapiq.approximator has no {name}; shapiq>=1.6 is required. "
            "The published setting is OddSHAP, and a run without it is a "
            "different method from the one in the paper.")
    return cls(n=n_features, random_state=random_state)


def _shapiq_importance(model, X_bg, Xe, budget, random_state, n_classes=None,
                       class_mode="all", approximator=None, ye=None):
    """First-order Shapley values from shapiq's imputation-based explainer.

    The training set stays fixed and each coalition costs one baseline
    imputation and one forward pass, so there is none of KernelSHAP's
    thousands of refits per sample.

    class_mode
        "all"       explain once per class and average |value|. This is the
                    same definition as the manuscript's "mean |SHAP|", but its
                    cost is proportional to the number of classes (seven times
                    over for GEA and LGGGBM, which have seven).
        "predicted" explain only the class the model actually chose for each
                    sample. The reading is "what contributed to the model's
                    decision", and the cost is 1/n_classes of the above.
    """
    import shapiq

    n_features = Xe.shape[1]
    proba = np.asarray(model.predict_proba(Xe))
    n_cls = n_classes or proba.shape[1]

    if class_mode == "predicted":
        pairs = list(zip(range(len(Xe)), proba.argmax(axis=1)))
    elif class_mode == "all":
        pairs = [(i, c) for c in range(n_cls) for i in range(len(Xe))]
    else:
        raise ValueError(f"class_mode must be 'all' or 'predicted': {class_mode}")

    # Build one explainer per class_index only, to avoid the construction cost.
    explainers = {}
    per_sample = np.zeros((len(Xe), n_features), dtype=np.float64)
    per_sample_n = np.zeros(len(Xe), dtype=np.int64)
    for i, c in pairs:
        c = int(c)
        if c not in explainers:
            # Without an explicit approximator, shapiq falls back to its own
            # default, and at index="SV", max_order=1 that default is
            # **KernelSHAP**. (Verified numerically identical to calling it
            # explicitly: IoU 0.117 / Spearman -0.022 either way.)
            #
            # Regression-based estimators (the KernelSHAP family) would have
            # to solve p=500 with O(500) coalitions and cannot recover the
            # ordering. At equal budget and equal time:
            #   KernelSHAP  IoU 0.117  Spearman -0.022
            #   OddSHAP     IoU 0.550  Spearman  0.432
            # which is why the approximator is selectable here.
            # The approximator is part of the definition of the method.
            # shapiq's TabularExplainer defaults to KernelSHAP at index="SV",
            # so leaving it unspecified quietly runs a different method from
            # the paper.
            kw = {}
            if approximator:
                kw["approximator"] = _make_approximator(
                    approximator, X_bg.shape[1], random_state)
            explainers[c] = shapiq.TabularExplainer(
                model=model.predict_proba, data=X_bg, class_index=c,
                index="SV", max_order=1, imputer=SHAP_IMPUTER,
                random_state=random_state, **kw)
        iv = explainers[c].explain(Xe[i], budget=budget)
        per_sample[i] += np.abs(_first_order_vector(iv, n_features))
        per_sample_n[i] += 1

    # Average over classes within each sample first.
    per_sample /= np.maximum(per_sample_n, 1)[:, None]

    if ye is None:
        return per_sample.mean(axis=0)

    # Then average **within each true subtype**, and average the subtypes with
    # equal weight. Even with balanced sampling, a small subtype can run out
    # (BLCA's smallest has six members), which changes how many were drawn; a
    # plain mean would then give the large subtypes more weight again. The
    # manuscript's definition is subtype-macro, so it is matched here.
    ye = np.asarray(ye)
    return np.mean([per_sample[ye == t].mean(axis=0)
                    for t in np.unique(ye)], axis=0)


def _first_order_vector(interaction_values, n_features):
    """Extract just the first-order terms from shapiq InteractionValues."""
    out = np.zeros(n_features, dtype=np.float64)
    values = np.asarray(interaction_values.values)
    lookup = getattr(interaction_values, "interaction_lookup", None)
    if lookup:
        for coalition, pos in lookup.items():
            if len(coalition) == 1:
                out[coalition[0]] = values[pos]
        return out
    if values.shape[-1] == n_features:      # already a flat first-order vector
        return values.reshape(-1, n_features)[0]
    raise ValueError(f"cannot extract first-order Shapley values: shape={values.shape}")


def _kernel_importance(model, X_bg, Xe, budget, random_state):
    """shap.KernelExplainer fallback; nsamples is set to budget to bound cost."""
    import shap

    background = shap.kmeans(X_bg, min(10, len(X_bg)))
    explainer = shap.KernelExplainer(model.predict_proba, background)
    values = explainer.shap_values(Xe, nsamples=int(budget), silent=True,
                                   l1_reg="num_features(%d)" % Xe.shape[1])
    return _reduce_shap(values, Xe.shape)


def _reduce_shap(values, shape):
    """Absorb SHAP output shapes, which vary by version: (n,p), (n,p,c), list[(n,p)]."""
    n, p = shape
    if isinstance(values, list):
        arr = np.stack([np.asarray(v) for v in values], axis=-1)   # (n,p,c)
    else:
        arr = np.asarray(values)
    if arr.ndim == 3:
        return np.abs(arr).mean(axis=(0, 2))
    if arr.ndim == 2:
        return np.abs(arr).mean(axis=0)
    raise ValueError(f"unexpected SHAP shape: {arr.shape}")


def _balanced_subsample(y, max_n, random_state):
    """Draw the same number from each class, returning at most max_n indices.

    This is **balanced**, not proportional. In a cohort like BLCA, where one
    subtype has 148 members and another has 6, proportional sampling would
    concentrate the explained samples in the large subtype and the Shapley
    ranking would reflect only that subtype's signal.

    When a small class runs out first, its unused quota is redistributed among
    the others. The total never exceeds max_n -- rounding per class
    independently can overshoot.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    if max_n < len(classes):
        # Dropping classes silently would remove their signal from the
        # Shapley ranking entirely, and because the averaging afterwards runs
        # over `np.unique(ye)`, the omission would not even be visible.
        # Better to stop.
        raise ValueError(
            f"shap_max_samples={max_n} is smaller than the number of classes "
            f"({len(classes)}); at least one sample per class must be explained.")
    if len(y) <= max_n:
        return np.arange(len(y))
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    pool = {c: np.flatnonzero(y == c) for c in classes}
    quota = {c: 0 for c in classes}

    remaining = int(max_n)
    active = [c for c in classes if len(pool[c]) > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        for c in list(active):
            if remaining <= 0:
                break
            take = min(share, len(pool[c]) - quota[c], remaining)
            quota[c] += take
            remaining -= take
            if quota[c] >= len(pool[c]):
                active.remove(c)

    picked = [rng.choice(pool[c], size=quota[c], replace=False)
              for c in classes if quota[c] > 0]
    return np.sort(np.concatenate(picked))


# ---------------------------------------------------------------------------
# Unified ranking pool
# ---------------------------------------------------------------------------
def build_rank_pool(rank_shap_global, rank_rf_global, rank_lr_global, n_total):
    """Build one global ranking, SHAP before RF before LR (Algorithm 1).

        R_pool = R_SHAP(F2)  ||  R_RF(F1 \\ F2)  ||  R_LR(~F1)

    All three arguments must be ranking arrays expressed in global indices.
    An index that already appeared earlier is skipped later.
    """
    seen = np.zeros(n_total, dtype=bool)
    out = []
    for part in (rank_shap_global, rank_rf_global, rank_lr_global):
        for g in part:
            if not seen[g]:
                seen[g] = True
                out.append(g)
    # Indices that appeared nowhere are appended last, in their original order.
    out.extend(np.flatnonzero(~seen))
    return np.asarray(out, dtype=int)
