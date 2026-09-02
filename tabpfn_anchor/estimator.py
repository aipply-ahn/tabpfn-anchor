# -*- coding: utf-8 -*-
"""The TabPFN-Anchor classifier.

It follows the scikit-learn estimator contract, so the experiment harness
evaluates it by exactly the same route as every other baseline: the same
cross-validation splits, the same preprocessing, the same metric computation.

    clf = TabPFNAnchorClassifier(stack_size=9, device="cuda:0")
    clf.fit(X_train, y_train)
    clf.predict_proba(X_test)

Wherever an early manuscript draft disagreed with itself, the choice is exposed
as a parameter and the default is stated here. For the three below the draft
contained contradictory statements, and the defaults follow what was actually
run.

    lr_penalty="l2"        Algorithm 1 said L2; the Step 1.1 prose said L1
    residual_source="pool" Algorithm 1 said R_pool; the Step 2.1 equation said R_1
    oof_meta=True          Algorithm 1 used in-sample predictions, which leak
"""

from __future__ import annotations

import logging
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from .cache import StageCache
from .selection import (
    NULL_RECORDER,
    SHAP_IMPUTER,
    build_rank_pool,
    STAGE1_SOURCES,
    SHAP_BACKENDS,
    run_stage1,
    stage2_random_forest,
    stage3_tabpfn_shap,
)
from .stacking import (build_blocks, oof_and_full_probs,
                       stack_meta_features, _align)

__all__ = ["TabPFNAnchorClassifier"]

# Part of the cache key. Bump it whenever code that affects the selection
# result changes; without a bump, an old computation is read back as though the
# new code had produced it.
__cache_version__ = "2026.08.20.anova1"


class TabPFNAnchorClassifier(ClassifierMixin, BaseEstimator):
    """Staged feature selection plus an anchor-based stacking ensemble.

    Parameters
    ----------
    n_stage1, n_stage2 : int
        How many features survive Steps 1.1 and 1.2. Defaults 2000 and 500.
    n_anchor : int
        Number of anchor genes. They enter every view model. Default 100.
    n_residual : int
        Residual features added afresh to each view model. Default 400.
    stack_size : int
        The number of view models, K. Total genes used is
        n_stage2 + n_residual*(K-1).
    pi_n_repeats, rf_n_estimators, tabpfn_n_estimators : int
        Repetition and ensemble sizes for each step. Defaults 50 / 2000 / 8.
        (tabpfn_n_estimators is used with the same value in the attribution
        step and the prediction step.)
    stage1_source : str
        The Step 1.1 selector. The published method uses **"anova_f"**, which
        is the only value the released package offers. An earlier version
        ("lr_pi": ridge logistic regression plus permutation importance) is
        registered only where its implementation is present, which it is not in
        the public release; selection.STAGE1_SOURCES lists what is available.
    lr_penalty, lr_C, pi_n_repeats, pi_oof_splits
        Settings of that earlier Step 1.1. They are accepted so that recorded
        configurations still load, and have no effect under "anova_f".
    residual_source : {"pool", "rf", "lr"}
        Whether residual features are drawn from the unified ranking (R_pool)
        or from the Step 1.1 ranking (R_1).
    oof_meta : bool
        Whether meta-features are built out of fold. False reproduces the early
        Algorithm 1 exactly, but then the meta-learner trains on leaked
        probabilities.
    nested_selection : bool
        If True, Step 1 (PI / RF / SHAP) is redone inside every inner fold.
        oof_meta=True alone withholds the inner validation split from base
        model fitting only; its labels have already entered feature selection.
        This option removes that residual dependence. The price is repeating
        the most expensive step oof_folds times (PI, a 2000-tree RF and
        KernelSHAP), which lengthens the run considerably. Either way, the
        model used at inference is refitted on the whole outer training split.
    view_strategy : {"anchor_ranked", "single", "random_disjoint", "ranked_disjoint"}
        How to build the feature view each view model sees. This is the axis of
        the A0-A4 ablation.

          single            one top-500 view from F2 (same as K=1)          -> A0
          random_disjoint   all genes shuffled under a fixed seed, then cut
                            into disjoint blocks of 500                     -> A1
          ranked_disjoint   the unified supervised ranking cut into disjoint
                            blocks of 500                                   -> A2
          anchor_ranked     100 shared anchors plus 400 ranked residuals
                            (the published method)                       -> A3/A4

        random_disjoint is not the comparator of Ye et al. That one *samples* a
        500-feature random subset m times and does not require disjointness.
        This is a controlled ablation, made disjoint on purpose so that its
        coverage structure matches A2 and the only thing that differs is random
        against supervised ordering.
    aggregation : {"stack", "vote", "none"}
        "none" exists for A0, which has a single view (one 500-feature
        learner). It uses that one prediction directly, with no meta-learner
        and no vote. "vote" would send it down the odd path of a one-voter
        majority, and "stack" would attach a pointless meta logistic regression
        that blurs what A0 is.
        "stack" is the logistic regression on out-of-fold meta-features (the
        published method); "vote" is a hard majority vote. A1-A3 must differ
        only in how views are built, so they all use "vote". Ties break
        deterministically towards the lower class index (the np.argmax rule).
    shap_max_samples : int
        Upper bound on how many samples SHAP explains, because of KernelSHAP's
        cost.
    """

    def __init__(self, n_stage1=2000, n_stage2=500, n_anchor=100, n_residual=400,
                 stack_size=9, pi_n_repeats=50, rf_n_estimators=2000,
                 tabpfn_n_estimators=8,
                 tabpfn_model_path="tabpfn-v3-classifier-v3_20260417_multiclass.ckpt",
                 lr_penalty="l2", lr_C=1.0,
                 pi_oof_splits=5,
                 stage1_source="anova_f",
                 residual_source="pool", anchor_source="shap", anchor_slot=1,
                 oof_meta=True, oof_folds=5, nested_selection=False,
                 view_strategy="anchor_ranked", aggregation="stack",
                 shap_max_samples=32, shap_budget=1024, shap_backend="shapiq", shap_approximator="OddSHAP",
                 shap_tabpfn_n_estimators=8, shap_class_mode="all",
                 meta_C=1.0, device="auto",
                 ignore_pretraining_limits=False, random_state=42,
                 n_jobs=-1, verbose=False, cache_dir=None, recorder=None,
                 # -- higher-level API, for humans -----------------------
                 mode="auto", n_views=None,
                 selection_kwargs=None, tabpfn_kwargs=None,
                 shap_kwargs=None, meta_kwargs=None):
        self.n_stage1 = n_stage1
        self.n_stage2 = n_stage2
        self.n_anchor = n_anchor
        self.n_residual = n_residual
        self.stack_size = stack_size
        self.pi_n_repeats = pi_n_repeats
        self.rf_n_estimators = rf_n_estimators
        self.tabpfn_n_estimators = tabpfn_n_estimators
        self.tabpfn_model_path = tabpfn_model_path
        self.lr_penalty = lr_penalty
        self.lr_C = lr_C
        self.pi_oof_splits = pi_oof_splits
        self.stage1_source = stage1_source
        self.residual_source = residual_source
        self.anchor_source = anchor_source
        # With anchor_source="slot", which 100-gene block of R_pool to repeat.
        # q=1 gives R[1:100], which equals the OddSHAP anchors (confirmed on
        # 180 of 180 splits).
        self.anchor_slot = anchor_slot
        self.oof_meta = oof_meta
        self.oof_folds = oof_folds
        self.nested_selection = nested_selection
        self.view_strategy = view_strategy
        self.aggregation = aggregation
        self.shap_max_samples = shap_max_samples
        self.shap_budget = shap_budget
        self.shap_backend = shap_backend
        self.shap_approximator = shap_approximator
        self.shap_tabpfn_n_estimators = shap_tabpfn_n_estimators
        self.shap_class_mode = shap_class_mode
        self.meta_C = meta_C
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cache_dir = cache_dir
        # Where the experiment harness plugs in a per-stage instrument. None
        # measures nothing (NULL_RECORDER). This keeps the package independent
        # of the runner.
        self.recorder = recorder
        # The higher-level API. The four below override the flat parameters
        # above. The flat ones exist for reproducing the paper (the YAML files
        # use them verbatim); these are the ones meant for public use.
        self.mode = mode
        self.n_views = n_views
        self.selection_kwargs = selection_kwargs
        self.tabpfn_kwargs = tabpfn_kwargs
        self.shap_kwargs = shap_kwargs
        self.meta_kwargs = meta_kwargs

    # -- constructing TabPFN ------------------------------------------------
    def _make_tabpfn(self, n_estimators=None):
        """Construct a view model.

        The default backbone is the **TabPFN-3 multiclass checkpoint**, which
        is the backbone the reported results were produced with.

        Each view holds 100 anchors plus 400 residuals, so a view model always
        receives its whole view. The external view width is fixed by this
        procedure rather than inferred from any backbone-internal limit;
        successive TabPFN releases share neither an identical architecture nor
        an identical per-estimator feature budget, so nothing here should be
        read as a claim about what the backbone does internally.

        Earlier development used a TabPFN-2.5 synthetic-only checkpoint, chosen
        so that no real gene-expression data had entered backbone training. The
        2.5 checkpoints are no longer the backbone or a baseline here; see
        configs/legacy/ for the superseded defaults.
        """
        from tabpfn import TabPFNClassifier
        kw = {}
        if self._cfg("tabpfn_model_path"):
            kw["model_path"] = self._resolve_ckpt(self._cfg("tabpfn_model_path"))
        return TabPFNClassifier(
            n_estimators=n_estimators or self._cfg("tabpfn_n_estimators"),
            device=self.device,
            random_state=self.random_state,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            **kw,
        )

    def _ckpt_id(self):
        """Identify the checkpoint by its **contents**, not by its path.

        If different weights are written over the same file name, the path
        alone cannot tell them apart and an old ranking is read back as though
        the new backbone had produced it. The identifier hashes the file size
        together with the first and last megabyte: enough to catch an
        overwrite without reading all 40 MB every time.
        """
        import hashlib
        import os
        if not self._cfg("tabpfn_model_path"):
            return "(tabpfn default)"
        path = self._resolve_ckpt(self._cfg("tabpfn_model_path"))
        try:
            size = os.path.getsize(path)
            h = hashlib.sha256(str(size).encode())
            with open(path, "rb") as f:
                h.update(f.read(1 << 20))
                if size > (1 << 21):
                    f.seek(-(1 << 20), os.SEEK_END)
                    h.update(f.read())
            return f"{os.path.basename(path)}:{h.hexdigest()[:16]}"
        except OSError:
            return str(self._cfg("tabpfn_model_path"))

    @staticmethod
    def _resolve_ckpt(name):
        """Resolve a checkpoint name against the cache directory.

        TabPFN treats a relative model_path as relative to the **current
        working directory**, and downloads there if the file is missing. Run
        the example from example/ and a 40 MB checkpoint appears inside
        example/ -- a file that must never reach the repository.

        When only a name is given, it is sent to TabPFN's own cache directory
        instead. One copy is then downloaded no matter where the code runs, and
        the working directory stays clean. A name containing a path separator
        means the user chose the location, so it is used as given.
        """
        import os
        if os.sep in str(name) or os.path.isabs(str(name)):
            return name
        cache = os.environ.get("TABPFN_MODEL_CACHE_DIR") or \
            os.path.join(os.path.expanduser("~"), ".cache", "tabpfn")
        os.makedirs(cache, exist_ok=True)
        return os.path.join(cache, str(name))

    def _log(self, msg):
        if self.verbose:
            logging.getLogger("tabpfn_anchor").info(msg)

    # -- STEP 1: anchor signature construction (standalone method) --------
    def _pv(self, name, default=None):
        """The effective value fixed by the execution plan, else the setting."""
        pl = getattr(self, "execution_plan_", None) or {}
        return pl.get(name, default if default is not None else self._cfg(name))

    def _select(self, X, y_enc, n_features, cache, logger):
        """Run Steps 1.1 to 1.3 on the given (X, y) alone, producing the
        anchors and the ranking pool.

        With nested_selection=True this method is called again inside every
        inner fold, so that the labels of the inner validation samples cannot
        leak into feature selection.
        """
        rec = self.recorder or NULL_RECORDER
        timings = {}
        # ---- STEP 1-1 ------------------------------------------------------
        t0 = time.perf_counter()
        k1 = min(self._cfg("n_stage1"), n_features)
        route = self._pv("route", "screen_rf_shap_anchor")
        if not self._pv("stage1_enabled", True):
            # Skipping Step 1.1 means **passing every feature straight to the
            # RF.** An earlier version kept k1, making F1 = rank[:2000]; since
            # the ordering is the identity, that was "the first 2,000 columns
            # in their original order" -- truncation, not selection. Skipping
            # is defined as not filtering, so k1 opens up to everything too.
            k1 = n_features
            rank_stage1 = np.arange(n_features, dtype=int)
            imp_stage1 = np.full(n_features, np.nan)
            timings["stage1"] = 0.0
            self._log(f"[1-1] skipped -- all {n_features} features to the RF (route={route})")
        else:
            src = self._cfg("stage1_source")
            # Put source into **both** the cache name and the cache key.
            # Splitting only the name leaves the old keys alive; splitting only
            # the key overwrites one method's result under the other's name.
            # Older lr_pi runs and newer anova_f runs must not mix even when
            # they share a cache_dir.
            if src == "lr_pi":
                kw = dict(n_keep=k1, n_repeats=self._cfg("pi_n_repeats"),
                          penalty=self._cfg("lr_penalty"), C=self._cfg("lr_C"),
                          oof_splits=self._cfg("pi_oof_splits"),
                          random_state=self.random_state)
                key = dict(source=src, n_repeats=self._cfg("pi_n_repeats"),
                           penalty=self._cfg("lr_penalty"), C=self._cfg("lr_C"),
                           oof_splits=self._cfg("pi_oof_splits"),
                           seed=self.random_state,
                           code_version=__cache_version__)
            else:
                # ANOVA-F is deterministic and takes no seed. Putting a seed
                # in the key would redo the same computation once per seed.
                kw = dict(n_keep=k1)
                key = dict(source=src, code_version=__cache_version__)
            rank_stage1, imp_stage1 = cache.get_or_compute(
                f"stage1_{src}",
                lambda: run_stage1(src, X, y_enc, recorder=rec, logger=logger, **kw),
                params=key, X=X, y=y_enc)
        F1 = rank_stage1[:k1]
        timings["stage1"] = time.perf_counter() - t0

        # ---- STEP 1-2 ------------------------------------------------------
        t0 = time.perf_counter()
        k2 = min(self._cfg("n_stage2"), len(F1))
        rank_rf_local, imp_rf = cache.get_or_compute(
            "stage2_rf",
            lambda: stage2_random_forest(
                X[:, F1], y_enc, n_keep=k2, n_estimators=self._cfg("rf_n_estimators"),
                recorder=rec,
                random_state=self.random_state, n_jobs=self._cfg("n_jobs"), logger=logger),
            params=dict(n_estimators=self._cfg("rf_n_estimators"), seed=self.random_state,
                        code_version=__cache_version__),
            X=X[:, F1], y=y_enc)
        rank_rf_global = F1[rank_rf_local]
        F2 = rank_rf_global[:k2]
        timings["stage2_rf"] = time.perf_counter() - t0

        # ---- STEP 1-3 ------------------------------------------------------
        # anchor_source exists for the step-by-step ablation.
        #   "shap" : the published method (ANOVA-F -> RF-FI -> OddSHAP)
        #   "rf"   : stop after Step 1.2; anchors are the top n_anchor of the
        #            Random Forest ranking
        #   "lr"   : an ablation ranker, available only where its
        #            implementation is present (see selection.STAGE1_SOURCES)
        # Separating them is what makes it possible to see whether raising the
        # interaction order actually contributes.
        k3 = min(self._cfg("n_anchor"), len(F2))
        if self._cfg("anchor_source") not in ("shap", "slot"):
            if self._cfg("anchor_source") == "lr":
                # Choose by LR rank **within** F2. Taking a global LR top-k
                # would let anchors enter from outside F2 and shift the
                # residual block boundaries as well. The ablation would then
                # compare whole view constructions rather than the one signal
                # that picks the anchors, and no difference could be
                # attributed to the anchors.
                in_f2 = np.isin(rank_stage1, F2)
                anchor_idx = np.asarray(rank_stage1)[in_f2][:k3]
            elif self._cfg("anchor_source") == "rf":
                anchor_idx = F2[:k3]
            elif self._cfg("anchor_source") == "random":
                # The control. Anchors are drawn at random from F2, the
                # survivors of Step 1.2. Asking whether Step 1.3 contributes to
                # performance means asking whether it matters what the anchors
                # are. If random anchors perform the same, the SHAP step adds
                # nothing to accuracy and only the interpretability claim
                # remains -- and that claim does not hold either while the SHAP
                # estimate varies from seed to seed.
                rs = np.random.RandomState(self.random_state)
                anchor_idx = F2[rs.choice(len(F2), size=k3, replace=False)]
            else:
                raise ValueError(
                    f"anchor_source must be one of shap/rf/lr/random: {self._cfg('anchor_source')}")
            rank_shap_global = F2                      # keep RF order, no SHAP re-ranking
            timings["stage3_shap"] = 0.0
            self._log(f"[1-3] skipped (anchor_source={self._cfg('anchor_source')})")
            imp_shap = None
            m1 = None
        else:
            # The TabPFN used for explanation is separate from the one used
            # for prediction. shapiq calls the model once per coalition, so the
            # cost is close to proportional to n_estimators (measured on BRCA
            # with 500 features: 13.5 s at 32, 3.6 s at 8, 0.8 s at 1, per
            # explanation). At the manuscript's setting (n_est=32, 200 samples)
            # that is three hours per fold, so 180 hours over 60 folds. SHAP is
            # here only to produce an ordering, so the explanation step uses a
            # small ensemble while prediction keeps tabpfn_n_estimators.
            t0 = time.perf_counter()
            # Time fitting and explanation separately. Combined, they cannot
            # later answer whether the cost of anchor construction lies in
            # fitting the proxy or in approximating the Shapley values.
            with rec.stage("TabPFN_proxy_fit", gpu=True):
                m1 = self._make_tabpfn()
                m1.fit(X[:, F2], y_enc)

                ne_shap = self._cfg("shap_tabpfn_n_estimators") or self._cfg("tabpfn_n_estimators")
                if ne_shap == self._cfg("tabpfn_n_estimators"):
                    m_shap = m1
                else:
                    m_shap = self._make_tabpfn(ne_shap)
                    m_shap.fit(X[:, F2], y_enc)

            with rec.stage("OddSHAP", gpu=True,
                           note=f"budget={self._cfg('shap_budget')} n_eval={self._cfg('shap_max_samples')}"):
                rank_shap_local, imp_shap = cache.get_or_compute(
                    "stage3_shap",
                    lambda: stage3_tabpfn_shap(
                        m_shap, X[:, F2], y_enc, n_keep=k3,
                            max_samples=self._pv("shap_n_eval"), budget=self._cfg("shap_budget"),
                        backend=self._cfg("shap_backend"), class_mode=self._cfg("shap_class_mode"),
                        approximator=self._cfg("shap_approximator"),
                        random_state=self.random_state, logger=logger),
                    # Without the approximator, the checkpoint and the imputer
                    # in the key, a ranking produced by KernelSHAP would be
                    # read back under an OddSHAP setting -- the kind of
                    # accident that voids a central decision of the paper to
                    # save a few minutes of computation.
                    params=dict(max_samples=self._cfg("shap_max_samples"),
                                budget=self._cfg("shap_budget"), backend=self._cfg("shap_backend"),
                                approximator=self._cfg("shap_approximator"),
                                class_mode=self._cfg("shap_class_mode"),
                                imputer=SHAP_IMPUTER,
                                checkpoint=self._ckpt_id(),
                                shap_tabpfn_n_estimators=ne_shap,
                                seed=self.random_state,
                                code_version=__cache_version__),
                    X=X[:, F2], y=y_enc)
            rank_shap_global = F2[rank_shap_local]
            anchor_idx = rank_shap_global[:k3]
            timings["stage3_shap"] = time.perf_counter() - t0

        # ---- unified ranking ------------------------------------------------
        with rec.stage("rank_pool_build"):
            if self._cfg("residual_source") == "pool":
                rank_pool = build_rank_pool(
                    rank_shap_global, rank_rf_global, rank_stage1, n_features)
            elif self._cfg("residual_source") == "lr":
                rank_pool = np.asarray(rank_stage1, dtype=int)
            elif self._cfg("residual_source") == "rf":
                # R_RF = RF-FI(F1) || ANOVA-F(G \ F1). Added 2026-08-29.
                # The second scheme in the gene-ranking comparison. It drops
                # the SHAP re-ranking, orders the inside of F1 by RF
                # importance, and leaves everything outside F1 in stage-1
                # (ANOVA-F) order. The only difference from "pool" is that the
                # top 500 are not re-ranked by OddSHAP.
                rank_pool = build_rank_pool(
                    rank_rf_global, rank_stage1, rank_stage1, n_features)
            else:
                raise ValueError(
                    "residual_source must be one of 'pool', 'rf', 'lr': "
                    f"{self._cfg('residual_source')}")

        # anchor_source="slot": use the q-th 100-gene block of the full
        # ranking R as the anchors,
        #   A_q = R[100(q-1)+1 : 100q]
        # This is overwritten here because A_q depends on rank_pool; the branch
        # above runs before rank_pool exists and so cannot cut the slot.
        # q=1 equals the OddSHAP anchors (confirmed on 180 of 180 splits), and
        # for q<=5 we have A_q inside F2, so S_1 = F2 still holds. The union at
        # K=9 is R[1:3700] for every q.
        if self._cfg("anchor_source") == "slot":
            q = int(self._cfg("anchor_slot"))
            na = int(self._cfg("n_anchor"))
            lo, hi = na * (q - 1), na * q
            if hi > len(rank_pool):
                raise ValueError(f"anchor_slot={q} exceeds the ranking length {len(rank_pool)}.")
            anchor_idx = np.asarray(rank_pool[lo:hi], dtype=int)
            self._log(f"[1-3] slot {q}: using R[{lo+1}:{hi}] as the anchors")

        # Kept for diagnostics and interpretation.
        out_rank_stage1 = rank_stage1
        out_imp_stage1 = imp_stage1
        out_F2 = F2


        return dict(anchor_idx=anchor_idx, rank_pool=rank_pool, F2=out_F2,
                    F1=F1, k2=k2, rank_stage1=out_rank_stage1, imp_stage1=out_imp_stage1,
                    imp_rf=imp_rf, imp_shap=imp_shap, m1=m1, timings=timings)

    # -- building the feature views -----------------------------------------
    def _build_views(self, rank_pool, anchor_idx, k2, n_features):
        """Build each view model's feature index list per view_strategy."""
        vs = self._cfg("view_strategy")
        if vs == "anchor_ranked":
            return build_blocks(rank_pool, anchor_idx, n_stage2=k2,
                                n_residual=self._cfg("n_residual"),
                                k=self._pv("effective_n_views"))
        if vs == "single":
            return [np.asarray(rank_pool[:k2])]
        if vs == "ranked_disjoint":
            pool = np.asarray(rank_pool)
        elif vs == "random_disjoint":
            # Shuffle reproducibly, with the seed fixed per outer-training
            # fold. No labels are involved, so nothing leaks.
            rng = np.random.default_rng(self.random_state)
            pool = rng.permutation(n_features)
        else:
            raise ValueError(f"unknown view_strategy: {vs}")
        size = k2
        views = [pool[i * size:(i + 1) * size] for i in range(self._cfg("stack_size"))]
        # The same rule as build_blocks: discard a final block that is too
        # small. Keeping it would give the learners different feature budgets
        # and blur the A1/A2 comparison.
        return [v for v in views if len(v) == size]

    # -- fitting -------------------------------------------------------------
    # -- resolving the configuration -----------------------------------------
    # Keys that kwargs may not set. They determine the structure of the method
    # or its reproducibility, so allowing them here would leave it unclear
    # whether they or the top-level argument won.
    _RESERVED = {
        "tabpfn_kwargs": {"device", "random_state", "model_path"},
        "shap_kwargs": {"random_state"},
        "meta_kwargs": {"random_state"},
        "selection_kwargs": {"random_state", "n_views", "stack_size"},
    }
    # kwargs name -> flat attribute name
    _MAP = {
        "tabpfn_kwargs": {"n_estimators": "tabpfn_n_estimators",
                          "checkpoint": "tabpfn_model_path"},
        "shap_kwargs": {"budget": "shap_budget", "n_eval": "shap_max_samples",
                        "approximator": "shap_approximator",
                        "class_mode": "shap_class_mode",
                        "backend": "shap_backend",
                        "n_estimators": "shap_tabpfn_n_estimators"},
        "meta_kwargs": {"C": "meta_C"},
        "selection_kwargs": {"n_stage1": "n_stage1", "n_stage2": "n_stage2",
                             "n_anchor": "n_anchor", "n_residual": "n_residual",
                             "residual_source": "residual_source",
                             "anchor_source": "anchor_source",
                             "anchor_slot": "anchor_slot",
                             "view_strategy": "view_strategy",
                             "oof_folds": "oof_folds",
                             "nested_selection": "nested_selection",
                             "direct_threshold": "_direct_threshold",
                             "skip_stage1_threshold": "_skip_stage1_threshold"},
    }
    # Nested groups inside selection_kwargs
    _SUB = {"lr": {"C": "lr_C", "penalty": "lr_penalty",
                   "pi_n_repeats": "pi_n_repeats",
                   "pi_oof_splits": "pi_oof_splits"},
            "rf": {"n_estimators": "rf_n_estimators", "n_jobs": "n_jobs"}}

    # Route thresholds for auto mode. Changeable through selection_kwargs, but
    # not advertised on the front page of the README.
    DIRECT_THRESHOLD = 2000      # at or below: TabPFN directly, no anchor split
    SKIP_STAGE1_THRESHOLD = 5000  # at or below: skip Step 1.1, start at the RF

    def _resolve(self):
        """Flatten the high-level API and kwargs into one configuration.

        There is one rule: **the top-level argument wins.** Structural settings
        such as device and random_state are forbidden in kwargs outright.
        Rejecting the call beats a configuration where it is ambiguous which
        side won.
        """
        r = {}
        for ns, kw in (("selection_kwargs", self.selection_kwargs),
                       ("tabpfn_kwargs", self.tabpfn_kwargs),
                       ("shap_kwargs", self.shap_kwargs),
                       ("meta_kwargs", self.meta_kwargs)):
            if not kw:
                continue
            bad = self._RESERVED[ns] & set(kw)
            if bad:
                raise ValueError(
                    f"{ns} cannot set: {sorted(bad)}. "
                    "Pass them as top-level arguments (device, random_state, "
                    "n_views, checkpoint).")
            for k, v in kw.items():
                if ns == "selection_kwargs" and k in self._SUB:
                    for k2, v2 in (v or {}).items():
                        if k2 not in self._SUB[k]:
                            raise ValueError(f"selection_kwargs['{k}'] has no key {k2}")
                        r[self._SUB[k][k2]] = v2
                    continue
                if k not in self._MAP[ns]:
                    raise ValueError(
                        f"{ns} has no key {k}. "
                        f"available: {sorted(self._MAP[ns])}")
                r[self._MAP[ns][k]] = v
        if self.n_views is not None:
            r["stack_size"] = int(self.n_views)
        return r

    def _cfg(self, name):
        """Read a resolved setting: the kwargs value if given, else the flat one."""
        return getattr(self, "_resolved", {}).get(name, getattr(self, name, None))

    def _validate_params_(self):
        """Stop before fitting if the configuration contradicts the method.

        With anchor + residual != n_stage2 the first view and the later views
        have different sizes. "The same feature budget for every learner" then
        no longer holds, and no result can be attributed to the view
        construction. The internal experiments happened to satisfy this, but
        for anyone outside it is a trap that breaks quietly.
        """
        g = self._cfg
        if g("stage1_source") not in STAGE1_SOURCES:
            raise ValueError(
                f"stage1_source must be one of {sorted(STAGE1_SOURCES)}: "
                f"{g('stage1_source')!r}")
        if g("anchor_source") == "shap" and g("shap_backend") not in SHAP_BACKENDS:
            raise ValueError(
                f"shap_backend must be one of {sorted(SHAP_BACKENDS)}: "
                f"{g('shap_backend')!r}")
        if not 0 < g("n_anchor") <= g("n_stage2") <= g("n_stage1"):
            raise ValueError(
                "require 0 < n_anchor <= n_stage2 <= n_stage1: "
                f"{self._cfg('n_anchor')}, {self._cfg('n_stage2')}, {self._cfg('n_stage1')}")
        if self._cfg("view_strategy") == "anchor_ranked" and \
                self._cfg("n_anchor") + self._cfg("n_residual") != self._cfg("n_stage2"):
            raise ValueError(
                "an anchor-conditioned view requires n_anchor + n_residual "
                f'== n_stage2: {g("n_anchor")} + {g("n_residual")} != {g("n_stage2")}')
        if g("stack_size") < 1:
            raise ValueError(f'n_views >= 1: {g("stack_size")}')
        if self._cfg("oof_meta") and g("oof_folds") < 2:
            raise ValueError(f'oof_folds >= 2: {g("oof_folds")}')
        if self._cfg("pi_oof_splits") is not None and 0 < self._cfg("pi_oof_splits") < 2:
            raise ValueError(
                f"pi_oof_splits must be >= 2, or <= 1 for in-sample: {self._cfg('pi_oof_splits')}")

    def _check_X(self, X, reset):
        """Check column names and column count against what fit saw.

        Without this check, a DataFrame at fit time and a DataFrame with a
        **different column order** at predict time would pass unnoticed. This
        classifier slices features by integer index, so the gene names would
        still look right while the values came from entirely different genes.
        That is the kind of error that returns a quietly wrong answer.

        Uses validate_data from sklearn 1.6+, and performs the same checks by
        hand when it is unavailable.
        """
        cols = getattr(X, "columns", None)
        if reset:
            if cols is not None:
                got = np.asarray(cols, dtype=object)
                # Duplicate names make the order check meaningless: two columns
                # answering to one name cannot be told apart later.
                uniq, counts = np.unique(got, return_counts=True)
                if (counts > 1).any():
                    raise ValueError(
                        "duplicate feature names: "
                        f"{sorted(uniq[counts > 1])[:5]}. This classifier "
                        "slices features by position and checks the names to "
                        "detect a reordering, which duplicates defeat.")
                self.feature_names_in_ = got
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        else:
            names = getattr(self, "feature_names_in_", None)
            if names is not None and cols is not None:
                got = np.asarray(cols, dtype=object)
                if got.shape != names.shape or (got != names).any():
                    raise ValueError(
                        "the feature names at predict differ from those at "
                        "fit. This classifier slices features by position, so a "
                        "changed order reads different genes. Pass the columns "
                        "in the same order as at fit.")
            elif names is not None and cols is None:
                # Fitted with names, predicting without them. The count check
                # below still applies, but nothing can verify the order, so say
                # so rather than letting a silent reordering through.
                warnings.warn(
                    "fitted with feature names but predicting on an array "
                    "without them; the column order cannot be checked. Pass a "
                    "DataFrame with the same columns as at fit.",
                    UserWarning, stacklevel=3)
            n_in = getattr(self, "n_features_in_", None)
            if n_in is not None and np.asarray(X).shape[1] != n_in:
                raise ValueError(
                    f"different number of features: fit {n_in}, predict "
                    f"{np.asarray(X).shape[1]}")
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if not np.isfinite(X).all():
            raise ValueError("the input contains NaN or inf.")
        return X

    def _plan(self, n_samples, n_features, y_enc, n_classes):
        """Decide the execution route and record why.

        auto adapts to the data; anchor -- the published profile -- fails on
        any deviation. If the route varied by cohort, the definition of the
        method would vary by cohort too.

        Three categories:
            algorithmically impossible  -> ERROR   (one class; a class of one)
            statistically doubtful      -> WARNING (very few samples)
            ordinary adaptation         -> INFO/WARNING plus execution_plan_
        """
        import warnings as _w
        g = self._cfg
        strict = self.mode == "anchor"
        counts = np.bincount(y_enc, minlength=n_classes)
        min_class = int(counts[counts > 0].min()) if counts.any() else 0

        if n_classes < 2:
            raise ValueError("classification needs at least two classes.")
        if min_class < 2:
            raise ValueError(
                "TabPFN-Anchor needs at least two training samples in every "
                f"class for out-of-fold estimation; the smallest class has "
                f"{min_class}.")
        if n_samples < 50:
            _w.warn(f"only {n_samples} training samples. Fitting will succeed, "
                    "but the feature ranking and the stability of the anchors "
                    "are hard to trust at this size.", stacklevel=3)

        direct = g("_direct_threshold") or self.DIRECT_THRESHOLD
        skip_s1 = g("_skip_stage1_threshold") or self.SKIP_STAGE1_THRESHOLD

        # Do not put the selector in the route name. stage1_source records
        # which selector ran; the old name ("lr_rf_shap_anchor") was printed
        # even on ANOVA-F runs, so the log lied.
        if self.mode == "tabpfn":
            route, use_stage1, use_rf = "direct_tabpfn", False, False
        elif self.mode == "auto" and n_features <= direct:
            route, use_stage1, use_rf = "direct_tabpfn", False, False
            _w.warn(f"the input has {n_features} features ({direct} or fewer), "
                    "so the anchor decomposition is skipped and TabPFN is used "
                    "directly.", stacklevel=3)
        elif self.mode == "auto" and n_features <= skip_s1:
            route, use_stage1, use_rf = "rf_shap_anchor", False, True
            _w.warn(f"the input has {n_features} features ({skip_s1} or fewer), "
                    f"so Step 1.1 ({g('stage1_source')}) is skipped and the run "
                    "starts at the RF.",
                    stacklevel=3)
        else:
            route, use_stage1, use_rf = "screen_rf_shap_anchor", True, True

        req_k = int(g("stack_size"))
        if route == "direct_tabpfn":
            eff_k = 1
        else:
            # The budget is anchor 100 + residual 400 x (K-1), so too small a
            # P cannot fill K views. auto reduces K; the published profile
            # fails instead.
            max_k = max(1, (n_features - g("n_anchor")) // max(1, g("n_residual")))
            eff_k = min(req_k, max_k)
            if eff_k < req_k:
                # View 1 uses anchors plus the rest of F2, i.e. n_stage2, not n_anchor.
                need = g("n_stage2") + g("n_residual") * (req_k - 1)
                if strict:
                    raise ValueError(
                        f"the paper profile requires n_views={req_k}, which "
                        f"needs at least {need} features, but received "
                        f"{n_features}.")
                _w.warn(f"n_views={req_k} was requested, but {n_features} "
                        f"features allow only {eff_k} complete views; "
                        f"continuing with n_views={eff_k}.", stacklevel=3)

        req_j = int(g("oof_folds"))
        eff_j = max(2, min(req_j, min_class))
        if eff_j < req_j:
            if strict:
                raise ValueError(
                    f"the paper profile uses oof_folds={req_j}, but the "
                    f"smallest class has {min_class} samples.")
            _w.warn(f"the smallest class has {min_class} samples, so the OOF "
                    f"folds are reduced from {req_j} to {eff_j}.", stacklevel=3)

        eff_ne = min(int(g("shap_max_samples")), n_samples)

        self.execution_plan_ = {
            "mode": self.mode, "route": route,
            "n_samples": int(n_samples), "n_features": int(n_features),
            "n_classes": int(n_classes), "min_class_count": min_class,
            "stage1_enabled": use_stage1, "stage2_rf": use_rf,
            "stage1_source": g("stage1_source"),
            "stage3_oddshap": route != "direct_tabpfn",
            "n_stage1": int(g("n_stage1")), "n_stage2": int(g("n_stage2")),
            "n_anchor": int(g("n_anchor")), "n_residual": int(g("n_residual")),
            "requested_n_views": req_k, "effective_n_views": int(eff_k),
            "requested_oof_folds": req_j, "effective_oof_folds": int(eff_j),
            "shap_n_eval": int(eff_ne),
            # Recorded for the run log: whether the input was wider than the
            # 500-feature views this procedure constructs.
            "native_feature_subsampling": bool(
                route == "direct_tabpfn" and n_features > 500),
        }
        self.requested_n_views_, self.effective_n_views_ = req_k, int(eff_k)
        self.requested_oof_folds_, self.effective_oof_folds_ = req_j, int(eff_j)
        self.resolved_config_ = {
            "mode": self.mode, "route": route,
            "selection": {"n_stage1": g("n_stage1"), "n_stage2": g("n_stage2"),
                          "n_anchor": g("n_anchor"), "n_residual": g("n_residual"),
                          "stage1_source": g("stage1_source"),
                          "lr": {"C": g("lr_C"), "penalty": g("lr_penalty"),
                                 "pi_n_repeats": g("pi_n_repeats"),
                                 "pi_oof_splits": g("pi_oof_splits")},
                          "rf": {"n_estimators": g("rf_n_estimators"),
                                 "n_jobs": g("n_jobs")}},
            "shap": {"approximator": g("shap_approximator"),
                     "budget": g("shap_budget"), "n_eval": eff_ne,
                     "class_mode": g("shap_class_mode"), "imputer": SHAP_IMPUTER},
            "tabpfn": {"checkpoint": self._ckpt_id(),
                       "n_estimators": g("tabpfn_n_estimators")},
            "stacking": {"n_views": int(eff_k), "oof_folds": int(eff_j),
                         "meta": {"type": "logistic_regression", "C": g("meta_C")}},
        }
        if self.verbose:
            self._log(f"[plan] {route}  P={n_features} K={eff_k} J={eff_j}")
        return self.execution_plan_

    def fit(self, X, y):
        self._resolved = self._resolve()
        self._validate_params_()
        X = self._check_X(X, reset=True)
        # LabelEncoder handles both string and integer labels. Forcing
        # astype(str) would make predict return strings rather than the
        # original dtype, breaking every comparison the caller makes.
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(np.asarray(y))
        self.classes_ = self._le.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.n_features_in_ = n_features

        self._plan(X.shape[0], n_features, y_enc, n_classes)
        if self.execution_plan_["route"] == "direct_tabpfn":
            # TabPFN is used directly, with no anchor decomposition. This
            # route has neither anchors nor a feature ranking, so the related
            # attributes are not created: faking what does not exist with an
            # empty table would let the user believe it does.
            rec0 = self.recorder or NULL_RECORDER
            m = self._make_tabpfn()
            with rec0.stage("view_refit", view=1, gpu=True):
                m.fit(X, y_enc)
            self.models_, self.meta_ = [m], None
            self.blocks_ = [np.arange(n_features)]
            # A constructor argument must not be changed in fit. Fitting one
            # object on a small P and then refitting it on a large P would
            # leave the second run stuck at "none", and get_params() would
            # differ before and after fit. Keep it as a fitted attribute.
            self.aggregation_ = "none"
            self.n_blocks_, self.n_genes_used_ = 1, int(n_features)
            self.timings_ = {"stage_direct_tabpfn": 0.0}
            return self
        rec = self.recorder or NULL_RECORDER
        logger = logging.getLogger("tabpfn_anchor") if self.verbose else None
        cache = StageCache(self.cache_dir, enabled=self.cache_dir is not None,
                           verbose=self.verbose)
        timings = {}

        sel = self._select(X, y_enc, n_features, cache, logger)
        self.anchor_idx_ = sel['anchor_idx']
        self.rank_pool_ = sel['rank_pool']
        self.rank_stage1_ = sel['rank_stage1']
        self.importance_stage1_ = sel['imp_stage1']
        # Backwards compatibility. The name is inaccurate, since it is filled
        # even when Step 1.1 is ANOVA-F. New code uses rank_stage1_ and
        # importance_stage1_.
        self.rank_lr_ = self.rank_stage1_
        self.importance_lr_ = self.importance_stage1_
        self.stage2_idx_ = sel['F2']
        self.stage1_idx_ = sel['F1']
        # The RF importances from Step 1.2. The paper's LR->RF-100 selector
        # (the partial pipeline without the SHAP step) is defined by these
        # values. They were in the returned dictionary but were not kept as an
        # attribute, so they were missing from the ranking dumps.
        self.imp_rf_ = sel.get('imp_rf')
        self._build_ranking_table(sel, n_features)
        F2, k2, m1 = sel['F2'], sel['k2'], sel['m1']
        imp_shap = sel['imp_shap']
        timings = dict(sel['timings'])

        # ---- STEP 2 --------------------------------------------------------
        t0 = time.perf_counter()
        with rec.stage("view_build"):
            self.blocks_ = self._build_views(self.rank_pool_, self.anchor_idx_,
                                             k2, n_features)
        if len(self.blocks_) < self._cfg("stack_size"):
            self._log(f"[STEP2] ranking pool exhausted: K={self._cfg('stack_size')} "
                      f"requested, only {len(self.blocks_)} built (P={n_features})")

        self.aggregation_ = self._cfg("aggregation")
        if self.aggregation_ == "none":
            # A single learner. Its prediction is used directly, with no meta-learner.
            if len(self.blocks_) != 1:
                raise ValueError(
                    f"aggregation='none' is for a single view, but {len(self.blocks_)} were "
                    f"built (stack_size={self._cfg('stack_size')}).")
            m = self._make_tabpfn()
            with rec.stage("view_refit", view=1, gpu=True):
                m.fit(np.ascontiguousarray(X[:, self.blocks_[0]]), y_enc)
            self.models_ = [m]
            self.meta_ = None
            timings["stage_stacking"] = time.perf_counter() - t0
            self.timings_ = timings
            self.n_blocks_ = 1
            self.n_genes_used_ = int(len(self.blocks_[0]))
            return self

        if self.aggregation_ == "vote":
            # For A1-A3. Aggregation is a hard majority vote at test time, so
            # neither a meta-learner nor out-of-fold meta-features are needed.
            self.models_ = []
            for i, feats in enumerate(self.blocks_, start=1):
                m = self._make_tabpfn()
                with rec.stage("view_refit", view=i, gpu=True):
                    m.fit(np.ascontiguousarray(X[:, feats]), y_enc)
                self.models_.append(m)
                self._log(f"[STEP2] M{i:02d} |F|={len(feats)} (vote)")
            self.meta_ = None
            timings["stage_stacking"] = time.perf_counter() - t0
            self.timings_ = timings
            self.n_blocks_ = len(self.blocks_)
            self.n_genes_used_ = int(len(np.unique(np.concatenate(self.blocks_))))
            return self

        if self._cfg("nested_selection"):
            # Fully nested: feature selection is redone from scratch in every
            # inner fold. With oof_meta alone, the inner validation samples'
            # labels stay out of base model fitting but have already entered
            # selection; this removes that residual dependence. The price is
            # repeating the most expensive step (PI / RF / SHAP) once per
            # fold.
            counts = np.bincount(y_enc, minlength=n_classes)
            nz = counts[counts > 0]
            n_splits = max(2, int(min(self._cfg("oof_folds"), nz.min()))) if len(nz) else self._cfg("oof_folds")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=self.random_state)
            self._log(f"[STEP2] nested_selection: reselecting in each of {n_splits} inner folds")

            folds, k_eff = [], len(self.blocks_)
            for tr, va in skf.split(X, y_enc):
                # Inner-fold selections are cached too. The key is not a file
                # path but (a fingerprint of X[tr] and y[tr] + the step's
                # parameters), so a different fold gives a different key. In
                # exchange, other variants sharing the same (dataset, outer
                # fold, inner fold) -- K = 1..15 in a K sweep, say -- share
                # Step 1, and the marginal cost of another variant is STEP 2
                # alone. Adding repeats does change the training partition
                # itself, so the cache is not reused there.
                s_f = self._select(X[tr], y_enc[tr], n_features, cache, logger)
                b_f = build_blocks(s_f["rank_pool"], s_f["anchor_idx"],
                                   n_stage2=s_f["k2"], n_residual=self._cfg("n_residual"),
                                   k=self._cfg("stack_size"))
                folds.append((tr, va, b_f))
                k_eff = min(k_eff, len(b_f))

            if k_eff < len(self.blocks_):
                self._log(f"[STEP2] block count aligned to K={k_eff} (pool size varies by fold)")
                self.blocks_ = self.blocks_[:k_eff]

            meta_parts = [np.zeros((len(y_enc), n_classes), dtype=np.float32)
                          for _ in range(k_eff)]
            for j, (tr, va, b_f) in enumerate(folds, start=1):
                for k in range(k_eff):
                    feats = b_f[k]
                    m = self._make_tabpfn()
                    with rec.stage("view_oof_fit", view=k + 1, inner_fold=j, gpu=True):
                        m.fit(np.ascontiguousarray(X[np.ix_(tr, feats)]), y_enc[tr])
                    with rec.stage("view_oof_predict", view=k + 1, inner_fold=j, gpu=True):
                        p = m.predict_proba(np.ascontiguousarray(X[np.ix_(va, feats)]))
                    meta_parts[k][va] = _align(p, m.classes_, n_classes)

            # The model used at inference is refitted from the selection made
            # on the whole outer training split.
            self.models_ = []
            for i, feats in enumerate(self.blocks_, start=1):
                m = self._make_tabpfn()
                with rec.stage("view_refit", view=i, gpu=True):
                    m.fit(np.ascontiguousarray(X[:, feats]), y_enc)
                self.models_.append(m)
                self._log(f"[STEP2] M{i:02d} |F|={len(feats)} refit (nested)")
            timings["stage_stacking"] = time.perf_counter() - t0
            return self._finish_fit(meta_parts, y_enc, timings, imp_shap)

        self.models_ = []
        meta_parts = []
        for i, feats in enumerate(self.blocks_, start=1):
            Xb = np.ascontiguousarray(X[:, feats])
            if self._cfg("oof_meta"):
                oof, full, ns = oof_and_full_probs(
                    self._make_tabpfn, Xb, y_enc, n_classes,
                    n_folds=self._pv("effective_oof_folds"), random_state=self.random_state,
                    recorder=rec, view=i)
                meta_parts.append(oof)
                self.models_.append(full)
                self._log(f"[STEP2] M{i:02d} |F|={len(feats)} oof_folds={ns}")
            else:
                # m1 was already fitted on F2 during Step 1.3, so it is
                # reused. When anchor_source is not shap there is no m1, and a
                # fresh model is fitted.
                reuse = (i == 1 and m1 is not None
                         and len(feats) == len(F2) and np.array_equal(feats, F2))
                m = m1 if reuse else self._make_tabpfn()
                if not reuse:
                    with rec.stage("view_refit", view=i, gpu=True):
                        m.fit(Xb, y_enc)
                with rec.stage("view_insample_predict", view=i, gpu=True):
                    meta_parts.append(m.predict_proba(Xb).astype(np.float32))
                self.models_.append(m)
                self._log(f"[STEP2] M{i:02d} |F|={len(feats)} (in-sample)")

        timings["stage_stacking"] = time.perf_counter() - t0
        return self._finish_fit(meta_parts, y_enc, timings, imp_shap)

    def _finish_fit(self, meta_parts, y_enc, timings, imp_shap=None):
        """Fit the meta-learner and record diagnostics; shared by both routes."""
        rec = self.recorder or NULL_RECORDER
        Z = stack_meta_features(meta_parts)
        # Keep the per-view out-of-fold probabilities as they are. This
        # changes no computation; it only retains a reference to arrays that
        # already exist.
        #
        # The development experiments use this. blocks_ is a prefix that does
        # not depend on k (see build_blocks: view j = anchor +
        # non_anchor[400(j-1):400j]), so one fit at stack_size=30 makes the
        # first K blocks exactly the meta-input for stack_size=K. Refitting
        # only the meta-learner then reproduces each K without a full refit.
        self.meta_parts_ = meta_parts
        # _cfg, not self.meta_C: meta_kwargs={"C": ...} resolves through the
        # same path as every other setting. Reading the flat attribute here
        # ignored the override silently, while the recorded configuration
        # reported the override -- so the record and the fitted model disagreed.
        self.meta_ = LogisticRegression(penalty="l2", C=self._cfg("meta_C"),
                                        solver="lbfgs", max_iter=5000,
                                        random_state=self.random_state)
        with rec.stage("meta_fit"):
            self.meta_.fit(Z, y_enc)
        self.timings_ = timings
        self.n_blocks_ = len(self.blocks_)
        self.n_genes_used_ = int(len(np.unique(np.concatenate(self.blocks_))))
        return self

    # -- prediction ----------------------------------------------------------
    def _meta_input(self, X):
        rec = getattr(self, "recorder", None) or NULL_RECORDER
        parts = []
        for i, (m, feats) in enumerate(zip(self.models_, self.blocks_), start=1):
            # TabPFN's fit is lazy, so most of the real computation is here.
            with rec.stage("view_test_predict", view=i, gpu=True):
                p = m.predict_proba(np.ascontiguousarray(X[:, feats]))
            parts.append(np.asarray(p, dtype=np.float32))
        return stack_meta_features(parts)

    def _vote_proba(self, X):
        """Vote shares over the view models' labels; ties go to the lower index."""
        rec = getattr(self, "recorder", None) or NULL_RECORDER
        n_classes = len(self.classes_)
        votes = np.zeros((len(X), n_classes), dtype=np.float32)
        for i, (m, feats) in enumerate(zip(self.models_, self.blocks_), start=1):
            with rec.stage("view_test_predict", view=i, gpu=True):
                lab = np.asarray(m.predict(np.ascontiguousarray(X[:, feats])),
                                 dtype=int)
            votes[np.arange(len(X)), lab] += 1.0
        return votes / max(1, len(self.models_))

    def predict_proba(self, X):
        rec = getattr(self, "recorder", None) or NULL_RECORDER
        if getattr(self, "aggregation_", self.aggregation) == "none":
            # A single learner: its prediction is the result.
            X = self._check_X(X, reset=False)
            with rec.stage("view_test_predict", view=1, gpu=True):
                p = self.models_[0].predict_proba(
                    np.ascontiguousarray(X[:, self.blocks_[0]]))
            return _align(p, self.models_[0].classes_, len(self.classes_))
        if getattr(self, "aggregation_", self.aggregation) == "vote":
            check_is_fitted(self, "models_")
            return self._vote_proba(self._check_X(X, reset=False))
        check_is_fitted(self, "meta_")
        Z = self._meta_input(self._check_X(X, reset=False))
        with rec.stage("meta_predict"):
            return self.meta_.predict_proba(Z)

    def predict(self, X):
        # The three routes use different column conventions.
        #   stack : predict_proba's column order follows meta_.classes_
        #   vote  : already in encoded class-index order
        #   none  : a single learner passed through _align, so index order
        # Handling none together with stack dies on A0, where meta_ is None.
        # Call predict_proba **once**. TabPFN's fit is lazy and the real
        # computation happens in predict, so calling it twice would run the
        # test-time inference of all K view models twice over, exactly
        # doubling the inference time.
        agg = getattr(self, "aggregation_", self.aggregation)
        col = np.argmax(self.predict_proba(X), axis=1)
        if agg in ("vote", "none"):
            return self._le.inverse_transform(col.astype(int))
        # Under stack the columns follow meta_.classes_, so undo that once.
        enc = np.asarray(self.meta_.classes_)[col]
        return self._le.inverse_transform(enc)

    # -- interpretation --------------------------------------------------------
    def _build_ranking_table(self, sel, n_features):
        """Build the unified feature ranking table as a fitted attribute.

        `feature_ranking_` holds every feature in unified-ranking (rank_pool)
        order, and `anchor_features_` is the anchor slice of it. Both are
        pandas DataFrames and, following the sklearn convention, exist only
        after fit.

        **score means something different in each step.** OddSHAP gives a
        Shapley magnitude, RF a Gini importance, LR a permutation importance
        (the increase in log loss). Sharing a column does not make them
        comparable. Only relative magnitude within one step is meaningful; do
        not place numbers side by side across different source_stage values.

        The step assignment follows build_rank_pool exactly.

            inside F2          -> OddSHAP  (the span Step 1.3 re-ranked)
            inside F1, not F2  -> RF       (the span Step 1.2 left)
            outside F1         -> LR       (only the Step 1.1 ranking exists)
        """
        import pandas as pd

        rank_pool = np.asarray(self.rank_pool_, dtype=int)
        F1 = np.asarray(sel["F1"], dtype=int)
        F2 = np.asarray(sel["F2"], dtype=int)

        # On a route that skipped Step 1.1, that span must not be labelled
        # LR: it would present an importance that was never computed as though
        # it had been. Record whatever Step 1.1 actually was. This used to
        # print "LR" unconditionally, so a table from an ANOVA-F run claimed
        # to be LR.
        if not self._pv("stage1_enabled", True):
            base = "SKIPPED"
        else:
            base = {"anova_f": "ANOVA-F", "lr_pi": "LR+PI"}.get(
                self._cfg("stage1_source"), str(self._cfg("stage1_source")))
        stage = np.full(n_features, base, dtype=object)
        stage[F1] = "RF"
        if sel.get("imp_shap") is not None:
            stage[F2] = "OddSHAP"
        else:
            stage[F2] = "RF"          # no re-ranking unless anchor_source == 'shap'

        # Spread each step's importances over global indices; NaN where uncomputed.
        score = np.full(n_features, np.nan)
        if sel.get("imp_stage1") is not None:
            score[:] = np.asarray(sel["imp_stage1"], dtype=float)
        if sel.get("imp_rf") is not None:
            score[F1] = np.asarray(sel["imp_rf"], dtype=float)
        if sel.get("imp_shap") is not None:
            score[F2] = np.asarray(sel["imp_shap"], dtype=float)

        anchors = set(int(i) for i in self.anchor_idx_)
        names = getattr(self, "feature_names_in_", None)
        feat = ([names[i] for i in rank_pool] if names is not None
                else rank_pool.tolist())

        self.feature_ranking_ = pd.DataFrame({
            "rank": np.arange(1, len(rank_pool) + 1),
            "feature": feat,
            "source_stage": [stage[i] for i in rank_pool],
            "score": score[rank_pool],
            "is_anchor": [int(int(i) in anchors) for i in rank_pool],
        })

        a = np.asarray(self.anchor_idx_, dtype=int)
        self.anchor_features_ = pd.DataFrame({
            "anchor_rank": np.arange(1, len(a) + 1),
            "feature": ([names[i] for i in a] if names is not None else a.tolist()),
            "importance": score[a],
            "source_stage": [stage[i] for i in a],
        })

    def anchor_features(self, feature_names=None):
        """Return the anchor genes in SHAP-ranking order.

        `anchor_features_` (a DataFrame) fits the sklearn convention better and
        carries more information. This method is kept for the experiment
        harness, which passes feature names in from outside.
        """
        check_is_fitted(self, "anchor_idx_")
        if feature_names is None:
            return self.anchor_idx_.copy()
        return [feature_names[i] for i in self.anchor_idx_]
