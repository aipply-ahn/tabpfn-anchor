# -*- coding: utf-8 -*-
"""Contract tests for the public estimator.

`sklearn.utils.estimator_checks.check_estimator` is not run as a whole.  It
fits the estimator many times on tiny low-dimensional arrays, and TabPFN-Anchor
needs a downloaded backbone checkpoint, a GPU to be practical, and an input
where the number of features exceeds the number of samples.  The checks that
matter and are meaningful here are asserted individually below; the ones that
are skipped, and why, are listed in README.md.

    pytest -q

Most tests run without the backbone. The ones that fit a model need the
checkpoint named in configs/paper and skip themselves without it.
"""

from __future__ import annotations

import io
import tarfile
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from sklearn.datasets import make_classification

from tabpfn_anchor import TabPFNAnchorClassifier
from tabpfn_anchor.cli import _pick_device

# Cut down hard so the fitting tests stay usable. Almost all of the cost is the
# OddSHAP stage, whose price is (explained samples) x (classes explained) x
# (coalition budget); `shap_class_mode="predicted"` drops the middle factor to
# one. These are test settings, not the published configuration.
FAST = dict(mode="anchor",
            n_stage1=400, n_stage2=60, n_anchor=20, n_residual=40, stack_size=2,
            # shap_max_samples must be at least the number of classes: the
            # attribution explains at least one sample per class.
            rf_n_estimators=50, shap_budget=16, shap_max_samples=3,
            shap_class_mode="predicted", oof_folds=2,
            tabpfn_n_estimators=1, shap_tabpfn_n_estimators=1, random_state=0,
            # Name one device. The default "auto" places a context on every
            # visible GPU, which on a multi-GPU machine is far slower.
            device=_pick_device())

PAPER_YAML = Path(__file__).resolve().parents[1] / "configs/paper/tabpfn_anchor.yaml"


def synthetic(n_features: int = 4000):
    """The same call as the README example, straight from scikit-learn.

    4,000 features is wide enough for two separate reasons: it clears the
    2,000-feature threshold at which mode="auto" hands the input to TabPFN and
    builds no ranking, and it leaves 100 + 400 * 9 = 3,700 for the published
    view construction, so K is not capped below 9 by the data.

    Everything not named here is left at its scikit-learn default. The
    informative and redundant counts are named because the defaults are two of
    each, which would be a trivial signal in this many columns.
    """
    X, y = make_classification(
        n_samples=300, n_features=n_features, n_informative=20, n_redundant=80,
        n_classes=3, random_state=0)
    names = [f"feature_{j + 1:05d}" for j in range(n_features)]
    return (pd.DataFrame(X, columns=names),
            pd.Series([f"Class_{chr(65 + c)}" for c in y], name="subtype"))


@pytest.fixture(scope="module")
def toy():
    return synthetic()


# --------------------------------------------------------------- no fitting
def test_is_a_sklearn_classifier():
    clf = TabPFNAnchorClassifier()
    assert isinstance(clf, BaseEstimator) and isinstance(clf, ClassifierMixin)


def test_clone_roundtrip():
    """`clone` is the contract that __init__ stores its arguments unchanged."""
    clf = TabPFNAnchorClassifier(n_anchor=37, stack_size=4)
    twin = clone(clf)
    assert twin.get_params() == clf.get_params()
    assert twin.n_anchor == 37 and twin.stack_size == 4


def test_get_set_params():
    clf = TabPFNAnchorClassifier()
    clf.set_params(n_anchor=11)
    assert clf.get_params()["n_anchor"] == 11


def test_defaults_match_the_published_configuration():
    """Every value in the published YAML must equal the estimator's default.

    A public default that silently differs from the published method would make
    every unconfigured run a different algorithm. This compares the whole file
    rather than a hand-picked subset, so a parameter added to the YAML cannot
    escape the check.
    """
    import yaml
    published = yaml.safe_load(PAPER_YAML.read_text())
    defaults = TabPFNAnchorClassifier().get_params()

    unknown = sorted(set(published) - set(defaults))
    assert not unknown, f"configs/paper names parameters the estimator lacks: {unknown}"

    mismatched = {k: (v, defaults[k]) for k, v in published.items() if defaults[k] != v}
    assert not mismatched, (
        "configs/paper disagrees with the estimator defaults "
        f"(yaml, default): {mismatched}")


def test_published_mode_is_not_the_strict_profile():
    """mode must be "auto".

    mode="anchor" is a strict profile: it raises rather than reducing
    oof_folds when a class is smaller than the requested number of folds. The
    reported runs do reduce it, so the strict profile cannot reproduce them.
    """
    assert TabPFNAnchorClassifier().get_params()["mode"] == "auto"


def test_only_runnable_stage1_sources_are_offered():
    """Every advertised stage1_source must actually be importable.

    The ablation implementation of "lr_pi" is not part of this release, so it
    must not appear in the registry either; otherwise selecting it raises
    ModuleNotFoundError at fit time.
    """
    from tabpfn_anchor.selection import STAGE1_SOURCES, run_stage1
    for name in STAGE1_SOURCES:
        X = np.random.default_rng(0).normal(size=(20, 8))
        y = np.repeat([0, 1], 10)
        run_stage1(name, X, y, n_keep=4)


def test_only_runnable_shap_backends_are_offered():
    """Same rule for Step 1.3: an offered backend must have an implementation.

    The legacy KernelSHAP path is not part of this release, so it must be
    rejected by name rather than reaching a missing import at fit time.
    """
    from tabpfn_anchor.selection import SHAP_BACKENDS
    for name in SHAP_BACKENDS:
        assert name in ("auto", "shapiq", "kernelshap", "batched")
    for name in ("kernelshap", "batched"):
        if name in SHAP_BACKENDS:
            continue
        clf = TabPFNAnchorClassifier(shap_backend=name, n_stage1=40, n_stage2=20,
                                     n_anchor=5, n_residual=15)
        X = np.random.default_rng(0).normal(size=(20, 30))
        with pytest.raises(ValueError, match="shap_backend"):
            clf.fit(X, np.repeat([0, 1], 10))


def test_unknown_stage1_source_is_rejected():
    clf = TabPFNAnchorClassifier(stage1_source="nope")
    X = np.random.default_rng(0).normal(size=(20, 30))
    with pytest.raises(ValueError, match="stage1_source"):
        clf.fit(X, np.repeat([0, 1], 10))


def test_view_budget_constraint_is_enforced():
    clf = TabPFNAnchorClassifier(n_anchor=50, n_residual=100, n_stage2=999)
    X, y = np.random.default_rng(0).normal(size=(40, 200)), np.repeat([0, 1], 20)
    with pytest.raises(ValueError, match="n_anchor"):
        clf.fit(X, y)


def test_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        TabPFNAnchorClassifier().predict(np.zeros((2, 10)))


def test_meta_kwargs_reach_the_meta_learner():
    """meta_kwargs must resolve like every other setting.

    Reading the flat attribute instead made the override silent: the recorded
    configuration reported the requested C while the fitted meta-learner used
    the default.
    """
    clf = TabPFNAnchorClassifier(meta_kwargs={"C": 0.25})
    # _resolve() is what fit() runs to fold the kwargs groups into flat
    # settings, and _cfg is the single read path used everywhere afterwards.
    clf._resolved = clf._resolve()
    assert clf._resolved["meta_C"] == 0.25
    assert clf._cfg("meta_C") == 0.25
    assert clf.meta_C == 1.0, "the constructor argument itself must not change"


def test_missing_ranking_attributes_explain_themselves():
    """A bare AttributeError here is the commonest way to be confused.

    `mode="auto"` hands a narrow input straight to TabPFN and builds no
    ranking, so `anchor_features_` genuinely does not exist. The message has to
    say that, and `hasattr` must still answer False rather than raise.
    """
    clf = TabPFNAnchorClassifier()
    assert not hasattr(clf, "anchor_features_")
    with pytest.raises(AttributeError, match="not fitted"):
        clf.anchor_features_

    # A direct fit is recorded in execution_plan_; no backbone needed to check
    # that the guard reads it.
    clf.execution_plan_ = {"route": "direct_tabpfn", "n_features": 1000}
    with pytest.raises(AttributeError, match="mode='anchor'"):
        clf.anchor_features_
    with pytest.raises(AttributeError, match="mode='anchor'"):
        clf.feature_ranking_

    # The guard must not swallow ordinary missing attributes, nor break clone.
    with pytest.raises(AttributeError):
        clf.no_such_attribute_
    assert clone(TabPFNAnchorClassifier()).get_params()


def test_view_blocks_keep_the_budget_and_share_the_anchors():
    """The two invariants of Step 2.1, checked without a backbone.

    Every view must carry the complete anchor set and exactly n_stage2
    features. Both are what makes a difference between views attributable to
    the residual block rather than to an unequal budget.
    """
    from tabpfn_anchor.stacking import build_blocks
    rank_pool = np.arange(5000)
    anchor = np.arange(100)
    blocks = build_blocks(rank_pool, anchor, n_stage2=500, n_residual=400, k=9)

    assert len(blocks) == 9
    assert all(len(b) == 500 for b in blocks)
    assert all(len(np.unique(b)) == 500 for b in blocks)
    for b in blocks:
        assert set(anchor) <= set(b)
    shared = set(blocks[0])
    for b in blocks[1:]:
        shared &= set(b)
    assert shared == set(anchor)
    used = len(np.unique(np.concatenate(blocks)))
    assert used == 100 + 400 * 9 == 3700


def test_duplicate_feature_names_are_rejected():
    clf = TabPFNAnchorClassifier()
    X = pd.DataFrame(np.zeros((4, 3)), columns=["a", "b", "a"])
    with pytest.raises(ValueError, match="duplicate feature names"):
        clf._check_X(X, reset=True)


def test_reordered_columns_are_rejected_and_bare_arrays_warn():
    """Features are sliced by position, so a changed order reads other genes."""
    clf = TabPFNAnchorClassifier()
    clf._check_X(pd.DataFrame(np.zeros((4, 2)), columns=["a", "b"]), reset=True)
    clf.n_features_in_ = 2

    with pytest.raises(ValueError, match="feature names at predict"):
        clf._check_X(pd.DataFrame(np.zeros((4, 2)), columns=["b", "a"]), reset=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf._check_X(np.zeros((4, 2)), reset=False)
    assert any("column order cannot be checked" in str(w.message) for w in caught)


# ------------------------------------------------------------ real-data example
def test_synthetic_fixture_is_deterministic_and_wide_enough(toy):
    """The fitting tests need a ranking, so the input must clear the threshold."""
    X, y = toy
    X2, y2 = synthetic()
    assert X.equals(X2) and y.equals(y2)
    assert X.shape[1] > TabPFNAnchorClassifier.DIRECT_THRESHOLD
    assert set(y) == {"Class_A", "Class_B", "Class_C"}
    assert all(c.startswith("feature_") for c in X.columns)


README = Path(__file__).resolve().parents[1] / "README.md"
TOY = Path(__file__).resolve().parents[1] / "tabpfn_anchor/demo/sklearn_toy.py"


def _readme_example() -> str:
    """The first python block under the pasteable-example heading."""
    body = README.read_text(encoding="utf-8").split(
        "### A complete example you can paste and run", 1)[1]
    return body.split("```python", 1)[1].split("```", 1)[0].strip("\n")


def _toy_example() -> str:
    """The marked region of sklearn_toy.main(), dedented by its indent."""
    lines = TOY.read_text(encoding="utf-8").splitlines()
    a = next(i for i, l in enumerate(lines) if "README EXAMPLE BEGIN" in l)
    b = next(i for i, l in enumerate(lines) if "README EXAMPLE END" in l)
    return textwrap.dedent("\n".join(lines[a + 1:b])).strip("\n")


def test_the_toy_script_is_the_readme_example():
    """The installation check and the documented example must be one thing.

    Two copies of the same snippet drift, and then the page tells you to run
    something the file does not do. Comparing them here is what lets the file
    say "identical to README.md" and mean it.
    """
    if not README.is_file():          # installed copy ships no README
        pytest.skip("README.md is not present next to the package")
    assert _toy_example() == _readme_example()


def test_release_pin_is_the_one_the_paper_used():
    """The publication page also links a newer bundle; that one is not it."""
    from tabpfn_anchor.demo.download_tcga import SOURCE
    assert SOURCE["release"] == "v12_20210228"
    assert SOURCE["uuid"] == "752b364a-601e-4006-9ef6-630b705b1d0e"
    assert SOURCE["md5"] == "5d052a54effa27fdc76bde8c73b1c0b3"
    assert SOURCE["uuid"] in SOURCE["url"]


def test_extract_takes_only_the_named_members(tmp_path):
    """Extraction matches on basename and writes only inside the output dir.

    A tar member may carry any path, including `../`. Nothing here is trusted:
    the basename is compared against the two expected names and the output
    path is built locally.
    """
    from tabpfn_anchor.demo.download_tcga import extract, SOURCE
    rel = SOURCE["release"]
    archive = tmp_path / "bundle.tar.gz"
    payload = {
        f"TMP/{'UCEC'}_{rel}.tsv": b"UCEC\tLabels\n",
        f"TMP/UCEC_CVfolds_5FOLD_{rel}.tsv": b"UCEC\tLabels\n",
        "../../evil.tsv": b"nope\n",
        f"TMP/BRCA_{rel}.tsv": b"other\n",
    }
    with tarfile.open(archive, "w:gz") as tf:
        for name, data in payload.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    out = tmp_path / "source"
    found = extract(archive, "UCEC", out)
    assert set(found) == {"matrix", "cv"}
    assert sorted(p.name for p in out.iterdir()) == [
        f"UCEC_CVfolds_5FOLD_{rel}.tsv", f"UCEC_{rel}.tsv"]
    assert not (tmp_path.parent / "evil.tsv").exists()


def test_tcga_example_splits_on_the_released_assignment(tmp_path):
    """The example must read the distributed fold column, not resample."""
    from tabpfn_anchor.demo import tcga_example as te

    X = pd.DataFrame(np.arange(40, dtype=float).reshape(10, 4),
                     index=[f"s{i}" for i in range(10)],
                     columns=[f"{te.GEXP_PREFIX}G{j}" for j in range(4)])
    y = pd.Series(list("AABBAABBAB"), index=X.index)
    cv = pd.DataFrame({"R1:F1": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]}, index=X.index)

    Xtr, Xte, ytr, yte = te.split(X, y, cv, "R1:F1")
    assert len(Xtr) == 6 and len(Xte) == 4
    assert list(Xte.index) == ["s6", "s7", "s8", "s9"]
    assert not set(Xtr.index) & set(Xte.index)

    with pytest.raises(SystemExit, match="R9:F9"):
        te.split(X, y, cv, "R9:F9")


# ------------------------------------------------------------------ fitting
#: Lowest tabpfn release that can load the TabPFN-3 checkpoint. Must match the
#: `tabpfn>=` floor in pyproject.toml. An older package finds the file and then
#: fails inside its own loader with `KeyError: 'tabpfn_v3'`, so the version is
#: part of "is the backbone available", not a separate concern.
MIN_TABPFN_MAJOR = 8


def _checkpoint_available() -> bool:
    """True only if *the* checkpoint this project uses is present and loadable.

    Three things have to hold, and each has been observed to fail on its own:

    - the tabpfn package imports;
    - it is new enough to know the TabPFN-3 architecture;
    - the exact file named by the published configuration is in the cache.

    Any .ckpt in the cache is not enough: a machine holding only a TabPFN-2
    checkpoint would run these tests against a different backbone. Checking the
    exact file also keeps the tests from triggering the first-use login and a
    203 MB download in CI.
    """
    try:
        import tabpfn
        major = int(str(tabpfn.__version__).split(".")[0])
    except Exception:
        return False
    if major < MIN_TABPFN_MAJOR:
        return False
    import os
    import yaml
    name = yaml.safe_load(PAPER_YAML.read_text())["tabpfn_model_path"]
    cache = os.environ.get("TABPFN_MODEL_CACHE_DIR") or str(
        Path.home() / ".cache" / "tabpfn")
    return (Path(cache) / name).is_file()


requires_backbone = pytest.mark.skipif(
    not _checkpoint_available(),
    reason=f"needs tabpfn >= {MIN_TABPFN_MAJOR} and the exact checkpoint named "
           "in configs/paper (downloading it requires login and licence "
           "acceptance)")


@requires_backbone
def test_fit_predict_contract(toy):
    X, y = toy
    clf = TabPFNAnchorClassifier(**FAST).fit(X, y)

    assert list(clf.classes_) == sorted(y.unique())
    assert clf.n_features_in_ == X.shape[1]
    assert list(clf.feature_names_in_) == list(X.columns)

    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), len(clf.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(clf.predict(X)) <= set(clf.classes_)
    assert 0.0 <= clf.score(X, y) <= 1.0


@requires_backbone
def test_fitted_selection_attributes(toy):
    X, y = toy
    clf = TabPFNAnchorClassifier(**FAST).fit(X, y)

    anchors = clf.anchor_features_
    assert len(anchors) == FAST["n_anchor"]
    assert set(anchors["feature"]) <= set(X.columns)

    ranking = clf.feature_ranking_
    assert ranking["rank"].is_monotonic_increasing
    assert ranking.loc[ranking.is_anchor == 1, "feature"].tolist() != []
    # every anchor must appear in the ranking it was drawn from
    assert set(anchors["feature"]) <= set(ranking["feature"])

    assert clf.effective_n_views_ <= FAST["stack_size"]
    assert clf.n_genes_used_ <= X.shape[1]


@requires_backbone
def test_meta_C_override_changes_the_fitted_meta_learner(toy):
    """The override must reach the model, not only the recorded configuration."""
    X, y = toy
    a = TabPFNAnchorClassifier(**FAST).fit(X, y)
    b = TabPFNAnchorClassifier(**dict(FAST, meta_kwargs={"C": 0.01})).fit(X, y)
    assert a.meta_.C == 1.0 and b.meta_.C == 0.01
    assert not np.allclose(a.meta_.coef_, b.meta_.coef_)
