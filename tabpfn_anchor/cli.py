# -*- coding: utf-8 -*-
"""Command line entry point. Three stages, in increasing order of realism.

    tabpfn-anchor demo                    300 x 1,000 synthetic   installation
    tabpfn-anchor demo --wide             500 x 10,000 synthetic  p >> n
    tabpfn-anchor download-tcga           fetch and verify the real release
    tabpfn-anchor tcga-example            one official split of one cohort

The first two generate their data with ``sklearn.datasets.make_classification``
and read nothing: they contain no patient observations, no gene identifiers and
no statistic derived from any cohort. The third runs on the data the paper
used, which you download yourself; nothing is redistributed by this repository.

None of the three reproduces the reported results. The first two run on
synthetic data, and the third is a single outer split, whereas the paper
reports means over twelve cohorts and many splits with a paired test across
them.

The demo presets are smaller than the published configuration so that the
commands finish. ``--paper-settings`` restores the published configuration,
which is far slower.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reduced settings so that a demo finishes quickly.
#
# Essentially all of the cost is Step 1.3, the OddSHAP attribution: the screen
# and the Random Forest refinement together take under a second even at
# transcriptome scale. Its cost is
#
#     (explained samples) x (classes explained) x (coalition budget)
#
# model evaluations, so the presets cut all three. `shap_class_mode="predicted"`
# explains only each sample's predicted class instead of every class, which the
# published configuration does not do -- these presets run a cheaper variant of
# the method, not the method itself.
#
# `oof_folds` is reduced for the same reason: each fold means another backbone
# instance, and on a machine where the checkpoint is not already in the page
# cache that load dominates Step 2.
PRESETS = {
    # mode="anchor" is forced here on purpose. The quick set has 1,000
    # features, below the threshold at which the default mode="auto" skips the
    # anchor decomposition and calls TabPFN directly -- which would demonstrate
    # nothing. The wider runs are above it, so auto takes the full cascade.
    "quick": dict(mode="anchor",
                  n_stage1=1000, n_stage2=150, n_anchor=50, n_residual=100,
                  stack_size=3, rf_n_estimators=300, oof_folds=2,
                  shap_budget=192, shap_max_samples=16,
                  shap_class_mode="predicted",
                  tabpfn_n_estimators=2, shap_tabpfn_n_estimators=2),
    # Same view construction as the paper -- 100 anchors, 400 residual genes per
    # view, K = 9, 3,700 distinct features -- with the attribution and backbone
    # budgets cut. The structure is the published one; the compute is not.
    "wide": dict(n_stage1=2000, n_stage2=500, n_anchor=100, n_residual=400,
                 stack_size=9, rf_n_estimators=500, oof_folds=3,
                 shap_budget=256, shap_max_samples=16,
                 shap_class_mode="predicted",
                 tabpfn_n_estimators=2, shap_tabpfn_n_estimators=2),
}

PAPER_YAML_NAME = "configs/paper/tabpfn_anchor.yaml"


def _paper_yaml() -> Path:
    """Locate the published configuration file.

    It lives beside the package in a source checkout, but a pip-installed copy
    has no configs/ directory, so the working directory is tried as well.
    """
    here = Path(__file__).resolve()
    for base in (here.parents[1], here.parents[2], Path.cwd()):
        candidate = base / PAPER_YAML_NAME
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"could not find {PAPER_YAML_NAME}. It ships with the source "
        "repository; run --paper-settings from a checkout, or pass the "
        "settings explicitly.")


def _paper_params() -> dict:
    """Read the published configuration from the YAML, not from the defaults.

    The defaults are asserted to equal this file by the test suite, but reading
    the file is what makes `--paper-settings` mean "the published
    configuration" rather than "whatever the defaults happen to be today".
    """
    import yaml
    return yaml.safe_load(_paper_yaml().read_text())


def _pick_device() -> str:
    """Choose one device explicitly rather than leaving the backbone on "auto".

    On a machine with several visible GPUs, "auto" was measured about fifty
    times slower than naming a single device (68 s against 1.4 s to fit one
    view model on 240 samples x 60 features). Pick the first GPU and say so.
    """
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n > 1:
                print(f"device        cuda:0 ({n} GPUs visible; naming one device "
                      "is far faster than 'auto'. Override with --device.)")
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _settings(a: argparse.Namespace, preset: str) -> dict:
    """The parameter dict for this run, plus the line that describes it."""
    if a.paper_settings:
        params = _paper_params()
        print(f"settings      published configuration from {_paper_yaml()} (slow)")
    else:
        params = dict(PRESETS[preset])
        print(f"settings      reduced '{preset}' preset: "
              f"{params['n_anchor']} anchors + {params['n_residual']} residual "
              f"genes per view, K={params['stack_size']}, cut attribution budget")
        print("              Verifies the method end to end. Does not reproduce "
              "the paper's results.")
    params["device"] = a.device or _pick_device()
    return params


def _license_note() -> None:
    print("\nThe backbone checkpoint is downloaded by the tabpfn package on first "
          "use.\nIt is licensed separately and not for commercial use; see "
          "THIRD_PARTY_NOTICES.md.")


def cmd_demo(a: argparse.Namespace) -> int:
    if a.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, force=True,
                            format="%(asctime)s %(message)s")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, classification_report

    from .demo.sklearn_toy import make_toy
    from .estimator import TabPFNAnchorClassifier

    preset = "wide" if a.wide else "quick"
    # The dataset is a fixed artefact, so --seed does not reach it: it controls
    # the train/test split and the estimator only. Otherwise every seed would
    # produce a different problem of a different difficulty, and the baseline
    # scores the preset is calibrated against would describe nothing.
    X, y, signal = make_toy(preset)
    print(f"data          synthetic ({preset}): {X.shape[0]} samples x "
          f"{X.shape[1]:,} features, {y.nunique()} classes")
    print(f"              sklearn.datasets.make_classification, "
          f"{len(signal['informative'])} informative + "
          f"{len(signal['redundant'])} correlated. No cohort data of any kind.")

    params = _settings(a, preset)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=a.seed)
    clf = TabPFNAnchorClassifier(random_state=a.seed, verbose=a.verbose, **params)
    print("\nfitting ...")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    print(f"\nmacro-F1      {f1_score(yte, pred, average='macro'):.3f}")
    print(f"views         {clf.effective_n_views_}   "
          f"distinct features used {clf.n_genes_used_:,}")
    anchors = set(clf.anchor_features_["feature"])
    # A redundant feature is a linear combination of the informative ones, so
    # picking it is not a miss; the two are counted together against what a
    # random draw of the same size would contain.
    planted = set(signal["informative"]) | set(signal["redundant"])
    chance = len(anchors) * len(planted) / X.shape[1]
    print(f"\nanchor set    {len(anchors)} features, of which "
          f"{len(anchors & planted)} carry planted signal "
          f"(a random draw of {len(anchors)} would contain about {chance:.0f})")
    print("full ranking in clf.feature_ranking_ ; anchors in clf.anchor_features_")
    if a.verbose:
        print("\n" + classification_report(yte, pred))
    _license_note()
    return 0


def cmd_download_tcga(a: argparse.Namespace) -> int:
    from .demo.download_tcga import SOURCE, download, extract, verify

    print(f"release       {SOURCE['release']}   ({SOURCE['filename']})")
    print(f"uuid          {SOURCE['uuid']}")
    archive = a.archive or (a.data_dir / "raw" / SOURCE["filename"])
    if a.archive:
        print(f"archive       {archive} (supplied)")
        if not verify(archive):
            raise SystemExit("The supplied archive does not match the manifest.")
    else:
        archive = download(archive)

    print(f"\nextracting {a.cohort} ...")
    extract(archive, a.cohort, a.data_dir / "source")
    print(f"\nSource files are local only; .gitignore keeps them out of git.\n"
          f"Next:  tabpfn-anchor tcga-example --cohort {a.cohort} "
          f"--data-dir {a.data_dir}")
    return 0


def cmd_tcga_example(a: argparse.Namespace) -> int:
    if a.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, force=True,
                            format="%(asctime)s %(message)s")
    from .demo.tcga_example import run

    params = _settings(a, "wide")
    out = a.out_dir or (Path("results") /
                        f"{a.cohort}_R{a.repeat:02d}_F{a.fold:02d}")
    run(cohort=a.cohort, data_dir=a.data_dir, repeat=a.repeat, fold=a.fold,
        params=params, out_dir=out, seed=a.seed, verbose=a.verbose)
    _license_note()
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--paper-settings", action="store_true",
                   help="use the published configuration instead of the demo preset")
    p.add_argument("--device", default=None, help='e.g. "cuda:0" or "cpu"')
    p.add_argument("--seed", type=int, default=42,
                   help="split and estimator seed; the demo dataset is fixed")
    p.add_argument("-v", "--verbose", action="store_true")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tabpfn-anchor", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="fit on synthetic data (no download needed)")
    d.add_argument("--wide", action="store_true",
                   help="500 x 10,000 instead of 300 x 1,000, with the "
                        "published view construction")
    _common(d)
    d.set_defaults(func=cmd_demo)

    g = sub.add_parser("download-tcga",
                       help="download and verify the official release archive, "
                            "then extract one cohort")
    g.add_argument("--cohort", default="UCEC")
    g.add_argument("--data-dir", type=Path, default=Path("data"))
    g.add_argument("--archive", type=Path, default=None,
                   help="use an archive you already have instead of downloading")
    g.set_defaults(func=cmd_download_tcga)

    t = sub.add_parser("tcga-example",
                       help="fit and evaluate on one official split of one cohort")
    t.add_argument("--cohort", default="UCEC")
    t.add_argument("--data-dir", type=Path, default=Path("data"))
    t.add_argument("--repeat", type=int, default=1, help="R in the R:F column")
    t.add_argument("--fold", type=int, default=1, help="F in the R:F column")
    t.add_argument("--out-dir", type=Path, default=None)
    _common(t)
    t.set_defaults(func=cmd_tcga_example)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
