# -*- coding: utf-8 -*-
"""Helpers for the examples.

    sklearn_toy     the README example as a runnable script, for checking an
                    installation. It generates its data in place with
                    sklearn.datasets.make_classification and reads nothing.
    download_tcga   fetches and verifies the official release archive and takes
                    one cohort out of it. Nothing is redistributed here.
    tcga_example    fits and evaluates on one official cross-validation split of
                    that cohort.

**No data ships with this package**, and neither does any statistic derived
from a real cohort. Everything these modules write goes to a local directory
that .gitignore excludes. See DATA.md.

An earlier release drew synthetic samples from per-subtype summary statistics
of a real cohort. That path is gone: nothing derived from a cohort leaves your
machine now.

``sklearn_toy`` is deliberately not a data generator with options. It holds one
``make_classification`` call with its arguments in plain sight, identical to the
block in README.md, and a test compares the two so they cannot drift apart. An
earlier version wrapped that call behind presets, which forwarded its arguments
and made chosen numbers look meaningful.
"""

from . import download_tcga, sklearn_toy, tcga_example

__all__ = ["download_tcga", "sklearn_toy", "tcga_example"]
