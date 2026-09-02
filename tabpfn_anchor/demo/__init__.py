# -*- coding: utf-8 -*-
"""Helpers for the real-data example.

    download_tcga   fetches and verifies the official release archive and takes
                    one cohort out of it. Nothing is redistributed here.
    tcga_example    fits and evaluates on one official cross-validation split of
                    that cohort.

**No data ships with this package**, and neither does any statistic derived
from a real cohort. Everything these modules write goes to a local directory
that .gitignore excludes. See DATA.md.

Earlier releases also carried synthetic demo data: first samples drawn from
per-subtype summary statistics of a real cohort, later a wrapper around
``sklearn.datasets.make_classification``. Both are gone. Anyone who wants a
synthetic matrix can call ``make_classification`` themselves in two lines, and
a module that only forwards arguments to it earns nothing while implying the
numbers mean something. README.md shows those two lines.
"""

from . import download_tcga, tcga_example

__all__ = ["download_tcga", "tcga_example"]
