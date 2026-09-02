# -*- coding: utf-8 -*-
"""Demo data for the three example stages.

    sklearn_toy     fully synthetic, from sklearn.datasets.make_classification.
                    Two sizes: a quick installation check and a wide p >> n run.
    download_tcga   fetches and verifies the official release archive and takes
                    one cohort out of it. Nothing is redistributed here.
    tcga_example    fits and evaluates on one official cross-validation split of
                    that cohort.

**No data ships with this package**, and neither does any statistic derived
from a real cohort. Everything these modules write goes to a local directory
that .gitignore excludes. See DATA.md.

An earlier release also derived synthetic samples from per-subtype summary
statistics of a real cohort. That path is gone: the synthetic stage is now
plainly synthetic, and the real stage uses the real data, so nothing needs to
sit in between.
"""

from . import download_tcga, sklearn_toy, tcga_example

__all__ = ["download_tcga", "sklearn_toy", "tcga_example"]
