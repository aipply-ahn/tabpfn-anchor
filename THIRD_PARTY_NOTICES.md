# Third-party notices

The MIT licence in `LICENSE` covers **the code in this repository only**. Two
things this project depends on are covered by other terms, and neither is
distributed here.

## TabPFN backbone weights — non-commercial licence

The predictive backbone is a TabPFN checkpoint published by Prior Labs. The
package code and the TabPFN-2 weights are under the Prior Labs License (Apache
2.0 with an added attribution requirement), but **the TabPFN-2.5, TabPFN-2.6
and TabPFN-3 weights are released under non-commercial licences**, TabPFN-3
under its own. This project's default backbone is the TabPFN-3 multiclass
checkpoint, so:

> Commercial or production use of this project as configured is governed by the
> TabPFN-3 model licence, not by the MIT licence of this code. Read that
> licence before deploying.

No checkpoint is redistributed here. The `tabpfn` package downloads it on first
use, which requires network access and licence acceptance:

- On first use TabPFN opens a browser window to log in with Prior Labs and
  accept the licence terms.
- Headless machines need the licence accepted through the web interface and a
  `TABPFN_TOKEN` environment variable set before running.
- The TabPFN-3 multiclass checkpoint is about 203 MB.

See <https://github.com/PriorLabs/TabPFN> for the current terms; the summary
above may lag behind them.

## Cancer cohort data — not distributed, terms set by the distributor

No cohort data, no cross-validation splits, and no statistic derived from them
is distributed here. The source archive is obtained by the user from the NCI
Genomic Data Commons, whose access and analysis policies govern its use. See
[DATA.md](DATA.md) for the release, the checksum, and the links.

## Python dependencies

numpy, pandas, scipy, scikit-learn, shapiq and PyYAML are installed from PyPI
under their own licences and are not vendored here.
