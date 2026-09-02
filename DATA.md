# Data

**No data is distributed with this repository.** Neither the cancer cohort
files nor anything derived from them is included here — not the expression
matrices, not the cross-validation splits, not per-subtype summary statistics,
and not synthetic samples built from such statistics.

The synthetic examples are generated locally from no input at all, and the real
example reads a file you obtain yourself from the original distributor and
never leaves your machine. Provenance therefore stays with the distributor, and
this repository stays a code repository.

---

## Three paths, kept separate

| Path | Data | Needs a download? | Purpose |
|---|---|---|---|
| `demo` | synthetic, 300 x 1,000 | no | verify the installation and the API |
| `demo --wide` | synthetic, 500 x 10,000 | no | the published view construction at `p >> n` |
| `tcga-example` | one real cohort, one official split | yes, once | the method on the real distribution |

None is a substitute for another, and **none reproduces the paper's results**.
The first two are synthetic; the third is a single outer split of a single
cohort, whereas the reported numbers are means over twelve cohorts and many
splits.

An earlier release had a fourth path that derived synthetic samples from
per-subtype summary statistics of a real cohort. **It has been removed.**
`demo --wide` covers the high-dimensional check without any cohort statistic,
and `tcga-example` covers the real distribution with the real data, so nothing
needed to sit in between. Removing it also removes the question of whether
aggregate statistics of a redistributed dataset may themselves be redistributed:
now nothing derived from a cohort leaves your machine.

---

## 1 and 2. Synthetic — generated locally, no inputs

```bash
tabpfn-anchor demo
tabpfn-anchor demo --wide
```

`tabpfn_anchor/demo/sklearn_toy.py` is a single call to
`sklearn.datasets.make_classification` under a fixed seed. It reads no file and
contains no constant taken from any cohort.

The data contain no patient observations, no gene identifiers, no subtype
labels, and no statistic derived from any cohort. The columns are named
`feature_00001` and the values are not on an expression scale. What they
reproduce is the shape of the task and nothing more:

- far more features than samples
- a small informative set (20 for `quick`, 100 for `wide`)
- many features that are random linear combinations of those, so that a
  marginal screen cannot separate an informative feature from its copies
- imbalanced classes

`wide` is sized so that the published view construction fits: 100 anchors plus
nine disjoint blocks of 400 is 3,700 distinct features out of 10,000.

## 3. One official split of one real cohort

### Getting the source file

The cohort matrices used in the paper come from the compact-feature-set
resource of the TCGA Genomic Data Analysis Network:

> Ellrott et al. (2025), companion data for *Classification of non-TCGA cancer
> samples to TCGA molecular subtypes*
> Publication page: <https://gdc.cancer.gov/about-data/publications/CCG-TMP-2022>

The paper uses release **v12_20210228**, distributed as one archive:

| | |
|---|---|
| File | `TMP_v12_20210228.tar.gz` |
| GDC UUID | `752b364a-601e-4006-9ef6-630b705b1d0e` |
| URL | <https://api.gdc.cancer.gov/data/752b364a-601e-4006-9ef6-630b705b1d0e> |
| Size | 609,027,948 bytes |
| MD5 | `5d052a54effa27fdc76bde8c73b1c0b3` |

> **Do not use the newer bundle.** The publication page links
> `TMP_20230209.tar.gz` (UUID `5116e86f-7646-4b7b-9d6e-dafddf2cc0f3`) as its
> source-data matrix. That is a later release than the one the paper used.
> `download-tcga` pins the v12 UUID so the two cannot be confused.

```bash
tabpfn-anchor download-tcga --cohort UCEC --data-dir ./data
```

This downloads the archive, checks its size and MD5 against the manifest above,
and extracts only two files for the cohort you ask for:

```
data/source/UCEC_v12_20210228.tsv                 the expression matrix
data/source/UCEC_CVfolds_5FOLD_v12_20210228.tsv   the cross-validation assignment
```

Both are excluded by `.gitignore`. Pass `--archive` if you already hold the
bundle.

In the matrix, rows are samples; the first column is the sample identifier, the
second is `Labels` (the molecular subtype), and the remaining columns are
features named `N:GEXP::SYMBOL:ENTREZ:` alongside other modalities. Only the
`N:GEXP:` columns are used.

### Running one split

```bash
tabpfn-anchor tcga-example --cohort UCEC --repeat 1 --fold 1 --data-dir ./data
```

**The split is not ours.** The release ships a 100 x 5 cross-validation
assignment, and `--repeat`/`--fold` name one column of it (`R1:F1`); a value of
0 marks a training sample and 1 a test sample. Reading the distributed column
rather than calling `train_test_split` is what makes a comparison against
another method on the same column meaningful.

Selection is fitted on the training partition alone: the ranking, the anchor
set and the view blocks are all produced inside `fit(X_train, y_train)`, and
the test partition is touched only by `predict`. The one transformation applied
outside the estimator is `log2(x + 1)`, a fixed monotone map with no fitted
parameter; microarray cohorts are already on a log scale and are left alone.

Results are written to `results/<cohort>_R01_F01/`: the anchor set, the full
ranking, the metrics, and the resolved configuration. That directory is
excluded by `.gitignore`.

## What this repository does not include

The benchmark harness is not here: the code that ran twelve cohorts across
repeated cross-validation, aggregated macro-F1, applied the multiple-comparison
correction, and produced the paper's tables and figures. There is no single
command that reproduces the reported numbers.

What is here is the method, the exact configuration it was run under
(`configs/paper/tabpfn_anchor.yaml`, which is also the estimator's default),
and the provenance of the data. The splits used in the paper are the first
three repeats for its 15-split analyses and the first ten for its 50-split
analyses, taken from the assignment file above.

---

## Licensing and terms

The cohort data are TCGA open-access molecular data redistributed by the NCI
Genomic Data Commons. Consult the distributor for the terms that apply to your
use:

- GDC data access policies: <https://gdc.cancer.gov/access-data/data-access-policies>
- GDC data analysis policies: <https://gdc.cancer.gov/about-data/data-analysis-policies>

Cite the resource above if you use it, in addition to citing this work.
