# Installation

## Requirements

| Item | Requirement |
|---|---|
| Python | 3.12 or later (needed by `shapiq >= 1.6`) |
| GPU | NVIDIA GPU recommended; CPU works but is very slow |
| Disk | ~200 MB for the TabPFN-3 checkpoint |

## Install

```bash
git clone https://github.com/aipply-ahn/tabpfn-anchor.git
cd tabpfn-anchor
pip install -e .            # add ".[test]" to also install the test tools
```

## TabPFN-3 checkpoint

1. Run any example. The `tabpfn` package opens a browser to sign in with Prior Labs.
2. Accept the TabPFN-3 model licence.
3. The checkpoint (~203 MB) downloads and is cached.

**Headless machine (no browser):** accept the licence on the web first, then set
`TABPFN_TOKEN` in the environment.

**Licence:** the TabPFN-3 weights are non-commercial.
See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and
[github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN).

## Troubleshooting

| Problem | Fix |
|---|---|
| Very slow on a multi-GPU machine | Pass one device: `TabPFNAnchorClassifier(device="cuda:0")` |
| `anchor_features_` does not exist | Input has ≤ 2,000 features, so native TabPFN-3 ran. Use `mode="anchor"` |
| Error with `mode="anchor"` on narrow input | Nine views need ≥ 3,700 features. Lower `stack_size` |
| Error with `preprocessing="rnaseq"` | Input has negative values. Use `"microarray"` or `"none"` |
| Python version error | Use Python 3.12 or later |

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

Tests that need the checkpoint skip themselves when it is absent.
