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

The checkpoint is licence-gated: it does not download until you accept the
Prior Labs licence. Choose one:

| Option | Steps |
|---|---|
| Automatic | Run any example → sign in with Prior Labs in the browser → accept the licence → the file downloads |
| Server, no browser | Accept the licence on the web → `export TABPFN_TOKEN=<your-token>` → run |
| Manual | Download `tabpfn-v3-classifier-v3_20260417_multiclass.ckpt` from [huggingface.co/Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3) → `tabpfn-anchor checkpoint --install /path/to/file.ckpt` |

```bash
tabpfn-anchor checkpoint      # expected file name, location, present or missing
```

- **Location:** `$TABPFN_MODEL_CACHE_DIR` if set, otherwise `~/.cache/tabpfn/` (all platforms).
- **Shared copy:** point `TABPFN_MODEL_CACHE_DIR` at a folder that already holds the file.
- **Any path:** `TabPFNAnchorClassifier(tabpfn_model_path="/full/path/to/file.ckpt")`.
- **Licence:** non-commercial. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and
  [github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN).

## Check the installation

```bash
tabpfn-anchor check              # package + checkpoint + a short TabPFN-3 run
tabpfn-anchor check --skip-run   # files only, no GPU needed
```

| Line | Meaning |
|---|---|
| `[1/3] Package installation OK` | `tabpfn-anchor` and `tabpfn` are importable |
| `[2/3] Checkpoint installation OK` | the TabPFN-3 checkpoint is in place (`MISSING` → install it first) |
| `[3/3] Backbone test OK` | TabPFN-3 loaded and predicted on a small synthetic set |

## Troubleshooting

| Problem | Fix |
|---|---|
| Checkpoint not found / licence error | Run `tabpfn-anchor checkpoint`; install the file manually (above) |
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
