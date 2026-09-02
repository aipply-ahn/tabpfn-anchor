# -*- coding: utf-8 -*-
"""Cache for intermediate stage results.

Step 1.1 as originally implemented (logistic regression plus permutation
importance) was the most expensive part of the pipeline: P is around 20,000 and
it repeats 50 times. For the same (data, fold, settings) there is no reason to
compute it twice.

The point is that **the key contains the whole configuration**. Look a cache
entry up by file name alone and a changed setting silently reuses the old
result, and a table built that way does not even reveal that it is wrong. The
key here is a hash of (a fingerprint of the input arrays + the stage name + the
parameters that actually affect that stage).

    cache = StageCache("cache/", enabled=True)
    rank, imp = cache.get_or_compute(
        "stage1_anova_f", lambda: stage1_anova_f(X, y, n_keep=2000),
        X=X, y=y, params=dict(source="anova_f", code_version=...))
"""

from __future__ import annotations

import hashlib
import os
import json
import time
from pathlib import Path

import numpy as np

__all__ = ["StageCache", "array_fingerprint"]


def array_fingerprint(*arrays):
    """SHA-256 fingerprint of the **entire** contents of the arrays.

    An earlier version sampled 20,000 values and appended the sum and standard
    deviation. Under that scheme a change to an unsampled value can leave the
    fingerprint unchanged -- a silent stale hit. Reading a wrong value out of
    the cache costs far more than the seconds it saves. The BRCA training
    matrix is 796 x 20,531 float32 = 65 MB and hashing it takes about 0.05 s,
    which is nothing in front of a Step 1 measured in minutes.

    Hashing proceeds in chunks so that large arrays cost no extra memory.
    """
    h = hashlib.sha256()
    CHUNK = 1 << 22                       # 4 MB
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        flat = a.reshape(-1)
        v = flat.view(np.uint8) if flat.dtype != object else None
        if v is None:
            h.update(repr(flat.tolist()).encode())
            continue
        for i in range(0, v.size, CHUNK):
            h.update(v[i:i + CHUNK].tobytes())
    return h.hexdigest()[:32]


class StageCache:
    """Stores each stage's output as an .npz and retrieves it by key."""

    def __init__(self, root=None, enabled=True, verbose=False):
        self.enabled = bool(enabled) and root is not None
        self.root = Path(root) if root is not None else None
        self.verbose = verbose
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # -- key -------------------------------------------------------------
    def key(self, stage, params, **arrays):
        payload = {
            "stage": stage,
            "params": {k: _jsonable(v) for k, v in sorted(params.items())},
            "data": array_fingerprint(*arrays.values()),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:24]
        return f"{stage}__{digest}", blob

    # -- lookup and store --------------------------------------------------
    def get_or_compute(self, stage, fn, params, **arrays):
        """Return the cached value, or run fn(), store it and return that.

        fn must return a tuple of ndarrays, for example (rank, importance).
        """
        if not self.enabled:
            return fn()

        name, blob = self.key(stage, params, **arrays)
        npz = self.root / f"{name}.npz"

        if npz.exists():
            try:
                with np.load(npz, allow_pickle=False) as z:
                    n = int(z["_n"])
                    out = tuple(z[f"a{i}"] for i in range(n))
                if self.verbose:
                    print(f"  [cache] hit  {stage}  {npz.name}", flush=True)
                return out if len(out) > 1 else out[0]
            except Exception as e:
                if self.verbose:
                    print(f"  [cache] corrupt, recomputing {npz.name}: {e}", flush=True)
                npz.unlink(missing_ok=True)

        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0

        out = result if isinstance(result, tuple) else (result,)
        # Write atomically. Several splits can share one cache directory, and
        # a half-written npz read by the next process either falls through as
        # "corrupt, recomputing" or is read partially. Write to a temporary
        # file and swap it in with os.replace, which is atomic within a
        # filesystem.
        # Pass the **open handle**, not the file name. np.savez_compressed
        # appends .npz on its own whenever the name does not already end in
        # .npz, so "stage.npz.tmp123" becomes "stage.npz.tmp123.npz" and the
        # os.replace on the next line dies with FileNotFoundError.
        tmp = npz.with_name(f"{npz.name}.tmp{os.getpid()}")
        try:
            with open(tmp, "wb") as fh:
                np.savez_compressed(
                    fh, _n=len(out),
                    **{f"a{i}": np.asarray(v) for i, v in enumerate(out)})
            os.replace(tmp, npz)
        finally:
            Path(tmp).unlink(missing_ok=True)
        # Record what the entry was built from, for debugging and auditing.
        jtmp = self.root / f"{name}.json.tmp{os.getpid()}"
        jtmp.write_text(blob, encoding="utf-8")
        os.replace(jtmp, self.root / f"{name}.json")
        if self.verbose:
            print(f"  [cache] miss {stage}  {elapsed:.1f}s -> {npz.name}", flush=True)
        return result


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return array_fingerprint(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in sorted(v.items())}
    return v
