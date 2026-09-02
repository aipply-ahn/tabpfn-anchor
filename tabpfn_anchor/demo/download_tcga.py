#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download and verify the official source archive, then extract one cohort.

    tabpfn-anchor download-tcga --cohort UCEC --data-dir ./data
    python3 -m tabpfn_anchor.demo.download_tcga --data-dir ./data --cohort UCEC

Nothing downloaded or extracted here is redistributed by this repository; the
files land in a local directory that .gitignore excludes.  See DATA.md.

RELEASE
-------
The paper uses release **v12_20210228**.  The publication page also links a
newer bundle, `TMP_20230209.tar.gz`
(UUID 5116e86f-7646-4b7b-9d6e-dafddf2cc0f3), which is *not* the release the
paper used.  This module pins the v12 UUID so the two cannot be confused.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

#: Socket timeout for the download. Without one a stalled connection hangs
#: forever rather than failing, and the archive is 609 MB.
TIMEOUT_S = 60

#: Official manifest entry for the release the paper used. Pinned by UUID, not
#: scraped from a page: the UUID is permanent, page layout is not.
SOURCE = {
    "release": "v12_20210228",
    "filename": "TMP_v12_20210228.tar.gz",
    "uuid": "752b364a-601e-4006-9ef6-630b705b1d0e",
    "url": ("https://api.gdc.cancer.gov/data/"
            "752b364a-601e-4006-9ef6-630b705b1d0e"),
    "size_bytes": 609_027_948,
    "md5": "5d052a54effa27fdc76bde8c73b1c0b3",
    "publication_page": ("https://gdc.cancer.gov/about-data/publications/"
                         "CCG-TMP-2022"),
}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path) -> bool:
    """Size first (cheap), then MD5. Both must match the official manifest."""
    if not path.exists():
        return False
    size = path.stat().st_size
    if size != SOURCE["size_bytes"]:
        print(f"  size mismatch: {size:,} != {SOURCE['size_bytes']:,}")
        return False
    print("  checking md5 ...", end=" ", flush=True)
    got = md5(path)
    ok = got == SOURCE["md5"]
    print("ok" if ok else f"MISMATCH ({got})")
    return ok


def download(dest: Path) -> Path:
    """Fetch to a .part file and rename only after verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if verify(dest):
        print(f"  already present and verified: {dest}")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {SOURCE['size_bytes'] / 1e6:.0f} MB from {SOURCE['url']}")
    with urllib.request.urlopen(SOURCE["url"], timeout=TIMEOUT_S) as r, open(part, "wb") as out:
        done = 0
        while chunk := r.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:8.0f} / {SOURCE['size_bytes'] / 1e6:.0f} MB",
                  end="", flush=True)
    print()
    if not verify(part):
        part.unlink(missing_ok=True)
        raise SystemExit(
            "Downloaded archive failed verification and was removed.\n"
            f"Retry, or fetch it manually from {SOURCE['publication_page']}")
    part.rename(dest)
    return dest


def extract(archive: Path, cohort: str, out_dir: Path) -> dict[str, Path]:
    """Pull just the one cohort's matrix and CV assignment out of the bundle."""
    wanted = {
        "matrix": f"{cohort}_{SOURCE['release']}.tsv",
        "cv": f"{cohort}_CVfolds_5FOLD_{SOURCE['release']}.tsv",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = Path(member.name).name
            for key, target in wanted.items():
                if name == target:
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    dst = out_dir / target
                    with open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)
                    found[key] = dst
                    print(f"  extracted {target}  "
                          f"({dst.stat().st_size / 1e6:.1f} MB)")
    missing = sorted(set(wanted) - set(found))
    if "matrix" in missing:
        raise SystemExit(
            f"{wanted['matrix']} not found inside {archive}.\n"
            f"Check that the archive is release {SOURCE['release']}.")
    if "cv" in missing:
        raise SystemExit(
            f"{wanted['cv']} not found inside {archive}.\n"
            "The example evaluates on the official cross-validation splits "
            "distributed with the release, so this file is required.")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--cohort", default="UCEC")
    ap.add_argument("--archive", type=Path, default=None,
                    help="use an archive you already have instead of downloading")
    a = ap.parse_args()

    print(f"release    {SOURCE['release']}   ({SOURCE['filename']})")
    print(f"uuid       {SOURCE['uuid']}")
    archive = a.archive or (a.data_dir / "raw" / SOURCE["filename"])
    if a.archive:
        print(f"archive    {archive} (supplied)")
        if not verify(archive):
            raise SystemExit("The supplied archive does not match the manifest.")
    else:
        archive = download(archive)

    print(f"\nextracting {a.cohort} ...")
    found = extract(archive, a.cohort, a.data_dir / "source")
    print(f"\nSource files are local only; .gitignore keeps them out of git.\n"
          f"Next:  tabpfn-anchor tcga-example --cohort {a.cohort} "
          f"--data-dir {a.data_dir}")


if __name__ == "__main__":
    main()
