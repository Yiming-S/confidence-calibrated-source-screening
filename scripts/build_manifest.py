#!/usr/bin/env python3
"""Rebuild the compact-result manifest and checksum list."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXCLUDED = {RESULTS / "MANIFEST.csv", RESULTS / "SHA256SUMS"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in RESULTS.rglob("*")
        if path.is_file() and path not in EXCLUDED
    )
    rows = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in files
    ]

    with (RESULTS / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["path", "bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"{row['sha256']}  {row['path']}" for row in rows]
    (RESULTS / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"indexed {len(rows)} result files")


if __name__ == "__main__":
    main()
