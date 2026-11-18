#!/usr/bin/env python3
"""Retain periodic and latest numbered checkpoints; preview unless --apply.

Only direct, regular files named exactly PREFIX_epoch<DIGITS>.pt are eligible.
Rolling, best, temporary, unrelated files and symlinks are never removed.
The newest epoch must be stable across two observations and at least 1 GB.
This checks filesystem stability, not checkpoint deserialization or integrity.
Use with atomically published checkpoints, as produced by scripts/utils.py.

Example:
    python3 scripts/tools/cleanup_checkpoints.py --out /absolute/experiment \
        --prefix model_name --keep-every 100
Add --apply after inspecting the preview. Deletion is permanent.
"""

import argparse
import math
import re
import stat
import sys
import time
from pathlib import Path


def fingerprint(path):
    """Do not follow symlinks; detect replacement as well as in-place writes."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Not a regular checkpoint file: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def numbered_checkpoints(out, prefix):
    pattern = re.compile(re.escape(prefix) + r"_epoch([0-9]+)\.pt")
    result = []
    for path in out.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        try:
            stamp = fingerprint(path)
        except (FileNotFoundError, ValueError):
            continue
        result.append((int(match[1]), path, stamp))
    return sorted(result, key=lambda item: (item[0], item[1].name))


def cleanup(out, prefix, *, keep_every=100, min_bytes=1_000_000_000,
            stability_seconds=5.0, apply=False):
    """Return removed paths. No numbered checkpoints is a successful no-op."""
    if not prefix or prefix in (".", "..") or "/" in prefix or "\\" in prefix:
        raise ValueError("--prefix must be a nonempty filename prefix, without paths")
    if keep_every < 1 or min_bytes < 1:
        raise ValueError("--keep-every and --min-bytes must be positive")
    if not math.isfinite(stability_seconds) or stability_seconds <= 0:
        raise ValueError("--stability-seconds must be finite and positive")
    out = Path(out).resolve(strict=True)
    if not out.is_dir() or out == Path(out.anchor) or out == Path.home().resolve():
        raise ValueError("--out must be a specific experiment directory")
    entries = numbered_checkpoints(out, prefix)
    if not entries:
        print(f"NO_NUMBERED_CHECKPOINTS:{out}")
        return []

    latest = entries[-1][0]
    # Preserve all aliases of the newest epoch, e.g. epoch5 and epoch005.
    newest = [entry for entry in entries if entry[0] == latest]
    candidates = [entry for entry in entries if entry[0] != latest and entry[0] % keep_every]
    print(f"MODE:{'APPLY' if apply else 'DRY_RUN'} OUT:{out}", flush=True)
    time.sleep(stability_seconds)

    def verify_newest():
        for _, path, before in newest:
            if fingerprint(path) != before:
                raise ValueError(f"Newest checkpoint changed; cleanup refused: {path}")
            if before[2] < min_bytes:
                raise ValueError(f"Newest checkpoint is smaller than {min_bytes} bytes: {path}")

    verify_newest()
    removed = []
    for _, path, before in candidates:
        verify_newest()
        try:
            if fingerprint(path) != before:
                print(f"SKIPPED_CHANGED:{path.name}", flush=True)
                continue
        except (FileNotFoundError, ValueError):
            print(f"SKIPPED_CHANGED:{path.name}", flush=True)
            continue
        if apply:
            path.unlink()
            removed.append(path)
            print(f"REMOVED:{path.name}", flush=True)
        else:
            print(f"WOULD_REMOVE:{path.name}", flush=True)

    print(f"LATEST:{latest}:{newest[-1][2][2]}")
    print(f"REMOVED_COUNT:{len(removed)}")
    # Include rolling/best checkpoints in the inventory, without touching them.
    for path in sorted(out.iterdir(), key=lambda item: item.name):
        if path.suffix == ".pt":
            try:
                size = fingerprint(path)[2]
            except (FileNotFoundError, ValueError):
                continue
            print(f"{path.name}|{size}")
    return removed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="Exact experiment directory")
    parser.add_argument("--prefix", required=True, help="Model filename without .pt or _epochN")
    parser.add_argument("--keep-every", type=int, default=100)
    parser.add_argument("--min-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--stability-seconds", type=float, default=5.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Permanently delete eligible checkpoints")
    mode.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    args = parser.parse_args(argv)
    try:
        cleanup(args.out, args.prefix, keep_every=args.keep_every, min_bytes=args.min_bytes,
                stability_seconds=args.stability_seconds, apply=args.apply)
    except (OSError, ValueError) as exc:
        print(f"ERROR:{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
