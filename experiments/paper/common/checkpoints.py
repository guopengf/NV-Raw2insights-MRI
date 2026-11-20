"""Checkpoint retention: latest.pt each epoch, rolling and permanent epoch_NNNN.pt copies, fallback on resume."""

from __future__ import annotations

from pathlib import Path
import re

import torch

REQUIRED = ("model", "optimizer", "epoch", "step")
EPOCH_FILE = re.compile(r"^epoch_(\d{4,})\.pt$")


def policy(paper):
    """(rolling_every, rolling_keep, permanent_every); 0 disables a tier."""
    values = [paper.get("checkpoint_rolling_every", 0), paper.get("checkpoint_rolling_keep", 2),
              paper.get("checkpoint_permanent_every", 0)]
    if any(type(v) is not int or v < 0 for v in values) or (values[0] and values[1] < 1):
        raise ValueError("paper.checkpoint_rolling_every/_keep/_permanent_every must be nonnegative integers (keep >= 1)")
    return tuple(values)


def epoch_files(directory):
    """{completed_epochs: path} for epoch_NNNN.pt files only."""
    found = {}
    for path in Path(directory).iterdir():
        match = EPOCH_FILE.match(path.name)
        if match and path.is_file():
            found[int(match.group(1))] = path
    return found


def load(path):
    """Return the state, or raise if the file is unreadable or not one of our checkpoints."""
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(state, dict) or any(key not in state for key in REQUIRED):
        raise ValueError(f"Not a training checkpoint: {path}")
    return state


def write(state, path, guard):
    """Atomic save followed by a read-back; the previous file survives any failure."""
    path = guard.check(Path(path))
    temporary = guard.check(path.with_name(path.name + ".tmp"))
    torch.save(state, temporary)
    if load(temporary)["epoch"] != state["epoch"]:
        raise RuntimeError(f"Checkpoint read-back mismatch: {temporary}")
    temporary.replace(path)
    return path


def retained(epochs, rolling_every, rolling_keep, permanent_every):
    """Epoch files to keep: every permanent multiple plus the newest rolling ones."""
    keep = {e for e in epochs if permanent_every and e % permanent_every == 0}
    rolling = sorted(e for e in epochs if rolling_every and e % rolling_every == 0)
    return keep | set(rolling[-rolling_keep:] if rolling_every else ())


def save_epoch(state, directory, guard, completed, paper):
    """Save after `completed` epochs. Old rolling files are removed only after the new one is verified."""
    rolling_every, rolling_keep, permanent_every = policy(paper)
    directory = Path(directory)
    latest = write(state, directory / "latest.pt", guard)
    if (rolling_every and completed % rolling_every == 0) or (permanent_every and completed % permanent_every == 0):
        write(state, directory / f"epoch_{completed:04d}.pt", guard)
        files = epoch_files(directory)
        keep = retained(files, rolling_every, rolling_keep, permanent_every)
        for epoch, path in files.items():
            if epoch not in keep and epoch < completed:
                guard.check(path).unlink()
    for stale in directory.glob("*.pt.tmp"):
        guard.check(stale).unlink()
    return latest


def resume_candidates(directory):
    """latest.pt first, then epoch files newest first."""
    directory = Path(directory)
    files = epoch_files(directory) if directory.is_dir() else {}
    return [directory / "latest.pt", *(files[e] for e in sorted(files, reverse=True))]
