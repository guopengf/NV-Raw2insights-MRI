"""Recreate isolated author checkouts/dependencies; never reorganize dataset files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .common.safety import OutputGuard, ROOT


def run(command):
    print(" ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def prepare_author(item, guard):
    destination = guard.check(ROOT / item["path"])
    guard.directory(destination.parent)
    existing = (destination / ".git").exists()
    if not existing:
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise RuntimeError(f"Author path contains non-repository files; refusing to overwrite: {destination}")
        run(["git", "-c", "http.version=HTTP/1.1", "clone", "--filter=blob:none", "--no-checkout", item["url"], destination])
    if existing and subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True).strip():
        raise RuntimeError(f"Author checkout has local edits; refusing to overwrite: {destination}")
    available = subprocess.run(["git", "-C", str(destination), "cat-file", "-e", f"{item['commit']}^{{commit}}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not available:
        run(["git", "-C", destination, "fetch", "origin", item["commit"]])
    if item.get("sparse_directories"):
        run(["git", "-C", destination, "sparse-checkout", "set", "--cone", *item["sparse_directories"]])
    head = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if not existing or head != item["commit"]:
        run(["git", "-C", destination, "checkout", "--detach", item["commit"]])
    print(f"PINNED {destination.name} {item['commit']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fetch-authors", action="store_true")
    p.add_argument("--install-deps", action="store_true")
    args = p.parse_args()
    guard = OutputGuard()
    if args.fetch_authors:
        lock = json.loads((ROOT / "experiments/paper/vendor_lock.json").read_text())
        if (ROOT / ".gitmodules").is_file() and (ROOT / ".git").exists():
            run(["git", "submodule", "init", "--", *(item["path"] for item in lock.values() if "url" in item)])
        for item in lock.values():
            if "url" not in item:
                continue
            prepare_author(item, guard)
    if args.install_deps:
        venv = guard.check(ROOT / "experiments/paper/.venv")
        run([sys.executable, "-m", "venv", "--system-site-packages", venv])
        cache = guard.directory(ROOT / "outputs/paper/pip_cache")
        run([venv / "bin/python", "-m", "pip", "install", "--cache-dir", cache,
             "-r", ROOT / "experiments/paper/requirements-extra.txt"])
    if not args.fetch_authors and not args.install_deps:
        p.error("Choose --fetch-authors and/or --install-deps")


if __name__ == "__main__":
    main()
