#!/usr/bin/env python3
"""Measure the existing pinned inference path; never alter its scientific inputs."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import threading
import time

GIB = 1024 ** 3
MIB = 1024 ** 2
EXPECTED_RAW = [4, 20, 10, 105, 128, 128]
LABEL = "memory_case33_e400_p149"


def load_driver(repo):
    spec = importlib.util.spec_from_file_location(
        "pinned_validation", repo / "scripts/tools/run_unified_checkpoint_validation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key_values(path):
    return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())}


def process_status(path):
    result = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"):
            fields = value.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise ValueError("Unexpected /proc memory unit")
            result[key] = int(fields[0]) * 1024
    return result


def process_tree(pid, proc=Path("/proc")):
    pending, visited, rows = [pid], set(), []
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        base = proc / str(current)
        try:
            row = process_status(base / "status")
            if row:
                rows.append(dict(pid=current, **row))
            for task in (base / "task").glob("*"):
                try:
                    pending.extend(int(p) for p in (task / "children").read_text().split())
                except (FileNotFoundError, ProcessLookupError):
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
    return rows


def resolve_cgroup(membership, mountinfo, extra_mount=Path("/host-cgroup"),
                   host_membership=None, job_id=None):
    """Resolve v2 membership against actual mount roots, including namespaces."""
    group = next((line[3:] for line in membership.splitlines() if line.startswith("0::")), None)
    if group is None:
        raise RuntimeError("GPU node is not cgroup v2; diagnose before changing profiler")
    group = PurePosixPath(group)
    if ".." in group.parts:
        raise ValueError("Unsafe cgroup membership")
    candidates = []
    # Enroot can report namespace membership '/' while /host-cgroup exposes
    # the host hierarchy. The batch shell supplies its real host membership.
    # Select the job ancestor (all its steps), never just step_batch or host.
    if host_membership is not None:
        host_group = next((line[3:] for line in host_membership.splitlines()
                           if line.startswith("0::")), None)
        if host_group is None or not job_id or not job_id.isdigit():
            raise ValueError("Missing host cgroup or numeric job identity")
        host_group = PurePosixPath(host_group)
        component = "job_" + job_id
        if ".." in host_group.parts or component not in host_group.parts:
            raise ValueError("Host membership does not identify this diagnostic job")
        job_group = PurePosixPath(*host_group.parts[:host_group.parts.index(component)+1])
        candidates.append(extra_mount / str(job_group).lstrip("/"))
        if not (candidates[0] / "memory.current").is_file():
            raise RuntimeError(f"No readable host job cgroup for {job_group}")
        return candidates[0]
    candidates.append(extra_mount / str(group).lstrip("/"))
    for line in mountinfo.splitlines():
        left, sep, right = line.partition(" - ")
        if not sep or right.split()[0] != "cgroup2":
            continue
        fields = left.split()
        mounted_root, mounted_at = PurePosixPath(fields[3]), Path(fields[4])
        if group.is_relative_to(mounted_root):
            candidates.append(mounted_at / str(group.relative_to(mounted_root)))
    for candidate in candidates:
        if (candidate / "memory.current").is_file():
            return candidate
    raise RuntimeError(f"No readable job cgroup for {group}")


def cgroup_snapshot(path):
    result = {"stat": key_values(path / "memory.stat"),
              "events": key_values(path / "memory.events")}
    for name in ("memory.current", "memory.peak", "memory.max", "memory.swap.current"):
        value = (path / name).read_text().strip()
        result[name] = value if value == "max" else int(value)
    return result


def allocation_cgroup(path):
    # Slurm may put the numeric RAM limit at the job parent and leave the
    # step's memory.max as "max". Use the first finite limit in our ancestry.
    for candidate in (path, *path.parents):
        maximum = candidate / "memory.max"
        if maximum.is_file():
            value = maximum.read_text().strip()
            if value != "max":
                if not 160*GIB <= int(value) <= 220*GIB:
                    raise ValueError("First finite cgroup limit is not diagnostic allocation")
                return candidate
    raise ValueError("No finite allocation RAM limit")


def gpu_snapshot():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    rows = []
    for fields in csv.reader(output.splitlines()):
        index, name, used, total, util = [x.strip() for x in fields]
        rows.append(dict(index=int(index), name=name, used_bytes=int(used)*MIB,
                         total_bytes=int(total)*MIB, utilization_percent=int(util)))
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one visible diagnostic GPU")
    return rows


def required_bytes(sample):
    """Conservative non-cache footprint; do not mistake reclaimable cache for anon."""
    stat = sample["cgroup"]["stat"]
    # kernel already includes slab and sockets. Dirty/writeback may overlap;
    # counting both is conservative. shmem is part of file, not anon.
    return sum(stat.get(k, 0) for k in
               ("anon", "shmem", "kernel", "file_dirty", "file_writeback"))


def summarize(samples):
    if len(samples) < 10:
        raise ValueError("Insufficient memory samples")
    limits = {s["cgroup"]["memory.max"] for s in samples}
    if len(limits) != 1:
        raise ValueError("Changing cgroup limit")
    limit = limits.pop()
    if not isinstance(limit, int) or not 160*GIB <= limit <= 220*GIB:
        raise ValueError("Not the bounded 180G diagnostic allocation cgroup")
    if any(not {"anon", "file", "kernel", "shmem"} <= set(s["cgroup"]["stat"])
           or not s["processes"] for s in samples):
        raise ValueError("Missing process/anonymous/cache measurements")
    first, last = samples[0]["cgroup"]["events"], samples[-1]["cgroup"]["events"]
    events = {k: last.get(k, 0) - first.get(k, 0) for k in set(first) | set(last)}
    gpu = [g for s in samples for g in s.get("gpu", [])]
    if not gpu:
        raise ValueError("Missing GPU measurements")
    peak_required = max(required_bytes(s) for s in samples)
    gpu_ratio = max(g["used_bytes"] / g["total_bytes"] for g in gpu)
    failures = []
    if any(last.get(k, 0) != 0 for k in ("oom", "oom_kill", "oom_group_kill")):
        failures.append("cgroup OOM event")
    if peak_required > .8 * limit:
        failures.append("less than 20 percent non-cache RAM headroom")
    if gpu_ratio > .8:
        failures.append("less than 20 percent GPU headroom")
    return dict(sample_count=len(samples), cgroup_limit_bytes=limit,
        cgroup_peak_bytes=max(s["cgroup"]["memory.peak"] for s in samples),
        peak_current_bytes=max(s["cgroup"]["memory.current"] for s in samples),
        peak_anon_bytes=max(s["cgroup"]["stat"]["anon"] for s in samples),
        peak_file_cache_bytes=max(s["cgroup"]["stat"]["file"] for s in samples),
        peak_non_cache_required_bytes=peak_required,
        non_cache_headroom_fraction=1-peak_required/limit,
        peak_process_hwm_bytes=max(p.get("VmHWM", 0) for s in samples for p in s["processes"]),
        peak_process_tree_rss_bytes=max(sum(p.get("VmRSS", 0) for p in s["processes"]) for s in samples),
        peak_process_tree_anon_bytes=max(sum(p.get("RssAnon", 0) for p in s["processes"]) for s in samples),
        events_delta=events, peak_gpu_bytes=max(g["used_bytes"] for g in gpu),
        gpu_headroom_fraction=1-gpu_ratio, memory_gate_passed=not failures,
        refusal_reasons=failures,
        accounting_note="memory.peak includes file cache; headroom uses anon + shmem + kernel + dirty/writeback, not total cache")


class Monitor:
    def __init__(self, path, cgroup, interval=1.):
        self.path, self.cgroup, self.interval = path, cgroup, interval
        self.stop = threading.Event()
        self.errors = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self, gpu=False):
        row = dict(unix=time.time(), cgroup=cgroup_snapshot(self.cgroup),
                   processes=process_tree(os.getpid()))
        if gpu:
            row["gpu"] = gpu_snapshot()
        return row

    def run(self):
        try:
            with self.path.open("x", buffering=1) as stream:
                count = 0
                while True:
                    stream.write(json.dumps(self.sample(gpu=count % 5 == 0), allow_nan=False)+"\n")
                    count += 1
                    if self.stop.wait(self.interval):
                        stream.write(json.dumps(self.sample(gpu=True), allow_nan=False)+"\n")
                        break
        except BaseException as error:
            self.errors.append(repr(error))

    def start(self):
        # Fail before expensive inference if monitoring is unavailable.
        self.sample(gpu=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=20)
        if self.thread.is_alive() or self.errors:
            raise RuntimeError(f"Memory sampler failed: {self.errors}")


def profile(repo, root):
    driver = load_driver(repo)
    root = driver.safe_root(repo, root)
    provenance, plan = driver.verify(repo, root)
    case = plan["cases"][33]
    if case["kspace_shape"] != EXPECTED_RAW or math.prod(EXPECTED_RAW) != max(
            math.prod(c["kspace_shape"]) for c in plan["cases"]):
        raise ValueError("Expected maximum-raw-memory case33")
    pair = next(p for p in plan["pairs"] if p["case_index"] == 33 and p["R"] == 10)
    if pair["index"] != 149:
        raise ValueError("Diagnostic pair identity changed")
    proof = root / "receipts/epoch400/pair149.json"
    if proof.exists() or (root / "work" / LABEL).exists():
        raise FileExistsError("Existing diagnostic or prediction; reconcile, never repeat")
    membership = Path("/proc/self/cgroup").read_text()
    host_membership = os.environ.get("VALIDATION_HOST_CGROUP")
    member_cgroup = resolve_cgroup(membership, Path("/proc/self/mountinfo").read_text(),
        host_membership=host_membership, job_id=os.environ.get("SLURM_JOB_ID"))
    cgroup = allocation_cgroup(member_cgroup)
    baseline = cgroup_snapshot(cgroup)
    if not isinstance(baseline["memory.max"], int) or not 160*GIB <= baseline["memory.max"] <= 220*GIB:
        raise ValueError("Profiler did not resolve bounded diagnostic job cgroup")
    directory = root / "memory/case33_epoch400"
    directory.mkdir(parents=True, exist_ok=False)
    samples_path = directory / "samples.jsonl"
    source_paths = [Path(__file__).resolve(), repo / "tests/test_unified_validation_memory.py",
                    repo / "scripts/slurm/profile_unified_validation_memory.slurm"]
    start = dict(status="running", job_id=os.environ.get("SLURM_JOB_ID"),
        cgroup_version=2, cgroup_path=str(cgroup), member_cgroup_path=str(member_cgroup),
        membership=membership, host_membership=host_membership,
        case_index=33, pair_index=149, epoch=400, kspace_shape=EXPECTED_RAW,
        checkpoint_sha256=provenance["checkpoints"]["400"]["sha256"],
        profiler_source_sha256={str(p.relative_to(repo)):driver.sha(p) for p in source_paths},
        pinned_driver_sha256=driver.sha(repo / "scripts/tools/run_unified_checkpoint_validation.py"),
        started_unix=time.time(), initial_cgroup=baseline)
    driver.write(directory / "started.json", start)
    print(json.dumps(start), flush=True)
    monitor = Monitor(samples_path, cgroup)
    monitor.start()
    failure = None
    try:
        with driver.lock(root, "inference"):
            driver.infer_pairs(repo, root, 400, [149], LABEL)
    except BaseException as error:
        failure = error
    finally:
        monitor.finish()
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    result = dict(start, **summarize(samples), finished_unix=time.time(),
                  samples_sha256=driver.sha(samples_path), status="failed" if failure else "complete")
    if failure:
        result["failure"] = repr(failure)
    else:
        driver.verify(repo, root)
        receipt = driver.read(proof)
        destination = root / "predictions/epoch400" / pair["rel_path"]
        digest = driver.verify_prediction(destination, case["image_shape"])
        if receipt["status"] != "complete" or receipt["sha256"] != digest or receipt["checkpoint_sha256"] != start["checkpoint_sha256"]:
            raise ValueError("Diagnostic prediction receipt mismatch")
        result.update(prediction_sha256=digest, prediction_receipt_sha256=driver.sha(proof),
                      prediction_shape=case["image_shape"], timing=driver.read(
                          root / "work" / LABEL / "epoch400/receipt.json"))
    driver.write(root / "memory_profile_case33.json", result)
    print(json.dumps(result), flush=True)
    if failure:
        raise failure
    if not result["memory_gate_passed"]:
        raise RuntimeError(result["refusal_reasons"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(WANDB_MODE="disabled", WANDB_DISABLED="true", HF_HUB_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    profile(args.repo.resolve(), args.run_root.resolve())


if __name__ == "__main__":
    main()
