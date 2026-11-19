#!/usr/bin/env python3
"""Convert the fully sampled final-v2 4D-flow train cases into per-case E1 HDF5 stores.

Each store holds hybrid/target [T,X,E,C,Z,Y,2] (kx-IFFT applied, float32, chunk
[1,1,1,C,Z,Y,2]), coilmap, segmask and the params.csv text. No undersampled inputs:
masks are generated online by the training code. The source dataset is read-only.

Subcommands: ready, run-pool, status, finalize, driver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "experiments/paper/inputs/final_dataset_manifest.csv"
REQUIRED = ("kdata_full.mat", "coilmap.mat", "segmask.mat", "params.csv")
SCHEMA = "raw2insights.4dflow.windowed_hdf5"
SCHEMA_VERSION = 2
PROFILE = "encoding_chunk_1"
RECENT_SECONDS = 900
MAX_ATTEMPTS = 2
STALE_CLAIM_SECONDS = 6 * 3600
USER = os.environ.get("USER", "pengfeig")


def _heavy():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import h5py
    import numpy as np
    from mra_utils import load_vessel_mask_prior, read_mat_array, to_complex_np
    from transforms import ifft1c_kx
    from windowed_4dflow import _coilmap_to_compact

    return {
        "h5py": h5py, "np": np, "load_vessel_mask_prior": load_vessel_mask_prior,
        "read_mat_array": read_mat_array, "to_complex_np": to_complex_np,
        "ifft1c_kx": ifft1c_kx, "coilmap_to_compact": _coilmap_to_compact,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def manifest_sha256(manifest=MANIFEST):
    return hashlib.sha256(Path(manifest).read_bytes()).hexdigest()


def load_cases(manifest=MANIFEST, group="train"):
    cases = {}
    with open(manifest, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            target = row["target_file"]
            if not target.startswith("/Center") or row["dataset_group"] != group:
                continue
            _, case_id, name = target.split("/", 2)
            case = cases.setdefault(case_id, {"sizes": {}, "kdata_shape": None})
            if name in REQUIRED:
                case["sizes"][name] = int(row["size_bytes"])
            if name == "kdata_full.mat":
                case["kdata_shape"] = json.loads(row["shape"])
    bad = [k for k, v in cases.items() if set(v["sizes"]) != set(REQUIRED) or not v["kdata_shape"]]
    if bad or not cases:
        raise ValueError(f"Manifest lacks required files for cases: {bad[:5]}")
    return cases


def scan(data_root, cases, now=None, recent=RECENT_SECONDS):
    """Stat-only check: ready = all required files at manifest size, no rsync temp, not recent."""
    now = time.time() if now is None else now
    out = {}
    for case_id, case in cases.items():
        directory = Path(data_root) / case_id
        reasons, sources = [], {}
        try:
            names = os.listdir(directory)
        except FileNotFoundError:
            names = []
            reasons.append("missing_dir")
        if any(name.startswith(".") for name in names):
            reasons.append("temp_file")
        for name, size in case["sizes"].items():
            try:
                stat = os.stat(directory / name)
            except FileNotFoundError:
                reasons.append(f"missing:{name}")
                continue
            if stat.st_size != size:
                reasons.append(f"size:{name}")
                continue
            if now - stat.st_mtime < recent:
                reasons.append(f"recent:{name}")
            sources[name] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        out[case_id] = {"ready": not reasons, "reasons": reasons, "sources": sources}
    return out


def ledger_path(out_root, case_id):
    return Path(out_root) / "_ledger" / f"{case_id}.json"


def store_path(out_root, case_id):
    return Path(out_root) / "cases" / f"{case_id}.h5"


def is_done(out_root, case_id, sources):
    led = read_json(ledger_path(out_root, case_id))
    return bool(led and led.get("status") == "done" and led.get("sources") == sources
                and store_path(out_root, case_id).is_file())


def pending_cases(out_root, cases, scanned):
    return [c for c, s in scanned.items() if s["ready"] and not is_done(out_root, c, s["sources"])]


def claim(out_root, case_id):
    path = Path(out_root) / "_claims" / f"{case_id}.claim"
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime < STALE_CLAIM_SECONDS:
                    return None
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as stream:
            stream.write(f"{socket.gethostname()} {os.getpid()} {os.environ.get('SLURM_JOB_ID', '')}\n")
        return path
    return None


def hybrid_frame(H, dataset, e, t):
    """[C,Z,Y,X] raw frame -> [X,C,Z,Y,2] float32, identical to raw_4dflow_to_joint_hybrid."""
    np = H["np"]
    hybrid = H["ifft1c_kx"](H["to_complex_np"](dataset[e, t]))
    out = np.stack([hybrid.real, hybrid.imag], axis=-1).astype(np.float32)
    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite hybrid k-space frame")
    return np.transpose(out, (3, 0, 1, 2, 4))


def source_attr(sources):
    return json.dumps(sources, sort_keys=True, separators=(",", ":"))


def verify_store(path, info, sources, *, source_dir=None, samples=0, seed=0):
    H = _heavy()
    h5py, np = H["h5py"], H["np"]
    e, t, c, z, y, x = info["kdata_shape"]
    with h5py.File(path, "r") as f:
        a = f.attrs
        if a.get("schema") != SCHEMA or not bool(a.get("complete")) or a.get("storage_profile") != PROFILE:
            raise ValueError(f"Incomplete or foreign store: {path}")
        if json.loads(a["sources_json"]) != sources:
            raise ValueError(f"Store sources differ from current sources: {path}")
        ds = f["hybrid/target"]
        if ds.shape != (t, x, e, c, z, y, 2) or ds.chunks != (1, 1, 1, c, z, y, 2) or ds.dtype != np.float32:
            raise ValueError(f"Bad hybrid/target layout {ds.shape} {ds.chunks} {ds.dtype}")
        if f["segmask"].shape != (x, z, y) or f["coilmap"].shape[0] != x:
            raise ValueError("Bad segmask/coilmap layout")
        if samples:
            rng = random.Random(f"{seed}:{path}")
            with h5py.File(Path(source_dir) / "kdata_full.mat", "r") as src:
                raw = src["kdata_full"]
                for _ in range(samples):
                    ti, ei, xi = rng.randrange(t), rng.randrange(e), rng.randrange(x)
                    if not np.array_equal(ds[ti, xi, ei], hybrid_frame(H, raw, ei, ti)[xi]):
                        raise ValueError(f"Sample (t={ti}, e={ei}, x={xi}) differs from source: {path}")


def convert_case(case_id, info, data_root, out_root, sources, *, samples=3):
    """Returns (status, seconds, bytes); status is 'skipped' when a matching store exists."""
    H = _heavy()
    h5py, np = H["h5py"], H["np"]
    src = Path(data_root) / case_id
    final = store_path(out_root, case_id)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.is_file():
        try:
            verify_store(final, info, sources)
            return "skipped", 0.0, final.stat().st_size
        except Exception:
            pass  # stale or damaged: rebuild
    for old in final.parent.glob(f".{case_id}.h5.*.tmp"):
        old.unlink()
    tmp = final.with_name(f".{case_id}.h5.{os.getpid()}.tmp")
    started = time.perf_counter()
    try:
        with h5py.File(src / "kdata_full.mat", "r") as source:
            raw = source["kdata_full"]
            e, t, c, z, y, x = raw.shape
            if e != 4 or list(raw.shape) != info["kdata_shape"]:
                raise ValueError(f"k-space shape {raw.shape} differs from manifest {info['kdata_shape']}")
            coil = H["read_mat_array"](src / "coilmap.mat", ("coilmap", "csm", "sensitivity_maps", "sens_maps"))
            compact, layout = H["coilmap_to_compact"](coil, nx=x, nc=c, n_enc=e, axis_order="auto", normalize=True)
            del coil
            seg = H["load_vessel_mask_prior"](src / "segmask.mat", None)
            if seg.shape != (x, z, y):
                raise ValueError(f"segmask {seg.shape} does not match k-space (X,Z,Y)={(x, z, y)}")
            with h5py.File(tmp, "w", libver="latest") as out:
                out.attrs.update({
                    "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "storage_profile": PROFILE,
                    "encoding_chunk_size": 1, "complete": False, "patient_key": case_id,
                    "fully_sampled_only": True, "sources_json": source_attr(sources),
                    "manifest_sha256": manifest_sha256(), "params_csv": (src / "params.csv").read_text(),
                    "created_unix": time.time(),
                })
                ds = out.create_dataset("hybrid/target", shape=(t, x, e, c, z, y, 2), dtype=np.float32,
                                        chunks=(1, 1, 1, c, z, y, 2))
                for ti in range(t):
                    for ei in range(e):
                        ds[ti, :, ei] = hybrid_frame(H, raw, ei, ti)
                co = out.create_dataset("coilmap", data=compact, dtype=np.float32, chunks=(1, *compact.shape[1:]))
                co.attrs["layout"] = layout
                out.create_dataset("segmask", data=seg, dtype=np.float32, chunks=(1, z, y))
                out.attrs["conversion_seconds"] = time.perf_counter() - started
                out.attrs["complete"] = True
        verify_store(tmp, info, sources, source_dir=src, samples=samples, seed=case_id)
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return "converted", time.perf_counter() - started, final.stat().st_size


def worker_loop(list_path, data_root, out_root, deadline, wid):
    items = json.loads(Path(list_path).read_text())["cases"]
    cases = load_cases()
    for item in items:
        if time.time() > deadline:
            return
        case_id = item["case_id"]
        if is_done(out_root, case_id, item["sources"]):
            continue
        led = read_json(ledger_path(out_root, case_id)) or {}
        if led.get("status") == "failed" and led.get("attempts", 0) >= MAX_ATTEMPTS:
            continue
        path = claim(out_root, case_id)
        if path is None:
            continue
        record = {"case_id": case_id, "sources": item["sources"], "host": socket.gethostname(),
                  "job": os.environ.get("SLURM_JOB_ID", ""), "time": time.time(), "attempts": led.get("attempts", 0)}
        try:
            status, seconds, size = convert_case(case_id, cases[case_id], data_root, out_root, item["sources"])
            record.update(status="done", seconds=seconds, bytes=size, outcome=status)
            print(f"[w{wid}] {case_id} {status} {seconds:.0f}s {size / 1e9:.2f}GB", flush=True)
        except Exception as error:  # noqa: BLE001 - recorded, retried once, then reported
            record.update(status="failed", attempts=record["attempts"] + 1, error=f"{type(error).__name__}: {error}")
            print(f"[w{wid}] {case_id} FAILED {record['error']}", flush=True)
        finally:
            write_json(ledger_path(out_root, case_id), record)
            path.unlink(missing_ok=True)


def cmd_ready(args):
    cases = load_cases()
    scanned = scan(args.data_root, cases)
    pend = pending_cases(args.out_root, cases, scanned)
    pend.sort(key=lambda c: -cases[c]["sizes"]["kdata_full.mat"])
    write_json(args.list_out, {"cases": [{"case_id": c, "sources": scanned[c]["sources"]} for c in pend],
                               "created": time.time()})
    ready = sum(s["ready"] for s in scanned.values())
    print(json.dumps({"train_cases": len(cases), "ready": ready, "pending_conversion": len(pend)}))


def cmd_run_pool(args):
    deadline = time.time() + args.deadline_seconds
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker_loop, args=(args.list, args.data_root, args.out_root, deadline, i))
             for i in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed = [p.exitcode for p in procs if p.exitcode]
    if failed:
        sys.exit(1)


def summarize(out_root, cases, scanned):
    ledgers = {c: read_json(ledger_path(out_root, c)) for c in cases}
    done = [c for c, s in scanned.items() if ledgers[c] and ledgers[c].get("status") == "done"
            and ledgers[c].get("sources") == s["sources"] and s["ready"]]
    failed = sorted(c for c, led in ledgers.items() if led and led.get("status") == "failed")
    reasons = {}
    for c, s in scanned.items():
        for r in s["reasons"]:
            key = r.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    return {"time": time.time(), "train_cases": len(cases), "source_ready": sum(s["ready"] for s in scanned.values()),
            "converted_current": len(done), "failed": failed, "not_ready_reasons": reasons}


def cmd_status(args):
    cases = load_cases()
    summary = summarize(args.out_root, cases, scan(args.data_root, cases))
    write_json(Path(args.out_root) / "_status.json", summary)
    print(json.dumps(summary, indent=2))


def cmd_finalize(args):
    cases = load_cases()
    scanned = scan(args.data_root, cases, recent=0)
    errors, entries = [], []
    ids = sorted(cases)
    deep = set(random.Random(2026).sample(ids, min(args.deep_samples, len(ids))))
    for case_id in ids:
        s = scanned[case_id]
        try:
            if not s["ready"] or "temp_file" in s["reasons"]:
                raise ValueError(f"source not ready: {s['reasons']}")
            path = store_path(args.out_root, case_id)
            verify_store(path, cases[case_id], s["sources"], source_dir=Path(args.data_root) / case_id,
                         samples=3 if case_id in deep else 0, seed=2026)
            entries.append({"case_id": case_id, "path": f"cases/{case_id}.h5", "dataset_group": "train",
                            "kdata_shape": cases[case_id]["kdata_shape"], "sources": s["sources"],
                            "store_bytes": path.stat().st_size})
        except Exception as error:  # noqa: BLE001
            errors.append(f"{case_id}: {type(error).__name__}: {error}")
    if errors:
        write_json(Path(args.out_root) / "_finalize_errors.json", {"errors": errors})
        print(f"FINALIZE FAILED: {len(errors)} errors, first: {errors[0]}")
        sys.exit(1)
    index = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "storage_profile": PROFILE,
             "fully_sampled_only": True, "dataset_group": "train", "num_cases": len(entries),
             "manifest_sha256": manifest_sha256(), "source_root": "fudan/data_v2", "cases": entries}
    write_json(Path(args.out_root) / "index.json", index)
    write_json(Path(args.out_root) / "_RECEIPT.json", {
        "complete": True, "num_cases": len(entries), "total_bytes": sum(e["store_bytes"] for e in entries),
        "deep_checked_cases": len(deep), "manifest_sha256": index["manifest_sha256"], "time": time.time()})
    print(f"FINALIZE OK: {len(entries)} cases, deep-checked {len(deep)}")


def squeue_names():
    out = subprocess.run(["squeue", "-u", USER, "-h", "-o", "%j"], capture_output=True, text=True).stdout
    return set(out.split())


def sbatch(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def cmd_driver(args):
    """One pass: refresh ready list, launch conversion/finalize as needed, resubmit itself."""
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "_driver_state.json"
    state = read_json(state_path, {"passes": 0, "started": time.time(), "history": [], "finalize_tries": 0})
    state["passes"] += 1
    cases = load_cases()
    scanned = scan(args.data_root, cases)
    pend = pending_cases(args.out_root, cases, scanned)
    n_ready = sum(s["ready"] for s in scanned.values())
    now = time.time()
    state["history"] = (state["history"] + [[now, n_ready]])[-400:]
    complete = n_ready == len(cases)
    state["complete_since"] = (state.get("complete_since") or now) if complete else None
    changed = [t for t, n in state["history"] if n != n_ready]
    state["stalled"] = (not complete) and bool(state["history"]) and now - state["history"][0][0] > 12 * 3600 \
        and not changed
    receipt = read_json(out / "_RECEIPT.json")
    active = squeue_names()
    action = "none"
    if pend and "fv2-convert" not in active:
        list_path = out / "_lists" / f"pending_{int(now)}.json"
        pend.sort(key=lambda c: -cases[c]["sizes"]["kdata_full.mat"])
        write_json(list_path, {"cases": [{"case_id": c, "sources": scanned[c]["sources"]} for c in pend], "created": now})
        job = sbatch(["sbatch", f"--array=0-{args.nodes - 1}%{args.nodes}", f"--export=ALL,FV2_LIST={list_path}",
                      args.convert_script])
        action = f"convert:{job}:{len(pend)}"
    elif not pend and complete and now - state["complete_since"] >= 3600 and not receipt \
            and "fv2-finalize" not in active and "fv2-convert" not in active and state["finalize_tries"] < 3:
        state["finalize_tries"] += 1
        action = "finalize:" + sbatch(["sbatch", args.finalize_script])
    state["last"] = {"time": now, "ready": n_ready, "pending": len(pend), "action": action,
                     "total": len(cases), "receipt": bool(receipt)}
    done = bool(receipt and receipt.get("complete"))
    state["done"] = done
    write_json(state_path, state)
    write_json(out / "_status.json", {**summarize(args.out_root, cases, scanned), "driver": state["last"],
                                       "stalled": state["stalled"], "passes": state["passes"]})
    print(json.dumps(state["last"]))
    if not done and state["passes"] < args.max_passes:
        sbatch(["sbatch", f"--begin=now+{args.interval_minutes}minutes", args.driver_script])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("ready", "run-pool", "status", "finalize", "driver"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", required=True)
        p.add_argument("--out-root", required=True)
        if name == "ready":
            p.add_argument("--list-out", required=True)
        if name == "run-pool":
            p.add_argument("--list", required=True)
            p.add_argument("--workers", type=int, default=24)
            p.add_argument("--deadline-seconds", type=float, default=11700)
        if name == "finalize":
            p.add_argument("--deep-samples", type=int, default=40)
        if name == "driver":
            p.add_argument("--convert-script", required=True)
            p.add_argument("--finalize-script", required=True)
            p.add_argument("--driver-script", required=True)
            p.add_argument("--nodes", type=int, default=10)
            p.add_argument("--max-passes", type=int, default=96)
            p.add_argument("--interval-minutes", type=int, default=30)
    args = parser.parse_args()
    {"ready": cmd_ready, "run-pool": cmd_run_pool, "status": cmd_status, "finalize": cmd_finalize,
     "driver": cmd_driver}[args.cmd](args)


if __name__ == "__main__":
    main()
