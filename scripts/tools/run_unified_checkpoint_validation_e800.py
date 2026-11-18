#!/usr/bin/env python3
"""Isolated epoch-800 validation; preserve existing 400/500/600 and 700 runs."""
import argparse
import csv
import importlib.util
import itertools
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

CORE = Path(__file__).with_name("run_unified_checkpoint_validation.py")
CORE_SHA = "7fa13684781599a35e2d61cfedf67f3ff643d3ae2f03f924ccf19357b877ecd9"
CHECKPOINT_SHA = "017384e1f40bfae94a4c22aabd94d0988e05412d13059c4c0c6cb6c9dd86172a"
BASE_REL = "outputs/4dflow/checkpoint_validation/unified_joint_e400_e500_e600_20261005T1820Z"
PROTECTED_E700_REL = "outputs/4dflow/checkpoint_validation/unified_joint_e700_extension_20261006T2104Z"
SPEC = importlib.util.spec_from_file_location("e800_validation_core", CORE)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)
v.EPOCHS = (800,)  # Private module only; on-disk pinned core remains unchanged.
NEW_FILES = (
    "scripts/tools/run_unified_checkpoint_validation_e800.py",
    "tests/test_unified_checkpoint_validation_e800.py",
    "scripts/slurm/run_unified_checkpoint_validation_e800_cpu.slurm",
    "scripts/slurm/run_unified_checkpoint_validation_e800_gpu.slurm",
)


def base_root(repo, root):
    base = v.safe_root(repo, repo / BASE_REL)
    for protected in (base, v.safe_root(repo, repo / PROTECTED_E700_REL)):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError("Extension and existing validation roots must be disjoint")
    return base


def same_plan(a, b):
    fields = ("index", "anatomy", "center", "vendor", "patient", "rel_dir", "kspace_shape", "image_shape")
    if len(a["cases"]) != 63 or len(a["pairs"]) != 259:
        raise ValueError("Expected 63 validation patients and 259 acceleration cases")
    if [{k: c[k] for k in fields} for c in a["cases"]] != [{k: c[k] for k in fields} for c in b["cases"]]:
        raise ValueError("Validation case identity drift")
    if a["pairs"] != b["pairs"]:
        raise ValueError("Validation acceleration-case coverage drift")


def verify(repo, root):
    if v.sha(CORE) != CORE_SHA:
        raise ValueError("Pinned original validation core changed")
    base = base_root(repo, root)
    old, old_plan = v.verify(repo, base)
    provenance, plan = v.verify(repo, root)
    if set(provenance["checkpoints"]) != {"800"} or provenance["checkpoints"]["800"]["sha256"] != CHECKPOINT_SHA:
        raise ValueError("Wrong epoch-800 checkpoint")
    if provenance["split_sha256"] != old["split_sha256"] or provenance["protocol"] != old["protocol"]:
        raise ValueError("Split or scientific protocol changed")
    same_plan(plan, old_plan)
    proof = v.read(root / "extension_receipt.json")
    for name, digest in proof["source_sha256"].items():
        if v.sha(repo / name) != digest:
            raise ValueError("Epoch-800 extension source changed")
    for name, digest in proof["base_metadata_sha256"].items():
        if v.sha(base / name) != digest:
            raise ValueError("Existing validation provenance changed")
    if not (root / "gt").is_symlink() or (root / "gt").resolve() != (base / "gt").resolve():
        raise ValueError("GT must reuse the existing validation cache")
    return provenance, plan, base


def gt_integrity(repo, root, plan, base):
    """Read-only full hashes and NPY header/shape gates; no GT reconstruction."""
    import numpy as np
    records = []
    for case in plan["cases"]:
        directory = base / "gt" / case["rel_dir"]
        metadata = v.read(directory / "img_gt.metadata.json")
        image = directory / "img_gt.npy"
        before = v.fingerprint(image)
        a = np.load(image, mmap_mode="r", allow_pickle=False)
        if a.dtype != np.dtype("complex64") or list(a.shape) != case["image_shape"] or image.stat().st_size != metadata["output_size"]:
            raise ValueError("Invalid reused GT header/shape/length")
        del a
        if v.sha(image) != metadata["output_sha256"] or v.fingerprint(image) != before:
            raise ValueError("Reused GT changed")
        record = dict(case_index=case["index"],sha256=metadata["output_sha256"],fingerprint=[str(x) for x in before])
        for name, key in (("segmask.mat", "segmask_sha256"), ("params.csv", "params_sha256")):
            path = directory / name
            fp = v.fingerprint(path)
            digest = v.sha(path)
            if digest != metadata[key] or digest != v.sha(Path(case["source_dir"]) / name) or v.fingerprint(path) != fp:
                raise ValueError("Reused GT metadata/source copy changed")
            record[name] = dict(sha256=digest,fingerprint=[str(x) for x in fp])
        records.append(record)
    return records


def prepare(repo, root, data):
    if v.sha(CORE) != CORE_SHA:
        raise ValueError("Original validation core changed")
    base = base_root(repo, root)
    old, old_plan = v.verify(repo, base)
    expected = repo / "outputs/4dflow/4dflow_joint_encoding" / v.EXPERIMENT / (v.PREFIX + "_epoch800.pt")
    before = v.fingerprint(expected)
    if v.sha(expected) != CHECKPOINT_SHA or v.fingerprint(expected) != before:
        raise ValueError("Requested epoch-800 checkpoint changed")
    pins = {name: v.sha(repo / name) for name in NEW_FILES}
    if (root / "extension_receipt.json").exists():
        verify(repo, root)
        return
    v.prepare(repo, root, data)  # Original strict finite-weight/key/shape/epoch/W&B audit.
    prepared, plan = v.verify(repo, root)
    same_plan(plan, old_plan)
    if prepared["checkpoints"]["800"]["sha256"] != CHECKPOINT_SHA or prepared["protocol"] != old["protocol"]:
        raise ValueError("Checkpoint/protocol drift")
    if not (root / "gt").exists():
        (root / "gt").symlink_to(base / "gt", target_is_directory=True)
    gt = gt_integrity(repo, root, plan, base)
    if pins != {name: v.sha(repo / name) for name in NEW_FILES}:
        raise ValueError("Extension sources changed during preparation")
    v.write(root / "extension_receipt.json", dict(status="complete",epoch=800,
        checkpoint_sha256=CHECKPOINT_SHA,base_run=str(base),source_sha256=pins,
        base_metadata_sha256={n:v.sha(base/n) for n in ("prepared.json","plan.json","shards.json")},
        gt_reuse=gt,cases=259,patients=63,job_id=os.environ.get("SLURM_JOB_ID"),
        gpu_gate="existing 777-score completion and recoveries scientifically verified first"))
    verify(repo, root)


def read_rows(path, expected):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    v.validate_rows(rows, expected)
    return rows


def require_base_complete(repo, root):
    _, plan, base = verify(repo, root)
    completion = v.read(base / "completion_receipt.json")
    comparison = v.read(base / "comparison.json")
    if completion.get("status") != "complete" or completion.get("valid_scores") != 777 or comparison.get("total_valid_scores") != 777:
        raise ValueError("Existing validation is not scientifically complete")
    for name, key in (("REPORT.md", "report_sha256"), ("comparison.json", "comparison_sha256")):
        if v.sha(base/name) != completion[key]:
            raise ValueError("Existing completion hash mismatch")
    expected = {p["rel_path"] for p in plan["pairs"]}
    for epoch in (400,500,600):
        path = base / "metrics" / f"epoch{epoch}.csv"
        receipt = v.read(path.with_suffix(".receipt.json"))
        if receipt.get("num_valid") != 259 or receipt["csv_sha256"] != v.sha(path):
            raise ValueError("Existing score CSV hash mismatch")
        read_rows(path, expected)
    maps = comparison["corrmap_sha256"]
    if len(maps) != 63 or {str(p.relative_to(base)) for p in (base/"corrmaps").rglob("corrmap.npz")} != set(maps):
        raise ValueError("Expected exact 63 existing GT-derived correction maps")
    for rel, digest in maps.items():
        if not rel.startswith("corrmaps/") or ".." in Path(rel).parts or v.sha(base/rel) != digest:
            raise ValueError("Existing correction-map drift")
    return plan, base, comparison


def pilot(repo, root):
    plan, _, _ = require_base_complete(repo, root)
    cases = (min(plan["cases"],key=lambda c:c["image_bytes"]), max(plan["cases"],key=lambda c:c["image_bytes"]))
    with v.lock(root, "inference"):
        measurements = []
        for case in cases:
            index = next(p["index"] for p in plan["pairs"] if p["case_index"] == case["index"])
            label = f"pilot_e800_p{index}"
            v.infer_pairs(repo, root, 800, [index], label)
            measurements.append(dict(v.read(root/"work"/label/"epoch800/receipt.json"),pair_index=index))
        v.write(root/"pilot.json",dict(status="complete",measurements=measurements,job_id=os.environ.get("SLURM_JOB_ID")))


def make_shards(repo, root):
    _, plan, base = verify(repo, root)
    pilot_result = v.read(root/"pilot.json")
    if pilot_result["status"] != "complete" or len(pilot_result["measurements"]) != 2:
        raise ValueError("Incomplete epoch-800 pilot")
    cases = [dict(c,samples=v.inference_center_samples(c)) for c in plan["cases"]]
    rate = max(v.read(base/"shards.json")["seconds_per_sample"],
        max(m["wall_seconds"]/cases[plan["pairs"][m["pair_index"]]["case_index"]]["samples"] for m in pilot_result["measurements"])*1.5)
    shards = v.partition(plan["pairs"],cases,rate,9000)
    if sorted(i for s in shards for i in s["indices"]) != list(range(259)):
        raise ValueError("Invalid epoch-800 shard coverage")
    v.write(root/"shards.json",dict(status="complete",epochs=[800],shards=shards,
        total_array_tasks=len(shards),seconds_per_sample=rate,budget_seconds=9000,
        workload_axes="raw T*kx, reader T*x centers",pilot_timings=pilot_result["measurements"]))


def infer_array(repo, root, index):
    require_base_complete(repo, root)
    shards = v.read(root/"shards.json")["shards"]
    if index is None or not 0 <= index < len(shards):
        raise IndexError("Epoch-800 array index outside plan")
    with v.lock(root,"inference",shared=True), v.lock(root,f"inference_task{index:03d}"):
        for pair in shards[index]["indices"]:
            v.infer_pairs(repo,root,800,[pair],f"full_task{index:03d}_p{pair:03d}")


def score(repo, root):
    import numpy as np
    plan, base, comparison = require_base_complete(repo,root)
    with v.lock(root,"score"):
        gt_integrity(repo,root,plan,base)
        expected = {p["rel_path"] for p in plan["pairs"]}
        predictions = root/"predictions/epoch800"
        if {p.relative_to(predictions).as_posix() for p in predictions.rglob("img_ktGaussian*.npz")} != expected:
            raise ValueError("Incomplete epoch-800 predictions")
        if {p.name for p in (root/"receipts/epoch800").glob("pair*.json")} != {f"pair{i:03d}.json" for i in range(259)}:
            raise ValueError("Incomplete epoch-800 normal receipts")
        for pair in plan["pairs"]:
            receipt = v.read(root/"receipts/epoch800"/f"pair{pair['index']:03d}.json")
            digest = v.verify_prediction(predictions/pair["rel_path"],plan["cases"][pair["case_index"]]["image_shape"])
            if receipt["sha256"] != digest or receipt["checkpoint_sha256"] != CHECKPOINT_SHA:
                raise ValueError("Epoch-800 prediction provenance mismatch")
        # Same exact maps, copied to extension-owned cache; never write base cache.
        for rel, digest in comparison["corrmap_sha256"].items():
            dest = root/rel
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(base/rel,dest)
            if v.sha(dest) != digest:
                raise ValueError("Copied correction map differs from existing epochs")
        evaluator = v.helper(repo,"evaluate_4dflow_submission")
        np.random.seed(v.SEED)
        original = sys.argv
        try:
            sys.argv = ["evaluate","--submission-root",str(predictions),"--gt-root",str(base/"gt"),
                "--eval-code-dir",str(repo/v.EVAL),"--out-csv",str(root/"metrics/epoch800.csv"),
                "--out-json",str(root/"metrics/epoch800.summary.json"),"--cache-dir",str(root/"corrmaps"),
                "--spatial-mode","full","--include-complex-diff"]
            evaluator.main()
        finally:
            sys.argv = original
        for rel,digest in comparison["corrmap_sha256"].items():
            if v.sha(root/rel) != digest or v.sha(base/rel) != digest:
                raise ValueError("Correction map changed during scoring")
        read_rows(root/"metrics/epoch800.csv",expected)
        v.write(root/"metrics/epoch800.receipt.json",dict(status="complete",num_valid=259,csv_sha256=v.sha(root/"metrics/epoch800.csv")))
        report(repo,root)


def report(repo, root):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan, base, old_comparison = require_base_complete(repo,root)
    expected = {p["rel_path"] for p in plan["pairs"]}
    epochs = (400,500,600,800)
    rows = {e:read_rows((root if e==800 else base)/"metrics"/f"epoch{e}.csv",expected) for e in epochs}
    new_receipt = v.read(root/"metrics/epoch800.receipt.json")
    if new_receipt["num_valid"] != 259 or new_receipt["csv_sha256"] != v.sha(root/"metrics/epoch800.csv"):
        raise ValueError("Epoch-800 score receipt mismatch")
    results = {str(e):dict(overall=v.means(rows[e])["all"],by_organ=v.means(rows[e],"anatomy"),
        by_acceleration=v.means(rows[e],"R"),patient_macro=v.means([dict(patient=p,**r["metrics"]) for p,r in v.means(rows[e],"patient").items()])["all"]) for e in epochs}
    paired = {}
    for a,b in itertools.combinations(epochs,2):
        first = {r["rel_path"]:r for r in rows[a]}; second = {r["rel_path"]:r for r in rows[b]}
        paired[f"{b}_minus_{a}"] = {m:dict(mean_delta=statistics.mean(float(second[p][m])-float(first[p][m]) for p in expected),
            improved_count=sum((float(second[p][m])-float(first[p][m]))*(1 if m=="SSIM" else -1)>0 for p in expected)) for m in v.METRICS}
    images = []
    evaluator = v.helper(repo,"evaluate_4dflow_submission")
    for organ in v.ORGANS:
        candidates = sorted((c for c in plan["cases"] if c["anatomy"]==organ),key=lambda c:c["image_bytes"])
        case = candidates[len(candidates)//2]
        candidates = [p for p in plan["pairs"] if p["case_index"]==case["index"]]
        pair = next((p for p in candidates if p["R"]==20),candidates[0])
        seg = evaluator.load_segmask(evaluator.load_mat_array,base/"gt"/case["rel_dir"]/"segmask.mat").squeeze().astype(bool)
        z = int(seg.sum(axis=(1,2)).argmax()); frame = case["image_shape"][1]//2
        gt = np.load(base/"gt"/case["rel_dir"]/"img_gt.npy",mmap_mode="r",allow_pickle=False)
        planes = [np.array(gt[:,frame,z])]
        for epoch in epochs:
            pred = evaluator.load_coo_npz((root if epoch==800 else base)/"predictions"/f"epoch{epoch}"/pair["rel_path"])
            planes.append(np.array(pred[:,frame,z])); del pred
        fig, axes = plt.subplots(2,5,figsize=(19,8),constrained_layout=True)
        vmax = float(np.percentile(np.abs(planes[0][0]),99.5))
        for col,(name,plane) in enumerate(zip(("GT",)+tuple(f"Epoch {e}" for e in epochs),planes)):
            axes[0,col].imshow(np.abs(plane[0]),cmap="gray",vmin=0,vmax=vmax); axes[0,col].set_title(name)
            phase = np.where(seg[z],np.angle(plane[1]*np.conj(plane[0])),np.nan)
            image = axes[1,col].imshow(phase,cmap="twilight",vmin=-np.pi,vmax=np.pi)
            axes[0,col].axis("off"); axes[1,col].axis("off")
        fig.colorbar(image,ax=axes[1,:],shrink=.7,label="X flow phase (rad), uncorrected representative slice")
        fig.suptitle(f"{organ}: R={pair['R']}, frame={frame}, z={z}, identical display scales")
        dest = root/"visuals"/f"{organ}_comparison.png"; dest.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(dest,dpi=140); plt.close(fig); images.append(str(dest))
    provenance, _, _ = verify(repo,root)
    v.write(root/"comparison.json",dict(status="complete",patients=63,cases_per_epoch=259,total_valid_scores=1036,
        results=results,paired=paired,protocol=provenance["protocol"],visuals=images,
        checkpoint_sha256=dict(old_comparison["checkpoint_sha256"],**{"800":CHECKPOINT_SHA}),
        corrmap_sha256=old_comparison["corrmap_sha256"],base_run=str(base)))
    lines = ["# Unified checkpoint validation: 400 / 500 / 600 / 800","",
        "63 held-out validation patients; 259 cases per epoch; 1,036 finite valid scores.",
        "Original training and the test split remain unchanged. GT and GT-derived MSAC maps are reused identically.",
        "Full-spatial official utilities, no VENC scaling. Local held-out results, not public-leaderboard parity.","",
        "| Epoch | SSIM higher | nRMSE lower | RelErr lower | AngErr lower | ComplexDiffErr lower |",
        "|---|---|---|---|---|---|"]
    for e in epochs:
        lines.append("| "+str(e)+" | "+" | ".join(f"{results[str(e)]['overall']['metrics'][m]:.6f}" for m in v.METRICS)+" |")
    lines += ["","comparison.json contains patient-macro, organ, acceleration, and all six paired epoch comparisons.",""]
    lines += [f"![{Path(p).stem}](visuals/{Path(p).name})" for p in images]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n")
    v.write(root/"completion_receipt.json",dict(status="complete",valid_scores=1036,epoch800_valid_scores=259,
        report_sha256=v.sha(root/"REPORT.md"),comparison_sha256=v.sha(root/"comparison.json"),
        job_id=os.environ.get("SLURM_JOB_ID")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","pilot","make-shards","infer-array","score","report","record-job","status"))
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,default=v.DATA)
    parser.add_argument("--task-index",type=int)
    parser.add_argument("--job-stage"); parser.add_argument("--job-id")
    args = parser.parse_args()
    repo = args.repo.resolve(); root = v.safe_root(repo,args.run_root,args.data_root); base_root(repo,root)
    os.environ.update(WANDB_MODE="disabled",WANDB_DISABLED="true",HF_HUB_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    if args.stage=="prepare": prepare(repo,root,args.data_root)
    elif args.stage=="record-job": v.record_job(root,args.job_stage,args.job_id)
    elif args.stage=="infer-array": infer_array(repo,root,args.task_index)
    elif args.stage=="status":
        import json
        print(json.dumps({n:v.read(root/n) for n in ("jobs.json","extension_receipt.json","pilot.json","shards.json","completion_receipt.json") if (root/n).exists()},indent=2))
    else: globals()[args.stage.replace("-","_")](repo,root)


if __name__=="__main__": main()
