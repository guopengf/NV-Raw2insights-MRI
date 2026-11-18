#!/usr/bin/env python3
"""Fail-closed validation only; never modifies training, checkpoints or raw data."""
from __future__ import annotations
import argparse
import collections
import contextlib
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

EPOCHS=(400,500,600)
ORGANS=("Aorta","Carotid","Cerebrovascular","PortalVein","RenalArtery")
METRICS=("SSIM","nRMSE","RelErr","AngErr","ComplexDiffErr")
SEED=20261005
HEAD="9504c29e50237d96eee89f3cf47693c64482bd2f"
CONFIG_SHA="d65c98b18b39ef4ee89c9e112b123d2b9ea6437e6703d6758fc5eb85706426a7"
CONFIG_NAME="nv_raw2insights_mri_small_4dflow_3d_flowvn_joint_venc_flow_unified_70_15_15.json"
EXPERIMENT="small_ft_4dflow_3d_flowvn_joint_venc_flow_unified_70_15_15"
PREFIX="nv_raw2insights_mri_"+EXPERIMENT
DATA=Path("/data/CMRx4DFlow2026-unified-70_15_15-seed20260914")
EVAL=Path("outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z/official/CMRx-evaluation-snippets/evaluation-2026")

def read(path):
    return json.loads(Path(path).read_text())

def write(path,payload):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def fingerprint(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def same_checkpoint_identity(current,recorded):
    # Device minor IDs are node/mount-namespace local. Inode/size/mtime/ctime
    # identify the Weka file across jobs; prepare still checks all five fields
    # unchanged within one namespace across its five-second stability gate.
    return current[1:]==recorded[1:]

def safe_root(repo,root,data=DATA):
    repo,root=Path(repo).resolve(),Path(root).resolve()
    allowed=repo/"outputs/4dflow/checkpoint_validation"
    if root==allowed or not root.is_relative_to(allowed) or root.is_relative_to(Path(data).resolve()):
        raise ValueError(f"Unsafe validation run root: {root}")
    return root

@contextlib.contextmanager
def lock(root,name,*,shared=False):
    (root/"locks").mkdir(parents=True,exist_ok=True)
    with (root/"locks"/(name+".lock")).open("a") as stream:
        mode=fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(stream,mode|fcntl.LOCK_NB)
        yield

def helper(repo,name):
    sys.path.insert(0,str(repo/"scripts"))
    sys.path.insert(0,str(repo/"scripts/tools"))
    spec=importlib.util.spec_from_file_location(name,repo/"scripts/tools"/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def effective_config(source):
    phase=source["phase3"]
    joint=phase["joint_encoding"]
    if not joint["enabled"] or joint["mode"]!="channel" or joint["count"]!=4 or joint["order"]!=[0,1,2,3]:
        raise ValueError("Wrong joint encoding architecture")
    if phase["recon_mode"]!="slab" or phase["num_slices"]!=3 or phase["enable_vaa"]:
        raise ValueError("Wrong slab/VAA architecture")
    result=json.loads(json.dumps(source))
    # Four encoding images = one joint sample; batch_size=1 is invalid.
    result.update(batch_size=4,num_workers=0,ddp=False,seed=SEED,data_aug=False,enable_onelogger=False)
    result["four_dflow_augmentation"]["enabled"]=False
    return result

def discover(manifest,data):
    ids=collections.defaultdict(set)
    origins=collections.defaultdict(set)
    for c in manifest["cases"]:
        if c["uid"] in ids[c["split"]]:raise ValueError("Duplicate identity")
        ids[c["split"]].add(c["uid"])
        origins[c["split"]].add(c.get("raw_origin_path",c["source_path"]))
    for a,b in (("train","val"),("train","test"),("val","test")):
        if ids[a]&ids[b] or origins[a]&origins[b]:raise ValueError("Split overlap")
    result=[]
    for c in sorted((c for c in manifest["cases"] if c["split"]=="val"),key=lambda c:c["uid"]):
        parts=(c["organ"],c["center"],c["scanner"],c["uid"])
        if any(Path(p).name!=p or p in ("",".","..") for p in parts):raise ValueError("Unsafe case name")
        source=data/"val"/Path(*parts)
        if not source.is_dir() or not source.resolve().is_relative_to(data.resolve()):raise ValueError("Escaped source")
        accelerations=sorted(c["available_accelerations"])
        if not accelerations or len(set(accelerations))!=len(accelerations) or not set(accelerations)<={10,20,30,40,50}:
            raise ValueError("Invalid acceleration set")
        required=["kdata_full.mat","coilmap.mat","segmask.mat","params.csv"]
        required += [f"{stem}_ktGaussian{r}.mat" for r in accelerations for stem in ("kdata","usmask")]
        for name in required:
            if not (source/name).is_file() or not (source/name).stat().st_size:raise FileNotFoundError(source/name)
        result.append(dict(index=len(result),uid=c["uid"],patient=c["uid"],anatomy=c["organ"],
                           center=c["center"],vendor=c["scanner"],source_dir=str(source),
                           task="TaskUnified",settype="ValidationSet",
                           rel_dir=(Path("TaskUnified/ValidationSet")/Path(*parts)).as_posix(),
                           accelerations=accelerations,source_dataset=c["source_dataset"]))
    return result

def input_payload(case,r):
    source=Path(case["source_dir"])
    us=str(source/f"kdata_ktGaussian{r}.mat")
    return dict(kspace=us,target_kspace=us,targetless=True,mask=[str(source/f"usmask_ktGaussian{r}.mat")],
                mask_type=f"ktGaussian{r}",acquisition="Flow4d",is_4dflow=True,
                coilmap=str(source/"coilmap.mat"),segmask=str(source/"segmask.mat"),
                joint_encodings=True,encoding_indices=[0,1,2,3])

def strict_state(state,shapes):
    result={}
    for key,value in state.items():
        key=key[7:] if key.startswith("module.") else key
        if key in result:raise ValueError("Normalized state collision")
        result[key]=value
    if set(result)!=set(shapes):raise ValueError("Missing or extra checkpoint keys")
    if any(tuple(result[k].shape)!=tuple(shapes[k]) for k in result):raise ValueError("Checkpoint shape mismatch")
    return result

def inference_center_samples(case):
    # CMRxReconReader reshapes raw [enc,T,coil,kz,ky,kx] into
    # [T,x,enc,coil,kz,ky]; joint inference iterates over T*x centers.
    shape=case["kspace_shape"]
    if len(shape)!=6 or shape[0]!=4 or any(int(n)<=0 for n in shape):
        raise ValueError("Invalid raw shape for inference workload")
    return int(shape[1])*int(shape[5])

def prepare(repo,root,data):
    import h5py
    import numpy as np
    import torch
    with lock(root,"prepare"):
        if (root/"prepared.json").exists():
            verify(repo,root);return
        branch=subprocess.check_output(["git","branch","--show-current"],cwd=repo,text=True).strip()
        head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
        cfg=repo/"configs"/CONFIG_NAME
        if branch!="pengfeig/4dflow-finetune" or head!=HEAD or sha(cfg)!=CONFIG_SHA:
            raise ValueError("Training provenance changed unexpectedly")
        cases=discover(read(data/"split_manifest.json"),data)
        if len(cases)!=63 or sum(len(c["accelerations"]) for c in cases)!=259:
            raise ValueError("Expected 63 patients and 259 acceleration cases")
        evaluator=helper(repo,"evaluate_4dflow_submission")
        pairs=[]
        for c in cases:
            source=Path(c["source_dir"])
            with h5py.File(source/"kdata_full.mat","r") as f:shape=list(f["kdata_full"].shape)
            if len(shape)!=6 or shape[0]!=4:raise ValueError("Invalid full k-space shape")
            with h5py.File(source/"coilmap.mat","r") as f:
                if tuple(f["coilmap"].shape)!=tuple(shape[2:]):raise ValueError("Coilmap mismatch")
            seg=evaluator.load_segmask(evaluator.load_mat_array,source/"segmask.mat").squeeze()
            if tuple(seg.shape)!=tuple(shape[-3:]) or not np.any(seg) or not np.isfinite(seg).all():
                raise ValueError("Invalid segmentation mask")
            evaluator.load_venc(source/"params.csv")
            image_shape=[shape[0],shape[1],*shape[3:]]
            c.update(kspace_shape=shape,image_shape=image_shape,samples=shape[1]*shape[5],
                     image_bytes=math.prod(image_shape)*8,
                     source_sizes={n:(source/n).stat().st_size for n in ("kdata_full.mat","coilmap.mat","segmask.mat","params.csv")})
            for r in c["accelerations"]:
                with h5py.File(source/f"kdata_ktGaussian{r}.mat","r") as f:
                    key=next(k for k in f if not k.startswith("#"))
                    if tuple(f[key].shape)!=tuple(shape):raise ValueError("Undersampled shape mismatch")
                stem=f"case{c['index']:03d}_R{r}_joint4"
                pairs.append(dict(index=len(pairs),case_index=c["index"],R=r,stem=stem,
                                  rel_path=c["rel_dir"]+f"/img_ktGaussian{r}.npz"))
                write(root/"inputs"/(stem+".json"),input_payload(c,r))
            print(f"HEADER OK {c['index']+1}/63 {c['anatomy']} {shape}",flush=True)
        view=root/"data_view"
        for c in cases:
            link=view/c["rel_dir"]
            link.parent.mkdir(parents=True,exist_ok=True)
            if link.is_symlink():
                if link.resolve()!=Path(c["source_dir"]).resolve():raise ValueError("Input alias collision")
            elif link.exists():raise FileExistsError(link)
            else:link.symlink_to(c["source_dir"],target_is_directory=True)
        write(root/"effective_config.json",effective_config(read(cfg)))
        write(root/"plan.json",dict(cases=cases,pairs=pairs,epochs=list(EPOCHS),data_root=str(data)))
        write(root/"gt_plan.json",dict(schema_version=1,data_root=str(view),case_count=63,
                                      task_counts={"TaskUnified":63},cases=cases))
        sys.path.insert(0,str(repo/"scripts"))
        from utils import load_config
        from models.latent_recon import create_mri_recon_model
        model=create_mri_recon_model(load_config(root/"effective_config.json"))
        shapes={k:tuple(v.shape) for k,v in model.state_dict().items()}
        checkpoints={}
        for epoch in EPOCHS:
            path=repo/"outputs/4dflow/4dflow_joint_encoding"/EXPERIMENT/f"{PREFIX}_epoch{epoch}.pt"
            before=fingerprint(path)
            if before[2]<1_000_000_000:raise ValueError("Small checkpoint")
            checkpoint=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
            if checkpoint.get("epoch")!=epoch or not checkpoint.get("epoch_finished"):raise ValueError("Wrong or incomplete epoch")
            state=strict_state(checkpoint["net_state_dict"],shapes)
            if any(not torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point() or v.is_complex()):
                raise ValueError("Nonfinite checkpoint weights")
            model.load_state_dict(state,strict=True)
            digest=sha(path)
            if fingerprint(path)!=before:raise ValueError("Checkpoint changed during hashing")
            checkpoints[str(epoch)]=dict(path=str(path),sha256=digest,fingerprint=before,epoch=epoch,
                       epoch_finished=True,global_step=checkpoint.get("global_step"),
                       wandb_run_id=checkpoint.get("wandb_run_id"),state_key_count=len(state))
            del checkpoint,state
            print(f"STRICT CHECKPOINT OK epoch={epoch}",flush=True)
        if {c["wandb_run_id"] for c in checkpoints.values()}!={"cvo4ei7y"}:raise ValueError("Wrong experiment run IDs")
        code=[Path("configs")/CONFIG_NAME,Path("scripts/inference.py"),Path("scripts/utils.py"),
              Path("scripts/joint_encoding.py"),Path("scripts/models/latent_recon.py"),
              Path("scripts/tools/run_unified_checkpoint_validation.py"),
              Path("scripts/tools/build_4dflow_validation_gt.py"),
              Path("scripts/tools/export_4dflow_submission.py"),Path("scripts/tools/evaluate_4dflow_submission.py")]
        official=list((repo/EVAL).glob("*.py"))
        if not official:raise FileNotFoundError("Official metric utilities")
        hashes={str(p):sha(repo/p) for p in code}
        hashes.update({str(p.relative_to(repo)):sha(p) for p in official})
        time.sleep(5)
        if any(fingerprint(c["path"])!=c["fingerprint"] for c in checkpoints.values()):raise ValueError("Unstable checkpoint")
        sizes=dict(gt_dense_bytes=sum(c["image_bytes"] for c in cases),
                   reconstruction_dense_bytes_per_epoch=sum(cases[p["case_index"]]["image_bytes"] for p in pairs))
        write(root/"prepared.json",dict(status="complete",head=head,branch=branch,config_sha256=CONFIG_SHA,
               hashes=hashes,checkpoints=checkpoints,split_sha256=sha(data/"split_manifest.json"),
               case_count=63,pair_count=259,byte_estimates=sizes,
               organ_counts=dict(collections.Counter(c["anatomy"] for c in cases)),
               acceleration_counts=dict(collections.Counter(p["R"] for p in pairs)),
               protocol=dict(spatial_mode="full",use_venc=False,seed=SEED,batch_size=4,joint_samples=1,
               num_workers=0,phase_correction="GT-derived MSAC cached once and shared across epochs",
               public_leaderboard_parity=False)))
        print(json.dumps(dict(status="PREPARED",byte_estimates=sizes)),flush=True)

def verify(repo,root):
    receipt=read(root/"prepared.json")
    for path,digest in receipt["hashes"].items():
        if sha(repo/path)!=digest:raise ValueError(f"Pinned source changed: {path}")
    plan=read(root/"plan.json")
    if sha(Path(plan["data_root"])/"split_manifest.json")!=receipt["split_sha256"]:raise ValueError("Split changed")
    for c in receipt["checkpoints"].values():
        if not same_checkpoint_identity(fingerprint(c["path"]),c["fingerprint"]):raise ValueError("Checkpoint changed")
    return receipt,plan

def amend_wrapper_pin(repo,root,reason):
    if not reason or not reason.strip():raise ValueError("Audited amendment reason required")
    with lock(root,"prepare"):
        receipt=read(root/"prepared.json")
        key="scripts/tools/run_unified_checkpoint_validation.py"
        for path,digest in receipt["hashes"].items():
            if path!=key and sha(repo/path)!=digest:raise ValueError("Non-wrapper source changed")
        plan=read(root/"plan.json")
        if sha(Path(plan["data_root"])/"split_manifest.json")!=receipt["split_sha256"]:raise ValueError("Split changed")
        for c in receipt["checkpoints"].values():
            if not same_checkpoint_identity(fingerprint(c["path"]),c["fingerprint"]):raise ValueError("Checkpoint changed")
        old,new=receipt["hashes"][key],sha(repo/key)
        amendment=dict(old_sha256=old,new_sha256=new,unix=time.time(),
            reason=reason)
        receipt.setdefault("wrapper_amendments",[]).append(amendment)
        receipt["hashes"][key]=new
        write(root/"prepared.json",receipt)
        verify(repo,root)
        print(json.dumps(amendment),flush=True)

def verify_prediction(path,shape):
    import numpy as np
    with np.load(path,allow_pickle=False) as f:
        coords,data=f["coords"],f["data"]
        if f["shape"].tolist()!=list(shape) or coords.ndim!=2 or coords.shape[1]!=5 or len(coords)!=len(data):
            raise ValueError("Invalid prediction shape/coordinates")
        if not np.iscomplexobj(data) or not len(data) or not np.isfinite(data).all():raise ValueError("Empty/nonfinite prediction")
        if np.any(coords<0) or np.any(coords>=np.asarray(shape)[None,:]):raise ValueError("Out-of-bounds coordinates")
    return sha(path)

def infer_pairs(repo,root,epoch,indices,label):
    provenance,plan=verify(repo,root)
    predictions=root/"predictions"/f"epoch{epoch}"
    selected=[]
    for index in indices:
        p=plan["pairs"][index]
        proof=root/"receipts"/f"epoch{epoch}"/f"pair{index:03d}.json"
        destination=predictions/p["rel_path"]
        if proof.exists():
            r=read(proof)
            digest=verify_prediction(destination,plan["cases"][p["case_index"]]["image_shape"])
            if r["sha256"]!=digest or r["checkpoint_sha256"]!=provenance["checkpoints"][str(epoch)]["sha256"]:
                raise ValueError("Existing prediction provenance mismatch")
        else:
            if destination.exists():raise FileExistsError(f"Unreceipted prediction: {destination}")
            selected.append(p)
    if not selected:return
    work=root/"work"/label/f"epoch{epoch}"
    inputs=work/"inputs"
    inputs.mkdir(parents=True,exist_ok=True)
    if list(inputs.glob("*.json")):raise FileExistsError("Existing attempted inputs; diagnose before retry")
    for p in selected:(inputs/(p["stem"]+".json")).symlink_to(root/"inputs"/(p["stem"]+".json"))
    log=work/"inference.log"
    temporary=work/"inference"
    command=[sys.executable,str(repo/"scripts/inference.py"),"--config",str(root/"effective_config.json"),
             "--model_ckpt",provenance["checkpoints"][str(epoch)]["path"],"--input_path",str(inputs),
             "--output_path",str(temporary),"--save-coil-combined-output"]
    started=time.time()
    print(f"INFERENCE START epoch={epoch} cases={len(selected)} log={log}",flush=True)
    with log.open("w") as stream:subprocess.run(command,cwd=repo,stdout=stream,stderr=subprocess.STDOUT,check=True)
    text=log.read_text(errors="replace")
    markers=("net skipped_shape_keys:","net unexpected checkpoint keys:","net shape-inflated compatible weights:","Traceback")
    if any(m in text for m in markers) or "net unchanged_keys: 0" not in text:raise ValueError("Incomplete checkpoint load/runtime failure")
    exporter=helper(repo,"export_4dflow_submission")
    for p in selected:
        c=plan["cases"][p["case_index"]]
        enc={e:temporary/"val_img4ranking"/f"{p['stem']}__enc{e}.mat" for e in range(4)}
        if not all(path.is_file() for path in enc.values()):raise FileNotFoundError("Missing encoding reconstruction")
        destination=exporter.export_case(enc,out_root=predictions,
            data_root=root/"data_view/TaskUnified/ValidationSet"/c["anatomy"],task="TaskUnified",
            split="ValidationSet",anatomy=c["anatomy"],center=c["center"],scanner=c["vendor"],
            patient=c["patient"],acc=p["R"],encodings=[0,1,2,3],recon_layout="yzxt",overwrite=False,verify=True)
        digest=verify_prediction(destination,c["image_shape"])
        write(root/"receipts"/f"epoch{epoch}"/f"pair{p['index']:03d}.json",dict(status="complete",
              epoch=epoch,pair_index=p["index"],sha256=digest,shape=c["image_shape"],
              checkpoint_sha256=provenance["checkpoints"][str(epoch)]["sha256"],
              inference_log=str(log),job_id=os.environ.get("SLURM_JOB_ID")))
        print(f"EXPORTED epoch={epoch} pair={p['index']} shape={c['image_shape']}",flush=True)
    write(work/"receipt.json",dict(status="complete",epoch=epoch,pair_indices=[p["index"] for p in selected],
          wall_seconds=time.time()-started,samples=sum(inference_center_samples(plan["cases"][p["case_index"]]) for p in selected)))

def pilot(repo,root):
    _,plan=verify(repo,root)
    smallest=min(plan["cases"],key=lambda c:c["image_bytes"])
    largest=max(plan["cases"],key=lambda c:c["image_bytes"])
    def pair(c):return next(p["index"] for p in plan["pairs"] if p["case_index"]==c["index"])
    small,large=pair(smallest),pair(largest)
    with lock(root,"inference"):
        measurements=[]
        for epoch,index in ((400,small),(400,large),(500,small),(600,small)):
            label=f"pilot_e{epoch}_p{index}"
            infer_pairs(repo,root,epoch,[index],label)
            timing=read(root/"work"/label/f"epoch{epoch}"/"receipt.json")
            measurements.append(dict(timing,pair_index=index))
        write(root/"pilot.json",dict(status="complete",measurements=measurements,
              smallest_case=smallest["index"],largest_case=largest["index"],job_id=os.environ.get("SLURM_JOB_ID")))

def partition(pairs,cases,rate,budget):
    if rate<=0 or not math.isfinite(rate) or budget<=0:raise ValueError("Invalid shard budget")
    shards=[];current=[];cost=0.
    for p in sorted(pairs,key=lambda p:(p["case_index"],p["R"])):
        estimate=cases[p["case_index"]]["samples"]*rate+30
        if estimate>budget:raise ValueError("One full case exceeds safe measured shard budget")
        if current and cost+estimate>budget:
            shards.append(dict(indices=current,estimated_seconds=cost));current=[];cost=0.
        current.append(p["index"]);cost+=estimate
    if current:shards.append(dict(indices=current,estimated_seconds=cost))
    return shards

def make_shards(repo,root):
    _,plan=verify(repo,root)
    measured=read(root/"pilot.json")
    if measured["status"]!="complete":raise ValueError("Incomplete pilot")
    cases=[dict(c,samples=inference_center_samples(c)) for c in plan["cases"]]
    # Re-derive pilot work from pinned raw shapes, including receipts emitted
    # before the workload-axis correction; their volumes remain complete.
    timings=[]
    for m in measured["measurements"]:
        case=cases[plan["pairs"][m["pair_index"]]["case_index"]]
        timings.append(dict(epoch=m["epoch"],pair_index=m["pair_index"],
            wall_seconds=m["wall_seconds"],reported_samples=m["samples"],
            inference_center_samples=case["samples"]))
    rate=max(m["wall_seconds"]/m["inference_center_samples"] for m in timings)*1.5
    shards=partition(plan["pairs"],cases,rate,9000)
    result=dict(status="complete",seconds_per_sample=rate,budget_seconds=9000,
                shards=shards,total_array_tasks=3*len(shards),epochs=list(EPOCHS),
                workload_axes="raw T*kx, reader T*x centers",pilot_timings=timings)
    write(root/"shards.json",result)
    print(json.dumps(result),flush=True)

def infer_array(repo,root,index,attempt_label="full"):
    shards=read(root/"shards.json")["shards"]
    count=len(shards)
    if not 0<=index<3*count:raise IndexError("Array index outside plan")
    if not attempt_label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in attempt_label):
        raise ValueError("Unsafe inference attempt label")
    epoch=EPOCHS[index//count]
    # The original inference loop retains its previous case's tensors while
    # Dataset loads the next case. Use the identical single-case subprocess
    # path measured by the pilot, releasing that process before the next case.
    # Keep an exclusive pilot out, but allow distinct array tasks to run together.
    # The task lock still rejects duplicates across original/recovery arrays.
    # Keep shard coverage, resources and scientific protocol fixed.
    with lock(root,"inference",shared=True), lock(root,f"inference_task{index:03d}"):
        for pair_index in shards[index%count]["indices"]:
            infer_pairs(repo,root,epoch,[pair_index],f"{attempt_label}_task{index:03d}_p{pair_index:03d}")

def validate_rows(rows,expected):
    paths=[r["rel_path"] for r in rows]
    if len(paths)!=len(set(paths)) or set(paths)!=set(expected):raise ValueError("Score coverage/duplicates")
    for r in rows:
        if r.get("comments") or any(not math.isfinite(float(r[m])) for m in METRICS):raise ValueError("Failed/nonfinite score")

def means(rows,key=None):
    groups=collections.defaultdict(list)
    for r in rows:groups[str(r[key]) if key else "all"].append(r)
    return {g:dict(count=len(rs),metrics={m:statistics.mean(float(r[m]) for r in rs) for m in METRICS})
            for g,rs in sorted(groups.items())}

def score(repo,root):
    import numpy as np
    with lock(root,"score"):
        _,plan=verify(repo,root)
        helper(repo,"build_4dflow_validation_gt").finalize(root/"gt_plan.json",root/"gt",root/"gt_receipt.json")
        evaluator=helper(repo,"evaluate_4dflow_submission")
        expected={p["rel_path"] for p in plan["pairs"]}
        for epoch in EPOCHS:
            predictions=root/"predictions"/f"epoch{epoch}"
            actual={p.relative_to(predictions).as_posix() for p in predictions.rglob("img_ktGaussian*.npz")}
            if actual!=expected or len(list((root/"receipts"/f"epoch{epoch}").glob("pair*.json")))!=259:
                raise ValueError(f"Incomplete predictions epoch {epoch}")
            np.random.seed(SEED)
            original=sys.argv
            try:
                sys.argv=["evaluate","--submission-root",str(predictions),"--gt-root",str(root/"gt"),
                    "--eval-code-dir",str(repo/EVAL),"--out-csv",str(root/"metrics"/f"epoch{epoch}.csv"),
                    "--out-json",str(root/"metrics"/f"epoch{epoch}.summary.json"),
                    "--cache-dir",str(root/"corrmaps"),"--spatial-mode","full","--include-complex-diff"]
                evaluator.main()
            finally:sys.argv=original
            with (root/"metrics"/f"epoch{epoch}.csv").open() as f:rows=list(csv.DictReader(f))
            validate_rows(rows,expected)
            write(root/"metrics"/f"epoch{epoch}.receipt.json",dict(status="complete",num_valid=259,
                  csv_sha256=sha(root/"metrics"/f"epoch{epoch}.csv")))
        report(repo,root)

def visuals(repo,root,plan):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    evaluator=helper(repo,"evaluate_4dflow_submission")
    paths=[]
    for organ in ORGANS:
        candidates=sorted((c for c in plan["cases"] if c["anatomy"]==organ),key=lambda c:c["image_bytes"])
        c=candidates[len(candidates)//2]
        candidates=[p for p in plan["pairs"] if p["case_index"]==c["index"]]
        pair=next((p for p in candidates if p["R"]==20),candidates[0])
        seg=evaluator.load_segmask(evaluator.load_mat_array,root/"gt"/c["rel_dir"]/"segmask.mat").squeeze().astype(bool)
        z=int(seg.sum(axis=(1,2)).argmax());frame=c["image_shape"][1]//2
        gt=np.load(root/"gt"/c["rel_dir"]/"img_gt.npy",mmap_mode="r",allow_pickle=False)
        planes=[np.array(gt[:,frame,z])]
        for epoch in EPOCHS:
            pred=evaluator.load_coo_npz(root/"predictions"/f"epoch{epoch}"/pair["rel_path"])
            planes.append(np.array(pred[:,frame,z]));del pred
        fig,axes=plt.subplots(2,4,figsize=(15,8),constrained_layout=True)
        vmax=float(np.percentile(np.abs(planes[0][0]),99.5))
        for col,(name,plane) in enumerate(zip(("GT","Epoch 400","Epoch 500","Epoch 600"),planes)):
            axes[0,col].imshow(np.abs(plane[0]),cmap="gray",vmin=0,vmax=vmax)
            axes[0,col].set_title(name)
            phase=np.where(seg[z],np.angle(plane[1]*np.conj(plane[0])),np.nan)
            image=axes[1,col].imshow(phase,cmap="twilight",vmin=-np.pi,vmax=np.pi)
            axes[0,col].axis("off");axes[1,col].axis("off")
        fig.colorbar(image,ax=axes[1,:],shrink=.7,label="X flow phase (rad), uncorrected representative slice")
        fig.suptitle(f"{organ}: R={pair['R']}, frame={frame}, z={z}, identical display scales")
        destination=root/"visuals"/f"{organ}_comparison.png"
        destination.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(destination,dpi=140);plt.close(fig)
        paths.append(str(destination))
    return paths

def report(repo,root):
    provenance,plan=verify(repo,root)
    expected={p["rel_path"] for p in plan["pairs"]}
    results={};rows_by_epoch={}
    for epoch in EPOCHS:
        with (root/"metrics"/f"epoch{epoch}.csv").open() as f:rows=list(csv.DictReader(f))
        validate_rows(rows,expected);rows_by_epoch[epoch]=rows
        patient_rows=[dict(patient=p,**v["metrics"]) for p,v in means(rows,"patient").items()]
        results[str(epoch)]=dict(overall=means(rows)["all"],by_organ=means(rows,"anatomy"),
                                by_acceleration=means(rows,"R"),patient_macro=means(patient_rows)["all"])
    paired={}
    for a,b in ((400,500),(500,600),(400,600)):
        first={r["rel_path"]:r for r in rows_by_epoch[a]};second={r["rel_path"]:r for r in rows_by_epoch[b]}
        paired[f"{b}_minus_{a}"]={m:dict(mean_delta=statistics.mean(float(second[p][m])-float(first[p][m]) for p in expected),
                improved_count=sum((float(second[p][m])-float(first[p][m]))*(1 if m=="SSIM" else -1)>0 for p in expected))
                for m in METRICS}
    corrmaps=list((root/"corrmaps").rglob("corrmap.npz"))
    if len(corrmaps)!=63:raise ValueError("Expected exactly 63 shared correction maps")
    images=visuals(repo,root,plan)
    write(root/"comparison.json",dict(status="complete",patients=63,cases_per_epoch=259,total_valid_scores=777,
          results=results,paired=paired,protocol=provenance["protocol"],
          checkpoint_sha256={e:c["sha256"] for e,c in provenance["checkpoints"].items()},
          corrmap_sha256={str(p.relative_to(root)):sha(p) for p in corrmaps},visuals=images))
    lines=["# Unified checkpoint validation","",
           "Complete: 63 held-out validation patients; 259 cases per epoch; 777 valid scores.",
           "The running training experiment and 62-patient test split are unchanged.",
           "Full-volume official utilities, no VENC scaling, GT-derived MSAC correction maps shared across epochs.",
           "Local held-out results, not public-leaderboard parity scores.","",
           "| Epoch | SSIM higher | nRMSE lower | RelErr lower | AngErr lower | ComplexDiffErr lower |",
           "|---|---|---|---|---|---|"]
    for epoch in EPOCHS:
        row=results[str(epoch)]["overall"]["metrics"]
        lines.append("| "+str(epoch)+" | "+" | ".join(f"{row[m]:.6f}" for m in METRICS)+" |")
    lines += ["","Overall is patient-acceleration weighted. comparison.json includes patient-macro means, organ/acceleration breakdowns and paired deltas.",""]
    lines += [f"![{Path(p).stem}](visuals/{Path(p).name})" for p in images]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n")
    write(root/"completion_receipt.json",dict(status="complete",valid_scores=777,
          report_sha256=sha(root/"REPORT.md"),comparison_sha256=sha(root/"comparison.json"),
          job_id=os.environ.get("SLURM_JOB_ID")))
    print(json.dumps(dict(status="COMPLETE",valid_scores=777,report=str(root/"REPORT.md"))),flush=True)

def record_job(root,stage,job):
    if not job or not job.isdigit():raise ValueError("Numeric job identity required")
    with lock(root,"jobs"):
        path=root/"jobs.json";jobs=read(path) if path.exists() else []
        if any(j["stage"]==stage and j["job_id"]!=job for j in jobs):raise ValueError("Reconcile existing stage before retry")
        if not any(j["stage"]==stage and j["job_id"]==job for j in jobs):
            jobs.append(dict(stage=stage,job_id=job,recorded_unix=time.time()))
        write(path,jobs)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","amend-wrapper-pin","pilot","make-shards","infer-array","gt","score","report","record-job","status"))
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,default=DATA)
    parser.add_argument("--task-index",type=int)
    parser.add_argument("--attempt-label",default="full")
    parser.add_argument("--job-stage");parser.add_argument("--job-id")
    parser.add_argument("--amendment-reason")
    args=parser.parse_args()
    repo=args.repo.resolve();root=safe_root(repo,args.run_root,args.data_root)
    os.environ.update(WANDB_MODE="disabled",WANDB_DISABLED="true",HF_HUB_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    if args.stage=="prepare":prepare(repo,root,args.data_root)
    elif args.stage=="amend-wrapper-pin":amend_wrapper_pin(repo,root,args.amendment_reason)
    elif args.stage=="pilot":pilot(repo,root)
    elif args.stage=="make-shards":make_shards(repo,root)
    elif args.stage=="infer-array":infer_array(repo,root,args.task_index,args.attempt_label)
    elif args.stage=="gt":
        verify(repo,root)
        helper(repo,"build_4dflow_validation_gt").build_case(root/"gt_plan.json",args.task_index,root/"gt",8,False,True)
    elif args.stage=="score":score(repo,root)
    elif args.stage=="report":report(repo,root)
    elif args.stage=="record-job":record_job(root,args.job_stage,args.job_id)
    else:print(json.dumps({n:read(root/n) for n in ("jobs.json","prepared.json","pilot.json","shards.json","completion_receipt.json") if (root/n).exists()},indent=2))

if __name__=="__main__":main()
