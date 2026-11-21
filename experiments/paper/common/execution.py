"""Minimal auditable loop shared by isolated experiment adapters."""

from __future__ import annotations

import argparse
import json
import random
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import checkpoints, distributed
from .data import TrainSlabs, h5_root
from .manifest import read_manifest, validate_train_index
from .online_sampling import sampling_log
from .path_mapping import protected_case_paths
from .safety import OutputGuard, ROOT, sha256
from .train_loader import CaseWindowLoader


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true", help="Train-only 1-3 optimizer steps, no full experiment")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--smoke-case-id", help="Use a specific Train case for a memory/throughput probe")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume-latest", action="store_true", help="Resume this run's latest.pt when it exists")
    p.add_argument("--probe-flow-loss", action="store_true", help="Smoke-only: test the fully enabled flow-loss curriculum")
    return p


def execute(args, factory):
    cfg = json.loads(args.config.read_text())
    # Under torchrun every rank runs this loop on its own share of cases and applies the rank-averaged gradient.
    world, rank, local_rank = distributed.environment()
    main = rank == 0
    if world > 1:
        if args.smoke:
            raise ValueError("Smoke tests are single-process; do not launch them with torchrun")
        args.device = f"cuda:{local_rank}"
    paper = cfg["paper"]
    if args.probe_flow_loss and not args.smoke:
        raise ValueError("--probe-flow-loss is a smoke-only override")
    if args.probe_flow_loss:
        paper["_smoke_flow_ramp_probe"] = True
    if paper["validation_enabled"] or paper["validation_manifest"] is not None:
        raise ValueError("Validation is disabled until an explicit independent validation manifest is provided")
    if args.smoke and not 1 <= args.steps <= 3:
        raise ValueError("Readiness smoke tests permit only 1-3 optimizer steps")
    if getattr(args, "smoke_case_id", None) and not args.smoke:
        raise ValueError("--smoke-case-id requires --smoke")
    manifest_path = ROOT / paper["manifest"]
    source_mapping = ROOT / paper["source_mapping"] if paper.get("source_mapping") else None
    source_cases, _ = read_manifest(manifest_path, source_mapping)
    protected = [manifest_path.parent, *protected_case_paths(source_cases), *paper["protected_roots"]]
    if source_mapping is not None:
        protected.append(source_mapping.parent)
    guard = OutputGuard(protected)
    output = guard.directory(ROOT / paper["output_dir"] / ("smoke" if args.smoke else "train"))
    report = {"experiment": paper["name"], "method": paper["method"], "status": "FAIL",
              "seed": paper["seed"], "smoke": args.smoke, "device": args.device,
              "config_sha256": sha256(args.config), "test_data_used": False,
              "source_mapping_sha256": sha256(source_mapping) if source_mapping else None,
              "validation_enabled": False, "steps": [], "world_size": world}
    report["runtime_overrides"] = {"fully_enabled_flow_loss": args.probe_flow_loss,
                                    "smoke_roi_sampling": "max_roi_train_slice" if args.smoke else None}
    start = time.monotonic()
    try:
        torch.manual_seed(paper["seed"])
        random.seed(paper["seed"])
        np.random.seed(paper["seed"])
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; choose --device cpu or run on the GPU server")
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
        if world > 1:
            distributed.initialize(world, rank, local_rank)
        dataset = TrainSlabs(ROOT / paper["train_index"], seed=paper["seed"],
                             num_slices=paper.get("smoke_num_slices", paper["num_slices"]) if args.smoke else paper["num_slices"],
                             num_frames=paper.get("smoke_num_frames", paper["num_frames"]) if args.smoke else paper["num_frames"],
                             samples_per_case=paper["samples_per_case"], accelerations=paper["accelerations"], smoke=args.smoke,
                             smoke_case_id=getattr(args, "smoke_case_id", None),
                             online_undersampling=paper.get("online_undersampling"),
                             h5_root=h5_root(paper))
        validate_train_index(dataset.cases, source_cases, require_complete=not args.smoke)
        if dataset.online_sampler is not None:
            report["online_undersampling"] = dataset.online_sampler.provenance()
        source_metadata = {}
        for case in dataset.cases if main else ():
            # The HDF5 backend never opens the raw files, so watch the stores it does read.
            sources = [dataset.h5_root / "cases" / (case["case_id"] + ".h5")] if dataset.h5_root else case["files"].values()
            for source in map(str, sources):
                stat = Path(source).stat()
                source_metadata[source] = (stat.st_size, stat.st_mtime_ns)
        loader = CaseWindowLoader(dataset, device, paper["method"], shuffle=not args.smoke, world=world, rank=rank,
                                  **paper.get("loader", {}))
        if main:
            guard.json(output / "effective_config.json", cfg)
        adapter = factory(cfg, device)
        model, optimizer = adapter.model, adapter.optimizer
        scheduler = getattr(adapter, "scheduler", None)
        accumulation = int(paper.get("gradient_accumulation", 1))
        if accumulation < 1 or (accumulation != 1 and getattr(adapter, "backward_in_step", False)):
            raise ValueError("Invalid accumulation, or native backward adapter requires accumulation=1")
        if hasattr(adapter, "pretrained"):
            report["pretrained"] = adapter.pretrained
            if main:
                print(f"PRETRAINED {json.dumps(adapter.pretrained)}", flush=True)
        report["parameters"] = sum(p.numel() for p in model.parameters())
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.train()
        start_epoch, step, resume_cases = 0, 0, 0
        checkpoint = guard.check(output / "latest.pt")
        candidates = [Path(paper["checkpoint"])] if paper.get("checkpoint") else []
        if getattr(args, "resume_latest", False):
            # latest.pt first; an unreadable one falls back to the newest epoch_NNNN.pt.
            candidates = [p for p in checkpoints.resume_candidates(output) if p.is_file()] or candidates
        if not args.smoke and not candidates and checkpoint.exists():
            raise FileExistsError(f"Training checkpoint already exists: {checkpoint}; use --resume-latest or a new output")
        failures = []
        for resume in candidates:
            try:
                state = checkpoints.load(resume)
                model.load_state_dict(state["model"], strict=True)
                optimizer.load_state_dict(state["optimizer"])
                if scheduler is not None and state.get("scheduler") is not None:
                    scheduler.load_state_dict(state["scheduler"])
                if hasattr(adapter, "load_state_dict"):
                    adapter.load_state_dict(state.get("adapter", {}))
            except Exception as exc:
                failures.append(f"{resume}: {type(exc).__name__}: {exc}")
                print(f"UNREADABLE_CHECKPOINT {failures[-1]}", flush=True)
                continue
            start_epoch, step = state["epoch"] + 1, state["step"]
            # A checkpoint written inside an epoch records how many of each rank's cases were already trained.
            resume_cases = int(state.get("cases_done", 0))
            report["resumed_from"] = str(resume)
            report["resume_epoch"], report["resume_step"], report["resume_cases"] = start_epoch, step, resume_cases
            del state
            break
        else:
            if candidates:
                raise RuntimeError(f"No readable checkpoint among {len(candidates)}: {failures}")
        if failures:
            report["unreadable_checkpoints"] = failures
        if world > 1:
            distributed.all_ranks_equal([start_epoch, step, resume_cases, *distributed.fingerprint(model)], "resume point or initial weights")
        end_epoch = start_epoch + 1 if args.smoke else int(cfg["num_epochs"])
        if end_epoch <= start_epoch:
            raise ValueError("Checkpoint already reached num_epochs; increase the total epoch budget explicitly")
        save_every = paper.get("checkpoint_every_epochs", 1)
        if type(save_every) is not int or save_every < 1:
            raise ValueError("paper.checkpoint_every_epochs must be a positive integer")
        checkpoints.policy(paper)
        report["batchnorm_modules"] = sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
        # PAPER_DEADLINE (epoch seconds, set by the job launcher): past it, save at the next case boundary and leave.
        deadline = float(os.environ.get("PAPER_DEADLINE", "inf"))
        # Inside long epochs, also save every PAPER_MIDEPOCH_MINUTES so a crash loses little work.
        mid_interval = 60 * float(os.environ.get("PAPER_MIDEPOCH_MINUTES", "30"))
        # Case boundaries are safe resume points only when no accumulated gradient spans them.
        mid_epoch = not args.smoke and loader.batches_per_case % accumulation == 0
        last_save, interrupted = time.time(), False

        def snapshot(completed_epoch, cases_done):
            return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "adapter": adapter.state_dict() if hasattr(adapter, "state_dict") else {},
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": completed_epoch, "cases_done": cases_done, "step": step, "config": cfg,
                    "seed": paper["seed"], "world_size": world}

        for epoch in range(start_epoch, end_epoch):
            epoch_start = time.monotonic()
            loader.skip_cases = resume_cases if epoch == start_epoch else 0
            dataset.epoch = epoch
            case_sampling = {}
            pending, accumulated_loss = 0, 0.0
            timing = {"windows": 0, "loader_wait_seconds": 0.0, "train_step_seconds": 0.0}
            with loader.epoch_batches() as batches:
                batch_done = time.perf_counter()
                for batch_index, cpu_batch in enumerate(batches, start=loader.skip_cases * loader.batches_per_case):
                    batch_start = time.perf_counter()
                    timing["loader_wait_seconds"] += batch_start - batch_done
                    timing["windows"] += len(cpu_batch["case_id"])
                    sampling = sampling_log(cpu_batch)
                    if sampling:
                        case_id = cpu_batch["case_id"][0]
                        if case_sampling.setdefault(case_id, sampling) != sampling:
                            raise RuntimeError(f"Inconsistent online mask settings for {case_id} in epoch {epoch}")
                    before = {p: p.detach().clone() for p in model.parameters() if p.requires_grad and p.numel() <= 100} if args.smoke and not report["steps"] else {}
                    batch = loader.to_device(cpu_batch, device)
                    if pending == 0:
                        optimizer.zero_grad(set_to_none=True)
                    loss, prediction, components = adapter.step(batch, epoch, step)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss: {loss}")
                    if prediction.shape != cpu_batch["target"].shape or not torch.is_complex(prediction):
                        raise AssertionError(f"Output must be four-encoding complex [B,E,T,S,Z,Y]: {prediction.shape}")
                    if args.smoke and not torch.isfinite(prediction).all():
                        raise FloatingPointError("Non-finite complex prediction")
                    if not getattr(adapter, "backward_in_step", False):
                        (loss / accumulation).backward()
                    pending += 1
                    accumulated_loss = accumulated_loss + loss.detach()
                    if pending < accumulation and batch_index + 1 < len(loader):
                        batch_done = time.perf_counter()
                        timing["train_step_seconds"] += batch_done - batch_start
                        continue
                    if world > 1:
                        # Time spent waiting for slower ranks, kept apart from this rank's own compute.
                        torch.cuda.synchronize(device)
                        sync_start = time.perf_counter()
                        distributed.sync_gradients(model.parameters(), world)
                        torch.cuda.synchronize(device)
                        timing["sync_seconds"] = timing.get("sync_seconds", 0.0) + time.perf_counter() - sync_start
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    if args.smoke and grad_norm <= 0:
                        raise AssertionError("All gradients are zero or absent")
                    optimizer.step()
                    if hasattr(adapter, "after_step"):
                        adapter.after_step()
                    if args.smoke and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timing["train_step_seconds"] += time.perf_counter() - batch_start
                    if main and (args.smoke or step % 10 == 0):
                        record = {"epoch": epoch, "step": step, "case": cpu_batch["case_id"],
                                  "source": cpu_batch["source"], "input_shape": list(batch["kspace"].shape),
                                  "output_shape": list(prediction.shape), "native_shape": cpu_batch["native_shape"].tolist(),
                                  "acceleration_nominal": cpu_batch["acceleration"].tolist(),
                                  "acceleration_actual": float(cpu_batch["mask"].numel() / cpu_batch["mask"].sum()),
                                  "venc_order": cpu_batch["venc_order"], "slice_indices": cpu_batch["slice_indices"].tolist(),
                                  "frame_indices": cpu_batch["frame_indices"].tolist(), "loss": float(accumulated_loss / pending),
                                  "accumulated_microbatches": pending, "gradient_norm": float(grad_norm),
                                  "parameters_with_grad": sum(p.grad is not None for p in model.parameters()),
                                  "small_parameters_changed": sum(not torch.equal(old, p.detach()) for p, old in before.items()) if before else None,
                                  "components": {k: float(v) for k, v in components.items()},
                                  "learning_rate": optimizer.param_groups[0]["lr"],
                                  "metadata_shape_warning": cpu_batch["metadata_shape_warning"],
                                  "roi_voxels": float(cpu_batch["segmask"].sum())}
                        if sampling:
                            record["sampling"] = sampling
                        if args.smoke:
                            mixer_gradients = [p.grad for name, p in model.named_parameters()
                                               if "flowvn_mixer.regularizers." in name and p.grad is not None]
                            if mixer_gradients:
                                record["mixer_regularizer_grad_abs_sum"] = sum(float(g.detach().abs().sum()) for g in mixer_gradients)
                                record["mixer_scale_values"] = {name: float(p.detach()) for name, p in model.named_parameters()
                                                                if name.endswith("flowvn_mixer.scale")}
                        report["steps"].append(record)
                        if not args.smoke:
                            report["steps"] = report["steps"][-10:]
                        print(json.dumps(record), flush=True)
                    pending, accumulated_loss = 0, 0.0
                    step += 1
                    if args.smoke:
                        model.eval()
                        with torch.no_grad():
                            inferred = adapter.reconstruct(batch)
                        if inferred.shape != prediction.shape or not torch.isfinite(inferred).all():
                            raise AssertionError("Inference output shape/finite-value check failed")
                        report["inference_shape"] = list(inferred.shape)
                        report["inference_complex"] = bool(torch.is_complex(inferred))
                        report["inference_uses_ema"] = bool(getattr(adapter, "inference_uses_ema", False))
                        model.train()
                        destination = guard.check(output / "train_window_complex.npy")
                        with destination.open("wb") as stream:
                            np.save(stream, inferred.detach().cpu().numpy())
                        if len(report["steps"]) >= args.steps:
                            break
                    if mid_epoch and (batch_index + 1) % loader.batches_per_case == 0 and batch_index + 1 < len(loader):
                        stop, save = time.time() > deadline, main and time.time() - last_save > mid_interval
                        if world > 1:
                            stop, save = distributed.any_rank(stop, save)
                        if stop or save:
                            if world > 1:
                                distributed.all_ranks_equal([step, *distributed.fingerprint(model)], f"weights inside epoch {epoch}")
                            if main:
                                checkpoints.write(snapshot(epoch - 1, (batch_index + 1) // loader.batches_per_case), checkpoint, guard)
                                print(f"MID_EPOCH_SAVE epoch={epoch} batch={batch_index + 1}/{len(loader)} stop={stop}", flush=True)
                            last_save = time.time()
                        if stop:
                            interrupted = True
                            break
                    batch_done = time.perf_counter()
            if interrupted:
                report["stopped_for_deadline"] = True
                break
            timing["epoch_seconds"] = time.monotonic() - epoch_start
            report["epoch_timing"] = timing
            if main:
                print(f"EPOCH {epoch + 1}/{end_epoch} {json.dumps(timing)}", flush=True)
            if scheduler is not None:
                scheduler.step()
            if world > 1:
                distributed.all_ranks_equal([step, *distributed.fingerprint(model)], f"weights after epoch {epoch}")
                if case_sampling:
                    # One append-only file per rank; masks are also reproducible from seed, case ID and epoch.
                    path = guard.check(guard.directory(output / "sampling") / f"rank_{rank:03d}.jsonl")
                    with path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"epoch": epoch, "cases": case_sampling}) + "\n")
            elif case_sampling:
                guard.json(output / "sampling" / f"epoch_{epoch:04d}.json",
                           {"epoch": epoch, "generator": report["online_undersampling"], "cases": case_sampling})
            # Past the job's deadline: save this epoch and leave; the next job resumes from here.
            out_of_time = epoch + 1 < end_epoch and time.time() > deadline
            if world > 1:
                out_of_time, = distributed.any_rank(out_of_time)
            if not args.smoke and (epoch + 1) % save_every != 0 and epoch + 1 != end_epoch and not out_of_time:
                continue
            if main:
                state = snapshot(epoch, 0)
                if args.smoke:
                    checkpoints.write(state, checkpoint, guard)
                else:
                    checkpoints.save_epoch(state, output, guard, epoch + 1, paper)
                del state
                last_save = time.time()
                report["checkpoint"] = str(checkpoint)
                report["checkpoint_bytes"] = checkpoint.stat().st_size
                report["completed_epochs"] = epoch + 1
            if out_of_time:
                report["stopped_for_deadline"] = True
                break
            if args.smoke:
                restored = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
                signature = {k: tuple(v.shape) for k, v in model.state_dict().items()}
                if signature != {k: tuple(v.shape) for k, v in restored["model"].items()}:
                    raise AssertionError("Checkpoint model shape signature changed during serialization")
                if not restored["optimizer"]["state"]:
                    raise AssertionError("Checkpoint lacks populated optimizer state")
                report["checkpoint_readback"] = {"model_shapes_match": True, "optimizer_state_populated": True,
                                                  "scheduler_saved": restored["scheduler"] is not None}
                del restored
            if args.smoke:
                break
        if main and not report["steps"]:
            raise RuntimeError("No training step completed")
        for source, before in source_metadata.items():
            stat = Path(source).stat()
            if before != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"Dataset file metadata changed during preflight: {source}")
        report["read_only_inputs_unchanged"] = len(source_metadata)
        if main and report.get("stopped_for_deadline"):
            print(f"DEADLINE_STOP epoch={epoch + 1}/{end_epoch}", flush=True)
        report["status"] = "PASS"
        if main and not args.smoke and report.get("completed_epochs") == end_epoch:
            # Marker for the job chain: the full epoch budget is trained.
            guard.json(output / "DONE.json", {"experiment": paper["name"], "epochs": end_epoch, "step": step,
                                              "checkpoint": str(checkpoint), "world_size": world})
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["seconds"] = time.monotonic() - start
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            report["peak_gpu_gib"] = torch.cuda.max_memory_allocated(args.device) / 1024**3
        # Rank 0 owns report.json; a failing worker rank leaves its own file for diagnosis.
        if main or report["status"] != "PASS":
            guard.json(output / ("report.json" if main else f"report.rank{rank:03d}.json"), report)
    if report["status"] != "PASS":
        print(report.get("traceback", report["error"]), flush=True)
        # A hard exit also takes down peers blocked in a collective call.
        if world > 1:
            os._exit(1)
        raise SystemExit(1)
    if main:
        print(f"PASS {paper['name']} report={output / 'report.json'}", flush=True)
