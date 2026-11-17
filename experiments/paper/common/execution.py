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
from torch.utils.data import DataLoader

from .data import TrainSlabs
from .manifest import read_manifest, validate_train_index
from .path_mapping import protected_case_paths
from .safety import OutputGuard, ROOT, sha256


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true", help="Train-only 1-3 optimizer steps, no full experiment")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--probe-flow-loss", action="store_true", help="Smoke-only: test the fully enabled flow-loss curriculum")
    return p


def execute(args, factory):
    cfg = json.loads(args.config.read_text())
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("This isolated paper entry is single-process; do not launch it with torchrun")
    paper = cfg["paper"]
    if args.probe_flow_loss and not args.smoke:
        raise ValueError("--probe-flow-loss is a smoke-only override")
    if args.probe_flow_loss:
        paper["_smoke_flow_ramp_probe"] = True
    if paper["validation_enabled"] or paper["validation_manifest"] is not None:
        raise ValueError("Validation is disabled until an explicit independent validation manifest is provided")
    if args.smoke and not 1 <= args.steps <= 3:
        raise ValueError("Readiness smoke tests permit only 1-3 optimizer steps")
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
              "validation_enabled": False, "steps": []}
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
        dataset = TrainSlabs(ROOT / paper["train_index"], seed=paper["seed"],
                             num_slices=paper.get("smoke_num_slices", paper["num_slices"]) if args.smoke else paper["num_slices"],
                             num_frames=paper.get("smoke_num_frames", paper["num_frames"]) if args.smoke else paper["num_frames"],
                             samples_per_case=paper["samples_per_case"], accelerations=paper["accelerations"], smoke=args.smoke)
        validate_train_index(dataset.cases, source_cases, require_complete=not args.smoke)
        source_metadata = {}
        for case in dataset.cases:
            for source in case["files"].values():
                stat = Path(source).stat()
                source_metadata[source] = (stat.st_size, stat.st_mtime_ns)
        loader = DataLoader(dataset, batch_size=1, shuffle=not args.smoke,
                            num_workers=int(cfg.get("num_workers", 0)), generator=torch.Generator().manual_seed(paper["seed"]))
        guard.json(output / "effective_config.json", cfg)
        adapter = factory(cfg, device)
        model, optimizer = adapter.model, adapter.optimizer
        scheduler = getattr(adapter, "scheduler", None)
        accumulation = int(paper.get("gradient_accumulation", 1))
        if accumulation < 1 or (accumulation != 1 and getattr(adapter, "backward_in_step", False)):
            raise ValueError("Invalid accumulation, or native backward adapter requires accumulation=1")
        report["parameters"] = sum(p.numel() for p in model.parameters())
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.train()
        start_epoch, step = 0, 0
        if paper.get("checkpoint"):
            state = torch.load(paper["checkpoint"], map_location=device, weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            if hasattr(adapter, "load_state_dict"):
                adapter.load_state_dict(state.get("adapter", {}))
            start_epoch, step = state["epoch"] + 1, state["step"]
            report["resumed_from"] = paper["checkpoint"]
            report["resume_epoch"], report["resume_step"] = start_epoch, step
            del state
        end_epoch = start_epoch + 1 if args.smoke else int(cfg["num_epochs"])
        if end_epoch <= start_epoch:
            raise ValueError("Checkpoint already reached num_epochs; increase the total epoch budget explicitly")
        for epoch in range(start_epoch, end_epoch):
            dataset.epoch = epoch
            pending, accumulated_loss = 0, 0.0
            for batch_index, cpu_batch in enumerate(loader):
                before = {p: p.detach().clone() for p in model.parameters() if p.requires_grad and p.numel() <= 100 and len(report["steps"]) == 0}
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu_batch.items()}
                if pending == 0:
                    optimizer.zero_grad(set_to_none=True)
                loss, prediction, components = adapter.step(batch, epoch, step)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: {loss}")
                if prediction.shape != batch["target"].shape or not torch.is_complex(prediction):
                    raise AssertionError(f"Output must be four-encoding complex [B,E,T,S,Z,Y]: {prediction.shape}")
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Non-finite complex prediction")
                if not getattr(adapter, "backward_in_step", False):
                    (loss / accumulation).backward()
                pending += 1
                accumulated_loss += float(loss.detach())
                if pending < accumulation and batch_index + 1 < len(loader):
                    continue
                gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Missing or non-finite gradients")
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if grad_norm <= 0:
                    raise AssertionError("All gradients are zero")
                optimizer.step()
                if hasattr(adapter, "after_step"):
                    adapter.after_step()
                record = {"epoch": epoch, "step": step, "case": cpu_batch["case_id"],
                          "source": cpu_batch["source"], "input_shape": list(batch["kspace"].shape),
                          "output_shape": list(prediction.shape), "native_shape": cpu_batch["native_shape"].tolist(),
                          "acceleration_nominal": cpu_batch["acceleration"].tolist(),
                          "acceleration_actual": float(batch["mask"].numel() / batch["mask"].sum()),
                          "venc_order": cpu_batch["venc_order"], "slice_indices": cpu_batch["slice_indices"].tolist(),
                          "frame_indices": cpu_batch["frame_indices"].tolist(), "loss": accumulated_loss / pending,
                          "accumulated_microbatches": pending,
                          "gradient_norm": float(grad_norm), "parameters_with_grad": len(gradients),
                          "small_parameters_changed": sum(not torch.equal(old, p.detach()) for p, old in before.items()) if before else None,
                          "components": {k: float(v) for k, v in components.items()}}
                record["learning_rate"] = optimizer.param_groups[0]["lr"]
                record["metadata_shape_warning"] = cpu_batch["metadata_shape_warning"]
                record["roi_voxels"] = float(batch["segmask"].sum())
                mixer_gradients = [p.grad for name, p in model.named_parameters()
                                   if "flowvn_mixer.regularizers." in name and p.grad is not None]
                if mixer_gradients:
                    record["mixer_regularizer_grad_abs_sum"] = sum(float(g.detach().abs().sum()) for g in mixer_gradients)
                    record["mixer_scale_values"] = {name: float(p.detach()) for name, p in model.named_parameters()
                                                    if name.endswith("flowvn_mixer.scale")}
                report["steps"].append(record)
                pending, accumulated_loss = 0, 0.0
                if args.smoke or step % 10 == 0:
                    print(json.dumps(record), flush=True)
                if not args.smoke:
                    report["steps"] = report["steps"][-10:]
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
                    # Export only a real train window; never label this as a complete case reconstruction.
                    destination = guard.check(output / "train_window_complex.npy")
                    with destination.open("wb") as stream:
                        np.save(stream, inferred.detach().cpu().numpy())
                    if len(report["steps"]) >= args.steps:
                        break
            if scheduler is not None:
                scheduler.step()
            checkpoint = guard.check(output / "latest.pt")
            temporary = guard.check(output / "latest.pt.tmp")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "adapter": adapter.state_dict() if hasattr(adapter, "state_dict") else {},
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "epoch": epoch, "step": step, "config": cfg, "seed": paper["seed"]}, temporary)
            temporary.replace(checkpoint)
            report["checkpoint"] = str(checkpoint)
            report["checkpoint_bytes"] = checkpoint.stat().st_size
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
        if not report["steps"]:
            raise RuntimeError("No training step completed")
        for source, before in source_metadata.items():
            stat = Path(source).stat()
            if before != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"Dataset file metadata changed during preflight: {source}")
        report["read_only_inputs_unchanged"] = len(source_metadata)
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["seconds"] = time.monotonic() - start
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            report["peak_gpu_gib"] = torch.cuda.max_memory_allocated(args.device) / 1024**3
        guard.json(output / "report.json", report)
    if report["status"] != "PASS":
        print(report.get("traceback", report["error"]), flush=True)
        raise SystemExit(1)
    print(f"PASS {paper['name']} report={output / 'report.json'}", flush=True)
