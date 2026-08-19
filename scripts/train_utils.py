# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import resource
import socket
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from monai.apps.reconstruction.complex_utils import convert_to_tensor_complex
from monai.apps.reconstruction.transforms.dictionary import ExtractDataKeyFromMetaKeyd
from monai.data.fft_utils import ifftn_centered
from monai.transforms import Compose, EnsureTyped, Identityd, Lambdad, LoadImaged, RandFlipd, ResizeWithPadOrCropd
from four_dflow_augmentation import FourDFlowOnlineAugmentd
from mri_data.data_utils import get_reader
from muon import MuonWithAuxAdam
from transforms import *
from utils import *


PERFORMANCE_TIMING_FIELDS = (
    "loader_wait_ms",
    "batch_setup_ms",
    "compute_cost_allreduce_ms",
    "num_samples_allreduce_ms",
    "num_batches_allreduce_ms",
    "prep_h2d_ms",
    "forward_ms",
    "loss_ms",
    "nan_guard_allreduce_ms",
    "backward_ddp_ms",
    "optimizer_ms",
    "loss_allreduce_ms",
    "loss_components_allreduce_ms",
    "step_total_ms",
)

WORKER_TIMING_FIELDS = (
    "worker_total_ms",
    "load_total_ms",
    "json_open_ms",
    "input_read_ms",
    "target_read_ms",
    "mask_read_ms",
    "coilmap_read_ms",
    "input_complex_ms",
    "target_complex_ms",
    "coilmap_complex_ms",
    "mask_hybrid_ms",
    "to_tensor_complex_ms",
    "ensure_typed_ms",
    "ifft_ms",
    "augmentation_ms",
    "rearrange_normalize_ms",
)

POST_WORKER_TIMING_FIELDS = (
    "collate_ms",
    "pre_worker_start_ms",
    "worker_on_critical_path_ms",
    "pre_collate_gap_ms",
    "collate_on_critical_path_ms",
    "post_collate_on_critical_path_ms",
    "critical_path_unattributed_ms",
    "ready_before_request_ms",
    "collate_to_consumer_ms",
)

ALL_PERFORMANCE_TIMING_FIELDS = (
    PERFORMANCE_TIMING_FIELDS + WORKER_TIMING_FIELDS + POST_WORKER_TIMING_FIELDS
)


def start_phase_timing(enabled, synchronize_cuda):
    if not enabled:
        return None
    if synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def stop_phase_timing(metrics, name, started, synchronize_cuda):
    if started is None:
        return
    if synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    metrics[name] = metrics.get(name, 0.0) + (time.perf_counter() - started) * 1000.0


def gather_rank_timing_stats(
    metrics,
    fields,
    final_shape,
    compute_cost,
    device,
    rank,
    world_size,
    is_ddp,
    synchronize_cuda,
):
    height, width = int(final_shape[-2]), int(final_shape[-1])
    local_values = [float(metrics.get(name, 0.0)) for name in fields]
    local_values.extend((float(height), float(width), float(compute_cost)))
    local_tensor = torch.tensor(local_values, dtype=torch.float32, device=device)

    gather_started = start_phase_timing(True, synchronize_cuda)
    if is_ddp:
        gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor)
        gathered_tensor = torch.stack(gathered)
    else:
        gathered_tensor = local_tensor.unsqueeze(0)
    gather_metrics = {}
    stop_phase_timing(gather_metrics, "timing_gather_ms", gather_started, synchronize_cuda)

    if rank != 0:
        return None, gather_metrics["timing_gather_ms"]

    values = gathered_tensor.detach().cpu().numpy()
    metadata_offset = len(fields)
    stats = {}
    for field_idx, name in enumerate(fields):
        field_values = values[:, field_idx]
        max_rank = int(np.argmax(field_values))
        stats[name] = {
            "mean_ms": float(np.mean(field_values)),
            "max_ms": float(field_values[max_rank]),
            "max_rank": max_rank,
            "height": int(values[max_rank, metadata_offset]),
            "width": int(values[max_rank, metadata_offset + 1]),
            "compute_cost": int(values[max_rank, metadata_offset + 2]),
        }
    return stats, gather_metrics["timing_gather_ms"]


def format_rank_timing_stats(stats, fields, include_rank_details):
    parts = []
    for name in fields:
        field = stats[name]
        detail = f"{name}={field['mean_ms']:.1f}/{field['max_ms']:.1f}ms"
        if include_rank_details:
            detail += f"@r{field['max_rank']}[{field['height']}x{field['width']}]"
        parts.append(detail)
    return " ".join(parts)


def log_step_performance(
    step_timings,
    final_shape,
    compute_cost,
    device,
    rank,
    world_size,
    is_ddp,
    synchronize_cuda,
    include_rank_details,
    global_step,
    writer=None,
):
    timing_stats, timing_gather_ms = gather_rank_timing_stats(
        step_timings,
        ALL_PERFORMANCE_TIMING_FIELDS,
        final_shape,
        compute_cost,
        device,
        rank,
        world_size,
        is_ddp,
        synchronize_cuda,
    )
    if rank != 0:
        return None

    print(
        f"[perf][step={global_step}] "
        f"{format_rank_timing_stats(timing_stats, PERFORMANCE_TIMING_FIELDS, include_rank_details)} "
        f"timing_gather={timing_gather_ms:.1f}ms"
    )
    print(
        f"[perf][worker_step={global_step}] "
        f"{format_rank_timing_stats(timing_stats, WORKER_TIMING_FIELDS, include_rank_details)}"
    )
    print(
        f"[perf][ipc_step={global_step}] "
        f"{format_rank_timing_stats(timing_stats, POST_WORKER_TIMING_FIELDS, include_rank_details)}"
    )
    if writer is not None:
        for timing_name, timing_value in timing_stats.items():
            writer.add_scalar(
                f"perf_step/{timing_name}_rank_mean_ms",
                timing_value["mean_ms"],
                global_step,
            )
            writer.add_scalar(
                f"perf_step/{timing_name}_rank_max_ms",
                timing_value["max_ms"],
                global_step,
            )
            writer.add_scalar(
                f"perf_step/{timing_name}_max_rank",
                timing_value["max_rank"],
                global_step,
            )
        writer.add_scalar("perf_step/timing_gather_ms", timing_gather_ms, global_step)
    return timing_stats


def log_epoch_performance(epoch_timing_samples, epoch, writer, epoch_log):
    if not epoch_timing_samples:
        return

    for timing_name in ALL_PERFORMANCE_TIMING_FIELDS:
        rank_mean_ms = float(
            np.mean([sample[timing_name]["mean_ms"] for sample in epoch_timing_samples])
        )
        rank_peak_ms = float(
            np.max([sample[timing_name]["max_ms"] for sample in epoch_timing_samples])
        )
        writer.add_scalar(f"perf_epoch/{timing_name}_rank_mean_ms", rank_mean_ms, epoch + 1)
        writer.add_scalar(f"perf_epoch/{timing_name}_rank_peak_ms", rank_peak_ms, epoch + 1)
        epoch_log[f"perf/{timing_name}_rank_mean_ms"] = rank_mean_ms
        epoch_log[f"perf/{timing_name}_rank_peak_ms"] = rank_peak_ms

    for label, fields in (
        ("epoch", PERFORMANCE_TIMING_FIELDS),
        ("worker_epoch", WORKER_TIMING_FIELDS),
        ("ipc_epoch", POST_WORKER_TIMING_FIELDS),
    ):
        print(
            f"[perf][{label}={epoch + 1}] samples={len(epoch_timing_samples)} "
            + " ".join(
                f"{timing_name}="
                f"{epoch_log[f'perf/{timing_name}_rank_mean_ms']:.1f}/"
                f"{epoch_log[f'perf/{timing_name}_rank_peak_ms']:.1f}ms"
                for timing_name in fields
            )
        )


def build_lightweight_performance_event(
    *,
    rank,
    epoch,
    global_step,
    batch_index,
    run_tag,
    file_name,
    final_shape,
    request_ns,
    received_ns,
    finished_ns,
):
    return {
        "job_id": os.getenv("SLURM_JOB_ID"),
        "rank": rank,
        "epoch": epoch + 1,
        "global_step": global_step,
        "batch_index": batch_index,
        "run_tag": run_tag,
        "filename": str(file_name),
        "final_shape": final_shape,
        "loader_wait_ms": (received_ns - request_ns) / 1.0e6,
        "compute_ms": (finished_ns - received_ns) / 1.0e6,
        "batch_interval_ms": (finished_ns - request_ns) / 1.0e6,
    }


def gather_and_log_lightweight_performance(
    local_events,
    rank,
    world_size,
    is_ddp,
    outpath,
    epoch,
    run_tag,
):
    payload = {"rank": rank, "events": local_events}
    if is_ddp:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
    else:
        gathered = [payload]

    if rank != 0:
        return

    events = [event for item in gathered if item is not None for event in item["events"]]
    events.sort(key=lambda event: (event["batch_index"], event["rank"]))
    performance_dir = Path(outpath) / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    event_path = performance_dir / "lightweight_events.jsonl"
    if events:
        with event_path.open("a", encoding="utf-8") as event_file:
            for event in events:
                event_file.write(json.dumps(event, sort_keys=True) + "\n")

    warm = [event for event in events if event["batch_index"] > 0]
    intervals = np.asarray([event["batch_interval_ms"] for event in warm], dtype=np.float64)
    waits = np.asarray([event["loader_wait_ms"] for event in warm], dtype=np.float64)
    if intervals.size:
        print(
            f"[perf-lite][epoch={epoch + 1}] run_tag={run_tag} events={len(events)} warm={len(warm)} "
            f"interval_mean/median/p95/max="
            f"{intervals.mean():.1f}/{np.median(intervals):.1f}/{np.percentile(intervals, 95):.1f}/{intervals.max():.1f}ms "
            f"loader_mean/median/p95/max="
            f"{waits.mean():.1f}/{np.median(waits):.1f}/{np.percentile(waits, 95):.1f}/{waits.max():.1f}ms "
            f"path={event_path}"
        )


def uncollate_worker_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return uncollate_worker_value(value[0])
        return [uncollate_worker_value(item) for item in value]
    return value


def extract_worker_profile(batch_data):
    meta = batch_data.get("kspace_meta_dict", {})
    timing_values = meta.get("worker_timing", {}) if isinstance(meta, dict) else {}
    context_values = meta.get("worker_context", {}) if isinstance(meta, dict) else {}
    resource_values = meta.get("worker_resource", {}) if isinstance(meta, dict) else {}
    return {
        "timing": {
            name: float(uncollate_worker_value(value))
            for name, value in timing_values.items()
            if not str(name).startswith("_")
        },
        "context": {
            name: uncollate_worker_value(value)
            for name, value in context_values.items()
        },
        "resource": {
            name: uncollate_worker_value(value)
            for name, value in resource_values.items()
        },
    }


def extract_post_worker_profile(batch_data):
    meta = batch_data.get("kspace_meta_dict", {})
    values = meta.get("post_worker_timing", {}) if isinstance(meta, dict) else {}
    return {name: uncollate_worker_value(value) for name, value in values.items()}


def record_loader_performance(
    batch_data,
    batch_timings,
    should_sample,
    request_ns,
    received_ns,
    loader_wait_ms,
    worker_timing_enabled,
    post_worker_timing_enabled,
    slow_loader_threshold_ms,
    slow_loader_max_events,
    worker_loader_max_events,
    slow_loader_events,
    worker_loader_events,
    epoch,
    global_step,
    batch_index,
    rank,
    run_tag,
    file_name,
    final_shape,
):
    worker_profile = extract_worker_profile(batch_data) if worker_timing_enabled else {
        "timing": {},
        "context": {},
        "resource": {},
    }
    post_worker_profile = (
        extract_post_worker_profile(batch_data) if post_worker_timing_enabled else {}
    )
    loader_critical_path = (
        derive_loader_critical_path(
            request_ns,
            received_ns,
            worker_profile["context"],
            post_worker_profile,
        )
        if post_worker_timing_enabled
        else {}
    )
    if should_sample:
        for timing_name in WORKER_TIMING_FIELDS:
            batch_timings[timing_name] = float(worker_profile["timing"].get(timing_name, 0.0))
        for timing_name in POST_WORKER_TIMING_FIELDS:
            value = (
                post_worker_profile.get(timing_name, 0.0)
                if timing_name == "collate_ms"
                else loader_critical_path.get(timing_name, 0.0)
            )
            batch_timings[timing_name] = float(value)

    consumer_slow = (
        slow_loader_threshold_ms > 0 and loader_wait_ms >= slow_loader_threshold_ms
    )
    worker_event = {
        "job_id": os.getenv("SLURM_JOB_ID"),
        "epoch": epoch + 1,
        "global_step": global_step,
        "batch_index": batch_index,
        "rank": rank,
        "hostname": socket.gethostname(),
        "run_tag": run_tag,
        "consumer_loader_wait_ms": round(loader_wait_ms, 3),
        "consumer_slow": consumer_slow,
        "filename": str(file_name),
        "final_shape": final_shape,
        "worker_timing": worker_profile["timing"],
        "worker_context": worker_profile["context"],
        "worker_resource": worker_profile["resource"],
        "post_worker_timing": post_worker_profile,
        "loader_critical_path": loader_critical_path,
    }
    if worker_timing_enabled and len(worker_loader_events) < worker_loader_max_events:
        worker_loader_events.append(worker_event)

    if not consumer_slow:
        return 0
    retain_slow_loader_event(
        slow_loader_events,
        {**worker_event, "loader_wait_ms": round(loader_wait_ms, 3)},
        slow_loader_max_events,
    )
    return 1


def derive_loader_critical_path(request_ns, received_ns, worker_context, post_worker):
    pipeline_start_ns = int(worker_context.get("pipeline_start_ns", 0) or 0)
    pipeline_end_ns = int(worker_context.get("pipeline_end_ns", 0) or 0)
    collate_start_ns = int(post_worker.get("collate_start_ns", 0) or 0)
    collate_end_ns = int(post_worker.get("collate_end_ns", 0) or 0)

    def overlap_ms(start_ns, end_ns):
        return max(0.0, (min(end_ns, received_ns) - max(start_ns, request_ns)) / 1.0e6)

    pre_worker_start_ms = (
        max(0.0, (min(pipeline_start_ns, received_ns) - request_ns) / 1.0e6)
        if pipeline_start_ns > 0
        else 0.0
    )
    worker_on_critical_path_ms = (
        overlap_ms(pipeline_start_ns, pipeline_end_ns)
        if pipeline_start_ns > 0 and pipeline_end_ns >= pipeline_start_ns
        else 0.0
    )
    pre_collate_gap_ms = (
        overlap_ms(pipeline_end_ns, collate_start_ns)
        if pipeline_end_ns > 0 and collate_start_ns >= pipeline_end_ns
        else 0.0
    )
    collate_on_critical_path_ms = (
        overlap_ms(collate_start_ns, collate_end_ns)
        if collate_start_ns > 0 and collate_end_ns >= collate_start_ns
        else 0.0
    )
    post_collate_on_critical_path_ms = (
        max(0.0, (received_ns - max(request_ns, collate_end_ns)) / 1.0e6)
        if collate_end_ns > 0
        else 0.0
    )
    loader_wait_ms = max(0.0, (received_ns - request_ns) / 1.0e6)
    attributed_ms = sum(
        (
            pre_worker_start_ms,
            worker_on_critical_path_ms,
            pre_collate_gap_ms,
            collate_on_critical_path_ms,
            post_collate_on_critical_path_ms,
        )
    )
    phases = {
        "pre_worker_start": pre_worker_start_ms,
        "worker_pipeline": worker_on_critical_path_ms,
        "pre_collate_gap": pre_collate_gap_ms,
        "collate": collate_on_critical_path_ms,
        "post_collate_delivery": post_collate_on_critical_path_ms,
    }
    dominant_phase = max(phases, key=phases.get) if any(phases.values()) else "unavailable"
    return {
        "pre_worker_start_ms": pre_worker_start_ms,
        "worker_on_critical_path_ms": worker_on_critical_path_ms,
        "pre_collate_gap_ms": pre_collate_gap_ms,
        "collate_on_critical_path_ms": collate_on_critical_path_ms,
        "post_collate_on_critical_path_ms": post_collate_on_critical_path_ms,
        "critical_path_unattributed_ms": max(0.0, loader_wait_ms - attributed_ms),
        "ready_before_request_ms": (
            max(0.0, (request_ns - collate_end_ns) / 1.0e6) if collate_end_ns > 0 else 0.0
        ),
        "collate_to_consumer_ms": (
            max(0.0, (received_ns - collate_end_ns) / 1.0e6) if collate_end_ns > 0 else 0.0
        ),
        "dominant_phase": dominant_phase,
        "request_ns": int(request_ns),
        "received_ns": int(received_ns),
    }


def log_checkpoint_timing(label, timing, writer, step):
    print(
        f"[perf][checkpoint] label={label} state_prepare={timing['state_prepare_s']:.2f}s "
        f"torch_save={timing['torch_save_s']:.2f}s total={timing['total_s']:.2f}s "
        f"size={timing['size_bytes'] / (1024**3):.2f}GiB"
    )
    writer.add_scalar(f"perf_checkpoint/{label}_state_prepare_s", timing["state_prepare_s"], step)
    writer.add_scalar(f"perf_checkpoint/{label}_torch_save_s", timing["torch_save_s"], step)
    writer.add_scalar(f"perf_checkpoint/{label}_total_s", timing["total_s"], step)


def retain_slow_loader_event(events, event, max_events):
    if max_events <= 0:
        return
    if len(events) < max_events:
        events.append(event)
        return
    fastest_index = min(range(len(events)), key=lambda idx: events[idx]["loader_wait_ms"])
    if event["loader_wait_ms"] > events[fastest_index]["loader_wait_ms"]:
        events[fastest_index] = event


def gather_and_log_slow_loader_events(
    local_events,
    local_event_count,
    rank,
    world_size,
    is_ddp,
    outpath,
    epoch,
    threshold_ms,
):
    payload = {
        "rank": rank,
        "total_count": local_event_count,
        "events": local_events,
    }
    if is_ddp:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
    else:
        gathered = [payload]

    if rank != 0:
        return

    rank_counts = {
        str(item["rank"]): int(item["total_count"])
        for item in gathered
        if item is not None and item["total_count"] > 0
    }
    events = [event for item in gathered if item is not None for event in item["events"]]
    events.sort(key=lambda event: event["loader_wait_ms"], reverse=True)

    performance_dir = Path(outpath) / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    event_path = performance_dir / "slow_loader_events.jsonl"
    if events:
        with event_path.open("a", encoding="utf-8") as event_file:
            for event in events:
                event_file.write(json.dumps(event, sort_keys=True) + "\n")

    top_events = "; ".join(
        f"r{event['rank']}:{event['loader_wait_ms']:.1f}ms:{Path(event['filename']).name}"
        for event in events[:5]
    )
    print(
        f"[perf][loader_slow][epoch={epoch + 1}] threshold={threshold_ms:.1f}ms "
        f"total={sum(rank_counts.values())} saved={len(events)} ranks={json.dumps(rank_counts, sort_keys=True)} "
        f"path={event_path} top={top_events or 'none'}"
    )


def gather_and_log_worker_loader_events(
    local_events,
    rank,
    world_size,
    is_ddp,
    outpath,
    epoch,
):
    payload = {"rank": rank, "events": local_events}
    if is_ddp:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
    else:
        gathered = [payload]

    if rank != 0:
        return

    events = [event for item in gathered if item is not None for event in item["events"]]
    events.sort(key=lambda event: (event["global_step"], event["rank"]))
    performance_dir = Path(outpath) / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    event_path = performance_dir / "worker_loader_events.jsonl"
    if events:
        with event_path.open("a", encoding="utf-8") as event_file:
            for event in events:
                event_file.write(json.dumps(event, sort_keys=True) + "\n")

    rank_counts = {
        str(event_rank): count
        for event_rank, count in sorted(Counter(event["rank"] for event in events).items())
    }
    slow_count = sum(bool(event["consumer_slow"]) for event in events)
    top_events = sorted(
        events,
        key=lambda event: event["worker_timing"].get("worker_total_ms", 0.0),
        reverse=True,
    )[:5]
    top_summary = "; ".join(
        f"r{event['rank']}/w{event['worker_context'].get('worker_id', -1)}:"
        f"{event['worker_timing'].get('worker_total_ms', 0.0):.1f}ms:"
        f"{Path(event['filename']).name}"
        for event in top_events
    )
    print(
        f"[perf][worker_loader][epoch={epoch + 1}] events={len(events)} slow={slow_count} "
        f"ranks={json.dumps(rank_counts, sort_keys=True)} path={event_path} "
        f"top={top_summary or 'none'}"
    )


def _worker_timing_enabled(args) -> bool:
    timing_cfg = getattr(args, "performance_timing", None)
    return timing_cfg is not None and bool(getattr(timing_cfg, "worker_timing_enabled", False))


def _current_cpu() -> int:
    getcpu = getattr(os, "sched_getcpu", None)
    if getcpu is not None:
        return int(getcpu())
    try:
        with open("/proc/self/stat", "r", encoding="utf-8") as stat_file:
            return int(stat_file.read().split()[38])
    except (OSError, IndexError, ValueError):
        return -1


def _format_cpu_affinity() -> str:
    if not hasattr(os, "sched_getaffinity"):
        return "unknown"
    cpus = sorted(os.sched_getaffinity(0))
    if not cpus:
        return "empty"
    ranges = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _resource_snapshot() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
        "input_blocks": int(usage.ru_inblock),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "max_rss_kb": int(usage.ru_maxrss),
    }


class WorkerTimedTransform:
    def __init__(self, transform, name: str, *, initialize: bool = False, finalize: bool = False):
        self.transform = transform
        self.name = name
        self.initialize = initialize
        self.finalize = finalize

    def __call__(self, data):
        monotonic_started_ns = time.monotonic_ns()
        started = time.perf_counter()
        resource_before = _resource_snapshot() if self.initialize else None
        cpu_before = _current_cpu() if self.initialize else None
        result = self.transform(data)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        meta = result.get("kspace_meta_dict")
        if not isinstance(meta, dict):
            return result

        timings = dict(meta.get("worker_timing", {}))
        context = dict(meta.get("worker_context", {}))
        timings[self.name] = elapsed_ms

        if self.initialize:
            worker_info = torch.utils.data.get_worker_info()
            timings["_pipeline_started"] = started
            timings["_pipeline_started_ns"] = monotonic_started_ns
            for resource_name, value in resource_before.items():
                timings[f"_resource_start_{resource_name}"] = value
            context.update(
                {
                    "rank": int(os.getenv("RANK", "0")),
                    "local_rank": int(os.getenv("LOCAL_RANK", "0")),
                    "worker_id": int(worker_info.id) if worker_info is not None else -1,
                    "worker_seed": int(worker_info.seed) if worker_info is not None else -1,
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "cpu_start": cpu_before,
                    "cpu_affinity": _format_cpu_affinity(),
                }
            )

        if self.finalize:
            pipeline_started = float(timings.pop("_pipeline_started", started))
            pipeline_started_ns = int(timings.pop("_pipeline_started_ns", monotonic_started_ns))
            timings["worker_total_ms"] = (time.perf_counter() - pipeline_started) * 1000.0
            resource_after = _resource_snapshot()
            worker_resource = {}
            for resource_name, value in resource_after.items():
                start_value = int(timings.pop(f"_resource_start_{resource_name}", value))
                if resource_name == "max_rss_kb":
                    worker_resource[resource_name] = value
                else:
                    worker_resource[resource_name] = value - start_value
            context["cpu_end"] = _current_cpu()
            context["pipeline_start_ns"] = pipeline_started_ns
            context["pipeline_end_ns"] = time.monotonic_ns()
            meta["worker_resource"] = worker_resource

        meta["worker_timing"] = timings
        meta["worker_context"] = context
        return result


def _maybe_timed(args, name: str, transform, *, initialize: bool = False, finalize: bool = False):
    if not _worker_timing_enabled(args):
        return transform
    return WorkerTimedTransform(transform, name, initialize=initialize, finalize=finalize)


def get_train_transforms(args):
    """
    Get the training transforms.
    """
    generic_data_aug = bool(args.data_aug) and not bool(getattr(args, "is_4dflow_aorta", False))
    train_transforms = Compose(
        [
            _maybe_timed(
                args,
                "load_total_ms",
                LoadImaged(
                    keys=["kspace"],
                    reader=get_reader(args, is_testing=True),
                    image_only=False,
                    dtype=np.complex64,
                ),
                initialize=True,
            ),
            (RandFlipd(keys=["kspace"], spatial_axis=-2, prob=0.25) if generic_data_aug else Identityd(keys=["kspace"])),
            (RandFlipd(keys=["kspace"], spatial_axis=-1, prob=0.25) if generic_data_aug else Identityd(keys=["kspace"])),
            (
                RandShiftKspaced(keys=["kspace"], prob=0.25, shift=(16, 32))
                if generic_data_aug
                else Identityd(keys=["kspace"])
            ),
            (
                RandPhaseShiftKspaced(keys=["kspace"], prob=0.25, angle=45)
                if generic_data_aug
                else Identityd(keys=["kspace"])
            ),
            (
                RandAdjustContrastKspaced(keys=["kspace"], prob=0.25, gamma=(0.5, 2))
                if generic_data_aug
                else Identityd(keys=["kspace"])
            ),
            (
                RandResizeWithPadOrCropd(keys=["kspace"], prob=0.25, spatial_size=(16, 32))
                if generic_data_aug
                else Identityd(keys=["kspace"])
            ),
            (
                RandSimulateNoReadoutOversampleKspaced(keys=["kspace"], prob=0.25)
                if args.dataset.lower() == "cmrxrecon" and not getattr(args, "is_4dflow_aorta", False)
                else Identityd(keys=["kspace"])
            ),
            ExtractDataKeyFromMetaKeyd(keys=["mask", "acquisition"], meta_key="kspace_meta_dict"),
            _maybe_timed(
                args,
                "mask_hybrid_ms",
                KspaceMaskd(
                    keys=["kspace"],
                    mask_types=args.train_mask_types,
                    center_fractions=args.center_fractions,
                    accelerations=args.accelerations,
                    acs_lines=args.acs_lines,
                    spatial_dims=2,
                    is_complex=True,
                ),
            ),
            _maybe_timed(
                args,
                "to_tensor_complex_ms",
                Lambdad(keys=["kspace"], func=lambda x: convert_to_tensor_complex(x)),
            ),
            (
                ResizeWithPadOrCropd(
                    keys=["kspace", "mask", "kspace_masked"],
                    spatial_size=[
                        -1,
                        -1,
                        args.uniform_input_kspace[0],
                        args.uniform_input_kspace[1],
                        2,
                    ],
                )
                if args.uniform_input_kspace
                else Identityd(keys=["kspace"])
            ),
            _maybe_timed(
                args,
                "ensure_typed_ms",
                EnsureTyped(keys=["kspace", "kspace_masked", "mask"]),
            ),
            _maybe_timed(
                args,
                "ifft_ms",
                Lambdad(
                    keys=["kspace", "kspace_masked"],
                    overwrite=["kspace_ifft", "kspace_masked_ifft"],
                    func=lambda x: ifftn_centered(x, spatial_dims=2, is_complex=True),
                ),
            ),
            _maybe_timed(
                args,
                "augmentation_ms",
                (
                    FourDFlowOnlineAugmentd(args)
                    if bool(getattr(args, "is_4dflow_aorta", False)) and bool(args.data_aug)
                    else Identityd(keys=["kspace"])
                ),
            ),
            _maybe_timed(
                args,
                "rearrange_normalize_ms",
                RearrangeAndNormalizeMRI(keys=["kspace_masked_ifft", "kspace_ifft", "mask"], args=args),
                finalize=True,
            ),
        ]
    )
    return train_transforms


def get_val_transforms(args):
    """
    Get the validation transforms.
    """
    val_transforms = Compose(
        [
            _maybe_timed(
                args,
                "load_total_ms",
                LoadImaged(
                    keys=["kspace"],
                    reader=get_reader(args, is_testing=True),
                    image_only=False,
                    dtype=np.complex64,
                ),
                initialize=True,
            ),
            ExtractDataKeyFromMetaKeyd(keys=["mask", "acquisition"], meta_key="kspace_meta_dict"),
            _maybe_timed(
                args,
                "mask_hybrid_ms",
                KspaceMaskd(
                    keys=["kspace"],
                    mask_types=args.val_mask_types,
                    center_fractions=args.center_fractions,
                    accelerations=args.accelerations,
                    spatial_dims=2,
                    is_complex=True,
                ),
            ),
            _maybe_timed(
                args,
                "to_tensor_complex_ms",
                Lambdad(keys=["kspace"], func=lambda x: convert_to_tensor_complex(x)),
            ),
            (
                ResizeWithPadOrCropd(
                    keys=["kspace", "mask", "kspace_masked"],
                    spatial_size=[
                        -1,
                        -1,
                        args.uniform_input_kspace[0],
                        args.uniform_input_kspace[1],
                        2,
                    ],
                )
                if args.uniform_input_kspace
                else Identityd(keys=["kspace"])
            ),
            _maybe_timed(
                args,
                "ensure_typed_ms",
                EnsureTyped(keys=["kspace", "kspace_masked", "mask"]),
            ),
            _maybe_timed(
                args,
                "ifft_ms",
                Lambdad(
                    keys=["kspace", "kspace_masked"],
                    overwrite=["kspace_ifft", "kspace_masked_ifft"],
                    func=lambda x: ifftn_centered(x, spatial_dims=2, is_complex=True),
                ),
            ),
            _maybe_timed(
                args,
                "rearrange_normalize_ms",
                RearrangeAndNormalizeMRI(keys=["kspace_masked_ifft", "kspace_ifft", "mask"], args=args),
                finalize=True,
            ),
        ]
    )
    return val_transforms


def get_optimizer(args, model):
    """
    Get the optimizer for the model.
    """
    if args.muon:
        excluded_patterns = [
            "unet.conv_0.conv_0.conv",
            "embed_conv",
            "dc_weight_map",
            "dc_weight",
            "feat_extract",
        ]

        def use_adam(name, parameter):
            return (
                parameter.ndim < 2
                or any(pattern in name for pattern in excluded_patterns)
                or (
                    "flowvn_mixer.regularizers." in name
                    and name.endswith(".weight")
                )
            )

        muon_params = [
            p
            for n, p in model.named_parameters()
            if not use_adam(n, p)
        ]
        adam_params = [
            p for n, p in model.named_parameters() if use_adam(n, p)
        ]
        param_groups = [
            dict(
                params=filter(lambda p: p.requires_grad, muon_params),
                use_muon=True,
                lr=args.lr,
                weight_decay=args.weight_decay,
            ),
            dict(
                params=filter(lambda p: p.requires_grad, adam_params),
                use_muon=False,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lr_scale=1.0,
            ),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    elif args.lookahead:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        optimizer = Lookahead(optimizer)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    print("using optimizer: ", optimizer.__class__.__name__)
    return optimizer


def apply_phase3_freeze(args, model):
    phase3 = getattr(args, "phase3", None)
    if phase3 is None:
        return
    if not bool(getattr(phase3, "enable_vaa", False)):
        return
    freeze = getattr(phase3, "freeze", None)
    if freeze is None:
        return

    freeze_backbone = bool(getattr(freeze, "backbone", False))
    freeze_vaa = bool(getattr(freeze, "vaa", False))
    gamma_cfg = getattr(phase3, "gamma", None)
    gamma_trainable = bool(getattr(gamma_cfg, "trainable", True)) if gamma_cfg is not None else True
    for name, param in model.named_parameters():
        is_vaa = "vaa_adapters" in name or "gamma_raw" in name
        if is_vaa:
            param.requires_grad = (not freeze_vaa) and (gamma_trainable if "gamma_raw" in name else True)
        elif freeze_backbone:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Phase3 freeze: backbone={freeze_backbone}, vaa={freeze_vaa}, "
        f"trainable_params={trainable / 1e6:.2f}M/{total / 1e6:.2f}M"
    )
