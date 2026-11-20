"""torchrun data-parallel helpers: every rank holds identical weights and applies the rank-averaged gradient."""

from __future__ import annotations

from datetime import timedelta
import os

import torch
import torch.distributed as dist


def environment():
    """(world_size, rank, local_rank) as exported by torchrun; (1, 0, 0) for a plain process."""
    return tuple(int(os.environ.get(key, default)) for key, default in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))


def initialize(world, rank, local_rank):
    # Mirrors scripts/train.py: bind the device first, then join the env:// group.
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world,
                            timeout=timedelta(seconds=1800))
    dist.barrier(device_ids=[local_rank])


def rank_share(items, world, rank):
    """Rank's slice of a shared order, wrapping the head so every rank gets the same count (as scripts/train.py pads)."""
    items = list(items)
    items += items[:-len(items) % world]
    return items[rank::world]


def size_buckets(order, cost, world, bucket_steps, rng):
    """Reorder a shared, already shuffled order so each step's `world` items have similar cost.

    Every rank waits for the slowest one at each step, so mixing a large and a small case wastes the small
    one's GPU. Items are taken in blocks of bucket_steps * world, sorted by cost inside the block and cut into
    steps; step order and the rank each item lands on stay random. Rank r reads positions r::world.
    """
    order = list(order)
    order += order[:-len(order) % world]
    result = []
    for start in range(0, len(order), bucket_steps * world):
        block = sorted(order[start:start + bucket_steps * world], key=cost)
        steps = [block[i:i + world] for i in range(0, len(block), world)]
        for index in rng.permutation(len(steps)):
            step = steps[int(index)]
            result.extend(step[int(i)] for i in rng.permutation(len(step)))
    return result


def sync_gradients(parameters, world):
    """Replace each gradient by its mean over ranks. A parameter without a gradient on every rank stays None,
    so the optimizer skips it exactly as in a single process."""
    params = [p for p in parameters if p.requires_grad]
    if world == 1 or not params:
        return
    # Complex parameters (CUT-4DFlow) travel as real/imaginary pairs; one real buffer holds everything.
    real = lambda t: torch.view_as_real(t) if t.is_complex() else t
    dtype = max((real(p).dtype for p in params), key=lambda d: torch.finfo(d).bits)
    flags = torch.tensor([p.grad is not None for p in params], device=params[0].device, dtype=dtype)
    flat = torch.cat([*(real(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).to(dtype) for p in params), flags])
    dist.all_reduce(flat)
    flat /= world
    used, offset = flat[-len(params):] > 0, 0
    for p, keep in zip(params, used.tolist()):
        size = real(p).numel()
        chunk = flat[offset:offset + size].to(real(p).dtype)
        offset += size
        if not keep:
            p.grad = None
        elif p.is_complex():
            p.grad = torch.view_as_complex(chunk.view(*p.shape, 2))
        else:
            p.grad = chunk.view_as(p)


def all_ranks_equal(values, what):
    """Raise unless every rank holds the same float64 vector (resume point, weight fingerprint)."""
    low = torch.as_tensor(values, dtype=torch.float64, device=torch.cuda.current_device())
    high = low.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    if not torch.equal(low, high):
        raise RuntimeError(f"Ranks disagree on {what}: min={low.tolist()} max={high.tolist()}")


def any_rank(*flags):
    """Each flag becomes True on every rank when any rank raises it, so all ranks act together."""
    value = torch.tensor([float(f) for f in flags], device=torch.cuda.current_device())
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return [bool(v) for v in value.tolist()]


def fingerprint(model):
    """Cheap parameter digest; buffers such as BatchNorm running statistics are rank-local and excluded."""
    total = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
    absolute = total.clone()
    for p in model.parameters():
        value = (torch.view_as_real(p.detach()) if p.is_complex() else p.detach()).double()
        total += value.sum()
        absolute += value.abs().sum()
    return [float(total), float(absolute)]
