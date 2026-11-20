"""Case-local sampling and one-window host prefetch without shared-memory IPC."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import torch

from .distributed import rank_share, size_buckets


def case_order(dataset, shuffle):
    rng = np.random.default_rng([dataset.seed, dataset.epoch])
    order = rng.permutation(len(dataset.cases)) if shuffle else range(len(dataset.cases))
    for case in order:
        offsets = rng.permutation(dataset.samples_per_case) if shuffle else range(dataset.samples_per_case)
        yield int(case), [int(case) * dataset.samples_per_case + int(offset) for offset in offsets]


class CaseWindowLoader:
    def __init__(self, dataset, device, method, *, shuffle, case_cache_mib=512,
                 prefetch=True, pin_limit_mib=128, world=1, rank=0, bucket_steps=0, stack=1):
        self.dataset, self.shuffle = dataset, shuffle
        self.world, self.rank = world, rank
        if type(stack) is not int or stack < 1:
            raise ValueError("paper.loader.stack must be a positive integer")
        # Windows of one case share a shape, so `stack` of them go through the network in one pass,
        # as scripts/train.py does with batch_size // encodings windows.
        self.stack = stack
        self.skip_cases = 0  # set by the training loop when it resumes inside an epoch
        if type(bucket_steps) is not int or bucket_steps < 0:
            raise ValueError("paper.loader.bucket_steps must be a nonnegative integer")
        # 0 keeps the plain shuffled split; N groups similar-size cases into the same step (see size_buckets).
        self.bucket_steps = bucket_steps if world > 1 and shuffle else 0
        self.cache_bytes = int(case_cache_mib * 1024**2)
        self.pin_limit = int(pin_limit_mib * 1024**2)
        if min(self.cache_bytes, self.pin_limit) < 0 or type(prefetch) is not bool:
            raise ValueError("paper.loader requires nonnegative MiB limits and a boolean prefetch")
        self.prefetch = prefetch
        self.cuda = torch.device(device).type == "cuda"
        self.root_keys = {"kspace", "coils", "mask"}
        self.device_keys = {
            "ours": {"target", "segmask"},
            "flowvn": {"target", "acceleration"},
            "flowmodl": {"target", "segmask", "venc", "acceleration"},
            "cut_4dflow": {"full_kspace", "frame_indices"},
        }[method] | self.root_keys
        self.use_full_kspace = method == "cut_4dflow"

    @property
    def batches_per_case(self):
        return -(-self.dataset.samples_per_case // self.stack)

    def __len__(self):
        # Every rank trains the same number of batches; the shared case order is padded by wrapping.
        return -(-len(self.dataset.cases) // self.world) * self.batches_per_case

    def _collate(self, group):
        batch = {k: torch.cat([b[k] for b in group]) if torch.is_tensor(group[0][k]) else [x for b in group for x in b[k]]
                 for k in group[0]}
        if self.cuda and self.prefetch:
            size = sum(batch[k].numel() * batch[k].element_size() for k in self.device_keys)
            if size <= self.pin_limit:
                for key in self.device_keys:
                    batch[key] = batch[key].pin_memory()
        return batch

    def _batches(self):
        order = case_order(self.dataset, self.shuffle)
        if self.bucket_steps:
            # TrainSlabs sorts cases by k-space size, so the case index is the cost rank.
            rng = np.random.default_rng([self.dataset.seed, self.dataset.epoch, 1])
            order = size_buckets(order, lambda item: item[0], self.world, self.bucket_steps, rng)
        for case, indices in rank_share(order, self.world, self.rank)[self.skip_cases:]:
            with self.dataset.case_windows(case, indices, cache_bytes=self.cache_bytes) as samples:
                group = []
                for sample in samples:
                    if not self.use_full_kspace:
                        sample.pop("full_kspace")
                    group.append({
                        k: v.unsqueeze(0) if torch.is_tensor(v) else
                        torch.tensor([v], dtype=torch.float64) if isinstance(v, float) else
                        torch.tensor([v]) if isinstance(v, int) else [v]
                        for k, v in sample.items()
                    })
                    if len(group) == self.stack:
                        yield self._collate(group)
                        group = []
                if group:
                    yield self._collate(group)

    def to_device(self, batch, device):
        return {k: v.to(device, non_blocking=True) if k in self.device_keys else v for k, v in batch.items()}

    @contextmanager
    def epoch_batches(self):
        batches = self._batches()
        if not self.prefetch:
            try:
                yield batches
            finally:
                batches.close()
            return
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="case-window") as pool:
            pending = pool.submit(next, batches, None)

            def ahead():
                nonlocal pending
                while (batch := pending.result()) is not None:
                    pending = pool.submit(next, batches, None)
                    yield batch

            iterator = ahead()
            try:
                yield iterator
            finally:
                iterator.close()
                # Drain the outstanding read before closing its HDF5 handles.
                pending.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                batches.close()
