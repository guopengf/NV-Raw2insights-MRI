"""Case-local sampling and one-window host prefetch without shared-memory IPC."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import torch


def case_order(dataset, shuffle):
    rng = np.random.default_rng([dataset.seed, dataset.epoch])
    order = rng.permutation(len(dataset.cases)) if shuffle else range(len(dataset.cases))
    for case in order:
        offsets = rng.permutation(dataset.samples_per_case) if shuffle else range(dataset.samples_per_case)
        yield int(case), [int(case) * dataset.samples_per_case + int(offset) for offset in offsets]


class CaseWindowLoader:
    def __init__(self, dataset, device, method, *, shuffle, case_cache_mib=512,
                 prefetch=True, pin_limit_mib=128):
        self.dataset, self.shuffle = dataset, shuffle
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

    def __len__(self):
        return len(self.dataset)

    def _batches(self):
        for case, indices in case_order(self.dataset, self.shuffle):
            with self.dataset.case_windows(case, indices, cache_bytes=self.cache_bytes) as samples:
                for sample in samples:
                    if not self.use_full_kspace:
                        sample.pop("full_kspace")
                    batch = {
                        k: v.unsqueeze(0) if torch.is_tensor(v) else
                        torch.tensor([v], dtype=torch.float64) if isinstance(v, float) else
                        torch.tensor([v]) if isinstance(v, int) else [v]
                        for k, v in sample.items()
                    }
                    if self.cuda and self.prefetch:
                        size = sum(batch[k].numel() * batch[k].element_size() for k in self.device_keys)
                        if size <= self.pin_limit:
                            for key in self.device_keys:
                                batch[key] = batch[key].pin_memory()
                    yield batch

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
