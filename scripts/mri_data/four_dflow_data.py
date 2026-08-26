"""Small, backwards-compatible data helpers for joint-VENC 4D Flow training."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_manifest(path: str | Path) -> dict:
    with open(path) as f:
        item = json.load(f)
    item["_manifest_path"] = str(path)
    return item


def joint_venc_group_key(item: dict) -> tuple[str, str, str]:
    target = Path(item["target_kspace"]).expanduser().resolve(strict=False)
    input_path = Path(item["kspace"]).expanduser().resolve(strict=False)
    return str(target.parent), str(item["mask_type"]), str(input_path)


def group_joint_venc_manifests(
    manifest_entries,
    *,
    encodings=(0, 1, 2, 3),
) -> list[list[dict]]:
    """Return strictly ordered complete VENC groups from per-encoding manifests."""

    expected = tuple(int(value) for value in encodings)
    groups: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for entry in manifest_entries:
        path = entry.get("kspace") if isinstance(entry, dict) else entry
        item = _read_manifest(path)
        enc = int(item["encoding_idx"])
        key = joint_venc_group_key(item)
        if enc in groups[key]:
            raise ValueError(f"Duplicate encoding {enc} for joint-VENC group {key}")
        groups[key][enc] = item

    ordered_groups = []
    for key in sorted(groups):
        found = groups[key]
        if set(found) != set(expected):
            raise ValueError(
                f"Incomplete joint-VENC group {key}: expected encodings {list(expected)}, "
                f"found {sorted(found)}"
            )
        ordered_groups.append([{"kspace": Path(found[enc]["_manifest_path"])} for enc in expected])
    return ordered_groups


def _same_value(a, b) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        a_tensor = torch.as_tensor(a)
        b_tensor = torch.as_tensor(b)
        return a_tensor.shape == b_tensor.shape and torch.equal(a_tensor, b_tensor)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


class Paired4DFlowRoll:
    """Safe image-domain periodic translation shared by inputs, targets and ROI.

    A spatial translation changes neither the acquired k-space support nor the
    velocity component coordinate system, so the sampling mask stays unchanged.
    """

    def __init__(self, *, enabled=False, probability=0.0, max_shift_zy=(0, 0)):
        self.enabled = bool(enabled)
        self.probability = float(probability)
        self.max_shift_zy = tuple(int(value) for value in max_shift_zy)
        if len(self.max_shift_zy) != 2 or any(value < 0 for value in self.max_shift_zy):
            raise ValueError("paired_roll.max_shift_zy must contain two non-negative integers")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("paired_roll.probability must be in [0,1]")

    def __call__(self, sample: dict) -> dict:
        if not self.enabled or random.random() >= self.probability:
            return sample
        shifts = tuple(random.randint(-limit, limit) if limit else 0 for limit in self.max_shift_zy)
        if shifts == (0, 0):
            return sample
        sample = dict(sample)
        for key in ("kspace_ifft", "kspace_masked_ifft", "sensitivity_maps"):
            if key in sample and sample[key] is not None:
                sample[key] = torch.roll(torch.as_tensor(sample[key]), shifts=shifts, dims=(-3, -2))
        for key in ("roi_mask", "mra_prior"):
            if key in sample and sample[key] is not None:
                sample[key] = torch.roll(torch.as_tensor(sample[key]), shifts=shifts, dims=(-2, -1))
        sample["paired_roll_zy"] = torch.tensor(shifts, dtype=torch.int64)
        return sample


class JointVencDataset(Dataset):
    """Apply the existing per-encoding transform, then stack encodings in order."""

    STACK_KEYS = ("kspace_ifft", "kspace_masked_ifft", "mask", "mean", "std", "sensitivity_maps")
    SHARED_KEYS = ("roi_mask", "mra_prior", "mask_type", "acc_factor", "acquisition", "temporal_shuffle")

    def __init__(self, manifest_entries, transform, *, encodings=(0, 1, 2, 3), paired_transform=None):
        self.encodings = tuple(int(value) for value in encodings)
        if manifest_entries and isinstance(manifest_entries[0], list):
            self.groups = list(manifest_entries)
        else:
            self.groups = group_joint_venc_manifests(manifest_entries, encodings=self.encodings)
        self.transform = transform
        self.paired_transform = paired_transform

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        result = {}
        metadata = None
        group = self.groups[index]
        if len(group) != len(self.encodings):
            raise ValueError(
                f"Joint-VENC group has {len(group)} entries, expected {len(self.encodings)}"
            )

        # These transformed volumes are several GiB each for real 4D Flow cases.
        # Fill the final tensors one encoding at a time so only one unstacked
        # sample, rather than all four samples, is resident at peak.
        for encoding_position, item in enumerate(group):
            sample = self.transform(item)

            for key in self.STACK_KEYS:
                value = sample.get(key)
                if encoding_position == 0:
                    if value is None:
                        continue
                    tensor = torch.as_tensor(value)
                    result[key] = tensor.new_empty((len(self.encodings), *tensor.shape))
                elif (key in result) != (value is not None):
                    raise ValueError(f"Joint-VENC samples disagree on presence of {key!r}")

                if value is not None:
                    tensor = torch.as_tensor(value)
                    expected_shape = result[key].shape[1:]
                    if tensor.shape != expected_shape:
                        raise ValueError(
                            f"Joint-VENC samples disagree on shape of {key!r}: "
                            f"expected {tuple(expected_shape)}, got {tuple(tensor.shape)}"
                        )
                    result[key][encoding_position].copy_(tensor)

            for key in self.SHARED_KEYS:
                value = sample.get(key)
                if encoding_position == 0:
                    if value is not None:
                        result[key] = value
                elif (key in result) != (value is not None) or (
                    value is not None and not _same_value(result[key], value)
                ):
                    raise ValueError(f"Joint-VENC samples disagree on shared value {key!r}")

            other_metadata = sample["kspace_meta_dict"]
            if metadata is None:
                metadata = dict(other_metadata)
            else:
                for key in ("shape", "acquisition"):
                    if (
                        key in metadata
                        and key in other_metadata
                        and not _same_value(metadata[key], other_metadata[key])
                    ):
                        raise ValueError(f"Joint-VENC metadata disagree on {key!r}")

            del sample

        assert metadata is not None
        metadata["encoding_indices"] = torch.tensor(self.encodings, dtype=torch.int64)
        metadata["joint_group_id"] = str(group[0]["kspace"]).rsplit("__enc", 1)[0]
        metadata["flow_corr_cache_id"] = metadata["joint_group_id"].rsplit("__ktGaussian", 1)[0]
        metadata["filename"] = Path(metadata["joint_group_id"]).name + ".json"
        result["kspace_meta_dict"] = metadata
        result["encoding_indices"] = torch.tensor(self.encodings, dtype=torch.int64)
        if self.paired_transform is not None:
            result = self.paired_transform(result)
        return result


class PostTransformDataset(Dataset):
    """Apply a paired transform after an existing per-encoding dataset/cache."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset[index])


def gather_joint_window(input_tensor: torch.Tensor, window_idx: torch.Tensor) -> torch.Tensor:
    """Use one [B,S,T] index tensor for every encoding and return [B,E,S,T,...]."""

    if input_tensor.ndim < 3:
        raise ValueError(f"Expected [E,N,...], got {tuple(input_tensor.shape)}")
    gathered = input_tensor[:, window_idx]
    permutation = (1, 0, 2, 3, *range(4, gathered.ndim))
    return gathered.permute(permutation).contiguous()


def select_spatial_mask_slab(case_mask, micro_b, final_shape, num_slices):
    """Select [B,S,Z,Y] ROI slabs using the same clamped raw-x indices as input."""

    if case_mask is None:
        return None
    mask = torch.as_tensor(case_mask, dtype=torch.float32)
    if mask.ndim != 3:
        raise ValueError(f"Expected ROI mask [x,z,y], got {tuple(mask.shape)}")
    total_slices = int(final_shape[-4])
    if mask.shape[0] != total_slices:
        raise ValueError(f"ROI x dimension {mask.shape[0]} does not match raw-x slices {total_slices}")
    half = num_slices // 2
    rows = []
    for flat_index in micro_b:
        slice_index = int(flat_index) % total_slices
        rows.append([max(0, min(total_slices - 1, slice_index + offset)) for offset in range(-half, num_slices - half)])
    indices = torch.tensor(rows, dtype=torch.long)
    return mask[indices]
