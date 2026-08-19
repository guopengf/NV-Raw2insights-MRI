"""Windowed HDF5 storage and loading for 4D-flow training."""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import h5py
import numpy as np
import torch
from monai.data.fft_utils import ifftn_centered

from four_dflow_augmentation import FourDFlowOnlineAugmenter
from mra_utils import load_vessel_mask_prior, read_mat_array, read_real_mat_array
from transforms import raw_4dflow_coilmap_to_hybrid, raw_4dflow_to_joint_hybrid
from utils import complex_zscore, is_slab_recon, slab_num_slices


WINDOWED_4DFLOW_SCHEMA = "raw2insights.4dflow.windowed_hdf5"
WINDOWED_4DFLOW_SCHEMA_VERSION = 2
LEGACY_WINDOWED_4DFLOW_SCHEMA_VERSION = 1
ENCODING_CHUNK_1 = "encoding_chunk_1"
ENCODING_CHUNK_ALL = "encoding_chunk_all"
WINDOWED_4DFLOW_STORAGE_PROFILES = (ENCODING_CHUNK_1, ENCODING_CHUNK_ALL)


def cfg_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def windowed_hdf5_enabled(args: Any, split: str = "train") -> bool:
    backend = str(cfg_get(args, f"four_dflow_storage.{split}_backend", "")).lower()
    if not backend:
        backend = str(cfg_get(args, "four_dflow_storage.backend", "raw_mat")).lower()
    return backend in {"windowed_hdf5", "hdf5", "h5"}


def _source_record(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    stat = source.stat()
    return {
        "path": str(source),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_patient_source_manifest(
    *,
    target_path: str | Path,
    acceleration_inputs: dict[int, str | Path],
    acceleration_masks: dict[int, str | Path],
    coilmap_path: str | Path | None = None,
    segmask_path: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "target": _source_record(target_path),
        "inputs": {str(key): _source_record(value) for key, value in acceleration_inputs.items()},
        "masks": {str(key): _source_record(value) for key, value in acceleration_masks.items()},
        "coilmap": _source_record(coilmap_path),
        "segmask": _source_record(segmask_path),
    }


def _json_attr(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _atomic_json_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _encoding_chunk_size(storage_profile: str, n_encodings: int) -> int:
    if storage_profile == ENCODING_CHUNK_1:
        return 1
    if storage_profile == ENCODING_CHUNK_ALL:
        return int(n_encodings)
    raise ValueError(f"Unsupported windowed 4D-flow storage profile: {storage_profile!r}")


def _dataset_chunks(shape: Sequence[int], storage_profile: str) -> tuple[int, ...]:
    if len(shape) != 7:
        raise ValueError(f"Expected [T,X,E,C,Z,Y,2], got {tuple(shape)}")
    return (
        1,
        1,
        _encoding_chunk_size(storage_profile, int(shape[2])),
        *tuple(int(value) for value in shape[3:]),
    )


def _mask_to_compact(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        return mask[None]
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4:
        if mask.shape[0] == 1:
            return mask[0]
        if mask.shape[-1] == 1:
            return mask[..., 0]
    if mask.ndim == 5:
        if mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 4 and mask.shape[1] == 1:
            return mask[:, 0]
        if mask.ndim == 4 and mask.shape[-1] == 1:
            return mask[..., 0]
    if mask.ndim == 6:
        return mask[0, :, 0, :, :, 0]
    raise ValueError(f"Unsupported 4D-flow mask shape: {mask.shape}")


def _coilmap_to_compact(
    coilmap: np.ndarray,
    *,
    nx: int,
    nc: int,
    n_enc: int,
    axis_order: str,
    normalize: bool,
) -> tuple[np.ndarray, str]:
    maps = [
        raw_4dflow_coilmap_to_hybrid(
            coilmap,
            nt=1,
            nx=nx,
            nc=nc,
            n_enc=n_enc,
            enc_idx=encoding_idx,
            axis_order=axis_order,
            normalize=normalize,
        )[0]
        for encoding_idx in range(n_enc)
    ]
    if all(np.array_equal(maps[0], value) for value in maps[1:]):
        return maps[0], "shared"
    return np.stack(maps, axis=1), "encoding"


def validate_patient_store(
    path: str | Path,
    *,
    deep: bool = False,
    expected_storage_profile: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    with h5py.File(path, "r", swmr=True) as store:
        if store.attrs.get("schema") != WINDOWED_4DFLOW_SCHEMA:
            raise ValueError(f"Unexpected schema in {path}: {store.attrs.get('schema')!r}")
        schema_version = int(store.attrs.get("schema_version", -1))
        if schema_version not in {
            LEGACY_WINDOWED_4DFLOW_SCHEMA_VERSION,
            WINDOWED_4DFLOW_SCHEMA_VERSION,
        }:
            raise ValueError(f"Unsupported schema version in {path}")
        if not bool(store.attrs.get("complete", False)):
            raise ValueError(f"Incomplete windowed store: {path}")
        storage_profile = str(store.attrs.get("storage_profile", ENCODING_CHUNK_1))
        if storage_profile not in WINDOWED_4DFLOW_STORAGE_PROFILES:
            raise ValueError(f"Unsupported storage profile in {path}: {storage_profile!r}")
        if expected_storage_profile is not None and storage_profile != expected_storage_profile:
            raise ValueError(
                f"Storage profile mismatch in {path}: expected {expected_storage_profile!r}, "
                f"found {storage_profile!r}"
            )
        target = store["hybrid/target"]
        if target.ndim != 7 or target.shape[-1] != 2 or target.dtype != np.float32:
            raise ValueError(f"Invalid target dataset in {path}: shape={target.shape}, dtype={target.dtype}")
        expected_chunks = _dataset_chunks(target.shape, storage_profile)
        if target.chunks != expected_chunks:
            raise ValueError(
                f"Target chunks mismatch in {path}: expected {expected_chunks}, found {target.chunks}"
            )
        encoding_chunk_size = _encoding_chunk_size(storage_profile, int(target.shape[2]))
        stored_encoding_chunk_size = int(store.attrs.get("encoding_chunk_size", encoding_chunk_size))
        if stored_encoding_chunk_size != encoding_chunk_size:
            raise ValueError(
                f"Encoding chunk metadata mismatch in {path}: expected {encoding_chunk_size}, "
                f"found {stored_encoding_chunk_size}"
            )
        acceleration_keys = sorted(store["hybrid/input"], key=int)
        if not acceleration_keys:
            raise ValueError(f"No acceleration inputs in {path}")
        for acceleration in acceleration_keys:
            input_dataset = store[f"hybrid/input/{acceleration}"]
            if input_dataset.shape != target.shape or input_dataset.dtype != np.float32:
                raise ValueError(
                    f"Input {acceleration} mismatch in {path}: "
                    f"shape={input_dataset.shape}, dtype={input_dataset.dtype}"
                )
            if input_dataset.chunks != expected_chunks:
                raise ValueError(
                    f"Input {acceleration} chunks mismatch in {path}: "
                    f"expected {expected_chunks}, found {input_dataset.chunks}"
                )
            compact_mask = store[f"mask/{acceleration}"]
            if compact_mask.shape != (target.shape[0], target.shape[4], target.shape[5]):
                raise ValueError(f"Mask {acceleration} mismatch in {path}: {compact_mask.shape}")
        if deep:
            probes = [(0, 0), (target.shape[0] - 1, target.shape[1] - 1)]
            for frame_idx, slice_idx in probes:
                if not np.isfinite(target[frame_idx, slice_idx]).all():
                    raise ValueError(f"Non-finite target chunk in {path} at {(frame_idx, slice_idx)}")
                for acceleration in acceleration_keys:
                    if not np.isfinite(store[f"hybrid/input/{acceleration}"][frame_idx, slice_idx]).all():
                        raise ValueError(
                            f"Non-finite input chunk in {path} for acceleration {acceleration}"
                        )
        return {
            "path": str(path),
            "patient_key": str(store.attrs["patient_key"]),
            "target_kspace": str(store.attrs["target_kspace"]),
            "shape": [int(value) for value in target.shape],
            "accelerations": [int(value) for value in acceleration_keys],
            "storage_profile": storage_profile,
            "encoding_chunk_size": encoding_chunk_size,
            "hybrid_chunks": [int(value) for value in expected_chunks],
            "size_bytes": int(path.stat().st_size),
            "source_manifest": json.loads(str(store.attrs["source_manifest_json"])),
        }


def _create_hybrid_dataset(
    store: h5py.File,
    name: str,
    data: np.ndarray,
    storage_profile: str,
) -> None:
    store.create_dataset(
        name,
        data=data,
        dtype=np.float32,
        chunks=_dataset_chunks(data.shape, storage_profile),
    )


def convert_patient_to_windowed_hdf5_profiles(
    *,
    patient_key: str,
    target_path: str | Path,
    acceleration_inputs: dict[int, str | Path],
    acceleration_masks: dict[int, str | Path],
    output_paths: dict[str, str | Path],
    coilmap_path: str | Path | None = None,
    segmask_path: str | Path | None = None,
    coilmap_axis_order: str = "auto",
    normalize_coilmap: bool = True,
    mask_args: Any = None,
    overwrite: bool = False,
) -> dict[str, dict[str, Any]]:
    if not output_paths:
        raise ValueError("At least one output storage profile is required")
    invalid_profiles = sorted(set(output_paths) - set(WINDOWED_4DFLOW_STORAGE_PROFILES))
    if invalid_profiles:
        raise ValueError(f"Unsupported storage profiles: {invalid_profiles}")

    normalized_paths = {profile: Path(path) for profile, path in output_paths.items()}
    if len(set(normalized_paths.values())) != len(normalized_paths):
        raise ValueError("Each storage profile must use a distinct output path")
    for output_path in normalized_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    source_manifest = build_patient_source_manifest(
        target_path=target_path,
        acceleration_inputs=acceleration_inputs,
        acceleration_masks=acceleration_masks,
        coilmap_path=coilmap_path,
        segmask_path=segmask_path,
    )
    records = {}
    profiles_to_write = []
    for storage_profile, output_path in normalized_paths.items():
        if output_path.exists() and not overwrite:
            record = validate_patient_store(
                output_path,
                expected_storage_profile=storage_profile,
            )
            if record["source_manifest"] != source_manifest:
                raise ValueError(
                    f"Source files changed after conversion for {output_path}; "
                    "use --overwrite to rebuild it explicitly"
                )
            records[storage_profile] = record
        else:
            profiles_to_write.append(storage_profile)
    if not profiles_to_write:
        return records

    temporaries = {
        profile: normalized_paths[profile].with_name(
            f".{normalized_paths[profile].name}.{os.getpid()}.tmp"
        )
        for profile in profiles_to_write
    }
    for temporary in temporaries.values():
        if temporary.exists():
            temporary.unlink()

    target_raw = read_mat_array(target_path, ("kdata_full", "kdata", "kspace_full", "kspace"))
    if target_raw.ndim != 6:
        raise ValueError(f"Expected target [E,T,C,Kz,Ky,Kx], got {target_raw.shape} in {target_path}")
    target_hybrid = raw_4dflow_to_joint_hybrid(target_raw)
    del target_raw
    nt, nx, n_enc, nc, nz, ny, two = target_hybrid.shape
    if two != 2:
        raise ValueError(f"Unexpected target hybrid shape: {target_hybrid.shape}")

    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            stores = {
                profile: stack.enter_context(h5py.File(temporaries[profile], "w", libver="latest"))
                for profile in profiles_to_write
            }
            for storage_profile, store in stores.items():
                store.attrs["schema"] = WINDOWED_4DFLOW_SCHEMA
                store.attrs["schema_version"] = WINDOWED_4DFLOW_SCHEMA_VERSION
                store.attrs["storage_profile"] = storage_profile
                store.attrs["encoding_chunk_size"] = _encoding_chunk_size(storage_profile, n_enc)
                store.attrs["complete"] = False
                store.attrs["patient_key"] = patient_key
                store.attrs["target_kspace"] = str(target_path)
                store.attrs["created_unix"] = time.time()
                store.attrs["source_manifest_json"] = _json_attr(source_manifest)
                store.attrs["coilmap_axis_order"] = coilmap_axis_order
                store.attrs["normalize_coilmap"] = bool(normalize_coilmap)
                _create_hybrid_dataset(store, "hybrid/target", target_hybrid, storage_profile)
            del target_hybrid

            for acceleration in sorted(acceleration_inputs):
                input_raw = read_mat_array(
                    acceleration_inputs[acceleration],
                    ("kdata", "kdata_ktGaussian", "kus", "kspace", "kspace_full"),
                )
                input_hybrid = raw_4dflow_to_joint_hybrid(input_raw)
                del input_raw
                if input_hybrid.shape != (nt, nx, n_enc, nc, nz, ny, 2):
                    raise ValueError(
                        f"Acceleration {acceleration} shape mismatch for {patient_key}: {input_hybrid.shape}"
                    )
                for storage_profile, store in stores.items():
                    _create_hybrid_dataset(
                        store,
                        f"hybrid/input/{acceleration}",
                        input_hybrid,
                        storage_profile,
                    )
                del input_hybrid

                mask_raw = read_real_mat_array(
                    acceleration_masks[acceleration],
                    ("mask", "usmask", "usmask_ktGaussian", "sampling_mask"),
                )
                compact_mask = _mask_to_compact(mask_raw)
                del mask_raw
                if compact_mask.shape != (nt, nz, ny):
                    raise ValueError(
                        f"Acceleration {acceleration} mask mismatch for {patient_key}: {compact_mask.shape}"
                    )
                for store in stores.values():
                    store.create_dataset(
                        f"mask/{acceleration}",
                        data=compact_mask,
                        dtype=np.float32,
                        chunks=(1, nz, ny),
                    )

            if coilmap_path is not None:
                coilmap = read_mat_array(
                    coilmap_path,
                    ("coilmap", "csm", "sensitivity_maps", "sens_maps"),
                )
                compact_coilmap, layout = _coilmap_to_compact(
                    coilmap,
                    nx=nx,
                    nc=nc,
                    n_enc=n_enc,
                    axis_order=coilmap_axis_order,
                    normalize=normalize_coilmap,
                )
                del coilmap
                for store in stores.values():
                    coil_dataset = store.create_dataset(
                        "coilmap",
                        data=compact_coilmap,
                        dtype=np.float32,
                        chunks=(1, *compact_coilmap.shape[1:]),
                    )
                    coil_dataset.attrs["layout"] = layout

            if segmask_path is not None:
                segmask = load_vessel_mask_prior(segmask_path, mask_args)
                if segmask.shape != (nx, nz, ny):
                    raise ValueError(f"Segmentation mask mismatch for {patient_key}: {segmask.shape}")
                for store in stores.values():
                    store.create_dataset("segmask", data=segmask, dtype=np.float32, chunks=(1, nz, ny))

            conversion_seconds = time.perf_counter() - started
            for store in stores.values():
                store.attrs["conversion_seconds"] = conversion_seconds
                store.attrs["complete"] = True
                store.flush()

        for storage_profile in profiles_to_write:
            os.replace(temporaries[storage_profile], normalized_paths[storage_profile])
    except BaseException:
        for temporary in temporaries.values():
            if temporary.exists():
                temporary.unlink()
        raise

    for storage_profile in profiles_to_write:
        records[storage_profile] = validate_patient_store(
            normalized_paths[storage_profile],
            deep=True,
            expected_storage_profile=storage_profile,
        )
    return records


def convert_patient_to_windowed_hdf5(
    *,
    patient_key: str,
    target_path: str | Path,
    acceleration_inputs: dict[int, str | Path],
    acceleration_masks: dict[int, str | Path],
    output_path: str | Path,
    coilmap_path: str | Path | None = None,
    segmask_path: str | Path | None = None,
    coilmap_axis_order: str = "auto",
    normalize_coilmap: bool = True,
    mask_args: Any = None,
    overwrite: bool = False,
    storage_profile: str = ENCODING_CHUNK_1,
) -> dict[str, Any]:
    records = convert_patient_to_windowed_hdf5_profiles(
        patient_key=patient_key,
        target_path=target_path,
        acceleration_inputs=acceleration_inputs,
        acceleration_masks=acceleration_masks,
        output_paths={storage_profile: output_path},
        coilmap_path=coilmap_path,
        segmask_path=segmask_path,
        coilmap_axis_order=coilmap_axis_order,
        normalize_coilmap=normalize_coilmap,
        mask_args=mask_args,
        overwrite=overwrite,
    )
    return records[storage_profile]


def discover_4dflow_patients(
    data_roots: Iterable[str | Path], accelerations: Sequence[int]
) -> list[dict[str, Any]]:
    patients = []
    for root_value in data_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for center_dir in sorted(root.glob("Center*")):
            for scanner_dir in sorted(path for path in center_dir.iterdir() if path.is_dir()):
                for patient_dir in sorted(path for path in scanner_dir.iterdir() if path.is_dir()):
                    target = patient_dir / "kdata_full.mat"
                    if not target.exists():
                        continue
                    inputs = {
                        int(acceleration): patient_dir / f"kdata_ktGaussian{int(acceleration)}.mat"
                        for acceleration in accelerations
                    }
                    masks = {
                        int(acceleration): patient_dir / f"usmask_ktGaussian{int(acceleration)}.mat"
                        for acceleration in accelerations
                    }
                    missing = [str(path) for path in [*inputs.values(), *masks.values()] if not path.exists()]
                    if missing:
                        continue
                    patients.append(
                        {
                            "patient_key": "/".join((center_dir.name, scanner_dir.name, patient_dir.name)),
                            "target_path": target,
                            "inputs": inputs,
                            "masks": masks,
                            "coilmap_path": patient_dir / "coilmap.mat" if (patient_dir / "coilmap.mat").exists() else None,
                            "segmask_path": patient_dir / "segmask.mat" if (patient_dir / "segmask.mat").exists() else None,
                        }
                    )
    return patients


def rebuild_windowed_index(output_root: str | Path, *, deep: bool = False) -> dict[str, Any]:
    output_root = Path(output_root)
    patient_records = [
        validate_patient_store(path, deep=deep)
        for path in sorted((output_root / "patients").glob("**/*.h5"))
    ]
    if not patient_records:
        raise ValueError(f"No converted patient stores found under {output_root}")
    storage_profiles = {record["storage_profile"] for record in patient_records}
    encoding_chunk_sizes = {record["encoding_chunk_size"] for record in patient_records}
    if len(storage_profiles) != 1 or len(encoding_chunk_sizes) != 1:
        raise ValueError(f"Mixed storage profiles found under {output_root}")
    payload = {
        "schema": WINDOWED_4DFLOW_SCHEMA,
        "schema_version": WINDOWED_4DFLOW_SCHEMA_VERSION,
        "storage_profile": next(iter(storage_profiles)),
        "encoding_chunk_size": next(iter(encoding_chunk_sizes)),
        "patients": patient_records,
    }
    _atomic_json_dump(payload, output_root / "index.json")
    return payload


def validate_windowed_profile_pair(
    e1_output_root: str | Path,
    e4_output_root: str | Path,
    *,
    deep: bool = False,
) -> dict[str, Any]:
    e1_index = rebuild_windowed_index(e1_output_root, deep=deep)
    e4_index = rebuild_windowed_index(e4_output_root, deep=deep)
    if e1_index["storage_profile"] != ENCODING_CHUNK_1:
        raise ValueError(f"Expected E1 profile under {e1_output_root}")
    if e4_index["storage_profile"] != ENCODING_CHUNK_ALL:
        raise ValueError(f"Expected E4 profile under {e4_output_root}")

    e1_patients = {record["patient_key"]: record for record in e1_index["patients"]}
    e4_patients = {record["patient_key"]: record for record in e4_index["patients"]}
    if set(e1_patients) != set(e4_patients):
        raise ValueError("E1 and E4 patient sets do not match")
    for patient_key in sorted(e1_patients):
        e1_record = e1_patients[patient_key]
        e4_record = e4_patients[patient_key]
        for field in ("shape", "accelerations", "source_manifest"):
            if e1_record[field] != e4_record[field]:
                raise ValueError(f"E1/E4 {field} mismatch for {patient_key}")
        if not deep:
            continue
        with h5py.File(e1_record["path"], "r", swmr=True) as e1_store, h5py.File(
            e4_record["path"], "r", swmr=True
        ) as e4_store:
            hybrid_names = ["hybrid/target"] + [
                f"hybrid/input/{acceleration}" for acceleration in e1_record["accelerations"]
            ]
            for name in hybrid_names:
                for frame_idx, slice_idx in (
                    (0, 0),
                    (e1_store[name].shape[0] - 1, e1_store[name].shape[1] - 1),
                ):
                    if not np.array_equal(
                        e1_store[name][frame_idx, slice_idx],
                        e4_store[name][frame_idx, slice_idx],
                    ):
                        raise ValueError(f"E1/E4 value mismatch for {patient_key} at {name}")
            compact_names = [f"mask/{acceleration}" for acceleration in e1_record["accelerations"]]
            compact_names.extend(name for name in ("coilmap", "segmask") if name in e1_store)
            for name in compact_names:
                if not np.array_equal(e1_store[name][0], e4_store[name][0]):
                    raise ValueError(f"E1/E4 compact value mismatch for {patient_key} at {name}")
                if not np.array_equal(e1_store[name][-1], e4_store[name][-1]):
                    raise ValueError(f"E1/E4 compact value mismatch for {patient_key} at {name}")

    return {
        "schema": WINDOWED_4DFLOW_SCHEMA,
        "schema_version": WINDOWED_4DFLOW_SCHEMA_VERSION,
        "patients": len(e1_patients),
        "e1_index": str(Path(e1_output_root) / "index.json"),
        "e4_index": str(Path(e4_output_root) / "index.json"),
        "deep": bool(deep),
    }


def build_windowed_4dflow_manifests(
    *,
    index_path: str | Path,
    data_roots: Iterable[str | Path],
    out_dir: str | Path,
    accelerations: Sequence[int],
    encodings: Sequence[int],
    joint_encodings: bool,
    expected_storage_profile: str | None = None,
    allow_profile_mismatch: bool = False,
) -> list[Path]:
    index_path = Path(index_path)
    with index_path.open() as stream:
        index = json.load(stream)
    if index.get("schema") != WINDOWED_4DFLOW_SCHEMA:
        raise ValueError(f"Unexpected windowed index schema: {index_path}")
    if int(index.get("schema_version", -1)) not in {
        LEGACY_WINDOWED_4DFLOW_SCHEMA_VERSION,
        WINDOWED_4DFLOW_SCHEMA_VERSION,
    }:
        raise ValueError(f"Unsupported windowed index version: {index_path}")
    storage_profile = str(index.get("storage_profile", ENCODING_CHUNK_1))
    profile_requirement = expected_storage_profile
    if profile_requirement is None:
        profile_requirement = ENCODING_CHUNK_ALL if joint_encodings else ENCODING_CHUNK_1
    if profile_requirement not in WINDOWED_4DFLOW_STORAGE_PROFILES:
        raise ValueError(f"Unsupported requested storage profile: {profile_requirement!r}")
    if not allow_profile_mismatch and storage_profile != profile_requirement:
        mode = "joint" if joint_encodings else "legacy"
        raise ValueError(
            f"{mode} training requires storage profile {profile_requirement!r}, "
            f"but {index_path} provides {storage_profile!r}"
        )

    roots = [os.path.normpath(str(value)) for value in data_roots]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    requested_accelerations = {int(value) for value in accelerations}
    encoding_groups = [tuple(int(value) for value in encodings)] if joint_encodings else [
        (int(value),) for value in encodings
    ]
    for patient in index["patients"]:
        target_path = os.path.normpath(str(patient["target_kspace"]))
        if roots and not any(target_path == root or target_path.startswith(root + os.sep) for root in roots):
            continue
        available = {int(value) for value in patient["accelerations"]}
        for acceleration in sorted(requested_accelerations & available):
            for encoding_group in encoding_groups:
                payload = {
                    "windowed_h5": patient["path"],
                    "target_kspace": patient["target_kspace"],
                    "kspace": f"{patient['path']}::hybrid/input/{acceleration}",
                    "mask_type": f"ktGaussian{acceleration}",
                    "acceleration": acceleration,
                    "acquisition": "Flow4d",
                    "is_4dflow": True,
                    "prewindowed_4dflow": True,
                    "storage_profile": storage_profile,
                    "encoding_chunk_size": int(
                        patient.get("encoding_chunk_size", index.get("encoding_chunk_size", 1))
                    ),
                }
                if joint_encodings:
                    payload["joint_encodings"] = True
                    payload["encoding_indices"] = list(encoding_group)
                    suffix = "joint4"
                else:
                    payload["joint_encodings"] = False
                    payload["encoding_idx"] = encoding_group[0]
                    suffix = f"enc{encoding_group[0]}"
                safe_patient = patient["patient_key"].replace("/", "__")
                path = out_dir / f"{safe_patient}__ktGaussian{acceleration}__{suffix}.json"
                _atomic_json_dump(payload, path)
                manifests.append(path)
    return manifests


def build_window_indices(
    centers: Sequence[int], *, total_frames: int, total_slices: int, num_frames: int, num_slices: int
) -> np.ndarray:
    frame_half = num_frames // 2
    slice_half = num_slices // 2
    result = []
    for center in centers:
        frame_idx = int(center) // total_slices
        slice_idx = int(center) % total_slices
        rows = []
        for slice_offset in range(-slice_half, num_slices - slice_half):
            selected_slice = max(0, min(total_slices - 1, slice_idx + slice_offset))
            row = []
            for frame_offset in range(-frame_half, num_frames - frame_half):
                selected_frame = (frame_idx + frame_offset) % total_frames
                row.append((selected_frame, selected_slice))
            rows.append(row)
        result.append(rows)
    return np.asarray(result, dtype=np.int64)


def _read_hybrid_windows(dataset: h5py.Dataset, indices: np.ndarray, encoding_idx: int | None) -> np.ndarray:
    values = {}
    for frame_idx, slice_idx in sorted({tuple(value) for value in indices.reshape(-1, 2)}):
        if encoding_idx is None:
            values[(frame_idx, slice_idx)] = dataset[frame_idx, slice_idx]
        else:
            values[(frame_idx, slice_idx)] = dataset[frame_idx, slice_idx, encoding_idx]
    nested = [
        [
            [values[tuple(indices[n, slice_offset, frame_offset])] for frame_offset in range(indices.shape[2])]
            for slice_offset in range(indices.shape[1])
        ]
        for n in range(indices.shape[0])
    ]
    return np.asarray(nested, dtype=np.float32)


def _read_mask_windows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    frames = {int(value) for value in indices[..., 0].reshape(-1)}
    values = {frame: dataset[frame] for frame in frames}
    return np.asarray(
        [
            [
                [values[int(indices[n, s, t, 0])] for t in range(indices.shape[2])]
                for s in range(indices.shape[1])
            ]
            for n in range(indices.shape[0])
        ],
        dtype=np.float32,
    )


class Windowed4DFlowDataset(torch.utils.data.Dataset):
    """Read only selected temporal/slice windows from converted patient stores."""

    def __init__(
        self,
        data: Sequence[dict[str, Any] | str | Path],
        args: Any,
        *,
        center_selector: Callable[[int, int], Sequence[int]] | None = None,
    ) -> None:
        self.data = list(data)
        self.args = args
        self.num_samples = int(args.num_samples_per_case)
        self.num_frames = int(args.num_frames)
        self.num_slices = slab_num_slices(args) if is_slab_recon(args) else 1
        self.center_selector = center_selector
        self.handle_capacity = max(1, int(cfg_get(args, "four_dflow_storage.handle_cache_entries", 1)))
        self.rdcc_nbytes = max(0, int(cfg_get(args, "four_dflow_storage.hdf5_chunk_cache_bytes", 256 << 20)))
        self.augmenter = FourDFlowOnlineAugmenter(args)
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()
        if not bool(getattr(args, "is_4dflow_aorta", False)):
            raise ValueError("windowed_hdf5 is only supported for is_4dflow_aorta=true")
        if [str(value).lower() for value in getattr(args, "train_mask_types", [])] != ["fixed"]:
            raise ValueError("windowed_hdf5 currently requires train_mask_types=['fixed']")
        if bool(cfg_get(args, "phase3.enable_vaa", False)) or bool(cfg_get(args, "phase3.loss.use_vascular", False)):
            raise ValueError("windowed_hdf5 v1 does not yet support MRA/VAA priors")

    def __len__(self) -> int:
        return len(self.data)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        return state

    def close(self) -> None:
        while self._handles:
            _, handle = self._handles.popitem(last=False)
            handle.close()

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass

    def _open(self, path: str, expected_storage_profile: str) -> h5py.File:
        if path in self._handles:
            handle = self._handles.pop(path)
            self._handles[path] = handle
        else:
            handle = h5py.File(path, "r", swmr=True, rdcc_nbytes=self.rdcc_nbytes)
            if handle.attrs.get("schema") != WINDOWED_4DFLOW_SCHEMA or not bool(
                handle.attrs.get("complete", False)
            ):
                handle.close()
                raise ValueError(f"Invalid or incomplete windowed store: {path}")
            self._handles[path] = handle
            while len(self._handles) > self.handle_capacity:
                _, evicted = self._handles.popitem(last=False)
                evicted.close()
        storage_profile = str(handle.attrs.get("storage_profile", ENCODING_CHUNK_1))
        if storage_profile != expected_storage_profile:
            raise ValueError(
                f"Manifest/store storage profile mismatch for {path}: "
                f"expected {expected_storage_profile!r}, found {storage_profile!r}"
            )
        return handle

    def _choose_centers(self, total: int) -> np.ndarray:
        count = min(total, self.num_samples)
        if self.center_selector is not None:
            centers = np.asarray(self.center_selector(total, count), dtype=np.int64)
        else:
            centers = np.random.choice(total, size=count, replace=False)
        if centers.shape != (count,) or np.any(centers < 0) or np.any(centers >= total):
            raise ValueError(f"Invalid selected centers: {centers}")
        return centers

    def __getitem__(self, index: int) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        item = self.data[index]
        manifest_path = Path(item["kspace"] if isinstance(item, dict) else item)
        with manifest_path.open() as stream:
            manifest = json.load(stream)
        store = self._open(
            str(manifest["windowed_h5"]),
            str(manifest.get("storage_profile", ENCODING_CHUNK_1)),
        )
        acceleration = str(int(manifest["acceleration"]))
        target_dataset = store["hybrid/target"]
        total_frames, total_slices, encoding_count, coils, height, width, _ = target_dataset.shape
        centers = self._choose_centers(total_frames * total_slices)
        indices = build_window_indices(
            centers,
            total_frames=total_frames,
            total_slices=total_slices,
            num_frames=self.num_frames,
            num_slices=self.num_slices,
        )
        joint = bool(manifest.get("joint_encodings", False))
        encoding_idx = None if joint else int(manifest["encoding_idx"])

        target_started = time.perf_counter()
        target_hybrid = _read_hybrid_windows(target_dataset, indices, encoding_idx)
        target_read_ms = (time.perf_counter() - target_started) * 1000.0

        mask_started = time.perf_counter()
        compact_mask = torch.from_numpy(_read_mask_windows(store[f"mask/{acceleration}"], indices))
        mask_read_ms = (time.perf_counter() - mask_started) * 1000.0

        sensitivity = None
        coilmap_read_ms = 0.0
        if "coilmap" in store:
            coil_started = time.perf_counter()
            coilmap = store["coilmap"]
            layout = str(coilmap.attrs["layout"])
            slice_values = sorted({int(value) for value in indices[..., 1].reshape(-1)})
            cached_slices = {slice_idx: coilmap[slice_idx] for slice_idx in slice_values}
            nested = np.asarray(
                [
                    [
                        [cached_slices[int(indices[n, s, t, 1])] for t in range(indices.shape[2])]
                        for s in range(indices.shape[1])
                    ]
                    for n in range(indices.shape[0])
                ],
                dtype=np.float32,
            )
            if joint:
                if layout == "shared":
                    nested = np.repeat(nested[:, :, :, None], encoding_count, axis=3)
                elif layout != "encoding":
                    raise ValueError(f"Unknown coilmap layout: {layout}")
                sensitivity = torch.from_numpy(nested).contiguous()
            else:
                if layout == "encoding":
                    nested = nested[:, :, :, encoding_idx]
                elif layout != "shared":
                    raise ValueError(f"Unknown coilmap layout: {layout}")
                sensitivity = torch.from_numpy(nested).contiguous()
            coilmap_read_ms = (time.perf_counter() - coil_started) * 1000.0

        joint_segmask = None
        if "segmask" in store:
            segmask = store["segmask"]
            joint_segmask = torch.from_numpy(
                np.asarray(
                    [
                        [segmask[int(indices[n, s, 0, 1])] for s in range(indices.shape[1])]
                        for n in range(indices.shape[0])
                    ],
                    dtype=np.float32,
                )
            )

        input_hybrid = None
        input_read_ms = 0.0
        if not self.augmenter.enabled:
            input_started = time.perf_counter()
            input_hybrid = _read_hybrid_windows(store[f"hybrid/input/{acceleration}"], indices, encoding_idx)
            input_read_ms = (time.perf_counter() - input_started) * 1000.0

        ifft_started = time.perf_counter()
        target_image = ifftn_centered(torch.from_numpy(target_hybrid), spatial_dims=2, is_complex=True)
        augmentation_ms = 0.0
        augmentation_params = None
        if self.augmenter.enabled:
            augmentation_started = time.perf_counter()
            augmented = self.augmenter(
                target_image,
                compact_mask,
                joint_encodings=joint,
                sensitivity_maps=sensitivity,
                segmask=joint_segmask,
            )
            input_image = augmented["input_image"]
            target_image = augmented["target_image"]
            sensitivity = augmented["sensitivity_maps"]
            joint_segmask = augmented["segmask"]
            augmentation_params = augmented["params"]
            augmentation_ms = (time.perf_counter() - augmentation_started) * 1000.0
        else:
            assert input_hybrid is not None
            input_image = ifftn_centered(torch.from_numpy(input_hybrid), spatial_dims=2, is_complex=True)
        ifft_ms = (time.perf_counter() - ifft_started) * 1000.0 - augmentation_ms

        if joint:
            input_image = input_image.permute(0, 3, 1, 2, 4, 5, 6, 7).contiguous()
            target_image = target_image.permute(0, 3, 1, 2, 4, 5, 6, 7).contiguous()
            if sensitivity is not None:
                sensitivity = sensitivity.permute(0, 3, 1, 2, 4, 5, 6, 7).contiguous()
            input_image, mean, std = complex_zscore(input_image, dim=[4, 5, 6])
            mask = compact_mask[:, None, :, :, None, :, :, None].expand(
                -1, encoding_count, -1, -1, 1, -1, -1, 1
            )
        else:
            input_image = input_image.contiguous()
            target_image = target_image.contiguous()
            input_image, mean, std = complex_zscore(input_image, dim=[3, 4, 5])
            mask = compact_mask[:, :, :, None, :, :, None]

        worker_timing = {
            "worker_total_ms": (time.monotonic_ns() - started_ns) / 1.0e6,
            "load_total_ms": (time.monotonic_ns() - started_ns) / 1.0e6,
            "input_read_ms": input_read_ms,
            "target_read_ms": target_read_ms,
            "mask_read_ms": mask_read_ms,
            "coilmap_read_ms": coilmap_read_ms,
            "ifft_ms": ifft_ms,
            "augmentation_ms": augmentation_ms,
            "input_read_skipped": float(self.augmenter.enabled),
            "window_count": int(centers.size),
            "unique_coordinate_count": int(len({tuple(value) for value in indices.reshape(-1, 2)})),
            "hybrid_read_call_count": int(len({tuple(value) for value in indices.reshape(-1, 2)})),
        }
        metadata = {
            "filename": manifest_path.name,
            "shape": np.asarray([total_frames, total_slices, coils, height, width], dtype=np.int32),
            "num_encodings": encoding_count,
            "joint_encodings": joint,
            "window_centers": centers,
            "worker_timing": worker_timing,
        }
        if augmentation_params is not None:
            metadata["four_dflow_augmentation"] = augmentation_params
        result = {
            "kspace_masked_ifft": input_image,
            "kspace_ifft": target_image,
            "mask": mask.contiguous(),
            "mask_type": manifest["mask_type"],
            "acc_factor": int(manifest["acceleration"]),
            "acquisition": manifest.get("acquisition", "Flow4d"),
            "mean": mean.contiguous(),
            "std": std.contiguous(),
            "kspace_meta_dict": metadata,
            "prewindowed_4dflow": True,
        }
        if sensitivity is not None:
            result["sensitivity_maps"] = sensitivity
        if joint_segmask is not None:
            result["joint_segmask"] = joint_segmask.contiguous()
        return result
