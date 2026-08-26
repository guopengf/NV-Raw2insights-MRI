from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import scipy.io
import torch
from monai.data.fft_utils import fftn_centered, ifftn_centered


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from joint_encoding import gather_joint_window, joint_windowed_input_x_slab
from transforms import (
    raw_4dflow_coilmap_to_hybrid,
    raw_4dflow_coilmap_to_joint_hybrid,
    raw_4dflow_mask_to_hybrid,
    raw_4dflow_mask_to_joint_hybrid,
    raw_4dflow_to_hybrid,
    raw_4dflow_to_joint_hybrid,
)
from utils import complex_zscore, load_config, windowed_input_x_slab
from windowed_4dflow import (
    ENCODING_CHUNK_1,
    ENCODING_CHUNK_ALL,
    Windowed4DFlowDataset,
    build_window_indices,
    build_windowed_4dflow_manifests,
    convert_patient_to_windowed_hdf5,
    convert_patient_to_windowed_hdf5_profiles,
    rebuild_windowed_index,
    validate_patient_store,
    validate_windowed_profile_pair,
    windowed_hdf5_enabled,
)


def _args():
    return SimpleNamespace(
        dataset="CMRxRecon",
        is_4dflow_aorta=True,
        is_multi_coil=True,
        num_samples_per_case=2,
        num_frames=3,
        data_aug=False,
        train_mask_types=["fixed"],
        phase3=SimpleNamespace(
            recon_mode="slab",
            num_slices=3,
            enable_vaa=False,
            loss=SimpleNamespace(use_vascular=False),
        ),
        four_dflow_storage=SimpleNamespace(
            handle_cache_entries=1,
            hdf5_chunk_cache_bytes=16 << 20,
        ),
    )


def _augmentation_args():
    args = _args()
    args.data_aug = True
    args.four_dflow_augmentation = SimpleNamespace(
        enabled=True,
        flip=SimpleNamespace(prob=1.0, axes=["z", "y"]),
        shift=SimpleNamespace(prob=1.0, max_pixels=[1, 2]),
        contrast=SimpleNamespace(prob=1.0, gamma=[0.9, 1.1]),
    )
    return args


def _write_synthetic_patient(root: Path):
    rng = np.random.default_rng(11)
    encodings, frames, coils, height, width, slices = 4, 3, 2, 4, 5, 6
    shape = (encodings, frames, coils, height, width, slices)
    target = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    masked = (target * (0.75 + 0.1j)).astype(np.complex64)
    mask = rng.integers(0, 2, size=(1, frames, 1, height, width, 1)).astype(np.float32)
    coilmap = (
        rng.normal(size=(coils, height, width, slices))
        + 1j * rng.normal(size=(coils, height, width, slices))
    ).astype(np.complex64)

    patient = root / "Center001" / "ScannerA" / "P001"
    patient.mkdir(parents=True)
    target_path = patient / "kdata_full.mat"
    input_path = patient / "kdata_ktGaussian10.mat"
    mask_path = patient / "usmask_ktGaussian10.mat"
    coilmap_path = patient / "coilmap.mat"
    scipy.io.savemat(target_path, {"kdata_full": target})
    scipy.io.savemat(input_path, {"kdata": masked})
    scipy.io.savemat(mask_path, {"mask": mask})
    scipy.io.savemat(coilmap_path, {"coilmap": coilmap})
    return {
        "root": root,
        "target": target,
        "masked": masked,
        "mask": mask,
        "coilmap": coilmap,
        "target_path": target_path,
        "input_path": input_path,
        "mask_path": mask_path,
        "coilmap_path": coilmap_path,
        "final_shape": [frames, slices, coils, height, width],
    }


def _convert_and_manifest(tmp_path: Path, *, joint: bool):
    source = _write_synthetic_patient(tmp_path / "raw")
    output_root = tmp_path / "converted"
    store_path = output_root / "patients" / "Center001" / "ScannerA" / "P001.h5"
    storage_profile = ENCODING_CHUNK_ALL if joint else ENCODING_CHUNK_1
    convert_patient_to_windowed_hdf5(
        patient_key="Center001/ScannerA/P001",
        target_path=source["target_path"],
        acceleration_inputs={10: source["input_path"]},
        acceleration_masks={10: source["mask_path"]},
        output_path=store_path,
        coilmap_path=source["coilmap_path"],
        storage_profile=storage_profile,
    )
    index = rebuild_windowed_index(output_root, deep=True)
    manifests = build_windowed_4dflow_manifests(
        index_path=output_root / "index.json",
        data_roots=[source["root"]],
        out_dir=tmp_path / ("joint_manifests" if joint else "single_manifests"),
        accelerations=[10],
        encodings=[0, 1, 2, 3],
        joint_encodings=joint,
    )
    return source, store_path, index, manifests


def test_build_window_indices_wraps_time_and_clamps_slices():
    result = build_window_indices(
        [0, 17], total_frames=3, total_slices=6, num_frames=3, num_slices=3
    )
    assert result[0].tolist() == [
        [[2, 0], [0, 0], [1, 0]],
        [[2, 0], [0, 0], [1, 0]],
        [[2, 1], [0, 1], [1, 1]],
    ]
    assert result[1].tolist() == [
        [[1, 4], [2, 4], [0, 4]],
        [[1, 5], [2, 5], [0, 5]],
        [[1, 5], [2, 5], [0, 5]],
    ]


def test_windowed_backend_is_opt_in_for_joint_and_legacy_configs():
    raw_config = SimpleNamespace(
        four_dflow_storage=SimpleNamespace(backend="raw_mat")
    )
    assert not windowed_hdf5_enabled(raw_config, "train")
    assert not windowed_hdf5_enabled(raw_config, "val")

    opt_in_names = (
        "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_windowed_h5_pg.json",
        "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json",
        "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_channel_pg.json",
    )
    for name in opt_in_names:
        config = load_config(REPO_ROOT / "configs" / name)
        assert windowed_hdf5_enabled(config, "train")
        assert not windowed_hdf5_enabled(config, "val")
        assert config.batch_size == 8
        assert config.num_samples_per_case == 8
    legacy = load_config(REPO_ROOT / "configs" / opt_in_names[0])
    joint = load_config(REPO_ROOT / "configs" / opt_in_names[1])
    joint_channel = load_config(REPO_ROOT / "configs" / opt_in_names[2])
    assert legacy.four_dflow_storage.storage_profile == ENCODING_CHUNK_1
    assert joint.four_dflow_storage.storage_profile == ENCODING_CHUNK_1
    assert joint_channel.four_dflow_storage.storage_profile == ENCODING_CHUNK_1
    assert "/windowed-e1-v2/" in legacy.four_dflow_storage.index_path
    assert "/windowed-e1-v2/" in joint.four_dflow_storage.index_path
    assert "/windowed-e1-v2/" in joint_channel.four_dflow_storage.index_path


def test_converter_is_restart_safe_and_manifests_cover_both_modes(tmp_path):
    source, store_path, index, joint_manifests = _convert_and_manifest(tmp_path, joint=True)
    assert len(index["patients"]) == 1
    assert len(joint_manifests) == 1
    joint_payload = json.loads(joint_manifests[0].read_text())
    assert joint_payload["encoding_indices"] == [0, 1, 2, 3]

    record = convert_patient_to_windowed_hdf5(
        patient_key="Center001/ScannerA/P001",
        target_path=source["target_path"],
        acceleration_inputs={10: source["input_path"]},
        acceleration_masks={10: source["mask_path"]},
        output_path=store_path,
        coilmap_path=source["coilmap_path"],
        storage_profile=ENCODING_CHUNK_ALL,
    )
    assert record["shape"] == [3, 6, 4, 2, 4, 5, 2]

    single_root = tmp_path / "converted_e1"
    convert_patient_to_windowed_hdf5(
        patient_key="Center001/ScannerA/P001",
        target_path=source["target_path"],
        acceleration_inputs={10: source["input_path"]},
        acceleration_masks={10: source["mask_path"]},
        output_path=single_root / "patients" / "Center001" / "ScannerA" / "P001.h5",
        coilmap_path=source["coilmap_path"],
        storage_profile=ENCODING_CHUNK_1,
    )
    rebuild_windowed_index(single_root, deep=True)
    single_manifests = build_windowed_4dflow_manifests(
        index_path=single_root / "index.json",
        data_roots=[source["root"]],
        out_dir=tmp_path / "single_manifests",
        accelerations=[10],
        encodings=[0, 1, 2, 3],
        joint_encodings=False,
    )
    assert len(single_manifests) == 4
    assert {json.loads(path.read_text())["encoding_idx"] for path in single_manifests} == {0, 1, 2, 3}
    assert validate_patient_store(store_path, deep=True)["size_bytes"] > 0


def test_paired_converter_writes_equal_e1_and_e4_profiles(tmp_path):
    source = _write_synthetic_patient(tmp_path / "raw")
    roots = {
        ENCODING_CHUNK_1: tmp_path / "e1",
        ENCODING_CHUNK_ALL: tmp_path / "e4",
    }
    output_paths = {
        profile: root / "patients" / "Center001" / "ScannerA" / "P001.h5"
        for profile, root in roots.items()
    }
    records = convert_patient_to_windowed_hdf5_profiles(
        patient_key="Center001/ScannerA/P001",
        target_path=source["target_path"],
        acceleration_inputs={10: source["input_path"]},
        acceleration_masks={10: source["mask_path"]},
        output_paths=output_paths,
        coilmap_path=source["coilmap_path"],
    )
    assert records[ENCODING_CHUNK_1]["hybrid_chunks"] == [1, 1, 1, 2, 4, 5, 2]
    assert records[ENCODING_CHUNK_ALL]["hybrid_chunks"] == [1, 1, 4, 2, 4, 5, 2]
    with h5py.File(output_paths[ENCODING_CHUNK_1], "r") as e1_store, h5py.File(
        output_paths[ENCODING_CHUNK_ALL], "r"
    ) as e4_store:
        assert e1_store["hybrid/target"].chunks == (1, 1, 1, 2, 4, 5, 2)
        assert e4_store["hybrid/target"].chunks == (1, 1, 4, 2, 4, 5, 2)
        np.testing.assert_array_equal(e1_store["hybrid/target"][:], e4_store["hybrid/target"][:])
        np.testing.assert_array_equal(e1_store["hybrid/input/10"][:], e4_store["hybrid/input/10"][:])
    pair = validate_windowed_profile_pair(roots[ENCODING_CHUNK_1], roots[ENCODING_CHUNK_ALL], deep=True)
    assert pair["patients"] == 1

    restarted = convert_patient_to_windowed_hdf5_profiles(
        patient_key="Center001/ScannerA/P001",
        target_path=source["target_path"],
        acceleration_inputs={10: source["input_path"]},
        acceleration_masks={10: source["mask_path"]},
        output_paths=output_paths,
        coilmap_path=source["coilmap_path"],
    )
    assert set(restarted) == {ENCODING_CHUNK_1, ENCODING_CHUNK_ALL}


def test_manifest_profile_guard_rejects_wrong_training_mode(tmp_path):
    source = _write_synthetic_patient(tmp_path / "raw")
    output_root = tmp_path / "e1"
    store_path = output_root / "patients" / "Center001" / "ScannerA" / "P001.h5"
    convert_patient_to_windowed_hdf5(
        patient_key="Center001/ScannerA/P001",
        target_path=source["target_path"],
        acceleration_inputs={10: source["input_path"]},
        acceleration_masks={10: source["mask_path"]},
        output_path=store_path,
        storage_profile=ENCODING_CHUNK_1,
    )
    rebuild_windowed_index(output_root)
    try:
        build_windowed_4dflow_manifests(
            index_path=output_root / "index.json",
            data_roots=[source["root"]],
            out_dir=tmp_path / "manifests",
            accelerations=[10],
            encodings=[0, 1, 2, 3],
            joint_encodings=True,
        )
    except ValueError as error:
        assert "joint training requires storage profile" in str(error)
    else:
        raise AssertionError("Expected profile mismatch to be rejected")

    manifests = build_windowed_4dflow_manifests(
        index_path=output_root / "index.json",
        data_roots=[source["root"]],
        out_dir=tmp_path / "configured_manifests",
        accelerations=[10],
        encodings=[0, 1, 2, 3],
        joint_encodings=True,
        expected_storage_profile=ENCODING_CHUNK_1,
    )
    assert len(manifests) == 1


def test_joint_windowed_dataset_matches_existing_full_volume_pipeline(tmp_path):
    source, _, _, manifests = _convert_and_manifest(tmp_path, joint=True)
    centers = np.asarray([0, 17], dtype=np.int64)
    dataset = Windowed4DFlowDataset(
        [{"kspace": manifests[0]}], _args(), center_selector=lambda total, count: centers
    )
    sample = dataset[0]

    input_full = ifftn_centered(
        torch.from_numpy(raw_4dflow_to_joint_hybrid(source["masked"])), spatial_dims=2, is_complex=True
    ).reshape(18, 4, 2, 4, 5, 2)
    target_full = ifftn_centered(
        torch.from_numpy(raw_4dflow_to_joint_hybrid(source["target"])), spatial_dims=2, is_complex=True
    ).reshape(18, 4, 2, 4, 5, 2)
    input_norm, mean, std = complex_zscore(input_full, dim=[2, 3, 4])
    expected_input, indices = joint_windowed_input_x_slab(
        input_norm, centers, source["final_shape"], num_frames=3, num_slices=3
    )
    expected_target = gather_joint_window(target_full, indices)
    mask_full = torch.from_numpy(
        raw_4dflow_mask_to_joint_hybrid(source["mask"], n_enc=4, nx=6)
    ).reshape(18, 4, 1, 4, 5, 1)
    expected_mask = gather_joint_window(mask_full, indices)
    expected_mean = gather_joint_window(mean, indices)
    expected_std = gather_joint_window(std, indices)
    csm_full = torch.from_numpy(
        raw_4dflow_coilmap_to_joint_hybrid(
            source["coilmap"], nt=3, nx=6, nc=2, n_enc=4, normalize=True
        )
    ).reshape(18, 4, 2, 4, 5, 2)
    expected_csm = gather_joint_window(csm_full, indices)

    torch.testing.assert_close(sample["kspace_masked_ifft"], expected_input)
    torch.testing.assert_close(sample["kspace_ifft"], expected_target)
    torch.testing.assert_close(sample["mask"], expected_mask)
    torch.testing.assert_close(sample["mean"], expected_mean)
    torch.testing.assert_close(sample["std"], expected_std)
    torch.testing.assert_close(sample["sensitivity_maps"], expected_csm)
    assert sample["kspace_meta_dict"]["window_centers"].tolist() == centers.tolist()
    timing = sample["kspace_meta_dict"]["worker_timing"]
    assert timing["hybrid_read_call_count"] == timing["unique_coordinate_count"]


def test_non_joint_windowed_dataset_matches_existing_full_volume_pipeline(tmp_path):
    source, _, _, manifests = _convert_and_manifest(tmp_path, joint=False)
    manifest = next(path for path in manifests if json.loads(path.read_text())["encoding_idx"] == 2)
    centers = np.asarray([0, 17], dtype=np.int64)
    dataset = Windowed4DFlowDataset(
        [{"kspace": manifest}], _args(), center_selector=lambda total, count: centers
    )
    sample = dataset[0]

    input_full = ifftn_centered(
        torch.from_numpy(raw_4dflow_to_hybrid(source["masked"][2:3])), spatial_dims=2, is_complex=True
    ).reshape(18, 2, 4, 5, 2)
    target_full = ifftn_centered(
        torch.from_numpy(raw_4dflow_to_hybrid(source["target"][2:3])), spatial_dims=2, is_complex=True
    ).reshape(18, 2, 4, 5, 2)
    input_norm, mean, std = complex_zscore(input_full, dim=[1, 2, 3])
    expected_input, indices = windowed_input_x_slab(
        input_norm, centers, source["final_shape"], num_frames=3, num_slices=3
    )
    expected_target = torch.as_tensor(target_full[indices])
    mask_full = torch.from_numpy(raw_4dflow_mask_to_hybrid(source["mask"], n_enc=1, nx=6)).reshape(
        18, 1, 4, 5, 1
    )
    expected_mask = torch.as_tensor(mask_full[indices])
    expected_mean = torch.as_tensor(mean[indices])
    expected_std = torch.as_tensor(std[indices])
    csm_full = torch.from_numpy(
        raw_4dflow_coilmap_to_hybrid(
            source["coilmap"], nt=3, nx=6, nc=2, n_enc=4, enc_idx=2, normalize=True
        )
    ).reshape(18, 2, 4, 5, 2)
    expected_csm = torch.as_tensor(csm_full[indices])

    torch.testing.assert_close(sample["kspace_masked_ifft"], expected_input)
    torch.testing.assert_close(sample["kspace_ifft"], expected_target)
    torch.testing.assert_close(sample["mask"], expected_mask)
    torch.testing.assert_close(sample["mean"], expected_mean)
    torch.testing.assert_close(sample["std"], expected_std)
    torch.testing.assert_close(sample["sensitivity_maps"], expected_csm)


def test_windowed_online_augmentation_regenerates_exact_masked_input(tmp_path):
    _, _, _, manifests = _convert_and_manifest(tmp_path, joint=True)
    centers = np.asarray([0, 17], dtype=np.int64)
    torch.manual_seed(31)
    sample = Windowed4DFlowDataset(
        [{"kspace": manifests[0]}],
        _augmentation_args(),
        center_selector=lambda total, count: centers,
    )[0]

    input_image = sample["kspace_masked_ifft"] * sample["std"] + sample["mean"]
    input_kspace = torch.view_as_complex(
        fftn_centered(input_image, spatial_dims=2, is_complex=True).contiguous()
    )
    target_kspace = torch.view_as_complex(
        fftn_centered(sample["kspace_ifft"], spatial_dims=2, is_complex=True).contiguous()
    )
    mask = sample["mask"][..., 0]
    torch.testing.assert_close(input_kspace, target_kspace * mask, atol=2e-5, rtol=2e-5)

    metadata = sample["kspace_meta_dict"]
    assert set(metadata["four_dflow_augmentation"]) == {
        "flip_z",
        "flip_y",
        "shift_z",
        "shift_y",
        "gamma",
    }
    assert metadata["worker_timing"]["input_read_skipped"] == 1.0
    assert metadata["worker_timing"]["input_read_ms"] == 0.0
