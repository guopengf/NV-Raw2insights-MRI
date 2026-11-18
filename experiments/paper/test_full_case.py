"""Full coverage, phase/scale/axis correctness and legacy prediction compatibility."""

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from .common.case_data import FullCase
from .common.data import sense_adjoint
from .common.prediction_io import load_prediction, read_index
from .common.safety import OutputGuard
from .common.windows import Assembly, windows
from .infer import output_lock, reconstruct


@pytest.fixture
def case_and_truth(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    e, nt, nc, z, y, x = 4, 7, 2, 4, 8, 5
    grid = np.indices((nt, z, y, x))
    mag = 1 + grid[0] * 0.1 + grid[3] * 0.03
    truth = np.stack([mag * np.exp(1j * (0.15 * ei + 0.01 * grid[0] + 0.02 * grid[3])) for ei in range(e)]).astype(np.complex64)
    maps = np.ones((nc, z, y, x), np.complex64) / np.sqrt(nc)
    maps[1] *= 1j
    coil_images = truth[:, :, None] * maps[None, None]
    k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(coil_images, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="ortho"), axes=(-3, -2, -1)).astype(np.complex64)
    arrays = {"kdata_full.mat": ("kdata_full", k), "coilmap.mat": ("coilmap", maps),
              "segmask.mat": ("segmask", np.ones((z, y, x), np.uint8)),
              "segmask_new.mat": ("segmask", np.ones((nt, z, y, x), np.uint8))}
    for filename, (key, array) in arrays.items():
        with h5py.File(source / filename, "w") as f:
            f[key] = array
    with (source / "params.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("VENC", "venc_order", "spatial_order", "matrix_size"))
        writer.writeheader()
        writer.writerow({"VENC": "100;150;200", "venc_order": "RL;AP;HF", "spatial_order": "HF;AP;RL",
                         "matrix_size": ";".join(map(str, (e, nt, nc, z, y, x)))})
    np.savez_compressed(source / "corrmap.npz", coords=np.zeros((0, 5), np.int32), data=np.array([], np.float32), shape=np.array((3, 1, z, y, x)))
    case = {"case_id": "Center000_P000001_Aorta", "dataset_group": "train", "directory": str(source),
            "files": {name: str(source / name) for name in (*arrays, "params.csv", "corrmap.npz")}}
    return case, truth, OutputGuard([source])


@pytest.mark.parametrize("merge,center_time,ns,nf", [("center", True, 3, 5), ("tile", True, 3, 5),
                                                  ("tile", False, 3, 5), ("tile", False, 64, 6),
                                                  ("overlap", False, 3, 5), ("overlap", True, 3, 5)])
def test_every_time_and_slice_is_assembled_once_or_averaged_without_phase_loss(case_and_truth, tmp_path, merge, center_time, ns, nf):
    case, truth, guard = case_and_truth
    data = FullCase(case, tmp_path / "cache", guard)
    np.testing.assert_allclose(data.gt, truth, atol=1e-6)
    nt, nx = truth.shape[1], truth.shape[-1]
    result = np.empty_like(truth)
    coverage = np.zeros((nt, nx), np.uint32)
    assembly = Assembly(result, coverage)
    full_mask = np.ones((nt, *truth.shape[2:4]), np.uint8)
    for window in windows(nt, nx, nf, ns, center_time=center_time, merge=merge):
        batch = data.batch(window.frames, window.slices, 10, full_mask)
        # Actual measured data through SENSE adjoint, without reading GT in this predictor.
        pred = sense_adjoint(batch["kspace"], batch["coils"], batch["mask"]).numpy()
        if center_time:
            pred = np.repeat(pred[:, :, nf // 2:nf // 2 + 1], nf, axis=2)
        assembly.add(pred, window, float(batch["scale"][0]))
    stats = assembly.finish()
    assert stats["complete"] and coverage.min() >= 1
    if merge in {"center", "tile"}:
        assert coverage.max() == 1
    np.testing.assert_allclose(result, truth, rtol=2e-6, atol=2e-6)
    data.assert_unchanged()


def test_mask_is_a_full_time_sequence_independent_of_context_length(case_and_truth, tmp_path):
    case, _, guard = case_and_truth
    a = FullCase(case, tmp_path / "cache", guard)
    m, info = a.fixed_mask(10, 2026)
    b = FullCase(case, tmp_path / "cache", guard)
    other, other_info = b.fixed_mask(10, 2026)
    assert info["sha256"] == other_info["sha256"]
    assert m.shape == (7, 4, 8)
    np.testing.assert_array_equal(m, other)
    small = a.batch([2, 3, 4], [0, 1, 2], 10, m)
    large = b.batch([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4], 10, other)
    np.testing.assert_array_equal(small["mask"][0], large["mask"][0, 2:5])


def test_assembly_rejects_holes():
    with pytest.raises(ValueError, match="uncovered"):
        Assembly(np.zeros((4, 7, 4, 8, 5), np.complex64), np.zeros((7, 5), np.uint32)).finish()


def test_legacy_four_mat_real_imag_files_keep_encoding_and_yzxt_order(case_and_truth, tmp_path):
    import scipy.io

    _, truth, _ = case_and_truth
    rows = []
    for ei in range(4):
        raw = truth[ei].transpose(2, 1, 3, 0)  # [y,z,x,t]
        path = tmp_path / f"enc{ei}.mat"
        scipy.io.savemat(path, {"img4ranking": np.stack((raw.real, raw.imag), axis=-1)})
        rows.append({"case_id": "Center000_P000001_Aorta", "R": 10, "encoding": ei,
                     "recon_path": path.name, "layout": "yzxt", "key": "img4ranking"})
    index = tmp_path / "predictions.csv"
    with index.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(reversed(rows))
    entries = read_index(index)
    np.testing.assert_array_equal(load_prediction(entries[0]), truth)


def test_completion_marker_skip_lock_and_failed_overwrite(case_and_truth, tmp_path):
    from .common.prediction_io import discover

    case, truth, guard = case_and_truth
    data = FullCase(case, tmp_path / "cache", guard)
    directory = guard.directory(tmp_path / "predictions")
    request = {"experiment": "synthetic", "method": "cg_sense", "case_id": case["case_id"], "R": 10,
               "dataset_group": "train", "shape": list(truth.shape), "axes": "etzyx",
               "config_sha256": "config-v1", "source_signature": data.signature,
               "window": {"frames": 5, "slices": 3, "merge": "tile"}, "mask": {"sha256": "test-mask"}}

    class MeasuredAdjoint:
        provenance = {}
        calls = 0

        def __call__(self, batch):
            self.calls += 1
            return sense_adjoint(batch["kspace"], batch["coils"], batch["mask"]).numpy()

    predictor = MeasuredAdjoint()
    mask = np.ones((truth.shape[1], *truth.shape[2:4]), np.uint8)
    reconstruct(data, predictor, directory, guard, request, mask)
    calls = predictor.calls
    np.testing.assert_allclose(load_prediction(discover(directory)[0]), truth, atol=2e-6)
    reconstruct(data, predictor, directory, guard, request, mask)
    assert predictor.calls == calls
    with pytest.raises(ValueError, match="different weights/mask/config/window/sources"):
        reconstruct(data, predictor, directory, guard, {**request, "config_sha256": "config-v2"}, mask)
    with output_lock(directory, guard):
        with pytest.raises(RuntimeError, match="locked"):
            reconstruct(data, predictor, directory, guard, request, mask)
        with pytest.raises(ValueError, match="incomplete|running|locked"):
            discover(directory)

    def fail(_):
        raise RuntimeError("synthetic predictor failure")

    fail.provenance = {}
    with pytest.raises(RuntimeError, match="predictor failure"):
        reconstruct(data, fail, directory, guard, request, mask, overwrite=True)
    assert not (directory / "reconstruction.json").exists()
    assert not (directory / "inference.lock").exists()
    assert not list(directory.glob("*.tmp.npy"))
    assert discover(directory) == []


def test_metric_adapter_passes_original_axes_mask_corrmap_and_case_venc(case_and_truth, tmp_path):
    from .evaluate import score

    case, truth, guard = case_and_truth
    data = FullCase(case, tmp_path / "cache", guard)
    recorded = {}

    def core(funcs, pred, gt, seg, **kwargs):
        recorded.update(kwargs)
        assert funcs == "official-functions"
        np.testing.assert_allclose(gt, truth, atol=2e-6)
        np.testing.assert_allclose(pred, truth, atol=2e-6)
        assert seg.shape == truth.shape[2:]
        return {"SSIM": 1.0}

    settings = {"mask_images": True, "use_venc": True, "spatial_mode": "bbox", "include_complex_diff": True}
    assert score(data, truth, settings, core, "official-functions") == {"SSIM": 1.0}
    np.testing.assert_array_equal(recorded["venc"], [100, 150, 200])
    assert recorded["corr_maps"].shape == (3, 1, *truth.shape[2:])
    assert recorded["spatial_mode"] == "bbox" and recorded["include_complex_diff"]


@pytest.mark.parametrize("imaginary", [0.0, 0.1])
def test_complex_stored_corrmap_requires_zero_imaginary_part(case_and_truth, tmp_path, imaginary):
    from .evaluate import score

    case, truth, guard = case_and_truth
    np.savez_compressed(case["files"]["corrmap.npz"], coords=np.array([[0, 0, 0, 0, 0]], np.int32),
                        data=np.array([0.2 + 1j * imaginary], np.complex64), shape=np.array((3, 1, *truth.shape[2:])))
    data = FullCase(case, tmp_path / "cache", guard)
    settings = {"mask_images": True, "use_venc": False, "spatial_mode": "bbox", "include_complex_diff": True}

    def core(*args, corr_maps, **kwargs):
        assert corr_maps.dtype == np.float32
        assert corr_maps[0, 0, 0, 0, 0] == np.float32(0.2)
        return {"SSIM": 1.0}

    if imaginary:
        with pytest.raises(ValueError, match="real phase offsets"):
            score(data, truth, settings, core, {})
    else:
        assert score(data, truth, settings, core, {}) == {"SSIM": 1.0}
