"""Export only a smoke train window, explicitly not a full challenge submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from .common.safety import OutputGuard, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    args = p.parse_args()
    guard = OutputGuard()
    # Restrict the supplied identifier to one directory, not a path.
    if Path(args.experiment).name != args.experiment or args.experiment in {".", ".."}:
        p.error("--experiment must be a registered experiment name")
    report = json.loads((ROOT / "outputs/paper" / args.experiment / "smoke/report.json").read_text())
    if report["status"] != "PASS" or report.get("test_data_used", True):
        raise ValueError("Export requires a passing Train-only readiness check")
    path = ROOT / "outputs/paper" / args.experiment / "smoke/train_window_complex.npy"
    array = np.load(path, allow_pickle=False)
    if array.ndim != 6 or array.shape[:2] != (1, 4) or not np.iscomplexobj(array):
        raise ValueError(f"Expected complex [1,4,T,X,Z,Y], got {array.shape}, {array.dtype}")
    temporal_selection = "all genuinely reconstructed window frames"
    if report.get("method") == "ours":
        center = array.shape[2] // 2
        array = array[:, :, center:center + 1]
        temporal_selection = "center frame only; native T repetitions are not separate reconstructions"
    array = array[0].transpose(0, 1, 3, 4, 2).astype(np.complex64)
    output = guard.directory(ROOT / "outputs/paper/test_output" / args.experiment)
    mat = guard.check(output / "train_window_complex.mat")
    savemat(mat, {"reconstruction": array}, do_compression=True)
    npz = guard.check(output / "train_window_complex.npz")
    np.savez_compressed(npz, reconstruction=array)
    roundtrip = np.load(npz, allow_pickle=False)["reconstruction"]
    if not np.array_equal(roundtrip, array):
        raise AssertionError("Complex export roundtrip changed data")
    guard.json(output / "export.json", {"shape": list(array.shape), "axes": "encoding,time,z,y,x_window",
                                        "amplitude_domain": "normalized acquired-kspace RMS, not raw scanner units",
                                        "temporal_selection": temporal_selection,
                                        "scope": "Train-only smoke window; NOT a challenge submission or full case",
                                        "real_imag_preserved": True, "source_report": report})
    print(f"PASS complex export {args.experiment}: {array.shape} -> {output}")


if __name__ == "__main__":
    main()
