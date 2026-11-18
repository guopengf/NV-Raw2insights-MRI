"""CG-SENSE (SciPy CG) and multi-coil CS-TV (SigPy's original application)."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, cg
from torch.utils.data import DataLoader

from ..common.data import TrainSlabs, fft2
from ..common.manifest import read_manifest, validate_train_index
from ..common.online_sampling import sampling_log
from ..common.path_mapping import protected_case_paths
from ..common.safety import OutputGuard, ROOT, sha256
from pathlib import Path


def cg_sense(batch, options):
    k = batch["kspace"].cpu()
    coils, mask = batch["coils"].cpu(), batch["mask"].cpu()
    _, e, t, _, s, h, w = k.shape
    output = torch.empty((1, e, t, s, h, w), dtype=torch.complex64)
    statuses = []
    for ei in range(e):
        for ti in range(t):
            for si in range(s):
                c = coils[0, :, si]
                m = mask[0, ti]
                rhs = (fft2(k[0, ei, ti, :, si], inverse=True) * c.conj()).sum(dim=0)

                def normal(vector):
                    x = torch.from_numpy(vector.reshape(h, w)).to(torch.complex64)
                    return ((fft2(fft2(x * c) * m, inverse=True) * c.conj()).sum(dim=0) + options["l2"] * x).numpy().ravel()

                operator = LinearOperator((h * w, h * w), matvec=normal, dtype=np.complex64)
                answer, info = cg(operator, rhs.numpy().ravel(), rtol=options["tolerance"], atol=0,
                                  maxiter=options["iterations"])
                output[0, ei, ti, si] = torch.from_numpy(answer.reshape(h, w))
                statuses.append(int(info))
    if any(v < 0 for v in statuses):
        raise RuntimeError(f"CG breakdown: {statuses}")
    return output, {"cg_converged": sum(i == 0 for i in statuses), "cg_solves": len(statuses), "cg_status": statuses}


def cs_tv(batch, options):
    import sigpy as sp
    import sigpy.mri as mr

    k = batch["kspace"].numpy()
    coils = batch["coils"][0].numpy()
    mask = batch["mask"][0].numpy()
    _, e, t, c, s, h, w = k.shape
    output = np.empty((1, e, t, s, h, w), np.complex64)
    # A unitary FFT along x turns the hybrid slab into a 3D encoding problem.
    # Mask is constant over x, so this does not change the measured data problem.
    raw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(k, axes=4), axis=4, norm="ortho"), axes=4).astype(np.complex64)
    for ei in range(e):
        for ti in range(t):
            app = mr.app.TotalVariationRecon(raw[0, ei, ti], coils, lamda=options["tv_weight"],
                                             weights=np.broadcast_to(mask[ti][None], (s, h, w)),
                                             device=sp.cpu_device, max_iter=options["iterations"],
                                             tol=options["tolerance"], show_pbar=False)
            output[0, ei, ti] = np.asarray(app.run())
    return torch.from_numpy(output), {"implementation": "sigpy.mri.app.TotalVariationRecon", "iterations": options["iterations"]}


def execute_solver(args):
    cfg = json.loads(args.config.read_text())
    p = cfg["paper"]
    source_mapping = ROOT / p["source_mapping"] if p.get("source_mapping") else None
    cases, _ = read_manifest(ROOT / p["manifest"], source_mapping)
    protected = [ROOT / p["manifest"], *protected_case_paths(cases), *p["protected_roots"]]
    if source_mapping is not None:
        protected.append(source_mapping.parent)
    guard = OutputGuard(protected)
    output = guard.directory(ROOT / p["output_dir"] / "smoke")
    report = {"experiment": p["name"], "status": "FAIL", "test_data_used": False,
              "config_sha256": sha256(args.config),
              "source_mapping_sha256": sha256(source_mapping) if source_mapping else None,
              "seed": p["seed"], "scope": "one train window, not a full study"}
    start = time.monotonic()
    try:
        dataset = TrainSlabs(ROOT / p["train_index"], seed=p["seed"], num_slices=p["num_slices"],
                             num_frames=p["num_frames"], accelerations=p["accelerations"], smoke=True,
                             online_undersampling=p.get("online_undersampling"))
        validate_train_index(dataset.cases, cases)
        before = {str(source): (Path(source).stat().st_size, Path(source).stat().st_mtime_ns)
                  for source in dataset.cases[0]["files"].values()}
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        if dataset.online_sampler is not None:
            report["online_undersampling"] = dataset.online_sampler.provenance()
            report["sampling"] = sampling_log(batch)
        pred, details = {"cg_sense": cg_sense, "cs_tv": cs_tv}[p["method"]](batch, cfg["solver"])
        if pred.shape != batch["target"].shape or not torch.isfinite(pred).all():
            raise ValueError(f"Invalid complex output: {pred.shape}")
        path = guard.check(output / "train_window_complex.npy")
        with path.open("wb") as f:
            np.save(f, pred.numpy())
        report.update(status="PASS", case=batch["case_id"], source=batch["source"],
                      output_shape=list(pred.shape), reconstruction=details,
                      nominal_R=batch["acceleration"].tolist(), output=str(path))
        if p["method"] == "cg_sense" and details["cg_converged"] < details["cg_solves"]:
            report["warnings"] = ["CG iteration budget exhausted before requested tolerance; workflow PASS is not a convergence/accuracy claim"]
        for source, metadata in before.items():
            stat = Path(source).stat()
            if metadata != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"Read-only input changed: {source}")
        report["read_only_inputs_unchanged"] = len(before)
    except Exception as exc:
        import traceback

        report["error"], report["traceback"] = f"{type(exc).__name__}: {exc}", traceback.format_exc()
    report["seconds"] = time.monotonic() - start
    guard.json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)
