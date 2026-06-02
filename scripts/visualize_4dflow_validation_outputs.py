import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def to_complex(arr):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return arr["real"] + 1j * arr["imag"]
    if arr.ndim > 0 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    raise ValueError(f"Unsupported complex array: dtype={arr.dtype}, shape={arr.shape}")


def robust01(x):
    x = np.abs(np.asarray(x, dtype=np.float32))
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)


def get_kspace_dataset(h5_file):
    if "kdata_ktGaussian" in h5_file:
        return h5_file["kdata_ktGaussian"]
    candidates = [key for key in h5_file.keys() if key.startswith("kdata")]
    if not candidates:
        raise KeyError(f"No kdata dataset found in {h5_file.filename}; keys={list(h5_file.keys())}")
    return h5_file[candidates[0]]


def zero_filled_frame(kspace_path: Path, enc: int, time_idx: int):
    with h5py.File(kspace_path, "r") as f:
        ds = get_kspace_dataset(f)
        kshape = ds.shape
        if len(kshape) == 6:
            raw = to_complex(ds[enc : enc + 1, time_idx : time_idx + 1])
        elif len(kshape) == 5:
            raw = to_complex(ds[time_idx : time_idx + 1])[None, ...]
        else:
            raise ValueError(f"Expected 5D/6D k-space in {kspace_path}, got shape={kshape}")

    hybrid = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(raw, axes=(-1,)), axis=-1, norm="ortho"), axes=(-1,))
    img = np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(hybrid, axes=(-3, -2)), axes=(-3, -2), norm="ortho"),
        axes=(-3, -2),
    )
    zfill = np.sqrt(np.sum(np.abs(img[0, 0]) ** 2, axis=0)).astype(np.float32)
    return zfill, kshape


def save_grid(enc, acc, case_label, kind, positions, zf, recon, time_idx, out_path):
    fig, axes = plt.subplots(2, len(positions), figsize=(2.35 * len(positions), 5.0), constrained_layout=True)
    for col, pos in enumerate(positions):
        if kind == "zy":
            us = zf[:, :, pos]
            rc = recon[:, :, pos, time_idx].T
            title = f"x={pos}"
        elif kind == "xy":
            us = zf[pos, :, :]
            rc = recon[:, pos, :, time_idx]
            title = f"z={pos}"
        else:
            raise ValueError(kind)

        axes[0, col].imshow(robust01(us), cmap="gray", origin="lower")
        axes[1, col].imshow(robust01(rc), cmap="gray", origin="lower")
        axes[0, col].set_title(title, fontsize=10)
        for row in range(2):
            axes[row, col].axis("off")

    axes[0, 0].set_ylabel("Undersampled\nIFFT", fontsize=11)
    axes[1, 0].set_ylabel("Recon", fontsize=11)
    fig.suptitle(f"{case_label} ktGaussian{acc} enc{enc} t={time_idx} {kind.upper()} slices", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def visualize_item(out_root: Path, item: dict):
    anatomy = item.get("anatomy")
    enc = int(item["encoding_idx"])
    acc = int(item["acc"])
    center = item["center"]
    scanner = item["scanner"]
    case_id = item["case_id"]
    case_label = f"{center}/{scanner}/{case_id}"
    if anatomy:
        case_label = f"{anatomy}/{case_label}"

    with open(item["json_path"]) as f:
        payload = json.load(f)

    kspace_path = Path(payload["kspace"])
    final_base = out_root / "final"
    if anatomy:
        final_base = final_base / anatomy
    recon_path = final_base / center / scanner / case_id / f"kdata_ktGaussian{acc}_enc{enc}_recon.mat"
    if not recon_path.exists():
        raise FileNotFoundError(f"Missing recon file: {recon_path}")

    recon = np.asarray(sio.loadmat(recon_path)["img4ranking"]).squeeze().astype(np.float32)
    if recon.ndim != 4:
        raise ValueError(f"Expected 4D recon in {recon_path}, got shape={recon.shape}")

    with h5py.File(kspace_path, "r") as f:
        kshape = get_kspace_dataset(f).shape
    nt = kshape[1] if len(kshape) == 6 else kshape[0]
    nz, _ny, nx = kshape[-3], kshape[-2], kshape[-1]
    time_idx = min(nt // 2, recon.shape[3] - 1)

    zf, _ = zero_filled_frame(kspace_path, enc, time_idx)
    x_positions = np.linspace(max(0, nx // 6), min(nx - 1, 5 * nx // 6), 7, dtype=int)
    z_positions = np.linspace(max(0, nz // 10), min(nz - 1, 9 * nz // 10), 7, dtype=int)

    safe_scanner = scanner.replace("/", "_")
    viz_dir = out_root / "viz"
    if anatomy:
        viz_dir = viz_dir / anatomy
    viz_dir = viz_dir / center / safe_scanner / case_id
    zy_path = viz_dir / f"{case_id}_ktGaussian{acc}_enc{enc}_t{time_idx}_zy_x7_us_vs_recon.png"
    xy_path = viz_dir / f"{case_id}_ktGaussian{acc}_enc{enc}_t{time_idx}_xy_z7_us_vs_recon.png"
    save_grid(enc, acc, case_label, "zy", x_positions, zf, recon, time_idx, zy_path)
    save_grid(enc, acc, case_label, "xy", z_positions, zf, recon, time_idx, xy_path)
    print(f"[VIZ] {zy_path}")
    print(f"[VIZ] {xy_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_root = args.out_root
    manifest_path = args.manifest or out_root / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    if args.limit is not None:
        manifest = manifest[: args.limit]

    for idx, item in enumerate(manifest, start=1):
        print(f"[INFO] Visualizing {idx}/{len(manifest)}: {item['stem']}")
        visualize_item(out_root, item)

    print(f"[DONE] Wrote {2 * len(manifest)} visualization PNGs under {out_root / 'viz'}")


if __name__ == "__main__":
    main()
