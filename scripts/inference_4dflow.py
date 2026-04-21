"""Run NV-Raw2Insights-MRI-Base on CMRx 4D-Flow data.

Pipeline (per case):
    kdata: (enc, t, coil, kz, ky, kx) complex
      1D IFFT on kx (fully-sampled)  ->  (enc, t, coil, kz, ky, x)
      reshape interleaving enc into t ->  (t*enc, x, coil, kz, ky)
      treated as model-5D kspace (T, Z, C, Y, X) where T=t*enc, Z=x, Y=kz, X=ky.
    usmask: (1, t, 1, kz, ky, 1) -> broadcast across enc, reshape to (t*enc, 1, 1, kz, ky, 1).

Each (t*enc, x) position is a 2D (kz, ky) kspace plane given to the model.
The model estimates its own coil-sensitivity map; final image is coil-RSS.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch
import torch.distributed as dist
from einops import rearrange
from monai.apps.reconstruction.complex_utils import complex_abs
from monai.data.fft_utils import fftn_centered, ifftn_centered
from torch.amp import autocast

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.latent_recon import create_mri_recon_model  # noqa: E402
from utils import load_config, load_net, resolve_checkpoint_path  # noqa: E402


def _first_key(f: h5py.File) -> str:
    for k in f.keys():
        if not k.startswith("#"):
            return k
    raise KeyError("no data key found")


def load_mat_complex(path: Path, key: str | None = None) -> np.ndarray:
    """Load a complex array from an HDF5 .mat file (v7.3). If `key` is None, pick first."""
    with h5py.File(path, "r", swmr=True) as f:
        k = key if key and key in f else _first_key(f)
        arr = f[k][()]
    if arr.dtype.names and "real" in arr.dtype.names and "imag" in arr.dtype.names:
        return arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)
    return np.asarray(arr, dtype=np.complex64)


def load_mat_real(path: Path, key: str | None = None) -> np.ndarray:
    with h5py.File(path, "r", swmr=True) as f:
        k = key if key and key in f else _first_key(f)
        return np.asarray(f[k][()])


def find_case_files(case_dir: Path) -> dict:
    """Locate mat files inside a case directory."""
    files = {p.name: p for p in case_dir.iterdir() if p.suffix == ".mat"}
    kdata_us = None
    mask_us = None
    mask_type = None
    for name, path in files.items():
        if name.startswith("kdata_kt") or name.startswith("kdata_Uniform") or name.startswith("kdata_us"):
            kdata_us = path
            mask_type = name.replace("kdata_", "").replace(".mat", "")
        if name.startswith("usmask"):
            mask_us = path
    return {
        "kdata_full": files.get("kdata_full.mat"),
        "kdata_us": kdata_us,
        "mask_us": mask_us,
        "coilmap": files.get("coilmap.mat"),
        "segmask": files.get("segmask.mat"),
        "mask_type": mask_type,
    }


def ifft1d_kx(kspace_6d: np.ndarray) -> np.ndarray:
    """Centered 1D IFFT along the last (kx) axis. Keeps input shape."""
    shifted = np.fft.ifftshift(kspace_6d, axes=-1)
    img_x = np.fft.ifft(shifted, axis=-1, norm="ortho")
    return np.fft.fftshift(img_x, axes=-1)


def reshape_one_enc(kspace_5d: np.ndarray) -> np.ndarray:
    """One enc direction: (t, coil, kz, ky, x) -> (t, x, coil, kz, ky).

    Maps cleanly to the model's 5D k-space convention (T=t, Z=x, C, Y=kz, X=ky)
    so the native `windowed_input` semantics hold — windows span adjacent cardiac phases.
    """
    return rearrange(kspace_5d, "t c kz ky x -> t x c kz ky")


def reshape_mask_one_enc(mask_6d: np.ndarray, enc_idx: int) -> np.ndarray:
    """Slice an enc from the usmask and flatten to (t, 1, kz, ky) float32.

    usmask is (enc_m, t, c_m, kz, ky, kx_m) with enc_m|c_m|kx_m possibly 1.
    We pick the enc if it's present (or use the broadcast slot if enc_m == 1),
    drop the coil/kx singletons, and return (t, 1, kz, ky).
    """
    em, tm, cm, kz, ky, kxm = mask_6d.shape
    assert cm == 1 and kxm == 1, f"unexpected mask shape {mask_6d.shape}"
    idx = 0 if em == 1 else enc_idx
    m = mask_6d[idx, :, 0, :, :, 0]  # (t, kz, ky)
    return m[:, None, :, :].astype(np.float32)  # (t, 1, kz, ky)


class AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def build_model(args, device) -> torch.nn.Module:
    model = create_mri_recon_model(args).to(device)
    try:
        args.is_multi_coil = model.use_csm or model.use_latent_csm
    except BaseException:
        args.is_multi_coil = True
    (model, *_) = load_net(model, args.model_ckpt, device, is_ddp=False, resume_rng_state=False)
    model.eval()
    return model


def init_dist():
    """Initialize (rank, world_size, local_rank, device). Works in single-proc and torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def prepare_model_inputs(
    kspace_us_5d: np.ndarray,
    usmask_4d: np.ndarray,
    device: torch.device,
) -> dict:
    """Returns image-domain input, mask, and normalization stats as torch tensors.

    Input shapes:
        kspace_us_5d: (T, Z, C, Y, X) complex - already masked k-space
        usmask_4d:    (T, 1, Y, X) float   - sampling mask (broadcast over Z, C)
    Output:
        inp:   (T*Z, C, Y, X, 2) image domain, z-scored
        mask:  (T*Z, C, Y, X, 2) bool (broadcast-expanded)
        mean/std: per-(T*Z) complex z-score stats for denormalization
    """
    # Add real/imag last dim.
    ks = np.stack((kspace_us_5d.real, kspace_us_5d.imag), axis=-1).astype(np.float32)
    ks_t = torch.from_numpy(ks)  # (T, Z, C, Y, X, 2)

    # IFFT along (Y, X) to image domain.
    img = ifftn_centered(ks_t, spatial_dims=2, is_complex=True)

    # Mask: (T, 1, Y, X) -> (T, 1, 1, Y, X, 1) then broadcast to image shape for the model.
    T, Z, C, Y, X = kspace_us_5d.shape
    mask = torch.from_numpy(usmask_4d).view(T, 1, 1, Y, X, 1).expand(T, Z, C, Y, X, 2).contiguous()

    # Rearrange to (T*Z, C, Y, X, 2), treating enc-interleaved t as "time" and x as "slice".
    img = rearrange(img, "t z c y x ri -> (t z) c y x ri")
    mask = rearrange(mask, "t z c y x ri -> (t z) c y x ri")

    # Z-score normalize over (C, Y, X) per flattened index, with eps to avoid /0.
    img_c = torch.view_as_complex(img.contiguous())  # (T*Z, C, Y, X) complex
    mean_c = img_c.mean(dim=(1, 2, 3), keepdim=True)  # (T*Z, 1, 1, 1) complex
    var = ((img_c - mean_c).abs() ** 2).mean(dim=(1, 2, 3), keepdim=True)  # (T*Z, 1, 1, 1) real
    std = torch.sqrt(var).clamp_min(1e-6)  # real, strictly positive
    img_norm_c = (img_c - mean_c) / std
    img_norm = torch.view_as_real(img_norm_c).contiguous()
    mean_ri = torch.view_as_real(mean_c).contiguous()
    std_r = std.unsqueeze(-1)  # add trailing real/imag dim for denorm shape consistency

    n_small = int((var < 1e-12).sum())
    if n_small:
        print(f"  warning: {n_small}/{var.numel()} slices had ~0 variance before eps clamp")

    return {
        "input": img_norm.to(device),
        "mask": mask.to(device).bool(),
        "mean": mean_ri.to(device),
        "std": std_r.to(device),
        "shape": (T, Z, C, Y, X),
    }


def windowed_indices(num_T: int, num_Z: int, num_frames: int, batch_size: int,
                     z_subset: list[int] | None = None,
                     rank: int = 0, world_size: int = 1):
    """Yield (window_idx LongTensor[B, num_frames], center_idx LongTensor[B]) batches.

    Windows are formed along the T axis, per Z-slice. Only z indices in `z_subset`
    (if provided) are reconstructed. Positions are striped across ranks.
    """
    half = num_frames // 2
    offsets = list(range(-half, num_frames - half))
    z_iter = z_subset if z_subset is not None else list(range(num_Z))

    flat = [(z, t) for t in range(num_T) for z in z_iter]
    flat = flat[rank::world_size]
    for start in range(0, len(flat), batch_size):
        batch = flat[start : start + batch_size]
        window_idx = []
        center_idx = []
        for z, t in batch:
            neigh_t = [(t + off) % num_T for off in offsets]
            idxs = [z + ti * num_Z for ti in neigh_t]  # (t, z) flat index = t * num_Z + z
            window_idx.append(idxs)
            center_idx.append(z + t * num_Z)
        yield torch.tensor(window_idx, dtype=torch.long), torch.tensor(center_idx, dtype=torch.long)


def run_inference_one_enc(
    kspace_5d_one_enc: np.ndarray,  # (t, coil, kz, ky, x) hybrid-domain (kx already IFFT'd)
    mask_4d_one_enc: np.ndarray,    # (t, 1, kz, ky) float32
    args, model, device,
    mask_type: str, acc_factor: int, acq_type: str,
    rank: int, world_size: int, tag: str,
):
    """Run the model on one encoding direction. Returns multi-coil image output
    as numpy array of shape (t, x, coil, kz, ky) complex.
    """
    kspace_5d = reshape_one_enc(kspace_5d_one_enc)  # (T=t, Z=x, C, Y=kz, X=ky)
    prepared = prepare_model_inputs(kspace_5d, mask_4d_one_enc, device)
    T, Z, C, Y, X = prepared["shape"]

    num_frames = args.num_frames
    batch_size = getattr(args, "batch_size", 1)

    out_img = torch.zeros(T * Z, C, Y, X, 2, dtype=torch.float32)

    input_all = prepared["input"]
    mask_all = prepared["mask"]
    mean_all = prepared["mean"]
    std_all = prepared["std"]

    tic = time.time()
    total_positions_all = T * Z
    total_positions_local = (total_positions_all + world_size - 1 - rank) // world_size
    done = 0
    last_print = time.time()
    first_batch = True
    nan_out = 0
    with torch.no_grad():
        for window_idx, center_idx in windowed_indices(T, Z, num_frames, batch_size,
                                                       rank=rank, world_size=world_size):
            window_idx = window_idx.to(device)
            center_idx = center_idx.to(device)
            inp = input_all[window_idx]
            mas = mask_all[window_idx]

            with autocast("cuda", torch.bfloat16, enabled=args.amp):
                out = model(inp, mas, mask_type, int(acc_factor), acq_type)
            out_center = out[:, num_frames // 2].float()
            if not torch.isfinite(out_center).all():
                nan_out += 1
            if first_batch and rank == 0:
                first_batch = False
                print(
                    f"  [{tag}] first batch: inp finite={torch.isfinite(inp).all().item()} "
                    f"inp range=[{inp.min().item():.3g},{inp.max().item():.3g}] "
                    f"out finite={torch.isfinite(out_center).all().item()} "
                    f"out range=[{out_center.min().item():.3g},{out_center.max().item():.3g}]",
                    flush=True,
                )

            mean_c = mean_all[center_idx]
            std_c = std_all[center_idx]
            out_center = out_center * std_c + mean_c

            out_img[center_idx.cpu()] = out_center.cpu()

            done += center_idx.shape[0]
            now = time.time()
            if now - last_print > 15 and rank == 0:
                rate = done / max(now - tic, 1e-6)
                eta = (total_positions_local - done) / max(rate, 1e-6)
                print(
                    f"  [{tag}] rank0 {done}/{total_positions_local} local pos "
                    f"(= {done*world_size}/{total_positions_all} global), "
                    f"{rate:.1f} pos/s, eta {eta:.0f}s",
                    flush=True,
                )
                last_print = now

    if world_size > 1:
        out_img = out_img.to(device)
        dist.all_reduce(out_img, op=dist.ReduceOp.SUM)
        out_img = out_img.cpu()

    if rank == 0:
        dur = time.time() - tic
        print(f"  [{tag}] done in {dur:.1f}s  nan_out_batches={nan_out}", flush=True)

    out_img = out_img.view(T, Z, C, Y, X, 2)
    return out_img  # torch tensor on CPU


def run_inference_one_case(case_dir: Path, args, model, device, x_stride: int = 1,
                           rank: int = 0, world_size: int = 1):
    info = find_case_files(case_dir)
    kdata_path = info["kdata_us"] if info["kdata_us"] is not None else info["kdata_full"]
    mask_path = info["mask_us"]

    kspace = load_mat_complex(kdata_path)
    if info["kdata_us"] is not None:
        mask_type = info["mask_type"] or "ktGaussian20"
    else:
        mask_type = "fully_sampled"

    kspace_full = load_mat_complex(info["kdata_full"]) if info["kdata_full"] is not None else None

    enc, t, coil, kz, ky, kx = kspace.shape
    if rank == 0:
        print(f"[{case_dir.name}] kspace shape (enc,t,c,kz,ky,kx) = {kspace.shape}")

    # 1D IFFT on kx (done once for all enc).
    kspace_hy = ifft1d_kx(kspace)  # (enc, t, c, kz, ky, x) hybrid

    # Mask (or full-sampled ones).
    if mask_path is not None:
        mask_raw = load_mat_real(mask_path).astype(np.float32)
    else:
        mask_raw = np.ones((1, t, 1, kz, ky, 1), dtype=np.float32)

    # Map the 20x-acc Gaussian mask to the closest trained acceleration if requested.
    try:
        acc_from_name = int("".join(ch for ch in mask_type if ch.isdigit()))
    except ValueError:
        acc_from_name = 16
    acc_factor = acc_from_name
    if getattr(args, "snap_acc_to_trained", False):
        known = [int(a) for a in getattr(args, "accelerations", [8, 16, 24])]
        acc_factor = min(known, key=lambda a: (abs(a - acc_from_name), -a))
    if rank == 0:
        print(f"[{case_dir.name}] mask_type={mask_type} acc_factor={acc_factor}")

    acq_type = "Flow2d"

    # Process each encoding direction sequentially.
    tic_case = time.time()
    per_enc_outputs = []
    for e in range(enc):
        ks_one = kspace_hy[e]                  # (t, coil, kz, ky, x)
        ms_one = reshape_mask_one_enc(mask_raw, e)  # (t, 1, kz, ky)
        out = run_inference_one_enc(
            ks_one, ms_one, args, model, device,
            mask_type=mask_type, acc_factor=acc_factor, acq_type=acq_type,
            rank=rank, world_size=world_size, tag=f"{case_dir.name}/enc{e}",
        )
        per_enc_outputs.append(out)

    if rank == 0:
        print(f"[{case_dir.name}] all enc done in {time.time() - tic_case:.1f}s total", flush=True)

    # Stack: (enc, t, x, coil, kz, ky, 2)
    full = torch.stack(per_enc_outputs, dim=0)  # (enc, T, Z, C, Y, X, 2)
    E, T_, Z_, C, Y, X, _ = full.shape
    # RSS over coils per (enc, t, x, kz, ky).
    mag_coil = complex_abs(full)  # (enc, T, Z, C, Y, X)
    recon_rss = torch.sqrt(torch.sum(mag_coil ** 2, dim=3)).numpy()  # (enc, T=t, Z=x, Y=kz, X=ky)

    # Transpose to (enc, t, kz, ky, x) for standard 4D-flow orientation.
    recon = np.transpose(recon_rss, (0, 1, 3, 4, 2))  # (enc, t, kz, ky, x)

    # Zero-filled baseline: per-enc IFFT of masked k-space, RSS over coils.
    zf_list = []
    for e in range(enc):
        ks_one = kspace_hy[e]
        ks_ri = np.stack((ks_one.real, ks_one.imag), axis=-1).astype(np.float32)
        img_zf = complex_abs(
            ifftn_centered(torch.from_numpy(rearrange(ks_ri, "t c kz ky x ri -> t x c kz ky ri")),
                           spatial_dims=2, is_complex=True)
        )  # (t, x, c, kz, ky)
        zf_list.append(torch.sqrt(torch.sum(img_zf ** 2, dim=2)).numpy())  # (t, x, kz, ky)
    zf_all = np.stack(zf_list, axis=0)  # (enc, t, x, kz, ky)
    zf = np.transpose(zf_all, (0, 1, 3, 4, 2))  # (enc, t, kz, ky, x)

    gt = None
    if kspace_full is not None:
        kf = kspace_full
        img_full = np.fft.fftshift(
            np.fft.ifftn(
                np.fft.ifftshift(kf, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="ortho"
            ),
            axes=(-3, -2, -1),
        )
        gt = np.sqrt(np.sum(np.abs(img_full) ** 2, axis=2))  # (enc, t, kz, ky, x)

    return {
        "recon": recon,
        "zf": zf,
        "gt": gt,
        "case": case_dir,
        "mask_type": mask_type,
        "acc_factor": acc_factor,
    }


def _to_uint8(img: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    x = img.astype(np.float32)
    if vmin is None:
        vmin = float(x.min())
    if vmax is None:
        vmax = float(x.max())
    rng = max(vmax - vmin, 1e-8)
    x = np.clip((x - vmin) / rng, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def save_visualizations(result: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    recon = result["recon"]  # (enc, t, kz, ky, x)
    zf = result["zf"]
    gt = result["gt"]
    enc, T, kz, ky, nx = recon.shape

    # The model's native recon plane is (kz, ky) per (t, x); pick middle x for each cardiac phase.
    mid_x = nx // 2
    mid_kz = kz // 2
    mid_t = T // 2

    case_name = "_".join(str(result["case"]).split("/")[-4:])

    # --- Panel 1: native (kz, ky) plane at middle x for each enc, middle t ---
    def _plane_kz_ky(arr, e, ti):
        # (kz, ky) image per enc, time, middle x
        return arr[e, ti, :, :, mid_x]

    ncols = 3 if gt is not None else 2
    fig, axes = plt.subplots(enc, ncols, figsize=(3 * ncols, 3 * enc))
    if enc == 1:
        axes = axes[None, :]
    col_titles = ["Model recon", "Zero-filled", "GT"][:ncols]
    for e in range(enc):
        imgs = [_plane_kz_ky(recon, e, mid_t), _plane_kz_ky(zf, e, mid_t)]
        if gt is not None:
            imgs.append(_plane_kz_ky(gt, e, mid_t))
        vmax = max(float(np.max(im)) for im in imgs)
        for c in range(ncols):
            axes[e, c].imshow(imgs[c], cmap="gray", vmin=0, vmax=vmax)
            axes[e, c].axis("off")
            if e == 0:
                axes[e, c].set_title(col_titles[c])
    fig.suptitle(
        f"{case_name} | (kz,ky) plane | x={mid_x} t={mid_t}/{T} "
        f"| {result['mask_type']} (acc={result['acc_factor']})"
    )
    fig.tight_layout()
    panel_path = out_dir / f"{case_name}_panel_kzky_x{mid_x}_t{mid_t}.png"
    fig.savefig(panel_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {panel_path}")

    # --- Panel 2: (x, ky) axial-style view at middle kz — shows readout direction ---
    fig, axes = plt.subplots(enc, ncols, figsize=(3 * ncols, 3 * enc))
    if enc == 1:
        axes = axes[None, :]
    for e in range(enc):
        imgs = [recon[e, mid_t, mid_kz, :, :], zf[e, mid_t, mid_kz, :, :]]
        if gt is not None:
            imgs.append(gt[e, mid_t, mid_kz, :, :])
        vmax = max(float(np.max(im)) for im in imgs)
        for c in range(ncols):
            axes[e, c].imshow(imgs[c].T, cmap="gray", vmin=0, vmax=vmax, aspect="auto")
            axes[e, c].axis("off")
            if e == 0:
                axes[e, c].set_title(col_titles[c])
    fig.suptitle(
        f"{case_name} | (ky,x) plane | kz={mid_kz} t={mid_t}/{T} "
        f"| {result['mask_type']} (acc={result['acc_factor']})"
    )
    fig.tight_layout()
    axial_path = out_dir / f"{case_name}_panel_kyx_kz{mid_kz}_t{mid_t}.png"
    fig.savefig(axial_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {axial_path}")

    # --- Animated GIF over time: enc=0, native plane at middle x ---
    vmax = float(recon[0, :, :, :, mid_x].max())
    frames = []
    for ti in range(T):
        panels = [_plane_kz_ky(recon, 0, ti), _plane_kz_ky(zf, 0, ti)]
        if gt is not None:
            panels.append(_plane_kz_ky(gt, 0, ti))
        frame = np.concatenate([_to_uint8(p, 0, vmax) for p in panels], axis=1)
        frames.append(frame)
    gif_path = out_dir / f"{case_name}_enc0_x{mid_x}_time.gif"
    imageio.mimsave(gif_path, frames, duration=0.12)
    print(f"  saved {gif_path}")

    # --- Grid of all enc directions at mid time ---
    fig, axes = plt.subplots(1, enc, figsize=(3 * enc, 3))
    if enc == 1:
        axes = [axes]
    for e in range(enc):
        axes[e].imshow(_plane_kz_ky(recon, e, mid_t), cmap="gray")
        axes[e].axis("off")
        axes[e].set_title(f"enc{e}")
    fig.suptitle(f"{case_name} recon | x={mid_x} t={mid_t}")
    fig.tight_layout()
    enc_path = out_dir / f"{case_name}_recon_enc_grid_x{mid_x}_t{mid_t}.png"
    fig.savefig(enc_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {enc_path}")

    # Save raw numpy for downstream evaluation / phase decoding.
    np.savez_compressed(out_dir / f"{case_name}_recon.npz", recon=recon, zf=zf, gt=gt)


def iter_cases(root: Path):
    """Yield leaf case directories that contain kdata_full.mat."""
    for p in sorted(root.rglob("kdata_full.mat")):
        yield p.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=Path, required=True)
    p.add_argument("-m", "--model_ckpt", type=Path, default=None)
    p.add_argument("-i", "--input_path", type=Path, required=True,
                   help="Root of CMRx 4D Flow data (scans for case dirs).")
    p.add_argument("-o", "--output_path", type=Path, required=True)
    p.add_argument("--max_cases", type=int, default=0, help="0 = all.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--x_stride", type=int, default=1,
                   help="Subsample x-slices by this stride (1 = all slices).")
    p.add_argument("--no_amp", action="store_true", help="Disable bf16 autocast.")
    p.add_argument("--snap_acc_to_trained", action="store_true",
                   help="Snap the mask's acceleration factor to the closest trained value "
                        "(one of config.accelerations, e.g. 20 → 24).")
    return p.parse_args()


def main():
    cli = parse_args()
    args = load_config(cli.config)
    args.ddp = False
    args.batch_size = cli.batch_size
    args.output_path = cli.output_path
    args.data_path_test = cli.input_path
    args.debug = False
    if cli.no_amp:
        args.amp = False
    args.snap_acc_to_trained = cli.snap_acc_to_trained
    model_variant = args.model_variant
    args.model_ckpt = resolve_checkpoint_path(model_variant, cli.model_ckpt)
    args.flow = True

    rank, world_size, local_rank, device = init_dist()
    if rank == 0:
        print(f"device={device}; ckpt={args.model_ckpt}; world_size={world_size} visible_gpus={torch.cuda.device_count()}")
    model = build_model(args, device)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters()) * 1e-6
        print(f"#model_params: {n_params:.1f}M")

    if rank == 0:
        cli.output_path.mkdir(parents=True, exist_ok=True)

    cases = list(iter_cases(cli.input_path))
    if cli.max_cases > 0:
        cases = cases[: cli.max_cases]
    if rank == 0:
        print(f"{len(cases)} case(s) to process")
    for case_dir in cases:
        if rank == 0:
            print(f"--- processing {case_dir} ---")
        result = run_inference_one_case(
            case_dir, args, model, device,
            x_stride=cli.x_stride, rank=rank, world_size=world_size,
        )
        if rank == 0:
            save_visualizations(result, cli.output_path)
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
