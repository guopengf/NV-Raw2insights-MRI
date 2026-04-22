"""
Visualize CMRx4Dflow reconstructions saved as img4ranking .mat files.

This script is specialized for 4D arrays with shape (H, W, slices, time) and
can generate:
  - one full slice x time PNG per encoding
  - one center-slice overview PNG across encodings
  - one animated GIF per slice for each encoding
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import scipy.io
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_mat(mat_file):
    """Read a MATLAB file, supporting both v7.3 and classic MAT formats."""
    try:
        with h5py.File(mat_file, "r", swmr=True) as f:
            return {key: f[key][()] for key in f}
    except Exception:
        data = scipy.io.loadmat(mat_file)
        return {key: value for key, value in data.items() if not key.startswith("__")}


def normalize_01(arr):
    """Normalize an array to [0, 1]."""
    arr = np.asarray(arr, dtype=np.float64)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - vmin) / (vmax - vmin)


def extract_img4ranking(mat_file, key):
    """Load and validate img4ranking-style data as a 4D array."""
    data = read_mat(mat_file)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {mat_file}. Keys: {sorted(data)}")

    arr = np.asarray(data[key]).squeeze()
    while arr.ndim > 4:
        arr = arr.squeeze()
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array for {mat_file}, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def enc_label(mat_path):
    """Extract a compact encoding label from a case filename."""
    for part in mat_path.stem.split("_"):
        if part.startswith("enc"):
            return part
    return mat_path.stem


def resolve_inputs(input_path):
    """Resolve a repo output dir, val_img4ranking dir, or one .mat file."""
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() != ".mat":
            raise ValueError(f"Expected a .mat file, got {input_path}")
        mat_files = [input_path]
        if input_path.parent.name == "val_img4ranking":
            output_root = input_path.parent.parent
        else:
            output_root = input_path.parent
        return mat_files, output_root

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    if (input_path / "val_img4ranking").is_dir():
        mat_dir = input_path / "val_img4ranking"
        output_root = input_path
    else:
        mat_dir = input_path
        output_root = input_path.parent if input_path.name == "val_img4ranking" else input_path

    mat_files = sorted(mat_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {mat_dir}")
    return mat_files, output_root


def save_grid_png(img4d, out_path, title, cmap, panel_size, dpi):
    """Save a compact slices x time montage for one encoding."""
    n_slices = img4d.shape[2]
    n_times = img4d.shape[3]

    fig, axes = plt.subplots(
        n_slices,
        n_times,
        figsize=(panel_size * n_times, panel_size * n_slices),
        squeeze=False,
    )

    for s in range(n_slices):
        for t in range(n_times):
            ax = axes[s, t]
            ax.imshow(normalize_01(img4d[:, :, s, t]), cmap=cmap, vmin=0.0, vmax=1.0)
            ax.axis("off")
            if s == 0:
                ax.set_title(f"t={t}", fontsize=5, pad=2)
            if t == 0:
                ax.set_ylabel(f"s={s}", fontsize=5, rotation=0, labelpad=16, va="center")

    fig.suptitle(title, fontsize=8)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG: {out_path}")


def save_center_slice_overview(cases, out_path, time_indices, tile_size):
    """Save a summary sheet with one center slice across representative times."""
    if not cases:
        return

    pad = 18
    header_h = 40
    label_w = 72
    row_gap = 18

    width = label_w + pad + len(time_indices) * (tile_size + pad) + pad
    height = header_h + pad + len(cases) * (tile_size + row_gap) + pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    y = pad
    for i, t in enumerate(time_indices):
        x = label_w + pad + i * (tile_size + pad)
        draw.text((x + tile_size // 2 - 14, y), f"t={t}", fill="black")

    y += header_h
    for label, img4d in cases:
        center_slice = img4d.shape[2] // 2
        draw.text((pad, y + tile_size // 2), label, fill="black")

        slice_block = normalize_01(img4d[:, :, center_slice, time_indices])
        x = label_w + pad
        for i, t in enumerate(time_indices):
            frame = (slice_block[:, :, i] * 255.0).clip(0, 255).astype(np.uint8)
            tile = Image.fromarray(frame, mode="L").resize((tile_size, tile_size)).convert("RGB")
            canvas.paste(tile, (x, y))
            x += tile_size + pad

        y += tile_size + row_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved overview: {out_path}")


def save_slice_gifs(label, img4d, gif_dir, duration_ms):
    """Save one animated GIF per slice, sweeping over time."""
    enc_dir = gif_dir / label
    enc_dir.mkdir(parents=True, exist_ok=True)

    for s in range(img4d.shape[2]):
        slice_data = normalize_01(img4d[:, :, s, :])
        frames = []
        for t in range(img4d.shape[3]):
            frame = (slice_data[:, :, t] * 255.0).clip(0, 255).astype(np.uint8)
            frames.append(Image.fromarray(frame, mode="L").convert("P"))

        out_path = enc_dir / f"slice_{s:02d}.gif"
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        print(f"Saved GIF: {out_path}")


def parse_time_indices(spec, n_times):
    """Parse comma-separated time indices and clamp them to the volume size."""
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if 0 <= idx < n_times:
            values.append(idx)
    if values:
        return values

    if n_times == 1:
        return [0]

    return sorted({0, n_times // 4, n_times // 2, (3 * n_times) // 4, n_times - 1})


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CMRx4Dflow img4ranking outputs as PNG grids, overview sheets, and GIFs."
    )
    parser.add_argument(
        "input",
        help="Inference output dir, val_img4ranking dir, or one .mat file.",
    )
    parser.add_argument(
        "-k",
        "--key",
        default="img4ranking",
        help="MAT variable name to visualize (default: img4ranking).",
    )
    parser.add_argument(
        "--fig-dir",
        default=None,
        help="Directory for per-encoding PNGs and overview PNG (default: <output>/figs).",
    )
    parser.add_argument(
        "--gif-dir",
        default=None,
        help="Directory for per-slice GIFs (default: <output>/gifs).",
    )
    parser.add_argument(
        "--summary-times",
        default="0,6,12,18,24",
        help="Comma-separated time indices for the center-slice overview.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=180,
        help="Tile size in pixels for the overview PNG (default: 180).",
    )
    parser.add_argument(
        "--panel-size",
        type=float,
        default=0.9,
        help="Panel size in inches for each slice/time cell in the full PNG grid.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="DPI for per-encoding PNGs (default: 140).",
    )
    parser.add_argument(
        "--cmap",
        default="gray",
        help="Matplotlib colormap for PNGs (default: gray).",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=120,
        help="Frame duration in milliseconds for GIFs (default: 120).",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip per-encoding PNG grids.",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Skip the center-slice overview PNG.",
    )
    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip per-slice GIF generation.",
    )
    args = parser.parse_args()

    mat_files, output_root = resolve_inputs(args.input)
    fig_dir = Path(args.fig_dir) if args.fig_dir else output_root / "figs"
    gif_dir = Path(args.gif_dir) if args.gif_dir else output_root / "gifs"

    cases = []
    for mat_file in mat_files:
        img4d = extract_img4ranking(mat_file, args.key)
        label = enc_label(mat_file)
        cases.append((label, img4d, mat_file))

    if not args.skip_grid:
        for label, img4d, mat_file in cases:
            save_grid_png(
                img4d=img4d,
                out_path=fig_dir / f"{mat_file.stem}.png",
                title=mat_file.name,
                cmap=args.cmap,
                panel_size=args.panel_size,
                dpi=args.dpi,
            )

    if not args.skip_overview:
        time_indices = parse_time_indices(args.summary_times, cases[0][1].shape[3])
        save_center_slice_overview(
            cases=[(label, img4d) for label, img4d, _ in cases],
            out_path=fig_dir / "overview_center_slice.png",
            time_indices=time_indices,
            tile_size=args.tile_size,
        )

    if not args.skip_gif:
        for label, img4d, _ in cases:
            save_slice_gifs(label=label, img4d=img4d, gif_dir=gif_dir, duration_ms=args.gif_duration_ms)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
