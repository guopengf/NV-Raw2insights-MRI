"""Derive HEL training configs from the ones build_virtual_dataset wrote: HDF5 reader, checkpoint retention,
multi-GPU baseline settings. Without --suite-path it writes benchmark-only configs that use the kt-Gaussian fallback.

    python experiments/paper/scripts/hel_configs.py --run-root RUN_ROOT --h5-root H5_ROOT [--suite-path .../mask_suite.py]
"""

import argparse
import json
from pathlib import Path

OURS = {"checkpoint_rolling_every": 5, "checkpoint_rolling_keep": 2, "checkpoint_permanent_every": 100}
BASELINE = {"checkpoint_permanent_every": 10}
NEURAL_BASELINES = {"flowvn", "flowmodl", "cut_4dflow"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--h5-root", type=Path, required=True)
    p.add_argument("--suite-path", type=Path)
    p.add_argument("--accumulate", nargs="*", default=[], help="Our-method runs that accumulate 2 windows instead of stacking them")
    p.add_argument("--bucket-steps", type=int, default=0, help="paper.loader.bucket_steps for multi-GPU runs (0 = plain shuffle)")
    args = p.parse_args()
    if not (args.h5_root / "cases").is_dir():
        p.error(f"No converted stores under {args.h5_root}")
    if args.suite_path and not args.suite_path.is_file():
        p.error(f"Mask suite not found: {args.suite_path}")
    kind = "hel" if args.suite_path else "bench"
    out = args.run_root / f"configs_{kind}"
    out.mkdir(parents=True, exist_ok=True)
    for source in sorted((args.run_root / "configs").glob("*.json")):
        cfg = json.loads(source.read_text())
        paper = cfg["paper"]
        method = paper["method"]
        paper.update(train_backend="windowed_hdf5", h5_root=str(args.h5_root),
                     output_dir=str(args.run_root / ("runs" if args.suite_path else "bench") / paper["name"]))
        paper["protected_roots"] = sorted({*paper["protected_roots"], str(args.h5_root)})
        if method == "ours":
            paper.update(OURS)
            # batch_size counts encoding images, as in the reference config: 8 = 2 windows per GPU per update.
            # The two windows are stacked into one pass (scripts/train.py's design). Runs named in
            # --accumulate would not fit two stacked windows in GPU memory, so they accumulate two passes.
            cfg["batch_size"] = 8
            stacked = paper["name"] not in args.accumulate
            paper["loader"]["stack"] = 2 if stacked else 1
            paper["gradient_accumulation"] = 1 if stacked else 2
        elif method in NEURAL_BASELINES:
            paper.update(BASELINE)
        if method == "ours" or method in NEURAL_BASELINES:
            paper["loader"]["bucket_steps"] = args.bucket_steps
        if method == "cut_4dflow":
            # One window per GPU per update; 8 GPUs give an effective batch close to the authors' 5.
            paper["gradient_accumulation"] = 1
        if "online_undersampling" in paper:
            if args.suite_path:
                paper["online_undersampling"]["suite_path"] = str(args.suite_path.resolve())
            else:
                paper["online_undersampling"]["enabled"] = False
                paper["benchmark_only"] = "kt-Gaussian fallback masks; not the paper protocol"
        (out / source.name).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
        print(f"{kind} {source.name} method={method} output={paper['output_dir']}")


if __name__ == "__main__":
    main()
