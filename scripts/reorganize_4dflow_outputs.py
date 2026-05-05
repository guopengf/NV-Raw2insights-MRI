import argparse
import shutil
from pathlib import Path

from path_safety import assert_not_in_known_raw_data_path, assert_outputs_not_in_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flat-output",
        type=Path,
        required=True,
        help="The original output root from inference.py, which contains val_img4ranking/",
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        required=True,
        help="Final organized output root",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying",
    )
    args = parser.parse_args()

    if args.move:
        assert_not_in_known_raw_data_path(args.flat_output, what="move source")
    final_output = assert_outputs_not_in_data([args.final_output], [args.flat_output])[0]

    src_dir = args.flat_output / "val_img4ranking"
    mats = sorted(src_dir.glob("*.mat"))

    if not mats:
        raise RuntimeError(f"No .mat found in {src_dir}")

    for mat_path in mats:
        stem = mat_path.stem
        parts = stem.split("__", 3)
        if len(parts) != 4:
            print(f"[skip] unexpected filename: {mat_path.name}")
            continue

        center, scanner, case_id, acc_tag = parts
        dst_dir = final_output / center / scanner / case_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_name = f"{case_id}_{acc_tag}.mat"
        dst_path = dst_dir / dst_name

        if args.move:
            shutil.move(str(mat_path), str(dst_path))
        else:
            shutil.copy2(str(mat_path), str(dst_path))

        print(f"{mat_path.name} -> {dst_path}")

    print("done.")

if __name__ == "__main__":
    main()
