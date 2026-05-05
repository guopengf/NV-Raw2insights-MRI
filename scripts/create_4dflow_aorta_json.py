#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from path_safety import assert_outputs_not_in_data

ACC_LIST = [10, 20, 30, 40, 50]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    out = assert_outputs_not_in_data([args.out], [root])[0]
    out.mkdir(parents=True, exist_ok=True)

    n_written = 0

    for center_dir in sorted(root.glob("Center*")):
        if not center_dir.is_dir():
            continue
        for vendor_dir in sorted(center_dir.iterdir()):
            if not vendor_dir.is_dir():
                continue
            for patient_dir in sorted(vendor_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue

                full_kspace = patient_dir / "kdata_full.mat"
                coilmap = patient_dir / "coilmap.mat"
                if not full_kspace.exists():
                    continue

                for acc in ACC_LIST:
                    us_kspace = patient_dir / f"kdata_ktGaussian{acc}.mat"
                    us_mask = patient_dir / f"usmask_ktGaussian{acc}.mat"

                    if not us_kspace.exists() or not us_mask.exists():
                        continue

                    for enc_idx in range(4):
                        item = {
                            "kspace": str(us_kspace),
                            "target_kspace": str(full_kspace),
                            "mask": [str(us_mask)],
                            "mask_type": f"ktGaussian{acc}",
                            "acquisition": "Flow2d",
                            "encoding_idx": enc_idx,
                            "is_4dflow": True,
                        }
                        if coilmap.exists():
                            item["coilmap"] = str(coilmap)

                        out_name = (
                            f"{center_dir.name}__{vendor_dir.name}__{patient_dir.name}"
                            f"__ktGaussian{acc}__enc{enc_idx}.json"
                        )
                        with open(out / out_name, "w") as f:
                            json.dump(item, f, indent=2)

                        n_written += 1

    print(f"Written {n_written} json files to: {out}")


if __name__ == "__main__":
    main()
