import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from path_safety import assert_outputs_not_in_data


ACCELS = [10, 20, 30, 40, 50]


def build_jsons(case_root: Path, json_dir: Path, acquisition: str, encoding_idx: int):
    """
    case_root:
      /SSDHome/share/4dFlow/ChallengeData/TaskR1&R2/ValidationSet/Aorta/Center007/GE_30T_Architect

    creates one json per (case, accel)
    """
    json_dir = assert_outputs_not_in_data([json_dir], [case_root])[0]
    json_dir.mkdir(parents=True, exist_ok=True)

    center = case_root.parent.name          # Center007
    scanner = case_root.name                # GE_30T_Architect

    manifest = []

    for case_dir in sorted(case_root.iterdir()):
        if not case_dir.is_dir():
            continue
        if not case_dir.name.startswith("P"):
            continue

        case_id = case_dir.name
        kspace_full = case_dir / "kdata_full.mat"

        if not kspace_full.exists():
            print(f"[WARN] Missing {kspace_full}, skip.")
            continue

        for acc in ACCELS:
            us_kspace = case_dir / f"kdata_ktGaussian{acc}.mat"
            mask_path = case_dir / f"usmask_ktGaussian{acc}.mat"
            if not us_kspace.exists() or not mask_path.exists():
                print(f"[WARN] Missing {us_kspace} or {mask_path}, skip.")
                continue
            coilmap = case_dir / "coilmap.mat"

            stem = f"{center}__{scanner}__{case_id}__ktGaussian{acc}"
            json_path = json_dir / f"{stem}.json"

            payload = {
                "kspace": str(us_kspace),
                "target_kspace": str(kspace_full),
                "mask": [str(mask_path)],
                "mask_type": f"ktGaussian{acc}",
                "acquisition": acquisition,
                "encoding_idx": encoding_idx,
                "is_4dflow": True,
            }
            if coilmap.exists():
                payload["coilmap"] = str(coilmap)

            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2)

            manifest.append(
                {
                    "stem": stem,
                    "center": center,
                    "scanner": scanner,
                    "case_id": case_id,
                    "acc": acc,
                    "json_path": str(json_path),
                }
            )

    return manifest


def run_inference(repo_root: Path, config_path: str, input_json_dir: Path, tmp_out_dir: Path, ckpt: str | None):
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "inference.py"),
        "-c",
        config_path,
        "-i",
        str(input_json_dir),
        "-o",
        str(tmp_out_dir),
    ]
    if ckpt:
        cmd.extend(["-m", ckpt])

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def reorganize_outputs(tmp_out_dir: Path, final_out_dir: Path, manifest):
    """
    Expect original inference outputs under:
      tmp_out_dir / val_img4ranking / <stem>.mat
    Copy to:
      final_out_dir / Center007 / GE_30T_Architect / P076 / kdata_ktGaussian10_recon.mat
    """
    src_dir = tmp_out_dir / "val_img4ranking"
    if not src_dir.exists():
        raise FileNotFoundError(f"Expected inference outputs in: {src_dir}")

    for item in manifest:
        src_file = src_dir / f"{item['stem']}.mat"
        if not src_file.exists():
            print(f"[WARN] Missing inference output: {src_file}")
            continue

        dst_dir = final_out_dir / item["center"] / item["scanner"] / item["case_id"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / f"kdata_ktGaussian{item['acc']}_recon.mat"

        shutil.copy2(src_file, dst_file)
        print(f"[OK] {src_file} -> {dst_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-root",
        type=str,
        required=True,
        help="Root folder that contains case folders P076, P077, ...",
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        required=True,
        help="Directory to write generated json descriptors",
    )
    parser.add_argument(
        "--tmp-out",
        type=str,
        required=True,
        help="Temporary output directory used by original inference.py",
    )
    parser.add_argument(
        "--final-out",
        type=str,
        required=True,
        help="Final reorganized output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nv_raw2insights_mri_base.json",
        help="Path to config json",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Local checkpoint path. Strongly recommended if your server cannot reach Hugging Face.",
    )
    parser.add_argument(
        "--acquisition",
        type=str,
        default="Flow2d",
        help="Acquisition token passed into the existing model",
    )
    parser.add_argument(
        "--encoding-idx",
        type=int,
        default=0,
        help="Which velocity encoding to use. Current baseline uses only one encoding.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    case_root = Path(args.case_root)
    json_dir, tmp_out_dir, final_out_dir = assert_outputs_not_in_data(
        [args.json_dir, args.tmp_out, args.final_out],
        [case_root],
    )

    json_dir.mkdir(parents=True, exist_ok=True)
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    final_out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_jsons(
        case_root=case_root,
        json_dir=json_dir,
        acquisition=args.acquisition,
        encoding_idx=args.encoding_idx,
    )

    print(f"[INFO] Generated {len(manifest)} json files")

    run_inference(
        repo_root=repo_root,
        config_path=args.config,
        input_json_dir=json_dir,
        tmp_out_dir=tmp_out_dir,
        ckpt=args.ckpt,
    )

    reorganize_outputs(
        tmp_out_dir=tmp_out_dir,
        final_out_dir=final_out_dir,
        manifest=manifest,
    )

    print("[DONE]")


if __name__ == "__main__":
    main()
