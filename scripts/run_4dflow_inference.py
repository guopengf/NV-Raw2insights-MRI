import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from path_safety import assert_outputs_not_in_data


ACCELS = [10, 20, 30, 40, 50]


def parse_accelerations(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def build_jsons(
    case_root: Path,
    json_dir: Path,
    acquisition: str,
    encoding_idx: int,
    accelerations: list[int],
    targetless: bool = False,
    max_cases: int | None = None,
):
    """
    case_root:
      /SSDHome/share/4dFlow/ChallengeData/TaskR1&R2/ValidationSet/Aorta/Center007/GE_30T_Architect

    Creates one json per (case, accel) for a single velocity encoding. Challenge validation/test
    folders usually do not contain kdata_full.mat; pass targetless=True to use the undersampled
    k-space as a shape placeholder for the reader while inference uses kspace_masked_ifft.
    """
    json_dir = assert_outputs_not_in_data([json_dir], [case_root])[0]
    json_dir.mkdir(parents=True, exist_ok=True)

    center = case_root.parent.name          # Center007
    scanner = case_root.name                # GE_30T_Architect

    manifest = []
    n_cases = 0

    for case_dir in sorted(case_root.iterdir()):
        if not case_dir.is_dir():
            continue
        if not case_dir.name.startswith("P"):
            continue

        case_id = case_dir.name
        kspace_full = case_dir / "kdata_full.mat"
        if not kspace_full.exists() and not targetless:
            print(f"[WARN] Missing {kspace_full}, skip. Use --targetless for challenge validation/test folders.")
            continue

        case_written = False
        for acc in accelerations:
            us_kspace = case_dir / f"kdata_ktGaussian{acc}.mat"
            mask_path = case_dir / f"usmask_ktGaussian{acc}.mat"
            if not us_kspace.exists() or not mask_path.exists():
                print(f"[WARN] Missing {us_kspace} or {mask_path}, skip.")
                continue
            coilmap = case_dir / "coilmap.mat"
            target_kspace = kspace_full if kspace_full.exists() else us_kspace

            stem = f"{center}__{scanner}__{case_id}__ktGaussian{acc}__enc{encoding_idx}"
            json_path = json_dir / f"{stem}.json"

            payload = {
                "kspace": str(us_kspace),
                "target_kspace": str(target_kspace),
                "mask": [str(mask_path)],
                "mask_type": f"ktGaussian{acc}",
                "acquisition": acquisition,
                "encoding_idx": encoding_idx,
                "is_4dflow": True,
                "targetless": not kspace_full.exists(),
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
                    "encoding_idx": encoding_idx,
                    "json_path": str(json_path),
                }
            )
            case_written = True

        if case_written:
            n_cases += 1
            if max_cases is not None and n_cases >= max_cases:
                break

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
      final_out_dir / Center007 / GE_30T_Architect / P076 / kdata_ktGaussian10_enc0_recon.mat
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
        dst_file = dst_dir / f"kdata_ktGaussian{item['acc']}_enc{item['encoding_idx']}_recon.mat"

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
    parser.add_argument(
        "--accelerations",
        type=parse_accelerations,
        default=ACCELS,
        help="Comma-separated acceleration list, for example 20 or 10,20,30.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit the number of patient folders that produce at least one JSON.",
    )
    parser.add_argument(
        "--targetless",
        action="store_true",
        help="Allow challenge validation/test folders without kdata_full.mat.",
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
        accelerations=args.accelerations,
        targetless=args.targetless,
        max_cases=args.max_cases,
    )

    print(f"[INFO] Generated {len(manifest)} json files")
    if not manifest:
        raise RuntimeError("No inference JSONs were generated.")

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
