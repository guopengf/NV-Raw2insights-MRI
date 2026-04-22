# Copyright (c) MONAI Consortium

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.



import argparse

import json

from pathlib import Path



import h5py

import numpy as np

import pandas as pd

from mri_data.ktSampling import kt_gaussian_sampling





DEFAULT_ACS_LINES = 20

DEFAULT_ALPHA = 0.2

DEFAULT_ACCELERATION = 24





def centered_ifft_1d(data: np.ndarray, axis: int) -> np.ndarray:

    return np.fft.fftshift(

        np.fft.ifft(np.fft.ifftshift(data, axes=axis), axis=axis, norm="ortho"),

        axes=axis,

    )





def to_mat_complex(data: np.ndarray) -> np.ndarray:

    mat_dtype = np.dtype([("real", np.float32), ("imag", np.float32)])

    mat_data = np.empty(data.shape, dtype=mat_dtype)

    mat_data["real"] = np.asarray(data.real, dtype=np.float32)

    mat_data["imag"] = np.asarray(data.imag, dtype=np.float32)

    return mat_data





def save_hdf5_mat(path: Path, key: str, data: np.ndarray) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:

        f.create_dataset(key, data=data)





def read_kdata_full(case_dir: Path) -> np.ndarray:

    kspace_file = case_dir / "kdata_full.mat"

    with h5py.File(kspace_file, "r") as f:

        if "kdata_full" not in f:

            keys = list(f.keys())

            raise KeyError(f"Expected 'kdata_full' in {kspace_file}, found keys: {keys}")

        kdata = f["kdata_full"][()]



    if kdata.dtype.fields is None or {"real", "imag"} - set(kdata.dtype.fields):

        raise TypeError(f"Expected compound real/imag data in {kspace_file}, got dtype {kdata.dtype}")



    return np.asarray(kdata["real"] + 1j * kdata["imag"], dtype=np.complex64)





def read_case_metadata(case_dir: Path) -> dict:

    metadata = {"case_dir": str(case_dir.resolve())}

    params_file = case_dir / "params.csv"

    if not params_file.exists():

        return metadata



    params_df = pd.read_csv(params_file)

    if params_df.empty:

        return metadata



    record = params_df.iloc[0].to_dict()

    metadata.update({k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()})

    return metadata





def create_retrospective_mask(nt: int, ny: int, nx: int, acceleration: int, acs_lines: int, alpha: float, seed: int):

    mask = kt_gaussian_sampling(

        nx,

        ny,

        nt,

        acs_lines,

        acceleration,

        alpha=alpha,

        seed=seed,

    )

    return mask.transpose().astype(np.float32)





def get_case_prefix(case_dir: Path) -> str:

    parent_names = case_dir.parts[-3:]

    return "_".join(parent_names)





def convert_case(

    case_dir: Path,

    output_dir: Path,

    acceleration: int,

    acs_lines: int,

    alpha: float,

    seed: int,

) -> None:

    kdata_full = read_kdata_full(case_dir)

    if kdata_full.ndim != 6:

        raise ValueError(f"Expected 6D k-space [venc, time, coil, kz, ky, kx], got shape {kdata_full.shape}")



    num_venc, nt, num_coils, nz, ny, nx = kdata_full.shape

    case_prefix = get_case_prefix(case_dir)

    metadata = read_case_metadata(case_dir)



    undersample_dir = output_dir / "MultiCoil" / "Flow2d" / "UnderSample_TaskR1"

    mask_dir = output_dir / "MultiCoil" / "Flow2d" / "Mask_TaskR1"

    undersample_dir.mkdir(parents=True, exist_ok=True)

    mask_dir.mkdir(parents=True, exist_ok=True)



    manifest = {

        "source_case_dir": str(case_dir.resolve()),

        "acquisition": "Flow2d",

        "acceleration": acceleration,

        "mask_type": f"ktGaussian{acceleration}",

        "shape_in": [int(v) for v in kdata_full.shape],

        "shape_out_per_case": [int(nt), int(nz), int(num_coils), int(ny), int(nx)],

        "metadata": metadata,

        "cases": [],

    }



    for venc_idx in range(num_venc):

        hybrid_kspace = centered_ifft_1d(kdata_full[venc_idx], axis=2)

        hybrid_kspace = np.transpose(hybrid_kspace, (0, 2, 1, 3, 4))



        retrospective_mask = create_retrospective_mask(

            nt=nt,

            ny=ny,

            nx=nx,

            acceleration=acceleration,

            acs_lines=acs_lines,

            alpha=alpha,

            seed=seed + venc_idx,

        )

        masked_kspace = hybrid_kspace * retrospective_mask[:, None, None, :, :]



        base_name = f"{case_prefix}_enc{venc_idx}"

        kspace_file = undersample_dir / f"{base_name}_kus_ktGaussian{acceleration}.mat"

        mask_file = mask_dir / f"{base_name}_mask_ktGaussian{acceleration}.mat"

        json_file = output_dir / f"{base_name}_kus_ktGaussian{acceleration}.json"



        save_hdf5_mat(kspace_file, "kus", to_mat_complex(masked_kspace))

        save_hdf5_mat(mask_file, "mask", retrospective_mask)



        json_payload = {"kspace": str(kspace_file.resolve()), "mask": [str(mask_file.resolve())]}

        json_file.write_text(json.dumps(json_payload, indent=4))



        manifest["cases"].append(

            {

                "venc_index": venc_idx,

                "json": str(json_file.resolve()),

                "kspace": str(kspace_file.resolve()),

                "mask": str(mask_file.resolve()),

                "shape": [int(nt), int(nz), int(num_coils), int(ny), int(nx)],

            }

        )



    metadata_dir = output_dir / "metadata"

    metadata_dir.mkdir(parents=True, exist_ok=True)

    (metadata_dir / "cmrx4dflow_manifest.json").write_text(json.dumps(manifest, indent=4))





def main():

    parser = argparse.ArgumentParser(

        description="Convert CMRx4Dflow full k-space into CMRxRecon-style inference inputs.",

    )

    parser.add_argument("--source_dir", type=Path, required=True, help="Path to one CMRx4Dflow case directory.")

    parser.add_argument(

        "--output_dir",

        type=Path,

        required=True,

        help="Output directory containing generated JSON, k-space, and mask files.",

    )

    parser.add_argument(

        "--acceleration",

        type=int,

        default=DEFAULT_ACCELERATION,

        help=f"Retrospective k-t Gaussian acceleration. Default: {DEFAULT_ACCELERATION}.",

    )

    parser.add_argument(

        "--acs_lines",

        type=int,

        default=DEFAULT_ACS_LINES,

        help=f"Number of fully sampled ACS lines. Default: {DEFAULT_ACS_LINES}.",

    )

    parser.add_argument(

        "--alpha",

        type=float,

        default=DEFAULT_ALPHA,

        help=f"Gaussian mask density alpha. Default: {DEFAULT_ALPHA}.",

    )

    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic retrospective masks.")

    args = parser.parse_args()



    source_dir = args.source_dir.resolve()

    output_dir = args.output_dir.resolve()

    if not source_dir.exists():

        raise FileNotFoundError(f"Source directory not found: {source_dir}")



    convert_case(

        case_dir=source_dir,

        output_dir=output_dir,

        acceleration=args.acceleration,

        acs_lines=args.acs_lines,

        alpha=args.alpha,

        seed=args.seed,

    )

    print(f"Converted {source_dir} into CMRxRecon-style inference inputs at {output_dir}")





if __name__ == "__main__":

    main()

