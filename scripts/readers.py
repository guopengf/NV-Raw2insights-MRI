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

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Sequence

import numpy as np
import scipy
from monai.config import PathLike
from monai.data.image_reader import ImageReader
from monai.data.utils import is_supported_format
from monai.utils import StrEnum, optional_import, require_pkg
from numpy import ndarray
from scipy.fft import fftn, fftshift, ifftn, ifftshift

h5py, has_h5py = optional_import("h5py")

__all__ = ["FastMRIReader", "CestMRIReader", "CMRxReconReader"]


class FastMRIKeys(StrEnum):
    """
    The keys to be used for extracting data from the fastMRI dataset
    """

    KSPACE = "kspace"
    MASK = "mask"
    FILENAME = "filename"
    RECON = "reconstruction_rss"
    ACQUISITION = "acquisition"
    MAX = "max"
    NORM = "norm"
    PID = "patient_id"


class CestMRIKeys(StrEnum):
    """
    The keys to be used for extracting data from the CestMRI dataset
    """

    KSPACE = "kspace"
    MASK = "mask"
    FILENAME = "filename"
    CSM = "sensitivity_maps"
    RECON_RSS = "reconstruction_rss"
    RECON_SENSE = "reconstruction_sense"
    ACQUISITION = "acquisition"
    MAX = "max"
    NORM = "norm"
    PID = "patient_id"
    SHAPE = "shape"


class CMRxReconKeys(StrEnum):
    """
    The keys to be used for extracting data from the CMRxRecon dataset
    """

    KSPACE = "kspace_full"
    MASK = "mask"
    MASK_TYPE = "mask_type"
    FILENAME = "filename"
    RECON = "reconstruction_rss"
    RECON_RAW = "reconstruction"
    ACQUISITION = "acquisition"
    MAX = "max"
    NORM = "norm"
    PID = "patient_id"
    NUM_SLICES = "num_slices"
    NUM_COILS = "num_coils"
    NUM_FRAMES = "num_frames"
    SHAPE = "shape"
    SENSITIVITY_MAPS = "sensitivity_maps"


@require_pkg(pkg_name="h5py")
class FastMRIReader(ImageReader):
    """
    Load fastMRI files with '.h5' suffix. fastMRI files, when loaded with "h5py",
    are HDF5 dictionary-like datasets. The keys are:

    - kspace: contains the fully-sampled kspace
    - reconstruction_rss: contains the root sum of squares of ifft of kspace. This
        is the ground-truth image.

    It also has several attributes with the following keys:

    - acquisition (str): acquisition mode of the data (e.g., AXT2 denotes T2 brain MRI scans)
    - max (float): dynamic range of the data
    - norm (float): norm of the kspace
    - patient_id (str): the patient's id whose measurements were recorded
    """

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
         Verify whether the specified file format is supported by h5py reader.

        Args:
             filename: file name
        """
        suffixes: Sequence[str] = [".h5"]
        return has_h5py and is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike) -> dict:  # type: ignore
        """
        Read data from specified h5 file.
        Note that the returned object is a dictionary.

        Args:
            data: file name to read.
        """
        if isinstance(data, (tuple, list)):
            data = data[0]

        with h5py.File(data, "r") as f:
            # extract everything from the ht5 file
            dat = dict(
                [(key, f[key][()]) for key in f]
                + [(key, f.attrs[key]) for key in f.attrs]
                + [(FastMRIKeys.FILENAME, os.path.basename(data))]  # type: ignore
            )
        f.close()

        return dat

    def get_data(self, dat: dict) -> tuple[ndarray, dict]:
        """
        Extract data array and metadata from the loaded data and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.

        Args:
            dat: a dictionary loaded from an h5 file
        """
        header = self._get_meta_dict(dat)
        data: ndarray = np.array(dat[FastMRIKeys.KSPACE])[np.newaxis, ...]
        header[FastMRIKeys.MASK] = (
            np.expand_dims(np.array(dat[FastMRIKeys.MASK]), 0)[None, ..., None]
            if FastMRIKeys.MASK in dat.keys()
            else np.zeros(data.shape)
        )
        data_shape = data.shape
        header[CMRxReconKeys.NUM_FRAMES] = data_shape[0]
        header[CMRxReconKeys.NUM_SLICES] = data_shape[1]
        header[CMRxReconKeys.NUM_COILS] = data_shape[2]
        header[CMRxReconKeys.SHAPE] = np.array(data_shape)
        mask = np.ones([1] * data.ndim)
        header[CMRxReconKeys.MASK] = mask.astype(np.float32)
        header[FastMRIKeys.ACQUISITION] = header[FastMRIKeys.ACQUISITION].replace("AXT1PRE", "AXT1")
        return data, header

    def _get_meta_dict(self, dat: dict) -> dict:
        """
        Get all the metadata of the loaded dict and return the meta dict.

        Args:
            dat: a dictionary object loaded from an h5 file.
        """
        return {k.value: dat[k.value] for k in FastMRIKeys if k.value in dat}


class CestMRIReader(ImageReader):
    """
    Load fastMRI files with '.h5' suffix. fastMRI files, when loaded with "h5py",
    are HDF5 dictionary-like datasets. The keys are:

    - kspace: contains the fully-sampled kspace
    - reconstruction_rss: contains the root sum of squares of ifft of kspace. This
        is the ground-truth image.

    It also has several attributes with the following keys:

    - acquisition (str): acquisition mode of the data (e.g., AXT2 denotes T2 brain MRI scans)
    - max (float): dynamic range of the data
    - norm (float): norm of the kspace
    - patient_id (str): the patient's id whose measurements were recorded
    """

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
         Verify whether the specified file format is supported by h5py reader.

        Args:
             filename: file name
        """
        suffixes: Sequence[str] = [".h5"]
        return has_h5py and is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike) -> dict:  # type: ignore
        """
        Read data from specified h5 file.
        Note that the returned object is a dictionary.

        Args:
            data: file name to read.
        """
        if isinstance(data, (tuple, list)):
            data = data[0]

        with h5py.File(data, "r") as f:
            # extract everything from the ht5 file
            dat = dict(
                [(key, f[key][()]) for key in f]
                + [(key, f.attrs[key]) for key in f.attrs]
                + [(CestMRIKeys.FILENAME, os.path.basename(data))]  # type: ignore
            )
        f.close()

        return dat

    def get_data(self, dat: dict) -> tuple[ndarray, dict]:
        """
        Extract data array and metadata from the loaded data and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.

        Args:
            dat: a dictionary loaded from an h5 file
        """
        header = self._get_meta_dict(dat)
        data: ndarray = np.array(dat[CestMRIKeys.KSPACE])
        data = fftshift(
            ifftn(ifftshift(data, axes=[-3, -2, -1]), axes=[-3, -2, -1], norm="ortho"),
            axes=[-3, -2, -1],
        ).transpose(0, 4, 1, 2, 3)
        data = fftshift(
            fftn(ifftshift(data, axes=[-2, -1]), axes=[-2, -1], norm="ortho"),
            axes=[-2, -1],
        )
        header[CestMRIKeys.MASK] = (
            np.array(dat[CestMRIKeys.MASK]) if CestMRIKeys.MASK in dat.keys() else np.zeros(data.shape)
        )
        header[CestMRIKeys.SHAPE] = np.array(data.shape)
        return data, header

    def _get_meta_dict(self, dat: dict) -> dict:
        """
        Get all the metadata of the loaded dict and return the meta dict.

        Args:
            dat: a dictionary object loaded from an h5 file.
        """
        return {k.value: dat[k.value] for k in CestMRIKeys if k.value in dat}

@require_pkg(pkg_name="h5py")
class CMRxReconReader(ImageReader):
    def __init__(self, fixed_mask_types=None):
        super().__init__()
        self.fixed_mask_types = fixed_mask_types if isinstance(fixed_mask_types, list) else [fixed_mask_types]

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        suffixes: Sequence[str] = [".json"]
        return has_h5py and is_supported_format(filename, suffixes)

    def read_mat(self, mat_file: Sequence[PathLike]) -> list:
        try:
            with h5py.File(mat_file, "r", swmr=True) as f:
                data_kv = [(key, f[key][()]) for key in f]
        except BaseException:
            data = scipy.io.loadmat(mat_file)
            data_kv = [(key, data[key]) for key in data if not key.startswith("__")]
        return data_kv

    def read_mat_dict(self, mat_file: Sequence[PathLike]) -> dict:
        return dict(self.read_mat(mat_file))

    def _select_mat_array_key(self, dat, preferred_keys: Sequence[str] = ()) -> str:
        for key in preferred_keys:
            if key in dat and hasattr(dat[key], "shape"):
                return key
        for key, value in dat.items():
            if not key.startswith("__") and hasattr(value, "shape"):
                return key
        raise ValueError("Could not find any array in MAT file.")

    def read_first_mat_array(
        self,
        mat_file: Sequence[PathLike],
        preferred_keys: Sequence[str] = (),
        selection=None,
        return_shape: bool = False,
    ) -> ndarray:
        try:
            with h5py.File(mat_file, "r", swmr=True) as f:
                key = self._select_mat_array_key(f, preferred_keys)
                dataset = f[key]
                shape = dataset.shape
                value = dataset[selection] if selection is not None else dataset[()]
        except OSError:
            dat = self.read_mat_dict(mat_file)
            key = self._select_mat_array_key(dat, preferred_keys)
            shape = dat[key].shape
            value = dat[key][selection] if selection is not None else dat[key]
        if return_shape:
            return value, shape
        return value

    def filter_masks_by_types(self, masks, fixed_mask_types):
        result = []
        if not all(fixed_mask_types):
            return masks
        for mask in masks:
            if any(mask_type in mask for mask_type in fixed_mask_types):
                result.append(mask)
        return result

    def _infer_mask_type(self, mask_path: str) -> str:
        base = os.path.basename(mask_path)
        if "_mask_" in base:
            return base.split("_mask_")[-1][:-4]
        if base.startswith("usmask_") and base.endswith(".mat"):
            return base[len("usmask_"):-4]
        if base.startswith("mask_") and base.endswith(".mat"):
            return base[len("mask_"):-4]
        return os.path.splitext(base)[0]

    def _to_complex_array(self, arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr)
        if np.issubdtype(arr.dtype, np.complexfloating):
            return arr
        if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
            return np.array(arr["real"] + 1j * arr["imag"])
        if arr.ndim > 0 and arr.shape[-1] == 2 and np.issubdtype(arr.dtype, np.floating):
            return np.array(arr[..., 0] + 1j * arr[..., 1])
        raise ValueError(f"Unsupported array dtype for complex conversion: {arr.dtype}")

    def _find_first_existing_key(self, dat: dict, candidates: list[str]) -> str | None:
        for k in candidates:
            if k in dat:
                return k
        return None

    def read(self, data: Sequence[PathLike] | PathLike) -> dict:  # type: ignore
        if isinstance(data, (tuple, list)):
            data = data[0]

        with open(data, "r") as f:
            json_data = json.load(f)

        if bool(json_data.get("is_4dflow", False)):
            kspace = json_data["kspace"]
            target_kspace = json_data.get("target_kspace", json_data.get("gt_kspace", json_data.get("full_kspace")))
            if target_kspace is None:
                raise ValueError(f"4D flow JSON must include target_kspace/full_kspace: {data}")

            all_masks = json_data.get("mask", [])
            masks = self.filter_masks_by_types(all_masks, self.fixed_mask_types)
            if len(masks) == 0:
                masks = all_masks
            mask = random.choice(masks) if len(masks) > 0 else ""
            mask_type = json_data.get("mask_type", self._infer_mask_type(mask) if mask else "fixed")

            enc_idx = int(json_data.get("encoding_idx", 0))
            encoding_selection = (slice(enc_idx, enc_idx + 1), Ellipsis)
            kspace_input, input_shape = self.read_first_mat_array(
                kspace,
                preferred_keys=("kdata", "kdata_ktGaussian", "kus", "kspace", "kspace_full"),
                selection=encoding_selection,
                return_shape=True,
            )
            kspace_target, target_shape = self.read_first_mat_array(
                target_kspace,
                preferred_keys=("kdata_full", "kdata", "kspace_full", "kspace"),
                selection=encoding_selection,
                return_shape=True,
            )

            dat = {
                CMRxReconKeys.FILENAME: os.path.basename(data),
                CMRxReconKeys.MASK_TYPE: mask_type,
                CMRxReconKeys.ACQUISITION: json_data.get("acquisition", "Flow4d"),
                "is_4dflow": True,
                "encoding_idx": enc_idx,
                "num_encodings": int(target_shape[0]) if len(target_shape) > 0 else int(input_shape[0]),
                "coilmap_axis_order": json_data.get("coilmap_axis_order", "auto"),
                "normalize_coilmap": bool(json_data.get("normalize_coilmap", True)),
                "kspace_4dflow_input": kspace_input,
                "kspace_4dflow_target": kspace_target,
            }
            if mask:
                dat[CMRxReconKeys.MASK] = self.read_first_mat_array(
                    mask,
                    preferred_keys=("mask", "usmask", "sampling_mask"),
                )
            if "coilmap" in json_data and json_data["coilmap"]:
                dat[CMRxReconKeys.SENSITIVITY_MAPS] = self.read_first_mat_array(
                    json_data["coilmap"],
                    preferred_keys=("coilmap", "csm", "sensitivity_maps", "sens_maps"),
                )
            return dat

        kspace = json_data["kspace"]
        all_masks = json_data.get("mask", [])
        masks = self.filter_masks_by_types(all_masks, self.fixed_mask_types)
        if len(masks) == 0:
            masks = all_masks
        mask = random.choice(masks) if len(masks) > 0 else ""

        mask_type = json_data.get("mask_type", self._infer_mask_type(mask) if mask else "fixed")
        acquisition_type = json_data.get("acquisition", None)
        if acquisition_type is None:
            m = re.search(r"(?:^|[/\\])MultiCoil[/\\]([^/\\]+)", kspace, flags=re.I)
            acquisition_type = m.group(1) if m else "Flow2d"

        kspace_kv = self.read_mat(kspace)
        mask_kv = self.read_mat(mask) if mask else [(None, None)]

        dat = dict(
            kspace_kv
            + mask_kv
            + [
                (CMRxReconKeys.FILENAME, os.path.basename(data)),
                (CMRxReconKeys.MASK_TYPE, mask_type),
                (CMRxReconKeys.ACQUISITION, acquisition_type),
            ]
        )

        dat["is_4dflow"] = bool(json_data.get("is_4dflow", False))
        if "encoding_idx" in json_data:
            dat["encoding_idx"] = int(json_data["encoding_idx"])
            
        return dat

    def get_data(self, dat: dict) -> tuple[ndarray, dict]:
        header = self._get_meta_dict(dat)

        # ----------------------------------------------------------
        # 4D flow path
        # raw k-space kept in original format:
        #   (enc, t, coil, kz, ky, kx)
        # raw mask kept in original format:
        #   (1,   t, 1,    kz, ky, 1)
        # IMPORTANT:
        #   do NOT convert kz->z here
        #   do NOT merge enc*t here
        # ----------------------------------------------------------
        if dat.get("is_4dflow", False):
            if "kspace_4dflow_input" in dat and "kspace_4dflow_target" in dat:
                raw_input = self._to_complex_array(dat["kspace_4dflow_input"])
                raw_target = self._to_complex_array(dat["kspace_4dflow_target"])
            else:
                kspace_key = self._find_first_existing_key(
                    dat,
                    ["kdata_full", "kdata", "kspace_full", "kspace", "kus", "kdata_ktGaussian"],
                )
                if kspace_key is None:
                    raise ValueError("Could not find 4D flow k-space key in .mat file.")
                raw_input = self._to_complex_array(dat[kspace_key])
                raw_target = raw_input

            if raw_input.ndim != 6 or raw_target.ndim != 6:
                raise ValueError(
                    "Expected 4D flow raw k-space to be 6D, "
                    f"got input={raw_input.shape}, target={raw_target.shape}"
                )
            
            # raw shape: (enc, t, coil, kz, ky, kx)
            n_enc_loaded, nt, nc, nkz, nky, nkx = raw_target.shape
            n_enc_all = int(dat.get("num_encodings", n_enc_loaded))
            if raw_input.shape != raw_target.shape:
                raise ValueError(
                    "4D flow input and target k-space must have the same shape, "
                    f"got input={raw_input.shape}, target={raw_target.shape}"
                )
            
            if "encoding_idx" not in dat:
                raise ValueError("encoding_idx is required for 4D flow when enc is split into batch.")
            
            enc_idx = int(dat["encoding_idx"])
            if not (0 <= enc_idx < n_enc_all):
                raise ValueError(f"encoding_idx={enc_idx} out of range for raw shape {raw_target.shape}")

            # keep singleton enc dim so transforms.py still sees a 6D tensor
            if n_enc_loaded == n_enc_all and n_enc_all > 1:
                raw_input = raw_input[enc_idx : enc_idx + 1]    # shape: (1, t, coil, kz, ky, kx)
                raw_target = raw_target[enc_idx : enc_idx + 1]  # shape: (1, t, coil, kz, ky, kx)
            elif n_enc_loaded == 1:
                raw_input = raw_input[:1]
                raw_target = raw_target[:1]
            else:
                raise ValueError(
                    "Unexpected loaded encoding dimension for 4D flow: "
                    f"loaded={n_enc_loaded}, total={n_enc_all}, encoding_idx={enc_idx}"
                )
            data = raw_target
            
            header[CMRxReconKeys.PID] = os.path.splitext(dat[CMRxReconKeys.FILENAME])[0]
            
            # downstream semantics for ONE encoding only:
            # (1, t, coil, kz, ky, kx)
            # -> (1, t, x, coil, kz, ky)
            # -> (t, x, coil, kz, ky)
            header[CMRxReconKeys.NUM_FRAMES] = nt
            header[CMRxReconKeys.NUM_SLICES] = nkx
            header[CMRxReconKeys.NUM_COILS] = nc
            header[CMRxReconKeys.SHAPE] = np.array([nt, nkx, nc, nkz, nky], dtype=np.int32)
            header["num_encodings"] = n_enc_all
            header["encoding_idx"] = enc_idx
            header["coilmap_axis_order"] = dat.get("coilmap_axis_order", "auto")
            header["normalize_coilmap"] = bool(dat.get("normalize_coilmap", True))

            mask_key = self._find_first_existing_key(dat, ["mask", "usmask_ktGaussian", "usmask", "sampling_mask"])
            if mask_key is not None:
                raw_mask = np.array(dat[mask_key]).astype(np.float32)

                if raw_mask.ndim == 2:
                    raw_mask = raw_mask[None, None, None, :, :, None]
                elif raw_mask.ndim == 3:
                    raw_mask = raw_mask[None, :, None, :, :, None]
                elif raw_mask.ndim == 4:
                    raw_mask = raw_mask[None, :, None, :, :, :] if raw_mask.shape[-1] == 1 else raw_mask[:, :, None, :, :, None]
                elif raw_mask.ndim == 5:
                    raw_mask = raw_mask[:, :, None, :, :, :] if raw_mask.shape[0] == 1 else raw_mask[None, :, :, :, :, :]

                if raw_mask.ndim != 6:
                    raise ValueError(f"Unsupported 4D flow mask shape: {raw_mask.shape}")

                header[CMRxReconKeys.MASK] = raw_mask.astype(np.float32)
            else:
                header[CMRxReconKeys.MASK] = np.ones((1, nt, 1, nkz, nky, 1), dtype=np.float32)

            header["kspace_4dflow_input"] = raw_input
            if CMRxReconKeys.SENSITIVITY_MAPS in dat:
                header[CMRxReconKeys.SENSITIVITY_MAPS] = self._to_complex_array(dat[CMRxReconKeys.SENSITIVITY_MAPS])

            return data, header

        # ----------------------------------------------------------
        # Original CMRxRecon path
        # ----------------------------------------------------------
        if "kus" in dat:
            kspace_key = "kus"
        else:
            kspace_key = CMRxReconKeys.KSPACE if CMRxReconKeys.KSPACE in dat else "kspace"

        if np.issubdtype(dat[kspace_key].dtype, np.complexfloating):
            data_shape = dat[kspace_key].shape[::-1]
            data_shape = (1,) * (5 - len(data_shape)) + data_shape
            data: ndarray = dat[kspace_key].transpose()
        else:
            data_shape = dat[kspace_key]["real"].shape
            data_shape = (1,) * (5 - len(data_shape)) + data_shape
            data: ndarray = np.array(dat[kspace_key]["real"] + 1j * dat[kspace_key]["imag"])
            data = data.reshape(data_shape)

        header[CMRxReconKeys.PID] = os.path.splitext(dat[CMRxReconKeys.FILENAME])[0].split("_")[0]
        header[CMRxReconKeys.NUM_FRAMES] = data_shape[0]
        header[CMRxReconKeys.NUM_SLICES] = data_shape[1]
        header[CMRxReconKeys.NUM_COILS] = data_shape[2]
        header[CMRxReconKeys.SHAPE] = np.array(data_shape)

        if CMRxReconKeys.MASK in dat.keys():
            mask = np.array(dat[CMRxReconKeys.MASK])
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=(0, 1))
            elif mask.ndim == 3:
                mask = np.expand_dims(mask, axis=(1, 2))
        else:
            mask = np.ones([1] * data.ndim)

        header[CMRxReconKeys.MASK] = mask.astype(np.float32)
        return data, header

    def _get_meta_dict(self, dat: dict) -> dict:
        return {k.value: dat[k.value] for k in CMRxReconKeys if k.value in dat and k != CMRxReconKeys.KSPACE}
