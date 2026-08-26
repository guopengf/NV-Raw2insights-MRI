"""Official-compatible case-level 4D Flow validation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tools.evaluate_4dflow_submission import compute_corrmap, evaluate_arrays, import_official_eval


DEFAULT_EVAL_CODE_DIR = Path("/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT/EvaluationCode")


def coil_combined_samples_to_official_volume(samples: torch.Tensor, final_shape) -> np.ndarray:
    """Convert [E,time*x,z,y,2] to official complex [E,time,z,y,x]."""

    if samples.ndim != 5 or samples.shape[-1] != 2:
        raise ValueError(f"Expected [E,N,z,y,2] with an optional singleton coil removed, got {tuple(samples.shape)}")
    num_frames = int(final_shape[-5])
    num_slices = int(final_shape[-4])
    if samples.shape[1] != num_frames * num_slices:
        raise ValueError(
            f"Sample count {samples.shape[1]} does not match time*x={num_frames}*{num_slices}"
        )
    values = torch.view_as_complex(samples.contiguous()).reshape(
        samples.shape[0], num_frames, num_slices, samples.shape[-3], samples.shape[-2]
    )
    return values.permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.complex64, copy=False)


class OfficialFlowMetricEvaluator:
    """Load official functions once and cache target-derived MSAC correction maps."""

    def __init__(self, eval_code_dir: str | Path = DEFAULT_EVAL_CODE_DIR):
        self.funcs = import_official_eval(Path(eval_code_dir))
        self._corr_cache: dict[str, np.ndarray] = {}

    def evaluate(
        self,
        case_id: str,
        pred: np.ndarray,
        target: np.ndarray,
        roi_mask_zyx: np.ndarray | None,
        *,
        corr_cache_id: str | None = None,
    ) -> dict[str, float]:
        if roi_mask_zyx is None:
            roi_mask_zyx = np.ones(target.shape[-3:], dtype=np.float32)
        else:
            roi_mask_zyx = np.asarray(roi_mask_zyx, dtype=np.float32)
        cache_id = case_id if corr_cache_id is None else corr_cache_id
        if cache_id not in self._corr_cache:
            self._corr_cache[cache_id] = compute_corrmap(self.funcs, target, roi_mask_zyx)
        row = evaluate_arrays(
            self.funcs,
            pred,
            target,
            roi_mask_zyx,
            corr_maps=self._corr_cache[cache_id],
        )
        return {
            "ssim": row["SSIM"],
            "nrmse": row["nRMSE"],
            "relerr": row["RelErr"],
            "angerr": row["AngErr"],
        }
