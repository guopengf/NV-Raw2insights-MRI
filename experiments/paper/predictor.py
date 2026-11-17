"""Inference-only forwards around existing models; no targets or loss computation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .common.data import sense_adjoint
from .common.safety import sha256


LEARNED = {"ours", "flowvn", "flowmodl", "cut_4dflow"}


class Predictor:
    def __init__(self, cfg, device, checkpoint=None):
        self.cfg, self.method, self.device = cfg, cfg["paper"]["method"], torch.device(device)
        self.adapter = None
        self.provenance = {"method": self.method, "checkpoint": None}
        if self.method not in LEARNED:
            if self.method not in {"cg_sense", "cs_tv"}:
                raise ValueError(f"Unknown reconstruction method: {self.method}")
            return
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError("Learned inference requires an explicit trained checkpoint; no random/HF fallback")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(self.device)
        if self.method == "ours":
            from .ours.run import Ours

            self.adapter = Ours(cfg, self.device)
        else:
            from .comparisons.learned import factory

            self.adapter = factory(cfg, self.device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        weights = None
        if self.method == "cut_4dflow":
            weights = state.get("adapter", {}).get("ema", state.get("ema_model"))
            if weights is None:
                raise ValueError("CUT Final inference requires saved EMA weights")
        if weights is None:
            for key in ("model", "net_state_dict", "model_state_dict", "state_dict"):
                if key in state and isinstance(state[key], dict):
                    weights = state[key]
                    break
        if weights is None and state and all(isinstance(v, torch.Tensor) for v in state.values()):
            weights = state
        if weights is None:
            raise ValueError("No recognized model state in checkpoint")
        weights = {key.removeprefix("module."): value for key, value in weights.items()}
        # Inference must not silently skip weights and score an initialized model.
        self.adapter.model.load_state_dict(weights, strict=True)
        self.adapter.model.eval()
        self.provenance = {"method": self.method, "checkpoint": str(Path(checkpoint).resolve()),
                           "checkpoint_sha256": sha256(checkpoint), "epoch": state.get("epoch"),
                           "optimizer_steps": state.get("step"), "uses_ema": self.method == "cut_4dflow"}
        del state, weights

    @torch.no_grad()
    def __call__(self, batch):
        if self.method in {"cg_sense", "cs_tv"}:
            from .comparisons.solvers import cg_sense, cs_tv

            solver = cg_sense if self.method == "cg_sense" else cs_tv
            value, details = solver(batch, self.cfg["solver"])
            self.provenance["last_solver_status"] = details
            return value.numpy()
        b = {key: v.to(self.device) if isinstance(v, torch.Tensor) else v for key, v in batch.items()}
        if self.method == "ours":
            value = self.adapter.reconstruct(b)
        elif self.method == "flowvn":
            n, e, t, c, s, h, w = b["kspace"].shape
            x0 = sense_adjoint(b["kspace"], b["coils"], b["mask"]).reshape(n * e, 1, t, s, h, w)
            k = b["kspace"].permute(0, 1, 3, 2, 4, 5, 6).reshape(n * e, 1, c, t, s, h, w)
            value = self.adapter.model(x0, k, b["coils"].repeat_interleave(e, 0), b["acceleration"].repeat_interleave(e)).reshape(n, e, t, s, h, w)
        elif self.method == "flowmodl":
            from utils.inference import reconstruct

            hybrid = b["kspace"].permute(0, 1, 3, 2, 5, 6, 4).contiguous()
            raw = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(hybrid, dim=-1), dim=-1, norm="ortho"), dim=-1)
            value = reconstruct(self.adapter.model, raw, b["mask"][:, None, None, :, :, :, None],
                                b["coils"].permute(0, 1, 3, 4, 2).contiguous(),
                                b["acceleration"].float(), per_encoding=False, decouple_readout=True)
            value = value.permute(0, 1, 2, 5, 3, 4)
        else:
            from src.models.conditioning import build_m

            initial = sense_adjoint(b["kspace"], b["coils"], b["mask"])
            model = self.adapter.model
            r = int(b["acceleration"][0])
            m = build_m(model.conditioning_inputs, r, None, self.device) if model.conditioning_enabled else None
            phases = b["frame_indices"].float() / b["kspace"].shape[2]
            outputs = []
            for si in range(b["kspace"].shape[4]):
                outputs.append(model(initial[:, :, :, si], b["kspace"][:, :, :, :, si].permute(0, 1, 3, 2, 4, 5),
                                     b["coils"][:, :, si], b["mask"][:, None, None], phases, m=m, R=r))
            value = torch.stack(outputs, dim=3)
        if not torch.is_complex(value):
            raise ValueError("Model inference lost complex phase")
        return value.detach().cpu().numpy().astype(np.complex64, copy=False)
