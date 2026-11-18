"""Small shape adapters around author implementations; no substitute networks."""

from __future__ import annotations

import importlib
import json
import sys

import torch

from ..common.data import fft2, sense_adjoint
from ..common.safety import ROOT


def prepend_author(path):
    path = ROOT / path
    if not path.is_dir():
        raise FileNotFoundError(f"Author repository missing: {path}; run bootstrap first")
    sys.path.insert(0, str(path))
    # Each comparison runs in its own process; refuse accidental main-model imports.
    if "utils" in sys.modules or "models" in sys.modules:
        raise RuntimeError("Comparison must run in a fresh process to avoid main-repo import collisions")
    return path


class FlowVNAdapter:
    def __init__(self, cfg, device):
        prepend_author(cfg["author"]["path"])
        from networks.flowvn import FlowVN

        self.model = FlowVN(**cfg["network"]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["optimizer"]["lr"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg["num_epochs"])

    def reconstruct(self, batch):
        return self.step(batch, 0, 0)[1]

    def step(self, batch, epoch, step):
        b, e, t, c, s, h, w = batch["kspace"].shape
        x0 = sense_adjoint(batch["kspace"], batch["coils"], batch["mask"]).reshape(b * e, 1, t, s, h, w)
        measured = batch["kspace"].permute(0, 1, 3, 2, 4, 5, 6).reshape(b * e, 1, c, t, s, h, w)
        coils = batch["coils"].repeat_interleave(e, dim=0)
        rate = batch["acceleration"].to(dtype=x0.real.dtype).repeat_interleave(e)
        pred = self.model(x0, measured, coils, rate).reshape(b, e, t, s, h, w)
        # The author's non-exp supervised objective is complex absolute L1.
        loss = torch.nn.functional.l1_loss(pred - batch["target"], torch.zeros_like(pred))
        return loss, pred, {"author_complex_l1": loss.detach()}


class FlowMoDLAdapter:
    backward_in_step = True

    def __init__(self, cfg, device):
        author = prepend_author(cfg["author"]["path"])
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from utils.metrics import extract_velocity
        from train import train_step

        with initialize_config_dir(config_dir=str(author / "configs"), version_base=None):
            native = compose(config_name=cfg["author"]["config_name"],
                             overrides=["seed=2026", "data.root=unused_readonly_manifest",
                                        *(f"{key}={value}" for key, value in cfg["hydra"].items())])
        self.native, self.train_step, self.velocity = native, train_step, extract_velocity
        self.probe_flow_loss = bool(cfg["paper"].get("_smoke_flow_ramp_probe"))
        self.model = instantiate(native.model).to(device)
        self.criterion = instantiate(native.loss).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **cfg["optimizer"])
        self.scheduler = instantiate(native.training.lr_scheduler, optimizer=self.optimizer)
        self.device = device

    def prepare(self, batch):
        # Native FlowMoDL is [B,E,C,T,Z,Y,X]; common contract is [B,E,T,C,X,Z,Y].
        hybrid = batch["kspace"].permute(0, 1, 3, 2, 5, 6, 4).contiguous()
        # Its decouple_readout=True forward performs IFFT(X). Undo only that transform
        # here so the author code receives raw kx of the already selected x slab.
        raw = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(hybrid, dim=-1), dim=-1, norm="ortho"), dim=-1)
        target = batch["target"].permute(0, 1, 2, 4, 5, 3).contiguous()
        data = {"kspace_us": raw, "target_image": target,
                "target_velocity": self.velocity(target, batch["venc"]),
                "mask": batch["mask"][:, None, None, :, :, :, None],
                "smaps": batch["coils"].permute(0, 1, 3, 4, 2).contiguous(),
                "segmask": batch["segmask"].permute(0, 2, 3, 1).contiguous(),
                "v_enc": batch["venc"], "usrate": batch["acceleration"].float()}
        return data

    def reconstruct(self, batch):
        from utils.inference import reconstruct

        data = self.prepare(batch)
        pred = reconstruct(self.model, data["kspace_us"], data["mask"], data["smaps"],
                           data["usrate"], per_encoding=False, decouple_readout=True)
        return pred.permute(0, 1, 2, 5, 3, 4).contiguous()

    def step(self, batch, epoch, step):
        data = self.prepare(batch)
        loss_epoch = max(epoch, 32) if self.probe_flow_loss else epoch
        value, pred = self.train_step(self.model, self.criterion, data,
                                      tau=step * self.native.training.tau_multiplier,
                                      K=self.native.model.num_layers, epoch=loss_epoch, cfg=self.native)
        pred = pred.permute(0, 1, 2, 5, 3, 4).contiguous()
        loss = torch.tensor(value, device=self.device)
        return loss, pred, {"author_composite_flow_loss": loss.detach(), "curriculum_epoch": loss_epoch,
                            **self.criterion.curriculum_scales}


class CUTAdapter:
    inference_uses_ema = True

    def __init__(self, cfg, device):
        author = prepend_author(cfg["author"]["path"])
        import yaml
        from src.models.cascade import CMRxTransformerCascade
        from src.training.train_cmrx import training_step, _EMA

        native = yaml.safe_load((author / "configs/train.yaml").read_text())
        native.update(lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"],
                      n_epochs=cfg["num_epochs"], seed=cfg["paper"]["seed"])
        self.native, self.device, self.training_step = native, device, training_step
        self.model = CMRxTransformerCascade(**native).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=native["lr"], weight_decay=native["weight_decay"])
        self.ema = _EMA(self.model, decay=native["ema"]["decay"])
        from src.training.train_cmrx import _lr_lambda

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda epoch: _lr_lambda(epoch, native["warmup_epochs"], native["n_epochs"], native["eta_min"], native["lr"]))
        self.prediction = None
        self.model.register_forward_hook(self._capture_prediction)

    def _capture_prediction(self, model, args, output):
        self.prediction = output

    def after_step(self):
        self.ema.update(self.model)

    def state_dict(self):
        return {"ema": self.ema.state_dict()}

    def load_state_dict(self, state):
        if state.get("ema") is not None:
            self.ema.load(state["ema"])

    def reconstruct(self, batch):
        raw = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        try:
            self.model.load_state_dict(self.ema.state_dict(), strict=True)
            return self.step(batch, 0, 0)[1]
        finally:
            self.model.load_state_dict(raw, strict=True)

    def step(self, batch, epoch, step):
        b, e, t, c, s, h, w = batch["kspace"].shape
        if b != 1:
            raise ValueError("The native CUT training_step uses one spatial slice per call")
        results, losses = [], []
        initial = sense_adjoint(batch["kspace"], batch["coils"], batch["mask"])
        for si in range(s):
            theta = batch["mask"][:, None, None]
            lam = 1 - theta
            rate = float(batch["acceleration"][0])
            if self.model.cond_table is not None:
                # The author architecture has an anchored table, not a continuous R input.
                rate = min(self.model.cond_table.r_values, key=lambda r: abs(r - rate))
            native_batch = {"img_zf": initial[:, :, :, si],
                            "kdata_theta": batch["kspace"][:, :, :, :, si].permute(0, 1, 3, 2, 4, 5),
                            "coilmap": batch["coils"][:, :, si],
                            "mask_theta": theta,
                            "kdata_lambda": batch["full_kspace"][:, :, :, :, si].permute(0, 1, 3, 2, 4, 5) * lam,
                            "mask_lambda": lam, "cardiac_bins": batch["frame_indices"][0],
                            "R": rate, "seg_idx": 0, "scanner_weight": 1.0}
            loss, _ = self.training_step(self.model, native_batch, self.device,
                                         apply_scanner_weight=False, lambda_v=self.native["lambda_v"])
            losses.append(loss)
            results.append(self.prediction)
        value = torch.stack(losses).mean()
        return value, torch.stack(results, dim=3), {"author_kspace_velocity_loss": value.detach()}


def factory(cfg, device):
    return {"flowvn": FlowVNAdapter, "flowmodl": FlowMoDLAdapter, "cut_4dflow": CUTAdapter}[cfg["paper"]["method"]](cfg, device)
