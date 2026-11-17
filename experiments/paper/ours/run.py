"""A0-A4/final-model training entry; imports the pinned project model and losses."""

from __future__ import annotations

import sys
import math

import torch

from ..common.data import fft2
from ..common.execution import execute, parser
from ..common.safety import ROOT


class Ours:
    def __init__(self, cfg, device):
        sys.path.insert(0, str(ROOT / "scripts"))
        from models.latent_recon import create_mri_recon_model
        from utils import Config, validate_phase3_config
        from joint_encoding import JointEncodingLoss
        from mri_data.data_utils import FlowVNPhaseLoss

        self.cfg, self.device = cfg, device
        p = cfg["phase3"]
        if p["enable_vaa"] or p["freeze"]["backbone"]:
            raise ValueError("This paper A0-A4 entry requires VAA off and backbone training on")
        if p["loss"]["use_ssim_zy"] or p["loss"]["use_vascular"]:
            raise ValueError("The fixed paper study uses complex L1/joint flow losses; SSIM/vascular VAA loss are disabled")
        args = Config(cfg)
        validate_phase3_config(args)
        self.model = create_mri_recon_model(args).to(device)
        self.optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        def lr_factor(epoch):
            warmup = int(cfg["warmup_epochs"])
            if epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            progress = min(1.0, (epoch - warmup) / max(1, cfg["num_epochs"] - warmup))
            minimum = cfg["min_lr"] / cfg["lr"]
            return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)
        self.phase = FlowVNPhaseLoss(method=cfg["phase3"]["loss"]["phase"]["method"])
        loss = cfg["phase3"]["loss"]
        self.joint = None
        if loss["joint"]["enabled"]:
            weight = lambda key: loss[key]["weight"] if loss[key]["enabled"] else 0.0
            self.joint = JointEncodingLoss(profile="official_aligned", complex_weight=weight("complex_roi"),
                                          circular_weight=weight("cross_venc_phase"), velocity_weight=weight("velocity"),
                                          relerr_weight=weight("relerr"), angular_weight=weight("angular"),
                                          velocity_type=loss["velocity"]["type"], eps=loss["joint"]["eps"])

    def reconstruct(self, batch):
        b, e, t, c, s, h, w = batch["kspace"].shape
        inp = torch.view_as_real(fft2(batch["kspace"], inverse=True)).permute(0, 1, 4, 2, 3, 5, 6, 7).reshape(b * e, s, t, c, h, w, 2).contiguous()
        coils = batch["coils"][:, None, None].expand(b, e, t, c, s, h, w).permute(0, 1, 4, 2, 3, 5, 6).reshape(b * e, s, t, c, h, w)
        sens = torch.view_as_real(coils.contiguous())
        mask = batch["mask"][:, None, None, :, None, :, :, None].expand(b, e, s, t, c, h, w, 2).reshape(b * e, s, t, c, h, w, 2).bool()
        acceleration = int(batch["acceleration"][0])
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda" and self.cfg["amp"]):
            out = self.model(inp, mask, "kt_gaussian", acceleration, "Flow4d", sensitivity_maps=sens, mra_prior=None)
        complex_out = torch.view_as_complex(out.float().contiguous())
        combined = (complex_out * coils.conj()).sum(dim=3).reshape(b, e, s, t, h, w).permute(0, 1, 3, 2, 4, 5)
        return combined

    def step(self, batch, epoch, step):
        combined = self.reconstruct(batch)
        b, e, t, s, h, w = combined.shape
        center_t = t // 2
        pred = torch.view_as_real(combined[:, :, center_t].contiguous())
        target = torch.view_as_real(batch["target"][:, :, center_t].contiguous())
        # Center frame, all S slices; preserve the original loss implementations.
        total = combined.real.sum() * 0.0
        components = {}
        if self.cfg["phase3"]["loss"]["use_phase"]:
            phase = self.phase(pred.reshape(b * e * s, 1, h, w, 2), target.reshape(b * e * s, 1, h, w, 2))
            total = total + phase * self.cfg["phase3"]["loss"]["phase"]["weight"]
            components["complex_l1_weighted"] = total.detach()
        if self.joint is not None:
            ramp_cfg = self.cfg["phase3"]["loss"]["flow_loss_ramp"]
            ramp = min(1.0, max(0.0, (epoch + 1 - ramp_cfg["start_epoch"]) / max(1, ramp_cfg["end_epoch"] - ramp_cfg["start_epoch"]))) if ramp_cfg["enabled"] else 1.0
            if self.cfg["paper"].get("_smoke_flow_ramp_probe"):
                ramp = 1.0
            joint, values = self.joint(pred.unsqueeze(3), target.unsqueeze(3), batch["segmask"], ramp=ramp)
            if self.cfg["paper"].get("_smoke_flow_ramp_probe") and torch.is_grad_enabled():
                gradient = torch.autograd.grad(joint, combined, retain_graph=True)[0]
                if not torch.isfinite(gradient).all() or gradient.abs().sum() <= 0:
                    raise AssertionError("Fully enabled joint flow loss has zero/non-finite output gradient")
                components["joint_output_gradient_norm"] = gradient.detach().norm()
            total = total + joint
            components.update({f"joint_{k}": v.detach() for k, v in values.items()})
            components["joint_weighted"], components["flow_ramp"] = joint.detach(), ramp
        return total, combined, components


if __name__ == "__main__":
    execute(parser().parse_args(), Ours)
