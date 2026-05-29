import os
import sys
import time
import csv
import argparse
from glob import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm import tqdm

from utils.misc_utils import *
from utils.dataloader_CMRx4DFlow import CMRx4DFlowDataSet

sys.path.append("../")
from Utils.utils_datasl import save_coo_npz
from Utils.utils_metrics import nRMSE, SSIM, RelErr, AngErr
from Utils.utils_flow import complex2magflow
from networks.flowvn_lowmem import FlowVNLowMem
from networks.flowvn import FlowVN


class ParamGradTensorBoardCallback(pl.Callback):
    """
    Log:
    - Histograms of each parameter (optional)
    - Histograms of each parameter gradient (optional)
    - Scalar stats such as norm, mean, std, max for parameters/gradients
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        log_hist: bool = False,
        log_stats: bool = True,
        max_params: int | None = None,
        grad_none_as_zero: bool = False,
    ):
        self.log_every_n_steps = log_every_n_steps
        self.log_hist = log_hist
        self.log_stats = log_stats
        self.max_params = max_params
        self.grad_none_as_zero = grad_none_as_zero

    def _should_log(self, trainer: "pl.Trainer"):
        return (trainer.global_step % self.log_every_n_steps) == 0 and trainer.global_step > 0

    @torch.no_grad()
    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if not trainer.training:
            return
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return
        if not self._should_log(trainer):
            return

        writer = trainer.logger.experiment
        step = trainer.global_step

        n = 0
        for name, p in pl_module.named_parameters():
            if (self.max_params is not None) and (n >= self.max_params):
                break
            n += 1

            if self.log_hist:
                writer.add_histogram(f"params/{name}", p.detach().float().cpu(), step)

            if self.log_stats:
                pdata = p.detach().float()
                writer.add_scalar(f"params_norm/{name}", pdata.norm().item(), step)
                writer.add_scalar(f"params_absmax/{name}", pdata.abs().max().item(), step)
                writer.add_scalar(f"params_mean/{name}", pdata.mean().item(), step)
                writer.add_scalar(f"params_std/{name}", pdata.std(unbiased=False).item(), step)

            g = p.grad
            if g is None:
                if not self.grad_none_as_zero:
                    writer.add_scalar(f"grads_is_none/{name}", 1.0, step)
                    continue
                g = torch.zeros_like(p)

            gdata = g.detach().float()
            if self.log_hist:
                writer.add_histogram(f"grads/{name}", gdata.cpu(), step)

            if self.log_stats:
                writer.add_scalar(f"grads_norm/{name}", gdata.norm().item(), step)
                writer.add_scalar(f"grads_absmax/{name}", gdata.abs().max().item(), step)
                writer.add_scalar(f"grads_mean/{name}", gdata.mean().item(), step)
                writer.add_scalar(f"grads_std/{name}", gdata.std(unbiased=False).item(), step)

        writer.flush()


class CMRSaveCallback(pl.Callback):
    def __init__(self):
        self._cache = {}

    def _key(self, meta: dict):
        out_dir = str(meta.get("out_dir", ""))
        case_dir = str(meta.get("case_dir", ""))
        R = int(meta.get("usrate", 0))
        base = out_dir if out_dir not in ("", "None") else case_dir
        return (base, R)

    def _flush_one(self, base_out: str, R: int):
        expected_segs = [0, 1, 2, 3]
        pack = self._cache.get((base_out, R), None)
        if pack is None:
            return

        recon_map = pack["recon"]
        seg_map = pack["seg"]
        missing = [i for i in expected_segs if i not in recon_map]
        if missing:
            return

        img = np.stack([recon_map[i] for i in expected_segs], axis=0)
        img = np.transpose(img, (0, 1, 4, 3, 2))

        s = seg_map[0]
        s = np.transpose(s, (2, 1, 0))
        s = s[None, None, :, :, :]

        out_dir = Path(base_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        print("SAVE", out_dir)
        save_coo_npz(str(out_dir / f"img_ktGaussian{R}.npz"), img * s)

        csv_path = out_dir / f"recontime_ktGaussian{R}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["recontime"])
            w.writerow([pack["recon_ms_sum"] / 1000.0])

        del self._cache[(base_out, R)]

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        meta = {k: outputs.get(k) for k in outputs.keys() if k != "recon"}
        seg_idx = int(meta.get("seg_idx", 0))
        recon_ms = meta.get("recon_ms", 0.0)

        k = self._key(meta)
        if k not in self._cache:
            self._cache[k] = {"recon": {}, "seg": {}, "meta": meta, "recon_ms_sum": 0.0}

        if recon_ms is not None:
            self._cache[k]["recon_ms_sum"] += float(recon_ms)

        recon = outputs["recon"]
        x = recon[0] if recon.ndim == 5 else recon
        x_np = x.numpy()

        seg = batch["segmentation"]
        if hasattr(seg, "detach"):
            seg = seg.detach().cpu().numpy()
        seg = seg.astype(bool)[0]

        self._cache[k]["recon"][seg_idx] = x_np
        self._cache[k]["seg"][seg_idx] = seg

        base_out, R = k
        self._flush_one(base_out, R)

    def on_test_epoch_end(self, trainer, pl_module):
        for (base_out, R) in list(self._cache.keys()):
            self._flush_one(base_out, R)
        self._cache.clear()


class UnrolledNetwork(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs
        self.log_img_count = 0

        if self.options["network"] == "FlowVN":
            use_lowmem = bool(self.options.get("lowmem", False)) and (self.options.get("mode") == "test")
            if use_lowmem:
                self.network = FlowVNLowMem(**self.options)
            else:
                self.network = FlowVN(**self.options)
        elif self.options["network"] == "FlowMRI_Net":
            self.network = FlowMRI_Net(**self.options)
        else:
            raise ValueError("Network does not exist")

        self.L1_loss = nn.L1Loss()

        self._vis_done_per_R = {}
        self._fixed_vis_case = None
        self.vis_nv_idx = 0
        self.vis_nt_idx = 0
    def on_fit_start(self):  # 不影响 test
        pass
    def _to_metrics_predgt(self, x: torch.Tensor) -> np.ndarray:
        if torch.is_tensor(x):
            x = x.detach().cpu()
        x_np = x.numpy()
        return np.transpose(x_np, (0, 1, 4, 3, 2))

    def _to_metrics_seg(self, seg: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(seg):
            seg = seg.detach().cpu().numpy()[0]
        seg = seg.astype(bool)
        return np.transpose(seg, (2, 1, 0))

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - x.min()
        d = x.max() - x.min()
        return x / (d if d > 0 else 1.0)

    def _log_val_images(self, recon, gt, batch, usrate):
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return
        if self.global_rank != 0:
            return

        subj = batch["subj"][0] if isinstance(batch["subj"], (list, tuple)) else batch["subj"]
        slice_start = int(batch["slice_start"][0]) if hasattr(batch["slice_start"], "__len__") else int(batch["slice_start"])
        seg_idx = int(batch["seg_idx"][0]) if hasattr(batch["seg_idx"], "__len__") else int(batch["seg_idx"])
        case_id = (subj, slice_start, seg_idx)

        if self._fixed_vis_case is None:
            self._fixed_vis_case = case_id

        if case_id != self._fixed_vis_case:
            return
        if self._vis_done_per_R.get(int(usrate), False):
            return

        self._vis_done_per_R[int(usrate)] = True

        pred = recon.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        if pred.ndim == 6:
            pred = pred[0]
            gt_np = gt_np[0]
        if pred.ndim == 5:
            Nv, Nt, FE, PE, SPE = pred.shape
        else:
            raise RuntimeError(f"Unexpected pred shape: {pred.shape}")

        nv = int(np.clip(self.vis_nv_idx, 0, Nv - 1))
        nt = int(np.clip(self.vis_nt_idx, 0, Nt - 1))
        spe = SPE // 2

        pred_mag = np.abs(pred[nv, nt, :, :, spe])
        gt_mag = np.abs(gt_np[nv, nt, :, :, spe])

        pred_vis = self._norm01(pred_mag)
        gt_vis = self._norm01(gt_mag)

        step = self.global_step
        base = f"val_images/usrate{int(usrate)}/{subj}_slice{slice_start}_seg{seg_idx}_nv{nv}_nt{nt}_spe{spe}"

        self.logger.experiment.add_image(
            f"{base}/gt", torch.from_numpy(gt_vis[None, :, :]), step
        )
        self.logger.experiment.add_image(
            f"{base}/pred", torch.from_numpy(pred_vis[None, :, :]), step
        )

    def training_step(self, batch):
        if self.options["loss"] == "ssdu":
            kdata_p2 = batch["kdata_p2"]
            loss_mask = abs(kdata_p2[:, :, 0, :, 0, :, :]) != 0

            recon_img_p1 = self.network(
                (batch["imdata_p1"]),
                batch["kdata_p1"],
                batch["coil_sens"],
                batch["usrate_true"],
            )
            kdata_p1 = mri_forward_op(recon_img_p1, batch["coil_sens"], loss_mask.float())
            loss = (
                0.5
                * torch.norm(torch.view_as_real(kdata_p2) - torch.view_as_real(kdata_p1), p=2)
                / torch.norm(torch.view_as_real(kdata_p2), p=2)
                + 0.5
                * torch.norm(torch.view_as_real(kdata_p2) - torch.view_as_real(kdata_p1), p=1)
                / torch.norm(torch.view_as_real(kdata_p2), p=1)
            )

        elif self.options["loss"] == "supervised":
            recon_img_p1 = self.network(
                Variable(batch["imdata_p1"], requires_grad=True),
                batch["kdata_p1"],
                batch["coil_sens"],
                batch["usrate_true"],
            )

            if self.options["exp_loss"]:
                tau = self.current_epoch / 10
                w = torch.exp(
                    torch.Tensor(
                        [
                            -tau * (self.options["num_stages"] - k + 1)
                            for k in range(self.options["num_stages"])
                        ]
                    ).to(self.device)
                )
                w /= torch.sum(w)
                loss = (
                    torch.sum(
                        w
                        * torch.norm(
                            recon_img_p1 - batch["gt"],
                            p=1,
                            dim=[1, 2, 3, 4, 5, 6],
                        )
                    )
                    / 40000
                )
            else:
                loss = self.L1_loss(
                    recon_img_p1 - batch["gt"][:, 0], torch.zeros_like(recon_img_p1)
                )

            if (not torch.isfinite(loss)) or (loss.item() > 10):
                print(
                    "[LOSS SPIKE]",
                    loss.item(),
                    batch["norm"],
                    batch["case_dir"],
                    batch["slice_start"],
                    "seg",
                    batch["seg_idx"],
                    "usrate",
                    batch["usrate"],
                )

        self.log_dict(
            {"train_loss_epoch": loss, "step": self.current_epoch * 1.0},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {"loss": loss}

    def on_validation_epoch_start(self):
        self._vis_done_per_R = {}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        usrate = int(batch["usrate"][0]) if hasattr(batch["usrate"], "__len__") else int(batch["usrate"])

        recon = self.network(
            batch["imdata_p1"], batch["kdata_p1"], batch["coil_sens"], batch["usrate_true"]
        )

        gt = batch["gt"]
        if torch.is_tensor(gt) and gt.ndim == 6 and gt.shape[1] == 1:
            gt = gt[:, 0]
        if recon.ndim == 6 and recon.shape[1] == 1:
            recon = recon[:, 0]

        loss = self.L1_loss(recon - gt, torch.zeros_like(recon))

        self.log(
            f"val/loss_usrate{usrate}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        recon_vis = recon * batch["norm"]
        gt_vis = gt * batch["norm"]
        self._log_val_images(recon_vis, gt_vis, batch, usrate)

        return {"val_loss": loss, "usrate": usrate}

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        recon_ms = None
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        if self.options["T_size"] == -1 or self.options["network"] == "FlowVN":
            recon_img = self.network(
                batch["imdata_p1"], batch["kdata_p1"], batch["coil_sens"], batch["usrate_true"]
            )
        else:
            window_t = self.options["T_size"] - 2
            n_bins = batch["imdata_p1"].shape[2]
            recon_img = torch.zeros_like(batch["imdata_p1"][:, :, 0:1, :, :, :]).repeat(
                1, 1, int(np.ceil(n_bins / window_t)) * window_t, 1, 1, 1
            )
            for t in range(0, n_bins, window_t):
                cardiac_bins = list(range(t - self.options["T_size"] + 1, t + 1))
                recon_img[:, :, t : t + window_t] = self.network(
                    batch["imdata_p1"][:, :, cardiac_bins],
                    batch["kdata_p1"][:, :, :, cardiac_bins],
                    batch["coil_sens"],
                    batch["usrate_true"],
                )[:, :, 1:-1]
            recon_img = torch.roll(recon_img[:, :, :n_bins], shifts=-window_t, dims=2)

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            recon_ms = float(start_event.elapsed_time(end_event))

        norm = batch["norm"].to(recon_img[0].device, non_blocking=True)
        recon_img_complex = (recon_img[0] * norm).detach().cpu()
        del recon_img
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        usrate = int(batch["usrate"][0]) if hasattr(batch["usrate"], "__len__") else int(batch["usrate"])
        seg_idx = int(batch["seg_idx"][0]) if hasattr(batch["seg_idx"], "__len__") else int(batch["seg_idx"])
        subj = batch["subj"][0] if isinstance(batch["subj"], (list, tuple)) else batch["subj"]

        case_dir = batch["case_dir"][0] if isinstance(batch["case_dir"], (list, tuple)) else batch["case_dir"]
        out_dir = batch["out_dir"][0] if isinstance(batch["out_dir"], (list, tuple)) else batch["out_dir"]
        torch.cuda.empty_cache()
        return {
            "recon": recon_img_complex,
            "subj": subj,
            "case_dir": case_dir,
            "out_dir": out_dir,
            "seg_idx": seg_idx,
            "usrate": usrate,
            "recon_ms": recon_ms,
        }

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.options["lr"])
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=self.options["epoch"]
                )
            },
        }


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Network arguments")

    parser.add_argument("--D_size", type=int, default=1, help="number of slices per volume (FlowVN only)")
    parser.add_argument("--T_size", type=int, default=5, help="number of cardiac bins per volume")
    parser.add_argument("--root_dir", type=str, default="data/own_card3d", help="directory of the data")
    parser.add_argument("--save_dir", type=str, default="results/exp", help="directory of the experiment")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint to test or restart training")
    parser.add_argument("--input", type=str, default="", help="name of network input file")

    parser.add_argument("--grad_check", type=bool, default=False, help="use gradient checkpointing")
    parser.add_argument("--network", type=str, default="", help="model to use (FlowVN or FlowMRI_Net)")
    parser.add_argument("--num_stages", type=int, default=10, help="number of stages in the network")
    parser.add_argument("--features_in", type=int, default=1, help="number of input dimensions")
    parser.add_argument("--features_out", type=int, default=24, help="number of filters for convolutional kernel")

    parser.add_argument("--kernel_size", type=int, default=7, help="xyz kernel size")
    parser.add_argument("--act", type=str, default="linear", help="what activation to use, rbf or linear")
    parser.add_argument("--num_act_weights", type=int, default=71, help="number of basis functions for activation")
    parser.add_argument("--grid", type=float, default=0.25, help="grid size for linear act")
    parser.add_argument("--weight", type=float, default=0.025, help="scale weights for RBF kernel")
    parser.add_argument("--vmin", type=float, default=-3.5, help="min value of filter response for rbf activation")
    parser.add_argument("--vmax", type=float, default=3.5, help="max value of filter response for rbf activation")
    parser.add_argument("--sgd_momentum", type=bool, default=True, help="use sgd momentum")
    parser.add_argument("--exp_loss", type=bool, default=False, help="use exponentially weighted loss")

    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epoch", type=int, default=100, help="number of training epoch")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--loss", type=str, default="", help="type of loss (ssdu or supervised)")
    parser.add_argument(
        "--usrate",
        type=int,
        nargs="+",
        default=None,
        help="test only: one or more ktGaussian undersampling rates, e.g. --usrate 10 20 30",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ids, e.g. --devices 0 or --devices 0 1 2 3",
    )
    parser.add_argument(
        "--test_roots",
        type=str,
        nargs="*",
        default=None,
        help="test mode: one or more roots to scan for cases (recursive)",
    )
    parser.add_argument(
        "--in_base_dir",
        type=str,
        default=None,
        help="base input dir used to compute relative path, e.g. .../ChallengeData/TaskR1&R2",
    )
    parser.add_argument(
        "--out_base_dir",
        type=str,
        default=None,
        help="base output dir used to mirror directory structure, e.g. .../ChallengeData_FlowVN/TaskR1&R2",
    )
    parser.add_argument(
        "--lowmem",
        action="store_true",
        help="test-only: use FlowVNLowMem for inference to reduce memory",
    )
    return parser


def _train(args: dict, parser: argparse.ArgumentParser):
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms = True

    save_dir = Path(args["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger("./results/lightning_logs", name="")

    n_run = str(max((int(p.split("_")[-1]) for p in glob("./results/lightning_logs/*")), default=0) + 1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir,
        filename=n_run + "-epoch{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=1,
        save_last=False,
        save_weights_only=True,
    )

    paramgrad_cb = ParamGradTensorBoardCallback(
        log_every_n_steps=50,
        log_hist=False,
        log_stats=True,
        max_params=None,
    )

    dataset = CMRx4DFlowDataSet(**args)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=True)

    args_val = vars(parser.parse_args())
    args_val["mode"] = "val"
    val_dataset = CMRx4DFlowDataSet(**args_val)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args["devices"],
        strategy="ddp_find_unused_parameters_true" if len(args["devices"]) > 1 else "auto",
        max_epochs=args["epoch"],
        logger=logger,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, paramgrad_cb],
        check_val_every_n_epoch=1,
    )

    if args["ckpt_path"] is not None:
        model = UnrolledNetwork.load_from_checkpoint(args["ckpt_path"], **args)
    else:
        model = UnrolledNetwork(**args)

    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)


def _test(args: dict):
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms = True

    if args.get("usrate", None) is None:
        raise ValueError("test mode requires --usrate, e.g. --usrate 10 or --usrate 10 20 30")

    if isinstance(args["usrate"], int):
        args["usrate"] = [int(args["usrate"])]
    else:
        args["usrate"] = [int(u) for u in args["usrate"]]

    if args.get("test_roots", None) is not None:
        args["test_roots"] = args["test_roots"]

    if args.get("out_base_dir", None) in (None, "", "None"):
        args["out_base_dir"] = args["save_dir"]

    if args.get("in_base_dir", None) in (None, "", "None"):
        tr = args.get("test_roots", None)
        if tr and len(tr) > 0:
            p = Path(tr[0]).resolve()
            args["in_base_dir"] = str(p.parent) if p.name in ("ValidationSet", "TrainSet") else str(p)

    dataset = CMRx4DFlowDataSet(**args)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)
    save_callback = CMRSaveCallback()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args["devices"][0]] if isinstance(args["devices"], (list, tuple)) else [args["devices"]],
        logger=False,
        callbacks=[save_callback],
    )

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda(0)
    else:
        map_location = "cpu"

    model = UnrolledNetwork.load_from_checkpoint(
    args["ckpt_path"],
    map_location="cpu",   # 关键：先别让 Lightning/torch 自动放 cuda
    **args,
    )
    model.eval()

    trainer.test(model, dataloaders=dataloader)


def main():
    parser = _build_arg_parser()
    args_ns = parser.parse_args()
    print_options(parser, args_ns)
    args = vars(args_ns)

    if args["mode"] == "train":
        _train(args, parser)
    elif args["mode"] == "test":
        _test(args)
    else:
        raise ValueError("mode must be 'train' or 'test'")


if __name__ == "__main__":
    main()