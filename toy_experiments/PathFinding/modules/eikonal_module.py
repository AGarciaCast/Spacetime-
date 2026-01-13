import pytorch_lightning as pl


import torch
from omegaconf import DictConfig
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn


import torch
from omegaconf import DictConfig
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
import time


from tqdm import tqdm

EPSILON = 1e-8


def _get_config_value(config, opt_config, name, default=None, alt_names=None):
    if alt_names is None:
        alt_names = []
    for key in [name] + alt_names:
        if opt_config is not None and hasattr(opt_config, key):
            return getattr(opt_config, key)
        if hasattr(config, key):
            return getattr(config, key)
    return default


def _create_optimizer(config, params):

    opt_config = getattr(config, "optimizer", None)
    optimizer_name = _get_config_value(
        config, opt_config, "optimizer", alt_names=["name", "type"]
    )
    learning_rate = _get_config_value(
        config, opt_config, "learning_rate", alt_names=["lr"], default=1e-3
    )
    weight_decay = _get_config_value(config, opt_config, "weight_decay", default=0.0)
    beta1 = _get_config_value(config, opt_config, "beta1", default=0.9)
    beta2 = _get_config_value(config, opt_config, "beta2", default=0.999)
    eps = _get_config_value(config, opt_config, "eps", default=1e-8)
    warmup_steps = _get_config_value(config, opt_config, "warmup_steps", default=0)
    decay_steps = _get_config_value(config, opt_config, "decay_steps", default=0)
    decay_rate = _get_config_value(config, opt_config, "decay_rate", default=1.0)
    staircase = _get_config_value(config, opt_config, "staircase", default=False)
    schedule_free = _get_config_value(
        config, opt_config, "schedule_free", default=False
    )

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "Soap":
        try:
            from soap import SOAP
        except ImportError as exc:
            raise ImportError(
                "Soap optimizer requested but not available. "
                "Provide a SOAP optimizer implementation or install its package."
            ) from exc

        optimizer = SOAP(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=0.0,
            precondition_frequency=2,
        )

    elif optimizer_name == "Muon":
        ns_coefficients = _get_config_value(
            config, opt_config, "ns_coefficients", default=(2, -1.5, 0.5)
        )
        ns_steps = _get_config_value(config, opt_config, "ns_steps", default=10)
        momentum = _get_config_value(
            config, opt_config, "beta", default=0.99, alt_names=["momentum"]
        )
        nesterov = _get_config_value(config, opt_config, "nesterov", default=True)

        try:
            optimizer = torch.optim.Muon(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_coefficients=ns_coefficients,
                eps=eps,
                ns_steps=ns_steps,
            )
        except AttributeError as exc:
            raise ImportError(
                "Muon optimizer requested but torch.optim.Muon is not available."
            ) from exc

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = None
    # if warmup_steps > 0 or decay_steps > 0:

    #     def lr_lambda(step):
    #         if warmup_steps > 0 and step < warmup_steps:
    #             return float(step) / float(max(1, warmup_steps))
    #         if decay_steps <= 0:
    #             return 1.0
    #         effective_step = step - warmup_steps if warmup_steps > 0 else step
    #         if staircase:
    #             exponent = effective_step // decay_steps
    #         else:
    #             exponent = effective_step / float(decay_steps)
    #         return decay_rate**exponent

    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler


class EikonalLightningModule(pl.LightningModule):

    def __init__(
        self,
        config: DictConfig,
        solver: torch.nn.Module,
    ):
        super().__init__()

        self.config = config
        self.solver = solver

        self.power = self.config.eikonal.power

        # Store autodecoder.
        self.mse = nn.MSELoss()

        self.epsilon = EPSILON

        # Set val and train loss
        self.best_val_loss = float("inf")
        self.best_val_epoch = 0
        self.best_train_loss = float("inf")
        self.best_train_epoch = 0

    def eiko_loss(self, norm_grad):
        eikp = torch.pow(norm_grad, self.power) - torch.ones_like(norm_grad)
        res = eikp / self.power
        return torch.sum(torch.abs(res)) / (res.shape[0])

    def on_validation_epoch_end(self) -> None:
        # Compute average loss over all samples

        # Reset accumulators for the next epoch
        self.total_test_mse = 0.0
        self.total_test_points = 0

        # Check if we obtained a new best validation loss
        if "val_eiko_epoch" in self.trainer.callback_metrics:
            if self.trainer.callback_metrics["val_eiko_epoch"] < self.best_val_loss:
                self.best_val_loss = self.trainer.callback_metrics["val_eiko_epoch"]
                self.best_val_epoch = self.trainer.current_epoch
                self.log(
                    "best_val_eiko",
                    self.best_val_loss,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=True,
                )

        # Check if we obtained a new best training loss
        if self.trainer.callback_metrics["train_eiko_epoch"] < self.best_train_loss:
            self.best_train_loss = self.trainer.callback_metrics["train_eiko_epoch"]
            self.best_train_epoch = self.trainer.current_epoch
            self.log(
                "best_train_eiko",
                self.best_train_loss,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val, gradient_clip_algorithm
    ):

        for param in self.solver.parameters():
            if param.grad is not None:
                # Apply gradient clipping
                torch.nn.utils.clip_grad_value_(param, gradient_clip_val)

    def forward(self, batch_coords, reuse_grad=False, aux_vel=False):
        # Forward pass through the model
        outputs = self.solver.times_grad_vel(batch_coords, reuse_grad, aux_vel)
        return outputs

    def training_step(self, batch, batch_idx):
        batch_coords, _ = batch

        # Forward pass through the model
        outputs = self.forward(batch_coords, reuse_grad=True, aux_vel=True)

        times, grads, norm_grad, pred_vel = outputs

        # Calculate loss
        loss = self.eiko_loss(norm_grad)
        self.log("train_eiko", loss, on_epoch=True, prog_bar=True)

        # Calculate metrics

        batch_mse = self.mse(pred_vel, torch.ones_like(pred_vel))
        self.total_train_mse += batch_mse.item()
        num_points_batch = pred_vel.shape[0] * pred_vel.shape[1] * 2.0
        self.total_train_points += num_points_batch

        self.log(
            "train_mse_step",
            batch_mse / num_points_batch,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass through the model
        batch_coords, _ = batch

        # Forward pass through the model
        batch_coords = batch_coords.to(self.device)
        outputs = self.solver.times_grad_vel(
            batch_coords, reuse_grad=False, aux_vel=True
        )

        times, grads, norm_grad, pred_vel = outputs

        # Calculate loss
        loss = self.eiko_loss(norm_grad)
        self.log(f"{self.name_test}_eiko", loss, on_epoch=True, prog_bar=True)

        # Calculate metrics
        batch_mse = self.mse(pred_vel, torch.ones_like(pred_vel))
        self.total_test_mse += batch_mse.item()
        num_points_batch = pred_vel.shape[0] * pred_vel.shape[1] * 2
        self.total_test_points += num_points_batch

        self.log(
            f"{self.name_test}_mse_step",
            batch_mse / num_points_batch,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer, scheduler = _create_optimizer(self.config, self.solver.parameters())
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
