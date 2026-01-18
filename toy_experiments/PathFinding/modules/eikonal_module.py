import time
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    SequentialLR,
)

EPSILON = 1e-8


class EikonalLightningModule(pl.LightningModule):

    def __init__(
        self,
        config: DictConfig,
        solver: torch.nn.Module,
    ):
        super().__init__()

        self.save_hyperparameters(
            {"config": OmegaConf.to_container(config, resolve=True)}
        )

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
        self._train_step_start = None

    def _log_batch_stats(self, prefix, norm_grad, pred_vel):
        with torch.no_grad():
            residual = (torch.pow(norm_grad, self.power) - 1.0) / self.power
            abs_residual = residual.abs()
            def _q(tensor, q):
                return torch.quantile(tensor, q)
            stats = {
                "residual_mean": abs_residual.mean(),
                "residual_max": abs_residual.max(),
                "residual_p90": _q(abs_residual, 0.9),
                "residual_p99": _q(abs_residual, 0.99),
                "norm_grad_mean": norm_grad.mean(),
                "norm_grad_max": norm_grad.max(),
                "norm_grad_p90": _q(norm_grad, 0.9),
                "norm_grad_p99": _q(norm_grad, 0.99),
                "vel_mean": pred_vel.mean(),
                "vel_std": pred_vel.std(unbiased=False),
                "vel_p90": _q(pred_vel, 0.9),
                "vel_p99": _q(pred_vel, 0.99),
            }

        for name, value in stats.items():
            self.log(
                f"{prefix}_{name}_step",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            self.log(
                f"{prefix}_{name}_epoch",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def eiko_loss(self, norm_grad):
        res1 = torch.pow(norm_grad, self.power) - torch.ones_like(norm_grad)
        res1 = res1 / self.power

        vel = 1.0 / (norm_grad + self.epsilon)
        res2 = torch.pow(vel, self.power) - torch.ones_like(norm_grad)
        res2 = res2 / self.power

        loss_type = getattr(self.config.eikonal, "residual_loss", "l1").lower()
        if loss_type == "huber":
            beta = float(getattr(self.config.eikonal, "residual_huber_beta", 1.0))
            return (
                F.smooth_l1_loss(res1, torch.zeros_like(res1), beta=beta)
                + F.smooth_l1_loss(res2, torch.zeros_like(res2), beta=beta)
            ) / 2.0

        if loss_type == "clipped_l1":
            clip = float(getattr(self.config.eikonal, "residual_clip", 10.0))
            res1 = torch.clamp(res1, min=-clip, max=clip)
            res2 = torch.clamp(res2, min=-clip, max=clip)

        return (torch.mean(torch.abs(res1)) + torch.mean(torch.abs(res2))) / 2.0

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
        train_eiko_epoch = self.trainer.callback_metrics.get("train_eiko_epoch")
        if train_eiko_epoch is not None and train_eiko_epoch < self.best_train_loss:
            self.best_train_loss = train_eiko_epoch
            self.best_train_epoch = self.trainer.current_epoch
            self.log(
                "best_train_eiko",
                self.best_train_loss,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )

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
        loss_eiko = self.eiko_loss(norm_grad)
        vel_reg_weight = float(getattr(self.config.eikonal, "vel_reg_weight", 0.0))
        vel_std = pred_vel.std(unbiased=False)
        loss = loss_eiko + vel_reg_weight * vel_std

        self.log("train_eiko", loss_eiko, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_total", loss, on_step=True, on_epoch=True, prog_bar=True)
        if vel_reg_weight > 0.0:
            self.log(
                "train_vel_reg",
                vel_reg_weight * vel_std,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )

        # Calculate metrics
        batch_mse = torch.mean((pred_vel - 1.0) ** 2, dim=(0, 1))

        self.log(
            "train_mse_step",
            batch_mse,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log(
            "train_mse_epoch",
            batch_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self._log_batch_stats("train", norm_grad, pred_vel)

        if self.logger and hasattr(self.logger, "experiment"):
            self.logger.experiment.log({"meta/heartbeat": 1.0})

        return loss

    def on_train_batch_start(self, batch, batch_idx):
        self._train_step_start = time.perf_counter()

    def on_fit_start(self):
        param_count = float(sum(p.numel() for p in self.solver.parameters()))
        train_size = None
        try:
            train_dl = self.trainer.train_dataloader
            train_size = float(len(train_dl.dataset)) if train_dl is not None else None
        except Exception:
            train_size = None

        payload = {"meta/param_count": param_count}
        if train_size is not None:
            payload["meta/train_dataset_size"] = train_size
        for logger in getattr(self.trainer, "loggers", []):
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                logger.experiment.log(payload)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._train_step_start is None:
            return
        step_time = time.perf_counter() - self._train_step_start
        self._train_step_start = None

        log_interval = max(1, self.trainer.log_every_n_steps)
        if (self.global_step + 1) % log_interval != 0:
            return

        batch_coords, _ = batch
        batch_size = batch_coords.shape[0]
        self.log("perf/step_time_sec", step_time, on_step=True, on_epoch=False)
        self.log(
            "perf/samples_per_sec",
            batch_size / max(step_time, 1e-12),
            on_step=True,
            on_epoch=False,
        )

        if torch.cuda.is_available():
            self.log(
                "perf/cuda_max_mem_alloc_mb",
                torch.cuda.max_memory_allocated() / (1024**2),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "perf/cuda_max_mem_reserved_mb",
                torch.cuda.max_memory_reserved() / (1024**2),
                on_step=True,
                on_epoch=False,
            )

        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0].get("lr")
            if lr is not None:
                self.log("optim/lr", lr, on_step=True, on_epoch=False)

    def on_after_backward(self):
        log_interval = max(1, self.trainer.log_every_n_steps)
        if (self.global_step + 1) % log_interval != 0:
            return

        total_norm_sq = 0.0
        for param in self.solver.parameters():
            if param.grad is None:
                continue
            param_norm = param.grad.data.norm(2)
            total_norm_sq += param_norm.item() ** 2
        total_norm = total_norm_sq**0.5
        self.log("optim/grad_norm", total_norm, on_step=True, on_epoch=False)

    def on_train_epoch_end(self):
        with torch.no_grad():
            total_norm_sq = 0.0
            for param in self.solver.parameters():
                param_norm = param.data.norm(2)
                total_norm_sq += param_norm.item() ** 2
            param_norm = total_norm_sq**0.5
        self.log("optim/param_norm", param_norm, on_step=False, on_epoch=True)
        self._maybe_update_weighted_sampling()

    def _maybe_update_weighted_sampling(self):
        data_cfg = getattr(self.config, "data", None)
        if data_cfg is None:
            return
        if not bool(getattr(data_cfg, "weighted_sampling_enabled", False)):
            return
        every = int(getattr(data_cfg, "weighted_sampling_every_n_epochs", 1))
        warmup = int(getattr(data_cfg, "weighted_sampling_warmup_epochs", 1))
        if (self.current_epoch + 1) <= warmup:
            return
        if every <= 0 or (self.current_epoch + 1) % every != 0:
            return

        train_loader = self.trainer.train_dataloader
        dataset = getattr(train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "update_coordinate_weights"):
            return

        min_weight = float(getattr(data_cfg, "weighted_sampling_min_weight", 0.1))
        max_weight = float(getattr(data_cfg, "weighted_sampling_max_weight", 0.9))
        batch_size = int(
            getattr(data_cfg, "weighted_sampling_batch_size", data_cfg.train_batch_size)
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        weight_scores = []
        was_training = self.solver.training
        self.solver.eval()
        for batch in loader:
            batch_coords, _ = batch
            batch_coords = batch_coords.to(self.device)
            with torch.enable_grad():
                _times, _grads, pred_vel, _norm_grad = self.solver.times_grad_vel(
                    batch_coords, reuse_grad=False, aux_vel=True
                )
            misfit = (pred_vel - 1.0).abs()
            weight_scores.append(misfit.reshape(-1).detach().cpu())
        if was_training:
            self.solver.train()

        if not weight_scores:
            return
        weight_scores = torch.cat(weight_scores, dim=0)
        dataset.update_coordinate_weights(weight_scores, min_weight, max_weight)
        dataset.reshuffle_pairs()

        sampler = getattr(train_loader, "sampler", None)
        if isinstance(sampler, WeightedRandomSampler):
            if sampler.weights.shape == dataset.sample_weights.shape:
                sampler.weights.copy_(dataset.sample_weights)
            else:
                sampler.weights = dataset.sample_weights

        self.log(
            "weighted_sampling/weight_mean",
            float(dataset.sample_weights.mean().item()),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "weighted_sampling/weight_min",
            float(dataset.sample_weights.min().item()),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "weighted_sampling/weight_p50",
            float(torch.quantile(dataset.sample_weights, 0.5).item()),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "weighted_sampling/weight_max",
            float(dataset.sample_weights.max().item()),
            on_step=False,
            on_epoch=True,
        )

    def validation_step(self, batch, batch_idx):
        batch_coords, _ = batch
        batch_coords = batch_coords.to(self.device)
        # Lightning runs validation under no_grad by default; enable grads for autograd.grad.
        with torch.enable_grad():
            outputs = self.solver.times_grad_vel(
                batch_coords, reuse_grad=False, aux_vel=True
            )

        times, grads, norm_grad, pred_vel = outputs

        # Calculate loss
        loss = self.eiko_loss(norm_grad)
        self.log("val_eiko", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Calculate metrics
        batch_mse = torch.mean((pred_vel - 1.0) ** 2, dim=(0, 1))

        self.log(
            "val_mse_step",
            batch_mse,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self._log_batch_stats("val", norm_grad, pred_vel)

        return loss

    def configure_optimizers(self):
        params = self.solver.parameters()
        config = self.config.optimizer

        optimizer_name = config.name.lower().capitalize()
        learning_rate = config.learning_rate
        beta1 = config.beta1
        beta2 = config.beta2
        warmup_steps = getattr(config, "warmup_steps", 0)
        decay_rate = getattr(config, "decay_rate", None)
        scheduler_type = getattr(config, "scheduler_type", "exponential")
        decay_steps = getattr(config, "decay_steps", warmup_steps)
        min_lr = getattr(config, "min_lr", 0.0)
        min_lr_at_pct = getattr(config, "min_lr_at_pct", None)
        warmup_pct = getattr(config, "warmup_pct", None)

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                params,
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=0.0,
            )

        elif optimizer_name == "Soap":
            try:
                from toy_experiments.PathFinding.modules.soap import SOAP
            except ImportError as exc:
                raise ImportError(
                    "Soap optimizer requested but not available. "
                    "Provide a SOAP optimizer implementation or install its package."
                ) from exc

            optimizer = SOAP(
                params,
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=0.0,
                precondition_frequency=2,
            )

        elif optimizer_name == "Muon":

            try:
                optimizer = torch.optim.Muon(
                    params,
                    lr=learning_rate,
                    weight_decay=0.0,
                    momentum=0.99,
                    ns_coefficients=(2, -1.5, 0.5),
                    ns_steps=10,
                )
            except AttributeError as exc:
                raise ImportError(
                    "Muon optimizer requested but torch.optim.Muon is not available."
                ) from exc

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler = None
        scheduler_active = config.use_lr_scheduler

        if scheduler_active:
            total_steps = None
            if self.trainer is not None:
                total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if total_steps is not None:
                if warmup_pct is not None:
                    warmup_steps = int(round(total_steps * float(warmup_pct) / 100.0))
                    warmup_steps = max(1, min(warmup_steps, int(total_steps)))
                if min_lr_at_pct is not None:
                    target_step = int(round(total_steps * float(min_lr_at_pct) / 100.0))
                    target_step = max(1, min(target_step, int(total_steps)))
                    decay_steps = max(1, target_step - int(warmup_steps))
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            )
            if scheduler_type.lower() == "cosine":
                decay_scheduler = CosineAnnealingLR(
                    optimizer, T_max=max(1, decay_steps), eta_min=min_lr
                )
            elif scheduler_type.lower() == "exponential":
                if min_lr_at_pct is not None and min_lr > 0:
                    gamma = (min_lr / learning_rate) ** (1.0 / max(1, decay_steps))
                else:
                    if decay_rate is None:
                        raise ValueError(
                            "optimizer.decay_rate is required for exponential scheduler "
                            "when optimizer.min_lr_at_pct is not set."
                        )
                    gamma = decay_rate ** (1.0 / max(1, decay_steps))
                decay_scheduler = ExponentialLR(optimizer, gamma=gamma)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps],
            )

        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
