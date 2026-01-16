import torch
import pytorch_lightning as pl
from torch import nn
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR

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
        loss = self.eiko_loss(norm_grad)
        self.log("train_eiko", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Calculate metrics
        batch_mse = self.mse(pred_vel, torch.ones_like(pred_vel))

        self.log(
            "train_mse_step",
            batch_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

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
        batch_mse = self.mse(pred_vel, torch.ones_like(pred_vel))

        self.log(
            "val_mse_step",
            batch_mse,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        params = self.solver.parameters()
        config = self.config.optimizer

        optimizer_name = config.name.lower().capitalize()
        learning_rate = config.learning_rate
        beta1 = config.beta1
        beta2 = config.beta2
        warmup_steps = config.warmup_steps
        decay_rate = config.decay_rate

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
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            )
            decay_scheduler = ExponentialLR(optimizer, gamma=decay_rate)
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
