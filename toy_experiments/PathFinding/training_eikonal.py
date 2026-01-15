import omegaconf
import argparse
from pathlib import Path
from hydra import initialize, compose


import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from toy_experiments.PathFinding.datasets import get_dataloaders
from toy_experiments.PathFinding.models.eikonal_solvers import get_solver
from toy_experiments.PathFinding.modules.eikonal_module import (
    EikonalLightningModule,
)


import argparse
from pathlib import Path

CHECKPOINT_PATH = Path("checkpoints").mkdir(parents=True, exist_ok=True)


def train(config_name):
    with initialize(config_path="configs"):
        cfg = compose(config_name=config_name)

    print(cfg)

    # Set device, seed and create log directory
    cfg.device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    pl.seed_everything(cfg.seed)

    # Create the dataset
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # Init model
    solver = get_solver(cfg)

    # Assuming `model` and `train_dataset` are defined and `config` is your configuration object
    lightning_model = EikonalLightningModule(config=cfg, solver=solver)
    aux = f"{cfg.geometry.dim_signal}d_signal_{cfg.solver.backbone_type}"
    # Setup logger
    logger = WandbLogger(
        name="fitting_" + aux,
        save_dir=cfg.logging.log_dir,
        project="SpaceTimeEikonal",
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
    )

    callbacks = []
    if cfg.training.model_checkpoint:
        save_callback = ModelCheckpoint(
            filename=aux, save_weights_only=True, mode="min", monitor="val_eiko"
        )
        callbacks.append(save_callback)

    if cfg.training.early_stopping:
        # Define early stopping criterion when the validation loss does not decrease for 5 epochs.
        early_stop_callback = EarlyStopping(
            monitor="val_eiko",
            min_delta=0.00000,
            patience=15,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stop_callback)

    if True:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    # Initialize a pytorch-lightning trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.training.num_epochs,
        accelerator=cfg.device,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        enable_progress_bar=False if cfg.logging.no_progress_bar else True,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        check_val_every_n_epoch=cfg.test.val_every_n_epochs,
        enable_checkpointing=True if cfg.training.model_checkpoint else False,
        num_sanity_val_steps=0,
        gradient_clip_val=cfg.training.gradient_clip_val,  # Clipping value
        gradient_clip_algorithm="norm",
    )

    # Train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Config file to load (without .yaml)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional overrides (e.g., database.port=1234)",
    )
    args = parser.parse_args()

    train(args.experiment)
