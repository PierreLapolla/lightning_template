import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from outdated.src.config import config


def get_trainer() -> Trainer:
    early_stopping_callback = EarlyStopping(
        monitor=config.early_stopping.monitor,
        min_delta=config.early_stopping.min_delta,
        patience=config.early_stopping.patience,
        verbose=config.verbose,
        mode=config.early_stopping.mode,
    )

    wandb.finish()
    logger = WandbLogger(
        project=config.wandb.project, entity=config.wandb.entity, save_dir=config.logdir
    )

    return Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=[
            early_stopping_callback,
        ],
        logger=logger,
    )
