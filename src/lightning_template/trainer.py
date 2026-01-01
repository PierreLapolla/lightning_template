from lightning.pytorch import Trainer
from pedros import has_dep

from lightning_template.settings import AppSettings
from lightning_template.wandb_manager import get_wandb


def get_trainer(settings: AppSettings) -> Trainer:
    wandb = get_wandb()

    callbacks = []
    if has_dep("rich"):
        from lightning.pytorch.callbacks import RichProgressBar

        callbacks.append(RichProgressBar())

    return Trainer(
        accelerator="cpu" if settings.train.force_cpu else "auto",
        max_epochs=settings.train.max_epochs,
        callbacks=callbacks,
        logger=wandb.get_logger(),
        fast_dev_run=settings.train.fast_dev_run,
    )
