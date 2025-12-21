from lightning.pytorch import Trainer
from pedros.dependency_check import check_dependency

from lightning_template.settings import get_settings
from lightning_template.wandb_manager import get_wandb


def get_trainer() -> Trainer:
    settings = get_settings()
    wandb = get_wandb()

    callbacks = []
    if check_dependency("rich"):
        from lightning.pytorch.callbacks import RichProgressBar

        callbacks.append(RichProgressBar())

    return Trainer(
        accelerator="cpu" if settings.train.force_cpu else "auto",
        max_epochs=settings.train.max_epochs,
        callbacks=callbacks,
        logger=wandb.get_logger(),
        fast_dev_run=settings.train.fast_dev_run,
    )
