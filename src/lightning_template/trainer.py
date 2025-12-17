from lightning.pytorch import Trainer

from lightning_template.settings import get_settings
from lightning_template.wandb_manager import get_wandb


def get_trainer() -> Trainer:
    settings = get_settings()
    wandb = get_wandb()

    callbacks = []
    try:
        from lightning.pytorch.callbacks import RichProgressBar
        callbacks.append(RichProgressBar())
    except ImportError:
        pass

    return Trainer(
        accelerator="cpu",
        max_epochs=settings.train.max_epochs,
        callbacks=callbacks,
        logger=wandb.get_logger(),
        fast_dev_run=settings.train.fast_dev_run,
    )
