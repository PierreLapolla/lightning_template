from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path


def get_trainer() -> Trainer:
    logger = WandbLogger(
        project="template_project",
        entity="deldrel",
        save_dir=str(Path(__file__).parent.parent.parent / "logs"),
    )

    return Trainer(
        max_epochs=20,
        callbacks=[],
        logger=logger,
    )
