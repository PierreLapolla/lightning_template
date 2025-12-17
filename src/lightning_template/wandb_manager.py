from functools import cache
from pathlib import Path

from lightning.pytorch.loggers import WandbLogger

from lightning_template.settings import get_settings


class WandbManager:
    def __init__(self):
        self.settings = get_settings()
        try:
            import wandb
            self.wandb = wandb if self.settings.wandb.use_wandb else None
        except ImportError:
            self.wandb = None

    def login(self):
        if self.wandb:
            self.wandb.login()

    def init(self):
        if self.wandb:
            self.wandb.init()

    def save(self, path: Path):
        if self.wandb:
            self.wandb.save(path, base_path=self.settings.wandb.root_path)

    def get_logger(self) -> WandbLogger | None:
        if self.wandb:
            return WandbLogger(
                project=self.settings.wandb.project,
                entity=self.settings.wandb.entity
            )
        return None


@cache
def get_wandb() -> WandbManager:
    return WandbManager()
