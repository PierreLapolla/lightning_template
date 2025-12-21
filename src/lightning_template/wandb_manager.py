from functools import cache
from pathlib import Path

from lightning.pytorch.loggers import WandbLogger
from pedros.dependency_check import check_dependency

from lightning_template.settings import get_settings


class WandbManager:
    def __init__(self):
        self.settings = get_settings()
        self.enabled = check_dependency("wandb") and self.settings.wandb.use_wandb
        self.wandb = __import__("wandb") if self.enabled else None

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
                project=self.settings.wandb.project, entity=self.settings.wandb.entity
            )
        return None


@cache
def get_wandb() -> WandbManager:
    return WandbManager()
