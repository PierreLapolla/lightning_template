import importlib.util
from functools import cache
from pathlib import Path
from typing import Any

from pedros.logger import get_logger
from pydantic import BaseModel, PositiveFloat, PositiveInt, computed_field
from pydantic_settings import BaseSettings


class EnvInfo(BaseModel):
    @computed_field
    @property
    def has_wandb(self) -> bool:
        return importlib.util.find_spec("wandb") is not None

    @computed_field
    @property
    def has_rich(self) -> bool:
        return importlib.util.find_spec("rich") is not None


class DataCfg(BaseModel):
    batch_size: PositiveInt = 64
    num_workers: PositiveInt = 7


class TrainCfg(BaseModel):
    force_cpu: bool = False
    learning_rate: PositiveFloat = 0.0001
    max_epochs: PositiveInt = 20
    fast_dev_run: bool = False


class WandbCfg(BaseModel):
    root_path: Path = Path(__file__).resolve().parents[2]

    use_wandb: bool = True
    project: str = "lightning_template"
    entity: str = "deldrel"


class AppSettings(BaseSettings):
    seed: int = 2002

    env: EnvInfo = EnvInfo()
    data: DataCfg = DataCfg()
    train: TrainCfg = TrainCfg()
    wandb: WandbCfg = WandbCfg()

    def __init__(self, **values: Any):
        super().__init__(**values)
        logger = get_logger()
        if self.wandb.use_wandb and not self.env.has_wandb:
            logger.warning(
                "Wandb is enabled in settings but wandb package is not installed. If you want to use it, make sure to add it to the environment with `pip install wandb`."
            )


@cache
def get_settings() -> AppSettings:
    return AppSettings()
