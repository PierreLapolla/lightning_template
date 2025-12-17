from functools import cache
from pathlib import Path

from pydantic import BaseModel, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings


class DataCfg(BaseModel):
    batch_size: PositiveInt = 64
    num_workers: PositiveInt = 7


class TrainCfg(BaseModel):
    learning_rate: PositiveFloat = 0.001
    max_epochs: PositiveInt = 10
    fast_dev_run: bool = False


class WandbCfg(BaseModel):
    root_path: Path = Path(__file__).resolve().parents[2]

    use_wandb: bool = False
    project: str = "template_project"
    entity: str = "deldrel"


class AppSettings(BaseSettings):
    seed: int = 2002

    data: DataCfg = DataCfg()
    train: TrainCfg = TrainCfg()
    wandb: WandbCfg = WandbCfg()


@cache
def get_settings() -> AppSettings:
    return AppSettings()
