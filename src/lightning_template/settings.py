from functools import cache
from pathlib import Path
from typing import Any

from pedros import has_dep, get_logger
from pydantic import BaseModel, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings


class DataCfg(BaseModel):
    batch_size: PositiveInt = 64
    num_workers: PositiveInt = 7


class TrainCfg(BaseModel):
    force_cpu: bool = False
    learning_rate: PositiveFloat = 0.001
    max_epochs: PositiveInt = 20
    fast_dev_run: bool = False


class WandbCfg(BaseModel):
    root_path: Path = Path(__file__).resolve().parents[2]

    use_wandb: bool = True
    project: str = "lightning_template"
    entity: str = "deldrel"

    sweep_config: dict[str, Any] = {
        "method": "random",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "train.learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
            "data.batch_size": {"values": [32, 64, 128]},
            "seed": {"values": [7, 42, 123]},
        },
    }
    sweep_count: int = 10


class AppSettings(BaseSettings):
    seed: int = 2002

    data: DataCfg = DataCfg()
    train: TrainCfg = TrainCfg()
    wandb: WandbCfg = WandbCfg()

    def __init__(self, **values: Any):
        super().__init__(**values)
        logger = get_logger()
        if self.wandb.use_wandb and not has_dep("wandb"):
            logger.warning(
                "Wandb is enabled in settings but wandb package is not installed. If you want to use it, make sure to add it to the environment with `pip install wandb`."
            )

    def update_from_wandb(self, config: dict[str, Any]):
        """Updates settings from a wandb config dictionary, handling nested keys."""
        for key, value in config.items():
            if "." in key:
                parts = key.split(".")
                obj = self
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        break
                else:
                    if hasattr(obj, parts[-1]):
                        setattr(obj, parts[-1], value)
            elif hasattr(self, key):
                setattr(self, key, value)


@cache
def get_settings() -> AppSettings:
    return AppSettings()
