from functools import lru_cache
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


class WandbConfig(BaseModel):
    entity: str = 'deldrel'
    project: str = 'lightning_template'
    sweep_config: str = 'src/sweep.yaml'


class DataModuleConfig(BaseModel):
    data_dir: str = 'data'
    batch_size: int = 32
    num_workers: int = 8
    persistent_workers: bool = True
    training_set_ratio: float = 0.8


class ModelConfig(BaseModel):
    architecture: str = 'MLP'
    learning_rate: float = 0.01
    loss_function: str = 'CrossEntropyLoss'
    optimizer: str = 'Adam'


class TrainerConfig(BaseModel):
    max_epochs: int = 1
    save_dir: str = 'logs/models'


class CheckpointConfig(BaseModel):
    dirpath: str = 'logs/checkpoints'
    filename: str = 'checkpoint-{epoch:02d}-{val_loss:.6f}'
    monitor: str = 'val_loss'
    save_last: bool = True
    save_top_k: int = 1
    mode: str = 'min'


class EarlyStoppingConfig(BaseModel):
    monitor: str = 'val_loss'
    min_delta: float = 0.001
    patience: int = 10
    mode: str = 'min'


class ReduceLROnPlateauConfig(BaseModel):
    monitor: str = 'val_loss'
    factor: float = 0.1
    patience: int = 5
    mode: str = 'min'


class Config(BaseModel):
    seed: int = 42
    logdir: str = 'logs'
    verbose: bool = True

    wandb: WandbConfig = WandbConfig()
    data_module: DataModuleConfig = DataModuleConfig()
    model: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    reduce_lr_on_plateau: ReduceLROnPlateauConfig = ReduceLROnPlateauConfig()

    def dump(self) -> Dict[str, Any]:
        return self.model_dump()

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, BaseModel):
                    current_value = current_value.copy(update=value)
                    setattr(self, key, current_value)
                else:
                    setattr(self, key, value)


@lru_cache(maxsize=1)
def get_config() -> Config:
    try:
        load_dotenv(verbose=True)
        return Config()
    except ValidationError as e:
        print(e)
        exit(1)


config: Config = get_config()
