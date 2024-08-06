import logging
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


class WandbConfig(BaseModel):
    entity: str = 'deldrel'
    project: str = 'lightning_template'
    sweep_config: str = 'src/sweep.yaml'


class DataModuleConfig(BaseModel):
    data_dir: str = 'data'
    batch_size: int = 64
    num_workers: int = 8
    use_max_workers: bool = True
    persistent_workers: bool = True
    training_set_ratio: float = 0.8


class ModelConfig(BaseModel):
    architecture: str = 'MNISTModel'
    learning_rate: float = 0.001
    loss_function: str = 'CrossEntropyLoss'
    optimizer: str = 'Adam'


class TrainerConfig(BaseModel):
    max_epochs: int = 1
    save_dir: str = 'logs/models'


class EarlyStoppingConfig(BaseModel):
    monitor: str = 'val_loss'
    min_delta: float = 0.001
    patience: int = 8
    mode: str = 'min'


class ReduceLROnPlateauConfig(BaseModel):
    monitor: str = 'val_loss'
    factor: float = 0.1
    patience: int = 4
    mode: str = 'min'


class Config(BaseModel):
    seed: int = 42
    logdir: str = 'logs'
    log_level: int = logging.INFO
    verbose: bool = True

    wandb: WandbConfig = WandbConfig()
    data_module: DataModuleConfig = DataModuleConfig()
    model: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    reduce_lr_on_plateau: ReduceLROnPlateauConfig = ReduceLROnPlateauConfig()

    def dump(self) -> Dict[str, Any]:
        return self.model_dump()

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, BaseModel):
                    current_value = current_value.model_copy(update=value)
                    setattr(self, key, current_value)
                else:
                    setattr(self, key, value)


def get_config() -> Config:
    try:
        load_dotenv(verbose=True)
        config = Config()
        return config
    except ValidationError as e:
        print(e)
        exit(1)


config: Config = get_config()
