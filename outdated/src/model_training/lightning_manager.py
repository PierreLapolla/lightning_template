import logging
from functools import lru_cache

import wandb
import yaml

from outdated.src.config import config
from outdated.src.helpers.decorators import timer
from .data_module import DataModule
from .mnistmodel import MNISTModel
from .trainer import get_trainer


class LightningManager:
    def __init__(self):
        self.data_module = None
        self.model = None
        self.trainer = None

    def setup(self) -> None:
        self.data_module = DataModule()

        if config.model.architecture == "MNISTModel":
            self.model = MNISTModel()
        else:
            message = f"Unknown architecture: {config.model.architecture}"
            logging.error(message)
            raise ValueError(message)

        self.trainer = get_trainer()

    @timer
    def train_model(self) -> None:
        self.setup()

        try:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                dir=config.logdir,
                config=config.dump(),
            )

            logging.info(
                "You can interrupt the training whenever you want with a keyboard interrupt (CTRL+C)"
            )
            self.trainer.fit(self.model, self.data_module)
            self.trainer.test(self.model, self.data_module)

        except Exception as e:
            if config.verbose:
                message = f"Error training model: {e}"
                logging.error(message)

        finally:
            wandb.finish()

    def sweep_train(self):
        wandb.init(dir=config.logdir, config=config.dump())
        config.update_from_dict(wandb.config)
        self.setup()
        self.trainer.fit(self.model, self.data_module)
        wandb.finish()

    @timer
    def start_sweep(self) -> None:
        with open(config.wandb.sweep_config, "r") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(
            sweep_config, project=config.wandb.project, entity=config.wandb.entity
        )
        wandb.agent(sweep_id, function=self.sweep_train)


@lru_cache(maxsize=1)
def get_lightning_manager() -> LightningManager:
    return LightningManager()


lightning_manager = get_lightning_manager()
