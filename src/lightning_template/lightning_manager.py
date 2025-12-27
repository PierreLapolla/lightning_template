from lightning import seed_everything
from pedros.logger import get_logger

from lightning_template.data_module import DataModule
from lightning_template.mnist_model import MNISTModel
from lightning_template.settings import get_settings
from lightning_template.trainer import get_trainer
from lightning_template.wandb_manager import get_wandb


class LightningManager:
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.wandb = get_wandb()

        self.wandb.login()

    def start_training(self) -> None:
        run = self.wandb.init()
        if run:
            self.settings.update_from_wandb(dict(run.config))

        seed_everything(self.settings.seed, workers=True)

        data_module = DataModule(self.settings)
        model = MNISTModel(self.settings)
        trainer = get_trainer(self.settings)

        trainer.fit(model, data_module)
        trainer.test(model, data_module)
        self.wandb.finish()

    def start_sweep(self) -> None:
        self.wandb.run_sweep(self.start_training)
