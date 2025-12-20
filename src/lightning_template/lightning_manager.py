from lightning_template.data_module import DataModule
from lightning_template.mnist_model import MNISTModel
from lightning_template.trainer import get_trainer
from pedros.logger import get_logger


class LightningManager:
    def __init__(self):
        self.logger = get_logger()
        self.data_module = DataModule()
        self.model = MNISTModel()
        self.trainer = get_trainer()

    def start_training(self) -> None:
        self.logger.info("Starting training...")
        self.trainer.fit(self.model, self.data_module)
        self.trainer.test(self.model, self.data_module)
        self.logger.info("Training finished.")
