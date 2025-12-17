from lightning_template.data_module import DataModule
from lightning_template.mnist_model import MNISTModel
from lightning_template.trainer import get_trainer


class LightningManager:
    def __init__(self):
        self.data_module = DataModule()
        self.model = MNISTModel()
        self.trainer = get_trainer()

    def start_training(self) -> None:
        self.trainer.fit(self.model, self.data_module)
        self.trainer.test(self.model, self.data_module)
