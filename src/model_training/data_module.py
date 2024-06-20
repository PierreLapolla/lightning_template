from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from src.config import config


class DataModule(LightningDataModule):
    def __init__(self) -> None:
        """
        This class is responsible for preparing the datasets and setting up the data loaders for the model.
        """
        super(DataModule, self).__init__()

    def prepare_data(self) -> None:
        MNIST(config.data_module.data_dir, train=True, download=True)
        MNIST(config.data_module.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(config.data_module.data_dir, train=True, transform=ToTensor())
            self.train_set, self.val_set = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.test_set = MNIST(config.data_module.data_dir, train=False, transform=ToTensor())

        if stage == 'predict' or stage is None:
            self.predict_set = MNIST(config.data_module.data_dir, train=False, transform=ToTensor())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set,
                          batch_size=config.data_module.batch_size,
                          num_workers=config.data_module.num_workers,
                          persistent_workers=config.data_module.persistent_workers,
                          shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set,
                          batch_size=config.data_module.batch_size,
                          num_workers=config.data_module.num_workers,
                          persistent_workers=config.data_module.persistent_workers,
                          shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set,
                          batch_size=config.data_module.batch_size,
                          num_workers=config.data_module.num_workers,
                          persistent_workers=config.data_module.persistent_workers,
                          shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_set,
                          batch_size=config.data_module.batch_size,
                          num_workers=config.data_module.num_workers,
                          persistent_workers=config.data_module.persistent_workers,
                          shuffle=False)
