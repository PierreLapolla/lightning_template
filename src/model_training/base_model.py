from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb
from lightning import LightningModule
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import config


class BaseModel(LightningModule):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.loss_func = getattr(nn, config.model.loss_function)()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer_class = getattr(optim, config.model.optimizer)
        optimizer = optimizer_class(self.parameters(),
                                    lr=config.model.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode=config.reduce_lr_on_plateau.mode,
                                      factor=config.reduce_lr_on_plateau.factor,
                                      patience=config.reduce_lr_on_plateau.patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': config.reduce_lr_on_plateau.monitor,
                'strict': True,
            }
        }

    def training_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def on_train_epoch_start(self) -> None:
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_end(self) -> None:
        path = Path(config.trainer.save_dir)
        path.mkdir(parents=True, exist_ok=True)
        model_name = f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        torch.save(self.state_dict(), path / model_name)
        wandb.save(str(path / model_name))
