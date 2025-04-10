import wandb
from lightning import seed_everything

from lightning_template.config import config
from lightning_template.lightning_manager import LightningManager
from utils.logger import log
from utils.timer import timer


@timer
def main():
    log.debug(config.model_dump())
    wandb.login(key=config.WANDB_API_KEY)
    seed_everything(config.seed, workers=True)

    lightning_manager = LightningManager()
    lightning_manager.start_training()


if __name__ == "__main__":
    main()
