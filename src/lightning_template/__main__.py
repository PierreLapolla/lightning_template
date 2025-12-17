from lightning import seed_everything

from lightning_template.lightning_manager import LightningManager
from lightning_template.settings import get_settings
from lightning_template.wandb_manager import get_wandb


def main():
    settings = get_settings()
    wandb = get_wandb()

    wandb.login()
    seed_everything(settings.seed, workers=True)

    lightning_manager = LightningManager()
    lightning_manager.start_training()


if __name__ == "__main__":
    main()
