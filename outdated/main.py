from lightning.pytorch import seed_everything

from src.config import config
from src.helpers.decorators import timer
from src.helpers.menu import Menu
from src.model_training.lightning_manager import lightning_manager


@timer
def main() -> None:
    seed_everything(config.seed, workers=True)
    Menu(
        {
            "1": ("Train model", lightning_manager.train_model),
            "2": ("Start sweep", lightning_manager.start_sweep),
        }
    ).start(timeout=30)


if __name__ == "__main__":
    main()
