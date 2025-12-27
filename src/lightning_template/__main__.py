import sys

from lightning_template.lightning_manager import LightningManager


def main():
    lightning_manager = LightningManager()
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        lightning_manager.start_sweep()
    else:
        lightning_manager.start_training()


if __name__ == "__main__":
    main()
