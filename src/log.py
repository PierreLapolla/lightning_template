import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import lightning

from src.config import config


class LogConfig:
    def __init__(self) -> None:
        self.log_dir = Path(config.logdir)
        self.log_file = self.log_dir / 'logs.log'

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._configure_logging()

    def _configure_logging(self) -> None:
        class CustomFormatter(logging.Formatter):
            grey = "\x1b[38;20m"
            green = "\x1b[32m"
            yellow = "\x1b[33m"
            red = "\x1b[31m"
            bold_red = "\x1b[31;1m"
            reset = "\x1b[0m"
            FORMATS = {
                logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                logging.INFO: green + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
            }

            def format(self, record):
                log_fmt = self.FORMATS.get(record.levelno)
                formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
                return formatter.format(record)

        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(config.log_level)

        file_handler = RotatingFileHandler(self.log_file, maxBytes=10 ** 6)
        file_handler.setLevel(config.log_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                                    datefmt='%Y-%m-%d %H:%M:%S'))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(config.log_level)
        console_handler.setFormatter(CustomFormatter())

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
