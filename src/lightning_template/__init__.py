import logging

from pedros.logger import setup_logging

setup_logging()

logging.getLogger("lightning").handlers.clear()
logging.getLogger("lightning").propagate = True

logging.getLogger("wandb").handlers.clear()
logging.getLogger("wandb").propagate = True
