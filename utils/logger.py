import logging
import logging.config
from pathlib import Path

from rich.logging import RichHandler

from utils.utils import read_json


def setup_logging(save_dir, log_config="configs/logger.json", default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)


def get_logging(name, verbosity=2):
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
        verbosity, log_levels.keys()
    )
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger()
    logger.handlers[0] = RichHandler(markup=True)  # set rich handler
    logger.name = name
    logger.setLevel(log_levels[verbosity])
    return logger
