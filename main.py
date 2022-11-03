import argparse
import collections

import scripts as scripts_
from utils.parse_config import ParseConfig


def main(config):
    logger = config.get_logger("main")
    try:
        script = getattr(scripts_, config.mode)
    except Exception:
        logger.error(f"Script >{config.mode}< not found.")

    script(config)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train traditional model")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-m",
        "--mode",
        default=None,
        type=str,
        help="mode in which to run: train/test (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["-vs", "--valid_split"], type=float, target="data_loader;args;train_valid_split_size"
        ),
    ]
    config = ParseConfig.from_args(args, options)
    main(config)
