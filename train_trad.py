import argparse
import collections
import glob

import ray
import sklearn.model_selection as model_selection_

import data_loader.data_loaders as data_loaders_
import models_trad.model as models_
import models_trad.optimizer as optimizers_
from utils.parse_config import ParseConfig
from utils.utils import read_json


def main(config):
    ray.init()
    logger = config.get_logger("train")

    # setup data_loader instances
    data_loader = config.init_obj("data_loader", data_loaders_)
    valid_data_loader = data_loader.split_validation()

    # initialize model and print to console
    model = config.init_obj("model", models_).create()

    # init validator and optimizer
    validator = config.init_obj("validation", model_selection_)
    Optimizer = config.import_module("optimizer", optimizers_)

    # log general info
    logger.info(model)
    logger.info(f"Total images for training: {len(data_loader.dataset)}"
                f" (train: {len(data_loader.sampler)}, valid: {len(data_loader.valid_sampler)})")

    # init optimizer, load data and do the optimization
    optimizer = Optimizer(
        model=model,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        validator=validator,
        config=config,
    )
    optimizer.load_data()
    optimizer.optimize()


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
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
    ]
    config = ParseConfig.from_args(args, options)
    main(config)
