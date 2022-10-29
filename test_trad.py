import argparse

import ray
import sklearn.model_selection as model_selection_
from sklearn.metrics import classification_report, confusion_matrix
from sympy import proper_divisor_count

import data_loader.data_loaders as data_loaders_
import data_loader.data_loaders as module_data
import models_trad.model as models_
import models_trad.optimizer as optimizers_
from models_trad.helpers import convert_images_to_1d
from utils.parse_config import ParseConfig
from utils.utils import read_pickle


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        config["data_loader"]["args"]["dataset"],
        config["data_loader"]["args"]["data_sampler"],
        config["data_loader"]["args"]["grouped_labels_filepath"],
        train_test_split_size=config["data_loader"]["args"]["train_test_split_size"],
        train_valid_split_size=0.0,
        batch_size=1,
        shuffle=False,
        training=False,
        num_workers=2,
    )
    # convert to signatures
    X_test, y_test = convert_images_to_1d(data_loader)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    model = read_pickle(config.resume / "model.pkl")
    logger.info(model)

    y_pred = model.predict(X_test)
    logger.info(f"Classification report on test data:\n{classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Test traditional model")
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

    config = ParseConfig.from_args(args)
    main(config)
