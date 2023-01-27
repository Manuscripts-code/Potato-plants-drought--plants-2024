import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import ray
import sklearn.model_selection as model_selection_
from sklearn.metrics import classification_report, confusion_matrix

import data_loader.data_loaders as data_loaders_
import data_loader.data_loaders as module_data
import models_trad.model as models_
import models_trad.optimizer as optimizers_
from models_trad.helpers import convert_images_to_1d
from utils.parse_config import ParseConfig
from utils.tools import calculate_classification_metrics
from utils.utils import ensure_dir, read_pickle, write_json, write_txt


def test_trad(config):
    logger = config.get_logger("test")
    logger.info("Loading checkpoint: {} ...".format(config.resume))

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        config["data_loader"]["args"]["dataset"],
        config["data_loader"]["args"]["data_sampler"],
        train_test_split_size=config["data_loader"]["args"]["train_test_split_size"],
        train_valid_split_size=0.0,
        batch_size=1,
        shuffle=False,
        training=False,
        num_workers=2,
    )
    # convert to signatures
    X_test, y_test = convert_images_to_1d(data_loader)

    # load model from registry
    model = read_pickle(config.resume / "artifacts/model/model.pkl")
    y_pred = model.predict(X_test)
    clf_report = classification_report(y_test, y_pred)

    logger.info(f"Model: {model}")
    logger.info(f"Total images for testing: {len(data_loader.dataset)}")
    logger.info(f"Classification report on test data:\n{clf_report}")

    mlflow.set_experiment(experiment_name=f"test_{config.exper_name}")
    with mlflow.start_run(run_name=config.run_id):
        performance = calculate_classification_metrics(y_test, y_pred)
        mlflow.log_metrics({"precision_avg": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall_avg": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1_avg": performance["overall"]["f1"]})

        with tempfile.TemporaryDirectory() as dp:
            ensure_dir(Path(dp) / "results")
            write_json(performance, Path(dp, "results/performance.json"))
            write_txt(clf_report, Path(dp, "results/classification_report.txt"))
            mlflow.log_artifacts(dp)


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
    test_trad(config)
