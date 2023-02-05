import argparse
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from rich.progress import track
from sklearn.metrics import classification_report

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from configs import configs
from utils.parse_config import ParseConfig
from utils.tools import calculate_classification_metrics
from utils.utils import ensure_dir, write_json, write_txt


def test(config):
    logger = config.get_logger("test")
    logger.info("Loading checkpoint: {} ...".format(config.resume))

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        configs.DATA_DIR,
        config["data_loader"]["args"]["dataset"],
        config["data_loader"]["args"]["data_sampler"],
        train_test_split_size=config["data_loader"]["args"]["train_test_split_size"],
        train_valid_split_size=0.0,
        batch_size=1,
        shuffle=False,
        training=False,
        num_workers=2,
    )

    # build model architecture
    model = config.init_obj("arch", module_arch)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    checkpoint = torch.load(config.resume / "artifacts/checkpoints" / "model_best.pth")
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # log general info
    logger.info(model)
    logger.info(f"Device: {device}")
    logger.info(f"Total images for testing: {len(data_loader.dataset)}")

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)

    target_list = []
    pred_list = []
    with torch.no_grad():
        for i, (data, target, _) in enumerate(track(data_loader, description="Loading data...")):
            # convert to column vector
            target = target.view(len(target), -1).float()
            data, target = data.to(device), target.to(device)

            pred_class = model(data)
            loss = loss_fn(pred_class, target)
            target_list.append(target.cpu().numpy())
            pred_list.append(pred_class.cpu().numpy())

            # computing loss, metrics on test set
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(pred_class, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})

    y_test, y_pred = np.array(target_list).flatten(), np.array(pred_list).flatten().round()
    clf_report = classification_report(y_test, y_pred)
    logger.info(log)
    logger.info(f"Classification report on test data:\n{clf_report}")

    mlflow.set_experiment(experiment_name=f"test_{config.exper_name}")
    with mlflow.start_run(run_name=f"{config.run_id}__test"):
        performance = calculate_classification_metrics(y_test, y_pred)
        mlflow.log_metrics({"precision_avg": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall_avg": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1_avg": performance["overall"]["f1"]})

        with tempfile.TemporaryDirectory() as dp:
            ensure_dir(Path(dp) / "results")
            ensure_dir(Path(dp) / "configs")
            write_json(performance, Path(dp, "results/performance.json"))
            write_txt(clf_report, Path(dp, "results/classification_report.txt"))
            write_json(config.config, Path(dp, "configs/config.json"))
            write_txt(config.resume.name, Path(dp, "configs/train_runID.txt"))
            mlflow.log_artifacts(dp)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Test convolutional model")
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
    test_conv(config)
