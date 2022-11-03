import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.progress import track
from sklearn.metrics import classification_report
from tqdm import tqdm

import data_loader.data_loaders as module_data
import models_conv.loss as module_loss
import models_conv.metric as module_metric
import models_conv.model as module_arch
from utils.parse_config import ParseConfig


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

    # build model architecture
    model = config.init_obj("arch", module_arch)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume / "model_best.pth")
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
        for i, (data, target) in enumerate(track(data_loader, description="Loading data...")):
            # replace nans with zeros
            data = torch.nan_to_num(data)
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
    logger.info(log)
    logger.info(
        f"Classification report on test data:\n"
        f"{classification_report(np.array(target_list).flatten(), np.array(pred_list).flatten().round())}"
    )


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
    main(config)
