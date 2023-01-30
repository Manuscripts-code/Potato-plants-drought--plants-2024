import argparse
import collections

import mlflow
import numpy as np
import torch

import data_loader.data_loaders as module_data
import models_conv.loss as module_loss
import models_conv.metric as module_metric
import models_conv.model as module_arch
from configs import configs
from models_conv.helpers import prepare_device
from models_conv.trainer import Trainer
from utils.parse_config import ParseConfig

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train_conv(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    data_loader = config.init_obj("data_loader", module_data, data_dir=configs.DATA_DIR)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    # log general info
    logger.info(model)
    logger.info(f"Device: {device}")
    logger.info(
        f"Total images for training: {len(data_loader.dataset)}"
        f" (train: {len(data_loader.sampler)}, "
        f"valid: {len(data_loader.valid_sampler) if data_loader.valid_sampler else 0})"
    )

    # initialize trainer and train
    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    # setup mlflow experiment name
    mlflow.set_experiment(experiment_name=f"train_{config.exper_name}")
    with mlflow.start_run(run_name=f"{config.run_id}__train"):
        trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train convolutional model")
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
    train_conv(config)
