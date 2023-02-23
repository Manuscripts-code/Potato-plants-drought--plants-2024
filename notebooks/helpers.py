import os
from pathlib import Path

import mlflow
import torch

import data_loader.data_loaders as module_data
import model.model as module_arch
from configs import configs
from utils.utils import read_json


def import_artifacts_from_runID(run_id):
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_base_path = Path(configs.MODEL_REGISTRY, experiment_id, run_id, "artifacts")
    config = read_json(artifacts_base_path / "configs/config.json")
    checkpoint = torch.load(artifacts_base_path / "checkpoints/model_best.pth")

    state_dict = checkpoint["state_dict"]
    model = getattr(module_arch, config["arch"]["type"])(**config["arch"]["args"])
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    data_loader = getattr(module_data, config["data_loader"]["type"])(
        configs.DATA_DIR,
        config["data_loader"]["args"]["imagings_used"],
        config["data_loader"]["args"]["dataset"],
        config["data_loader"]["args"]["data_sampler"],
        train_test_split_size=config["data_loader"]["args"]["train_test_split_size"],
        train_valid_split_size=config["data_loader"]["args"]["train_valid_split_size"],
        batch_size=1,
        shuffle=False,
        training=False,
    )

    artifacts = {"model": model, "data_loader": data_loader, "device": device, "config": config}
    return artifacts


def load_ids_from_registry():
    # get experiments ids-es and remove .trash
    experiments_ids = [f for f in os.listdir(configs.MODEL_REGISTRY) if not f.startswith(".")]
    # write data dict where key represent experiment by name and value correspond to runs under that experiment
    data_all = {
        mlflow.get_experiment(exp_id).name: mlflow.search_runs(exp_id) for exp_id in experiments_ids
    }
    # extract train ids
    run_ids = data_all["train_CNN"]["run_id"].tolist()
    return run_ids


def get_plot_name(config):
    imagings_str = "".join(config["data_loader"]["args"]["imagings_used"])
    imagings_str = "".join(list(filter(str.isdigit, imagings_str)))
    sampler_str = config["data_loader"]["args"]["data_sampler"]
    name = sampler_str + imagings_str
    return name
