import os
from functools import partial
from operator import itemgetter
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from rich.progress import track
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import data_loader.data_loaders as module_data
import model.model as module_arch
from configs import configs
from utils.tools import calculate_metric_and_confidence_interval
from utils.utils import read_json


def import_artifacts_from_runID(run_id, training=False):
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
        train_valid_split_size=0,
        batch_size=1,
        shuffle=False,
        training=training,
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
    sampler_str = config["data_loader"]["args"]["data_sampler"]
    imagings_str = "".join(config["data_loader"]["args"]["imagings_used"])
    imagings_str = "".join(list(filter(str.isdigit, imagings_str)))
    penalty_str = str(config["loss"]["args"]["l1_lambda"])
    name = sampler_str + "-i" + imagings_str + "-p" + penalty_str
    return name


def create_per_imaging_report(test_df, add_counts=False):
    f1_score_ = partial(f1_score, average="weighted")
    precision_score_ = partial(precision_score, average="weighted")
    recall_score_ = partial(recall_score, average="weighted")
    roc_auc_score_ = partial(roc_auc_score, average="weighted")

    msg = ""
    ST_SPACE = 20
    for name, df in test_df.groupby("imaging"):
        try:
            msg += f"{name:<{ST_SPACE}}"
            num_samples = "/".join(df['target'].value_counts().sort_index().astype("string").tolist())
            msg += f"{num_samples:<{ST_SPACE}}"
            mean, ci = calculate_metric_and_confidence_interval(
                df, roc_auc_score_, prediction_key="prediction_proba"
            )
            msg += f"{mean:.2f} ({ci[0]:.2f}, {ci[1]:.2f})    "
            mean, ci = calculate_metric_and_confidence_interval(df, f1_score_)
            msg += f"{mean:.2f} ({ci[0]:.2f}, {ci[1]:.2f})    "
            mean, ci = calculate_metric_and_confidence_interval(df, precision_score_)
            msg += f"{mean:.2f} ({ci[0]:.2f}, {ci[1]:.2f})   "
            mean, ci = calculate_metric_and_confidence_interval(df, recall_score_)
            msg += f"{mean:.2f} ({ci[0]:.2f}, {ci[1]:.2f})   \n"
        except ValueError:
            continue
    msg += f"{'Pooled':<{ST_SPACE}}"
    num_samples = "/".join(test_df['target'].value_counts().sort_index().astype("string").tolist())
    msg += f"{num_samples:<{ST_SPACE}}"
    mean, ci = calculate_metric_and_confidence_interval(
        test_df, roc_auc_score_, prediction_key="prediction_proba"
    )
    msg += f"{mean:.2f} ({ci[0]:.2f}, {ci[1]:.2f})    "
    mean, ci = calculate_metric_and_confidence_interval(test_df, f1_score_)
    msg += f"{mean:.2f} ({ci[0]:.2f}, {ci[1]:.2f})    "
    mean, ci = calculate_metric_and_confidence_interval(test_df, precision_score_)
    msg += f"{mean:.2f} ({ci[0]:.2f}, {ci[1]:.2f})   "
    mean, ci = calculate_metric_and_confidence_interval(test_df, recall_score_)
    msg += f"{mean:.2f} ({ci[0]:.2f}, {ci[1]:.2f})   "
    if add_counts:
        msg += test_df.astype("object").groupby("imaging").count().to_string()
    msg += "\n\n"
    return msg


def load_test_df(run_id, training=False):
    artifacts = import_artifacts_from_runID(run_id, training)
    model, data_loader, device, config = itemgetter("model", "data_loader", "device", "config")(
        artifacts
    )

    signatures_list = []
    predictions_test = []
    targets_test = []
    labels_test = []
    imagings_test = []
    relevances_list = []
    with torch.no_grad():
        for data, target, metadata in track(data_loader, description="Loading data..."):
            data = data.to(device)

            # signatures
            signature = torch.mean(data, dim=(2, 3))
            signatures_list.append(signature.detach().cpu().numpy().flatten())

            # metrics
            prediction = model(data)
            predictions_test.append(prediction.cpu().numpy())
            targets_test.append(target.numpy())
            labels_test.append(metadata["label"])
            imagings_test.append(metadata["imaging"])

            # relevances
            output = model.spectral.fc1(signature)
            output = model.spectral.act1(output)
            output = model.spectral.fc2(output)
            output = model.spectral.act2(output)
            relevances_list.append(output.detach().cpu().numpy().flatten())

    predictions_test = np.concatenate(predictions_test).flatten()
    targets_test = np.concatenate(targets_test).flatten()
    labels_test = np.concatenate(labels_test).flatten()
    imagings_test = np.concatenate(imagings_test).flatten()

    test_df = pd.DataFrame.from_dict(
        {
            "signature": signatures_list,
            "relevance": relevances_list,
            "imaging": imagings_test,
            "label": labels_test,
            "target": targets_test,
            "prediction": predictions_test.round().astype("int"),
            "prediction_proba": predictions_test,
        }
    )
    return test_df, config


def extract_info_from_absolute_path(filepath):
    indices, labels, imaging, _ = Path(filepath).stem.split("__")
    idx = int(indices.split("_")[1])
    labels = labels.split("_")
    identifier = labels[idx - 1]
    variety, class_ = identifier.split("-")[:2]
    class_map = {"K": "Control", "S": "Drought"}
    class_ = class_map[class_]
    return imaging, identifier, class_, variety


def create_dataframe_from_absolute_paths(filepaths):
    data = []
    for filepath in filepaths:
        imaging, identifier, class_, variety = extract_info_from_absolute_path(filepath)
        data.append([imaging, identifier, class_, variety])
    return pd.DataFrame(data, columns=["imaging", "identifier", "treatment", "variety"])
