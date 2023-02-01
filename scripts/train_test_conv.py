import copy
import time as time
from pathlib import Path

import mlflow

from configs import configs

from . import test_conv, train_conv


def train_test_conv(config_orig):
    logger = config_orig.get_logger("evaluate")
    config = copy.deepcopy(config_orig)

    try:
        run_id = train_conv(config)
        experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
        config.resume = Path(configs.MODEL_REGISTRY, experiment_id, run_id)
        test_conv(config)

    except Exception as e:
        logger.error(f"Exception during training: {e}")
