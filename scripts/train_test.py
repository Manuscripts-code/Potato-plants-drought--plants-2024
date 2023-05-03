import copy
import time as time
from pathlib import Path

import mlflow

from configs import configs
from notebooks.produce_results import produce_results

from . import test, train


def train_test(config_orig):
    logger = config_orig.get_logger("evaluate")
    config = copy.deepcopy(config_orig)

    try:
        run_id = train(config)
        experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
        config.resume = Path(configs.MODEL_REGISTRY, experiment_id, run_id)
        test(config)
        produce_results(run_id)

    except Exception as e:
        logger.error(f"Exception during training: {e}")
