import copy
import json
import os
import tempfile
from abc import abstractmethod
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from ray import tune
from ray.air import session
from ray.air.callbacks.mlflow import MLflowLoggerCallback
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.base import clone

from configs import configs
from utils.tools import calculate_classification_metrics
from utils.utils import ensure_dir, write_json, write_txt

from .helpers import convert_images_to_1d
from .scorer import make_scorer_ftn, objective_cv, objective_split


class BaseOptimizer:
    def __init__(self, model, config):
        self.config = copy.deepcopy(config.config)
        self.logger = config.get_logger("trainer", verbosity=2)
        self.model = model
        self.scoring_metric = config["scoring_metric"]
        self.scoring_mode = config["scoring_mode"]
        self.tuned_params = self._modify_params(config)
        self.debug = config["debug"]
        self.num_samples = config["num_samples"]
        self.name = config["name"]
        self.save_dir = config.save_dir
        self.exper_name = config.exper_name
        self.data = None

    def optimize(self):
        if self.data is None:
            raise Exception("use self.load_data() before self.optimize()")
        if self.debug:
            print("Not implemented.")
        else:
            self._debug_false()

    def _debug_false(self):
        results, best_results = self.perform_search()
        X_data, y_data = self._pool_data()
        self._refit_model(best_results.config, X_data, y_data)
        y_pred = self._predict_pooled_data(X_data)
        self.log_session(results, best_results, y_data, y_pred)

    def _pool_data(self):
        X_data, y_data = self.data.X_train, self.data.y_train
        if self.data.X_valid is not None:
            X_data += self.data.X_valid
            y_data += self.data.y_valid
        return X_data, y_data

    def _refit_model(self, best_config, X_data, y_data):
        self.model = clone(self.model)
        self.model.set_params(**best_config)
        self.model.fit(X_data, y_data)

    def _predict_pooled_data(self, X_data):
        return self.model.predict(X_data)

    @staticmethod
    def _modify_params(config):
        tuned_parameters = config["tuned_parameters"]

        for method_name in tuned_parameters:
            temp = tuned_parameters[method_name]
            if temp[0] == "CHOICE":
                temp.pop(0)
                tuned_parameters[method_name] = tune.choice(temp)
            elif len(temp) == 3 and temp[0] == "LOG_UNIFORM":
                tuned_parameters[method_name] = tune.loguniform(temp[1], temp[2])
            elif len(temp) == 3 and temp[0] == "RAND_INT":
                tuned_parameters[method_name] = tune.randint(temp[1], temp[2])
            else:
                raise Exception("Parameters not configured properly")

        return tuned_parameters

    @abstractmethod
    def load_data(self):
        """Data needs to be loaded into self.data structure"""
        raise NotImplementedError

    @abstractmethod
    def perform_search(self):
        """Seach of hyperparameters implemented here"""
        raise NotImplementedError

    @abstractmethod
    def log_session(self, results, best_results, y_test, y_pred):
        """Log session implemented here"""
        raise NotImplementedError


class OptimizerClassification(BaseOptimizer):
    def __init__(self, model, data_loader, valid_data_loader, validator, config):
        super().__init__(model, config)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.validator = validator
        self.scorer = None

    def load_data(self):
        X_train, y_train = convert_images_to_1d(self.data_loader)
        X_valid, y_valid = convert_images_to_1d(self.valid_data_loader)
        # create data structure i.e.: self.data.X_train, self.data.y_train, etc.
        self.data = SimpleNamespace(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
        self.data_loader = None
        self.valid_data_loader = None

    def perform_search(self):
        self._init_scorer()
        search_alg = HyperOptSearch()
        scheduler = HyperBandScheduler()
        tune_config = tune.TuneConfig(
            mode=self.scoring_mode,
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=self.num_samples,
        )
        mlflow_callback = MLflowLoggerCallback(
            tracking_uri=configs.TRACKING_URI,
            experiment_name=self.exper_name,
            save_artifact=True,
        )
        run_config = RunConfig(
            name=self.name,
            #    callbacks=[mlflow_callback]
        )
        tuner = tune.Tuner(
            trainable=self._trainable,
            param_space=self.tuned_params,
            tune_config=tune_config,
            run_config=run_config,
        )
        results = tuner.fit()
        best_results = results.get_best_result(metric=self.scoring_metric, mode=self.scoring_mode)
        return results, best_results

    def _init_scorer(self):
        if self.data.X_valid is None:
            self.logger.info("No validation data provided. Using train data for cross-validation.")
            scoring_metric = make_scorer_ftn(self.scoring_metric, init=False)
            self.scorer = partial(
                objective_cv,
                X_data=self.data.X_train,
                y_data=self.data.y_train,
                validator=self.validator,
                scoring_metric=scoring_metric,
            )
        else:
            self.logger.info("Using validation data for scoring.")
            scoring_metric_ftn = make_scorer_ftn(self.scoring_metric, init=True)
            self.scorer = partial(
                objective_split,
                X_train=self.data.X_train,
                y_train=self.data.y_train,
                X_valid=self.data.X_valid,
                y_valid=self.data.y_valid,
                scoring_metric_ftn=scoring_metric_ftn,
            )

    def _trainable(self, config):
        score = self._objective(config)
        session.report({self.scoring_metric: score, "_metric": score})

    def _objective(self, config):
        self.model = clone(self.model)
        self.model.set_params(**config)
        return self.scorer(self.model)

    def log_session(self, results, best_results, y_test, y_pred):
        best_params = best_results.config
        best_metric = best_results.metrics[self.scoring_metric]
        study_best_params = best_results.metrics

        # get study result and order by scoring metric
        ascending = False if self.scoring_mode == "max" else True
        study_df = results.get_dataframe().sort_values(by=self.scoring_metric, ascending=ascending)
        study_df = study_df.to_string()

        # Log metrics and parameters and model
        mlflow.sklearn.log_model(self.model, "model")
        mlflow.log_params(best_params)
        mlflow.log_metrics({self.scoring_metric: best_metric})

        performance = calculate_classification_metrics(y_test, y_pred)
        mlflow.log_metrics({"precision_avg": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall_avg": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1_avg": performance["overall"]["f1"]})

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            ensure_dir(Path(dp) / "results")
            ensure_dir(Path(dp) / "configs")
            ensure_dir(Path(dp) / "study")

            write_json(best_params, Path(dp, "configs/best_params.json"))
            write_json(study_best_params, Path(dp, "study/study_best_params.json"))
            write_json({self.scoring_metric: best_metric}, Path(dp, "results/best_valid_metric.json"))
            write_json(performance, Path(dp, "results/performance.json"))
            write_json(self.config, Path(dp, "configs/config.json"))
            write_txt(study_df, Path(dp, "study/study_df.txt"))
            mlflow.log_artifacts(dp)

        # log info
        self.logger.info(f"Best hyperparameters found were: {best_params}")
        self.logger.info(f"Best {self.scoring_metric}: {best_metric}")
        self.logger.info(f"Run ID: {mlflow.active_run().info.run_id}")
        self.logger.info(json.dumps(performance, indent=4))
