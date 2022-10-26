import os
import pickle
from abc import abstractmethod
from functools import partial
from types import SimpleNamespace

import numpy as np
from ray import tune
from ray.air import session
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from rich.progress import track
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    mean_absolute_error,
)
from sklearn.utils import shuffle

from .scorer import make_scorer_ftn, objective_cv, objective_split


class BaseOptimizer:
    def __init__(self, model, config):
        self.logger = config.get_logger("trainer", verbosity=2)
        self.model = model
        self.scoring_metric = config["scoring_metric"]
        self.scoring_mode = config["scoring_mode"]
        self.tuned_params = self._modify_params(config)
        self.debug = config["debug"]
        self.num_samples = config["num_samples"]
        self.name = config["name"]
        self.save_dir = config.save_dir
        self.data = None

    def optimize(self):
        if self.data is None:
            raise Exception("use self.load_data() before self.optimize()")
        if self.debug:
            print("Not implemented.")
        else:
            self._debug_false()

    def _debug_false(self):
        results = self.perform_search()
        results = self.create_train_report(results)
        self.model.set_params(**results.config)
        self.model.fit(self.data.X_train, self.data.y_train)
        self.save_model(self.model)
        # self.save_report(train_report, "report_train.txt")

    def save_model(self, model):
        save_path = os.path.join(self.save_dir, "model.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

    def load_model(self):
        load_path = os.path.join(self.save_dir, "model.pkl")
        with open(load_path, "rb") as f:
            model = pickle.load(f)
            print(model)
        return model

    def save_report(self, report, name_txt):
        save_path = os.path.join(self.save_dir, name_txt)
        with open(save_path, "w") as text_file:
            text_file.write(report)

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

    def create_train_report(self, analysis):
        """Should return report from training"""
        return "Train report not configured."

    @abstractmethod
    def load_data(self):
        """Load of data implemented here"""
        raise NotImplementedError

    @abstractmethod
    def perform_search(self):
        """Seach of hyperparameters implemented here"""
        raise NotImplementedError


class OptimizerClassification(BaseOptimizer):
    def __init__(self, model, data_loader, valid_data_loader, validator, config):
        super().__init__(model, config)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.validator = validator
        self.scorer = None

    def load_data(self):
        X_train, y_train = self.convert_images_to_1d(self.data_loader)
        X_valid, y_valid = self.convert_images_to_1d(self.valid_data_loader)
        # create data structure i.e.: self.data.X_train, self.data.y_train, etc.
        self.data = SimpleNamespace(
            X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid
        )
        self.data_loader = None
        self.valid_data_loader = None

    def convert_images_to_1d(self, data_loader):
        if data_loader is None:
            return None, None
        X, y = [], []
        for images, targets in track(data_loader, description="Converting images..."):
            # average spatial dimension of images
            signatures = images.mean((2, 3)).numpy()
            targets = targets.numpy()
            [X.append(sig) for sig in signatures]
            [y.append(tar) for tar in targets]
        return X, y

    def init_scorer(self):
        if self.data.X_valid is None:
            scoring_metric = make_scorer_ftn(self.scoring_metric, init=False)
            self.scorer = partial(
                objective_cv,
                X_data=self.data.X_train,
                y_data=self.data.y_train,
                validator=self.validator,
                scoring_metric=scoring_metric,
            )
        else:
            scoring_metric_ftn = make_scorer_ftn(self.scoring_metric, init=True)
            self.scorer = partial(
                objective_split,
                X_train=self.data.X_train,
                y_train=self.data.y_train,
                X_valid=self.data.X_valid,
                y_valid=self.data.y_valid,
                scoring_metric_ftn=scoring_metric_ftn,
            )

    def perform_search(self):
        self.init_scorer()
        search_alg = HyperOptSearch()
        scheduler = HyperBandScheduler()
        tune_config = tune.TuneConfig(
            mode=self.scoring_mode,
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=self.num_samples,
        )
        run_config = RunConfig(
            name=self.name,
        )
        tuner = tune.Tuner(
            trainable=self.trainable,
            param_space=self.tuned_params,
            tune_config=tune_config,
            run_config=run_config,
        )
        results = tuner.fit()
        return results

    def trainable(self, config):
        score = self.objective(config)
        session.report({self.scoring_metric: score, "_metric": score})

    def objective(self, config):
        self.model.set_params(**config)
        return self.scorer(self.model)

    def create_train_report(self, results):
        best_results = results.get_best_result(metric=self.scoring_metric, mode=self.scoring_mode)
        self.logger.info(f"Best hyperparameters found were: {best_results.config}")
        self.logger.info(f"Best results all: {best_results.metrics}")
        self.logger.info(
            f"Best {self.scoring_metric}: {best_results.metrics[self.scoring_metric]}\n"
        )
        self.save_report(str(best_results.metrics[self.scoring_metric]), "scoring_train.txt")
        return best_results
