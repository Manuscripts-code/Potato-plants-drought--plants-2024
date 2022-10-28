import os
from abc import abstractmethod
from functools import partial
from types import SimpleNamespace

import numpy as np
from ray import tune
from ray.air import session
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from utils.utils import write_txt, write_pickle

from .helpers import convert_images_to_1d
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
        results, best_results = self.perform_search()
        self._create_train_report(results, best_results)
        self._refit_model(best_results.config)
        self._save_model()

    def _refit_model(self, best_config):
        X_data, y_data = self.data.X_train, self.data.y_train
        if self.data.X_valid is not None:
            X_data += self.data.X_valid
            y_data += self.data.y_valid

        self.model.set_params(**best_config)
        self.model.fit(X_data, y_data)

    def _save_model(self):
        write_pickle(self.model, self.save_dir / "model.pkl")

    def _save_report(self, report, name_txt):
        report = str(report)
        save_path = os.path.join(self.save_dir, name_txt)
        write_txt(report, save_path)

    def _create_train_report(self, results, best_results):
        best_config = best_results.config
        best_metric = best_results.metrics[self.scoring_metric]
        best_all_params = best_results.metrics
        all_df = results.get_dataframe().to_string()

        self.logger.info(f"Best hyperparameters found were: {best_config}")
        self.logger.info(f"Best {self.scoring_metric}: {best_metric}\n")
        self._save_report(best_config, "best_config_train.txt")
        self._save_report(best_metric, "best_metric_train.txt")
        self._save_report(best_all_params, "best_all_params_train.txt")
        self._save_report(all_df, "all_df_train.txt")

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
        X_train, y_train = convert_images_to_1d(self.data_loader)
        X_valid, y_valid = convert_images_to_1d(self.valid_data_loader)
        # create data structure i.e.: self.data.X_train, self.data.y_train, etc.
        self.data = SimpleNamespace(
            X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid
        )
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
        run_config = RunConfig(
            name=self.name,
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

    def _trainable(self, config):
        score = self._objective(config)
        session.report({self.scoring_metric: score, "_metric": score})

    def _objective(self, config):
        self.model.set_params(**config)
        return self.scorer(self.model)
