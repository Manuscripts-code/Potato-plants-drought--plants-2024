
import argparse
import os

import matplotlib.pyplot as plt
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
import pysensors as ps
import ray
import torch
from parse_config import ConfigParser
from pysensors.classification import SSPOC
from ray import tune
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from utils import read_json

from scripts.helpers import (basis_func, cross_val_model, get_sparse_coefs,
                             import_data, model_cs_func)

from data_loader import data_loaders as module_data


def objective_func(config):
    basis_idx = config["basis_idx"]
    n_basis_modes = config["n_basis_modes"]
    n_sensors = config["n_sensors"]
    model_clf = config["model_clf"]
    basis = basis_func(basis_idx, n_basis_modes)
    model_cs = model_cs_func(basis, n_sensors)

    model_cs.fit(X_train, y_train)
    _, sparse_sensors_idx = get_sparse_coefs(model_cs)
    # model_cs.update_sensors(n_sensors=n_sensors, method=method, quiet=True)
    result_mean = cross_val_model(model_clf, X_train[:, sparse_sensors_idx], y_train)
    # model_clf = clone(model_clf)
    # model_clf.fit(X_train[:, sparse_sensors_idx], y_train)
    # result_mean = model_clf.score(X_test[:, sparse_sensors_idx], y_test)

    tune.report(mean_accuracy=result_mean)


def perform_search():
    ray.init()
    analysis = tune.run(
        objective_func,
        name="Optimizer",
        resources_per_trial={"cpu":1, "gpu":0.1},
        metric="mean_accuracy",
        mode="max",
        num_samples=1,
        config=param_grid
        )
    return analysis



if __name__ == '__main__':

    ### PARAMETERS ###
    param_grid = {
        "basis_idx": tune.grid_search([1]),
        # "n_basis_modes": tune.grid_search(list(range(2,100))),
        "n_basis_modes": tune.grid_search([10]),
        "n_sensors": tune.grid_search([3]),
        # "n_basis_modes": tune.randint(2, 100),
        # "n_sensors": tune.randint(50,500),
        "model_clf": tune.grid_search([
                # LinearDiscriminantAnalysis(),
                # SVC(random_state=0),
                MLPClassifier(random_state=1, hidden_layer_sizes=(100,), alpha=0.0005, max_iter=10000, early_stopping=True)
            ])
    }
    path_config = "configs/conv/config_autoencoder_hyp.json"
    path_resume = "saved/models/Potato_plants/best"
    

    ### LAOD MODEL AND DATA ###
    config = read_json(path_config)
    config = ConfigParser(config, path_resume)
    model = config.init_obj('arch', module_arch)
    checkpoint = torch.load(os.path.join(config.resume,"model_best.pth"))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # setup data_loader instances
    train_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['data_sampler'],
        config['data_loader']['args']['grouped_labels_filepath'],
        batch_size=1,
        shuffle=True,
        validation_split=0,
        training=True,
        num_workers=2
    )
    # valid_data_loader = train_data_loader.split_validation()
    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['data_sampler'],
        config['data_loader']['args']['grouped_labels_filepath'],
        batch_size=1,
        shuffle=True,
        validation_split=0,
        training=False,
        num_workers=2
    )

    X_train, y_train = import_data(train_data_loader, model, device)
    X_test, y_test = import_data(test_data_loader, model, device)

    # find sparse columns that are set to zero and delete them
    # zero_col_indeces = np.where(~X_train.any(axis=0))[0]
    # X_train = np.delete(X_train, zero_col_indeces, axis=1)
    # X_test = np.delete(X_test, zero_col_indeces, axis=1)


    ### PERFORM SEARCH ###
    analysis = perform_search()
    config = analysis.best_config

    basis_idx = config["basis_idx"]
    n_basis_modes = config["n_basis_modes"]
    n_sensors = config["n_sensors"]
    model_clf = config["model_clf"]

    basis = basis_func(basis_idx, n_basis_modes)
    model_cs = model_cs_func(basis, n_sensors)

    model_cs.fit(X_train, y_train)
    sensor_coefs, sparse_sensors_idx = get_sparse_coefs(model_cs)


    ### RESULTS ###
    # CV train
    score_train_cv = cross_val_model(model_clf, X_train, y_train)
    score_train_cv_red = cross_val_model(model_clf, X_train[:, sparse_sensors_idx], y_train)

    # full data
    model_clf = clone(model_clf)
    model_clf.fit(X_train, y_train)
    score_test = model_clf.score(X_test, y_test)
    y_pred = model_clf.predict(X_test)

    # reduced data
    model_clf = clone(model_clf)
    model_clf.fit(X_train[:, sparse_sensors_idx], y_train)
    score_test_red = model_clf.score(X_test[:, sparse_sensors_idx], y_test)
    y_pred_red = model_clf.predict(X_test[:, sparse_sensors_idx])

    print("------------------------------------------------------------------------------")
    print("Best parameters:")
    for k, v in config.items():
        print(f"    {k}:  {v}")
    # print("Best estimator", search.best_estimator_)

    print("Best sensor locations:")
    print("     by indexes:", np.array(sparse_sensors_idx))
    # print("     by wavelengths:", np.array(BANDS)[sparse_sensors_idx])
    print("Scores:")
    print(" -> whole space:")
    print("       score train (CV):", score_train_cv)
    print("       score test:", score_test)
    print(" -> reduced dimensions:")
    print("       score train (CV):", score_train_cv_red)
    print("       score test:", score_test_red)
    print("------------------------------------------------------------------------------")
    print("Report:")
    print(" -> whole space:")
    print(classification_report(y_test, y_pred))
    print(" -> reduced dimensions:")
    print(classification_report(y_test, y_pred_red))
    print("------------------------------------------------------------------------------")
    # Plot relevances
    # for idx, sensor_coef in enumerate(sensor_coefs.transpose()):
    #     plot_relevances_fig(sensor_coef, sparse_sensors_idx, height=peak_height_label, distance=distance_between_peaks)
