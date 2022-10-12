import numpy as np
import pysensors as ps
import torch
from pysensors.classification import SSPOC
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from tqdm import tqdm


def cross_val_model(model, X, y):
    model = clone(model)
    cross_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    result = cross_val_score(model, X, y,
                                cv=cross_val,
                                error_score=0,
                                n_jobs=1,
                                scoring='roc_auc')
                                # scoring="balanced_accuracy")
    result_mean = result.mean()
    return round(result_mean, 2)


def get_sparse_coefs(model):
    sensor_coefs = np.abs(model.sensor_coef_)
    if sensor_coefs.ndim == 1:
        sensor_coefs = np.expand_dims(sensor_coefs, axis=1)
    sparse_sensors_idx = sorted(model.selected_sensors)
    # print("Number of sparse sensor locations: ", len(sparse_sensors_idx))
    return sensor_coefs, sparse_sensors_idx



def basis_func(basis_idx, n_basis_modes):
    basis_list = [  ps.basis.Identity(n_basis_modes=n_basis_modes),
                    ps.basis.SVD(n_basis_modes=n_basis_modes, random_state=0),
                    ps.basis.RandomProjection(n_basis_modes=n_basis_modes, random_state=0)]
    return basis_list[basis_idx]


def model_cs_func(basis, n_sensors):
    model_cs = SSPOC(
        classifier=LinearDiscriminantAnalysis(),
        basis=basis,
        n_sensors=n_sensors,
    )
    return model_cs


def import_data(data_loader, model, device):
    X, y = [],[]
    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            x = model.encoder(data)
            x = model.encoder2(x)
            output = x.flatten()

            X.append(output.cpu().numpy())
            y.append(target.cpu().numpy())

    return np.array(X), np.array(y).flatten()
