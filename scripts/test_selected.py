
import argparse
import os

import data_loader.data_loaders as module_data
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

if __name__ == '__main__':

	### PARAMETERS ###
	path_config = "configs/conv/config_autoencoder_hyp.json"
	path_resume = "saved/models/Potato_plants/best"
	basis_idx = 1
	n_basis_modes = 20
	n_sensors = 5
	model_clf = LinearDiscriminantAnalysis()
	# model_clf = SVC(random_state=0)
	# model_clf = MLPClassifier(random_state=1, hidden_layer_sizes=(100,), alpha=0.0005, max_iter=10000, early_stopping=True)


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



