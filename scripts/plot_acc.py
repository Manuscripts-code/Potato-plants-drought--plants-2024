import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn import preprocessing


def plot_methods_acc(signatures, labels, *, no_sensors_list, title="", x_label="", y_label="", vertical_lines=None):
	""" Plot signatures with corresponding wavelenght or consequtive wavelength number
	Args:
		signatures (list): list of lists/signatures
		labels (list): list of labels
	"""
	transformer = preprocessing.LabelEncoder()
	transformer.fit(labels)
	labels = transformer.transform(labels)

	fig = plt.figure() 
	fig.set_figheight(5)
	fig.set_figwidth(15)

	ax = fig.add_subplot(1, 1, 1)
	ax.set_title(title, fontsize=14)
	ax.set_ylabel(y_label, fontsize=12)
	ax.set_xlabel(x_label, fontsize=12)
	ax.tick_params(axis='both', which='major', labelsize=22)
	ax.tick_params(axis='both', which='minor', labelsize=22)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_ylim(0.7, 0.9)

	cmap = plt.get_cmap('viridis')
	no_colors = len(np.unique(labels))
	colors = cmap(np.linspace(0, 1, no_colors))

	x_scat = list(np.arange(len(signatures[0])))

	def format_func(value, tick_number):
		if tick_number is not None:
			if tick_number < len(no_sensors_list):
				return str(no_sensors_list[int(tick_number)])
			else:
				return ""

	ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

	for obj in range(len(signatures)):
		ax.plot(x_scat, signatures[obj], "-gD", color=colors[labels[obj]], alpha=0.7)

	custom_lines = []
	for idx in range(no_colors):
		custom_lines.append(Line2D([0], [0], color=colors[idx], lw=2))

	labels = transformer.inverse_transform(list(range(no_colors)))
	ax.legend(custom_lines, [str(num) for num in labels], fontsize=30)
	plt.show()


if __name__ == '__main__':
	acc_dict = {
		"LDA": [0.85, 0.85, 0.82, 0.85, 0.74, 0.74],
		"SVM": [0.82, 0.79, 0.82, 0.82, 0.79, 0.90],
		"NN": [0.77, 0.72, 0.77, 0.72, 0.82, 0.87],
	}
	no_sensors_list = [1, 1, 3, 5, 10, 20, 160]
	kwargs = {
		"no_sensors_list": no_sensors_list,
		# "title": "Accuracy on train data at different numbers of sensors",
		# "x_label": "Number of sensors",
		# "y_label": "Accuracy"
	}
	plot_methods_acc(list(acc_dict.values()), list(acc_dict.keys()), **kwargs)

