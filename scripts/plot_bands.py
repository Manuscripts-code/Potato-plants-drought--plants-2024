import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def plot_color_gradients(features_rang):
	fig, ax = plt.subplots(nrows=1, figsize=(13, 1))
	fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
	# ax.set_title("Relavant features", fontsize=14)

	w_start, w_stop = 410, 988

	cmap = cm.get_cmap('Greens', 8)
	cmap.set_under("white")
	gradient = np.vstack((features_rang, features_rang))
	ax.imshow(gradient, aspect='auto', cmap=cmap,
				extent=[w_start, w_stop, 1, 0], vmin=1)

	ax.set_xlabel("Wavelength [nm]")
	ax.xaxis.label.set_size(12)
	ax.set_yticklabels([])
	plt.show()


if __name__ == '__main__':

	lda_1_bands = [1]
	lda_3_bands = [11, 15, 50]
	lda_5_bands = [11, 15, 18, 40, 108]
	lda_10_bands = [4, 5, 8, 20, 25, 72, 81, 89, 95, 128]
	lda_20_bands = [0, 2, 5, 21, 23, 30, 47, 68, 71, 75, 80, 81, 91, 114, 115, 116, 117, 125, 139, 144]

	svm_1_bands = [1]
	svm_3_bands = [11, 15, 50]
	svm_5_bands = [11, 15, 18, 40, 108]
	svm_10_bands = [4, 5, 8, 20, 25, 72, 81, 89, 95, 128]
	svm_20_bands = [4, 5, 8, 16, 20, 25, 40, 42, 46, 55, 57, 60, 68, 79, 97, 102, 115, 119, 123, 139]

	nn_1_bands = [9]
	nn_3_bands = [0, 5, 102]
	nn_5_bands = [0, 11, 15, 50, 156]
	nn_10_bands = [0, 8, 11, 12, 15, 50, 116, 133, 155, 156]
	nn_20_bands = [4, 5, 8, 12, 13, 20, 25, 30, 31, 40, 42, 68, 72, 78, 95, 102, 123, 128, 135, 152]

	all_bands = [lda_1_bands, lda_3_bands, lda_5_bands, lda_10_bands, lda_20_bands]
	all_bands = [svm_1_bands, svm_3_bands, svm_5_bands, svm_10_bands, svm_20_bands]
	all_bands = [nn_1_bands, nn_3_bands, nn_5_bands, nn_10_bands, nn_20_bands]


	for idx, bands in enumerate(all_bands):
		relevances = np.array([0] * 160)
		relevances[bands] = 10
		plot_color_gradients(relevances)



