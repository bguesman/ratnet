import numpy as np
import tensorflow as tf
import os
import wavio
from sklearn.model_selection import train_test_split
from scipy.signal import lfilter, butter

SAMPLES_PER_DATUM = 128

def get_data(file_directory):
	"""

	Brad: This is where we should load the data from the file and preprocess
	it.

	Typically in DL we'd also split it into training and testing examples.
	So we'd return something like 4 numpy arrays:
		return train_inputs, train_ground_truth, test_inputs, test_ground_truth

	"""
	clean_signal = []
	distorted_signal = []
	for clean_file_path in os.listdir(file_directory):
		if clean_file_path.endswith("_clean.wav"):
			# find corresponding _distorted file
			distorted_file_path = clean_file_path[:-10] + "_distorted.wav"

			# if this corresponding file exists, process both clean and distorted
			if distorted_file_path in os.listdir(file_directory):
				clean_file_path = os.path.join(file_directory, clean_file_path)
				distorted_file_path = os.path.join(file_directory, distorted_file_path)
				print(clean_file_path)
				print(distorted_file_path)

				# extract signals
				cf = wavio.read(clean_file_path)
				df = wavio.read(distorted_file_path)


				# create padding
				cf_data = cf.data.flatten()
				df_data = df.data.flatten()
				np.pad(cf_data, (0, cf_data.size % SAMPLES_PER_DATUM), 'constant', constant_values=(0, 0))
				np.pad(df_data, (0, df_data.size % SAMPLES_PER_DATUM), 'constant', constant_values=(0, 0))

				# filter distorted signal
				# TODO: apply a HP filter to df_data (the distorted file's data)

				# add to list of samples after splitting on samples_per_datum
				clean_signal.extend(np.split(cf_data, SAMPLES_PER_DATUM))
				distorted_signal.extend(np.split(df_data, SAMPLES_PER_DATUM))

	X_train, X_test, y_train, y_test = train_test_split(clean_signal, distorted_signal, test_size = 0.2, random_state = 31)

	# Test it's working
	# wavio.write("xtrain1.wav", X_train[11], 20500)
	# wavio.write("ytrain1.wav", y_train[11], 20500)
	# wavio.write("xtest1.wav", X_test[11], 20500)
	# wavio.write("ytest1.wav", y_test[11], 20500)

	return X_train, y_train, X_test, y_test