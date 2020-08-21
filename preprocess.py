import numpy as np
import tensorflow as tf
import os
import wavio
from sklearn.model_selection import train_test_split
from scipy.signal import lfilter, butter

SAMPLES_PER_DATUM = 512

def get_data(file_directory):
	"""

	Brad: This is where we should load the data from the file and preprocess
	it.

	Typically in DL we'd also split it into training and testing examples.
	So we'd return something like 4 numpy arrays:
		return train_inputs, train_ground_truth, test_inputs, test_ground_truth

	"""

	clean_train_signal = []
	clean_test_signal = []
	distorted_train_signal = []
	distorted_test_signal = []
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

				cf_data = cf.data.flatten()
				df_data = df.data.flatten()

				# shorten distorted file to match clean file
				df_data = df_data[:cf_data.shape[0]]

				# create padding
				np.pad(cf_data, (0, cf_data.size % SAMPLES_PER_DATUM), 'constant', constant_values=(0, 0))
				np.pad(df_data, (0, df_data.size % SAMPLES_PER_DATUM), 'constant', constant_values=(0, 0))

				# convert to normalized float arrays
				# TODO: this fails on non-16-bit bit depths
				cf_data = cf_data.astype(np.float32, order='C') / 32768.0
				df_data = df_data.astype(np.float32, order='C') / 32768.0

				# add to list of samples after splitting on samples_per_datum
				# also exclude the last element just in case it's short
				# via [:-1] slice
				clean_splits = np.array_split(cf_data, cf_data.shape[0]/SAMPLES_PER_DATUM)[:-1]
				dist_splits = np.array_split(df_data, df_data.shape[0]/SAMPLES_PER_DATUM)[:-1]
				split_index = int(len(clean_splits)*0.9)
				clean_train_signal.extend(clean_splits[:split_index])
				clean_test_signal.extend(clean_splits[split_index:])
				distorted_train_signal.extend(dist_splits[:split_index])
				distorted_test_signal.extend(dist_splits[split_index:])

	# Test it's working
	# wavio.write("xtrain1.wav", X_train[11], 44100)
	# wavio.write("ytrain1.wav", y_train[11], 44100)
	# wavio.write("xtest1.wav", X_test[11], 44100)
	# wavio.write("ytest1.wav", y_test[11], 44100)

	return tf.convert_to_tensor(clean_train_signal), tf.convert_to_tensor(distorted_train_signal), \
		tf.convert_to_tensor(clean_test_signal), tf.convert_to_tensor(distorted_test_signal)
