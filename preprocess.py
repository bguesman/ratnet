import numpy as np
import tensorflow as tf
import os
import wavio
from sklearn.model_selection import train_test_split
from scipy.signal import lfilter, butter

def get_data(file_directory, frame_size, receptive_field):
	"""

	Brad: This is where we should load the data from the file and preprocess
	it.

	Typically in DL we'd also split it into training and testing examples.
	So we'd return something like 4 numpy arrays:
		return train_inputs, train_ground_truth, test_inputs, test_ground_truth

	"""

	clean_signal = []
	dist_signal = []
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

				# make files same length just in case they are somehow different
				minlen = min(cf_data.shape[0], df_data.shape[0])
				df_data = df_data[:minlen]
				cf_data = cf_data[:minlen]

				# create padding
				# pad clean and distorted data at the end with frame size
				cf_data = np.pad(cf_data, (0, cf_data.size % frame_size), 'constant', constant_values=(0, 0))
				df_data = np.pad(df_data, (0, df_data.size % frame_size), 'constant', constant_values=(0, 0))


				# convert to normalized float arrays
				# TODO: this fails on non-16-bit bit depths
				cf_data = cf_data.astype(np.float32, order='C') / 32768.0
				df_data = df_data.astype(np.float32, order='C') / 32768.0

				# add to list of samples after splitting on samples_per_datum
				# TODO: idk how to do this without using a list comprehension
				clean_splits = np.array_split(cf_data, cf_data.shape[0]/frame_size)
				dist_splits = np.array_split(df_data, df_data.shape[0]/frame_size)


				clean_signal.extend(clean_splits)
				dist_signal.extend(dist_splits)

	# Test it's working
	# wavio.write("xtrain1.wav", X_train[11], 44100)
	# wavio.write("ytrain1.wav", y_train[11], 44100)
	# wavio.write("xtest1.wav", X_test[11], 44100)
	# wavio.write("ytest1.wav", y_test[11], 44100)

	# We can use the sklearn split now. We'll shuffle ourselves.
	# That way we can actually write out the data to a wav file
	# and make sense of it if we want to.
	print(len(clean_signal))
	print(len(dist_signal))
	X_train, X_test, y_train, y_test = train_test_split(clean_signal,
		dist_signal, test_size=0.25, shuffle=False)

	return tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train), \
		tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_test)
