import numpy as np
import tensorflow as tf
import os
import re
import wavio
from sklearn.model_selection import train_test_split
from scipy.signal import lfilter, butter


def get_train_test_data(file_directory, frame_size, receptive_field, controls=0):
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
			clean_file_path = os.path.join(file_directory, clean_file_path)
			# Process clean file
			cf = wavio.read(clean_file_path)
			cf_data = cf.data.flatten()
			cf_data = np.pad(cf_data, ((0, cf_data.size % frame_size)), 'constant', constant_values=(0, 0))
			cf_data = np.pad(cf_data, (receptive_field, 0), 'constant', constant_values=(0, 0))
			cf_data = np.expand_dims(cf_data, axis=1)
			cf_data = cf_data.astype(np.float32, order='C') / 32768.0

			# If we're training with controlable parameters (eg. Filter cutoff)
			if controls:

				# For each controlable parameter's data subdirectory
				effected_files = [f.path for f in os.scandir(file_directory) if f.path.startswith(clean_file_path[:-10]) and f.path != clean_file_path]
				for ef in effected_files:

					# split on underscore
					split_ef = ef[:-4].split("_")

					# make sure we have right number of parameters in filename
					if len(split_ef) != controls + 1:
						print("File name does not match number of specified control conditions (eg. filter cutoff)")

					# build parameter matrix
					contr = np.zeros((1, controls))
					print("Processing data with:")
					for i in range(1, len(split_ef)):
						sliced = split_ef[i].split('=')
						val = float(sliced[1])
						print(sliced[0] + " value: " + str(val))
						contr[0,i-1] = val

					# make same length as input
					contr = np.repeat(contr, cf_data.shape[0], axis=0)
					# add params to clean signal data
					cf_with_controls = np.concatenate([cf_data, contr], axis=1)
					#ef = os.path.join(file_directory, ef)
					df = wavio.read(ef)
					df_data = df.data.flatten()
					# create padding
					# pad clean and distorted data at the end with frame size


					df_data = np.pad(df_data, (0, df_data.size % frame_size), 'constant', constant_values=(0, 0))

					# convert to normalized float arrays
					# TODO: this fails on non-16-bit bit depths

					df_data = df_data.astype(np.float32, order='C') / 32768.0


					# add to list of samples after splitting on samples_per_datum
					# TODO: idk how to do this without using a list comprehension

					dist_splits = np.array([df_data[i*frame_size:(i+1)*frame_size] for i in range(int(0.1 * (df_data.shape[0]/frame_size)))])
					clean_splits = np.array([cf_with_controls[i*frame_size:receptive_field+(i+1)*frame_size, :] for i in range(int(0.1 * ((cf_with_controls.shape[0]-receptive_field)/frame_size)))])


					clean_signal.extend(clean_splits)
					dist_signal.extend(dist_splits)

			else:
				# find corresponding _distorted file
				distorted_file_path = clean_file_path[:-10] + "_distorted.wav"

				# if this corresponding file exists, process both clean and effected
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
					df_data = df_data[:cf_data.shape[0]]
					cf_data = cf_data[:df_data.shape[0]]

					# create padding
					# pad clean and distorted data at the end with frame size
					cf_data = np.pad(cf_data, (0, cf_data.size % frame_size), 'constant', constant_values=(0, 0))
					df_data = np.pad(df_data, (0, df_data.size % frame_size), 'constant', constant_values=(0, 0))
					# pad clean beginning to match receptive field
					cf_data = np.pad(cf_data, (receptive_field, 0), 'constant', constant_values=(0, 0))

					print(cf_data.shape)

					# convert to normalized float arrays
					# TODO: this fails on non-16-bit bit depths
					cf_data = cf_data.astype(np.float32, order='C') / 32768.0
					df_data = df_data.astype(np.float32, order='C') / 32768.0

					# add to list of samples after splitting on samples_per_datum
					# TODO: idk how to do this without using a list comprehension
					clean_splits = np.array([cf_data[i*frame_size:receptive_field+(i+1)*frame_size] for i in range(int((cf_data.shape[0]-receptive_field)/frame_size))])
					dist_splits = np.array_split(df_data, df_data.shape[0]/frame_size)

					clean_signal.extend(clean_splits)
					dist_signal.extend(dist_splits)

	# Test it's working
	# wavio.write("xtrain1.wav", X_train[11], 44100)
	# wavio.write("ytrain1.wav", y_train[11], 44100)
	# wavio.write("xtest1.wav", X_test[11], 44100)
	# wavio.write("ytest1.wav", y_test[11], 44100)
	print("Running train_test_split...")
	# We can use the sklearn split now.
	X_train, X_test, y_train, y_test = train_test_split(clean_signal,
		dist_signal, test_size=0.25, shuffle=True)

	print("Returning tensors")
	return tf.convert_to_tensor(np.array(X_train, dtype=np.float32)), tf.convert_to_tensor(np.array(y_train, dtype=np.float32)), \
		tf.convert_to_tensor(np.array(X_test, dtype=np.float32)), tf.convert_to_tensor(np.array(y_test, dtype=np.float32))

def get_run_data(file_path, frame_size, receptive_field):
	cf = wavio.read(file_path)

	cf_data = cf.data.flatten()

	# create padding
	# pad clean and distorted data at the end with frame size
	cf_data = np.pad(cf_data, (0, cf_data.size % frame_size), 'constant', constant_values=(0, 0))
	# pad clean beginning to match receptive field
	cf_data = np.pad(cf_data, (receptive_field, 0), 'constant', constant_values=(0, 0))

	# convert to normalized float arrays
	# TODO: this fails on non-16-bit bit depths
	cf_data = cf_data.astype(np.float32, order='C') / 32768.0

	# add to list of samples after splitting on samples_per_datum
	# TODO: idk how to do this without using a list comprehension
	clean_splits = np.array([cf_data[i*frame_size:receptive_field+(i+1)*frame_size] for i in range(int((cf_data.shape[0]-receptive_field)/frame_size))])

	return clean_splits
