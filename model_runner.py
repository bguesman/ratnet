import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
import wavio
import argparse
from audio_device_model import AudioDeviceModel

# @brief: Utility class for printing different colors to the terminal.
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    ITALICS = '\033[3m'
    UNDERLINE = '\033[4m'

# @brief: Use argparse to parse the command line arguments. Exits and prints
# an error message upon receiving invalid arguments.
# @return: argparse args struct.
def parse_command_line():
    # Use argparse for command line arguments.
    parser = argparse.ArgumentParser(description='Train, test, or run an audio device model.')
    parser.add_argument('mode', help='Mode to run the script in, one of <TRAIN, TEST, RUN>', type=str, choices=['TRAIN', 'TEST', 'RUN'])
    parser.add_argument('--weight_store_path', type=str, default=None, help='specify directory to store weight checkpoints in')
    parser.add_argument('--weight_load_path', type=str, default=None, help='specify directory to load weights from')
    parser.add_argument('--data_path', type=str, default=None, help='specify directory that data is in, if in TRAIN or TEST mode')
    parser.add_argument('--signal_path', type=str, default=None, help='specify the path to the signal to process, if in RUN mode')
    parser.add_argument('--out_path', type=str, default=None, help='specify the path to output the processed signal to')
    parser.add_argument('--epochs', type=int, default=1, help='specify how many epochs to train the model for')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='specify the learning rate of the optimizer')
    parser.add_argument('--frame_size', type=int, default=128, help='specify the frame size of the model')

    args = parser.parse_args()

    # If we are running in test or train mode, we need data.
    if ((args.mode == 'TRAIN' or args.mode == 'TEST') and args.data_path == None):
        print("ERROR: in both TRAIN and TEST mode, a data path must be specified with --data_path")
        sys.exit()

    # If we are running in run mode, we need a signal.
    if (args.mode == 'RUN' and (args.signal_path == None or args.out_path == None)):
        print("ERROR: in RUN mode, a signal path and out path must be specified with --signal_path and --out_path")
        sys.exit()

    print('')
    print(bcolors.OKGREEN + bcolors.BOLD + 'Running in', args.mode, "mode with parameters: " + bcolors.ENDC)
    print(bcolors.BOLD + 'weight store path:' + bcolors.ENDC, args.weight_store_path)
    print(bcolors.BOLD + 'weight load path:' + bcolors.ENDC, args.weight_load_path)
    print(bcolors.BOLD + 'data path:' + bcolors.ENDC, args.data_path)
    print(bcolors.BOLD + 'signal path:' + bcolors.ENDC, args.signal_path)
    print(bcolors.BOLD + 'out path:' + bcolors.ENDC, args.out_path)
    print(bcolors.BOLD + 'epochs:' + bcolors.ENDC, args.epochs)
    print(bcolors.BOLD + 'learning rate:' + bcolors.ENDC, args.learning_rate)
    print(bcolors.BOLD + 'frame size:' + bcolors.ENDC, args.frame_size)
    print('')

    return args

# @brief: Given command line arguments, setup an AudioDeviceModel.
# @return: AudioDeviceModel object.
def setup_model(args):
    print(bcolors.OKGREEN + "Creating model..." + bcolors.ENDC)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # Create model with specified learning rate and frame size.
    model = AudioDeviceModel(learning_rate=args.learning_rate, \
        frame_size=args.frame_size)
    # Load weights if a path is specified.
    if args.weight_load_path is not None:
        print("Loading weights from path", args.weight_load_path + "...")
        model.load_weights(args.weight_load_path)
    print(bcolors.OKGREEN + "Done." + bcolors.ENDC)
    return model

# @brief: Utility class for storing info about each file.
class FileInfo:
    def __init__(self, local_path, global_path, directory, batches_per_file):
        # Set the global path so we can load the file.
        self.global_path = global_path
        self.local_path = local_path
        self.directory = directory

        # Extract parameter vector from local filepath name.
        # Split on underscore. Skip first slipt
        parameter_splits = (local_path[:-4].split("_"))[1:]

        # Build parameter vector.
        self.parameters = np.zeros((len(parameter_splits)))
        print("Indexing file", bcolors.ITALICS + local_path + bcolors.ENDC, "with parameters:")
        for i in range(len(parameter_splits)):
            # Split on the equals sign.
            sliced = parameter_splits[i].split('=')
            # To the right of the equals sign is the parameter value.
            val = float(sliced[1])
            # To the left is the parameter name.
            print(bcolors.BOLD + sliced[0] + ": " + bcolors.ENDC + str(val))
            self.parameters[i] = val

        # Flags to specify which batches in this file have been processed
        # (trained on).
        self.batches_processed = [False for _ in range(batches_per_file)]

# @brief: indexes files in data_path and returns a list of FileInfo structs.
# @return: list of FileInfo structs for each file in directory data_path.
def get_data_index(data_path, batches_per_file=10):
    print('')
    print(bcolors.OKGREEN + 'Building data index...' + bcolors.ENDC)
    data_index = []
    for processed_path in os.listdir(data_path):
        if not processed_path.endswith("_clean.wav"):
            global_path = os.path.join(data_path, processed_path)
            data_index.append(FileInfo(local_path=processed_path, \
                global_path=global_path, directory=data_path, \
                batches_per_file=batches_per_file))
    print(bcolors.OKGREEN + 'Done.')
    print('')
    return data_index

# @brief: clears the file processed flags in data index.
def clear_data_index(index):
    for file_info in index:
        for i in range(len(file_info.batches_processed)):
            file_info.batches_processed[i] = False

# @brief: Given a file info object and batch number, returns the clean and
# processed batch.
def get_input_processed_pair(model, file_info, batch, total_batches):
    # Get the clean file path by searching through the file's parent directory.
    clean_path = None
    for possibly_clean_file in os.listdir(file_info.directory):
        if possibly_clean_file.endswith("_clean.wav"):
            clean_path = os.path.join(file_info.directory, possibly_clean_file)

    # Load both the clean and distorted files.
    x = wavio.read(clean_path).data
    y = wavio.read(file_info.global_path).data

    # Pad the data.
    x = np.pad(x, ((0, x.shape[0] % model.frame_size), (0, 0)), 'constant', constant_values=(0, 0))
    y = np.pad(y, ((0, y.shape[0] % model.frame_size), (0, 0)), 'constant', constant_values=(0, 0))
    # Pad clean beginning to match receptive field.
    x = np.pad(x, ((model.R, 0), (0, 0)), 'constant', constant_values=(0, 0))

    # Take the batch. Use the distorted data shape since it hasn't been
    # padded with the receptive field.
    start = int((batch / total_batches) * y.shape[0])
    end = int(((batch + 1.0) / total_batches) * y.shape[0])
    y = y[start:end,...]
    x = x[start:end+model.R,...]

    # Normalize to [-1.0, 1.0]
    y = y.astype(np.float32, order='C') / 32768.0
    x = x.astype(np.float32, order='C') / 32768.0

    # TODO: this is fucking SLOW!
    x = np.array([x[i*model.frame_size:model.R+(i+1)*model.frame_size] for i in range(int((x.shape[0]-model.R)/model.frame_size))])
    y = np.array([y[i*model.frame_size:(i+1)*model.frame_size] for i in range(int(y.shape[0]/model.frame_size))])

    return x, y

# @brief: Trains model for one batch with inputs x and ground truth y.
def train_batch(model, x, y, mini_batch_size=32):
    # Loop through the batch splitting into mini batches.
    for i in range(0, int(x.shape[0]/mini_batch_size)):
        # Grab the input and corresponding ground truth batches.
        batch_start = i*mini_batch_size
        batch_end = (i+1)*mini_batch_size
        input = x[batch_start:batch_end]
        ground_truth = y[batch_start:batch_end]

        # Compute the model prediction and the loss within the scope of the
        # gradient tape.
        with tf.GradientTape(persistent=True) as tape:
            model_prediction = model(input)
            # TODO: squeezing here won't work for stereo!!
            # Divide by mini batch size to compute average loss across batch.
            loss = model.loss(model_prediction, tf.squeeze(ground_truth)) / mini_batch_size

        if (i % 100 == 0):
            print("Loss on iteration " + str(i) + ": " + str(loss))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# @brief: Trains model for one epoch on files in data index index.
def train_epoch(model, index):
    # Iterate over the batches in a random order.
    # TODO: To make this fully random we could create some kind of matrix
    # of iteration order. Right now it doesn't really go through in a random
    # order.
    random_batch_order = list(range(len(index[0].batches_processed)))
    random.shuffle(random_batch_order)
    for b in random_batch_order:
        # Iterate over the files in a random order.
        random_file_order = list(range(len(index)))
        random.shuffle(random_file_order)
        for f in random_file_order:
            # Get the file info.
            file_info = index[f]

            print(bcolors.BOLD + "Training on file", file_info.local_path, ", batch", str(b) + bcolors.ENDC)

            # Get the training pair of clean and distorted data for this file
            # info and batch struct.
            x, y = get_input_processed_pair(model, file_info, b, len(file_info.batches_processed))

            # Shuffle inputs and ground truth in the same order.
            shuffle_order = list(range(x.shape[0]))
            random.shuffle(shuffle_order)
            x = x[shuffle_order,:,:]
            y = y[shuffle_order,:,:]

            # Tile parameters to be the same dimensions as x.
            params = np.tile(file_info.parameters, x.shape)
            # Stitch the parameters vector onto the clean data as new channels.
            x = np.concatenate([x, params], axis=2)

            # Train the model on this batch.
            train_batch(model, np.array(x, dtype=np.float32), np.array(y, dtype=np.float32))

            # Set flag to true indicating that this was processed.
            file_info.batches_processed[b] = True

# @brief: Trains the model on the data in the folder data_path for specified
# number of epochs. Checkpoints model after every epoch.
def train(model, data_path, weight_store_path, epochs):
    # Get the data index, which is a list of FileInfo objects that specify
    # the path to each file, the parameters associated with the file, and
    # a list of booleans specified whether or not each chunk in the file
    # has been processed.
    data_index = get_data_index(data_path)

    # Train for specified number of epochs.
    for i in range(epochs):
        print(bcolors.BOLD + bcolors.OKGREEN + "EPOCH", str(i), bcolors.ENDC)
        # Train for an epoch.
        train_epoch(model, data_index)
        # Clear out the data index.
        clear_data_index(data_index)
        # Save the weights.
        if weight_store_path is not None:
            model.save_weights(weight_store_path + "_epoch_" + str(i), save_format='tf')

    print(bcolors.BOLD + bcolors.OKGREEN + "DONE TRAINING" + bcolors.ENDC)

# @brief: tests model for one batch with inputs x and ground truth y.
# Computes TOTAL loss, not AVERAGE loss.
def test_batch(model, x, y, mini_batch_size=32):
    # Loop through the batch splitting into mini batches.
    total_loss = 0
    for i in range(0, int(x.shape[0]/mini_batch_size)):
        if (i % 100 == 0):
            print("minibatch", i)
        # Grab the input and corresponding ground truth batches.
        batch_start = i*mini_batch_size
        batch_end = (i+1)*mini_batch_size
        input = x[batch_start:batch_end]
        ground_truth = y[batch_start:batch_end]

        model_prediction = model(input)
        # TODO: squeezing here won't work for stereo!!
        total_loss += model.loss(model_prediction, tf.squeeze(ground_truth))
    return total_loss

# @brief: tests the model.
def test(model, data_path):
    # Get the data index, which is a list of FileInfo objects that specify
    # the path to each file, the parameters associated with the file, and
    # a list of booleans specified whether or not each chunk in the file
    # has been processed.
    index = get_data_index(data_path)
    total_loss = 0.0
    i = 0
    for b in range(len(index[0].batches_processed)):
        for file_info in index:
            print(bcolors.BOLD + "Testing on file", file_info.local_path, ", batch", str(b) + bcolors.ENDC)

            # Get the training pair of clean and distorted data for this file
            # info and batch struct.
            x, y = get_input_processed_pair(model, file_info, b, len(file_info.batches_processed))

            # Tile parameters to be the same dimensions as x.
            params = np.tile(file_info.parameters, x.shape)
            # Stitch the parameters vector onto the clean data as new channels.
            x = np.concatenate([x, params], axis=2)

            # TODO: squeezing here won't work for stereo!!
            # Divide by the number of items to make sure this is average loss.
            total_loss += test_batch(model, x, y) / x.shape[0]
            i += 1
    return total_loss / i

# @brief: main function.
def main():
    # Parse arguments from command line. This function guarantees that we
    # have everything we need for the mode we are running in.
    args = parse_command_line()

    # Create the model.
    model = setup_model(args)

    # Train the model.
    if (args.mode == 'TRAIN'):
        train(model, data_path=args.data_path, \
            weight_store_path=args.weight_store_path,
            epochs=args.epochs)

    # TODO: test the model.
    if (args.mode == 'TEST' or args.mode == 'TRAIN'):
        print(bcolors.BOLD + bcolors.OKGREEN + "Computing loss on test data." + bcolors.ENDC)
        loss = test(model, args.data_path)
        print(bcolors.BOLD + bcolors.OKGREEN + "Loss on test data:", loss, bcolors.ENDC)

    # TODO: run the model.
    if (args.mode == 'RUN'):
        pass

if __name__ == '__main__':
   main()


# def test(model, test_inputs, test_ground_truth, batch_size=32):
#     """
#     This computes the average loss across the testing sample.
#     """
#     total_loss = 0
#     for i in range(0, int(test_inputs.shape[0]/batch_size)):
#         # Grab the input and corresponding ground truth. TODO: we can batch
#         # this if we want to make it faster.
#         batch_start = i*batch_size
#         batch_end = (i+1)*batch_size
#         input = test_inputs[batch_start:batch_end]
#         ground_truth = test_ground_truth[batch_start:batch_end]
#
#         # Run the model on the input to get the predicted output.
#         model_prediction = model(input)
#
#         # Compute the loss.
#         total_loss += model.loss(model_prediction, ground_truth)
#     return total_loss / float(int(test_inputs.shape[0]/batch_size))
#
# def test_wav(model, test_inputs, out_path, batch_size=32):
#     output = np.zeros((test_inputs.shape[0], 128))
#     for i in range(0, int(test_inputs.shape[0]/batch_size)):
#         # Grab the input and corresponding ground truth. TODO: we can batch
#         # this if we want to make it faster.
#         batch_start = i*batch_size
#         batch_end = (i+1)*batch_size
#         batched_input = test_inputs[batch_start:batch_end]
#
#         # Run the model on the input to get the predicted output.
#         output[batch_start:batch_end] = model(batched_input)
#
#     # Flatten the output.
#     output = np.reshape(output, (-1))
#
#     # Convert to wav.
#     output = np.clip(output, -1.0, 1.0) * 32768.0
#     output = output.astype(np.int16, order='C')
#     wavio.write(out_path + "_wet.wav", output, 44100)
#
