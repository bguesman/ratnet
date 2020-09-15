import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
import wavio
import argparse
from audio_utils import resample
from audio_device_model import AudioDeviceModel
import time

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
    parser.add_argument('--model_store_path', type=str, default=None, help='specify directory to store model checkpoints in')
    parser.add_argument('--model_load_path', type=str, default=None, help='specify directory to load model from')
    parser.add_argument('--train_data_path', type=str, default=None, help='specify directory that training data is in, if in TRAIN mode')
    parser.add_argument('--test_data_path', type=str, default=None, help='specify directory that test data is in, if in TRAIN or TEST mode')
    parser.add_argument('--signal_path', type=str, default=None, help='specify the path to the signal to process, if in RUN mode')
    parser.add_argument('--out_path', type=str, default=None, help='specify the path to output the processed signal to')
    parser.add_argument('--epochs', type=int, default=1, help='specify how many epochs to train the model for')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='specify the learning rate of the optimizer')
    parser.add_argument('--frame_size', type=int, default=128, help='specify the frame size of the model')
    parser.add_argument('--parameters', type=float, nargs='+', help='specify the parameters to use in run mode, separated by spaces')

    args = parser.parse_args()

    # If we are running in test or train mode, we need data.
    if ((args.mode == 'TRAIN') and args.train_data_path == None):
        print("ERROR: in TRAIN mode, a train data path must be specified with --train_data_path")
        sys.exit()

    if ((args.mode == 'TEST') and args.test_data_path == None):
        print("ERROR: in TEST mode, a test data path must be specified with --test_data_path")
        sys.exit()

    # If we are running in run mode, we need a signal.
    if (args.mode == 'RUN' and (args.signal_path == None or args.out_path == None)):
        print("ERROR: in RUN mode, a signal path and out path must be specified with --signal_path and --out_path")
        sys.exit()

    print('')
    print(bcolors.OKGREEN + bcolors.BOLD + 'Running in', args.mode, "mode with parameters: " + bcolors.ENDC)
    print(bcolors.BOLD + 'model store path:' + bcolors.ENDC, args.model_store_path)
    print(bcolors.BOLD + 'model load path:' + bcolors.ENDC, args.model_load_path)
    print(bcolors.BOLD + 'train data path:' + bcolors.ENDC, args.train_data_path)
    print(bcolors.BOLD + 'test data path:' + bcolors.ENDC, args.test_data_path)
    print(bcolors.BOLD + 'signal path:' + bcolors.ENDC, args.signal_path)
    print(bcolors.BOLD + 'out path:' + bcolors.ENDC, args.out_path)
    print(bcolors.BOLD + 'epochs:' + bcolors.ENDC, args.epochs)
    print(bcolors.BOLD + 'learning rate:' + bcolors.ENDC, args.learning_rate)
    print(bcolors.BOLD + 'frame size:' + bcolors.ENDC, args.frame_size)
    print(bcolors.BOLD + 'parameters:' + bcolors.ENDC, args.parameters)
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

    # Load model if a path is specified.
    if args.model_load_path is not None:
        print("Loading model from path", args.model_load_path + "...")
        model.load_weights(args.model_load_path)

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
        if (len(parameter_splits) == 0):
            self.parameters = None
        else:
            for i in range(len(parameter_splits)):
                # Split on the equals sign.
                sliced = parameter_splits[i].split('=')
                if (len(sliced) > 1):
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
def get_data_index(data_path, batches_per_file=50):
    print('')
    print(bcolors.OKGREEN + 'Building data index...' + bcolors.ENDC)
    data_index = []
    for processed_path in os.listdir(data_path):
        if not processed_path.endswith("_clean.wav"):
            global_path = os.path.join(data_path, processed_path)
            data_index.append(FileInfo(local_path=processed_path, \
                global_path=global_path, directory=data_path, \
                batches_per_file=batches_per_file))
    print(bcolors.OKGREEN + 'Done.' + bcolors.ENDC)
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
    file_object = wavio.read(clean_path)

    #Calculate bit depth for normalization
    bitdepth_divisor = float(2**(file_object.sampwidth*8 - 1))
    x = file_object.data
    y = wavio.read(file_info.global_path).data

    assert (x.shape[1] == y.shape[1]), "Clean file has different number of channels than non-clean file."
    num_channels = x.shape[1]

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
    y = y.astype(np.float32, order='C') / bitdepth_divisor
    x = x.astype(np.float32, order='C') / bitdepth_divisor

    # Dimension 0 is the number of frame_sizes that fit in x[0] - receptive field samples
    # Dimension 1 is framesize + receptive field (what is necessary for lookback on a given sample)
    # Dimension 2 is channels
    new_x_shape = (int((x.shape[0]-model.R)/model.frame_size), model.frame_size + model.R, num_channels)

    # stride specifies how many bits we have to move in each dimension for new view
    xstride = (model.frame_size*4*num_channels, 4*num_channels, 4)
    x = np.lib.stride_tricks.as_strided(x, new_x_shape, xstride)


    # Dimension 0 is the number of frame_sizes that fit in y[0] samples
    # Dimension 1 is framesize  (what is necessary for lookback on a given sample)
    # Dimension 2 is left and right channels
    new_y_shape = (int((y.shape[0])/model.frame_size), model.frame_size, num_channels)
    ystride = (model.frame_size*4*num_channels, 4*num_channels, 4)
    y = np.lib.stride_tricks.as_strided(y, new_y_shape, ystride)

    return x, y

# @brief: Trains model for one mini-batch.
def train_minibatch(model, input, ground_truth, mini_batch_size, i):
    with tf.GradientTape(persistent=True) as tape:
        model_prediction = model(input)
        loss = model.loss(model_prediction, ground_truth) / mini_batch_size

    if (i % 100 == 0):
        if (loss < 1.0):
            print("loss on mini-batch " + str(i) + ": " + str(loss))
        else:
            print(bcolors.WARNING + bcolors.BOLD + "Warning: loss uncharacteristically high: " + str(loss) + bcolors.ENDC)
            print("magnitude of ground truth signal: ", tf.reduce_sum(ground_truth * ground_truth))
            print("magnitude of prediction: ", tf.reduce_sum(model_prediction * model_prediction))
            # print("ground truth:", input[0])
            # print("prediction:", model_prediction[0])


    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# @brief: Trains model for one batch with inputs x and ground truth y.
def train_batch(model, x, y, mini_batch_size=32):
    # Loop through the batch splitting into mini batches. Use a list
    # comprehension to speed things up.
    _ = [train_minibatch(model, \
        x[i*mini_batch_size:(i+1)*mini_batch_size], \
        y[i*mini_batch_size:(i+1)*mini_batch_size], \
        mini_batch_size, i) for i in range(0, int(x.shape[0]/mini_batch_size))]

# @brief: Trains model for one epoch on files in data index index.
def train_epoch(model, index, start=0.0, end=1.0):
    # Iterate over the batches in a random order.
    # TODO: To make this fully random we could create some kind of matrix
    # of iteration order. Right now it doesn't really go through in a random
    # order.
    idx_start = int(start * len(index[0].batches_processed))
    idx_end = int(end * len(index[0].batches_processed))

    # Create a different random batch order for each file.
    random_batch_order = [list(range(idx_start, idx_end)) for _ in range(len(index))]
    for i in range(len(random_batch_order)):
        random.shuffle(random_batch_order[i])

    for b_i in range(len(random_batch_order[0])):
        print('')
        print(bcolors.BOLD + "Training on batch", b_i, "of total", str(len(random_batch_order[0])) + bcolors.ENDC)

        # Accumulate batch from each parameter file.
        x = None
        y = None
        num_channels = None
        for f in range(len(index)):
            file_info = index[f]
            # Get the training pair of clean and distorted data for this file
            # info and batch struct.
            f_batch = (random_batch_order[f])[b_i]
            x_f, y_f = get_input_processed_pair(model, file_info, f_batch, len(file_info.batches_processed))

            # Take note of the number of audio channels of our input
            if not num_channels:
                num_channels = x_f.shape[-1]

            # Tile parameters to be the same dimensions as x.
            if (file_info.parameters is not None):
                params = np.tile(file_info.parameters, x_f.shape[:-1] + tuple([1]))
                # Stitch the parameters vector onto the clean data as new channels.
                x_f = np.concatenate([x_f, params], axis=2)

            if x is None:
                x = x_f
                y = y_f
            else:
                x = np.concatenate([x, x_f], axis=0)
                y = np.concatenate([y, y_f], axis=0)

            # Set flag to true indicating that this was processed.
            file_info.batches_processed[f_batch] = True

        # set model's channels to the number of channels we see in the input
        if not model.channelSet:
            model.num_channels = num_channels
            print(bcolors.OKBLUE + "Set model's number of channels to", str(num_channels) + bcolors.ENDC)

        # Shuffle inputs and ground truth in the same order.
        shuffle_order = list(range(x.shape[0]))
        random.shuffle(shuffle_order)
        x = x[shuffle_order,:,:]
        y = y[shuffle_order,:,:]

        print("x shape: ", x.shape)
        print("y shape: ", y.shape)

        # Train the model on this batch.
        train_batch(model, np.array(x, dtype=np.float32), np.array(y, dtype=np.float32))

        # Do a test.
        # if (b_i % 12 == 0):
        #     test_width = 0.3
        #     test_start = 0.1
        #     loss = test(model, index=index, start=test_start, end=test_start+test_width)
        #     print(bcolors.BOLD + bcolors.OKGREEN + \
        #         "Loss on random contiguous train data subset", \
        #         str(test_start) + ", " + str(test_start + test_width), \
        #         "for batch " + str(b_i) + ":", loss, bcolors.ENDC)
        #     print('')


# @brief: Trains the model on the data in the folder data_path for specified
# number of epochs. Checkpoints model after every epoch.
def train(model, data_path, model_store_path, epochs, start=0.0, end=1.0):
    # Get the data index, which is a list of FileInfo objects that specify
    # the path to each file, the parameters associated with the file, and
    # a list of booleans specified whether or not each chunk in the file
    # has been processed.
    data_index = get_data_index(data_path)

    # Train for specified number of epochs.
    for i in range(epochs):
        print(bcolors.BOLD + bcolors.OKGREEN + "EPOCH", str(i), bcolors.ENDC)
        # Train for an epoch.
        start_time = time.time()
        train_epoch(model, data_index, start=start, end=end)
        end_time = time.time()
        print("Epoch took", (end_time - start_time) / 3600, "hours")
        # Clear out the data index.
        clear_data_index(data_index)
        # Save the model.
        if model_store_path is not None:
            print(bcolors.OKGREEN + "Saving model at path ", model_store_path + "..." + bcolors.ENDC)
            model.save_weights(model_store_path + "_epoch_" + str(i), save_format='tf')
            print(bcolors.OKGREEN + "Done." + bcolors.ENDC)
        # Do a test on a subset of the training data.
        print(bcolors.BOLD + "Computing test loss..." + bcolors.ENDC)
        test_width = 1.0
        test_start = 0.0
        start_time = time.time()
        loss = test(model, index=data_index, start=test_start, end=test_start+test_width)
        end_time = time.time()
        print(bcolors.BOLD + bcolors.OKGREEN + \
            "Loss on training set", str(test_start) + ", " + str(test_start + test_width), " for epoch " + str(i) + ":", \
            loss, bcolors.ENDC)
        print("Test on training set took", (end_time - start_time) / 3600, "hours")
        print('')

    print(bcolors.BOLD + bcolors.OKGREEN + "DONE TRAINING" + bcolors.ENDC)

# @brief: tests model for one batch with inputs x and ground truth y.
# Computes AVERAGE loss.
def test_batch(model, x, y, mini_batch_size=32):
    # Loop through the batch splitting into mini batches.
    total_loss = 0
    for i in range(0, int(x.shape[0]/mini_batch_size)):
        # Grab the input and corresponding ground truth batches.
        batch_start = i*mini_batch_size
        batch_end = (i+1)*mini_batch_size
        input = x[batch_start:batch_end]
        ground_truth = y[batch_start:batch_end]

        model_prediction = model(input)
        loss = model.loss(model_prediction, ground_truth) / mini_batch_size

        total_loss += loss
    return total_loss / np.floor(x.shape[0]/mini_batch_size)

# @brief: tests the model on data specified by data_path or
# precomputed data index.
def test(model, data_path=None, index=None, start=0.0, end=0.2):
    # Get the data index, which is a list of FileInfo objects that specify
    # the path to each file, the parameters associated with the file, and
    # a list of booleans specified whether or not each chunk in the file
    # has been processed.
    if index == None:
        index = get_data_index(data_path)
    total_loss = 0.0
    i = 0
    start = int(start * (len(index[0].batches_processed)))
    end = int(end * (len(index[0].batches_processed)))
    for b in range(start, end):
        for file_info in index:
            # Get the training pair of clean and distorted data for this file
            # info and batch struct.
            x, y = get_input_processed_pair(model, file_info, b, len(file_info.batches_processed))

            # Tile parameters to be the same dimensions as x.
            if (file_info.parameters is not None):
                params = np.tile(file_info.parameters, x.shape[:-1] + tuple([1]))
                # Stitch the parameters vector onto the clean data as new channels.
                x = np.array(np.concatenate([x, params], axis=2), dtype=np.float32)

            # Divide by the number of items to make sure this is average loss.
            total_loss += test_batch(model, x, y)
            i += 1.0
    return total_loss / i

# @brief: runs the model on an input signal, writes out result to
# specified output path.
def run(model, signal_path, out_path, parameters):
    # Read in the data.
    file_object = wavio.read(signal_path)
    x = file_object.data
    bitdepth_divisor = float(2**((file_object.sampwidth * 8)- 1))
    # Pad the data.
    x = np.pad(x, ((model.R, x.shape[0] % model.frame_size), (0, 0)), 'constant', constant_values=(0, 0))
    # Normalize to [-1.0, 1.0]
    x = x.astype(np.float32) / bitdepth_divisor
    # determine shape of final output
    print("input shape:", x.shape)
    num_channels = x.shape[1]
    new_x_shape = (int((x.shape[0]-model.R)/model.frame_size), model.frame_size + model.R, num_channels)

    # stride specifies how many bits we have to move in each dimension for new view
    xstride = (model.frame_size*4*num_channels, 4*num_channels, 4)

    # apply new view so that we get suffient lookback data for each example
    x = np.lib.stride_tricks.as_strided(x, new_x_shape, xstride)

    output = None
    batch_size = 32
    for i in range(int(x.shape[0]/batch_size)):
        # Collect batch.
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size
        input = x[batch_start:batch_end]

        # Tile parameters to be the same dimensions as x.

        if (parameters.ndim > 0):
            params = np.tile(parameters, input.shape[:-1] + tuple([1]))
            # Stitch the parameters vector onto the clean data as new channels.
            input = np.concatenate([input, params], axis=2)

        input = np.array(input, dtype=np.float32)

        # Run the model.
        model_prediction = model(input)[...,0:1] # HACK: for some reason model seems to be stereo.


        # Append to output list.
        if output is None:
            output = model_prediction
        else:
            output = np.concatenate([output, model_prediction], axis=0)

    # Scale to int16 range.
    output = output * 32768.0

    output = output.reshape(-1, output.shape[-1])
    print("output shape: ", output.shape)

    # Convert to int-16 and write out.
    output = output.astype(np.int16)
    wavio.write(out_path, output, 44100)

# @brief: main function.
def main():

    # Parse arguments from command line. This function guarantees that we
    # have everything we need for the mode we are running in.
    args = parse_command_line()

    # Create the model.
    model = setup_model(args)

    # Train the model.
    if (args.mode == 'TRAIN'):
        start = time.time()
        train(model, data_path=args.train_data_path, \
            model_store_path=args.model_store_path,
            epochs=args.epochs)
        end = time.time()
        print("Training took", (end - start) / 3600, "hours")

    # Test the model.
    if ((args.mode == 'TEST' or args.mode == 'TRAIN') and args.test_data_path is not None):
        print(bcolors.BOLD + bcolors.OKGREEN + "Computing loss on test data." + bcolors.ENDC)
        loss = test(model, data_path=args.test_data_path, start=0.0, end=1.0)
        print(bcolors.BOLD + bcolors.OKGREEN + "Final loss on test data:", loss, bcolors.ENDC)

    # TODO: run the model.
    if (args.mode == 'RUN'):
        run(model, args.signal_path, args.out_path, np.array(args.parameters))
        print(bcolors.BOLD + bcolors.OKGREEN + "Wrote out result to", args.out_path, bcolors.ENDC)

if __name__ == '__main__':
   main()
