import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
import wavio
import argparse
from audio_device_model import AudioDeviceModel

# Utility class for printing different colors to the terminal.
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

# Utility class for storing info about each file.
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

# Given a file info object and batch number, returns the clean and processed
# batch. We need the model so we can get the frame size and receptive field.
def get_training_pair(model, file_info, batch, total_batches):
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

# Trains model for one batch with inputs x and ground truth y.
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
            loss = model.loss(model_prediction, tf.squeeze(ground_truth))

        if (i % 100 == 0):
            print("Loss on iteration " + str(i) + ": " + str(loss))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Trains model for one epoch on files in data index index.
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
            x, y = get_training_pair(model, file_info, b, len(file_info.batches_processed))

            # Tile parameters to be the same dimensions as x.
            params = np.tile(file_info.parameters, x.shape)
            # Stitch the parameters vector onto the clean data as new channels.
            x = np.concatenate([x, params], axis=2)

            # Train the model on this batch.
            train_batch(model, np.array(x, dtype=np.float32), np.array(y, dtype=np.float32))

            # Set flag to true indicating that this was processed.
            file_info.batches_processed[b] = True

# Trains the model on the data in the folder data_path for specified number
# of epochs. Checkpoints model after every epoch.
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
        pass

    # TODO: run the model.
    if (args.mode == 'RUN'):
        pass

if __name__ == '__main__':
   main()

# def train(model, train_inputs, train_ground_truth, model_discriminator=None, batch_size=32, pruning=False):
#     """
#
#     This runs through one epoch of the training process. It takes as input:
#
#     @param model: the model object we're trying to train.
#
#     @param train_inputs: vector of all the inputs we want to train on.
#
#     @param train_ground_truth: vector of all the corresponding ground truth
#         outputs to use when computing the loss.
#
#     @param batch_size: how many examples to use to compute the gradients.
#
#     @return: nothing, since we're just modifying the model object.
#
#     """
#
#     # Loop through all the batches.
#     for i in range(0, int(train_inputs.shape[0]/batch_size)):
#         # Grab the input and corresponding ground truth batches.
#         batch_start = i*batch_size
#         batch_end = (i+1)*batch_size
#         input = train_inputs[batch_start:batch_end]
#         ground_truth = train_ground_truth[batch_start:batch_end]
#
#         # Start a "gradient tape".
#         #
#         # This stores all the derivatives for every calculation that takes
#         # place in its scope.
#         with tf.GradientTape(persistent=True) as tape:
#             # Run the model on the input to get the predicted output. Since
#             # this is happening in the scope of the gradient tape, all the
#             # derivatives of the output w.r.t. every parameter of the model
#             # are stored.
#             model_prediction = model(input)
#
#             # Compute the loss. Since this is also happening in the gradient
#             # tape scope, it now has recorded all the derivatives of the loss
#             # w.r.t. the model parameters.
#             loss = model.loss(model_prediction, ground_truth)
#
#             # Use the discriminator
#             if (model_discriminator is not None):
#                 frame_size = model_prediction.shape[1]
#                 in_frame = input[:,-frame_size:]
#                 discriminator_input = tf.concat([in_frame, model_prediction], axis=0)
#                 discriminator_prediction = model_discriminator(discriminator_input)
#                 discriminator_ground_truth = tf.one_hot(tf.concat([tf.ones(batch_size, dtype=tf.int32), tf.zeros(batch_size, dtype=tf.int32)], axis=0), 2)
#                 discriminator_loss = model_discriminator.loss(discriminator_prediction, discriminator_ground_truth)
#
#                 # Subtract from the total loss---when the discriminator does poorly,
#                 # we do really well! When it does well, we do poorly.
#                 loss -= 0.1 * discriminator_loss
#
#         if (model_discriminator is not None):
#             if (i % 2 == 0):
#                 # Do the same with the discriminator.
#                 discriminator_gradients = tape.gradient(discriminator_loss, model_discriminator.trainable_variables)
#                 model_discriminator.optimizer.apply_gradients(zip(discriminator_gradients, model_discriminator.trainable_variables))
#             else:
#                 # The gradient tape now has the derivatives of the loss function
#                 # w.r.t. to the model parameters. We grab them here.
#                 gradients = tape.gradient(loss, model.trainable_variables)
#                 # Now, we update the model parameters according to the gradients,
#                 # using the optimizer we defined in the model.
#                 model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         else:
#             # The gradient tape now has the derivatives of the loss function
#             # w.r.t. to the model parameters. We grab them here.
#             gradients = tape.gradient(loss, model.trainable_variables)
#             # Now, we update the model parameters according to the gradients,
#             # using the optimizer we defined in the model.
#             model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#         # This is called "checkpointing". It will save the weights for us
#         # so we can load them later. TODO: I think this overwrites it every
#         # time tho.
#         if (i % 80 == 0 and not pruning):
#            model.save_weights('model_weights/model_weights', save_format='tf')
#
#         # At this point we would usually test our model on some random
#         # chunk of the training data and print out the loss. Just to keep
#         # track of how it's doing. TODO: I think this chunk needs to be
#         # contiguous, since our model has state? If we store state in the
#         # model, will this fuck up that state and fuck up the results?
#         if (i == 0 or i % 50 == 0):
#             if (model_discriminator is not None):
#                 print ("DISCRIMINATOR LOSS on iteration ", i, ": ", discriminator_loss)
#             # Random test.
#             test_size = 100 * batch_size
#             random_index = int(random.uniform(0, 1) * (train_inputs.shape[0]-test_size))
#             test_inputs = train_inputs[random_index:random_index+test_size]
#             test_ground_truth = train_ground_truth[random_index:random_index+test_size]
#             test_result = test(model, test_inputs, test_ground_truth, 500)
#             print("LOSS on iteration ", i, ": ", test_result)
#
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
# def main():
#     """
#     Main function... so where we actually run everything from.
#     """

    # TODO: turn this into an argparse template.
    # if (len(sys.argv) < 3):
    #     print('usage: python model_runner.py <mode> <filepath> <num epochs>')
    #
    # mode = sys.argv[1]
    # filepath = sys.argv[2]
    #
    # if mode not in ['TRAIN', 'LOAD-AND-TRAIN', 'TEST', 'RUN', 'PRUNE']:
    #     print('mode must be one of <TRAIN, LOAD-AND-TRAIN, TEST, RUN, PRUNE>')
    #     return
    #
    # # Print out working directory, since that's useful for CoLab, and
    # # print out whether or not we have access to a GPU.
    # print("Working directory: ", os.getcwd())
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #
    # # Create the model object. See audio_device_model.py.
    # model = AudioDeviceModel()
    #
    # # Get data.
    # if (mode == 'RUN'):
    #     print("Getting and preprocessing audio data from path " + filepath)
    #     start = time.time()
    #     # Get the data using preprocess.py.
    #     run_data = get_run_data(filepath, model.frame_size, model.R)
    #     end = time.time()
    #     print("Done getting audio data, took", (end - start) / 60, "minutes.")
    # else:
    #     print("Getting and preprocessing audio data from path " + filepath)
    #     start = time.time()
    #     # Get the data using preprocess.py.
    #     train_inputs, train_ground_truth, test_inputs, test_ground_truth = get_train_test_data(filepath, 128, 2047, 1)
    #     print("train inputs shape: ", train_inputs.shape)
    #     print("train ground truth shape: ", train_ground_truth.shape)
    #     print("test inputs shape: ", test_inputs.shape)
    #     print("test ground truth shape: ", test_ground_truth.shape)
    #     end = time.time()
    #     print("Done getting audio data, took", (end - start) / 60, "minutes.")
    #
    # # Load weights if we are testing or resuming training.
    # if mode == "LOAD-AND-TRAIN" or mode == "TEST" or mode == "RUN" or mode == "PRUNE":
    #     print("Loading model weights...")
    #     model.load_weights('model_weights/model_weights')
    #     print("Done.")
    #
    # if mode == 'PRUNE':
    #     # Make bogus prediction to force build model.
    #     #bogus_prediction = model(train_inputs[0:2])
    #
    #     # Prune each convolutional layer.
    #     #print(model.summary())
    #     for i in range(len(model.c)):
    #         print("Pruning layer ", i)
    #         model.c[i] = tfmot.sparsity.keras.prune_low_magnitude(model.c[i])
    #         if (i < len(model.c) - 1):
    #             model.io[i] = tfmot.sparsity.keras.prune_low_magnitude(model.io[i])
    #     model.mixer = tfmot.sparsity.keras.prune_low_magnitude(model.mixer)
    #
    #     # Make another bogus prediction to force build model again.
    #     # bogus_prediction = model(train_inputs[0:2])
    #
    # # Train the model if we are training or resuming training.
    # if mode == "TRAIN" or mode == "LOAD-AND-TRAIN" or mode == 'PRUNE':
    #     # Create discriminator.
    #     pruning = (mode == 'PRUNE')
    #     use_discriminator = (sys.argv[3] == "TRUE")
    #     model_discriminator = None
    #     if (use_discriminator):
    #         model_discriminator = AudioDeviceDiscriminator()
    #
    #     # Train the model for some number of epochs, and time how long it takes.
    #     epochs = int(sys.argv[4])
    #     start = time.time()
    #     for i in range(epochs):
    #         print("EPOCH ", i)
    #         shuffle_order = list(range(train_inputs.shape[0]))
    #         random.shuffle(shuffle_order)
    #         X = (train_inputs.numpy())[shuffle_order,:,:]
    #         y = (train_ground_truth.numpy())[shuffle_order,:]
    #         train(model, X, y, model_discriminator, pruning=pruning)
    #         now = time.time()
    #         print("Been training for ", (now - start) / 60 , " minutes.")
    #     end = time.time()
    #     print("Done training, took", (end - start) / 60, "minutes.")
    #
    #     # Save the weights for later use.
    #     if not pruning:
    #         model.save_weights('model_weights/model_weights', save_format='tf')
    #
    # if mode == 'RUN':
    #     # Write out a wav file.
    #     print("Processing file...")
    #     start = time.time()
    #     test_wav(model, run_data, "test", 1024)
    #     end = time.time()
    #     data_length_in_minutes = (run_data.shape[0] * model.frame_size) / (44100 * 60)
    #     processing_time_in_minutes = (end-start)/60
    #     print("Done. Took ", processing_time_in_minutes, " minutes.")
    #     print("This is ", data_length_in_minutes/processing_time_in_minutes, "x realtime.")
    #     print("Wrote out wav to test_wet.wav")
    # else:
    #     # Test the model.
    #     test_loss = test(model, test_inputs, test_ground_truth, 1024)
    #     print("FINAL LOSS ON TEST DATA:", test_loss)

# if __name__ == '__main__':
#    main()
