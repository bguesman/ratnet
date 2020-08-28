import numpy as np
import tensorflow as tf
from preprocess import *
from audio_device_model import AudioDeviceModel
from audio_device_discriminator import AudioDeviceDiscriminator
import os
import sys
import time
import random
import tensorflow_model_optimization as tfmot

def train(model, train_inputs, train_ground_truth, model_discriminator=None, batch_size=32, pruning=False):
    """

    This runs through one epoch of the training process. It takes as input:

    @param model: the model object we're trying to train.

    @param train_inputs: vector of all the inputs we want to train on.

    @param train_ground_truth: vector of all the corresponding ground truth
        outputs to use when computing the loss.

    @param batch_size: how many examples to use to compute the gradients.

    @return: nothing, since we're just modifying the model object.

    """

    # Loop through all the batches.
    for i in range(0, int(train_inputs.shape[0]/batch_size)):
        # Grab the input and corresponding ground truth batches.
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size
        input = train_inputs[batch_start:batch_end]
        ground_truth = train_ground_truth[batch_start:batch_end]

        # Start a "gradient tape".
        #
        # This stores all the derivatives for every calculation that takes
        # place in its scope.
        with tf.GradientTape(persistent=True) as tape:
            # Run the model on the input to get the predicted output. Since
            # this is happening in the scope of the gradient tape, all the
            # derivatives of the output w.r.t. every parameter of the model
            # are stored.
            model_prediction = model(input)

            # Compute the loss. Since this is also happening in the gradient
            # tape scope, it now has recorded all the derivatives of the loss
            # w.r.t. the model parameters.
            loss = model.loss(model_prediction, ground_truth)

            # Use the discriminator
            if (model_discriminator is not None):
                frame_size = model_prediction.shape[1]
                in_frame = input[:,-frame_size:]
                discriminator_input = tf.concat([in_frame, model_prediction], axis=0)
                discriminator_prediction = model_discriminator(discriminator_input)
                discriminator_ground_truth = tf.one_hot(tf.concat([tf.ones(batch_size, dtype=tf.int32), tf.zeros(batch_size, dtype=tf.int32)], axis=0), 2)
                discriminator_loss = model_discriminator.loss(discriminator_prediction, discriminator_ground_truth)

                # Subtract from the total loss---when the discriminator does poorly,
                # we do really well! When it does well, we do poorly.
                loss -= 0.1 * discriminator_loss

        if (model_discriminator is not None):
            if (i % 2 == 0):
                # Do the same with the discriminator.
                discriminator_gradients = tape.gradient(discriminator_loss, model_discriminator.trainable_variables)
                model_discriminator.optimizer.apply_gradients(zip(discriminator_gradients, model_discriminator.trainable_variables))
            else:
                # The gradient tape now has the derivatives of the loss function
                # w.r.t. to the model parameters. We grab them here.
                gradients = tape.gradient(loss, model.trainable_variables)
                # Now, we update the model parameters according to the gradients,
                # using the optimizer we defined in the model.
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            # The gradient tape now has the derivatives of the loss function
            # w.r.t. to the model parameters. We grab them here.
            gradients = tape.gradient(loss, model.trainable_variables)
            # Now, we update the model parameters according to the gradients,
            # using the optimizer we defined in the model.
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # This is called "checkpointing". It will save the weights for us
        # so we can load them later. TODO: I think this overwrites it every
        # time tho.
        if (i % 80 == 0 and not pruning):
           model.save_weights('model_weights/model_weights', save_format='tf')

        # At this point we would usually test our model on some random
        # chunk of the training data and print out the loss. Just to keep
        # track of how it's doing. TODO: I think this chunk needs to be
        # contiguous, since our model has state? If we store state in the
        # model, will this fuck up that state and fuck up the results?
        if (i == 0 or i % 50 == 0):
            if (model_discriminator is not None):
                print ("DISCRIMINATOR LOSS on iteration ", i, ": ", discriminator_loss)
            # Random test.
            test_size = 100 * batch_size
            random_index = int(random.uniform(0, 1) * (train_inputs.shape[0]-test_size))
            test_inputs = train_inputs[random_index:random_index+test_size]
            test_ground_truth = train_ground_truth[random_index:random_index+test_size]
            test_result = test(model, test_inputs, test_ground_truth, 500)
            print("LOSS on iteration ", i, ": ", test_result)

def test(model, test_inputs, test_ground_truth, batch_size=32):
    """
    This computes the average loss across the testing sample.
    """
    total_loss = 0
    for i in range(0, int(test_inputs.shape[0]/batch_size)):
        # Grab the input and corresponding ground truth. TODO: we can batch
        # this if we want to make it faster.
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size
        input = test_inputs[batch_start:batch_end]
        ground_truth = test_ground_truth[batch_start:batch_end]

        # Run the model on the input to get the predicted output.
        model_prediction = model(input)

        # Compute the loss.
        total_loss += model.loss(model_prediction, ground_truth)
    return total_loss / float(int(test_inputs.shape[0]/batch_size))

def test_wav(model, test_inputs, out_path, batch_size=32):
    output = np.zeros((test_inputs.shape[0], 128))
    for i in range(0, int(test_inputs.shape[0]/batch_size)):
        # Grab the input and corresponding ground truth. TODO: we can batch
        # this if we want to make it faster.
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size
        batched_input = test_inputs[batch_start:batch_end]

        # Run the model on the input to get the predicted output.
        output[batch_start:batch_end] = model(batched_input)

    # Flatten the output.
    output = np.reshape(output, (-1))

    # Convert to wav.
    output = np.clip(output, -1.0, 1.0) * 32768.0
    output = output.astype(np.int16, order='C')
    wavio.write(out_path + "_wet.wav", output, 44100)

def main():
    """
    Main function... so where we actually run everything from.
    """

    # TODO: turn this into an argparse template.
    if (len(sys.argv) < 3):
        print('usage: python model_runner.py <mode> <filepath> <num epochs>')

    mode = sys.argv[1]
    filepath = sys.argv[2]

    if mode not in ['TRAIN', 'LOAD-AND-TRAIN', 'TEST', 'RUN', 'PRUNE']:
        print('mode must be one of <TRAIN, LOAD-AND-TRAIN, TEST, RUN, PRUNE>')
        return

    # Print out working directory, since that's useful for CoLab, and
    # print out whether or not we have access to a GPU.
    print("Working directory: ", os.getcwd())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Create the model object. See audio_device_model.py.
    model = AudioDeviceModel()

    # Get data.
    if (mode == 'RUN'):
        print("Getting and preprocessing audio data from path " + filepath)
        start = time.time()
        # Get the data using preprocess.py.
        run_data = get_run_data(filepath, model.frame_size, model.R)
        end = time.time()
        print("Done getting audio data, took", (end - start) / 60, "minutes.")
    else:
        print("Getting and preprocessing audio data from path " + filepath)
        start = time.time()
        # Get the data using preprocess.py.
        train_inputs, train_ground_truth, test_inputs, test_ground_truth = get_train_test_data(filepath, 128, 2047, 1)
        print("train inputs shape: ", train_inputs.shape)
        print("train ground truth shape: ", train_ground_truth.shape)
        print("test inputs shape: ", test_inputs.shape)
        print("test ground truth shape: ", test_ground_truth.shape)
        end = time.time()
        print("Done getting audio data, took", (end - start) / 60, "minutes.")

    # Load weights if we are testing or resuming training.
    if mode == "LOAD-AND-TRAIN" or mode == "TEST" or mode == "RUN" or mode == "PRUNE":
        print("Loading model weights...")
        model.load_weights('model_weights/model_weights')
        print("Done.")

    if mode == 'PRUNE':
        # Make bogus prediction to force build model.
        #bogus_prediction = model(train_inputs[0:2])

        # Prune each convolutional layer.
        #print(model.summary())
        for i in range(len(model.c)):
            print("Pruning layer ", i)
            model.c[i] = tfmot.sparsity.keras.prune_low_magnitude(model.c[i])
            if (i < len(model.c) - 1):
                model.io[i] = tfmot.sparsity.keras.prune_low_magnitude(model.io[i])
        model.mixer = tfmot.sparsity.keras.prune_low_magnitude(model.mixer)

        # Make another bogus prediction to force build model again.
        # bogus_prediction = model(train_inputs[0:2])

    # Train the model if we are training or resuming training.
    if mode == "TRAIN" or mode == "LOAD-AND-TRAIN" or mode == 'PRUNE':
        # Create discriminator.
        pruning = (mode == 'PRUNE')
        use_discriminator = (sys.argv[3] == "TRUE")
        model_discriminator = None
        if (use_discriminator):
            model_discriminator = AudioDeviceDiscriminator()

        # Train the model for some number of epochs, and time how long it takes.
        epochs = int(sys.argv[4])
        start = time.time()
        for i in range(epochs):
            print("EPOCH ", i)
            shuffle_order = list(range(train_inputs.shape[0]))
            random.shuffle(shuffle_order)
            X = (train_inputs.numpy())[shuffle_order,:,:]
            y = (train_ground_truth.numpy())[shuffle_order,:,:]
            train(model, X, y, model_discriminator, pruning=pruning)
            now = time.time()
            print("Been training for ", (now - start) / 60 , " minutes.")
        end = time.time()
        print("Done training, took", (end - start) / 60, "minutes.")

        # Save the weights for later use.
        if not pruning:
            model.save_weights('model_weights/model_weights', save_format='tf')

    if mode == 'RUN':
        # Write out a wav file.
        print("Processing file...")
        start = time.time()
        test_wav(model, run_data, "test", 1024)
        end = time.time()
        data_length_in_minutes = (run_data.shape[0] * model.frame_size) / (44100 * 60)
        processing_time_in_minutes = (end-start)/60
        print("Done. Took ", processing_time_in_minutes, " minutes.")
        print("This is ", data_length_in_minutes/processing_time_in_minutes, "x realtime.")
        print("Wrote out wav to test_wet.wav")
    else:
        # Test the model.
        test_loss = test(model, test_inputs, test_ground_truth, 1024)
        print("FINAL LOSS ON TEST DATA:", test_loss)

if __name__ == '__main__':
   main()
