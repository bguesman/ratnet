import numpy as np
import tensorflow as tf
from preprocess import *
from audio_device_model import AudioDeviceModel
import os
import sys
import time
import random
from audio_utils import hpf


def train(model, train_inputs, train_ground_truth, batch_size=10):
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
    for i in range(int(train_inputs.shape[0]/batch_size)):
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

            # Apply in-place hpf.
            # TODO: should the hpf be something we apply in the loss
            # function? I think that could actually make sense, since it's
            # kind of like a part of the loss function.
            model_prediction = hpf(model_prediction)

            # Compute the loss. Since this is also happening in the gradient
            # tape scope, it now has recorded all the derivatives of the loss
            # w.r.t. the model parameters.
            loss = model.loss(model_prediction, ground_truth)

        # The gradient tape now has the derivatives of the loss function
        # w.r.t. to the model parameters. We grab them here.
        gradients = tape.gradient(loss, model.trainable_variables)
        # Now, we update the model parameters according to the gradients,
        # using the optimizer we defined in the model.
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # This is called "checkpointing". It will save the weights for us
        # so we can load them later. TODO: I think this overwrites it every
        # time tho.
        if (i % 40 == 0):
           model.save_weights('model_weights/model_weights', save_format='tf')

        # At this point we would usually test our model on some random
        # chunk of the training data and print out the loss. Just to keep
        # track of how it's doing. TODO: I think this chunk needs to be
        # contiguous, since our model has state? If we store state in the
        # model, will this fuck up that state and fuck up the results?

        if (i % 200 == 0):
            # Random test.
            test_size = 100 * batch_size
            random_index = int(random.uniform(0, 1) * (train_inputs.shape[0]-test_size))
            test_inputs = train_inputs[random_index:random_index+test_size]
            test_ground_truth = train_ground_truth[random_index:random_index+test_size]
            test_result = test(model, test_inputs, test_ground_truth, batch_size)
            print("LOSS on iteration ", i, ": ", test_result)

def test(model, test_inputs, test_ground_truth, batch_size=10):
    """
    This computes the average loss across the testing sample.
    """
    total_loss = 0
    for i in range(int(test_inputs.shape[0]/batch_size)):
        # Grab the input and corresponding ground truth. TODO: we can batch
        # this if we want to make it faster.
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size
        input = test_inputs[batch_start:batch_end]
        ground_truth = test_ground_truth[batch_start:batch_end]

        # Run the model on the input to get the predicted output.
        model_prediction = model(input)

        # Apply in-place hpf.
        # TODO: should the hpf be something we apply in the loss
        # function? I think that could actually make sense, since it's
        # kind of like a part of the loss function.
        model_prediction = hpf(model_prediction)

        # Compute the loss.
        total_loss += model.loss(model_prediction, ground_truth)
    return total_loss / test_inputs.shape[0]

def main():
    """
    Main function... so where we actually run everything from.
    """

    filepath = sys.argv[1]

    print("Getting and preprocessing audio data from path " + filepath)
    start = time.time()
    # Get the data using preprocess.py.
    train_inputs, train_ground_truth, test_inputs, test_ground_truth = get_data(filepath)
    print("train inputs shape: ", train_inputs.shape)
    print("train ground truth shape: ", train_ground_truth.shape)
    print("test inputs shape: ", test_inputs.shape)
    print("test ground truth shape: ", test_ground_truth.shape)
    end = time.time()
    print("Done getting audio data, took", (end - start) / 60, "minutes.")

    # Create the model object. See audio_device_model.py.
    model = AudioDeviceModel()

    # Train the model for some number of epochs, and time how long it takes.
    epochs = 1
    start = time.time()

    for _ in range(epochs):
        train(model, train_inputs, train_ground_truth)

    end = time.time()
    print("Done training, took", (end - start) / 60, "minutes.")

    # Test the model.
    test_loss = test(model, test_inputs, test_ground_truth)
    print("FINAL LOSS ON TEST DATA:", test_loss)

    # Save the weights for later use.
    # model.save_weights('model_weights/model_weights', save_format='tf')

if __name__ == '__main__':
   main()
