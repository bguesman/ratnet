import numpy as np
import tensorflow as tf
from preprocess import *
from audio_device_model import AudioDeviceModel
import os
import sys
import time


def train(model, train_inputs, train_ground_truth):
    """
    This runs through one epoch of the training process. It takes as input:

    @param model: the model object we're trying to train.

    @param train_inputs: vector of all the inputs we want to train on.

    @param train_ground_truth: vector of all the corresponding ground truth
        outputs to use when computing the loss.

    Doesn't return anything, since we're just modifying the model object.

    """

    # Loop through all the examples.
    for i in range(train_inputs.shape[0]):
        # Grab the input and corresponding ground truth. This would usually be
        # batched but for example I'm just grabbing one input.
        input = train_inputs[i]
        ground_truth = train_ground_truth[i]

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

def test(model, test_inputs, test_ground_truth):
    """
    This computes the average loss across the testing sample.
    """
    total_loss = 0
    for i in range(train_inputs.shape[0]):
        # Grab the input and corresponding ground truth. This would usually be
        # batched but for example I'm just grabbing one input.
        input = test_inputs[i]
        ground_truth = test_ground_truth[i]

        # Run the model on the input to get the predicted output.
        model_prediction = model(input)
        # Compute the loss.
        total_loss += model.loss(model_prediction, ground_truth)
    return total_loss / train_inputs.shape[0]

def main():
    """
    Main function... so where we actually run everything from.
    """
    # Get the data using preprocess.py.
    train_inputs, train_ground_truth, test_inputs, test_ground_truth = get_data("some/filepath/butts/")

    # Create the model object. See audio_device_model.py.
    model = AudioDeviceModel()

    # Train the model for some number of epochs, and time how long it takes.
    epochs = 5
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
