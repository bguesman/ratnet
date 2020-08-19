import numpy as np
import tensorflow as tf

"""

Brad: This class is where we define the network itself as an object.

"""

class AudioDeviceModel(tf.keras.Model):
    def __init__(self):

        # Brad: I don't know why this is here, but I'm afraid to touch it.
        ######vvv DO NOT CHANGE vvvv##############
        super(AudioDeviceModel, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################

        # Brad: here's an example of what the member variables might look
        # like for a model.

        # Brad: we define the optimizer we want to use to train the model.
        #
        # Theoretically, we could do this outside the model, in
        # "model-runner.py". But it's good to have it set up in here because
        # optimization algorithms like Adam maintain state across training
        # examples (just like, "how fast should I be descending right now").
        #
        # Basically, if we set it up outside of the model, it would
        # need to be a global variable defined outside the training loop.
        # But setting it up here makes sure we can't make a mistake that
        # breaks the optimizer's state maintenance.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

        # Brad: here's where the real bulk of the model definition happens.
        # We define a member variable for each layer in the network. This
        # example just has 2 two dimensional convolutional layers. For our
        # model, we'll have a member variable for each 1D dilated convolution
        # layer.

        # TODO: do these automatically include a bias term?
        self.conv1 = tf.keras.layers.Conv2D(
            filters=2,      # <-- How many channels do we output.
            kernel_size=5,  # <-- How big is our filter kernel.
            strides=2,      # <-- How big is our stride.
            padding='same', # <-- "Same" or "Zero" padding?
            activation=tf.keras.layers.LeakyReLU(alpha=0.2) # <-- Non-linear function applied after the convolution.
        )
        self.conv2 = tf.keras.layers.Conv2D(filters=4,
            kernel_size=5,
            strides=2,
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2)
        )

    @tf.function
    def call(self, input):
        # This function actually runs the model on a given input.
        # Because we use the layers API, this is super easy.

        # Apply convolutional layer 1 to the input.
        conv1 = self.conv1(input)
        # Apply convolutional layer 2 to the output of convolutional layer 1.
        conv2 = self.conv2(conv1)
        # Return the final result.
        return conv2

    def loss(self, prediction, ground_truth):
        # This is the model's loss function, given a vector of predictions and
        # a corresponding vector of ground truths.

        # Here's an example of L2 loss.
        tf.reduce_sum((prediction - ground_truth) ** 2)
