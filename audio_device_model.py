import numpy as np
import tensorflow as tf
from audio_utils import hpf

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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Brad: here's where the real bulk of the model definition happens.
        # We define a member variable for each layer in the network. This
        # example just has 2 linear layers.

        self.c1 = tf.keras.layers.Conv1D(filters=1, kernel_size=2, dilation_rate=1, padding="same")
        self.lr1 = tf.keras.layers.LeakyReLU()
        self.c2 = tf.keras.layers.Conv1D(filters=1, kernel_size=2, dilation_rate=4, padding="same")
        self.lr2 = tf.keras.layers.LeakyReLU()
        self.c3 = tf.keras.layers.Conv1D(filters=1, kernel_size=2, dilation_rate=8, padding="same")
        self.lr3 = tf.keras.layers.LeakyReLU()
        self.c4 = tf.keras.layers.Conv1D(filters=1, kernel_size=2, dilation_rate=32, padding="same")
        self.lr4 = tf.keras.layers.LeakyReLU()
        self.c5 = tf.keras.layers.Conv1D(filters=1, kernel_size=2, dilation_rate=128, padding="same")
        self.d1 = tf.keras.layers.Dense(128);

    @tf.function
    def call(self, input):
        # This function actually runs the model on a given input.
        # Because we use the layers API, this is super easy.

        # Apply conv1d layer 1 to the input.
        c1 = self.c1(tf.expand_dims(input, axis=1))
        # Apply conv1d layer 2 to the output of conv1d layer 1.
        c2 = self.c2(self.lr1(c1))
        # Apply conv1d layer 3 to the output of conv1d layer 2.
        c3 = self.c3(self.lr2(c2))
        # Apply conv1d layer 4 to the output of conv1d layer 3.
        c4 = self.c4(self.lr3(c3))
        # Apply conv1d layer 4 to the output of conv1d layer 3.
        c5 = self.c5(self.lr4(c4))
        # Return the final result.
        return tf.squeeze(self.d1(c1 + c2 + c3 + c4 + c5))

    def loss(self, prediction, ground_truth):
        # This is the model's loss function, given a vector of predictions and
        # a corresponding vector of ground truths.

        # High pass both signals.
        hpf_ground_truth = hpf(ground_truth)
        hpf_prediction = hpf(prediction)

        # This is the "error to signal ratio" they describe in the paper.
        # It's just a normalized L2 loss.
        # Avoid dividing by zero.
        if (tf.reduce_sum(hpf_ground_truth ** 2) == 0.0):
            # TODO: is this the right thing to do though?
            return tf.reduce_sum((hpf_prediction - hpf_ground_truth) ** 2)
        return tf.reduce_sum((hpf_prediction - hpf_ground_truth) ** 2) / tf.reduce_sum(hpf_ground_truth ** 2)
