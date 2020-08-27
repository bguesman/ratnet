import numpy as np
import tensorflow as tf
from audio_utils import hpf

class AudioDeviceDiscriminator(tf.keras.Model):
    def __init__(self):

        # Brad: I don't know why this is here, but I'm afraid to touch it.
        ######vvv DO NOT CHANGE vvvv##############
        super(AudioDeviceDiscriminator, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################

        # Define the optimizer we want to use to train the model.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # How many samples we are trying to predict at once.
        self.frame_size = 128

        # For now, just use a few causal convolutional layers.
        self.c1 = tf.keras.layers.Conv1D(filters=4, kernel_size=5, dilation_rate=1, padding="causal")
        self.c2 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, dilation_rate=4, padding="causal")
        self.c3 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, dilation_rate=8, padding="causal")
        self.c4 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, dilation_rate=16, padding="causal")
        self.f = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.d2 = tf.keras.layers.Dense(128, activation="relu")
        self.d3 = tf.keras.layers.Dense(2)

        # Since activations have no learnable parameters, I think we only
        # need one that we can reuse?
        self.lr = tf.keras.layers.ReLU()

    @tf.function
    def call(self, input):
        # high pass to emphasize high frequencies
        # EDIT: try not high passing.
        # input = hpf(input)
        input = tf.expand_dims(input, axis=2)
        # do the convolution
        # TODO: there should probably be skip connections here?
        conv_out = self.c4(self.lr(self.c3(self.lr(self.c2(self.lr(self.c1(input)))))))
        # flatten and apply dense layer
        return self.d3(self.d2(self.d1(self.f(conv_out))))

    def loss(self, prediction, ground_truth):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(ground_truth, prediction))
