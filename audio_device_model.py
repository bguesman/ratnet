import numpy as np
import tensorflow as tf
from audio_utils import hpf

"""

Brad: This class is where we define the network itself as an object.

"""

class AudioDeviceModel(tf.keras.Model):
    def __init__(self, learning_rate=1e-3, frame_size=128):

        # Brad: I don't know why this is here, but I'm afraid to touch it.
        ######vvv DO NOT CHANGE vvvv##############
        super(AudioDeviceModel, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################

        # Define the optimizer we want to use to train the model.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # How many samples we are trying to predict at once.
        self.frame_size = frame_size

        # Set number of signal channels
        self.num_channels = 1

        # Set num channels flag
        self.channelSet = False

        # Set the parameters for each convolutional layer.
        self.d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.k = [3 for _ in self.d]
        self.chan = [16 for _ in self.d]

        # Compute the receptive field.
        self.R = sum(d * (k - 1) for d, k in zip(self.d, self.k)) + 1

        # Layer input mixers.
        self.io = []
        # Convolutional layers.
        self.c = []

        for i in range(len(self.d)):
            # Convolutional layer.
            self.c.append(tf.keras.layers.Conv1D(filters=self.chan[i],
                kernel_size=self.k[i], dilation_rate=self.d[i], padding="causal"))

            # IO mixer (convolutional layer with kernel size 1). Final
            # layer does not need one.
            if (i != len(self.d) - 1):
                self.io.append(tf.keras.layers.Conv1D(filters=self.chan[i],
                    kernel_size=1, padding="causal"))

        # Nonlinearity.
        self.lr = tf.keras.layers.ReLU()

        # Linear mixer (convolutional layer with kernel size 1).
        self.mixer = tf.keras.layers.Conv1D(filters=self.num_channels, kernel_size=1, padding="same")

    @tf.function
    def call(self, input):
        # Accumulator variable that we'll use to implement skip connections.
        accumulator = None
        signal = input[:,:,0:self.num_channels]
        controls = input[:,:,self.num_channels:]
        for i in range(len(self.c)):
            # Apply convolutional layer i and non-linearity
            layer_output = self.c[i](tf.concat([signal, controls], axis=2))
            layer_output_nonlinear = self.lr(layer_output)

            # Add the result to the total output of the network. This is a
            # "skip connection".
            if accumulator is None:
                accumulator = layer_output_nonlinear
            else:
                accumulator = tf.concat([accumulator, layer_output_nonlinear], axis=2)
            if (i != len(self.c) - 1):
                # Only compute input if there's a next layer to feed it to.
                # Blend between the output and input via a 1x1 convolution.
                signal = self.io[i](layer_output_nonlinear) + tf.expand_dims((tf.reduce_sum(signal, axis=2)/2), axis=2)

        # Concatenate along the channel dimension and apply the 1x1
        # convolution.
        mixed = self.mixer(accumulator)
        # Squeeze out the last few samples and take only the last frame_size
        # frames.
        return mixed[:,-self.frame_size:, :]

    def loss(self, prediction, ground_truth):
        # High pass both signals.
        hpf_ground_truth = hpf(ground_truth)
        hpf_prediction = hpf(prediction)

        # Compute normalized L2 loss.
        l2 = tf.reduce_sum((hpf_prediction - hpf_ground_truth) ** 2)
        denominator = tf.reduce_sum(hpf_ground_truth ** 2)
        if (denominator != 0.0):
            # Avoid dividing by zero.
            l2 = l2 / denominator

        # Compute dc offset loss.
        N = tf.size(prediction, out_type=tf.dtypes.float32)
        dc_offset = (1.0/N) * tf.reduce_sum(hpf_prediction - hpf_ground_truth)
        dc_offset = dc_offset * dc_offset
        if (denominator != 0.0):
            dc_offset = (N * dc_offset) / denominator

        return l2 + dc_offset
