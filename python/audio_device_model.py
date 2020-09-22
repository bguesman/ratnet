import numpy as np
import tensorflow as tf
from audio_utils import hpf, parameterized_hpf

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

        # Threshold for avoiding spurious loss calculations.
        self.loss_divisor_threshold = 0.001
        self.dc_loss_multiplier = 1
        # Set number of signal channels. HACK: this actually doesn't ever get set, have to set manually.
        self.num_channels = 1

        # Set num channels flag
        self.channelSet = False

        # Set the parameters for each convolutional layer.

        # Distortion setup:
        # self.d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # self.k = [3 for _ in self.d]
        # self.chan = [16 for _ in self.d]
        # HRTF setup:
        # self.d = [1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64]
        # self.k = [3 for _ in self.d]
        # self.chan = [16 for _ in self.d]

        # self.d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.k = [3 for _ in self.d]
        self.chan = [16 for _ in self.d]

        # Compute the receptive field.
        self.R = sum(d * (k - 1) for d, k in zip(self.d, self.k)) + 1

        # Layer input mixers.
        self.linear_io = []
        self.nonlinear_io = []
        # Convolutional layers.
        self.linear_c = []
        self.nonlinear_c = []

        for i in range(len(self.d)):
            # Convolutional layer.
            self.linear_c.append(tf.keras.layers.Conv1D(filters=self.chan[i],
                kernel_size=self.k[i], dilation_rate=self.d[i], padding="causal", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)))
            self.nonlinear_c.append(tf.keras.layers.Conv1D(filters=self.chan[i],
                kernel_size=self.k[i], dilation_rate=self.d[i], padding="causal", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)))

            if (i != len(self.d) - 1):
                self.linear_io.append(tf.keras.layers.Conv1D(filters=self.chan[i],
                    kernel_size=1, padding="causal", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)))
                self.nonlinear_io.append(tf.keras.layers.Conv1D(filters=self.chan[i],
                    kernel_size=1, padding="causal", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)))

        # Nonlinearity.
        self.lr = tf.keras.layers.ReLU()

        # Linear mixer (convolutional layer with kernel size 1).
        self.mixer = tf.keras.layers.Conv1D(filters=self.num_channels, kernel_size=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))

    @tf.function
    def call(self, input):
        accumulator = None
        nonlinear_signal = input[:,:,0:self.num_channels]
        linear_signal = input[:,:,0:self.num_channels]
        controls = input[:,:,self.num_channels:]
        for i in range(len(self.nonlinear_c)):
            # Apply convolutional layer to linear signal for full linear path.
            linear_output = self.linear_c[i](tf.concat([linear_signal, controls], axis=2))

            # Apply convolutional layer to distorted signal for linear shaping
            # of distorted signal.
            linearly_shaped_output = self.nonlinear_c[i](tf.concat([nonlinear_signal, controls], axis=2))

            # Apply nonlinearity for further distortion of shaped distorted signal.
            nonlinear_output = self.lr(linearly_shaped_output)

            # Add the result to the total output of the network. This is a
            # "skip connection".
            if accumulator is None:
                accumulator = tf.concat([linear_output, linearly_shaped_output, nonlinear_output], axis=2)
                # accumulator = nonlinear_output
            else:
                accumulator = tf.concat([accumulator, linear_output, linearly_shaped_output, nonlinear_output], axis=2)
                # accumulator = tf.concat([accumulator, nonlinear_output], axis=2)

            if (i != len(self.nonlinear_c) - 1):
                # Only compute input if there's a next layer to feed it to.
                # Input to the next layer is linear combination of
                # input to this layer and output of last layer.
                if (i == 0):
                    nonlinear_signal = self.nonlinear_io[i](nonlinear_output) + tf.expand_dims((tf.reduce_sum(nonlinear_signal, axis=2)/2.0), axis=2)
                    linear_signal = self.linear_io[i](linear_output) + tf.expand_dims((tf.reduce_sum(linear_signal, axis=2)/2.0), axis=2)
                else:
                    nonlinear_signal = self.nonlinear_io[i](nonlinear_output) + nonlinear_signal
                    linear_signal = self.linear_io[i](linear_output) + linear_signal

        # Concatenate along the channel dimension and apply the 1x1
        # convolution.
        mixed = self.mixer(accumulator)
        # Squeeze out the last few samples and take only the last frame_size
        # frames.
        return mixed[:,-self.frame_size:,:]

    def dc_loss(self, prediction, ground_truth):
        # High pass both signals.
        hpf_ground_truth = hpf(ground_truth)
        hpf_prediction = hpf(prediction)

        denominator = tf.reduce_sum(hpf_ground_truth ** 2)

        # Compute dc offset loss.
        N = tf.size(prediction, out_type=tf.dtypes.float32)
        dc_offset = (1.0/N) * tf.reduce_sum(hpf_prediction - hpf_ground_truth)
        dc_offset = dc_offset * dc_offset
        if (denominator > self.loss_divisor_threshold):
            dc_offset = (N * dc_offset) / denominator

        return self.dc_loss_multiplier * dc_offset

    def l2_loss(self, prediction, ground_truth):
        # High pass both signals.
        hpf_ground_truth = hpf(ground_truth)
        hpf_prediction = hpf(prediction)

        # Compute normalized L2 loss.
        l2 = tf.reduce_sum((hpf_prediction - hpf_ground_truth) ** 2)
        denominator = tf.reduce_sum(hpf_ground_truth ** 2)
        if (denominator > self.loss_divisor_threshold):
            # Avoid dividing by zero.
            l2 = l2 / denominator
        return l2

    def loss(self, prediction, ground_truth):
        # High pass both signals.
        hpf_ground_truth = hpf(ground_truth)
        hpf_prediction = hpf(prediction)


        # Compute normalized L2 loss.
        l2 = tf.reduce_sum((hpf_prediction - hpf_ground_truth) ** 2)
        denominator = tf.reduce_sum(hpf_ground_truth ** 2)
        if (denominator > self.loss_divisor_threshold):
            # Avoid dividing by zero.
            l2 = l2 / denominator

        # Compute dc offset loss.
        N = tf.size(prediction, out_type=tf.dtypes.float32)
        dc_offset = (1.0/N) * tf.reduce_sum(hpf_prediction - hpf_ground_truth)
        dc_offset = dc_offset * dc_offset
        if (denominator > self.loss_divisor_threshold):
            dc_offset = (N * dc_offset) / denominator

        return l2 + self.dc_loss_multiplier * dc_offset
