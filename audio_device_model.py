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

        # How many samples we are trying to predict at once.
        self.frame_size = 128

        # Brad: here's where the real bulk of the model definition happens.
        # We define a member variable for each layer in the network. This
        # example just has 2 linear layers.

        self.numConvolutionalLayers = 10;
        self.io = []
        self.c = []
        for i in range(self.numConvolutionalLayers):
            # Convolutional layer.
            self.c.append(tf.keras.layers.Conv1D(filters=8, kernel_size=3, dilation_rate=2**i, padding="causal"))
            # IO mixer (convolutional layer with kernel size 1)
            if (i != self.numConvolutionalLayers - 1):
                # Last kernel has no input.
                self.io.append(tf.keras.layers.Conv1D(filters=8, kernel_size=1, padding="causal"))

        # Since activations have no learnable parameters, I think we only
        # need one that we can reuse?
        self.lr = tf.keras.layers.ReLU()

        # Linear mixer (convolutional layer with kernel size 1).
        self.mixer = tf.keras.layers.Conv1D(filters=1, kernel_size=1, padding="same")

        # TODO: this dense layer should actually be a conv1d with a kernel_size
        # of 5 and dilation rate of 128? It just needs to mix the vectors.
        # self.d1 = tf.keras.layers.Dense(128);

    @tf.function
    def call(self, input):
        # This function actually runs the model on a given input.
        # Because we use the layers API, this is super easy.

        # Our input is shape [batch_size, 128]. But conv1D expects a
        # 3D tensor, where the last dimension is the number of channels.
        # We only have one channel, so we need to use expand_dims to
        # add a 3rd dimension that's just length 1 to the input.
        # The following gives us a tensor of size [batch_size, 128, 1].
        input = tf.expand_dims(input, axis=2)

        # Accumulator variable that we'll add each layer output to.
        accumulator = []
        for i in range(self.numConvolutionalLayers):
            # Apply convolutional layer i and non-linearity
            layer_output = self.c[i](input)
            layer_output_nonlinear = self.lr(layer_output)
            # Add the result to the total output of the network. This is a
            # "skip connection".
            accumulator.append(layer_output_nonlinear)
            if (i != self.numConvolutionalLayers - 1):
                # Only compute input if there's a next layer to feed it to.
                # Apply non-linearity and blend between the output and input
                # via a 1x1 convolution.
                #io_channels = tf.concat([layer_output_nonlinear, input], axis=2)
                input = self.io[i](layer_output_nonlinear) + input

        # Squeeze the singleton channel dimension out of the tensor and take
        # just the last 128 samples so the output shape is [batch_size, 256]
        mixer_channels = tf.concat(accumulator, axis=2)
        mixed = self.mixer(mixer_channels)
        result = (tf.squeeze(mixed))[:,-self.frame_size:]
        return result

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
