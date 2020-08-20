import numpy as np
import tensorflow as tf

@tf.function
def hpf(signal):
    # Assume last dimension is signal dimension. We might have
    # batches and channels before it.
    # Also, Tensorflow doesn't let us do this as an in-place operation, so
    # we have to get wacky with the slicing.
    hpf_signal = signal[...,1:] - 0.95*signal[...,:-1]
    reshape_size = tf.concat([signal.shape[:-1], [1]], axis=0)
    return tf.concat([tf.reshape(signal[:,0], reshape_size), hpf_signal], len(signal.shape)-1)
