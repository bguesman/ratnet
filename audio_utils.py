import numpy as np
import tensorflow as tf

@tf.function
def hpf(signal):
    # Signal's dimensions are minibatch X receptive field X channels
    hpf_signal = signal[...,1:,:] - 0.95*signal[...,:-1,:]
    return tf.concat([hpf_signal, signal[:,0:1,:]], 1)