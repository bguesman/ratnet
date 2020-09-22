import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

@tf.function
def hpf(signal):
    # Signal's dimensions are minibatch X receptive field X channels
    hpf_signal = signal[...,1:,:] - 0.95*signal[...,:-1,:]
    return tf.concat([hpf_signal, signal[:,0:1,:]], 1)

@tf.function
def parameterized_hpf(signal, cutoff):
    # Signal's dimensions are minibatch X receptive field X channels
    hpf_signal = signal[...,1:,:] - cutoff*signal[...,:-1,:]
    return tf.concat([hpf_signal, signal[:,0:1,:]], 1)

def resample(signal, start_sr, target_sr):
    batch_size = signal.shape[0]
    signal = tf.reshape(signal, [-1, signal.shape[-1]])
    resampled = tfio.audio.resample(signal, rate_in=start_sr, rate_out=target_sr)
    # Reshape to original.
    resampled = tf.reshape(resampled, [batch_size, -1, resampled.shape[1]])
    return resampled
