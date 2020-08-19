import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from fluid_autoencoder import FluidAutoencoder
from fluid_discriminator import FluidDiscriminator
import sys

import time


def train(model, discrim, train_low, train_hi, train_d):
    """
    Runs through one epoch - all training examples.
    """

    print("Training", train_low.shape[0]-1, "batches.")
    
    for i in range(train_low.shape[0]-model.batch_size-1):
        # Collect batch.
        if(model.batch_size + i + 1 > train_low.shape[0]):
            break
        inputs = train_low[i+1:model.batch_size + i + 1, :]
        density_tn1 = train_d[i:model.batch_size + i, :]
        labels_tn1 = train_hi[i:model.batch_size + i, :]
        labels_t0 = train_hi[i+1:model.batch_size + i+1, :]
        labels_t1 = train_hi[i+2:model.batch_size + i + 2, :]
        with tf.GradientTape(persistent=True) as tape:
            # print("model")
            upsampled = inputs + model(inputs)
            # print("loss")
            loss, advect_for, advect_back = model.loss(upsampled, labels_tn1, labels_t0, labels_t1, density_tn1, True)
            dreal_n1 = discrim(labels_tn1)
            dreal_0 = discrim(labels_t0)
            dreal_1 = discrim(labels_t1)
            dfake_n1 = discrim(advect_back)
            dfake_0 = discrim(upsampled)
            dfake_1 = discrim(advect_for)
            d_loss = discrim.loss(dreal_0, dfake_0) + discrim.loss(dreal_1, dfake_1) + discrim.loss(dreal_n1, dfake_n1) 
            loss = loss + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dfake_0), logits=dfake_0))/2.0 \
                        + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dfake_1), logits=dfake_1))/2.0 \
                        + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dfake_n1), logits=dfake_n1))/2.0 
        # Optimize.
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        gradients = tape.gradient(d_loss, discrim.trainable_variables)
        discrim.optimizer.apply_gradients(zip(gradients, discrim.trainable_variables))

        if (i % 40 == 0):
            model.save_weights('model_weights/model_weights', save_format='tf')

        if (i % 1 == 0):
            # Pick random contiguous datapoints.
            datapoints = model.batch_size*10
            random_index = np.random.randint(1, high=train_low.shape[0]-1-datapoints, size=1)
            random_data = np.arange(random_index, random_index + model.batch_size*10)
            test_loss = test(model, discrim, tf.gather(train_low, random_data),
                tf.gather(train_hi, random_data-1), 
                tf.gather(train_hi, random_data), 
                tf.gather(train_hi, random_data+1),
                tf.gather(train_d, random_data-1))
            print("Batch", i, ", average loss on random", model.batch_size*10,
                "datapoints: ", test_loss)
            print("Index of loss:", random_index)

def test(model, discrim, test_low, test_hi_tn1, test_hi_t0, test_hi_t1, test_d):
    """
    Runs through one epoch - all testing examples.
    """
    avg_loss = 0
    avg_l2_loss = 0
    num_batches = int(test_low.shape[0] / model.batch_size)
    for i in range(num_batches):
        # Collect batch.
        if (model.batch_size + i +1> test_low.shape[0]):
            break
        batch_inputs = test_low[i:model.batch_size + i, :]
        batch_d = test_d[i:model.batch_size + i, :]
        batch_labels_tn1 = test_hi_tn1[i:model.batch_size + i, :]
        batch_labels_t0 = test_hi_t0[i:model.batch_size + i, :]
        batch_labels_t1 = test_hi_t1[i:model.batch_size + i, :]
        # Compute loss.
        upsampled = batch_inputs + model(batch_inputs)
        # print("Low max:", np.max(batch_inputs))
        # print("Upsampled max:", np.max(batch_labels_t0))
        loss, advect_for, advect_back = model.loss(upsampled, batch_labels_tn1, batch_labels_t0, batch_labels_t1, batch_d, True)
        dfake_n1 = discrim(advect_back)
        dfake_0 = discrim(upsampled)
        dfake_1 = discrim(advect_for)
        avg_l2_loss += loss / model.batch_size
        loss = loss + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dfake_0), logits=dfake_0))/2.0 \
                    + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dfake_1), logits=dfake_1))/2.0 \
                    + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dfake_n1), logits=dfake_n1))/2.0 
        # Accumulate loss.
        avg_loss += loss / model.batch_size

    print("L2 loss: ", avg_l2_loss / num_batches)

    return avg_loss / num_batches

def main():

    model = FluidAutoencoder()
    discrim = FluidDiscriminator()
    # Train and Test Model.
    start = time.time()
    epochs = 5
    frame_block_size = 160
    frame_blocks = 12000 // frame_block_size
    for i in range(epochs):
        for j in range(frame_blocks):
            print("Loading frame block", j, "...")
            train_low, train_hi, train_d, test_low, test_hi, test_d = \
                get_data('../data/lo_res/', '../data/hi_res/', j * frame_block_size, \
                frame_block_size)
            print("Frame block loaded.")
            print("Lo-res dimension:", test_low.shape[:])
            print("Hi-res dimension:", test_hi.shape[:])
            train(model, discrim, train_low, train_hi, train_d)
    end = time.time()
    print("Done training, took", (end - start) / 60, "minutes.")

    loss = test(model, discrim, test_low, test_hi)
    print("FINAL LOSS ON TEST DATA:", loss)

    model.save_weights('model_weights/model_weights', save_format='tf')

if __name__ == '__main__':
   main()
