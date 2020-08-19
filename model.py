import numpy as np
import tensorflow as tf
from scipy import ndimage

class FluidAutoencoder(tf.keras.Model):
    def __init__(self):

    ######vvv DO NOT CHANGE vvvv##############
        super(FluidAutoencoder, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################


        # TODO:
        # 1) Define any hyperparameters

        # Define batch size and optimizer/learning rate
        self.batch_size = 8
        
        self.dt = 2 # HACK! In smokemultires.py

        self.num_inner_features = 16

        self.rnn_size = 10 * 10
        self.RNNs = [tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
                recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
                for i in range(self.num_inner_features)]

        #2) Define convolutional layers + batch norms.
        # RNN: 2, 4, 8, 16 (yes, a duplicate 2). For some reason, deconv1 has 32 channels??
        # No RNN: 8, 16, 32, 64
        self.conv1 = tf.keras.layers.Conv2D(filters=2, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
        self.conv2 = tf.keras.layers.Conv2D(filters=4, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
        self.conv3 = tf.keras.layers.Conv2D(filters=8, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
        self.conv4 = tf.keras.layers.Conv2D(filters=self.num_inner_features, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.batch_norm_4 = tf.keras.layers.BatchNormalization(axis=3, scale=True)

        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.batch_norm_5 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.batch_norm_6 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
        self.deconv3 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.batch_norm_7 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
        self.deconv4 = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2))

        self.deconv5 = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=5, \
            strides=1, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        
        # Global scale to put velocities in right range.
        self.global_scale = tf.Variable(1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

    @tf.function
    def call(self, lo_res, skip_connections=True):
        conv1 = self.batch_norm_1(self.conv1(lo_res))
        conv2 = self.batch_norm_2(self.conv2(conv1))
        conv3 = self.batch_norm_3(self.conv3(conv2))
        conv4 = self.batch_norm_4(self.conv4(conv3))
        frames = tf.reshape(conv4, (conv2.shape[0], -1, conv4.shape[3]))
        frames = tf.transpose(frames, (2, 0, 1))

        rnn_results = tf.stack([tf.squeeze(self.RNNs[i](tf.expand_dims(frames[i,:,:], axis=0))) \
          for i in range(self.num_inner_features)])

        if (len(rnn_results.shape) == 2):
          rnn_results = tf.transpose(rnn_results, (1, 0)) # HACK FOR RUNNING
        else:
          rnn_results = tf.transpose(rnn_results, (1, 2, 0))

        rnn_results = tf.reshape(rnn_results, conv4.shape)

        # Skip connection.
        if (skip_connections):
            deconv1 = self.batch_norm_5(self.deconv1(rnn_results))
            deconv2 = self.batch_norm_6(self.deconv2(tf.keras.layers.concatenate([deconv1, conv3], axis=3)))
            deconv3 = self.batch_norm_7(self.deconv3(tf.keras.layers.concatenate([deconv2, conv2], axis=3)))
            deconv4 = self.global_scale * self.deconv4(tf.keras.layers.concatenate([deconv3, conv1], axis=3))
            return self.deconv5(tf.keras.layers.concatenate([deconv4, lo_res], axis=3))
        else:
            deconv1 = self.batch_norm_5(self.deconv1(rnn_results))
            deconv2 = self.batch_norm_6(self.deconv2(deconv1))
            deconv3 = self.batch_norm_7(self.deconv3(deconv2))
            deconv4 = self.global_scale * self.deconv4(deconv3)
            return self.deconv5(deconv4)

    def advect(self, data, v, dim, fill, interp_method, collision=True):
        # Get a grid of cell indices (cell center point locations).
        x_range = np.arange(0, data.shape[0])
        y_range = np.arange(0, data.shape[1])
        xx, yy = np.meshgrid(x_range, y_range)

        # Use x, y to fit with velocity grid's order.
        grid = np.stack([np.transpose(xx), np.transpose(yy)], axis=-1)

        # Trace those points backward in time using the velocity field.
        backtraced_locations = grid - self.dt * v 
        if (collision):
            backtraced_locations = np.abs(backtraced_locations)

        # Sample the velocity at those points, set it to the new velocity.
        backtraced_locations_reshaped = backtraced_locations.reshape(-1,2).transpose()
        if (dim == 2):
            interpolated_x = ndimage.map_coordinates(data[:,:,0],
                backtraced_locations_reshaped, order=1, mode='constant', cval=fill)
            interpolated_y = ndimage.map_coordinates(data[:,:,1],
                backtraced_locations_reshaped, order=1, mode='constant', cval=fill)
            interpolated = np.stack([interpolated_x, interpolated_y], axis=-1)
        else:
            interpolated = ndimage.map_coordinates(data,
                backtraced_locations_reshaped, order=1, mode='constant', cval=fill)

        # Make sure to reshape back to a grid!
        interpolated = interpolated.reshape(data.shape)

        return interpolated

    def loss(self, upsampled, hi_res_tn1, hi_res_t0, hi_res_t1, hi_res_d, ret_vals=False):
        advected_backward = np.zeros(upsampled.shape, dtype=np.float32)
        advected_backward[0,:,:,:] = self.advect(upsampled[0,:,:,:], -upsampled[0,:,:,:], 2, 0.0, 'linear')
        advected_backward[1,:,:,:] = self.advect(upsampled[1,:,:,:], -upsampled[1,:,:,:], 2, 0.0, 'linear')
        advected_backward[2,:,:,:] = self.advect(upsampled[2,:,:,:], -upsampled[2,:,:,:], 2, 0.0, 'linear')
        advected_backward[3,:,:,:] = self.advect(upsampled[3,:,:,:], -upsampled[3,:,:,:], 2, 0.0, 'linear')
        advected_backward[4,:,:,:] = self.advect(upsampled[4,:,:,:], -upsampled[4,:,:,:], 2, 0.0, 'linear')
        advected_backward[5,:,:,:] = self.advect(upsampled[5,:,:,:], -upsampled[5,:,:,:], 2, 0.0, 'linear')
        advected_backward[6,:,:,:] = self.advect(upsampled[6,:,:,:], -upsampled[6,:,:,:], 2, 0.0, 'linear')
        advected_backward[7,:,:,:] = self.advect(upsampled[7,:,:,:], -upsampled[7,:,:,:], 2, 0.0, 'linear')

        advected_forward = np.zeros(upsampled.shape, dtype=np.float32)
        advected_forward[0,:,:,:] = self.advect(upsampled[0,:,:,:], upsampled[0,:,:,:], 2, 0.0, 'linear')
        advected_forward[1,:,:,:] = self.advect(upsampled[1,:,:,:], upsampled[1,:,:,:], 2, 0.0, 'linear')
        advected_forward[2,:,:,:] = self.advect(upsampled[2,:,:,:], upsampled[2,:,:,:], 2, 0.0, 'linear')
        advected_forward[3,:,:,:] = self.advect(upsampled[3,:,:,:], upsampled[3,:,:,:], 2, 0.0, 'linear')
        advected_forward[4,:,:,:] = self.advect(upsampled[4,:,:,:], upsampled[4,:,:,:], 2, 0.0, 'linear')
        advected_forward[5,:,:,:] = self.advect(upsampled[5,:,:,:], upsampled[5,:,:,:], 2, 0.0, 'linear')
        advected_forward[6,:,:,:] = self.advect(upsampled[6,:,:,:], upsampled[6,:,:,:], 2, 0.0, 'linear')
        advected_forward[7,:,:,:] = self.advect(upsampled[7,:,:,:], upsampled[7,:,:,:], 2, 0.0, 'linear')
        
        # Advect forward and backward.
        # advected_backward = [self.advect(upsampled[i,:,:,:], -upsampled[i,:,:,:], 2, 0.0, 'linear') for i in range(upsampled.shape[0])]
        # advected_forward = [self.advect(upsampled[i,:,:,:], upsampled[i,:,:,:], 2, 0.0, 'linear') for i in range(upsampled.shape[0])]
        
        advected_backward = tf.convert_to_tensor(advected_backward)
        advected_forward = tf.convert_to_tensor(advected_forward)

        # Advect density according to true high res and upsampled velocity fields.
        advected_hi_res_density = np.zeros(hi_res_d.shape, dtype=np.float32)
        advected_hi_res_density[0,:,:] = self.advect(hi_res_d[0,:,:], hi_res_t0[0,:,:,:], 1, 0.0, 'linear')
        advected_hi_res_density[1,:,:] = self.advect(hi_res_d[1,:,:], hi_res_t0[1,:,:,:], 1, 0.0, 'linear')
        advected_hi_res_density[2,:,:] = self.advect(hi_res_d[2,:,:], hi_res_t0[2,:,:,:], 1, 0.0, 'linear')
        advected_hi_res_density[3,:,:] = self.advect(hi_res_d[3,:,:], hi_res_t0[3,:,:,:], 1, 0.0, 'linear')
        advected_hi_res_density[4,:,:] = self.advect(hi_res_d[4,:,:], hi_res_t0[4,:,:,:], 1, 0.0, 'linear')
        advected_hi_res_density[5,:,:] = self.advect(hi_res_d[5,:,:], hi_res_t0[5,:,:,:], 1, 0.0, 'linear')
        advected_hi_res_density[6,:,:] = self.advect(hi_res_d[6,:,:], hi_res_t0[6,:,:,:], 1, 0.0, 'linear')
        advected_hi_res_density[7,:,:] = self.advect(hi_res_d[7,:,:], hi_res_t0[7,:,:,:], 1, 0.0, 'linear')
        
        advected_upsampled_density = np.zeros(hi_res_d.shape, dtype=np.float32)
        advected_upsampled_density[0,:,:] = self.advect(hi_res_d[0,:,:], upsampled[0,:,:,:], 1, 0.0, 'linear')
        advected_upsampled_density[1,:,:] = self.advect(hi_res_d[1,:,:], upsampled[1,:,:,:], 1, 0.0, 'linear')
        advected_upsampled_density[2,:,:] = self.advect(hi_res_d[2,:,:], upsampled[2,:,:,:], 1, 0.0, 'linear')
        advected_upsampled_density[3,:,:] = self.advect(hi_res_d[3,:,:], upsampled[3,:,:,:], 1, 0.0, 'linear')
        advected_upsampled_density[4,:,:] = self.advect(hi_res_d[4,:,:], upsampled[4,:,:,:], 1, 0.0, 'linear')
        advected_upsampled_density[5,:,:] = self.advect(hi_res_d[5,:,:], upsampled[5,:,:,:], 1, 0.0, 'linear')
        advected_upsampled_density[6,:,:] = self.advect(hi_res_d[6,:,:], upsampled[6,:,:,:], 1, 0.0, 'linear')
        advected_upsampled_density[7,:,:] = self.advect(hi_res_d[7,:,:], upsampled[7,:,:,:], 1, 0.0, 'linear')
        
        advected_hi_res_density = tf.convert_to_tensor(advected_hi_res_density)
        advected_upsampled_density = tf.convert_to_tensor(advected_upsampled_density)

        density_loss = 1000 * tf.reduce_sum((advected_upsampled_density - advected_hi_res_density)**2)
        forward_temporal_loss = tf.reduce_sum((advected_forward - hi_res_t1) ** 2)
        backward_temporal_loss = tf.reduce_sum((advected_backward - hi_res_tn1) ** 2)
        spatial_loss = tf.reduce_sum((upsampled - hi_res_t0)**2)
        #discrim_0 = discrim(upsampled)
        #discrim_1 = discrim(advected_forward)
        #discrim_n1 = discrim(advected_backward)
        
        #discrim_loss0 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(discrim_0), logits=discrim_0))/2.0
        #discrim_loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(discrim_1), logits=discrim_1))/2.0
        #discrim_lossn1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(discrim_n1), logits=discrim_n1))/2.0
        # tf.print("density loss:", density_loss)
        # tf.print("forward temporal loss:", forward_temporal_loss)
        # tf.print("backward temporal loss:", backward_temporal_loss)
        # tf.print("spatial loss:", spatial_loss)
        if ret_vals:
            return 0.25 * forward_temporal_loss + 0.25 * backward_temporal_loss + spatial_loss + density_loss, advected_forward, advected_backward
        return 0.25 * forward_temporal_loss + 0.25 * backward_temporal_loss + spatial_loss + density_loss
