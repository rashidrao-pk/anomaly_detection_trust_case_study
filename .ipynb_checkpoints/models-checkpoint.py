import utils as ut
import tensorflow.keras
from tensorflow.keras import backend as K
# from keras import backend as K
import tensorflow as tf


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
# import tensorflow_probability as tfp

import tensorflow.keras.layers as layers
class Model_VAE_GAN_functions():
    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
        ####              TODO
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_encoder(latent_dim=None,image_size=[128, 128],print_summary=False):
        a,b        = image_size
        shape      = (a, b,3)
        encoder_inputs = keras.Input(shape=shape)
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(160, activation="tanh")(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Model_VAE_GAN_functions.Sampling()([z_mean, z_log_var])

        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        if print_summary:
            encoder.summary()
        return encoder

    def get_decoder(latent_dim=None,print_summary=False):
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(8* 8 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((8, 8, 64))(x)
        
        x = layers.Conv2DTranspose(256, 2, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(256, 3, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2DTranspose(128, 2, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2DTranspose(64, 2, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2DTranspose(32, 2, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        if print_summary:
            decoder.summary()
        return decoder
    def get_discriminator(shape=None,print_summary=False):
        discriminator_inputs = keras.Input(shape=shape)

        x = layers.Conv2D(128, 8, activation="relu", strides=2, padding="same")(discriminator_inputs)
        x = layers.MaxPool2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 5, activation="relu", strides=2, padding="same")(x)
        x = layers.MaxPool2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 4, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)

        discriminator_outputs = layers.Dense(1,activation="sigmoid")(x)
        discriminator = keras.Model(discriminator_inputs, discriminator_outputs, name="discriminator")
        if print_summary:
            discriminator.summary()
        return discriminator


###########################################################################

# class Model_AE_GAN():
#     def get_encoder(latent_dim=None,image_size=[128, 128],print_summary=False):
#         a,b        = image_size
#         shape      = (a, b,3)
#         encoder_inputs = tf.keras.Input(shape=shape)
#         x = layers.Conv2D(32, 4, activation="relu", strides=2, padding="same")(encoder_inputs)
#         x = layers.GaussianNoise(stddev=15)(x)
#         x = layers.MaxPooling2D(2)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Dropout(.2)(x)

#         x = layers.Conv2D(64, 4, activation="relu", strides=2, padding="same")(x)
#         x = layers.MaxPooling2D(2)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Dropout(.2)(x)

#         x = layers.Conv2D(65, 3, activation="relu", strides=1, padding="same")(x)

#         encoder = tf.keras.Model(encoder_inputs, x, name="encoder")

#         if print_summary:
#             encoder.summary()
#         return encoder,x

#     def get_decoder(x=None,print_summary=False):
#         a,b,c,d = x.shape
#         latent_inputs = keras.Input(shape=(b,c,d))

#         x = layers.Conv2DTranspose(256, 3, activation="relu", strides=1, padding="same")(latent_inputs)
#         x = layers.BatchNormalization()(x)
#         x = layers.UpSampling2D(2)(x)
#         x = layers.Dropout(.2)(x)

#         x = layers.Conv2DTranspose(128, 5, activation="relu", strides=2, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#         #x = layers.UpSampling2D(2)(x)
#         x = layers.Dropout(.2)(x)

#         x = layers.Conv2DTranspose(64, 9, activation="relu", strides=4, padding="same")(x)
#         x = layers.BatchNormalization()(x)

#         x = layers.Conv2D(32, 5, activation="relu", strides=1, padding="same")(x)
#         decoder_outputs = layers.Conv2D(3, 5, activation="sigmoid", padding="same")(x)
#         decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
#         if print_summary:
#             decoder.summary()
#         return decoder
#     def get_discriminator(shape=None,print_summary=False):
#         discriminator_inputs = keras.Input(shape=shape)

#         x = layers.Conv2D(32, 4, activation="relu", strides=2, padding="same")(discriminator_inputs)
#         x = layers.MaxPooling2D(2)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Dropout(.2)(x)

#         x = layers.Conv2D(64, 4, activation="relu", strides=2, padding="same")(x)
#         x = layers.MaxPooling2D(2)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Dropout(.2)(x)
#         x = layers.Flatten()(x)
#         x = layers.Dense(128,activation="relu")(x)
#         x = layers.Dropout(.2)(x)
#         x = layers.Dense(128,activation="relu")(x)
#         discriminator_outputs = layers.Dense(1,activation="sigmoid")(x)
#         discriminator = keras.Model(discriminator_inputs, discriminator_outputs, name="discriminator")
#         if print_summary:
#             discriminator.summary()
#         return discriminator
#     def get_lr_callback(epoch,lr):
#         lr_start   = 0.00001
#         lr_max     = 0.005#0.00000125 * 1 * batch_size
#         lr_min     = 0.0001
#         lr_ramp_ep = 10
#         lr_sus_ep  = 2
#         lr_decay   = 0.9
#         cycle = 5
#         def lrfn(epoch):
#             if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
#             elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
#             else:
#                 lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
#             return lr    
#         return lrfn(epoch)
#     def diff(a,b):
#         img = abs(a-b)
#         treshhold = .2
#         img = layers.Activation("relu")(img-treshhold)
#         #m,M = tf.reduce_min(img),tf.reduce_max(img)
#         #norm = (img-m)/(M-m)
#         return layers.Activation("relu")(img)*255
###########################################################################################
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
    def call(self,x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z,reconstruction

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            #coorelation_loss = mdl.Model_VAE_GAN.corr_loss(z)
            
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
###########################################################################################
class VAE_GAN(keras.Model):    

    def __init__(self, vae, discriminator, opti1=keras.optimizers.Adam(), opti2=keras.optimizers.Adam(), opti3=keras.optimizers.Adam(), **kwargs):
        super(VAE_GAN, self).__init__(**kwargs)
        
        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.discriminator = discriminator
        self.vae = vae
        
        self.vae_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.correlation_loss_tracker = keras.metrics.Mean(name="cr_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss = keras.losses.BinaryCrossentropy()
        
        self.vae_optimizer = opti1
        self.gen_optimizer = opti2
        self.disc_optimizer = opti3
        
    def call(self,x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z,reconstruction

    @property
    def metrics(self):
        return [
            self.vae_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.correlation_loss_tracker,
            self.disc_loss_tracker,
            self.gen_loss_tracker
        ]

    def train_step(self, data):        
        batch_size = K.shape(data)[0]    
        
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
            
            z_mean, z_log_var, z = self.encoder(data)
            
            reconstruction = self.decoder(z)
            
            reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # coorelation_loss = Model_VAE_GAN.corr_loss(z)
            
            
            # GAN
            #batch_size = 12
            recon_vect = z#tf.random.normal((batch_size, latent_dim))
            contruction = self.decoder(recon_vect)
            combined_images = tf.concat([data, contruction], axis=0)
            data_l,recon_l = tf.zeros((batch_size, 1)),tf.ones((batch_size, 1))
            combined_l = tf.concat([data_l, recon_l], axis=0)
            tot_predictions = self.discriminator(combined_images)
            r_prediction = self.discriminator(contruction)

            discr_loss = self.disc_loss(combined_l,tot_predictions)
            #fake labels : 
            #gen_loss =  self.disc_loss(recon_l,r_prediction)
            gen_loss = tf.math.maximum(self.disc_loss(data_l,r_prediction) - discr_loss,.0001)
        
            #=========
            vae_loss = reconstruction_loss + kl_loss + gen_loss #+.1*coorelation_loss 

               
        grad_discr = disc_tape.gradient(discr_loss, self.discriminator.trainable_weights)
        grad_vae = enc_tape.gradient(vae_loss, self.vae.trainable_weights)
        #grad_gen = dec_tape.gradient(gen_loss, self.decoder.trainable_weights)
        
        
        #self.gen_optimizer.apply_gradients(zip(grad_gen, self.decoder.trainable_weights))
        self.disc_optimizer.apply_gradients(zip(grad_discr, self.discriminator.trainable_weights))
        self.vae_optimizer.apply_gradients(zip(grad_vae, self.vae.trainable_weights))

                                           
        self.vae_loss_tracker.update_state(vae_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # self.correlation_loss_tracker.update_state(coorelation_loss)
        self.disc_loss_tracker.update_state(discr_loss)
        self.gen_loss_tracker.update_state(gen_loss)
        
        return {
            "vae_loss": self.vae_loss_tracker.result(),
            "disc_loss": self.disc_loss_tracker.result(),
            "gen_los": self.gen_loss_tracker.result(),
        }