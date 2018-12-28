'''Example of VAE on MNIST dataset using CNN
Inspired from Keras website
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''
import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import csv
import numpy as np
import argparse
import os
import plot_vae
from Helper import data_loader


MODELS_PATH = '../Models/VAE/'
RESULTS_PATH = '../Results/VAE/'

class VAE:
    
    def __init__(self, data):
        # network parameters
        self.batch_size = 128
        self.kernel_size = 3
        self.filters = 16
        self.latent_dim = 2
        self.epochs = 2
        self.shape = 0
        self.data = data
        self.input_shape = (data.img_size, data.img_size, 1)
        self.encoder = None
        self.decoder = None
        self.vae = None

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # then z = z_mean + sqrt(var)*eps
    def sampling(self, args):
        """Reparameterization trick by sampling for an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def build_encoder(self):
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = inputs
        for i in range(2):
            self.filters *= 2
            x = Conv2D(filters=self.filters,
                    kernel_size=self.kernel_size,
                    activation='relu',
                    strides=2,
                    padding='same')(x)
        # shape info needed to build decoder model
        self.shape = K.int_shape(x)
        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        return encoder, inputs, z_mean, z_log_var

    def build_decoder(self):
        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.shape[1] * self.shape[2] * self.shape[3], activation='relu')(latent_inputs)
        x = Reshape((self.shape[1], self.shape[2], self.shape[3]))(x)
        for i in range(2):
            x = Conv2DTranspose(filters=self.filters,
                                kernel_size=self.kernel_size,
                                activation='relu',
                                strides=2,
                                padding='same')(x)
            self.filters //= 2
        outputs = Conv2DTranspose(filters=1,
                                kernel_size=self.kernel_size,
                                activation='sigmoid',
                                padding='same',
                                name='decoder_output')(x)
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        return decoder

    def add_custom_loss_function(self, inputs, outputs, z_mean, z_log_var):
        # VAE loss = mse_loss or xent_loss + kl_loss
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
        #reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
        reconstruction_loss *= self.data.img_size * self.data.img_size
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)

    def build_vae(self):
        self.encoder, inputs, z_mean, z_log_var = self.build_encoder()
        self.decoder = self.build_decoder()
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae')
        self.add_custom_loss_function(inputs, outputs, z_mean, z_log_var)
        self.vae.summary()
        self.vae.compile(optimizer='rmsprop')

    def train(self):
        history_callback = self.vae.fit(self.data.x_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.data.x_test, None),
            callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        loss_history = history_callback.history
        print(type(loss_history))
        print(loss_history['val_loss'][0])
        with open(RESULTS_PATH + 'loss_log.csv', 'a') as f:
            writer = csv.DictWriter(f, loss_history.keys())
            writer.writeheader()
            for epoch in range(self.epochs):
                writer.writerow({
                'val_loss':loss_history['val_loss'][epoch],
                'loss':loss_history['loss'][epoch]
                })


def load_vae(model):
    model.encoder.load_weights(MODELS_PATH + 'encoder_w.h5')
    model.decoder.load_weights(MODELS_PATH + 'decoder_w.h5')
    model.vae.load_weights(MODELS_PATH + 'vae_w.h5')


def save_vae(model):
    model.encoder.save_weights(MODELS_PATH + 'encoder_w.h5')
    model.decoder.save_weights(MODELS_PATH + 'decoder_w.h5')
    model.vae.save_weights(MODELS_PATH + 'vae_w.h5')
    

if __name__ == '__main__':
    data = data_loader.DataStructure('vae')
    model = VAE(data)
    model.build_vae()
    try:
        load_vae(model)
    except OSError:
        model.train()
        save_vae(model)
    plot_vae.plot_results(
        encoder=model.encoder, 
        decoder=model.decoder,
        x_test=model.data.x_test, 
        y_test=model.data.y_test, 
        batch_size=model.batch_size, 
        model_name="vae_cnn")