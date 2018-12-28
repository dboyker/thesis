"""
An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028
Inspired from https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
"""

import sys
sys.path.append("..")
import argparse
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt
import csv
import numpy as np
from Helper import data_loader


MODELS_PATH = '../Models/GAN/'
RESULTS_PATH = '../Results/GAN/'
EPOCHS = 100
BATCH_SIZE = 64
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


class GAN():

    def __init__(self, data):
        self.data = data
        self.img_shape = (data.img_size, data.img_size, 1)
        self.latent_dim = 100
       
        generator_m = self.build_generator()
        discriminator_m = self.build_discriminator()


        # The generator_model is used when we want to train the generator layers.
        # As such, we ensure that the discriminator layers are not trainable.
        # Note that once we compile this model, updating .trainable will have no effect within it. As such, it
        # won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
        # as we compile the generator_model first.
        for layer in discriminator_m.layers:
            layer.trainable = False
        discriminator_m.trainable = False
        generator_input = Input(shape=(100,))
        generator_layers = generator_m(generator_input)
        discriminator_layers_for_generator = discriminator_m(generator_layers)
        self.generator = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
        # We use the Adam paramaters from Gulrajani et al.
        self.generator.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

        # Now that the generator is compiled, we can make the discriminator layers trainable.
        for layer in discriminator_m.layers:
            layer.trainable = True
        for layer in generator_m.layers:
            layer.trainable = False
        discriminator_m.trainable = True
        generator_m.trainable = False

        # The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
        # The noise seed is run through the generator model to get generated images. Both real and generated images
        # are then run through the discriminator. Although we could concatenate the real and generated images into a
        # single tensor, we don't (see model compilation for why).
        real_samples = Input(shape=self.data.x_train.shape[1:])
        generator_input_for_discriminator = Input(shape=(100,))
        generated_samples_for_discriminator = generator_m(generator_input_for_discriminator)
        discriminator_output_from_generator = discriminator_m(generated_samples_for_discriminator)
        discriminator_output_from_real_samples = discriminator_m(real_samples)

        # We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
        averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
        # We then run these samples through the discriminator as well. Note that we never really use the discriminator
        # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
        averaged_samples_out = discriminator_m(averaged_samples)

        # The gradient penalty loss function requires the input averaged samples to get gradients. However,
        # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
        # of the function with the averaged samples here.
        partial_gp_loss = partial(gradient_penalty_loss,
                                averaged_samples=averaged_samples,
                                gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

        # Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
        # real samples and generated samples before passing them to the discriminator: If we had, it would create an
        # output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
        # would have only BATCH_SIZE samples.

        # If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
        # samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
        self.discriminator = Model(inputs=[real_samples, generator_input_for_discriminator],
                                    outputs=[discriminator_output_from_real_samples,
                                            discriminator_output_from_generator,
                                            averaged_samples_out])
        # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
        # samples, and the gradient penalty loss for the averaged samples.
        self.discriminator.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                                    loss=[wasserstein_loss,
                                        wasserstein_loss,
                                        partial_gp_loss])
        self.generator.summary()
        self.discriminator.summary()


    def build_generator(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=100))
        model.add(LeakyReLU())
        model.add(Dense(128 * 7 * 7))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        if K.image_data_format() == 'channels_first':
            model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
            bn_axis = 1
        else:
            model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
            bn_axis = -1
        model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        model.add(Convolution2D(64, (5, 5), padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))  
        # Because we normalized training inputs to lie in the range [-1, 1], the tanh function should be used
        return model


    def build_discriminator(self):
        model = Sequential()
        if K.image_data_format() == 'channels_first':
            model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(1, 28, 28)))
        else:
            model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
        model.add(LeakyReLU())
        model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2]))
        model.add(LeakyReLU())
        model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(1024, kernel_initializer='he_normal'))
        model.add(LeakyReLU())
        model.add(Dense(1, kernel_initializer='he_normal'))
        return model


    def train(self, epochs):
        valid = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        fake = -valid
        dummy = np.zeros((BATCH_SIZE, 1), dtype=np.float32)   # gradient_penalty loss function and is not used.
        for epoch in range(epochs):
            np.random.shuffle(self.data.x_train)
            d_loss = []
            g_loss = []
            minibatches_size = BATCH_SIZE * TRAINING_RATIO
            for i in range(int(self.data.x_train.shape[0] // minibatches_size)):
                discriminator_minibatches = self.data.x_train[i * minibatches_size:(i + 1) * minibatches_size]
                for j in range(TRAINING_RATIO):  # Number of discriminator update per generator update
                    image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                    noise = np.random.rand(BATCH_SIZE, 100).astype(np.float32)
                    d_loss.append(self.discriminator.train_on_batch([image_batch, noise],
                                                                                [valid, fake, dummy]))
                g_loss.append(self.generator.train_on_batch(np.random.rand(BATCH_SIZE, 100), valid))
                print("Epoch: %d, Minibatch: %d/%d" % (epoch, i, int(self.data.x_train.shape[0] // minibatches_size)))

  

def load_gan(model):
    model.discriminator.load_weights(MODELS_PATH + 'discriminator_w.h5')
    model.generator.load_weights(MODELS_PATH + 'generator_w.h5')


def save_gan(model):
    model.discriminator.save_weights(MODELS_PATH + 'discriminator_w.h5')
    model.generator.save_weights(MODELS_PATH + 'generator_w.h5')
    model.discriminator.save_model(MODELS_PATH + 'discriminator_m.h5')
    model.generator.save_model(MODELS_PATH + 'generator_m.h5')


if __name__ == '__main__':
    data = data_loader.DataStructure('dcgan')
    model = GAN(data)
    model.train(epochs=EPOCHS)
    #try:
    #    load_gan(model)
    #except OSError:
        #model.train(epochs=EPOCHS)
        #save_gan(model)