import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K


class DataStructure:

    def __init__(self, type):
        x_train, x_test, y_train, y_test, img_size = self.load_mnist(type)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.img_size = img_size


    def load_mnist(self, type):
        if type == 'ae':
            # -> AE
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            img_size = x_train.shape[1]
            x_train = x_train.astype('float32') / 255.
            x_test = x_test.astype('float32') / 255.
            x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
            x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
            return x_train, x_test, y_train, y_test, img_size
        elif type == 'vae':
            # MNIST dataset -> VAE
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            img_size = x_train.shape[1]
            x_train = np.reshape(x_train, [-1, img_size, img_size, 1])
            x_test = np.reshape(x_test, [-1, img_size, img_size, 1])
            x_train = x_train.astype('float32') / 255
            x_test = x_test.astype('float32') / 255
            return x_train, x_test, y_train, y_test, img_size
        elif type == 'gan':
            # -> GAN
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            img_size = x_train.shape[1]
            x_train = x_train / 127.5 - 1.  # Rescale -1 to 1
            x_train = np.expand_dims(x_train, axis=3)
            return x_train, x_test, y_train, y_test, img_size
        elif type == 'dcgan':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            img_size = x_train.shape[1]
            if K.image_data_format() == 'channels_first':
                x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
            else:
                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
            x_train = (x_train.astype(np.float32)) / 127.5 - 1
            return x_train, x_test, y_train, y_test, img_size