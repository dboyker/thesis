import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def load_mnist():
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_size = x_train.shape[1]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, x_test, y_train, y_test, img_size
    """
    # MNIST dataset -> VAE
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, img_size, img_size, 1])
    x_test = np.reshape(x_test, [-1, img_size, img_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, x_test, y_train, y_test, img_size
    #return x_train[:100], x_test[:100], y_train[:100], y_test[:100], img_size