import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns;  sns.set()
sns.set_palette(sns.color_palette("husl", 8))
import pandas as pd
import data_loader

INPUT_SHAPE = (28, 28)
INPUT_DIM = 784  # Dimension of the input data
ENCODING_DIM = 32  # Latent representation have dimension 1
MODEL_PATH = 'Model/Autoencoder/'


def create_ae():
    input_data = Input(shape=(INPUT_DIM,))  # Input placeholder. Inputs have dimension 2
    encoded = Dense(ENCODING_DIM, activation='relu')(input_data)  # Encoding of the input
    decoded = Dense(INPUT_DIM, activation='sigmoid')(encoded)  # Reconstruction of the input
    autoencoder = Model(input_data, decoded)  # Autoencoder -> identity function
    encoder = Model(input_data, encoded)  # Encoder model
    encoded_input = Input(shape=(ENCODING_DIM,))  # Encoded input placeholder
    decoder_layer = autoencoder.layers[-1]  # Last layer of autoencoder
    decoder = Model(encoded_input, decoder_layer(encoded_input))  # Decoder model
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')  # Compile
    return encoder, decoder, autoencoder

def train_ae(autoencoder, x_train, x_test):
    autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),   
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

def load_ae():
    encoder = load_model(MODEL_PATH + 'encoder.h5')
    decoder = load_model(MODEL_PATH + 'decoder.h5')
    autoencoder = load_model(MODEL_PATH + 'autoencoder.h5')
    return encoder, decoder, autoencoder

def save_ae(encoder, decoder, autoencoder):
    encoder.save(MODEL_PATH + 'encoder.h5')
    decoder.save(MODEL_PATH + 'decoder.h5')
    autoencoder.save(MODEL_PATH + 'autoencoder.h5')

def get_ae(x_train, x_test):
    try:
        encoder, decoder, autoencoder = load_ae()
    except OSError:
        encoder, decoder, autoencoder = create_ae()
        train_ae(autoencoder, x_train, x_test)
        save_ae(encoder, decoder, autoencoder)
    encoder.summary()
    decoder.summary()
    autoencoder.summary()
    return encoder, decoder, autoencoder



x_train, x_test, y_train, y_test, _ = data_loader.load_mnist()  # Load data
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
encoder, decoder, autoencoder = get_ae(x_train, x_test)  # Load model if exist
z = encoder.predict(x_test)  # Prediction - latent space
print(z.shape)
x_synthetic = decoder.predict(z)  # Prediction - synthetic data

"""
# Visualization
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))  # display original
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_synthetic[i].reshape(28, 28))  # display reconstruction
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
"""