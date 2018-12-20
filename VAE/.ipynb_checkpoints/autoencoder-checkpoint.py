import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

INPUT_DIM = 784  # Dimension of the input data
ENCODING_DIM = 32  # Latent representation have dimension 1

# Creating model
input_data = Input(shape=(INPUT_DIM,))  # Input placeholder. Inputs have dimension 2
encoded = Dense(ENCODING_DIM, activation='relu')(input_data)  # Encoding of the input
decoded = Dense(INPUT_DIM, activation='sigmoid')(encoded)  # Reconstruction of the input
autoencoder = Model(input_data, decoded)  # The autoencoder represent a model of the identity function from inp to out
encoder = Model(input_data, encoded)  # Encoder model
encoded_input = Input(shape=(ENCODING_DIM,))  # Encoded input placeholder
decoder_layer = autoencoder.layers[-1]  # Last layer of autoencoder
decoder = Model(encoded_input, decoder_layer(encoded_input))  # Decoder model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')  # Compile

# Preparing data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),   
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# Prediction
encoded_data = encoder.predict(x_test)
decoded_data = decoder.predict(encoded_data)

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
    plt.imshow(decoded_data[i].reshape(28, 28))  # display reconstruction
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()