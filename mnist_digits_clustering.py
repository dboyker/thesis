import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np 
import matplotlib.pyplot as plt


def customize_mnist(mnist):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = np.around(train_images / 255.0, decimals=0).tolist()
    test_images = np.around(test_images / 255.0, decimals=0).tolist()
    # Create noisy data
    noisy_images = np.random.randint(2, size=(7000, 28, 28))
    noisy_labels = np.add(np.zeros(7000), 0)
    return ((
            np.concatenate((train_images, noisy_images[:6000])), 
            np.concatenate((train_labels, noisy_labels[:6000]))
        ), 
        (
            np.concatenate((test_images, noisy_images[6000:])),
            np.concatenate((test_labels, noisy_labels[6000:]))
        ))


def get_model():
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = customize_mnist(mnist)
    try:
        model = load_model('model/mnist.h5')
    except OSError:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),  # Transform the 2D imates into 1D array
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)  # Output 10 probabilities
        ])
        model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=5)
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)
        print('Test loss:', test_loss)
        model.save('model/mnist.h5')
    return model
    #print(model.predict(np.array([test_images[0]])))

get_model()