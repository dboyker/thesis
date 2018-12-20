import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import data as dt


class Evaluator:

    def __init__(self):
        try:
            self.model = keras.models.load_model('model/mnist.h5')
        except OSError:
            self.create_evaluator_model()
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        print(test_images.shape)
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print(self.model.predict(test_images))
        print('Test accuracy:', test_acc)
        print('Test loss:', test_loss)

    def create_evaluator_model(self):
        # Load and prepare dataset
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        synthetic_images, synthetic_labels = dt.load_synthetic_images()
        train_images = np.concatenate((train_images, synthetic_images), axis=0)
        train_labels = np.concatenate((train_labels, synthetic_labels), axis=0)
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),  # Transform the 2D imates into 1D array
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)  # Output 10 probabilities
        ])  # Build model
        self.model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        self.model.fit(train_images, train_labels, epochs=5)  # Train model
        self.model.save('model/mnist.h5')  # To save model

    def evaluate(self, synthetic_images, synthetic_labels):
        test_loss, test_acc = self.model.evaluate(synthetic_images, synthetic_labels)
        print('evaluate')
        pred = self.model.predict(synthetic_images)
        print(pred)
        print('Synthetic accuracy:', test_acc)
        print('Synthetic loss:', test_loss)