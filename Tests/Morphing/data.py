from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import csv
import cv2 as cv
from os import walk



def download_mnist():
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        for i, image in enumerate(train_images):
                result = Image.fromarray(image)
                print(i)
                result.save('train/' + str(i) + '-' + str(train_labels[i]) + '.png')
        for i, image in enumerate(test_images):
                print(i)
                result = Image.fromarray(image)
                result.save('test/' + str(i) + '-' + str(test_labels[i]) + '.png')


def create_index():   
        with open('train.csv', 'w') as csvfile:
                w = csv.writer(csvfile, delimiter=';')
                for root, dirs, filenames  in walk('train/'):
                        for file in filenames:
                                file = file.replace('.png', '')
                                file = file.split('-')
                                w.writerow([file[0], file[1]])


def load_synthetic_images():
        synthetic_images = []
        synthetic_labels = []
        for root, dirs, filenames in walk('synthetic/'):
                for file in filenames:
                        if file == '.DS_Store':
                                continue
                        """
                        if file not in ['2-1.png']:
                                continue
                        """
                        img = cv.imread('synthetic/' + file, cv.IMREAD_GRAYSCALE)
                        file = file.replace('.png', '')
                        file = file.split('-')
                        synthetic_images.append(img)
                        synthetic_labels.append(int(file[1]))
        return np.array(synthetic_images), np.array(synthetic_labels)                   


#create_index()
#download_mnist()
#synthetic_images, synthetic_labels = load_synthetic_images()