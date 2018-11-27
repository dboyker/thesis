import numpy as np 
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2 as cv
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.models import load_model


def plot_results(img0, img1, features0, features1, tri0, tri1):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(img0)
        for f in features0:
                plt.scatter(f[1], f[0], s=10, c='red', marker='o')
        plt.triplot(features0[:,1], features0[:,0], tri0.simplices.copy())
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(img1)
        for f in features1:
                plt.scatter(f[1], f[0], s=10, c='red', marker='o')
        plt.triplot(features1[:,1], features1[:,0], tri1.simplices.copy())
        plt.show()


def find_features(img):
        features = np.zeros((10, 2))
        for i in range(0, 5):
                row = 7 + 3*i
                for l in range(0,28):  # Left-side features
                        r = 27 - l
                        if img[row][l] > 0:
                                features[i] = [row, l]
                        if img[row][r] > 0:
                                features[i+5] = [row, r]
        return features

img0 = mpimg.imread('im0.png')
img1 = mpimg.imread('im1.png')

# Find features
features0 = find_features(img0)
features1 = find_features(img1)
print(features0)
# Match features - Delaunay triangulation
tri0 = Delaunay(features0)
tri1 = Delaunay(features1)

# Transform

# Plot
plot_results(img0, img1, features0, features1, tri0, tri1)



"""
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
for i in range(0,40):
    if train_labels[i] == 5:
        im = train_images[i]
        result = Image.fromarray(im)
        result.save('im' + str(i) + '.png')
        """


