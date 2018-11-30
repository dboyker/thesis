import numpy as np 
import pandas as pd
from scipy.spatial import Delaunay
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2 as cv
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
import warpTriangle as wt
import warnings
warnings.filterwarnings("ignore")

"""
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
for i in range(0,40):
    if train_labels[i] == 7:
        im = train_images[i]
        result = Image.fromarray(im)
        result.save('im' + str(i) + '.png')
"""

def find_features(img):
        """
        Do clustering on image.
        Return image with clusters.
        """
        data = []
        for col in range(img.shape[1]):
                for row in range(img.shape[0]):
                        if img[row][col] > 0.1:
                                data.append([col, row])
        kmeans = KMeans(5)
        kmeans.fit(data)  # Training
        corners = [[0, 0], [0, 27], [27, 0], [27, 27]]
        return np.append(kmeans.cluster_centers_,np.array(corners)).reshape(9, 2)  # Add corners


def create_graph(features0, features1):
        """
        Create and return a bipartite graph base on two set of features.
        The nodes of the graph are the features and the edges are the distance proximity.
        """
        graph = {}
        # Find two closest f1 for each f0 and construct graph
        for j in range(len(features0)):
                f0 = features0[j]
                distances_to_f1 = []
                for i in range(len(features1)):
                        f1 = features1[i]
                        d = np.sqrt(np.power((f0[0]-f1[0]),2) + np.power((f0[1]-f1[1]),2))  # Compute distance 
                        distances_to_f1.append((d,i))  # Add distances
                graph[j] = sorted(distances_to_f1, key=lambda x: x[0])[:4]
        return graph


def match_features(features0, features1):
        """
        Create 1-1 correspondance between features from to sets.
        Do matching by distance optimization.
        """
        graph = create_graph(features0, features1)
        # Create cost matrix
        costs = []
        for k in graph:
                cost = np.full(len(features0), 1000)
                cost[graph[k][0][1]] = graph[k][0][0]
                cost[graph[k][1][1]] = graph[k][1][0]
                costs.append(cost.tolist())
        row_ind, col_ind = linear_sum_assignment(costs)
        new_centers1 = []
        for col in col_ind:
                new_centers1.append(features1[col])
        features1 = new_centers1
        return features0, features1
        

def morph(centers0, centers1, tri0):
        synthetic_img = np.zeros(shape=(28,28))
        for i in range(len(tri0.simplices.copy())):
                tri = []
                for s in tri0.simplices.copy()[i]:
                        tri.append(centers0[s])
                triangle0 = np.float32([tri])

                tri = []
                for s in tri0.simplices.copy()[i]:
                        tri.append(centers1[s])
                triangle1 = np.float32([tri])
                imgIn = cv.imread("im0.png")
                imgOut = np.zeros(imgIn.shape, dtype = imgIn.dtype)
                triIn = triangle0
                triOut = triangle1
                wt.warpTriangle(imgIn, imgOut, triIn, triOut)   # Warp all pixels inside input triangle to output triangle
                synthetic_img += cv.cvtColor(imgOut, cv.COLOR_BGR2GRAY)
        synthetic_img[synthetic_img > 255] = 255
        return synthetic_img, imgOut


def plot(img0, img1, centers0, centers1, tri0, synthetic_img):
        # Draw centers and corresponding labels
        fig = plt.figure(figsize=(16,8))
        ax0, ax1, ax2 = fig.add_subplot(2,2,1), fig.add_subplot(2,2,2), fig.add_subplot(2,2,3)
        ax0.imshow(img0)
        ax1.imshow(img1)
        for i in range(len(centers0)):
                ax0.annotate(i, centers0[i])
                ax1.annotate(i, centers1[i])
        # Draw triangles
        edges = []
        for i, tri in enumerate(tri0.simplices.copy()):
                edge1 = (tri[0], tri[1]) 
                edge2 = (tri[1], tri[2])
                edge3 = (tri[2], tri[0])
                for e in (edge1, edge2, edge3):
                        if e not in edges and e[::-1] not in edges:
                                edges.append(e)
        for e in edges:
                for c in [(centers0, ax0), (centers1, ax1)]:
                        x0, y0 = c[0][e[0]][0], c[0][e[0]][1]
                        x1 = c[0][e[1]][0]
                        y1 = c[0][e[1]][1]
                        c[1].plot([x0, x1], [y0, y1], 'ro-')
        ax2.imshow(synthetic_img)
        plt.show()



img0 = mpimg.imread('im0.png')
img1 = mpimg.imread('im1.png')
# Find and match features -> Find clusters and clusters center then match them
centers0, centers1 = match_features(find_features(img0), find_features(img1))
# Morph using Delaunay triangulation of images
tri0 = Delaunay(centers0)
synthetic_img, imgOut = morph(centers0, centers1, tri0)
plt.imsave('synthetic.png', synthetic_img, cmap=matplotlib.cm.gray)
plot(img0, img1, centers0, centers1, tri0, synthetic_img)
