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
        #labels = kmeans.predict(data)  # Prediction
        return kmeans.cluster_centers_


def match_features(features0, features1):
        """
        Create 1-1 correspondance between features from to sets.
        Do matching by distance optimization.
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
        print(graph)
        # Create cost matrix
        costs = []
        for k in graph:
                #cost = np.zeros(len(features0))
                cost = np.full(len(features0), 1000)
                cost[graph[k][0][1]] = graph[k][0][0]
                cost[graph[k][1][1]] = graph[k][1][0]
                costs.append(cost.tolist())
        row_ind, col_ind = linear_sum_assignment(costs)
        return col_ind, row_ind


img0 = mpimg.imread('im0.png')
img1 = mpimg.imread('im1.png')

# Find features -> clusters and clusters center
centers0 = find_features(img0)
centers1 = find_features(img1)
centers0 = np.append(centers0, np.array([[0, 0], [0, 27], [27, 0], [27, 27]])).reshape(9, 2)  # Add corners
centers1 = np.append(centers1, np.array([[0, 0], [0, 27], [27, 0], [27, 27]])).reshape(9, 2)

# Match features 1 by 1
matching0, matching1 = match_features(centers0, centers1)
# Change centers1 order to match centers0
new_centers1 = []
for match in matching0:
        new_centers1.append(centers1[match])
centers1 = new_centers1
# Draw centers and corresponding labels/colors
fig = plt.figure(figsize=(16,8))
ax0, ax1, ax2 = fig.add_subplot(2,2,1), fig.add_subplot(2,2,2), fig.add_subplot(2,2,3)
ax0.imshow(img0)
ax0.scatter([c[0] for c in centers0], [c[1] for c in centers0], c=matching0)
ax1.imshow(img1)
ax1.scatter([c[0] for c in centers1], [c[1] for c in centers1], c=matching0)
for i in range(len(centers0)):
        ax0.annotate(i, centers0[i])
for i in range(len(centers1)):
        ax1.annotate(i, centers1[i])


# Apply Delaunay triangulation to divide picture 1 and picture 2 accordingly
tri0 = Delaunay(centers0)

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
        x0 = centers0[e[0]][0]
        y0 = centers0[e[0]][1]
        x1 = centers0[e[1]][0]
        y1 = centers0[e[1]][1]
        ax0.plot([x0, x1], [y0, y1], 'ro-')
        x0 = centers1[e[0]][0]
        y0 = centers1[e[0]][1]
        x1 = centers1[e[1]][0]
        y1 = centers1[e[1]][1]
        ax1.plot([x0, x1], [y0, y1], 'ro-')

# Morph
print(img0.shape)
#synthetic_img = np.zeros(shape=(28,28,3))
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
        #imgIn = img0
        # Output image is set to white
        imgOut = np.zeros(imgIn.shape, dtype = imgIn.dtype)
        #imgOut = cv.imread("im1.png")
        # Input triangle
        triIn = triangle0
        # Output triangle
        triOut = triangle1
        # Warp all pixels inside input triangle to output triangle
        wt.warpTriangle(imgIn, imgOut, triIn, triOut)
        # Draw triangles in input and output images.
        print(imgOut.shape)
        synthetic_img += cv.cvtColor(imgOut, cv.COLOR_BGR2GRAY)
        #plt.figure()
        #synthetic_img = imgOut
        #plt.imshow(imgOut)
ax2.imshow(synthetic_img)
plt.show()
