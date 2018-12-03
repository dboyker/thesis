import numpy as np 
import pandas as pd
from scipy.spatial import Delaunay
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


class FeaturesMorphing:
        

        def __init__(self, img0, img1):
                self.img0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
                self.img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
                self.features0 = []
                self.features1 = []
                self.triangulation0 = []
                self.triangulation1 = []
                self.features_graph = None
                self.synthetic_img = None
                self.tri = None


        def create_synthetic_image(self):
                self.features0 = self.find_features(self.img0)
                self.features1 = self.find_features(self.img1)
                self.match_features()
                self.apply_triangulation()
                self.morph()


        def find_features(self, img):
                """
                Do clustering on image.
                Return clusters center.
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


        def apply_triangulation(self):
                tri = Delaunay(self.features0)
                self.tri = tri
                for i in range(len(tri.simplices.copy())):
                        tri0 = []
                        tri1 = []
                        for s in tri.simplices.copy()[i]:
                                tri0.append(self.features0[s])
                                tri1.append(self.features1[s])
                        self.triangulation0.append(np.float32([tri0]))
                        self.triangulation1.append(np.float32([tri1]))


        def create_features_graph(self):
                """
                Create and return a bipartite graph base on two set of features.
                The nodes of the graph are the features and the edges are the distance proximity.
                """
                graph = {}
                # Find closest f1 for each f0 and construct graph
                for j in range(len(self.features0)):
                        f0 = self.features0[j]
                        distances_to_f1 = []
                        for i in range(len(self.features1)):
                                f1 = self.features1[i]
                                d = np.sqrt(np.power((f0[0]-f1[0]),2) + np.power((f0[1]-f1[1]),2))  # Compute distance 
                                distances_to_f1.append((d,i))  # Add distances
                        graph[j] = sorted(distances_to_f1, key=lambda x: x[0])[:4]
                self.features_graph = graph


        def match_features(self):
                """
                Create 1-1 correspondance between features from to sets.
                Do matching by distance optimization.
                """
                self.create_features_graph()
                # Create cost matrix
                costs = []
                for k in self.features_graph:
                        cost = np.full(len(self.features0), 1000)
                        cost[self.features_graph[k][0][1]] = self.features_graph[k][0][0]
                        cost[self.features_graph[k][1][1]] = self.features_graph[k][1][0]
                        costs.append(cost.tolist())
                col_ind = linear_sum_assignment(costs)[1]
                new_features1 = []
                for col in col_ind:
                        new_features1.append(self.features1[col])
                self.features1 = new_features1
                

        def match_triangles(self, tri0, tri1) :
                img0_bis = cv.cvtColor(self.img0, cv.COLOR_GRAY2BGR)
                img_out = np.zeros(img0_bis.shape, dtype = img0_bis.dtype)
                r1 = cv.boundingRect(tri0)
                r2 = cv.boundingRect(tri1)
                tri0_cropped = []
                tri1_cropped = []
                for i in range(0, 3):
                        tri0_cropped.append(((tri0[0][i][0] - r1[0]),(tri0[0][i][1] - r1[1])))
                        tri1_cropped.append(((tri1[0][i][0] - r2[0]),(tri1[0][i][1] - r2[1])))
                img0_cropped = img0_bis[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
                warp_mat = cv.getAffineTransform( np.float32(tri0_cropped), np.float32(tri1_cropped) )
                img1_cropped = cv.warpAffine( img0_cropped, warp_mat, (r2[2], r2[3]), None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101 )
                mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
                cv.fillConvexPoly(mask, np.int32(tri1_cropped), (1.0, 1.0, 1.0), 16, 0);
                img1_cropped = img1_cropped * mask
                img_out[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img_out[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
                img_out[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img_out[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img1_cropped
                return cv.cvtColor(img_out, cv.COLOR_BGR2GRAY)

        def morph(self):
                self.synthetic_img = np.zeros(shape=(28,28))
                for i in range(len(self.triangulation0)):
                        self.synthetic_img += self.match_triangles(self.triangulation0[i], self.triangulation1[i])
                self.synthetic_img[self.synthetic_img > 255] = 255


"""
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
"""

#img0 = cv.imread("im0.png")
#img1 = cv.imread("im1.png")
#model = FeaturesMorphing(img0, img1)
#model.create_synthetic_image()
#plot(img0, img1, model.features0, model.features1, model.tri, model.synthetic_img)
