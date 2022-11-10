"""
CS 351 - Artificial Intelligence 
Assignment 3, Question 1

Student 1(Name and ID): Ali Asghar Yousuf ay06993
Student 2(Name and ID): Ali Zain Sardar

"""
import numpy as np
import math
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.animation as anm


class KMeansClustering:
    
    def __init__(self, filename: str, K:int):
        self.image = mpimg.imread(filename)        
        self.K = K
        self.height = self.image.shape[0] - 1
        self.width = self.image.shape[1] - 1
        # print(self.width, self.height)
        self.centroids = list()
        self.clusters = dict()
        self.index = dict()
        self.error = 9999
        self.total_error = 0
        self.result = self.image.copy()
        self.animation = None
    
    
    def __generate_initial_centroids(self) -> list :
        #write your code here to return initial random centroids
        for i in range(self.K):
            h, w = random.randint(0, self.height), random.randint(0, self.width)
            # self.centroids.append((h, w, self.image[h][w]))
            self.centroids.append(self.image[h][w])
            self.clusters[i] = list()
            self.index[i] = list()
        return self.centroids
    
    def __calculate_distance(self, p1: tuple, p2: tuple) -> float:
        #This function computes and returns distances between two data points
        return math.dist(p1, p2)

    def __assign_clusters(self)->dict:
        #assign each data point to its nearest cluster (centroid)
        for i in range(self.height):
            for j in range(self.width):
                d = 99999999
                a, b, pos = 0, 0, 0
                for k in range(self.K):
                    dist = self.__calculate_distance(self.image[i][j], self.centroids[k])
                    if dist < d:
                        d, a, b, pos = dist, i, j, k
                # self.clusters[pos].append((a, b, self.image[a][b]))
                self.clusters[pos].append(self.image[a][b])
                self.index[pos].append((a, b))
                self.result[i][j] = self.centroids[pos]
        # self.show_result()
        return self.clusters
    
    def average(self, arr):
        r, g, b, count = 0, 0, 0, len(arr)
        for i in arr:
            r += i[0]
            g += i[1]
            b += i[2]
        if count:
            return (r/count, g/count, b/count)
        return (r, g, b)
    
    def __recompute_centroids(self)->list:
        #your code here to return new centroids based on cluster formation
        for i in range(self.K):
            # rgb = self.clusters[i][2]
            rgb = self.clusters[i]
            avg = self.average(rgb)
            # avg = np.average(rgb, axis=(0,1,2))
            new_cluster = (int(avg[0]), int(avg[1]), int(avg[2]))
            # new_cluster = np.average(rgb, axis=0)
            # print(new_cluster)
            # new = (new_cluster[0], new_cluster[1])
            self.error = self.__calculate_distance(self.centroids[i], new_cluster)
            # print(self.error)
            self.centroids[i] = new_cluster
            self.clusters[i] = list()
            self.index[i] = list()
        return self.centroids

   
    def apply(self):
        #your code here to apply kmeans algorithm to cluster data loaded from the image file.
        self.__generate_initial_centroids()
        self.__assign_clusters()
        # self.show_result()
        while self.error > 0.5:
            # self.show_result()
            self.__recompute_centroids()
            self.__assign_clusters()
        # self.show_result()
        # self.print_centroids()
        self.__save_image()
        
 
    def __save_image(self):
        #This function overwrites original image with segmented image to be shown later.
        # self.result = self.image.copy()
        for i in self.index:
            count = 0
            for j in self.index[i]:
                # result[j[0]][j[1]] = j[2]
                self.clusters[i][count] = self.centroids[i]
                self.result[j[0]][j[1]] = self.clusters[i][count]
                count += 1
        # print(self.clusters[1][10])
    
    def show_result(self):
        # self.animation = plt.imshow(self.result)
        plt.imshow(self.result)
        plt.imsave("results\sample5.jpg", self.result)
        plt.show()
        
        
    def print_centroids(self):
        #This function prints all centroids formed by Kmeans clustering
        print(self.centroids)
       
        
kmeans = KMeansClustering("images\sample5.jpg", 5)
kmeans.apply()
kmeans.show_result()
# kmeans.print_centroids()
