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


class KMeansClustering:
    
    def __init__(self, filename: str, K:int):
        self.image = mpimg.imread(filename)        
        self.K = K
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        # print(self.width, self.height)
        self.centroids = list()
        self.clusters = dict()
        self.error = 9999
    
    
    def __generate_initial_centroids(self) -> list :
        #write your code here to return initial random centroids
        for i in range(self.K):
            h, w = random.randint(0, self.height), random.randint(0, self.width)
            self.centroids.append((h, w, self.image[h][w]))
            self.clusters[i] = list()
        return self.centroids
    
    def __calculate_distance(self, p1: tuple, p2: tuple) -> float:
        #This function computes and returns distances between two data points
        return math.dist(p1, p2)

    def __assign_clusters(self)->dict:
        #assign each data point to its nearest cluster (centroid)
        for i in range(self.width):
            for j in range(self.height):
                d = 99999999
                a, b, pos = 0, 0, 0
                for k in range(self.K):
                    dist = self.__calculate_distance((i, j), self.centroids[k])
                    if dist < d:
                        d, a, b, pos = dist, i, j, k
                self.clusters[pos].append((a, b, self.image[a][b]))
        return self.clusters
    
    
    def __recompute_centroids(self)->list:
        #your code here to return new centroids based on cluster formation
        for i in range(self.K):
            rgb = self.clusters[i][2]
            new_cluster = tuple(np.average(rgb, axis=0))
            # new = (new_cluster[0], new_cluster[1])
            self.error = self.__calculate_distance(self.centroids[i][2], new_cluster)
            print(self.error)
            self.centroids[i] = new_cluster
        return list()

   
    def apply(self):
        #your code here to apply kmeans algorithm to cluster data loaded from the image file.
        self.__generate_initial_centroids()
        while self.error > 0.5:
            self.__assign_clusters()
            self.__recompute_centroids()
        print(self.centroids)
        # print(self.clusters)
        
       
 
    def __save_image(self):
        #This function overwrites original image with segmented image to be shown later.
        pass
    
    def show_result(self):
        plt.imshow(self.image)
        
    def print_centroids(self):
        #This function prints all centroids formed by Kmeans clustering
        print(self.centroids)
       
        
kmeans = KMeansClustering("images\sample1.jpg", 2)
kmeans.apply()
kmeans.show_result()
# kmeans.print_centroids()    
    
