"""

@author: wangyizhi
@reference: https://github.com/annoviko/pyclustering/blob/bf4f51a472622292627ec8c294eb205585e50f52/pyclustering/cluster/cure.py
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time,os
from COS_Funcs.dist import calc_cov_i,calc_dist
from COS_Funcs.kdtree import kdtree_

from . import visualize as V

class Cluster:
    def __init__(self,index,point,num = 1):
        self.index = [index]
        self.points = [point]
        self.center = point
        self.rep_points = [point]
        self.num = num
        self.closest = None
        self.distance = None
        
    def copy_(self,another_cluster):
        self.index = another_cluster.index
        self.points = another_cluster.points
        self.center = another_cluster.center
        self.rep_points = another_cluster.rep_points
        self.num = another_cluster.num
        self.closest = another_cluster.closest
        self.distance = another_cluster.distance
    
    def renew_center(self,another_cluster):
        '''
        Calculate the center : list
        It calculated by weight(num of points) * mean values
        '''
        dimension = len(self.center)
        self.center = [0] * dimension
        # Because points stored in lists, so should be updated in each dimensions
        for dim in range(dimension):
            self.center[dim] = (self.num * self.center[dim] + another_cluster.num * another_cluster.center[dim] ) / (another_cluster.num+self.num)
        
    def renew_paras(self,another_cluster):
        '''
        Renew the parameters of the merged cluster
        '''
        self.num += another_cluster.num # renew the number of points  num:int
        
        self.index += another_cluster.index # renew the indices of points  index:list

        self.points += another_cluster.points # renew the values of points  points:list
            
    def add_shrink(self,tmp_repset,alpha):
        '''
        Shrink the representative points to the centroid
        tmp_repset: the representative points which is satisfied after merging two clusters
        alpha: the given shrink parameter, the bigger alpha is the closer the representative to the centroid
        '''
        self.rep_points = []
        dimension = len(self.center)
        for tmp_rep in tmp_repset:
            rep = [0] * dimension
        # Because points stored in lists, so should be updated in each dimensions
            for dim in range(dimension):
                rep[dim] = tmp_rep[dim] + alpha * (self.center[dim] - tmp_rep[dim])
                # self.rep_points = tmp_repset + alpha * (self.center - tmp_repset) 
            self.rep_points.append(rep)    
                
    def merge(self,another_cluster,c,alpha,L,cov_i):
        '''
        Merge two cluster into one,
        firstly renew the parameters and center,
        secondly renew the representative points: 1st rep - farest to centroid, 2nd rep - farest to representative set (the minimum distance from this point to the representative points is maximum), 3rd rep - same to 2 
        c: the maximum number of representative points each cluster
        alpha: the given shrink parameter
        '''
        self.renew_center(another_cluster)
        self.renew_paras(another_cluster)
        # If total number of points less than c, the representative points will be points itselves
        if self.num <= c: 
            tmpSet = self.points 
        else:
            tmpSet = []
            for i in range(c):
                maxDist = 0
                for p in self.points:
                    if i==0:
                        minDist = calc_dist(p,self.center,L=L,cov_i=cov_i)
                    else:
                        # for a given p, if p's min distance to any q in tmpset is biggest, then p is next representative point 
                        minDist = np.min([calc_dist(p,q,L=L,cov_i=cov_i) for q in tmpSet])
                    if minDist >= maxDist:
                        maxPoint = p
                        maxDist = minDist
                tmpSet.append(maxPoint)
        self.add_shrink(tmpSet,alpha)
        
    def clus_dist(self,another_cluster,linkage='cure_single',L=2,cov_i=None):
        '''
        Calculate the distance between two clusters
        linkage: 
            if 'cure_single' - calculate a nearest distance using representative points as the distance of two clusters, default value;
            if 'cure_complete' - calculate a furthest distance using representative points as the distance of two clusters;  
            if 'cure_average' - calculate a average distance using representative points as the distance of two clusters;
            if 'cure_centroid' or 'centroid' - calculate a distance using centorids as the distance of two clusters; 
            if 'cure_ward' - calculate the variance if merging two clusters as the distance of two clusters;  
        L: the distance metric, L=1 the Manhattan distance, L=2 the Euclidean distance, by default L=2
        '''

               # min_dist = calc_dist(self.rep_points[0],another_cluster.rep_points[0],L)
        if linkage == 'cure_centroid' or linkage == 'centroid' :
            min_dist = calc_dist(self.center,another_cluster.center,L=L,cov_i=cov_i)
        elif linkage == 'cure_ward':
            min_dist = np.var(np.vstack([self.points,another_cluster.points]),axis=0).mean() - (np.var(self.points,axis=0) + np.var(another_cluster.points,axis=0)).mean()
            
        else:
            # For average link
            num_dists = len(self.rep_points) * len(another_cluster.rep_points)

            for ind1,i in enumerate(self.rep_points):
                for ind2,j in enumerate(another_cluster.rep_points):
                    
                    if ind1 == 0 and ind2 == 0:
                        min_dist = calc_dist(i,j,L=L,cov_i=cov_i)
                    
                    dist_i = calc_dist(i,j,L=L,cov_i=cov_i)
                    
                    if linkage == 'cure_complete':
                        if dist_i > min_dist:
                            min_dist = dist_i
                    
                    elif linkage == 'cure_average':
                        if ind1 == 0 and ind2 == 0:
                            min_dist = 0
                        min_dist += dist_i/num_dists 
                    
                    else: # 'cure_single' or others
                        if dist_i < min_dist:
                            min_dist = dist_i
        return min_dist
    
    def closest_cluster(self,tree,L,cov_i):
        # Find one more closest point than the representative points in the cluster
        # to make sure at least one another cluster's rep point will be found
        K = len(self.rep_points) + 1
        
        nearest_cluster = None
        nearest_distance = float('inf')

        for rep_point in self.rep_points:
            # Nearest nodes should be returned (at least it will return itself).
            nearest_nodes = tree.search_knn(rep_point, K, L, cov_i)
            for (kdtree_node,candidate_distance) in nearest_nodes:
                if (candidate_distance < nearest_distance) and (kdtree_node is not None) and (kdtree_node.payload is not self):
                    nearest_distance = candidate_distance
                    nearest_cluster = kdtree_node.payload
                    
        return (nearest_cluster, nearest_distance)

    @staticmethod
    def visualize_cluster(clusters):
        '''
        Visualize the clusters in a scatter figure for testing (only works for 2D)
        '''
        plt.figure()
        color_list = list(V.color_dict.keys())
        mod_ = len(color_list)
        for ind,cluster in enumerate(clusters):
            points = np.array(cluster.points)
            rep_points = np.array(cluster.rep_points)
            center = cluster.center
            c_ind = (ind*3)%mod_
            plt.scatter(points[:,0],points[:,1],label=ind,c=color_list[c_ind])
            plt.scatter(rep_points[:,0],rep_points[:,1],c=color_list[c_ind],marker='x')
            plt.scatter(center[0],center[1],c=color_list[c_ind],marker='+')

        plt.legend(loc='lower left',bbox_to_anchor=[1,0])
        plt.show()
    
    
class Cure():
    def __init__(self,N,c,alpha,linkage='cure_single',L=2,visualize = False):
        self.N = N
        self.c = c
        self.alpha = alpha
        self.linkage = linkage
        self.L = L
        self.visualize = visualize
        # Will be assigned later
        self.data = None
        self.tree = None
        self.queue = None
        self.cov_i = None
        # Outputs
        self.labels_ = None
        self.clusters_ = None
        self.reps_ = None
        self.centers_ = None 
        self.num_reps_ = None 
          
    def create_tree(self):
        if self.L!=1 and self.L!=2:
            # Or change it to minority class
            self.cov_i = calc_cov_i(self.data)
        
        representatives, payloads = [], []
        for cluster in self.queue:
            for rep in cluster.rep_points:
                representatives.append(rep)
                payloads.append(cluster)
        # initialize it using constructor to have balanced tree at the beginning to ensure the highest performance
        # when we have the biggest amount of nodes in the tree.
        self.tree = kdtree_(representatives,payloads=payloads,cov_i=self.cov_i)
        
    def create_queue(self):
        # each point in one cluster
        self.queue = [Cluster(i,self.data[i],1) for i in range(len(self.data))]
        
        # set closest clusters
        for i in range(0, len(self.queue)):
            minimal_distance = float('inf')
            closest_index_cluster = -1
            
            for k in range(0, len(self.queue)):
                if i != k:
                    dist = self.queue[i].clus_dist(self.queue[k],linkage=self.linkage,L=self.L,cov_i=self.cov_i)
                    if dist < minimal_distance:
                        minimal_distance = dist
                        closest_index_cluster = k
            
            self.queue[i].closest = self.queue[closest_index_cluster]
            self.queue[i].distance = minimal_distance
            
        # sort clusters: minimal distance one to be the first
        self.queue.sort(key=lambda x: x.distance, reverse = False) 
    
    def return_list(self,X):
        if isinstance(X,np.ndarray):
            return X.tolist()
        return X
         
    def remove_rep(self,cluster):
        for rep_point in cluster.rep_points:
            self.tree.remove(rep_point, payload=cluster)
            
    def insert_rep(self,cluster):
        for rep_point in cluster.rep_points:
            self.tree.insert(rep_point, payload=cluster)
    
    def merge(self,neighbor1,neighbor2):
        merged_cluster = Cluster(neighbor1.index,neighbor1.points,neighbor1.num)
        merged_cluster.copy_(neighbor1)
        merged_cluster.merge(neighbor2,self.c,self.alpha,self.L,self.cov_i)
        return merged_cluster
    
    def insert_cluster(self, cluster):
        for index in range(len(self.queue)):
            if cluster.distance < self.queue[index].distance:
                self.queue.insert(index, cluster)
                return
        # or append to the end
        self.queue.append(cluster)

    def relocate_cluster(self, cluster):
        self.queue.remove(cluster)
        self.insert_cluster(cluster)

    def get_outputs(self):
        '''
        @outputs: clusters_,reps_,num_reps_,centers_,labels_
        '''
        self.clusters_ = [cluster for cluster in self.queue]
        self.reps_ = []
        self.centers_ = []
        for cluster in self.clusters_:
            for rep in cluster.rep_points:
                self.reps_.append(rep)
            for center in cluster.center:
                self.centers_.append(center)
        self.num_reps_  = len(self.reps_)
        
        self.labels_ = [0] * len(self.data)
        for cluster_id,cluster in enumerate(self.clusters_):
            for point_index in cluster.index:
                self.labels_[point_index] = cluster_id
            
        return self.clusters_,self.reps_,self.num_reps_,self.centers_,self.labels_
        
    def fit(self,X):
        self.data = self.return_list(X)
        self.create_queue()
        self.create_tree()
        # Merge two clusters if they are nearest to each other until N cluster satisfies
        while(len(self.queue) > self.N):       
            neighbor1 = self.queue[0]
            neighbor2 = self.queue[0].closest
            
            self.queue.remove(neighbor1)
            self.queue.remove(neighbor2)
            
            self.remove_rep(neighbor1)
            self.remove_rep(neighbor2)
            
            merged_cluster = self.merge(neighbor1,neighbor2)
            
            self.insert_rep(merged_cluster)
            
            cluster_relocation_requests = []
            if len(self.queue) > 0:
                merged_cluster.closest = self.queue[0]  # arbitrary cluster from queue
                merged_cluster.distance = merged_cluster.clus_dist(merged_cluster.closest,linkage=self.linkage,L=self.L,cov_i=self.cov_i)

                for item in self.queue:
                    distance = merged_cluster.clus_dist(item,linkage=self.linkage,L=self.L,cov_i=self.cov_i)
                    # Check if distance between new cluster and current is the best than now.
                    if distance < merged_cluster.distance:
                        merged_cluster.closest = item
                        merged_cluster.distance = distance

    
                    # Check if current cluster has removed neighbor.
                    if (item.closest is neighbor1) or (item.closest is neighbor2):
                        
                        # original closest distance < new distance to merged cluster 
                        # find other clusters with the distance in the interval [original closest distance,new distance to merged cluster]
                        if item.distance < distance: 
                            (item.closest, item.distance) = item.closest_cluster(self.tree,self.L,self.cov_i)

                            # TODO: investigation is required. There is assumption that itself and merged cluster
                            # should be always in list of neighbors in line with specified radius. But merged cluster
                            # may not be in list due to error calculation, therefore it should be added manually.
                            # if item.closest is None: 
                            #     item.closest = merged_cluster
                            #     item.distance = distance

                        else: # original closest distance > new distance to merged cluster -> No other cluster can be closer
                            item.closest = merged_cluster
                            item.distance = distance

                        cluster_relocation_requests.append(item) 
            
            self.insert_cluster(merged_cluster)
            for item in cluster_relocation_requests:
                self.relocate_cluster(item)
        self.get_outputs()
        return self.clusters_,self.reps_,self.num_reps_
            #     visualize_cure(clusters,dist,neighbor1,neighbor2,min_dist)

