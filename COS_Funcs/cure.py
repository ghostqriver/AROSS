"""

@author: wangyizhi
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time,os
# from COS_Funcs.dist import calc_cov_i,calc_dist
# from . import visualize as V
from dist import calc_cov_i,calc_dist
import visualize as V

cov_i = None

class Cluster:
    '''
    The object of clusters during the CURE algorithm running
    .index: the indices of points in this cluster, initially it is each point itselves' index 
    .points: the values of points in this cluster, initially it is each point itself 
    .center: the mean value of all points in the cluster, will be used when shrinking the rep points, initially it is each point it self 
    .rep_points: the shrinked representative points in the cluster, initially it is each point itself 
    .num: how many points in this cluster now

    renew_center(self,another_cluster): when merge two cluster together, calculate the mean value of all points to get a center of the cluster
    renew_paras(self,another_cluster): when merge two cluster together, renew the number of points
    add_shrink(self,tmp_repset,alpha): shrink the representative points to the centroid after the merging
    merge(self,another_cluster,c,alpha): merge two cluster into one
    clus_dist(self,another_cluster): calculate the minimum distance between representative points between two clusters
    
    '''
    def __init__(self,index,point,num = 1):
        self.index = [index]
        self.points = [point]
        self.center = point
        self.rep_points = [point]
        self.num = num
   
    def renew_center(self,another_cluster):
        '''
        Calculate the center 
        It calculated by weight(num of points) * mean values
        '''
        self.center = (another_cluster.num*another_cluster.center+self.num*self.center)/(another_cluster.num+self.num)
        
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
        self.rep_points = tmp_repset + alpha * (self.center - tmp_repset) # in np.array self.center - tmp_repset will be calculate as [self.center-tmp_repset[0],self.center-tmp_repset[1],self.center-tmp_repset[2]....]
    
    def merge(self,another_cluster,c,alpha,L):
        '''
        Merge two cluster into one,
        firstly renew the parameters and center,
        secondly renew the representative points: 1st rep - farest to centroid, 2nd rep - farest to representative set (the minimum distance from this point to the representative points is maximum), 3rd rep - same to 2 
        c: the maximum number of representative points each cluster
        alpha: the given shrink parameter
        '''
        self.renew_center(another_cluster)
        self.renew_paras(another_cluster)
        if self.num<=c: # if total number of points less than c, the representative points will be points itselves
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
        
        
    def clus_dist(self,another_cluster,linkage='cure_single',L=2):
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
    
    
    @staticmethod
    def gen_clusters(X):
        '''
        Initialize each points being in one cluster 
        '''
        num_clusters = len(X)
        clusters = [Cluster(i,X[i],1) for i in range(num_clusters)] # i:index   X[i]:point   num:number of points in the cluster
        return clusters

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
    
    
class dist_matrix():
    '''
    The object of distance matrix during the CURE algorithm running
    .matrix    : the nxn distance matrix, n is the number of clusters in this iteratio
    '''
    def __init__(self,X,linkage,L):
        self.matrix = np.zeros([len(X),len(X)])
        self.gen_matrix(X,linkage,L)
        
    def gen_matrix(self,X,linkage,L):
        
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if i>=j:
                    self.matrix[i][j]=float('inf')
                else:
                    if linkage=='cure_ward':
                        self.matrix[i][j] = np.var(np.vstack([X[i],X[j]]),axis=0).mean()
                    else:
                        self.matrix[i][j] = calc_dist(X[i],X[j],L=L,cov_i=cov_i)  
                    
    def nearest_neighbor(self):
        min_dist = np.min(self.matrix)
        neighbors = np.where(self.matrix == min_dist)
        neighbor1 = neighbors[0][0]
        neighbor2 = neighbors[1][0]      
        return (neighbor1,neighbor2,min_dist)
      
    def renew_matrix(self,clusters,neighbor1,neighbor2,linkage,L):
        '''
        Change the distance matrix information, calculate the distance use the representative pts
        '''
        for i in range(self.matrix.shape[0]):
            if i < neighbor1:
                self.matrix[i,neighbor1] = clusters[neighbor1].clus_dist(clusters[i],linkage,L)
            if i > neighbor1:
                self.matrix[neighbor1,i] = clusters[neighbor1].clus_dist(clusters[i],linkage,L)
        self.matrix = np.delete(self.matrix,neighbor2,axis=1)
        self.matrix = np.delete(self.matrix,neighbor2,axis=0)
    
    def visualize_matrix(self):
        '''
        Visualize the distance matrix in a table for testing
        '''
        pd.set_option('precision',2)  
        pd.set_option('display.width', 100)
        pd.set_option('expand_frame_repr', False)
        df = pd.DataFrame(self.matrix)
        print(df)
        
        
def visualize_cure(clusters,dist,neighbor1 = None,neighbor2 = None,min_dist = None):
    '''
    Will be called when the visualize = True in Cure
    '''
    Cluster.visualize_cluster(clusters)
    dist.visualize_matrix()   
    if neighbor1 != None:
        print(neighbor1,'and',neighbor2,'cluster will be merged, with distance',min_dist,'.')
    else:
        print('Over.')
    print('-'*60)
    
    
def Cure(X,num_expected_clusters,c,alpha,linkage='cure_single',L=2,visualize = False):
    '''
    The Cure algorithm, will return clusters,all_reps,num_reps(list of cluster objects,all representative points,number of representative points)
    X: the data
    num_expected_clusters: N, how many clusters you want
    c: number of representative points in each cluster
    alpha: the given shrink parameter, the bigger alpha is the closer the representative to the centroid
    linkage: 
        if 'cure_single' - calculate a nearest distance using representative points as the distance of two clusters, default value;
        if 'cure_complete' - calculate a furthest distance using representative points as the distance of two clusters;  
        if 'cure_average' - calculate a average distance using representative points as the distance of two clusters;
        if 'cure_centroid' or 'centroid' - calculate a distance using centorids as the distance of two clusters; 
        if 'cure_ward' - calculate the variance if merging two clusters as the distance of two clusters;  
    L: the distance metric, L=1 the Manhattan distance, L=2 the Euclidean distance, by default L=2
    visualize: if set to true, it will show the clusters and distance matrix after each merging, only works for 2d dataset for testing
    '''
    clusters = Cluster.gen_clusters(X)
    
    if L!=1 and L!=2:
        # Or change it to minority class
        global cov_i
        cov_i = calc_cov_i(X)
        
    dist = dist_matrix(X,linkage,L)
    
    num_clusters = len(clusters)
        
    # Merge two cluster until reach num_expected_clusters(N)
    while(num_clusters > num_expected_clusters):
        
        neighbor1,neighbor2,min_dist = dist.nearest_neighbor()

        if visualize == True:
            visualize_cure(clusters,dist,neighbor1,neighbor2,min_dist)
            
        clusters[neighbor1].merge(clusters[neighbor2],c,alpha,L)
        
        dist.renew_matrix(clusters,neighbor1,neighbor2,linkage,L)
               
        # Drop the unused clusters' informations
        del(clusters[neighbor2])
        
        # Decrease the total number of clusters
        num_clusters=num_clusters-1

    if visualize == True:
        visualize_cure(clusters,dist)
    
    # Flatten all representative points 
    lst = [] 
    for i in clusters:
        for j in i.rep_points:
            lst.append(j)  
    all_reps = np.array(lst) 
    num_reps = (all_reps.shape[0])
         

    return clusters,all_reps,num_reps