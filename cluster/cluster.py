'''
@brief extract reps from clusters
@author yizhi
'''
import numpy as np
import math

from utils.dist import calc_dist


def choose_c(cluster):
    N = cluster.num
    num_min = cluster.num_min
    c = sample_size(N,num_min)
    return c

def sample_size(N,num_min):
    p = num_min/N
    # size1 might be 0 when num_min == 0 or num_min == N, then set c = 1, choose centroid as rep_point directly
    if p==0 and N >= 9:
        size1 = 0
    elif p==0 or p==1:
        size1 = 1
    else:
        Z = 1.64
        epsilon = 0.05
        e = epsilon + np.log(N)/N
        x = (Z**2 * p * (1-p)) / (e**2)
        size1 = (N * x) / (x + N - 1)
    return math.ceil(size1)


class Cluster:
    def __init__(self):
        self.index = []
        self.points = []
        self.labels = []
        self.num = 0
        self.center = 0
        self.rep_points = []
        self.num_min = 0

    def renew_point(self,index,point,label):
        '''
        @brief renew the points in the cluster by iteration after the clustering
        '''
        self.index.append(index)
        self.points.append(point)
        self.labels.append(label)

    def renew_para(self,minlabel):
        '''
        @brief renew the paras after finishing renewing all points
        '''
        self.center =  np.mean(self.points,axis=0)
        self.num = len(self.points)   
        self.points = np.array(self.points)
        self.labels = np.array(self.labels)
        self.num_min = len(self.labels[self.labels == minlabel])
        self.c = choose_c(self)
        

    def add_shrink(self,tmp_repset,alpha):
        '''
        @brief shrink the representative point toward centroid
        '''
        if len(tmp_repset) != 0:
            self.rep_points = tmp_repset + alpha * (self.center - tmp_repset) 
    
    def renew_rep(self,alpha,L,cov_i):
        '''
        @brief renew the representative points of the cluster
        '''
        if self.c == 0:
            self.rep_points = []
        elif self.c == 1:
            self.rep_points = [self.center]
        # if total number of points less than c, the representative points will be points
        elif self.num <= self.c: 
            tmpSet = self.points
            self.add_shrink(tmpSet,alpha)
        else:
            tmpSet = []
            for i in range(self.c):
                maxDist = 0
                for p in self.points:
                    if i==0:
                        minDist = calc_dist(p,self.center,L,cov_i)
                    else:
                        minDist = np.min([calc_dist(p,q,L,cov_i) for q in tmpSet])
                    if minDist >= maxDist:
                        maxPoint = p
                        maxDist = minDist
                tmpSet.append(maxPoint)
            self.add_shrink(tmpSet,alpha)

    @staticmethod
    def gen_clusters(N):
        '''
        @brief Initialize clusters 
        '''
        clusters = [Cluster() for i in range(N)] 
        return clusters

    @staticmethod
    def renew_clusters(X,y,labels,clusters,alpha,L,cov_i,minlabel): 
        '''
        @brief Put the output of Agglomerative clustering into Cluster objects
        '''
        for index,label in enumerate(labels):
            clusters[label].renew_point(index,X[index],y[index])
        for cluster in clusters:
            cluster.renew_para(minlabel)
            cluster.renew_rep(alpha,L,cov_i)
        return clusters

    @staticmethod
    def flatten_rep(clusters):
        '''
        Iterate all clusters and flatten all representative points as a list
        '''
        lst = []
        for i in clusters:
            for j in i.rep_points:
                lst.append(j)  
        all_reps = np.array(lst,dtype=object) 
        num_reps = (all_reps.shape[0])
        return all_reps,num_reps
