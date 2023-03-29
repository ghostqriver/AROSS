import numpy as np
from COS_Funcs.utils.dist import calc_dist
from COS_Funcs.cos import optmize as choose

class Cluster:
    def __init__(self,c):
        self.index = []
        self.points = []
        self.labels = []
        self.num = 0
        self.center = 0
        self.rep_points = []
        self.c = c 
        self.num_min = 0

    def renew_point(self,index,point,label):
        '''
        renew the points in the cluster by iteration after the clustering
        '''
        self.index.append(index)
        self.points.append(point)
        self.labels.append(label)

    def renew_para(self,minlabel,majlabel):
        '''
        renew the paras after finishing renewing all points
        '''
        self.center =  np.mean(self.points,axis=0)
        self.num = len(self.points)   
        self.points = np.array(self.points)
        self.labels = np.array(self.labels)
        self.num_min = len(self.labels[self.labels == minlabel])
        if self.c == 0:
            self.c = choose.choose_c(self)
        

    def add_shrink(self,tmp_repset,alpha):
        if len(tmp_repset) != 0:
            self.rep_points = tmp_repset + alpha * (self.center - tmp_repset) 
    
    def renew_rep(self,alpha,L,cov_i):
        '''
        renew the para and representative points of the cluster
        '''
        if self.c == 1:
            tmpSet = self.center
        elif self.num <= self.c: # if total number of points less than c, the representative points will be points itselves
            tmpSet = self.points
        else:
            tmpSet = []
            for i in range(self.c):
                maxDist = 0
                for p in self.points:
                    if i==0:
                        minDist = calc_dist(p,self.center,L,cov_i)
                    else:
                        # for a given p, if p's min distance to any q in tmpset is biggest, then p is next representative point 
                        minDist = np.min([calc_dist(p,q,L,cov_i) for q in tmpSet])
                    if minDist >= maxDist:
                        maxPoint = p
                        maxDist = minDist
                tmpSet.append(maxPoint)
        self.add_shrink(tmpSet,alpha)

    @staticmethod
    def gen_clusters(N,c):
        '''
        Initialize clusters 
        '''
        num_clusters = N
        if isinstance(c,int):
            clusters = [Cluster(c) for i in range(num_clusters)] 
        else:
            clusters = [Cluster(0) for i in range(num_clusters)] 
        return clusters

    @staticmethod
    def renew_clusters(X,y,labels,clusters,alpha,L,cov_i,minlabel,majlabel): 
        '''
        Put the output of Agglomerative clustering into Cluster objects
        '''
        for index,label in enumerate(labels):
            clusters[label].renew_point(index,X[index],y[index])
        for cluster in clusters:
            cluster.renew_para(minlabel,majlabel)
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
        all_reps = np.array(lst) 
        num_reps = (all_reps.shape[0])
        return all_reps,num_reps