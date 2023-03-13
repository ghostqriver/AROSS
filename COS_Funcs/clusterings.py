from sklearn.cluster import AgglomerativeClustering
from pyclustering.cluster.cure import cure as pyc_cure
import numpy as np
import pandas as pd

from . import choose_para as choose

def get_labels(y):
    '''
    Return the minority class's label and majority class's label when given all labels y
    Only works well on binary dataset as yet
    return minlabel,int(majlabel)
    '''
    valuecounts=pd.Series(y).value_counts().index
    majlabel=valuecounts[0]
    minlabel=valuecounts[1:]
    if len(minlabel)==1:
        minlabel=int(minlabel[0])
    return minlabel,int(majlabel)


def calc_dist(vecA, vecB, L=2):
    '''
    Calculate the distance between points
    also can be implenmented using scipy.spatial.distance.pdist((vecA,vecB),'minkowski',p=1)/scipy.spatial.distance.pdist((vecA, vecB),'euclidean')
    Because when there is no string type feature this function can now work, but if there is, then we need to consider what is the distance between 0 and 3 which should be 1 but not 3 and some times 1 is too big for our dataset
    L: L=1 the Manhattan distance, L=2 the Euclidean distance, by default L=2.
    '''
    if L ==1 : 
        return np.abs(vecA - vecB).sum()
    else :
        return np.sqrt(np.power(vecA - vecB, 2).sum())


class Cluster:
    def __init__(self,c):
        self.index = []
        self.points = []
        self.labels = []
        self.num = 0
        self.center = 0
        self.rep_points = []
        self.c = c
        

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
        if self.c == 0:
            self.c = choose.choose_c(self)

    def add_shrink(self,tmp_repset,alpha):
        if len(tmp_repset) != 0:
            self.rep_points = tmp_repset + alpha * (self.center - tmp_repset) 
    
    def renew_rep(self,alpha,L):
        '''
        renew the para and representative points of the cluster
        '''
        if self.num <= self.c: # if total number of points less than c, the representative points will be points itselves
            tmpSet = self.points
        else:
            tmpSet = []
            for i in range(self.c):
                maxDist = 0
                for p in self.points:
                    if i==0:
                        minDist = calc_dist(p,self.center,L)
                    else:
                        # for a given p, if p's min distance to any q in tmpset is biggest, then p is next representative point 
                        minDist = np.min([calc_dist(p,q,L) for q in tmpSet])
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
    def renew_clusters(X,y,labels,clusters,c,alpha,L,minlabel,majlabel): 
        '''
        Put the output of Agglomerative clustering into Cluster objects
        '''
        for index,label in enumerate(labels):
            clusters[label].renew_point(index,X[index],y[index])
        for cluster in clusters:
            cluster.renew_para(minlabel,majlabel)
            cluster.renew_rep(alpha,L)
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


# Functions for pyclustering.cluster.cure
def pyc_cure2label(length,clusters):
    '''
    Transform pyclustering.cluster.cure 's output cluster to cluster labels of data points
    '''
    labels = [0 for i in range(length)]

    for cluster_id,point_indexs in enumerate(clusters):
        for point_index in point_indexs:
            labels[point_index] = cluster_id
            
    return labels


def pyc_cure_flatten(rep_points):
    '''
    flatten pyclustering.cluster.cure 's representative points
    '''
    all_reps = []
    for i in rep_points:
        for j in i:
            all_reps.append(j)
    return all_reps,len(all_reps)


def clustering(X,y,N,c,alpha,linkage,L=2,minlabel=None,majlabel=None):
    '''
    linkage = ward,single,complete,average
    '''
    if minlabel == None and majlabel == None:
        minlabel,majlabel = get_labels(y)
        
    clusters = Cluster.gen_clusters(N,c)
    
    if linkage in ['ward','single','complete','average']:
        agg = AgglomerativeClustering(n_clusters=N, linkage=linkage).fit(X)
        labels = agg.labels_
        # into cluster objects
        clusters = Cluster.renew_clusters(X,y,labels,clusters,c,alpha,L,minlabel=minlabel,majlabel=majlabel)
        # Flatten all representative points 
        all_reps,num_reps = Cluster.flatten_rep(clusters)
    
    elif linkage == 'pyc_cure':
        cure_instance = pyc_cure(X,N,c,alpha)
        cure_instance.process()
        cure_clusters = cure_instance.get_clusters()
        labels = pyc_cure2label(len(X),cure_clusters)
        # Be careful here, the representative points in clusters are generate by Cluster class it self
        clusters = Cluster.renew_clusters(X,labels,clusters,c,alpha,L)
        # Use its own representative points
        rep_points = cure_instance.get_representors()
        all_reps,num_reps = pyc_cure_flatten(rep_points)

    return clusters,all_reps,num_reps


