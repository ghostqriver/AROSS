'''
@brief Agglomerativeclustering's implemented by sklearn 
@param linkage should be 'ward','single','complete','average'
@reference https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b/sklearn/cluster/_agglomerative.py
@example 
    from COS_Funcs.cluster import agg
    from COS_Funcs.utils import read_data

    X,y = read_data('Datasets\\sampledata_new_3.csv')
    num_expected_clusters = 30
    c = 3
    alpha = 0.5
    linkage = 'ward' #'ward'/'single'/'complete'/'average'
    L = 2 #/1/3 L only serves as distance for extracting representatives, won't affect clustering result
    clusters,all_reps,num_reps = agg.Agglomerativeclustering(X,y,N,alpha,linkage,L)
@author yizhi
'''

from sklearn.cluster import AgglomerativeClustering as AgglomerativeClustering_

from utils import get_labels
from utils.dist import calc_cov_i
from .cluster import Cluster

def Agglomerativeclustering(X,y,N,alpha,linkage,L=2):
    
    cov_i = None
    if L not in (1,2):
        # Or change it to minority class
        cov_i = calc_cov_i(X)
        
    minlabel,majlabel = get_labels(y)
        
    # Initialize empty clusters    
    clusters = Cluster.gen_clusters(N)
    
    agg = AgglomerativeClustering_(n_clusters=N,linkage=linkage).fit(X)
    labels = agg.labels_
    
    # For saving cost in parameter optimization
    if alpha is None:
        return None,None,None,labels
    
    # Update cluster objects
    clusters = Cluster.renew_clusters(X,y,labels,clusters,alpha,L,cov_i,minlabel)
    
    # Flatten reps 
    all_reps,num_reps = Cluster.flatten_rep(clusters)
    
    return clusters,all_reps,num_reps,labels


