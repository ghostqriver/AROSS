'''
@brief Agglomerativeclustering's implemented by sklearn 
@details Linkage can be 'ward','single','complete','average'
@reference https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b/sklearn/cluster/_agglomerative.py
@example 
    from COS_Funcs.cluster import agg
    from COS_Funcs.utils import read_data

    X,y = read_data('Datasets\\sampledata_new_3.csv')
    num_expected_clusters = 30
    c = 3
    alpha = 0.5
    linkage = 'ward' #'ward'/'single'/'complete'/'average'
    L = 2 #/1/3 Here the L only serves as distance for extracting representatives, won't affect clustering result
    clusters,all_reps,num_reps = agg.Agglomerativeclustering(X,y,N,c,alpha,linkage,L)
'''


from sklearn.cluster import AgglomerativeClustering as AgglomerativeClustering_
from COS_Funcs.utils import get_labels
from COS_Funcs.utils.dist import calc_cov_i
from .cluster import Cluster

def Agglomerativeclustering(X,y,N,c,alpha,linkage,L=2):
    '''
    linkage = ward,single,complete,average
    '''
    
    cov_i = None
    if L not in (1,2):
        # Or change it to minority class
        cov_i = calc_cov_i(X)
        
    minlabel,majlabel = get_labels(y)        
    clusters = Cluster.gen_clusters(N,c)
    
    agg = AgglomerativeClustering_(n_clusters=N, linkage=linkage).fit(X)
    labels = agg.labels_
    # into cluster objects
    clusters = Cluster.renew_clusters(X,y,labels,clusters,alpha,L,cov_i,minlabel,majlabel)
    # Flatten all representative points 
    all_reps,num_reps = Cluster.flatten_rep(clusters)
    
    return clusters,all_reps,num_reps


