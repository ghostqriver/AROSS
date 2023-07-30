'''
@brief CURE's implemented by pyclustering
@details Fast when using ccore
@note Should be called when L = 2 and linkage = 'cure_single'
@reference https://github.com/annoviko/pyclustering
@example 
    from COS_Funcs.cluster import cure_pyc
    from COS_Funcs.utils import read_data

    X,y = read_data('Datasets\\sampledata_new_3.csv')
    num_expected_clusters = 30
    c = 3
    alpha = 0.5
    L = 2 #/1/3 Here the L only serves as distance for extracting representatives, won't affect clustering result
    clusters,all_reps,num_reps = cure.Cure(X,num_expected_clusters,c,alpha,linkage,L,visualize = False)
@author yizhi
'''

from pyclustering.cluster.cure import cure as pyc_cure
from utils import get_labels
from utils.dist import calc_cov_i
from .cluster import Cluster

def Cure(X,y,N,c,alpha,L=2,ccore=True):

    cov_i = None
    if L not in (1,2):
        # Or change it to minority class
        cov_i = calc_cov_i(X)
        
    if c == 0:
        # set a default c for pyclustering's cure
        c_ = c
        c = 5
        
    minlabel,majlabel = get_labels(y)        
    clusters = Cluster.gen_clusters(N,c_)
    cure_instance = pyc_cure(X,N,c,alpha,ccore=ccore)
    cure_instance.process()
    cure_clusters = cure_instance.get_clusters()
    labels = pyc_cure2label(len(X),cure_clusters)
    if alpha is None and c is None:
        # For saving running cost in parameter optimization
        return None,None,None,labels
    # Be careful here, the representative points in clusters are generate by Cluster class
    clusters = Cluster.renew_clusters(X,y,labels,clusters,alpha,L,cov_i,minlabel,majlabel)
    # Use its own representative points
    rep_points = cure_instance.get_representors()
    all_reps,num_reps = pyc_cure_flatten(rep_points)

    return clusters,all_reps,num_reps,labels

# Functions for transforming pyclustering.cluster.cure to COS needed format
def pyc_cure2label(length,clusters):
    '''
    Transform pyclustering.cluster.cure 's output cluster to cluster labels of data points
    '''
    labels = [0 for i in range(length)]

    for cluster_id,point_indexs in enumerate(clusters):
        for point_index in point_indexs:
            labels[point_index] = cluster_id
            
    return np.array(labels)

def pyc_cure_flatten(rep_points):
    '''
    flatten pyclustering.cluster.cure 's representative points
    '''
    all_reps = []
    for i in rep_points:
        for j in i:
            all_reps.append(j)
    return all_reps,len(all_reps)

