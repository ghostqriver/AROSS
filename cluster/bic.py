'''
@brief decide n_cluster by BIC
@author yizhi
'''

import math
import numpy as np
from kneed import KneeLocator
from scipy import stats

from utils.visualize import plot_line_color,plot_line_width,figsize,vline_color,vline_width,vline_style,def_figure
from . import clustering


def choose_N(X,y,linkage,L=2):
    '''
    @brief determine n_cluster by BIC or PURITY
    '''
    step = 5
    div = 3
    max_N = math.ceil(len(X)/div)
    N = BIC(X,y,max_N,step,linkage,L)
    while N >= max_N-step and div > 1:
        div = div - 1
        max_N = math.ceil(len(X)/div)
        N = BIC(X,y,max_N,step,linkage,L)
    while N == 1 and step > 1:
        step -= 2
        N = BIC(X,y,max_N,step,linkage,L)
    
    # If the BIC does NOT works at all
    if N == 1 or N >= max_N-step:
        step = 5
        div = 3
        max_N = math.ceil(len(X)/div)
        N = PURITY(X,y,max_N,step,linkage,L) 
    # To avoid error in cleveland
    if N is None:
        N = 3
    return int(N)


def PURITY(X,y,max_N,step,linkage,L=2):
    
    alpha = None
    c = None

    p_scores = {}
    N_list = list(range(10,max_N,step))    
    
    for n in N_list: 
        _,_,_,labels = clustering(X,y,n,alpha,linkage,L)
        p_score = purity_score(y, labels)
        p_scores[n] = (p_score) 

    # def_figure()
    # plt.plot(list(p_scores.keys()),list(p_scores.values()),label='Purity scores',color=plot_line_color,linewidth=plot_line_width,)
    kn = KneeLocator(list(p_scores.keys()),list(p_scores.values()),curve='concave', direction='increasing',online=True,)
    # plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1],  linestyles=vline_style,linewidth=vline_width,colors=vline_color)
    # plt.legend()
    # plt.show()
    return kn.knee    
  
  
def BIC(X,y,max_N,step,linkage,L=2):

    alpha = None
    c = None
        
    BIC_scores = {}
    N_list = list(range(1,max_N,step))    
    max_bic = 0 - np.inf
    max_bic_n = 1
    
    for n in N_list: 
        _,_,_,labels = clustering(X,y,n,alpha,linkage,L)
        BIC_score = bic_score(X,labels)
        if BIC_score > max_bic:
            max_bic_n = n
            max_bic = BIC_score
        BIC_scores[n] = BIC_score
    
    # def_figure()
    # plt.plot(list(BIC_scores.keys()),list(BIC_scores.values()),label='BIC scores',color=plot_line_color,linewidth=plot_line_width,)
    # plt.vlines(max_bic_n,plt.ylim()[0], plt.ylim()[1], linestyles=vline_style,linewidth=vline_width,colors=vline_color)
    # plt.legend()
    # plt.show()
    return max_bic_n 


def bic_score(X: np.ndarray, labels: np.array):
    """
    @brief BIC score for the goodness of fit of clusters (The greater the better)
    @source https://github.com/smazzanti/are_you_still_using_elbow_method/blob/main/are-you-still-using-elbow-method.ipynb
    @detail This Python function is translated from the Golang implementation by the author of the paper. 
            The original code is available here: https://github.com/bobhancock/goxmeans/blob/a78e909e374c6f97ddd04a239658c7c5b7365e5c/km.go#L778
    """
    n_points = len(labels)
    n_clusters = len(set(labels))
    n_dimensions = X.shape[1]

    n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1

    loglikelihood = 0
    for label_name in set(labels):
        X_cluster = X[labels == label_name]
        n_points_cluster = len(X_cluster)
        centroid = np.mean(X_cluster, axis=0)
        if len(X_cluster) == 1 :
            variance = 0
            loglikelihood += 0
        else :
            variance = np.sum((X_cluster - centroid) ** 2) / (len(X_cluster) - 1)
            if variance == 0:
                loglikelihood += 0
            else:
                loglikelihood += \
                n_points_cluster * np.log(n_points_cluster) \
                - n_points_cluster * np.log(n_points) \
                - n_points_cluster * n_dimensions / 2 * np.log(2 * math.pi * variance) \
                - (n_points_cluster - 1) / 2
    
    bic = loglikelihood - (n_parameters / 2) * np.log(n_points)
        
    return bic


def purity_score(y: np.ndarray, labels: np.ndarray):
    '''
    @brief purity score for clusters (elbow is better)
    '''
    cluster_purities = []
    # loop through clusters and calculate purity for each
    for pred_cluster in np.unique(labels):
        
        filter_ = labels == pred_cluster
#         print(filter_)
        gt_partition = y[filter_]
        pred_partition = labels[filter_]
        
        # figure out which gt partition this predicted cluster contains the most points of
        mode_ = stats.mode(gt_partition)
        max_gt_cluster = mode_[0][0]
        
        # how many points in the max cluster does the current cluster contain
        pure_members = np.sum(gt_partition == max_gt_cluster)
        cluster_purity = pure_members / len(pred_partition)
        
        cluster_purities.append(pure_members)
    
    purity = np.sum(cluster_purities) / len(labels)
    return purity