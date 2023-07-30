import math
from scipy import stats
import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
from cluster import clustering

from utils.visualize import plot_line_color,plot_line_width,figsize,vline_color,vline_width,vline_style,def_figure
from utils import get_labels
from baseline.classifiers import do_classification
from baseline.metrics import calc_score


def choose_para(X_train,y_train,X_test,y_test,classifier,metric,N,linkage,L=2):
    '''
    @brief Optimize the necessary parameters of COS
    @return N,alpha,c
    '''
    N = choose_N(X_train,y_train,linkage=linkage,L=2)
    alpha,_ = choose_alpha(X_train,y_train,X_test,y_test,classifier,metric,N,linkage,L)
    return N,alpha,0

def choose_alpha(X_train,y_train,X_test,y_test,classifier,metrics,N,linkage,L=2,all_safe_weight=1,IR=1):

    # Pick the best alpha based on 'recall'
    det_metric = 'recall'
    if det_metric not in metrics:
        metrics.append(det_metric)
        
    pos_label = get_labels(y_train)[0]
    best_score = {}
    best_score[det_metric] = 0 - np.inf
    best_alpha = 0
    
    score_ls = []
    safe_min_neighbor_ls = []
    all_min_neighbor_ls = []
        
    for alpha in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        X_gen,y_gen,safe_min_neighbors,all_min_neighbors = COS(X_train,y_train,N,0,alpha,linkage=linkage,L=L,all_safe_weight=all_safe_weight,IR=IR)
        y_pred,y_pred_proba = do_classification(X_gen,y_gen,X_test,classifier)#,metric)
        
        score = {}
        for metric in metrics:
            score[metric] = calc_score(metric,y_test,y_pred,y_pred_proba,pos_label)
        
        score_ls.append(score)
        safe_min_neighbor_ls.append(safe_min_neighbors)
        all_min_neighbor_ls.append(all_min_neighbors)
        # print('alpha:',alpha,'| score:',score)
        if score[det_metric] > best_score[det_metric]:
            best_score = score
            best_alpha = alpha
            best_x = X_gen
            best_y = y_gen
    return best_alpha,best_score,score_ls,safe_min_neighbor_ls,all_min_neighbor_ls,best_x,best_y

def choose_N(X_train,y_train,linkage,L=2):
    step = 5
    div = 3
    max_N = math.ceil(len(X_train)/div)
    N = BIC(X_train,y_train,max_N,step,linkage,L)
    while N >= max_N-step and div > 1:
        div = div - 1
        max_N = math.ceil(len(X_train)/div)
        N = BIC(X_train,y_train,max_N,step,linkage,L)
    while N == 1 and step > 1:
        step -= 2
        N = BIC(X_train,y_train,max_N,step,linkage,L)
    
    if N == 1 or N >= max_N-step:
    # If the BIC NOT works at all
        step = 5
        div = 3
        max_N = math.ceil(len(X_train)/div)
        N = PURITY(X_train,y_train,max_N,step,linkage,L) 
    
    # To avoid error in cleveland
    if N is None:
        N = 3
    return N

def PURITY(X_train,y_train,max_N=None,step=5,linkage='ward',L=2):
    # Won't extract reps when optimizing para
    alpha = None
    c = None
    if max_N == None:
        max_N = math.ceil(len(X_train)/3)

    p_scores = {}
    N_list = list(range(10,max_N,step))    
    
    for n in N_list: 
        _,_,_,labels = clustering(X_train,y_train,n,c,alpha,linkage,L)
        p_score = purity_score(y_train, labels)
        p_scores[n] = (p_score) 

    # def_figure()
    # plt.plot(list(p_scores.keys()),list(p_scores.values()),label='Purity scores',color=plot_line_color,linewidth=plot_line_width,)
    kn = KneeLocator(list(p_scores.keys()),list(p_scores.values()),curve='concave', direction='increasing',online=True,)
    # plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1],  linestyles=vline_style,linewidth=vline_width,colors=vline_color)
    # plt.legend()
    # plt.show()
    return kn.knee    
  
def BIC(X_train,y_train,max_N=None,step=5,linkage='ward',L=2):
    # Won't extract reps when optimizing para
    alpha = None
    c = None
    if max_N == None:
        max_N = math.ceil(len(X_train)/3)
        
    BIC_scores = {}
    N_list = list(range(1,max_N,step))    
    max_bic = 0 - np.inf
    max_bic_n = 1
    
    for n in N_list: 
        _,_,_,labels = clustering(X_train,y_train,n,c,alpha,linkage,L)
        BIC_score = bic_score(X_train,labels)
        if BIC_score > max_bic:
            max_bic_n = n
            max_bic = BIC_score
        BIC_scores[n] = BIC_score
    
    # def_figure()
    plt.plot(list(BIC_scores.keys()),list(BIC_scores.values()),label='BIC scores',color=plot_line_color,linewidth=plot_line_width,)
    plt.vlines(max_bic_n,plt.ylim()[0], plt.ylim()[1], linestyles=vline_style,linewidth=vline_width,colors=vline_color)
    plt.legend()
    plt.show()
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



# def cos_para_show(datasets,N,c,alpha,linkage='ward',all_safe_gen=G.Smote_Generator,half_safe_gen=G.Smote_Generator,metric='recall',classification_model='random_forest',k=10,pos_label=None):
#     '''
#     Did not choose generator yet
#     '''
#     dataset_path = 'Dataset/'
#     changing_para = {}
#     if isinstance(c,list):
#         changing_para['c'] = c
#     elif isinstance(N,list):
#         changing_para['N'] = N
#     elif isinstance(alpha,list):
#         changing_para['alpha'] = alpha
#     elif isinstance(linkage,list):
#         changing_para['linkage'] = linkage
        
        
#     dataset_results = {}
#     for dataset in datasets: 
#         dataset_results[dataset] = []
#         for paras in changing_para.values():
#             for para in tqdm(paras):
                
#                 X,y = baseline.read_data(dataset_path,dataset)

#                 if pos_label == None:
#                     pos_label = cos_bf.get_labels(y)[0]

#                 for random_state in range(k):
#                     scores = [] 
#                     X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=random_state)

#                     if 'c' in changing_para.keys():
#                         c = para 
#                         X_train,y_train,num_all_safe,num_half_safe = cos_bf.COS(X_train,y_train,N,c,alpha,linkage=linkage,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen)
#                     elif 'N' in changing_para.keys():
#                         N = para
#                         X_train,y_train,num_all_safe,num_half_safe = cos_bf.COS(X_train,y_train,N,c,alpha,linkage=linkage,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen)
#                     elif 'alpha' in changing_para.keys():
#                         alpha = para
#                         X_train,y_train,num_all_safe,num_half_safe = cos_bf.COS(X_train,y_train,N,c,alpha,linkage=linkage,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen)
#                     elif 'linkage' in changing_para.keys():
#                         linkage = para
#                         X_train,y_train,num_all_safe,num_half_safe = cos_bf.COS(X_train,y_train,N,c,alpha,linkage=linkage,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen)

#                     y_pred = baseline.do_classification(X_train,y_train,X_test,classification_model)
#                     scores.append(baseline.calc_score(metric,y_test,y_pred,pos_label))
                
#                 dataset_results[dataset].append(np.mean(scores))

#     para_name = list(changing_para.keys())[0]
#     paras = list(changing_para.values())[0]
#     plt.figure(figsize=(8,8))
#     plt.xlabel(para_name)
#     plt.ylabel(metric)
#     for dataset in dataset_results.keys():
#         results = dataset_results[dataset]
#         plt.plot(paras,results,label=dataset)
#     plt.legend()
# #     return changing_para,dataset_results











# def choose_alpha(dataset,N=None,c='sample',alpha_list=None,linkage='ward',all_safe_gen=G.Smote_Generator,half_safe_gen=G.Smote_Generator,metric='recall',classification_model='random_forest',k=10,pos_label=None):
    
#     dataset_path = 'Dataset/'
#     best_alpha = -1
#     best_recall = -1
#     if alpha_list == None:
#         alpha_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    
#     if N == None:
#         N = choose_N(dataset)

#     results = []
#     for alpha in tqdm(alpha_list):
  
#         X,y = baseline.read_data(dataset_path,dataset)

#         if pos_label == None:
#             pos_label = cos.get_labels(y)[0]

#         for random_state in range(k):
#             scores = [] 
#             X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=random_state)

            
#             X_train,y_train,num_all_safe,num_half_safe = cos.COS(X_train,y_train,N,c,alpha,linkage=linkage,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen)

#             y_pred = baseline.do_classification(X_train,y_train,X_test,classification_model)

#             scores.append(baseline.calc_score(metric,y_test,y_pred,pos_label))
        
#         result = np.mean(scores)
#         if result > best_recall:
#             best_alpha = alpha
#             best_recall = result

#         results.append(np.mean(scores))

    
#     plt.figure(figsize=(8,8))
#     plt.xlabel('alpha')
#     plt.ylabel(metric)
#     plt.plot(alpha_list,results,label=dataset)
#     plt.vlines(best_alpha, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
#     plt.legend()
    
#     return N,best_alpha


# def sample_size_(cluster):
#     # print('All instances:',cluster.num,end=', ')
#     Z_score = 2.58
#     margin_error = 0.05
#     std_dev = 0.015 #np.sqrt(np.mean(np.var(cluster.points,axis=0)))
#     N = cluster.num
#     size = ((Z_score**2 * std_dev * (1-std_dev)) / (margin_error**2)) / (1 + ((Z_score**2 * std_dev * (1-std_dev))/(margin_error**2 * N)))
#     # print('extract representative points',size)
#     return math.ceil(size)








