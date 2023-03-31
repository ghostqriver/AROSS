import COS_Funcs.baseline as baseline
# from COS_Funcs.cos import cos
import COS_Funcs.cos.generate as G

import math
from scipy import stats
import numpy as np
from tqdm import tqdm
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

def choose_c(cluster):
    
    N = cluster.num
    num_min = cluster.num_min
    c = sample_size(N,num_min)
    
    return c

def sample_size(N,num_min):
    p = num_min/N
    # size1 might be 0 when num_min == 0 or num_min == N, then set c = 1, choose centroid as rep_point directly
    if p==0 or p==1:
        size1 = 1
    else:
        Z = 1.64
        epsilon = 0.05
        e = epsilon + np.log(N)/N
        x = (Z**2 * p * (1-p)) / (e**2)
        size1 = (N * x) / (x + N - 1)
    return math.ceil(size1)







def cos_para_show(datasets,N,c,alpha,linkage='ward',all_safe_gen=G.Smote_Generator,half_safe_gen=G.Smote_Generator,metric='recall',classification_model='random_forest',k=10,pos_label=None):
    '''
    Did not choose generator yet
    '''
    dataset_path = 'Dataset/'
    changing_para = {}
    if isinstance(c,list):
        changing_para['c'] = c
    elif isinstance(N,list):
        changing_para['N'] = N
    elif isinstance(alpha,list):
        changing_para['alpha'] = alpha
    elif isinstance(linkage,list):
        changing_para['linkage'] = linkage
        
        
    dataset_results = {}
    for dataset in datasets: 
        dataset_results[dataset] = []
        for paras in changing_para.values():
            for para in tqdm(paras):
                
                X,y = baseline.read_data(dataset_path,dataset)

                if pos_label == None:
                    pos_label = cos_bf.get_labels(y)[0]

                for random_state in range(k):
                    scores = [] 
                    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=random_state)

                    if 'c' in changing_para.keys():
                        c = para 
                        X_train,y_train,num_all_safe,num_half_safe = cos_bf.COS(X_train,y_train,N,c,alpha,linkage=linkage,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen)
                    elif 'N' in changing_para.keys():
                        N = para
                        X_train,y_train,num_all_safe,num_half_safe = cos_bf.COS(X_train,y_train,N,c,alpha,linkage=linkage,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen)
                    elif 'alpha' in changing_para.keys():
                        alpha = para
                        X_train,y_train,num_all_safe,num_half_safe = cos_bf.COS(X_train,y_train,N,c,alpha,linkage=linkage,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen)
                    elif 'linkage' in changing_para.keys():
                        linkage = para
                        X_train,y_train,num_all_safe,num_half_safe = cos_bf.COS(X_train,y_train,N,c,alpha,linkage=linkage,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen)

                    y_pred = baseline.do_classification(X_train,y_train,X_test,classification_model)
                    scores.append(baseline.calc_score(metric,y_test,y_pred,pos_label))
                
                dataset_results[dataset].append(np.mean(scores))

    para_name = list(changing_para.keys())[0]
    paras = list(changing_para.values())[0]
    plt.figure(figsize=(8,8))
    plt.xlabel(para_name)
    plt.ylabel(metric)
    for dataset in dataset_results.keys():
        results = dataset_results[dataset]
        plt.plot(paras,results,label=dataset)
    plt.legend()
#     return changing_para,dataset_results



## Cluster purity
def purity(truth, pred):
    cluster_purities = []
    # loop through clusters and calculate purity for each
    for pred_cluster in np.unique(pred):
        
        filter_ = pred == pred_cluster
#         print(filter_)
        gt_partition = truth[filter_]
        pred_partition = pred[filter_]
        
        # figure out which gt partition this predicted cluster contains the most points of
        mode_ = stats.mode(gt_partition)
        max_gt_cluster = mode_[0][0]
        
        # how many points in the max cluster does the current cluster contain
        pure_members = np.sum(gt_partition == max_gt_cluster)
        cluster_purity = pure_members / len(pred_partition)
        
        cluster_purities.append(pure_members)
    
    return np.sum(cluster_purities) / len(pred)


def Purity(dataset,model,max_cluster=None):
    '''
    Choose the best number of clusters by checking Purity of clusters, use ward linkage by default
    '''
    X,y = baseline.read_data(baseline.dataset_path,dataset)
    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y)
    X_train,y_train = X,y
    if max_cluster == None:
        max_cluster = math.ceil(len(X_train)/3)

    p_scores = {}
    N_list = list(range(10,max_cluster ,10))    
    
    for n in N_list: 
        model = AgglomerativeClustering
        agg = model(n_clusters=n).fit(X_train)
        labels = agg.labels_
        p_score = purity(y_train, labels)
#         print(p_score)
        p_scores[n] = (p_score) 

    plt.figure(figsize=(8,8))
    plt.plot(list(p_scores.keys()),list(p_scores.values()))

    kn = KneeLocator(list(p_scores.keys()),list(p_scores.values()),curve='concave', direction='increasing',online=True,)
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cluster Purity Value')
    plt.show()
    return kn.knee


def choose_N(dataset,max_cluster=None):
    
    N = Purity(dataset,max_cluster)
    
    return N


def choose_alpha(dataset,N=None,c='sample',alpha_list=None,linkage='ward',all_safe_gen=G.Smote_Generator,half_safe_gen=G.Smote_Generator,metric='recall',classification_model='random_forest',k=10,pos_label=None):
    
    dataset_path = 'Dataset/'
    best_alpha = -1
    best_recall = -1
    if alpha_list == None:
        alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    
    if N == None:
        N = choose_N(dataset)

    results = []
    for alpha in tqdm(alpha_list):
  
        X,y = baseline.read_data(dataset_path,dataset)

        if pos_label == None:
            pos_label = cos_bf.get_labels(y)[0]

        for random_state in range(k):
            scores = [] 
            X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=random_state)

            
            X_train,y_train,num_all_safe,num_half_safe = cos_bf.COS(X_train,y_train,N,c,alpha,linkage=linkage,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen)

            y_pred = baseline.do_classification(X_train,y_train,X_test,classification_model)

            scores.append(baseline.calc_score(metric,y_test,y_pred,pos_label))
        
        result = np.mean(scores)
        if result > best_recall:
            best_alpha = alpha
            best_recall = result

        results.append(np.mean(scores))

    
    plt.figure(figsize=(8,8))
    plt.xlabel('alpha')
    plt.ylabel(metric)
    plt.plot(alpha_list,results,label=dataset)
    plt.vlines(best_alpha, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.legend()
    
    return N,best_alpha


def sample_size_(cluster):
    # print('All instances:',cluster.num,end=', ')
    Z_score = 2.58
    margin_error = 0.05
    std_dev = 0.015 #np.sqrt(np.mean(np.var(cluster.points,axis=0)))
    N = cluster.num
    size = ((Z_score**2 * std_dev * (1-std_dev)) / (margin_error**2)) / (1 + ((Z_score**2 * std_dev * (1-std_dev))/(margin_error**2 * N)))
    # print('extract representative points',size)
    return math.ceil(size)





def bic_score(X: np.ndarray, labels: np.array):
    """
    # higher is better
    @reference: https://github.com/smazzanti/are_you_still_using_elbow_method/blob/main/are-you-still-using-elbow-method.ipynb
    BIC score for the goodness of fit of clusters.
    This Python function is translated from the Golang implementation by the author of the paper. 
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
        variance = np.sum((X_cluster - centroid) ** 2) / (len(X_cluster) - 1)
        loglikelihood += \
          n_points_cluster * np.log(n_points_cluster) \
          - n_points_cluster * np.log(n_points) \
          - n_points_cluster * n_dimensions / 2 * np.log(2 * math.pi * variance) \
          - (n_points_cluster - 1) / 2
    
    bic = loglikelihood - (n_parameters / 2) * np.log(n_points)
        
    return bic



