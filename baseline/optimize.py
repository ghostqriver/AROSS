import numpy as np

import matplotlib.pyplot as plt

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








