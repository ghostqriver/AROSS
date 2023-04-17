dataset_path = 'Datasets/'

oversamplers = ['original','smote','db_smote','smote_d','cure_smote','kmeans_smote','adasyn','somo','symprod',
                'smote_enn','smote_tl','nras','g_smote','rwo_sampling','ans','svm_smote',
                # 'wgan', 'wgan_filter' 
                # 'cos'
                # 'random'
                ] 

classifiers = ['knn','svm','decision_tree','random_forest','mlp','naive_bayes'] 

metrics = ['recall','f1_score','g_mean','kappa','auc','accuracy','precision']

save_path = 'test/'
cos_save_path = 'costest/'
gan_save_path = 'gantest/'
random_save_path = 'test/random/'

import os
import glob
datasets = glob.glob(os.path.join(dataset_path,'*.csv'))

from .ccpc import linkages
from COS_Funcs.utils import *
from COS_Funcs.baseline.classifiers import do_classification
from COS_Funcs.baseline.oversamplers import do_oversampling
from COS_Funcs.baseline.metrics import calc_score
from COS_Funcs.cos import optimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import time
import math
import sys
from tqdm import tqdm
import copy

def baseline(classifiers=classifiers,metrics=metrics,k=10,oversamplers=oversamplers,datasets=datasets,show_folds=True,**args):
    #pd.set_option('precision',5)  
    #pd.set_option('display.width', 100)
    #pd.set_option('expand_frame_repr', False)
    warnings.filterwarnings("ignore") 
    
    if 'cos' in oversamplers:
        path = cos_save_path
    elif 'random' in oversamplers:
        path = random_save_path
    elif 'wgan' in oversamplers:
        path = gan_save_path
    else:
        path = save_path
        
    make_dir(path)

    writers = create_writer(classifiers,metrics,k,oversamplers=oversamplers,datasets=datasets,path=path)
    sys.stdout = Logger(path=path)
    for random_state in tqdm(range(k)):
        
        for dataset in datasets: 
            
            X,y = read_data(dataset)
#             print(dataset)

            for oversampler in oversamplers:
                print(oversampler,end='|')

                X_train,X_test,y_train,y_test = split_data(X,y,random_state=random_state)

                pos_label = get_labels(y)[0]
                
                # try:
                start = time.time()
                if oversampler == 'cos':
                    X_train,y_train,num_all_safe,num_half_safe = do_oversampling(oversampler,X_train,y_train,args) 

                elif oversampler == 'wgan' or oversampler == 'wgan_filter':
                    X_train,y_train = do_oversampling(oversampler,X_train,y_train,X_test=X_test,y_test=y_test,classifier=classifiers[0]) 
                    
                else:
                    X_train,y_train = do_oversampling(oversampler,X_train,y_train) 
                    
                end = time.time()
                
                print('cost:',end-start)
                
                for classifier in classifiers:     
                    y_pred = do_classification(X_train,y_train,X_test,classifier)
                    for metric in metrics:
                        score = calc_score(metric,y_test,y_pred,pos_label)
                        if show_folds:
                            print(random_state+1,'|',dataset,'|',oversampler,'|',classifier,'|',metric,':',end='')
                            print(score)

                        writers[classifier][metric]['scores_df'][random_state][oversampler].loc[dataset] = score
                    
                # except BaseException as e: 
                #     print(oversampler,'cause an error on',dataset)
                #     continue
                
    write_writer(writers,classifiers,metrics,k,oversamplers,datasets)
    return writers

def cos_baseline(classifiers,metrics,datasets=datasets,k=10,linkage=None,L=2,all_safe_weight=1,IR=1,show_folds=True):

    path = cos_save_path
    make_dir(path)
    sys.stdout = Logger(path=path)
    if isinstance(classifiers,str):
        classifiers = [classifiers]
    if isinstance(metrics,str):
        metrics = [metrics]

    K_fold_dict = {}

    file_name = os.path.join(path,cos_file_name(classifiers,metrics))
    dict_file_name = file_name + '_folds.json'
    file_name = file_name + '.xlsx'
    writer = pd.ExcelWriter(file_name)

    index = list(map(lambda x:os.path.basename(x).split('.')[0],datasets))
    
    for classifier in classifiers:     

        for metric in metrics:

            df = pd.DataFrame(columns=['cos'],index=datasets)
            sheet_name_ = sheet_name(metric,classifier,k)
            
            for dataset in datasets: 
                print(dataset)
                if linkage == None:
                    # Choose the linkage from CCPC
                    linkage = linkages[base_file(dataset)]
                try:
                    if 'page-blocks1vs2345' in  dataset or 'oil_spill' in dataset:
                        # The dataset might meet the error
                        scores,score,alphas = None,None,None
                    else:
                        scores,score,alphas = cos_baseline_(dataset,metric,classifier,k=k,linkage=linkage,L=L,all_safe_weight=all_safe_weight,IR=IR,show_folds=show_folds)

                except BaseException as e: 
                    print('COS cause an error on',dataset,'with',classifier)
                    scores = []
                    score = None
                    continue 

                key_name_ = key_name(dataset,sheet_name_)
                K_fold_dict[key_name_] = scores
                K_fold_dict[key_name_+'alpha'] = alphas
                df['cos'].loc[dataset] = score

            df.index = index
            df.to_excel(writer,sheet_name= sheet_name_)

    writer.save()
    save_json(K_fold_dict,dict_file_name)
    print("File saved in",file_name,'and',dict_file_name)
    
def cos_baseline_(dataset,metric,classifier,k=10,linkage='ward',L=2,all_safe_weight=1,IR=1,show_folds=True):
    scores = []
    # For recommend best alpha interval
    alphas = []
    X,y = read_data(dataset)
    for random_state in range(k):
        X_train,X_test,y_train,y_test = split_data(X,y)
        pos_label = get_labels(y)[0]

        # Choose N
        N = optimize.choose_N(X_train, y_train, linkage=linkage, L=L)
        
        # Choose alpha
        best_alpha,best_score = optimize.choose_alpha(X_train,y_train,X_test,y_test,classifier,metric,N,linkage=linkage,L=L,all_safe_weight=all_safe_weight,IR=IR)
        if show_folds:      
            print(random_state+1,'|',dataset,'|',classifier,'|',metric,':',end=' ')
            print(best_score)
        scores.append(best_score)
        alphas.append(best_alpha)
    avg = np.mean(scores)
    if show_folds:      
        print(dataset,'|',classifier,'|',metric,':',end=' ')
        print(avg)
    return scores,avg,alphas


# Class
def create_writer(classifiers,metrics,k,oversamplers,datasets,path):
    writers = dict()
    for classifier in classifiers:
        writers[classifier] = {}
        for metric in metrics:
            writers[classifier][metric] = dict()
            if 'cos' not in oversamplers:
                file_name = base_name(metric,classifier,k)
            file_name = os.path.join(path+file_name)
            writers[classifier][metric]['file_name'] = file_name
            writers[classifier][metric]['writer'] =  pd.ExcelWriter(file_name)
            writers[classifier][metric]['scores_df'] = []
            for i in range(k):
                writers[classifier][metric]['scores_df'].append(pd.DataFrame(columns=oversamplers,index=datasets))
    return writers

def write_writer(writers,classifiers,metrics,k,oversamplers,datasets):
    columns = list(map(lambda x:str.upper(x),oversamplers))
    index = list(map(lambda x:os.path.basename(x).split('.')[0],datasets))
    for classifier in classifiers:
        for metric in metrics:
            writer = writers[classifier][metric]['writer']
            file_name = writers[classifier][metric]['file_name']
            for random_state in range(k): 
                df = writers[classifier][metric]['scores_df'][random_state]
                df.columns = columns
                df.index = index
                df.to_excel(writer,sheet_name= 'fold_'+str(random_state+1))
                tmp_df = copy.deepcopy(df)
                if random_state == 0:
                    avg = tmp_df
                else:
                    avg += tmp_df
            avg = avg/k
            avg.to_excel(writer,sheet_name= 'avg')
            writer.save()
            print("File saved in",file_name)    

# Modify it because parameter name changed
def show_baseline(dataset,random_state=None,pos_label=None,img_name=None,**args):
    
    path = 'test/'
    make_dir(path)

    if img_name == None:
        fn = ''
        for str_ in [j[0]+str(j[1]) for j in [(i,args['args'][i]) for i in args['args']]]:
            fn = fn  + str_ +'_'
        # version = time.strftime('%m%d_%H%M%S')
        img_name  = fn+dataset+'_k='+str(random_state)+'.png'

    num_models = len(models)
    num_columns = 4
    num_rows = math.ceil(num_models/num_columns)
    
    plt.figure(figsize=(5*num_columns,5*num_rows))
    
    
    for ind,model in enumerate(models):

        X,y = read_data(dataset)

        X_train,X_test,y_train,y_test = split_data(X,y)

        pos_label = get_labels(y_test)[0]
        if model == 'cos':
            X_oversampled,y_oversampled,_,_ = oversampling(model,X_train,y_train,args['args']) 
        else:
            X_oversampled,y_oversampled = oversampling(model,X_train,y_train,args['args'])

        plt.subplot(num_rows,num_columns,ind+1)
        
        V.show_oversampling(X_train,y_train,X_oversampled,y_oversampled)
        plt.title(model)
    plt.savefig(path+img_name)
    print("The image stored in",path+img_name)
    plt.show()
    
