dataset_path = 'Datasets/'

models = [
        'original','smote','db_smote','smote_d','cure_smote','kmeans_smote','adasyn','somo','symprod',
        'smote_enn','smote_tl','d_smote','nras','g_smote','rwo_sampling','ans','svm_smote',
         # 'wgan', 'wgan_filter' need to be modify a bit, 'tabgan' error
         # 'cos'
         ] 

classifiers = ['knn','svm','decision_tree','random_forest','mlp','naive_bayes'] 

metrics = ['recall','f1_score','g_mean','kappa','auc','accuracy','precision']

import glob
datasets = glob.glob(os.path.join(dataset_path,'*.csv'))


from COS_Funcs.utils import *
from COS_Funcs.baseline.classifiers import *
from COS_Funcs.baseline.oversamplers import *
from COS_Funcs.baseline.metrics import *

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
import math
from tqdm import tqdm
import copy

def gen_df(models,datasets,avg_scores):
    # avg_scores_df = pd.DataFrame(columns=models+['all safe area','half safe area'],index=datasets)
    avg_scores_df = pd.DataFrame(columns=models,index=datasets)

    for index,i in zip(avg_scores_df.index,range(len(avg_scores))):
        avg_scores_df.loc[index] = avg_scores[i]
    return avg_scores_df

def base_name(metric,classification_model,k):
    return metric+'_'+classification_model+'_k'+str(k)+'.xlsx'

def baseline(metric,classification_model,k=10,pos_label=None,excel_name=None,show_folds=False,dataset_path=dataset_path,datasets=datasets,**args):
    '''
    '''
    print(args)
    pd.set_option('precision',5)  
    pd.set_option('display.width', 100)
    pd.set_option('expand_frame_repr', False)
    warnings.filterwarnings("ignore") 
    
    path = 'test/'
    make_dir(path)
    
    if excel_name == None:
       
        # version = time.strftime('%m%d_%H%M%S')
        # fn = gen_file_name(args)
        # excel_name = fn+metric+'_'+classification_model+'_k'+str(k)+'.xlsx'
        excel_name = base_name(metric,classification_model,k)
        
    writer = pd.ExcelWriter(os.path.join(path+excel_name))
    
    for random_state in tqdm(range(k)):
        
        # scores_df = pd.DataFrame(columns=models+['all safe area','half safe area'],index=datasets)
        scores_df = pd.DataFrame(columns=models,index=datasets)
        
        for dataset in datasets: 
            
            print(dataset)
            scores = [] 

            for model in models:

                X,y = read_data(dataset)

                X_train,X_test,y_train,y_test = split_data(X,y)

                # if pos_label == None:
                pos_label = get_labels(y_test)[0]
                
                try:
                    
                    if model == 'cos':
                        X_train,y_train,num_all_safe,num_half_safe = oversampling(model,X_train,y_train,args) 
                        
                    elif model == 'wgan':
                        pass
                        # X_train,y_train,X_train_f,y_train_f = 
                    elif model == 'wgan_filter':
                        pass
                        # X_train,y_train = X_train_f,y_train_f
                    else:
                        X_train,y_train = oversampling(model,X_train,y_train,args) 
                    
                    
                    # for classificaion_model in classfication_models:
                             
                    y_pred = do_classification(X_train,y_train,X_test,classification_model)
                    # for metric in metrics:   
                    scores.append(calc_score(metric,y_test,y_pred,pos_label))
                    
                except BaseException as e: 
                    scores.append(None)
                    continue
                                    
                if model == 'cos':
                    scores.append(num_all_safe)
                    scores.append(num_half_safe)

            scores_df.loc[dataset] = scores
        
        tmp_df = copy.deepcopy(scores_df)
        tmp_df.fillna(0)
        if random_state == 0:
            avg_scores = tmp_df.values
        else:
            avg_scores += tmp_df.values
        
        if show_folds == True:
            print(random_state+1,'fold:')
            print(scores_df)
            
        scores_df.columns = list(map(lambda x:str.upper(x),models))
        scores_df.index = list(map(lambda x:os.path.basename(x).split('.')[0],datasets))
        scores_df.to_excel(writer,sheet_name= 'fold_'+str(random_state+1))
        
    avg_scores = avg_scores/k
    
    avg_scores_df = gen_df(models,datasets,avg_scores)
    avg_scores_df.columns = list(map(lambda x:str.upper(x),models))
    avg_scores_df.index = list(map(lambda x:os.path.basename(x).split('.')[0],datasets))
    avg_scores_df.to_excel(writer,sheet_name= 'avg')
    writer.save()
    print("The scores in each fold stored in",path+excel_name)
    print("The average scores:")
    print(avg_scores_df)

    return path+excel_name

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