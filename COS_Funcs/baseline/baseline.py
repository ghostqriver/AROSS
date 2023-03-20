dataset_path = 'Datasets/'

oversamplers = ['original','smote','db_smote','smote_d','cure_smote','kmeans_smote','adasyn','somo','symprod',
                'smote_enn','smote_tl','nras','g_smote','rwo_sampling','ans','svm_smote',
                # 'd_smote' also slow 
                # 'wgan', 'wgan_filter' need to be modify a bit, 'tabgan' error
                # 'cos'
                ] 

classifiers = ['knn','svm','decision_tree','random_forest','mlp','naive_bayes'] 

metrics = ['recall','f1_score','g_mean','kappa','auc','accuracy','precision']

save_path = 'test/'
cos_save_path = 'costest/'
gan_save_path = 'gantest/'

import os
import glob
datasets = glob.glob(os.path.join(dataset_path,'*.csv'))


from COS_Funcs.utils import *
from COS_Funcs.baseline.classifiers import do_classification
from COS_Funcs.baseline.oversamplers import do_oversampling
from COS_Funcs.baseline.metrics import calc_score

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import time
import math
from tqdm import tqdm
import copy

def baseline(classifiers=classifiers,metrics=metrics,k=10,oversamplers=oversamplers,dataset_path=dataset_path,datasets=datasets,show_folds=False,**args):
    #pd.set_option('precision',5)  
    #pd.set_option('display.width', 100)
    #pd.set_option('expand_frame_repr', False)
    warnings.filterwarnings("ignore") 
    
    if 'cos' in oversamplers:
        path = cos_save_path
    elif 'wgan' in oversamplers:
        path = gan_save_path
    else:
        path = save_path
        
    make_dir(path)

    writers = create_writer(classifiers,metrics,k,oversamplers=oversamplers,datasets=datasets,path=path)
    
    for random_state in tqdm(range(k)):
        
        for dataset in datasets: 
            
#             print(dataset)
            scores = [] 

            for oversampler in oversamplers:
                print(oversampler,end='|')
                X,y = read_data(dataset)

                X_train,X_test,y_train,y_test = split_data(X,y)

                pos_label = get_labels(y)[0]
                
                try:
                    start = time.time()
                    if oversampler == 'cos':
                        X_train,y_train,num_all_safe,num_half_safe = do_oversampling(oversampler,X_train,y_train,args) 

                    else:
                        X_train,y_train = do_oversampling(oversampler,X_train,y_train,args) 
                    end = time.time()
                    print('cost:',end-start)
                    for classifier in classifiers:     
                        y_pred = do_classification(X_train,y_train,X_test,classifier)
                        for metric in metrics:
                            print(random_state+1,'|',dataset,'|',oversampler,'|',classifier,'|',metric,':',end='')
                            score = calc_score(metric,y_test,y_pred,pos_label)
                            print(score)
                            writers[classifier][metric]['scores_df'][random_state][oversampler].loc[dataset] = score
                            
                    
                except BaseException as e: 
                    print(oversampler,'cause an error on',dataset)
                    continue
                
    write_writer(writers,classifiers,metrics,k,oversamplers,datasets)
    return writers

def base_name(metric,classifier,k):
    return classifier+'_'+metric+'_k'+str(k)+'.xlsx'

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
    
def gen_df(models,datasets,avg_scores):
    # avg_scores_df = pd.DataFrame(columns=models+['all safe area','half safe area'],index=datasets)
    avg_scores_df = pd.DataFrame(columns=models,index=datasets)

    for index,i in zip(avg_scores_df.index,range(len(avg_scores))):
        avg_scores_df.loc[index] = avg_scores[i]
    return avg_scores_df