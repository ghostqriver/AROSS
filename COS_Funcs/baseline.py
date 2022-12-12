from sklearn import metrics
import COS_Funcs.metrics as M
import COS_Funcs.visualize as V
import COS_Funcs.cos as cos
import COS_Funcs.generate as G

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE,SVMSMOTE,ADASYN
from imblearn.combine import SMOTETomek,SMOTEENN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from smote_variants import (DBSMOTE,DSMOTE,SMOTE_D,CURE_SMOTE,kmeans_SMOTE,SOMO,NRAS,SYMPROD)
from dtosmote.dto_smote import DTO

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import time
import os
import math
from tqdm import tqdm

# Global variables
dataset_path = 'Dataset/'
datasets = ['Sampledata_new_1','Sampledata_new_2','Sampledata_new_3','Sampledata1','yeast','pima-indians-diabetes',
            'haberman',
            'ecoli2','glass1'
            ]


models = [#'original','smote','db_smote'
        #   ,'smote_d'
        #,'cure_smote','kmeans_smote'
        #   ,'adasyn','somo','symprod'
          #,
        'cos'] 
        # 'smote_enn','smote_tl','d_smote','nras' was moved
        # 'dto_smote' can not worked on 'Sampledata1'


def calc_score(metric,y_test,y_pred,pos_label):
    '''
    Works only in binary classification case because in COS_Funcs.metrics we only considered about binary cases
    g_mean function is from COS_Funcs.metrics, all others are from sklearn.metrics
    if the value of metric is not in ['recall','f1_score','g_mean','kappa','auc','accuracy','precision'],
    it will return the Recall value by default
    
    pos_label: the positive label, should be set as the minority label in our case
    '''
    
    if metric == 'recall':
        return metrics.recall_score(y_test,y_pred,pos_label=pos_label)
    
    elif metric == 'f1_score':
        return metrics.f1_score(y_test,y_pred,pos_label=pos_label)
    
    elif metric == 'g_mean':
        return M.g_mean(y_test,y_pred,pos_label=pos_label)
    
    elif metric == 'kappa':
        return metrics.cohen_kappa_score(y_test,y_pred)
    
    elif metric == 'auc':
        # Put the auc here when getting what it is 
        return None
    
    elif metric == 'accuracy':
        return metrics.accuracy_score(y_test,y_pred)
    
    elif metric == 'precision':
        return metrics.precision_score(y_test,y_pred,pos_label=pos_label)
    
    else:
        return metrics.recall_score(y_test,y_pred,pos_label=pos_label)
    
    
def oversampling(model,X_train,y_train,*args): # !!!! Donia
    
    if model == 'original':
        return X_train,y_train
    
    elif model == 'smote':
        smote = SMOTE()
        return smote.fit_resample(X_train,y_train)
    
    elif model == 'smote_enn':
        smoteenn = SMOTEENN()
        return smoteenn.fit_resample(X_train,y_train)
    
    elif model == 'smote_tl':
        smotetl = SMOTETomek()
        return smotetl.fit_resample(X_train,y_train,)
    
    elif model == 'adasyn':
        adasyn = ADASYN()
        return adasyn.fit_resample(X_train,y_train)
    
    elif model == 'db_smote':
        dbsmote = DBSMOTE()
        return dbsmote.sample(X_train,y_train)
    
    elif model == 'd_smote':
        dsmote = DSMOTE()
        return dsmote.sample(X_train,y_train)
    
    elif model == 'smote_d':
        smoted = SMOTE_D()
        return smoted.sample(X_train,y_train)
    
    elif model == 'cure_smote':
        curesmote = CURE_SMOTE()
        return curesmote.sample(X_train,y_train)
    
    elif model == 'kmeans_smote':
        kmeanssmote = kmeans_SMOTE()
        return kmeanssmote.sample(X_train,y_train)
    
    elif model == 'somo':
        somo = SOMO()
        return somo.sample(X_train,y_train)
    
    elif model == 'nras':
        nras = NRAS()
        return nras.sample(X_train,y_train)
    
    elif model == 'symprod':
        symprod = SYMPROD()
        return symprod.sample(X_train,y_train)
    
    elif model == 'dto_smote':
        delaunay = DTO('test','solid_angle',7.5)
        return delaunay.fit_resample(X_train,y_train)
    
    elif model == 'cos':
        N,c,alpha,linkage,L,shrink_half,expand_half,all_safe_weight,all_safe_gen,half_safe_gen,Gaussian_scale,IR,minlabel,majlabel,visualize = get_cos_para(args[0])
        return cos.COS(X_train,y_train,N,c,alpha,linkage,L,shrink_half,expand_half,all_safe_weight,all_safe_gen,half_safe_gen,Gaussian_scale,IR,minlabel,majlabel,visualize)
    
    else:
        return 0
    
    
def do_classification(X_train,y_train,X_test,classification_model):
    
    if classification_model == 'knn':
        model = KNeighborsClassifier()
       
    
    elif classification_model == 'svm':
        model = SVC()
        
    elif classification_model == 'decision_tree':
        model = DecisionTreeClassifier()
    
    elif classification_model == 'random_forest':
        model = RandomForestClassifier()
    
    elif classification_model == 'neural_network':
        model = MLPClassifier()
    
    elif classification_model == 'c_classifier':
        pass
    
    model.fit(X_train,y_train)
    return model.predict(X_test)


def read_data(dataset_path,dataset):
    df = pd.read_csv(dataset_path+dataset+'.csv')
    
     # All dataset frame should in the format that the final column should be the labels
    X = df.values[:,:-1]
    y = df.values[:,-1]
    
    # Data preprocessing
    ss = StandardScaler()
    X = ss.fit_transform(X)
    
    return X,y


def check_dir(dir):
    if not os.path.exists(dir.split('/')[0]):
        os.mkdir(dir.split('/')[0])
        

def gen_df(models,datasets,avg_scores):
    avg_scores_df = pd.DataFrame(columns=models+['all safe area','half safe area'],index=datasets)
    for index,i in zip(avg_scores_df.index,range(len(avg_scores))):
        avg_scores_df.loc[index] = avg_scores[i]
    return avg_scores_df


def get_cos_para(args):
    
    c = args['c']
    N = args['N']
    alpha = args['alpha']

    if 'linkage' in args.keys():
        linkage = args['linkage']
    else:
        linkage = 'single'

    if 'l' in args.keys():
        L = args['l']
    else:
        L = 2

    if 'shrink_half' in args.keys():
        shrink_half = args['shrink_half']
    else:
        shrink_half = None

    if 'expand_half' in args.keys():
        expand_half = args['expand_half']
    else:
        expand_half = None

    if 'all_safe_weight' in args.keys():
        all_safe_weight = args['all_safe_weight']
    else:
        all_safe_weight = 2

    if 'all_safe_gen' in args.keys():
        all_safe_gen = args['all_safe_gen']
    else:
        all_safe_gen = G.Smote_Generator
        
    if 'half_safe_gen' in args.keys():
        half_safe_gen = args['half_safe_gen']
    else:
        half_safe_gen = G.Smote_Generator   

    if 'gaussian_scale' in args.keys():
        Gaussian_scale = args['gaussian_scale']
    else:
        Gaussian_scale = 0.8  

    if 'ir' in args.keys():
        IR = args['ir']
    else:
        IR=1

    if 'minlabel' in args.keys():
        minlabel = args['minlabel']
    else:
        minlabel = None

    if 'majlabel' in args.keys():
        majlabel = args['majlabel']
    else:
        majlabel = None

    if 'visualize' in args.keys():
        visualize = args['visualize']
    else:
        visualize = False

    return N,c,alpha,linkage,L,shrink_half,expand_half,all_safe_weight,all_safe_gen,half_safe_gen,Gaussian_scale,IR,minlabel,majlabel,visualize


def gen_file_name(args):
    '''
    Generate a string by the given parameters of COS
    '''
    fn = ''
    # exclude 'all_safe_gen','half_safe_gen' in automatical way, or it will error because the generator function name is invalid when creating a file
    for str_ in [j[0]+str(j[1]) for j in list(filter(lambda i:i[0] not in ['all_safe_gen','half_safe_gen'],[(i,args['args'][i]) for i in args['args']]))]:
        fn = fn  + str_ +'_'
    # add generator in
    if 'all_safe_gen' in args['args'].keys():
        if args['args']['all_safe_gen'] == G.Gaussian_Generator:
            fn = fn + 'all_safe_genGaussian' + '_'
    else:
        fn = fn + 'all_safe_genSMOTE' + '_'
            
    if 'half_safe_gen' in args['args'].keys():
        if args['args']['half_safe_gen'] == G.Gaussian_Generator:
            fn = fn + 'half_safe_genGaussian' + '_'
    else:
        fn = fn + 'half_safe_genSMOTE' + '_'
    return fn

def baseline(metric,classification_model,k=10,pos_label=None,excel_name=None,show_folds=False,**args):
    '''
    '''
    pd.set_option('precision',5)  
    pd.set_option('display.width', 100)
    pd.set_option('expand_frame_repr', False)
    warnings.filterwarnings("ignore") 
    
    path = 'baselines/'
    check_dir(path)
    
    if excel_name == None:
       
        # version = time.strftime('%m%d_%H%M%S')
        fn = gen_file_name(args)
        excel_name = fn+metric+'_'+classification_model+'_k'+str(k)+'.xlsx'

    writer = pd.ExcelWriter(path+excel_name)
    
    for random_state in tqdm(range(k)):
        
        scores_df = pd.DataFrame(columns=models+['all safe area','half safe area'],index=datasets)
        
        for dataset in datasets: 

            # print(dataset)
            scores = [] 

            for model in models:

                X,y = read_data(dataset_path,dataset)

                X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=random_state)

                pos_label = cos.get_labels(y_test)[0]

                if model == 'cos':
                    X_train,y_train,num_all_safe,num_half_safe = oversampling(model,X_train,y_train,args['args']) # Donia
                else:
                    X_train,y_train = oversampling(model,X_train,y_train,args['args']) # Donia

                y_pred = do_classification(X_train,y_train,X_test,classification_model)

                scores.append(calc_score(metric,y_test,y_pred,pos_label))

                if model == 'cos':
                    scores.append(num_all_safe)
                    scores.append(num_half_safe)

            scores_df.loc[dataset] = scores
        
        if random_state == 0:
            avg_scores = scores_df.values
        else:
            avg_scores += scores_df.values
        
        if show_folds == True:
            print(random_state+1,'fold:')
            print(scores_df)
            
        scores_df.to_excel(writer,sheet_name= 'fold_'+str(random_state+1))
        
    avg_scores = avg_scores/k
    
    avg_scores_df = gen_df(models,datasets,avg_scores)
    
    avg_scores_df.to_excel(writer,sheet_name= 'avg')
    writer.save()
    print("The scores in each fold stored in",path+excel_name)
    print("The average scores:")
    print(avg_scores_df)

    return path+excel_name


def show_baseline(dataset,random_state=None,pos_label=None,img_name=None,**args):
    
    path = 'baselines/'
    check_dir(path)

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

        X,y = read_data(dataset_path,dataset)

        X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=random_state)

        pos_label = cos.get_labels(y_test)[0]
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



def show_baseline_cos(dataset,random_state=None,pos_label=None,**args): 
    
    model = 'cos'

    args['args']['visualize'] = True

    X,y = read_data(dataset_path,dataset)

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=random_state)

    pos_label = cos.get_labels(y_test)[0]

    X_oversampled,y_oversampled,_,_ = oversampling(model,X_train,y_train,args['args'])

    plt.show()