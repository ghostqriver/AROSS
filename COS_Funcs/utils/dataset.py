label_name = 'label'
min_label = 1
maj_label = 0


from sklearn.datasets import load_svmlight_file
import pandas as pd
import numpy as np
import os
import glob
from COS_Funcs.utils import *


def dataset_description(dataset_path,save=True):
    path = dataset_path
    if os.path.isdir(path):
        dict_ls = []
        for file_name in glob.glob(os.path.join(path+'*.csv')):
            dict_ls.append(read_info(file_name))
        dataset_df = pd.DataFrame.from_dict(dict_ls)
        if save:
            save_path = os.path.join(path+'dataset_description.xlsx')
            dataset_df.to_excel(save_path,index=False)
            print('Dataset description saved in',save_path)   
        else:
            return dataset_df
    else: # is file
        dataset_df = pd.DataFrame.from_dict([read_info(path)])
        return dataset_df

def read_info(file_name):
    df = pd.read_csv(file_name)
    y = df.values[:,-1]
    
    dict = {}
    # print(file_name)
    dict['dataset'] = os.path.basename(file_name).split('.')[0]
    minlabel,majlabel = get_labels(y)
    
    dict['minority_class'] = 'Class' + str(minlabel)
    dict['majority_class'] = 'Class' + str(majlabel)
    dict['num_of_attributes'] = len(df.columns) - 1
    dict['num_of_minority_samples'] = len(y[y==minlabel])
    dict['num_of_majority_samples'] = len(y[y==majlabel])
    dict['imbalance_ratio'] = round(dict['num_of_majority_samples']/dict['num_of_minority_samples'],2)
    dict['label_name'] = df.columns.values[-1]
    return dict
       
def dataset_transform(file_path,format='svmlight'):
    if format == 'svmlight':
        X, y = load_svmlight_file(file_path)
        X = np.array(X.todense())
        y = pd.Series(y.astype(np.int))
        df = pd.DataFrame(X,columns=range(1,X.shape[1]+1))
        min_label_,maj_label_=get_labels(y)
        y = pd.Series(y)
        y.replace(maj_label_,maj_label,inplace=True)
        y.replace(min_label_,min_label_,inplace=True)
        df['label'] = y
    save_path = file_path.split('.')[0] + '.csv'
    df.to_csv(save_path,index=False)
    print('Dataset saved in',save_path) 
    return dataset_description(save_path)