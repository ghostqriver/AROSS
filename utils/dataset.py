'''
@brief functions dealing with the dataset
@author yizhi
@note 
    label_name = 'label'
    min_label = 1
    maj_label = 0
'''
label_name = 'label'
min_label = 1
maj_label = 0
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import glob

from utils import *


def dataset_description(dataset_path,save=True):
    '''
    @brief generate the excel describing the datasets saved in dataset_path
    '''
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
    dict['num_of_rows'] = len(df)

    dict['num_of_minority_samples'] = len(y[y==minlabel])
    dict['num_of_majority_samples'] = len(y[y==majlabel])
    dict['imbalance_ratio'] = round(dict['num_of_majority_samples']/dict['num_of_minority_samples'],2)
    dict['label_name'] = df.columns.values[-1]
    return dict
       
       
def dataset_transform(file_path,format='svmlight'):
    '''
    @brief given a folder saving datasets in a certain format, transform all datasets into csv
    '''
    if format == 'svmlight':
        X, y = load_svmlight_file(file_path)
        X = np.array(X.todense())
        y = pd.Series(y.astype(np.int))
        df = pd.DataFrame(X,columns=range(1,X.shape[1]+1))
        min_label_,maj_label_=get_labels(y)
        y = pd.Series(y)
        y.replace(maj_label_,maj_label,inplace=True)
        y.replace(min_label_,min_label,inplace=True)
        df['label'] = y
    save_path = file_path.split('.')[0] + '.csv'
    df.to_csv(save_path,index=False)
    print('Dataset saved in',save_path) 
    return dataset_description(save_path)


def split_datasets(path='Datasets/',k=5):
    '''
    @brief split and standardize dataset saved as csv in the given path into k folds
    '''
    datasets = glob.glob(os.path.join(path,'*.csv'))
    for dataset in datasets:
        dir_name = dataset.split('.')[0]
    #         print(dir_name)
        make_dir(dir_name)

        # read original file in
        X,y = read_data(dataset,norm=False)

        # Split data
        skf = StratifiedKFold(n_splits=k)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            # Standardize
            ss = StandardScaler()
            X_train = ss.fit_transform(X_train)
            X_test = ss.transform(X_test)

            fold_dir_name = os.path.join(dir_name,str(i))
            print(fold_dir_name)
            make_dir(fold_dir_name)

            # Save training and test set
            df_train = pd.DataFrame(X_train)
            df_train['label'] = y_train
            df_train.to_csv(os.path.join(fold_dir_name,'train.csv'),index=None)
            df_test = pd.DataFrame(X_test)
            df_test['label'] = y_test
            df_test.to_csv(os.path.join(fold_dir_name,'test.csv'),index=None)