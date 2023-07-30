'''
@brief tools
@author yizhi
'''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import glob
import tqdm
import json
import sys


def read_data(dataset,norm=True):
    '''
    @brief read the data saved in the path dataset
    @note dataset should in the csv format that the last column should be the labels
    '''
    df = pd.read_csv(dataset)
    # 
    X = df.values[:,:-1]
    y = df.values[:,-1]
    if norm:
        # Data preprocessing
        X = standard(X)
    return X,y


def read_fold(dataset,k):
    '''
    @brief Read one of the fold of splitted and standardized dataset we saved
    @return X_train,X_test,y_train,y_test
    @note I saved 5 folds(0~4)
    '''
    # We save  for test, for runing 10 fold validation 10 times, we set k to 100, if k>9 it read data from the beginning
    if k >= 5:
       k = k % 5 
    path = dataset.split('.')[0]
    path = os.path.join(path,str(k))
    # print(path)
    X_train,y_train = read_data(os.path.join(path,'train.csv'),norm=False)
    X_test,y_test = read_data(os.path.join(path,'test.csv'),norm=False)
    return X_train,X_test,y_train,y_test


def standard(X):
    ss = StandardScaler()
    return ss.fit_transform(X)

def get_labels(y):
    '''
    @return minority class's label,majority class's label
    @note only works for binary dataset
    '''
    valuecounts = pd.Series(y).value_counts().index
    majlabel = valuecounts[0]
    minlabel = valuecounts[1:]
    if len(minlabel) == 1:
        minlabel=int(minlabel[0])
    return minlabel,int(majlabel)

def split_data(X,y,random_state=None):
    '''
    @return X_train,X_test,y_train,y_test
    '''
    return train_test_split(X,y,stratify=y,random_state=random_state)
    
    
# Functions to deal the test result
def to_excel(file_name,dfs,sheet_names):    
    writer = pd.ExcelWriter(file_name)
    if len(dfs) != len(sheet_names):
        raise "sheet name should correspond to the number of dataframes!"
    for df,sheet_name in zip(dfs,sheet_names):
        df.to_excel(writer,sheet_name=sheet_name)
    writer.save()
    
class Logger(object):
    def __init__(self, filename="baseline.log", path="./"):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
def base_file(path):
    if '\\' in path:
        return path.split('\\')[-1]
    elif '/' in path:
        return path.split('/')[-1]
        
def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def base_name(metric,classifier,k):
    return classifier+'_'+metric+'_k'+str(k)+'.xlsx'

def sheet_name(metric,classifier,k):
    return classifier+'_'+metric+'_k'+str(k)

def key_name(dataset,sheet_name):
    return dataset+'_'+sheet_name

def cos_file_name(classifiers,metrics):
    file_name = 'cos'
    for i in classifiers:
        file_name += '_' + i
    for i in metrics:
        file_name += '_' + i
    return file_name

def save_json(json_,file_name):
    with open(file_name,"w") as outfile:
        json.dump(json_, outfile)
    return file_name

def read_all_avg(path,file_name= 'all_avg_k10.xlsx'):
    '''
    @brief Extract the avg sheet of all test xlsx files into one xlsx file
    '''
    xlsxs = glob.glob(os.path.join(path,'*.xlsx'))
    file_name = os.path.join(path,file_name)
    writer = pd.ExcelWriter(file_name)
    if os.path.isdir(path):
        for test_file_name in tqdm.tqdm(xlsxs):
            sheet_name =  os.path.basename(test_file_name)
            sheet_name = sheet_name.split('_k10')[0]
            df = pd.read_excel(test_file_name,'avg',)
            df.columns = np.r_[['Datasets'],df.columns.values[1:]]
            df.to_excel(writer,sheet_name=sheet_name,index=False,index_label='')
        writer.save()
    print('File saved in',file_name)