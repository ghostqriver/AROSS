from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import glob
import tqdm
import json

def read_data(dataset,norm=True):
    df = pd.read_csv(dataset)
    # All dataset frame should in the format that the final column should be the labels
    X = df.values[:,:-1]
    y = df.values[:,-1]
    if norm:
        # Data preprocessing
        X = standard(X)
    
    return X,y

def standard(X):
    ss = StandardScaler()
    return ss.fit_transform(X)

def get_labels(y):
    '''
    @return minority class's label,majority class's label
    @note Only works for binary dataset
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
    return train_test_split(X,y,stratify=y,random_state=random_state,)#)test_size=0.33)


# Baseline functions
def make_dir(dir):
    if not os.path.exists(dir.split('/')[0]):
        os.mkdir(dir.split('/')[0])

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
    # file_name = os.path.join(path,file_name)
    writer = pd.ExcelWriter(file_name)
    if os.path.isdir(path):
        for test_file_name in tqdm.tqdm(xlsxs):
            sheet_name =  os.path.basename(test_file_name)
            sheet_name = sheet_name.split('_k10')[0]
            df = pd.read_excel(test_file_name,'avg',)
            df.columns = np.r_[['Datasets'],df.columns.values[1:]]
            df.to_excel(writer,sheet_name=sheet_name,index=False,index_label='')
        writer.save()
    print('File saved in,',file_name)