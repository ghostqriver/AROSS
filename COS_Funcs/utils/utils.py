import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    @ Return minority class's label,majority class's label
    @ Only works on binary dataset
    '''
    valuecounts = pd.Series(y).value_counts().index
    majlabel = valuecounts[0]
    minlabel = valuecounts[1:]
    if len(minlabel) == 1:
        minlabel=int(minlabel[0])
    return minlabel,int(majlabel)

def split_data(X,y,random_state=None):
    '''
    @  Return: X_train,X_test,y_train,y_test
    '''
    return train_test_split(X,y,stratify=y,random_state=random_state,)
