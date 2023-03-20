'''
@source: https://github.com/LyudaK/msc_thesis_imblearn
@author: Liudmila Kopeikina
'''
import pandas as pd 
import numpy as np
import pandas as pd
import seaborn as sns
import collections 
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors

# function to load data 
def data_loader(filename):
    data=pd.read_csv(filename+".csv")
    return data
# check data on null values
def check_notnull(data):
    plt.figure(figsize=(15, 5))
    plt.xticks(rotation=90)
    plt.ylabel('Number')
    plt.title('Non-Missing Values in columns within %d instances ' % data.shape[0])
    plt.bar(data.columns, data.notnull().sum())

# functions for EDA
def plot_displot(data):
    fig = plt.figure(1, figsize=(20, 40))
    for i in range(len(data.columns)):
        fig.add_subplot(10, 5, i + 1)
        sns.histplot(data.iloc[i], kde=True)
        plt.axvline(data[data.columns[i]].mean(), c='green')
        plt.axvline(data[data.columns[i]].median(), c='blue')

def plot_scatter(data, x, y, target):
    fig = plt.figure(1, figsize=(8, 5))
    sns.scatterplot(data=data, x=x, y=y, hue=target)
    plt.xlabel('ftr# {}'.format(x))
    plt.ylabel('ftr# {}'.format(y))
    plt.show()


def plot_class_dist(target_column):
    ax = target_column.value_counts().plot(kind='bar', figsize=(12, 8), 
                                           fontsize=12, 
                                           color=['#6ca5ce','#a06cce','#6cb4ce',
                                                  '#6cce81','#c92c4c','#c726c9'])
    ax.set_title('Target class\n', size=20, pad=30)
    ax.set_ylabel('Number of samples', fontsize=12)
    for i in ax.patches:
        ax.text(i.get_x() + 0.19, i.get_height(), str(round(i.get_height(), 2)), 
                fontsize=12)

def plot_pie(data,labels,title):
    #Usage:
    # data = df[target].value_counts()
    # print(df[target].value_counts(True)*100)
    # plot_pie(data,classes,'Gallagher Dataset')
    fig, ax = plt.subplots(figsize =(20, 10))
    colors = sns.color_palette('pastel')
    ax.pie(data, labels = labels, colors = colors)
    ax.set_title(title,fontsize=14)
    plt.show()

def plot_class_dist(target_column):
  # Usage: plot_class_dist(df[target])
    ax = target_column.value_counts().plot(kind='bar', figsize=(12, 6),
         fontsize=12, 
         color=['#6ca5ce','#a06cce','#6cb4ce','#6cce81','#c92c4c','#c726c9'])
    ax.set_title('Target class\n', size=16, pad=30)
    ax.set_ylabel('Number of samples', fontsize=12)
    for i in ax.patches:
        ax.text(i.get_x() + 0.19, i.get_height(), str(round(i.get_height(), 2)),
                fontsize=12)

def fill_missing_values(data, num_features, cat_features):
    for f in num_features:
        median = data[f].mean()
        data[f].fillna(median, inplace=True)
    for col in cat_features:
        most_frequent_category = data[col].mode()[0]
        data[col].fillna(most_frequent_category, inplace=True)


def encode_target(data, target):
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(data[target])
    return target_encoded


def standardize_data(data, num_features):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[num_features])
    return scaled_data


def transfrom_cat_features(data, cat_features):
    for c in cat_features:
        data = data.merge(pd.get_dummies(data[c], prefix=c), 
                          left_index=True, right_index=True)
    data.drop(cat_features, axis=1, inplace=True)


def split_data(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, 
                                                        stratify=target, 
                                                        test_size=0.33, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

# get the oversampler model by key from dict
def get_oversampler(oversamplers_dict, oversampler_num, proportion):
    if proportion == None:
        return oversamplers_dict[oversampler_num]()
    else:
        return oversamplers_dict[oversampler_num](proportion=proportion)

def get_filter(filters_dict, filter_num):
    return filters_dict[filter_num]

# function to preprocess input data
def preprocess(data, target, num_features, cat_features):
    # check_notnull(data.drop(target, 1))
    fill_missing_values(data.drop(target, 1), num_features, cat_features)
    data[target] = encode_target(data, target)
    data[num_features] = standardize_data(data, num_features)
    transfrom_cat_features(data, cat_features)
    return data

# function to print evaluation metrics values
def print_eval_results(y_test, preds):
    print('Classification report:')
    print(classification_report(y_test, preds))
    print('Geometric mean:', geometric_mean_score(y_test, preds, 
                                                  average='weighted'))
    print('Geometric mean default:', geometric_mean_score(y_test, preds))
    print('Cohen Kappa', cohen_kappa_score(y_test, preds))

# function to get optimal DT model
def get_model(X_train, y_train):
    param_grid = { 'criterion':['gini','entropy'],
                  'max_depth': np.arange(3, 200),
                  'max_features': ['auto', 'log2'],
                  }
    model=DecisionTreeClassifier()
    adb = GridSearchCV(model, param_grid, cv=5,scoring='f1_weighted')
    adb.fit(X_train, y_train)
    return adb.best_estimator_ 

# function for calculation of error per class
def error_per_class(y_test,preds,classes):
    cm = confusion_matrix(y_test, preds)
    # to store the results in a dictionary for easy access later
    per_class_accuracies = {}
    per_class_error={}
    # Calculate the accuracy for each one of our classes
    for idx, cls in enumerate(classes):
        # TN - all the samples that are not current GT class 
        # and not predicted as the current class
        true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        # TP are all the samples of current GT class that were predicted as such
        true_positives = cm[idx, idx]
        # accuracy for the current class
        per_class_accuracies[cls] = (true_positives) / np.sum(cm[idx,:])
        per_class_error[cls] = 1-(true_positives) / np.sum(cm[idx,:])
    print('PER CLASS ERROR', per_class_error)
    return per_class_error

# functions to perform KS test for gen/real data
def ks_test(real, gen):
    df_a=np.array(real.values)
    df_b=np.array(gen.values)
    ks_scores=ks_2samp(df_a, df_b)
    print("Gen vs Real: ks statistic",ks_scores.statistic)
    print("Gen vs Real: ks pvalue",ks_scores.pvalue)
    print("Gen & Real distributions are equal",ks_scores.pvalue>0.05)

def run_kstwo(X_sample,X_train):
    df_gen=pd.DataFrame(X_sample.copy(),columns=X_train.columns)
    df_gen=df_gen[~df_gen.isin(X_train)].dropna()
    df_gen['gen']='generated'
    df_real=pd.DataFrame(X_train.copy())
    df_real['gen']='real'
    df=pd.concat([df_real,df_gen])
    for col in df_gen.drop(['gen'],1).columns.to_list():
        print('Feature',col)
        ks_test(df_gen[col], df_real[col])

# functions for filtering data points 

# function to find N neighbors for a point
def get_neighbours(X_train,X_gen):
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', p=2).fit(np.array(X_train.values))
    _,ind=nbrs.kneighbors(np.array(X_gen.values))
    return ind

# filtering
def filter_data(X_train,y_train,X_gen,y_gen,X_test,init_error,c,y_test,classes):
  # init empty F set
    n=len(X_gen)
    X_filtered=pd.DataFrame()
    y_filtered=pd.Series()
    i=0
    # find n-neighbors for the data point
    k_neighbours=get_neighbours(X_train,X_gen)

    for kn in k_neighbours:
        X_tmp=X_train.copy()
        y_tmp=y_train.copy()
        # find class of the neigborhood 
        max_class=max(collections.Counter(y_train[kn]))
        # if gen_sample class equals to neighborhood's class we append sample to F set
        if max_class==y_gen.iloc[i]:
            X_filtered=X_filtered.append(X_gen.iloc[i])
            y_filtered=pd.concat([pd.Series(y_filtered),pd.Series(y_gen.iloc[i])])
        # otherwise we check whether there is an improvement in error rate
        else:
            X_tmp=X_tmp.append(X_gen.iloc[i])
            y_tmp=pd.concat([pd.Series(y_tmp),pd.Series(y_gen.iloc[i])])
            clf_model=get_model(X_tmp,y_tmp)
            preds = clf_model.predict(X_test)
            error=error_per_class(y_test,preds,classes)
            # if there is an improvement
            # we append sample to F set
            if init_error[c]>error[c]:
                X_filtered.append(X_gen.iloc[i])
                y_filtered.append(y_gen.iloc[i])
    i+=1
    return pd.DataFrame(X_filtered),pd.Series(y_filtered)