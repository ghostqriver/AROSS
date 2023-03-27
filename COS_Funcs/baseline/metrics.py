import COS_Funcs.cos.cos_bf as cos_bf
from COS_Funcs.utils import *

from sklearn import metrics
import numpy as np

def calc_score(metric,y_test,y_pred,pos_label):
    '''
    @ Works only in binary classification case.
    @ metric: ['recall','f1_score','g_mean','kappa','auc','accuracy','precision'], for any other values it will return the Recall value by default
    @ pos_label: the positive label, should be set as the minority label in our case
    '''
    
    if metric == 'recall':
        return metrics.recall_score(y_test,y_pred,pos_label=pos_label)
    
    elif metric == 'f1_score':
        return metrics.f1_score(y_test,y_pred,pos_label=pos_label)
    elif metric == 'f2_score':
        pass
    
    elif metric == 'g_mean':
        return g_mean(y_test,y_pred,pos_label=pos_label)
    
    elif metric == 'kappa':
        return metrics.cohen_kappa_score(y_test,y_pred)
    
    elif metric == 'auc':
        return auc(y_test,y_pred,pos_label=pos_label)
    
    elif metric == 'accuracy':
        return metrics.accuracy_score(y_test,y_pred)
    
    elif metric == 'precision':
        return metrics.precision_score(y_test,y_pred,pos_label=pos_label)
    
    else:
        return metrics.recall_score(y_test,y_pred,pos_label=pos_label)
    
def auc(y_test,y_pred,pos_label=None):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=pos_label)
    return metrics.auc(fpr, tpr)

def confusion_matrix(y_test,y_pred,pos_label=None):
    '''
    Will return TP,FP,FN,TN
    pos_label: the positive class(minority class)'s label
    '''
    if pos_label == None:
        pos_label = get_labels(y_test)[0]
        
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index in range(len(y_test)):
        pred = y_pred[index]
        true = y_test[index]
        if pred == true:
            if true == pos_label:
                TP += 1
            else:
                TN += 1 
        else:
            if pred == pos_label:
                FP += 1
            else:
                FN +=1
    return TP,FP,FN,TN

def accuracy(y_test,y_pred,pos_label=None):
    '''
    In accuracy score these is not necessay to set a pos_label, we just put it here for a None
    '''
    TP,FP,FN,TN = confusion_matrix(y_test,y_pred,pos_label)
    return (TP+TN) / len(y_test)
    
def precision(y_test,y_pred,pos_label=None):
    TP,FP,FN,TN = confusion_matrix(y_test,y_pred,pos_label)
    return TP / (TP + FP)

def recall(y_test,y_pred,pos_label=None):
    TP,FP,FN,TN = confusion_matrix(y_test,y_pred,pos_label)
    return TP / (TP + FN)

def f1_score(y_test,y_pred,pos_label=None):
    precision_ = precision(y_test,y_pred,pos_label)
    recall_ = recall(y_test,y_pred,pos_label)
    return 2 * (precision_ * recall_) / (precision_ + recall_)

def g_mean(y_test,y_pred,pos_label=None):
    TP,FP,FN,TN = confusion_matrix(y_test,y_pred,pos_label)
    # Equals to  np.sqrt(sklearn.metrics.recall_score(y_test,y_pred,pos_label=0) *  sklearn.metrics.recall_score(y_test,y_pred,pos_label=1))
    # 0 and 1 are minority and majority classes here
    return np.sqrt((TP / (TP + FN))*(TN/(TN + FP))) 

def kappa(y_test,y_pred,pos_label=None):
    '''
    In kappa score these is not necessay to set a pos_label, we just put it here for a None
    '''
    TP,FP,FN,TN = confusion_matrix(y_test,y_pred,pos_label)
    acc = accuracy(y_test,y_pred,pos_label)
    N = len(y_test)
    pe = ((TP + FN) / N) * ((TP + FP) / N) + ((TN + FN) / N) * ((TN + FP) / N)
    return (acc - pe) / (1 - pe)