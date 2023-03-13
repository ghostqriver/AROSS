'''
This is the implementation of metrics we will use, the main implementation is the g_mean which wasn't implemented in sklearn.metrics.
Other implementations mainly use for test and checking the score.

'''

import numpy as np
# import COS_Funcs.cos as cos
import cos as cos



def confusion_matrix(y_test,y_pred,pos_label=None):
    '''
    Will return TP,FP,FN,TN
    pos_label: the positive class(minority class)'s label
    '''
    if pos_label == None:
        pos_label = cos.get_labels(y_test)[0]
        
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