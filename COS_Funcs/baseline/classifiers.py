from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy as np

scoring_dict = {'recall':'recall','f1_score':'f1','g_mean':'g_mean','kappa':'kappa','auc':'roc_auc','accuracy':'accuracy','precision':'precision'}

def get_dt(X_train,y_train,metric):
    param_grid = { 'criterion':['gini','entropy'],
                  'max_depth': np.arange(3, 200),
                  'max_features': ['auto', 'log2'],
                  }
    model=DecisionTreeClassifier()
    adb = GridSearchCV(model, param_grid, cv=5,scoring=scoring_dict['recall'])
    adb.fit(X_train, y_train)
    return adb.best_estimator_ 

def get_svm(X_train,y_train,metric):
    print('*'*15,'Classifier Optimizing','*'*15)
    grid = GridSearchCV(estimator=SVC(),param_grid={'C': [0.1,1, 10, 100],
                                                    'kernel': ['rbf', 'sigmoid'],
                                                    'gamma': [1,0.1,0.01,0.001],},scoring=scoring_dict[metric],n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_ 

def get_knn(X_train,y_train,metric):
    print('*'*15,'Classifier Optimizing','*'*15)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid = {'n_neighbors':list(range(1, 31, 2))}, scoring=scoring_dict[metric],n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_ 

def do_classification(X_train,y_train,X_test,classification_model,metric=None):
    
    if metric is None:
        if classification_model == 'knn':
            model = KNeighborsClassifier()
        
        elif classification_model == 'svm':
            model = SVC(probability=True)
            
        elif classification_model == 'decision_tree':
            model = DecisionTreeClassifier(random_state=23,)
            
        elif classification_model == 'random_forest':
            model = RandomForestClassifier(random_state=23,)
                        
        elif classification_model == 'mlp':
            model = MLPClassifier()
            
        elif classification_model == 'naive_bayes':
            model = GaussianNB()
            
    else:
        if classification_model == 'knn':
            model = get_knn(X_train, y_train, metric)
        
        elif classification_model == 'svm':
            model = get_svm(X_train, y_train, metric)
            
        elif classification_model == 'decision_tree':
            model = get_dt(X_train, y_train, metric)
        
        elif classification_model == 'random_forest':
            model = RandomForestClassifier()
        
        elif classification_model == 'mlp':
            model = MLPClassifier()
        
        elif classification_model == 'naive_bayes':
            model = GaussianNB()
            
        # LR
        
    model.fit(X_train,y_train)         
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return y_pred,y_pred_proba